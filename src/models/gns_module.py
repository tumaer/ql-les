#src.models.gns_module.py
from typing import Any, Dict, Tuple
from functools import partial
import torch
from torch import Tensor
from lightning import LightningModule
from src.models.components.gns import get_random_walk_noise_for_position_sequence
from src.utils.train_utils import (
    push_forward_sample_steps, eval_rollout, integrate, kolm_eval_rollout
)

class GNSLitModule(LightningModule):
    """A LightningModule for training a Graph Network Simulator (GNS) model."""
    
    def __init__(
        self,
        net: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        seed: int = 0,
        pushforward: Dict[str, Any] = None,
        num_rollout_steps: int = 1,
        num_eval_steps: int = 1,
        pbc: bool = True,
        vel_solver: str = "simple",
        alpha_u: float = 1.0,
        alpha_v: float = 1.0,
    ) -> None:
        """Initialize the GNS model's LightningModule.

        :param net: The GNS model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # Save hyperparameters and initialize the network
        # For model checkpointing
        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net
        self.pbc = pbc
        
        # Loss weights
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.net.alpha_u = alpha_u

        # Push forward configuration
        self.seed = seed
        self.pushforward = pushforward
        if (self.pushforward is not None) and (self.alpha_u != 0.0):
            raise NotImplementedError(
                "Pushforward is only implemented for alpha_u = 0.0, i.e. LagrangeBench setting."
            )
        
        # Number of eval steps
        self.num_rollout_steps = num_rollout_steps
        self.num_eval_steps = num_eval_steps
        self.vel_solver = vel_solver
        
    def forward(
        self,
        features: Dict[str, Tensor],
    ) -> Tensor:
        sampled_noise = get_random_walk_noise_for_position_sequence(
            features["position"], 
            boundaries=self.net._boundaries,
            noise_std_last_step=self.net.noise_std,
            ).to(self.device)
        
        non_kinematic_mask = (features["particle_type"] != 3).clone().detach()
        sampled_noise *= non_kinematic_mask.view(-1, 1, 1)
        if not self.training:
            sampled_noise *= 0.0

        if self.alpha_u != 0.0:  # if true, we recover LagrangeBench; TODO: think of better condition
            pred, target_normalized_acceleration = self.net.predict_accelerations(
                next_position=features["next_position"],
                position_sequence_noise=sampled_noise,
                position_sequence=features["position"],
                n_particles_per_trajectory=features["n_particles_per_trajectory"],
                particle_types=features["particle_type"],
                pbc=features["pbc"],
                u_velocity=features["u_velocity"],
                next_u_velocity=features["next_u_velocity"],
            )
        else:
            pred, target_normalized_acceleration = self.net.predict_accelerations(
                next_position=features["next_position"],
                position_sequence_noise=sampled_noise,
                position_sequence=features["position"],
                n_particles_per_trajectory=features["n_particles_per_trajectory"],
                particle_types=features["particle_type"],
                pbc=features["pbc"]
            )
        
        return pred, target_normalized_acceleration, non_kinematic_mask
    
    def model_step(self, features: Dict[str, Tensor]) -> Tensor:
        """Perform a single forward pass through the model and compute the loss for a_u and a_v."""
        # Determine the number of pushforward steps, if applicable
        unroll_steps = 0
        if self.global_step != 0 and self.pushforward is not None:
            updated_seed, unroll_steps = push_forward_sample_steps(
                seed=self.seed, step=self.global_step, pushforward=self.pushforward
            )
            self.seed = updated_seed

        # Perform pushforward integration if unroll_steps > 0
        if unroll_steps > 0:
            # print(f"Pushing forward {unroll_steps} steps!!!")
            for _ in range(unroll_steps):
                pred, target, non_kinematic_mask = self.forward(features)
                next_pos = integrate(
                    normalized_acceleration=pred,
                    position_sequence=features["position"],
                    normalization_stats=features["normalization_stats"]["v_acceleration"],
                    boundaries=features["boundaries"],
                )
                features["position"] = torch.cat(
                    [features["position"][:, 1:], next_pos[:, None, :]], dim=1
                )
        else:
            # Forward pass
            # pred and target should be tuples: (a_u_pred, a_v_pred), (a_u_target, a_v_target)
            pred, target, non_kinematic_mask = self.forward(features)

        def particle_mse(pred, target, non_kinematic_mask):
            loss = (pred - target) ** 2
            loss = loss.sum(dim=-1)
            num_non_kinematic = non_kinematic_mask.sum()
            loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
            loss = loss.sum() / num_non_kinematic
            return loss

        if self.alpha_u == 0.0:
            # Calculate loss
            loss = particle_mse(pred, target, non_kinematic_mask)
        else:
            # Split predictions and targets into a_u and a_v
            a_u_pred, a_v_pred = pred
            a_u_target, a_v_target = target

            # Calculate MSE loss for a_u
            loss_u = particle_mse(a_u_pred, a_u_target, non_kinematic_mask)
            # Calculate MSE loss for a_v
            loss_v = particle_mse(a_v_pred, a_v_target, non_kinematic_mask)

            # Weighted combined loss
            loss = self.alpha_u * loss_u + self.alpha_v * loss_v

        return loss

    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Evaluate the model on the training batch 
        features = {
            "next_position": batch.target_pos,
            "position": batch.enc_pos,
            "n_particles_per_trajectory": batch.n_particles_per_trajectory,
            "particle_type": batch.particle_type,
            "pbc": self.pbc,
            "normalization_stats": self.net.normalization_stats,
            "boundaries": self.net._boundaries,         
        }
        if self.alpha_u != 0.0:
            features["u_velocity"] = batch.enc_u
            features["next_u_velocity"] = batch.target_u
        loss = self.model_step(features)
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, batch_size=batch.batch_size)   
        return loss 
    
    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """Perform a single validation step, using the forward method to infer positions."""
        #ROLLOUT EVALUATION
        # Evaluate validation loss, meaning a N-Step rollout
        # TODO: add further metrics, e.g., kinetic energy MSE.
        if self.alpha_u == 0.0:
            loss = eval_rollout(batch, self.net, self.net.metadata, self.num_rollout_steps, self.num_eval_steps, self.pbc, batch.batch_size)
            self.log("val/loss", loss, prog_bar=True, batch_size=batch.batch_size)
        else:
            position_loss, u_vel_loss = kolm_eval_rollout(batch, 
                                                        self.net,
                                                        self.num_rollout_steps, 
                                                        self.num_eval_steps, 
                                                        self.pbc,
                                                        vel_solver=self.vel_solver)
            
            self.log("val/postion_loss", position_loss, prog_bar=True, batch_size=batch.batch_size)
            self.log("val/u_velocity_loss", u_vel_loss, prog_bar=True, batch_size=batch.batch_size)
            self.log("val/loss", position_loss + u_vel_loss, prog_bar=True, batch_size=batch.batch_size)
            return position_loss, u_vel_loss

    def test_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """Perform a single test step, using the forward method to infer positions."""
        # Evaluate the trajectory rollout
        if self.alpha_u == 0.0:
            # if batch_idx >= self.num_eval_steps:
            #     return None  # Skip batches beyond num_eval_steps
            batch_idx = 0 
            loss = eval_rollout(batch, self.net, self.net.metadata, self.num_rollout_steps, self.num_eval_steps, self.pbc, batch.batch_size)
            self.log("test/loss", loss, prog_bar=True, batch_size=batch.batch_size)
            batch_idx += 1
        else:
            position_loss, u_vel_loss = kolm_eval_rollout(batch, 
                                                            self.net,
                                                            self.num_rollout_steps, 
                                                            self.num_eval_steps, 
                                                            self.pbc,
                                                            vel_solver=self.vel_solver)
                
            self.log("test/postion_loss", position_loss, prog_bar=True, batch_size=batch.batch_size)
            self.log("test/u_velocity_loss", u_vel_loss, prog_bar=True, batch_size=batch.batch_size)

    def on_fit_start(self) -> None:
        self.net.set_metadata_device(self.net._device)
        
    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        print(f"Current epoch learning rate: {lr}") 
        return lr
        
    def on_test_start(self) -> None:
        self.net.set_metadata_device(self.net._device)

    def setup(self, stage: str) -> None:
        """Setup model for training or evaluation."""

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        # This overengineered solution is needed if we want to increment the learning
        # rate at every gradient descent step, while specifying the learning rate
        # schedule in terms of epochs.
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            if (type(self.hparams.scheduler) is partial) and (
                self.hparams.scheduler.func.__name__ in [
                    "LinearWarmupCosineAnnealingLR", "ExpDecayLR"
                ]
            ):
                interval = "step"
            else:
                interval = "epoch"
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    