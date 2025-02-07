#src.models.gns_module.py
from typing import Any, Dict, Tuple
from functools import partial
import torch
from torch import Tensor
from lightning import LightningModule
from src.models.components.gns import get_random_walk_noise_for_position_sequence
from src.utils.train_utils import (
    pushforward_sample_steps, pushforward_fn, eval_rollout, particle_mse
)

class GNSLitModule(LightningModule):
    """A LightningModule for training a Graph Network Simulator (GNS) model."""
    
    def __init__(
        self,
        net: torch.nn.Module,
        accelerator: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        visualize: Dict[str, Dict[str, Any]] = None,
        compile: bool = False,
        seed: int = 0,
        pushforward: Dict[str, Any] = None,
        num_rollout_steps: int = 1,
        pbc: bool = True,
        vel_solver: str = "simple",
        alpha_u: float = 1.0,
        alpha_v: float = 1.0,
        active_metrics: Dict[str, Any] = None,
        
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
        self.net = net(device="cuda" if accelerator == "gpu" else "cpu")
        self.pbc = pbc
        
        # Loss weights
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v

        if (alpha_u != 0.0) and ("KOLM" not in self.net._case):
            raise NotImplementedError(
                "Alpha_u > 0.0 is only implemented for the Kolmogorov dataset."
            )
        # Push forward configuration
        self.seed = seed
        self.pushforward = pushforward
        if (self.pushforward is not None) and (self.alpha_u != 0.0):
            raise NotImplementedError(
                "Pushforward is only implemented for alpha_u = 0.0, i.e. LagrangeBench setting."
            )
        
        # Number of eval steps
        self.num_rollout_steps = num_rollout_steps
        self.vel_solver = vel_solver
        self.active_metrics = active_metrics
        self.visualize = visualize
        self.trajectory_idx = 0

        
    def forward(
        self,
        features: Dict[str, Tensor],
    ) -> Tensor:
        if self.alpha_u != 0.0:  # if true, we recover LagrangeBench; TODO: think of better condition
            pred, target_normalized_acceleration = self.net.predict_accelerations(
                next_position=features["next_position"],
                position_sequence_noise=features["sampled_noise"],
                position_sequence=features["position"],
                n_particles_per_trajectory=features["n_particles_per_trajectory"],
                particle_types=features["particle_type"],
                pbc=features["pbc"],
                u_velocity=features["u_velocity"],
                next_u_velocity=features["next_u_velocity"],
                vel_solver = self.vel_solver
            )
        else:
            pred, target_normalized_acceleration = self.net.predict_accelerations(
                next_position=features["next_position"],
                position_sequence_noise=features["sampled_noise"],
                position_sequence=features["position"],
                n_particles_per_trajectory=features["n_particles_per_trajectory"],
                particle_types=features["particle_type"],
                pbc=features["pbc"]
            )
        
        return pred, target_normalized_acceleration
    
    def model_step(self, batch: Any) -> Tensor:
        """Perform a single forward pass through the model and compute the loss for a_u and a_v."""
        # Determine the number of pushforward steps, if applicable
        unroll_steps = 0
        if self.global_step != 0 and self.pushforward is not None:
            updated_seed, unroll_steps = pushforward_sample_steps(
                seed=self.seed, step=self.global_step, pushforward=self.pushforward
            )
            self.seed = updated_seed
        #Random walk noise
        sampled_noise = get_random_walk_noise_for_position_sequence(
            position_sequence=batch.enc_pos, 
            boundaries=self.net._boundaries,
            noise_std_last_step=self.net.noise_std,
            ).to(self.device)
        
        non_kinematic_mask = (batch.particle_type != 3).clone().detach()
        sampled_noise *= non_kinematic_mask.view(-1, 1, 1)
        if not self.training:
            sampled_noise *= 0.0
        
        # Prepare features for the forward pass
        features = {
            "next_position": batch.target_pos[:, 0],  # Only the first target position
            "position": batch.enc_pos,
            "n_particles_per_trajectory": batch.n_particles_per_trajectory,
            "particle_type": batch.particle_type,
            "pbc": self.pbc,
            "normalization_stats": self.net.normalization_stats,
            "boundaries": self.net._boundaries,
            "sampled_noise": sampled_noise
        }
        if self.alpha_u != 0.0:
            features["u_velocity"] = batch.enc_u
            features["next_u_velocity"] = batch.target_u

        # Perform pushforward if unroll_steps > 0
        if unroll_steps > 0:
             # print(f"Pushing forward {unroll_steps} steps!!!")
            target_positions = batch.target_pos
            pred, target = pushforward_fn(features, target_positions, unroll_steps, self.forward)
        else:
            # Forward pass
            # pred and target should be tuples: (a_u_pred, a_v_pred), (a_u_target, a_v_target)
            pred, target = self.forward(features)
            
        if self.alpha_u != 0.0:
            # Split predictions and targets into a_u and a_v
            a_v_pred, a_u_pred = pred
            a_v_target, a_u_target = target

            # Calculate MSE loss
            loss_v = particle_mse(a_v_pred, a_v_target, non_kinematic_mask)
            loss_u = particle_mse(a_u_pred, a_u_target, non_kinematic_mask)

            # Weighted combined loss
            loss = self.alpha_v * loss_v + self.alpha_u * loss_u
        else:
            # Calculate loss
            loss = particle_mse(pred, target, non_kinematic_mask)

        return loss

    
    def training_step(self, batch: Any) -> torch.Tensor:
        # Evaluate the model on the training batch and calculate the loss
        loss = self.model_step(batch)
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, batch_size=batch.batch_size, on_epoch=True)   
        return loss 
    
    def on_validation_start(self) -> None:
        self.trajectory_idx = 0
    
    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """Perform a single validation step, using the forward method to infer positions."""
        #ROLLOUT EVALUATION
        # Evaluate validation loss, meaning a N-Step rollout
       
                   
        if self.alpha_u != 0.0:
            position_loss, u_vel_loss = eval_rollout(batch=batch,
                                                        simulator=self.net,
                                                        metadata=self.net.metadata, 
                                                        num_rollout_steps=self.num_rollout_steps,
                                                        pbc=self.pbc,
                                                        vel_solver=self.vel_solver,
                                                        device=self.net._device,
                                                        active_metrics=self.active_metrics,
                                                        u_vel=True,
                                                        vis_config=self.visualize["vis_val"],
                                                        trajectory_idx=self.trajectory_idx)
            
            self.log("val/postion_loss", position_loss["mse"].mean(), prog_bar=True, on_epoch=True, batch_size=batch.batch_size)
            self.log("val/u_velocity_loss", u_vel_loss.mean(), prog_bar=True, on_epoch=True, batch_size=batch.batch_size)
            self.log("val/loss", position_loss["mse"].mean() + u_vel_loss.mean(), prog_bar=True, on_epoch=True, batch_size=batch.batch_size)
            self.log("val/loss_ekin", position_loss["e_kin"]["mse"].mean(), prog_bar=True, on_epoch=True, batch_size=batch.batch_size) #only shifting velocity currently
        else:
            loss = eval_rollout(batch=batch,
                                simulator=self.net, 
                                metadata=self.net.metadata, 
                                num_rollout_steps=self.num_rollout_steps,
                                pbc=self.pbc, 
                                device=self.net._device,
                                active_metrics=self.active_metrics,
                                u_vel=False,
                                vis_config=self.visualize["vis_val"],
                                trajectory_idx=self.trajectory_idx)

            #MSE
            for k in ["mse", "mse1", "mse5", "mse10"]:  #, "mse20", "mse50", "mse100"]:
                self.log(f"val/{k}", loss[k].mean(), prog_bar=True, on_epoch=True, batch_size=batch.batch_size)
                self.log(f"val/{k}std", loss[k].std(), prog_bar=True, on_epoch=True, batch_size=batch.batch_size)

            self.log("val/loss", loss["mse"].mean(), prog_bar=True, on_epoch=True, batch_size=batch.batch_size)
            
            #MAE
            
            #E_KIN
            self.log("val/loss_ekin", loss["e_kin"]["mse"].mean(), prog_bar=True, on_epoch=True, batch_size=batch.batch_size)     

        self.trajectory_idx += 1
        
    def on_test_epoch_start(self):
        self.trajectory_idx = 0

    def test_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """Perform a single test step, using the forward method to infer positions."""
        # Evaluate the trajectory rollout
        if self.alpha_u != 0.0:
            position_loss, u_vel_loss = eval_rollout(batch=batch,
                                                        simulator=self.net,
                                                        metadata=self.net.metadata, 
                                                        num_rollout_steps=self.num_rollout_steps, 
                                                        pbc=self.pbc,
                                                        vel_solver=self.vel_solver,
                                                        device=self.net._device,
                                                        active_metrics=self.active_metrics,
                                                        u_vel=True,
                                                        vis_config=self.visualize["vis_test"],
                                                        trajectory_idx=self.trajectory_idx)
                
            self.log("test/postion_loss", position_loss["mse"].mean(), prog_bar=True, on_step=True, batch_size=batch.batch_size)
            self.log("test/u_velocity_loss", u_vel_loss.mean(), prog_bar=True, on_step=True, batch_size=batch.batch_size)
            self.log("test/loss", position_loss["mse"].mean() + u_vel_loss.mean(), prog_bar=True, on_step=True, batch_size=batch.batch_size)
            self.log("test/loss_ekin", position_loss["e_kin"]["mse"].mean(), prog_bar=True, on_step=True, batch_size=batch.batch_size) #only shifting velocity currently
        else:
            loss = eval_rollout(batch=batch,
                                simulator=self.net, 
                                metadata=self.net.metadata, 
                                num_rollout_steps=self.num_rollout_steps,
                                pbc=self.pbc, 
                                device=self.net._device,
                                active_metrics=self.active_metrics,
                                u_vel=False,
                                vis_config=self.visualize["vis_test"],
                                trajectory_idx=self.trajectory_idx)
            self.log("test/loss", loss["mse"].mean(), prog_bar=True, on_step=True, batch_size=batch.batch_size)
        
        self.trajectory_idx += 1
        
    def on_fit_start(self) -> None:
        self.net.set_metadata_device(self.net._device)    
    
    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
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
    
    
    