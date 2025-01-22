#src.models.gns_module.py
from typing import Any, Dict, Tuple
from functools import partial
import torch
from torch import Tensor
from lightning import LightningModule
from src.models.components.gns import get_random_walk_noise_for_position_sequence
from src.utils.train_utils import kolm_eval_rollout

class GNSLitModule(LightningModule):
    """A LightningModule for training a Graph Neural Simulator (GNS) model."""
    
    def __init__(
        self,
        net: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        noise_std: float = 3e-4,
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
        self.noise_std = noise_std
        self.pbc = pbc
        
        # Push forward configuration
        self.seed = seed
        
        # Number of eval steps
        self.num_rollout_steps = num_rollout_steps
        self.num_eval_steps = num_eval_steps
        self.vel_solver = vel_solver
        
        # Loss weights
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        
    def forward(
        self,
        features: Dict[str, Tensor],
    ) -> Tensor:
        sampled_noise = get_random_walk_noise_for_position_sequence(
            features["position"], 
            boundaries=self.net._boundaries,
            noise_std_last_step=self.noise_std,
            ).to(self.device)
        
        non_kinematic_mask = (features["particle_type"] != 3).clone().detach()
        sampled_noise *= non_kinematic_mask.view(-1, 1, 1)
        if not self.training:
            sampled_noise *= 0.0

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
        
        return pred, target_normalized_acceleration, non_kinematic_mask
    
    def model_step(self, features: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform a single forward pass through the model and compute the loss for a_u and a_v."""
        # Forward pass
        pred, target, non_kinematic_mask = self.forward(features)  # pred and target should be tuples: (a_u_pred, a_v_pred), (a_u_target, a_v_target)

        # Split predictions and targets into a_u and a_v
        a_u_pred, a_v_pred = pred
        a_u_target, a_v_target = target

        # Calculate MSE loss for a_u
        loss_u = (a_u_pred - a_u_target) ** 2
        loss_u = loss_u.sum(dim=-1)
        num_non_kinematic = non_kinematic_mask.sum()
        loss_u = torch.where(non_kinematic_mask.bool(), loss_u, torch.zeros_like(loss_u))
        loss_u = loss_u.sum() / num_non_kinematic

        # Calculate MSE loss for a_v
        loss_v = (a_v_pred - a_v_target) ** 2
        loss_v = loss_v.sum(dim=-1)
        loss_v = torch.where(non_kinematic_mask.bool(), loss_v, torch.zeros_like(loss_v))
        loss_v = loss_v.sum() / num_non_kinematic

        # Weighted combined loss
        total_loss = self.alpha_u * loss_u + self.alpha_v * loss_v

        return total_loss

    
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
            "u_velocity": batch.enc_u,
            "next_u_velocity": batch.target_u,
          
        }
        loss = self.model_step(features)
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, batch_size=batch.batch_size)   
        return loss 
    
    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """Perform a single validation step, using the forward method to infer positions."""
        #ROLLOUT EVALUATION
        # Evaluate validation loss, meaning a N-Step rollout
        # loss = kolm_eval_rollout(batch, self.net, self.net.metadata, self.num_rollout_steps, self.num_eval_steps, self.pbc, batch.batch_size)
        
        position_loss, u_vel_loss = kolm_eval_rollout(batch, 
                                                      self.net,
                                                      self.num_rollout_steps, 
                                                      self.num_eval_steps, 
                                                      self.pbc,
                                                      vel_solver=self.vel_solver)
        
        self.log("val/postion_loss", position_loss, prog_bar=True, batch_size=batch.batch_size)
        self.log("val/u_velocity_loss", u_vel_loss, prog_bar=True, batch_size=batch.batch_size)
        return position_loss, u_vel_loss

    def test_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """Perform a single test step, using the forward method to infer positions."""
        # Evaluate the trajectory rollout
        position_loss, u_vel_loss = kolm_eval_rollout(batch, 
                                                        self.net,
                                                        self.num_rollout_steps, 
                                                        self.num_eval_steps, 
                                                        self.pbc,
                                                        vel_solver=self.vel_solver)
            
        self.log("test/postion_loss", position_loss, prog_bar=True, batch_size=batch.batch_size)
        self.log("test/u_velocity_loss", u_vel_loss, prog_bar=True, batch_size=batch.batch_size)
        return position_loss, u_vel_loss
         
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
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            if (type(self.hparams.scheduler) is partial) and (
                self.hparams.scheduler.func.__name__ == "LinearWarmupCosineAnnealingLR"
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
    
    