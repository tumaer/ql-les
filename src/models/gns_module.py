#src.models.gns_module.py
from typing import Any, Dict, Tuple
from functools import partial
import torch
from torch import Tensor
from lightning import LightningModule
from src.models.components.gns import get_random_walk_noise_for_position_sequence
from src.utils.train_utils import push_forward_sample_steps, eval_rollout, integrate

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
        pushforward: Dict[str, Any] = None,
        seed: int = 0,
        num_rollout_steps: int = 1,
        num_eval_steps: int = 1,
        pbc: bool = True,
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
        self.pushforward = pushforward
        
        # Number of eval steps
        self.num_rollout_steps = num_rollout_steps
        self.num_eval_steps = num_eval_steps     
        
    def forward(
        self,
        features: Dict[str, Tensor],
        unroll_steps: int,
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
            batch_size=features["batch_size"],
            unroll_steps=unroll_steps,
        )
        
        return pred, target_normalized_acceleration, non_kinematic_mask

    def model_step(self, features: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Perform a single forward pass through the model and compute the loss."""
        # Determine the number of pushforward steps, if applicable
        unroll_steps = 0
        if self.global_step != 0 and self.pushforward is not None:
            updated_seed, unroll_steps = push_forward_sample_steps(
                seed=self.seed, step=self.global_step, pushforward=self.pushforward
            )
            self.seed = updated_seed

        # Perform pushforward integration if unroll_steps > 0
        if unroll_steps > 0:
            print(f"Pushing forward {unroll_steps} steps!!!")
            for _ in range(unroll_steps):
                pred, target, non_kinematic_mask = self.forward(features, unroll_steps)
                next_pos = integrate(
                    normalized_acceleration=pred,
                    position_sequence=features["position"],
                    normalization_stats=features["normalization_stats"],
                    boundaries=features["boundaries"],
                )
                features["position"] = torch.cat(
                    [features["position"][:, 1:], next_pos[:, None, :]], dim=1
                )
        else:
            pred, target, non_kinematic_mask = self.forward(features, unroll_steps)

        # Calculate loss
        loss = (pred - target) ** 2
        loss = loss.sum(dim=-1)
        num_non_kinematic = non_kinematic_mask.sum()
        loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
        loss = loss.sum() / num_non_kinematic

        return loss, pred
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Evaluate the model on the training batch 
        features = {
            "next_position": batch.target_pos,
            "position": batch.enc_pos,
            "n_particles_per_trajectory": batch.n_particles_per_trajectory,
            "particle_type": batch.particle_type,
            "batch_size": batch.batch_size,
            "pbc": self.pbc,
            "batch_size": batch.batch_size,
            "normalization_stats": self.net.normalization_stats,
            "boundaries": self.net._boundaries,
        }
        loss, preds = self.model_step(features)
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, batch_size=batch.batch_size)        
        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """Perform a single validation step, using the forward method to infer positions."""
        # Evaluate validation loss, meaning a N-Step rollout
        loss = eval_rollout(batch, self.net, self.net.metadata, self.num_rollout_steps, self.num_eval_steps, self.pbc, batch.batch_size)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch.batch_size)
        
    def test_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """Perform a single test step, using the forward method to infer positions."""
        # Evaluate the trajectory rollout
        # if batch_idx >= self.num_eval_steps:
        #     return None  # Skip batches beyond num_eval_steps
        batch_idx = 0 
        loss = eval_rollout(batch, self.net, self.net.metadata, self.num_rollout_steps, self.num_eval_steps, self.pbc, batch.batch_size)
        self.log("test/loss", loss, prog_bar=True, batch_size=batch.batch_size)
        batch_idx += 1
         
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