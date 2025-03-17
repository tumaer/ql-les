#src.models.gns_module.py
from typing import Any, Dict, Tuple
from functools import partial
import time
import pickle
import os

import torch
from torch import Tensor
from lightning import LightningModule
from src.models.components.gns import get_random_walk_noise_for_position_sequence
from src.utils.train_utils import pushforward_sample_steps, pushforward_fn
from src.utils.metrics import particle_mse
from src.utils.eval_utils import eval_rollout

class SimulatorLitModule(LightningModule):
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
        neuralsph: Dict[str, Any] = None,
        num_rollout_steps: int = 1,
        vel_solver: str = "simple",
        alpha_u: float = 1.0,
        alpha_v: float = 1.0,
        active_metrics: Dict[str, Any] = None,
        metric_space: Dict[str, str] = "norm",
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
        self.neuralsph = neuralsph

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
        self.metric_space = metric_space
        self.visualize = visualize
        self.trajectory_idx = 0

    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        return self.net.predict_accelerations(**features)
    
    def model_step(self, batch: Any) -> Tensor:
        """Forward pass through the model and compute the loss for a_u and a_v."""
        # Determine the number of pushforward steps, if applicable
        unroll_steps = 0
        if self.global_step != 0 and self.pushforward is not None:
            updated_seed, unroll_steps = pushforward_sample_steps(
                seed=self.seed, step=self.global_step, pushforward=self.pushforward
            )
            self.seed = updated_seed
        #Random walk noise
        position_sequence_noise = get_random_walk_noise_for_position_sequence(
            position_sequence=batch.enc_pos, 
            boundaries=self.net._boundaries,
            noise_std_last_step=self.net.noise_std,
            ).to(self.device)

        non_kinematic_mask = (batch.particle_types != 3).clone().detach()
        position_sequence_noise *= non_kinematic_mask.view(-1, 1, 1)
        if not self.training:
            position_sequence_noise *= 0.0
        
        # Prepare features for the forward pass
        features = {
            "next_position": batch.target_pos[:, 0],  # Only the first target position
            "position_sequence": batch.enc_pos,
            "n_particles_per_trajectory": batch.n_particles_per_trajectory,
            "particle_types": batch.particle_types,
            "pbc": any(self.net._pbc),
            "position_sequence_noise": position_sequence_noise,
        }
        if self.pushforward is not None:
            features["normalization_stats"] = self.net.normalization_stats
            features["boundaries"] = self.net._boundaries,
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
            self.log("train/loss_u", loss_u, prog_bar=True, batch_size=batch.batch_size, on_epoch=True)   
            self.log("train/loss_v", loss_v, prog_bar=True, batch_size=batch.batch_size, on_epoch=True)   

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
        self.metrics_dump = {}
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], **kwargs) -> Dict[str, Tensor]:
        """Perform a single validation step, using the forward method to infer positions."""
        #ROLLOUT EVALUATION
        # Evaluate validation loss, meaning a N-Step rollout
        is_test = (("testing" in kwargs) and kwargs["testing"])
        split = "test" if is_test else "val"
        self.net.neuralsph=self.neuralsph[split]
        loss = eval_rollout(
            batch=batch,
            simulator=self.net, 
            metadata=self.net.metadata, 
            num_rollout_steps=self.num_rollout_steps,
            pbc=any(self.net._pbc), 
            device=self.net._device,
            active_metrics=self.active_metrics,
            vis_config=self.visualize[f"vis_{split}"],
            trajectory_idx=self.trajectory_idx,
            u_vel=True if self.alpha_u != 0.0 else False,
            metric_space=self.metric_space.test if is_test else self.metric_space.val,
        )

        kwargs_log = {"prog_bar": True, "on_epoch": True, "batch_size": batch.batch_size}
        if self.alpha_u != 0.0:
            position_loss, u_vel_loss = loss
            self.log(f"{split}/loss", position_loss["mse"].mean() + u_vel_loss.mean(), **kwargs_log)
            self.log(f"{split}/u_loss", u_vel_loss.mean(), **kwargs_log)
        else:
            position_loss = loss
            self.log(f"{split}/loss", position_loss["mse"].mean(), **kwargs_log)
            # for k in ["mse", "mse1", "mse5", "mse10"]:  #, "mse20", "mse50", "mse100"]:
            #     self.log(f"val/{k}", position_loss[k].mean(), **kwargs_log)
            #     self.log(f"val/{k}std", position_loss[k].std(), **kwargs_log)
    
        self.log(f"{split}/v_loss", position_loss["mse"].mean(), **kwargs_log)
        self.log(f"{split}/loss_ekin", position_loss["e_kin"]["mse"].mean(), **kwargs_log) #only shifting velocity currently
        if "mse_pos" in position_loss:
            self.log(f"{split}/mse_pos", position_loss["mse_pos"].mean(), **kwargs_log)

        self.metrics_dump[self.trajectory_idx] = loss
        self.trajectory_idx += 1

    def on_test_start(self):
        self.net.set_metadata_device(self.net._device)
        self.trajectory_idx = 0
        self.metrics_dump = {}

    def on_test_end(self):
        # write full metrics to file
        metrics_path = os.path.join(
            self.visualize.vis_test.rollout_dir,
            f"metrics_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}.pkl"
        )
        print(f"Writing metrics to {metrics_path}")
        with open(metrics_path, "wb") as f:
            pickle.dump(self.metrics_dump, f)


    def test_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """Perform a single test step, using the forward method to infer positions."""
        # Evaluate the trajectory rollout
        self.validation_step(batch, testing=True)
        
    def on_fit_start(self) -> None:
        self.net.set_metadata_device(self.net._device)    
    
    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        return lr

    def setup(self, stage: str) -> None:
        """Setup model for training or evaluation."""

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            print("Model compiled!")

    def configure_optimizers(self) -> Dict[str, Any]:
        # To increment the learning rate at every gradient descent step
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
    
    
    