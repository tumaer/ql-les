from typing import Any, Dict, List, Tuple

import os
import hydra
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

from src.utils.eval_utils import update_wandb_id

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path, "Checkpoint path must be specified in the config!"

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Replace the wandb id in the config file
    update_wandb_id(cfg, logger)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    checkpoint = torch.load(
        cfg.ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    # Compiled models are saved with the prefix "net._orig_mod." in the state_dict keys
    updated_state_dict = {
        k.replace("net._orig_mod.", "net."): v for k, v in checkpoint["state_dict"].items()
    }

    model.load_state_dict(updated_state_dict, strict=False)
    log.info(f"Loaded checkpoint from {cfg.ckpt_path}")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(
        model=model, datamodule=datamodule, ckpt_path=None
    )  # ckpt_path=None since we manually loaded weights

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


def main():
    """Main entry point for evaluation."""

    # Capture CLI overrides
    cli_overrides = OmegaConf.from_cli()
    assert "ckpt_path" in cli_overrides, (
        "You must specify the checkpoint path with `ckpt_path=<path_to_ckpt>`"
    )

    # print(f"Captured CLI Overrides: {OmegaConf.to_yaml(cli_overrides)}")  # Debugging info

    if "experiment" not in cli_overrides:
        ckpt_path = cli_overrides["ckpt_path"]
        ckpt_root = os.path.dirname(os.path.dirname(ckpt_path))

        # Checkpoint configs path
        ckpt_cfgs = os.path.join(ckpt_root, ".hydra")

        # Extract the experiment name from overrides.yaml
        overrides_path = os.path.join(ckpt_cfgs, "overrides.yaml")
        with open(overrides_path, "r") as f:
            overrides = f.readlines()

        # Base cfg path, used to generate the checkpoint
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_cfgs, "config.yaml"))

        # Hardcode the task name to eval
        ckpt_cfg["task_name"] = "eval"
        ckpt_cfg.pop("train")  # or set to False
        ckpt_cfg.pop("test")  # or set to True

        experiment = next(
            (line.split("=")[1].strip() for line in overrides if "experiment=" in line)
        )
        if not experiment:
            raise ValueError("Experiment name not found in overrides.yaml")

        # First load the default config to the experiment used to generate the checkpoint.
        # Then merge the checkpoint config and CLI overrides.
        hydra.initialize(config_path="../configs", version_base="1.3")
        experiment_cfg = hydra.compose(
            config_name="eval_default.yaml",
            overrides=[f"experiment={experiment}"],
        )
        OmegaConf.set_struct(experiment_cfg, False)  # allow adding new keys
        cfg = OmegaConf.merge(experiment_cfg, ckpt_cfg, cli_overrides)

        # record logs during testing
        try:
            test_logs = os.path.join(cli_overrides.model.visualize.vis_test.rollout_dir, "logs")
        except Exception:
            test_logs = os.path.join(ckpt_root, "rollout_logs")
        os.makedirs(test_logs, exist_ok=True)
        # The following two depend on the hydra run path, which we do not have
        cfg.paths.output_dir = test_logs
        cfg.paths.work_dir = test_logs
        cfg.paths.log_dir = test_logs
    else:
        raise NotImplementedError(
            "'experiment' cannot be specified! It is inferred from the checkpoint."
        )

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
