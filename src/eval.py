from typing import Any, Dict, List, Tuple
    
import os
import hydra
import torch
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

from src.utils.eval_utils import update_wandb_id

log = RankedLogger(__name__, rank_zero_only=True)

def resolve_experiment_defaults(experiment_cfg: DictConfig, config_path: str) -> DictConfig:
    """
    Resolves the `defaults:` section in an experiment config by loading referenced YAML files.
    :param experiment_cfg: The loaded but unresolved experiment config.
    :param config_path: The path to the Hydra config directory.
    :return: A fully resolved experiment configuration.
    """
    resolved_cfg = experiment_cfg.copy()
    
    if "defaults" in experiment_cfg:
        resolved_defaults = []
        
        for entry in experiment_cfg.defaults:
            if isinstance(entry, DictConfig) and "override /" in list(entry.keys())[0]:
                # Extract the key and filename
                key = list(entry.keys())[0]
                filename = entry[key]  # e.g., "gns"
                key = key.replace("override /", "")  # e.g., "data"
                # Construct the correct path to load the referenced YAML
                ref_config_path = os.path.join(config_path, key, f"{filename}.yaml")
                ref_config_path = os.path.normpath(ref_config_path)
                if not os.path.exists(ref_config_path):
                    raise FileNotFoundError(f"Referenced config not found: {ref_config_path}")

                # Load the referenced YAML
                ref_config = OmegaConf.load(ref_config_path)
                
                if "defaults" in ref_config:
                    # Resolve the referenced config's defaults (sub-configs)
                    for sub_entry in ref_config.defaults:
                        if "_self_" in sub_entry: # Skip self reference
                            continue      
                        sub_config_path = os.path.join(config_path, key, f"{sub_entry}.yaml")
                        if not os.path.exists(sub_config_path):
                            raise FileNotFoundError(f"Referenced config not found: {sub_config_path}")
                        sub_config = OmegaConf.load(sub_config_path)
                        ref_config = OmegaConf.merge(ref_config, sub_config)
                        
                    ref_config.pop("defaults") # Remove the defaults section after resolving
                        
                # Merge the referenced config into experiment.yaml
                resolved_cfg = OmegaConf.merge(resolved_cfg, {key: ref_config})
                
                # Store resolved default
                resolved_defaults.append({key: filename})

        # Replace `defaults:` with fully resolved ones
        resolved_cfg.defaults = resolved_defaults
        
        resolved_cfg.pop("defaults")
    
    return resolved_cfg

def load_and_resolve_experiment_cfg(experiment_name: str, config_path: str) -> DictConfig:
    """
    Loads an experiment config and resolves its `defaults:` section.
    :param experiment_name: The name of the experiment YAML file (without `.yaml`).
    :param config_path: The path to the Hydra config directory.
    :return: The fully resolved experiment configuration.
    """
    
    exp_cfg = OmegaConf.load(os.getcwd() + f"/configs/experiment/{experiment_name}")

    resolved_exp_cfg = resolve_experiment_defaults(exp_cfg, config_path)
    
    
    return resolved_exp_cfg

def merge_without_overwrite(target_cfg, source_cfg):
    """
    Recursively merges source_cfg into target_cfg,
    adding keys that are missing in target_cfg,
    and does NOT attempt to resolve interpolation.
    """
    # Convert source_cfg to a plain dict without resolution
    src_dict = OmegaConf.to_container(source_cfg, resolve=False)

    # Convert target_cfg similarly, so you can manipulate
    tgt_dict = OmegaConf.to_container(target_cfg, resolve=False)

    # Recursively merge these two *plain python* dicts
    def _merge_dicts(tgt, src):
        for k, v in src.items():
            if k not in tgt:
                tgt[k] = v
            elif isinstance(v, dict) and isinstance(tgt[k], dict):
                _merge_dicts(tgt[k], v)
            # else: do not overwrite

    _merge_dicts(tgt_dict, src_dict)

    #Build a new DictConfig from the merged python dict
    merged_cfg = OmegaConf.create(tgt_dict)
    return merged_cfg


def capture_cli_overrides() -> DictConfig:
    """Capture CLI overrides before Hydra merges them into the default config."""
    cli_overrides_raw = OmegaConf.from_cli()# Raw list of CLI overrides
    for override in cli_overrides_raw.keys():
        strip_key = str(override).strip("+")  # Remove the '+' prefix
        cli_overrides_raw[strip_key] = cli_overrides_raw.pop(override)  # Replace the key


    return cli_overrides_raw  # Returns only explicit CLI overrides

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

    #Replace the wandb id in the config file
    update_wandb_id(cfg, logger)
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    
    checkpoint = torch.load(cfg.ckpt_path, map_location= "cuda" if torch.cuda.is_available() else "cpu")
    #Compiled models are saved with the prefix "net._orig_mod." in the state_dict keys
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
    trainer.test(model=model, datamodule=datamodule, ckpt_path=None)  # ckpt_path=None since we manually loaded weights

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics
    
    return metric_dict, object_dict



@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_default.yaml")
def main(cfg: DictConfig):
    """Main entry point for evaluation."""
    
    #Capture CLI overrides
    cli_overrides = capture_cli_overrides()
 
    # print(f"Captured CLI Overrides: {OmegaConf.to_yaml(cli_overrides)}")  # Debugging info
    
    if "experiment" not in cli_overrides:
        
        ckpt_path = cfg.get("ckpt_path", None)
        if not ckpt_path:
            raise ValueError("You must specify the checkpoint path with ++ckpt_path=<path_to_ckpt>")
        
        #Checkpoint configs path
        ckpt_cfgs = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), ".hydra")
        
        #Extract the experiment name from overrides.yaml
        overrides_path = os.path.join(ckpt_cfgs, "overrides.yaml")
        with open(overrides_path, "r") as f:
            overrides = f.readlines()
            
        #Base cfg path, used to generate the checkpoint
        base_cfg_path = os.path.join(ckpt_cfgs, "config.yaml")
        base_cfg = OmegaConf.load(base_cfg_path)
        
        #Hardcode the task name to eval
        base_cfg["task_name"] = "eval"
        base_cfg.pop("train")
        base_cfg.pop("test")
        
        experiment_name = None
        for line in overrides:
            if "experiment=" in line:
                experiment_name = line.split("=")[1].strip()
                break
        if not experiment_name:
            raise ValueError("Experiment name not found in overrides.yaml")

        #Load and resolve experiment.yaml
        configs_abs_path = os.path.abspath("./configs")
        #Experiment cfg
        experiment_cfg = load_and_resolve_experiment_cfg(experiment_name, configs_abs_path)  
        
    
        final_cfg = merge_without_overwrite(base_cfg, experiment_cfg)
        
        #Merge experiment cfg with base_cfg
        final_cfg = OmegaConf.merge(final_cfg, cli_overrides)

        extras(final_cfg)

        # Continue with evaluation
        evaluate(final_cfg)
        
    else: 
        # apply extra utilities
        # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
        extras(cfg)

        evaluate(cfg)

if __name__ == "__main__":
    main()