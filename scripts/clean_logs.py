"""Script for cleaning up local logs by removing runs that are not present in the WANDB API."""

import argparse
import os
import shutil
from tqdm import tqdm
from wandb.apis.public import Api
from datetime import datetime


def delete_local_runs_not_on_wandb(from_date: int, entity: str, local_dir: str, dry_run: bool):
    """
    Delete local runs that are not present in the WANDB API.

    Args:
        api (Api): The WANDB API object.
        project (str): The name of the WANDB project.
        entity (str): The WANDB entity (username or team name).
        local_dir (str): The local directory containing the runs.
    """
    ### Get the runs from the WANDB API
    api = Api()
    format = "%Y-%m-%d"
    from_date = datetime.strptime(from_date, format)
    # list all projects within the same entity
    projects = api.projects(entity=entity)
    # discard projects inactive since from_date
    projects = [p for p in projects if datetime.strptime(p.created_at[:10], format) > from_date]
    # get the list of runs from all these projects on the WANDB API
    api_dirs = set()
    for p in projects:
        # Get the list of runs from the WANDB API
        runs = api.runs(path=f"{entity}/{p.name}")
        for run in runs:
            try:
                # For each of these runs, extract their trainer.default_root_dir
                api_dirs.update({run.config["trainer"]["default_root_dir"]})
            except Exception as e:
                print(f"Error processing project {p.name}: {e}")
                continue
    print(f"Found {len(api_dirs)} runs on WANDB API.")

    ### Get a list of the local runs
    local_runs = []
    for task in ["debug/runs", "train/runs"]:
        dir = os.path.abspath(os.path.join(local_dir, task))
        if os.path.exists(dir):
            local_runs.extend([os.path.join(dir, d) for d in os.listdir(dir)])
    local_runs.sort()

    # Delete local runs that are not present in the WANDB API
    for local_dir in tqdm(local_runs, desc="Deleting local runs not on WANDB"):
        if local_dir not in api_dirs:
            print(f"Deleting: {local_dir}")
            if not dry_run:
                # Delete the local run
                shutil.rmtree(local_dir)
    print("Finished deleting local runs not on WANDB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete local runs not present in WANDB.")
    parser.add_argument("--from_date", type=str, default="2025-01-01")
    parser.add_argument("--entity", type=str, default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--local_dir", type=str, default="logs/")
    parser.add_argument("--dry_run", action="store_true", help="If set, do not delete any files.")
    args = parser.parse_args()

    delete_local_runs_not_on_wandb(args.from_date, args.entity, args.local_dir, args.dry_run)

    # Dry run:
    # python scripts/clean_logs.py --dry_run
