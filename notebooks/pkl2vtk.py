import argparse

from src.utils.eval_utils import pkl2vtk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, help="Source path to .pkl file.")
    parser.add_argument("--dst_path", type=str, help="Destination directory path.")
    args = parser.parse_args()

    pkl2vtk(args.src_path, args.dst_path)

    # python notebooks/pkl2vtk.py \
    #     --src_path="logs/rollouts/ckr4tejc/test/rollout_0000.pkl" \
    #     --dst_path="logs/rollouts/ckr4tejc/test_vtk_0"
