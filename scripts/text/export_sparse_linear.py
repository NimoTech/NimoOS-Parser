"""Convert BAAI's sparse_linear.pt (torch state dict) to sparse_linear.npz.

Runs inside the conversion venv (torch available there); the runtime venv
then loads plain numpy arrays and never needs torch for the sparse head.
Usage: python export_sparse_linear.py <sparse_linear.pt> <out_dir>
"""
import sys

import numpy as np
import torch


def main() -> None:
    pt_path, out_dir = sys.argv[1], sys.argv[2]
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    weight = state["weight"].float().numpy()   # (1, 1024)
    bias = state["bias"].float().numpy()       # (1,)
    np.savez(f"{out_dir}/sparse_linear.npz", weight=weight, bias=bias)
    print(f"saved {out_dir}/sparse_linear.npz weight={weight.shape} bias={bias.shape}")


if __name__ == "__main__":
    main()
