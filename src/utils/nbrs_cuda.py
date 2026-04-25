"""
Custom CUDA neighbor search for periodic self-radius queries.

This module is intentionally strict:
- CUDA only
- batch size 1 only (single position tensor)
- self-neighbors only (no separate query tensor)
- inputs: (positions, cutoff, box_side_length)
"""

import os

import torch

from src.utils.nbrs_cuda_ext import pbc_radius_edges_self_cuda


def radius_search_pbc_cuda_self(
    positions: torch.Tensor,
    cutoff: float,
    box_side_length: float,
) -> torch.Tensor:
    """Compute self-neighbors under periodic BCs using the custom CUDA extension.

    Args:
        positions: Tensor of shape (N, dim), dim in {2, 3}, float32, CUDA.
        cutoff: Radius cutoff (float > 0).
        box_side_length: Side length of periodic square/cube box (float > 0).

    Returns:
        edge_index: Tensor of shape (2, E) with [receivers, senders].
    """
    if not positions.is_cuda:
        raise RuntimeError("positions must be a CUDA tensor")
    if positions.dtype != torch.float32:
        raise RuntimeError("positions must be float32")
    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise RuntimeError("positions must have shape (N, 2) or (N, 3)")
    if cutoff <= 0:
        raise RuntimeError("cutoff must be > 0")
    if box_side_length <= 0:
        raise RuntimeError("box_side_length must be > 0")

    max_neighbors = int(os.environ.get("SPH_CUDA_MAX_NEIGHBORS", "1000"))
    if max_neighbors <= 0:
        raise RuntimeError("SPH_CUDA_MAX_NEIGHBORS must be > 0")

    return pbc_radius_edges_self_cuda(
        positions.contiguous(),
        float(cutoff),
        float(box_side_length),
        max_neighbors=max_neighbors,
    )


def nearest_pbc_cuda_self(
    positions: torch.Tensor,
    cutoff: float,
    box_side_length: float,
) -> torch.Tensor:
    """Alias for radius_search_pbc_cuda_self."""
    return radius_search_pbc_cuda_self(positions, cutoff, box_side_length)


def _sort_directed_edges(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Sort edge_index lexicographically by (receiver, sender)."""
    sort_key = edge_index[0].to(torch.int64) * int(n_nodes) + edge_index[1].to(torch.int64)
    sort_idx = torch.argsort(sort_key)
    return edge_index[:, sort_idx]


if __name__ == "__main__":
    import time
    from torch_geometric.nn import radius

    from src.utils.nbrs_utils import _pbc_halo_duplication_single

    print("Comparing baseline vs custom CUDA self-neighbor implementation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this benchmark")

    dtype = torch.float32

    n_particles = 30000
    ndim = 3
    box_side_length = 10.0
    cutoff = 1.0
    n_repeats = 10

    positions = torch.rand(n_particles, ndim, dtype=dtype, device=device) * box_side_length
    n_pptr = torch.tensor([n_particles], dtype=torch.long, device=device)
    box_vec = torch.tensor([box_side_length] * ndim, dtype=dtype, device=device)

    print(f"device={device}, n_particles={n_particles}, ndim={ndim}")

    def baseline_radius_pbc_halo(
        pos: torch.Tensor, radius_cutoff: float, box: torch.Tensor
    ) -> torch.Tensor:
        """Reference baseline: halo duplication + torch_geometric radius.

        This intentionally bypasses src.utils.nbrs_utils.nearest because that path
        may dispatch to the custom CUDA extension.
        """
        combined_positions, sender_index = _pbc_halo_duplication_single(pos, box, radius_cutoff)
        edge_index = radius(
            x=combined_positions,
            y=pos,
            r=radius_cutoff,
            max_num_neighbors=1000,
        )
        edge_index[1] = sender_index[edge_index[1]]
        return edge_index

    # Warm up both implementations.
    _ = baseline_radius_pbc_halo(positions, cutoff, box_vec)
    _ = radius_search_pbc_cuda_self(positions, cutoff, box_side_length)

    # Time baseline.
    torch.cuda.synchronize()
    t0 = time.time()
    edge_index_baseline = None
    for _ in range(n_repeats):
        edge_index_baseline = baseline_radius_pbc_halo(positions, cutoff, box_vec)
    torch.cuda.synchronize()
    t1 = time.time()
    baseline_ms = (t1 - t0) * 1000.0 / n_repeats

    # Time custom CUDA.
    torch.cuda.synchronize()
    t2 = time.time()
    edge_index_new = None
    for _ in range(n_repeats):
        edge_index_new = radius_search_pbc_cuda_self(positions, cutoff, box_side_length)
    torch.cuda.synchronize()
    t3 = time.time()
    new_ms = (t3 - t2) * 1000.0 / n_repeats

    n_edges_baseline = edge_index_baseline.shape[1]
    n_edges_new = edge_index_new.shape[1]
    speedup = baseline_ms / max(new_ms, 1e-12)

    print(f"Baseline edges: {n_edges_baseline}, avg time: {baseline_ms:.2f} ms")
    print(f"New edges: {n_edges_new}, avg time: {new_ms:.2f} ms")
    print(f"Nbrs per particle: {n_edges_new / n_particles:.2f}")
    print(f"Speedup (baseline/new): {speedup:.2f}x")

    # Validate correctness by sorting both directed edge lists and comparing.
    print("\n--- Correctness Validation ---")

    edges_baseline_sorted = _sort_directed_edges(edge_index_baseline, n_particles)
    edges_new_sorted = _sort_directed_edges(edge_index_new, n_particles)

    # Compare.
    if torch.equal(edges_baseline_sorted, edges_new_sorted):
        print("OK: edge lists match exactly after sorting")
    else:
        if edges_baseline_sorted.shape[1] != edges_new_sorted.shape[1]:
            print("MISMATCH: different number of edges")
            print(
                f"  Baseline: {edges_baseline_sorted.shape[1]}, New: {edges_new_sorted.shape[1]}"
            )
        else:
            matches = torch.all(edges_baseline_sorted == edges_new_sorted, dim=0)
            n_mismatch = (~matches).sum().item()
            print(f"MISMATCH: {n_mismatch} edges differ out of {matches.shape[0]}")
            if n_mismatch > 0:
                mismatch_idx = torch.where(~matches)[0][: min(5, n_mismatch)]
                for idx in mismatch_idx:
                    b = edges_baseline_sorted[:, idx]
                    n = edges_new_sorted[:, idx]
                    print(f"  Edge {idx}: baseline ({b[0]}, {b[1]}) vs new ({n[0]}, {n[1]})")
