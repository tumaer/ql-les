from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_EXT: Optional[object] = None
_LOAD_ERROR: Optional[Exception] = None


def get_nbrs_cuda_ext(verbose: bool = False):
    global _EXT, _LOAD_ERROR
    if _EXT is not None:
        return _EXT
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR

    this_dir = Path(__file__).resolve().parent
    src_dir = this_dir / "cuda"
    cpp = src_dir / "nbrs_cuda_ext.cpp"
    cu = src_dir / "nbrs_cuda_kernel.cu"

    try:
        _EXT = load(
            name="sph_les_nbrs_cuda_ext",
            sources=[str(cpp), str(cu)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=verbose,
        )
    except Exception as exc:
        msg = str(exc)
        if "Ninja is required" in msg:
            _LOAD_ERROR = RuntimeError(
                "Ninja is required to build the CUDA extension. "
                "Install it in the active environment, e.g. `pip install ninja`."
            )
        else:
            _LOAD_ERROR = exc
        raise _LOAD_ERROR

    return _EXT


def pbc_radius_edges_self_cuda(
    positions: torch.Tensor,
    cutoff: float,
    box_side_length: float,
    max_neighbors: int = 1000,
):
    ext = get_nbrs_cuda_ext()
    return ext.pbc_radius_edges_self(
        positions,
        float(cutoff),
        float(box_side_length),
        int(max_neighbors),
    )
