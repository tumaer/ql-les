import torch
import numpy as np
from torch_scatter import scatter_add
from src.utils.nbrs_utils import displ_fn, nearest


class XsqinvKernel:
    """The 1/x^2 kernel function from PointNet++."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h
        self._normalized_cutoff = 3.0  # arbitrarily chosen
        self.cutoff = self._normalized_cutoff * h

    def w(self, r):
        res = 1 / (r * self._one_over_h) ** 2
        # if any r is zero, set corresponding res to a very large number
        if torch.any(r == 0):
            res[r == 0] = 10**10
        return res


class QuinticKernel:
    """The quintic kernel function of Morris."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h

        self._normalized_cutoff = 3.0
        self.cutoff = self._normalized_cutoff * h
        if dim == 1:
            self._sigma = 1.0 / 120.0 * self._one_over_h
        elif dim == 2:
            self._sigma = 7.0 / 478.0 / np.pi * self._one_over_h**2
        elif dim == 3:
            self._sigma = 3.0 / 359.0 / np.pi * self._one_over_h**3

    def w(self, r):
        q = r * self._one_over_h
        q1 = torch.clamp(1.0 - q, min=0.0)
        q2 = torch.clamp(2.0 - q, min=0.0)
        q3 = torch.clamp(3.0 - q, min=0.0)

        return self._sigma * (q3**5 - 6.0 * q2**5 + 15.0 * q1**5)

    def grad_w(self, r):
        """Evaluates the 1D kernel gradient at the radial distance r."""
        q = r * self._one_over_h
        q1 = torch.clamp(1.0 - q, min=0.0)
        q2 = torch.clamp(2.0 - q, min=0.0)
        q3 = torch.clamp(3.0 - q, min=0.0)

        grad_w_q = self._sigma * (-5.0 * (q3**4 - 6.0 * q2**4 + 15.0 * q1**4))
        grad_w_r = grad_w_q * self._one_over_h

        return grad_w_r


class Interpolator(torch.nn.Module):
    """Base class for interpolators."""

    def __init__(
        self,
        is_periodic,
        domain_size,
        dim,
        dx,
        condition,  # "radius" or "knn"
        k=None,
        cutoff_factor=3,  # for Quintic
        kernel="quintic",  # "xsqinv" or "quintic"
    ):
        super().__init__()

        # Dataset metadata
        self.is_periodic = is_periodic  # e.g. True
        self.register_buffer("domain_size", torch.tensor(domain_size))  # e.g. [1.0, 1.0]
        self.dim = dim  # e.g. 2

        # Neighbors search algorithm
        self.condition = condition
        if self.condition == "radius":
            assert cutoff_factor is not None, "cutoff must be provided for radius condition"
        elif self.condition == "knn":
            assert k is not None, "k must be provided for knn condition"
        self.k = k
        self.cutoff = cutoff_factor * dx

        if kernel == "xsqinv":
            self.kernel_fn = XsqinvKernel(dx)
        elif kernel == "quintic":
            self.kernel_fn = QuinticKernel(self.cutoff / 3, dim=self.dim)

    def displ_fn(self, r1, r2):
        return displ_fn(r1, r2, self.domain_size, self.is_periodic)

    def __call__(self, r, r_target, f, npptr=None, npptr_target=None):
        """Shepard interpolation between two point clouds.

        Args:
            r (torch.Tensor): Positions of the source point cloud (shape: (N, D)).
            r_target (torch.Tensor): Positions of the target point cloud (shape: (M, D)).
            f (torch.Tensor): Features of the source point cloud (shape: (N, F)).
            npptr (torch.Tensor, optional): number of particles per trajectory (shape: (B,)).
            npptr_target (torch.Tensor, optional): as `npptr`, but for target point cloud.

        Returns:
            torch.Tensor: Interpolated features at the target point cloud (shape: (M, F)).
        """
        # Compute distances
        # TODO: works only for batch size 1 for now

        # assert that if npptr is not None, then also npptr_target is not None, and vice versa
        assert (npptr is None) == (npptr_target is None), (
            "Either both or none of npptr and npptr_target must be provided"
        )

        if npptr is None:
            npptr = torch.tensor([r.shape[0]], dtype=torch.int64, device=r.device)
        else:
            assert npptr.shape[0] == npptr_target.shape[0], "len(npptr) must be len(npptr_target)"
        if npptr_target is None:
            npptr_target = torch.tensor([r_target.shape[0]], dtype=torch.int64, device=r.device)

        edge_index = nearest(
            r,
            npptr,
            self.is_periodic,
            self.domain_size,
            condition=self.condition,
            k=self.k,
            cutoff=self.cutoff,
            query=r_target,
            n_pptr_query=npptr_target,
        )
        i_s, j_s = edge_index
        r_i, r_j = r_target[i_s], r[j_s]
        dr_ij = self.displ_fn(r_i, r_j)
        dist = torch.norm(dr_ij, dim=-1)

        # Compute weights
        w_dist = self.kernel_fn.w(dist)
        w_dist_sum = scatter_add(w_dist, i_s, dim=0, dim_size=len(r_target))

        # Interpolate
        num_targets = r_target.shape[0]  # Shape (M, D)
        f_interp = scatter_add(
            w_dist[:, None] * f[j_s], i_s, dim=0, dim_size=num_targets
        )  # TODO: check j_s
        f_interp /= w_dist_sum[:, None]
        # assert no nan or inf numbers
        assert torch.all(torch.isfinite(f_interp)), "Interpolation resulted in NaN or Inf values"
        assert torch.all(w_dist_sum > 0), "Interpolation resulted in zero weights"

        return f_interp


if __name__ == "__main__":
    # Parameters
    condition = "radius"  # "radius" or "knn"
    kernel = "quintic"  # "xsqinv" or "quintic"
    k, cutoff_factor = 10, 3.0

    Nx, L, dim = 10, 1, 2
    N, dx = Nx**dim, L / Nx
    box_size = np.ones(dim, dtype=np.float32) * L

    import random
    import matplotlib.pyplot as plt
    from src.utils.nbrs_utils import pos_init_cartesian_2d, shift_fn

    random.seed(0)
    np.random.seed(0)
    noise = torch.tensor(np.random.normal(0, dx / 5, (N, dim)), dtype=torch.float32)
    noise[0] = 0  # don't move the first particle

    r_target = torch.tensor(pos_init_cartesian_2d(box_size, dx), dtype=torch.float32)
    r = shift_fn(r_target, noise, box_size, pbc=True)
    # Taylor-Green vortex velocity field
    f = torch.stack(
        [
            torch.cos(2.0 * np.pi * r[:, 0]) * torch.sin(2.0 * np.pi * r[:, 1]),
            torch.sin(2.0 * np.pi * r[:, 0]) * torch.cos(2.0 * np.pi * r[:, 1]),
        ]
    ).T
    f[r[:, 0] > 0.5] = 1  # set right half to a constant

    def plt_scatter(ax, i, r, f, r_target, f_target, circle_radius, pbc):
        """Helper function."""
        # plot a grid where the query particles are
        for k in np.arange(dx / 2, L, dx):
            ax.axhline(k, color="gray", lw=0.5, zorder=0)
            ax.axvline(k, color="gray", lw=0.5, zorder=0)

        ax.scatter(r[:, 0], r[:, 1], c=f, s=40, marker="x")
        ax.scatter(r_target[:, 0], r_target[:, 1], c=f_target, s=20, marker="+")
        # draw a circle around particle i with cutoff radius
        pos_i = r_target[i]
        for s in [[0, 0], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            center = (pos_i[0] + s[0] * box_size[0], pos_i[1] + s[1] * box_size[1])
            circle = plt.Circle(center, circle_radius, color="r", fill=False)
            ax.add_artist(circle)
            if not pbc:
                break
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_title("PBC" if pbc else "no PBC")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    ind = 0  # target particle index
    for ax, pbc in zip(axs, [False, True]):
        interp = Interpolator(pbc, box_size, dim, dx, condition, k, cutoff_factor, kernel)
        f_interp = interp(r, r_target, f)
        plt_scatter(ax, ind, r, f[:, 0], r_target, f_interp[:, 0], cutoff_factor * dx, pbc=pbc)
    plt.tight_layout()
    fig.savefig(f"interp_{condition}_{kernel}.png")
    plt.close(fig)
