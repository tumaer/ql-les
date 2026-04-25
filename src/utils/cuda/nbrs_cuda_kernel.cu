#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__device__ __forceinline__ float wrap_disp(float d, float box) {
    if (d > 0.5f * box) {
        d -= box;
    } else if (d < -0.5f * box) {
        d += box;
    }
    return d;
}

__global__ void pbc_radius_self_count_fill_kernel_2d(
    const float* __restrict__ x,
    const int64_t* __restrict__ sorted_indices,
    const int64_t* __restrict__ cell_starts,
    const int64_t* __restrict__ cell_counts,
    const int64_t* __restrict__ n_cells_per_dim,
    int64_t n_particles,
    float box_side,
    float cutoff_sq,
    int64_t max_neighbors,
    int64_t* __restrict__ edges_tmp,
    int32_t* __restrict__ counts) {
    int64_t q = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= n_particles) {
        return;
    }

    const int64_t nx = n_cells_per_dim[0];
    const int64_t ny = n_cells_per_dim[1];

    const float qx = x[q * 2 + 0];
    const float qy = x[q * 2 + 1];

    int64_t cx = static_cast<int64_t>(floorf((qx / box_side) * static_cast<float>(nx)));
    int64_t cy = static_cast<int64_t>(floorf((qy / box_side) * static_cast<float>(ny)));

    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cx >= nx) cx = nx - 1;
    if (cy >= ny) cy = ny - 1;

    int32_t local_count = 0;
    const int64_t out_base = q * max_neighbors;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int64_t ncx = (cx + dx) % nx;
            int64_t ncy = (cy + dy) % ny;
            if (ncx < 0) ncx += nx;
            if (ncy < 0) ncy += ny;

            int64_t cell_id = ncx + ncy * nx;
            int64_t start = cell_starts[cell_id];
            int64_t count = cell_counts[cell_id];

            for (int64_t i = 0; i < count; ++i) {
                int64_t p = sorted_indices[start + i];

                float dxp = qx - x[p * 2 + 0];
                float dyp = qy - x[p * 2 + 1];

                dxp = wrap_disp(dxp, box_side);
                dyp = wrap_disp(dyp, box_side);

                float dist_sq = dxp * dxp + dyp * dyp;
                if (dist_sq <= cutoff_sq && local_count < max_neighbors) {
                    edges_tmp[out_base + local_count] = p;
                    local_count += 1;
                }
            }
        }
    }

    counts[q] = local_count;
}

__global__ void pbc_radius_self_count_fill_kernel_3d(
    const float* __restrict__ x,
    const int64_t* __restrict__ sorted_indices,
    const int64_t* __restrict__ cell_starts,
    const int64_t* __restrict__ cell_counts,
    const int64_t* __restrict__ n_cells_per_dim,
    int64_t n_particles,
    float box_side,
    float cutoff_sq,
    int64_t max_neighbors,
    int64_t* __restrict__ edges_tmp,
    int32_t* __restrict__ counts) {
    int64_t q = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= n_particles) {
        return;
    }

    const int64_t nx = n_cells_per_dim[0];
    const int64_t ny = n_cells_per_dim[1];
    const int64_t nz = n_cells_per_dim[2];

    const float qx = x[q * 3 + 0];
    const float qy = x[q * 3 + 1];
    const float qz = x[q * 3 + 2];

    int64_t cx = static_cast<int64_t>(floorf((qx / box_side) * static_cast<float>(nx)));
    int64_t cy = static_cast<int64_t>(floorf((qy / box_side) * static_cast<float>(ny)));
    int64_t cz = static_cast<int64_t>(floorf((qz / box_side) * static_cast<float>(nz)));

    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cz < 0) cz = 0;
    if (cx >= nx) cx = nx - 1;
    if (cy >= ny) cy = ny - 1;
    if (cz >= nz) cz = nz - 1;

    int32_t local_count = 0;
    const int64_t out_base = q * max_neighbors;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int64_t ncx = (cx + dx) % nx;
                int64_t ncy = (cy + dy) % ny;
                int64_t ncz = (cz + dz) % nz;
                if (ncx < 0) ncx += nx;
                if (ncy < 0) ncy += ny;
                if (ncz < 0) ncz += nz;

                int64_t cell_id = ncx + ncy * nx + ncz * nx * ny;
                int64_t start = cell_starts[cell_id];
                int64_t count = cell_counts[cell_id];

                for (int64_t i = 0; i < count; ++i) {
                    int64_t p = sorted_indices[start + i];

                    float dxp = qx - x[p * 3 + 0];
                    float dyp = qy - x[p * 3 + 1];
                    float dzp = qz - x[p * 3 + 2];

                    dxp = wrap_disp(dxp, box_side);
                    dyp = wrap_disp(dyp, box_side);
                    dzp = wrap_disp(dzp, box_side);

                    float dist_sq = dxp * dxp + dyp * dyp + dzp * dzp;
                    if (dist_sq <= cutoff_sq && local_count < max_neighbors) {
                        edges_tmp[out_base + local_count] = p;
                        local_count += 1;
                    }
                }
            }
        }
    }

    counts[q] = local_count;
}

__global__ void compact_edges_kernel(
    const int64_t* __restrict__ edges_tmp,
    const int32_t* __restrict__ counts,
    const int64_t* __restrict__ offsets,
    int64_t n_particles,
    int64_t total_edges,
    int64_t max_neighbors,
    int64_t* __restrict__ edge_index) {
    int64_t q = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= n_particles) {
        return;
    }

    int32_t c = counts[q];
    int64_t out_start = offsets[q];
    int64_t in_start = q * max_neighbors;

    for (int32_t j = 0; j < c; ++j) {
        int64_t out_col = out_start + j;
        edge_index[out_col] = q;
        edge_index[total_edges + out_col] = edges_tmp[in_start + j];
    }
}

}  // namespace

torch::Tensor pbc_radius_edges_self_cuda(
    torch::Tensor positions,
    double cutoff,
    double box_side_length,
    int64_t max_neighbors) {
    c10::cuda::CUDAGuard device_guard(positions.device());

    const int64_t n_particles = positions.size(0);
    const int64_t ndim = positions.size(1);

    if (n_particles == 0) {
        return torch::empty(
            {2, 0},
            torch::TensorOptions().dtype(torch::kInt64).device(positions.device()));
    }

    const int64_t n_cells_1d = std::max<int64_t>(
        1, static_cast<int64_t>(std::floor(box_side_length / cutoff)));

    auto n_cells_per_dim = torch::full(
        {ndim},
        n_cells_1d,
        torch::TensorOptions().dtype(torch::kInt64).device(positions.device()));

    auto scaled = (positions / static_cast<float>(box_side_length)) * static_cast<float>(n_cells_1d);
    auto cell_coords = torch::floor(scaled).to(torch::kInt64);
    cell_coords = torch::clamp(cell_coords, 0, n_cells_1d - 1);
    cell_coords = torch::remainder(cell_coords, n_cells_1d);

    torch::Tensor cell_ids;
    if (ndim == 2) {
        cell_ids = cell_coords.select(1, 0) + cell_coords.select(1, 1) * n_cells_1d;
    } else {
        cell_ids =
            cell_coords.select(1, 0) +
            cell_coords.select(1, 1) * n_cells_1d +
            cell_coords.select(1, 2) * n_cells_1d * n_cells_1d;
    }

    auto sorted_indices = torch::argsort(cell_ids);
    auto cell_ids_sorted = cell_ids.index_select(0, sorted_indices);

    int64_t n_cells_total = (ndim == 2)
        ? (n_cells_1d * n_cells_1d)
        : (n_cells_1d * n_cells_1d * n_cells_1d);

    auto cell_counts = torch::bincount(cell_ids_sorted, {}, n_cells_total);
    auto cell_starts = torch::cumsum(cell_counts, 0) - cell_counts;

    auto edges_tmp = torch::empty(
        {n_particles * max_neighbors},
        torch::TensorOptions().dtype(torch::kInt64).device(positions.device()));
    auto counts = torch::zeros(
        {n_particles},
        torch::TensorOptions().dtype(torch::kInt32).device(positions.device()));

    const int threads = 256;
    const int blocks = static_cast<int>((n_particles + threads - 1) / threads);
    const float cutoff_sq = static_cast<float>(cutoff * cutoff);
    const float box_side = static_cast<float>(box_side_length);

    if (ndim == 2) {
        pbc_radius_self_count_fill_kernel_2d<<<blocks, threads>>>(
            positions.data_ptr<float>(),
            sorted_indices.data_ptr<int64_t>(),
            cell_starts.data_ptr<int64_t>(),
            cell_counts.data_ptr<int64_t>(),
            n_cells_per_dim.data_ptr<int64_t>(),
            n_particles,
            box_side,
            cutoff_sq,
            max_neighbors,
            edges_tmp.data_ptr<int64_t>(),
            counts.data_ptr<int32_t>());
    } else {
        pbc_radius_self_count_fill_kernel_3d<<<blocks, threads>>>(
            positions.data_ptr<float>(),
            sorted_indices.data_ptr<int64_t>(),
            cell_starts.data_ptr<int64_t>(),
            cell_counts.data_ptr<int64_t>(),
            n_cells_per_dim.data_ptr<int64_t>(),
            n_particles,
            box_side,
            cutoff_sq,
            max_neighbors,
            edges_tmp.data_ptr<int64_t>(),
            counts.data_ptr<int32_t>());
    }

    auto counts_long = counts.to(torch::kInt64);
    auto offsets = torch::cumsum(counts_long, 0) - counts_long;
    int64_t total_edges = counts_long.sum().item<int64_t>();

    auto edge_index = torch::empty(
        {2, total_edges},
        torch::TensorOptions().dtype(torch::kInt64).device(positions.device()));

    if (total_edges == 0) {
        return edge_index;
    }

    compact_edges_kernel<<<blocks, threads>>>(
        edges_tmp.data_ptr<int64_t>(),
        counts.data_ptr<int32_t>(),
        offsets.data_ptr<int64_t>(),
        n_particles,
        total_edges,
        max_neighbors,
        edge_index.data_ptr<int64_t>());

    return edge_index;
}
