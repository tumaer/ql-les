#include <stdexcept>
#include <torch/extension.h>

torch::Tensor pbc_radius_edges_self_cuda(
    torch::Tensor positions,
    double cutoff,
    double box_side_length,
    int64_t max_neighbors);

static void check_input(const torch::Tensor& t, const char* name) {
    if (!t.is_cuda()) {
        throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
}

torch::Tensor pbc_radius_edges_self(
    torch::Tensor positions,
    double cutoff,
    double box_side_length,
    int64_t max_neighbors) {
    check_input(positions, "positions");

    if (positions.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("positions must be float32");
    }

    if (positions.dim() != 2) {
        throw std::runtime_error("positions must be rank-2");
    }

    const int64_t ndim = positions.size(1);
    if (ndim != 2 && ndim != 3) {
        throw std::runtime_error("only 2D/3D supported");
    }

    if (cutoff <= 0.0) {
        throw std::runtime_error("cutoff must be > 0");
    }
    if (box_side_length <= 0.0) {
        throw std::runtime_error("box_side_length must be > 0");
    }
    if (max_neighbors <= 0) {
        throw std::runtime_error("max_neighbors must be > 0 (silently capped per particle)");
    }

    return pbc_radius_edges_self_cuda(
        positions,
        cutoff,
        box_side_length,
        max_neighbors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pbc_radius_edges_self", &pbc_radius_edges_self, "PBC self radius edges (CUDA)");
}
