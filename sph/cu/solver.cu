// Single-GPU SPH simulation in C++/CUDA.
//
// Supports 2D and 3D periodic SPH simulations via a single
// template <int DIM> class / kernel set.  Select dimension in the
// config file with  dim = 2  (default) or  dim = 3.
//
// Build:
//   nvcc -O3 -std=c++17 solver.cu -o build/solver
//
// Run example:
//   ./solver --config cfg/case.conf
//
// NOTE: saved state files (.bin) now use Vec3 (double3) for both 2D and 3D.

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace minimal_sph {

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ") +               \
                                     cudaGetErrorString(err__) +                  \
                                     " at " + __FILE__ + ":" +                   \
                                     std::to_string(__LINE__));                   \
        }                                                                         \
    } while (0)

using Real = double;
using Vec3 = double3;

constexpr Real kPi  = 3.141592653589793238462643383279502884;
constexpr Real kEps = 2.2204460492503131e-16;

// ─── helpers ────────────────────────────────────────────────────────────────

inline std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    return s.substr(begin, s.find_last_not_of(" \t\r\n") - begin + 1);
}

__device__ inline Real wrap_periodic(Real x, Real L) {
    return x - floor(x / L) * L;
}

__device__ inline Vec3 periodic_displacement(const Vec3& a, const Vec3& b, Real L) {
    Vec3 d = make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
    d.x -= nearbyint(d.x / L) * L;
    d.y -= nearbyint(d.y / L) * L;
    d.z -= nearbyint(d.z / L) * L;
    return d;
}

// ─── quintic kernel (sigma depends on dimension) ────────────────────────────

template <int DIM>
struct QuinticKernel {
    Real one_over_h, cutoff, sigma;

    explicit QuinticKernel(Real h) : one_over_h(1.0 / h), cutoff(3.0 * h) {
        if constexpr (DIM == 2)
            sigma = (7.0 / 478.0) / kPi / (h * h);
        else
            sigma = (3.0 / 359.0) / kPi / (h * h * h);
    }

    __device__ Real w(Real r) const {
        Real q = r * one_over_h;
        Real q1 = fmax(1.0 - q, 0.0), q2 = fmax(2.0 - q, 0.0), q3 = fmax(3.0 - q, 0.0);
        return sigma * (q3*q3*q3*q3*q3 - 6.0*q2*q2*q2*q2*q2 + 15.0*q1*q1*q1*q1*q1);
    }

    __device__ Real grad_w(Real r) const {
        Real q = r * one_over_h;
        Real q1 = fmax(1.0 - q, 0.0), q2 = fmax(2.0 - q, 0.0), q3 = fmax(3.0 - q, 0.0);
        return sigma * (-5.0) * (q3*q3*q3*q3 - 6.0*q2*q2*q2*q2 + 15.0*q1*q1*q1*q1) * one_over_h;
    }
};

// ─── GPU-visible parameter bundle (passed by value to kernels) ───────────────

template <int DIM>
struct SphParams {
    int n, cells_per_dim;
    Real L, mass, nu, p_eos, tvf_factor, cell_size;
    bool is_tvf;
    QuinticKernel<DIM> kernel;
};

// ─── dim-independent kernels ─────────────────────────────────────────────────

__global__ void reset_int_kernel(int* data, int n, int val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = val;
}

__global__ void scatter_to_cells_kernel(
    const int* particle_cell, const int* cell_offsets,
    int* cell_write, int* cell_particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c    = particle_cell[i];
    int slot = cell_offsets[c] + atomicAdd(&cell_write[c], 1);
    cell_particles[slot] = i;
}

__global__ void pressure_kernel(Real* p, const Real* rho, int n, Real p_ref, Real rho_ref) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = p_ref * (rho[i] / rho_ref - 1.0);
}

// ─── dim-templated kernels ────────────────────────────────────────────────────

// Cell index helpers: 2D uses cx + cpd*cy, 3D uses cx + cpd*(cy + cpd*cz).
template <int DIM>
__device__ inline int cell_index(int cx, int cy, int cz, int cpd) {
    if constexpr (DIM == 2) return cx + cpd * cy;
    else                    return cx + cpd * (cy + cpd * cz);
}

template <int DIM>
__global__ void init_tgv_kernel(Vec3* pos, Vec3* vel, int nx, Real L, Real dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n   = (DIM == 2) ? nx * nx : nx * nx * nx;
    if (tid >= n) return;

    int ix = tid % nx;
    int iy = (tid / nx) % nx;
    Real x = (ix + 0.5) * dx;
    Real y = (iy + 0.5) * dx;
    Real k0  = 2.0 * kPi / L;

    if constexpr (DIM == 2) {
        pos[tid] = make_double3(x, y, 0.0);
        vel[tid] = make_double3(-cos(k0*x)*sin(k0*y),
                                +sin(k0*x)*cos(k0*y), 0.0);
    } else {
        int  iz  = tid / (nx * nx);
        Real z   = (iz + 0.5) * dx;
        pos[tid] = make_double3(x, y, z);
        vel[tid] = make_double3(+sin(k0*x)*cos(k0*y)*cos(k0*z),
                                -cos(k0*x)*sin(k0*y)*cos(k0*z), 0.0);
    }
}

__global__ void copy_vec3_kernel(const Vec3* src, Vec3* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

template <int DIM>
__global__ void compute_particle_cells_kernel(
    const Vec3* pos, int* particle_cell, int* cell_counts,
    int n, Real cell_size, int cells_per_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec3 p  = pos[i];
    int  cx = max(0, min(cells_per_dim - 1, (int)floor(p.x / cell_size)));
    int  cy = max(0, min(cells_per_dim - 1, (int)floor(p.y / cell_size)));
    int  cz = 0;
    if constexpr (DIM == 3)
        cz = max(0, min(cells_per_dim - 1, (int)floor(p.z / cell_size)));

    int c = cell_index<DIM>(cx, cy, cz, cells_per_dim);
    particle_cell[i] = c;
    atomicAdd(&cell_counts[c], 1);
}

template <int DIM>
__global__ void density_kernel(
    const Vec3* pos, const int* cell_offsets, const int* cell_particles,
    Real* rho, SphParams<DIM> p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    Vec3 ri = pos[i];
    int  cx = max(0, min(p.cells_per_dim - 1, (int)floor(ri.x / p.cell_size)));
    int  cy = max(0, min(p.cells_per_dim - 1, (int)floor(ri.y / p.cell_size)));
    int  cz = 0;
    if constexpr (DIM == 3)
        cz = max(0, min(p.cells_per_dim - 1, (int)floor(ri.z / p.cell_size)));

    Real rhoi = 0.0;
    for (int oz = (DIM == 3 ? -1 : 0); oz <= (DIM == 3 ? 1 : 0); ++oz) {
        int nz = (DIM == 3) ? (cz + oz + p.cells_per_dim) % p.cells_per_dim : 0;
        for (int oy = -1; oy <= 1; ++oy) {
            int ny = (cy + oy + p.cells_per_dim) % p.cells_per_dim;
            for (int ox = -1; ox <= 1; ++ox) {
                int nx_c = (cx + ox + p.cells_per_dim) % p.cells_per_dim;
                int c    = cell_index<DIM>(nx_c, ny, nz, p.cells_per_dim);
                for (int k = cell_offsets[c]; k < cell_offsets[c + 1]; ++k) {
                    int  j   = cell_particles[k];
                    Vec3 rij = periodic_displacement(ri, pos[j], p.L);
                    Real d   = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);
                    if (d <= p.kernel.cutoff) rhoi += p.mass * p.kernel.w(d);
                }
            }
        }
    }
    rho[i] = rhoi;
}

template <int DIM>
__global__ void accel_kernel(
    const Vec3* pos, const Vec3* vel_u, const Vec3* vel_v,
    const Real* rho, const Real* press,
    const int* cell_offsets, const int* cell_particles,
    Vec3* acc, Vec3* acc_tvf, SphParams<DIM> p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    Vec3 ri = pos[i], ui = vel_u[i], vi = vel_v[i];
    Real rhoi = rho[i], pi = press[i];
    int  cx = max(0, min(p.cells_per_dim - 1, (int)floor(ri.x / p.cell_size)));
    int  cy = max(0, min(p.cells_per_dim - 1, (int)floor(ri.y / p.cell_size)));
    int  cz = 0;
    if constexpr (DIM == 3)
        cz = max(0, min(p.cells_per_dim - 1, (int)floor(ri.z / p.cell_size)));

    Vec3 ai = make_double3(0, 0, 0), atvfi = make_double3(0, 0, 0);

    for (int oz = (DIM == 3 ? -1 : 0); oz <= (DIM == 3 ? 1 : 0); ++oz) {
        int nz = (DIM == 3) ? (cz + oz + p.cells_per_dim) % p.cells_per_dim : 0;
        for (int oy = -1; oy <= 1; ++oy) {
            int ny = (cy + oy + p.cells_per_dim) % p.cells_per_dim;
            for (int ox = -1; ox <= 1; ++ox) {
                int nx_c = (cx + ox + p.cells_per_dim) % p.cells_per_dim;
                int c    = cell_index<DIM>(nx_c, ny, nz, p.cells_per_dim);
                for (int k = cell_offsets[c]; k < cell_offsets[c + 1]; ++k) {
                    int  j    = cell_particles[k];
                    Vec3 rj   = pos[j], uj = vel_u[j], vj = vel_v[j];
                    Real rhoj = rho[j], pj = press[j];

                    Vec3 rij = periodic_displacement(ri, rj, p.L);
                    Real d   = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);
                    if (d > p.kernel.cutoff) continue;

                    Real inv_d     = 1.0 / (d + kEps);
                    Real kd        = p.kernel.grad_w(d);
                    Real inv_rhoi  = 1.0 / (rhoi + kEps);
                    Real inv_rhoj  = 1.0 / (rhoj + kEps);
                    Real prefactor = p.mass * (inv_rhoi*inv_rhoi + inv_rhoj*inv_rhoj);
                    Real pij       = (rhoj*pi + rhoi*pj) / (rhoi + rhoj + kEps);

                    Real coeff = -prefactor * pij * kd * inv_d;
                    ai.x += coeff * rij.x;
                    ai.y += coeff * rij.y;
                    ai.z += coeff * rij.z;

                    if (p.is_tvf) {
                        Vec3 vi_minus_ui = make_double3(vi.x - ui.x, vi.y - ui.y, vi.z - ui.z);
                        Vec3 vj_minus_uj = make_double3(vj.x - uj.x, vj.y - uj.y, vj.z - uj.z);

                        Real A_xx_term = 0.5 * (rhoi * ui.x * vi_minus_ui.x + rhoj * uj.x * vj_minus_uj.x);
                        Real A_xy_term = 0.5 * (rhoi * ui.x * vi_minus_ui.y + rhoj * uj.x * vj_minus_uj.y);
                        Real A_xz_term = 0.5 * (rhoi * ui.x * vi_minus_ui.z + rhoj * uj.x * vj_minus_uj.z);
                        Real A_yx_term = 0.5 * (rhoi * ui.y * vi_minus_ui.x + rhoj * uj.y * vj_minus_uj.x);
                        Real A_yy_term = 0.5 * (rhoi * ui.y * vi_minus_ui.y + rhoj * uj.y * vj_minus_uj.y);
                        Real A_yz_term = 0.5 * (rhoi * ui.y * vi_minus_ui.z + rhoj * uj.y * vj_minus_uj.z);
                        Real A_zx_term = 0.5 * (rhoi * ui.z * vi_minus_ui.x + rhoj * uj.z * vj_minus_uj.x);
                        Real A_zy_term = 0.5 * (rhoi * ui.z * vi_minus_ui.y + rhoj * uj.z * vj_minus_uj.y);
                        Real A_zz_term = 0.5 * (rhoi * ui.z * vi_minus_ui.z + rhoj * uj.z * vj_minus_uj.z);

                        Real A_vec_x = A_xx_term * rij.x + A_xy_term * rij.y + A_xz_term * rij.z;
                        Real A_vec_y = A_yx_term * rij.x + A_yy_term * rij.y + A_yz_term * rij.z;
                        Real A_vec_z = A_zx_term * rij.x + A_zy_term * rij.y + A_zz_term * rij.z;

                        Real A_coeff = prefactor * kd * inv_d;
                        ai.x += A_coeff * A_vec_x;
                        ai.y += A_coeff * A_vec_y;
                        ai.z += A_coeff * A_vec_z;
                    }

                    if (p.nu != 0.0) {
                        Real visc = p.nu * prefactor * kd * inv_d;
                        ai.x += visc * (ui.x - uj.x);
                        ai.y += visc * (ui.y - uj.y);
                        ai.z += visc * (ui.z - uj.z);
                    }

                    if (p.is_tvf) {
                        Real tc = prefactor * kd * inv_d * (-p.p_eos);
                        atvfi.x += tc * rij.x;
                        atvfi.y += tc * rij.y;
                        atvfi.z += tc * rij.z;
                    }
                }
            }
        }
    }
    acc[i]     = ai;
    acc_tvf[i] = atvfi;
}

template <int DIM>
__global__ void integrate_kernel(
    Vec3* pos, Vec3* vel_u, Vec3* vel_v, const Vec3* acc, const Vec3* acc_tvf,
    int n, Real dt, Real L, Real tvf_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec3 u = vel_u[i];
    u.x += dt * acc[i].x;
    u.y += dt * acc[i].y;
    u.z += dt * acc[i].z;
    vel_u[i] = u;

    Vec3 v = u;
    if (tvf_factor != 0.0) {
        v.x += tvf_factor * 0.5 * dt * acc_tvf[i].x;
        v.y += tvf_factor * 0.5 * dt * acc_tvf[i].y;
        v.z += tvf_factor * 0.5 * dt * acc_tvf[i].z;
    }
    vel_v[i] = v;

    Vec3 x = pos[i];
    x.x = wrap_periodic(x.x + dt * v.x, L);
    x.y = wrap_periodic(x.y + dt * v.y, L);
    if constexpr (DIM == 3)
        x.z = wrap_periodic(x.z + dt * v.z, L);
    pos[i] = x;
}

struct SpeedNorm {
    __host__ __device__ Real operator()(const Vec3& u) const {
        return sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
    }
};

template <int DIM>
struct SpeedSqDim {
    __host__ __device__ Real operator()(const Vec3& u) const {
        if constexpr (DIM == 2) return u.x*u.x + u.y*u.y;
        else                    return u.x*u.x + u.y*u.y + u.z*u.z;
    }
};

// ─── unified simulation class ────────────────────────────────────────────────

template <int DIM>
class SphSolver {
public:
    struct Params {
        int  nx          = 512;
        Real L           = 1.0;
        Real u_ref       = 1.0;
        Real nu          = 0.01;
        Real dt          = 5e-4;
        Real t_end       = 3.0;
        Real tvf_factor  = 1.0;
        int  print_every = 200;
        int  save_every  = 0;
        Real noise_std_factor = 0.0;
        std::string save_dir        = "res/out";
        std::string init_state_file = "";
    };

    static Params from_conf(const std::string& path) {
        Params p;
        std::ifstream in(path);
        if (!in) throw std::runtime_error("Failed to open config file: " + path);

        std::string line;
        int line_no = 0;
        while (std::getline(in, line)) {
            ++line_no;
            auto hpos = line.find('#');
            if (hpos != std::string::npos) line = line.substr(0, hpos);
            line = trim(line);
            if (line.empty()) continue;

            auto epos = line.find('=');
            if (epos == std::string::npos)
                throw std::runtime_error("Invalid config entry at line " + std::to_string(line_no));

            const std::string key = trim(line.substr(0, epos));
            const std::string val = trim(line.substr(epos + 1));
            if (key.empty())
                throw std::runtime_error("Empty key at line " + std::to_string(line_no));

            if      (key == "dim")             { /* handled in main() */ }
            else if (key == "nx")              p.nx          = std::stoi(val);
            else if (key == "L")               p.L           = std::stod(val);
            else if (key == "u_ref")           p.u_ref       = std::stod(val);
            else if (key == "nu")              p.nu          = std::stod(val);
            else if (key == "dt")              p.dt          = std::stod(val);
            else if (key == "t_end")           p.t_end       = std::stod(val);
            else if (key == "tvf_factor")      p.tvf_factor  = std::stod(val);
            else if (key == "print_every")     p.print_every = std::stoi(val);
            else if (key == "save_every")      p.save_every  = std::stoi(val);
            else if (key == "noise_std_factor") p.noise_std_factor = std::stod(val);
            else if (key == "save_dir")        p.save_dir    = val;
            else if (key == "init_state_file") p.init_state_file = val;
            else throw std::runtime_error("Unknown config key: " + key);
        }
        return p;
    }

    explicit SphSolver(const Params& p)
        : params_(p)
        , dx_(p.L / p.nx)
        , rho_ref_(1.0)
        , c_eos_(10.0 * p.u_ref)
        , p_eos_(c_eos_ * c_eos_ * rho_ref_)
        , kernel_(dx_)
        , n_(ipow(p.nx))
        , cells_per_dim_(std::max(1, (int)std::floor(p.L / kernel_.cutoff)))
        , n_cells_(ipow(cells_per_dim_))
        , sp_{n_, cells_per_dim_, p.L,
              ipow_real(dx_) * rho_ref_,
              p.nu, p_eos_, p.tvf_factor,
              p.L / cells_per_dim_,
              std::abs(p.tvf_factor) > 0.0,
              kernel_} {
        if (params_.print_every <= 0) params_.print_every = 1;
        if (params_.save_every  <  0) params_.save_every  = 0;
        allocate();
        init();
        open_outputs();
    }

    ~SphSolver() {
        if (diag_csv_.is_open()) diag_csv_.close();
        release();
    }

    void run() {
        int steps      = (int)std::round(params_.t_end / params_.dt);
        int grid_n     = (n_       + 255) / 256;
        int grid_cells = (n_cells_ + 255) / 256;
        const auto t0  = std::chrono::steady_clock::now();

        std::cout << (DIM == 2 ? "2D" : "3D") << " SPH: N=" << n_
                  << " particles, nx=" << params_.nx
                  << ", cells_per_dim=" << cells_per_dim_
                  << ", total_cells=" << n_cells_ << "\n";

        for (int step = 0; step <= steps; ++step) {
            reset_int_kernel<<<grid_cells, 256>>>(d_cell_counts_, n_cells_, 0);

            compute_particle_cells_kernel<DIM><<<grid_n, 256>>>(
                d_pos_, d_particle_cell_, d_cell_counts_,
                n_, sp_.cell_size, sp_.cells_per_dim);

            // Build cell offsets with an exclusive prefix sum.
            thrust::exclusive_scan(
                thrust::device_ptr<int>(d_cell_counts_),
                thrust::device_ptr<int>(d_cell_counts_) + n_cells_,
                thrust::device_ptr<int>(d_cell_offsets_));
            // Sentinel: cell_offsets[n_cells_] = n.
            CUDA_CHECK(cudaMemcpy(d_cell_offsets_ + n_cells_, &n_, sizeof(int),
                                  cudaMemcpyHostToDevice));

            reset_int_kernel<<<grid_cells, 256>>>(d_cell_write_, n_cells_, 0);
            scatter_to_cells_kernel<<<grid_n, 256>>>(
                d_particle_cell_, d_cell_offsets_, d_cell_write_, d_cell_particles_, n_);

            density_kernel<DIM><<<grid_n, 256>>>(
                d_pos_, d_cell_offsets_, d_cell_particles_, d_rho_, sp_);

            pressure_kernel<<<grid_n, 256>>>(d_p_, d_rho_, n_, p_eos_, rho_ref_);

            accel_kernel<DIM><<<grid_n, 256>>>(
                d_pos_, d_vel_u_, d_vel_v_, d_rho_, d_p_,
                d_cell_offsets_, d_cell_particles_,
                d_acc_, d_acc_tvf_, sp_);

            integrate_kernel<DIM><<<grid_n, 256>>>(
                d_pos_, d_vel_u_, d_vel_v_, d_acc_, d_acc_tvf_,
                n_, params_.dt, params_.L, params_.tvf_factor);

            if (step % params_.print_every == 0 || step == steps) {
                CUDA_CHECK(cudaDeviceSynchronize());
                Real t = step * params_.dt;
                Real rho_max = thrust::reduce(
                    thrust::device_ptr<Real>(d_rho_),
                    thrust::device_ptr<Real>(d_rho_) + n_,
                    0.0, thrust::maximum<Real>());
                Real umax = thrust::transform_reduce(
                    thrust::device_pointer_cast(d_vel_v_),
                    thrust::device_pointer_cast(d_vel_v_) + n_,
                    SpeedNorm(), 0.0, thrust::maximum<Real>());
                Real u2_mean = thrust::transform_reduce(
                    thrust::device_pointer_cast(d_vel_v_),
                    thrust::device_pointer_cast(d_vel_v_) + n_,
                    SpeedSqDim<DIM>(), 0.0, thrust::plus<Real>()) / static_cast<Real>(n_);
                Real kinetic_energy = 0.5 * u2_mean * ipow_real(params_.L);

                std::cout << "step=" << step << " t=" << t << " umax=" << umax;
                if constexpr (DIM == 2) {
                    Real uref = std::exp(-8.0 * kPi * kPi * params_.nu * t) * params_.u_ref;
                    std::cout << " uref=" << uref;
                }
                std::cout << " rho_max=" << rho_max << " ekin=" << kinetic_energy << "\n";

                if (diag_csv_.is_open()) {
                    diag_csv_ << step << "," << std::setprecision(17) << t
                              << "," << umax;
                    if constexpr (DIM == 2) {
                        Real uref = std::exp(-8.0 * kPi * kPi * params_.nu * t) * params_.u_ref;
                        diag_csv_ << "," << uref;
                    }
                    diag_csv_ << "," << rho_max << "," << kinetic_energy << "\n";
                    diag_csv_.flush();
                }
            }

            if (params_.save_every > 0 &&
                (step % params_.save_every == 0 || step == steps))
                save_state(step, step * params_.dt);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - t0;
        std::cout << "total_simulation_time_s=" << elapsed.count() << "\n";
    }

private:
    // Compile-time integer / real powers of nx or dx.
    static int  ipow(int  x) { return (DIM == 2) ? x*x     : x*x*x;         }
    static Real ipow_real(Real x) { return (DIM == 2) ? x*x : x*x*x;        }

    void allocate() {
        CUDA_CHECK(cudaMalloc(&d_pos_,            sizeof(Vec3) * n_));
        CUDA_CHECK(cudaMalloc(&d_vel_u_,          sizeof(Vec3) * n_));
        CUDA_CHECK(cudaMalloc(&d_vel_v_,          sizeof(Vec3) * n_));
        CUDA_CHECK(cudaMalloc(&d_acc_,            sizeof(Vec3) * n_));
        CUDA_CHECK(cudaMalloc(&d_acc_tvf_,        sizeof(Vec3) * n_));
        CUDA_CHECK(cudaMalloc(&d_rho_,            sizeof(Real) * n_));
        CUDA_CHECK(cudaMalloc(&d_p_,              sizeof(Real) * n_));
        CUDA_CHECK(cudaMalloc(&d_particle_cell_,  sizeof(int)  * n_));
        CUDA_CHECK(cudaMalloc(&d_cell_particles_, sizeof(int)  * n_));
        CUDA_CHECK(cudaMalloc(&d_cell_counts_,    sizeof(int)  * n_cells_));
        CUDA_CHECK(cudaMalloc(&d_cell_write_,     sizeof(int)  * n_cells_));
        CUDA_CHECK(cudaMalloc(&d_cell_offsets_,   sizeof(int)  * (n_cells_ + 1)));
    }

    void init() {
        if (params_.init_state_file.empty()) {
            init_tgv_kernel<DIM><<<(n_ + 255) / 256, 256>>>(
                d_pos_, d_vel_u_, params_.nx, params_.L, dx_);
            copy_vec3_kernel<<<(n_ + 255) / 256, 256>>>(d_vel_u_, d_vel_v_, n_);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (params_.noise_std_factor > 0.0) {
                const Real noise_std = params_.noise_std_factor * dx_;
                std::vector<Vec3> h_pos(n_);
                CUDA_CHECK(cudaMemcpy(h_pos.data(), d_pos_, sizeof(Vec3) * n_,
                                      cudaMemcpyDeviceToHost));
                std::mt19937_64 rng(42);
                std::normal_distribution<Real> dist(0.0, noise_std);
                auto wrap = [](Real x, Real L) { return x - std::floor(x / L) * L; };
                for (int i = 0; i < n_; ++i) {
                    h_pos[i].x = wrap(h_pos[i].x + dist(rng), params_.L);
                    h_pos[i].y = wrap(h_pos[i].y + dist(rng), params_.L);
                    if constexpr (DIM == 3)
                        h_pos[i].z = wrap(h_pos[i].z + dist(rng), params_.L);
                }
                CUDA_CHECK(cudaMemcpy(d_pos_, h_pos.data(), sizeof(Vec3) * n_,
                                      cudaMemcpyHostToDevice));
                std::cout << "Applied position noise: std=" << noise_std
                          << " (noise_std_factor=" << params_.noise_std_factor << ")\n";
            }
            return;
        }

        std::ifstream in(params_.init_state_file, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open init_state_file: " + params_.init_state_file);

        int32_t nx_i32 = 0, n_i32 = 0;
        Real    t_init = 0.0;
        in.read(reinterpret_cast<char*>(&nx_i32), sizeof(nx_i32));
        in.read(reinterpret_cast<char*>(&n_i32),  sizeof(n_i32));
        in.read(reinterpret_cast<char*>(&t_init), sizeof(t_init));
        if (!in) throw std::runtime_error("Failed to read header: " + params_.init_state_file);
        if (n_i32  != n_)          throw std::runtime_error("Particle count mismatch in init_state_file");
        if (nx_i32 != params_.nx)  throw std::runtime_error("nx mismatch in init_state_file");

        std::vector<Vec3> h_pos(n_), h_vel(n_);
        in.read(reinterpret_cast<char*>(h_pos.data()), sizeof(Vec3) * n_);
        in.read(reinterpret_cast<char*>(h_vel.data()), sizeof(Vec3) * n_);
        if (!in) throw std::runtime_error("Failed to read x/u from init_state_file");

        CUDA_CHECK(cudaMemcpy(d_pos_, h_pos.data(), sizeof(Vec3) * n_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel_u_, h_vel.data(), sizeof(Vec3) * n_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel_v_, h_vel.data(), sizeof(Vec3) * n_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Initialized from state file: " << params_.init_state_file
                  << " (t=" << t_init << ")\n";
    }

    void release() {
        cudaFree(d_pos_);  cudaFree(d_vel_u_);  cudaFree(d_vel_v_);  cudaFree(d_acc_);  cudaFree(d_acc_tvf_);
        cudaFree(d_rho_);  cudaFree(d_p_);
        cudaFree(d_particle_cell_);  cudaFree(d_cell_particles_);
        cudaFree(d_cell_counts_);    cudaFree(d_cell_write_);  cudaFree(d_cell_offsets_);
    }

    void open_outputs() {
        std::filesystem::create_directories(params_.save_dir);
        const std::string csv = params_.save_dir + "/diagnostics.csv";
        diag_csv_.open(csv, std::ios::out | std::ios::trunc);
        if (!diag_csv_) throw std::runtime_error("Failed to open diagnostics CSV: " + csv);
        diag_csv_ << "step,time,umax";
        if constexpr (DIM == 2) diag_csv_ << ",uref";
        diag_csv_ << ",rho_max,ekin\n";
    }

    void save_state(int step, Real t) {
        std::vector<Vec3> h_pos(n_), h_vel(n_);
        CUDA_CHECK(cudaMemcpy(h_pos.data(), d_pos_, sizeof(Vec3) * n_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vel.data(), d_vel_u_, sizeof(Vec3) * n_, cudaMemcpyDeviceToHost));

        std::ostringstream oss;
        oss << params_.save_dir << "/state_step_"
            << std::setw(8) << std::setfill('0') << step << ".bin";
        std::ofstream out(oss.str(), std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("Failed to open state output: " + oss.str());

        const int32_t nx_i32 = (int32_t)params_.nx, n_i32 = (int32_t)n_;
        out.write(reinterpret_cast<const char*>(&nx_i32), sizeof(nx_i32));
        out.write(reinterpret_cast<const char*>(&n_i32),  sizeof(n_i32));
        out.write(reinterpret_cast<const char*>(&t),      sizeof(t));
        out.write(reinterpret_cast<const char*>(h_pos.data()), sizeof(Vec3) * n_);
        out.write(reinterpret_cast<const char*>(h_vel.data()), sizeof(Vec3) * n_);
        if (!out) throw std::runtime_error("Failed while writing state output");
    }

    // Members are declared in initialization order (C++ initializes in declaration order).
    Params              params_;
    Real                dx_, rho_ref_, c_eos_, p_eos_;
    QuinticKernel<DIM>  kernel_;
    int                 n_, cells_per_dim_, n_cells_;
    SphParams<DIM>      sp_;

    Vec3* d_pos_            = nullptr;
    Vec3* d_vel_u_          = nullptr;
    Vec3* d_vel_v_          = nullptr;
    Vec3* d_acc_            = nullptr;
    Vec3* d_acc_tvf_        = nullptr;
    Real* d_rho_            = nullptr;
    Real* d_p_              = nullptr;
    int*  d_particle_cell_  = nullptr;
    int*  d_cell_particles_ = nullptr;
    int*  d_cell_counts_    = nullptr;
    int*  d_cell_write_     = nullptr;
    int*  d_cell_offsets_   = nullptr;

    std::ofstream diag_csv_;
};

}  // namespace minimal_sph

namespace {

struct CliOverrides {
    bool has_dim = false;
    bool has_nx = false;
    bool has_L = false;
    bool has_u_ref = false;
    bool has_nu = false;
    bool has_dt = false;
    bool has_save_dir = false;
    bool has_init_state_file = false;
    bool has_t_end = false;
    bool has_tvf_factor = false;
    bool has_print_every = false;
    bool has_save_every = false;
    bool has_noise_std_factor = false;
    int dim = 2;
    int nx = 0;
    int print_every = 0;
    int save_every = 0;
    minimal_sph::Real L = 0.0;
    minimal_sph::Real u_ref = 0.0;
    minimal_sph::Real nu = 0.0;
    minimal_sph::Real dt = 0.0;
    std::string save_dir;
    std::string init_state_file;
    minimal_sph::Real t_end = 0.0;
    minimal_sph::Real tvf_factor = 0.0;
    minimal_sph::Real noise_std_factor = 0.0;
};

void print_usage() {
    std::cerr << "Usage: ./solver [config_path] [--config path]"
              << " [--dim value] [--nx value] [--L value] [--u_ref value] [--nu value]"
              << " [--dt value] [--t_end value] [--tvf_factor value]"
              << " [--print_every value] [--save_every value]"
              << " [--noise_std_factor value] [--save_dir path]"
              << " [--init_state_file path]\n";
}

std::string require_arg_value(const std::string& flag, int argc, char** argv, int& i) {
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value after " + flag);
    }
    return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
    std::string conf_path = "case.conf";
    CliOverrides overrides;
    bool have_positional_config = false;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--config") {
                conf_path = require_arg_value(arg, argc, argv, i);
                have_positional_config = true;
            } else if (arg == "--dim") {
                overrides.dim = std::stoi(require_arg_value(arg, argc, argv, i));
                overrides.has_dim = true;
            } else if (arg == "--nx") {
                overrides.nx = std::stoi(require_arg_value(arg, argc, argv, i));
                overrides.has_nx = true;
            } else if (arg == "--L") {
                overrides.L = std::stod(require_arg_value(arg, argc, argv, i));
                overrides.has_L = true;
            } else if (arg == "--u_ref") {
                overrides.u_ref = std::stod(require_arg_value(arg, argc, argv, i));
                overrides.has_u_ref = true;
            } else if (arg == "--nu") {
                overrides.nu = std::stod(require_arg_value(arg, argc, argv, i));
                overrides.has_nu = true;
            } else if (arg == "--dt") {
                overrides.dt = std::stod(require_arg_value(arg, argc, argv, i));
                overrides.has_dt = true;
            } else if (arg == "--save_dir") {
                overrides.save_dir = require_arg_value(arg, argc, argv, i);
                overrides.has_save_dir = true;
            } else if (arg == "--init_state_file") {
                overrides.init_state_file = require_arg_value(arg, argc, argv, i);
                overrides.has_init_state_file = true;
            } else if (arg == "--t_end") {
                overrides.t_end = std::stod(require_arg_value(arg, argc, argv, i));
                overrides.has_t_end = true;
            } else if (arg == "--tvf_factor") {
                overrides.tvf_factor = std::stod(require_arg_value(arg, argc, argv, i));
                overrides.has_tvf_factor = true;
            } else if (arg == "--print_every") {
                overrides.print_every = std::stoi(require_arg_value(arg, argc, argv, i));
                overrides.has_print_every = true;
            } else if (arg == "--save_every") {
                overrides.save_every = std::stoi(require_arg_value(arg, argc, argv, i));
                overrides.has_save_every = true;
            } else if (arg == "--noise_std_factor") {
                overrides.noise_std_factor = std::stod(require_arg_value(arg, argc, argv, i));
                overrides.has_noise_std_factor = true;
            } else if (!arg.empty() && arg[0] != '-' && !have_positional_config) {
                conf_path = arg;
                have_positional_config = true;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        print_usage();
        return 1;
    }

    // Peek at the config to read dim= before instantiating the template.
    int dim = overrides.has_dim ? overrides.dim : 2;
    if (!overrides.has_dim) {
        std::ifstream in(conf_path);
        std::string line;
        while (std::getline(in, line)) {
            auto h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);
            auto e = line.find('='); if (e == std::string::npos) continue;
            if (minimal_sph::trim(line.substr(0, e)) == "dim")
                dim = std::stoi(minimal_sph::trim(line.substr(e + 1)));
        }
    }

    try {
        auto apply_overrides = [&overrides](auto& params) {
            if (overrides.has_nx) params.nx = overrides.nx;
            if (overrides.has_L) params.L = overrides.L;
            if (overrides.has_u_ref) params.u_ref = overrides.u_ref;
            if (overrides.has_nu) params.nu = overrides.nu;
            if (overrides.has_dt) params.dt = overrides.dt;
            if (overrides.has_save_dir) params.save_dir = overrides.save_dir;
            if (overrides.has_init_state_file) params.init_state_file = overrides.init_state_file;
            if (overrides.has_t_end) params.t_end = overrides.t_end;
            if (overrides.has_tvf_factor) params.tvf_factor = overrides.tvf_factor;
            if (overrides.has_print_every) params.print_every = overrides.print_every;
            if (overrides.has_save_every) params.save_every = overrides.save_every;
            if (overrides.has_noise_std_factor) params.noise_std_factor = overrides.noise_std_factor;
        };

        if (dim != 2 && dim != 3)
            throw std::runtime_error("Unsupported dim: " + std::to_string(dim));

        if (dim == 3) {
            using Sim = minimal_sph::SphSolver<3>;
            auto params = Sim::from_conf(conf_path);
            apply_overrides(params);
            Sim sim(params);
            sim.run();
        } else {
            using Sim = minimal_sph::SphSolver<2>;
            auto params = Sim::from_conf(conf_path);
            apply_overrides(params);
            Sim sim(params);
            sim.run();
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n"
                  << "Usage: ./solver [config_path] [--config path]"
                  << " [--dim value] [--nx value] [--L value] [--u_ref value] [--nu value]"
                  << " [--dt value] [--t_end value] [--tvf_factor value]"
                  << " [--print_every value] [--save_every value]"
                  << " [--noise_std_factor value] [--save_dir path]"
                  << " [--init_state_file path]\n";
        return 1;
    }

    return 0;
}
