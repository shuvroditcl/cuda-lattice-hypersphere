/*
 * 159.735 Assignment 5: Lattice Points in an N-Dimensional Hypersphere
 * Student ID: 26017361
 * Target GPU: NVIDIA Tesla T4 (hpc-parallelgpu01, Turing Architecture, CC 7.5)
 * Processing Units: 40 SMs x 64 cores/SM = 2,560 CUDA Cores, 16 GB GDDR6 VRAM
 *
 * Compile: nvcc -O3 -arch=sm_75 nsphere_cuda.cu -o nsphere_cuda
 * Usage:   ./nsphere_cuda [ndim] [radius] [run_cpu: 1=yes, 0=no] [run_naive_gpu: 1=yes, 0=no]
 */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

typedef unsigned long long ULONGINT;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

/* ============================================================================
 * HOST SEQUENTIAL IMPLEMENTATIONS (from starter code nsphere.cpp)
 * ============================================================================ */

long powlong(long n, long k) {
    long p = 1;
    for (long i = 0; i < k; ++i) p *= n;
    return p;
}

void convert(long num, long base, std::vector<long>& index) {
    const long ndim = index.size();
    for (long i = 0; i < ndim; ++i) index[i] = 0;
    long idx = 0;
    while (num != 0) {
        long rem = num % base;
        num = num / base;
        index[idx] = rem;
        ++idx;
    }
}

long count_in_v1(long ndim, double radius) {
    const long halfb = static_cast<long>(std::floor(radius));
    const long base = 2 * halfb + 1;
    const double rsquare = radius * radius;
    const long ntotal = powlong(base, ndim);

    long count = 0;
    std::vector<long> index(ndim, 0);

    for (long n = 0; n < ntotal; ++n) {
        convert(n, base, index);
        double rtestsq = 0.0;
        for (long k = 0; k < ndim; ++k) {
            double xk = index[k] - halfb;
            rtestsq += xk * xk;
        }
        if (rtestsq < rsquare) ++count;
    }
    return count;
}

void addone(std::vector<long>& index, long base, long i) {
    long ndim = index.size();
    long newv = index[i] + 1;
    if (newv >= base) {
        index[i] = 0;
        if (i < ndim - 1) addone(index, base, i + 1);
    } else {
        index[i] = newv;
    }
}

long count_in_v2(long ndim, double radius) {
    const long halfb = static_cast<long>(std::floor(radius));
    const long base = 2 * halfb + 1;
    const double rsquare = radius * radius;
    const long ntotal = powlong(base, ndim);

    long count = 0;
    std::vector<long> index(ndim, 0);

    for (long n = 0; n < ntotal; ++n) {
        double rtestsq = 0.0;
        for (long k = 0; k < ndim; ++k) {
            double xk = index[k] - halfb;
            rtestsq += xk * xk;
        }
        if (rtestsq < rsquare) ++count;
        addone(index, base, 0);
    }
    return count;
}

/* ============================================================================
 * GPU KERNEL 1: Naive Global atomicAdd (64-bit division + FP64 + global atomic)
 * ============================================================================ */
__global__ void d_count_naive_kernel(ULONGINT ntotal, int ndim, long base, long halfb,
                                     double rsquare, ULONGINT* d_count) {
    ULONGINT tid = static_cast<ULONGINT>(blockIdx.x) * blockDim.x + threadIdx.x;
    ULONGINT stride = static_cast<ULONGINT>(gridDim.x) * blockDim.x;

    for (ULONGINT n = tid; n < ntotal; n += stride) {
        ULONGINT num = n;
        double rtestsq = 0.0;

        for (int k = 0; k < ndim; ++k) {
            long rem = static_cast<long>(num % base);
            num /= base;
            double xk = static_cast<double>(rem - halfb);
            rtestsq += xk * xk;
        }

        if (rtestsq < rsquare) {
            atomicAdd(d_count, 1ULL);
        }
    }
}

/* ============================================================================
 * GPU KERNEL 2: Optimized Architecture-Aware Kernel
 * ============================================================================ */
template <typename UIntT>
__global__ void d_count_optimized_kernel(UIntT ntotal, int ndim, UIntT base,
                                         int halfb, long long max_r2_int, ULONGINT* d_count) {
    __shared__ ULONGINT warp_sums[8];

    UIntT tid = static_cast<UIntT>(blockIdx.x) * blockDim.x + threadIdx.x;
    UIntT stride = static_cast<UIntT>(gridDim.x) * blockDim.x;
    ULONGINT local_count = 0;

    for (UIntT n = tid; n < ntotal; n += stride) {
        UIntT num = n;
        long long rtestsq = 0;

        for (int k = 0; k < ndim; ++k) {
            int rem = static_cast<int>(num % base);
            num /= base;
            int xk = rem - halfb;
            rtestsq += static_cast<long long>(xk) * xk;

            if (rtestsq > max_r2_int) {
                break;
            }
        }

        if (rtestsq <= max_r2_int) {
            ++local_count;
        }
    }

    unsigned int lane = threadIdx.x & 31;
    unsigned int warp_id = threadIdx.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }

    if (lane == 0) {
        warp_sums[warp_id] = local_count;
    }
    __syncthreads();

    if (warp_id == 0) {
        ULONGINT block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0ULL;
        for (int offset = 4; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0 && block_sum > 0) {
            atomicAdd(d_count, block_sum);
        }
    }
}

/* ============================================================================
 * HOST WRAPPER FOR GPU EXECUTION
 * ============================================================================ */
ULONGINT h_count_in_gpu(double radius, int ndim, bool use_optimized,
                        float& kernel_ms, float& total_ms) {
    const long halfb = static_cast<long>(std::floor(radius));
    const long base = 2 * halfb + 1;
    const double rsquare = radius * radius;
    const long long max_r2_int = static_cast<long long>(std::ceil(rsquare)) - 1LL;

    ULONGINT ntotal = 1;
    for (int i = 0; i < ndim; ++i) {
        ntotal *= static_cast<ULONGINT>(base);
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    ULONGINT* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(ULONGINT)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(ULONGINT)));

    const int tpb = 256;
    ULONGINT needed_blocks = (ntotal + tpb - 1) / tpb;
    ULONGINT max_blocks = 65535ULL;
    int bpg = static_cast<int>(needed_blocks < max_blocks ? needed_blocks : max_blocks);
    if (bpg < 1) bpg = 1;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    if (use_optimized) {
        if (ntotal <= 0xFFFFFFFFULL) {
            d_count_optimized_kernel<unsigned int><<<bpg, tpb>>>(
                static_cast<unsigned int>(ntotal), ndim,
                static_cast<unsigned int>(base), static_cast<int>(halfb),
                max_r2_int, d_count);
        } else {
            d_count_optimized_kernel<ULONGINT><<<bpg, tpb>>>(
                ntotal, ndim, static_cast<ULONGINT>(base),
                static_cast<int>(halfb), max_r2_int, d_count);
        }
    } else {
        d_count_naive_kernel<<<bpg, tpb>>>(
            ntotal, ndim, base, halfb, rsquare, d_count);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));

    ULONGINT h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(ULONGINT), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    auto t_end = std::chrono::high_resolution_clock::now();
    total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    return h_count;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <ndim> <radius> [run_cpu=1] [run_naive_gpu=1]\n";
        return 1;
    }

    const int ndim = std::atoi(argv[1]);
    const double r = std::atof(argv[2]);
    const int run_cpu = (argc >= 4) ? std::atoi(argv[3]) : 1;
    const int run_naive = (argc >= 5) ? std::atoi(argv[4]) : 1;

    const long halfb = static_cast<long>(std::floor(r));
    const long base = 2 * halfb + 1;
    double log10_ntotal = ndim * std::log10(static_cast<double>(base));
    if (log10_ntotal > 19.2) {
        std::cerr << "Error: base^ndim exceeds 64-bit unsigned integer limit (2^64 - 1)!\n";
        return 1;
    }

    ULONGINT ntotal = 1;
    for (int i = 0; i < ndim; ++i) ntotal *= static_cast<ULONGINT>(base);

    CUDA_CHECK(cudaFree(0));
    float warmup_k = 0.0f, warmup_t = 0.0f;
    h_count_in_gpu(1.0, 1, true, warmup_k, warmup_t);
    h_count_in_gpu(1.0, 1, false, warmup_k, warmup_t);

    float gpu_opt_kernel_ms = 0.0f, gpu_opt_total_ms = 0.0f;
    ULONGINT gpu_opt_count = h_count_in_gpu(r, ndim, true, gpu_opt_kernel_ms, gpu_opt_total_ms);

    float gpu_naive_kernel_ms = 0.0f, gpu_naive_total_ms = 0.0f;
    ULONGINT gpu_naive_count = gpu_opt_count;
    if (run_naive) {
        gpu_naive_count = h_count_in_gpu(r, ndim, false, gpu_naive_kernel_ms, gpu_naive_total_ms);
    }

    double cpu_v1_ms = 0.0, cpu_v2_ms = 0.0;
    long cpu_v1_count = 0, cpu_v2_count = 0;
    if (run_cpu) {
        auto c1_start = std::chrono::high_resolution_clock::now();
        cpu_v1_count = count_in_v1(ndim, r);
        auto c1_end = std::chrono::high_resolution_clock::now();
        cpu_v1_ms = std::chrono::duration<double, std::milli>(c1_end - c1_start).count();

        auto c2_start = std::chrono::high_resolution_clock::now();
        cpu_v2_count = count_in_v2(ndim, r);
        auto c2_end = std::chrono::high_resolution_clock::now();
        cpu_v2_ms = std::chrono::duration<double, std::milli>(c2_end - c2_start).count();
    }

    bool all_match = (gpu_opt_count == gpu_naive_count);
    if (run_cpu) {
        all_match = all_match &&
                    (gpu_opt_count == static_cast<ULONGINT>(cpu_v1_count)) &&
                    (cpu_v1_count == cpu_v2_count);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "N=" << ndim << " | R=" << r << " | Base=" << base << " | NTotal=" << ntotal
              << " | Count=" << gpu_opt_count
              << " | GPU_Opt_Ker=" << gpu_opt_kernel_ms << "ms"
              << " | GPU_Opt_Tot=" << gpu_opt_total_ms << "ms";
    if (run_naive) {
        std::cout << " | GPU_Naive_Ker=" << gpu_naive_kernel_ms << "ms"
                  << " | Opt_vs_Naive=" << (gpu_naive_kernel_ms / gpu_opt_kernel_ms) << "x";
    }
    if (run_cpu) {
        std::cout << " | CPU_v1=" << cpu_v1_ms << "ms"
                  << " | CPU_v2=" << cpu_v2_ms << "ms"
                  << " | Speedup_v1=" << (cpu_v1_ms / gpu_opt_kernel_ms) << "x"
                  << " | Speedup_v2=" << (cpu_v2_ms / gpu_opt_kernel_ms) << "x";
    }
    std::cout << " | Match=" << (all_match ? "YES" : "NO") << std::endl;

    return 0;
}
