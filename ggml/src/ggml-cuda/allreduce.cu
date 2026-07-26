#include "allreduce.cuh"

// HIP compatibility: map the CUDA host-mapped-pinned-memory APIs that the
// vendor hip.h header does not provide.  hipHostMalloc has the same signature
// as cudaHostAlloc (ptr, size, flags), and the flag constants share values.
#if defined(GGML_USE_HIP)
#ifndef cudaHostAlloc
#define cudaHostAlloc(ptr, size, flags) hipHostMalloc(ptr, size, flags)
#endif
#ifndef cudaHostAllocPortable
#define cudaHostAllocPortable hipHostMallocPortable
#endif
#ifndef cudaHostAllocMapped
#define cudaHostAllocMapped hipHostMallocMapped
#endif
#ifndef cudaHostGetDevicePointer
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#endif
#endif // GGML_USE_HIP

#include "convert.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

// ---------------------------------------------------------------------------
// Internal AllReduce for tensor-parallel inference across N GPUs.
//
// Provides an in-place sum reduction over matching tensors on N CUDA/HIP
// devices in the same process.  Used by the tensor-split path alongside
// NCCL.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single kernel both
//     stages data through host-mapped pinned memory and performs the local
//     sum.  Cross-GPU synchronization happens *inside the kernel* (busy-wait
//     on a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.
//
//   * Copy-engine path (large reductions): each GPU pulls peer data via
//     hipMemcpyPeerAsync / cudaMemcpyPeerAsync and runs a small device-side
//     add kernel.  On MI100/XGMI this uses the high-bandwidth peer-to-peer
//     fabric directly.
//
// Both paths work on CUDA (NVLink / PCIe) and HIP (XGMI / PCIe).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank, block) in pinned host memory.  Each AR call writes
// a strictly increasing token (= the AR call number) into its own arrival int.
// Peers spin until their read of the other's arrival int equals the token they
// expect for this call.
//
// There is exactly one writer (the owning GPU) and N-1 readers, so we don't
// need atomics.  A volatile store paired with __threadfence_system() provides
// the release ordering.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p, int token) {
    *(volatile int *)p = token;
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
    return *(const volatile int *)p;
}

// Portable low-power busy-wait yield.  __nanosleep is CUDA-only (sm70+);
// HIP uses the AMDGCN s_sleep intrinsic.
static __device__ __forceinline__ void ggml_cuda_ar_sleep() {
#if defined(GGML_USE_HIP)
    __builtin_amdgcn_s_sleep(0x3FFF);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    __nanosleep(100);
#else
    NO_DEVICE_CODE;
#endif
}

// Byte spacing between adjacent arrival ints.  64 bytes (one cache line)
// ensures each GPU/block's arrival slot lives on its own line, preventing
// false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 64;

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slot so multiple SMs can pump host-memory stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// ---------------------------------------------------------------------------
// Pointer table for the N-GPU chunked kernel, passed by value.
//
// Host-mapped memory pointers are *universal*: the same numeric address is
// valid on every device (confirmed on MI100/XGMI via hipHostGetDevicePointer).
// This lets us pass an array of per-rank host-buf and arrival pointers in a
// single struct, and the kernel can dereference any rank's pointer directly.
// ---------------------------------------------------------------------------
struct ggml_cuda_ar_kernel_ptrs {
    void * host_bufs[GGML_CUDA_MAX_DEVICES];
    int  * arrivals[GGML_CUDA_MAX_DEVICES];
};

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- N GPUs, supports float, half, and bfloat16.
//
// All GPUs run this kernel simultaneously, each on its own compute stream.
// sendbuf and recvbuf live in T_dst (the caller's tensor type); host_bufs
// carry data in T_wire (the on-wire type, possibly narrower than T_dst).
//
// Each GPU runs three phases:
//
//   Phase 1 (all threads): cast sendbuf (T_dst) -> T_wire and store into
//                          host_bufs[my_rank] (my own host-mapped buffer).
//                          __threadfence_system() commits these writes.
//   Phase 2 (thread 0):    write token to arrivals[my_rank]; spin until
//                          arrivals[peer] == token for ALL peers.
//   Phase 3 (all threads): read T_wire data from each peer's host_bufs,
//                          cast to float, sum all contributions, and write
//                          back to recvbuf.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(
        const T_dst           * sendbuf,
        T_dst                 * recvbuf,
        ggml_cuda_ar_kernel_ptrs ptrs,
        int                     my_rank,
        int                     n_devices,
        int                     count,
        int                     token) {

    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int ARRIVAL_INTS  = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

    T_wire * my_host = reinterpret_cast<T_wire *>(ptrs.host_bufs[my_rank]);

    // Phase 1: cast sendbuf -> my_host_buf (T_wire), store as vectors.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
            }
            ggml_cuda_memcpy_1<sizeof(wire)>(&my_host[off], wire);
        }
        if (bid == 0 && tid < count - tail) {
            my_host[tail + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
        }
    }

    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 of each block signals on its own arrival slot, then
    // spins for the matching token from ALL peers.
    if (tid == 0) {
        int * my_slot = ptrs.arrivals[my_rank] + bid * ARRIVAL_INTS;

        ggml_cuda_ar_signal_set(my_slot, token);
        __threadfence_system();

        for (int d = 1; d < n_devices; ++d) {
            const int peer = (my_rank + d) % n_devices;
            const int * peer_slot = ptrs.arrivals[peer] + bid * ARRIVAL_INTS;
            while (ggml_cuda_ar_signal_get(peer_slot) != token) {
                ggml_cuda_ar_sleep();
            }
        }
    }

    __syncthreads();
    __threadfence_system();

    // Phase 3: sum contributions from self (rounded through T_wire) and all
    // peers.  Each value is read from host_bufs as T_wire, cast to float,
    // summed, and the result written back as T_dst.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;

            // Start with my own value rounded through T_wire for bit-equivalence.
            float sum[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                sum[k] = ggml_cuda_cast<float>(ggml_cuda_cast<T_wire>(sendbuf[off + k]));
            }

            // Add each peer's contribution.
            for (int d = 1; d < n_devices; ++d) {
                const int peer = (my_rank + d) % n_devices;
                T_wire wire[ELEMS_PER_VEC];
                ggml_cuda_memcpy_1<sizeof(wire)>(
                    wire, &reinterpret_cast<const T_wire *>(ptrs.host_bufs[peer])[off]);
                #pragma unroll
                for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                    sum[k] += ggml_cuda_cast<float>(wire[k]);
                }
            }

            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                recvbuf[off + k] = ggml_cuda_cast<T_dst>(sum[k]);
            }
        }
        // Tail elements.
        if (bid == 0 && tid < count - tail) {
            float s = ggml_cuda_cast<float>(ggml_cuda_cast<T_wire>(sendbuf[tail + tid]));
            for (int d = 1; d < n_devices; ++d) {
                const int peer = (my_rank + d) % n_devices;
                s += ggml_cuda_cast<float>(
                    reinterpret_cast<const T_wire *>(ptrs.host_bufs[peer])[tail + tid]);
            }
            recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(s);
        }
    }
}

// Combined load-convert-add kernel for the copy-engine path.  Adds src into
// dst, rounding dst through T_src for bit-equivalence between GPUs.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_add_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src d_low = ggml_cuda_cast<T_src>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i]));
    }
}

// Type-convert kernel: dst[i] = (T_dst)(float)src[i].  Used by the ring path
// to convert the BF16-reduced result into the F32 destination tensor.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_cvt_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        dst[i] = ggml_cuda_cast<T_dst>(ggml_cuda_cast<float>(src[i]));
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  Two slots is sufficient:
// the pipeline is barrier-like, so slot[N%2] is always safe to reuse.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 2;

// Maximum chunk size (bytes per GPU) handled by one chunked kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB

// Copy-engine path: largest tensor accepted on this path; sets dev_tmp
// allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copy-engine path takes over from the chunked-
// kernel path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.
struct ggml_cuda_ar_host_mapping {
    uint8_t * host = nullptr;
    uint8_t * dev  = nullptr;

    cudaError_t alloc(size_t bytes) {
        cudaError_t rc = cudaHostAlloc(reinterpret_cast<void **>(&host), bytes,
                                       cudaHostAllocPortable | cudaHostAllocMapped);
        if (rc != cudaSuccess) {
            host = nullptr;
            return rc;
        }
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), host, 0);
        if (rc != cudaSuccess) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
        return rc;
    }

    void free() {
        if (host) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
    }
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in dev_tmp[]
    size_t   copy_threshold;
    size_t   bf16_threshold;
    uint64_t call_count;

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path
    ggml_cuda_ar_event_slot   ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

    // Per-device "my add_kernel is done with dev_tmp" event.  Recorded on
    // the compute stream after each AR; the next AR waits on it before
    // overwriting dev_tmp.  Single-buffered dev_tmp is safe because of this.
    cudaEvent_t              dev_tmp_kernel_done[GGML_CUDA_MAX_DEVICES];
    bool                     dev_tmp_kernel_done_valid;

    // Ring-allreduce step events: double-buffered [slot][buf][device].
    // buf = step%2 for the current step's recording, (step-1)%2 for waiting.
    cudaEvent_t ring_ev[GGML_CUDA_AR_POOL_SIZE][2][GGML_CUDA_MAX_DEVICES];

    // Arrival ring: ARRIVAL_STRIDE bytes between adjacent ints.  Mapped pinned
    // memory; CPU never reads/writes -- only the kernel.
    ggml_cuda_ar_host_mapping arrival;
};

// Base pointer for the (slot, rank) per-block token block.  The kernel adds
// blockIdx.x * (ARRIVAL_STRIDE/sizeof(int)) internally to land on its own slot.
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival.dev + offset);
}

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

struct ggml_cuda_ar_slot_info {
    int slot;
    int token;
};

static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    if (pool_lapped) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    return { slot, (int) p->call_count };
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices < 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce requires n_devices >= 2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }
    if (n_devices > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_DEBUG("%s: internal AllReduce: n_devices=%zu exceeds GGML_CUDA_MAX_DEVICES=%d; "
                       "falling back\n", __func__, n_devices, GGML_CUDA_MAX_DEVICES);
        return nullptr;
    }

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices        = (int) n_devices;
    p->copy_bytes       = GGML_CUDA_AR_COPY_MAX_BYTES;
    p->copy_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    // Enable peer-to-peer access between all device pairs.  This is required
    // for hipMemcpyPeerAsync on the copy-engine path.  On XGMI-connected
    // MI100s this enables ~37 GB/s peer bandwidth.  Peer access may already
    // be enabled by ggml_cuda_init (when GGML_CUDA_P2P is set), but calling
    // again is harmless (returns cudaErrorPeerAccessAlreadyEnabled).
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(devices[i]);
        for (size_t j = 0; j < n_devices; ++j) {
            if (i == j) continue;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]));
            if (can_access) {
                cudaError_t rc = cudaDeviceEnablePeerAccess(devices[j], 0);
                // On HIP, hipDeviceEnablePeerAccess sets a sticky error even
                // for the benign "already enabled" case (hipErrorPeerAccess-
                // AlreadyEnabled = 704).  If left uncleared, subsequent kernel
                // launches report LAUNCHFAIL err=704.  Always clear it.
                if (rc != cudaSuccess && rc != cudaErrorPeerAccessAlreadyEnabled) {
                    GGML_LOG_WARN("%s: cudaDeviceEnablePeerAccess(%d->%d) failed: %s\n",
                                  __func__, devices[i], devices[j], cudaGetErrorString(rc));
                }
                (void) cudaGetLastError(); // clear sticky error unconditionally
            } else {
                GGML_LOG_WARN("%s: peer access not supported: dev %d -> dev %d\n",
                              __func__, devices[i], devices[j]);
            }
        }
    }

    // Per-device event pools.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }

        if (cudaEventCreateWithFlags(&p->dev_tmp_kernel_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for dev_tmp_kernel_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Ring-allreduce step events: double-buffered per-slot per-device.
    for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
        for (int b = 0; b < 2; ++b) {
            for (size_t i = 0; i < n_devices; ++i) {
                ggml_cuda_set_device(p->devices[i]);
                if (cudaEventCreateWithFlags(&p->ring_ev[s][b][i], cudaEventDisableTiming) != cudaSuccess) {
                    GGML_LOG_ERROR("%s: cudaEventCreate for ring_ev failed (slot %d buf %d dev %d)\n",
                                   __func__, s, b, p->devices[i]);
                    ggml_cuda_ar_pipeline_free(p);
                    return nullptr;
                }
            }
        }
    }

    // Arrival ring: cache-line padded so each GPU/block's int is on its own line.
    const size_t arrival_bytes =
        (size_t)GGML_CUDA_AR_POOL_SIZE * n_devices *
        GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    if (p->arrival.alloc(arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
    ggml_cuda_set_device(p->devices[0]);
    if (cudaMemset(p->arrival.dev, 0, arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemset for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }

    // Per-device pinned staging buffers -- POOL_SIZE-deep ring so the chunked-
    // kernel can write the next slot's data while peers are still reading
    // the previous slot's.
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t host_buf_total = (size_t) GGML_CUDA_AR_POOL_SIZE * p->buf_bytes;
    for (size_t i = 0; i < n_devices; ++i) {
        if (p->host_buf[i].alloc(host_buf_total) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n",
                           __func__, host_buf_total);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Copy-engine path: device scratch, sized for the largest tensor we
    // accept on this path (GGML_CUDA_AR_COPY_MAX_BYTES).
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                           __func__, p->copy_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %d GPUs, "
                  "%zu KB chunked-kernel staging + %zu MB copy-engine scratch per GPU\n",
                  __func__, p->n_devices, p->buf_bytes >> 10, p->copy_bytes >> 20);

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    for (int i = 0; i < p->n_devices; ++i) {
        p->host_buf[i].free();
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaFree(p->dev_tmp[i]);
        }
        ggml_cuda_set_device(p->devices[i]);
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            if (p->ev_pool[i][s].app) { cudaEventDestroy(p->ev_pool[i][s].app); }
            if (p->ev_pool[i][s].ker) { cudaEventDestroy(p->ev_pool[i][s].ker); }
        }
        if (p->dev_tmp_kernel_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaEventDestroy(p->dev_tmp_kernel_done[i]);
        }
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            for (int b = 0; b < 2; ++b) {
                if (p->ring_ev[s][b][i]) {
                    ggml_cuda_set_device(p->devices[i]);
                    cudaEventDestroy(p->ring_ev[s][b][i]);
                }
            }
        }
    }
    p->arrival.free();
    delete p;
}

// ---------------------------------------------------------------------------
// Copy-engine path: N-GPU ring allreduce
//
// Implements a bandwidth-optimal ring allreduce in two phases:
//
//   Scatter-reduce (n-1 steps):
//     GPU i sends chunk[(i-s)%n] to GPU (i+1)%n, and GPU (i+1)%n adds it to
//     its local copy of chunk[(i-s)%n].  After n-1 steps, GPU i holds the
//     full sum for chunk[(i+1)%n].
//
//   Allgather (n-1 steps):
//     GPU i sends its fully-reduced chunk to GPU (i+1)%n.  After n-1 steps,
//     all GPUs have all chunks.
//
// Ring topology: GPU i sends to GPU (i+1)%n, receives from GPU (i-1+n)%n.
//
// Data moved per GPU: 2*(n-1)/n * nbytes  (vs (n-1)*nbytes for the old
// N*(N-1) approach).  For 4 GPUs: 1.5*nbytes vs 3*nbytes — 2x less traffic.
//
// Synchronization: step s of GPU i waits on step s-1 of GPU (i-1+n)%n via
// double-buffered ring events.  dev_tmp[i] serves as the receive scratch
// for the scatter-reduce phase (copy peer chunk, then add into local buf).
// The allgather phase copies directly peer-to-peer (no add needed).
//
// The ring operates entirely in T_src.  If T_dst != T_src (BF16 ring, F32
// output), a final conversion kernel writes src_buf -> dst_buf.
// ---------------------------------------------------------------------------

template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    const int n = p->n_devices;
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const int slot = ggml_cuda_ar_acquire_slot(p).slot;

    ggml_backend_cuda_context * cuda_ctx[GGML_CUDA_MAX_DEVICES] = {};

    // Record compute-done events for all GPUs.  GPU i needs to wait on its
    // left neighbor's upstream compute before reading from it in step 0.
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].app, cuda_ctx[i]->stream()));
    }

    // Per-GPU setup: protect dev_tmp from prior AR, zero inactive shards.
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cudaStream_t stream = cuda_ctx[i]->stream();

        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, p->dev_tmp_kernel_done[i]));
        }
        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, stream));
        }
    }

    // Chunk layout: n equal chunks; the last absorbs the remainder so every
    // GPU agrees on offsets without extra communication.
    const int64_t  chunk_base = ne / n;
    const int64_t  chunk_rem  = ne - chunk_base * n;
    const size_t   elem_bytes = sizeof(T_src);
    const int      block_size = 256;

    // total_step indexes the double-buffered ring events (total_step % 2).
    int total_step = 0;

    // ---- Phase 1: Scatter-reduce (n-1 steps) ----
    // Step s: GPU i receives chunk c_recv = (i-s-1+n)%n from left = (i-1+n)%n,
    // copies it into dev_tmp[i], and adds dev_tmp[i] into src_buf[i][c_recv].
    // After n-1 steps, GPU i holds the full sum for chunk (i+1)%n.
    //
    // We use the copy engine (hipMemcpyPeerAsync) for the data transfer and a
    // separate add kernel, rather than a single P2P-read kernel.  On MI100/XGMI
    // the SDMA copy engine runs in parallel with compute and is more efficient
    // for small transfers than uncached loads from a compute kernel.
    //
    // Loop order is step-outer so all GPUs issue their work for a given step
    // before advancing — this keeps the ring pipeline balanced.
    for (int s = 0; s < n - 1; ++s, ++total_step) {
        const int b_cur = total_step % 2;

        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStream_t stream = cuda_ctx[i]->stream();
            const int left = (i - 1 + n) % n;

            // Wait for left neighbor's data to be ready.
            if (total_step == 0) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, p->ev_pool[left][slot].app));
            } else {
                const int b_prev = (total_step - 1) % 2;
                CUDA_CHECK(cudaStreamWaitEvent(stream, p->ring_ev[slot][b_prev][left]));
            }

            // Chunk to receive and accumulate this step.
            const int     c_recv = (((i - s - 1) % n) + n) % n;
            const int64_t off    = (int64_t)c_recv * chunk_base;
            const int64_t cn     = (c_recv == n - 1) ? (chunk_base + chunk_rem) : chunk_base;
            const size_t  cbytes = (size_t)cn * elem_bytes;

            // Copy left neighbor's chunk into dev_tmp (receive scratch).
            CUDA_CHECK(cudaMemcpyPeerAsync(
                p->dev_tmp[i], p->devices[i],
                src_buf[left] + off, p->devices[left],
                cbytes, stream));

            // Add dev_tmp into local chunk.
            int n_blocks = (int)((cn + block_size - 1) / block_size);
            if (n_blocks > 1024) n_blocks = 1024;
            if (n_blocks > 0) {
                ggml_cuda_ar_add_kernel<T_src, T_src><<<n_blocks, block_size, 0, stream>>>(
                    src_buf[i] + off,
                    reinterpret_cast<const T_src *>(p->dev_tmp[i]),
                    (int)cn);
                CUDA_CHECK(cudaGetLastError());
            }

            CUDA_CHECK(cudaEventRecord(p->ring_ev[slot][b_cur][i], stream));
        }
    }

    // ---- Phase 2: Allgather (n-1 steps) ----
    // Step s: GPU i receives chunk c_recv = (i-s+n)%n from left = (i-1+n)%n.
    // This is a direct copy (overwrite, no add): left neighbor's fully-reduced
    // chunk replaces the local copy.  After n-1 steps, all GPUs have all chunks.
    for (int s = 0; s < n - 1; ++s, ++total_step) {
        const int b_cur  = total_step % 2;
        const int b_prev = (total_step - 1) % 2;

        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStream_t stream = cuda_ctx[i]->stream();
            const int left = (i - 1 + n) % n;

            CUDA_CHECK(cudaStreamWaitEvent(stream, p->ring_ev[slot][b_prev][left]));

            const int     c_recv = (((i - s) % n) + n) % n;
            const int64_t off    = (int64_t)c_recv * chunk_base;
            const int64_t cn     = (c_recv == n - 1) ? (chunk_base + chunk_rem) : chunk_base;
            const size_t  cbytes = (size_t)cn * elem_bytes;

            CUDA_CHECK(cudaMemcpyPeerAsync(
                src_buf[i] + off, p->devices[i],
                src_buf[left] + off, p->devices[left],
                cbytes, stream));

            CUDA_CHECK(cudaEventRecord(p->ring_ev[slot][b_cur][i], stream));
        }
    }

    // If T_dst != T_src (BF16 ring → F32 output), convert the result.
    if (!std::is_same_v<T_src, T_dst>) {
        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStream_t stream = cuda_ctx[i]->stream();
            int n_blocks = (int)((ne + block_size - 1) / block_size);
            if (n_blocks > 1024) n_blocks = 1024;
            if (n_blocks > 0) {
                ggml_cuda_ar_cvt_kernel<T_dst, T_src><<<n_blocks, block_size, 0, stream>>>(
                    dst_buf[i], src_buf[i], (int)ne);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    // Record final events for pool-wraparound and dev_tmp safety.
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cudaStream_t stream = cuda_ctx[i]->stream();
        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], stream));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
    }
    p->dev_tmp_kernel_done_valid = true;

    return true;
}

// Outer-level chunker: copy_impl handles up to copy_bytes per call (limited by
// the dev_tmp allocation size).  When the full AR exceeds that, slice the
// tensor into copy_bytes-sized pieces and call copy_impl repeatedly.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    const int64_t outer_max_elems = (int64_t) (p->copy_bytes / sizeof(T_src));
    GGML_ASSERT(outer_max_elems > 0);

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne     = std::min(outer_max_elems, ne - outer_start);
        const size_t  outer_nbytes = (size_t) outer_ne * sizeof(T_src);

        T_src * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start;
            dst[i] = dst_buf[i] + outer_start;
        }
        ok = ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>(
            p, backends, src, dst, compute, outer_ne, outer_nbytes);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const size_t   input_nbytes = ggml_nbytes(tensors[0]);

    // BF16 round-trip: F32 inputs >= bf16_threshold are converted to BF16 for
    // the reduction, halving on-wire bytes.  The pre-conversion zeroes
    // inactive shards so the inner paths see them as already-prepared.
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;

    const ggml_type kernel_type = use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t    type_size   = ggml_type_size(kernel_type);
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes      = (size_t) ne * type_size;

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    const bool use_copy_engine =
        p->copy_threshold > 0 &&
        nbytes >= p->copy_threshold;

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // and the add kernel both accumulate into the F32 tensor data directly,
    // so an inactive shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    bool ok = true;
    if (use_copy_engine) {
        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES] = {};
            float       * dst[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, ne);
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<float *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<float, float>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_BF16: {
                    nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<nv_bfloat16 *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_F16: {
                    half * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<half *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<half, half>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        // Chunked kernel path: runs entirely on the caller's compute stream.
        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t input_type_size = ggml_type_size(input_type);

        for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
            const size_t remaining_elems = (size_t) (ne - chunk_start);
            const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;

            // Build the pointer table for the kernel.  Host-mapped pointers
            // are universal (same numeric address on all devices), so the
            // kernel can dereference any rank's entry.
            ggml_cuda_ar_kernel_ptrs kptrs;
            for (int i = 0; i < n; ++i) {
                kptrs.host_bufs[i] = p->host_buf[i].dev + (size_t) slot * p->buf_bytes;
                kptrs.arrivals[i]  = ggml_cuda_ar_arrival_ptr(p, slot, i);
            }

            for (int i = 0; i < n; ++i) {
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                cudaStream_t stream = cuda_ctx->stream();

                char * data = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) input_type_size;

                // Inactive shards contribute zeros.  On the BF16 path the F32
                // tensor data was already zeroed up-front.
                if (!compute_flag[i] && !use_bf16) {
                    CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_dst_bytes, stream));
                }

#define LAUNCH_AR_KERNEL(T_dst, T_wire) \
                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \
                    reinterpret_cast<const T_dst *>(data), \
                    reinterpret_cast<T_dst *>(data), \
                    kptrs, \
                    i, \
                    n, \
                    static_cast<int>(chunk_elems), \
                    token)

                if (use_bf16) {
                    GGML_ASSERT(input_type == GGML_TYPE_F32);
                    LAUNCH_AR_KERNEL(float, nv_bfloat16);
                } else {
                    switch (input_type) {
                        case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float,       float);       break;
                        case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half,        half);        break;
                        case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(nv_bfloat16, nv_bfloat16); break;
                        default: GGML_ASSERT(false);
                    }
                }

#undef LAUNCH_AR_KERNEL
                CUDA_CHECK(cudaGetLastError());

                if (last_chunk) {
                    CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
                }
            }
        }
    }

    return ok;
}
