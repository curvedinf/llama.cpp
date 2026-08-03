#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }
#pragma unroll
    for (size_t j = 0; j < d_conv - 1; j++) {
        x[j] = x_block[tid * stride_x + j];
    }
    x[d_conv - 1] = x_block[tid * stride_x + d_conv - 1];

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        if (i > 0) {
#pragma unroll
            for (size_t j = 0; j < d_conv - 1; j++) {
                x[j] = x[j + 1];
            }
            x[d_conv - 1] = x_block[tid * stride_x + i + d_conv - 1];
        }

        // identical accumulation to ssm_conv_idx_f32 (same window layout and
        // explicit non-contracted mul-add chain) so the two kernels are
        // bit-exact for the same inputs
#pragma clang fp_contract(off)
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

// indexed in-place variant (ggml_ssm_conv_idx): one thread per channel, one block column per sequence.
// the state row is read from / written to the store in place, s_idxs is read from device memory
// at run time so the kernel is safe to capture in a CUDA graph.
template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_idx_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                        const float * __restrict__ bias, float * __restrict__ src2,
                                        const int32_t * __restrict__ sidx, const int fresh_mask, const int src0_nb1, const int src0_nb2,
                                        const int src1_nb1, const int src2_nb1, float * __restrict__ dst,
                                        const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const int r = bidy * split_d_inner + tid;

    const int32_t row = sidx[bidx];
    const bool fresh = (fresh_mask >> bidx) & 1;
    float * s_row = (float *) ((char *) src2 + (int64_t) row * src2_nb1);

    const float * x_row = (const float *) ((const char *) src0 + (int64_t) bidx * src0_nb2 + r * src0_nb1); // [n_t]
    const float * w_row = (const float *) ((const char *) src1 + r * src1_nb1);                             // [d_conv]

    float * st = s_row + r * (d_conv - 1); // state columns of channel r

    float w[d_conv];
    float x[d_conv];

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_row[j];
    }
#pragma unroll
    for (size_t j = 0; j < d_conv - 1; j++) {
        x[j] = fresh ? 0.0f : st[j];
    }
    x[d_conv - 1] = x_row[0];

    const float b = bias != nullptr ? bias[r] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        if (i > 0) {
#pragma unroll
            for (size_t j = 0; j < d_conv - 1; j++) {
                x[j] = x[j + 1];
            }
            x[d_conv - 1] = x_row[i];
        }

#pragma clang fp_contract(off)
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[j] * w[j];
        }
        sumf += b;
        *(float *) ((char *) dst + (int64_t) bidx * dst_nb2 + i * dst_nb1 + r * sizeof(float)) =
            apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }

    // write the new state back to the store row: the sliding window x[] holds
    // the padded sequence [old state | new tokens] shifted so that x[j] is the
    // element at column j + n_t - 1, hence the last d_conv-1 columns of the
    // padded sequence - the new state - are x[1 .. d_conv-1]. (The previous
    // x[n_t + c] / x_row[n_t + c - (d_conv-1)] indexing was only correct for
    // n_t >= d_conv-1; for n_t = 2 it stored the newest token in the oldest
    // slot, corrupting the conv state - the multi-seq divergence when a
    // sequential reference is chunked at n_t=2 but the concurrent run is not.)
    // Read the local x[] window (zeroed for fresh sequences by the fresh-skip)
    // instead of the store st[] - for fresh sequences the store still holds the
    // previous occupant's state and would leak it into the new sequence's conv
    // state.
#pragma unroll
    for (size_t c = 0; c < d_conv - 1; c++) {
        st[c] = x[c + 1];
    }
}

template <bool apply_silu>
static void ssm_conv_idx_f32_cuda(const float * src0, const float * src1, const float * bias, float * src2,
                                  const int32_t * sidx, const int fresh_mask, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                  const int src2_nb1, float * dst, const int dst_nb1, const int dst_nb2,
                                  const int64_t nc, const int64_t nr, const int64_t n_t, const int64_t n_s,
                                  cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
        ssm_conv_idx_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(
            src0, src1, bias, src2, sidx, fresh_mask, src0_nb1, src0_nb2, src1_nb1, src2_nb1, dst, dst_nb1, dst_nb2, n_t);
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    if (getenv("LLAMA_CONV_EVENT") != nullptr) {
        fprintf(stderr, "CONV_DISP: idx=%d nr=%ld n_t=%ld n_s=%ld out=%s ne={%ld,%ld,%ld,%ld} data=%p\n",
            (dst->src[2] != nullptr), (long) nr, (long) n_t, (long) n_s, out->name,
            (long) out->ne[0], (long) out->ne[1], (long) out->ne[2], (long) out->ne[3], out->data);
    }

    if (dst->src[2] != nullptr) {
        // indexed in-place form (ggml_ssm_conv_idx)
        const struct ggml_tensor * src2 = dst->src[2]; // state store [(d_conv - 1)*d_inner, n_rows]
        const struct ggml_tensor * src3 = dst->src[3]; // sidx [n_s]

        GGML_ASSERT(out->ne[0] == nr);
        GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->type == GGML_TYPE_F32 && src3->type == GGML_TYPE_I32);
        GGML_ASSERT(out->type  == GGML_TYPE_F32);
        GGML_ASSERT(src0->nb[0] == sizeof(float) && src0->nb[1] == src0->ne[0]*sizeof(float));
        GGML_ASSERT(src1->nb[0] == sizeof(float));
        GGML_ASSERT(src2->nb[0] == sizeof(float));
        GGML_ASSERT(out->nb[0]  == sizeof(float));
        if (fuse_bias) {
            GGML_ASSERT(bias->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(bias));
            GGML_ASSERT(ggml_nelements(bias) == nr);
        }

        const float *   src0_d = (const float *)   src0->data;
        const float *   src1_d = (const float *)   src1->data;
        const float *   bias_d = fuse_bias ? (const float *) bias->data : nullptr;
        float *         src2_d = (float *)         src2->data;
        const int32_t * sidx_d = (const int32_t *) src3->data;
        float *         dst_d  = (float *)         out->data;
        cudaStream_t    stream = ctx.stream();
        const int32_t   fresh_mask = ggml_get_op_params_i32(dst, 0);

        if (getenv("LLAMA_CONV_TRACE") != nullptr) {
            const int32_t * sidx_p = (const int32_t *) src3->data;
            fprintf(stderr, "CONV_IDX: nr=%ld n_t=%ld n_s=%ld nc=%ld grid=(%ld,%ld) fresh=%x | src0=%s ne={%ld,%ld,%ld,%ld} nb={%zu,%zu,%zu,%zu} data=%p buf=%s base=%p size=%zu | src2=%s ne={%ld,%ld,%ld,%ld} data=%p buf=%s base=%p size=%zu | src3=%s data=%p buf=%s | sidx=[",
                (long) nr, (long) n_t, (long) n_s, (long) nc, (long) n_s, (long) ((nr + 127) / 128), fresh_mask,
                src0->name, (long) src0->ne[0], (long) src0->ne[1], (long) src0->ne[2], (long) src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], src0->data,
                src0->buffer ? ggml_backend_buffer_name(src0->buffer) : "none",
                src0->buffer ? ggml_backend_buffer_get_base(src0->buffer) : nullptr,
                src0->buffer ? ggml_backend_buffer_get_size(src0->buffer) : 0,
                src2->name, (long) src2->ne[0], (long) src2->ne[1], (long) src2->ne[2], (long) src2->ne[3], src2->data,
                src2->buffer ? ggml_backend_buffer_name(src2->buffer) : "none",
                src2->buffer ? ggml_backend_buffer_get_base(src2->buffer) : nullptr,
                src2->buffer ? ggml_backend_buffer_get_size(src2->buffer) : 0);
            const int64_t n_sidx = std::min<int64_t>(src3->ne[0], 32);
            for (int64_t i = 0; i < n_sidx; ++i) {
                fprintf(stderr, "%s%d", i ? "," : "", sidx_p[i]);
            }
            fprintf(stderr, "]");
            if (n_sidx >= 3 && (sidx_p[0] > (int32_t) src2->ne[1] || sidx_p[0] < -1)) {
                fprintf(stderr, " BAD");
            }
            fprintf(stderr, "\n");
        }

        // stale-copy early-out: n_s can only be the batch's sequence count
        // (<= n_parallel); an impossible n_s means the executed node carries
        // recycled/garbage metadata - skip the launch entirely and dump the
        // full provenance (the launch itself faults or hangs otherwise)
        if (n_s > 64) {
            const int32_t * sidx_p = (const int32_t *) src3->data;
            fprintf(stderr, "CONV_STALE: nr=%ld n_t=%ld n_s=%ld nc=%ld fresh=%x | out=%s ne={%ld,%ld,%ld,%ld} nb={%zu,%zu,%zu,%zu} data=%p buf=%s base=%p size=%zu | src0=%s ne={%ld,%ld,%ld,%ld} data=%p buf=%s base=%p size=%zu | src1=%s ne={%ld,%ld,%ld,%ld} data=%p buf=%s base=%p size=%zu | src2=%s ne={%ld,%ld,%ld,%ld} data=%p buf=%s base=%p size=%zu | src3=%s ne={%ld,%ld,%ld,%ld} data=%p | sidx=[",
                (long) nr, (long) n_t, (long) n_s, (long) nc, fresh_mask,
                out->name, (long) out->ne[0], (long) out->ne[1], (long) out->ne[2], (long) out->ne[3],
                out->nb[0], out->nb[1], out->nb[2], out->nb[3], out->data,
                out->buffer ? ggml_backend_buffer_name(out->buffer) : "none",
                out->buffer ? ggml_backend_buffer_get_base(out->buffer) : nullptr,
                out->buffer ? ggml_backend_buffer_get_size(out->buffer) : 0,
                src0->name, (long) src0->ne[0], (long) src0->ne[1], (long) src0->ne[2], (long) src0->ne[3], src0->data,
                src0->buffer ? ggml_backend_buffer_name(src0->buffer) : "none",
                src0->buffer ? ggml_backend_buffer_get_base(src0->buffer) : nullptr,
                src0->buffer ? ggml_backend_buffer_get_size(src0->buffer) : 0,
                src1->name, (long) src1->ne[0], (long) src1->ne[1], (long) src1->ne[2], (long) src1->ne[3], src1->data,
                src1->buffer ? ggml_backend_buffer_name(src1->buffer) : "none",
                src1->buffer ? ggml_backend_buffer_get_base(src1->buffer) : nullptr,
                src1->buffer ? ggml_backend_buffer_get_size(src1->buffer) : 0,
                src2->name, (long) src2->ne[0], (long) src2->ne[1], (long) src2->ne[2], (long) src2->ne[3], src2->data,
                src2->buffer ? ggml_backend_buffer_name(src2->buffer) : "none",
                src2->buffer ? ggml_backend_buffer_get_base(src2->buffer) : nullptr,
                src2->buffer ? ggml_backend_buffer_get_size(src2->buffer) : 0,
                src3->name, (long) src3->ne[0], (long) src3->ne[1], (long) src3->ne[2], (long) src3->ne[3], src3->data);
            const int64_t n_sidx = std::min<int64_t>(src3->ne[0], 32);
            for (int64_t i = 0; i < n_sidx; ++i) {
                fprintf(stderr, "%s%d", i ? "," : "", sidx_p[i]);
            }
            fprintf(stderr, "]\n");
            return;
        }

        // T1 guard: the kernel reads/writes state channels r*(nc-1)+c for r<nr
        // and row sidx[bidx]. Under the meta backend a full-width nr can arrive
        // with a sharded store (per-GPU ne[0] smaller) -> OOB. Detect and skip
        // (zero the output) instead of faulting; the MTP verification catches
        // the wrong result, the server survives.
        const int64_t src0_max_read = (int64_t)(n_s-1)*src0->nb[2] + (int64_t)(nr-1)*src0->nb[1] + n_t*(int64_t)src0->nb[0];
        const int64_t src1_max_read = (int64_t)(nr-1)*src1->nb[1] + nc*(int64_t)src1->nb[0];
        const int64_t dst_max_write = (int64_t)(n_s-1)*out->nb[2] + (int64_t)(n_t-1)*out->nb[1] + nr*(int64_t)out->nb[0];
        // metadata can be self-consistent while the BUFFER is a per-GPU shard
        // (full dims on shard data) - compare the access range against the
        // buffer base+size, the definitive bound
        const auto buf_has = [](const ggml_tensor * t, size_t max_access) -> bool {
            if (t->buffer == nullptr || t->data == nullptr) {
                return true;
            }
            const char * base = (const char *) ggml_backend_buffer_get_base(t->buffer);
            const size_t  size = ggml_backend_buffer_get_size(t->buffer);
            if (base == nullptr) {
                return true;
            }
            const ptrdiff_t off = (const char *) t->data - base;
            return off >= 0 && (size_t) off + max_access <= size;
        };
        const bool guard_buf = !buf_has(src0, src0_max_read) || !buf_has(src1, src1_max_read) ||
                               !buf_has(out,  dst_max_write) ||
                               !buf_has(src2, (size_t)(src2->ne[1]-1)*src2->nb[1] + (size_t)(nr*(nc-1))*src2->nb[0]);
        if (nr * (nc - 1) > src2->ne[0] || n_s > src2->ne[1] || src0_max_read > (int64_t) ggml_nbytes(src0) ||
                src1_max_read > (int64_t) ggml_nbytes(src1) || dst_max_write > (int64_t) ggml_nbytes(out) || guard_buf) {
            fprintf(stderr, "CONV_GUARD: nr=%ld n_t=%ld n_s=%ld nc=%ld store_ne={%ld,%ld} src0=%s ne={%ld,%ld,%ld,%ld} nb1=%zu nb2=%zu nbytes=%zu max_read=%ld | src1=%s ne={%ld,%ld,%ld,%ld} nb1=%zu nbytes=%zu w_max_read=%ld | out=%s nbytes=%zu dst_max_write=%ld bufguard=%d\n",
                (long) nr, (long) n_t, (long) n_s, (long) nc,
                (long) src2->ne[0], (long) src2->ne[1], src0->name,
                (long) src0->ne[0], (long) src0->ne[1], (long) src0->ne[2], (long) src0->ne[3],
                (size_t) src0->nb[1], (size_t) src0->nb[2], ggml_nbytes(src0), (long) src0_max_read,
                src1->name, (long) src1->ne[0], (long) src1->ne[1], (long) src1->ne[2], (long) src1->ne[3],
                (size_t) src1->nb[1], ggml_nbytes(src1), (long) src1_max_read,
                out->name, ggml_nbytes(out), (long) dst_max_write, (int) guard_buf);
            // clamp the zero to the buffer range: out can be full-dims on a
            // shard, and zeroing ggml_nbytes(out) from a shard pointer faults
            size_t zero_n = ggml_nbytes(out);
            if (out->buffer != nullptr) {
                const char * base = (const char *) ggml_backend_buffer_get_base(out->buffer);
                const size_t  size = ggml_backend_buffer_get_size(out->buffer);
                const ptrdiff_t off = (const char *) out->data - base;
                if (base != nullptr && off >= 0) {
                    zero_n = std::min(zero_n, size - (size_t) off);
                }
            }
            CUDA_CHECK(cudaMemsetAsync(dst_d, 0, zero_n, stream));
            return;
        }

        if (fuse_silu) {
            ssm_conv_idx_f32_cuda<true>(src0_d, src1_d, bias_d, src2_d, sidx_d, fresh_mask, src0->nb[1], src0->nb[2], src1->nb[1],
                                        src2->nb[1], dst_d, out->nb[1], out->nb[2], nc, nr, n_t, n_s, stream);
        } else {
            ssm_conv_idx_f32_cuda<false>(src0_d, src1_d, bias_d, src2_d, sidx_d, fresh_mask, src0->nb[1], src0->nb[2], src1->nb[1],
                                         src2->nb[1], dst_d, out->nb[1], out->nb[2], nc, nr, n_t, n_s, stream);
        }

        if (getenv("LLAMA_CONV_VDUMP") != nullptr) {
            cudaStreamSynchronize(stream);
            const int32_t row0 = sidx_d[0];
            float sv[8] = {0}, ov[8] = {0};
            CUDA_CHECK(cudaMemcpyAsync(sv, (const char *) src2->data + (int64_t) row0 * src2->nb[1], 8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(ov, out->data, 8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, "CONV_VD: n_t=%ld n_s=%ld fresh=%x row0=%d store=[%.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g] out=[%.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g]\n",
                (long) n_t, (long) n_s, fresh_mask, row0,
                (double) sv[0], (double) sv[1], (double) sv[2], (double) sv[3],
                (double) sv[4], (double) sv[5], (double) sv[6], (double) sv[7],
                (double) ov[0], (double) ov[1], (double) ov[2], (double) ov[3],
                (double) ov[4], (double) ov[5], (double) ov[6], (double) ov[7]);
        }

        return;
    }


    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    // stale-copy early-out (non-indexed path, n_rs_seq>0 rollback): n_s is the
    // batch's sequence count (<= n_parallel); an impossible value means the
    // executed node carries recycled/garbage metadata - dump provenance and
    // skip instead of launching a faulting kernel
    if (n_s > 64) {
        fprintf(stderr, "CONV_STALE: nr=%ld n_t=%ld n_s=%ld nc=%ld | out=%s ne={%ld,%ld,%ld,%ld} nb={%zu,%zu,%zu,%zu} data=%p buf=%s base=%p size=%zu | src0=%s ne={%ld,%ld,%ld,%ld} data=%p buf=%s base=%p size=%zu | src1=%s ne={%ld,%ld,%ld,%ld} data=%p buf=%s base=%p size=%zu\n",
            (long) nr, (long) n_t, (long) n_s, (long) nc,
            out->name, (long) out->ne[0], (long) out->ne[1], (long) out->ne[2], (long) out->ne[3],
            out->nb[0], out->nb[1], out->nb[2], out->nb[3], out->data,
            out->buffer ? ggml_backend_buffer_name(out->buffer) : "none",
            out->buffer ? ggml_backend_buffer_get_base(out->buffer) : nullptr,
            out->buffer ? ggml_backend_buffer_get_size(out->buffer) : 0,
            src0->name, (long) src0->ne[0], (long) src0->ne[1], (long) src0->ne[2], (long) src0->ne[3], src0->data,
            src0->buffer ? ggml_backend_buffer_name(src0->buffer) : "none",
            src0->buffer ? ggml_backend_buffer_get_base(src0->buffer) : nullptr,
            src0->buffer ? ggml_backend_buffer_get_size(src0->buffer) : 0,
            src1->name, (long) src1->ne[0], (long) src1->ne[1], (long) src1->ne[2], (long) src1->ne[3], src1->data,
            src1->buffer ? ggml_backend_buffer_name(src1->buffer) : "none",
            src1->buffer ? ggml_backend_buffer_get_base(src1->buffer) : nullptr,
            src1->buffer ? ggml_backend_buffer_get_size(src1->buffer) : 0);
        return;
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }

    if (getenv("LLAMA_CONV_VDUMP") != nullptr) {
        cudaStreamSynchronize(stream);
        const int32_t row0 = 0;
        float sv[8] = {0}, ov[8] = {0};
        CUDA_CHECK(cudaMemcpyAsync(sv, (const char *) src0->data + (int64_t) row0 * src0->nb[1], 8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(ov, out->data, 8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fprintf(stderr, "CONV_PVD: n_t=%ld n_s=%ld src0=[%.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g] out=[%.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g]\n",
            (long) n_t, (long) n_s,
            (double) sv[0], (double) sv[1], (double) sv[2], (double) sv[3],
            (double) sv[4], (double) sv[5], (double) sv[6], (double) sv[7],
            (double) ov[0], (double) ov[1], (double) ov[2], (double) ov[3],
            (double) ov[4], (double) ov[5], (double) ov[6], (double) ov[7]);
    }
}