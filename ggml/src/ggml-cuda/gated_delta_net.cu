#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

// IDX: curr_state is the full recurrent store [S_v, S_v, H, n_rows] (f32 or f16) and the initial
// state of batch seq `sequence` is read from store row s_idxs[sequence] (read from device memory,
// never baked, so the kernel stays CUDA-graph capture safe). state_ip (K == 1 only): write the
// final state back into the store row it was read from (converting to f16 for an f16 store)
// instead of the output snapshot area, which is then left unwritten.
template <int S_v, bool KDA, bool keep_rs_t, bool IDX, bool STATE_F16>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 1)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const void *  curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * s_idxs,
                                     int           state_ip,
                                     int           fresh_mask) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state: dense [S_v, S_v, H, n_seqs] row `sequence`, or (IDX) the full store
    // [S_v, S_v, H, n_rows] row s_idxs[sequence] — seq stride is D = H * S_v * S_v either way.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    int64_t iseq = sequence;
    if constexpr (IDX) {
        iseq = s_idxs[sequence];
    }
    const int64_t state_in_offset      = iseq * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    // fresh sequences (fresh_mask bit set) start from a zero state: skip the store read
    const bool fresh = (fresh_mask >> sequence) & 1;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        if (fresh) {
            s_shard[r] = 0.0f;
        } else if constexpr (STATE_F16) {
            s_shard[r] = __half2float(((const half *) curr_state)[state_in_offset + col * S_v + i]);
        } else {
            s_shard[r] = ((const float *) curr_state)[state_in_offset + col * S_v + i];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        if (IDX && state_ip) {
            // write the final state back into the store row it was read from; the output
            // snapshot area is left unwritten. Distinct s_idxs rows => no write conflicts.
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                if constexpr (STATE_F16) {
                    ((half *) curr_state)[state_in_offset + col * S_v + i] = __float2half(s_shard[r]);
                } else {
                    ((float *) curr_state)[state_in_offset + col * S_v + i] = s_shard[r];
                }
            }
        } else {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i          = r * warp_size + lane;
                state[col * S_v + i] = s_shard[r];
            }
        }
    }
}

template <bool KDA, bool keep_rs_t, bool IDX, bool STATE_F16>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const void * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K,
        const int32_t * sidx_d, int state_ip, int fresh_mask, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, IDX, STATE_F16>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, IDX, STATE_F16>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, IDX, STATE_F16>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, IDX, STATE_F16>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const void *  s_d   = (const void *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, nullptr, 0, 0, stream);
        } else {
            launch_gated_delta_net<true, false, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, nullptr, 0, 0, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, nullptr, 0, 0, stream);
        } else {
            launch_gated_delta_net<false, false, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, nullptr, 0, 0, stream);
        }
    }
}

// indexed variant (GGML_OP_GATED_DELTA_NET_IDX): src[5] is the full recurrent store
// [S_v, S_v, H, n_rows] (f32 or f16), src[6] the I32 store-row indices s_idxs[n_seqs].
// state_ip (K == 1 only) writes the final state back into the store rows in place.
static void ggml_cuda_op_gated_delta_net_idx_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (getenv("LLAMA_GDN_CALLTRACE") != nullptr) {
        fprintf(stderr, "GDN_CALL: dst=%s nt=%lld ns=%lld state=%s\n",
            dst->name, (long long) dst->src[0]->ne[2], (long long) dst->src[0]->ne[3],
            dst->src[5]->name);
    }
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];
    ggml_tensor * src_sidx  = dst->src[6];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda       = (src_g->ne[0] == S_v);
    const bool state_f16 = (src_state->type == GGML_TYPE_F16);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float *   q_d    = (const float *)   src_q->data;
    const float *   k_d    = (const float *)   src_k->data;
    const float *   v_d    = (const float *)   src_v->data;
    const float *   g_d    = (const float *)   src_g->data;
    const float *   b_d    = (const float *)   src_beta->data;
    const void *    s_d    = (const void *)    src_state->data;
    const int32_t * sidx_d = (const int32_t *) src_sidx->data;
    float *         dst_d  = (float *)         dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    if (!ggml_is_contiguous(src_state)) {
        fprintf(stderr, "GDN_STATE_NONCONTIG: name=%s ne={%ld,%ld,%ld,%ld} nb={%zu,%zu,%zu,%zu} data=%p vsrc=%p voffs=%zu\n",
            src_state->name,
            (long) src_state->ne[0], (long) src_state->ne[1], (long) src_state->ne[2], (long) src_state->ne[3],
            src_state->nb[0], src_state->nb[1], src_state->nb[2], src_state->nb[3],
            src_state->data, (void *) src_state->view_src, src_state->view_offs);
    }
    GGML_ASSERT(ggml_is_contiguous(src_state));
    GGML_ASSERT(src_state->type == GGML_TYPE_F32 || state_f16);
    GGML_ASSERT(src_sidx->type == GGML_TYPE_I32 && ggml_is_contiguous(src_sidx));
    GGML_ASSERT(src_sidx->ne[0] >= n_seqs);

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) and state_ip are op params; the state output layout is
    // dense by batch seq, same as the non-indexed op.
    const int K        = ggml_get_op_params_i32(dst, 0);
    const int state_ip = ggml_get_op_params_i32(dst, 1);
    const int fresh_mask = ggml_get_op_params_i32(dst, 2);
    const bool keep_rs = K > 1;
    GGML_ASSERT(!state_ip || K == 1);
    if (getenv("LLAMA_GDN_PTRS") != nullptr) {
        fprintf(stderr, "GDN_PTRS: nt=%lld ns=%lld sidx=%d fresh=%x ip=%d state=%p stateobj=%p state_ne={%ld,%ld,%ld,%ld} nb={%zu,%zu,%zu,%zu} view_src=%p offs=%zu\n",
            (long long) n_tokens, (long long) n_seqs, src_sidx->type == GGML_TYPE_I32 ? ((const int32_t *) src_sidx->data)[0] : -1,
            fresh_mask, state_ip, src_state->data, (void *) src_state,
            (long) src_state->ne[0], (long) src_state->ne[1], (long) src_state->ne[2], (long) src_state->ne[3],
            src_state->nb[0], src_state->nb[1], src_state->nb[2], src_state->nb[3],
            src_state->view_src, src_state->view_offs);
    }

    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;

    if (kda) {
        if (keep_rs) {
            if (state_f16) {
                launch_gated_delta_net<true, true, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask, stream);
            } else {
                launch_gated_delta_net<true, true, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask, stream);
            }
        } else {
            if (state_f16) {
                launch_gated_delta_net<true, false, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask, stream);
            } else {
                launch_gated_delta_net<true, false, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask, stream);
            }
        }
    } else {
        if (keep_rs) {
            if (state_f16) {
                launch_gated_delta_net<false, true, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask, stream);
            } else {
                launch_gated_delta_net<false, true, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask, stream);
            }
        } else {
            if (state_f16) {
                launch_gated_delta_net<false, false, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask, stream);
            } else {
                launch_gated_delta_net<false, false, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, sidx_d, state_ip, fresh_mask, stream);
            }
        }
    }

    // T1 debug: dump the state row the op just read/wrote (sync readback). The read is
    // the row state_in_offset for seq 0 (head 0) - the first floats of the row.
    // Restricted to layer 0 (the store's source tensor is named cache_s_l0).
    const char * gdn_dbg = getenv("LLAMA_GDN_DEBUG");
    const char * store_name = src_state->view_src != nullptr ? src_state->view_src->name : src_state->name;
    if (gdn_dbg != nullptr && strstr(store_name, "cache_s_l0") != nullptr) {
        cudaStreamCaptureStatus capture_status;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
        if (capture_status == cudaStreamCaptureStatusNone) {
            int32_t sidx0 = 0;
            CUDA_CHECK(cudaMemcpyAsync((void *) &sidx0, (const void *) sidx_d, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float dbg[4] = {0, 0, 0, 0};
            if (src_state->type == GGML_TYPE_F16) {
                half dbg_h[4] = {0, 0, 0, 0};
                CUDA_CHECK(cudaMemcpyAsync(dbg_h, (const char *) s_d + (int64_t) sidx0 * S_v * S_v * H * sizeof(half), 4*sizeof(half), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int i = 0; i < 4; ++i) dbg[i] = __half2float(dbg_h[i]);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dbg, (const char *) s_d + (int64_t) sidx0 * S_v * S_v * H * sizeof(float), 4*sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            fprintf(stderr, "GDN_STATE: nt=%lld ns=%lld sidx0=%d fresh=%x ip=%d K=%d row0=[%.4g %.4g %.4g %.4g]\n",
                (long long) n_tokens, (long long) n_seqs, sidx0, fresh_mask, state_ip, K,
                (double) dbg[0], (double) dbg[1], (double) dbg[2], (double) dbg[3]);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

void ggml_cuda_op_gated_delta_net_idx(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_idx_impl(ctx, dst);
}
