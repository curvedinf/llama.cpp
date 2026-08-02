#include "common.cuh"
#include "fattn-common.cuh"

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
// Block-table (paged attention) addressing: k/v for logical position p of a sequence lives at
//     flat row block_table[(p >> bt_log2_bs)*bt_nb0 + sequence*bt_nb1]*(1 << bt_log2_bs) + (p & ((1 << bt_log2_bs) - 1))
// of the k/v buffers. The table is read from device memory (CUDA-graph capture safe).
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool use_block_table> // D == head size
static __device__ __forceinline__ void flash_attn_ext_vec_impl(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const int32_t * block_table, const int64_t bt_nb0, const int64_t bt_nb1, const int32_t bt_log2_bs,
        const int32_t n_pool_blocks) {
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33,
            block_table, bt_nb0, bt_nb1, bt_log2_bs, n_pool_blocks);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr int nthreads_KQ = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_BF16;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // Paged attention: resolve a logical KV position to a physical row via the block table.
    // Entries outside [0, n_pool_blocks) (-1 "unused" sentinels, stale/garbage contents) resolve
    // to -1 and are treated as masked-out positions instead of addressing out of bounds.
    const int32_t bt_bs = 1 << bt_log2_bs;
    auto fattn_paged_row = [&](const int p) -> int64_t {
        const int32_t blk = block_table[(p >> bt_log2_bs)*bt_nb0 + sequence*bt_nb1];
        if (blk < 0 || blk >= n_pool_blocks) {
            return (int64_t) -1;
        }
        return (int64_t) blk*bt_bs + (p & (bt_bs - 1));
    };
    GGML_UNUSED(fattn_paged_row);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];

    ggml_cuda_pdl_sync();
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    if constexpr (!use_block_table) {
        K     += blockIdx.y*nthreads * nb11;
        V     += blockIdx.y*nthreads * nb21;
    }
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += use_block_table ? 0 : gridDim.y*nthreads*nb11,
             V += use_block_table ? 0 : gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

            // Paged attention: logical position p -> physical row; positions >= k_VKQ_max and
            // invalid block table entries are masked out below.
            int64_t k_row  = i_KQ;
            bool    k_valid = true;
            if constexpr (use_block_table) {
                const int p = k_VKQ_0 + i_KQ;
                k_row   = p < k_VKQ_max ? fattn_paged_row(p) : (int64_t) -1;
                k_valid = k_row >= 0;
                k_row   = k_valid ? k_row : 0;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(use_block_table ? K + k_row*nb11 : K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (use_block_table && !k_valid) {
                    sum = -INFINITY;
                }

                if (mask && k_valid && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope*__half2float(maskh[j*ne11 + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif // GGML_USE_HIP

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

            // Paged attention: logical position p -> physical row; invalid positions have KQ == 0.
            int64_t v_row = k;
            if constexpr (use_block_table) {
                const int p = k_VKQ_0 + k;
                const int64_t bt_row = p < k_VKQ_max ? fattn_paged_row(p) : (int64_t) -1;
                v_row = bt_row >= 0 ? bt_row : 0;
            }
            const char * V_k = use_block_table ? V + v_row*nb21 : V + k*nb21;

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    dequantize_V(V_k, tmp_f,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else {
                    dequantize_V(V_k, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                dequantize_V(V_k, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                // the last block's columns can be past the real token count (ne01.z is not
                // padded to a multiple of ncols); write them nowhere - the reads above are
                // guarded the same way and the mask zeroes the padded KQ sums
                if (ncols == 1 || ic0 + j_VKQ < int(ne01.z)) {
                    dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
                }
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33,
        block_table, bt_nb0, bt_nb1, bt_log2_bs, n_pool_blocks);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    flash_attn_ext_vec_impl<D, ncols, type_K, type_V, use_logit_softcap, false>(
        Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33,
        nullptr, 0, 0, 0, 0);
}

// Paged (block-table) variant of flash_attn_ext_vec; same signature plus the block table descriptor.
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec_paged(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const int32_t * block_table, const int64_t bt_nb0, const int64_t bt_nb1, const int32_t bt_log2_bs,
        const int32_t n_pool_blocks) {
    flash_attn_ext_vec_impl<D, ncols, type_K, type_V, use_logit_softcap, true>(
        Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33,
        block_table, bt_nb0, bt_nb1, bt_log2_bs, n_pool_blocks);
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

// -------------------------------------------------------------------------
// Paged (block-table) flash attention.
//
// When dst->src[5] != nullptr, K/V are physical block pools and the row for logical KV position p
// of sequence s is resolved inside the kernel from device memory (CUDA-graph capture safe):
//     physical row = block_table[(p >> 5)*bt_nb0 + s*bt_nb1]*32 + (p & 31)
// with bt_nb0/bt_nb1 being the block table strides in elements (block size is fixed at 32).
// The logical KV length is taken from mask->ne[0]; K->ne[1]/V->ne[1] is the physical pool size.
//

#define FATTN_PAGED_BLOCK_SIZE_LOG2 5

typedef void (* fattn_paged_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const int32_t * block_table, const int64_t bt_nb0, const int64_t bt_nb1, const int32_t bt_log2_bs,
        const int32_t n_pool_blocks);

// Host-side helper: the number of parallel KV blocks the paged launcher will use for this op.
// Deterministic for a given (device, dst) because the op's scratch allocation is sized from it.
template <int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
static int ggml_cuda_fattn_vec_paged_parallel_blocks_impl(const int device, const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * mask = dst->src[3];

    const int n_kv_log   = mask->ne[0];
    const int ntiles_x   = ((Q->ne[1] + ncols - 1) / ncols);
    const int ntiles_dst = ntiles_x * Q->ne[2] * Q->ne[3];
    const int nsm        = ggml_cuda_info().devices[device].nsm;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(ggml_cuda_info().devices[device].cc);
    const dim3 block_dim(WARP_SIZE, nthreads / WARP_SIZE, 1);
    fattn_paged_kernel_t fattn_kernel = flash_attn_ext_vec_paged<D, ncols, type_K, type_V, use_logit_softcap>;
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, 0));
    GGML_ASSERT(max_blocks_per_sm > 0);

    const int ntiles_KV = (n_kv_log + D - 1) / D; // Max. number of parallel blocks limited by KV cache length.

    // parallel_blocks must not be larger than what the tensor size allows:
    int parallel_blocks = std::min(max_blocks_per_sm, ntiles_KV);

    // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
    // Test whether parallel_blocks can be set to a higher value for better efficiency.
    const int blocks_per_wave = nsm * max_blocks_per_sm;
    int nwaves_best = 0;
    int efficiency_percent_best = 0;
    for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
        const int nblocks_total = ntiles_dst * parallel_blocks_test;
        const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
        const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

        // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
        if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
            break;
        }

        if (efficiency_percent > efficiency_percent_best) {
            nwaves_best = nwaves;
            efficiency_percent_best = efficiency_percent;
            parallel_blocks = parallel_blocks_test;
        }
    }

    return parallel_blocks;
}

// Dispatch over the supported paged instantiations; used both when sizing the op's scratch
// allocation and when launching, so the two always agree.
static int ggml_cuda_fattn_vec_paged_parallel_blocks(const int device, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int  old_device = ggml_cuda_get_device();
    if (old_device != device) {
        ggml_cuda_set_device(device);
    }

    int parallel_blocks = -1;
    if (Q->ne[1] == 1) {
        if (K->type == GGML_TYPE_F16) {
            parallel_blocks = logit_softcap == 0.0f ?
                (Q->ne[0] == 128 ? ggml_cuda_fattn_vec_paged_parallel_blocks_impl<128, 1, GGML_TYPE_F16,  GGML_TYPE_F16,  false>(device, dst) :
                                   ggml_cuda_fattn_vec_paged_parallel_blocks_impl<256, 1, GGML_TYPE_F16,  GGML_TYPE_F16,  false>(device, dst)) :
                (Q->ne[0] == 128 ? ggml_cuda_fattn_vec_paged_parallel_blocks_impl<128, 1, GGML_TYPE_F16,  GGML_TYPE_F16,  true >(device, dst) :
                                   ggml_cuda_fattn_vec_paged_parallel_blocks_impl<256, 1, GGML_TYPE_F16,  GGML_TYPE_F16,  true >(device, dst));
        } else {
            parallel_blocks = logit_softcap == 0.0f ?
                (Q->ne[0] == 128 ? ggml_cuda_fattn_vec_paged_parallel_blocks_impl<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>(device, dst) :
                                   ggml_cuda_fattn_vec_paged_parallel_blocks_impl<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>(device, dst)) :
                (Q->ne[0] == 128 ? ggml_cuda_fattn_vec_paged_parallel_blocks_impl<128, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true >(device, dst) :
                                   ggml_cuda_fattn_vec_paged_parallel_blocks_impl<256, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true >(device, dst));
        }
    } else {
        if (K->type == GGML_TYPE_F16) {
            parallel_blocks = logit_softcap == 0.0f ?
                (Q->ne[0] == 128 ? ggml_cuda_fattn_vec_paged_parallel_blocks_impl<128, 2, GGML_TYPE_F16,  GGML_TYPE_F16,  false>(device, dst) :
                                   ggml_cuda_fattn_vec_paged_parallel_blocks_impl<256, 2, GGML_TYPE_F16,  GGML_TYPE_F16,  false>(device, dst)) :
                (Q->ne[0] == 128 ? ggml_cuda_fattn_vec_paged_parallel_blocks_impl<128, 2, GGML_TYPE_F16,  GGML_TYPE_F16,  true >(device, dst) :
                                   ggml_cuda_fattn_vec_paged_parallel_blocks_impl<256, 2, GGML_TYPE_F16,  GGML_TYPE_F16,  true >(device, dst));
        } else {
            parallel_blocks = logit_softcap == 0.0f ?
                (Q->ne[0] == 128 ? ggml_cuda_fattn_vec_paged_parallel_blocks_impl<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>(device, dst) :
                                   ggml_cuda_fattn_vec_paged_parallel_blocks_impl<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>(device, dst)) :
                (Q->ne[0] == 128 ? ggml_cuda_fattn_vec_paged_parallel_blocks_impl<128, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true >(device, dst) :
                                   ggml_cuda_fattn_vec_paged_parallel_blocks_impl<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true >(device, dst));
        }
    }

    if (old_device != device) {
        ggml_cuda_set_device(old_device);
    }

    return parallel_blocks;
}

// Paged scratch appended to the op's own tensor allocation (like the f16 conversion scratch).
// Unlike the per-stream ggml_cuda_pool, this memory is part of the graph's tensor buffer and stays
// valid for the graph's lifetime: pool buffers can be recycled or flushed (clear_pool on OOM)
// between graph capture and replay, which turns captured scratch pointers into use-after-free.
static size_t ggml_cuda_fattn_vec_paged_scratch_nbytes(const int device, const ggml_tensor * dst) {
    const int parallel_blocks = ggml_cuda_fattn_vec_paged_parallel_blocks(device, dst);
    if (parallel_blocks <= 1) {
        return 0;
    }
    return (size_t) parallel_blocks * (ggml_nelements(dst)*sizeof(float) + ggml_nrows(dst)*sizeof(float2));
}

// Offset of the paged scratch area relative to dst->data (paged K/V are never converted to f16,
// so the f16 scratch area is empty and the paged scratch sits right after the padded tensor data).
static inline size_t ggml_cuda_fattn_vec_paged_scratch_offset(const ggml_tensor * dst) {
    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, false, false);
    return (size_t) (GGML_PAD(f16_extra.end, 128) - (uintptr_t) dst->data);
}

// Restricted variant of launch_fattn for the paged vector kernel: no f16 conversion of K/V,
// no mask-to-KV_max scan, no stream-k. ne11 is the logical KV length (mask->ne[0]).
template <int D, int ncols>
static void launch_fattn_vec_paged(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_paged_kernel_t fattn_kernel, const int nwarps
) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * bt    = dst->src[5];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(mask && mask->type == GGML_TYPE_F16);
    GGML_ASSERT(bt != nullptr && bt->type == GGML_TYPE_I32);
    GGML_ASSERT(bt->ne[2] == 1 && bt->ne[3] == 1);
    GGML_ASSERT(bt->nb[0] == sizeof(int32_t));
    GGML_ASSERT(bt->ne[1] == Q->ne[3]);
    GGML_ASSERT(K->ne[1] % (1 << FATTN_PAGED_BLOCK_SIZE_LOG2) == 0);
    GGML_ASSERT(V->ne[1] == K->ne[1]);

    cudaStream_t main_stream = ctx.stream();

    const int n_kv_log = mask->ne[0]; // Logical KV length; the physical pool size is K->ne[1].

    const int ntiles_x = ((Q->ne[1] + ncols - 1) / ncols);

    // Same value that ggml_cuda_flash_attn_ext_get_alloc_size reserved scratch for:
    const int parallel_blocks = ggml_cuda_fattn_vec_paged_parallel_blocks(ggml_cuda_get_device(), dst);

    const dim3 block_dim(WARP_SIZE, nwarps, 1);
    const dim3 blocks_num(ntiles_x, parallel_blocks, Q->ne[2]*Q->ne[3]);

    if (getenv("LLAMA_FA_TRACE") != nullptr) {
        fprintf(stderr, "FA_PAGED: Q ne={%ld,%ld,%ld,%ld} nb={%zu,%zu,%zu,%zu} data=%p | K ne={%ld,%ld,%ld,%ld} data=%p | bt ne={%ld,%ld,%ld,%ld} nb={%zu,%zu,%zu,%zu} data=%p | mask ne0=%ld | grid={%d,%d,%d} n_kv_log=%d dst_nbytes=%zu\n",
            (long) Q->ne[0], (long) Q->ne[1], (long) Q->ne[2], (long) Q->ne[3],
            Q->nb[0], Q->nb[1], Q->nb[2], Q->nb[3], (const void *) Q->data,
            (long) K->ne[0], (long) K->ne[1], (long) K->ne[2], (long) K->ne[3], (const void *) K->data,
            (long) dst->src[5]->ne[0], (long) dst->src[5]->ne[1], (long) dst->src[5]->ne[2], (long) dst->src[5]->ne[3],
            dst->src[5]->nb[0], dst->src[5]->nb[1], dst->src[5]->nb[2], dst->src[5]->nb[3], (const void *) dst->src[5]->data,
            (long) mask->ne[0], blocks_num.x, blocks_num.y, blocks_num.z, n_kv_log, (size_t) ggml_nbytes(KQV));
    }

    // Scratch from the op's own tensor allocation (graph-capture safe, see above):
    char   * scratch      = (char *) KQV->data + ggml_cuda_fattn_vec_paged_scratch_offset(dst);
    float  * dst_tmp      = (float  *) scratch;
    float2 * dst_tmp_meta = (float2 *) (scratch + (size_t) parallel_blocks*ggml_nelements(KQV)*sizeof(float));

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % WARP_SIZE == 0);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, 0, main_stream);
    ggml_cuda_kernel_launch(fattn_kernel, launch_params,
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        (const char *) mask->data,
        sinks ? ((const char *) sinks->data) : nullptr,
        (const int *) nullptr, // KV_max
        parallel_blocks > 1 ? dst_tmp : (float *) KQV->data, parallel_blocks > 1 ? dst_tmp_meta : nullptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], n_kv_log, K->ne[2], K->ne[3], K->nb[1], K->nb[2], K->nb[3],
        V->nb[1], V->nb[2], V->nb[3],
        mask->ne[1], mask->ne[2], mask->ne[3],
        mask->nb[1], mask->nb[2], mask->nb[3],
        (const int32_t *) bt->data, (int64_t) bt->nb[0]/sizeof(int32_t), (int64_t) bt->nb[1]/sizeof(int32_t),
        FATTN_PAGED_BLOCK_SIZE_LOG2, (int32_t) (K->ne[1] >> FATTN_PAGED_BLOCK_SIZE_LOG2));
    CUDA_CHECK(cudaGetLastError());

    if (parallel_blocks > 1) {
        const dim3 block_dim_combine(D, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params_combine =
            ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<D>, launch_params_combine,
            dst_tmp, dst_tmp_meta, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
static void ggml_cuda_flash_attn_ext_vec_paged_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_paged_kernel_t fattn_kernel = flash_attn_ext_vec_paged<D, ncols, type_K, type_V, use_logit_softcap>;
    launch_fattn_vec_paged<D, ncols>(ctx, dst, fattn_kernel, nwarps);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_paged_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_paged_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_paged_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_paged_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_paged_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)
