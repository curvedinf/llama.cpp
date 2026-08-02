#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*(1 + n_stream)*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;

        paged = other->paged;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;

    const char * LLAMA_PREFIX_CACHE_DISABLE = getenv("LLAMA_PREFIX_CACHE_DISABLE");
    prefix_enabled = LLAMA_PREFIX_CACHE_DISABLE ? !atoi(LLAMA_PREFIX_CACHE_DISABLE) : true;

    // env: LLAMA_KV_PAGED - GPU-side paged attention: the flash-attention kernel receives a
    // vLLM-style block table (src[5] of GGML_OP_FLASH_ATTN_EXT) and gathers K/V through it.
    // switches the cell allocator to seq-aligned blocks and the K/V views to full width.
    // default OFF. requires flash attention, non-SWA and a HIP(ROCm)/CUDA KV buffer - the
    // CPU backend asserts on src[5] != nullptr, so refuse to enable anywhere else.
    if (!other) {
        const char * LLAMA_KV_PAGED = getenv("LLAMA_KV_PAGED");

        paged = LLAMA_KV_PAGED && atoi(LLAMA_KV_PAGED) != 0;
    }

    if (paged) {
        bool ok = true;

        if (v_trans) {
            LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED requires flash attention - falling back to the legacy path\n", __func__);
            ok = false;
        }

        if (n_swa > 0) {
            LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED is only supported for non-SWA caches - falling back to the legacy path\n", __func__);
            ok = false;
        }

        if (n_stream != 1) {
            // the block table addresses the physical pool of a single (unified) stream
            LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED requires a unified KV cache (-kvu) - falling back to the legacy path\n", __func__);
            ok = false;
        }

        if (kv_size % llama_kv_cells::block_size != 0) {
            LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED requires kv_size = %u to be a multiple of the block size %u - falling back to the legacy path\n",
                    __func__, kv_size, llama_kv_cells::block_size);
            ok = false;
        }

        // the HIP/CUDA paged FA kernel currently supports head size 128/256 and F16/Q8_0 K/V only
        // (mirrors ggml_cuda_fattn_paged_supported - attaching the block table for any other
        // configuration would make the op unsupported and crash the CPU fallback)
        const bool head_ok = (n_embd_head_k_all == 128 || n_embd_head_k_all == 256) &&
                             (n_embd_head_v_all == 128 || n_embd_head_v_all == 256) &&
                             n_embd_head_k_all == n_embd_head_v_all;
        if (!head_ok) {
            LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED requires head size 128 or 256 (got k = %d, v = %d) - falling back to the legacy path\n",
                    __func__, n_embd_head_k_all, n_embd_head_v_all);
            ok = false;
        }

        if (type_k != type_v || (type_k != GGML_TYPE_F16 && type_k != GGML_TYPE_Q8_0)) {
            LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED requires F16 or Q8_0 K/V cache types (got %s/%s) - falling back to the legacy path\n",
                    __func__, ggml_type_name(type_k), ggml_type_name(type_v));
            ok = false;
        }

        if (ok && !other) {
            for (const auto & [buft, ctx] : ctx_map) {
                const char * buft_name = ggml_backend_buft_name(buft);

                // "Meta" wraps per-device (simple) buffers of the compute backend, e.g.
                // the ROCm bufts under -sm tensor. The per-GPU paged FA nodes dispatch to
                // those simple backends, so the meta buft is as safe as the direct one -
                // the final say is the GPU-side support check at dispatch. Anything else
                // (the CPU backend in particular) asserts on a block table, so refuse.
                if (strncmp(buft_name, "ROCm", 4) != 0 && strncmp(buft_name, "CUDA", 4) != 0 &&
                        strncmp(buft_name, "Meta", 4) != 0) {
                    LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED is only supported on the HIP/CUDA backend (got buffer type '%s') - falling back to the legacy path\n",
                            __func__, buft_name);
                    ok = false;
                }
            }
        }

        paged = ok;

        if (paged) {
            LLAMA_LOG_INFO("%s: paged attention enabled (LLAMA_KV_PAGED): block size = %u, %u blocks/stream\n",
                    __func__, llama_kv_cells::block_size, kv_size/llama_kv_cells::block_size);
        }
    }

    chains.resize(LLAMA_MAX_SEQ);
}

void llama_kv_cache::clear(bool data) {
    prefix.clear();

    for (auto & chain : chains) {
        chain.reset();
    }

    inflight_valid = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (getenv("LLAMA_RS_DEBUG") != nullptr) {
        fprintf(stderr, "KV_SEQRM: seq=%d p0=%lld p1=%lld caller=%p\n", seq_id, (long long) p0, (long long) p1,
            (void *) __builtin_return_address(0));
    }

    if (seq_id >= 0) {
        // registered blocks are retained by the pinned cells - only the token chain is affected
        chains[seq_id].truncate(p0);

        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        const uint32_t new_head = cells.seq_rm_range(seq_id, p0, p1);

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence - hard removal, the registered blocks are unregistered first
        for (uint32_t s = 0; s < n_stream; ++s) {
            std::vector<llama_prefix_cache::loc_t> locs;
            prefix.remove_range(s, p0, p1, locs);
            prefix_unregister(locs);

            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            const uint32_t new_head = cells.rm_range(p0, p1);

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        // the destination sequence sees the same tokens - it can share the token chain
        chains[seq_id_dst] = chains[seq_id_src];

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t b = cells.block_lo(); b < cells.block_hi(); ++b) {
            if (cells.block_is_free(b) || cells.block_pos_max(b) < p0 || cells.block_pos_min(b) >= p1) {
                continue;
            }

            for (uint32_t i = cells.cell_begin(b); i < cells.cell_end(b); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                if (cells.seq_has(i, seq_id_src)) {
                    cells.seq_add(i, seq_id_dst);
                }
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // the destination stream is fully reset - unregister its cached blocks first
    {
        std::vector<llama_prefix_cache::loc_t> locs;
        prefix.remove_stream(s1, locs);
        prefix_unregister(locs);
    }

    chains[seq_id_dst].reset();

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();

    const auto & cells_src = v_cells[s0];

    for (uint32_t b = cells_src.block_lo(); b < cells_src.block_hi(); ++b) {
        if (cells_src.block_is_free(b)) {
            continue;
        }

        for (uint32_t i = cells_src.cell_begin(b); i < cells_src.cell_end(b); ++i) {
            if (!cells_src.seq_has(i, seq_id_src)) {
                continue;
            }

            llama_pos pos   = cells_src.pos_get(i);
            llama_pos shift = cells_src.get_shift(i);

            llama_kv_cell_ext ext = cells_src.ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const uint32_t strm = seq_to_stream[seq_id];

    // cells of other sequences in the stream are removed - conservatively drop all
    //   cached blocks of the stream, since they may become partially filled
    {
        std::vector<llama_prefix_cache::loc_t> locs;
        prefix.remove_stream(strm, locs);
        prefix_unregister(locs);
    }

    for (llama_seq_id s = 0; s < (llama_seq_id) chains.size(); ++s) {
        if (s != seq_id && s < (llama_seq_id) seq_to_stream.size() && seq_to_stream[s] == strm) {
            chains[s].reset();
        }
    }

    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    const uint32_t new_head = cells.seq_keep_all(seq_id);

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    if (paged && shift % (llama_pos) llama_kv_cells::block_size != 0) {
        // a non-block-aligned shift breaks the seq-aligned invariant that the block table
        // relies on (cell offset == position offset) - the K-shift graph still runs, but
        // the gathered positions will be wrong
        LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED does not support non-block-aligned position shifts (shift = %d) - results will be wrong\n",
                __func__, shift);
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    // shifted cells change positions - drop the affected cached blocks and the token chain
    {
        std::vector<llama_prefix_cache::loc_t> locs;
        prefix.remove_range(seq_to_stream[seq_id], p0, p1, locs);
        prefix_unregister(locs);
    }

    chains[seq_id].kill();

    for (uint32_t b = cells.block_lo(); b < cells.block_hi(); ++b) {
        if (cells.block_is_free(b) || cells.block_pos_max(b) < p0 || cells.block_pos_min(b) >= p1) {
            continue;
        }

        for (uint32_t i = cells.cell_begin(b); i < cells.cell_end(b); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id)) {
                if (cells.pos_add(i, shift)) {
                    if (new_head == cells.size()) {
                        new_head = i;
                    }
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (paged) {
        // position division breaks the seq-aligned invariant that the block table relies on
        LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED does not support position division - results will be wrong\n", __func__);
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // divided cells change positions - drop the affected cached blocks and the token chain
    {
        std::vector<llama_prefix_cache::loc_t> locs;
        prefix.remove_range(seq_to_stream[seq_id], p0, p1, locs);
        prefix_unregister(locs);
    }

    chains[seq_id].kill();

    for (uint32_t b = cells.block_lo(); b < cells.block_hi(); ++b) {
        if (cells.block_is_free(b) || cells.block_pos_max(b) < p0 || cells.block_pos_min(b) >= p1) {
            continue;
        }

        for (uint32_t i = cells.cell_begin(b); i < cells.cell_end(b); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id)) {
                cells.pos_div(i, d);
            }
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        auto sinfo_new = find_slot(ubatch, false);

        // on allocation pressure, evict unreferenced cached prefix blocks (LRU) and retry
        while (sinfo_new.empty()) {
            bool progress = false;

            for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                progress = progress || prefix_evict_one(seq_to_stream[ubatch.seq_id_unq[s]]);
            }

            if (!progress) {
                break;
            }

            sinfo_new = find_slot(ubatch, false);
        }

        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

//
// prefix cache
//

llama_prefix_match llama_kv_cache::prefix_match(const llama_token * tokens, int32_t n_tokens) {
    uint64_t handle = 0;

    llama_prefix_match res { prefix_match_ex(tokens, n_tokens, false, handle), handle };

    return res;
}

uint32_t llama_kv_cache::prefix_match_ex(const llama_token * tokens, int32_t n_tokens, bool need_state, uint64_t & out_handle) {
    out_handle = 0;

    if (!prefix_enabled || other) {
        return 0;
    }

    return prefix.match(tokens, n_tokens, need_state, out_handle);
}

bool llama_kv_cache::prefix_copy(llama_seq_id seq_id, uint64_t handle) {
    const auto * hashes = prefix.handle_hashes(handle);

    bool ok = false;

    if (hashes && prefix_enabled && !other) {
        ok = prefix_copy_impl(seq_id, *hashes);
    }

    prefix.release(handle);

    return ok;
}

void llama_kv_cache::prefix_release(uint64_t handle) {
    prefix.release(handle);
}

void llama_kv_cache::prefix_notify(const llama_ubatch & ubatch) {
    std::vector<seq_end> ends;

    prefix_notify_ex(ubatch, ends);
}

void llama_kv_cache::prefix_notify_ex(const llama_ubatch & ubatch, std::vector<seq_end> & out) {
    if (!prefix_enabled || !inflight_valid) {
        return;
    }

    inflight_valid = false;

    // embedding-only ubatches carry no token ids
    if (other || ubatch.token == nullptr) {
        return;
    }

    const auto & sinfo = inflight_sinfo;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const uint32_t strm = sinfo.strm[s];

        auto & cells = v_cells[strm];

        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            const llama_seq_id seq = ubatch.seq_id[i][0];

            auto & chain = chains[seq];

            if (!chain.feed(ubatch.pos[i], ubatch.token[i])) {
                continue;
            }

            // a full block was completed - register it if its cells are physically aligned
            const uint32_t b = sinfo.idxs[s][ii]/llama_kv_cells::block_size;

            if (!prefix_block_matches(cells, b, chain.last_p0)) {
                continue;
            }

            const llama_prefix_cache::loc_t loc { strm, b, chain.last_p0 };

            const auto res = prefix.insert(chain.last_hash, chain.last_tokens, loc);
            if (res.second) {
                cells.block_pin(b);
            }
        }
    }

    // report aligned sequence ends (used by hybrid memory for recurrent state snapshots)
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq = ubatch.seq_id_unq[s];

        const auto & chain = chains[seq];

        if (chain.next_pos <= 0 || chain.next_pos%(llama_pos) llama_prefix_cache::block_size != 0 || chain.hashes.empty()) {
            continue;
        }

        seq_end end;

        end.seq_id  = seq;
        end.pos_end = chain.next_pos;
        end.hash    = chain.last_hash;

        memcpy(end.tokens, chain.last_tokens, sizeof(end.tokens));

        out.push_back(end);
    }
}

void llama_kv_cache::prefix_set_state(uint64_t hash, const llama_token * tokens, std::vector<uint8_t> state) {
    prefix.set_state(hash, tokens, std::move(state));
}

void llama_kv_cache::prefix_set_state(uint64_t hash, const llama_token * tokens, ggml_backend_buffer_ptr buf, size_t size) {
    prefix.set_state(hash, tokens, std::move(buf), size);
}

const llama_prefix_cache::state_view * llama_kv_cache::prefix_handle_state(uint64_t handle) const {
    return prefix.handle_state(handle);
}

void llama_kv_cache::prefix_set_enabled(bool enabled) {
    prefix_enabled = enabled;
}

void llama_kv_cache::prefix_note_applied(const slot_info & sinfo) {
    if (!prefix_enabled) {
        return;
    }

    inflight_sinfo = sinfo;
    inflight_valid = true;
}

bool llama_kv_cache::prefix_evict_one(uint32_t stream) {
    if (!prefix_enabled) {
        return false;
    }

    auto & cells = v_cells[stream];

    std::vector<llama_prefix_cache::loc_t> cands;

    prefix.evict_candidates(stream, std::numeric_limits<uint32_t>::max(), cands);

    for (const auto & loc : cands) {
        // only cache-owned blocks actually free memory
        if (!cells.block_all_seqless(loc.block)) {
            continue;
        }

        prefix.remove_loc(loc);

        cells.block_unpin(loc.block);

        return true;
    }

    return false;
}

void llama_kv_cache::prefix_unregister(const std::vector<llama_prefix_cache::loc_t> & locs) {
    for (const auto & loc : locs) {
        if (prefix.remove_loc(loc)) {
            v_cells[loc.stream].block_unpin(loc.block);
        }
    }
}

bool llama_kv_cache::prefix_block_matches(const llama_kv_cells & cells, uint32_t b, llama_pos p0) const {
    if (cells.cell_end(b) - cells.cell_begin(b) != llama_kv_cells::block_size) {
        return false;
    }

    if (!cells.block_is_full(b)) {
        return false;
    }

    for (uint32_t j = 0; j < llama_kv_cells::block_size; ++j) {
        const uint32_t i = cells.cell_begin(b) + j;

        if (cells.is_empty(i) || cells.pos_get(i) != p0 + (llama_pos) j) {
            return false;
        }
    }

    return true;
}

bool llama_kv_cache::prefix_copy_impl(llama_seq_id seq_id, const std::vector<uint64_t> & hashes) {
    const uint32_t n_block = hashes.size();

    if (n_block == 0) {
        return false;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const uint32_t s1 = seq_to_stream[seq_id];

    auto & cells = v_cells[s1];

    // the destination sequence must be empty
    if (cells.seq_pos_max(seq_id) != -1) {
        return false;
    }

    std::vector<uint32_t> attached;
    std::vector<uint32_t> allocated;
    std::vector<llama_prefix_cache::loc_t> registered;

    bool ok = true;

    for (uint32_t k = 0; k < n_block && ok; ++k) {
        const llama_pos p0 = (llama_pos) k*llama_prefix_cache::block_size;

        const auto * e = prefix.find(hashes[k]);
        if (!e || e->locs.empty()) {
            ok = false;
            break;
        }

        // prefer a location in the destination stream (zero-copy re-attach)
        const llama_prefix_cache::loc_t * loc_same = nullptr;
        const llama_prefix_cache::loc_t * loc_any  = nullptr;

        for (const auto & loc : e->locs) {
            if (loc.stream == s1) {
                if (prefix_block_matches(v_cells[loc.stream], loc.block, p0)) {
                    loc_same = &loc;
                    break;
                }
            } else if (!loc_any && prefix_block_matches(v_cells[loc.stream], loc.block, p0)) {
                loc_any = &loc;
            }
        }

        if (loc_same) {
            // the cells already hold the right contents and positions - re-attach them.
            // in paged mode this is the entire prefix copy: pure bookkeeping - the block
            // stays shared (read-only, pinned) and the sequence's block table simply maps
            // the range to it (see llama_kv_cells::seq_block_table)
            for (uint32_t j = 0; j < llama_kv_cells::block_size; ++j) {
                const uint32_t i = cells.cell_begin(loc_same->block) + j;

                cells.seq_add(i, seq_id);

                attached.push_back(i);
            }

            continue;
        }

        if (!loc_any) {
            ok = false;
            break;
        }

        if (paged) {
            // paged mode never performs physical prefix copies: blocks are shared
            // read-only via pin handles and re-attached (loc_same), which already covers
            // every location of the single (unified) stream, so this point is only
            // reachable if the matching block was concurrently mutated or evicted.
            // the caller falls back to processing the prompt normally
            LLAMA_LOG_DEBUG("%s: skipping physical prefix copy in paged mode (hash %d, pos %d)\n",
                    __func__, (int) k, p0);
            ok = false;
            break;
        }

        // allocate 32 cells in the destination stream, evicting cached blocks on pressure
        std::vector<uint32_t> idxs;

        cells.alloc_find(llama_kv_cells::block_size, idxs);

        while (idxs.size() < llama_kv_cells::block_size && prefix_evict_one(s1)) {
            idxs.clear();
            cells.alloc_find(llama_kv_cells::block_size, idxs);
        }

        if (idxs.size() < llama_kv_cells::block_size) {
            ok = false;
            break;
        }

        std::sort(idxs.begin(), idxs.end());

        const auto & cells_src = v_cells[loc_any->stream];

        const uint32_t c0 = cells_src.cell_begin(loc_any->block);

        for (uint32_t j = 0; j < llama_kv_cells::block_size; ++j) {
            cells.pos_set(idxs[j], p0 + (llama_pos) j);
            cells.seq_add(idxs[j], seq_id);
            cells.ext_set(idxs[j], cells_src.ext_get(c0 + j));
        }

        // copy the K/V data in contiguous runs
        for (uint32_t j0 = 0; j0 < llama_kv_cells::block_size; ) {
            uint32_t j1 = j0 + 1;
            while (j1 < llama_kv_cells::block_size && idxs[j1] == idxs[j1 - 1] + 1) {
                ++j1;
            }

            copy_cells(loc_any->stream, c0 + j0, s1, idxs[j0], j1 - j0);

            j0 = j1;
        }

        allocated.insert(allocated.end(), idxs.begin(), idxs.end());

        // register the destination block as an additional location if it is physically aligned
        if (idxs.front()%llama_kv_cells::block_size == 0 && idxs.back() - idxs.front() == llama_kv_cells::block_size - 1) {
            const llama_prefix_cache::loc_t loc { s1, idxs.front()/llama_kv_cells::block_size, p0 };

            const auto res = prefix.insert(hashes[k], e->tokens, loc);
            if (res.second) {
                cells.block_pin(loc.block);
                registered.push_back(loc);
            }
        }
    }

    if (!ok) {
        prefix_unregister(registered);

        for (const auto i : allocated) {
            cells.rm(i);
        }

        for (const auto i : attached) {
            cells.seq_rm(i, seq_id);
        }

        return false;
    }

    // the destination sequence now owns the same token chain
    auto & chain = chains[seq_id];

    chain.reset();
    chain.hashes   = hashes;
    chain.next_pos = (llama_pos) n_block*llama_prefix_cache::block_size;

    // move the stream head past the copied cells
    uint32_t idx_max = 0;

    for (const auto i : allocated) {
        idx_max = std::max(idx_max, i);
    }

    for (const auto i : attached) {
        idx_max = std::max(idx_max, i);
    }

    if (idx_max + 1 > v_heads[s1]) {
        v_heads[s1] = idx_max + 1;
    }

    return true;
}

void llama_kv_cache::copy_cells(uint32_t ssrc, uint32_t csrc, uint32_t sdst, uint32_t cdst, uint32_t n) {
    size_t n_views = 0;

    for (const auto & layer : layers) {
        n_views += 2; // k

        if (layer.v) {
            n_views += v_trans ? 2*(size_t) layer.v->ne[0] : 2;
        }
    }

    ggml_init_params params = {
        /*.mem_size   =*/ n_views*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: failed to create ggml context for cell copies\n", __func__);
        return;
    }

    const uint32_t kv_size = get_size();

    // views created in a fresh context have no storage - point them at the parent tensor's
    auto view_2d = [&](ggml_tensor * t, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) {
        ggml_tensor * v = ggml_view_2d(ctx.get(), t, ne0, ne1, nb1, offset);
        v->buffer = t->buffer;
        v->data   = (char *) t->data + offset;
        return v;
    };

    auto view_1d = [&](ggml_tensor * t, int64_t ne0, size_t offset) {
        ggml_tensor * v = ggml_view_1d(ctx.get(), t, ne0, offset);
        v->buffer = t->buffer;
        v->data   = (char *) t->data + offset;
        return v;
    };

    for (const auto & layer : layers) {
        ggml_tensor * k = layer.k;

        ggml_tensor * k_src = view_2d(k, k->ne[0], n, k->nb[1], ssrc*k->nb[2] + csrc*k->nb[1]);
        ggml_tensor * k_dst = view_2d(k, k->ne[0], n, k->nb[1], sdst*k->nb[2] + cdst*k->nb[1]);

        ggml_backend_tensor_copy(k_src, k_dst);

        ggml_tensor * v = layer.v;
        if (!v) {
            continue;
        }

        if (!v_trans) {
            ggml_tensor * v_src = view_2d(v, v->ne[0], n, v->nb[1], ssrc*v->nb[2] + csrc*v->nb[1]);
            ggml_tensor * v_dst = view_2d(v, v->ne[0], n, v->nb[1], sdst*v->nb[2] + cdst*v->nb[1]);

            ggml_backend_tensor_copy(v_src, v_dst);
        } else {
            // cells are strided across the transposed V cache - copy row by row
            const int64_t n_embd_v_gqa = v->ne[0];

            for (int64_t j = 0; j < n_embd_v_gqa; ++j) {
                ggml_tensor * v_src = view_1d(v, n, ssrc*v->nb[2] + ((size_t) j*kv_size + csrc)*v->nb[0]);
                ggml_tensor * v_dst = view_1d(v, n, sdst*v->nb[2] + ((size_t) j*kv_size + cdst)*v->nb[0]);

                ggml_backend_tensor_copy(v_src, v_dst);
            }
        }
    }
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        if (!cont) {
            if (paged) {
                // seq-aligned allocation: the token with logical position p goes to cell
                // block_table[seq][p/32]*32 + p%32 of a block owned by its sequence - never
                // shares a partial-block tail with another sequence or aligned range.
                // the blocks themselves are updated in apply_ubatch()
                const uint32_t i0 = n_stream > 1 ? s*n_tokens : 0;

                std::vector<llama_seq_id> seqs(n_tokens);
                for (uint32_t ii = 0; ii < n_tokens; ++ii) {
                    seqs[ii] = ubatch.seq_id[i0 + ii][0];
                }

                cells.alloc_find_seq_aligned(ubatch.pos + i0, seqs.data(), n_tokens, res.idxs[s]);
            } else {
                // allocate empty cells with the block allocator - partially-used blocks first,
                //   then free blocks. the blocks themselves are updated in apply_ubatch()
                cells.alloc_find(n_tokens, res.idxs[s]);
            }

            if (res.idxs[s].size() < n_tokens && n_swa > 0) {
                // not enough empty cells - reuse cells with SWA-masked positions, starting
                //   from the current head
                uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

                // if we have enough unused cells before the current head ->
                //   better to start searching from the beginning of the cache, hoping to fill it
                if (head_cur > cells.get_used() + 2*n_tokens) {
                    head_cur = 0;
                }

                uint32_t n_tested = 0;

                while (res.idxs[s].size() < n_tokens && n_tested < cells.size()) {
                    if (head_cur >= cells.size()) {
                        head_cur = 0;
                    }

                    const auto idx = head_cur;

                    head_cur++;
                    n_tested++;

                    // the cell can be reused if it is occupied only by one sequence and its
                    //   position is masked by the SWA, using the current max pos for that
                    //   sequence in the cache
                    // pinned (prefix-cached) blocks are never reused
                    if (cells.is_empty(idx) || cells.seq_count(idx) != 1 || cells.block_ref(idx/llama_kv_cells::block_size) != 0) {
                        continue;
                    }

                    const llama_seq_id seq_id_cell = cells.seq_get(idx);
                    const llama_pos    pos_cell    = cells.pos_get(idx);

                    if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                        res.idxs[s].push_back(idx);
                    }
                }
            }

            // we didn't find a suitable slot - return empty result
            if (res.idxs[s].size() < n_tokens) {
                return { };
            }

            // keep the indices sorted - helps the contiguous fast path of the state restore
            // note: in paged mode the token -> cell mapping is positional (the token with
            //       logical position p must go to cell offset p%32), so the ubatch order
            //       of the indices must be preserved
            if (!paged) {
                std::sort(res.idxs[s].begin(), res.idxs[s].end());
            }

            continue;
        }

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        const uint32_t n_test = n_tokens;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence and its position is masked by
                //    the SWA, using current max pos for that sequence in the cache
                //    pinned (prefix-cached) blocks are never reused
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1 && cells.block_ref(idx/llama_kv_cells::block_size) == 0) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    const llama_seq_id seq_id_cell = cells.seq_get(idx);

                    // SWA mask
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                        can_use = true;
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    break;
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            res.idxs[s].clear();

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    if (getenv("LLAMA_KV_FIND_TRACE") != nullptr) {
        fprintf(stderr, "FIND_SLOT: n_seqs=%u n_tokens=%u cont=%d |", ubatch.n_seqs_unq, ubatch.n_tokens, (int) cont);
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            fprintf(stderr, " seq%d(strm%d,head%d)[", (int) seq_id, (int) stream_id, (int) v_heads[stream_id]);
            for (size_t i = 0; i < res.idxs[s].size() && i < 8; ++i) {
                fprintf(stderr, "%s%u", i ? "," : "", res.idxs[s][i]);
            }
            fprintf(stderr, "]");
        }
        fprintf(stderr, "\n");
    }

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            // the paged allocation must place every token at its seq-aligned cell -
            // checked because apply_ubatch pairs indices and positions by ubatch order
            GGML_ASSERT(!paged || (uint32_t) ubatch.pos[i]%llama_kv_cells::block_size == idx%llama_kv_cells::block_size);

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    if (paged) {
        // position shifts move cells off their seq-aligned offsets, which the paged
        // block table cannot represent (cell offset == position offset). disabling
        // shifts makes the server fall back to reprocessing instead of shifting
        return false;
    }
    return true;
}

bool llama_kv_cache::get_prefix_cache_enabled() const {
    return prefix_enabled;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

bool llama_kv_cache::get_paged() const {
    return paged;
}

void llama_kv_cache::set_paged(bool enabled) {
    paged = enabled;
}

bool llama_kv_cache::paged_ubatch(const llama_ubatch & ubatch) const {
    if (!paged) {
        return false;
    }

    // the paged FA kernel addresses each sequence through its own block table column:
    // the tokens of the ubatch must be grouped per sequence (seq-major), with an equal
    // number of tokens per sequence and no multi-sequence (coupled) tokens
    const uint32_t n_seq = ubatch.n_seqs_unq;

    if (n_seq == 0 || ubatch.n_tokens % n_seq != 0) {
        return false;
    }

    const uint32_t n_tps = ubatch.n_tokens/n_seq;

    // the paged FA kernel is used only for single-token-per-sequence ubatches (decode).
    // the n_tps > 1 shapes (prefill chunks, multi-token MTP verify) take the legacy path:
    // the HIP paged vec kernel with ncols = 2 intermittently hits an illegal memory
    // access on gfx908 (see the crash logs of the bench validation)
    if (n_tps != 1) {
        return false;
    }

    llama_seq_id seq_cur = -1;

    for (uint32_t g = 0; g < n_seq; ++g) {
        // note: reserved ubatches (graph_reserve) carry null seq_id pointers past the
        // first few tokens - they are never paged-eligible
        if (ubatch.n_seq_id[g*n_tps] != 1 || ubatch.seq_id[g*n_tps] == nullptr) {
            return false;
        }

        const llama_seq_id seq_id = ubatch.seq_id[g*n_tps][0];

        if (seq_id == seq_cur) {
            // the tokens of this sequence are not grouped in a single contiguous block
            return false;
        }

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = g*n_tps + ii;

            if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i] == nullptr || ubatch.seq_id[i][0] != seq_id) {
                return false;
            }
        }

        seq_cur = seq_id;
    }

    return true;
}

uint32_t llama_kv_cache::get_n_kv_paged(const llama_ubatch & ubatch) const {
    uint32_t result = 0;

    // the logical KV length is the padded max sequence length of the ubatch
    // note: called after apply_ubatch(), so the current positions are included
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        result = std::max(result, (uint32_t) (v_cells[seq_to_stream[seq_id]].seq_pos_max(seq_id) + 1));
    }

    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    return std::min(get_size(), std::max(n_pad_cur, GGML_PAD(result, n_pad_cur)));
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_k_paged(ggml_context * ctx, int32_t il, uint32_t n_seq) const {
    GGML_UNUSED(n_seq);

    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    // the full physical pool (a single slice) - the kernel resolves the actual rows for
    // each sequence through the block table (src[5]). the pool is seq-broadcast:
    // ne[3] == 1 with nb[3] == 0 (ggml cannot alias it into one view per sequence - its
    // view-size assert counts the aliased dim against the source size), so multi-sequence
    // ubatches use a single batched FA node whose kernel derives the sequence from
    // blockIdx.z (see build_attn_mha)
    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), kv_size, 1,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            0,
            0);
}

ggml_tensor * llama_kv_cache::get_v_paged(ggml_context * ctx, int32_t il, uint32_t n_seq) const {
    GGML_UNUSED(n_seq);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    // paged mode requires flash attention, so the V cache is not transposed
    assert(!v_trans);

    // see get_k_paged
    return ggml_view_4d(ctx, v,
            hparams.n_embd_head_v(il), hparams.n_head_kv(il), kv_size, 1,
            ggml_row_size(v->type, hparams.n_embd_head_v(il)),
            ggml_row_size(v->type, n_embd_v_gqa),
            0,
            0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_block_table(ggml_context * ctx, const llama_ubatch & ubatch) const {
    if (!paged_ubatch(ubatch)) {
        return nullptr;
    }

    // I32 [n_block, n_seq] - one column per sequence of the ubatch, in the order in which
    // the sequences appear in the (seq-major) token axis of q. entry = physical block id,
    // k/v for logical position p of sequence s lives at row table[p/32][s]*32 + p%32 of
    // the physical pool. unused entries are filled with 0 by set_input_block_table (the
    // kernel dereferences the table unconditionally; the mask makes the gathered rows
    // ineffective). the mask is logically indexed: mask column p covers logical position p.
    const uint32_t n_block = get_size()/llama_kv_cells::block_size;

    ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_block, ubatch.n_seqs_unq);

    ggml_set_input(block_table);

    ggml_set_name(block_table, "attn_inp_block_table");

    return block_table;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    if (getenv("LLAMA_SETINP_TRACE") != nullptr) {
        fprintf(stderr, "SETINP: dst=%s data=%p n_tokens=%u size=%u n_stream=%u idxs=[", dst->name, (void *) dst->data, n_tokens, sinfo.size(), sinfo.n_stream());
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                fprintf(stderr, "%s%lld", (s*sinfo.size() + i) ? "," : "",
                        (long long) (sinfo.strm[s]*get_size() + sinfo.idxs[s][i]));
            }
        }
        fprintf(stderr, "]\n");
    }

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }

    // KV diagnostic: read back the leaf right after the write, to detect a second writer
    if (getenv("LLAMA_SETINP_WR") != nullptr) {
        fprintf(stderr, "SETINP_WR: dst=%s data=%p leaf=[", dst->name, (void *) dst->data);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            fprintf(stderr, "%s%lld", i ? "," : "", (long long) data[i]);
        }
        fprintf(stderr, "]\n");
    }

    // KV diagnostic: dump the token-to-cell mapping for single-token (decode) ubatches
    if (getenv("LLAMA_RS_DEBUG") != nullptr && sinfo.size() == 1) {
        fprintf(stderr, "KV_IDXS: size=%u n_stream=%u idxs=[", sinfo.size(), sinfo.n_stream());
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            fprintf(stderr, "%s%lld", s ? "," : "", (long long) (sinfo.idxs[s][0] + sinfo.strm[s]*get_size()));
        }
        fprintf(stderr, "]\n");
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_block_table(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    GGML_ASSERT(paged);

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const uint32_t n_block = get_size()/llama_kv_cells::block_size;
    const uint32_t n_seq   = ubatch->n_seqs_unq;
    const uint32_t n_tps   = ubatch->n_tokens/n_seq;

    GGML_ASSERT((uint32_t) dst->ne[0] == n_block);
    GGML_ASSERT((uint32_t) dst->ne[1] == n_seq);

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t g = 0; g < n_seq; ++g) {
        // the sequence of column g - the tokens are seq-major (see paged_ubatch)
        const llama_seq_id seq_id = ubatch->seq_id[g*n_tps][0];

        int32_t * dst_col = data + (size_t) g*n_block;

        // unused entries must address valid memory - the kernel reads them unconditionally
        std::fill(dst_col, dst_col + n_block, 0);

        std::vector<int32_t> table(n_block, -1);

        if (!v_cells[seq_to_stream[seq_id]].seq_block_table(seq_id, table)) {
            // cells of one (seq, aligned range) pair in multiple blocks - the table
            // cannot represent this. should not happen with the seq-aligned
            // allocation policy - attention for this sequence will be wrong
            LLAMA_LOG_ERROR("%s: ambiguous block table for sequence %d - paged allocation policy violated\n",
                    __func__, seq_id);
        }

        for (uint32_t b = 0; b < n_block; ++b) {
            if (table[b] >= 0) {
                dst_col[b] = table[b];
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }

    // KV diagnostic: dump the attended cell range for single-token (decode) ubatches
    if (getenv("LLAMA_RS_DEBUG") != nullptr && n_tps == 1) {
        for (uint32_t s = 0; s < n_stream; ++s) {
            const uint32_t i = s*n_tps;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            const int64_t off = n_kv*i;
            int32_t n_keep = 0;
            int32_t p_max = -1;
            for (int64_t j = 0; j < n_kv; ++j) {
                if (data[off + j] == mask_keep) {
                    n_keep++;
                    p_max = (int32_t) j;
                }
            }
            fprintf(stderr, "KV_MASK: seq=%d n_kv=%lld n_keep=%d p_max=%d pos=%lld\n",
                seq_id, (long long) n_kv, n_keep, p_max, (long long) ubatch->pos[i]);
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

// paged variant of set_input_kq_mask: the mask is indexed by LOGICAL position (the FA
// kernel gathers the physical rows through the block table), with one slice per sequence
// of the ubatch (dst->ne[3] == n_seqs_unq)
template<typename T>
static void set_input_kq_mask_paged_impl(
        const llama_hparams & hparams,
        const llama_ubatch  * ubatch,
        const llama_kv_cells_vec & v_cells,
        const std::vector<uint32_t> & seq_to_stream,
        bool causal_attn,
        ggml_tensor * dst,
        T * data) {
    const int64_t n_kv  = dst->ne[0]; // logical KV length
    const int64_t n_seq = dst->ne[3];
    const int64_t n_tps = dst->ne[1];

    const uint32_t bs = llama_kv_cells::block_size;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    std::vector<int32_t> table;

    for (int64_t g = 0; g < n_seq; ++g) {
        const llama_seq_id seq_id = ubatch->seq_id[g*n_tps][0];

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        // resolve the logical positions of this sequence through its block table:
        // has_pos[p] == true iff a cell with position p of this sequence exists
        table.resize((n_kv + bs - 1)/bs);

        cells.seq_block_table(seq_id, table);

        std::vector<uint8_t> has_pos(n_kv, 0);

        for (int64_t p = 0; p < n_kv; ++p) {
            const int32_t b = table[p/bs];

            if (b < 0) {
                continue;
            }

            const uint32_t c = (uint32_t) b*bs + (uint32_t) p%bs;

            if (c < cells.size() && !cells.is_empty(c) && cells.pos_get(c) == p && cells.seq_has(c, seq_id)) {
                has_pos[p] = 1;
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t i = g*n_tps + ii;

            const llama_pos p1 = ubatch->pos[i];

            T * row = data + ((size_t) g*n_tps + ii)*n_kv;

            int64_t p = 0;

            if (hparams.use_alibi) {
                for (; p <= p1 && p < n_kv; ++p) {
                    row[p] = has_pos[p] ? llama_cast<T>(static_cast<float>(-std::abs(p - p1))) : mask_drop;
                }
            } else {
                for (; p <= p1 && p < n_kv; ++p) {
                    row[p] = has_pos[p] ? mask_keep : mask_drop;
                }
            }

            for (; p < n_kv; ++p) {
                if (has_pos[p] && !causal_attn) {
                    row[p] = hparams.use_alibi ? llama_cast<T>(static_cast<float>(-std::abs(p - p1))) : mask_keep;
                } else {
                    row[p] = mask_drop;
                }
            }
        }
    }
}

void llama_kv_cache::set_input_kq_mask_paged(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    GGML_ASSERT(paged);

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    GGML_ASSERT(dst->ne[3] == ubatch->n_seqs_unq);
    GGML_ASSERT(dst->ne[1]*dst->ne[3] == ubatch->n_tokens);

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_paged_impl<ggml_fp16_t>(hparams, ubatch, v_cells, seq_to_stream, causal_attn, dst, (ggml_fp16_t *) dst->data);
    } else {
        set_input_kq_mask_paged_impl<float>      (hparams, ubatch, v_cells, seq_to_stream, causal_attn, dst, (float       *) dst->data);
    }
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

uint32_t llama_kv_cache::get_n_kv_layers() const {
    return (uint32_t) layers.size();
}

const ggml_tensor * llama_kv_cache::get_k_l(int32_t il) const {
    return layers[il].k;
}

const ggml_tensor * llama_kv_cache::get_v_l(int32_t il) const {
    return layers[il].v;
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (seq_id != -1) {
        // the restored cells have unknown tokens - the chain restarts empty
        chains[seq_id].reset();
    }

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);
        res = res && state_read_data(io, strm, cell_count, sinfo);

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (hparams.n_pos_per_embd() > 1) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Fast path: contiguous cells, single memcpy
                io.read_tensor(k, sinfo.head() * k_size_row, cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                    io.read_tensor(k, dst_offset, k_size_row);
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    io.read_tensor(v, sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        io.read_tensor(v, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        io.read_tensor(v, dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            io.read_tensor(v, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    kv->prefix_note_applied(sinfos[i_cur]);

    paged = kv->paged_ubatch(ubatches[i_cur]);

    n_kv = paged ? kv->get_n_kv_paged(ubatches[i_cur]) : kv->get_n_kv(sinfos[i_cur]);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

bool llama_kv_cache_context::get_paged() const {
    return paged;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    if (paged) {
        return kv->get_k_paged(ctx, il, ubatches[i_cur].n_seqs_unq);
    }

    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    if (paged) {
        return kv->get_v_paged(ctx, il, ubatches[i_cur].n_seqs_unq);
    }

    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_block_table(ggml_context * ctx, const llama_ubatch & ubatch) const {
    // note: gate on the context state (set in apply()), not just on the ubatch layout, so
    // that the block table and the mask/view shapes are always built consistently.
    // non-batch contexts (init_full/update, e.g. graph reserve) always take the legacy path
    if (!paged) {
        return nullptr;
    }

    return kv->build_input_block_table(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_block_table(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_block_table(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    if (paged) {
        kv->set_input_kq_mask_paged(dst, ubatch, causal_attn);
        return;
    }

    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

const llama_kv_cache_context::slot_info_vec_t & llama_kv_cache_context::get_slot_infos() const {
    return sinfos;
}

size_t llama_kv_cache_context::get_i_cur() const {
    return i_cur;
}

const llama_kv_cache * llama_kv_cache_context::get_kv_cache() const {
    return kv;
}
