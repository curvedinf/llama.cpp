// End-to-end test for the block-hash prefix cache through llama_context:
//   - seq 0 prefills a shared block-aligned prefix, registering its blocks
//   - seq 1 matches the prefix and reuses it via llama_memory_prefix_copy
//   - post-prefix logits of seq 1 must be bit-identical to a full-prefill reference
//   - the donor sequence can keep decoding and can be removed without corrupting seq 1
//   - hybrid (recurrent) archs reuse only prefixes with a snapshotted state
//
// covers LLM_ARCH_LLAMA (pure attention) and LLM_ARCH_QWEN35 (hybrid GDN + attention)

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// minimal synthetic model, same approach as test-graph-cache.cpp / test-llama-archs.cpp
static gguf_context_ptr get_gguf_ctx(const llm_arch arch) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 512;

    const uint32_t n_vocab = 128;
    const uint32_t n_embd  = 256;
    const uint32_t n_head  = 2;
    const uint32_t n_ff    = 384;
    const uint32_t n_layer = 2;

    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      uint32_t(8));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

// 3 blocks of shared prefix + a divergent suffix (not block-aligned)
static const int n_prefix = 96;
static const int n_suffix = 8;
static const int n_extra  = 16; // donor continuation / final continuation

static std::vector<llama_token> make_prompt() {
    std::vector<llama_token> res(n_prefix + n_suffix + n_extra);
    for (size_t i = 0; i < res.size(); ++i) {
        res[i] = (llama_token) (i*7%113 + 1);
    }
    return res;
}

// decode tokens[pos0, pos0 + n) of seq_id, collecting the logits of the last n_out tokens
static bool decode_range(llama_context * ctx, llama_model * model, llama_seq_id seq_id,
        const std::vector<llama_token> & tokens, int32_t pos0, int32_t n, int32_t n_out, std::vector<float> & logits) {
    llama_batch batch = llama_batch_init(n, 0, 1);

    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch.token    [i]    = tokens[pos0 + i];
        batch.pos      [i]    = pos0 + i;
        batch.n_seq_id [i]    = 1;
        batch.seq_id   [i][0] = seq_id;
        batch.logits   [i]    = i >= n - n_out;
    }

    const int rc = llama_decode(ctx, batch);

    llama_batch_free(batch);

    if (rc != 0) {
        fprintf(stderr, "%s: llama_decode failed: %d\n", __func__, rc);
        return false;
    }

    const int64_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int32_t i = n - n_out; i < n; ++i) {
        const float * logits_i = llama_get_logits_ith(ctx, i);
        if (!logits_i) {
            fprintf(stderr, "%s: failed to get logits\n", __func__);
            return false;
        }
        logits.insert(logits.end(), logits_i, logits_i + n_vocab);
    }

    return true;
}

struct run_result {
    std::vector<float> logits_suffix; // logits of the divergent suffix
    std::vector<float> logits_extra;  // logits of the final continuation

    uint32_t n_matched = 0;

    bool ok = false;
};

// reference: seq 1 full-prefills the whole prompt
static run_result run_reference(llama_model * model, const std::vector<llama_token> & tokens) {
    run_result result;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 512;
    ctx_params.n_batch   = 128;
    ctx_params.n_ubatch  = 64;
    ctx_params.n_seq_max = 2;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;

    llama_context_ptr ctx(llama_init_from_model(model, ctx_params));
    if (!ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return result;
    }

    bool ok = true;

    // seq 0 prefills the shared prefix
    ok = ok && decode_range(ctx.get(), model, 0, tokens, 0, n_prefix, 1, result.logits_extra);

    // seq 1 full-prefills prefix + suffix
    result.logits_extra.clear();
    ok = ok && decode_range(ctx.get(), model, 1, tokens, 0, n_prefix + n_suffix, n_suffix, result.logits_suffix);

    // seq 1 continues
    ok = ok && decode_range(ctx.get(), model, 1, tokens, n_prefix + n_suffix, n_extra, n_extra, result.logits_extra);

    result.ok = ok;

    return result;
}

// test: seq 1 reuses seq 0's cached prefix, then seq 0 is removed mid-flight
static run_result run_copy(llama_model * model, const std::vector<llama_token> & tokens, bool rm_donor) {
    run_result result;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 512;
    ctx_params.n_batch   = 128;
    ctx_params.n_ubatch  = 64;
    ctx_params.n_seq_max = 2;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;

    llama_context_ptr ctx(llama_init_from_model(model, ctx_params));
    if (!ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return result;
    }

    bool ok = true;

    // seq 0 prefills the shared prefix
    std::vector<float> tmp;
    ok = ok && decode_range(ctx.get(), model, 0, tokens, 0, n_prefix, 1, tmp);

    // seq 1 matches the cached prefix
    const auto pm = llama_memory_prefix_match(ctx.get(), tokens.data(), n_prefix + n_suffix);

    result.n_matched = pm.n_tokens;

    if (pm.handle == 0) {
        fprintf(stderr, "%s: no prefix match\n", __func__);
        return result;
    }

    // the donor keeps decoding past the prefix before the copy
    ok = ok && decode_range(ctx.get(), model, 0, tokens, n_prefix, n_suffix, 1, tmp);

    if (!llama_memory_prefix_copy(ctx.get(), 1, pm.handle)) {
        fprintf(stderr, "%s: llama_memory_prefix_copy failed\n", __func__);
        return result;
    }

    if (rm_donor) {
        llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, -1);
    }

    // seq 1 decodes only the suffix
    ok = ok && decode_range(ctx.get(), model, 1, tokens, n_prefix, n_suffix, n_suffix, result.logits_suffix);

    // seq 1 continues
    ok = ok && decode_range(ctx.get(), model, 1, tokens, n_prefix + n_suffix, n_extra, n_extra, result.logits_extra);

    result.ok = ok;

    return result;
}

// same-stream reuse: after the donor is removed, a new sequence on the same stream
// re-attaches the retained cache-owned blocks without any data copy
static run_result run_reattach(llama_model * model, const std::vector<llama_token> & tokens) {
    run_result result;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 512;
    ctx_params.n_batch   = 128;
    ctx_params.n_ubatch  = 64;
    ctx_params.n_seq_max = 2;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;

    llama_context_ptr ctx(llama_init_from_model(model, ctx_params));
    if (!ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return result;
    }

    bool ok = true;

    // seq 0 prefills the shared prefix
    std::vector<float> tmp;
    ok = ok && decode_range(ctx.get(), model, 0, tokens, 0, n_prefix, 1, tmp);

    // match the prefix before removing the donor
    const auto pm = llama_memory_prefix_match(ctx.get(), tokens.data(), n_prefix + n_suffix);

    result.n_matched = pm.n_tokens;

    if (pm.handle == 0) {
        fprintf(stderr, "%s: no prefix match\n", __func__);
        return result;
    }

    // remove the donor - its registered blocks stay cached (cache-owned)
    llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, -1);

    // a new sequence on the same stream reuses the retained blocks
    if (!llama_memory_prefix_copy(ctx.get(), 0, pm.handle)) {
        fprintf(stderr, "%s: llama_memory_prefix_copy failed\n", __func__);
        return result;
    }

    ok = ok && decode_range(ctx.get(), model, 0, tokens, n_prefix, n_suffix, n_suffix, result.logits_suffix);
    ok = ok && decode_range(ctx.get(), model, 0, tokens, n_prefix + n_suffix, n_extra, n_extra, result.logits_extra);

    result.ok = ok;

    return result;
}

// the donor prefill ends mid-block within a single ubatch: no aligned sequence end is
// reported, so a hybrid model cannot reuse the prefix, while a pure-attention model can
static bool test_unaligned(const llm_arch arch, llama_model * model, const std::vector<llama_token> & tokens) {
    printf("=== %s: arch = %s ===\n", __func__, llm_arch_name(arch));

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 512;
    ctx_params.n_batch   = 128;
    ctx_params.n_ubatch  = 128; // single ubatch - the prefill ends at an unaligned position
    ctx_params.n_seq_max = 2;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;

    llama_context_ptr ctx(llama_init_from_model(model, ctx_params));
    if (!ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return false;
    }

    std::vector<float> tmp;

    // 101 tokens - ends mid-block at position 100
    if (!decode_range(ctx.get(), model, 0, tokens, 0, n_prefix + 5, 1, tmp)) {
        return false;
    }

    const auto pm = llama_memory_prefix_match(ctx.get(), tokens.data(), n_prefix + 5);

    const bool is_hybrid = arch == LLM_ARCH_QWEN35;

    const uint32_t expect = is_hybrid ? 0 : n_prefix;

    printf("matched: %d tokens (expected %d)\n", pm.n_tokens, expect);

    if (pm.handle != 0) {
        llama_memory_prefix_release(ctx.get(), pm.handle);
    }

    if (pm.n_tokens != expect) {
        fprintf(stderr, "%s: unexpected match length\n", __func__);
        return false;
    }

    printf("%s: OK\n", __func__);

    return true;
}

// eviction under allocation pressure: the retained cached blocks are evicted LRU when a
// new sequence needs the space, invalidating the corresponding prefix entries
static bool test_evict(const llm_arch arch, llama_model * model, const std::vector<llama_token> & tokens) {
    printf("=== %s: arch = %s ===\n", __func__, llm_arch_name(arch));

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 256; // 8 blocks per sequence (n_ctx_seq padded to 256 cells)
    ctx_params.n_batch   = 256;
    ctx_params.n_ubatch  = 64;
    ctx_params.n_seq_max = 2;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;

    llama_context_ptr ctx(llama_init_from_model(model, ctx_params));
    if (!ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return false;
    }

    bool ok = true;

    std::vector<float> tmp;

    // seq 0 fills 3 of the 8 blocks of its stream with a cacheable prefix
    ok = ok && decode_range(ctx.get(), model, 0, tokens, 0, n_prefix, 1, tmp);

    // remove the donor - the 3 blocks stay cached (cache-owned)
    llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, -1);

    // seq 0 prefills 6 blocks of a different prompt on the same stream: it needs 6 blocks
    //   but only 5 are free - the oldest retained block must be evicted for the decode to
    //   succeed
    std::vector<llama_token> other(6*32);
    for (size_t i = 0; i < other.size(); ++i) {
        other[i] = (llama_token) (i*11%97 + 31);
    }

    ok = ok && decode_range(ctx.get(), model, 0, other, 0, (int32_t) other.size(), 1, tmp);

    // the evicted blocks invalidate their entries: block 0 is gone -> no match
    const auto pm = llama_memory_prefix_match(ctx.get(), tokens.data(), n_prefix);

    if (pm.handle != 0) {
        llama_memory_prefix_release(ctx.get(), pm.handle);
    }

    printf("matched after eviction: %d tokens (expected 0)\n", pm.n_tokens);

    if (pm.n_tokens != 0) {
        fprintf(stderr, "%s: expected no match after eviction\n", __func__);
        ok = false;
    }

    if (!ok) {
        fprintf(stderr, "%s: FAILED\n", __func__);
        return false;
    }

    printf("%s: OK\n", __func__);

    return true;
}

static bool test_arch(const llm_arch arch, const size_t seed) {
    printf("=== %s: arch = %s ===\n", __func__, llm_arch_name(arch));

    gguf_context_ptr gguf_ctx = get_gguf_ctx(arch);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;

    size_t tmp = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tmp, model_params));
    if (!model) {
        fprintf(stderr, "%s: failed to create model\n", __func__);
        return false;
    }

    const auto tokens = make_prompt();

    const run_result ref      = run_reference(model.get(), tokens);
    const run_result copy     = run_copy(model.get(), tokens, false);
    const run_result rm       = run_copy(model.get(), tokens, true);
    const run_result reattach = run_reattach(model.get(), tokens);

    bool ok = true;

    if (!ref.ok || !copy.ok || !rm.ok || !reattach.ok) {
        fprintf(stderr, "%s: run failed\n", __func__);
        return false;
    }

    printf("matched: %d tokens (expected %d)\n", copy.n_matched, n_prefix);

    if (copy.n_matched != (uint32_t) n_prefix || rm.n_matched != (uint32_t) n_prefix ||
        reattach.n_matched != (uint32_t) n_prefix) {
        fprintf(stderr, "%s: unexpected match length\n", __func__);
        ok = false;
    }

    auto cmp = [&](const char * what, const std::vector<float> & a, const std::vector<float> & b) {
        if (a.size() != b.size()) {
            fprintf(stderr, "%s: %s size mismatch\n", __func__, what);
            ok = false;
            return;
        }
        // the copy and reference paths compute the same values through different shapes/paths,
        // which can reorder float reductions (esp. on GPU backends) - allow ulp-level noise
        size_t i_bad = 0;
        size_t n_bad = 0;
        float  d_max = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            const float d = std::fabs(a[i] - b[i]);
            if (d > 1e-5f*std::max(1.0f, std::fabs(a[i]))) {
                if (n_bad == 0) {
                    i_bad = i;
                }
                n_bad++;
                d_max = std::max(d_max, d);
            }
        }
        if (n_bad > 0) {
            fprintf(stderr, "%s: %s mismatch at %zu: %a vs %a (%zu bad, max diff %g)\n", __func__, what, i_bad, (double) a[i_bad], (double) b[i_bad], n_bad, (double) d_max);
            ok = false;
        }
    };

    cmp("suffix logits (copy)",       ref.logits_suffix, copy.logits_suffix);
    cmp("suffix logits (donor rm)",   ref.logits_suffix, rm.logits_suffix);
    cmp("suffix logits (reattach)",   ref.logits_suffix, reattach.logits_suffix);
    cmp("extra logits  (copy)",       ref.logits_extra,  copy.logits_extra);
    cmp("extra logits  (donor rm)",   ref.logits_extra,  rm.logits_extra);
    cmp("extra logits  (reattach)",   ref.logits_extra,  reattach.logits_extra);

    ok = ok && test_unaligned(arch, model.get(), tokens);
    ok = ok && test_evict(arch, model.get(), tokens);

    printf("%s: %s\n", __func__, ok ? "OK" : "FAILED");

    return ok;
}

int main() {
    llama_backend_init();

    bool ok = true;

    ok = ok && test_arch(LLM_ARCH_LLAMA,   1234);
    ok = ok && test_arch(LLM_ARCH_QWEN35,  5678);

    llama_backend_free();

    if (!ok) {
        fprintf(stderr, "%s: FAILED\n", __func__);
        return 1;
    }

    printf("%s: OK\n", __func__);
    return 0;
}
