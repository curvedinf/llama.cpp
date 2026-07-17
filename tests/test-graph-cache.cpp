// Test the llama_context graph cache:
// - alternating ubatch shapes (prompt/verify/draft, like speculative decoding) hit the cache
// - logits are bit-identical with and without graph reuse (incl. cache eviction and context shift)

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// minimal synthetic model, same approach as test-llama-archs.cpp
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

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    if (value) {
        _putenv_s(name, value);
    } else {
        _putenv_s(name, "");
    }
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// speculative-decoding-like loop with alternating ubatch shapes:
//   prompt (32 tokens, 1 output), then verify (24 tokens, 4 outputs) / draft (8 tokens, 1 output) pairs
// positions are continuous, total stays within one n_kv padding window (256)
static const int n_prompt      = 32;
static const int n_verify      = 24;
static const int n_draft       = 8;
static const int n_iter        = 7;
static const int n_iter_shift  = 3;  // apply context shift after this many iterations
static const int n_shift       = 64; // shift the context down by this many positions

struct run_result {
    std::vector<float> logits;

    int32_t n_reused = 0; // cache hits, from llama_perf_context
    int32_t n_decode = 0;

    // counters around the first decode after the context shift
    int32_t n_reused_pre_shift  = -1;
    int32_t n_reused_post_shift = -1;

    bool ok = false;
};

static run_result run_loop(llama_model * model, const char * env_reuse_disable, const char * env_cache_size, bool do_shift) {
    run_result result;

    set_env("LLAMA_GRAPH_REUSE_DISABLE", env_reuse_disable);
    set_env("LLAMA_GRAPH_CACHE_SIZE",    env_cache_size);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 512;
    ctx_params.n_batch   = 64;
    ctx_params.n_ubatch  = 64;
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;

    llama_context_ptr ctx(llama_init_from_model(model, ctx_params));

    set_env("LLAMA_GRAPH_REUSE_DISABLE", nullptr);
    set_env("LLAMA_GRAPH_CACHE_SIZE",    nullptr);

    if (!ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return result;
    }

    llama_memory_t mem = llama_get_memory(ctx.get());

    llama_batch batch = llama_batch_init(n_prompt > n_verify ? n_prompt : n_verify, 0, 1);

    int32_t n_past = 0;

    auto decode = [&](int32_t n_tokens, int32_t n_out) -> bool {
        batch.n_tokens = n_tokens;
        for (int32_t i = 0; i < n_tokens; i++) {
            batch.token    [i]   = (n_past + i) % 113 + 1;
            batch.pos      [i]   = n_past + i;
            batch.n_seq_id [i]   = 1;
            batch.seq_id   [i][0] = 0;
            batch.logits   [i]   = i >= n_tokens - n_out;
        }

        if (llama_decode(ctx.get(), batch) != 0) {
            fprintf(stderr, "%s: llama_decode failed\n", __func__);
            return false;
        }

        const int64_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        for (int32_t i = n_tokens - n_out; i < n_tokens; i++) {
            const float * logits_i = llama_get_logits_ith(ctx.get(), i);
            if (!logits_i) {
                fprintf(stderr, "%s: failed to get logits\n", __func__);
                return false;
            }
            result.logits.insert(result.logits.end(), logits_i, logits_i + n_vocab);
        }

        n_past += n_tokens;
        result.n_decode++;

        return true;
    };

    bool ok = true;

    // prompt
    ok = ok && decode(n_prompt, 1);

    for (int i = 0; ok && i < n_iter; i++) {
        if (do_shift && i == n_iter_shift) {
            llama_memory_seq_rm (mem, 0, 0,       n_shift);
            llama_memory_seq_add(mem, 0, n_shift, -1, -n_shift);
            n_past -= n_shift;

            result.n_reused_pre_shift = llama_perf_context(ctx.get()).n_reused;
        }

        ok = ok && decode(n_verify, 4);

        if (do_shift && i == n_iter_shift) {
            // the verify shape was cached before the shift - it must still hit afterwards
            result.n_reused_post_shift = llama_perf_context(ctx.get()).n_reused;
        }

        ok = ok && decode(n_draft, 1);
    }

    llama_synchronize(ctx.get());

    result.n_reused = llama_perf_context(ctx.get()).n_reused;
    result.ok       = ok;

    llama_batch_free(batch);

    return result;
}

static bool test_arch(const llm_arch arch, const size_t seed, bool do_shift) {
    printf("=== %s: arch = %s, shift = %d ===\n", __func__, llm_arch_name(arch), (int) do_shift);

    gguf_context_ptr gguf_ctx = get_gguf_ctx(arch);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;

    size_t tmp = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tmp, model_params));
    if (!model) {
        fprintf(stderr, "%s: failed to create model\n", __func__);
        return false;
    }

    // same model, 3 contexts: cache enabled (default), reuse disabled, cache size 1 (eviction on every switch)
    const run_result res_cached   = run_loop(model.get(), nullptr, nullptr, do_shift);
    const run_result res_disabled = run_loop(model.get(), "1",     nullptr, do_shift);
    const run_result res_evict    = run_loop(model.get(), nullptr, "1",     do_shift);

    bool ok = true;

    if (!res_cached.ok || !res_disabled.ok || !res_evict.ok) {
        fprintf(stderr, "%s: decode loop failed\n", __func__);
        return false;
    }

    const int32_t n_miss_cached = res_cached.n_decode - res_cached.n_reused;

    printf("cached:   decodes = %d, hits = %d, misses = %d\n", res_cached.n_decode, res_cached.n_reused, n_miss_cached);
    printf("disabled: decodes = %d, hits = %d\n", res_disabled.n_decode, res_disabled.n_reused);
    printf("evict:    decodes = %d, hits = %d\n", res_evict.n_decode, res_evict.n_reused);

    // 3 distinct shapes (prompt, verify, draft) -> 3 misses, everything else must hit
    if (res_cached.n_reused < res_cached.n_decode - 3 || n_miss_cached > 4) {
        fprintf(stderr, "%s: unexpected cache hit/miss counts\n", __func__);
        ok = false;
    }

    if (res_disabled.n_reused != 0 || res_evict.n_reused != 0) {
        fprintf(stderr, "%s: expected zero cache hits\n", __func__);
        ok = false;
    }

    // the memory update (context shift) must not invalidate the cached graphs
    if (do_shift) {
        if (res_cached.n_reused_post_shift != res_cached.n_reused_pre_shift + 1) {
            fprintf(stderr, "%s: cache did not survive the memory update (hits %d -> %d)\n", __func__,
                    res_cached.n_reused_pre_shift, res_cached.n_reused_post_shift);
            ok = false;
        }
    }

    // logits must be bit-identical between all runs
    if (res_cached.logits.size() != res_disabled.logits.size() || res_cached.logits.size() != res_evict.logits.size()) {
        fprintf(stderr, "%s: logits size mismatch\n", __func__);
        ok = false;
    } else {
        const size_t n = res_cached.logits.size();
        if (memcmp(res_cached.logits.data(), res_disabled.logits.data(), n*sizeof(float)) != 0 ||
            memcmp(res_cached.logits.data(), res_evict.logits.data(),    n*sizeof(float)) != 0) {
            size_t i_bad = 0;
            for (size_t i = 0; i < n; i++) {
                if (res_cached.logits[i] != res_disabled.logits[i] || res_cached.logits[i] != res_evict.logits[i]) {
                    i_bad = i;
                    break;
                }
            }
            fprintf(stderr, "%s: logits mismatch at %zu: %a vs %a vs %a\n", __func__, i_bad,
                    (double) res_cached.logits[i_bad], (double) res_disabled.logits[i_bad], (double) res_evict.logits[i_bad]);
            ok = false;
        }
    }

    printf("%s: %s\n", __func__, ok ? "OK" : "FAILED");

    return ok;
}

int main() {
    llama_backend_init();

    bool ok = true;

    ok = ok && test_arch(LLM_ARCH_LLAMA,  1234, true);
    ok = ok && test_arch(LLM_ARCH_QWEN35, 5678, false);

    llama_backend_free();

    if (!ok) {
        fprintf(stderr, "%s: FAILED\n", __func__);
        return 1;
    }

    printf("%s: OK\n", __func__);
    return 0;
}
