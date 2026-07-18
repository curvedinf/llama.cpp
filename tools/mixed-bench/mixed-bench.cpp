// mixed-bench: simulate a mixed prefill + decode workload the way a continuous
// batching server sees it. Warm up N_decode sequences so they are mid-decode,
// then admit N_arrive fresh prompts of PP_arrive tokens each, prefilling them
// in LLAMA_PREFILL_CHUNK-sized chunks while the warm decoders keep emitting.
//
// Reports: overall tok/s, decode-only tok/s during the mixed phase, average
// per-iteration wall time, and the average decode token latency.
//
// The chunk cap is read from LLAMA_PREFILL_CHUNK (same env var as the server),
// so this bench exercises the same knob the server scheduler exposes.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct seq_state {
    int32_t id;
    bool    is_decoding;       // true once prompt is fully prefilled
    int32_t prompt_pos;        // tokens prefilled so far (arriving seqs)
    int32_t prompt_target;     // arriving prompt length
    int32_t n_decoded;
    int32_t n_decode_target;
    llama_pos pos;             // current seq position
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m model.gguf -c 67584 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024\n", argv[0]);
    LOG("  env: N_DECODE=8 N_ARRIVE=8 PP_ARRIVE=4096 TG_PER_SEQ=128 LLAMA_PREFILL_CHUNK=512\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    const int n_decode = []{ const char * e = getenv("N_DECODE");     return e ? atoi(e) : 8;  }();
    const int n_arrive = []{ const char * e = getenv("N_ARRIVE");     return e ? atoi(e) : 8;  }();
    const int pp_arrive = []{ const char * e = getenv("PP_ARRIVE");   return e ? atoi(e) : 4096; }();
    const int tg_per_seq = []{ const char * e = getenv("TG_PER_SEQ"); return e ? atoi(e) : 128; }();
    const int prefill_chunk = []{
        const char * e = getenv("LLAMA_PREFILL_CHUNK");
        return e ? atoi(e) : 0;
    }();

    const int n_seq = n_decode + n_arrive;
    const int n_batch  = params.n_batch;
    const int n_ubatch = params.n_ubatch;

    if (n_seq <= 0 || n_decode <= 0) {
        LOG_ERR("N_DECODE and total N_SEQ must be positive\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.n_seq_max = n_seq;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("%s: failed to create llama_context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    auto get_token_rand = [n_vocab]() -> llama_token {
        return std::rand() % n_vocab;
    };

    auto * mem = llama_get_memory(ctx);
    const int32_t n_kv_max = llama_n_ctx(ctx);

    const int32_t kv_needed = n_decode * (64 + tg_per_seq) + n_arrive * (pp_arrive + tg_per_seq);
    if (kv_needed > n_kv_max) {
        LOG_ERR("kv_needed = %d > n_kv_max = %d (increase -c)\n", kv_needed, n_kv_max);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_batch batch = llama_batch_init(n_batch, 0, n_seq);

    std::vector<seq_state> seqs(n_seq);
    for (int i = 0; i < n_seq; ++i) {
        seqs[i].id             = i;
        seqs[i].n_decoded      = 0;
        seqs[i].n_decode_target = tg_per_seq;
        seqs[i].pos            = 0;
        seqs[i].prompt_pos     = 0;
        seqs[i].prompt_target  = (i < n_decode) ? 64 : pp_arrive;
        seqs[i].is_decoding    = false;
    }

    // Warmup: prefill the first n_decode seqs (small 64-token prompt), one seq
    // at a time, then mark them as decoding. Done outside the measured window.
    for (int i = 0; i < n_decode; ++i) {
        common_batch_clear(batch);
        auto & s = seqs[i];
        for (int t = 0; t < s.prompt_target; ++t) {
            common_batch_add(batch, get_token_rand(), s.pos++, { s.id }, t == s.prompt_target - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("warmup prefill failed for seq %d\n", i);
            return 1;
        }
        s.is_decoding = true;
        s.n_decoded   = 0;
    }
    llama_synchronize(ctx);
    (void) mem;

    LOG("\nmixed-bench: N_DECODE=%d N_ARRIVE=%d PP_ARRIVE=%d TG_PER_SEQ=%d chunk=%d (n_batch=%d n_ubatch=%d)\n",
        n_decode, n_arrive, pp_arrive, tg_per_seq, prefill_chunk, n_batch, n_ubatch);

    // Measured mixed phase: loop until every seq has finished its decode budget.
    // Each iteration assembles one batch: 1 decode token for each decoding seq,
    // plus up to `prefill_chunk` prompt tokens for at most one arriving seq.
    int64_t tokens_decoded_mixed = 0;
    int64_t tokens_prefilled_mixed = 0;
    int     iterations = 0;
    int     next_arrive_to_prefill = n_decode; // index of next arriving seq needing prefill

    const auto t_start = ggml_time_us();

    while (true) {
        // termination: all seqs decoded enough
        bool any_active = false;
        for (auto & s : seqs) {
            if (s.n_decoded < s.n_decode_target) { any_active = true; break; }
        }
        if (!any_active) break;

        common_batch_clear(batch);

        // 1) one decode token per decoding seq (still under budget)
        for (auto & s : seqs) {
            if (!s.is_decoding) continue;
            if (s.n_decoded >= s.n_decode_target) continue;
            common_batch_add(batch, get_token_rand(), s.pos++, { s.id }, true);
            s.n_decoded++;
            tokens_decoded_mixed++;
        }

        // 2) up to `prefill_chunk` prompt tokens from one arriving seq
        //    (round-robin: finish one slot's prompt before starting the next,
        //     matching the server scheduler's anti-starvation policy)
        if (next_arrive_to_prefill < n_seq) {
            auto & s = seqs[next_arrive_to_prefill];
            const int remaining_prompt = s.prompt_target - s.prompt_pos;
            const int this_chunk = prefill_chunk > 0
                ? std::min({remaining_prompt, prefill_chunk, n_batch - batch.n_tokens})
                : std::min({remaining_prompt,            n_batch - batch.n_tokens});
            for (int t = 0; t < this_chunk; ++t) {
                const bool last = (s.prompt_pos + t + 1 == s.prompt_target);
                common_batch_add(batch, get_token_rand(), s.pos++, { s.id }, last);
                tokens_prefilled_mixed++;
            }
            s.prompt_pos += this_chunk;
            if (s.prompt_pos >= s.prompt_target) {
                s.is_decoding = true;
                next_arrive_to_prefill++;
            }
        }

        if (batch.n_tokens == 0) break;

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("mixed decode failed at iteration %d (n_tokens=%d)\n", iterations, batch.n_tokens);
            break;
        }
        llama_synchronize(ctx);
        iterations++;
    }

    const auto t_end = ggml_time_us();
    const double t_total_s = (t_end - t_start) / 1e6;

    const double t_decoded   = tokens_decoded_mixed;
    const double t_prefilled = tokens_prefilled_mixed;
    const double t_all       = t_decoded + t_prefilled;

    LOG("mixed phase: %d iters, %.3f s, decode=%lld prefill=%lld total=%lld tokens\n",
        iterations, t_total_s,
        (long long) tokens_decoded_mixed, (long long) tokens_prefilled_mixed, (long long) t_all);
    LOG("| %5s | %5s | %5s | %8s | %8s | %8s | %8s | %8s |\n",
        "chunk", "iters", "T_s", "D_tok", "P_tok", "D_t/s", "P_t/s", "all_t/s");
    LOG("| %5d | %5d | %5.3f | %8lld | %8lld | %8.1f | %8.1f | %8.1f |\n",
        prefill_chunk, iterations, t_total_s,
        (long long) tokens_decoded_mixed, (long long) tokens_prefilled_mixed,
        t_decoded / t_total_s, t_prefilled / t_total_s, t_all / t_total_s);
    LOG("decode latency: %.3f ms/token (avg per decoding seq)\n",
        1e3 * t_total_s * n_decode / std::max<double>(1.0, t_decoded));

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
