// ux-bench: multi-user UX benchmark.
//
// Schedules N_USERS virtual users against the llama library, replicating the
// server's pre_decode scheduler behavior with toggleable policy knobs. Each
// user has a deterministic but random-ish profile (arrival offset, prompt
// length, generation length) so the workload is reproducible across runs and
// scheduler variants.
//
// Reports user-experience metrics, not raw throughput:
//   - TTFT: time-to-first-token from each user's arrival
//   - per-token inter-arrival p50/p90/p99 (the "did it stutter?" signal)
//   - aggregate throughput as a secondary sanity metric
//
// Policy knobs (env, all default off = baseline behavior):
//   LLAMA_UX_DYNAMIC_BUDGET=1   P1: prefill chunk = max(MIN, n_batch - n_generating)
//   LLAMA_UX_FIRST_TOKEN=1      P4: decode the first token in the same iteration
//                                 as the last prefill chunk (saves one tick)
//   LLAMA_UX_SLO_ADMIT=1        P3: admit shortest-prompt-first within priority
//   LLAMA_UX_MIN_CHUNK=<N>      dynamic budget floor (default 64)
//
// Workload knobs (env):
//   N_USERS=16        SEED=0x5EED5EED
//   MEAN_GAP_MS=200   PROMPT_MU=5.55  PROMPT_SIGMA=1.2  PROMPT_MIN=64   PROMPT_MAX=8192
//   GEN_MU=4.16       GEN_SIGMA=0.9   GEN_MIN=16        GEN_MAX=512

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// --- deterministic PRNG (xorshift64) ---
struct rng_t {
    uint64_t s;
    explicit rng_t(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t u64() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    double f64() { return (double)(u64() >> 11) / (double)(1ULL << 53); }
    // Box-Muller for a normal sample
    double normal() {
        double u1 = std::max(1e-12, f64());
        double u2 = f64();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
    // lognormal sample, clamped
    int lognormal_clamped(double mu, double sigma, int lo, int hi) {
        double v = std::exp(mu + sigma * normal());
        int iv = (int) std::llround(v);
        return std::max(lo, std::min(hi, iv));
    }
    // exponential sample in microseconds, given mean in microseconds
    int64_t exponential_us(double mean_us) {
        double u = std::max(1e-12, f64());
        return (int64_t) std::llround(-mean_us * std::log(u));
    }
};

// --- env helpers ---
static int env_int(const char * k, int def) {
    const char * e = getenv(k); return e ? atoi(e) : def;
}
static double env_double(const char * k, double def) {
    const char * e = getenv(k); return e ? atof(e) : def;
}
// default-on flag: enabled unless explicitly set to "0"
static bool env_flag_on(const char * k) {
    const char * e = getenv(k); return e == nullptr || e[0] != '0';
}

// --- user state ---
enum user_state_t { US_WAITING, US_PREFILLING, US_GENERATING, US_DONE };

struct user_t {
    int      id;
    int64_t  arrival_us;   // relative to bench start
    int      prompt_len;
    int      gen_len;

    user_state_t state = US_WAITING;
    int      prompt_pos  = 0;
    int      n_generated = 0;
    llama_pos pos = 0;

    // metrics (microseconds since bench start)
    int64_t  t_admit       = -1;
    int64_t  t_first_token = -1;
    std::vector<int64_t> token_times;   // timestamps of each generated token (relative to bench start)
};

// --- stats helpers ---
static double percentile(const std::vector<double> & sorted, double p) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    double idx = p * (sorted.size() - 1);
    size_t lo = (size_t) std::floor(idx);
    size_t hi = (size_t) std::min(sorted.size() - 1, lo + 1);
    double frac = idx - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

struct summary_t {
    double p50, p90, p99, max;
};

static summary_t summarize(std::vector<double> xs) {
    std::sort(xs.begin(), xs.end());
    return { percentile(xs, 0.50), percentile(xs, 0.90), percentile(xs, 0.99),
             xs.empty() ? 0.0 : xs.back() };
}

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("  %s -m model.gguf -c 67584 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024\n", argv[0]);
    LOG("  env: N_USERS MEAN_GAP_MS PROMPT_MU PROMPT_SIGMA GEN_MU GEN_SIGMA SEED\n");
    LOG("       LLAMA_UX_DYNAMIC_BUDGET LLAMA_UX_FIRST_TOKEN LLAMA_UX_SLO_ADMIT LLAMA_UX_MIN_CHUNK\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    // workload config
    const int    N_USERS      = env_int   ("N_USERS",      16);
    const int    SEED          = env_int   ("SEED",         (int)0x5EED5EED);
    const double MEAN_GAP_MS   = env_double("MEAN_GAP_MS",  200.0);
    const double PROMPT_MU     = env_double("PROMPT_MU",    5.55);
    const double PROMPT_SIGMA  = env_double("PROMPT_SIGMA", 1.2);
    const int    PROMPT_MIN    = env_int   ("PROMPT_MIN",   64);
    const int    PROMPT_MAX    = env_int   ("PROMPT_MAX",   8192);
    const double GEN_MU        = env_double("GEN_MU",       4.16);
    const double GEN_SIGMA     = env_double("GEN_SIGMA",    0.9);
    const int    GEN_MIN       = env_int   ("GEN_MIN",      16);
    const int    GEN_MAX       = env_int   ("GEN_MAX",      512);

    // scheduler policy (default ON; set to "0" to disable for A/B comparison)
    const bool P1_DYNAMIC_BUDGET = env_flag_on("LLAMA_UX_DYNAMIC_BUDGET");
    const bool P4_FIRST_TOKEN    = env_flag_on("LLAMA_UX_FIRST_TOKEN");
    const bool P3_SLO_ADMIT      = env_flag_on("LLAMA_UX_SLO_ADMIT");
    const int  MIN_CHUNK         = env_int ("LLAMA_UX_MIN_CHUNK", 256);

    const int n_batch  = params.n_batch;
    const int n_ubatch = params.n_ubatch;

    if (N_USERS <= 0) {
        LOG_ERR("N_USERS must be positive\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) { LOG_ERR("model load failed\n"); return 1; }

    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.n_seq_max = N_USERS;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) { LOG_ERR("context create failed\n"); return 1; }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    // deterministic token source: per-call idx mixed with seed
    auto gen_token = [n_vocab, SEED](int user_id, int token_idx) -> llama_token {
        uint64_t s = (uint64_t)SEED ^ ((uint64_t)user_id * 0x9E3779B97F4A7C15ULL)
                                     ^ ((uint64_t)token_idx * 0xBF58476D1CE4E5B9ULL);
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (llama_token)(s % n_vocab);
    };

    // generate user profiles deterministically
    std::vector<user_t> users(N_USERS);
    {
        rng_t rng((uint64_t)SEED);
        int64_t arrival_accum = 0;
        // generate one at a time so arrival_offsets are cumulative
        for (int i = 0; i < N_USERS; ++i) {
            users[i].id          = i;
            users[i].arrival_us  = arrival_accum;
            users[i].prompt_len  = rng.lognormal_clamped(PROMPT_MU, PROMPT_SIGMA, PROMPT_MIN, PROMPT_MAX);
            users[i].gen_len     = rng.lognormal_clamped(GEN_MU,    GEN_SIGMA,    GEN_MIN,    GEN_MAX);
            arrival_accum += rng.exponential_us(MEAN_GAP_MS * 1000.0);
        }
    }

    // sanity: total KV needed
    const int32_t n_kv_max = llama_n_ctx(ctx);
    int64_t kv_needed = 0;
    for (auto & u : users) kv_needed += u.prompt_len + u.gen_len;
    if (kv_needed > n_kv_max) {
        LOG_ERR("kv_needed=%lld > n_kv_max=%d (reduce N_USERS / prompt / gen or raise -c)\n",
                (long long)kv_needed, n_kv_max);
        return 1;
    }

    llama_batch batch = llama_batch_init(n_batch, 0, N_USERS);

    LOG("\nux-bench: N_USERS=%d seed=0x%08X mean_gap=%.0fms  prompt=N(%.2f,%.2f)[%d,%d] gen=N(%.2f,%.2f)[%d,%d]\n",
        N_USERS, (unsigned)SEED, MEAN_GAP_MS, PROMPT_MU, PROMPT_SIGMA, PROMPT_MIN, PROMPT_MAX,
        GEN_MU, GEN_SIGMA, GEN_MIN, GEN_MAX);
    LOG("policies: dynamic_budget=%c first_token=%c slo_admit=%c  min_chunk=%d  (n_batch=%d n_ubatch=%d)\n",
        P1_DYNAMIC_BUDGET ? 'Y' : 'N', P4_FIRST_TOKEN ? 'Y' : 'N', P3_SLO_ADMIT ? 'Y' : 'N',
        MIN_CHUNK, n_batch, n_ubatch);
    LOG("user profiles (id: arrival_ms prompt_len gen_len):\n");
    for (auto & u : users) {
        LOG("  u%d: %8.1f %5d %4d\n", u.id, u.arrival_us/1000.0, u.prompt_len, u.gen_len);
    }

    const auto t_start_us = ggml_time_us();

    int64_t tokens_decoded_total = 0;
    int64_t tokens_prefilled_total = 0;
    int     iterations = 0;

    auto any_active = [&]() -> bool {
        for (auto & u : users) if (u.state != US_DONE) return true;
        return false;
    };

    while (any_active()) {
        const int64_t t_now = ggml_time_us() - t_start_us;

        // 1) admit users whose arrival time has passed
        for (auto & u : users) {
            if (u.state == US_WAITING && t_now >= u.arrival_us) {
                u.state    = US_PREFILLING;
                u.t_admit  = t_now;
            }
        }

        // 2) build the batch
        common_batch_clear(batch);

        // 2a) one decode token for each generating user (under budget)
        int n_generating_this_iter = 0;
        for (auto & u : users) {
            if (u.state != US_GENERATING) continue;
            if (u.n_generated >= u.gen_len) continue;
            common_batch_add(batch, gen_token(u.id, u.prompt_len + u.n_generated),
                             u.pos++, { u.id }, true);
            u.n_generated++;
            tokens_decoded_total++;
            n_generating_this_iter++;
        }

        // 2b) prefill tokens for prefilling users, under the chunk budget.
        // policy: order = priority class (none here) then SJF if SLO_ADMIT, else FIFO.
        std::vector<user_t *> queue;
        for (auto & u : users) if (u.state == US_PREFILLING) queue.push_back(&u);
        if (P3_SLO_ADMIT) {
            std::sort(queue.begin(), queue.end(),
                      [](const user_t * a, const user_t * b) { return a->prompt_len < b->prompt_len; });
        }

        for (user_t * up : queue) {
            if (batch.n_tokens >= n_batch) break;
            const size_t n_before = batch.n_tokens;

            // budget: how many tokens may THIS user add this iteration?
            // baseline: n_batch (legacy cap-on-loop), or env-fixed cap via LLAMA_PREFILL_CHUNK
            // P1 dynamic: each prefilling slot gets an equal share of (n_batch - n_generating),
            //   floored at MIN_CHUNK, so multiple slots can prefill per iteration instead of
            //   one slot consuming the whole batch. This is the vLLM/Sarathi-Serve pattern.
            int budget;
            {
                const char * e = getenv("LLAMA_PREFILL_CHUNK");
                int fixed = e ? atoi(e) : 0;
                if (P1_DYNAMIC_BUDGET) {
                    const int n_queue = std::max(1, (int) queue.size());
                    budget = std::max(MIN_CHUNK, (n_batch - n_generating_this_iter) / n_queue);
                } else if (fixed > 0) {
                    budget = fixed;
                } else {
                    budget = n_batch;  // legacy: one slot may consume the whole batch
                }
                // also bounded by what's left of n_batch this iteration
                budget = std::min(budget, n_batch - (int)batch.n_tokens);
            }

            const int remaining_prompt = up->prompt_len - up->prompt_pos;
            const int this_chunk = std::min(remaining_prompt, budget);

            for (int t = 0; t < this_chunk; ++t) {
                const bool last_prompt_token = (up->prompt_pos + t + 1 == up->prompt_len);
                common_batch_add(batch, gen_token(up->id, up->prompt_pos + t),
                                 up->pos++, { up->id }, last_prompt_token);
                up->prompt_pos++;
                tokens_prefilled_total++;
            }
            (void) n_before;
        }

        if (batch.n_tokens == 0) {
            // no work to do this tick (all waiting users haven't arrived yet); sleep briefly
            // to avoid a busy spin
            ggml_time_us(); // ensure time advances
            iterations++;
            continue;
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("decode failed at iter %d (n_tokens=%d)\n", iterations, batch.n_tokens);
            break;
        }
        llama_synchronize(ctx);

        // 3) per-iteration bookkeeping
        const int64_t t_post_decode = ggml_time_us() - t_start_us;

        for (auto & u : users) {
            if (u.state == US_PREFILLING && u.prompt_pos >= u.prompt_len) {
                // prompt finished this iteration
                u.state = US_GENERATING;
                if (P4_FIRST_TOKEN) {
                    // the last prompt token's logits ARE the first generated token's logits;
                    // emit it now (we'll record the timestamp below using the post-decode time)
                    u.token_times.push_back(t_post_decode);
                    u.t_first_token = t_post_decode;
                    u.n_generated = 1;  // count this as the first generated token
                    tokens_decoded_total++;  // matches: we treat last-prompt-token-eval as gen
                    if (u.n_generated >= u.gen_len) {
                        u.state = US_DONE;
                    }
                }
            } else if (u.state == US_GENERATING) {
                // a decode happened for this user this iteration
                if (!P4_FIRST_TOKEN || u.token_times.empty() || u.token_times.size() < (size_t)u.n_generated) {
                    u.token_times.push_back(t_post_decode);
                    if (u.t_first_token < 0) u.t_first_token = t_post_decode;
                }
                if (u.n_generated >= u.gen_len) {
                    u.state = US_DONE;
                }
            }
        }

        iterations++;
    }

    const int64_t t_total_us = ggml_time_us() - t_start_us;
    const double  t_total_s  = t_total_us / 1e6;

    // --- compute UX metrics ---
    // TTFT per user: t_first_token - arrival_us
    std::vector<double> ttft_ms;
    std::vector<double> queue_ms;        // t_admit - arrival
    std::vector<double> all_gaps_ms;     // all inter-token gaps across all users
    std::vector<double> per_user_p90_gap_ms;  // p90 gap per user, then aggregated
    for (auto & u : users) {
        if (u.t_first_token < 0) continue;
        ttft_ms.push_back((u.t_first_token - u.arrival_us) / 1000.0);
        queue_ms.push_back((u.t_admit - u.arrival_us) / 1000.0);
        // inter-token gaps within this user
        std::vector<double> gaps;
        for (size_t i = 1; i < u.token_times.size(); ++i) {
            double g = (u.token_times[i] - u.token_times[i-1]) / 1000.0;
            gaps.push_back(g);
            all_gaps_ms.push_back(g);
        }
        if (!gaps.empty()) {
            per_user_p90_gap_ms.push_back(percentile(gaps, 0.90));  // note: not sorted yet, percentile sorts copy
        }
    }

    auto ttft_s     = summarize(ttft_ms);
    auto queue_s    = summarize(queue_ms);
    auto gaps_s     = summarize(all_gaps_ms);
    auto user_p90_s = summarize(per_user_p90_gap_ms);

    const double total_gen_tokens = (double) tokens_decoded_total;
    const double total_pp_tokens  = (double) tokens_prefilled_total;

    LOG("\n");
    LOG("======================== ux-bench results ========================\n");
    LOG("iterations          : %d\n", iterations);
    LOG("wall clock          : %.3f s\n", t_total_s);
    LOG("tokens prefill      : %lld\n", (long long)tokens_prefilled_total);
    LOG("tokens decode       : %lld\n", (long long)tokens_decoded_total);
    LOG("throughput decode   : %.1f tok/s\n", total_gen_tokens / t_total_s);
    LOG("throughput total    : %.1f tok/s\n", (total_gen_tokens + total_pp_tokens) / t_total_s);
    LOG("..................................................................\n");
    LOG("UX metrics (lower is better, ms):\n");
    LOG("  TTFT              p50=%7.1f  p90=%7.1f  p99=%7.1f  max=%7.1f   (n=%zu)\n",
        ttft_s.p50, ttft_s.p90, ttft_s.p99, ttft_s.max, ttft_ms.size());
    LOG("  queue wait        p50=%7.1f  p90=%7.1f  p99=%7.1f  max=%7.1f   (n=%zu)\n",
        queue_s.p50, queue_s.p90, queue_s.p99, queue_s.max, queue_ms.size());
    LOG("  inter-token gap   p50=%7.1f  p90=%7.1f  p99=%7.1f  max=%7.1f   (n=%zu)\n",
        gaps_s.p50, gaps_s.p90, gaps_s.p99, gaps_s.max, all_gaps_ms.size());
    LOG("  per-user p90 gap  p50=%7.1f  p90=%7.1f  p99=%7.1f  max=%7.1f   (n=%zu, worse-case user view)\n",
        user_p90_s.p50, user_p90_s.p90, user_p90_s.p99, user_p90_s.max, per_user_p90_gap_ms.size());
    LOG("..................................................................\n");
    LOG("per-user detail (id: ttft_ms  p50_gap  p90_gap  p99_gap  n_tokens):\n");
    for (auto & u : users) {
        std::vector<double> gaps;
        for (size_t i = 1; i < u.token_times.size(); ++i) {
            gaps.push_back((u.token_times[i] - u.token_times[i-1]) / 1000.0);
        }
        auto g = summarize(gaps);
        double ttft = u.t_first_token >= 0 ? (u.t_first_token - u.arrival_us) / 1000.0 : -1.0;
        LOG("  u%d: ttft=%7.1f  gap_p50=%5.1f  p90=%5.1f  p99=%6.1f  n=%zu  (arrival=%5.0fms prompt=%d gen=%d)\n",
            u.id, ttft, g.p50, g.p90, g.p99, u.token_times.size(),
            u.arrival_us/1000.0, u.prompt_len, u.gen_len);
    }
    LOG("==================================================================\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
