// virtual-multigpu: simulate TP / PP communication overhead on a single GPU.
//
// Runs the real model with the same ux-bench workload (deterministic 24-user
// concurrent) and injects the would-be cross-GPU communication cost at the
// right points per iteration. Compute itself is NOT split - this is an
// overhead estimator, not a real multi-GPU run. The output tells you whether
// a given split strategy + link speed would have paid off.
//
// Modeled costs:
//   TP (tensor parallel): each layer's matmuls need ~4 all-reduce / all-gather
//   ops per layer on the activation tensor (size = n_embd * n_tokens * 4 B).
//   Steady-state compute speedup assumed = N (linear), so effective compute
//   time per iter = compute_time / N.
//
//   PP (pipeline parallel): N-1 stage-boundary transfers per iter, each of
//   activation size. With enough concurrent sequences to keep the pipeline
//   full, steady-state throughput is ~1x (no speedup from PP alone); the
//   added cost is just the N-1 transfers.
//
// Link presets (latency, bandwidth) are configurable; defaults approximate
// NVLink-NVSwitch (5us, 300 GB/s) and PCIe Gen4 x16 (15us, 32 GB/s).
//
// env knobs:
//   N_USERS=24, MEAN_GAP_MS=50, etc. (same as ux-bench workload knobs)
//   VMG_LINK=nvlink|pcie|custom      VMG_LATENCY_US=5  VMG_BW_GBS=300
//   VMG_TP=N (try 2,4,8)             VMG_PP=N (try 2,4)
//   VMG_COMPARE=1  -> run baseline + all configured strategies in one shot
//
// Example:
//   VMG_TP=4 VMG_PP=2 VMG_LINK=nvlink VMG_COMPARE=1 ./llama-virtual-multigpu -m ...

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
#include <thread>
#include <vector>

struct rng_t {
    uint64_t s;
    explicit rng_t(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t u64() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double f64() { return (double)(u64() >> 11) / (double)(1ULL << 53); }
    double normal() {
        double u1 = std::max(1e-12, f64());
        double u2 = f64();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
    int lognormal_clamped(double mu, double sigma, int lo, int hi) {
        int iv = (int) std::llround(std::exp(mu + sigma * normal()));
        return std::max(lo, std::min(hi, iv));
    }
    int64_t exponential_us(double mean_us) {
        return (int64_t) std::llround(-mean_us * std::log(std::max(1e-12, f64())));
    }
};

static int env_int(const char * k, int def) {
    const char * e = getenv(k); return e ? atoi(e) : def;
}
static double env_double(const char * k, double def) {
    const char * e = getenv(k); return e ? atof(e) : def;
}

enum user_state_t { US_WAITING, US_PREFILLING, US_GENERATING, US_DONE };
struct user_t {
    int id; int64_t arrival_us; int prompt_len; int gen_len;
    user_state_t state = US_WAITING;
    int prompt_pos = 0; int n_generated = 0; llama_pos pos = 0;
};

struct link_t {
    double latency_us;
    double bw_bytes_per_us;  // bytes/us = GB/s * 1e3
    const char * name;
};

static constexpr int    N_USERS_DFLT     = 24;
static constexpr double MEAN_GAP_MS_DFLT = 50.0;
static constexpr double PROMPT_MU        = 5.55;
static constexpr double PROMPT_SIGMA     = 1.2;
static constexpr int    PROMPT_MIN       = 64;
static constexpr int    PROMPT_MAX       = 8192;
static constexpr double GEN_MU           = 4.16;
static constexpr double GEN_SIGMA        = 0.9;
static constexpr int    GEN_MIN          = 16;
static constexpr int    GEN_MAX          = 512;

// Per-layer TP comm ops. Rough estimate for a transformer layer:
//   - after QKV proj: all-gather activation
//   - after attention output proj: all-reduce
//   - after FFN gate/up: all-gather (for column-parallel)
//   - after FFN down: all-reduce (for row-parallel)
// 4 collective ops per layer; activation size each.
static constexpr int TP_COLLECTIVES_PER_LAYER = 4;

// Per-iter PP transfers: one activation handoff per stage boundary.
// For N pipeline stages there are N-1 boundaries.
static int pp_transfers_per_iter(int n_pp_stages) { return std::max(0, n_pp_stages - 1); }

static double bytes_per_collective(int n_embd, int n_tokens_in_iter) {
    // activation tensor size for one collective
    return (double) n_embd * (double) n_tokens_in_iter * 4.0;  // f32
}

static double link_transfer_us(const link_t & link, double bytes) {
    return link.latency_us + bytes / link.bw_bytes_per_us;
}

struct run_config {
    int   tp_n = 1;     // tensor parallel degree
    int   pp_n = 1;     // pipeline parallel degree
    link_t link;
};

struct run_result {
    double wall_s;
    double compute_s;       // time spent in llama_decode
    double sim_comm_s;      // time spent in injected comm sleeps
    int    iterations;
    long   tokens_prefilled;
    long   tokens_decoded;
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("  %s -m model.gguf -c 131072 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024\n", argv[0]);
    LOG("  env: VMG_TP=4 VMG_PP=1 VMG_LINK=nvlink  (or pcie, custom)\n");
    LOG("       VMG_LATENCY_US=5 VMG_BW_GBS=300 (for custom link)\n");
    LOG("       VMG_COMPARE=1 to run baseline + all configured strategies in one shot\n");
    LOG("       N_USERS MEAN_GAP_MS SEED PROMPT_* GEN_* (workload knobs, same as ux-bench)\n\n");
}

// Run one configuration. Builds the workload deterministically from the seed,
// runs the same iteration loop as ux-bench, but injects a sleep after each
// llama_decode for the simulated comm cost.
static run_result run_one(
        llama_context * ctx, llama_model * model,
        int n_users, double mean_gap_ms, uint64_t seed,
        const run_config & cfg, int n_batch, int n_embd, int n_layers) {

    // regenerate the user profiles (same as ux-bench)
    std::vector<user_t> users(n_users);
    {
        rng_t rng(seed);
        int64_t arrival_accum = 0;
        for (int i = 0; i < n_users; ++i) {
            users[i].id = i;
            users[i].arrival_us = arrival_accum;
            users[i].prompt_len = rng.lognormal_clamped(PROMPT_MU, PROMPT_SIGMA, PROMPT_MIN, PROMPT_MAX);
            users[i].gen_len    = rng.lognormal_clamped(GEN_MU, GEN_SIGMA, GEN_MIN, GEN_MAX);
            arrival_accum += rng.exponential_us(mean_gap_ms * 1000.0);
        }
    }

    llama_batch batch = llama_batch_init(n_batch, 0, n_users);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    auto gen_token = [n_vocab, seed](int uid, int idx) -> llama_token {
        uint64_t s = (uint64_t)seed ^ ((uint64_t)uid * 0x9E3779B97F4A7C15ULL)
                                     ^ ((uint64_t)idx * 0xBF58476D1CE4E5B9ULL);
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (llama_token)(s % n_vocab);
    };

    const auto t_start = ggml_time_us();
    double compute_total_us = 0;
    double sim_comm_total_us = 0;
    int iterations = 0;
    long tok_pp = 0, tok_tg = 0;

    auto any_active = [&]() -> bool {
        for (auto & u : users) if (u.state != US_DONE) return true;
        return false;
    };

    while (any_active()) {
        const int64_t t_now = ggml_time_us() - t_start;

        for (auto & u : users) {
            if (u.state == US_WAITING && t_now >= u.arrival_us) {
                u.state = US_PREFILLING;
            }
        }

        common_batch_clear(batch);
        int n_gen_this_iter = 0;
        for (auto & u : users) {
            if (u.state != US_GENERATING) continue;
            if (u.n_generated >= u.gen_len) continue;
            common_batch_add(batch, gen_token(u.id, u.prompt_len + u.n_generated),
                             u.pos++, { u.id }, true);
            u.n_generated++;
            tok_tg++;
            n_gen_this_iter++;
        }

        for (auto & u : users) {
            if (u.state != US_PREFILLING) continue;
            if (batch.n_tokens >= n_batch) break;
            const int budget = n_batch - batch.n_tokens;
            const int remaining = u.prompt_len - u.prompt_pos;
            const int chunk = std::min(remaining, budget);
            for (int t = 0; t < chunk; ++t) {
                const bool last = (u.prompt_pos + t + 1 == u.prompt_len);
                common_batch_add(batch, gen_token(u.id, u.prompt_pos + t),
                                 u.pos++, { u.id }, last);
                u.prompt_pos++;
                tok_pp++;
            }
        }

        if (batch.n_tokens == 0) {
            iterations++;
            continue;
        }

        const int64_t t_pre_decode = ggml_time_us();
        if (llama_decode(ctx, batch) != 0) break;
        llama_synchronize(ctx);
        const int64_t t_post_decode = ggml_time_us();
        compute_total_us += t_post_decode - t_pre_decode;

        // simulate comm: TP collectives (activation * tp_collectives * layers) + PP transfers
        // The activation size uses the iter's n_tokens. Smaller iter -> smaller comm.
        const int n_tokens_iter = batch.n_tokens;
        const double bytes = bytes_per_collective(n_embd, n_tokens_iter);
        double comm_us = 0.0;
        if (cfg.tp_n > 1) {
            const int n_collectives = TP_COLLECTIVES_PER_LAYER * n_layers;
            // For TP, all-reduce has a "ring" cost ~ 2*(N-1)*latency + 2*N*size/bw per collective
            // (ring all-reduce). Approximate as link_transfer * (2 * (N-1)/N) for the bandwidth
            // part (each GPU sends/receives 2*(N-1)/N of the data) plus 2*(N-1) latencies.
            const double ar_us = cfg.link.latency_us * 2.0 * (cfg.tp_n - 1)
                               + link_transfer_us(cfg.link, bytes) * (2.0 * (cfg.tp_n - 1) / cfg.tp_n);
            comm_us += n_collectives * ar_us;
        }
        if (cfg.pp_n > 1) {
            const int n_xfers = pp_transfers_per_iter(cfg.pp_n);
            comm_us += n_xfers * link_transfer_us(cfg.link, bytes);
        }
        if (comm_us > 0) {
            // sleep for the simulated comm cost. Use busy_wait for sub-millisecond accuracy.
            const auto t_comm_start = ggml_time_us();
            while (ggml_time_us() - t_comm_start < (int64_t) comm_us) {
                // spin
            }
            sim_comm_total_us += comm_us;
        }

        // state transitions
        for (auto & u : users) {
            if (u.state == US_PREFILLING && u.prompt_pos >= u.prompt_len) {
                u.state = US_GENERATING;
            }
            if (u.state == US_GENERATING && u.n_generated >= u.gen_len) {
                u.state = US_DONE;
            }
        }
        iterations++;
    }

    const int64_t t_end = ggml_time_us();
    llama_batch_free(batch);

    return run_result{
        (t_end - t_start) / 1e6,
        compute_total_us / 1e6,
        sim_comm_total_us / 1e6,
        iterations,
        tok_pp, tok_tg
    };
}

static link_t parse_link() {
    const char * preset = getenv("VMG_LINK");
    if (preset == nullptr || !strcmp(preset, "nvlink")) {
        return {5.0, 300.0 * 1e3, "nvlink"};  // 5us, 300 GB/s
    }
    if (!strcmp(preset, "pcie")) {
        return {15.0, 32.0 * 1e3, "pcie"};    // 15us, 32 GB/s
    }
    // custom
    double lat = env_double("VMG_LATENCY_US", 5.0);
    double bw  = env_double("VMG_BW_GBS", 300.0) * 1e3;
    return {lat, bw, "custom"};
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    const int    N_USERS     = env_int("N_USERS", N_USERS_DFLT);
    const int    SEED        = env_int("SEED", (int)0x5EED5EED);
    const double MEAN_GAP_MS = env_double("MEAN_GAP_MS", MEAN_GAP_MS_DFLT);
    const bool   COMPARE     = env_int("VMG_COMPARE", 0) != 0;

    link_t link = parse_link();

    // Build list of configs to run. If VMG_COMPARE=1, run baseline + TP variants + PP variants.
    // Otherwise, run just the one specified by VMG_TP / VMG_PP.
    std::vector<run_config> configs;
    if (COMPARE) {
        configs.push_back({1, 1, link});                // baseline
        for (int n : {2, 4, 8}) configs.push_back({n, 1, link});
        for (int n : {2, 4})    configs.push_back({1, n, link});
    } else {
        run_config cfg;
        cfg.tp_n = std::max(1, env_int("VMG_TP", 1));
        cfg.pp_n = std::max(1, env_int("VMG_PP", 1));
        cfg.link = link;
        configs.push_back(cfg);
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

    const int n_batch = params.n_batch;
    const int n_embd  = llama_model_n_embd(model);
    const int n_layers = llama_model_n_layer(model);

    LOG("\nvirtual-multigpu: N_USERS=%d mean_gap=%.0fms seed=0x%08X  n_embd=%d n_layers=%d n_batch=%d\n",
        N_USERS, MEAN_GAP_MS, (unsigned)SEED, n_embd, n_layers, n_batch);
    LOG("link: %s (latency=%.1fus, bw=%.0f GB/s)\n", link.name, link.latency_us, link.bw_bytes_per_us / 1e3);
    LOG("activation per collective @n_tokens=24: %.0f bytes (%.2f KB)\n",
        bytes_per_collective(n_embd, 24), bytes_per_collective(n_embd, 24) / 1024.0);
    LOG("\n");

    // header
    LOG("| %-12s | %-12s | %-10s | %-10s | %-10s | %-12s | %-12s | %-10s |\n",
        "config", "wall_s", "compute_s", "comm_s", "comm_pct", "pp_tok/s", "all_tok/s", "iters");
    LOG("|--------------|--------------|------------|------------|------------|--------------|--------------|------------|\n");

    for (const auto & cfg : configs) {
        // Reset KV / recurrent state between configs so each starts from a clean slate.
        llama_memory_clear(llama_get_memory(ctx), true);

        // For TP, the compute time should ideally be /cfg.tp_n (linear speedup).
        // We model that by running the actual compute and reporting it as-is, then
        // noting the projected-with-speedup time separately.
        run_result r = run_one(ctx, model, N_USERS, MEAN_GAP_MS, (uint64_t)SEED, cfg, n_batch, n_embd, n_layers);

        char cfg_str[64];
        if (cfg.tp_n == 1 && cfg.pp_n == 1) {
            snprintf(cfg_str, sizeof(cfg_str), "baseline");
        } else if (cfg.tp_n > 1) {
            snprintf(cfg_str, sizeof(cfg_str), "TP=%d", cfg.tp_n);
        } else {
            snprintf(cfg_str, sizeof(cfg_str), "PP=%d", cfg.pp_n);
        }

        const long total_tok = r.tokens_prefilled + r.tokens_decoded;
        const double pp_tok_s = r.tokens_prefilled / r.wall_s;
        const double all_tok_s = total_tok / r.wall_s;
        const double comm_pct = r.wall_s > 0 ? 100.0 * r.sim_comm_s / r.wall_s : 0.0;

        LOG("| %-12s | %12.3f | %10.3f | %10.3f | %9.1f%% | %12.1f | %12.1f | %10d |\n",
            cfg_str, r.wall_s, r.compute_s, r.sim_comm_s, comm_pct, pp_tok_s, all_tok_s, r.iterations);

        // For TP, also report the projected time with linear compute speedup
        if (cfg.tp_n > 1) {
            const double projected_compute_s = r.compute_s / cfg.tp_n;
            const double projected_wall_s = projected_compute_s + r.sim_comm_s + (r.wall_s - r.compute_s - r.sim_comm_s);
            const double proj_all_tok_s = total_tok / std::max(0.001, projected_wall_s);
            LOG("| %-12s | %12.3f | %10.3f | %10.3f | %9.1f%% | %12.1f | %12.1f | %10d |\n",
                "(+linear sp)", projected_wall_s, projected_compute_s, r.sim_comm_s,
                100.0 * r.sim_comm_s / std::max(0.001, projected_wall_s),
                r.tokens_prefilled / std::max(0.001, projected_wall_s),
                proj_all_tok_s, r.iterations);
        }
    }

    LOG("\n");
    LOG("notes:\n");
    LOG("  - 'baseline' is single-GPU, no comm overhead (the current state of the engine)\n");
    LOG("  - 'TP=N' is the *measured* wall time = real compute + simulated N-way comm cost (NO speedup modeled)\n");
    LOG("  - '+linear sp' for TP projects wall time assuming linear compute speedup (compute/N + comm)\n");
    LOG("    this is the BEST CASE for TP - real speedup is less due to the slowest layer / non-matmul ops\n");
    LOG("  - 'PP=N' adds N-1 stage-boundary transfers per iter; pipeline speedup is not modeled\n");
    LOG("    (PP helps when concurrent sequences keep the pipeline full, which the bench does exercise)\n");
    LOG("  - comm uses ring all-reduce for TP: 2*(N-1) latencies + 2*(N-1)/N * (size/bw) per collective\n");
    LOG("  - %d collectives per layer (after QKV, after attn-out, after ffn-up, after ffn-down)\n",
        TP_COLLECTIVES_PER_LAYER);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
