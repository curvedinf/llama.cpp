// CPU-only bit-exactness test for ggml_gated_delta_net_idx.
//
// The indexed op reads the initial recurrent state from the full state store
// through a row-index tensor; the gathered reference path materializes the same
// rows with ggml_get_rows and runs ggml_gated_delta_net. Both paths must produce
// bit-identical outputs (attn scores + K state snapshots), and the indexed op
// must not modify the state store.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct test_cfg {
    int64_t S_v;
    int64_t H_v;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t n_rows;
    int64_t K;
    int     v_repeat; // H_k = H_v / v_repeat
    bool    kda;
    bool    alias;    // allow duplicate read rows
};

static void fill(float * dst, int64_t n, float lo, float hi, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (int64_t i = 0; i < n; i++) {
        dst[i] = dist(rng);
    }
}

static bool run_case(ggml_backend_t backend, const test_cfg & cfg) {
    char name[128];
    snprintf(name, sizeof(name), "S=%d H=%d T=%d B=%d rows=%d K=%d vr=%d kda=%d alias=%d",
        (int) cfg.S_v, (int) cfg.H_v, (int) cfg.n_tokens, (int) cfg.n_seqs, (int) cfg.n_rows,
        (int) cfg.K, cfg.v_repeat, (int) cfg.kda, (int) cfg.alias);

    const int64_t H_k = cfg.H_v / cfg.v_repeat;

    struct ggml_init_params iparams = { 32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(iparams);

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.S_v, H_k, cfg.n_tokens, cfg.n_seqs);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.S_v, H_k, cfg.n_tokens, cfg.n_seqs);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.S_v, cfg.H_v, cfg.n_tokens, cfg.n_seqs);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.kda ? cfg.S_v : 1, cfg.H_v, cfg.n_tokens, cfg.n_seqs);
    ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, cfg.H_v, cfg.n_tokens, cfg.n_seqs);
    ggml_tensor * store = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.S_v*cfg.S_v*cfg.H_v, cfg.n_rows);
    ggml_tensor * idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_seqs);

    // gathered reference path
    ggml_tensor * gathered = ggml_get_rows(ctx, store, idx);
    ggml_tensor * state_g  = ggml_reshape_4d(ctx, gathered, cfg.S_v, cfg.S_v, cfg.H_v, cfg.n_seqs);
    ggml_tensor * out_g    = ggml_gated_delta_net(ctx, q, k, v, g, beta, state_g, cfg.K);

    // indexed path
    ggml_tensor * state_i = ggml_reshape_4d(ctx, store, cfg.S_v, cfg.S_v, cfg.H_v, cfg.n_rows);
    ggml_tensor * out_i   = ggml_gated_delta_net_idx(ctx, q, k, v, g, beta, state_i, idx, cfg.K, false);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_g);
    ggml_build_forward_expand(gf, out_i);
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "%s: alloc failed\n", name);
        return false;
    }

    std::mt19937 rng(1234);
    std::vector<float> buf(ggml_nelements(store));
    fill(buf.data(), ggml_nelements(store), -1.0f, 1.0f, rng);
    ggml_backend_tensor_set(store, buf.data(), 0, buf.size()*sizeof(float));

    const std::vector<float> store_snapshot = buf;

    std::vector<int32_t> idx_data(cfg.n_seqs);
    for (int64_t i = 0; i < cfg.n_seqs; i++) {
        // deterministic permutation-like rows spanning the whole store
        idx_data[i] = (int32_t) ((3*i + 1) % cfg.n_rows);
    }
    if (cfg.alias && cfg.n_seqs > 1) {
        idx_data[cfg.n_seqs - 1] = idx_data[0];
    }
    ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size()*sizeof(int32_t));

    buf.resize(ggml_nelements(q));
    fill(buf.data(), ggml_nelements(q), -1.0f, 1.0f, rng);
    ggml_backend_tensor_set(q, buf.data(), 0, buf.size()*sizeof(float));
    buf.resize(ggml_nelements(k));
    fill(buf.data(), ggml_nelements(k), -1.0f, 1.0f, rng);
    ggml_backend_tensor_set(k, buf.data(), 0, buf.size()*sizeof(float));
    buf.resize(ggml_nelements(v));
    fill(buf.data(), ggml_nelements(v), -0.3f, 5.0f, rng);
    ggml_backend_tensor_set(v, buf.data(), 0, buf.size()*sizeof(float));
    buf.resize(ggml_nelements(g));
    fill(buf.data(), ggml_nelements(g), -20.0f, -1e-4f, rng);
    ggml_backend_tensor_set(g, buf.data(), 0, buf.size()*sizeof(float));
    buf.resize(ggml_nelements(beta));
    fill(buf.data(), ggml_nelements(beta), 0.0f, 1.0f, rng);
    ggml_backend_tensor_set(beta, buf.data(), 0, buf.size()*sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: compute failed\n", name);
        return false;
    }

    bool ok = true;

    const int64_t n_out = ggml_nelements(out_g);
    std::vector<float> res_g(n_out), res_i(n_out);
    ggml_backend_tensor_get(out_g, res_g.data(), 0, n_out*sizeof(float));
    ggml_backend_tensor_get(out_i, res_i.data(), 0, n_out*sizeof(float));

    // compare the contracted region: attn scores + the n_written written snapshot slots;
    // slots beyond min(n_tokens, K) are caller-owned (not written by the op)
    const int64_t attn_elems = cfg.S_v*cfg.H_v*cfg.n_tokens*cfg.n_seqs;
    const int64_t snap_elems = cfg.S_v*cfg.S_v*cfg.H_v*cfg.n_seqs;
    const int64_t n_written  = cfg.n_tokens < cfg.K ? cfg.n_tokens : cfg.K;
    const int64_t n_cmp      = attn_elems + n_written*snap_elems;

    int64_t n_nan = 0;
    for (int64_t i = 0; i < n_out; i++) {
        n_nan += std::isnan(res_g[i]) || std::isnan(res_i[i]);
    }
    if (n_nan > 0) {
        fprintf(stderr, "%s: %d NaNs\n", name, (int) n_nan);
        ok = false;
    }
    if (std::memcmp(res_g.data(), res_i.data(), n_cmp*sizeof(float)) != 0) {
        int64_t first = -1;
        for (int64_t i = 0; i < n_cmp; i++) {
            if (std::memcmp(&res_g[i], &res_i[i], sizeof(float)) != 0) { first = i; break; }
        }
        fprintf(stderr, "%s: indexed vs gathered mismatch at element %d (%g != %g)\n",
                name, (int) first, (double) res_g[first], (double) res_i[first]);
        ok = false;
    }

    // the op must not write into the state store
    buf.resize(ggml_nelements(store));
    ggml_backend_tensor_get(store, buf.data(), 0, buf.size()*sizeof(float));
    if (std::memcmp(buf.data(), store_snapshot.data(), buf.size()*sizeof(float)) != 0) {
        fprintf(stderr, "%s: state store was modified\n", name);
        ok = false;
    }

    fprintf(stderr, "%s: %s\n", name, ok ? "OK" : "FAIL");

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return ok;
}

// in-place form (K=1): the op writes the new state back into the store rows s_idxs[i]
// instead of the output snapshot area. verify: attn output matches the gathered
// reference, the store rows s_idxs[i] hold the reference new state, all other rows
// are untouched.
static bool run_case_inplace(ggml_backend_t backend, const test_cfg & cfg) {
    char name[128];
    snprintf(name, sizeof(name), "S=%d H=%d B=%d rows=%d vr=%d kda=%d (in-place)",
        (int) cfg.S_v, (int) cfg.H_v, (int) cfg.n_seqs, (int) cfg.n_rows, cfg.v_repeat, (int) cfg.kda);

    const int64_t H_k = cfg.H_v / cfg.v_repeat;

    struct ggml_init_params iparams = { 32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(iparams);

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.S_v, H_k, 1, cfg.n_seqs);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.S_v, H_k, 1, cfg.n_seqs);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.S_v, cfg.H_v, 1, cfg.n_seqs);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.kda ? cfg.S_v : 1, cfg.H_v, 1, cfg.n_seqs);
    ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, cfg.H_v, 1, cfg.n_seqs);
    ggml_tensor * store = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.S_v*cfg.S_v*cfg.H_v, cfg.n_rows);
    ggml_tensor * idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_seqs);

    // gathered reference path
    ggml_tensor * gathered = ggml_get_rows(ctx, store, idx);
    ggml_tensor * state_g  = ggml_reshape_4d(ctx, gathered, cfg.S_v, cfg.S_v, cfg.H_v, cfg.n_seqs);
    ggml_tensor * out_g    = ggml_gated_delta_net(ctx, q, k, v, g, beta, state_g, 1);

    // indexed in-place path
    ggml_tensor * state_i = ggml_reshape_4d(ctx, store, cfg.S_v, cfg.S_v, cfg.H_v, cfg.n_rows);
    ggml_tensor * out_i   = ggml_gated_delta_net_idx(ctx, q, k, v, g, beta, state_i, idx, 1, true);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_g);
    ggml_build_forward_expand(gf, out_i);
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "%s: alloc failed\n", name);
        return false;
    }

    std::mt19937 rng(1234);
    std::vector<float> buf(ggml_nelements(store));
    fill(buf.data(), ggml_nelements(store), -1.0f, 1.0f, rng);
    ggml_backend_tensor_set(store, buf.data(), 0, buf.size()*sizeof(float));

    const std::vector<float> store_snapshot = buf;

    std::vector<int32_t> idx_data(cfg.n_seqs);
    for (int64_t i = 0; i < cfg.n_seqs; i++) {
        // the in-place form requires distinct rows (each workgroup writes back what it read)
        idx_data[i] = (int32_t) ((2*i + 1) % cfg.n_rows);
    }
    ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size()*sizeof(int32_t));

    buf.resize(ggml_nelements(q));
    fill(buf.data(), ggml_nelements(q), -1.0f, 1.0f, rng);
    ggml_backend_tensor_set(q, buf.data(), 0, buf.size()*sizeof(float));
    buf.resize(ggml_nelements(k));
    fill(buf.data(), ggml_nelements(k), -1.0f, 1.0f, rng);
    ggml_backend_tensor_set(k, buf.data(), 0, buf.size()*sizeof(float));
    buf.resize(ggml_nelements(v));
    fill(buf.data(), ggml_nelements(v), -0.3f, 5.0f, rng);
    ggml_backend_tensor_set(v, buf.data(), 0, buf.size()*sizeof(float));
    buf.resize(ggml_nelements(g));
    fill(buf.data(), ggml_nelements(g), -20.0f, -1e-4f, rng);
    ggml_backend_tensor_set(g, buf.data(), 0, buf.size()*sizeof(float));
    buf.resize(ggml_nelements(beta));
    fill(buf.data(), ggml_nelements(beta), 0.0f, 1.0f, rng);
    ggml_backend_tensor_set(beta, buf.data(), 0, buf.size()*sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: compute failed\n", name);
        return false;
    }

    bool ok = true;

    const int64_t n_out = ggml_nelements(out_g);
    std::vector<float> res_g(n_out), res_i(n_out);
    ggml_backend_tensor_get(out_g, res_g.data(), 0, n_out*sizeof(float));
    ggml_backend_tensor_get(out_i, res_i.data(), 0, n_out*sizeof(float));

    const int64_t attn_elems = cfg.S_v*cfg.H_v*cfg.n_seqs;
    const int64_t row_elems  = cfg.S_v*cfg.S_v*cfg.H_v;

    int64_t n_nan = 0;
    for (int64_t i = 0; i < n_out; i++) {
        n_nan += std::isnan(res_g[i]) || std::isnan(res_i[i]);
    }
    if (n_nan > 0) {
        fprintf(stderr, "%s: %d NaNs\n", name, (int) n_nan);
        ok = false;
    }

    // attn output must match the reference
    if (std::memcmp(res_g.data(), res_i.data(), attn_elems*sizeof(float)) != 0) {
        fprintf(stderr, "%s: attn output mismatch\n", name);
        ok = false;
    }

    // store rows s_idxs[i] must hold the reference new state; other rows untouched
    buf.resize(ggml_nelements(store));
    ggml_backend_tensor_get(store, buf.data(), 0, buf.size()*sizeof(float));

    std::vector<bool> touched(cfg.n_rows, false);
    for (int64_t s = 0; s < cfg.n_seqs; s++) {
        touched[idx_data[s]] = true;
    }

    for (int64_t r = 0; r < cfg.n_rows; r++) {
        const float * expected = touched[r]
            ? res_g.data() + attn_elems + ([&]() -> int64_t {
                for (int64_t s = 0; s < cfg.n_seqs; s++) {
                    if (idx_data[s] == r) {
                        return s*row_elems;
                    }
                }
                return (int64_t) 0;
              })()
            : store_snapshot.data() + r*row_elems;

        if (std::memcmp(buf.data() + r*row_elems, expected, row_elems*sizeof(float)) != 0) {
            fprintf(stderr, "%s: store row %d mismatch\n", name, (int) r);
            ok = false;
        }
    }

    fprintf(stderr, "%s: %s\n", name, ok ? "OK" : "FAIL");

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return ok;
}

int main(int, char **) {
    ggml_backend_load_all();

    bool ok = true;

    for (int dt = 0; dt <= 1; dt++) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(dt == 0 ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev == nullptr) {
            if (dt == 0) {
                fprintf(stderr, "no CPU backend found\n");
                return 1;
            }
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            fprintf(stderr, "failed to init backend\n");
            return 1;
        }

        fprintf(stderr, "== backend: %s\n", ggml_backend_dev_name(dev));

        const std::vector<test_cfg> cases = {
            // S_v, H_v, T, B, rows, K, v_repeat, kda, alias
            {  64,  4,  1, 3,  7, 1, 1, false, false}, // decode (K=1)
            {  64,  4,  8, 3,  7, 1, 1, false, false}, // chunked prefill (K=1)
            {  64,  4,  8, 3, 21, 4, 1, false, false}, // rollback snapshots, plane-spanning rows
            {  32,  2,  2, 2, 10, 4, 1, false, false}, // n_tokens < K: partial snapshot writes
            {  64,  2,  4, 2,  5, 2, 1, true,  false}, // KDA gate
            {  32,  4,  4, 2,  5, 1, 2, false, false}, // H_k != H_v broadcast
            {  64,  4,  1, 4,  9, 1, 1, false, true }, // aliased read rows
            { 128, 16,  1, 8, 18, 1, 1, false, false}, // Qwen3.5-like decode
            { 128, 16, 16, 4, 20, 4, 1, false, false}, // Qwen3.5-like prefill + snapshots
        };

        for (const auto & cfg : cases) {
            ok &= run_case(backend, cfg);
        }

        const std::vector<test_cfg> ip_cases = {
            // S_v, H_v, T, B, rows, K, v_repeat, kda, alias
            {  64,  4, 1, 3,  7, 1, 1, false, false}, // small decode
            { 128, 16, 1, 8, 18, 1, 1, false, false}, // Qwen3.5-like decode
            {  64,  2, 1, 2,  5, 1, 1, true,  false}, // KDA gate
        };
        for (const auto & cfg : ip_cases) {
            ok &= run_case_inplace(backend, cfg);
        }

        ggml_backend_free(backend);
    }

    fprintf(stderr, "%s\n", ok ? "all cases passed" : "FAILURES");
    return ok ? 0 : 1;
}
