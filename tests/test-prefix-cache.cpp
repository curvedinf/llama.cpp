// Unit tests for the block-hash prefix cache:
//   - chained block hashing (prefix-consistency)
//   - registry insert/find/match/release/evict
//   - per-sequence token chain tracker
//   - KV cell block pinning and cache-owned retention

#include "llama-kv-cells.h"
#include "llama-prefix-cache.h"

#include <cstdio>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            n_fail++; \
        } \
    } while (0)

static constexpr uint32_t BS = llama_prefix_cache::block_size;

static std::vector<llama_token> make_tokens(int32_t n, int32_t base) {
    std::vector<llama_token> res(n);
    for (int32_t i = 0; i < n; ++i) {
        res[i] = base + i%100 + 1;
    }
    return res;
}

static uint64_t hash_n_blocks(const std::vector<llama_token> & tokens, uint32_t n_blocks, std::vector<uint64_t> * out = nullptr) {
    uint64_t h = llama_prefix_cache::seed;
    for (uint32_t k = 0; k < n_blocks; ++k) {
        h = llama_prefix_cache::hash_block(h, tokens.data() + k*BS);
        if (out) {
            out->push_back(h);
        }
    }
    return h;
}

// chained hashing is deterministic and prefix-consistent
static void test_hash() {
    fprintf(stderr, "%s\n", __func__);

    const auto ta = make_tokens(3*BS, 0);
    const auto tb = make_tokens(3*BS, 0);

    CHECK(ta == tb);

    std::vector<uint64_t> ha, hb;
    hash_n_blocks(ta, 3, &ha);
    hash_n_blocks(tb, 3, &hb);

    CHECK(ha == hb);

    // each block hash differs
    CHECK(ha[0] != ha[1] && ha[1] != ha[2] && ha[0] != ha[2]);

    // changing one token in block 1 changes block 1 and all subsequent blocks
    auto tc = ta;
    tc[BS + 5] += 1;

    std::vector<uint64_t> hc;
    hash_n_blocks(tc, 3, &hc);

    CHECK(hc[0] == ha[0]);
    CHECK(hc[1] != ha[1]);
    CHECK(hc[2] != ha[2]);
}

// registry insert/find/match/release
static void test_registry_match() {
    fprintf(stderr, "%s\n", __func__);

    llama_prefix_cache reg;

    const auto prompt_a = make_tokens(4*BS, 0);
    const auto prompt_b = make_tokens(4*BS, 0);

    std::vector<uint64_t> h;
    hash_n_blocks(prompt_a, 4, &h);

    // register blocks 0..2 of the prompt in stream 0
    for (uint32_t k = 0; k < 3; ++k) {
        const llama_prefix_cache::loc_t loc { 0, k, (llama_pos) (k*BS) };
        const auto res = reg.insert(h[k], prompt_a.data() + k*BS, loc);
        CHECK(res.first != nullptr);
        CHECK(res.second == true);
    }

    CHECK(reg.size() == 3);

    // exact find by hash + tokens
    CHECK(reg.find(h[1], prompt_a.data() + BS) != nullptr);
    CHECK(reg.find(h[1], prompt_b.data() + BS) != nullptr); // same tokens

    auto wrong = prompt_a;
    wrong[BS] += 1;
    CHECK(reg.find(h[1], wrong.data() + BS) == nullptr); // token mismatch

    // match the full 4-block prompt - only 3 blocks are cached
    uint64_t handle = 0;
    const uint32_t n = reg.match(prompt_b.data(), 4*BS, false, handle);

    CHECK(n == 3*BS);
    CHECK(handle != 0);
    CHECK(reg.handle_hashes(handle) != nullptr);
    CHECK(reg.handle_hashes(handle)->size() == 3);
    CHECK(reg.handle_state(handle) == nullptr); // no state attached

    // the entries are pinned while the handle is alive
    CHECK(reg.find(h[0])->n_pin == 1);

    reg.release(handle);
    CHECK(reg.find(h[0])->n_pin == 0);
    CHECK(reg.handle_hashes(handle) == nullptr);

    // need_state: no state attached -> no match
    handle = 0;
    CHECK(reg.match(prompt_b.data(), 4*BS, true, handle) == 0);
    CHECK(handle == 0);

    // attach state to block 1: need_state matches up to block 1 (longest with state)
    reg.set_state(h[1], prompt_a.data() + BS, { 1, 2, 3 });
    CHECK(reg.find(h[1])->has_state);

    handle = 0;
    CHECK(reg.match(prompt_b.data(), 4*BS, true, handle) == 2*BS);
    CHECK(handle != 0);
    CHECK(reg.handle_state(handle) != nullptr);
    CHECK(reg.handle_state(handle)->size() == 3);

    reg.release(handle);

    // attach state to block 2 as well: matches all 3 cached blocks
    reg.set_state(h[2], prompt_a.data() + 2*BS, { 4, 5 });
    handle = 0;
    CHECK(reg.match(prompt_b.data(), 4*BS, true, handle) == 3*BS);
    reg.release(handle);

    // divergent prompt matches only the common blocks
    auto prompt_c = prompt_b;
    prompt_c[2*BS] += 1;

    handle = 0;
    CHECK(reg.match(prompt_c.data(), 4*BS, false, handle) == 2*BS);
    reg.release(handle);

    // short prompts never match
    handle = 0;
    CHECK(reg.match(prompt_b.data(), BS - 1, false, handle) == 0);
}

// multiple locations per entry, removal by location and LRU eviction
static void test_registry_evict() {
    fprintf(stderr, "%s\n", __func__);

    llama_prefix_cache reg;

    const auto prompt = make_tokens(2*BS, 0);

    std::vector<uint64_t> h;
    hash_n_blocks(prompt, 2, &h);

    // block 0 cached in streams 0 and 1, block 1 in stream 1 only
    CHECK(reg.insert(h[0], prompt.data(),      { 0, 0, 0 })          .second);
    CHECK(reg.insert(h[0], prompt.data(),      { 1, 5, 0 })          .second);
    CHECK(reg.insert(h[1], prompt.data() + BS, { 1, 6, (llama_pos) BS }).second);

    CHECK(reg.size() == 2);
    CHECK(reg.find(h[0])->locs.size() == 2);

    // duplicate insert of the same location does not duplicate
    CHECK(reg.insert(h[0], prompt.data(),      { 0, 0, 0 }).second == false);
    CHECK(reg.find(h[0])->locs.size() == 2);

    // eviction candidates: oldest first, only the requested stream
    std::vector<llama_prefix_cache::loc_t> cands;
    reg.evict_candidates(0, 16, cands);
    CHECK(cands.size() == 1 && cands[0].stream == 0 && cands[0].block == 0);

    cands.clear();
    reg.evict_candidates(1, 16, cands);
    CHECK(cands.size() == 2);

    // pinned entries are skipped
    uint64_t handle = 0;
    CHECK(reg.match(prompt.data(), 2*BS, false, handle) == 2*BS);
    cands.clear();
    reg.evict_candidates(1, 16, cands);
    CHECK(cands.empty());
    reg.release(handle);

    // remove one location of a multi-location entry - the entry survives
    CHECK(reg.remove_loc({ 0, 0, 0 }));
    CHECK(reg.find(h[0]) != nullptr);
    CHECK(reg.find(h[0])->locs.size() == 1);

    // remove the last location - the entry dies
    CHECK(reg.remove_loc({ 1, 5, 0 }));
    CHECK(reg.find(h[0]) == nullptr);
    CHECK(reg.size() == 1);

    // remove_range by position
    std::vector<llama_prefix_cache::loc_t> out;
    reg.remove_range(1, 0, BS, out); // block 1 has p0 = BS - outside
    CHECK(out.empty());
    reg.remove_range(1, BS, 2*BS, out);
    CHECK(out.size() == 1);
    CHECK(reg.empty());

    // remove_stream
    reg.insert(h[0], prompt.data(), { 0, 0, 0 });
    reg.insert(h[0], prompt.data(), { 1, 1, 0 });
    out.clear();
    reg.remove_stream(1, out);
    CHECK(out.size() == 1);
    CHECK(reg.find(h[0])->locs.size() == 1);
    out.clear();
    reg.remove_stream(0, out);
    CHECK(out.size() == 1);
    CHECK(reg.empty());
}

// per-sequence token chain tracker
static void test_chain() {
    fprintf(stderr, "%s\n", __func__);

    llama_prefix_chain chain;

    const auto prompt = make_tokens(3*BS, 0);

    // feed 2.5 blocks - 2 completions
    int n_complete = 0;
    for (uint32_t i = 0; i < 2*BS + BS/2; ++i) {
        n_complete += chain.feed((llama_pos) i, prompt[i]) ? 1 : 0;
    }

    CHECK(n_complete == 2);
    CHECK(chain.hashes.size() == 2);
    CHECK(chain.next_pos == (llama_pos) (2*BS + BS/2));
    CHECK(chain.last_p0 == (llama_pos) BS);

    std::vector<uint64_t> href;
    hash_n_blocks(prompt, 2, &href);
    CHECK(chain.hashes == href);

    // out-of-order feed kills the chain
    CHECK(chain.feed(999, 1) == false);
    CHECK(chain.next_pos == -1);
    CHECK(chain.feed((llama_pos) (2*BS + BS/2), prompt[2*BS + BS/2]) == false);

    // feeding position 0 restarts
    chain.kill();
    CHECK(chain.feed(0, prompt[0]) == false); // block not complete yet
    CHECK(chain.next_pos == 1);

    // truncate at a block boundary keeps the chain alive
    chain.reset();
    for (uint32_t i = 0; i < 3*BS; ++i) {
        chain.feed((llama_pos) i, prompt[i]);
    }
    CHECK(chain.hashes.size() == 3);

    chain.truncate(2*BS);
    CHECK(chain.next_pos == (llama_pos) 2*BS);
    CHECK(chain.hashes.size() == 2);

    // refill completes the truncated block with the same hash
    bool done = false;
    for (uint32_t i = 2*BS; i < 3*BS; ++i) {
        done = chain.feed((llama_pos) i, prompt[i]);
    }
    CHECK(done);
    CHECK(chain.hashes.size() == 3);
    CHECK(chain.hashes[2] == llama_prefix_cache::hash_block(href[1], prompt.data() + 2*BS));

    // mid-block truncate kills the chain
    chain.truncate(2*BS + 5);
    CHECK(chain.next_pos == -1);
}

// block pinning and cache-owned retention in llama_kv_cells
static void test_cells_pin_retention() {
    fprintf(stderr, "%s\n", __func__);

    llama_kv_cells cells;
    cells.resize(128); // 4 blocks

    // fill 2 blocks with seq 0
    std::vector<uint32_t> idxs;
    CHECK(cells.alloc_find(2*BS, idxs) == 2*BS);
    std::sort(idxs.begin(), idxs.end());
    for (uint32_t i = 0; i < 2*BS; ++i) {
        cells.pos_set(idxs[i], (llama_pos) i);
        cells.seq_add(idxs[i], 0);
    }

    CHECK(cells.block_is_full(0));
    CHECK(cells.block_is_full(1));
    CHECK(cells.block_ref(0) == 0);

    // pin block 0
    cells.block_pin(0);
    CHECK(cells.block_ref(0) == 1);

    // removing the sequence frees the unpinned block 1, but block 0 is retained cache-owned
    cells.seq_rm_range(0, 0, std::numeric_limits<llama_pos>::max());

    CHECK(cells.seq_pos_max(0) == -1);
    CHECK(cells.get_used() == BS);
    CHECK(cells.block_is_full(0));
    CHECK(cells.block_all_seqless(0));
    CHECK(cells.block_is_free(1));

    // the retained cells keep their positions
    for (uint32_t j = 0; j < BS; ++j) {
        CHECK(cells.pos_get(j) == (llama_pos) j);
        CHECK(cells.seq_count(j) == 0);
    }

    // the retained block is not handed out by the allocator
    idxs.clear();
    CHECK(cells.alloc_find(3*BS, idxs) == 3*BS);
    for (const auto i : idxs) {
        CHECK(i >= BS);
    }

    // unpin frees the cache-owned cells
    CHECK(cells.block_unpin(0));
    CHECK(cells.get_used() == 0);
    CHECK(cells.block_is_free(0));

    idxs.clear();
    CHECK(cells.alloc_find(4*BS, idxs) == 4*BS);

    // two pins: the block survives the first unpin
    cells.pos_set(idxs[0], 0);
    cells.seq_add(idxs[0], 0);
    const uint32_t b0 = idxs[0]/BS;
    cells.block_pin(b0);
    cells.block_pin(b0);
    CHECK(!cells.block_unpin(b0));
    CHECK(cells.block_ref(b0) == 1);
    cells.seq_rm(idxs[0], 0); // retained again (block not full - no free)
    CHECK(cells.block_ref(b0) == 1);
    CHECK(!cells.block_unpin(b0)); // not all cells used/seqless -> no free
    CHECK(cells.block_ref(b0) == 0);
}

// paged-mode prefix copy (Phase 2): re-attaching a matched prefix to another sequence is
// pure bookkeeping - the blocks are shared via the registry pin, no cells are allocated
// or copied, and both sequences' block tables map the shared ranges to the same blocks
static void test_paged_reattach_bookkeeping() {
    fprintf(stderr, "%s\n", __func__);

    llama_prefix_cache reg;
    llama_kv_cells     cells;

    cells.resize(8*BS);

    const auto prompt = make_tokens(2*BS, 0);

    std::vector<uint64_t> hashes;
    hash_n_blocks(prompt, 2, &hashes);

    // seq 0 fills the first two blocks with aligned cells
    {
        std::vector<llama_pos>    poss(2*BS);
        std::vector<llama_seq_id> seqs(2*BS, 0);

        for (uint32_t i = 0; i < 2*BS; ++i) {
            poss[i] = (llama_pos) i;
        }

        std::vector<uint32_t> idxs;

        CHECK(cells.alloc_find_seq_aligned(poss.data(), seqs.data(), 2*BS, idxs) == 2*BS);

        for (uint32_t i = 0; i < 2*BS; ++i) {
            cells.pos_set(idxs[i], poss[i]);
            cells.seq_add(idxs[i], 0);
        }

        CHECK(idxs[0] == 0 && idxs.back() == 2*BS - 1);
    }

    // register both blocks as prefix locations (pinned)
    for (uint32_t b = 0; b < 2; ++b) {
        const auto res = reg.insert(hashes[b], prompt.data() + b*BS, { 0, b, (llama_pos) (b*BS) });
        CHECK(res.second);

        cells.block_pin(b);
    }

    const uint32_t used_before = cells.get_used();

    // seq 1 matches the prefix (registry level)
    uint64_t handle = 0;

    const uint32_t n_match = reg.match(prompt.data(), 2*BS, false, handle);

    CHECK(n_match == 2*BS);
    CHECK(handle != 0);

    // re-attach at the cells level (llama_kv_cache::prefix_copy_impl loc_same in paged
    // mode): bookkeeping only - no allocation, no copies
    for (uint32_t k = 0; k < 2; ++k) {
        const auto * e = reg.find(hashes[k]);

        CHECK(e != nullptr && !e->locs.empty());

        const auto & loc = e->locs.front();

        CHECK(loc.stream == 0 && loc.block == k);

        for (uint32_t j = 0; j < BS; ++j) {
            cells.seq_add(loc.block*BS + j, 1);
        }
    }

    // no cells were allocated or copied
    CHECK(cells.get_used() == used_before);

    // both sequences' block tables map the shared ranges to the same physical blocks
    {
        std::vector<int32_t> t0(8, -2), t1(8, -2);

        CHECK(cells.seq_block_table(0, t0));
        CHECK(cells.seq_block_table(1, t1));

        CHECK(t0[0] == 0 && t0[1] == 1);
        CHECK(t1[0] == 0 && t1[1] == 1);

        // the cells are shared read-only
        for (uint32_t j = 0; j < 2*BS; ++j) {
            CHECK(cells.seq_count(j) == 2);
        }
    }

    // unregistering + unpinning keeps the cells alive while seq 1 still uses them
    reg.release(handle);

    reg.remove_loc({ 0, 0, 0 });
    reg.remove_loc({ 0, 1, (llama_pos) BS });

    CHECK(!cells.block_unpin(0)); // ref reaches 0, but the cells are not seqless
    CHECK(!cells.block_unpin(1));

    for (uint32_t j = 0; j < 2*BS; ++j) {
        CHECK(!cells.is_empty(j));
        CHECK(cells.seq_has(j, 1));
    }

    // removing both sequences then frees everything (blocks are unpinned now)
    cells.seq_rm_range(1, 0, std::numeric_limits<llama_pos>::max());
    cells.seq_rm_range(0, 0, std::numeric_limits<llama_pos>::max());

    CHECK(cells.get_used() == 0);
}

int main() {
    test_hash();
    test_registry_match();
    test_registry_evict();
    test_chain();
    test_cells_pin_retention();
    test_paged_reattach_bookkeeping();

    if (n_fail > 0) {
        fprintf(stderr, "FAILED: %d checks failed\n", n_fail);
        return 1;
    }

    printf("%s: OK\n", __func__);
    return 0;
}
