// Unit tests for the block-granular KV cell allocator (llama_kv_cells)

#include "llama-kv-cells.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            n_fail++; \
        } \
    } while (0)

// fill n cells drawn from the block allocator, assigning positions p0, p0 + 1, ... to seq_id
// mimics llama_kv_cache::find_slot() + apply_ubatch()
static void fill_seq(llama_kv_cells & cells, uint32_t n, llama_seq_id seq_id, llama_pos p0) {
    std::vector<uint32_t> idxs;

    CHECK(cells.alloc_find(n, idxs) == n);

    std::sort(idxs.begin(), idxs.end());

    for (uint32_t i = 0; i < n; ++i) {
        cells.pos_set(idxs[i], p0 + (llama_pos) i);
        cells.seq_add(idxs[i], seq_id);
    }
}

// block alloc/free, high-water mark trimming, rollback
static void test_alloc_rollback() {
    fprintf(stderr, "%s\n", __func__);

    llama_kv_cells cells;
    cells.resize(128); // 4 blocks

    CHECK(cells.size() == 128);
    CHECK(cells.get_used() == 0);
    CHECK(cells.used_max_p1() == 0);

    fill_seq(cells, 100, 0, 0);

    CHECK(cells.get_used() == 100);
    CHECK(cells.used_max_p1() == 100);
    CHECK(cells.seq_pos_min(0) == 0);
    CHECK(cells.seq_pos_max(0) == 99);

    // rollback the tail: keep positions [0, 50)
    const uint32_t min_freed = cells.seq_rm_range(0, 50, std::numeric_limits<llama_pos>::max());

    CHECK(min_freed == 50);
    CHECK(cells.get_used() == 50);
    CHECK(cells.used_max_p1() == 50);
    CHECK(cells.seq_pos_min(0) == 0);
    CHECK(cells.seq_pos_max(0) == 49);

    // the freed blocks must be reusable: 14 free cells in the boundary block, then 32 in the
    // next one and 4 more in the last block
    std::vector<uint32_t> idxs;

    CHECK(cells.alloc_find(46, idxs) == 46);
    for (uint32_t i = 0; i < 46; ++i) {
        CHECK(idxs[i] == 50 + i);
    }

    fill_seq(cells, 46, 0, 100);

    CHECK(cells.get_used() == 96);
    CHECK(cells.seq_pos_max(0) == 145);

    // only 4 cells in the last block remain available
    idxs.clear();

    CHECK(cells.alloc_find(4, idxs) == 4);
    CHECK(idxs[0] == 96 && idxs[3] == 99);
}

// full-cache exhaustion behavior
static void test_exhaustion() {
    fprintf(stderr, "%s\n", __func__);

    llama_kv_cells cells;
    cells.resize(64); // 2 blocks

    fill_seq(cells, 64, 0, 0);

    CHECK(cells.get_used() == 64);
    CHECK(cells.used_max_p1() == 64);

    // no empty cells left
    std::vector<uint32_t> idxs;

    CHECK(cells.alloc_find(1, idxs) == 0);
    CHECK(cells.alloc_find(64, idxs) == 0);

    // free exactly one cell in the middle - the block becomes partial and reusable
    cells.rm(33);

    CHECK(cells.get_used() == 63);
    CHECK(cells.used_max_p1() == 64);
    CHECK(cells.alloc_find(2, idxs) == 1);
    CHECK(idxs[0] == 33);

    // full clear via rm_range
    CHECK(cells.rm_range(0, std::numeric_limits<llama_pos>::max()) == 0);
    CHECK(cells.get_used() == 0);
    CHECK(cells.used_max_p1() == 0);
    CHECK(cells.seq_pos_max(0) == -1);
    CHECK(cells.alloc_find(64, idxs) == 64);
}

// multiple sequences sharing the same cells
static void test_multi_seq() {
    fprintf(stderr, "%s\n", __func__);

    llama_kv_cells cells;
    cells.resize(64);

    fill_seq(cells, 32, 0, 0);

    // share cells [0, 16) with seq 1
    for (uint32_t i = 0; i < 16; ++i) {
        cells.seq_add(i, 1);
    }

    CHECK(cells.seq_count(0) == 2);
    CHECK(cells.seq_has(0, 0));
    CHECK(cells.seq_has(0, 1));
    CHECK(!cells.seq_has(0, 2));
    CHECK(cells.seq_count(20) == 1);
    CHECK(cells.seq_get(20) == 0);

    CHECK(cells.seq_pos_min(1) == 0);
    CHECK(cells.seq_pos_max(1) == 15);
    CHECK(cells.seq_pos_max(0) == 31);

    // removing one sequence keeps the cells
    CHECK(!cells.seq_rm(10, 1));
    CHECK(cells.seq_count(10) == 1);
    CHECK(cells.seq_get(10) == 0);
    CHECK(!cells.is_empty(10));
    CHECK(cells.seq_pos_max(1) == 15);

    // remove seq 1 everywhere - the used cells of seq 0 must remain
    CHECK(cells.seq_rm_range(1, 0, std::numeric_limits<llama_pos>::max()) == 0);
    CHECK(cells.get_used() == 32);
    CHECK(cells.seq_pos_max(1) == -1);
    CHECK(cells.seq_pos_min(1) == -1);
    CHECK(cells.seq_pos_max(0) == 31);

    // seq_keep drops other sequences but keeps the cells
    for (uint32_t i = 0; i < 8; ++i) {
        cells.seq_add(i, 2);
    }
    CHECK(!cells.seq_keep(3, 0));
    CHECK(cells.seq_count(3) == 1);
    CHECK(cells.seq_get(3) == 0);

    // seq_keep on a cell that does not have the sequence frees it
    CHECK(cells.seq_keep(40, 0) == false);
    CHECK(cells.seq_keep_all(2) == 3);
    CHECK(cells.get_used() == 7);
    CHECK(cells.seq_pos_max(0) == -1);
    CHECK(cells.seq_pos_max(2) == 7);
}

// interior removal leaves a reusable hole
static void test_interior_hole() {
    fprintf(stderr, "%s\n", __func__);

    llama_kv_cells cells;
    cells.resize(128);

    fill_seq(cells, 128, 0, 0);

    // remove the second block entirely
    CHECK(cells.seq_rm_range(0, 32, 64) == 32);

    CHECK(cells.get_used() == 96);
    CHECK(cells.seq_pos_min(0) == 0);
    CHECK(cells.seq_pos_max(0) == 127);
    CHECK(cells.used_max_p1() == 128);

    // the hole is reusable
    std::vector<uint32_t> idxs;

    CHECK(cells.alloc_find(32, idxs) == 32);
    for (uint32_t i = 0; i < 32; ++i) {
        CHECK(idxs[i] == 32 + i);
    }

    // fill the hole with fresh (higher) positions
    fill_seq(cells, 32, 0, 200);

    CHECK(cells.get_used() == 128);
    CHECK(cells.seq_pos_min(0) == 0);
    CHECK(cells.seq_pos_max(0) == 231);

    // remove everything except the hole positions
    CHECK(cells.seq_rm_range(0, 0, 200) == 0);

    CHECK(cells.get_used() == 32);
    CHECK(cells.seq_pos_min(0) == 200);
    CHECK(cells.seq_pos_max(0) == 231);
    CHECK(cells.used_max_p1() == 64);
}

// position shifts and divisions
static void test_shift_div() {
    fprintf(stderr, "%s\n", __func__);

    llama_kv_cells cells;
    cells.resize(64);

    fill_seq(cells, 64, 0, 100);

    CHECK(!cells.get_has_shift());

    // shift everything down by 50
    for (uint32_t b = cells.block_lo(); b < cells.block_hi(); ++b) {
        for (uint32_t i = cells.cell_begin(b); i < cells.cell_end(b); ++i) {
            if (!cells.is_empty(i)) {
                CHECK(!cells.pos_add(i, -50));
            }
        }
    }

    CHECK(cells.get_has_shift());
    CHECK(cells.get_used() == 64);
    CHECK(cells.seq_pos_min(0) == 50);
    CHECK(cells.seq_pos_max(0) == 113);
    CHECK(cells.get_shift(0) == -50);

    // shift below zero - the cells with positions that become negative are freed
    uint32_t n_freed = 0;

    for (uint32_t b = cells.block_lo(); b < cells.block_hi(); ++b) {
        for (uint32_t i = cells.cell_begin(b); i < cells.cell_end(b); ++i) {
            if (!cells.is_empty(i) && cells.pos_add(i, -60)) {
                n_freed++;
            }
        }
    }

    CHECK(n_freed == 10);
    CHECK(cells.get_used() == 54);
    CHECK(cells.seq_pos_min(0) == 0);
    CHECK(cells.seq_pos_max(0) == 53);

    // shift the remaining cells below zero
    for (uint32_t b = cells.block_lo(); b < cells.block_hi(); ++b) {
        for (uint32_t i = cells.cell_begin(b); i < cells.cell_end(b); ++i) {
            if (!cells.is_empty(i)) {
                CHECK(cells.pos_add(i, -120));
            }
        }
    }

    CHECK(cells.get_used() == 0);
    CHECK(cells.used_max_p1() == 0);
    CHECK(cells.seq_pos_max(0) == -1);

    std::vector<uint32_t> idxs;
    CHECK(cells.alloc_find(64, idxs) == 64);

    cells.reset_shift();
    CHECK(!cells.get_has_shift());

    // divide positions
    fill_seq(cells, 64, 0, 0);

    for (uint32_t b = cells.block_lo(); b < cells.block_hi(); ++b) {
        for (uint32_t i = cells.cell_begin(b); i < cells.cell_end(b); ++i) {
            if (!cells.is_empty(i)) {
                cells.pos_div(i, 2);
            }
        }
    }

    CHECK(cells.get_used() == 64);
    CHECK(cells.seq_pos_min(0) == 0);
    CHECK(cells.seq_pos_max(0) == 31);
    CHECK(cells.pos_get(40) == 20);
}

// duplicate positions for the same sequence (e.g. vision inputs)
static void test_duplicate_pos() {
    fprintf(stderr, "%s\n", __func__);

    llama_kv_cells cells;
    cells.resize(64);

    std::vector<uint32_t> idxs;

    CHECK(cells.alloc_find(4, idxs) == 4);

    for (uint32_t i = 0; i < 4; ++i) {
        cells.pos_set(idxs[i], 7);
        cells.seq_add(idxs[i], 0);
    }

    CHECK(cells.seq_pos_min(0) == 7);
    CHECK(cells.seq_pos_max(0) == 7);

    // removing some of the duplicates keeps min/max
    cells.rm(idxs[0]);
    cells.rm(idxs[1]);

    CHECK(cells.seq_pos_min(0) == 7);
    CHECK(cells.seq_pos_max(0) == 7);

    cells.rm(idxs[2]);
    cells.rm(idxs[3]);

    CHECK(cells.seq_pos_min(0) == -1);
    CHECK(cells.seq_pos_max(0) == -1);
    CHECK(cells.get_used() == 0);
}

// save/restore of cell ranges with cp()/set()
static void test_cp_set() {
    fprintf(stderr, "%s\n", __func__);

    llama_kv_cells cells;
    cells.resize(64);

    fill_seq(cells, 64, 0, 0);

    // share some cells with seq 1
    for (uint32_t i = 4; i < 8; ++i) {
        cells.seq_add(i, 1);
    }

    // snapshot the full cache and a scattered subset
    const auto full = cells.cp(0, 64);

    std::vector<uint32_t> subset = { 3, 17, 40, 63 };
    const auto part = cells.cp(subset);

    // mutate: rollback + fresh fill
    CHECK(cells.seq_rm_range(0, 32, std::numeric_limits<llama_pos>::max()) == 32);
    fill_seq(cells, 32, 0, 200);

    CHECK(cells.get_used() == 64);
    CHECK(cells.seq_pos_max(0) == 231);

    // restore the snapshot
    cells.set(0, full);

    CHECK(cells.get_used() == 64);
    CHECK(cells.seq_pos_min(0) == 0);
    CHECK(cells.seq_pos_max(0) == 63);
    CHECK(cells.seq_pos_max(1) == 7);

    for (uint32_t i = 0; i < 64; ++i) {
        CHECK(!cells.is_empty(i));
        CHECK(cells.pos_get(i) == (llama_pos) i);
        CHECK(cells.seq_has(i, 0));
        CHECK(cells.seq_count(i) == (i >= 4 && i < 8 ? 2 : 1));
    }

    // restore the scattered subset on top of a modified range
    CHECK(cells.seq_rm_range(0, 0, 10) == 0);
    cells.set(subset, part);

    for (const auto i : subset) {
        CHECK(!cells.is_empty(i));
        CHECK(cells.pos_get(i) == (llama_pos) i);
        CHECK(cells.seq_has(i, 0));
    }

    CHECK(cells.is_empty(0));
    CHECK(cells.is_empty(9));
}

// differential test against a naive reference implementation
static void test_randomized() {
    fprintf(stderr, "%s\n", __func__);

    const uint32_t size = 256; // 8 blocks

    llama_kv_cells cells;
    cells.resize(size);

    // reference state
    std::set<llama_pos> ref_pos[2]; // positions per sequence
    uint32_t ref_used = 0;

    srand(12345);

    for (int step = 0; step < 2000; ++step) {
        const int op = rand()%100;

        if (op < 45) {
            // allocate + fill
            const uint32_t n = 1 + rand()%40;

            std::vector<uint32_t> idxs;

            const uint32_t got = cells.alloc_find(n, idxs);

            CHECK(got == std::min(n, size - ref_used));

            const llama_seq_id seq_id = rand()%2;

            llama_pos p0 = 0;
            if (!ref_pos[seq_id].empty()) {
                p0 = *ref_pos[seq_id].rbegin() + 1;
            }

            for (uint32_t i = 0; i < got; ++i) {
                cells.pos_set(idxs[i], p0 + (llama_pos) i);
                cells.seq_add(idxs[i], seq_id);

                ref_pos[seq_id].insert(p0 + (llama_pos) i);
            }

            ref_used += got;
        } else if (op < 75) {
            // tail rollback (spec-decode rejection)
            const llama_seq_id seq_id = rand()%2;

            if (ref_pos[seq_id].empty()) {
                continue;
            }

            const llama_pos p_min = *ref_pos[seq_id].begin();
            const llama_pos p_max = *ref_pos[seq_id].rbegin();

            const llama_pos p0 = p_min + rand()%(p_max - p_min + 1);

            cells.seq_rm_range(seq_id, p0, std::numeric_limits<llama_pos>::max());

            uint32_t n_rm = 0;
            for (auto it = ref_pos[seq_id].lower_bound(p0); it != ref_pos[seq_id].end();) {
                it = ref_pos[seq_id].erase(it);
                n_rm++;
            }

            ref_used -= n_rm;
        } else if (op < 90) {
            // interior removal
            const llama_seq_id seq_id = rand()%2;

            if (ref_pos[seq_id].empty()) {
                continue;
            }

            const llama_pos p_min = *ref_pos[seq_id].begin();
            const llama_pos p_max = *ref_pos[seq_id].rbegin();

            const llama_pos p0 = p_min + rand()%(p_max - p_min + 1);
            const llama_pos p1 = p0   + rand()%(p_max - p0   + 1);

            cells.seq_rm_range(seq_id, p0, p1);

            uint32_t n_rm = 0;
            for (auto it = ref_pos[seq_id].lower_bound(p0); it != ref_pos[seq_id].end() && *it < p1;) {
                it = ref_pos[seq_id].erase(it);
                n_rm++;
            }

            ref_used -= n_rm;
        } else {
            // full removal of one sequence
            const llama_seq_id seq_id = rand()%2;

            cells.seq_rm_range(seq_id, 0, std::numeric_limits<llama_pos>::max());

            ref_used -= ref_pos[seq_id].size();
            ref_pos[seq_id].clear();
        }

        CHECK(cells.get_used() == ref_used);

        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            const llama_pos ref_min = ref_pos[seq_id].empty() ? -1 : *ref_pos[seq_id].begin();
            const llama_pos ref_max = ref_pos[seq_id].empty() ? -1 : *ref_pos[seq_id].rbegin();

            CHECK(cells.seq_pos_min(seq_id) == ref_min);
            CHECK(cells.seq_pos_max(seq_id) == ref_max);
        }

        // the high-water mark must cover exactly the used cells
        uint32_t n_below_hwm = 0;
        for (uint32_t i = 0; i < cells.used_max_p1(); ++i) {
            if (!cells.is_empty(i)) {
                n_below_hwm++;
            }
        }

        CHECK(n_below_hwm == ref_used);
        CHECK(cells.used_max_p1() <= size);
    }
}

// seq-aligned (paged) allocation: a token with logical position p lands at cell
// b*block_size + p%block_size of a block covering the same aligned range and owned by
// the token's sequence
static void test_seq_aligned_alloc() {
    fprintf(stderr, "%s\n", __func__);

    constexpr uint32_t bs = llama_kv_cells::block_size;

    llama_kv_cells cells;
    cells.resize(9*bs);

    auto fill_aligned = [&](llama_seq_id seq_id, llama_pos p0, uint32_t n) {
        std::vector<llama_pos>    poss(n);
        std::vector<llama_seq_id> seqs(n, seq_id);

        for (uint32_t i = 0; i < n; ++i) {
            poss[i] = p0 + (llama_pos) i;
        }

        std::vector<uint32_t> idxs;

        CHECK(cells.alloc_find_seq_aligned(poss.data(), seqs.data(), n, idxs) == n);

        for (uint32_t i = 0; i < n; ++i) {
            CHECK(idxs[i]%bs == (uint32_t) poss[i]%bs);

            cells.pos_set(idxs[i], poss[i]);
            cells.seq_add(idxs[i], seq_id);
        }

        return idxs;
    };

    // seq 0 takes positions [0, 16): block 0, cells [0, 16)
    const auto idxs0 = fill_aligned(0, 0, 16);
    for (uint32_t i = 0; i < 16; ++i) {
        CHECK(idxs0[i] == i);
    }

    // seq 1 takes positions [16, 32): the tail of block 0 is free, but the block belongs
    // to seq 0 - seq 1 gets its own block (at the matching offsets)
    const auto idxs1 = fill_aligned(1, 16, 16);
    for (uint32_t i = 0; i < 16; ++i) {
        CHECK(idxs1[i] == bs + 16 + i);
    }

    // seq 0 continues with positions [16, 32): extends its own block 0
    const auto idxs2 = fill_aligned(0, 16, 16);
    for (uint32_t i = 0; i < 16; ++i) {
        CHECK(idxs2[i] == 16 + i);
    }

    // seq 1 continues with positions [32, 48): a new aligned range - a new block, never
    // the tail of a block covering a different range
    const auto idxs3 = fill_aligned(1, 32, 16);
    for (uint32_t i = 0; i < 16; ++i) {
        CHECK(idxs3[i] == 2*bs + i);
    }

    // share seq 1's range-1 block with seq 2 (mimics a prefix re-attach), then seq 2
    // continues the same range - it extends the shared block since every cell of the
    // block has seq 2 as a member
    for (uint32_t j = 2*bs; j < 3*bs; ++j) {
        cells.seq_add(j, 2);
    }

    const auto idxs4 = fill_aligned(2, 48, 8);
    for (uint32_t i = 0; i < 8; ++i) {
        CHECK(idxs4[i] == 2*bs + 16 + i);
    }

    // ... but seq 0 cannot extend that block - it is not a member of its cells
    const auto idxs5 = fill_aligned(0, 32, 8);
    for (uint32_t i = 0; i < 8; ++i) {
        CHECK(idxs5[i] == 3*bs + i);
    }

    // tokens of different sequences in one call never share a fresh block
    {
        const llama_pos    poss2[2] = { 64, 65 };
        const llama_seq_id seqs2[2] = { 3, 4 };

        std::vector<uint32_t> idxs6;

        CHECK(cells.alloc_find_seq_aligned(poss2, seqs2, 2, idxs6) == 2);
        CHECK(idxs6[0]/bs != idxs6[1]/bs);

        for (uint32_t i = 0; i < 2; ++i) {
            cells.pos_set(idxs6[i], poss2[i]);
            cells.seq_add(idxs6[i], seqs2[i]);
        }
    }

    // a shared multi-sequence run (one cell, several members) can be continued by any
    // member sequence
    {
        const llama_pos    poss3[2] = { 96, 97 };
        const llama_seq_id seqs3[2] = { 5, 5 };

        std::vector<uint32_t> idxs7;

        CHECK(cells.alloc_find_seq_aligned(poss3, seqs3, 2, idxs7) == 2);

        for (uint32_t i = 0; i < 2; ++i) {
            cells.pos_set(idxs7[i], poss3[i]);
            cells.seq_add(idxs7[i], 5);
            cells.seq_add(idxs7[i], 6);
        }

        const auto idxs8 = fill_aligned(6, 98, 4);
        for (uint32_t i = 0; i < 4; ++i) {
            CHECK(idxs8[i] == idxs7[0] + 2 + i);
        }
    }

    // exhaust the remaining free blocks, then a new range must fail even though
    // individual cells are still free
    fill_aligned(7, 128, 32);
    fill_aligned(7, 160, 32);

    const llama_pos    poss_full = 192;
    const llama_seq_id seqs_full = 7;

    std::vector<uint32_t> idxs_full;

    CHECK(cells.alloc_find_seq_aligned(&poss_full, &seqs_full, 1, idxs_full) == 0);
}

// per-sequence paged-attention block table built from the cell metadata
static void test_seq_block_table() {
    fprintf(stderr, "%s\n", __func__);

    constexpr uint32_t bs = llama_kv_cells::block_size;

    llama_kv_cells cells;
    cells.resize(4*bs);

    auto fill_aligned = [&](llama_seq_id seq_id, llama_pos p0, uint32_t n) {
        std::vector<llama_pos>    poss(n);
        std::vector<llama_seq_id> seqs(n, seq_id);

        for (uint32_t i = 0; i < n; ++i) {
            poss[i] = p0 + (llama_pos) i;
        }

        std::vector<uint32_t> idxs;

        CHECK(cells.alloc_find_seq_aligned(poss.data(), seqs.data(), n, idxs) == n);

        for (uint32_t i = 0; i < n; ++i) {
            cells.pos_set(idxs[i], poss[i]);
            cells.seq_add(idxs[i], seq_id);
        }
    };

    fill_aligned(0, 0, 32);  // block 0
    fill_aligned(0, 32, 40); // block 1 + block 2, offsets [0, 8)
    fill_aligned(1, 0, 32);  // block 3 (blocks 0/1 are owned by seq 0)

    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        std::vector<int32_t> table(4, -2);

        CHECK(cells.seq_block_table(seq_id, table));

        const llama_pos p_min = cells.seq_pos_min(seq_id);
        const llama_pos p_max = cells.seq_pos_max(seq_id);

        CHECK(p_min >= 0);

        // every position of the sequence resolves through the table
        for (llama_pos p = p_min; p <= p_max; ++p) {
            const int32_t b = table[p/bs];

            CHECK(b >= 0);

            const uint32_t cell = (uint32_t) b*bs + (uint32_t) p%bs;

            CHECK(!cells.is_empty(cell));
            CHECK(cells.pos_get(cell) == p);
            CHECK(cells.seq_has(cell, seq_id));
        }

        // logical blocks outside the sequence range are -1
        for (uint32_t lb = 0; lb < table.size(); ++lb) {
            const bool has = (llama_pos) (lb*bs) <= p_max && (llama_pos) (lb*bs + bs - 1) >= p_min;

            if (!has) {
                CHECK(table[lb] == -1);
            }
        }
    }

    // an absent sequence has an empty table
    {
        std::vector<int32_t> table(4, -2);

        CHECK(cells.seq_block_table(2, table));

        for (const auto b : table) {
            CHECK(b == -1);
        }
    }

    // sharing cells of another block covering the same aligned range makes the table of
    // the sequence ambiguous - must be detected
    for (uint32_t j = 3*bs; j < 4*bs; ++j) {
        cells.seq_add(j, 0);
    }

    {
        std::vector<int32_t> table(4, -2);

        CHECK(!cells.seq_block_table(0, table));
    }
}

// re-application of already-cached positions (MTP draft/verify, rollback re-writes):
// the aligned cell occupied by the same sequence is reused instead of spilling to a new
// block - this is what keeps one block per (sequence, aligned range)
static void test_seq_aligned_reuse() {
    fprintf(stderr, "%s\n", __func__);

    constexpr uint32_t bs = llama_kv_cells::block_size;

    llama_kv_cells cells;
    cells.resize(4*bs);

    auto fill_aligned = [&](llama_seq_id seq_id, llama_pos p0, uint32_t n) {
        std::vector<llama_pos>    poss(n);
        std::vector<llama_seq_id> seqs(n, seq_id);

        for (uint32_t i = 0; i < n; ++i) {
            poss[i] = p0 + (llama_pos) i;
        }

        std::vector<uint32_t> idxs;

        CHECK(cells.alloc_find_seq_aligned(poss.data(), seqs.data(), n, idxs) == n);

        for (uint32_t i = 0; i < n; ++i) {
            if (!cells.is_empty(idxs[i])) {
                cells.rm(idxs[i]);
            }

            cells.pos_set(idxs[i], poss[i]);
            cells.seq_add(idxs[i], seq_id);
        }

        return idxs;
    };

    const auto idxs0 = fill_aligned(0, 0, 40); // blocks 0 and 1 (offsets [0, 8))

    // re-applying positions [32, 40) of seq 0 reuses exactly the same cells
    const auto idxs1 = fill_aligned(0, 32, 8);
    for (uint32_t i = 0; i < 8; ++i) {
        CHECK(idxs1[i] == idxs0[32 + i]);
    }

    // a different sequence does not reuse those cells - it gets its own block
    const auto idxs2 = fill_aligned(1, 32, 8);
    for (uint32_t i = 0; i < 8; ++i) {
        CHECK(idxs2[i] == 2*bs + i);
    }

    // the block table of seq 0 is still unambiguous
    {
        std::vector<int32_t> table(4, -2);

        CHECK(cells.seq_block_table(0, table));
        CHECK(table[0] == 0 && table[1] == 1 && table[2] == -1);
    }
}

// paged prefix sharing (Phase 2): a prefix block is shared read-only across sequences
// via pure bookkeeping (seq_add, the cells-level equivalent of the loc_same re-attach
// in llama_kv_cache::prefix_copy_impl) - no cells are allocated or copied, and both
// sequences' block tables map the shared ranges to the same physical blocks
static void test_paged_prefix_share() {
    fprintf(stderr, "%s\n", __func__);

    constexpr uint32_t bs = llama_kv_cells::block_size;

    llama_kv_cells cells;
    cells.resize(8*bs);

    auto fill_aligned = [&](llama_seq_id seq_id, llama_pos p0, uint32_t n) {
        std::vector<llama_pos>    poss(n);
        std::vector<llama_seq_id> seqs(n, seq_id);

        for (uint32_t i = 0; i < n; ++i) {
            poss[i] = p0 + (llama_pos) i;
        }

        std::vector<uint32_t> idxs;

        CHECK(cells.alloc_find_seq_aligned(poss.data(), seqs.data(), n, idxs) == n);

        for (uint32_t i = 0; i < n; ++i) {
            if (!cells.is_empty(idxs[i])) {
                cells.rm(idxs[i]);
            }

            cells.pos_set(idxs[i], poss[i]);
            cells.seq_add(idxs[i], seq_id);
        }

        return idxs;
    };

    // seq 0 owns positions [0, 96): blocks 0, 1, 2
    fill_aligned(0, 0, 96);

    // blocks 0 and 1 are registered in the prefix cache (pinned)
    cells.block_pin(0);
    cells.block_pin(1);

    const uint32_t used_before = cells.get_used();

    // re-attach the shared prefix to seq 1 (pure bookkeeping - no allocation)
    for (uint32_t j = 0; j < 2*bs; ++j) {
        cells.seq_add(j, 1);
    }

    // no cells were allocated or copied by the re-attach
    CHECK(cells.get_used() == used_before);

    // seq 1 continues past the shared prefix with its own tail
    const auto idxs1 = fill_aligned(1, 64, 32);
    for (uint32_t i = 0; i < 32; ++i) {
        CHECK(idxs1[i] == 3*bs + i); // a new block - block 2 is owned by seq 0
    }

    // both sequences' block tables map the shared ranges to the same physical blocks
    {
        std::vector<int32_t> t0(8, -2), t1(8, -2);

        CHECK(cells.seq_block_table(0, t0));
        CHECK(cells.seq_block_table(1, t1));

        CHECK(t0[0] == 0 && t0[1] == 1 && t0[2] == 2);
        CHECK(t1[0] == 0 && t1[1] == 1 && t1[2] == 3);
        CHECK(t0[3] == -1 && t1[3] == -1);

        // every position of both sequences resolves through its table
        for (llama_seq_id s = 0; s < 2; ++s) {
            const auto & t = s == 0 ? t0 : t1;

            for (llama_pos p = 0; p < 96; ++p) {
                const uint32_t c = (uint32_t) t[p/bs]*bs + (uint32_t) p%bs;

                CHECK(!cells.is_empty(c));
                CHECK(cells.pos_get(c) == p);
                CHECK(cells.seq_has(c, s));
            }
        }
    }

    // removing seq 0 keeps the shared cells (seq 1 still references them) and its table
    CHECK(cells.seq_rm_range(0, 0, std::numeric_limits<llama_pos>::max()) == 0);

    {
        std::vector<int32_t> t1(8, -2);

        CHECK(cells.seq_block_table(1, t1));
        CHECK(t1[0] == 0 && t1[1] == 1 && t1[2] == 3);

        for (llama_pos p = 0; p < 96; ++p) {
            const uint32_t c = (uint32_t) t1[p/bs]*bs + (uint32_t) p%bs;

            CHECK(!cells.is_empty(c));
            CHECK(cells.seq_has(c, 1));
            CHECK(!cells.seq_has(c, 0));
        }
    }

    // removing seq 1 leaves the pinned blocks retained as cache-owned, the rest freed
    CHECK(cells.seq_rm_range(1, 0, std::numeric_limits<llama_pos>::max()) == 0);
    CHECK(cells.get_used() == 2*bs);
    CHECK(cells.block_all_seqless(0));
    CHECK(cells.block_all_seqless(1));
    CHECK(cells.block_is_free(2));
    CHECK(cells.block_is_free(3));
}

int main() {
    test_alloc_rollback();
    test_exhaustion();
    test_multi_seq();
    test_interior_hole();
    test_shift_div();
    test_duplicate_pos();
    test_cp_set();
    test_randomized();
    test_seq_aligned_alloc();
    test_seq_block_table();
    test_seq_aligned_reuse();
    test_paged_prefix_share();

    if (n_fail == 0) {
        fprintf(stderr, "all tests passed\n");
        return 0;
    }

    fprintf(stderr, "%d checks failed\n", n_fail);
    return 1;
}
