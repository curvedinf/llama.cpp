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

int main() {
    test_alloc_rollback();
    test_exhaustion();
    test_multi_seq();
    test_interior_hole();
    test_shift_div();
    test_duplicate_pos();
    test_cp_set();
    test_randomized();

    if (n_fail == 0) {
        fprintf(stderr, "all tests passed\n");
        return 0;
    }

    fprintf(stderr, "%d checks failed\n", n_fail);
    return 1;
}
