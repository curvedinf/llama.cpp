#pragma once

#include "llama.h"
#include "llama-cparams.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_kv_cell_ext {
    // 2D spatial positions, typically used for M-RoPE
    llama_pos x = 0;
    llama_pos y = 0;

    // return true if the current 2D spatial position is greater than other
    bool is_2d_gt(llama_pos ox, llama_pos oy) const {
        return (y > oy) || (y == oy && x > ox);
    }

    void reset() {
        static_assert(std::is_trivially_copyable_v<llama_kv_cell_ext>);

        memset(this, 0, sizeof(*this));
    }
};

// meta information about KV cells that can be part of multiple sequences at the same time
//
// the cells are addressed by a flat index (required by the dense K/V tensor views), but the
// bookkeeping on top of them is block-granular (vLLM-style):
//
//   - the cells are partitioned into fixed-size blocks of block_size cells
//   - each block is in one of 3 states - free / partial / full - tracked by intrusive lists
//   - allocation draws from the partial list first, then from the free list
//   - a high-water mark (hwm) delimits the range [lwm, hwm) of blocks that can contain
//     used cells, which makes used_max_p1() and range removals cheap
//
// per-cell metadata is slim: pos + ext + shift + (seq0, n_seq). cells shared by more than one
// sequence keep the extra seq ids in a hash map (seq_extra), empty in the common case.
//
// per-sequence (min, max) positions are updated incrementally on insert and repaired lazily
// on removal of an extremum, using block-granular scans with position-range skips.
//
class llama_kv_cells {
public:
    // number of cells per block
    // note: keep a multiple of 32 for compatibility with quantized K/V types (e.g. Q8_0)
    static constexpr uint32_t block_size = 32;

    void reset() {
        std::fill(pos.begin(),   pos.end(),   -1);
        std::fill(shift.begin(), shift.end(),  0);
        std::fill(seq0.begin(),  seq0.end(),   0);
        std::fill(n_seq.begin(), n_seq.end(),  0);

        for (auto & e : ext) {
            e.reset();
        }

        has_shift = false;

        seq_extra.clear();

        n_used = 0;

        hwm = 0;
        lwm = 0;

        partial_head = -1;

        // rebuild the free list so that blocks are drawn in ascending order
        free_head = blocks.empty() ? (uint32_t) -1 : 0;

        for (uint32_t b = 0; b < blocks.size(); ++b) {
            auto & blk = blocks[b];

            blk.pos_min = -1;
            blk.pos_max = -1;
            blk.n_used  = 0;
            blk.ref     = 0;

            blk.list = LIST_FREE;
            blk.prev = b == 0 ? (uint32_t) -1 : b - 1;
            blk.next = b + 1 < blocks.size() ? b + 1 : (uint32_t) -1;
        }

        seq_n.fill(0);
        seq_min.fill(-1);
        seq_max.fill(-1);
        seq_dirty.reset();
    }

    void reset_shift() {
        has_shift = false;

        std::fill(shift.begin(), shift.end(), 0);
    }

    uint32_t size() const {
        return pos.size();
    }

    void resize(uint32_t n) {
        pos.resize(n);
        ext.resize(n);
        shift.resize(n);
        seq0.resize(n);
        n_seq.resize(n);

        blocks.resize((n + block_size - 1)/block_size);

        reset();
    }

    bool is_empty(uint32_t i) const {
        assert(i < pos.size());
        assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

        return pos[i] == -1;
    }

    uint32_t get_used() const {
        return n_used;
    }

    // the index of the last cell that is used + 1
    // return 0 if no cells are used
    uint32_t used_max_p1() const {
        if (hwm == 0) {
            return 0;
        }

        const uint32_t b = hwm - 1;

        assert(blocks[b].n_used > 0);

        for (uint32_t j = cell_end(b); j-- > cell_begin(b);) {
            if (pos[j] != -1) {
                return j + 1;
            }
        }

        assert(false);
        return 0;
    }

    bool get_has_shift() const {
        return has_shift;
    }

    //
    // block allocator
    //

    uint32_t n_blocks() const {
        return blocks.size();
    }

    // block indices in [block_lo, block_hi) can contain used cells - all blocks outside are free
    uint32_t block_lo() const {
        return lwm;
    }

    uint32_t block_hi() const {
        return hwm;
    }

    bool block_is_free(uint32_t b) const {
        return blocks[b].list == LIST_FREE;
    }

    bool block_is_full(uint32_t b) const {
        return blocks[b].n_used == block_len(b);
    }

    // number of prefix-cache pins on the block (see llama-prefix-cache.h)
    uint32_t block_ref(uint32_t b) const {
        return blocks[b].ref;
    }

    // true if no cell of the block belongs to a sequence (cache-owned block)
    bool block_all_seqless(uint32_t b) const {
        for (uint32_t j = cell_begin(b); j < cell_end(b); ++j) {
            if (n_seq[j] != 0) {
                return false;
            }
        }

        return true;
    }

    // pin a block for the prefix cache - pinned blocks are never reallocated and
    //   their cells are retained (cache-owned) when their last sequence is removed
    void block_pin(uint32_t b) {
        blocks[b].ref++;
    }

    // unpin a block; if it was the last pin and all cells are cache-owned, free them
    // return true if the cells were freed
    bool block_unpin(uint32_t b) {
        auto & blk = blocks[b];

        assert(blk.ref > 0);

        if (--blk.ref > 0) {
            return false;
        }

        if (blk.n_used != block_len(b) || !block_all_seqless(b)) {
            return false;
        }

        for (uint32_t j = cell_begin(b); j < cell_end(b); ++j) {
            rm(j);
        }

        return true;
    }

    // min/max positions of the used cells in the block (any sequence)
    llama_pos block_pos_min(uint32_t b) const {
        return blocks[b].pos_min;
    }

    llama_pos block_pos_max(uint32_t b) const {
        return blocks[b].pos_max;
    }

    uint32_t cell_begin(uint32_t b) const {
        return b*block_size;
    }

    uint32_t cell_end(uint32_t b) const {
        return std::min(b*block_size + block_size, (uint32_t) pos.size());
    }

    // collect up to n empty cell indices, drawing from partially-used blocks first, then from
    //   free blocks. the block state itself is not modified - that happens in pos_set()
    // return the number of indices collected
    uint32_t alloc_find(uint32_t n, std::vector<uint32_t> & out) const {
        uint32_t res = 0;

        for (uint32_t b = partial_head; b != (uint32_t) -1 && res < n; b = blocks[b].next) {
            res += block_empty_cells(b, n - res, out);
        }

        for (uint32_t b = free_head; b != (uint32_t) -1 && res < n; b = blocks[b].next) {
            res += block_empty_cells(b, n - res, out);
        }

        return res;
    }

    // seq-aligned ("paged") variant of alloc_find() for GPU paged attention (LLAMA_KV_PAGED):
    //
    // the token with logical position poss[i] (belonging to sequence seqs[i]) is placed at
    // cell
    //
    //   b*block_size + poss[i] % block_size
    //
    // of a block b whose used cells all (a) have positions in the same aligned block_size
    // range and (b) belong to sequence seqs[i]. this is what the paged-attention block
    // table requires:
    //
    //   cell_of(seq, p) == block_table[seq][p/block_size]*block_size + p%block_size
    //
    // constraint (a) keeps every block mapped to a single aligned range; constraint (b)
    // keeps all cells of one (seq, range) pair in a single block, so the per-sequence
    // block table is unambiguous. cells shared by multiple sequences (prefix sharing)
    // satisfy (b) for every member sequence and can therefore be extended by any of them.
    //
    // matching partial blocks are drawn first, then free blocks. like alloc_find(), the
    // block state itself is not modified - that happens in pos_set(). return the number of
    // indices collected
    uint32_t alloc_find_seq_aligned(const llama_pos * poss, const llama_seq_id * seqs, uint32_t n, std::vector<uint32_t> & out) const {
        // true if every used cell of block b belongs to sequence seq_id
        auto block_owned_by = [&](uint32_t b, llama_seq_id seq_id) {
            for (uint32_t j = cell_begin(b); j < cell_end(b); ++j) {
                if (!is_empty(j) && !seq_has(j, seq_id)) {
                    return false;
                }
            }

            return true;
        };

        // assignments made during this call, since the cell state changes only in pos_set():
        //   - blk_range[b]: block b (originally free) is reserved for this aligned range
        //   - blk_seqs[b]:  intersection of the seq sets of the tokens placed in b - a new
        //                   token can join only if its own sequence is in the intersection
        //                   (the in-call analogue of constraint (b) above)
        //   - cell_taken:   individual cells already handed out
        std::unordered_map<uint32_t, uint32_t>                   blk_range;
        std::unordered_map<uint32_t, std::vector<llama_seq_id>>  blk_seqs;
        std::unordered_set<uint32_t>                             cell_taken;

        uint32_t res = 0;

        for (uint32_t i = 0; i < n; ++i) {
            const llama_pos    p = poss[i];
            const llama_seq_id s = seqs[i];

            const uint32_t range = (uint32_t) p/block_size;
            const uint32_t off   = (uint32_t) p%block_size;

            const llama_pos p_min = (llama_pos) (range*block_size);
            const llama_pos p_max = (llama_pos) (range*block_size + block_size - 1);

            uint32_t b_sel = (uint32_t) -1;

            // pass 0: the aligned cell may already be occupied by the same sequence - e.g.
            // MTP draft tokens waiting for verification in the shared cells, or positions
            // re-applied after a rollback. reusing it keeps the sequence in a single block
            // per aligned range. never reuse cells shared with other sequences - the
            // caller rewrites the cell (apply_ubatch), which requires single membership
            for (uint32_t b = lwm; b < hwm; ++b) {
                if (blocks[b].list == LIST_FREE) {
                    continue;
                }

                if (blocks[b].pos_min < p_min || blocks[b].pos_max > p_max) {
                    continue;
                }

                const uint32_t c = b*block_size + off;

                if (c < pos.size() && !is_empty(c) && cell_taken.count(c) == 0 &&
                        seq_count(c) == 1 && seq_get(c) == s) {
                    b_sel = b;
                    break;
                }
            }

            // pass 1: used blocks covering the same aligned range, owned by this sequence
            for (uint32_t b = lwm; b < hwm && b_sel == (uint32_t) -1; ++b) {
                if (blocks[b].list == LIST_FREE) {
                    continue;
                }

                if (blocks[b].pos_min < p_min || blocks[b].pos_max > p_max) {
                    continue;
                }

                if (!block_owned_by(b, s)) {
                    continue;
                }

                const uint32_t c = b*block_size + off;

                if (c >= pos.size() || !is_empty(c) || cell_taken.count(c) != 0) {
                    continue;
                }

                b_sel = b;
                break;
            }

            // pass 2: free blocks
            if (b_sel == (uint32_t) -1) {
                for (uint32_t b = 0; b < blocks.size(); ++b) {
                    if (blocks[b].list != LIST_FREE) {
                        continue;
                    }

                    // a block reserved earlier in this call can only grow within its range
                    //   and only for sequences that own every cell placed there so far
                    const auto it = blk_range.find(b);
                    if (it != blk_range.end()) {
                        if (it->second != range) {
                            continue;
                        }

                        const auto & sb = blk_seqs[b];

                        if (std::find(sb.begin(), sb.end(), s) == sb.end()) {
                            continue;
                        }
                    }

                    const uint32_t c = b*block_size + off;

                    if (c >= pos.size() || cell_taken.count(c) != 0) {
                        continue;
                    }

                    b_sel = b;
                    break;
                }
            }

            if (b_sel == (uint32_t) -1) {
                break;
            }

            if (blk_range.emplace(b_sel, range).second) {
                blk_seqs[b_sel] = { s };
            }

            cell_taken.insert(b_sel*block_size + off);

            out.push_back(b_sel*block_size + off);

            res++;
        }

        return res;
    }

    // fill out with the paged-attention block table of the given sequence:
    //   out[p/block_size] = physical block id of the cell holding position p, -1 elsewhere
    // only meaningful with seq-aligned allocation (alloc_find_seq_aligned)
    // return false if the table is ambiguous - i.e. cells of the same aligned range of this
    // sequence ended up in more than one block (should not happen with the allocation
    // policy above - treated as an error by the caller)
    bool seq_block_table(llama_seq_id seq_id, std::vector<int32_t> & out) const {
        std::fill(out.begin(), out.end(), -1);

        bool res = true;

        for (uint32_t b = lwm; b < hwm; ++b) {
            if (blocks[b].list == LIST_FREE) {
                continue;
            }

            for (uint32_t j = cell_begin(b); j < cell_end(b); ++j) {
                if (is_empty(j) || !seq_has(j, seq_id)) {
                    continue;
                }

                const llama_pos p = pos[j];

                // seq-aligned allocation guarantees that the cell offset matches the position
                assert((uint32_t) p%block_size == j%block_size);

                const uint32_t lb = (uint32_t) p/block_size;

                if (lb < out.size()) {
                    if (out[lb] != -1 && out[lb] != (int32_t) b) {
                        res = false;
                    }

                    out[lb] = (int32_t) b;
                }
            }
        }

        return res;
    }

    // remove all cells with positions in [p0, p1)
    // return the lowest freed cell index (or size() if none was freed)
    uint32_t rm_range(llama_pos p0, llama_pos p1) {
        uint32_t res = size();

        for (uint32_t b = hwm; b-- > lwm;) {
            const auto & blk = blocks[b];

            if (blk.list == LIST_FREE) {
                continue;
            }

            if (blk.pos_max < p0 || blk.pos_min >= p1) {
                continue;
            }

            for (uint32_t j = cell_end(b); j-- > cell_begin(b);) {
                if (pos[j] == -1 || !pos_in(j, p0, p1)) {
                    continue;
                }

                rm(j);

                res = j;
            }
        }

        return res;
    }

    // remove sequence seq_id from all cells with positions in [p0, p1)
    // return the lowest freed cell index (or size() if none was freed)
    uint32_t seq_rm_range(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        uint32_t res = size();

        for (uint32_t b = hwm; b-- > lwm;) {
            const auto & blk = blocks[b];

            if (blk.list == LIST_FREE) {
                continue;
            }

            if (blk.pos_max < p0 || blk.pos_min >= p1) {
                continue;
            }

            for (uint32_t j = cell_end(b); j-- > cell_begin(b);) {
                if (pos[j] == -1 || !pos_in(j, p0, p1) || !seq_has(j, seq_id)) {
                    continue;
                }

                seq_rm(j, seq_id);

                res = j;
            }
        }

        return res;
    }

    // keep only sequence seq_id in the cache
    // return the lowest freed cell index (or size() if none was freed)
    uint32_t seq_keep_all(llama_seq_id seq_id) {
        uint32_t res = size();

        for (uint32_t b = lwm; b < hwm; ++b) {
            if (blocks[b].list == LIST_FREE) {
                continue;
            }

            for (uint32_t j = cell_begin(b); j < cell_end(b); ++j) {
                if (pos[j] == -1) {
                    continue;
                }

                if (seq_keep(j, seq_id)) {
                    res = std::min(res, j);
                }
            }
        }

        return res;
    }

    // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
    // note: the copy is a plain container of per-cell data - only set() can consume it
    llama_kv_cells cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llama_kv_cells res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            const auto idx = i + j;

            res.pos[j]   = pos[idx];
            res.ext[j]   = ext[idx];
            res.seq0[j]  = seq0[idx];
            res.n_seq[j] = n_seq[idx];

            const auto it = seq_extra.find(idx);
            if (it != seq_extra.end()) {
                res.seq_extra[j] = it->second;
            }

            assert(shift[idx] == 0);
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j]   = pos[idx];
            res.ext[j]   = ext[idx];
            res.seq0[j]  = seq0[idx];
            res.n_seq[j] = n_seq[idx];

            const auto it = seq_extra.find(idx);
            if (it != seq_extra.end()) {
                res.seq_extra[j] = it->second;
            }

            assert(shift[idx] == 0);
        }

        return res;
    }

    // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
    void set(uint32_t i, const llama_kv_cells & other) {
        assert(i + other.pos.size() <= pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            set_cell(i + j, other, j);
        }
    }

    // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    void set(const std::vector<uint32_t> & idxs, const llama_kv_cells & other) {
        assert(idxs.size() == other.pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            set_cell(idxs[j], other, j);
        }
    }

    // clear a non-empty cell
    void rm(uint32_t i) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        // pinned (prefix-cached) blocks must be unpinned before their cells are removed
        assert(blocks[i/block_size].ref == 0);

        const llama_pos p = pos[i];

        seq_cell_rm_all(i);

        seq0[i]  = 0;
        n_seq[i] = 0;

        seq_extra.erase(i);

        pos[i]   = -1;
        shift[i] = 0;

        ext[i].reset();

        n_used--;

        block_cell_rm(i, p);
    }

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    bool seq_rm(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(seq_has(i, seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        const llama_pos p = pos[i];

        seq_stat_rm(i, seq_id);

        uint32_t cnt = 0;

        if (seq0[i] == (uint8_t) seq_id) {
            const auto it = seq_extra.find(i);
            if (it != seq_extra.end()) {
                // promote an extra seq to seq0
                seq0[i] = it->second.back();
                it->second.pop_back();

                cnt = 1 + it->second.size();

                if (it->second.empty()) {
                    seq_extra.erase(it);
                }
            }
        } else {
            const auto it = seq_extra.find(i);
            assert(it != seq_extra.end());

            auto & v = it->second;

            const auto iv = std::find(v.begin(), v.end(), (uint8_t) seq_id);
            assert(iv != v.end());

            v.erase(iv);

            cnt = 1 + v.size();

            if (v.empty()) {
                seq_extra.erase(it);
            }
        }

        n_seq[i] = cnt > 254 ? 255 : (uint8_t) cnt;

        if (cnt == 0) {
            if (blocks[i/block_size].ref > 0) {
                // the block is pinned by the prefix cache - retain the cell as cache-owned:
                // the position and the K/V data are preserved, only the membership is dropped
                return false;
            }

            pos[i]   = -1;
            shift[i] = 0;

            ext[i].reset();

            n_used--;

            block_cell_rm(i, p);

            return true;
        }

        return false;
    }

    // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
    bool seq_keep(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());

        if (seq_has(i, seq_id)) {
            // remove all other sequences from the cell
            llama_seq_id seqs[LLAMA_MAX_SEQ];

            const int n = cell_seqs(i, seqs);

            for (int k = 0; k < n; ++k) {
                if (seqs[k] != seq_id) {
                    seq_stat_rm(i, seqs[k]);
                }
            }

            seq0[i]  = (uint8_t) seq_id;
            n_seq[i] = 1;

            seq_extra.erase(i);

            return false;
        }

        if (n_seq[i] > 0) {
            rm(i);

            return true;
        }

        assert(pos[i] == -1);

        return false;
    }

    // number of different sequences in the cell
    // note: the result saturates at 255 - such cells are only reachable with > 254 sequences
    int seq_count(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return n_seq[i];
    }

    // check if the cell contains seq_id
    bool seq_has(uint32_t i, llama_seq_id seq_id) const {
        assert(i < pos.size());
        assert(seq_id >= 0);

        if (n_seq[i] == 0) {
            return false;
        }

        if (seq0[i] == (uint8_t) seq_id) {
            return true;
        }

        const auto it = seq_extra.find(i);
        if (it == seq_extra.end()) {
            return false;
        }

        const auto & v = it->second;

        return std::find(v.begin(), v.end(), (uint8_t) seq_id) != v.end();
    }

    // note: call only if the cell is not empty and the seq_id is not in the cell
    void seq_add(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(!seq_has(i, seq_id));

        if (n_seq[i] == 0) {
            seq0[i] = (uint8_t) seq_id;
            n_seq[i] = 1;
        } else {
            seq_extra[i].push_back((uint8_t) seq_id);

            if (n_seq[i] < 255) {
                n_seq[i]++;
            }
        }

        seq_stat_add(i, seq_id);
    }

    // return the sequence id of this cell
    // note: call only for cells with exactly one sequence
    llama_seq_id seq_get(uint32_t i) const {
        assert(n_seq[i] == 1);

        return seq0[i];
    }

    // the minimum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_min(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_dirty.test(seq_id)) {
            seq_repair(seq_id);
        }

        return seq_min[seq_id];
    }

    // the maximum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_max(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_dirty.test(seq_id)) {
            seq_repair(seq_id);
        }

        return seq_max[seq_id];
    }

    // note: call only if the cell is not empty
    llama_pos pos_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return pos[i];
    }

    const llama_kv_cell_ext & ext_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return ext[i];
    }

    // note: call only if the cell is not empty
    llama_pos get_shift(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return shift[i];
    }

    // check if a cell is not empty and its position is within [p0, p1)
    bool pos_in(uint32_t i, llama_pos p0, llama_pos p1) const {
        assert(i < pos.size());

        return pos[i] >= p0 && pos[i] < p1;
    }

    // set the position of an empty cell
    // does not modify "has_shift"
    // note: call only if the cell is empty
    void pos_set(uint32_t i, llama_pos p) {
        assert(i < pos.size());
        assert(pos[i] == -1);
        assert(n_seq[i] == 0);

        pos[i] = p;

        n_used++;

        block_cell_add(i, p);
    }

    void ext_set(uint32_t i, llama_kv_cell_ext p) {
        assert(i < ext.size());
        ext[i] = p;
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    // return true if the cell became empty (i.e. its position became negative)
    bool pos_add(uint32_t i, llama_pos d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llama_pos p_old = pos[i];

        pos[i]   += d;
        shift[i] += d;

        has_shift = true;

        if (pos[i] < 0) {
            // the cell becomes empty - restore the old position so that rm() can
            //   update the seq stats correctly
            pos[i] = p_old;

            rm(i);

            return true;
        }

        block_pos_update(i, p_old);

        seq_stat_move(i, p_old);

        return false;
    }

    // pos[i] = pos[i] / d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    void pos_div(uint32_t i, int d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llama_pos p_old = pos[i];

        pos[i]   /= d;
        shift[i] += p_old - pos[i];

        has_shift = true;

        block_pos_update(i, p_old);

        seq_stat_move(i, p_old);
    }

private:
    bool has_shift = false;

    std::vector<llama_pos> pos;

    // stores extra info per cell
    std::vector<llama_kv_cell_ext> ext;

    // this array accumulates any applied shifts to the pos array since the last reset_shift() call
    // (see the K-shift graph in llama-kv-cache)
    std::vector<llama_pos> shift;

    // the first sequence of each cell and the total number of sequences in it
    // n_seq[i] == 0 means the cell has no sequences, n_seq[i] == 255 means 255 or more
    std::vector<uint8_t> seq0;
    std::vector<uint8_t> n_seq;

    // the remaining sequences of cells with more than one sequence: cell idx -> extra seq ids
    // note: empty in the common case of single-sequence cells
    std::unordered_map<uint32_t, std::vector<uint8_t>> seq_extra;

    uint32_t n_used = 0;

    enum block_list : uint8_t {
        LIST_NONE    = 0,
        LIST_FREE    = 1,
        LIST_PARTIAL = 2,
    };

    struct block_t {
        llama_pos pos_min = -1; // min/max positions of the used cells in the block (any sequence)
        llama_pos pos_max = -1;

        uint32_t next = -1;     // intrusive list links
        uint32_t prev = -1;

        uint32_t n_used = 0;    // number of used cells in the block

        uint32_t ref = 0;       // reserved for future block sharing (prefix cache)

        block_list list = LIST_NONE;
    };

    std::vector<block_t> blocks;

    uint32_t free_head    = -1;
    uint32_t partial_head = -1;

    // blocks in [lwm, hwm) can contain used cells - all blocks outside are free
    uint32_t hwm = 0;
    uint32_t lwm = 0;

    // per-sequence stats: number of cells and min/max positions
    std::array<uint32_t, LLAMA_MAX_SEQ> seq_n;

    // the min/max positions are repaired lazily on access (see seq_repair)
    mutable std::array<llama_pos, LLAMA_MAX_SEQ> seq_min;
    mutable std::array<llama_pos, LLAMA_MAX_SEQ> seq_max;

    // sequences whose min/max positions need to be recomputed
    mutable std::bitset<LLAMA_MAX_SEQ> seq_dirty;

    uint32_t block_len(uint32_t b) const {
        return cell_end(b) - cell_begin(b);
    }

    uint32_t block_empty_cells(uint32_t b, uint32_t n, std::vector<uint32_t> & out) const {
        uint32_t res = 0;

        for (uint32_t j = cell_begin(b); j < cell_end(b) && res < n; ++j) {
            if (pos[j] == -1) {
                out.push_back(j);
                res++;
            }
        }

        return res;
    }

    void list_insert(block_list list, uint32_t b) {
        auto & blk = blocks[b];

        assert(blk.list == LIST_NONE);

        uint32_t & head = list == LIST_FREE ? free_head : partial_head;

        blk.list = list;
        blk.prev = -1;
        blk.next = head;

        if (head != (uint32_t) -1) {
            blocks[head].prev = b;
        }

        head = b;
    }

    void list_remove(uint32_t b) {
        auto & blk = blocks[b];

        if (blk.list == LIST_NONE) {
            return;
        }

        uint32_t & head = blk.list == LIST_FREE ? free_head : partial_head;

        if (blk.prev != (uint32_t) -1) {
            blocks[blk.prev].next = blk.next;
        }

        if (blk.next != (uint32_t) -1) {
            blocks[blk.next].prev = blk.prev;
        }

        if (head == b) {
            head = blk.next;
        }

        blk.list = LIST_NONE;
        blk.prev = -1;
        blk.next = -1;
    }

    // account for a new used cell i with position p
    void block_cell_add(uint32_t i, llama_pos p) {
        const uint32_t b = i/block_size;

        auto & blk = blocks[b];

        if (blk.n_used == 0) {
            list_remove(b);

            blk.pos_min = p;
            blk.pos_max = p;
            blk.n_used  = 1;

            if (b + 1 > hwm) {
                hwm = b + 1;
            }

            if (b < lwm) {
                lwm = b;
            }

            if (block_len(b) > 1) {
                list_insert(LIST_PARTIAL, b);
            }

            return;
        }

        blk.pos_min = std::min(blk.pos_min, p);
        blk.pos_max = std::max(blk.pos_max, p);

        if (++blk.n_used == block_len(b)) {
            // the block is full - it stays out of the lists
            list_remove(b);
        }
    }

    // account for a removed used cell i that had position p
    void block_cell_rm(uint32_t i, llama_pos p) {
        const uint32_t b = i/block_size;

        auto & blk = blocks[b];

        assert(blk.n_used > 0);

        if (blk.n_used == 1) {
            // the block became free
            blk.n_used  = 0;
            blk.pos_min = -1;
            blk.pos_max = -1;

            list_remove(b);
            list_insert(LIST_FREE, b);

            // trim the water marks
            if (b + 1 == hwm) {
                while (hwm > 0 && blocks[hwm - 1].list == LIST_FREE) {
                    hwm--;
                }
                if (lwm > hwm) {
                    lwm = hwm;
                }
            }

            if (b == lwm) {
                while (lwm < hwm && blocks[lwm].list == LIST_FREE) {
                    lwm++;
                }
            }

            return;
        }

        if (--blk.n_used == block_len(b) - 1) {
            // the block is no longer full
            list_insert(LIST_PARTIAL, b);
        }

        if (p == blk.pos_min || p == blk.pos_max) {
            // recompute the block position range
            blk.pos_min = std::numeric_limits<llama_pos>::max();
            blk.pos_max = -1;

            for (uint32_t j = cell_begin(b); j < cell_end(b); ++j) {
                if (pos[j] == -1) {
                    continue;
                }

                blk.pos_min = std::min(blk.pos_min, pos[j]);
                blk.pos_max = std::max(blk.pos_max, pos[j]);
            }
        }
    }

    // update the block position range after the position of cell i changed from p_old to pos[i]
    void block_pos_update(uint32_t i, llama_pos p_old) {
        const uint32_t b = i/block_size;

        auto & blk = blocks[b];

        assert(blk.n_used > 0);

        if (p_old != blk.pos_min && p_old != blk.pos_max) {
            blk.pos_min = std::min(blk.pos_min, pos[i]);
            blk.pos_max = std::max(blk.pos_max, pos[i]);

            return;
        }

        blk.pos_min = std::numeric_limits<llama_pos>::max();
        blk.pos_max = -1;

        for (uint32_t j = cell_begin(b); j < cell_end(b); ++j) {
            if (pos[j] == -1) {
                continue;
            }

            blk.pos_min = std::min(blk.pos_min, pos[j]);
            blk.pos_max = std::max(blk.pos_max, pos[j]);
        }
    }

    // write the seq ids of cell i into out, return the count
    int cell_seqs(uint32_t i, llama_seq_id * out) const {
        if (n_seq[i] == 0) {
            return 0;
        }

        int n = 0;

        out[n++] = seq0[i];

        const auto it = seq_extra.find(i);
        if (it != seq_extra.end()) {
            for (const auto s : it->second) {
                out[n++] = s;
            }
        }

        return n;
    }

    void set_cell(uint32_t i, const llama_kv_cells & other, uint32_t j) {
        if (pos[i] != -1) {
            rm(i);
        }

        if (other.pos[j] != -1) {
            pos_set(i, other.pos[j]);

            ext_set(i, other.ext[j]);

            llama_seq_id seqs[LLAMA_MAX_SEQ];

            const int n = other.cell_seqs(j, seqs);

            for (int k = 0; k < n; ++k) {
                seq_add(i, seqs[k]);
            }
        }

        assert(shift[i] == 0);
    }

    // per-seq stats updates. the min/max positions are exact after each add, while removals of
    //   an extremum only flag the sequence - the exact values are recomputed lazily on access

    void seq_stat_add(uint32_t i, llama_seq_id s) {
        if (seq_n[s] == 0) {
            seq_min[s] = pos[i];
            seq_max[s] = pos[i];

            seq_dirty.reset(s);
        } else if (!seq_dirty.test(s)) {
            seq_min[s] = std::min(seq_min[s], pos[i]);
            seq_max[s] = std::max(seq_max[s], pos[i]);
        }

        seq_n[s]++;
    }

    void seq_stat_rm(uint32_t i, llama_seq_id s) {
        assert(seq_n[s] > 0);

        seq_n[s]--;

        if (seq_n[s] == 0) {
            seq_min[s] = -1;
            seq_max[s] = -1;

            seq_dirty.reset(s);
        } else if (!seq_dirty.test(s) && (pos[i] == seq_min[s] || pos[i] == seq_max[s])) {
            seq_dirty.set(s);
        }
    }

    // update the per-seq position stats after the position of cell i changed from p_old to pos[i]
    void seq_stat_move(uint32_t i, llama_pos p_old) {
        llama_seq_id seqs[LLAMA_MAX_SEQ];

        const int n = cell_seqs(i, seqs);

        for (int k = 0; k < n; ++k) {
            const auto s = seqs[k];

            if (seq_dirty.test(s)) {
                continue;
            }

            if (p_old == seq_min[s] || p_old == seq_max[s]) {
                seq_dirty.set(s);
                continue;
            }

            seq_min[s] = std::min(seq_min[s], pos[i]);
            seq_max[s] = std::max(seq_max[s], pos[i]);
        }
    }

    void seq_cell_rm_all(uint32_t i) {
        llama_seq_id seqs[LLAMA_MAX_SEQ];

        const int n = cell_seqs(i, seqs);

        for (int k = 0; k < n; ++k) {
            seq_stat_rm(i, seqs[k]);
        }
    }

    // recompute the exact min/max positions of sequence s with a block-granular scan
    void seq_repair(llama_seq_id s) const {
        assert(seq_n[s] > 0);

        seq_min[s] = seq_scan_min(s);
        seq_max[s] = seq_scan_max(s);

        seq_dirty.reset(s);
    }

    // find the min position of sequence s, skipping blocks that cannot improve the result
    llama_pos seq_scan_min(llama_seq_id s) const {
        llama_pos best = std::numeric_limits<llama_pos>::max();

        for (uint32_t b = lwm; b < hwm; ++b) {
            const auto & blk = blocks[b];

            if (blk.list == LIST_FREE || blk.pos_min >= best) {
                continue;
            }

            for (uint32_t j = cell_begin(b); j < cell_end(b); ++j) {
                if (pos[j] == -1 || !seq_has(j, s)) {
                    continue;
                }

                best = std::min(best, pos[j]);
            }
        }

        return best == std::numeric_limits<llama_pos>::max() ? -1 : best;
    }

    // find the max position of sequence s, skipping blocks that cannot improve the result
    llama_pos seq_scan_max(llama_seq_id s) const {
        llama_pos best = -1;

        for (uint32_t b = hwm; b-- > lwm;) {
            const auto & blk = blocks[b];

            if (blk.list == LIST_FREE || blk.pos_max <= best) {
                continue;
            }

            for (uint32_t j = cell_end(b); j-- > cell_begin(b);) {
                if (pos[j] == -1 || !seq_has(j, s)) {
                    continue;
                }

                best = std::max(best, pos[j]);
            }
        }

        return best;
    }
};

using llama_kv_cells_vec = std::vector<llama_kv_cells>;
