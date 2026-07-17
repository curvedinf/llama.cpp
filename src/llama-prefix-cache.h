#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

// block-hash prefix cache registry (vLLM APC-style)
//
// maps chained hashes of full 32-token blocks to physical KV block locations
// {(stream, block)}. the hash of a block chains in the hash of the previous
// block, so a run of consecutive matches guarantees that the whole prefix is
// token-identical. blocks are registered when they become full and removed when
// their cells are modified or reclaimed.
//
// the registry is pure bookkeeping: the owner (llama_kv_cache) pins/unpins the
// physical blocks (llama_kv_cells::block_t::ref) around insert/remove calls,
// exactly one pin per registered location.
//
// optionally, an entry can carry an opaque state snapshot (e.g. the recurrent
// state of a hybrid model at the end of the block's prefix). a match with
// need_state only reuses prefixes whose last block has a snapshot attached.
class llama_prefix_cache {
public:
    static constexpr uint32_t block_size = 32;

    struct loc_t {
        uint32_t  stream;
        uint32_t  block;
        llama_pos p0; // position of the first token of the block (multiple of block_size)

        bool operator==(const loc_t & other) const {
            return stream == other.stream && block == other.block && p0 == other.p0;
        }
    };

    // read-only view of a state snapshot attached to an entry
    struct state_view {
        const uint8_t * ptr = nullptr;
        size_t          n   = 0;

        const uint8_t * data() const { return ptr; }
        size_t          size() const { return n; }
    };

    struct entry {
        uint64_t hash = 0;

        llama_token tokens[block_size] = {}; // kept for exact match verification

        std::vector<loc_t> locs;

        uint32_t n_pin = 0; // pins held by outstanding match handles

        bool has_state = false;
        std::vector<uint8_t> state; // state snapshot at the end of this block's prefix

        // alternative state storage: a backend host buffer (e.g. Vulkan pinned memory) that
        //   an async device-to-host copy may still be filling. the copy is ordered before any
        //   later backend work on the same device, so restoring with ggml_backend_tensor_set
        //   after synchronizing the backend is safe. never read the bytes from the CPU without
        //   synchronizing the backend first.
        ggml_backend_buffer_ptr state_buf;

        state_view sv; // view of state / state_buf, valid while the entry is alive

        std::list<uint64_t>::iterator lru_it;
    };

    static constexpr uint64_t seed = 0xCBF29CE484222325ULL;

    // xxhash64-style chained block hash
    static uint64_t hash_block(uint64_t prev, const llama_token * tokens) {
        uint64_t h = prev + 0x9E3779B185EBCA87ULL;

        for (uint32_t i = 0; i < block_size; ++i) {
            uint64_t k = (uint64_t) (uint32_t) tokens[i]*0xC2B2AE3D27D4EB4FULL;
            k = (k << 31) | (k >> 33);
            k *= 0x9E3779B185EBCA87ULL;
            h ^= k;
            h = (h << 27) | (h >> 37);
            h = h*5 + 0x52DCE729;
        }

        h ^= h >> 33;
        h *= 0xFF51AFD7ED558CCDULL;
        h ^= h >> 33;
        h *= 0xC4CEB9FE1A85EC53ULL;
        h ^= h >> 33;

        return h;
    }

    bool empty() const {
        return map.empty();
    }

    size_t size() const {
        return map.size();
    }

    // find an entry by hash (nullptr if not present)
    const entry * find(uint64_t hash) const {
        const auto it = map.find(hash);
        return it == map.end() ? nullptr : &it->second;
    }

    // find an entry by hash and exact tokens (nullptr if not present or tokens differ)
    const entry * find(uint64_t hash, const llama_token * tokens) const {
        const entry * e = find(hash);
        if (e && tokens_equal(e->tokens, tokens)) {
            return e;
        }
        return nullptr;
    }

    // register a block location
    // returns {entry, true} if the location was newly added (the caller pins the block)
    // returns {nullptr, false} on a hash collision with different tokens (insert dropped)
    std::pair<entry *, bool> insert(uint64_t hash, const llama_token * tokens, const loc_t & loc) {
        auto it = map.find(hash);

        if (it == map.end()) {
            entry e;

            e.hash = hash;
            memcpy(e.tokens, tokens, sizeof(e.tokens));
            e.locs.push_back(loc);

            lru.push_back(hash);
            e.lru_it = std::prev(lru.end());

            const auto res = map.emplace(hash, std::move(e));

            loc_index[loc_key(loc)] = hash;

            return { &res.first->second, true };
        }

        entry & e = it->second;

        if (!tokens_equal(e.tokens, tokens)) {
            return { nullptr, false };
        }

        if (std::find(e.locs.begin(), e.locs.end(), loc) == e.locs.end()) {
            e.locs.push_back(loc);
            loc_index[loc_key(loc)] = hash;
            touch(e);
            return { &e, true };
        }

        touch(e);

        return { &e, false };
    }

    // attach a state snapshot to the entry at the end of a prefix
    void set_state(uint64_t hash, const llama_token * tokens, std::vector<uint8_t> && state) {
        auto it = map.find(hash);
        if (it == map.end() || !tokens_equal(it->second.tokens, tokens)) {
            return;
        }

        it->second.state     = std::move(state);
        it->second.state_buf = nullptr;
        it->second.has_state = true;
        it->second.sv        = { it->second.state.data(), it->second.state.size() };

        touch(it->second);
    }

    // attach a state snapshot held by a backend host buffer (see entry::state_buf)
    void set_state(uint64_t hash, const llama_token * tokens, ggml_backend_buffer_ptr && buf, size_t size) {
        auto it = map.find(hash);
        if (it == map.end() || !tokens_equal(it->second.tokens, tokens)) {
            return;
        }

        it->second.state.clear();
        it->second.state_buf = std::move(buf);
        it->second.has_state = true;
        it->second.sv        = { (const uint8_t *) ggml_backend_buffer_get_base(it->second.state_buf.get()), size };

        touch(it->second);
    }

    // remove a single location (the caller unpins the block)
    // return true if the location was registered
    bool remove_loc(const loc_t & loc) {
        const auto it = loc_index.find(loc_key(loc));
        if (it == loc_index.end()) {
            return false;
        }

        const uint64_t hash = it->second;

        loc_index.erase(it);

        auto & e = map.at(hash);

        const auto il = std::find(e.locs.begin(), e.locs.end(), loc);
        if (il != e.locs.end()) {
            e.locs.erase(il);
        }

        if (e.locs.empty()) {
            erase_entry(hash);
        }

        return true;
    }

    // remove all locations of the given stream with p0 in [p0, p1), appending them to out
    void remove_range(uint32_t stream, llama_pos p0, llama_pos p1, std::vector<loc_t> & out) {
        const uint64_t k0 = loc_key({ stream, 0, 0 });
        const uint64_t k1 = loc_key({ stream + 1, 0, 0 });

        for (auto it = loc_index.lower_bound(k0); it != loc_index.end() && it->first < k1; ) {
            const loc_t loc = loc_from_key(it->first, map.at(it->second));

            if (loc.p0 >= p0 && loc.p0 < p1) {
                out.push_back(loc);
                it = erase_loc(it);
            } else {
                ++it;
            }
        }
    }

    // remove all locations of the given stream, appending them to out
    void remove_stream(uint32_t stream, std::vector<loc_t> & out) {
        const uint64_t k0 = loc_key({ stream, 0, 0 });
        const uint64_t k1 = loc_key({ stream + 1, 0, 0 });

        for (auto it = loc_index.lower_bound(k0); it != loc_index.end() && it->first < k1; ) {
            out.push_back(loc_from_key(it->first, map.at(it->second)));
            it = erase_loc(it);
        }
    }

    // collect up to n_max eviction candidates (unpinned locations of the given stream,
    //   oldest first) without removing them
    void evict_candidates(uint32_t stream, uint32_t n_max, std::vector<loc_t> & out) const {
        for (const uint64_t hash : lru) {
            if (out.size() >= n_max) {
                return;
            }

            const auto & e = map.at(hash);

            if (e.n_pin > 0) {
                continue;
            }

            for (const auto & loc : e.locs) {
                if (loc.stream == stream) {
                    out.push_back(loc);
                }
            }
        }
    }

    // find the longest fully-cached block-aligned prefix of tokens[0, n_tokens)
    // if need_state, the prefix must end with an entry that has a state snapshot
    // pins the matched entries and returns a handle via out_handle (0 = no match)
    uint32_t match(const llama_token * tokens, int32_t n_tokens, bool need_state, uint64_t & out_handle) {
        out_handle = 0;

        if (n_tokens < (int32_t) block_size) {
            return 0;
        }

        const int32_t n_blocks = n_tokens/block_size;

        uint64_t h = seed;

        uint32_t best = 0;

        std::vector<uint64_t> hashes;
        hashes.reserve(n_blocks);

        for (int32_t k = 0; k < n_blocks; ++k) {
            h = hash_block(h, tokens + k*block_size);

            const entry * e = find(h, tokens + k*block_size);
            if (!e) {
                break;
            }

            touch(const_cast<entry &>(*e));

            hashes.push_back(h);

            if (!need_state || e->has_state) {
                best = k + 1;
            }
        }

        if (best == 0) {
            return 0;
        }

        hashes.resize(best);

        for (const uint64_t hash : hashes) {
            map.at(hash).n_pin++;
        }

        out_handle = ++handle_counter;

        handles.emplace(out_handle, std::move(hashes));

        return best*block_size;
    }

    // the block hashes of a matched prefix (nullptr if the handle is invalid)
    const std::vector<uint64_t> * handle_hashes(uint64_t handle) const {
        const auto it = handles.find(handle);
        return it == handles.end() ? nullptr : &it->second;
    }

    // the state snapshot at the end of a matched prefix (nullptr if none or invalid handle)
    const state_view * handle_state(uint64_t handle) const {
        const auto * hashes = handle_hashes(handle);
        if (!hashes || hashes->empty()) {
            return nullptr;
        }

        const entry * e = find(hashes->back());
        if (!e || !e->has_state) {
            return nullptr;
        }

        return &e->sv;
    }

    // unpin and forget a handle
    void release(uint64_t handle) {
        const auto it = handles.find(handle);
        if (it == handles.end()) {
            return;
        }

        for (const uint64_t hash : it->second) {
            const auto ie = map.find(hash);
            if (ie != map.end() && ie->second.n_pin > 0) {
                ie->second.n_pin--;
            }
        }

        handles.erase(it);
    }

    void clear() {
        map.clear();
        lru.clear();
        loc_index.clear();
        handles.clear();
    }

private:
    static bool tokens_equal(const llama_token * a, const llama_token * b) {
        return memcmp(a, b, block_size*sizeof(llama_token)) == 0;
    }

    static uint64_t loc_key(const loc_t & loc) {
        return ((uint64_t) loc.stream << 32) | loc.block;
    }

    // recover a full loc from its key; p0 is taken from the matching entry location
    static loc_t loc_from_key(uint64_t key, const entry & e) {
        const uint32_t stream = (uint32_t) (key >> 32);
        const uint32_t block  = (uint32_t) key;

        for (const auto & loc : e.locs) {
            if (loc.stream == stream && loc.block == block) {
                return loc;
            }
        }

        return { stream, block, -1 };
    }

    void touch(entry & e) {
        lru.splice(lru.end(), lru, e.lru_it);
    }

    void erase_entry(uint64_t hash) {
        auto & e = map.at(hash);
        lru.erase(e.lru_it);
        map.erase(hash);
    }

    // remove the location referenced by the given loc_index iterator
    // return the iterator following it
    std::map<uint64_t, uint64_t>::iterator erase_loc(std::map<uint64_t, uint64_t>::iterator it) {
        const uint64_t key  = it->first;
        const uint64_t hash = it->second;

        auto res = loc_index.erase(it);

        auto & e = map.at(hash);

        const loc_t loc = loc_from_key(key, e);

        const auto il = std::find(e.locs.begin(), e.locs.end(), loc);
        if (il != e.locs.end()) {
            e.locs.erase(il);
        }

        if (e.locs.empty()) {
            erase_entry(hash);
        }

        return res;
    }

    std::unordered_map<uint64_t, entry> map;

    // hashes, least-recently used first
    std::list<uint64_t> lru;

    // (stream, block) -> hash
    std::map<uint64_t, uint64_t> loc_index;

    // match handle -> pinned block hashes
    std::unordered_map<uint64_t, std::vector<uint64_t>> handles;

    uint64_t handle_counter = 0;
};

// per-sequence token chain tracker used to compute block hashes at registration time
// tracks only prefixes that start at position 0 and advance contiguously - anything
// else (mid-sequence starts, non-block-aligned rollbacks) kills the chain
struct llama_prefix_chain {
    std::vector<uint64_t> hashes; // hash of each completed block

    llama_token partial[llama_prefix_cache::block_size] = {};
    uint32_t    n_partial = 0;

    llama_pos next_pos = 0; // next expected position, -1 = dead

    // last completed block
    uint64_t   last_hash = 0;
    llama_token last_tokens[llama_prefix_cache::block_size] = {};
    llama_pos  last_p0 = -1;

    void reset() {
        hashes.clear();
        n_partial = 0;
        next_pos  = 0;
        last_hash = 0;
        last_p0   = -1;
    }

    void kill() {
        next_pos = -1;
    }

    // remove everything at positions >= p0 (sequence rollback)
    void truncate(llama_pos p0) {
        if (next_pos < 0 || p0 < 0 || p0 >= next_pos) {
            return;
        }

        if (p0%llama_prefix_cache::block_size == 0) {
            hashes.resize(p0/llama_prefix_cache::block_size);
            n_partial = 0;
            next_pos  = p0;
        } else {
            // mid-block truncation - cannot re-chain without the dropped tokens
            kill();
        }
    }

    // feed one token, return true when a full block was just completed
    bool feed(llama_pos p, llama_token t) {
        if (p != next_pos) {
            if (p == 0) {
                reset();
            } else {
                kill();
                return false;
            }
        }

        partial[n_partial++] = t;
        next_pos = p + 1;

        if (n_partial == llama_prefix_cache::block_size) {
            last_hash = llama_prefix_cache::hash_block(hashes.empty() ? llama_prefix_cache::seed : hashes.back(), partial);
            last_p0   = p - (llama_prefix_cache::block_size - 1);

            memcpy(last_tokens, partial, sizeof(partial));

            hashes.push_back(last_hash);

            n_partial = 0;

            return true;
        }

        return false;
    }
};
