#pragma once

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "grain_pipeline.h"

inline uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

struct Slice {
    int id;
    uint64_t corpus_start;  // sample index in corpus buffer
    uint32_t length;        // length of slice in samples
    SliceFeatures features;
};

struct SliceStore {
    std::mutex mtx;
    std::vector<int16_t> corpus;
    std::unordered_map<int, Slice> slices;
    int next_id = 0;

    int add_slice(const int16_t* data, uint32_t n_samples, SliceFeatures features = {}) {
        std::lock_guard<std::mutex> lk(mtx);
        uint64_t start = corpus.size();
        corpus.insert(corpus.end(), data, data + n_samples);
        int id = next_id++;
        slices[id] = Slice{id, start, n_samples, features};
        return id;
    }

    void list() {
        std::lock_guard<std::mutex> lk(mtx);
        std::fprintf(stderr, "Slices (%zu):\n", slices.size());
        for (auto& [id, s] : slices) {
            std::fprintf(stderr,
                "  id=%d start=%llu lenSamples=%u"
                "  rms=%.4f f0=%.1fHz conf=%.2f flat=%.4f rolloff=%.1fHz tonal=%.4f\n",
                id, (unsigned long long)s.corpus_start, s.length,
                s.features.rms, s.features.f0, s.features.pitch_confidence,
                s.features.spectral_flatness, s.features.rolloff_freq,
                s.features.tonal_alignment_score);
        }
    }

    bool get(int id, Slice& out) {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = slices.find(id);
        if (it == slices.end()) return false;
        out = it->second;
        return true;
    }

    bool random_id(uint32_t& rng, int& out_id) {
        std::lock_guard<std::mutex> lk(mtx);
        if (slices.empty()) return false;

        size_t k = (size_t)(rng % slices.size());
        auto it = slices.begin();
        std::advance(it, k);
        out_id = it->first;
        return true;
    }


    const int16_t* ptr(uint64_t corpusIndex) const {
        return corpus.data() + corpusIndex;
    }
};