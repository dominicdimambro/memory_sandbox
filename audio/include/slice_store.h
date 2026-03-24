#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

        if (slices.size() < 2) return;

        // per-feature stats + ASCII histogram
        static constexpr int kBuckets = 10;
        struct FeatStat {
            const char* name;
            float min, max, sum;
            int buckets[kBuckets];
        };

        FeatStat stats[] = {
            {"rms    ", 1e9f, -1e9f, 0.f, {}},
            {"flat   ", 1e9f, -1e9f, 0.f, {}},
            {"tonal  ", 1e9f, -1e9f, 0.f, {}},
            {"rolloff", 1e9f, -1e9f, 0.f, {}},
            {"f0     ", 1e9f, -1e9f, 0.f, {}},
        };

        // first pass: min/max/sum
        for (auto& [id, s] : slices) {
            float vals[] = {
                s.features.rms,
                s.features.spectral_flatness,
                s.features.tonal_alignment_score,
                s.features.rolloff_freq,
                s.features.f0,
            };
            for (int i = 0; i < 5; i++) {
                if (vals[i] < stats[i].min) stats[i].min = vals[i];
                if (vals[i] > stats[i].max) stats[i].max = vals[i];
                stats[i].sum += vals[i];
            }
        }

        // second pass: fill buckets
        for (auto& [id, s] : slices) {
            float vals[] = {
                s.features.rms,
                s.features.spectral_flatness,
                s.features.tonal_alignment_score,
                s.features.rolloff_freq,
                s.features.f0,
            };
            for (int i = 0; i < 5; i++) {
                float range = stats[i].max - stats[i].min;
                if (range < 1e-12f) { stats[i].buckets[0]++; continue; }
                int b = (int)((vals[i] - stats[i].min) / range * kBuckets);
                if (b >= kBuckets) b = kBuckets - 1;
                stats[i].buckets[b]++;
            }
        }

        std::fprintf(stderr, "\n[feature stats]  (n=%zu)\n", slices.size());
        for (int i = 0; i < 5; i++) {
            float mean = stats[i].sum / (float)slices.size();
            std::fprintf(stderr, "  %s  min=%-9.4f max=%-9.4f mean=%-9.4f  [",
                stats[i].name, stats[i].min, stats[i].max, mean);
            int peak = *std::max_element(stats[i].buckets, stats[i].buckets + kBuckets);
            for (int b = 0; b < kBuckets; b++) {
                int bar = (peak > 0) ? (stats[i].buckets[b] * 8 / peak) : 0;
                const char* blocks[] = {" ","▁","▂","▃","▄","▅","▆","▇","█"};
                std::fprintf(stderr, "%s", blocks[bar]);
            }
            std::fprintf(stderr, "]\n");
        }
        std::fprintf(stderr, "\n");
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