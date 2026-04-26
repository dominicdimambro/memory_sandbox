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

    // returns a random pick from the k nearest grains in normalized feature space
    // normalization matches grain_visualizer snapshot(): tonal*0.5, rolloff log-scale, rms linear
    bool closest_k_id(float q_tonal, float q_rolloff, float q_rms,
                      int k, uint32_t& rng, int& out_id) {
        std::lock_guard<std::mutex> lk(mtx);
        if (slices.empty()) return false;
        if (k < 1) k = 1;
        static constexpr int kMaxK = 10;
        if (k > kMaxK) k = kMaxK;

        auto norm_tonal = [](float t) { return t * 0.5f; };
        auto norm_rolloff = [](float r) {
            float v = (std::log(r < 47.f ? 47.f : r) - 3.850f) / (9.393f - 3.850f) - 0.5f;
            return v < -0.5f ? -0.5f : v > 0.5f ? 0.5f : v;
        };
        auto norm_rms = [](float r) {
            float v = std::sqrt(r / 0.12f);
            return (v > 1.f ? 1.f : v) - 0.5f;
        };

        float qx = norm_tonal(q_tonal);
        float qy = norm_rolloff(q_rolloff);
        float qz = norm_rms(q_rms);

        int   best_ids[kMaxK]  = {};
        float best_dist[kMaxK] = {};
        int   found = 0;

        for (auto& [id, slice] : slices) {
            const auto& f = slice.features;
            float dx = norm_tonal(f.tonal_alignment_score) - qx;
            float dy = norm_rolloff(f.rolloff_freq) - qy;
            float dz = norm_rms(f.rms) - qz;
            float d2 = dx*dx + dy*dy + dz*dz;

            if (found < k) {
                best_ids[found] = id;
                best_dist[found++] = d2;
            } else {
                int wi = 0;
                for (int i = 1; i < k; i++)
                    if (best_dist[i] > best_dist[wi]) wi = i;
                if (d2 < best_dist[wi]) {
                    best_ids[wi] = id;
                    best_dist[wi] = d2;
                }
            }
        }

        if (found == 0) return false;
        out_id = best_ids[rng % (uint32_t)found];
        return true;
    }

    // Like closest_k_id but query is already in projected space (matches snapshot() coords).
    // out_radius = distance to the k-th nearest grain — used by visualizer to draw the sphere.
    bool closest_k_id_xyz(float qx, float qy, float qz,
                          int k, uint32_t& rng, int& out_id, float& out_radius) {
        std::lock_guard<std::mutex> lk(mtx);
        if (slices.empty()) return false;
        if (k < 1) k = 1;
        static constexpr int kMaxK = 10;
        if (k > kMaxK) k = kMaxK;

        auto norm_tonal = [](float t) { return t * 0.5f; };
        auto norm_rolloff = [](float r) {
            float v = (std::log(r < 47.f ? 47.f : r) - 3.850f) / (9.393f - 3.850f) - 0.5f;
            return v < -0.5f ? -0.5f : v > 0.5f ? 0.5f : v;
        };
        auto norm_rms = [](float r) {
            float v = std::sqrt(r / 0.12f);
            return (v > 1.f ? 1.f : v) - 0.5f;
        };

        int   best_ids[kMaxK]  = {};
        float best_dist[kMaxK] = {};
        int   found = 0;

        for (auto& [id, slice] : slices) {
            const auto& f = slice.features;
            float dx = norm_tonal(f.tonal_alignment_score) - qx;
            float dy = norm_rolloff(f.rolloff_freq) - qy;
            float dz = norm_rms(f.rms) - qz;
            float d2 = dx*dx + dy*dy + dz*dz;

            if (found < k) {
                best_ids[found] = id;
                best_dist[found++] = d2;
            } else {
                int wi = 0;
                for (int i = 1; i < k; i++)
                    if (best_dist[i] > best_dist[wi]) wi = i;
                if (d2 < best_dist[wi]) {
                    best_ids[wi] = id;
                    best_dist[wi] = d2;
                }
            }
        }

        if (found == 0) return false;
        out_id = best_ids[rng % (uint32_t)found];
        float max_d2 = 0.0f;
        for (int i = 0; i < found; i++) if (best_dist[i] > max_d2) max_d2 = best_dist[i];
        out_radius = std::sqrt(max_d2);
        return true;
    }


    const int16_t* ptr(uint64_t corpusIndex) const {
        return corpus.data() + corpusIndex;
    }
};