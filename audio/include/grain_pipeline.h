#pragma once

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "slicer.h"

// ---- feature data ----
// populated by the feature analysis stage; fields added as analyzers are implemented
struct SliceFeatures {
    float rms = 0.0f;
    float tonal_alignment_score = 0.0f;
    float rolloff_freq = 0.0f;
    float f0 = 0.0f;
    float pitch_confidence = 0.0f;
    float spectral_flatness = 0.0f;
};

// ---- candidate ----
// a slice region plus its audio context and extracted features,
// passed through the preprocessing → analysis → filter pipeline
struct SliceCandidate {
    SliceRegion      region;    // start_frame / length_frames (may be modified by preprocessors)
    const int16_t*   window;    // pointer to start of the audio window (not owned)
    unsigned int     channels;
    unsigned int     rate;
    SliceFeatures    features;  // populated by the analysis stage

    // normalized [-1, 1] mono signal covering exactly region.length_frames frames;
    // populated by StereoToMonoPreprocessor — analyzers should prefer this over window when non-empty
    std::vector<float> mono_buf;
};

// ---- stage interfaces ----

// preprocessing: may modify a candidate's region before feature extraction
// return false to discard the candidate entirely
class SlicePreprocessor {
public:
    virtual ~SlicePreprocessor() = default;
    virtual bool process(SliceCandidate& candidate) = 0;
};

// feature analysis: populates candidate.features from the audio data
class SliceAnalyzer {
public:
    virtual ~SliceAnalyzer() = default;
    virtual void analyze(SliceCandidate& candidate) = 0;
};

// filtering: return false to discard a candidate after feature extraction
class SliceFilter {
public:
    virtual ~SliceFilter() = default;
    virtual bool passes(const SliceCandidate& candidate) = 0;
};

// ---- preprocessor implementations ----

// downmixes stereo to mono for analysis: x_mono[i] = 0.5 * (L[i] + R[i])
// result is normalized to [-1, 1] and written to candidate.mono_buf
// no-op for mono input; always returns true (never discards)
class StereoToMonoPreprocessor : public SlicePreprocessor {
public:
    bool process(SliceCandidate& candidate) override {
        if (candidate.channels < 2) return true;

        const size_t n_frames = candidate.region.length_frames;
        const int16_t* src = candidate.window + candidate.region.start_frame * candidate.channels;

        candidate.mono_buf.resize(n_frames);
        for (size_t i = 0; i < n_frames; i++) {
            float l = src[i * 2 + 0] / 32768.0f;
            float r = src[i * 2 + 1] / 32768.0f;
            candidate.mono_buf[i] = 0.5f * (l + r);
        }

        return true;
    }
};

// ---- analyzer implementations ----

// computes root mean square (RMS) amplitude of the slice; populates candidate.features.rms
// operates on candidate.mono_buf if available, otherwise falls back to candidate.window
class RMSAnalyzer : public SliceAnalyzer {
public:
    void analyze(SliceCandidate& candidate) override {
        // only operate on mono buffer if available, otherwise fall back to raw window
        const float* x = candidate.mono_buf.empty() ? nullptr : candidate.mono_buf.data();
        const int16_t* x_raw = candidate.window + candidate.region.start_frame * candidate.channels;
        size_t n_frames = candidate.region.length_frames;

        float sum_squares = 0.0f;
        if (x) {
            for (size_t i = 0; i < n_frames; i++) {
                sum_squares += x[i] * x[i];
            }
        }
        else {
            for (size_t i = 0; i < n_frames; i++) {
                int16_t s = x_raw[i * candidate.channels]; // take left channel if stereo
                float fs = s / 32768.0f;
                sum_squares += fs * fs;
            }
        }
        candidate.features.rms = sqrtf(sum_squares / n_frames);
    }
};

// computes fundamental frequency (f0) of the slice using NSDF (McLeod Pitch Method); populates candidate.features.f0 and candidate.features.pitch_confidence
// operates on candidate.mono_buf if available, otherwise falls back to candidate.window
class F0Analyzer : public SliceAnalyzer {
public:
    void analyze(SliceCandidate& candidate) override {
        // only operate on mono buffer if available, otherwise fall back to raw window
        const float* x = candidate.mono_buf.empty() ? nullptr : candidate.mono_buf.data();
        const int16_t* x_raw = candidate.window + candidate.region.start_frame * candidate.channels;
        size_t n_frames = candidate.region.length_frames;

        // create instrument lag bounds based on expected pitch range
        float min_expected_f0 = 50.0f;
        float max_expected_f0 = 2000.0f;
        size_t min_lag = (size_t)(candidate.rate / max_expected_f0);
        size_t max_lag = (size_t)(candidate.rate / min_expected_f0);

        // normalize autocorrelation for lag in range [min_lag, max_lag]
        // find lag with maximum normalized autocorrelation
        float max_R = 0.0f;
        size_t best_lag = 0;
        
        // NSDF (normalized square difference function): n[lag] = 2*sum(s1*s2) / (sum(s1^2) + sum(s2^2))
        // bounded [-1, 1]; arithmetic denominator avoids the low-lag bias of Pearson's sqrt(e1*e2)
        float eps = 1e-8f;
        for (size_t lag = min_lag; lag <= max_lag; lag++) {
            float sum = 0.0f;
            float e1 = 0.0f, e2 = 0.0f;
            for (size_t i = 0; i < n_frames - lag; i++) {
                float s1 = x ? x[i]       : (x_raw[i * candidate.channels] / 32768.0f);
                float s2 = x ? x[i + lag] : (x_raw[(i + lag) * candidate.channels] / 32768.0f);
                sum += s1 * s2;
                e1 += s1 * s1;
                e2 += s2 * s2;
            }

            float r_norm = (2.0f * sum) / (e1 + e2 + eps);

            if (r_norm > max_R) {
                max_R = r_norm;
                best_lag = lag;
            }
        }

        // prevent division by zero on silent input
        if (best_lag == 0) {
            candidate.features.f0 = 0.0f;
            candidate.features.pitch_confidence = 0.0f;
            return;
        }

        candidate.features.f0 = candidate.rate / best_lag;
        candidate.features.pitch_confidence = max_R;
    }
};

// computes spectral rolloff frequency (rolloff_freq) at rolloff_threshold energy; populates candidate.features.rolloff_freq
// operates on candidate.mono_buf if available, otherwise falls back to candidate.window
// uses a 1024-point Hann-windowed DFT; bin resolution = rate / 1024 (~46.9 Hz at 48kHz)
class SpectralRolloffAnalyzer : public SliceAnalyzer {
public:
    float rolloff_threshold = 0.85f;

    void analyze(SliceCandidate& candidate) override {
        const float* x = candidate.mono_buf.empty() ? nullptr : candidate.mono_buf.data();
        const int16_t* x_raw = candidate.window + candidate.region.start_frame * candidate.channels;
        size_t n_frames = candidate.region.length_frames;

        static constexpr float kPi = 3.14159265358979f;
        const size_t N = 1024;
        const size_t n = std::min(n_frames, N);
        if (n < 2) { candidate.features.rolloff_freq = 0.0f; return; }

        // apply Hann window to first n samples
        std::vector<float> buf(n);
        for (size_t i = 0; i < n; i++) {
            float hann = 0.5f * (1.0f - cosf(2.0f * kPi * i / (n - 1)));
            float s = x ? x[i] : (x_raw[i * candidate.channels] / 32768.0f);
            buf[i] = s * hann;
        }

        // magnitude squared at each of the n/2 positive-frequency bins
        const size_t n_bins = n / 2;
        std::vector<float> mag2(n_bins);
        float total_energy = 0.0f;
        for (size_t k = 0; k < n_bins; k++) {
            float re = 0.0f, im = 0.0f;
            const float phase_step = 2.0f * kPi * k / n;
            for (size_t i = 0; i < n; i++) {
                re += buf[i] * cosf(phase_step * i);
                im -= buf[i] * sinf(phase_step * i);
            }
            mag2[k] = re * re + im * im;
            total_energy += mag2[k];
        }

        if (total_energy < 1e-10f) { candidate.features.rolloff_freq = 0.0f; return; }

        // find lowest bin where cumulative energy crosses rolloff_threshold
        float cumulative = 0.0f;
        const float threshold = rolloff_threshold * total_energy;
        for (size_t k = 0; k < n_bins; k++) {
            cumulative += mag2[k];
            if (cumulative >= threshold) {
                candidate.features.rolloff_freq = (float)k * candidate.rate / n;
                return;
            }
        }
        candidate.features.rolloff_freq = (float)(n_bins - 1) * candidate.rate / n;
    }
};


// computes spectral flatness of the slice; populates candidate.features.spectral_flatness
// operates on candidate.mono_buf if available, otherwise falls back to candidate.window
class SpectralFlatnessAnalyzer : public SliceAnalyzer {
public:
    void analyze(SliceCandidate& candidate) override {
        // only operate on mono buffer if available, otherwise fall back to raw window
        const float* x = candidate.mono_buf.empty() ? nullptr : candidate.mono_buf.data();
        const int16_t* x_raw = candidate.window + candidate.region.start_frame * candidate.channels;
        size_t n_frames = candidate.region.length_frames;



    }

};