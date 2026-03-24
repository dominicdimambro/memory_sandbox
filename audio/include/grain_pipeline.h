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
    float gain = 1.0f;  // set by GainNormalizerAnalyzer; applied at playback
    // normalized per-pitch-class energy relative to A=440, root-independent;
    // index 0=A 1=Bb ... 11=Ab; used to recompute tonal_alignment_score on root/scale change
    float chroma_energy[12] = {};
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

    // Hann-windowed FFT power spectrum (kFFTSize/2 positive-frequency bins);
    // populated lazily by ensure_power_spectrum() — shared across all spectral analyzers
    std::vector<float> power_spectrum;
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

// ---- shared spectrum helper ----

static constexpr size_t kFFTSize = 1024;  // must be a power of 2
static constexpr float  kPi      = 3.14159265358979f;

// Computes and caches a Hann-windowed radix-2 FFT power spectrum in candidate.power_spectrum.
// Returns a reference to the cached result (no-op on subsequent calls).
// kFFTSize/2 bins; bin k -> frequency k * rate / kFFTSize Hz.
// Returns an empty vector if the slice has fewer than 2 frames.
inline const std::vector<float>& ensure_power_spectrum(SliceCandidate& candidate) {
    if (!candidate.power_spectrum.empty()) return candidate.power_spectrum;

    const float* x = candidate.mono_buf.empty() ? nullptr : candidate.mono_buf.data();
    const int16_t* x_raw = candidate.window + candidate.region.start_frame * candidate.channels;
    const size_t n = std::min(candidate.region.length_frames, kFFTSize);

    if (n < 2) return candidate.power_spectrum;  // leave empty

    // interleaved real/imag, zero-initialized (zero-pads to kFFTSize)
    std::vector<float> buf(kFFTSize * 2, 0.0f);
    for (size_t i = 0; i < n; i++) {
        float hann = 0.5f * (1.0f - cosf(2.0f * kPi * i / (n - 1)));
        float s = x ? x[i] : (x_raw[i * candidate.channels] / 32768.0f);
        buf[i * 2] = s * hann;
    }

    // bit-reversal permutation
    for (size_t i = 1, j = 0; i < kFFTSize; i++) {
        size_t bit = kFFTSize >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(buf[i * 2],     buf[j * 2]);
            std::swap(buf[i * 2 + 1], buf[j * 2 + 1]);
        }
    }

    // Cooley-Tukey radix-2 DIT butterfly stages
    for (size_t len = 2; len <= kFFTSize; len <<= 1) {
        float ang      = -2.0f * kPi / len;
        float wre_step = cosf(ang), wim_step = sinf(ang);
        for (size_t i = 0; i < kFFTSize; i += len) {
            float wre = 1.0f, wim = 0.0f;
            for (size_t k = 0; k < len / 2; k++) {
                size_t u = (i + k) * 2, v = (i + k + len / 2) * 2;
                float t_re = wre * buf[v] - wim * buf[v + 1];
                float t_im = wre * buf[v + 1] + wim * buf[v];
                buf[v]     = buf[u]     - t_re;
                buf[v + 1] = buf[u + 1] - t_im;
                buf[u]     += t_re;
                buf[u + 1] += t_im;
                float new_wre = wre * wre_step - wim * wim_step;
                wim = wre * wim_step + wim * wre_step;
                wre = new_wre;
            }
        }
    }

    // magnitude squared for positive-frequency bins
    const size_t n_bins = kFFTSize / 2;
    candidate.power_spectrum.resize(n_bins);
    for (size_t k = 0; k < n_bins; k++) {
        float re = buf[k * 2], im = buf[k * 2 + 1];
        candidate.power_spectrum[k] = re * re + im * im;
    }

    return candidate.power_spectrum;
}

// ---- analyzer implementations ----

// computes root mean square (RMS) amplitude of the slice; populates candidate.features.rms
// operates on candidate.mono_buf if available, otherwise falls back to candidate.window
class RMSAnalyzer : public SliceAnalyzer {
public:
    void analyze(SliceCandidate& candidate) override {
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
        const float* x = candidate.mono_buf.empty() ? nullptr : candidate.mono_buf.data();
        const int16_t* x_raw = candidate.window + candidate.region.start_frame * candidate.channels;
        // cap to 2048 frames — pitch detection doesn't benefit from longer windows
        // and the NSDF is O(n_frames * lag_range)
        size_t n_frames = std::min(candidate.region.length_frames, (size_t)2048);

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

// computes spectral rolloff frequency at rolloff_threshold energy; populates candidate.features.rolloff_freq
// bin resolution = rate / kFFTSize (~46.9 Hz at 48kHz)
class SpectralRolloffAnalyzer : public SliceAnalyzer {
public:
    float rolloff_threshold = 0.85f;

    void analyze(SliceCandidate& candidate) override {
        const auto& mag2 = ensure_power_spectrum(candidate);
        if (mag2.empty()) { candidate.features.rolloff_freq = 0.0f; return; }

        float total_energy = 0.0f;
        for (float p : mag2) total_energy += p;
        if (total_energy < 1e-10f) { candidate.features.rolloff_freq = 0.0f; return; }

        float cumulative = 0.0f;
        const float threshold = rolloff_threshold * total_energy;
        for (size_t k = 0; k < mag2.size(); k++) {
            cumulative += mag2[k];
            if (cumulative >= threshold) {
                candidate.features.rolloff_freq = (float)k * candidate.rate / kFFTSize;
                return;
            }
        }
        candidate.features.rolloff_freq = (float)(mag2.size() - 1) * candidate.rate / kFFTSize;
    }
};

// computes spectral flatness (Wiener entropy) of the slice; populates candidate.features.spectral_flatness
// result is in [0, 1]: 0 = pure tone, 1 = white noise
class SpectralFlatnessAnalyzer : public SliceAnalyzer {
public:
    void analyze(SliceCandidate& candidate) override {
        const auto& mag2 = ensure_power_spectrum(candidate);
        if (mag2.empty()) { candidate.features.spectral_flatness = 0.0f; return; }

        float log_sum = 0.0f;
        float arith_sum = 0.0f;
        const float eps = 1e-10f;
        for (float p : mag2) {
            log_sum   += logf(p + eps);
            arith_sum += p;
        }

        float arith_mean = arith_sum / mag2.size();
        if (arith_mean < eps) { candidate.features.spectral_flatness = 0.0f; return; }

        float geo_mean = expf(log_sum / mag2.size());
        candidate.features.spectral_flatness = geo_mean / arith_mean;
    }
};

enum class ScaleType { Major, Minor };

// computes tonal alignment score relative to a root note and scale
// analyze() builds a root-independent normalized chroma vector (chroma_energy[12])
// then projects it onto the current root/scale via tonal_score_from_chroma()
// calling tonal_score_from_chroma() on stored slices after a root/scale change updates
// tonal_alignment_score without re-running the FFT
class TonalAlignmentAnalyzer : public SliceAnalyzer {
public:
    int       root_idx        = 0;              // 0=A 1=Bb ... 11=Ab; set from engine atomic each epoch
    ScaleType scale_type      = ScaleType::Major;
    float     low_freq_cutoff = 80.0f;

    void analyze(SliceCandidate& candidate) override {
        const auto& mag2 = ensure_power_spectrum(candidate);
        if (mag2.empty()) { candidate.features.tonal_alignment_score = 0.0f; return; }

        // accumulate energy per absolute pitch class relative to A=440 (root-independent)
        float raw_chroma[12] = {};
        float total_energy   = 0.0f;

        for (size_t k = 0; k < mag2.size(); k++) {
            float bin_freq = (float)k * candidate.rate / kFFTSize;
            if (bin_freq < low_freq_cutoff) continue;
            if (mag2[k] < 1e-12f) continue;

            float raw_st = 12.0f * log2f(bin_freq / 440.0f);  // relative to A=440, not root
            int sc = (int)roundf(raw_st) % 12;
            if (sc < 0) sc += 12;

            raw_chroma[sc] += mag2[k];
            total_energy   += mag2[k];
        }

        if (total_energy < 1e-10f) { candidate.features.tonal_alignment_score = 0.0f; return; }

        for (int i = 0; i < 12; i++)
            candidate.features.chroma_energy[i] = raw_chroma[i] / total_energy;

        candidate.features.tonal_alignment_score =
            tonal_score_from_chroma(candidate.features.chroma_energy, root_idx, scale_type);
    }

    // project a stored chroma vector onto a root/scale; O(12) — call on all slices after root change
    static float tonal_score_from_chroma(const float* chroma, int root_idx, ScaleType scale) {
        const float* weights = (scale == ScaleType::Major) ? kMajorWeights : kMinorWeights;
        float score = 0.0f;
        for (int i = 0; i < 12; i++) {
            int semitone_from_root = (i - root_idx + 12) % 12;
            score += chroma[i] * weights[semitone_from_root];
        }
        return score;
    }

private:
    // semitone weights [0..11]: unison, m2, M2, m3, M3, P4, tritone, P5, m6, M6, m7, M7
    static constexpr float kMajorWeights[12] = { +1.0f, -0.5f, -0.1f, -0.2f, +0.6f, +0.4f, -0.8f, +0.9f, -0.2f, +0.5f, -0.3f, +0.2f };
    static constexpr float kMinorWeights[12] = { +1.0f, -0.5f, -0.1f, +0.6f, -0.2f, +0.4f, -0.8f, +0.9f, +0.4f, -0.2f, +0.3f, -0.4f };
};

// Computes a playback gain to normalise each grain toward target_rms.
// Must be registered after RMSAnalyzer so features.rms is already populated.
// gain is capped at max_gain to avoid over-amplifying near-silent grains.
class GainNormalizerAnalyzer : public SliceAnalyzer {
public:
    float target_rms = 0.09f;  // target loudness (~median of observed corpus)
    float max_gain   = 6.0f;   // hard ceiling: never boost more than 6×

    void analyze(SliceCandidate& candidate) override {
        float rms = candidate.features.rms;
        if (rms < 1e-6f) {
            candidate.features.gain = 1.0f;
            return;
        }
        float g = target_rms / rms;
        if (g > max_gain) g = max_gain;
        candidate.features.gain = g;
    }
};
