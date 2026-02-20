#include "slicer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

struct OnsetEvent {
    int sample_index;
    float env;
    float delta;
};

std::vector<OnsetEvent> detect_onsets_stereo_s16(
    const int16_t* x, size_t n_samples, unsigned int channels, unsigned int rate,
    float sensitivity
) {
    const size_t n_frames = n_samples / channels;

    // smoothing ~10ms time constant
    const float tau_ms = 10.0f;
    const float a = std::exp(-1.0f / ((tau_ms * 0.001f) * rate));

    // thresholds and sensitivity
    const float base_level = 0.005f;
    const float base_delta = 0.00025f;

    const float level_thresh = base_level / std::sqrt(sensitivity);
    const float delta_thresh = base_delta / sensitivity;

    // prevent spamming
    const int refractory_ms = 120;
    const size_t refractory_frames = (size_t)(rate * refractory_ms / 1000);

    std::vector<OnsetEvent> events;
    events.reserve(32);

    float env = 0.0f;
    float prev_env = 0.0f;
    size_t last_onset_frame = (size_t)-refractory_frames;
    float max_d = 0.0f;

    for (size_t i = 0; i < n_frames; i++) {
        int16_t left = x[i * channels + 0];
        int16_t right = (channels > 1) ? x[i * channels + 1] : left;

        float f_left = std::abs((int)left) / 32768.0f;
        float f_right = std::abs((int)right) / 32768.0f;
        float inst = 0.5f * (f_left + f_right);

        env = a * env + (1.0f - a) * inst;

        float d_env = env - prev_env;
        prev_env = env;
        max_d = std::max(max_d, d_env);

        if (i >= last_onset_frame + refractory_frames) {
            if (env >= level_thresh && d_env >= delta_thresh) {
                events.push_back(OnsetEvent{(int)i, env, d_env});
                last_onset_frame = i;
            }
        }
    }

    return events;
}

} // anonymous namespace

std::vector<SliceRegion> OnsetSlicer::process(
    const int16_t* data, size_t n_samples,
    unsigned int channels, unsigned int rate
) {
    const size_t n_frames = n_samples / channels;
    const size_t pre_frames = (size_t)rate * pre_onset_ms / 1000;
    const size_t target_slice_frames = (size_t)rate * slice_ms / 1000;
    const size_t min_frames = (size_t)rate * min_slice_ms / 1000;

    auto events = detect_onsets_stereo_s16(data, n_samples, channels, rate, sensitivity);

    std::vector<SliceRegion> regions;
    regions.reserve(events.size());

    for (auto& o : events) {
        size_t onset_frame = (size_t)o.sample_index;

        // start frame with preroll, clamped to 0
        size_t start_frame = (onset_frame > pre_frames) ? (onset_frame - pre_frames) : 0;

        // clamp end within window
        size_t end_frame = std::min(start_frame + target_slice_frames, n_frames);
        size_t n_frames_slice = end_frame - start_frame;

        if (n_frames_slice < min_frames) continue;

        regions.push_back(SliceRegion{start_frame, n_frames_slice});
    }

    return regions;
}
