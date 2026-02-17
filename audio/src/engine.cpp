#include <alsa/asoundlib.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "alsa_util.h"
#include "ringbuffer.h"
#include "slice_store.h"
#include "slicer.h"
#include "terminal.h"

static std::atomic<bool> g_run{true};

int main(int argc, char** argv) {

    // capture and playback devices with default fallbacks
    const char* cap_dev = (argc > 1) ? argv[1] : "plughw:2,0";
    const char* pb_dev  = (argc > 2) ? argv[2] : "plughw:2,0";

    // hw parameters
    const unsigned int rate = 48000;
    const unsigned int channels = 2;
    const snd_pcm_uframes_t period_frames = 256;
    const snd_pcm_uframes_t buffer_frames = period_frames * 16;

    const size_t samples_per_period = (size_t)period_frames * channels;

    // ring buffer: 1 second of interleaved stereo
    const size_t rb_capacity_seconds = 1;
    const size_t rb_capacity_samples = (size_t)rate * channels * rb_capacity_seconds;
    RingBuffer<int16_t> rb(rb_capacity_samples);
    std::mutex rb_mtx;

    // cross-thread control flags
    std::atomic<bool> record_enabled{false};
    std::atomic<bool> play_enabled{false};

    // open ALSA devices
    snd_pcm_t* cap = nullptr;
    snd_pcm_t* pb  = nullptr;

    std::fprintf(stderr, "opening capture=%s playback=%s\n", cap_dev, pb_dev);
    if (!open_pcm(&cap, cap_dev, SND_PCM_STREAM_CAPTURE, rate, channels, period_frames, buffer_frames)) return 1;
    if (!open_pcm(&pb, pb_dev, SND_PCM_STREAM_PLAYBACK, rate, channels, period_frames, buffer_frames)) return 1;

    std::fprintf(stderr,
        "\nControls:\n"
        "   r = start recording into ringbuffer\n"
        "   s = stop recording\n"
        "   m = toggle monitor\n"
        "   e = toggle explicit/debug mode\n"
        "   l = list slice store\n"
        "   1-6 = grain interval (1s/200ms/100ms/50ms/30ms/15ms)\n"
        "   , / . = decrease / increase grain length (25ms steps, 20-2000ms)\n"
        "   [ / ] = decrease / increase onset sensitivity\n"
        "   q = quit\n"
    );

    TermRawMode raw;
    raw.enable();

    // slicer: swappable algorithm (default: onset detection)
    auto slicer = std::make_unique<OnsetSlicer>();

    SliceStore store;

    // slicer control + state
    std::atomic<uint64_t> rb_epoch{0};
    std::vector<int16_t> one_sec_window(rb_capacity_samples);

    // slicer controls
    std::atomic<float> slicer_sensitivity{1.0f};

    // playback controls
    std::atomic<int> grain_interval_ms{1000};
    std::atomic<int> grain_length_ms{200};
    std::atomic<bool> explicit_mode{false};
    std::atomic<int> current_grain_id{-1};

    // ---- capture thread ----
    std::thread cap_thread([&] {
        std::vector<int16_t> in(samples_per_period);

        uint64_t samples_since_epoch = 0;
        bool was_recording = false;

        while (g_run.load(std::memory_order_relaxed)) {
            snd_pcm_sframes_t n = snd_pcm_readi(cap, in.data(), period_frames);

            if (n < 0) { recover_if_xrun(cap, (int)n, "capture"); continue; }
            if (n == 0) continue;

            bool rec = record_enabled.load(std::memory_order_relaxed);

            // detect recording start: reset epoch accounting
            if (rec && !was_recording) {
                samples_since_epoch = 0;
                rb_epoch.store(0, std::memory_order_release);
            }
            was_recording = rec;

            if (!rec) continue;

            const size_t nsamp = (size_t)n * channels;
            {
                std::lock_guard<std::mutex> lk(rb_mtx);
                rb.write_overwrite(in.data(), nsamp);
            }

            // epoch accounting
            samples_since_epoch += nsamp;
            while (samples_since_epoch >= rb_capacity_samples) {
                samples_since_epoch -= rb_capacity_samples;
                rb_epoch.fetch_add(1, std::memory_order_release);
            }
        }
    });

    // ---- slicer thread ----
    std::thread slicer_thread([&] {
        uint64_t last_epoch = 0;

        while (g_run.load(std::memory_order_relaxed)) {

            uint64_t e = rb_epoch.load(std::memory_order_acquire);

            if (e != last_epoch) {
                size_t got = 0;
                {
                    std::lock_guard<std::mutex> lk(rb_mtx);
                    got = rb.copy_latest(one_sec_window.data(), rb_capacity_samples, 0);
                }

                if (got == rb_capacity_samples) {
                    slicer->sensitivity = slicer_sensitivity.load(std::memory_order_relaxed);
                    auto regions = slicer->process(
                        one_sec_window.data(), got, channels, rate
                    );

                    for (auto& region : regions) {
                        size_t start_sample = region.start_frame * channels;
                        size_t n_samples_slice = region.length_frames * channels;

                        int id = store.add_slice(
                            one_sec_window.data() + start_sample,
                            (uint32_t)n_samples_slice
                        );

                        std::fprintf(stderr, "[slice] id=%d start=%.1fms len=%zu samples\n",
                            id, 1000.0 * region.start_frame / rate, n_samples_slice);
                    }
                }

                last_epoch = e;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    });

    // ---- playback thread ----
    std::thread pb_thread([&] {
        std::vector<int16_t> out(samples_per_period, 0);

        Slice cur{};
        bool cur_valid = false;
        uint32_t pos = 0;
        int grain_jitter_ms = 0;
        uint32_t rng = 0x12345678u;

        auto next_switch = std::chrono::steady_clock::now();

        while (g_run.load(std::memory_order_relaxed)) {
            size_t got = 0;

            bool playing = play_enabled.load(std::memory_order_relaxed);
            bool rec     = record_enabled.load(std::memory_order_relaxed);
            bool expl    = explicit_mode.load(std::memory_order_relaxed);
            int interval = grain_interval_ms.load(std::memory_order_relaxed);
            int glen_ms  = grain_length_ms.load(std::memory_order_relaxed);

            if (!playing) {
                got = 0;
            }
            else if (rec) {
                // monitor mode: peek from ring buffer with delay
                if (rb_mtx.try_lock()) {
                    const size_t delay = samples_per_period * 2;
                    got = rb.copy_latest(out.data(), samples_per_period, delay);
                    rb_mtx.unlock();
                } else {
                    got = 0;
                }
            }
            else {
                // grains mode: random slice playback
                auto now = std::chrono::steady_clock::now();
                if (now >= next_switch) {
                    rng = xorshift32(rng);
                    int new_id = -1;
                    if (store.random_id(rng, new_id)) {
                        Slice tmp;
                        if (store.get(new_id, tmp)) {
                            cur = tmp;
                            cur_valid = true;
                            pos = 0;
                            current_grain_id.store(new_id, std::memory_order_relaxed);

                            // per-grain jitter: ±25% of glen_ms at grain start
                            rng = xorshift32(rng);
                            int jitter_range = glen_ms / 4;
                            grain_jitter_ms = (jitter_range > 0)
                                ? (int)(rng % (2 * jitter_range + 1)) - jitter_range
                                : 0;

                            if (expl) {
                                std::fprintf(stderr, "[grain] id=%d slice_samples=%u jitter=%dms interval=%dms\n",
                                    cur.id, cur.length, grain_jitter_ms, interval);
                            }
                        }
                    } else {
                        cur_valid = false;
                    }
                    next_switch = now + std::chrono::milliseconds(interval);
                }

                if (cur_valid) {
                    int actual_ms = std::max(20, glen_ms + grain_jitter_ms);
                    uint32_t play_end = std::min(
                        (uint32_t)((uint64_t)rate * channels * actual_ms / 1000),
                        cur.length);
                    uint32_t remain = (pos < play_end) ? (play_end - pos) : 0;
                    uint32_t want = (uint32_t)samples_per_period;
                    uint32_t take = std::min(remain, want);

                    if (take > 0) {
                        std::lock_guard<std::mutex> lk(store.mtx);
                        const int16_t* base = store.corpus.data() + cur.corpus_start;
                        std::copy(base + pos, base + pos + take, out.begin());

                        got = take;
                        pos += take;
                    } else {
                        got = 0;
                        cur_valid = false;
                    }
                }
            }

            if (got < samples_per_period) {
                std::fill(out.begin() + got, out.end(), 0);
            }

            snd_pcm_sframes_t w = snd_pcm_writei(pb, out.data(), period_frames);
            if (w < 0) recover_if_xrun(pb, (int)w, "playback");
        }
    });

    // ---- key input loop ----
    while (g_run.load(std::memory_order_relaxed)) {
        int k = read_key_nonblocking();

        if (k != -1) {
            if (k == 'q') { g_run.store(false); break; }

            else if (k == 'r') {
                record_enabled.store(true);
                {
                    std::lock_guard<std::mutex> lk(rb_mtx);
                    rb.clear();
                }
                std::fprintf(stderr, "record_enabled = true\n");
            }

            else if (k == 's') {
                record_enabled.store(false);
                std::fprintf(stderr, "record_enabled = false\n");
            }

            else if (k == 'm') {
                bool ns = !play_enabled.load();
                play_enabled.store(ns);
                std::fprintf(stderr, "play_enabled = %s\n", ns ? "true" : "false");
            }

            else if (k == 'e') {
                bool v = !explicit_mode.load();
                explicit_mode.store(v);
                std::fprintf(stderr, "[key] explicit_mode = %s\n", v ? "true" : "false");
            }

            else if (k == 'l') {
                store.list();
            }

            else if (k == '.') {
                int v = std::min(grain_length_ms.load() + 25, slicer->slice_ms);
                grain_length_ms.store(v);
                std::fprintf(stderr, "[key] grain_length=%dms\n", v);
            }
            else if (k == ',') {
                int v = std::max(grain_length_ms.load() - 25, 20);
                grain_length_ms.store(v);
                std::fprintf(stderr, "[key] grain_length=%dms\n", v);
            }

            else if (k == ']') {
                float v = std::min(slicer_sensitivity.load() + 0.5f, 10.0f);
                slicer_sensitivity.store(v);
                std::fprintf(stderr, "[key] sensitivity=%.1f\n", v);
            }
            else if (k == '[') {
                float v = std::max(slicer_sensitivity.load() - 0.5f, 0.5f);
                slicer_sensitivity.store(v);
                std::fprintf(stderr, "[key] sensitivity=%.1f\n", v);
            }
            else if (k == '1') { grain_interval_ms.store(1000); std::fprintf(stderr, "[key] interval=1000ms\n"); }
            else if (k == '2') { grain_interval_ms.store(200);  std::fprintf(stderr, "[key] interval=200ms\n"); }
            else if (k == '3') { grain_interval_ms.store(100);  std::fprintf(stderr, "[key] interval=100ms\n"); }
            else if (k == '4') { grain_interval_ms.store(50);   std::fprintf(stderr, "[key] interval=50ms\n"); }
            else if (k == '5') { grain_interval_ms.store(30);   std::fprintf(stderr, "[key] interval=30ms\n"); }
            else if (k == '6') { grain_interval_ms.store(15);   std::fprintf(stderr, "[key] interval=15ms\n"); }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // shutdown
    g_run.store(false);
    if (cap_thread.joinable()) cap_thread.join();
    if (pb_thread.joinable()) pb_thread.join();
    if (slicer_thread.joinable()) slicer_thread.join();

    snd_pcm_close(cap);
    snd_pcm_close(pb);
    std::fprintf(stderr, "Done.\n");
    return 0;
}
