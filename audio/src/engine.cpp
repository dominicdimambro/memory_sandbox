#include <alsa/asoundlib.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <linux/gpio.h>

#include "alsa_util.h"
#include "grain_pipeline.h"
#include "grain_visualizer.h"
#include "ringbuffer.h"
#include "slice_store.h"
#include "slicer.h"
#include "terminal.h"

static std::atomic<bool> g_run{true};

// Returns cumulative CPU time (user + sys) for this process in clock ticks.
// Reads /proc/self/stat; fields 14+15 (utime, stime) follow the comm field.
static uint64_t read_cpu_ticks() {
    FILE* f = fopen("/proc/self/stat", "r");
    if (!f) return 0;
    char buf[512];
    bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    fclose(f);
    if (!ok) return 0;
    const char* p = strrchr(buf, ')');  // comm ends at last ')'
    if (!p) return 0;
    p += 2;  // skip ') '
    unsigned long utime = 0, stime = 0;
    // after comm: state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt utime stime
    sscanf(p, "%*c %*d %*d %*d %*d %*d %*lu %*lu %*lu %*lu %*lu %lu %lu",
           &utime, &stime);
    return (uint64_t)(utime + stime);
}

int main(int argc, char** argv) {

    // capture and playback devices with default fallbacks
    const char* cap_dev = (argc > 1) ? argv[1] : "plughw:2,0";
    const char* pb_dev  = (argc > 2) ? argv[2] : "plughw:2,0";

    // hw parameters
    const unsigned int rate = 48000;
    const unsigned int channels = 2;
    const snd_pcm_uframes_t period_frames = 256;
    const snd_pcm_uframes_t buffer_frames = period_frames * 8;

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
        "   - / = = envelope shape: percussive -> sustained -> swell\n"
        "   p = toggle passthrough (dry signal mixed with grains)\n"
        "   9 / 0 = decrease / increase dry/wet (0=all dry  1=all grains)\n"
        "   z / x = tonal root down / up (semitone, wraps A..Ab)\n"
        "   v = toggle tonal scale (major / minor)\n"
        "   k = cycle kNN k (1/3/5/10 nearest grains to live audio)\n"
        "   q = quit\n"
    );

    TermRawMode raw;
    raw.enable();

    // slicer: swappable algorithm (default: onset detection)
    auto slicer = std::make_unique<OnsetSlicer>();

    // grain pipeline stages
    std::vector<std::unique_ptr<SlicePreprocessor>> preprocessors;
    std::vector<std::unique_ptr<SliceAnalyzer>>     analyzers;
    std::vector<std::unique_ptr<SliceFilter>>        filters;

    // register preprocessors, analyzers, filters in the desired order
    preprocessors.push_back(std::make_unique<StereoToMonoPreprocessor>());

    static constexpr float kRootFreqs[12] = {
        440.000f, 466.164f, 493.883f, 523.251f, 554.365f, 587.330f,
        622.254f, 659.255f, 698.456f, 739.989f, 783.991f, 830.609f,
    //  A         Bb        B         C         C#        D
    //  Eb        E         F         F#        G         Ab
    };

    analyzers.push_back(std::make_unique<RMSAnalyzer>());
    analyzers.push_back(std::make_unique<GainNormalizerAnalyzer>());
    analyzers.push_back(std::make_unique<F0Analyzer>());
    analyzers.push_back(std::make_unique<SpectralRolloffAnalyzer>());
    analyzers.push_back(std::make_unique<SpectralFlatnessAnalyzer>());

    auto tonal_ptr_own = std::make_unique<TonalAlignmentAnalyzer>();
    TonalAlignmentAnalyzer* tonal_ptr = tonal_ptr_own.get();
    analyzers.push_back(std::move(tonal_ptr_own));


    SliceStore store;

    // slicer control + state
    std::atomic<uint64_t> rb_epoch{0};
    std::vector<int16_t> one_sec_window(rb_capacity_samples);

    // slicer controls
    std::atomic<float> slicer_sensitivity{1.0f};

    // playback controls
    // tonal alignment controls
    // root index: 0=A 1=Bb 2=B 3=C 4=C# 5=D 6=Eb 7=E 8=F 9=F# 10=G 11=Ab
    std::atomic<int>  tonal_root_idx{0};   // default: A=440Hz
    std::atomic<bool> tonal_minor{false};  // false=major, true=minor

    std::atomic<int> grain_interval_ms{1000};
    std::atomic<int> grain_length_ms{200};
    std::atomic<bool> explicit_mode{false};
    std::atomic<int> current_grain_id{-1};
    // 0.0=percussive  1.0=sustained  2.0=swell  (interpolated between 3 presets)
    std::atomic<float> env_pos{1.0f};
    std::atomic<bool> passthrough_enabled{false};
    std::atomic<float> dry_wet{1.0f};  // 0.0=all dry (passthrough)  1.0=all grains (wet)

    // kNN grain selection
    std::atomic<int>   grain_k{5};          // k=1/3/5/10, cycled with 'k' key
    std::atomic<float> live_tonal{0.f};
    std::atomic<float> live_rolloff{500.f};
    std::atomic<float> live_rms{0.05f};
    std::atomic<bool>  live_valid{false};   // true once first valid (non-silent) analysis done

    GrainVisualizer viz(store, g_run, current_grain_id);

    // passthrough ring buffer: small, always written by capture thread
    const size_t pt_rb_capacity = samples_per_period * 8;
    RingBuffer<int16_t> pt_rb(pt_rb_capacity);
    std::mutex pt_rb_mtx;

    // ---- capture thread ----
    std::thread cap_thread([&] {
        std::vector<int16_t> in(samples_per_period);

        uint64_t samples_since_epoch = 0;
        bool was_recording = false;

        // live feature analysis: accumulates directly from capture into a local buffer,
        // no mutex needed — only the capture thread reads/writes this
        const size_t kLiveFrames  = kFFTSize;  // 1024 frames
        const size_t kLiveSamples = kLiveFrames * channels;
        std::vector<int16_t> live_buf(kLiveSamples, 0);
        size_t live_fill = 0;
        StereoToMonoPreprocessor live_s2m;
        RMSAnalyzer              live_rms_an;
        SpectralRolloffAnalyzer  live_rolloff_an;
        TonalAlignmentAnalyzer   live_tonal_an;

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

            const size_t nsamp = (size_t)n * channels;

            // always feed passthrough buffer regardless of record state
            {
                std::lock_guard<std::mutex> lk(pt_rb_mtx);
                pt_rb.write_overwrite(in.data(), nsamp);
            }

            // accumulate capture samples into live_buf; analyze when full
            {
                size_t to_copy = std::min(nsamp, kLiveSamples - live_fill);
                std::memcpy(live_buf.data() + live_fill, in.data(), to_copy * sizeof(int16_t));
                live_fill += to_copy;
            }
            if (live_fill >= kLiveSamples) {
                live_fill = 0;
                if (true) {
                    SliceCandidate lc;
                    lc.window   = live_buf.data();
                    lc.channels = channels;
                    lc.rate     = rate;
                    lc.region   = {0, kLiveFrames};
                    live_s2m.process(lc);
                    live_rms_an.analyze(lc);
                    // always update rms so silence gate in playback thread tracks current level
                    live_rms.store(lc.features.rms, std::memory_order_relaxed);
                    if (lc.features.rms >= 0.01f) {
                        // only update tonal/rolloff when signal is present (meaningless on silence)
                        live_rolloff_an.analyze(lc);
                        live_tonal_an.root_idx   = tonal_root_idx.load(std::memory_order_relaxed);
                        live_tonal_an.scale_type = tonal_minor.load(std::memory_order_relaxed)
                                                    ? ScaleType::Minor : ScaleType::Major;
                        live_tonal_an.analyze(lc);
                        live_tonal.store(lc.features.tonal_alignment_score, std::memory_order_relaxed);
                        live_rolloff.store(lc.features.rolloff_freq, std::memory_order_relaxed);
                        live_valid.store(true, std::memory_order_release);
                    }
                }
            }

            if (!rec) continue;

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
                    tonal_ptr->root_idx   = tonal_root_idx.load(std::memory_order_relaxed);
                    tonal_ptr->scale_type = tonal_minor.load(std::memory_order_relaxed)
                                            ? ScaleType::Minor : ScaleType::Major;
                    auto regions = slicer->process(
                        one_sec_window.data(), got, channels, rate
                    );

                    // build initial candidate list from raw slice regions
                    std::vector<SliceCandidate> slice_candidates;
                    slice_candidates.reserve(regions.size());
                    for (auto& region : regions) {
                        slice_candidates.push_back(SliceCandidate{
                            region,
                            one_sec_window.data(),
                            channels,
                            rate,
                            {} // features populated below
                        });
                    }

                    // ---- preprocessing stage ----
                    // each preprocessor may modify a candidate's region or discard it
                    for (auto& pre : preprocessors) {
                        std::vector<SliceCandidate> surviving;
                        surviving.reserve(slice_candidates.size());
                        for (auto& c : slice_candidates) {
                            if (pre->process(c)) surviving.push_back(c);
                        }
                        slice_candidates = std::move(surviving);
                    }

                    // ---- feature analysis stage ----
                    // each analyzer populates fields in candidate.features
                    for (auto& c : slice_candidates) {
                        for (auto& an : analyzers) {
                            an->analyze(c);
                        }
                    }

                    // ---- filter stage ----
                    // discard candidates that fail any filter
                    for (auto& filt : filters) {
                        std::vector<SliceCandidate> surviving;
                        surviving.reserve(slice_candidates.size());
                        for (auto& c : slice_candidates) {
                            if (filt->passes(c)) surviving.push_back(c);
                        }
                        slice_candidates = std::move(surviving);
                    }

                    // ---- commit surviving candidates to store ----
                    for (auto& c : slice_candidates) {
                        size_t start_sample = c.region.start_frame * channels;
                        size_t n_samples_slice = c.region.length_frames * channels;

                        int id = store.add_slice(
                            c.window + start_sample,
                            (uint32_t)n_samples_slice,
                            c.features
                        );

                        std::fprintf(stderr, "[slice] id=%d start=%.1fms len=%zu samples\n",
                            id, 1000.0 * c.region.start_frame / rate, n_samples_slice);
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
        std::vector<int32_t> mix(samples_per_period, 0);
        std::vector<int16_t> pt_buf(samples_per_period, 0);

        // voice pool for overlapping grain playback
        const int MAX_VOICES = 30;
        struct GrainVoice {
            Slice slice;
            uint32_t pos;
            uint32_t play_end;
            bool active;
            float attack_frac;  // fraction of play_end spent in attack ramp
            float decay_frac;   // fraction of play_end spent in decay ramp
        };
        GrainVoice voices[MAX_VOICES] = {};

        // envelope shape presets: {attack_frac, decay_frac}
        // shape 0 (percussive): short attack, long decay ramp
        // shape 1 (sustained):  short attack, sustain plateau, short decay
        // shape 2 (swell):      long attack ramp, short decay
        struct EnvPreset { float attack; float decay; };
        const EnvPreset kEnvShapes[3] = {
            {0.05f, 0.95f},
            {0.10f, 0.10f},
            {0.85f, 0.15f},
        };
        int next_voice = 0;

        uint32_t rng = 0x12345678u;

        // master output envelope: ramps up on signal, decays slowly on silence
        // per-period increments at 256 frames / 48kHz ≈ 5.3ms per period
        float master_env = 0.0f;
        static constexpr float kMasterAttack  = 0.10f;   // ~53ms to full
        static constexpr float kMasterRelease = 0.008f;  // ~660ms tail

        auto next_switch = std::chrono::steady_clock::now();

        while (g_run.load(std::memory_order_relaxed)) {
            size_t got = 0;

            bool playing = play_enabled.load(std::memory_order_relaxed);
            bool rec     = record_enabled.load(std::memory_order_relaxed);
            bool expl    = explicit_mode.load(std::memory_order_relaxed);
            int interval = grain_interval_ms.load(std::memory_order_relaxed);
            int glen_ms  = grain_length_ms.load(std::memory_order_relaxed);
            float env_p  = env_pos.load(std::memory_order_relaxed);

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
                // grains mode: overlapping random slice playback
                auto now = std::chrono::steady_clock::now();
                // silence gate: don't trigger new grains when live input is quiet
                bool live_active = !live_valid.load(std::memory_order_acquire)
                                   || live_rms.load(std::memory_order_relaxed) >= 0.01f;
                if (now >= next_switch && live_active) {
                    rng = xorshift32(rng);
                    int new_id = -1;
                    bool found_id = false;
                    if (live_valid.load(std::memory_order_acquire)) {
                        found_id = store.closest_k_id(
                            live_tonal.load(std::memory_order_relaxed),
                            live_rolloff.load(std::memory_order_relaxed),
                            live_rms.load(std::memory_order_relaxed),
                            grain_k.load(std::memory_order_relaxed),
                            rng, new_id);
                    }
                    if (!found_id) found_id = store.random_id(rng, new_id);
                    if (found_id) {
                        Slice tmp;
                        if (store.get(new_id, tmp)) {
                            // per-grain jitter: ±25% of glen_ms at grain start
                            rng = xorshift32(rng);
                            int jitter_range = glen_ms / 4;
                            int jitter_ms = (jitter_range > 0)
                                ? (int)(rng % (2 * jitter_range + 1)) - jitter_range
                                : 0;
                            int actual_ms = std::max(20, glen_ms + jitter_ms);
                            uint32_t play_end = std::min(
                                (uint32_t)((uint64_t)rate * channels * actual_ms / 1000),
                                tmp.length);

                            // interpolate envelope shape between the three presets
                            float ep = std::max(0.0f, std::min(2.0f, env_p));
                            int   lo = (int)ep;
                            int   hi = std::min(lo + 1, 2);
                            float t  = ep - (float)lo;
                            float att = kEnvShapes[lo].attack * (1.0f - t) + kEnvShapes[hi].attack * t;
                            float dec = kEnvShapes[lo].decay  * (1.0f - t) + kEnvShapes[hi].decay  * t;

                            // round-robin voice allocation (overwrites oldest if all busy)
                            GrainVoice& v = voices[next_voice % MAX_VOICES];
                            v.slice       = tmp;
                            v.pos         = 0;
                            v.play_end    = play_end;
                            v.active      = true;
                            v.attack_frac = att;
                            v.decay_frac  = dec;
                            next_voice++;
                            current_grain_id.store(new_id, std::memory_order_relaxed);

                            if (expl) {
                                std::fprintf(stderr, "[grain] id=%d slice_samples=%u jitter=%dms interval=%dms\n",
                                    tmp.id, tmp.length, jitter_ms, interval);
                            }
                        }
                    }
                    next_switch = now + std::chrono::milliseconds(interval);
                }

                // mix all active voices into int32 accumulator, then clamp to int16
                std::fill(mix.begin(), mix.end(), 0);
                {
                    std::lock_guard<std::mutex> lk(store.mtx);
                    for (auto& v : voices) {
                        if (!v.active) continue;
                        uint32_t remain = (v.pos < v.play_end) ? (v.play_end - v.pos) : 0;
                        uint32_t take = std::min(remain, (uint32_t)samples_per_period);
                        if (take > 0) {
                            const int16_t* base = store.corpus.data() + v.slice.corpus_start;
                            for (uint32_t i = 0; i < take; i++) {
                                float grain_t = (float)(v.pos + i) / (float)v.play_end;
                                float env;
                                if (grain_t < v.attack_frac) {
                                    env = grain_t / v.attack_frac;
                                } else if (grain_t > 1.0f - v.decay_frac) {
                                    env = (1.0f - grain_t) / v.decay_frac;
                                } else {
                                    env = 1.0f;
                                }
                                mix[i] += (int32_t)((float)base[v.pos + i] * env * v.slice.features.gain);
                            }
                            v.pos += take;
                        }
                        if (v.pos >= v.play_end) v.active = false;
                    }
                }
                // advance master envelope toward target (1 when active, 0 when silent)
                float env_target = live_active ? 1.0f : 0.0f;
                float env_step   = live_active ? kMasterAttack : kMasterRelease;
                if (master_env < env_target) {
                    master_env = std::min(master_env + env_step, 1.0f);
                } else {
                    master_env = std::max(master_env - env_step, 0.0f);
                }

                for (size_t i = 0; i < samples_per_period; i++) {
                    int32_t s = (int32_t)(mix[i] * master_env);
                    out[i] = (int16_t)(s > 32767 ? 32767 : s < -32768 ? -32768 : s);
                }
                got = samples_per_period;
            }

            if (got < samples_per_period) {
                std::fill(out.begin() + got, out.end(), 0);
            }

            // passthrough: crossfade dry (live) and wet (grains)
            // dry_wet: 0.0=all dry  1.0=all grains
            if (passthrough_enabled.load(std::memory_order_relaxed)) {
                float dw = dry_wet.load(std::memory_order_relaxed);
                size_t pt_got = 0;
                {
                    std::lock_guard<std::mutex> lk(pt_rb_mtx);
                    pt_got = pt_rb.copy_latest(pt_buf.data(), samples_per_period, samples_per_period * 2);
                }
                for (size_t i = 0; i < samples_per_period; i++) {
                    float dry_s = (i < pt_got) ? (float)pt_buf[i] * (1.0f - dw) : 0.0f;
                    float wet_s = (float)out[i] * dw;
                    int32_t s = (int32_t)(dry_s + wet_s);
                    out[i] = (int16_t)(s > 32767 ? 32767 : s < -32768 ? -32768 : s);
                }
            }

            snd_pcm_sframes_t w = snd_pcm_writei(pb, out.data(), period_frames);
            if (w < 0) recover_if_xrun(pb, (int)w, "playback");
        }
    });

    // ---- encoder thread (MCP23017 via I2C) ----
    // enc1: A=GPA7 B=GPA6 btn=GPB0
    // enc2: A=GPA5 B=GPA4 btn=GPB1
    // enc3: A=GPA3 B=GPA2 btn=GPB2
    // enc4: A=GPA1 B=GPA0 btn=GPB3
    std::thread encoder_thread([&] {
        int fd = open("/dev/i2c-1", O_RDWR);
        if (fd < 0) {
            std::fprintf(stderr, "[encoder] failed to open /dev/i2c-1\n");
            return;
        }
        if (ioctl(fd, I2C_SLAVE, 0x20) < 0) {
            std::fprintf(stderr, "[encoder] failed to set I2C slave address\n");
            close(fd);
            return;
        }

        auto i2c_write_reg = [&](uint8_t reg, uint8_t val) {
            uint8_t buf[2] = {reg, val};
            write(fd, buf, 2);
        };
        auto i2c_read_reg = [&](uint8_t reg) -> uint8_t {
            write(fd, &reg, 1);
            uint8_t val = 0;
            read(fd, &val, 1);
            return val;
        };

        // all GPA pins as inputs (all encoder A/B lines)
        i2c_write_reg(0x00, 0xFF);        // IODIRA
        i2c_write_reg(0x01, 0xFF);        // IODIRB
        // pull-ups on all 8 GPA encoder pins and 4 button pins (GPB0,1,2,3)
        i2c_write_reg(0x0C, 0xFF);        // GPPUA: GPA0-7
        i2c_write_reg(0x0D, 0b00001111);  // GPPUB: GPB0,1,2,3

        // A-pin bit positions in port_a (odd bits), B-pin (even bits below each A)
        static constexpr uint8_t kABit[4] = { 7, 5, 3, 1 };
        static constexpr uint8_t kBBit[4] = { 6, 4, 2, 0 };
        // button bit positions in port_b
        static constexpr uint8_t kBtnBit[4] = { 0, 1, 2, 3 };

        uint8_t init_a = i2c_read_reg(0x12);
        uint8_t prev_a[4];
        for (int i = 0; i < 4; i++)
            prev_a[i] = (init_a >> kABit[i]) & 1;

        int      counters[4]  = {};
        uint8_t  prev_btn_raw = 0xFF;  // all released (pulled high)

        while (g_run.load(std::memory_order_relaxed)) {
            uint8_t port_a = i2c_read_reg(0x12);  // GPIOA
            uint8_t port_b = i2c_read_reg(0x13);  // GPIOB

            // decode all four encoders
            for (int i = 0; i < 4; i++) {
                uint8_t a = (port_a >> kABit[i]) & 1;
                uint8_t b = (port_a >> kBBit[i]) & 1;
                if (a != prev_a[i]) {
                    counters[i] += (a == 1) ? (b == 0 ? -1 : 1)
                                            : (b == 1 ? -1 : 1);
                    std::fprintf(stderr, "[encoder%d] counter=%d\n", i + 1, counters[i]);
                    prev_a[i] = a;
                }
            }

            // check all four buttons
            for (int i = 0; i < 4; i++) {
                uint8_t mask = 1u << kBtnBit[i];
                uint8_t cur  = port_b  & mask;
                uint8_t prev = prev_btn_raw & mask;
                if (cur != prev) {
                    std::fprintf(stderr, "[encoder%d] button=%s\n",
                        i + 1, cur == 0 ? "pressed" : "released");
                }
            }
            prev_btn_raw = port_b;

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        close(fd);
    });

    // ---- input gain presets (dB, PCM1863 PGA via ALSA enum control) ----
    static constexpr const char* kMixerCard      = "hw:2";
    static constexpr float kGainLine_dB       =  0.0f;  // TRS line level
    static constexpr float kGainInstrument_dB = 20.0f;  // TRS instrument level
    static constexpr float kGainDualJack_dB   =  0.0f;  // dual jack stereo (line)
    static constexpr float kGainMic_dB        = 30.0f;  // XLR mic mono

    std::atomic<int> current_input_mode{-1};

    // ---- rotary switch thread (GPIO 5 + 6 direct to Pi) ----
    std::thread switch_thread([&] {
        int chip_fd = open("/dev/gpiochip4", O_RDONLY);
        if (chip_fd < 0) {
            std::fprintf(stderr, "[switch] failed to open /dev/gpiochip4\n");
            return;
        }

        struct gpio_v2_line_request req = {};
        req.offsets[0] = 16;
        req.offsets[1] = 26;
        req.num_lines   = 2;
        req.config.flags = GPIO_V2_LINE_FLAG_INPUT | GPIO_V2_LINE_FLAG_BIAS_PULL_UP;
        std::strncpy(req.consumer, "engine-switch", sizeof(req.consumer) - 1);

        if (ioctl(chip_fd, GPIO_V2_GET_LINE_IOCTL, &req) < 0) {
            std::fprintf(stderr, "[switch] failed to get GPIO lines\n");
            close(chip_fd);
            return;
        }
        close(chip_fd);

        int prev_mode = -1;

        while (g_run.load(std::memory_order_relaxed)) {
            struct gpio_v2_line_values vals = {};
            vals.mask = 0b11;
            if (ioctl(req.fd, GPIO_V2_LINE_GET_VALUES_IOCTL, &vals) < 0) {
                std::fprintf(stderr, "[switch] read error\n");
                break;
            }

            int gpio5 = (vals.bits >> 0) & 1;
            int gpio6 = (vals.bits >> 1) & 1;
            int mode  = (gpio6 << 1) | gpio5;  // 2-bit position: 0-3

            if (mode != prev_mode) {
                const char* names[] = {"???", "XLR mic mono", "Dual jack stereo", "TRS stereo"};
                std::fprintf(stderr, "[switch] mode=%s (GPIO5=%d GPIO6=%d)\n",
                    names[mode], gpio5, gpio6);

                float gain = kGainLine_dB;
                switch (mode) {
                    case 1: gain = kGainMic_dB;       break;
                    case 2: gain = kGainDualJack_dB;  break;
                    case 3: gain = kGainLine_dB;       break;
                }
                if (mode != 0) {
                    if (set_pga_gain_db(kMixerCard, gain))
                        std::fprintf(stderr, "[gain] set %.1fdB for mode %s\n", gain, names[mode]);
                }
                current_input_mode.store(mode, std::memory_order_relaxed);
                prev_mode = mode;
            }

            // TRS stereo (mode 3): auto-detect instrument vs line level with hysteresis
            if (current_input_mode.load(std::memory_order_relaxed) == 3) {
                static constexpr float kLowRms          = 0.03f;  // below = instrument level
                static constexpr float kHighRms         = 0.08f;  // above = line level
                static constexpr int   kHoldMs          = 3000;
                static constexpr float kSignalPresent   = 0.01f;  // must be above this to count as signal (not silence)
                static bool trs_instrument = false;
                static auto low_since  = std::chrono::steady_clock::time_point{};
                static auto high_since = std::chrono::steady_clock::time_point{};
                static bool low_timing  = false;
                static bool high_timing = false;

                float rms = live_rms.load(std::memory_order_relaxed);
                auto now = std::chrono::steady_clock::now();

                if (!trs_instrument) {
                    // currently at line gain — watch for quiet signal
                    if (rms > kSignalPresent && rms < kLowRms) {
                        if (!low_timing) { low_since = now; low_timing = true; }
                        else if (std::chrono::duration_cast<std::chrono::milliseconds>(now - low_since).count() >= kHoldMs) {
                            trs_instrument = true;
                            low_timing = false;
                            set_pga_gain_db(kMixerCard, kGainInstrument_dB);
                            std::fprintf(stderr, "[gain] TRS: instrument level detected → %.1fdB\n", kGainInstrument_dB);
                        }
                    } else {
                        low_timing = false;
                    }
                } else {
                    // currently at instrument gain — watch for loud signal
                    if (rms > kHighRms) {
                        if (!high_timing) { high_since = now; high_timing = true; }
                        else if (std::chrono::duration_cast<std::chrono::milliseconds>(now - high_since).count() >= kHoldMs) {
                            trs_instrument = false;
                            high_timing = false;
                            set_pga_gain_db(kMixerCard, kGainLine_dB);
                            std::fprintf(stderr, "[gain] TRS: line level detected → %.1fdB\n", kGainLine_dB);
                        }
                    } else {
                        high_timing = false;
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        close(req.fd);
    });

    // ---- LED thread (GPIO 23 = green anode, GPIO 24 = red anode) ----
    // recording: green on, red off  |  stopped: green off, red on
    std::thread led_thread([&] {
        int chip_fd = open("/dev/gpiochip4", O_RDONLY);
        if (chip_fd < 0) {
            std::fprintf(stderr, "[led] failed to open /dev/gpiochip4\n");
            return;
        }

        struct gpio_v2_line_request led_req = {};
        led_req.offsets[0] = 23;  // green anode
        led_req.offsets[1] = 24;  // red anode
        led_req.num_lines   = 2;
        led_req.config.flags = GPIO_V2_LINE_FLAG_OUTPUT;
        std::strncpy(led_req.consumer, "engine-led", sizeof(led_req.consumer) - 1);

        if (ioctl(chip_fd, GPIO_V2_GET_LINE_IOCTL, &led_req) < 0) {
            std::fprintf(stderr, "[led] failed to get GPIO lines\n");
            close(chip_fd);
            return;
        }
        close(chip_fd);

        bool prev_rec = !record_enabled.load();  // force update on first iteration

        while (g_run.load(std::memory_order_relaxed)) {
            bool rec = record_enabled.load(std::memory_order_relaxed);
            if (rec != prev_rec) {
                struct gpio_v2_line_values vals = {};
                vals.mask = 0b11;
                vals.bits = rec ? 0b01 : 0b10;  // rec: green=1 red=0 | idle: green=0 red=1
                ioctl(led_req.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &vals);
                prev_rec = rec;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        // turn both off on shutdown
        struct gpio_v2_line_values off = {};
        off.mask = 0b11;
        off.bits = 0b00;
        ioctl(led_req.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &off);
        close(led_req.fd);
    });

    // ---- record button thread (GPIO 12, active-low, toggles record_enabled) ----
    std::thread button_thread([&] {
        int chip_fd = open("/dev/gpiochip4", O_RDONLY);
        if (chip_fd < 0) {
            std::fprintf(stderr, "[btn] failed to open /dev/gpiochip4\n");
            return;
        }

        struct gpio_v2_line_request btn_req = {};
        btn_req.offsets[0] = 12;
        btn_req.num_lines   = 1;
        btn_req.config.flags = GPIO_V2_LINE_FLAG_INPUT | GPIO_V2_LINE_FLAG_BIAS_PULL_UP;
        std::strncpy(btn_req.consumer, "engine-recbtn", sizeof(btn_req.consumer) - 1);

        if (ioctl(chip_fd, GPIO_V2_GET_LINE_IOCTL, &btn_req) < 0) {
            std::fprintf(stderr, "[btn] failed to get GPIO line\n");
            close(chip_fd);
            return;
        }
        close(chip_fd);

        int prev_level = 1;  // assume released (pulled high)

        while (g_run.load(std::memory_order_relaxed)) {
            struct gpio_v2_line_values vals = {};
            vals.mask = 0b1;
            if (ioctl(btn_req.fd, GPIO_V2_LINE_GET_VALUES_IOCTL, &vals) < 0) {
                std::fprintf(stderr, "[btn] read error\n");
                break;
            }

            int level = vals.bits & 1;

            // falling edge = button pressed (pin pulled to GND)
            if (level == 0 && prev_level == 1) {
                bool rec = !record_enabled.load(std::memory_order_relaxed);
                record_enabled.store(rec, std::memory_order_relaxed);
                if (rec) {
                    std::lock_guard<std::mutex> lk(rb_mtx);
                    rb.clear();
                }
                std::fprintf(stderr, "[btn] record_enabled = %s\n", rec ? "true" : "false");
            }

            prev_level = level;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));  // debounce
        }

        close(btn_req.fd);
    });

    viz.start();

    // ---- key input loop ----
    auto last_stats_time  = std::chrono::steady_clock::now();
    uint64_t last_cpu_ticks = read_cpu_ticks();

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

            else if (k == 'p') {
                bool v = !passthrough_enabled.load();
                passthrough_enabled.store(v);
                std::fprintf(stderr, "[key] passthrough = %s\n", v ? "on" : "off");
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
            else if (k == '=') {
                float v = std::min(env_pos.load() + 0.25f, 2.0f);
                env_pos.store(v);
                const char* name = (v < 0.75f) ? "percussive" : (v < 1.25f) ? "sustained" : (v < 1.75f) ? "->swell" : "swell";
                std::fprintf(stderr, "[key] env_shape=%.2f (%s)\n", v, name);
            }
            else if (k == '-') {
                float v = std::max(env_pos.load() - 0.25f, 0.0f);
                env_pos.store(v);
                const char* name = (v < 0.25f) ? "percussive" : (v < 0.75f) ? "->sustained" : (v < 1.25f) ? "sustained" : "->swell";
                std::fprintf(stderr, "[key] env_shape=%.2f (%s)\n", v, name);
            }

            else if (k == '9') {
                float v = std::max(dry_wet.load() - 0.1f, 0.0f);
                dry_wet.store(v);
                std::fprintf(stderr, "[key] dry_wet=%.1f (%s)\n", v, v < 0.15f ? "all dry" : v > 0.85f ? "all grains" : "mix");
            }
            else if (k == '0') {
                float v = std::min(dry_wet.load() + 0.1f, 1.0f);
                dry_wet.store(v);
                std::fprintf(stderr, "[key] dry_wet=%.1f (%s)\n", v, v < 0.15f ? "all dry" : v > 0.85f ? "all grains" : "mix");
            }

            else if (k == 'z' || k == 'x') {
                static const char* kNames[12] = {"A","Bb","B","C","C#","D","Eb","E","F","F#","G","Ab"};
                int idx = (k == 'z')
                    ? (tonal_root_idx.load() + 11) % 12   // step down
                    : (tonal_root_idx.load() +  1) % 12;  // step up
                tonal_root_idx.store(idx);
                ScaleType sc = tonal_minor.load() ? ScaleType::Minor : ScaleType::Major;
                {
                    std::lock_guard<std::mutex> lk(store.mtx);
                    for (auto& [id, slice] : store.slices)
                        slice.features.tonal_alignment_score =
                            TonalAlignmentAnalyzer::tonal_score_from_chroma(
                                slice.features.chroma_energy, idx, sc);
                }
                std::fprintf(stderr, "[key] tonal root=%s (%.1fHz)\n", kNames[idx], kRootFreqs[idx]);
            }
            else if (k == 'v') {
                bool minor = !tonal_minor.load();
                tonal_minor.store(minor);
                int idx = tonal_root_idx.load();
                ScaleType sc = minor ? ScaleType::Minor : ScaleType::Major;
                {
                    std::lock_guard<std::mutex> lk(store.mtx);
                    for (auto& [id, slice] : store.slices)
                        slice.features.tonal_alignment_score =
                            TonalAlignmentAnalyzer::tonal_score_from_chroma(
                                slice.features.chroma_energy, idx, sc);
                }
                std::fprintf(stderr, "[key] tonal scale=%s\n", minor ? "minor" : "major");
            }

            else if (k == 'k') {
                static const int kSteps[] = {1, 3, 5, 10};
                int cur = grain_k.load();
                int next_k = kSteps[0];
                for (int i = 0; i < 3; i++)
                    if (kSteps[i] == cur) { next_k = kSteps[i + 1]; break; }
                grain_k.store(next_k);
                std::fprintf(stderr, "[key] grain_k=%d\n", next_k);
            }

            else if (k == '1') { grain_interval_ms.store(1000); std::fprintf(stderr, "[key] interval=1000ms\n"); }
            else if (k == '2') { grain_interval_ms.store(200);  std::fprintf(stderr, "[key] interval=200ms\n"); }
            else if (k == '3') { grain_interval_ms.store(100);  std::fprintf(stderr, "[key] interval=100ms\n"); }
            else if (k == '4') { grain_interval_ms.store(50);   std::fprintf(stderr, "[key] interval=50ms\n"); }
            else if (k == '5') { grain_interval_ms.store(30);   std::fprintf(stderr, "[key] interval=30ms\n"); }
            else if (k == '6') { grain_interval_ms.store(15);   std::fprintf(stderr, "[key] interval=15ms\n"); }
        }

        // periodic stats: cpu usage + corpus size every 5 seconds
        {
            auto now_s = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now_s - last_stats_time).count();
            if (elapsed >= 5.0) {
                uint64_t ticks = read_cpu_ticks();
                double cpu_pct = ((double)(ticks - last_cpu_ticks) / (double)sysconf(_SC_CLK_TCK))
                                 / elapsed * 100.0;
                size_t corpus_samples, n_slices;
                {
                    std::lock_guard<std::mutex> lk(store.mtx);
                    corpus_samples = store.corpus.size();
                    n_slices       = store.slices.size();
                }
                std::fprintf(stderr, "[stats] cpu=%.1f%%  corpus=%zu slices / %zu samples (%.1f KB)\n",
                    cpu_pct, n_slices, corpus_samples,
                    corpus_samples * sizeof(int16_t) / 1024.0);
                last_stats_time = now_s;
                last_cpu_ticks  = ticks;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // shutdown
    g_run.store(false);
    viz.stop();
    if (cap_thread.joinable()) cap_thread.join();
    if (pb_thread.joinable()) pb_thread.join();
    if (slicer_thread.joinable()) slicer_thread.join();
    if (encoder_thread.joinable()) encoder_thread.join();
    if (switch_thread.joinable()) switch_thread.join();
    if (led_thread.joinable()) led_thread.join();
    if (button_thread.joinable()) button_thread.join();

    snd_pcm_close(cap);
    snd_pcm_close(pb);
    std::fprintf(stderr, "Done.\n");
    return 0;
}
