#include <alsa/asoundlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <pthread.h>
#include <sched.h>
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

static uint64_t read_cpu_ticks() {
    FILE* f = fopen("/proc/self/stat", "r");
    if (!f) return 0;
    char buf[512];
    bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    fclose(f);
    if (!ok) return 0;
    const char* p = strrchr(buf, ')');
    if (!p) return 0;
    p += 2;
    unsigned long utime = 0, stime = 0;
    sscanf(p, "%*c %*d %*d %*d %*d %*d %*lu %*lu %*lu %*lu %*lu %lu %lu",
           &utime, &stime);
    return (uint64_t)(utime + stime);
}

// ---- bank persistence ----

#pragma pack(push, 1)
struct GranHeader {
    char     magic[4];
    uint32_t version;
    uint32_t slice_count;
    uint64_t corpus_samples;
    char     name[64];
};
struct GranSliceRec {
    int32_t  id;
    uint64_t corpus_start;
    uint32_t length;
    SliceFeatures features;
};
#pragma pack(pop)

static std::string banks_dir() {
    const char* home = getenv("HOME");
    return std::string(home ? home : "/tmp") + "/.granular/banks/";
}
static std::string cfg_path() {
    const char* home = getenv("HOME");
    return std::string(home ? home : "/tmp") + "/.granular/last.cfg";
}
static std::string timestamp_name() {
    auto t = std::time(nullptr);
    struct tm tm_info;
    localtime_r(&t, &tm_info);
    char buf[64];
    std::strftime(buf, sizeof(buf), "bank_%Y%m%d_%H%M%S", &tm_info);
    return std::string(buf) + ".gran";
}

static bool save_bank(SliceStore& s, const std::string& path, const std::string& name) {
    std::lock_guard<std::mutex> lk(s.mtx);
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    GranHeader hdr = {};
    hdr.magic[0]='G'; hdr.magic[1]='R'; hdr.magic[2]='A'; hdr.magic[3]='N';
    hdr.version = 1;
    hdr.slice_count    = (uint32_t)s.slices.size();
    hdr.corpus_samples = (uint64_t)s.corpus.size();
    std::strncpy(hdr.name, name.c_str(), 63);
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(s.corpus.data()),
            (std::streamsize)(s.corpus.size() * sizeof(int16_t)));
    for (auto& [id, sl] : s.slices) {
        GranSliceRec rec = {};
        rec.id           = (int32_t)sl.id;
        rec.corpus_start = sl.corpus_start;
        rec.length       = sl.length;
        rec.features     = sl.features;
        f.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
    }
    return f.good();
}

static bool load_bank(SliceStore& s, const std::string& path, std::string& out_name) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    GranHeader hdr = {};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f || std::strncmp(hdr.magic, "GRAN", 4) != 0 || hdr.version != 1) return false;
    std::vector<int16_t> corpus(hdr.corpus_samples);
    f.read(reinterpret_cast<char*>(corpus.data()),
           (std::streamsize)(hdr.corpus_samples * sizeof(int16_t)));
    if (!f) return false;
    std::vector<GranSliceRec> recs(hdr.slice_count);
    for (uint32_t i = 0; i < hdr.slice_count; i++) {
        f.read(reinterpret_cast<char*>(&recs[i]), sizeof(GranSliceRec));
        if (!f) return false;
    }
    std::lock_guard<std::mutex> lk(s.mtx);
    s.corpus = std::move(corpus);
    s.slices.clear();
    s.next_id = 0;
    for (auto& rec : recs) {
        Slice sl;
        sl.id           = (int)rec.id;
        sl.corpus_start = rec.corpus_start;
        sl.length       = rec.length;
        sl.features     = rec.features;
        s.slices[sl.id] = sl;
        if (sl.id >= s.next_id) s.next_id = sl.id + 1;
    }
    out_name = std::string(hdr.name, strnlen(hdr.name, 64));
    return true;
}

static std::vector<std::string> list_banks() {
    std::vector<std::string> result;
    std::error_code ec;
    if (!std::filesystem::exists(banks_dir(), ec)) return result;
    for (auto& entry : std::filesystem::directory_iterator(banks_dir(), ec)) {
        if (entry.path().extension() == ".gran")
            result.push_back(entry.path().filename().string());
    }
    std::sort(result.begin(), result.end());
    return result;
}

static void save_config(const std::string& a_file, const std::string& b_file, int rec) {
    std::ofstream f(cfg_path());
    if (f) {
        f << "slot_a=" << a_file << "\n";
        f << "slot_b=" << b_file << "\n";
        f << "record=" << rec  << "\n";
    }
}

static void load_config(std::string& a_file, std::string& b_file, int& rec) {
    std::ifstream f(cfg_path());
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.size() > 7 && line.substr(0,7) == "slot_a=") a_file = line.substr(7);
        else if (line.size() > 7 && line.substr(0,7) == "slot_b=") b_file = line.substr(7);
        else if (line.size() > 7 && line.substr(0,7) == "record=") {
            try { rec = std::stoi(line.substr(7)); } catch (...) {}
        }
    }
}

// ---- main ----

int main(int argc, char** argv) {
    const char* cap_dev = (argc > 1) ? argv[1] : "hw:2,0";
    const char* pb_dev  = (argc > 2) ? argv[2] : "hw:2,0";

    const unsigned int rate = 48000;
    const unsigned int channels = 2;
    const snd_pcm_uframes_t period_frames = 256;
    const snd_pcm_uframes_t buffer_frames = period_frames * 8;

    const size_t samples_per_period = (size_t)period_frames * channels;

    const size_t rb_capacity_seconds = 1;
    const size_t rb_capacity_samples = (size_t)rate * channels * rb_capacity_seconds;
    RingBuffer<int16_t> rb(rb_capacity_samples);
    std::mutex rb_mtx;

    std::atomic<bool> record_enabled{false};
    std::atomic<bool> play_enabled{true};

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

    auto slicer = std::make_unique<OnsetSlicer>();

    std::vector<std::unique_ptr<SlicePreprocessor>> preprocessors;
    std::vector<std::unique_ptr<SliceAnalyzer>>     analyzers;
    std::vector<std::unique_ptr<SliceFilter>>        filters;

    preprocessors.push_back(std::make_unique<StereoToMonoPreprocessor>());

    static constexpr float kRootFreqs[12] = {
        440.000f, 466.164f, 493.883f, 523.251f, 554.365f, 587.330f,
        622.254f, 659.255f, 698.456f, 739.989f, 783.991f, 830.609f,
    };

    analyzers.push_back(std::make_unique<RMSAnalyzer>());
    {
        auto gn = std::make_unique<GainNormalizerAnalyzer>();
        gn->target_rms = 0.07f;
        gn->max_gain   = 1.5f;  // cap at 1.5× to keep peaks below limiter
        analyzers.push_back(std::move(gn));
    }
    analyzers.push_back(std::make_unique<F0Analyzer>());
    analyzers.push_back(std::make_unique<SpectralRolloffAnalyzer>());
    analyzers.push_back(std::make_unique<SpectralFlatnessAnalyzer>());

    auto tonal_ptr_own = std::make_unique<TonalAlignmentAnalyzer>();
    TonalAlignmentAnalyzer* tonal_ptr = tonal_ptr_own.get();
    analyzers.push_back(std::move(tonal_ptr_own));

    SliceStore store_a;
    SliceStore store_b;
    // pre-reserve to prevent allocations under mutex during recording
    store_a.corpus.reserve(48000 * 2 * 180);   // ~3 min stereo
    store_b.corpus.reserve(48000 * 2 * 180);
    store_a.slices.reserve(4000);               // prevent unordered_map rehash
    store_b.slices.reserve(4000);

    std::atomic<uint64_t> rb_epoch{0};
    std::vector<int16_t> one_sec_window(rb_capacity_samples);

    std::atomic<float> slicer_sensitivity{3.0f};
    std::atomic<float> activity_halflife_ms{50.0f};  // ms; 1e9 = hold forever

    std::atomic<int>   engine_mode{0};        // 0=analysis, 1=exploration
    std::atomic<float> explore_x{0.0f};       // centroid in projected space [-0.5, 0.5]
    std::atomic<float> explore_y{0.0f};
    std::atomic<float> explore_z{0.0f};
    std::atomic<float> search_radius{0.0f};   // dist to k-th nearest grain; for viz sphere
    std::atomic<float> view_theta_y{0.4f};     // analysis-mode view rotation (encoder-controlled)
    std::atomic<float> view_theta_x{0.3f};
    std::atomic<float> view_zoom{2.0f};       // scale multiplier [0.3, 3.0]

    std::atomic<int>  tonal_root_idx{0};
    std::atomic<bool> tonal_minor{false};

    std::atomic<int> grain_interval_ms{200};
    std::atomic<int> grain_length_ms{200};
    std::atomic<bool> explicit_mode{false};
    std::atomic<int> current_grain_id{-1};
    std::atomic<int> current_bank{0};       // 0=store_a, 1=store_b (for viz highlight)
    std::atomic<float> env_pos{1.0f};
    std::atomic<bool> passthrough_enabled{false};
    std::atomic<float> dry_wet{0.0f};
    std::atomic<bool> mono_passthrough{false};
    std::atomic<bool> hpf_enabled{false};

    std::atomic<float> fx_amount{0.0f};
    std::atomic<float> crossfade_pos{0.0f};  // 0=all A, 1=all B
    std::atomic<int>   record_target{0};     // 0=store_a, 1=store_b
    std::atomic<int>   new_slice_signal{0};
    std::atomic<bool>  trs_instrument_gain{false};  // true=instrument 40dB, false=line 0dB

    std::atomic<int>   grain_k{5};
    std::atomic<float> live_tonal{0.f};
    std::atomic<float> live_rolloff{500.f};
    std::atomic<float> live_rms{0.05f};
    std::atomic<bool>  live_valid{false};

    // bank menu shared state (encoder thread writes, visualizer reads)
    BankMenuDisplay bank_menu_disp;
    std::mutex      bank_menu_mtx;
    std::string     bank_name_a, bank_name_b;

    // parameter toast (pot thread writes, visualizer reads)
    ParamToast toast_disp;
    std::mutex toast_mtx;

    // create banks directory if missing
    {
        std::error_code ec;
        std::filesystem::create_directories(banks_dir(), ec);
    }

    // auto-load last banks from config
    std::string cur_bank_a_file, cur_bank_b_file;
    {
        std::string a_file, b_file;
        int rec_tgt = 0;
        load_config(a_file, b_file, rec_tgt);
        if (!a_file.empty()) {
            std::string loaded_name;
            if (load_bank(store_a, banks_dir() + a_file, loaded_name)) {
                bank_name_a     = loaded_name;
                cur_bank_a_file = a_file;
                std::fprintf(stderr, "[bank] loaded slot A: %s (%s)\n", a_file.c_str(), loaded_name.c_str());
            }
        }
        if (!b_file.empty()) {
            std::string loaded_name;
            if (load_bank(store_b, banks_dir() + b_file, loaded_name)) {
                bank_name_b     = loaded_name;
                cur_bank_b_file = b_file;
                std::fprintf(stderr, "[bank] loaded slot B: %s (%s)\n", b_file.c_str(), loaded_name.c_str());
            }
        }
        if (rec_tgt == 1) record_target.store(1);
        {
            std::lock_guard<std::mutex> lk(bank_menu_mtx);
            bank_menu_disp.bank_name_a = bank_name_a;
            bank_menu_disp.bank_name_b = bank_name_b;
            bank_menu_disp.record_target = rec_tgt;
        }
    }

    GrainVisualizer viz(store_a, store_b, g_run,
                        current_grain_id, current_bank,
                        crossfade_pos,
                        bank_menu_mtx, bank_menu_disp,
                        toast_mtx, toast_disp,
                        engine_mode,
                        explore_x, explore_y, explore_z,
                        search_radius,
                        view_theta_y, view_theta_x, view_zoom);

    const size_t pt_rb_capacity = samples_per_period * 8;
    RingBuffer<int16_t> pt_rb(pt_rb_capacity);
    std::mutex pt_rb_mtx;

    // ---- capture thread ----
    std::thread cap_thread([&] {
        std::vector<int16_t> in(samples_per_period);

        uint64_t samples_since_epoch = 0;
        bool was_recording = false;

        const size_t kLiveFrames  = kFFTSize;
        const size_t kLiveSamples = kLiveFrames * channels;
        std::vector<int16_t> live_buf(kLiveSamples, 0);
        size_t live_fill = 0;
        StereoToMonoPreprocessor live_s2m;
        RMSAnalyzer              live_rms_an;
        SpectralRolloffAnalyzer  live_rolloff_an;
        TonalAlignmentAnalyzer   live_tonal_an;

        struct BiquadHP {
            float b0, b1, b2, a1, a2;
            float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
            void init(float fc, float fs) {
                float w0    = 2.0f * 3.14159265f * fc / fs;
                float cw    = std::cos(w0);
                float alpha = std::sin(w0) / (2.0f * 0.7071f);
                float a0    = 1.0f + alpha;
                b0 =  (1.0f + cw) / 2.0f / a0;
                b1 = -(1.0f + cw)        / a0;
                b2 =  (1.0f + cw) / 2.0f / a0;
                a1 = -2.0f * cw          / a0;
                a2 =  (1.0f - alpha)     / a0;
            }
            float process(float x) {
                float y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2;
                x2 = x1; x1 = x; y2 = y1; y1 = y;
                return y;
            }
        } hpf_L, hpf_R;
        hpf_L.init(50.0f, (float)rate);
        hpf_R.init(50.0f, (float)rate);

        while (g_run.load(std::memory_order_relaxed)) {
            snd_pcm_sframes_t n = snd_pcm_readi(cap, in.data(), period_frames);

            if (n < 0) { recover_if_xrun(cap, (int)n, "capture"); continue; }
            if (n == 0) continue;

            bool rec = record_enabled.load(std::memory_order_relaxed);

            if (rec && !was_recording) {
                samples_since_epoch = 0;
                rb_epoch.store(0, std::memory_order_release);
            }
            was_recording = rec;

            const size_t nsamp = (size_t)n * channels;

            {
                size_t to_copy = std::min(nsamp, kLiveSamples - live_fill);
                std::memcpy(live_buf.data() + live_fill, in.data(), to_copy * sizeof(int16_t));
                live_fill += to_copy;
            }

            if (hpf_enabled.load(std::memory_order_relaxed)) {
                static constexpr float kSoftGain = 4.0f;
                for (snd_pcm_sframes_t i = 0; i < n; ++i) {
                    float L = hpf_L.process((float)in[i * 2])     * kSoftGain;
                    float R = hpf_R.process((float)in[i * 2 + 1]) * kSoftGain;
                    in[i * 2]     = (int16_t)(L >  32767 ?  32767 : L < -32768 ? -32768 : L);
                    in[i * 2 + 1] = (int16_t)(R >  32767 ?  32767 : R < -32768 ? -32768 : R);
                }
            }

            {
                std::lock_guard<std::mutex> lk(pt_rb_mtx);
                pt_rb.write_overwrite(in.data(), nsamp);
            }

            if (live_fill >= kLiveSamples) {
                live_fill = 0;
                SliceCandidate lc;
                lc.window   = live_buf.data();
                lc.channels = channels;
                lc.rate     = rate;
                lc.region   = {0, kLiveFrames};
                live_s2m.process(lc);
                live_rms_an.analyze(lc);
                live_rms.store(lc.features.rms, std::memory_order_relaxed);
                if (lc.features.rms >= 0.01f) {
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

            if (!rec) continue;

            {
                std::lock_guard<std::mutex> lk(rb_mtx);
                rb.write_overwrite(in.data(), nsamp);
            }

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

                    std::vector<SliceCandidate> slice_candidates;
                    slice_candidates.reserve(regions.size());
                    for (auto& region : regions) {
                        slice_candidates.push_back(SliceCandidate{
                            region, one_sec_window.data(), channels, rate, {}
                        });
                    }

                    for (auto& pre : preprocessors) {
                        std::vector<SliceCandidate> surviving;
                        surviving.reserve(slice_candidates.size());
                        for (auto& c : slice_candidates) {
                            if (pre->process(c)) surviving.push_back(c);
                        }
                        slice_candidates = std::move(surviving);
                    }

                    for (auto& c : slice_candidates) {
                        for (auto& an : analyzers) {
                            an->analyze(c);
                        }
                    }

                    for (auto& filt : filters) {
                        std::vector<SliceCandidate> surviving;
                        surviving.reserve(slice_candidates.size());
                        for (auto& c : slice_candidates) {
                            if (filt->passes(c)) surviving.push_back(c);
                        }
                        slice_candidates = std::move(surviving);
                    }

                    for (auto& c : slice_candidates) {
                        size_t start_sample    = c.region.start_frame * channels;
                        size_t n_samples_slice = c.region.length_frames * channels;

                        // route to record target bank
                        SliceStore& target = (record_target.load(std::memory_order_relaxed) == 0)
                                             ? store_a : store_b;
                        int id = target.add_slice(
                            c.window + start_sample,
                            (uint32_t)n_samples_slice,
                            c.features
                        );

                        new_slice_signal.fetch_add(1, std::memory_order_relaxed);
                        std::fprintf(stderr, "[slice] id=%d bank=%c start=%.1fms len=%zu samples\n",
                            id,
                            (record_target.load() == 0) ? 'A' : 'B',
                            1000.0 * c.region.start_frame / rate, n_samples_slice);
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

        const int MAX_VOICES = 30;
        struct GrainVoice {
            Slice       slice;
            SliceStore* src;        // which store's corpus to read from
            uint32_t    pos;
            uint32_t    play_end;
            bool        active;
            float       attack_frac;
            float       decay_frac;
        };
        GrainVoice voices[MAX_VOICES] = {};
        for (auto& v : voices) v.src = &store_a;  // safe default

        struct EnvPreset { float attack; float decay; };
        const EnvPreset kEnvShapes[3] = {
            {0.05f, 0.95f},
            {0.10f, 0.10f},
            {0.85f, 0.15f},
        };
        int next_voice = 0;

        uint32_t rng = 0x12345678u;

        // reverb + ping-pong delay state
        static constexpr size_t kDelayL = 14400;
        static constexpr size_t kDelayR = 21600;
        std::vector<float> dly_L(kDelayL, 0.0f), dly_R(kDelayR, 0.0f);
        size_t dly_wL = 0, dly_wR = 0;
        size_t dly_fill_L = 0, dly_fill_R = 0;

        struct CombFilter {
            std::vector<float> buf;
            size_t w = 0;
            void init(size_t n) { buf.assign(n, 0.0f); }
            float process(float x, float fb) {
                float y = buf[w];
                buf[w] = x + y * fb;
                if (++w >= buf.size()) w = 0;
                return y;
            }
        };
        struct AllpassFilter {
            std::vector<float> buf;
            size_t w = 0;
            const float g = 0.5f;
            void init(size_t n) { buf.assign(n, 0.0f); }
            float process(float x) {
                float d = buf[w];
                float y = -g * x + d;
                buf[w] = x + g * d;
                if (++w >= buf.size()) w = 0;
                return y;
            }
        };

        const size_t kCombs[4] = {2473, 2767, 3109, 3557};
        CombFilter    combs_L[4], combs_R[4];
        AllpassFilter aps_L[2],   aps_R[2];
        for (int i = 0; i < 4; i++) {
            combs_L[i].init(kCombs[i]);
            combs_R[i].init(kCombs[i] + 23 * (i + 1));
        }
        aps_L[0].init(480); aps_L[1].init(161);
        aps_R[0].init(500); aps_R[1].init(171);

        float master_env        = 0.0f;
        float activity_level    = 0.0f;  // stays 0 until live audio is detected
        float voice_scale_smooth = 1.0f;  // smoothed 1/sqrt(n_active) to avoid per-period amplitude jumps
        // frozen query features: only update when signal is present so sustained
        // retriggering keeps targeting the last audible sound, not silence
        float frozen_tonal   = 0.0f;
        float frozen_rolloff = 1000.0f;
        float frozen_rms     = 0.05f;
        static constexpr float kMasterAttack  = 0.10f;
        static constexpr float kMasterRelease = 0.008f;
        static constexpr float kMasterGain    = 0.7f;              // ~-3dB overall level
        static constexpr float kClipKnee      = 26000.0f;          // transparent below ~-2dBFS
        static constexpr float kClipHeadroom  = 32767.0f - kClipKnee;

        auto next_switch = std::chrono::steady_clock::now();

        while (g_run.load(std::memory_order_relaxed)) {
            size_t got = 0;

            bool playing = play_enabled.load(std::memory_order_relaxed);
            bool expl    = explicit_mode.load(std::memory_order_relaxed);
            int interval = grain_interval_ms.load(std::memory_order_relaxed);
            int glen_ms  = grain_length_ms.load(std::memory_order_relaxed);
            float env_p  = env_pos.load(std::memory_order_relaxed);

            if (!playing) {
                got = 0;
            }
            else {
                float dw_check  = dry_wet.load(std::memory_order_relaxed);
                bool  grains_on = (dw_check >= 0.01f);  // skip grain engine entirely when fully dry

                auto now = std::chrono::steady_clock::now();
                bool  lv_valid  = live_valid.load(std::memory_order_acquire);
                float lv_rms    = live_rms.load(std::memory_order_relaxed);
                if (lv_valid && lv_rms >= 0.01f) {
                    activity_level = 1.0f;
                } else {
                    float hl = activity_halflife_ms.load(std::memory_order_relaxed);
                    if (hl < 1e8f) {
                        float period_ms = (float)samples_per_period / (float)rate * 1000.0f;
                        activity_level *= powf(0.5f, period_ms / hl);
                    }
                    // hl >= 1e8: hold — don't decay
                }
                bool live_active = (activity_level >= 0.01f);
                if (grains_on && now >= next_switch && live_active) {
                    rng = xorshift32(rng);

                    // probabilistic crossfade: roll dice, pick from one bank, fallback to other
                    float cf   = crossfade_pos.load(std::memory_order_relaxed);
                    float roll = (float)(rng % 10000) / 10000.0f;
                    SliceStore* pick_store     = (roll < cf) ? &store_b : &store_a;
                    SliceStore* fallback_store = (roll < cf) ? &store_a : &store_b;

                    int new_id = -1;
                    SliceStore* chosen_store = nullptr;
                    bool found_id = false;

                    float lt  = live_tonal.load(std::memory_order_relaxed);
                    float lr  = live_rolloff.load(std::memory_order_relaxed);
                    float lrm = live_rms.load(std::memory_order_relaxed);
                    int   gk  = grain_k.load(std::memory_order_relaxed);

                    // freeze query features while signal is present; hold last value during silence
                    if (lv_valid && lv_rms >= 0.01f) {
                        frozen_tonal   = lt;
                        frozen_rolloff = lr;
                        frozen_rms     = lrm;
                    }

                    float cur_radius = 0.0f;
                    if (engine_mode.load(std::memory_order_relaxed) == 1) {
                        // exploration: query by encoder XYZ centroid
                        float ex = explore_x.load(std::memory_order_relaxed);
                        float ey = explore_y.load(std::memory_order_relaxed);
                        float ez = explore_z.load(std::memory_order_relaxed);
                        if (pick_store->closest_k_id_xyz(ex, ey, ez, gk, rng, new_id, cur_radius)) {
                            found_id = true; chosen_store = pick_store;
                        } else if (fallback_store->closest_k_id_xyz(ex, ey, ez, gk, rng, new_id, cur_radius)) {
                            found_id = true; chosen_store = fallback_store;
                        }
                    } else if (lv_valid) {
                        // analysis: query by frozen features (hold last audible sound during silence)
                        if (pick_store->closest_k_id(frozen_tonal, frozen_rolloff, frozen_rms, gk, rng, new_id)) {
                            found_id = true; chosen_store = pick_store;
                        } else if (fallback_store->closest_k_id(frozen_tonal, frozen_rolloff, frozen_rms, gk, rng, new_id)) {
                            found_id = true; chosen_store = fallback_store;
                        }
                    }
                    if (!found_id) {
                        if (pick_store->random_id(rng, new_id)) {
                            found_id = true; chosen_store = pick_store;
                        } else if (fallback_store->random_id(rng, new_id)) {
                            found_id = true; chosen_store = fallback_store;
                        }
                    }

                    search_radius.store(cur_radius, std::memory_order_relaxed);

                    if (found_id && chosen_store) {
                        Slice tmp;
                        if (chosen_store->get(new_id, tmp)) {
                            rng = xorshift32(rng);
                            int jitter_range = glen_ms / 4;
                            int jitter_ms = (jitter_range > 0)
                                ? (int)(rng % (2 * jitter_range + 1)) - jitter_range : 0;
                            int actual_ms = std::max(20, glen_ms + jitter_ms);
                            uint32_t play_end = std::min(
                                (uint32_t)((uint64_t)rate * channels * actual_ms / 1000),
                                tmp.length);

                            float ep = std::max(0.0f, std::min(2.0f, env_p));
                            int   lo = (int)ep;
                            int   hi = std::min(lo + 1, 2);
                            float t_ep = ep - (float)lo;
                            float att = kEnvShapes[lo].attack * (1.0f - t_ep) + kEnvShapes[hi].attack * t_ep;
                            float dec = kEnvShapes[lo].decay  * (1.0f - t_ep) + kEnvShapes[hi].decay  * t_ep;

                            GrainVoice& v = voices[next_voice % MAX_VOICES];
                            v.slice       = tmp;
                            v.src         = chosen_store;
                            v.pos         = 0;
                            v.play_end    = play_end;
                            v.active      = true;
                            v.attack_frac = att;
                            v.decay_frac  = dec;
                            next_voice++;
                            current_grain_id.store(new_id, std::memory_order_relaxed);
                            current_bank.store((chosen_store == &store_b) ? 1 : 0,
                                               std::memory_order_relaxed);

                            if (expl) {
                                std::fprintf(stderr, "[grain] id=%d bank=%c slice=%u jitter=%dms interval=%dms\n",
                                    tmp.id, (chosen_store == &store_b) ? 'B' : 'A',
                                    tmp.length, jitter_ms, interval);
                            }
                        }
                    }
                    next_switch = now + std::chrono::milliseconds(interval);
                }

                std::fill(mix.begin(), mix.end(), 0);
                if (grains_on) {
                    int n_active = 0;
                    for (auto& v : voices) if (v.active) n_active++;
                    float target_scale = (n_active > 1) ? 1.0f / sqrtf((float)n_active) : 1.0f;
                    voice_scale_smooth += (target_scale - voice_scale_smooth) * 0.25f;
                    float voice_scale = voice_scale_smooth;

                    // lock both stores for the entire mixing loop
                    std::lock_guard<std::mutex> la(store_a.mtx);
                    std::lock_guard<std::mutex> lb(store_b.mtx);
                    for (auto& v : voices) {
                        if (!v.active) continue;
                        uint32_t remain = (v.pos < v.play_end) ? (v.play_end - v.pos) : 0;
                        uint32_t take = std::min(remain, (uint32_t)samples_per_period);
                        if (take > 0) {
                            const int16_t* base = v.src->corpus.data() + v.slice.corpus_start;
                            bool mono_grain = mono_passthrough.load(std::memory_order_relaxed);
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
                                uint32_t src = (mono_grain && (i % 2 == 1)) ? (v.pos + i - 1) : (v.pos + i);
                                mix[i] += (int32_t)((float)base[src] * env * v.slice.features.gain * voice_scale);
                            }
                            v.pos += take;
                        }
                        if (v.pos >= v.play_end) v.active = false;
                    }
                }  // end grains_on mixing block

                float env_target = (grains_on && live_active) ? 1.0f : 0.0f;
                float env_step   = live_active ? kMasterAttack : kMasterRelease;
                if (master_env < env_target) {
                    master_env = std::min(master_env + env_step, 1.0f);
                } else {
                    master_env = std::max(master_env - env_step, 0.0f);
                }

                for (size_t i = 0; i < samples_per_period; i++) {
                    float s     = (float)mix[i] * master_env * kMasterGain;
                    float abs_s = s < 0.0f ? -s : s;
                    float limited;
                    if (abs_s <= kClipKnee) {
                        limited = s;   // transparent below knee
                    } else {
                        float excess = abs_s - kClipKnee;
                        float sat    = kClipKnee + kClipHeadroom * tanhf(excess / kClipHeadroom);
                        limited = s < 0.0f ? -sat : sat;
                    }
                    int32_t ls = (int32_t)limited;
                    out[i] = (int16_t)(ls > 32767 ? 32767 : ls < -32768 ? -32768 : ls);
                }
                got = samples_per_period;
            }

            if (got < samples_per_period) {
                std::fill(out.begin() + got, out.end(), 0);
            }

            {
                float dw   = dry_wet.load(std::memory_order_relaxed);
                bool  mono = mono_passthrough.load(std::memory_order_relaxed);
                size_t pt_got = 0;
                {
                    std::lock_guard<std::mutex> lk(pt_rb_mtx);
                    pt_got = pt_rb.copy_latest(pt_buf.data(), samples_per_period, samples_per_period * 2);
                }
                for (size_t i = 0; i < samples_per_period; i += 2) {
                    float dry_L = (i     < pt_got) ? (float)pt_buf[i]     * (1.0f - dw) : 0.0f;
                    float dry_R = (i + 1 < pt_got) ? (float)pt_buf[i + 1] * (1.0f - dw) : 0.0f;
                    if (mono) dry_R = dry_L;
                    float wet_L = (float)out[i]     * dw;
                    float wet_R = (float)out[i + 1] * dw;
                    int32_t sL = (int32_t)(dry_L + wet_L);
                    int32_t sR = (int32_t)(dry_R + wet_R);
                    out[i]     = (int16_t)(sL > 32767 ? 32767 : sL < -32768 ? -32768 : sL);
                    out[i + 1] = (int16_t)(sR > 32767 ? 32767 : sR < -32768 ? -32768 : sR);
                }
            }

            // reverb + ping-pong delay send
            float fx = fx_amount.load(std::memory_order_relaxed);
            if (fx > 0.005f) {
                float delay_fb     = fx * 0.65f;
                float reverb_decay = 0.5f + fx * 0.45f;

                for (size_t i = 0; i < samples_per_period; i += 2) {
                    float inL  = (float)out[i];
                    float inR  = (float)out[i + 1];
                    float mono = (inL + inR) * 0.5f;

                    float dL = (dly_fill_L >= kDelayL) ? dly_L[dly_wL] : 0.0f;
                    float dR = (dly_fill_R >= kDelayR) ? dly_R[dly_wR] : 0.0f;
                    dly_L[dly_wL] = mono + dR * delay_fb;
                    dly_R[dly_wR] = dL;
                    if (++dly_wL >= kDelayL) dly_wL = 0;
                    if (++dly_wR >= kDelayR) dly_wR = 0;
                    if (dly_fill_L < kDelayL) ++dly_fill_L;
                    if (dly_fill_R < kDelayR) ++dly_fill_R;

                    float rvL = 0.0f, rvR = 0.0f;
                    for (int c = 0; c < 4; c++) {
                        rvL += combs_L[c].process(mono, reverb_decay);
                        rvR += combs_R[c].process(mono, reverb_decay);
                    }
                    float rv_norm = (1.0f - reverb_decay) * 0.25f;
                    rvL = aps_L[1].process(aps_L[0].process(rvL * rv_norm));
                    rvR = aps_R[1].process(aps_R[0].process(rvR * rv_norm));

                    float wetL = (dL + rvL) * fx * 0.5f;
                    float wetR = (dR + rvR) * fx * 0.5f;

                    int32_t sL = (int32_t)out[i]     + (int32_t)wetL;
                    int32_t sR = (int32_t)out[i + 1] + (int32_t)wetR;
                    out[i]     = (int16_t)(sL >  32767 ?  32767 : sL < -32768 ? -32768 : sL);
                    out[i + 1] = (int16_t)(sR >  32767 ?  32767 : sR < -32768 ? -32768 : sR);
                }
            }

            snd_pcm_sframes_t w = snd_pcm_writei(pb, out.data(), period_frames);
            if (w < 0) recover_if_xrun(pb, (int)w, "playback");
        }
    });

    // ---- input gain presets (used by encoder menu and switch thread) ----
    static constexpr const char* kMixerCard       = "hw:2";
    static constexpr float kGainLine_dB        =  0.0f;
    static constexpr float kGainInstrument_dB  = 40.0f;
    static constexpr float kGainDualJack_dB    =  0.0f;
    static constexpr float kGainMic_dB         = 30.0f;

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

        i2c_write_reg(0x00, 0xFF);
        i2c_write_reg(0x01, 0xFF);
        i2c_write_reg(0x0C, 0xFF);
        i2c_write_reg(0x0D, 0b00001111);

        static constexpr uint8_t kABit[4] = { 7, 5, 3, 1 };
        static constexpr uint8_t kBBit[4] = { 6, 4, 2, 0 };
        static constexpr uint8_t kBtnBit[4] = { 0, 1, 2, 3 };

        uint8_t init_a = i2c_read_reg(0x12);
        uint8_t prev_a[4];
        for (int i = 0; i < 4; i++)
            prev_a[i] = (init_a >> kABit[i]) & 1;

        int     counters[4]  = {};
        uint8_t prev_btn_raw = 0xFF;

        // enc4 long-press detection
        auto enc4_press_start = std::chrono::steady_clock::time_point{};
        bool enc4_was_down    = false;

        // local menu state (encoder thread is the sole writer)
        BankMenuDisplay menu;

        // name-entry state (pages 5=SaveA, 6=SaveB)
        static const char kNameChars[] = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_";
        static constexpr int kNameCharsLen = 39;
        static constexpr int kNameMaxLen   = 12;
        int name_char_idx[kNameMaxLen] = {};
        auto apply_name = [&]() {
            for (int i = 0; i < kNameMaxLen; i++)
                menu.name_buf[i] = kNameChars[name_char_idx[i]];
            menu.name_buf[kNameMaxLen] = '\0';
        };
        auto enter_name_page = [&](int page) {
            std::memset(name_char_idx, 0, sizeof(name_char_idx));
            menu.cursor = 0;
            menu.page   = page;
            apply_name();
        };

        while (g_run.load(std::memory_order_relaxed)) {
            // sync local menu state with any external changes (e.g. mbtn back press)
            {
                std::lock_guard<std::mutex> lk(bank_menu_mtx);
                if (menu.open && !bank_menu_disp.open) {
                    menu.open = false;
                } else if (menu.open && bank_menu_disp.page != menu.page) {
                    menu.page   = bank_menu_disp.page;
                    menu.cursor = bank_menu_disp.cursor;
                    std::memcpy(menu.delete_file, bank_menu_disp.delete_file, sizeof(menu.delete_file));
                }
            }

            uint8_t port_a = i2c_read_reg(0x12);
            uint8_t port_b = i2c_read_reg(0x13);

            // decode all four encoders, compute deltas
            int deltas[4] = {};
            for (int i = 0; i < 4; i++) {
                uint8_t a = (port_a >> kABit[i]) & 1;
                uint8_t b = (port_a >> kBBit[i]) & 1;
                if (a != prev_a[i]) {
                    int d = (a == 1) ? (b == 0 ? -1 : 1) : (b == 1 ? -1 : 1);
                    counters[i] += d;
                    deltas[i] = d;
                    prev_a[i] = a;
                }
            }

            // enc4 rotation: navigate menu when open
            if (deltas[3] != 0) {
                if (menu.open) {
                    if (menu.page == 5 || menu.page == 6) {
                        // name entry: cycle character at cursor
                        int pos = menu.cursor;
                        name_char_idx[pos] = ((name_char_idx[pos] + deltas[3]) % kNameCharsLen
                                              + kNameCharsLen) % kNameCharsLen;
                        apply_name();
                        std::lock_guard<std::mutex> lk(bank_menu_mtx);
                        std::memcpy(bank_menu_disp.name_buf, menu.name_buf, sizeof(menu.name_buf));
                        bank_menu_disp.cursor = pos;
                    } else {
                        int max_items = 0;
                        if (menu.page == 0) max_items = 13;
                        else if (menu.page == 1 || menu.page == 2)
                            max_items = (int)menu.file_list.size();
                        if (max_items > 0) {
                            menu.cursor = (menu.cursor + deltas[3] + max_items) % max_items;
                            std::lock_guard<std::mutex> lk(bank_menu_mtx);
                            bank_menu_disp.cursor = menu.cursor;
                        }
                    }
                } else {
                    // menu closed: cycle tonic root
                    static const char* kKeyNames[12] = {"A","Bb","B","C","C#","D","Eb","E","F","F#","G","Ab"};
                    int idx = (deltas[3] < 0)
                        ? (tonal_root_idx.load() + 11) % 12
                        : (tonal_root_idx.load() +  1) % 12;
                    tonal_root_idx.store(idx);
                    ScaleType sc = tonal_minor.load() ? ScaleType::Minor : ScaleType::Major;
                    auto update_tonal_rot = [&](SliceStore& st) {
                        std::lock_guard<std::mutex> lk(st.mtx);
                        for (auto& [id, slice] : st.slices)
                            slice.features.tonal_alignment_score =
                                TonalAlignmentAnalyzer::tonal_score_from_chroma(
                                    slice.features.chroma_energy, idx, sc);
                    };
                    update_tonal_rot(store_a);
                    update_tonal_rot(store_b);
                    { std::lock_guard<std::mutex> lk(toast_mtx);
                      std::snprintf(toast_disp.name,  sizeof(toast_disp.name),  "Key");
                      std::snprintf(toast_disp.value, sizeof(toast_disp.value), "%s %s",
                          kKeyNames[idx], tonal_minor.load() ? "minor" : "major");
                      toast_disp.updated = std::chrono::steady_clock::now(); }
                }
            }
            // enc1-3 rotation: mode-dependent
            {
                int mode = engine_mode.load(std::memory_order_relaxed);
                auto clamp = [](float v, float lo, float hi) {
                    return v < lo ? lo : v > hi ? hi : v;
                };
                for (int i = 0; i < 3; i++) {
                    if (deltas[i] == 0) continue;
                    if (mode == 0) {
                        // analysis: enc1=theta_y enc2=theta_x enc3=zoom
                        if (i == 0) {
                            view_theta_y.store(view_theta_y.load() + deltas[i] * 0.05f,
                                               std::memory_order_relaxed);
                        } else if (i == 1) {
                            view_theta_x.store(clamp(view_theta_x.load() + deltas[i] * 0.02f,
                                                     -1.2f, 1.2f), std::memory_order_relaxed);
                        } else {
                            view_zoom.store(clamp(view_zoom.load() + deltas[i] * 0.05f,
                                                  0.3f, 3.0f), std::memory_order_relaxed);
                        }
                    } else {
                        // exploration: enc1=X enc2=Y enc3=Z centroid position
                        if (i == 0)
                            explore_x.store(clamp(explore_x.load() + deltas[i] * 0.01f,
                                                  -0.5f, 0.5f), std::memory_order_relaxed);
                        else if (i == 1)
                            explore_y.store(clamp(explore_y.load() + deltas[i] * 0.01f,
                                                  -0.5f, 0.5f), std::memory_order_relaxed);
                        else
                            explore_z.store(clamp(explore_z.load() + deltas[i] * 0.01f,
                                                  -0.5f, 0.5f), std::memory_order_relaxed);
                    }
                }
            }

            // enc1-3 buttons: reset view (analysis) or individual axis (exploration)
            for (int i = 0; i < 3; i++) {
                uint8_t mask = 1u << kBtnBit[i];
                uint8_t cur  = port_b & mask;
                uint8_t prev = prev_btn_raw & mask;
                if (cur != prev && cur == 0) {  // falling edge = press (active low)
                    int mode = engine_mode.load(std::memory_order_relaxed);
                    if (mode == 0) {
                        if      (i == 0) { view_theta_y.store(0.4f, std::memory_order_relaxed); std::fprintf(stderr, "[encoder1] theta_y reset\n"); }
                        else if (i == 1) { view_theta_x.store(0.3f, std::memory_order_relaxed); std::fprintf(stderr, "[encoder2] theta_x reset\n"); }
                        else             { view_zoom.store(2.0f,    std::memory_order_relaxed); std::fprintf(stderr, "[encoder3] zoom reset\n"); }
                    } else {
                        if      (i == 0) { explore_x.store(0.0f, std::memory_order_relaxed); std::fprintf(stderr, "[encoder1] x reset\n"); }
                        else if (i == 1) { explore_y.store(0.0f, std::memory_order_relaxed); std::fprintf(stderr, "[encoder2] y reset\n"); }
                        else             { explore_z.store(0.0f, std::memory_order_relaxed); std::fprintf(stderr, "[encoder3] z reset\n"); }
                    }
                }
            }

            // enc4 button: long-press = bank menu, short-press = gain confirm / menu advance
            {
                uint8_t mask = 1u << kBtnBit[3];
                bool enc4_down = (port_b & mask) == 0;  // active low

                if (enc4_down && !enc4_was_down) {
                    enc4_press_start = std::chrono::steady_clock::now();
                    enc4_was_down    = true;
                }
                if (!enc4_down && enc4_was_down) {
                    enc4_was_down = false;
                    auto held_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - enc4_press_start).count();

                    if (held_ms >= 500) {
                        // long press: toggle bank menu
                        menu.open = !menu.open;
                        if (menu.open) {
                            menu.page   = 0;
                            menu.cursor = 0;
                            menu.bank_name_a    = bank_name_a;
                            menu.bank_name_b    = bank_name_b;
                            menu.has_file_a     = !cur_bank_a_file.empty();
                            menu.has_file_b     = !cur_bank_b_file.empty();
                            menu.record_target  = record_target.load();
                            menu.trs_instrument = trs_instrument_gain.load();
                            menu.file_list.clear();
                            std::fprintf(stderr, "[menu] opened\n");
                        } else {
                            std::fprintf(stderr, "[menu] closed\n");
                        }
                        std::lock_guard<std::mutex> lk(bank_menu_mtx);
                        bank_menu_disp = menu;
                    } else {
                        // short press
                        if (menu.open) {
                            // advance / select current item
                            if (menu.page == 0) {
                                switch (menu.cursor) {
                                    case 0: {  // Slot A — overwrite if file loaded
                                        if (!cur_bank_a_file.empty()) {
                                            if (save_bank(store_a, banks_dir() + cur_bank_a_file, bank_name_a)) {
                                                menu.bank_name_a = bank_name_a;
                                                save_config(cur_bank_a_file, cur_bank_b_file, record_target.load());
                                                std::fprintf(stderr, "[bank] overwrite A: %s\n", cur_bank_a_file.c_str());
                                            }
                                        }
                                        break;
                                    }
                                    case 1: {  // Slot B — overwrite if file loaded
                                        if (!cur_bank_b_file.empty()) {
                                            if (save_bank(store_b, banks_dir() + cur_bank_b_file, bank_name_b)) {
                                                menu.bank_name_b = bank_name_b;
                                                save_config(cur_bank_a_file, cur_bank_b_file, record_target.load());
                                                std::fprintf(stderr, "[bank] overwrite B: %s\n", cur_bank_b_file.c_str());
                                            }
                                        }
                                        break;
                                    }
                                    case 2:  // Load -> A
                                        menu.page = 1;
                                        menu.cursor = 0;
                                        menu.file_list = list_banks();
                                        break;
                                    case 3:  // Load -> B
                                        menu.page = 2;
                                        menu.cursor = 0;
                                        menu.file_list = list_banks();
                                        break;
                                    case 4:  // Save A as new → name entry
                                        enter_name_page(5);
                                        break;
                                    case 5:  // Save B as new → name entry
                                        enter_name_page(6);
                                        break;
                                    case 6:  // Clear A
                                        menu.page   = 3;
                                        menu.cursor = 0;
                                        break;
                                    case 7:  // Clear B
                                        menu.page   = 4;
                                        menu.cursor = 0;
                                        break;
                                    case 8: {  // Record target toggle
                                        int rt = record_target.load();
                                        rt = 1 - rt;
                                        record_target.store(rt);
                                        menu.record_target = rt;
                                        std::fprintf(stderr, "[bank] record target -> %c\n", rt == 0 ? 'A' : 'B');
                                        break;
                                    }
                                    case 9: {  // Gain toggle (TRS instrument / line)
                                        bool instr = !trs_instrument_gain.load();
                                        trs_instrument_gain.store(instr);
                                        mono_passthrough.store(instr, std::memory_order_relaxed);
                                        hpf_enabled.store(instr,      std::memory_order_relaxed);
                                        set_pga_gain_db(kMixerCard,
                                            instr ? kGainInstrument_dB : kGainLine_dB);
                                        menu.trs_instrument = instr;
                                        {
                                            std::lock_guard<std::mutex> lk(toast_mtx);
                                            const char* label = instr ? "instrument" : "line";
                                            std::strncpy(toast_disp.name,  "gain",  31);
                                            std::strncpy(toast_disp.value, label,   31);
                                            toast_disp.updated = std::chrono::steady_clock::now();
                                        }
                                        std::fprintf(stderr, "[gain] -> %s %.1fdB\n",
                                            instr ? "instrument" : "line",
                                            instr ? kGainInstrument_dB : kGainLine_dB);
                                        break;
                                    }
                                    case 10: {  // Mode toggle
                                        int m = 1 - engine_mode.load();
                                        engine_mode.store(m);
                                        std::fprintf(stderr, "[mode] -> %s\n",
                                            m == 0 ? "analysis" : "exploration");
                                        break;
                                    }
                                    case 11:  // Shutdown → confirm page
                                        menu.page   = 9;
                                        menu.cursor = 0;
                                        break;
                                    case 12:  // Exit
                                        menu.open = false;
                                        std::fprintf(stderr, "[menu] closed\n");
                                        break;
                                }
                            } else if (menu.page == 1 || menu.page == 2) {
                                // load selected file
                                if (!menu.file_list.empty()) {
                                    std::string fname = menu.file_list[menu.cursor];
                                    std::string path  = banks_dir() + fname;
                                    std::string loaded_name;
                                    SliceStore& target = (menu.page == 1) ? store_a : store_b;
                                    if (load_bank(target, path, loaded_name)) {
                                        if (menu.page == 1) {
                                            bank_name_a = loaded_name;
                                            cur_bank_a_file = fname;
                                            menu.bank_name_a = loaded_name;
                                        } else {
                                            bank_name_b = loaded_name;
                                            cur_bank_b_file = fname;
                                            menu.bank_name_b = loaded_name;
                                        }
                                        save_config(cur_bank_a_file, cur_bank_b_file,
                                                    record_target.load());
                                        std::fprintf(stderr, "[bank] loaded slot %c: %s\n",
                                            (menu.page == 1) ? 'A' : 'B', fname.c_str());
                                    }
                                }
                                menu.page   = 0;
                                menu.cursor = 0;
                            } else if (menu.page == 3) {
                                // confirm clear A
                                {
                                    std::lock_guard<std::mutex> lk(store_a.mtx);
                                    store_a.corpus.clear();
                                    store_a.slices.clear();
                                    store_a.next_id = 0;
                                }
                                bank_name_a = "";
                                cur_bank_a_file = "";
                                menu.bank_name_a = "";
                                save_config(cur_bank_a_file, cur_bank_b_file, record_target.load());
                                std::fprintf(stderr, "[bank] cleared slot A\n");
                                menu.page   = 0;
                                menu.cursor = 0;
                            } else if (menu.page == 4) {
                                // confirm clear B
                                {
                                    std::lock_guard<std::mutex> lk(store_b.mtx);
                                    store_b.corpus.clear();
                                    store_b.slices.clear();
                                    store_b.next_id = 0;
                                }
                                bank_name_b = "";
                                cur_bank_b_file = "";
                                menu.bank_name_b = "";
                                save_config(cur_bank_a_file, cur_bank_b_file, record_target.load());
                                std::fprintf(stderr, "[bank] cleared slot B\n");
                                menu.page   = 0;
                                menu.cursor = 0;
                            } else if (menu.page == 9) {
                                // confirm shutdown
                                std::fprintf(stderr, "[menu] shutdown confirmed\n");
                                std::lock_guard<std::mutex> lk(bank_menu_mtx);
                                bank_menu_disp = menu;
                                std::system("poweroff");
                            } else if (menu.page == 7 || menu.page == 8) {
                                // confirm delete
                                if (menu.delete_file[0]) {
                                    std::error_code ec;
                                    std::filesystem::remove(banks_dir() + menu.delete_file, ec);
                                    std::fprintf(stderr, "[bank] deleted: %s\n", menu.delete_file);
                                    menu.delete_file[0] = '\0';
                                }
                                menu.page      = (menu.page == 7) ? 1 : 2;
                                menu.cursor    = 0;
                                menu.file_list = list_banks();
                            } else if (menu.page == 5 || menu.page == 6) {
                                // name entry: advance cursor or confirm save
                                if (menu.cursor < kNameMaxLen - 1) {
                                    menu.cursor++;
                                } else {
                                    // last position: confirm save
                                    std::string raw(menu.name_buf, kNameMaxLen);
                                    while (!raw.empty() && raw.back() == ' ') raw.pop_back();
                                    if (raw.empty()) raw = "bank";
                                    auto t = std::time(nullptr);
                                    struct tm tmi; localtime_r(&t, &tmi);
                                    char ts[32]; std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tmi);
                                    std::string fname = raw + "_" + ts + ".gran";
                                    bool saving_a = (menu.page == 5);
                                    SliceStore& tgt = saving_a ? store_a : store_b;
                                    if (save_bank(tgt, banks_dir() + fname, raw)) {
                                        if (saving_a) {
                                            bank_name_a = raw; cur_bank_a_file = fname;
                                            menu.bank_name_a = raw;
                                        } else {
                                            bank_name_b = raw; cur_bank_b_file = fname;
                                            menu.bank_name_b = raw;
                                        }
                                        save_config(cur_bank_a_file, cur_bank_b_file, record_target.load());
                                        std::fprintf(stderr, "[bank] saved %c: %s\n", saving_a?'A':'B', fname.c_str());
                                    }
                                    menu.page   = 0;
                                    menu.cursor = 0;
                                }
                            }
                            std::lock_guard<std::mutex> lk(bank_menu_mtx);
                            bank_menu_disp = menu;
                        } else {
                            // enc4 short-press, menu closed: toggle major/minor
                            static const char* kKeyNames[12] = {"A","Bb","B","C","C#","D","Eb","E","F","F#","G","Ab"};
                            bool minor = !tonal_minor.load();
                            tonal_minor.store(minor);
                            int idx = tonal_root_idx.load();
                            ScaleType sc = minor ? ScaleType::Minor : ScaleType::Major;
                            auto update_tonal_btn = [&](SliceStore& st) {
                                std::lock_guard<std::mutex> lk(st.mtx);
                                for (auto& [id, slice] : st.slices)
                                    slice.features.tonal_alignment_score =
                                        TonalAlignmentAnalyzer::tonal_score_from_chroma(
                                            slice.features.chroma_energy, idx, sc);
                            };
                            update_tonal_btn(store_a);
                            update_tonal_btn(store_b);
                            { std::lock_guard<std::mutex> lk(toast_mtx);
                              std::snprintf(toast_disp.name,  sizeof(toast_disp.name),  "Key");
                              std::snprintf(toast_disp.value, sizeof(toast_disp.value), "%s %s",
                                  kKeyNames[idx], minor ? "minor" : "major");
                              toast_disp.updated = std::chrono::steady_clock::now(); }
                        }
                    }
                }
            }

            prev_btn_raw = port_b;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        close(fd);
    });

    // ---- MCP3008 potentiometer thread ----
    std::thread mcp3008_thread([&] {
        int chip_fd = open("/dev/gpiochip4", O_RDONLY);
        if (chip_fd < 0) {
            std::fprintf(stderr, "[pot] failed to open /dev/gpiochip4\n");
            return;
        }

        struct gpio_v2_line_request out_req = {};
        out_req.offsets[0] = 11;
        out_req.offsets[1] = 10;
        out_req.offsets[2] = 25;
        out_req.num_lines  = 3;
        out_req.config.flags = GPIO_V2_LINE_FLAG_OUTPUT;
        out_req.config.attrs[0].attr.id     = GPIO_V2_LINE_ATTR_ID_OUTPUT_VALUES;
        out_req.config.attrs[0].attr.values = 0b100;
        out_req.config.attrs[0].mask        = 0b111;
        out_req.config.num_attrs = 1;
        std::strncpy(out_req.consumer, "engine-mcp-out", sizeof(out_req.consumer) - 1);

        if (ioctl(chip_fd, GPIO_V2_GET_LINE_IOCTL, &out_req) < 0) {
            std::fprintf(stderr, "[pot] failed to get output GPIO lines\n");
            close(chip_fd);
            return;
        }

        struct gpio_v2_line_request in_req = {};
        in_req.offsets[0] = 9;
        in_req.num_lines  = 1;
        in_req.config.flags = GPIO_V2_LINE_FLAG_INPUT;
        std::strncpy(in_req.consumer, "engine-mcp-in", sizeof(in_req.consumer) - 1);

        if (ioctl(chip_fd, GPIO_V2_GET_LINE_IOCTL, &in_req) < 0) {
            std::fprintf(stderr, "[pot] failed to get input GPIO line\n");
            close(chip_fd);
            close(out_req.fd);
            return;
        }
        close(chip_fd);

        auto set_pin = [&](uint64_t bit, int val) {
            struct gpio_v2_line_values v = {};
            v.bits = val ? bit : 0;
            v.mask = bit;
            ioctl(out_req.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &v);
        };
        auto get_miso = [&]() -> int {
            struct gpio_v2_line_values v = {};
            v.mask = 1;
            ioctl(in_req.fd, GPIO_V2_LINE_GET_VALUES_IOCTL, &v);
            return v.bits & 1;
        };

        auto read_channel = [&](int ch) -> int {
            set_pin(4, 0);
            int bits_out[] = {1, 1, (ch >> 2) & 1, (ch >> 1) & 1, ch & 1};
            for (int b : bits_out) {
                set_pin(1, 0);
                set_pin(2, b);
                set_pin(1, 1);
            }
            int result = 0;
            for (int i = 0; i < 12; i++) {
                set_pin(1, 0);
                set_pin(1, 1);
                if (i >= 2) result = (result << 1) | get_miso();
            }
            set_pin(1, 0);
            set_pin(4, 1);
            return result;
        };

        int prev[8] = {};
        static constexpr int kThreshold = 8;

        auto update_toast = [&](const char* name, const char* val) {
            std::lock_guard<std::mutex> lk(toast_mtx);
            std::strncpy(toast_disp.name,  name, 31); toast_disp.name[31]  = '\0';
            std::strncpy(toast_disp.value, val,  31); toast_disp.value[31] = '\0';
            toast_disp.updated = std::chrono::steady_clock::now();
        };

        while (g_run.load(std::memory_order_relaxed)) {
            for (int ch = 0; ch < 8; ch++) {
                int val = read_channel(ch);
                int diff = val - prev[ch];
                if (diff < 0) diff = -diff;
                if (diff >= kThreshold) {
                    prev[ch] = val;
                    if (ch == 0) {
                        float norm = val / 1023.0f;
                        float s    = 0.5f + norm * 9.5f;
                        slicer_sensitivity.store(s, std::memory_order_relaxed);

                        // activity: exp decay halflife 50ms→5000ms over 0–90%, hold above 90%
                        float hl;
                        if (norm >= 0.90f) {
                            hl = 1e9f;
                        } else {
                            float t = norm / 0.90f;              // 0→1 over usable range
                            hl = 50.0f * powf(100.0f, t);        // 50ms → 5000ms
                        }
                        activity_halflife_ms.store(hl, std::memory_order_relaxed);

                        char buf[32];
                        if (norm >= 0.90f)
                            std::snprintf(buf, sizeof(buf), "%.1f / hold", s);
                        else if (hl < 1000.f)
                            std::snprintf(buf, sizeof(buf), "%.1f / %.0fms", s, hl);
                        else
                            std::snprintf(buf, sizeof(buf), "%.1f / %.1fs", s, hl / 1000.f);
                        update_toast("sense/activity", buf);
                        std::fprintf(stderr, "[pot] sensitivity=%.2f activity_hl=%.0fms\n", s, hl);
                    } else if (ch == 2) {
                        float ep = val / 1023.0f * 2.0f;
                        env_pos.store(ep, std::memory_order_relaxed);
                        const char* name = (ep < 0.75f) ? "percussive" : (ep < 1.25f) ? "sustained" : "swell";
                        // mirror kEnvShapes interpolation to show real att/dec values
                        static constexpr float kAtt[3] = {0.05f, 0.10f, 0.85f};
                        static constexpr float kDec[3] = {0.95f, 0.10f, 0.15f};
                        int lo = (ep >= 1.0f) ? 1 : 0;
                        float t = ep - (float)lo;
                        int att_pct = (int)((kAtt[lo] * (1.0f-t) + kAtt[lo+1] * t) * 100.0f + 0.5f);
                        int dec_pct = (int)((kDec[lo] * (1.0f-t) + kDec[lo+1] * t) * 100.0f + 0.5f);
                        char buf[32];
                        std::snprintf(buf, sizeof(buf), "%s A:%d%% D:%d%%", name, att_pct, dec_pct);
                        update_toast("envelope", buf);
                        std::fprintf(stderr, "[pot] env_pos=%.2f (%s) att=%d%% dec=%d%%\n", ep, name, att_pct, dec_pct);
                    } else if (ch == 1) {
                        int ms = (int)(2000.0f * std::powf(15.0f / 2000.0f, val / 1023.0f));
                        grain_interval_ms.store(ms, std::memory_order_relaxed);
                        char buf[32]; std::snprintf(buf, sizeof(buf), "%dms", ms);
                        update_toast("grain interval", buf);
                        std::fprintf(stderr, "[pot] grain_interval=%dms\n", ms);
                    } else if (ch == 3) {
                        int ms = (int)(20.0f * std::powf(100.0f, val / 1023.0f));
                        ms = std::min(ms, slicer->slice_ms);
                        grain_length_ms.store(ms, std::memory_order_relaxed);
                        char buf[32]; std::snprintf(buf, sizeof(buf), "%dms", ms);
                        update_toast("grain length", buf);
                        std::fprintf(stderr, "[pot] grain_length=%dms\n", ms);
                    } else if (ch == 5) {
                        // scale k with combined corpus size across both banks
                        size_t n = 0;
                        if (store_a.mtx.try_lock()) { n += store_a.slices.size(); store_a.mtx.unlock(); }
                        if (store_b.mtx.try_lock()) { n += store_b.slices.size(); store_b.mtx.unlock(); }
                        int k = std::max(1, (int)std::roundf(val / 1023.0f * 9.0f) + 1);
                        if (k > 10) k = 10;
                        grain_k.store(k, std::memory_order_relaxed);
                        char buf[32]; std::snprintf(buf, sizeof(buf), "%d", k);
                        update_toast("grain neighbors", buf);
                        std::fprintf(stderr, "[pot] grain_k=%d (corpus=%zu)\n", k, n);
                    } else if (ch == 4) {
                        float fx = val / 1023.0f;
                        fx_amount.store(fx, std::memory_order_relaxed);
                        char buf[32]; std::snprintf(buf, sizeof(buf), "%.0f%%", fx * 100.0f);
                        update_toast("reverb/delay", buf);
                        std::fprintf(stderr, "[pot] fx=%.2f\n", fx);
                    } else if (ch == 6) {
                        float dw = val / 1023.0f;
                        dry_wet.store(dw, std::memory_order_relaxed);
                        char buf[32];
                        if (dw < 0.02f)       std::snprintf(buf, sizeof(buf), "dry");
                        else if (dw > 0.98f)  std::snprintf(buf, sizeof(buf), "wet");
                        else                  std::snprintf(buf, sizeof(buf), "%.0f%% wet", dw * 100.0f);
                        update_toast("dry/wet", buf);
                        std::fprintf(stderr, "[pot] dry_wet=%.2f\n", dw);
                    } else if (ch == 7) {
                        float cf = 1.0f - val / 1023.0f;
                        crossfade_pos.store(cf, std::memory_order_relaxed);
                        char buf[32];
                        if (cf < 0.02f)       std::snprintf(buf, sizeof(buf), "full A");
                        else if (cf > 0.98f)  std::snprintf(buf, sizeof(buf), "full B");
                        else                  std::snprintf(buf, sizeof(buf), "A %.0f%%  B %.0f%%", (1.0f-cf)*100, cf*100);
                        update_toast("crossfade", buf);
                        std::fprintf(stderr, "[pot] crossfade=%.2f\n", cf);
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        close(out_req.fd);
        close(in_req.fd);
    });

    std::atomic<int> current_input_mode{-1};

    // ---- rotary switch thread ----
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
        auto gain_settled_at = std::chrono::steady_clock::now();
        static constexpr int kSettleMs = 1500;

        while (g_run.load(std::memory_order_relaxed)) {
            struct gpio_v2_line_values vals = {};
            vals.mask = 0b11;
            if (ioctl(req.fd, GPIO_V2_LINE_GET_VALUES_IOCTL, &vals) < 0) {
                std::fprintf(stderr, "[switch] read error\n");
                break;
            }

            int gpio5 = (vals.bits >> 0) & 1;
            int gpio6 = (vals.bits >> 1) & 1;
            int mode  = (gpio6 << 1) | gpio5;

            if (mode != prev_mode) {
                const char* names[] = {"???", "XLR mic mono", "Dual jack stereo", "TRS stereo"};
                std::fprintf(stderr, "[switch] mode=%s (GPIO5=%d GPIO6=%d)\n",
                    names[mode], gpio5, gpio6);

                float gain = kGainLine_dB;
                switch (mode) {
                    case 1: gain = kGainMic_dB;      break;
                    case 2: gain = kGainDualJack_dB; break;
                    case 3: gain = trs_instrument_gain.load() ? kGainInstrument_dB : kGainLine_dB; break;
                }
                if (mode != 0) {
                    if (set_pga_gain_db(kMixerCard, gain))
                        std::fprintf(stderr, "[gain] set %.1fdB for mode %s\n", gain, names[mode]);
                }
                mono_passthrough.store(mode == 1, std::memory_order_relaxed);
                hpf_enabled.store(mode == 1, std::memory_order_relaxed);
                current_input_mode.store(mode, std::memory_order_relaxed);
                prev_mode = mode;
                gain_settled_at = std::chrono::steady_clock::now();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        close(req.fd);
    });

    // ---- LED thread ----
    std::thread led_thread([&] {
        int chip_fd = open("/dev/gpiochip4", O_RDONLY);
        if (chip_fd < 0) {
            std::fprintf(stderr, "[led] failed to open /dev/gpiochip4\n");
            return;
        }

        struct gpio_v2_line_request led_req = {};
        led_req.offsets[0] = 17;
        led_req.offsets[1] = 27;
        led_req.num_lines   = 2;
        led_req.config.flags = GPIO_V2_LINE_FLAG_OUTPUT;
        std::strncpy(led_req.consumer, "engine-led", sizeof(led_req.consumer) - 1);

        if (ioctl(chip_fd, GPIO_V2_GET_LINE_IOCTL, &led_req) < 0) {
            std::fprintf(stderr, "[led] failed to get GPIO lines\n");
            close(chip_fd);
            return;
        }
        close(chip_fd);

        int  last_signal = new_slice_signal.load();
        auto flash_until = std::chrono::steady_clock::time_point{};

        while (g_run.load(std::memory_order_relaxed)) {
            bool rec = record_enabled.load(std::memory_order_relaxed);
            auto now = std::chrono::steady_clock::now();

            int sig = new_slice_signal.load(std::memory_order_relaxed);
            if (sig != last_signal) {
                flash_until = now + std::chrono::milliseconds(100);
                last_signal = sig;
            }

            struct gpio_v2_line_values vals = {};
            vals.mask = 0b11;
            if (now < flash_until) {
                vals.bits = 0b11;
            } else {
                vals.bits = rec ? 0b01 : 0b10;
            }
            ioctl(led_req.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &vals);

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        struct gpio_v2_line_values off = {};
        off.mask = 0b11;
        off.bits = 0b00;
        ioctl(led_req.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &off);
        close(led_req.fd);
    });

    // ---- momentary button thread (GPIO 13, active-low) ----
    std::thread moment_btn_thread([&] {
        int chip_fd = open("/dev/gpiochip4", O_RDONLY);
        if (chip_fd < 0) {
            std::fprintf(stderr, "[mbtn] failed to open /dev/gpiochip4\n");
            return;
        }

        struct gpio_v2_line_request mbtn_req = {};
        mbtn_req.offsets[0] = 13;
        mbtn_req.num_lines   = 1;
        mbtn_req.config.flags = GPIO_V2_LINE_FLAG_INPUT | GPIO_V2_LINE_FLAG_BIAS_PULL_UP;
        std::strncpy(mbtn_req.consumer, "engine-mbtn", sizeof(mbtn_req.consumer) - 1);

        if (ioctl(chip_fd, GPIO_V2_GET_LINE_IOCTL, &mbtn_req) < 0) {
            std::fprintf(stderr, "[mbtn] failed to get GPIO line\n");
            close(chip_fd);
            return;
        }
        close(chip_fd);

        int prev_level = 1;

        while (g_run.load(std::memory_order_relaxed)) {
            struct gpio_v2_line_values vals = {};
            vals.mask = 0b1;
            if (ioctl(mbtn_req.fd, GPIO_V2_LINE_GET_VALUES_IOCTL, &vals) < 0) {
                std::fprintf(stderr, "[mbtn] read error\n");
                break;
            }

            int level = vals.bits & 1;

            if (level == 0 && prev_level == 1) {
                std::fprintf(stderr, "[mbtn] pressed\n");
                std::lock_guard<std::mutex> lk(bank_menu_mtx);
                if (bank_menu_disp.open) {
                    if (bank_menu_disp.page == 0) {
                        bank_menu_disp.open = false;
                        std::fprintf(stderr, "[menu] closed via mbtn\n");
                    } else if ((bank_menu_disp.page == 1 || bank_menu_disp.page == 2)
                               && !bank_menu_disp.file_list.empty()) {
                        // mbtn on load list = delete confirmation for highlighted file
                        int ci = bank_menu_disp.cursor;
                        if (ci < (int)bank_menu_disp.file_list.size()) {
                            std::strncpy(bank_menu_disp.delete_file,
                                         bank_menu_disp.file_list[ci].c_str(),
                                         sizeof(bank_menu_disp.delete_file) - 1);
                            bank_menu_disp.page   = (bank_menu_disp.page == 1) ? 7 : 8;
                            bank_menu_disp.cursor = 0;
                        }
                    } else {
                        bank_menu_disp.page   = 0;
                        bank_menu_disp.cursor = 0;
                    }
                }
            }

            prev_level = level;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        close(mbtn_req.fd);
    });

    // ---- record button thread (GPIO 12) ----
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

        int prev_level = 1;

        while (g_run.load(std::memory_order_relaxed)) {
            struct gpio_v2_line_values vals = {};
            vals.mask = 0b1;
            if (ioctl(btn_req.fd, GPIO_V2_LINE_GET_VALUES_IOCTL, &vals) < 0) {
                std::fprintf(stderr, "[btn] read error\n");
                break;
            }

            int level = vals.bits & 1;

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
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        close(btn_req.fd);
    });

    // set RT priority
    {
        struct sched_param sp{};
        sp.sched_priority = 80;
        if (pthread_setschedparam(cap_thread.native_handle(), SCHED_FIFO, &sp) != 0)
            std::fprintf(stderr, "[rt] failed to set RT priority for cap thread (run as root?)\n");
        if (pthread_setschedparam(pb_thread.native_handle(), SCHED_FIFO, &sp) != 0)
            std::fprintf(stderr, "[rt] failed to set RT priority for pb thread (run as root?)\n");
    }

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
                store_a.list();
                store_b.list();
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
                    ? (tonal_root_idx.load() + 11) % 12
                    : (tonal_root_idx.load() +  1) % 12;
                tonal_root_idx.store(idx);
                ScaleType sc = tonal_minor.load() ? ScaleType::Minor : ScaleType::Major;
                auto update_store_tonal = [&](SliceStore& s) {
                    std::lock_guard<std::mutex> lk(s.mtx);
                    for (auto& [id, slice] : s.slices)
                        slice.features.tonal_alignment_score =
                            TonalAlignmentAnalyzer::tonal_score_from_chroma(
                                slice.features.chroma_energy, idx, sc);
                };
                update_store_tonal(store_a);
                update_store_tonal(store_b);
                std::fprintf(stderr, "[key] tonal root=%s (%.1fHz)\n", kNames[idx], kRootFreqs[idx]);
            }
            else if (k == 'v') {
                bool minor = !tonal_minor.load();
                tonal_minor.store(minor);
                int idx = tonal_root_idx.load();
                ScaleType sc = minor ? ScaleType::Minor : ScaleType::Major;
                auto update_store_tonal = [&](SliceStore& s) {
                    std::lock_guard<std::mutex> lk(s.mtx);
                    for (auto& [id, slice] : s.slices)
                        slice.features.tonal_alignment_score =
                            TonalAlignmentAnalyzer::tonal_score_from_chroma(
                                slice.features.chroma_energy, idx, sc);
                };
                update_store_tonal(store_a);
                update_store_tonal(store_b);
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

        // periodic stats: cpu + corpus sizes every 5s
        {
            auto now_s = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now_s - last_stats_time).count();
            if (elapsed >= 5.0) {
                uint64_t ticks = read_cpu_ticks();
                double cpu_pct = ((double)(ticks - last_cpu_ticks) / (double)sysconf(_SC_CLK_TCK))
                                 / elapsed * 100.0;
                size_t sa_slices, sa_samp, sb_slices, sb_samp;
                {
                    std::lock_guard<std::mutex> la(store_a.mtx);
                    sa_slices = store_a.slices.size();
                    sa_samp   = store_a.corpus.size();
                }
                {
                    std::lock_guard<std::mutex> lb(store_b.mtx);
                    sb_slices = store_b.slices.size();
                    sb_samp   = store_b.corpus.size();
                }
                std::fprintf(stderr,
                    "[stats] cpu=%.1f%%  A=%zu slices/%.1fKB  B=%zu slices/%.1fKB  cf=%.2f\n",
                    cpu_pct,
                    sa_slices, sa_samp * sizeof(int16_t) / 1024.0,
                    sb_slices, sb_samp * sizeof(int16_t) / 1024.0,
                    crossfade_pos.load());
                last_stats_time = now_s;
                last_cpu_ticks  = ticks;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    g_run.store(false);
    viz.stop();
    if (cap_thread.joinable())       cap_thread.join();
    if (pb_thread.joinable())        pb_thread.join();
    if (slicer_thread.joinable())    slicer_thread.join();
    if (encoder_thread.joinable())   encoder_thread.join();
    if (mcp3008_thread.joinable())   mcp3008_thread.join();
    if (moment_btn_thread.joinable()) moment_btn_thread.join();
    if (switch_thread.joinable())    switch_thread.join();
    if (led_thread.joinable())       led_thread.join();
    if (button_thread.joinable())    button_thread.join();

    snd_pcm_close(cap);
    snd_pcm_close(pb);
    std::fprintf(stderr, "Done.\n");
    return 0;
}
