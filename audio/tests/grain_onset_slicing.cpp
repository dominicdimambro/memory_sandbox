#include <alsa/asoundlib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "ringbuffer.h"

// Terminal input helpers

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

static inline uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s^= s >> 17;
    s ^= s << 5;
    return s;
}

struct TermRawMode {
    termios orig{};
    bool ok{false};

    bool enable() {
        if (tcgetattr(STDIN_FILENO, &orig) != 0) return false;
        termios raw = orig;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) return false;

        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (flags >= 0) fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

        ok = true;
        return true;
    }

    ~TermRawMode() {
        if (ok) tcsetattr(STDIN_FILENO, TCSANOW, &orig);
    }
};

static int read_key_nonblocking() {
    unsigned char c;
    ssize_t n = ::read(STDIN_FILENO, &c, 1);
    if (n == 1) return (int)c;
    return -1;
}

// slice struct
struct Slice {
    int id;
    uint64_t corpus_start;  // sample index in corpus buffer
    uint32_t length;        // length of slice in samples
};

// struct for pool of slices
struct SliceStore {
    std::mutex mtx;
    std::vector<int16_t> corpus;
    std::unordered_map<int, Slice> slices;
    int next_id = 0;

    int add_slice(const int16_t* data, uint32_t n_samples) {
        std::lock_guard<std::mutex> lk(mtx);
        uint64_t start = corpus.size();
        corpus.insert(corpus.end(), data, data + n_samples);
        int id = next_id++;
        slices[id] = Slice{id, start, n_samples};
        return id;
    }

    void list() {
        std::lock_guard<std::mutex> lk(mtx);
        std::fprintf(stderr, "Slices (%zu):\n", slices.size());
        for (auto& [id, s] : slices) {
            std::fprintf(stderr, "  id=%d start=%llu lenSamples=%u\n",
                         id, (unsigned long long)s.corpus_start, s.length);
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

        // pick k-th element
        size_t k = (size_t)(rng % slices.size());
        auto it = slices.begin();
        std::advance(it, k);
        out_id = it->first;
        return true;
    }

    // don't capture while slice playback is enabled
    const int16_t* ptr(uint64_t corpusIndex) const {
        return corpus.data() + corpusIndex;
    }
};

// ALSA helpers

// helper function for setting hw params when we open a new pcm
static bool set_hw_params(snd_pcm_t* pcm,
                          unsigned int rate,
                          unsigned int channels,
                          snd_pcm_format_t fmt,
                          snd_pcm_uframes_t period_frames,
                          snd_pcm_uframes_t buffer_frames) {
    snd_pcm_hw_params_t* hw = nullptr;
    snd_pcm_hw_params_alloca(&hw);

    if (snd_pcm_hw_params_any(pcm, hw) < 0) return false;
    if (snd_pcm_hw_params_set_access(pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED) < 0) return false;
    if (snd_pcm_hw_params_set_format(pcm, hw, fmt) < 0) return false;
    if (snd_pcm_hw_params_set_channels(pcm, hw, channels) < 0) return false;

    unsigned int r = rate;
    if (snd_pcm_hw_params_set_rate_near(pcm, hw, &r, nullptr) < 0) return false;

    snd_pcm_uframes_t p = period_frames;
    if (snd_pcm_hw_params_set_period_size_near(pcm, hw, &p, nullptr) < 0) return false;

    snd_pcm_uframes_t b = buffer_frames;
    if (snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &b) < 0) return false;

    if (snd_pcm_hw_params(pcm, hw) < 0) return false;

    return true;
}

// open pcm, set parameters
static bool open_pcm(snd_pcm_t** out,
                     const char* dev,
                     snd_pcm_stream_t stream,
                     unsigned int rate,
                     unsigned int channels,
                     snd_pcm_uframes_t period_frames,
                     snd_pcm_uframes_t buffer_frames) {
    *out = nullptr;
    int err = snd_pcm_open(out, dev, stream, 0);
    if (err < 0) {
        std::fprintf(stderr, "snd_pcm_open(%s) failed: %s\n", dev, snd_strerror(err));
        return false;
    }

    if (!set_hw_params(*out, rate, channels, SND_PCM_FORMAT_S16_LE, period_frames, buffer_frames)) {
        std::fprintf(stderr, "set_hw_params(%s) failed\n", dev);
        snd_pcm_close(*out);
        *out = nullptr;
        return false;
    }

    err = snd_pcm_prepare(*out);
    if (err < 0) {
        std::fprintf(stderr, "snd_pcm_prepare(%s) failed: %s\n", dev, snd_strerror(err));
        snd_pcm_close(*out);
        *out = nullptr;
        return false;
    }
    return true;
}

// if overrun or underrun, prepare the pcm again
static void recover_if_xrun(snd_pcm_t* pcm, int err, const char* tag) {
    if (err == -EPIPE) { // xrun
        std::fprintf(stderr, "[%s] XRUN, preparing device...\n", tag);
        snd_pcm_prepare(pcm);
    } else if (err == -ESTRPIPE) {
        std::fprintf(stderr, "[%s] ESTRPIPE, trying resume...\n", tag);
        while ((err = snd_pcm_resume(pcm)) == -EAGAIN) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (err < 0) snd_pcm_prepare(pcm);
    }
}

// struct for detecting onsets for slicing
struct OnsetEvent {
    int sample_index;
    float env;
    float delta;
};

static std::vector<OnsetEvent> detect_onsets_stereo_s16(
    const int16_t* x, size_t n_samples, unsigned int channels, unsigned int rate, 
    float sensitivity
) {

    // n_samples is interleaved samples: frames * channels
    const size_t n_frames = n_samples / channels;

    // smoothing ~10ms time constant
    const float tau_ms = 10.0f;
    const float a = std::exp(-1.0f / ( (tau_ms * 0.001f) * rate ));

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
                events.push_back(OnsetEvent{ (int)i, env, d_env });
                last_onset_frame = i;
            }
        }
    }

    std::fprintf(stderr, "[onset] max_d=%.6f level_thresh=%.6f delta_thresh=%.6f\n",
             max_d, level_thresh, delta_thresh);

    return events;
}

// good run atomic bool
static std::atomic<bool> g_run{true};

int main(int argc, char** argv) {
    
    // set capture and playback devices with default fallbacks
    const char* cap_dev = (argc > 1) ? argv[1] : "plughw:2,0";
    const char* pb_dev = (argc > 2) ? argv[2] : "plughw:2,0";

    // hw parameters
    const unsigned int rate = 48000;
    const unsigned int channels = 2;
    const snd_pcm_uframes_t period_frames = 256;
    const snd_pcm_uframes_t buffer_frames = period_frames * 16;

    const size_t samples_per_period = (size_t)period_frames * channels;

    // set up ring buffer
    const size_t rb_capacity_seconds = 1;
    const size_t rb_capacity_samples = (size_t)rate * channels * rb_capacity_seconds;
    RingBuffer<int16_t> rb(rb_capacity_samples);
    std::mutex rb_mtx;

    // keeping track of if we are recording or playing across threads
    std::atomic<bool> record_enabled{false};
    std::atomic<bool> play_enabled{false};

    // delcare and open pcms
    snd_pcm_t* cap = nullptr;
    snd_pcm_t* pb = nullptr;

    // try opening pcm, catch openning errors
    std::fprintf(stderr, "opening capture=%s playback=%s\n", cap_dev, pb_dev);
    if (!open_pcm(&cap, cap_dev, SND_PCM_STREAM_CAPTURE, rate, channels, period_frames, buffer_frames)) return 1;
    if (!open_pcm(&pb, pb_dev, SND_PCM_STREAM_PLAYBACK, rate, channels, period_frames, buffer_frames)) return 1;

    // display controls
    std::fprintf(stderr,
        "\nControls:\n"
        "   r = start recording into ringbuffer\n"
        "   s = stop recording\n"
        "   m = toggle monitor\n"
        "   q = quit\n"
    );

    // start looking for keyboard inputs
    TermRawMode raw;
    raw.enable();
    

    // allocate window
    std::vector<int16_t> one_sec_window(rb_capacity_samples);

    // slicer control + state 
    std::atomic<bool> slicer_enabled{true};
    std::atomic<uint64_t> windows_captured{0};

    // epoch counter
    std::atomic<uint64_t> rb_epoch{0};

    // playback controls and debug flags
    std::atomic<int> grain_interval_ms{200};    // [1]=1000, [2]=200, etc.
    std::atomic<bool> explicit_mode{false};
    std::atomic<int> current_grain_id{-1};

    SliceStore store;

    // capture thread: fill ringbuffer when recording enabled
    std::thread cap_thread([&] {

        std::vector<int16_t> in(samples_per_period);

        uint64_t samples_since_epoch = 0;
        bool was_recording = false;

        // while run is still good read frames to in, write to ringbuffer if record enabled
        while (g_run.load(std::memory_order_relaxed)) {

            // read in 1 period of frames
            snd_pcm_sframes_t n = snd_pcm_readi(cap, in.data(), period_frames);

            // deal with xruns and empty reads
            if (n < 0) { recover_if_xrun(cap, (int)n, "capture"); continue; }
            if (n == 0) continue;

            bool rec = record_enabled.load(std::memory_order_relaxed);

            // detect recording start: reset epoch accounting
            if (rec && !was_recording) {
                samples_since_epoch = 0;
                rb_epoch.store(0, std::memory_order_release);
            }
            was_recording = rec;

            // do nothing else if we aren't recording
            if (!rec) continue;

            // write to ringbuffer
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

    // slicer config
    const int pre_ms = 10;
    const int slice_ms = 200;
    const size_t pre_frames = (size_t)rate * pre_ms / 1000;
    const size_t slice_frames = (size_t)rate * slice_ms / 1000;
    const size_t slice_samples = slice_frames * channels;

    // slicer thread
    std::thread slicer_thread([&] {
        uint64_t last_epoch = 0;

        while (g_run.load(std::memory_order_relaxed)) {
            
            uint64_t e = rb_epoch.load(std::memory_order_acquire);

            // capture window = latest second
            if (e != last_epoch) {
                
                size_t got = 0;
                {
                    std::lock_guard<std::mutex> lk(rb_mtx);
                    got = rb.copy_latest(one_sec_window.data(), rb_capacity_samples, 0);
                }

                auto stats = [&](const int16_t* x, size_t n, unsigned ch) {
                    size_t frames = n / ch;
                    int16_t maxL = 0, maxR = 0;
                    double meanAbs = 0.0;

                    for (size_t i = 0; i < frames; i++) {
                        int16_t l = x[i*ch + 0];
                        int16_t r = (ch > 1) ? x[i*ch + 1] : l;
                        maxL = std::max<int16_t>(maxL, (int16_t)std::abs((int)l));
                        maxR = std::max<int16_t>(maxR, (int16_t)std::abs((int)r));
                        meanAbs += 0.5 * (std::abs((int)l) + std::abs((int)r));
                    }
                    meanAbs /= (double)frames; // in int16 units
                    std::fprintf(stderr, "[slicer] maxL=%d maxR=%d meanAbs=%.1f\n", maxL, maxR, meanAbs);
                };
                stats(one_sec_window.data(), got, channels);


                if (got == rb_capacity_samples) {
                    
                    const size_t n_frames = got / channels;

                    // TODO: make adjustable
                    float sensitivity = 1.0f;
                    auto ev = detect_onsets_stereo_s16(
                        one_sec_window.data(), got, channels, rate, sensitivity
                    );

                    std::fprintf(stderr, "[slicer] onsets=%zu\n", ev.size());

                    for (auto &o : ev) {
                        size_t onset_frame = (size_t)o.sample_index;

                        // start frame with preroll with clamp
                        size_t start_frame = (onset_frame > pre_frames) ? (onset_frame - pre_frames) : 0;

                        // clamp end within 1s window
                        size_t end_frame = std::min(start_frame + slice_frames, n_frames);
                        size_t n_frames_slice = end_frame - start_frame;
                        if (n_frames_slice < (size_t)(rate * 30 / 1000)) continue;  // skip tiny (<30ms)

                        size_t start_sample = start_frame * channels;
                        size_t n_samples_slice = n_frames_slice * channels;

                        int id = store.add_slice(one_sec_window.data() + start_sample, (uint32_t)n_samples_slice);

                        std::fprintf(stderr, "[slice] id=%d start=%.1fms len=%zu samples\n",
                            id, 1000.0*start_frame/rate, n_samples_slice);
                    }
                }

                windows_captured.fetch_add(1, std::memory_order_relaxed);

                std::fprintf(stderr, "[slicer] epoch=%llu got=%zu samples (expected=%zu)\n", (unsigned long long)e, got, rb_capacity_samples);

                last_epoch = e;
            }
            else {
                // don't busy spin
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    });

    // playback thread: monitor latest audio with small delay
    std::thread pb_thread([&] {

        // initialize as 0
        std::vector<int16_t> out(samples_per_period, 0);

        Slice cur{};
        bool cur_valid = false;
        uint32_t pos = 0;
        uint32_t rng = 0x12345678u;

        auto next_switch = std::chrono::steady_clock::now();

        while (g_run.load(std::memory_order_relaxed)) {

            // only sending output when we are recording and monitoring
            size_t got = 0;
            
            bool playing = play_enabled.load(std::memory_order_relaxed);
            bool rec     = record_enabled.load(std::memory_order_relaxed);
            bool expl    = explicit_mode.load(std::memory_order_relaxed);
            int interval = grain_interval_ms.load(std::memory_order_relaxed);

            

            if (!playing) {
                got = 0;    // silence
            }
            else if (rec) {
            
                // read in a period of samples into out with 2 periods of delay
                if (rb_mtx.try_lock()) {
                    const size_t delay = samples_per_period * 2;
                    got = rb.copy_latest(out.data(), samples_per_period, delay);
                    rb_mtx.unlock();
                } else {
                    got = 0;
                }
            }
            else {
                // grains mode while not recording
                auto now = std::chrono::steady_clock::now();
                if (now >= next_switch || !cur_valid) {
                    rng = xorshift32(rng);
                    int new_id = -1;
                    if (store.random_id(rng, new_id)) {
                        Slice tmp;
                        if (store.get(new_id, tmp)) {
                            cur = tmp;
                            cur_valid = true;
                            pos = 0;
                            current_grain_id.store(new_id, std::memory_order_relaxed);

                            if (expl) {
                                std::fprintf(stderr, "[grain] id=%d len_samples=%u inteval=%dms\n",
                                        cur.id, cur.length, interval);
                            }
                        }
                    }
                    else {
                        cur_valid = false;
                    }
                    next_switch = now + std::chrono::milliseconds(interval);
                }

                if (cur_valid) {
                    // copy as much as we can from this slice
                    uint32_t remain = (pos < cur.length) ? (cur.length - pos) : 0;
                    uint32_t want = (uint32_t)samples_per_period;
                    uint32_t take = std::min(remain, want);

                    if (take > 0) {
                        // lock store while reading from corpus pointer
                        std::lock_guard<std::mutex> lk(store.mtx);
                        const int16_t* base = store.corpus.data() + cur.corpus_start;
                        std::copy(base + pos, base + pos + take, out.begin());

                        got = take;     // tell ALSA how many samples are valid
                        pos += take;    // advance within slice
                    }
                    else {
                        got = 0;
                        cur_valid = false;
                    }
                }
            }

            if (got < samples_per_period) {
                std::fill(out.begin() + got, out.end(), 0);
            }

            // write to playback pcm
            snd_pcm_sframes_t w = snd_pcm_writei(pb, out.data(), period_frames);
            if (w < 0) recover_if_xrun(pb, (int)w, "playback");
        }
    });

    // key input loop
    while (g_run.load(std::memory_order_relaxed)) {

        // set reading key
        int k = read_key_nonblocking();

        // cases for controls
        if (k != -1) {

            // quit
            if (k == 'q') { g_run.store(false); break; }

            // enable recording, clear ringbuffer
            else if (k == 'r') {
                record_enabled.store(true);
                {
                    std::lock_guard<std::mutex> lk_guard(rb_mtx);
                    rb.clear();
                }
                std::fprintf(stderr, "record_enabled = true\n");
            }

            // stop recording
            else if (k == 's') {
                record_enabled.store(false);
                std::fprintf(stderr, "record_enabled = false\n");
            }

           // monitor recording
            else if (k == 'm') {
                bool ns = !play_enabled.load();
                play_enabled.store(ns);
                std::fprintf(stderr, "play_enabled = %s\n", ns ? "true" : "false");
                
            }

            // explicit mode
            else if (k == 'e') {
                bool v = !explicit_mode.load();
                explicit_mode.store(v);
                std::fprintf(stderr, "[key] explicit_mode = %s\n", v ? "true" : "false");
            }

            // grain freq kets
            else if (k == '1') { grain_interval_ms.store(1000); std::fprintf(stderr, "[key] interval=1000ms\n"); }
            else if (k == '2') { grain_interval_ms.store(200); std::fprintf(stderr, "[key] interval=200ms\n"); }
            else if (k == '3') { grain_interval_ms.store(100); std::fprintf(stderr, "[key] interval=100ms\n"); }
            else if (k == '4') { grain_interval_ms.store(50); std::fprintf(stderr, "[key] interval=50ms\n"); }
        }
        
        // sleep this thread
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // if reach this point, join threads and close pcms
    g_run.store(false);
    if (cap_thread.joinable()) cap_thread.join();
    if (pb_thread.joinable()) pb_thread.join();
    if (slicer_thread.joinable()) slicer_thread.join();

    snd_pcm_close(cap);
    snd_pcm_close(pb);
    std::fprintf(stderr, "Done.\n");
    return 0;
}