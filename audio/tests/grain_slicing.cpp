#include <alsa/asoundlib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
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

// ALSA helpers

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

    // don't capture while slice playback is enabled
    const int16_t* ptr(uint64_t corpusIndex) const {
        return corpus.data() + corpusIndex;
    }
};

// add a playback mode and slice player state
enum class PlayMode { Monitor, Slice };

std::atomic<PlayMode> play_mode{PlayMode::Monitor};
std::atomic<int> active_slice_id{-1};

// Playback-thread owned state
Slice activeSlice{};
bool activeSliceValid = false;
uint32_t slicePos = 0;

// main test
static std::atomic<bool> g_run{true};

int main(int argc, char** argv) {

    // set capture and playback devices
    const char* cap_dev = (argc > 1) ? argv[1] : "plughw:2,0";
    const char* pb_dev  = (argc > 2) ? argv[2] : "plughw:2,0";

    // Audio config
    const unsigned int rate = 48000;
    const unsigned int channels = 2;
    const snd_pcm_uframes_t period_frames = 256;
    const snd_pcm_uframes_t buffer_frames = period_frames * 16;

    const size_t frames_per_period = (size_t)period_frames;
    const size_t samples_per_period = frames_per_period * channels;

    // Ring buffer holds *samples* (interleaved int16)
    // Example: 5 seconds at 48k stereo => 48000*2*5 = 480k samples
    const size_t seconds = 5;
    const size_t rb_capacity_samples = (size_t)rate * channels * seconds;

    RingBuffer<int16_t> rb(rb_capacity_samples);
    
    // mutex for making changes to the ringbuffer
    std::mutex rb_mtx;

    // Control flags
    std::atomic<bool> record_enabled{false};
    std::atomic<bool> play_enabled{false};

    // loop read head and size
    size_t loop_read_pos = 0;
    size_t loop_len = 0;
    size_t loop_start = 0;

    // ALSA devices
    snd_pcm_t* cap = nullptr;
    snd_pcm_t* pb  = nullptr;

    // opening pcm for playback and capture devices
    std::fprintf(stderr, "Opening capture=%s playback=%s\n", cap_dev, pb_dev);
    if (!open_pcm(&cap, cap_dev, SND_PCM_STREAM_CAPTURE, rate, channels, period_frames, buffer_frames)) {
        return 1;
    }
    if (!open_pcm(&pb, pb_dev, SND_PCM_STREAM_PLAYBACK, rate, channels, period_frames, buffer_frames)) {
        snd_pcm_close(cap);
        return 1;
    }

    // printing controls to test user
    std::fprintf(stderr,
        "\nControls:\n"
        "  r = start recording into ring buffer (drop-old)\n"
        "  s = stop recording and freeze current buffer (snapshot)\n"
        "  p = toggle playback (monitor if recording, loop if stopped)\n"
        "  g = capture last 200ms as a slice\n"
        "  l = list slices\n"
        "  m = monitor mode\n"
        "  t = slice mode (plays selected slice)\n"
        "  [0-9] = select slice id\n"
        "  q = quit\n\n"
        "Ring buffer: %zu seconds (%zu samples)\n\n",
        seconds, rb_capacity_samples
    );

    // enable keyboard input
    TermRawMode raw;
    if (!raw.enable()) {
        std::fprintf(stderr, "Warning: couldn't enable raw keyboard input (still running).\n");
    }

    // create slice for "grab slice"
    SliceStore store;
    const int slice_ms = 200;
    const size_t slice_samples = (size_t)rate * channels * slice_ms / 1000;
    std::vector<int16_t> tmp_slice(slice_samples);

    // Thread: capture
    std::thread cap_thread([&] {
        std::vector<int16_t> in(samples_per_period);

        while (g_run.load()) {
            snd_pcm_sframes_t n = snd_pcm_readi(cap, in.data(), period_frames);
            if (n < 0) {
                recover_if_xrun(cap, (int)n, "capture");
                continue;
            }
            if (n == 0) continue;

            // if recording enabled, append samples to ringbuffer
            if (record_enabled.load()) {
                const size_t nsamp = (size_t)n * channels;
                std::lock_guard<std::mutex> lk(rb_mtx);
                // drop-old behavior
                rb.write_overwrite(in.data(), nsamp);
            }
        }
    });

    // Thread: playback
    std::thread pb_thread([&] {
        std::vector<int16_t> out(samples_per_period, 0);

        while (g_run.load()) {
            if (!play_enabled.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            size_t got = 0;

            PlayMode mode = play_mode.load();

            // monitor mode
            if (mode == PlayMode::Monitor) {

                if (record_enabled.load()) {
                    // MONITOR MODE: non-destructive peek from ring buffer
                    {
                        std::lock_guard<std::mutex> lk(rb_mtx);
                        // copy latest with 2 periods of delay
                        const size_t monitor_delay_samples = samples_per_period * 2;
                        got = rb.copy_latest(out.data(), samples_per_period, monitor_delay_samples);
                    }
                } else {
                    // FREEZE MODE: loop through the valid window
                    {
                        std::lock_guard<std::mutex> lk(rb_mtx);
                        if (loop_len == 0) {
                            got = 0;
                        } else {
                            got = rb.copy_range(out.data(), loop_read_pos, samples_per_period);
                            
                            const size_t cap = rb.capacity();

                            // advance within [loop_start, loop_start + loop_len)
                            size_t rel = (loop_read_pos + cap - loop_start) % cap; // position within ring relative to loop_start
                            rel = (rel + got) % loop_len;                          // wrap within loop_len
                            loop_read_pos = (loop_start + rel) % cap;              // back to absolute ring index
                        }
                    }
                }  
            }
            // Slice mode
            else {
                // latch slice if changed
                int want = active_slice_id.load();
                static int last = -999;

                if (want != last) {
                    Slice s;
                    if (store.get(want, s)) {
                        activeSlice = s;
                        activeSliceValid = true;
                        slicePos = 0;
                        last = want;
                        std::fprintf(stderr, "[pb] latched slice id=%d len=%u\n", s.id, s.length);
                    } else {
                        activeSliceValid = false;
                        last = want;
                        std::fprintf(stderr, "[pb] slice id=%d not found\n", want);
                    }
                }

                if (!activeSliceValid || activeSlice.length == 0) {
                    got = 0;
                } else {
                    // loop-copy from corpus into out
                    // activeSlice.length is in samples (interleaved)
                    const int16_t* base = store.ptr(activeSlice.corpus_start);

                    size_t remain = samples_per_period;
                    size_t outPos = 0;

                    while (remain > 0) {
                        uint32_t avail = activeSlice.length - slicePos;
                        uint32_t ncopy = (uint32_t)std::min<size_t>(remain, avail);

                        std::memcpy(out.data() + outPos, base + slicePos, ncopy * sizeof(int16_t));

                        slicePos += ncopy;
                        if (slicePos >= activeSlice.length) slicePos = 0;

                        outPos += ncopy;
                        remain -= ncopy;
                    }
                    got = samples_per_period;
                }
            }         

            // Always pad if short
            if (got < samples_per_period) {
                std::fill(out.begin() + got, out.end(), 0);
            }

            snd_pcm_sframes_t w = snd_pcm_writei(pb, out.data(), period_frames);
            if (w < 0) recover_if_xrun(pb, (int)w, "playback");
        }
    });


    // Main: key handling
    auto last_print = std::chrono::steady_clock::now();

    while (g_run.load()) {
        int k = read_key_nonblocking();
        if (k != -1) {
            if (k == 'q') {
                g_run.store(false);
                break;
            } else if (k == 'r') {
                record_enabled.store(true);
                std::fprintf(stderr, "[key] record_enabled = true\n");
            } else if (k == 's') {
                record_enabled.store(false);

                {
                    std::lock_guard<std::mutex> lk(rb_mtx);
                    loop_len = rb.size();
                    if (loop_len == 0) {
                        loop_start = loop_read_pos = 0;
                    } else {
                        const size_t cap = rb.capacity();
                        const size_t w = rb.write_pos();
                        loop_start = (w + cap - (loop_len % cap)) % cap;
                        loop_read_pos = loop_start;
                    }
                }
            
                std::fprintf(stderr, "[key] record_enabled = false, (loop_len=%zu)\n", loop_len);

            } else if (k == 'p') {
                bool new_state = !play_enabled.load();
                play_enabled.store(new_state);
                std::fprintf(stderr, "[key] play_enabled = %s\n", new_state ? "true" : "false");

            } else if (k == 'g') {
                if (play_mode.load() == PlayMode::Slice) {
                    std::fprintf(stderr, "[key] stop slice mode before capturing (avoids corpus realloc during playback)\n");
                    continue;
                }

                size_t got = 0;
                {
                    std::lock_guard<std::mutex> lk(rb_mtx);
                    // small delay to avoid capturing "future" samples relative to monitor
                    const size_t delay = samples_per_period * 2;
                    got = rb.copy_latest(tmp_slice.data(), slice_samples, delay);
                }
                if (got == 0) {
                    std::fprintf(stderr, "[key] capture failed: ring buffer empty\n");
                } else {
                    int id = store.add_slice(tmp_slice.data(), (uint32_t)got);
                    std::fprintf(stderr, "[key] captured slice id=%d (%zu samples ~ %.1fms)\n",
                        id, got, 1000.0 * (double)got / (double)(rate * channels));
                    active_slice_id.store(id);

                }
            } else if (k == 'l') {
                store.list();
            } else if (k == 'm') {
                play_mode.store(PlayMode::Monitor);
                std::fprintf(stderr, "[key] play_mode = Monitor\n");
            } else if (k == 't') {
                play_mode.store(PlayMode::Slice);
                play_enabled.store(true);
                std::fprintf(stderr, "[key] play_mode = Slice\n");
            } else if (k >= '0' && k <= '9') {
                int id = k - '0';
                active_slice_id.store(id);
                std::fprintf(stderr, "[key] active_slice_id = %d\n", id);
            }
        }

        // poll every 5ms
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // shutdown
    g_run.store(false);
    if (cap_thread.joinable()) cap_thread.join();
    if (pb_thread.joinable()) pb_thread.join();

    snd_pcm_close(cap);
    snd_pcm_close(pb);

    std::fprintf(stderr, "Done.\n");
    return 0;
}
