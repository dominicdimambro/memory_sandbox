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

    // capture thread: fill ringbuffer when recording enabled
    std::thread cap_thread([&] {

        std::vector<int16_t> in(samples_per_period);

        // while run is still good read frames to in, write to ringbuffer if record enabled
        while (g_run.load(std::memory_order_relaxed)) {

            // read in 1 period of frames
            snd_pcm_sframes_t n = snd_pcm_readi(cap, in.data(), period_frames);

            // deal with xruns and empty reads
            if (n < 0) { recover_if_xrun(cap, (int)n, "capture"); continue; }
            if (n == 0) continue;

            // do nothing else if we aren't recording
            if (!record_enabled.load(std::memory_order_relaxed)) continue;

            // write to ringbuffer
            const size_t nsamp = (size_t)n * channels;
            std::lock_guard<std::mutex> lk(rb_mtx);
            rb.write_overwrite(in.data(), nsamp);
        }
    });

    // playback thread: monitor latest audio with small delay
    std::thread pb_thread([&] {

        // initialize as 0
        std::vector<int16_t> out(samples_per_period, 0);

        while (g_run.load(std::memory_order_relaxed)) {

            bool playing = play_enabled.load(std::memory_order_relaxed);
            bool rec     = record_enabled.load(std::memory_order_relaxed);

            // only sending output when we are recording and monitoring
            size_t got = 0;

            if (playing && rec) {
            
                // read in a period of samples into out with 2 periods of delay
                if (rb_mtx.try_lock()) {
                    const size_t delay = samples_per_period * 2;
                    got = rb.copy_latest(out.data(), samples_per_period, delay);
                    rb_mtx.unlock();
                }
            }

            // pad if we didn't get a full period of samples 
            if (got < samples_per_period) std::fill(out.begin() + got, out.end(), 0);

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
                    std::fprintf(stderr, "record_enabled = true\n");
                }
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
        }
        
        // sleep this thread
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // if reach this point, join threads and close pcms
    g_run.store(false);
    if (cap_thread.joinable()) cap_thread.join();
    if (pb_thread.joinable()) pb_thread.join();

    snd_pcm_close(cap);
    snd_pcm_close(pb);
    std::fprintf(stderr, "Done.\n");
    return 0;
}