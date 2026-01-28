#include <alsa/asoundlib.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// helper function for displaying errors
static void die(const char* msg, int err) {
    std::fprintf(stderr, "%s: %s\n", msg, snd_strerror(err));
    std::exit(1);
}

// set hardware parameters, check for errors
static void set_hw_params(snd_pcm_t* pcm,
                          snd_pcm_stream_t stream,
                          unsigned rate,
                          unsigned channels,
                          snd_pcm_format_t format,
                          snd_pcm_uframes_t period_frames,
                          snd_pcm_uframes_t buffer_frames) {
    snd_pcm_hw_params_t* hw = nullptr;
    snd_pcm_hw_params_alloca(&hw);

    int err = snd_pcm_hw_params_any(pcm, hw);
    if (err < 0) die("hw_params_any failed", err);

    err = snd_pcm_hw_params_set_access(pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) die("set_access failed", err);

    err = snd_pcm_hw_params_set_format(pcm, hw, format);
    if (err < 0) die("set_format failed", err);

    err = snd_pcm_hw_params_set_channels(pcm, hw, channels);
    if (err < 0) die("set_channels failed", err);

    unsigned exact_rate = rate;
    err = snd_pcm_hw_params_set_rate_near(pcm, hw, &exact_rate, nullptr);
    if (err < 0) die("set_rate_near failed", err);
    if (exact_rate != rate) {
        std::fprintf(stderr, "Warning: requested %u Hz, got %u Hz\n", rate, exact_rate);
    }

    snd_pcm_uframes_t exact_period = period_frames;
    err = snd_pcm_hw_params_set_period_size_near(pcm, hw, &exact_period, nullptr);
    if (err < 0) die("set_period_size_near failed", err);

    snd_pcm_uframes_t exact_buffer = buffer_frames;
    err = snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &exact_buffer);
    if (err < 0) die("set_buffer_size_near failed", err);

    err = snd_pcm_hw_params(pcm, hw);
    if (err < 0) die("hw_params failed", err);

    // print what we actually got
    snd_pcm_hw_params_get_period_size(hw, &exact_period, nullptr);
    snd_pcm_hw_params_get_buffer_size(hw, &exact_buffer);
    std::fprintf(stderr, "[%s] period=%lu frames, buffer=%lu frames\n",
                 stream == SND_PCM_STREAM_CAPTURE ? "capture" : "playback",
                 (unsigned long)exact_period, (unsigned long)exact_buffer);
}

int main(int argc, char** argv) {
    // You can pass device names as args:
    //   ./passthrough plughw:0,0 plughw:0,0
    const char* cap_dev = (argc > 1) ? argv[1] : "plughw:0,0";
    const char* pb_dev  = (argc > 2) ? argv[2] : "plughw:0,0";

    const unsigned rate = 48000;
    const unsigned channels = 2;
    const snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;

    // moderate latency for a quick test
    const snd_pcm_uframes_t period_frames = 256;
    const snd_pcm_uframes_t buffer_frames = period_frames * 16;

    snd_pcm_t* cap = nullptr;
    snd_pcm_t* pb  = nullptr;

    // capture pcm
    int err = snd_pcm_open(&cap, cap_dev, SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) die("snd_pcm_open capture failed", err);

    // planyback pcm
    err = snd_pcm_open(&pb, pb_dev, SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) die("snd_pcm_open playback failed", err);

    // set params of pcms
    set_hw_params(cap, SND_PCM_STREAM_CAPTURE,  rate, channels, format, period_frames, buffer_frames);
    set_hw_params(pb,  SND_PCM_STREAM_PLAYBACK, rate, channels, format, period_frames, buffer_frames);

    err = snd_pcm_prepare(cap);
    if (err < 0) die("capture prepare failed", err);
    err = snd_pcm_prepare(pb);
    if (err < 0) die("playback prepare failed", err);

    // set up buffers for capture and playback
    const size_t bytes_per_frame = channels * snd_pcm_format_physical_width(format) / 8;
    std::vector<char> buf(period_frames * bytes_per_frame);
    std::vector<char> zbuf(period_frames * bytes_per_frame, 0);

    // prime playback with silence to avoid startup underrun
    for (int i = 0; i < 8; i++) {
        snd_pcm_sframes_t m = snd_pcm_writei(pb, zbuf.data(), period_frames);
        if (m == -EPIPE) { snd_pcm_prepare(pb); --i; continue; }
        if (m < 0)       { snd_pcm_prepare(pb); --i; continue; }
    }

    std::fprintf(stderr, "Passthrough running: %s -> %s (Ctrl+C to stop)\n", cap_dev, pb_dev);

    // looping functionality:
    //    read in from capture
    //    write samples to playback
    while (true) {
        snd_pcm_sframes_t n = snd_pcm_readi(cap, buf.data(), period_frames);
        if (n == -EPIPE) {
            std::fprintf(stderr, "Capture overrun\n");
            snd_pcm_prepare(cap);
            continue;
        } else if (n < 0) {
            std::fprintf(stderr, "Capture read error: %s\n", snd_strerror((int)n));
            snd_pcm_prepare(cap);
            continue;
        }

        snd_pcm_sframes_t written = 0;
        while (written < n) {
            snd_pcm_sframes_t m = snd_pcm_writei(pb,
                                                 buf.data() + (written * bytes_per_frame),
                                                 n - written);
            if (m == -EPIPE) {
                std::fprintf(stderr, "Playback underrun\n");
                snd_pcm_prepare(pb);
                continue;
            } else if (m < 0) {
                std::fprintf(stderr, "Playback write error: %s\n", snd_strerror((int)m));
                snd_pcm_prepare(pb);
                continue;
            }
            written += m;
        }
    }

    // close pcms on exit
    snd_pcm_close(cap);
    snd_pcm_close(pb);
    return 0;
}
