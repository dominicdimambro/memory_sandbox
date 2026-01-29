#include <alsa/asoundlib.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "test_util.h"

int main(int argc, char** argv) {

    // define devices through args:
    //   ./passthrough plughw:2,0 plughw:2,0 (uses card #2, device 0)
    const char* cap_dev = (argc > 1) ? argv[1] : "plughw:0,0";
    const char* pb_dev  = (argc > 2) ? argv[2] : "plughw:0,0";

    // define other hardware params
    const unsigned rate = 48000;
    const unsigned channels = 2;
    const snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;

    // moderate latency for testing
    const snd_pcm_uframes_t period_frames = 256;
    const snd_pcm_uframes_t buffer_frames = period_frames * 16;

    snd_pcm_t* cap = nullptr;
    snd_pcm_t* pb  = nullptr;

    // capture pcm
    int err = snd_pcm_open(&cap, cap_dev, SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) die("snd_pcm_open capture failed", err);

    // playback pcm
    err = snd_pcm_open(&pb, pb_dev, SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) die("snd_pcm_open playback failed", err);

    // set params of pcms
    set_hw_params(cap, SND_PCM_STREAM_CAPTURE,  rate, channels, format, period_frames, buffer_frames);
    set_hw_params(pb,  SND_PCM_STREAM_PLAYBACK, rate, channels, format, period_frames, buffer_frames);

    // prep hardware pcms
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

    while (true) {

        // read in from capture
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

        // write samples to playback
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
