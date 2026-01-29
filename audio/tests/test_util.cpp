#include <alsa/asoundlib.h>
#include <cstdio>
#include <cstdlib>
#include "test_util.h"

// helper function for displaying errors
void die(const char* msg, int err) {
    std::fprintf(stderr, "%s: %s\n", msg, snd_strerror(err));
    std::exit(1);
}


// set hardware parameters, check for errors
void set_hw_params(snd_pcm_t* pcm,
                          snd_pcm_stream_t stream,
                          unsigned rate,
                          unsigned channels,
                          snd_pcm_format_t format,
                          snd_pcm_uframes_t period_frames,
                          snd_pcm_uframes_t buffer_frames) {
    // allocate space for hardware pointers                            
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

    // set sample rate
    unsigned exact_rate = rate;
    err = snd_pcm_hw_params_set_rate_near(pcm, hw, &exact_rate, nullptr);
    if (err < 0) die("set_rate_near failed", err);
    if (exact_rate != rate) {
        std::fprintf(stderr, "Warning: requested %u Hz, got %u Hz\n", rate, exact_rate);
    }

    // set number of frames / period
    snd_pcm_uframes_t exact_period = period_frames;
    err = snd_pcm_hw_params_set_period_size_near(pcm, hw, &exact_period, nullptr);
    if (err < 0) die("set_period_size_near failed", err);

    // set number of frames / buffer
    snd_pcm_uframes_t exact_buffer = buffer_frames;
    err = snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &exact_buffer);
    if (err < 0) die("set_buffer_size_near failed", err);

    err = snd_pcm_hw_params(pcm, hw);
    if (err < 0) die("hw_params failed", err);

    // print our hardware configuration
    snd_pcm_hw_params_get_period_size(hw, &exact_period, nullptr);
    snd_pcm_hw_params_get_buffer_size(hw, &exact_buffer);
    std::fprintf(stderr, "[%s] period=%lu frames, buffer=%lu frames\n",
                 stream == SND_PCM_STREAM_CAPTURE ? "capture" : "playback",
                 (unsigned long)exact_period, (unsigned long)exact_buffer);
}