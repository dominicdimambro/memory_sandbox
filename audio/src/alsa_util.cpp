#include "alsa_util.h"

#include <cstdio>
#include <chrono>
#include <thread>

bool set_hw_params(snd_pcm_t* pcm,
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

bool open_pcm(snd_pcm_t** out,
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

void recover_if_xrun(snd_pcm_t* pcm, int err, const char* tag) {
    if (err == -EPIPE) {
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
