#pragma once
#include <alsa/asoundlib.h>

// set hardware parameters for a PCM device
// returns true on success
bool set_hw_params(snd_pcm_t* pcm,
                   unsigned int rate,
                   unsigned int channels,
                   snd_pcm_format_t fmt,
                   snd_pcm_uframes_t period_frames,
                   snd_pcm_uframes_t buffer_frames);

// open a PCM device, configure hw params, and prepare it
// returns true on success, sets *out to the opened handle
bool open_pcm(snd_pcm_t** out,
              const char* dev,
              snd_pcm_stream_t stream,
              unsigned int rate,
              unsigned int channels,
              snd_pcm_uframes_t period_frames,
              snd_pcm_uframes_t buffer_frames);

// recover from xrun (underrun/overrun) or suspend
void recover_if_xrun(snd_pcm_t* pcm, int err, const char* tag);
