#pragma once
#include <alsa/asoundlib.h>

// helper function for displaying errors
void die(const char* msg, int err);

// set hardware parameters, check for errors
void set_hw_params(snd_pcm_t* pcm,
                          snd_pcm_stream_t stream,
                          unsigned rate,
                          unsigned channels,
                          snd_pcm_format_t format,
                          snd_pcm_uframes_t period_frames,
                          snd_pcm_uframes_t buffer_frames);