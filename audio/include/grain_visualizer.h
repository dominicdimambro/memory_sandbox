#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <SDL2/SDL.h>

#include "slice_store.h"

struct GrainPoint {
    int   id;
    float x, y, z;        // remapped to [-0.5, +0.5]
    float rolloff_norm;    // for color (hue)
    float rms_norm;        // for brightness
};

class GrainVisualizer {
public:
    GrainVisualizer(SliceStore&, std::atomic<bool>& run, std::atomic<int>& current_id);
    void start();   // launch render thread
    void stop();    // join render thread

private:
    void render_loop();
    void snapshot();      // try_lock → copy → unlock
    void render_frame();  // project + SDL draw
    static SDL_Color hsv_to_rgb(float h, float s, float v);

    SliceStore&         store_;
    std::atomic<bool>&  run_;
    std::atomic<int>&   current_id_;
    std::thread         thread_;
    std::vector<GrainPoint> points_;   // render-thread-only, no mutex

    float theta_y_ = 0.0f;
    float theta_x_ = 0.15f;   // slight downward tilt

    SDL_Window*   window_   = nullptr;
    SDL_Renderer* renderer_ = nullptr;

    int   w_ = 800, h_ = 540;  // set from actual renderer output size after init
    float scale_ = 250.0f;    // derived from display size after init

    static constexpr float kFocal = 1.5f, kCameraZ = 2.5f;
};
