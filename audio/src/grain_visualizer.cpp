#include "grain_visualizer.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unistd.h>

using namespace std::chrono;

GrainVisualizer::GrainVisualizer(SliceStore& store,
                                  std::atomic<bool>& run,
                                  std::atomic<int>& current_id)
    : store_(store), run_(run), current_id_(current_id) {}

void GrainVisualizer::start() {
    thread_ = std::thread(&GrainVisualizer::render_loop, this);
}

void GrainVisualizer::stop() {
    if (thread_.joinable()) thread_.join();
}

void GrainVisualizer::render_loop() {
    // Always auto-detect Wayland socket if not already set — covers VNC sessions
    // and SSH where the compositor is running but the env var isn't exported.
    if (!getenv("WAYLAND_DISPLAY")) {
        char path[128];
        uid_t uid = getuid();
        for (int n = 0; n <= 3; ++n) {
            std::snprintf(path, sizeof(path), "/run/user/%u/wayland-%d", uid, n);
            if (access(path, F_OK) == 0) {
                char name[32];
                std::snprintf(name, sizeof(name), "wayland-%d", n);
                setenv("WAYLAND_DISPLAY", name, 0);
                std::fprintf(stderr, "[viz] auto-set WAYLAND_DISPLAY=%s\n", name);
                break;
            }
        }
    }

    // Try drivers in preference order: Wayland (local/VNC) → kmsdrm → x11.
    static const char* kDrivers[] = {"wayland", "kmsdrm", "x11", nullptr};
    bool inited = false;
    for (int i = 0; kDrivers[i]; ++i) {
        SDL_SetHint(SDL_HINT_VIDEODRIVER, kDrivers[i]);
        if (SDL_Init(SDL_INIT_VIDEO) == 0) {
            std::fprintf(stderr, "[viz] SDL video driver: %s\n", kDrivers[i]);
            inited = true;
            break;
        }
        std::fprintf(stderr, "[viz] %s failed: %s\n", kDrivers[i], SDL_GetError());
        SDL_Quit();
    }
    if (!inited) {
        std::fprintf(stderr, "[viz] no usable SDL video driver (tried kmsdrm/wayland/x11) — visualizer disabled\n");
        return;
    }

    window_ = SDL_CreateWindow("Grain Space", 0, 0, w_, h_, 0);
    if (!window_) {
        std::fprintf(stderr, "[viz] SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return;
    }

    renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer_) {
        renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_SOFTWARE);
    }
    if (!renderer_) {
        std::fprintf(stderr, "[viz] SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window_);
        SDL_Quit();
        return;
    }

    // Use actual output size (handles HiDPI and compositor scaling).
    SDL_GetRendererOutputSize(renderer_, &w_, &h_);
    // Scale so ±0.5 world units fill ~80% of the shorter screen dimension.
    scale_ = static_cast<float>(std::min(w_, h_)) / (kFocal / kCameraZ) * 0.45f;
    std::fprintf(stderr, "[viz] render size: %dx%d  scale: %.0f\n", w_, h_, scale_);

    while (run_.load(std::memory_order_relaxed)) {
        auto t0 = steady_clock::now();

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) run_.store(false);
        }

        snapshot();
        render_frame();
        SDL_RenderPresent(renderer_);

        auto dt = steady_clock::now() - t0;
        if (dt < milliseconds(33))
            std::this_thread::sleep_for(milliseconds(33) - dt);
    }

    SDL_DestroyRenderer(renderer_);
    SDL_DestroyWindow(window_);
    SDL_Quit();
}

void GrainVisualizer::snapshot() {
    if (!store_.mtx.try_lock()) return;

    points_.clear();
    for (auto& [id, slice] : store_.slices) {
        const auto& f = slice.features;
        GrainPoint p;
        p.id = id;
        // tonal: centred at 0 (neutral), ±1.0 maps to ±0.5 — stable across key changes
        p.x  = f.tonal_alignment_score * 0.5f;
        // flatness: nearly all data in [0, 0.008]; sqrt-transform to spread the cluster
        float flat_n = f.spectral_flatness / 0.008f;
        if (flat_n > 1.0f) flat_n = 1.0f;
        p.y  = std::sqrt(flat_n) - 0.5f;
        // rms: observed [0.009, 0.164] → map full range to [-0.5, 0.5]
        float rms_n = (f.rms - 0.009f) / 0.155f;
        if (rms_n > 1.0f) rms_n = 1.0f;
        if (rms_n < 0.0f) rms_n = 0.0f;
        p.z           = rms_n - 0.5f;
        p.rolloff_norm = f.rolloff_freq / 20000.0f;
        if (p.rolloff_norm > 1.0f) p.rolloff_norm = 1.0f;
        if (p.rolloff_norm < 0.0f) p.rolloff_norm = 0.0f;
        p.rms_norm = rms_n;
        points_.push_back(p);
    }

    store_.mtx.unlock();
}

void GrainVisualizer::render_frame() {
    SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
    SDL_RenderClear(renderer_);

    float ty  = theta_y_;
    float tx  = theta_x_;
    float cty = std::cos(ty), sty = std::sin(ty);
    float ctx = std::cos(tx), stx = std::sin(tx);

    auto rotate = [&](float x, float y, float z, float& xr, float& yr, float& zr) {
        xr =  x * cty + z * sty;
        yr =  x * stx * sty + y * ctx - z * stx * cty;
        zr = -x * ctx * sty + y * stx + z * ctx * cty;
    };

    auto project = [&](float xr, float yr, float zr, int& sx, int& sy) -> bool {
        float denom = kCameraZ - zr;
        if (denom <= 0.01f) return false;
        float w = kFocal / denom;
        sx = (int)(w_ / 2 + xr * w * scale_);
        sy = (int)(h_ / 2 - yr * w * scale_);
        return true;
    };

    // axis lines: draw full span -0.5 to +0.5 so they pass through the data cloud.
    // +half is bright, -half is dim to preserve orientation sense.
    struct Axis { float dx, dy, dz; uint8_t r, g, b, rd, gd, bd; };
    static const Axis axes[] = {
        {0.5f,0,0, 255, 80, 80,  100,30,30},   // X: bright red / dim red
        {0,0.5f,0,  80,255, 80,   30,100,30},   // Y: bright green / dim green
        {0,0,0.5f,  80, 80,255,   30, 30,100},  // Z: bright blue / dim blue
    };
    for (auto& a : axes) {
        float ax0,ay0,az0, ax1,ay1,az1;
        rotate(-a.dx,-a.dy,-a.dz, ax0,ay0,az0);
        rotate( a.dx, a.dy, a.dz, ax1,ay1,az1);
        int sx0,sy0, sx1,sy1;
        // negative half (dim)
        float ox,oy,oz; rotate(0,0,0,ox,oy,oz);
        int cx,cy; project(ox,oy,oz,cx,cy);
        if (project(ax0,ay0,az0,sx0,sy0)) {
            SDL_SetRenderDrawColor(renderer_, a.rd, a.gd, a.bd, 255);
            SDL_RenderDrawLine(renderer_, cx, cy, sx0, sy0);
        }
        // positive half (bright)
        if (project(ax1,ay1,az1,sx1,sy1)) {
            SDL_SetRenderDrawColor(renderer_, a.r, a.g, a.b, 255);
            SDL_RenderDrawLine(renderer_, cx, cy, sx1, sy1);
        }
    }

    // draw grain points
    int current_id = current_id_.load(std::memory_order_relaxed);
    for (auto& p : points_) {
        float xr, yr, zr;
        rotate(p.x, p.y, p.z, xr, yr, zr);
        int sx, sy;
        if (!project(xr, yr, zr, sx, sy)) continue;
        if (sx < 0 || sx >= w_ || sy < 0 || sy >= h_) continue;

        if (p.id == current_id) {
            SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);
            SDL_Rect rect = {sx - 3, sy - 3, 7, 7};
            SDL_RenderDrawRect(renderer_, &rect);
        } else {
            // hue: blue (low rolloff/bass) → red (high rolloff/bright)
            float hue = (1.0f - p.rolloff_norm) * (240.0f / 360.0f);
            float val = 0.5f + 0.5f * p.rms_norm;
            SDL_Color col = hsv_to_rgb(hue, 0.9f, val);
            SDL_SetRenderDrawColor(renderer_, col.r, col.g, col.b, 255);
            if (zr > 0.1f) {
                SDL_Rect rect = {sx - 1, sy - 1, 3, 3};
                SDL_RenderFillRect(renderer_, &rect);
            } else {
                SDL_RenderDrawPoint(renderer_, sx, sy);
            }
        }
    }

    theta_y_ += 0.004f;
}

SDL_Color GrainVisualizer::hsv_to_rgb(float h, float s, float v) {
    float r, g, b;
    int   i = (int)(h * 6.0f);
    float f = h * 6.0f - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
    return SDL_Color{
        (uint8_t)(r * 255),
        (uint8_t)(g * 255),
        (uint8_t)(b * 255),
        255
    };
}
