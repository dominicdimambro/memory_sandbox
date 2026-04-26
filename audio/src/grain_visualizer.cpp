#include "grain_visualizer.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unistd.h>

using namespace std::chrono;

GrainVisualizer::GrainVisualizer(SliceStore& store_a, SliceStore& store_b,
                                  std::atomic<bool>&  run,
                                  std::atomic<int>&   current_id,
                                  std::atomic<int>&   current_bank,
                                  std::atomic<float>& crossfade,
                                  std::mutex&         bank_menu_mtx,
                                  BankMenuDisplay&    bank_menu_disp,
                                  std::mutex&         toast_mtx,
                                  ParamToast&         toast_disp,
                                  std::atomic<int>&   engine_mode,
                                  std::atomic<float>& explore_x,
                                  std::atomic<float>& explore_y,
                                  std::atomic<float>& explore_z,
                                  std::atomic<float>& search_radius,
                                  std::atomic<float>& view_theta_y,
                                  std::atomic<float>& view_theta_x,
                                  std::atomic<float>& view_zoom)
    : store_a_(store_a), store_b_(store_b),
      run_(run), current_id_(current_id), current_bank_(current_bank),
      crossfade_(crossfade),
      bank_menu_mtx_(bank_menu_mtx), bank_menu_disp_(bank_menu_disp),
      toast_mtx_(toast_mtx), toast_disp_(toast_disp),
      engine_mode_(engine_mode),
      explore_x_(explore_x), explore_y_(explore_y), explore_z_(explore_z),
      search_radius_(search_radius),
      view_theta_y_(view_theta_y), view_theta_x_(view_theta_x), view_zoom_(view_zoom) {}

void GrainVisualizer::start() {
    thread_ = std::thread(&GrainVisualizer::render_loop, this);
}

void GrainVisualizer::stop() {
    if (thread_.joinable()) thread_.join();
}

void GrainVisualizer::render_loop() {
    // Auto-detect Wayland socket — covers SSH sessions where the compositor runs
    // but the env var isn't exported.
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

    // Try drivers in preference order, with retries for service startup timing.
    // kmsdrm is tried on both card0 and card1 (Pi 4/5 often uses card1 for HDMI).
    // Retries handle the case where the DRM subsystem isn't ready at service start.
    // The reinit label allows re-entry after persistent pageflip failures (wrong card).
    reinit:
    w_ = 800; h_ = 540;
    bool inited = false;
    static constexpr int kMaxAttempts = 15;
    for (int attempt = 0; attempt < kMaxAttempts && !inited && run_.load(); ++attempt) {
        if (attempt > 0)
            std::this_thread::sleep_for(std::chrono::seconds(1));

        // wayland
        if (!inited) {
            SDL_SetHint(SDL_HINT_VIDEODRIVER, "wayland");
            if (SDL_Init(SDL_INIT_VIDEO) == 0) {
                std::fprintf(stderr, "[viz] SDL video driver: wayland\n");
                inited = true;
            } else { SDL_Quit(); }
        }

        // kmsdrm — try card index 0 then 1
        for (int card = 0; card <= 1 && !inited; ++card) {
            char idx[2] = {(char)('0' + card), '\0'};
            SDL_SetHint(SDL_HINT_VIDEODRIVER, "kmsdrm");
            SDL_SetHint("SDL_VIDEO_KMSDRM_DEVICE_INDEX", idx);
            if (SDL_Init(SDL_INIT_VIDEO) == 0) {
                std::fprintf(stderr, "[viz] SDL video driver: kmsdrm (card%d)\n", card);
                inited = true;
            } else { SDL_Quit(); }
        }

        // x11 / local fallback
        if (!inited) {
            SDL_SetHint(SDL_HINT_VIDEODRIVER, "x11");
            if (SDL_Init(SDL_INIT_VIDEO) == 0) {
                std::fprintf(stderr, "[viz] SDL video driver: x11\n");
                inited = true;
            } else {
                SDL_Quit();
                static const char* kLocal[] = {":0", ":1", nullptr};
                for (int i = 0; kLocal[i] && !inited; ++i) {
                    char path[64];
                    std::snprintf(path, sizeof(path), "/tmp/.X11-unix/X%s", kLocal[i] + 1);
                    if (access(path, F_OK) != 0) continue;
                    setenv("DISPLAY", kLocal[i], 1);
                    SDL_SetHint(SDL_HINT_VIDEODRIVER, "x11");
                    if (SDL_Init(SDL_INIT_VIDEO) == 0) {
                        std::fprintf(stderr, "[viz] SDL video driver: x11 (local %s)\n", kLocal[i]);
                        inited = true;
                    } else { SDL_Quit(); }
                }
            }
        }

        if (!inited && attempt == 0)
            std::fprintf(stderr, "[viz] display not ready, retrying...\n");
    }

    if (!inited) {
        std::fprintf(stderr, "[viz] no usable SDL video driver after %d attempts — visualizer disabled\n", kMaxAttempts);
        return;
    }

    window_ = SDL_CreateWindow("Grain Space", 0, 0, w_, h_, SDL_WINDOW_FULLSCREEN_DESKTOP);
    if (!window_) {
        std::fprintf(stderr, "[viz] SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return;
    }

    SDL_ShowCursor(SDL_DISABLE);

    TTF_Init();
    font_ = TTF_OpenFont("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20);
    if (!font_)
        font_ = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20);

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

    SDL_GetRendererOutputSize(renderer_, &w_, &h_);
    scale_ = static_cast<float>(std::min(w_, h_)) / (kFocal / kCameraZ) * 0.45f;
    std::fprintf(stderr, "[viz] render size: %dx%d  scale: %.0f\n", w_, h_, scale_);

    // If SDL opened a card but page-flipping fails (wrong DRM device), we detect it by
    // measuring wall time: if 3 seconds pass and the loop is still running far faster
    // than 30fps (because SDL_RenderPresent returns immediately on error), reinitialise.
    auto init_time = steady_clock::now();
    int  frame_count = 0;

    while (run_.load(std::memory_order_relaxed)) {
        auto t0 = steady_clock::now();

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) run_.store(false);
        }

        snapshot();
        render_frame();
        SDL_RenderPresent(renderer_);
        frame_count++;

        // detect pageflip failure: if we've rendered >150 frames in under 3s (>>30fps)
        // the present is returning immediately rather than syncing to vsync
        auto elapsed = duration<float>(steady_clock::now() - init_time).count();
        if (elapsed > 3.0f && frame_count > 150 && (frame_count / elapsed) > 45.0f) {
            std::fprintf(stderr, "[viz] pageflip failure detected (%.0f fps), trying other card\n",
                         frame_count / elapsed);
            SDL_DestroyRenderer(renderer_); renderer_ = nullptr;
            SDL_DestroyWindow(window_);     window_   = nullptr;
            SDL_Quit();
            std::this_thread::sleep_for(std::chrono::seconds(2));
            goto reinit;
        }

        auto dt = steady_clock::now() - t0;
        if (dt < milliseconds(33))
            std::this_thread::sleep_for(milliseconds(33) - dt);
    }

    SDL_DestroyRenderer(renderer_);
    SDL_DestroyWindow(window_);
    if (font_) TTF_CloseFont(font_);
    TTF_Quit();
    SDL_Quit();
}

void GrainVisualizer::snapshot() {
    auto now = steady_clock::now();
    points_.clear();

    auto add_grains = [&](SliceStore& s, bool from_b,
                           std::unordered_map<int, steady_clock::time_point>& btimes) {
        if (!s.mtx.try_lock()) return;
        for (auto& [id, slice] : s.slices) {
            const auto& f = slice.features;
            GrainPoint p;
            p.id     = id;
            p.from_b = from_b;
            if (btimes.find(id) == btimes.end())
                btimes[id] = now;
            float age = duration<float>(now - btimes[id]).count();
            p.flash = std::max(0.0f, 1.0f - age / 2.0f);
            p.x = f.tonal_alignment_score * 0.5f;
            {
                const float log_min = 3.850f, log_max = 9.393f;
                float lr = std::log(std::max(f.rolloff_freq, 47.f));
                p.y = (lr - log_min) / (log_max - log_min) - 0.5f;
                if (p.y > 0.5f) p.y = 0.5f;
                if (p.y < -0.5f) p.y = -0.5f;
            }
            float rms_n = std::sqrt(f.rms / 0.12f);
            if (rms_n > 1.0f) rms_n = 1.0f;
            p.z = rms_n - 0.5f;
            p.rolloff_norm = f.rolloff_freq / 20000.0f;
            if (p.rolloff_norm > 1.0f) p.rolloff_norm = 1.0f;
            if (p.rolloff_norm < 0.0f) p.rolloff_norm = 0.0f;
            float flat_n = std::sqrt(f.spectral_flatness / 0.008f);
            p.flatness_norm = flat_n > 1.0f ? 1.0f : flat_n;
            points_.push_back(p);
        }
        s.mtx.unlock();
    };

    add_grains(store_a_, false, birth_times_a_);
    add_grains(store_b_, true,  birth_times_b_);
}

void GrainVisualizer::render_frame() {
    SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
    SDL_RenderClear(renderer_);

    int   mode = engine_mode_.load(std::memory_order_relaxed);
    float ty, tx, zoom;
    if (mode == 0) {
        ty   = view_theta_y_.load(std::memory_order_relaxed);
        tx   = view_theta_x_.load(std::memory_order_relaxed);
        zoom = view_zoom_.load(std::memory_order_relaxed);
    } else {
        ty   = theta_y_;
        tx   = theta_x_;
        zoom = 2.0f;
    }

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
        sx = (int)(w_ / 2 + xr * w * scale_ * zoom);
        sy = (int)(h_ / 2 - yr * w * scale_ * zoom);
        return true;
    };

    // reusable text helper
    auto draw_text = [&](const char* txt, int x, int y, SDL_Color col) {
        if (!font_) return;
        SDL_Surface* surf = TTF_RenderUTF8_Blended(font_, txt, col);
        if (!surf) return;
        SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer_, surf);
        if (tex) {
            SDL_Rect dst = {x, y, surf->w, surf->h};
            SDL_RenderCopy(renderer_, tex, nullptr, &dst);
            SDL_DestroyTexture(tex);
        }
        SDL_FreeSurface(surf);
    };

    // axis lines
    struct Axis { float dx, dy, dz; uint8_t r, g, b, rd, gd, bd; };
    static const Axis axes[] = {
        {0.5f,0,0, 255, 80, 80,  100,30,30},
        {0,0.5f,0,  80,255, 80,   30,100,30},
        {0,0,0.5f,  80, 80,255,   30, 30,100},
    };
    for (auto& a : axes) {
        float ax0,ay0,az0, ax1,ay1,az1;
        rotate(-a.dx,-a.dy,-a.dz, ax0,ay0,az0);
        rotate( a.dx, a.dy, a.dz, ax1,ay1,az1);
        int sx0,sy0, sx1,sy1;
        float ox,oy,oz; rotate(0,0,0,ox,oy,oz);
        int cx,cy; project(ox,oy,oz,cx,cy);
        if (project(ax0,ay0,az0,sx0,sy0)) {
            SDL_SetRenderDrawColor(renderer_, a.rd, a.gd, a.bd, 255);
            SDL_RenderDrawLine(renderer_, cx, cy, sx0, sy0);
        }
        if (project(ax1,ay1,az1,sx1,sy1)) {
            SDL_SetRenderDrawColor(renderer_, a.r, a.g, a.b, 255);
            SDL_RenderDrawLine(renderer_, cx, cy, sx1, sy1);
        }
    }

    // draw grain points — bank A: cyan, bank B: orange
    // density-adaptive size: half=2 (5×5) when sparse, half=1 (3×3) medium, point when dense
    int n_pts = (int)points_.size();
    int grain_half = (n_pts < 80) ? 2 : (n_pts < 300) ? 1 : 0;

    // crossfader alpha: full brightness from center to active side, fade only past center
    float cf = crossfade_.load(std::memory_order_relaxed);
    float scale_a = (cf < 0.5f) ? 1.0f : 2.0f * (1.0f - cf);
    float scale_b = (cf > 0.5f) ? 1.0f : 2.0f * cf;
    uint8_t alpha_a = (uint8_t)(50 + (int)(205 * scale_a));
    uint8_t alpha_b = (uint8_t)(50 + (int)(205 * scale_b));

    SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);

    int cur_id   = current_id_.load(std::memory_order_relaxed);
    int cur_bank = current_bank_.load(std::memory_order_relaxed);
    for (auto& p : points_) {
        float xr, yr, zr;
        rotate(p.x, p.y, p.z, xr, yr, zr);
        int sx, sy;
        if (!project(xr, yr, zr, sx, sy)) continue;
        if (sx < 0 || sx >= w_ || sy < 0 || sy >= h_) continue;

        uint8_t alpha = p.from_b ? alpha_b : alpha_a;
        bool is_current = (p.id == cur_id && (p.from_b ? 1 : 0) == cur_bank);

        if (is_current) {
            SDL_SetRenderDrawColor(renderer_, 255, 255, 255, alpha);
            SDL_Rect rect = {sx - 3, sy - 3, 7, 7};
            SDL_RenderDrawRect(renderer_, &rect);
        } else if (p.flash > 0.0f) {
            uint8_t br = p.from_b ? 255 :   0;
            uint8_t bg = p.from_b ? 150 : 200;
            uint8_t bb = p.from_b ?   0 : 220;
            float t = p.flash;
            SDL_SetRenderDrawColor(renderer_,
                (uint8_t)(255 * t + br * (1.0f - t)),
                (uint8_t)(255 * t + bg * (1.0f - t)),
                (uint8_t)(255 * t + bb * (1.0f - t)), alpha);
            int half = (int)(1.0f + 3.0f * t);
            SDL_Rect rect = {sx - half, sy - half, half * 2 + 1, half * 2 + 1};
            SDL_RenderFillRect(renderer_, &rect);
        } else {
            if (p.from_b) SDL_SetRenderDrawColor(renderer_, 255, 150,   0, alpha);
            else          SDL_SetRenderDrawColor(renderer_,   0, 200, 220, alpha);
            if (grain_half > 0) {
                SDL_Rect rect = {sx - grain_half, sy - grain_half,
                                 grain_half * 2 + 1, grain_half * 2 + 1};
                SDL_RenderFillRect(renderer_, &rect);
            } else {
                SDL_RenderDrawPoint(renderer_, sx, sy);
            }
        }
    }

    SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_NONE);

    if (mode == 1) theta_y_ += 0.004f;  // auto-rotate only in exploration mode

    // exploration mode: centroid cursor + density-adaptive sphere
    if (mode == 1) {
        static constexpr float kPi = 3.14159265f;
        float cx = explore_x_.load(std::memory_order_relaxed);
        float cy = explore_y_.load(std::memory_order_relaxed);
        float cz = explore_z_.load(std::memory_order_relaxed);
        float r  = search_radius_.load(std::memory_order_relaxed);

        float cxr, cyr, czr;
        rotate(cx, cy, cz, cxr, cyr, czr);
        int csx, csy;
        if (project(cxr, cyr, czr, csx, csy)) {
            // crosshair
            SDL_SetRenderDrawColor(renderer_, 0, 255, 210, 255);
            SDL_RenderDrawLine(renderer_, csx - 14, csy, csx + 14, csy);
            SDL_RenderDrawLine(renderer_, csx, csy - 14, csx, csy + 14);
            SDL_Rect dot = {csx - 3, csy - 3, 7, 7};
            SDL_RenderDrawRect(renderer_, &dot);
        }

        if (r > 0.005f) {
            static constexpr int kSteps = 48;
            SDL_SetRenderDrawColor(renderer_, 0, 180, 160, 255);
            // draw 3 great circles: XY, XZ, YZ planes
            for (int plane = 0; plane < 3; plane++) {
                int prev_sx = 0, prev_sy = 0;
                bool prev_valid = false;
                for (int s = 0; s <= kSteps; s++) {
                    float angle = s * 2.0f * kPi / kSteps;
                    float ca = std::cos(angle), sa = std::sin(angle);
                    float px, py, pz;
                    if      (plane == 0) { px = cx + r*ca; py = cy + r*sa; pz = cz; }
                    else if (plane == 1) { px = cx + r*ca; py = cy;        pz = cz + r*sa; }
                    else                 { px = cx;        py = cy + r*ca; pz = cz + r*sa; }
                    float xr, yr, zr;
                    rotate(px, py, pz, xr, yr, zr);
                    int sx, sy;
                    bool valid = project(xr, yr, zr, sx, sy);
                    if (valid && prev_valid)
                        SDL_RenderDrawLine(renderer_, prev_sx, prev_sy, sx, sy);
                    prev_sx = sx; prev_sy = sy; prev_valid = valid;
                }
            }
        }
    }

    // crossfader bar (always visible at bottom)
    {
        float cf = crossfade_.load(std::memory_order_relaxed);
        const int kBarH = 8;
        const int kBarW = (int)(w_ * 0.7f);
        const int kBarX = (w_ - kBarW) / 2;
        const int kBarY = h_ - kBarH - 20;

        SDL_SetRenderDrawColor(renderer_, 50, 50, 50, 255);
        SDL_Rect bg = {kBarX, kBarY, kBarW, kBarH};
        SDL_RenderFillRect(renderer_, &bg);

        int thumb_x = kBarX + (int)(cf * (float)kBarW);
        SDL_SetRenderDrawColor(renderer_, 220, 220, 220, 255);
        SDL_Rect thumb = {thumb_x - 4, kBarY - 3, 8, kBarH + 6};
        SDL_RenderFillRect(renderer_, &thumb);

        if (font_) {
            SDL_Color wh = {200, 200, 200, 255};
            draw_text("A", kBarX - 20, kBarY - 4, wh);
            draw_text("B", kBarX + kBarW + 6, kBarY - 4, wh);
        }
    }

    // axis legend (top-left)
    if (font_) {
        struct LegendEntry { uint8_t r, g, b; const char* label; bool square; };
        static const LegendEntry kLegend[] = {
            {  0, 200, 220, "A  bank A", true},
            {255, 150,   0, "B  bank B", true},
            {255,  80,  80, "X  tonal",  false},
            { 80, 255,  80, "Y  rolloff",false},
            { 80,  80, 255, "Z  rms",    false},
        };
        int lx = 12, ly = 10;
        for (int i = 0; i < 5; i++) {
            if (i == 2) ly += 8;  // gap between banks and axes
            const auto& e = kLegend[i];
            SDL_SetRenderDrawColor(renderer_, e.r, e.g, e.b, 255);
            if (e.square) {
                SDL_Rect swatch = {lx + 2, ly + 5, 12, 12};
                SDL_RenderFillRect(renderer_, &swatch);
            } else {
                SDL_Rect swatch = {lx, ly + 8, 20, 4};
                SDL_RenderFillRect(renderer_, &swatch);
            }
            SDL_Color col = {e.r, e.g, e.b, 255};
            draw_text(e.label, lx + 28, ly, col);
            ly += 26;
        }
    }

    // bank menu overlay
    {
        BankMenuDisplay snap;
        bool got = false;
        if (bank_menu_mtx_.try_lock()) {
            snap = bank_menu_disp_;
            got  = true;
            bank_menu_mtx_.unlock();
        }
        if (got && snap.open && font_) {
            const int kLineH  = 26;
            const int kPanelW = std::min(420, w_ - 40);
            const int kPanelX = (w_ - kPanelW) / 2;
            const int kPanelY = 20;
            const int kPad    = 8;

            // count items for height
            int n_items = 0;
            if      (snap.page == 0)                   n_items = 8;
            else if (snap.page == 1 || snap.page == 2) n_items = (int)snap.file_list.size() + (snap.file_list.empty() ? 1 : 0);
            else if (snap.page == 5 || snap.page == 6) n_items = 2;
            else if (snap.page == 7 || snap.page == 8) n_items = 3;
            else if (snap.page == 9)                   n_items = 3;
            else if (snap.page == 10 || snap.page == 11) n_items = 4;
            else if (snap.page == 12)                  n_items = 1;
            else n_items = 2;  // ConfirmClear

            int panel_h = kPad + kLineH + 4 + n_items * kLineH + kLineH + kPad;

            SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 210);
            SDL_Rect panel = {kPanelX - kPad, kPanelY - kPad, kPanelW + kPad * 2, panel_h};
            SDL_RenderFillRect(renderer_, &panel);
            SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_NONE);

            SDL_Color white  = {255, 255, 255, 255};
            SDL_Color yellow = {255, 220,  60, 255};
            SDL_Color gray   = {140, 140, 140, 255};
            SDL_Color green  = { 80, 220,  80, 255};
            SDL_Color red    = {220,  80,  80, 255};

            int y = kPanelY;

            if (snap.page == 0) {
                draw_text("MENU", kPanelX, y, yellow); y += kLineH + 4;

                std::string name_a = snap.bank_name_a.empty() ? "<empty>" : snap.bank_name_a;
                std::string name_b = snap.bank_name_b.empty() ? "<empty>" : snap.bank_name_b;
                std::string rec_toggle = std::string("Record -> ") +
                    (snap.record_target == 0 ? "B (currently A)" : "A (currently B)");
                int cur_mode = engine_mode_.load(std::memory_order_relaxed);
                std::string mode_item = std::string("Mode: ") +
                    (cur_mode == 0 ? "Analysis" : "Exploration");
                std::string gain_item = snap.trs_instrument
                    ? "Gain: INSTRUMENT"
                    : "Gain: LINE";
                const std::string items[8] = {
                    "Bank A: " + name_a,
                    "Bank B: " + name_b,
                    rec_toggle,
                    gain_item,
                    mode_item,
                    "Parameters",
                    "Shutdown",
                    "Exit"
                };
                for (int i = 0; i < 8; i++) {
                    bool selected = (i == snap.cursor);
                    if (selected) {
                        SDL_SetRenderDrawColor(renderer_, 60, 55, 0, 255);
                        SDL_Rect hl = {kPanelX - 4, y - 2, kPanelW + 8, kLineH};
                        SDL_RenderFillRect(renderer_, &hl);
                    }
                    SDL_Color col = selected ? yellow : white;
                    draw_text(items[i].c_str(), kPanelX, y, col);
                    y += kLineH;
                }
            } else if (snap.page == 10 || snap.page == 11) {
                bool is_a = (snap.page == 10);
                draw_text(is_a ? "BANK A" : "BANK B", kPanelX, y, yellow); y += kLineH + 4;
                const char* sub_items[4] = {"Load", "Save as new",
                    snap.has_file_a && is_a ? "Overwrite" :
                    snap.has_file_b && !is_a ? "Overwrite" : "Overwrite",
                    "Clear"};
                bool overwrite_active = is_a ? snap.has_file_a : snap.has_file_b;
                for (int i = 0; i < 4; i++) {
                    bool selected = (i == snap.cursor);
                    if (selected) {
                        SDL_SetRenderDrawColor(renderer_, 60, 55, 0, 255);
                        SDL_Rect hl = {kPanelX - 4, y - 2, kPanelW + 8, kLineH};
                        SDL_RenderFillRect(renderer_, &hl);
                    }
                    SDL_Color col = selected ? yellow
                                  : (i == 2 && !overwrite_active) ? gray : white;
                    draw_text(sub_items[i], kPanelX, y, col);
                    y += kLineH;
                }
            } else if (snap.page == 12) {
                draw_text("PARAMETERS", kPanelX, y, yellow); y += kLineH + 4;
                SDL_SetRenderDrawColor(renderer_, 60, 55, 0, 255);
                SDL_Rect hl = {kPanelX - 4, y - 2, kPanelW + 8, kLineH};
                SDL_RenderFillRect(renderer_, &hl);
                char dbuf[48];
                std::snprintf(dbuf, sizeof(dbuf), "Delay time:  %d ms", snap.delay_ms);
                draw_text(dbuf, kPanelX, y, yellow); y += kLineH;
                draw_text("enc4=adjust  M=back", kPanelX, y + 4, gray);
                y += kLineH;
            } else if (snap.page == 1 || snap.page == 2) {
                const char* title = (snap.page == 1) ? "LOAD INTO SLOT A" : "LOAD INTO SLOT B";
                draw_text(title, kPanelX, y, yellow); y += kLineH + 4;
                if (snap.file_list.empty()) {
                    draw_text("<no banks saved>", kPanelX, y, gray); y += kLineH;
                } else {
                    for (int i = 0; i < (int)snap.file_list.size(); i++) {
                        bool selected = (i == snap.cursor);
                        if (selected) {
                            SDL_SetRenderDrawColor(renderer_, 60, 55, 0, 255);
                            SDL_Rect hl = {kPanelX - 4, y - 2, kPanelW + 8, kLineH};
                            SDL_RenderFillRect(renderer_, &hl);
                        }
                        SDL_Color col = selected ? yellow : white;
                        draw_text(snap.file_list[i].c_str(), kPanelX, y, col);
                        y += kLineH;
                    }
                    draw_text("btn=load  M=delete", kPanelX, y + 4, gray); y += kLineH;
                }
            } else if (snap.page == 9) {
                draw_text("POWER OFF?", kPanelX, y, red); y += kLineH + 4;
                draw_text("Enc4 = confirm shutdown", kPanelX, y, green); y += kLineH;
                draw_text("MBtn = cancel",           kPanelX, y, gray);  y += kLineH;
            } else if (snap.page == 7 || snap.page == 8) {
                const char* title = (snap.page == 7) ? "DELETE FILE?" : "DELETE FILE?";
                draw_text(title, kPanelX, y, red); y += kLineH + 4;
                draw_text(snap.delete_file, kPanelX, y, white); y += kLineH;
                draw_text("Enc4 = confirm delete", kPanelX, y, green); y += kLineH;
                draw_text("MBtn = cancel",         kPanelX, y, gray);  y += kLineH;
            } else if (snap.page == 5 || snap.page == 6) {
                const char* title = (snap.page == 5) ? "SAVE A AS:" : "SAVE B AS:";
                draw_text(title, kPanelX, y, yellow); y += kLineH + 4;

                // draw 12 character cells
                static constexpr int kCellW = 22, kCellH = 28;
                int cx = kPanelX;
                for (int ci = 0; ci < 12; ci++) {
                    bool cur = (ci == snap.cursor);
                    if (cur) {
                        SDL_SetRenderDrawColor(renderer_, 70, 60, 0, 255);
                        SDL_Rect bg = {cx, y, kCellW, kCellH};
                        SDL_RenderFillRect(renderer_, &bg);
                    }
                    SDL_SetRenderDrawColor(renderer_, cur ? 255 : 100, cur ? 220 : 100, 0, 255);
                    SDL_Rect border = {cx, y, kCellW, kCellH};
                    SDL_RenderDrawRect(renderer_, &border);
                    char ch[2] = {snap.name_buf[ci] ? snap.name_buf[ci] : ' ', '\0'};
                    SDL_Color cc = cur ? yellow : white;
                    draw_text(ch, cx + 4, y + 4, cc);
                    cx += kCellW + 2;
                }
                y += kCellH + 6;
                draw_text("enc4=char  btn=next  last=save  M=back", kPanelX, y, gray);
                y += kLineH;
            } else {
                const char* title = (snap.page == 3) ? "CLEAR SLOT A?" : "CLEAR SLOT B?";
                draw_text(title, kPanelX, y, yellow); y += kLineH + 4;
                draw_text("Enc4 = confirm", kPanelX, y, green); y += kLineH;
                draw_text("MBtn = cancel",  kPanelX, y, red);   y += kLineH;
            }

            draw_text("enc4=scroll  btn=select  M=back", kPanelX, y + 4, gray);
        }
    }

    // parameter toast: shows current value while a pot is being turned, fades after 1.5s
    if (font_) {
        ParamToast tsnap;
        bool tgot = false;
        if (toast_mtx_.try_lock()) {
            tsnap = toast_disp_;
            tgot  = true;
            toast_mtx_.unlock();
        }
        if (tgot && tsnap.name[0] != '\0') {
            float age   = duration<float>(steady_clock::now() - tsnap.updated).count();
            float alpha = (age < 1.0f) ? 1.0f : 1.0f - (age - 1.0f) / 0.5f;
            if (alpha > 0.0f) {
                uint8_t a = (uint8_t)(alpha * 230.0f);

                char line[72];
                std::snprintf(line, sizeof(line), "%s: %s", tsnap.name, tsnap.value);

                int tw = 0, th = 0;
                TTF_SizeUTF8(font_, line, &tw, &th);
                int tx = (w_ - tw) / 2;
                int ty = 12;

                SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);
                SDL_SetRenderDrawColor(renderer_, 20, 20, 20, a);
                SDL_Rect bg = {tx - 12, ty - 5, tw + 24, th + 10};
                SDL_RenderFillRect(renderer_, &bg);
                SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_NONE);

                SDL_Color col = {255, 255, 255, 255};
                SDL_Surface* surf = TTF_RenderUTF8_Blended(font_, line, col);
                if (surf) {
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer_, surf);
                    if (tex) {
                        SDL_SetTextureAlphaMod(tex, a);
                        SDL_Rect dst = {tx, ty, surf->w, surf->h};
                        SDL_RenderCopy(renderer_, tex, nullptr, &dst);
                        SDL_DestroyTexture(tex);
                    }
                    SDL_FreeSurface(surf);
                }
            }
        }
    }
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
