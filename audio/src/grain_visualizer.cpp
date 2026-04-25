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
                                  std::atomic<int>&   gain_pending,
                                  std::atomic<float>& crossfade,
                                  std::mutex&         bank_menu_mtx,
                                  BankMenuDisplay&    bank_menu_disp,
                                  std::mutex&         toast_mtx,
                                  ParamToast&         toast_disp)
    : store_a_(store_a), store_b_(store_b),
      run_(run), current_id_(current_id), current_bank_(current_bank),
      gain_pending_(gain_pending), crossfade_(crossfade),
      bank_menu_mtx_(bank_menu_mtx), bank_menu_disp_(bank_menu_disp),
      toast_mtx_(toast_mtx), toast_disp_(toast_disp) {}

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

    // Try drivers in preference order: Wayland → kmsdrm → x11.
    // If x11 fails and we're in an SSH session with a forwarded/mismatched DISPLAY,
    // retry x11 with the local Xorg socket directly.
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

    // Fallback: SSH X11 forwarding sets DISPLAY to a tunnel whose auth cookie
    // hostname often doesn't match. If there's a local Xorg socket, point at it.
    if (!inited) {
        static const char* kLocalDisplays[] = {":0", ":1", nullptr};
        for (int i = 0; kLocalDisplays[i] && !inited; ++i) {
            char path[64];
            std::snprintf(path, sizeof(path), "/tmp/.X11-unix/X%s", kLocalDisplays[i] + 1);
            if (access(path, F_OK) != 0) continue;
            setenv("DISPLAY", kLocalDisplays[i], 1);
            if (!getenv("XAUTHORITY")) {
                const char* home = getenv("HOME");
                if (home) {
                    char xauth[256];
                    std::snprintf(xauth, sizeof(xauth), "%s/.Xauthority", home);
                    if (access(xauth, F_OK) == 0)
                        setenv("XAUTHORITY", xauth, 0);
                }
            }
            SDL_SetHint(SDL_HINT_VIDEODRIVER, "x11");
            if (SDL_Init(SDL_INIT_VIDEO) == 0) {
                std::fprintf(stderr, "[viz] SDL video driver: x11 (local fallback %s)\n", kLocalDisplays[i]);
                inited = true;
            } else {
                std::fprintf(stderr, "[viz] x11 local fallback %s failed: %s\n", kLocalDisplays[i], SDL_GetError());
                SDL_Quit();
            }
        }
    }

    if (!inited) {
        std::fprintf(stderr, "[viz] no usable SDL video driver — visualizer disabled\n");
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

    // draw grain points — bank A: blue→red, bank B: hue+120° (green→magenta)
    int cur_id   = current_id_.load(std::memory_order_relaxed);
    int cur_bank = current_bank_.load(std::memory_order_relaxed);
    for (auto& p : points_) {
        float xr, yr, zr;
        rotate(p.x, p.y, p.z, xr, yr, zr);
        int sx, sy;
        if (!project(xr, yr, zr, sx, sy)) continue;
        if (sx < 0 || sx >= w_ || sy < 0 || sy >= h_) continue;

        bool is_current = (p.id == cur_id && (p.from_b ? 1 : 0) == cur_bank);

        if (is_current) {
            SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);
            SDL_Rect rect = {sx - 3, sy - 3, 7, 7};
            SDL_RenderDrawRect(renderer_, &rect);
        } else if (p.flash > 0.0f) {
            float base_hue = (1.0f - p.rolloff_norm) * (240.0f / 360.0f);
            float hue = p.from_b ? std::fmod(base_hue + 0.333f, 1.0f) : base_hue;
            float val = 0.3f + 0.7f * p.flatness_norm;
            SDL_Color hue_col = hsv_to_rgb(hue, 0.9f, val);
            float t = p.flash;
            uint8_t r = (uint8_t)(255 * t + hue_col.r * (1.0f - t));
            uint8_t g = (uint8_t)(255 * t + hue_col.g * (1.0f - t));
            uint8_t b = (uint8_t)(255 * t + hue_col.b * (1.0f - t));
            SDL_SetRenderDrawColor(renderer_, r, g, b, 255);
            int half = (int)(1.0f + 3.0f * t);
            SDL_Rect rect = {sx - half, sy - half, half * 2 + 1, half * 2 + 1};
            SDL_RenderFillRect(renderer_, &rect);
        } else {
            float base_hue = (1.0f - p.rolloff_norm) * (240.0f / 360.0f);
            float hue = p.from_b ? std::fmod(base_hue + 0.333f, 1.0f) : base_hue;
            float val = 0.3f + 0.7f * p.flatness_norm;
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

    // gain-change prompt overlay
    int pending = gain_pending_.load(std::memory_order_relaxed);
    if (pending != 0 && font_) {
        const int kBarH = 52;
        const int kPad  = 8;

        SDL_SetRenderDrawColor(renderer_, 180, 120, 0, 255);
        SDL_Rect bar = {0, h_ - kBarH, w_, kBarH};
        SDL_RenderFillRect(renderer_, &bar);

        SDL_Color white  = {255, 255, 255, 255};
        SDL_Color green  = { 80, 220,  80, 255};
        SDL_Color red    = {220,  80,  80, 255};

        const char* action = (pending == 1) ? "Switch to INSTRUMENT gain?" : "Switch to LINE gain?";
        draw_text(action,           kPad,           h_ - kBarH + kPad,      white);
        draw_text("Enc4 = confirm", kPad,           h_ - kBarH + kPad + 26, green);
        draw_text("MBtn = cancel",  w_ / 2 + kPad,  h_ - kBarH + kPad + 26, red);
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
            if (snap.page == 0) n_items = 10;
            else if (snap.page == 1 || snap.page == 2)
                n_items = (int)snap.file_list.size() + (snap.file_list.empty() ? 1 : 0);
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
                draw_text("BANK MENU", kPanelX, y, yellow); y += kLineH + 4;

                std::string name_a = snap.bank_name_a.empty() ? "<empty>" : snap.bank_name_a;
                std::string name_b = snap.bank_name_b.empty() ? "<empty>" : snap.bank_name_b;
                std::string rec_toggle = std::string("Record -> ") +
                    (snap.record_target == 0 ? "B (currently A)" : "A (currently B)");
                const std::string items[10] = {
                    "Slot A: " + name_a,
                    "Slot B: " + name_b,
                    "Load -> A",
                    "Load -> B",
                    "Save A as new",
                    "Save B as new",
                    "Clear A",
                    "Clear B",
                    rec_toggle,
                    "Exit"
                };
                for (int i = 0; i < 10; i++) {
                    bool selected = (i == snap.cursor);
                    if (selected) {
                        SDL_SetRenderDrawColor(renderer_, 60, 55, 0, 255);
                        SDL_Rect hl = {kPanelX - 4, y - 2, kPanelW + 8, kLineH};
                        SDL_RenderFillRect(renderer_, &hl);
                    }
                    SDL_Color col = selected ? yellow : (i < 2 ? gray : white);
                    draw_text(items[i].c_str(), kPanelX, y, col);
                    y += kLineH;
                }
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
                }
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
