#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "slice_store.h"

struct GrainPoint {
    int   id;
    float x, y, z;        // remapped to [-0.5, +0.5]
    float rolloff_norm;    // for color (hue): blue=bass → red=bright
    float flatness_norm;   // for brightness: dim=pure tone, bright=noisy
    float flash;           // 1.0=just born → 0.0=fully aged (over 2s)
    bool  from_b;          // true = from store_b (rendered with shifted hue)
};

// Parameter toast — written by pot thread on each change, fades out after ~1.5s.
// Protected by a dedicated mutex in main().
struct ParamToast {
    char   name[32]  = {};
    char   value[32] = {};
    std::chrono::steady_clock::time_point updated;
};

// Shared menu display state — written by encoder thread, read by visualizer.
// Protected by a dedicated mutex in main().
struct BankMenuDisplay {
    bool     open          = false;
    int      page          = 0;     // 0=Main,1=LoadA,2=LoadB,3=ConfirmClearA,4=ConfirmClearB,5=NameA,6=NameB,7=ConfirmDelA,8=ConfirmDelB
    int      cursor        = 0;
    char     name_buf[13]  = {};    // pages 5/6: name being entered
    char     delete_file[64] = {};  // pages 7/8: filename pending deletion
    bool     has_file_a    = false; // cur_bank_a_file is non-empty (overwrite available)
    bool     has_file_b    = false;
    int      record_target   = 0;     // 0=A, 1=B
    bool     trs_instrument  = false; // reflects trs_instrument_gain atomic
    std::string bank_name_a;
    std::string bank_name_b;
    std::vector<std::string> file_list;  // populated for Load sub-pages
};

class GrainVisualizer {
public:
    GrainVisualizer(SliceStore& store_a, SliceStore& store_b,
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
                    std::atomic<float>& view_zoom);
    void start();   // launch render thread
    void stop();    // join render thread

private:
    void render_loop();
    void snapshot();      // try_lock → copy → unlock
    void render_frame();  // project + SDL draw
    static SDL_Color hsv_to_rgb(float h, float s, float v);

    SliceStore&          store_a_;
    SliceStore&          store_b_;
    std::atomic<bool>&   run_;
    std::atomic<int>&    current_id_;
    std::atomic<int>&    current_bank_;
    std::atomic<float>&  crossfade_;
    std::mutex&          bank_menu_mtx_;
    BankMenuDisplay&     bank_menu_disp_;
    std::mutex&          toast_mtx_;
    ParamToast&          toast_disp_;
    std::atomic<int>&    engine_mode_;
    std::atomic<float>&  explore_x_;
    std::atomic<float>&  explore_y_;
    std::atomic<float>&  explore_z_;
    std::atomic<float>&  search_radius_;
    std::atomic<float>&  view_theta_y_;
    std::atomic<float>&  view_theta_x_;
    std::atomic<float>&  view_zoom_;

    std::thread          thread_;
    std::vector<GrainPoint> points_;   // render-thread-only, no mutex

    std::unordered_map<int, std::chrono::steady_clock::time_point> birth_times_a_;
    std::unordered_map<int, std::chrono::steady_clock::time_point> birth_times_b_;

    float theta_y_ = 0.4f;
    float theta_x_ = 0.3f;

    SDL_Window*   window_   = nullptr;
    SDL_Renderer* renderer_ = nullptr;
    TTF_Font*     font_     = nullptr;

    int   w_ = 800, h_ = 540;  // set from actual renderer output size after init
    float scale_ = 250.0f;    // derived from display size after init

    static constexpr float kFocal = 1.5f, kCameraZ = 2.5f;
};
