#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// region returned by a slicer algorithm
struct SliceRegion {
    size_t start_frame;
    size_t length_frames;
};

// base class for swappable slicing algorithms
class Slicer {
public:
    virtual ~Slicer() = default;

    // given a window of interleaved audio, return regions to slice
    virtual std::vector<SliceRegion> process(
        const int16_t* data, size_t n_samples,
        unsigned int channels, unsigned int rate) = 0;
};

// amplitude onset-based slicer
class OnsetSlicer : public Slicer {
public:
    float sensitivity = 1.0f;
    int pre_onset_ms = 10;
    int slice_ms = 1000;
    int min_slice_ms = 30;

    std::vector<SliceRegion> process(
        const int16_t* data, size_t n_samples,
        unsigned int channels, unsigned int rate) override;
};
