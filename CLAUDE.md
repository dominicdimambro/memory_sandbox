# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Memory Sandbox is a hardware granular synthesis instrument running on a Raspberry Pi (ARM64/aarch64). It captures live audio via ALSA, stores it in ring buffers, detects amplitude onsets to slice audio into grains, and plays back grains at variable speeds/intervals. The project is in early prototyping phase.

## Build Commands

All compilation is done manually with g++ from the `audio/` directory. There is no Makefile or CMake.

```bash
cd audio

# Compile the main engine
g++ -Iinclude src/alsa_util.cpp src/slicer.cpp src/engine.cpp -o build/engine -lasound
```

Legacy test builds (self-contained, don't use the modular src/ files):
```bash
g++ -Iinclude tests/grain_onset_slicing.cpp -o build/grain_onset_slicing_test -lasound
g++ -Iinclude tests/grain_slicing.cpp -o build/grain_slicing_test -lasound
g++ -Iinclude tests/ringbuffer_audio.cpp -o build/ringbuffer_audio_test -lasound
g++ -Iinclude tests/passthrough.cpp src/test_util.cpp -lasound -o build/test_passthrough
```

Unit tests (no ALSA dependency):
```bash
g++ -Iinclude tests/ringbuffer.cpp -o build/ringbuffer_test && ./build/ringbuffer_test
g++ -Iinclude tests/ringbuffer_overwrite.cpp -o build/ringbuffer_overwrite_test && ./build/ringbuffer_overwrite_test
```

## Architecture

### Threading Model (4 threads)

- **Capture thread**: Reads interleaved stereo S16LE from ALSA (`snd_pcm_readi`) → writes to shared ring buffer using `write_overwrite()` (drops oldest on overflow)
- **Slicer thread**: Periodically scans ring buffer epochs, runs onset detection (`detect_onsets_stereo_s16`), creates `Slice` entries in the `SliceStore`
- **Playback thread**: Reads from ring buffer (monitor mode) or from slices (grain mode) → writes to ALSA (`snd_pcm_writei`)
- **Main thread**: Non-blocking keyboard input via `TermRawMode` (termios), dispatches commands by setting atomic flags

Thread communication uses `std::atomic` for flags/modes and `std::mutex` for shared data structures (ring buffer, slice store).

### Key Components

- **`RingBuffer<T>`** (`audio/include/ringbuffer.h`): Templated circular buffer. Supports destructive read, drop-old overwrite, and non-destructive `copy_latest()`/`copy_range()` for zero-copy monitoring with delay offset. All audio data is interleaved stereo (multiply frame counts by 2 for sample counts).

- **`SliceStore`** (`audio/include/slice_store.h`): Thread-safe grain storage. Contains a `corpus` vector (all recorded audio) and a map of `Slice` metadata (id, start index, length). Provides `add_slice()`, `get()`, `random_id()`.

- **`Slicer`** (`audio/include/slicer.h`): Abstract base class for swappable slicing algorithms. `process()` takes a window of interleaved audio and returns `SliceRegion` structs (start_frame, length_frames). `OnsetSlicer` (`audio/src/slicer.cpp`) implements amplitude onset detection with exponential envelope follower, ~10ms time constant, configurable sensitivity, and 120ms refractory period.

- **ALSA utilities** (`audio/include/alsa_util.h`, `audio/src/alsa_util.cpp`): `set_hw_params()`, `open_pcm()`, `recover_if_xrun()` for configuring and managing ALSA PCM devices.

- **Terminal input** (`audio/include/terminal.h`): `TermRawMode` for non-blocking keyboard input via termios.

- **Engine** (`audio/src/engine.cpp`): Main program orchestrating all threads with a `std::unique_ptr<Slicer>` for the active slicing algorithm.

### Audio Parameters

- Sample rate: 48000 Hz
- Format: S16LE interleaved stereo (2 channels)
- Period: 256 frames (~5.3ms latency)
- Buffer: 4096 frames
- Default ALSA device: `plughw:2,0` (HiFiBerry or similar)

### Conventions

- Struct names: PascalCase (`SliceStore`, `OnsetEvent`, `TermRawMode`)
- Private members: snake_case with trailing underscore (`buf_`, `cap_`, `r_`, `w_`)
- Global atomics: snake_case (`g_run`, `record_enabled`, `play_enabled`)
- Tests double as integration programs — the `tests/` directory contains both unit tests and interactive audio programs (self-contained, predate the modular src/ layout)
- New slicing algorithms subclass `Slicer` and override `process()` — see `OnsetSlicer` for the pattern
