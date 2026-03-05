# memory_sandbox
A C++ real-time audio engine for an embedded granular synthesis instrument
running on Linux with ALSA.

## Key features
- callback-driven real-time architecture
- lock-free ring buffers
- multithreaded analysis and synthesis pipeline
- feature-space grain retrieval with KD-tree search

## System constraints
- 48 kHz sample rate
- 256 frame periods
- ~5.3 ms callback deadline
