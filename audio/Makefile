CXX      := g++
SDL2_CFLAGS := $(shell sdl2-config --cflags)
SDL2_LIBS   := $(shell sdl2-config --libs)
CXXFLAGS := -Iinclude $(SDL2_CFLAGS)
LDFLAGS  := -lasound $(SDL2_LIBS)

BUILDDIR := build

# ---- primary target ----
.PHONY: all
all: $(BUILDDIR)/engine

HEADERS := $(wildcard include/*.h)

$(BUILDDIR)/engine: src/alsa_util.cpp src/slicer.cpp src/engine.cpp src/grain_visualizer.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) src/alsa_util.cpp src/slicer.cpp src/engine.cpp src/grain_visualizer.cpp -o $@ $(LDFLAGS)

# ---- unit tests (no ALSA) ----
.PHONY: test
test: $(BUILDDIR)/ringbuffer_test $(BUILDDIR)/ringbuffer_overwrite_test
	./$(BUILDDIR)/ringbuffer_test
	./$(BUILDDIR)/ringbuffer_overwrite_test

$(BUILDDIR)/ringbuffer_test: tests/ringbuffer.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILDDIR)/ringbuffer_overwrite_test: tests/ringbuffer_overwrite.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

# ---- legacy integration tests (require ALSA) ----
.PHONY: legacy
legacy: \
	$(BUILDDIR)/grain_onset_slicing_test \
	$(BUILDDIR)/grain_slicing_test \
	$(BUILDDIR)/ringbuffer_audio_test \
	$(BUILDDIR)/test_passthrough

$(BUILDDIR)/grain_onset_slicing_test: tests/grain_onset_slicing.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILDDIR)/grain_slicing_test: tests/grain_slicing.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILDDIR)/ringbuffer_audio_test: tests/ringbuffer_audio.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILDDIR)/test_passthrough: tests/passthrough.cpp src/test_util.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# ---- clean ----
.PHONY: clean
clean:
	rm -f $(BUILDDIR)/*
