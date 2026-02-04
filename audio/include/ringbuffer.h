#pragma once
#include <vector>
#include <cstddef>
#include <algorithm>

// simple ring buffer class
template <typename T>
class RingBuffer {
public:
    // initialize buf_ (vector) and cap_ (size_t)
    explicit RingBuffer(size_t capacity)
        : buf_(capacity), cap_(capacity) {}

    size_t capacity() const { return cap_; }
    size_t size() const { return size_; }              // number of valid items stored
    size_t free_space() const { return cap_ - size_; } // remaining space
    bool empty() const { return size_ == 0; }   
    bool full() const { return size_ == cap_; }

    // Debug helpers
    size_t write_pos() const { return w_; }
    size_t read_pos() const { return r_; } 

    // {function} write: write up to n items into the ring buffer
    // {param} *in: pointer to the start of input data to write
    // {param} n: requested number of elements to write
    // {return} to_write: number of elements successfully written
    size_t write(const T* in, size_t n) {
        const size_t to_write = std::min(n, free_space());
        if (to_write == 0) return 0;

        // first chunk: from w_ to end
        const size_t first_chunk_size = std::min(to_write, cap_ - w_);
        std::copy(in, in + first_chunk_size, buf_.begin() + w_);

        // second chunk: from 0
        const size_t second_chunk_size = to_write - first_chunk_size;
        if (second_chunk_size > 0) {
            std::copy(in + first_chunk_size, in + first_chunk_size + second_chunk_size, buf_.begin());
        }

        // update the write head position
        w_ = (w_ + to_write) % cap_;
        size_ += to_write;
        return to_write;
    }

    // {function} write_overwrite: write n items into ring buffer with "drop-old" behavior
    // {param} const T* in: pointer to the start of the input data to write
    // {param} size_t n: requested number of elements to write
    // {return} const size_t written: number of elements successfully written
    size_t write_overwrite(const T* in, size_t n) {
        if (cap_ == 0 || n == 0) return 0;

        // if n >= capacity, keep only last cap_ elements
        if (n >= cap_) {
            // drop everything in the buffer
            r_ = 0;
            w_ = 0;
            size_ = 0;

            // write in the last cap_ elements from input data
            const T* tail = in + (n - cap_);
            // use normal write behavior since this will fit
            return write(tail, cap_);
        }

        // drop old elements if we need to make room
        const size_t needed = n;
        if (needed > free_space()) {
            const size_t overflow = needed - free_space();
            // advance read head; reduces size
            drop(overflow);
        }

        // now that it fits, write n elements
        const size_t written = write(in, n);
        return written;
    }
    
    // {function} read: read up to n items from ring buffer
    // {param} *out: pointer to the start of data to read
    // {param} n: requested number of elements to read
    // {return} to_read: number of elements successfully read
    size_t read(T* out, size_t n) {
        const size_t to_read = std::min(n, size_);
        if (to_read == 0) return 0;

        // first chunk: from r_ to end
        const size_t first_chunk_size = std::min(to_read, cap_ - r_);
        std::copy(buf_.begin() + r_, buf_.begin() + r_ + first_chunk_size, out);

        // second chunk: from 0
        const size_t second_chunk_size = to_read - first_chunk_size;
        if (second_chunk_size > 0) {
            std::copy(buf_.begin(), buf_.begin() + second_chunk_size, out + first_chunk_size);
        }

        // update read head position
        r_ = (r_ + to_read) % cap_;
        size_ -= to_read;
        return to_read;
    }

    // {function} drop: drop elements without reading them
    // {param} n: number of elements to drop from the read head without reading them
    // {return} to_drop: number of elements successfully dropped
    size_t drop(size_t n) {
        const size_t to_drop = std::min(n, size_);
        r_ = (r_ + to_drop) % cap_;
        size_ -= to_drop;
        return to_drop;
    }

    // {function} clear: set size to 0, reset read/write heads to beginning
    void clear() {
        r_ = w_ = size_ = 0;
    }

    // ----------------------------
    // Non-destructive read methods
    // ----------------------------

    // {function}: copy 'n' items ending at (write_pos - delay). Does NOT modify r_/w_/size_.
    // if buffer has fewer than n valid items, copies what it has and returns that count.
    // {param} T* out: pointer to start of data to read
    // {param} size_t n: number of items to read
    // {param} size_t delay: how far back from write head to copy items from (0 = newest written sample)
    // {return}: number of items copied
    size_t copy_latest(T* out, size_t n, size_t delay = 0) const {
        if (cap_ == 0 || n == 0 || size_ == 0) return 0;

        // limit what we can read: valid sample history ends size_ backwards from write head w_
        const size_t can_read = std::min(n, size_);

        // oldest valid sample is size_ before w_
        const size_t max_delay = size_ > 0 ? (size_ - 1) : 0;
        if (delay > max_delay) delay = max_delay;

        // our copy region excludes emd: [start:end)
        size_t end = (w_ + cap_ - (delay % cap_)) % cap_;

        // start = end - can_read
        size_t start = (end + cap_ - (can_read % cap_)) % cap_;

        return copy_range(out, start, can_read);
    }

    // {function}: copy 'n' items starting at absolute ring index 'start' (0..cap-1)
    // {param} T* out: pointer to start of data to read
    // {param} size_t n: number of items to read
    // return: number of items copied
    size_t copy_range(T* out, size_t start, size_t n) const {
        if (cap_ == 0 || n == 0 || size_ == 0) return 0;

        const size_t to_copy = std::min(n, size_);
        const size_t s = start % cap_;

        // copy first chunk from s to end of buffer
        const size_t first_chunk_size = std::min(to_copy, cap_ - s);
        std::copy(buf_.begin() + s, buf_.begin() + s + first_chunk_size, out);

        // if there is more to copy, finish in this chunk
        const size_t second_chunk_size = to_copy - first;
        if (second_chunk_size > 0) {
            std::copy(buf_.begin(), buf_.begin() + second_chunk_size, out + first);
        }
        return to_copy;
    }

private:
    std::vector<T> buf_;
    size_t cap_{0};     // total capacity of buffer
    size_t r_{0};       // read head position
    size_t w_{0};       // write head position
    size_t size_{0};    // current size of buffer
};
