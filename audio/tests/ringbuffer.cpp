#include <cassert>
#include <cstdio>
#include <vector>
#include <numeric>
#include "ringbuffer.h"

// {function} test_basic_fifo: test if ring buffer reads and writes in correct order
static void test_basic_fifo() {
    RingBuffer<float> rb(8);

    float in[4]  = {1,2,3,4};
    float out[4] = {0,0,0,0};

    // check if we successfully write 4 elements; check if size is incremented correctly
    assert(rb.write(in, 4) == 4);
    assert(rb.size() == 4);

    // check if we successfully read 4 elements; check if size is decremented correctly
    assert(rb.read(out, 4) == 4);
    assert(rb.size() == 0);

    // check if elements of in and out match in the correct order
    for (int i = 0; i < 4; ++i) assert(out[i] == in[i]);
}

// {function} test_wraparound: test if we can wraparound in read and write functions
static void test_wraparound() {
    RingBuffer<float> rb(8);

    // write 6 elements
    float in1[6] = {0,1,2,3,4,5};
    assert(rb.write(in1, 6) == 6);

    // read 4 elements
    float out1[4];
    assert(rb.read(out1, 4) == 4);
    // out1 should be 0,1,2,3
    for (int i = 0; i < 4; ++i) assert(out1[i] == (float)i);

    // Now read head is at 4,5 (last 2 items). Write 6 more -> should wrap.
    float in2[6] = {6,7,8,9,10,11};
    assert(rb.write(in2, 6) == 6);
    // check the buffer reflects that it's full
    assert(rb.size() == 8);

    float out2[8];
    assert(rb.read(out2, 8) == 8);

    // Expected sequence: 4,5,6,7,8,9,10,11
    float expected[8] = {4,5,6,7,8,9,10,11};
    for (int i = 0; i < 8; ++i) assert(out2[i] == expected[i]);
}

// {function} tests_overflow_behavior_drop_new: check if write only writes elements when there is room
static void test_overflow_behavior_drop_new() {
    RingBuffer<float> rb(5);

    float in[10];
    for (int i = 0; i < 10; ++i) in[i] = (float)i;

    // check we only write till capacity is full; check buffer is reflected as full
    assert(rb.write(in, 10) == 5);
    assert(rb.full());

    // check we only read 5 values (0..4)
    float out[5];
    assert(rb.read(out, 5) == 5);
    for (int i = 0; i < 5; ++i) assert(out[i] == (float)i);
}

// {function} test_partial_read_write: check accuracy of partial read/write of buffer
static void test_partial_read_write() {
    RingBuffer<float> rb(8);

    // write 8 elements
    float in[8] = {10,11,12,13,14,15,16,17};
    assert(rb.write(in, 8) == 8);

    // read 3 elements
    float outA[3];
    assert(rb.read(outA, 3) == 3);
    assert(outA[0] == 10 && outA[1] == 11 && outA[2] == 12);

    // only write 3 of 5 elements; check buffer is full
    float inB[5] = {100,101,102,103,104};
    assert(rb.write(inB, 5) == 3);
    assert(rb.full());

    // read 8 elements
    float outB[8];
    assert(rb.read(outB, 8) == 8);

    // Remaining from first block: 13..17 (5 items), then 100..102 (3 items)
    float expected[8] = {13,14,15,16,17,100,101,102};
    for (int i = 0; i < 8; ++i) assert(outB[i] == expected[i]);
}

// {function} test_drop: check we can drop elements without reading them
static void test_drop() {
    RingBuffer<float> rb(8);

    // write 6 elements
    float in[6] = {1,2,3,4,5,6};
    assert(rb.write(in, 6) == 6);

    // drop 2 elements; check size is updated
    assert(rb.drop(2) == 2);
    assert(rb.size() == 4);

    // read 4 elements; check we begin reading after dropped elements
    float out[4];
    assert(rb.read(out, 4) == 4);
    float expected[4] = {3,4,5,6};
    for (int i = 0; i < 4; ++i) assert(out[i] == expected[i]);
}

int main() {
    test_basic_fifo();
    test_wraparound();
    test_overflow_behavior_drop_new();
    test_partial_read_write();
    test_drop();

    std::puts("RingBuffer tests: PASS");
    return 0;
}
