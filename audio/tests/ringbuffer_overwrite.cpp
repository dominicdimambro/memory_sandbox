#include <cassert>
#include <cstdio>
#include "ringbuffer.h"

// {function}: check we can overwrite ring buffer when it is full
static void test_overwrite_when_full() {
    RingBuffer<int> rb(5);

    // write 5 elements (should fit); check buffer is full
    int a[5] = {0,1,2,3,4};
    assert(rb.write_overwrite(a, 5) == 5);
    assert(rb.full());

    // write 2 elements (overwrite oldest elements); check the buffer is still full
    int b[2] = {5,6};
    assert(rb.write_overwrite(b, 2) == 2);
    assert(rb.full());

    // read out 5 elements
    int out[5];
    assert(rb.read(out, 5) == 5);

    // oldest 2 (0,1) should have been dropped; buffer should hold 2,3,4,5,6
    int expected[5] = {2,3,4,5,6};
    for (int i = 0; i < 5; ++i) assert(out[i] == expected[i]);
}

// {function}: check that large write requests fill the buffer with only the tail
static void test_overwrite_large_write_keeps_tail() {
    RingBuffer<int> rb(5);

    // write 10 elements into 5 element buffer; check buffer is full
    int x[10] = {0,1,2,3,4,5,6,7,8,9};
    assert(rb.write_overwrite(x, 10) == 5);
    assert(rb.full());

    // read 5 elements
    int out[5];
    assert(rb.read(out, 5) == 5);

    // if you write 10 into cap=5, you should keep the last 5: 5,6,7,8,9
    int expected[5] = {5,6,7,8,9};
    for (int i = 0; i < 5; ++i) assert(out[i] == expected[i]);
}

// {function}: check overwrite behavior when wrapping around end of buffer
static void test_overwrite_wraparound_interaction() {
    RingBuffer<int> rb(8);

    // write 6 elements
    int a[6] = {0,1,2,3,4,5};
    assert(rb.write_overwrite(a, 6) == 6);

    // read 4 elements
    int tmp[4];
    assert(rb.read(tmp, 4) == 4);

    // write 6 elements; check buffer is full
    int b[6] = {6,7,8,9,10,11};
    assert(rb.write_overwrite(b, 6) == 6);
    assert(rb.full());

    // overwrite oldest 3 elements; check buffer is full
    int c[3] = {12,13,14};
    assert(rb.write_overwrite(c, 3) == 3);
    assert(rb.full());

    // read 8 elements
    int out[8];
    assert(rb.read(out, 8) == 8);

    // remaining from previous full buffer: 7,8,9,10,11 then 12,13,14
    int expected[8] = {7,8,9,10,11,12,13,14};
    for (int i = 0; i < 8; ++i) assert(out[i] == expected[i]);
}

int main() {
    test_overwrite_when_full();
    test_overwrite_large_write_keeps_tail();
    test_overwrite_wraparound_interaction();

    std::puts("RingBuffer overwrite tests: PASS");
    return 0;
}
