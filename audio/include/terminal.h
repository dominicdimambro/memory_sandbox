#pragma once

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

struct TermRawMode {
    termios orig{};
    bool ok{false};

    bool enable() {
        if (tcgetattr(STDIN_FILENO, &orig) != 0) return false;
        termios raw = orig;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) return false;

        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (flags >= 0) fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

        ok = true;
        return true;
    }

    ~TermRawMode() {
        if (ok) tcsetattr(STDIN_FILENO, TCSANOW, &orig);
    }
};

inline int read_key_nonblocking() {
    unsigned char c;
    ssize_t n = ::read(STDIN_FILENO, &c, 1);
    if (n == 1) return (int)c;
    return -1;
}
