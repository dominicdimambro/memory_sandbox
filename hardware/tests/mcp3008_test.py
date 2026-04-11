from gpiozero import OutputDevice, InputDevice
import time

SCLK = OutputDevice(11)
MOSI = OutputDevice(10)
MISO = InputDevice(9, pull_up=False)
CS = OutputDevice(25, active_high=False, initial_value=False)

def read_channel(ch):
    CS.on()
    time.sleep(0.000002)

    for bit in [1, 1, (ch >> 2) & 1, (ch >> 1) & 1, ch & 1]:
        SCLK.off()
        MOSI.value = bit
        time.sleep(0.000002)
        SCLK.on()
        time.sleep(0.000002)

    result = 0
    for i in range(12):
        SCLK.off()
        time.sleep(0.000002)
        SCLK.on()
        time.sleep(0.000002)
        if i >= 2:
            result = (result << 1) | MISO.value

    SCLK.off()
    CS.off()
    return result

while True:
    for ch in range(2):
        raw = read_channel(ch)
        volts = raw * 3.3 / 1023
        print(f"CH{ch}: {raw:4d} | {volts:.3f}V", end="  ")
    print()
    time.sleep(0.1)
