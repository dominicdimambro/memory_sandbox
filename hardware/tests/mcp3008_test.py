import spidev, time

spi = spidev.SpiDev()
spi.open(0, 0)  # bus 0, CE0
spi.max_speed_hz = 1_000_000

def read_channel(ch):
    r = spi.xfer2([1, (8 + ch) << 4, 0])
    return ((r[1] & 3) << 8) | r[2]
while True:
    for ch in range(2):
        raw = read_channel(ch)
        volts = raw * 3.3 / 1023
        print(f"CH{ch}: {raw:4d} | {volts:.3f}V", end="  ")
    print()
    time.sleep(0.1)