import smbus2, time

# MCP23017 default I2C address (A0/A1/A2 all GND)
ADDR = 0x20

IODIRA = 0x00  # direction port A (1=input)
IODIRB = 0x01  # direction port B
GPPUA  = 0x0C  # pull-ups port A
GPPUB  = 0x0D  # pull-ups port B
GPIOA  = 0x12  # read port A
GPIOB  = 0x13  # read port B

bus = smbus2.SMBus(1)  # I2C bus 1 (GPIO 2/3)

# Set all pins as inputs
bus.write_byte_data(ADDR, IODIRA, 0xFF)
bus.write_byte_data(ADDR, IODIRB, 0xFF)

# Enable pull-ups on GPA6, GPA7 (encoder) and GPB0 (button)
bus.write_byte_data(ADDR, GPPUA, 0b11000000)  # GPA6 and GPA7
bus.write_byte_data(ADDR, GPPUB, 0b00000001)  # GPB0

port_a = bus.read_byte_data(ADDR, GPIOA)
prev_a = (port_a >> 7) & 1
prev_b = (port_a >> 6) & 1
prev_btn = None
counter = 0

while True:
    port_a = bus.read_byte_data(ADDR, GPIOA)
    port_b = bus.read_byte_data(ADDR, GPIOB)

    enc_a = (port_a >> 7) & 1  # GPA7
    enc_b = (port_a >> 6) & 1  # GPA6
    btn   = (port_b >> 0) & 1  # GPB0 (low when pressed with pull-up)

    # Decode direction on both edges of A
    if enc_a != prev_a:
        if enc_a == 1:
            counter += 1 if enc_b == 0 else -1
        else:
            counter += 1 if enc_b == 1 else -1
        print(f"counter: {counter}")

    if btn != prev_btn:
        print(f"BTN: {'pressed' if btn == 0 else 'released'}")
        prev_btn = btn

    prev_a, prev_b = enc_a, enc_b
    time.sleep(0.005)
