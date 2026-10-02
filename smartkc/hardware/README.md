# Phone Attachment

## Requirement:
1. **Placido head**: 3D-print `placido_head.stl` (to modify the design, use `placido_head.step`) by following the 3D-print related instructions [below](#placido-head)
1. **Placido base**: 3D-print `placido_base.stl` (to modify the design, use `placido_base.step`) by following the 3D-print related instructions [below](#placido-base)
1. **Diffuser**: 3D-print `diffuser.stl` (to modify the design, use `diffuser.step`) by following the 3D-print related instructions [below](#diffuser)
1. **Circular LED**: Use WS2812/WS2812B 12-bit RGB LED Ring (available for Rs 250 or 3$ on DigiKey, [Robocraze](https://robocraze.com/products/ws2812-12-bit-rgb-led-round), [ThinkRobotics](https://thinkrobotics.in/products/ws2812-5050-rgb-led-ring), etc.)
1. **Development board**: Use ATtiny85 (Digispark) USB Development Board (available for Rs 325 or 4$ on DigiKey, [Robocraze](https://robocraze.com/products/attiny85-usb-development-board), [Robu](https://robu.in/product/attiny85-usb-development-board/), etc.)
1. **Source code for Development board**: Use `attiny85_led_setup_code/attiny85_led_setup_code.ino` and upload it to the ATtiny85 development board by following the instructions [below](#attiny85-development-board-related-instructions)
1. **OTG cable**: To connect ATtiny85 Development Board with the smartphone charging input, use a USB-A Female to USB-B/C Male cable (available for Rs 200 or 2.5$ on [Amazon](https://www.amazon.in/gp/product/B012V56C8K))
1. **Double-sided tape**: Use 3M Scotch Double Sided Foam Tape (available for Rs 100 or 1.2$ on [Amazon](https://www.amazon.in/gp/product/B00N1U9AJS/))
1. **Smartphone Case**: Specific to your smartphone
1. **Smartphone**: Please check the smartphone specification related requirement [below](#smartphone-specifications)

Note: Kindly follow the guidelines mentioned in the `guidelines_for_data_collector.pdf` document to collect high-quality data from the SmartKC device.

## Ring Distribution
File `./ring_distribution.txt` contains the radius of each ring and its height from the base.

## 3D-print Instructions:

### Placido head:
MJF (MultiJet Fusion) print with PA-12 material in Stone Grey color (Matt finish). We used the default settings.

### Placido base:
MJF (MultiJet Fusion) print with PA-12 material in Stone Grey color (Matt finish). We used the default settings.

### Diffuser:
Print using PLA material (white filament): [1.75mm PLA_3D Filament White 1KG](https://robu.in/product/esun-pla-1-75mm-3d-printing-filament-1kg-white/). White-colored filament is required to achieve translucency. Note: Convert the `.stl` file to `.gx` file before printing.

We used the FlashForge Finder 3D printer with [FlashPrint](https://flashforge-usa.com/pages/download) software (version 4.20, 64 Bits), with the following settings: 
* First Layer Height: 0.25 mm, 
* Layer Height: 0.20 mm, 
* Infill: 30%, 
* Fill pattern: Hexagonal, 
* Temperature: 225C,
* Resolution: High.

## ATtiny85 Development Board-related Instructions:

The firmware controls a 12-pixel WS2812/WS2812B ring from Digispark pin P1. It performs a short startup chase through all 12 pixels and then leaves the ring at steady, current-limited white illumination.

### Wiring

Disconnect USB power while wiring.

1. Connect Digispark `5V` to the LED ring `5V`.
2. Connect Digispark `GND` to the LED ring `GND`.
3. Connect Digispark `P1` through a 330–500 ohm resistor to the LED ring `DIN`. Do not connect to `DOUT`.
4. Add a 500–1000 µF capacitor rated for at least 6.3 V across the ring's `5V` and `GND`, observing capacitor polarity.
5. Keep the data wire short and confirm that the board and ring share ground.

### Arduino IDE setup on Windows

1. Install the current [Arduino IDE](https://www.arduino.cc/en/software).
2. Open `File` → `Preferences` and add this Boards Manager URL:

   ```text
   https://raw.githubusercontent.com/ArminJo/DigistumpArduino/master/package_digistump_index.json
   ```

3. Open `Tools` → `Board` → `Boards Manager`, search for `Digistump AVR Boards`, and install it.
4. Open `Tools` → `Manage Libraries`, search for `Adafruit NeoPixel`, and install it.
5. Select `Tools` → `Board` → `Digistump AVR Boards` → `Digispark (Default - 16.5 MHz)`.
6. Open `attiny85_led_setup_code/attiny85_led_setup_code.ino`.

### Upload sequence

1. Unplug the Digispark from USB.
2. Click `Verify` and wait for compilation to finish.
3. Click `Upload` while the board remains unplugged.
4. Wait until the output asks you to plug in the device.
5. Plug the Digispark directly into USB. The Micronucleus uploader should detect it and finish automatically. A COM port is not selected for this board.
6. After upload, disconnect USB, connect the LED ring, and reconnect power.

### Firmware verification

The firmware was compiled and uploaded to a physical Digispark ATtiny85 on 2 October 2026.

- Board target: `digistump:avr:digispark-tiny`
- Digistump AVR Boards: `1.7.5`
- Adafruit NeoPixel: `1.15.5`
- Program storage: 2,210 of 6,650 bytes (33%)
- Dynamic memory: 40 of 512 bytes (7%)
- Micronucleus detected firmware version 1.6, erased the board, wrote the application, and started it successfully.
- LED-ring behavior remains to be tested when the WS2812/WS2812B ring is available.

### Hardware test

Keep the light away from a person's eye during this first test.

1. On power-up, count 12 LEDs illuminating in order in about one second.
2. Confirm that all 12 LEDs then remain steadily illuminated.
3. Leave it powered for two minutes and verify that there is no flicker, repeated restarting, USB disconnection, unusual smell, or excessive heating.
4. Disconnect it from the computer and repeat the test from the smartphone through the OTG adapter.

If the chase stops partway around the ring, inspect that pixel and the connection immediately before it. If no LEDs illuminate, check `DIN` versus `DOUT`, common ground, 5 V power, and the P1 data connection. Repeated startup chases indicate a power reset; inspect the power source, capacitor, wiring, and solder joints.

## Smartphone Specifications:
* Android smartphone with 12 MP (or more) back camera with a minimum focusing distance of 75mm (or less).
* Devices we have tested SmartKC to work well on: OnePlus 7T, OnePlus 6T, Samsung Galaxy A52s 5G, Xiaomi Pocophone F1.
* Devices we have tested SmartKC to not work on: OnePlus 9 5G, Redmi 9 Power, Pixel 4a. Reason: minimum focusing distance too high.

## Assembly Instructions (putting it all together):
1. Upload `attiny85_led_setup_code/attiny85_led_setup_code.ino` to the `Development board`.
1. Connect the `Circular LED` to the `Development board` by soldering ATtiny85 GND to WS2812 GND, ATtiny85 5V to WS2812 5V, and ATtiny85 P1 to WS2812 D1.
1. Use `Double-sided tape` to attach the `Circular LED` on the `Placido base` (aligning it with the center).
1. Use `Double-sided tape` to attach the `Development board` on the `Placido base` (inside the igloo shaped enclosure)
1. Use `Double-sided tape` to attach the `Placido base` over the `Smartphone Case`.
   * Important: Make sure to align the hole on the `Placido base` with the smartphone's main back camera.
   * Also, orient the `Placido base` such that the `Development board` enclosure aligns vertically with the phone.
1. Connect the `Development board` to the smartphone using the `OTG cable`.
1. Place the `Diffuser` inside the `Placido head`.
1. Attach the `Placido head` (with `Diffuser` inside it) to the `Placido base` by placing the three pegs and rotating it.
