// SmartKC Digispark ATtiny85 firmware for a 12-pixel WS2812/WS2812B ring.
// Wiring: P1 -> 330-500 ohm resistor -> ring DIN, 5V -> 5V, GND -> GND.
#include <Adafruit_NeoPixel.h>

constexpr uint8_t LED_PIN = 1;
constexpr uint8_t LED_COUNT = 12;

// 48/255 limits worst-case LED current to about 136 mA for 12 white pixels.
// This leaves useful margin when the Digispark is powered through USB or phone OTG.
constexpr uint8_t LED_BRIGHTNESS = 48;
constexpr uint16_t PIXEL_TEST_DELAY_MS = 60;

Adafruit_NeoPixel ring(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

void setAllPixels(uint32_t colour) {
  for (uint8_t pixel = 0; pixel < LED_COUNT; ++pixel) {
    ring.setPixelColor(pixel, colour);
  }
  ring.show();
}

void setup() {
  ring.begin();
  ring.setBrightness(LED_BRIGHTNESS);
  ring.clear();
  ring.show();
  delay(200);

  // Startup self-test: the 12 pixels illuminate in order so a failed pixel,
  // reversed DIN/DOUT connection, or incorrect LED count is easy to identify.
  const uint32_t white = ring.Color(255, 255, 255);
  for (uint8_t pixel = 0; pixel < LED_COUNT; ++pixel) {
    ring.setPixelColor(pixel, white);
    ring.show();
    delay(PIXEL_TEST_DELAY_MS);
  }

  // Stable, even illumination for Placido-ring image capture.
  setAllPixels(white);
}

void loop() {
  // The illumination remains steady until power is removed.
}
