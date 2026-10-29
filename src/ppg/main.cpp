/*
 * src/ppg/main.cpp  -  MAX30102 PPG logger (ESP32, Arduino framework)
 * ---------------------------------------------------------------
 * MAX30102 breakout  ->  ESP32
 *   VIN (3V3)  -> 3V3
 *   GND        -> GND
 *   SDA        -> GPIO21
 *   SCL        -> GPIO22
 *   INT        -> not connected (polling is used)
 *
 * Shares the I2C bus with the MPU6050 later (addresses 0x57 and 0x68 differ).
 *
 * Output (one line per sample, 100 Hz), raw 18-bit ADC counts:
 *   timestamp_us,red,ir
 *   e.g.  1234567,81234,102345
 *
 * Lines starting with '#' are status messages. Every 2 s the firmware prints
 *   # PPG_META fs_hz=100 adc_bits=18 led_red_ma=7.2 led_ir_ma=7.2 adc_range_na=4096 ...
 * so the viewer always knows the real settings.
 *
 * How it works
 *  - The MAX30102 samples by itself (400 sps, 4x averaging => 100 Hz) and keeps
 *    samples in a 32-entry FIFO (6 bytes per sample: 3 bytes RED, 3 bytes IR).
 *  - An esp_timer polls the FIFO pointers every POLL_PERIOD_US, reads whatever is
 *    new in one burst, and pushes it to a queue. Serial.print only happens in loop().
 *  - Sample timestamps are RECONSTRUCTED: the newest sample gets the time of the
 *    pointer read, older ones are spaced by exactly one sample period. The error
 *    is at most about POLL_PERIOD_US (3 ms). For precise ECG->PPG timing (PAT) a
 *    later stage can use the sensor's data-ready interrupt (INT pin) instead.
 *  - Same esp_timer clock as the ECG and IMU firmware, so streams can be aligned.
 *
 * Board: "ESP32 Dev Module".  PlatformIO env: ppg.
 * Close the Serial Monitor before running the Python viewer.
 */

#include <Arduino.h>
#include <Wire.h>
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

// ---------------- configuration ----------------
static const int      PIN_SDA          = 21;
static const int      PIN_SCL          = 22;
static const uint8_t  MAX_ADDR         = 0x57;
static const uint32_t I2C_HZ           = 400000;
static const uint32_t SERIAL_BAUD      = 230400;

// Output rate = SAMPLE_RATE_SPS / AVERAGING  (here 400 / 4 = 100 Hz)
static const uint16_t SAMPLE_RATE_SPS  = 400;   // 50,100,200,400,800,1000,1600,3200
static const uint8_t  AVERAGING        = 4;     // 1,2,4,8,16,32
// 18-bit resolution (411 us pulse) allows at most 400 sps. Shorter pulses allow higher rates.
static const uint8_t  PULSE_WIDTH_CODE = 3;     // 0:69us/15bit 1:118us/16bit 2:215us/17bit 3:411us/18bit
static const uint8_t  ADC_RANGE_CODE   = 1;     // 0:2048 nA  1:4096 nA  2:8192 nA  3:16384 nA
static const uint8_t  LED_RED_PA       = 0x24;  // LED current = value * 0.2 mA  (0x24 = 7.2 mA)
static const uint8_t  LED_IR_PA        = 0x24;

static const uint32_t POLL_PERIOD_US   = 3000;  // how often the FIFO is checked
static const int      QUEUE_LEN        = 128;   // ~1.3 s of buffering at 100 Hz
// -------------------------------------------------

static_assert(SAMPLE_RATE_SPS % AVERAGING == 0, "AVERAGING must divide SAMPLE_RATE_SPS");

static const uint32_t OUTPUT_RATE_HZ   = SAMPLE_RATE_SPS / AVERAGING;
static const int64_t  SAMPLE_PERIOD_US = 1000000LL / OUTPUT_RATE_HZ;

// MAX30102 registers
static const uint8_t REG_INT_STATUS1 = 0x00;
static const uint8_t REG_INT_STATUS2 = 0x01;
static const uint8_t REG_FIFO_WR_PTR = 0x04;
static const uint8_t REG_FIFO_OVF    = 0x05;
static const uint8_t REG_FIFO_RD_PTR = 0x06;
static const uint8_t REG_FIFO_DATA   = 0x07;
static const uint8_t REG_FIFO_CFG    = 0x08;
static const uint8_t REG_MODE_CFG    = 0x09;
static const uint8_t REG_SPO2_CFG    = 0x0A;
static const uint8_t REG_LED1_PA     = 0x0C;   // RED
static const uint8_t REG_LED2_PA     = 0x0D;   // IR
static const uint8_t REG_PART_ID     = 0xFF;   // 0x15 for MAX30102

struct Sample {
  int64_t  t_us;
  uint32_t red;
  uint32_t ir;
};

static QueueHandle_t      sampleQueue  = nullptr;
static esp_timer_handle_t pollTimer    = nullptr;
static volatile uint32_t  droppedCount = 0;   // queue-full events
static volatile uint32_t  i2cErrors    = 0;   // failed transfers
static volatile uint32_t  fifoOverflow = 0;   // last read of the FIFO overflow counter
static bool               maxReady     = false;
static uint8_t            partId       = 0;

// ---------------- I2C helpers ----------------
static bool writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MAX_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool readRegs(uint8_t reg, uint8_t *buf, size_t len) {
  Wire.beginTransmission(MAX_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;    // repeated start
  size_t got = Wire.requestFrom((int)MAX_ADDR, (int)len);
  if (got != len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)Wire.read();
  return true;
}

static bool sampleRateCode(uint16_t sps, uint8_t &code) {
  switch (sps) {
    case 50:   code = 0; return true;
    case 100:  code = 1; return true;
    case 200:  code = 2; return true;
    case 400:  code = 3; return true;
    case 800:  code = 4; return true;
    case 1000: code = 5; return true;
    case 1600: code = 6; return true;
    case 3200: code = 7; return true;
    default:   return false;
  }
}

static bool averagingCode(uint8_t avg, uint8_t &code) {
  switch (avg) {
    case 1:  code = 0; return true;
    case 2:  code = 1; return true;
    case 4:  code = 2; return true;
    case 8:  code = 3; return true;
    case 16: code = 4; return true;
    case 32: code = 5; return true;
    default: return false;
  }
}

// Reset, configure for SpO2 mode (RED + IR), clear the FIFO.
static bool initMax() {
  uint8_t srCode = 0, aveCode = 0;
  if (!sampleRateCode(SAMPLE_RATE_SPS, srCode) || !averagingCode(AVERAGING, aveCode)) return false;
  if (!readRegs(REG_PART_ID, &partId, 1)) return false;    // nothing answering at MAX_ADDR

  bool ok = true;
  ok &= writeReg(REG_MODE_CFG, 0x40);                      // soft reset (bit clears itself)
  for (int i = 0; i < 50; i++) {
    delay(2);
    uint8_t m = 0xFF;
    if (readRegs(REG_MODE_CFG, &m, 1) && (m & 0x40) == 0) break;
  }
  ok &= writeReg(REG_FIFO_WR_PTR, 0);
  ok &= writeReg(REG_FIFO_OVF, 0);
  ok &= writeReg(REG_FIFO_RD_PTR, 0);
  ok &= writeReg(REG_FIFO_CFG, (uint8_t)((aveCode << 5) | (1 << 4) | 0x0F));  // averaging, rollover on
  ok &= writeReg(REG_SPO2_CFG, (uint8_t)((ADC_RANGE_CODE << 5) | (srCode << 2) | PULSE_WIDTH_CODE));
  ok &= writeReg(REG_LED1_PA, LED_RED_PA);
  ok &= writeReg(REG_LED2_PA, LED_IR_PA);
  ok &= writeReg(REG_MODE_CFG, 0x03);                      // SpO2 mode: RED then IR per sample
  uint8_t dummy[2];
  readRegs(REG_INT_STATUS1, dummy, 1);                     // clear interrupt flags
  readRegs(REG_INT_STATUS2, dummy + 1, 1);
  return ok;
}

// ---------------- sampling ----------------
// Runs in the esp_timer task every POLL_PERIOD_US. It is the ONLY code that talks to
// the sensor after setup (important when the IMU shares the bus: keep both in this task).
static void onPollTimer(void *arg) {
  static uint32_t pollCount = 0;
  int64_t t = esp_timer_get_time();

  uint8_t wr = 0, rd = 0;
  if (!readRegs(REG_FIFO_WR_PTR, &wr, 1) || !readRegs(REG_FIFO_RD_PTR, &rd, 1)) {
    i2cErrors++;
    return;
  }
  int n = (int)((wr - rd) & 0x1F);

  if ((pollCount++ & 63) == 0) {                           // statistics only, every ~190 ms
    uint8_t ovf = 0;
    if (readRegs(REG_FIFO_OVF, &ovf, 1)) fifoOverflow = ovf & 0x1F;
  }
  if (n == 0) return;

  uint8_t b[6 * 16];
  int done = 0;
  while (done < n) {
    int chunk = (n - done < 16) ? (n - done) : 16;   // <= 16 samples = 96 bytes (Wire buffer is 128)
    if (!readRegs(REG_FIFO_DATA, b, (size_t)chunk * 6)) {
      i2cErrors++;
      return;
    }
    for (int i = 0; i < chunk; i++) {
      const uint8_t *p = b + i * 6;
      Sample s;
      s.red  = (((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2]) & 0x3FFFF;
      s.ir   = (((uint32_t)p[3] << 16) | ((uint32_t)p[4] << 8) | p[5]) & 0x3FFFF;
      s.t_us = t - (int64_t)(n - 1 - (done + i)) * SAMPLE_PERIOD_US;
      if (xQueueSend(sampleQueue, &s, 0) != pdTRUE) droppedCount++;
    }
    done += chunk;
  }
}

static void startSampling() {
  esp_timer_create_args_t args = {};
  args.callback        = &onPollTimer;
  args.arg             = nullptr;
  args.dispatch_method = ESP_TIMER_TASK;
  args.name            = "ppg_poll";
  esp_timer_create(&args, &pollTimer);
  esp_timer_start_periodic(pollTimer, POLL_PERIOD_US);
}

static void printMeta() {
  Serial.printf("# PPG_META fs_hz=%lu adc_bits=18 led_red_ma=%.1f led_ir_ma=%.1f adc_range_na=%d "
                "part_id=0x%02X i2c_errors=%lu fifo_ovf=%lu dropped=%lu\n",
                (unsigned long)OUTPUT_RATE_HZ, LED_RED_PA * 0.2, LED_IR_PA * 0.2,
                2048 << ADC_RANGE_CODE, partId,
                (unsigned long)i2cErrors, (unsigned long)fifoOverflow, (unsigned long)droppedCount);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);
  Serial.println("# PPG_LOGGER v1");
  Serial.println("# columns=timestamp_us,red,ir (raw 18-bit counts)");

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(I2C_HZ);
  sampleQueue = xQueueCreate(QUEUE_LEN, sizeof(Sample));
}

void loop() {
  static uint32_t lastTryMs  = 0;
  static uint32_t lastMetaMs = 0;
  uint32_t now = millis();

  // (Re)try to find and configure the MAX30102 until it answers.
  if (!maxReady) {
    if (now - lastTryMs >= 2000 || lastTryMs == 0) {
      lastTryMs = now ? now : 1;
      if (initMax()) {
        maxReady = true;
        if (partId != 0x15) {
          Serial.printf("# WARNING part_id=0x%02X (MAX30102 should be 0x15). Other chip or clone?\n", partId);
        }
        startSampling();
      } else {
        Serial.println("# ERROR no MAX30102 answering at 0x57. Check VIN/GND/SDA(21)/SCL(22).");
      }
    }
    delay(20);
    return;
  }

  // Drain everything that is waiting; block up to 20 ms when empty.
  Sample s;
  char   line[48];
  while (xQueueReceive(sampleQueue, &s, pdMS_TO_TICKS(20)) == pdTRUE) {
    int n = snprintf(line, sizeof(line), "%lld,%lu,%lu\n",
                     (long long)s.t_us, (unsigned long)s.red, (unsigned long)s.ir);
    Serial.write((const uint8_t *)line, n);
  }

  // Periodic meta line (the viewer opens the port after boot, so it would miss a one-time banner).
  if (now - lastMetaMs >= 2000) {
    lastMetaMs = now;
    printMeta();
  }
}