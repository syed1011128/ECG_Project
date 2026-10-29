/*
 * src/imu/main.cpp  -  MPU6050 IMU logger (ESP32, Arduino framework)
 * ---------------------------------------------------------------
 * MPU6050 (GY-521)  ->  ESP32
 *   VCC  -> 3V3
 *   GND  -> GND
 *   SDA  -> GPIO21
 *   SCL  -> GPIO22
 *   AD0  -> GND      (I2C address 0x68)
 *   INT  -> not connected
 *
 * Output (one line per sample, 100 Hz), RAW signed 16-bit integers:
 *   timestamp_us,ax,ay,az,gx,gy,gz
 *   e.g.  1234567,120,-45,8210,15,-3,7
 * The desktop viewer converts to g and deg/s (depends on the ranges below).
 *
 * Lines starting with '#' are status messages. Every 2 s the firmware prints
 *   # IMU_META fs_hz=100 accel_range_g=4 gyro_range_dps=500 who_am_i=0x68 i2c_errors=0
 * so the viewer always knows the real sampling rate and ranges.
 *
 * Design (same pattern as the ECG firmware)
 *  - esp_timer fires every 10 ms. The callback reads 14 bytes in ONE I2C burst
 *    (accel 6 + temp 2 + gyro 6), stamps the sample and pushes it to a queue.
 *  - Serial.print only happens in loop(), never in the timer callback.
 *  - After setup, only the timer callback touches the I2C bus.
 *
 * Board: "ESP32 Dev Module".  PlatformIO env: imu.
 * Close the Serial Monitor before running the Python viewer.
 */

#include <Arduino.h>
#include <Wire.h>
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

// ---------------- configuration ----------------
static const int      PIN_SDA        = 21;
static const int      PIN_SCL        = 22;
static const uint8_t  MPU_ADDR       = 0x68;     // AD0 = GND (0x69 if AD0 = 3V3)
static const uint32_t I2C_HZ         = 400000;

static const uint32_t SAMPLE_RATE_HZ = 100;      // must divide 1000 (DLPF on => 1 kHz base)
static const uint32_t SERIAL_BAUD    = 230400;
static const int      QUEUE_LEN      = 256;      // ~2.5 s of buffering

static const uint8_t  ACCEL_FS_SEL   = 1;        // 0:+-2g  1:+-4g   2:+-8g    3:+-16g
static const uint8_t  GYRO_FS_SEL    = 1;        // 0:+-250 1:+-500  2:+-1000  3:+-2000 deg/s
static const uint8_t  DLPF_CFG       = 3;        // 3 => ~44 Hz accel / ~42 Hz gyro, 1 kHz base
// -------------------------------------------------

static_assert(SAMPLE_RATE_HZ >= 4 && SAMPLE_RATE_HZ <= 1000 && (1000 % SAMPLE_RATE_HZ) == 0,
              "SAMPLE_RATE_HZ must divide 1000");

static const uint8_t  REG_SMPLRT_DIV   = 0x19;
static const uint8_t  REG_CONFIG       = 0x1A;
static const uint8_t  REG_GYRO_CONFIG  = 0x1B;
static const uint8_t  REG_ACCEL_CONFIG = 0x1C;
static const uint8_t  REG_ACCEL_XOUT_H = 0x3B;
static const uint8_t  REG_PWR_MGMT_1   = 0x6B;
static const uint8_t  REG_WHO_AM_I     = 0x75;

static const uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;
static const int      ACCEL_RANGE_G    = 2 << ACCEL_FS_SEL;
static const int      GYRO_RANGE_DPS   = 250 << GYRO_FS_SEL;

struct Sample {
  int64_t t_us;
  int16_t v[6];          // ax, ay, az, gx, gy, gz
};

static QueueHandle_t      sampleQueue  = nullptr;
static esp_timer_handle_t sampleTimer  = nullptr;
static volatile uint32_t  droppedCount = 0;   // queue-full events
static volatile uint32_t  i2cErrors    = 0;   // failed bursts
static bool               mpuReady     = false;
static uint8_t            whoAmI       = 0;

// ---------------- I2C helpers ----------------
static bool writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool readRegs(uint8_t reg, uint8_t *buf, size_t len) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;    // repeated start
  size_t got = Wire.requestFrom((int)MPU_ADDR, (int)len);
  if (got != len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)Wire.read();
  return true;
}

// Reset, wake (the chip powers up ASLEEP, so every register reads 0 until woken), configure.
static bool initMpu() {
  if (!readRegs(REG_WHO_AM_I, &whoAmI, 1)) return false;   // nothing answering at MPU_ADDR
  bool ok = true;
  ok &= writeReg(REG_PWR_MGMT_1, 0x80);                    // device reset
  delay(100);
  ok &= writeReg(REG_PWR_MGMT_1, 0x01);                    // sleep off, clock = gyro X PLL
  delay(50);
  ok &= writeReg(REG_CONFIG,       DLPF_CFG);
  ok &= writeReg(REG_SMPLRT_DIV,   (uint8_t)(1000 / SAMPLE_RATE_HZ - 1));
  ok &= writeReg(REG_GYRO_CONFIG,  (uint8_t)(GYRO_FS_SEL  << 3));
  ok &= writeReg(REG_ACCEL_CONFIG, (uint8_t)(ACCEL_FS_SEL << 3));
  return ok;
}

// ---------------- sampling ----------------
// Runs in the esp_timer task every SAMPLE_PERIOD_US.
static void onSampleTimer(void *arg) {
  uint8_t b[14];
  int64_t t = esp_timer_get_time();
  if (!readRegs(REG_ACCEL_XOUT_H, b, sizeof(b))) {
    i2cErrors++;
    return;
  }
  Sample s;
  s.t_us = t;
  s.v[0] = (int16_t)((b[0]  << 8) | b[1]);    // ax
  s.v[1] = (int16_t)((b[2]  << 8) | b[3]);    // ay
  s.v[2] = (int16_t)((b[4]  << 8) | b[5]);    // az
  // b[6], b[7] = temperature (not logged)
  s.v[3] = (int16_t)((b[8]  << 8) | b[9]);    // gx
  s.v[4] = (int16_t)((b[10] << 8) | b[11]);   // gy
  s.v[5] = (int16_t)((b[12] << 8) | b[13]);   // gz
  if (xQueueSend(sampleQueue, &s, 0) != pdTRUE) droppedCount++;
}

static void startSampling() {
  esp_timer_create_args_t args = {};
  args.callback        = &onSampleTimer;
  args.arg             = nullptr;
  args.dispatch_method = ESP_TIMER_TASK;
  args.name            = "imu_sample";
  esp_timer_create(&args, &sampleTimer);
  esp_timer_start_periodic(sampleTimer, SAMPLE_PERIOD_US);
}

static void printMeta() {
  Serial.printf("# IMU_META fs_hz=%lu accel_range_g=%d gyro_range_dps=%d who_am_i=0x%02X "
                "i2c_errors=%lu dropped=%lu\n",
                (unsigned long)SAMPLE_RATE_HZ, ACCEL_RANGE_G, GYRO_RANGE_DPS, whoAmI,
                (unsigned long)i2cErrors, (unsigned long)droppedCount);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);
  Serial.println("# IMU_LOGGER v1");
  Serial.println("# columns=timestamp_us,ax,ay,az,gx,gy,gz (raw int16)");

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(I2C_HZ);
  sampleQueue = xQueueCreate(QUEUE_LEN, sizeof(Sample));
}

void loop() {
  static uint32_t lastTryMs  = 0;
  static uint32_t lastMetaMs = 0;
  uint32_t now = millis();

  // (Re)try to find and configure the MPU6050 until it answers.
  if (!mpuReady) {
    if (now - lastTryMs >= 2000 || lastTryMs == 0) {
      lastTryMs = now ? now : 1;
      if (initMpu()) {
        mpuReady = true;
        if (whoAmI != 0x68) {
          Serial.printf("# WARNING who_am_i=0x%02X (MPU6050 should be 0x68). Clone or other chip?\n", whoAmI);
        }
        startSampling();
      } else {
        Serial.println("# ERROR no MPU6050 answering at 0x68. Check VCC/GND/SDA(21)/SCL(22)/AD0.");
      }
    }
    delay(20);
    return;
  }

  // Drain everything that is waiting; block up to 20 ms when empty.
  Sample s;
  char   line[72];
  while (xQueueReceive(sampleQueue, &s, pdMS_TO_TICKS(20)) == pdTRUE) {
    int n = snprintf(line, sizeof(line), "%lld,%d,%d,%d,%d,%d,%d\n",
                     (long long)s.t_us,
                     (int)s.v[0], (int)s.v[1], (int)s.v[2],
                     (int)s.v[3], (int)s.v[4], (int)s.v[5]);
    Serial.write((const uint8_t *)line, n);
  }

  // Periodic meta line (the viewer opens the port after boot, so it would miss a one-time banner).
  if (now - lastMetaMs >= 2000) {
    lastMetaMs = now;
    printMeta();
  }
}