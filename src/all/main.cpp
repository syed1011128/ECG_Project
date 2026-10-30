/*
 * src/all/main.cpp  -  ECG + IMU + PPG combined logger (ESP32, Arduino framework)
 * ---------------------------------------------------------------
 * AD8232 (ECG)        MPU6050 (IMU)         MAX30102 (PPG)           ESP32
 *   OUTPUT ------------------------------------------------------->  GPIO34
 *   LO+    ------------------------------------------------------->  GPIO32
 *   LO-    ------------------------------------------------------->  GPIO33
 *   3.3V / GND (all three sensors: 3V3 and GND only, never 5 V on the AD8232)
 *   SDA   (MPU6050 SDA and MAX30102 SDA, same two wires)  -------->  GPIO21
 *   SCL   (MPU6050 SCL and MAX30102 SCL, same two wires)  -------->  GPIO22
 *   MPU6050 AD0 -> GND (address 0x68);  MAX30102 address is 0x57;  INT pins not used.
 *
 * ONE serial stream, one tagged CSV line per sample, all on the SAME clock
 * (esp_timer_get_time, microseconds since boot):
 *   E,timestamp_us,adc,lo_plus,lo_minus                 ECG   250 Hz
 *   I,timestamp_us,ax,ay,az,gx,gy,gz                    IMU   100 Hz  (raw int16)
 *   P,timestamp_us,red,ir                               PPG   100 Hz  (raw 18-bit counts)
 * Lines starting with '#' are status messages; every 2 s the firmware prints
 *   # ECG_META ...   # IMU_META ...   # PPG_META ...   # ALL_META ...
 * (the viewer opens the port after boot, so a one-time banner would be lost).
 *
 * Why the work is split like this
 *  - ECG: an esp_timer callback every 4 ms reads the ADC and stamps the sample. It does NO
 *    I2C, so a slow I2C transfer can never delay an ECG sample.
 *  - I2C (PPG FIFO poll every 3 ms, IMU read every 10 ms): ONE FreeRTOS task. It is the only
 *    code that touches the I2C bus after setup, so the two I2C sensors never collide.
 *  - Everything goes into ONE FreeRTOS queue. loop() drains it and prints. Serial.print never
 *    runs in a sampling path, so sampling timing does not depend on serial speed.
 *  - PPG timestamps are reconstructed (newest = FIFO read time, older ones one sample period
 *    apart), error about +-3 ms. The IMU runs at 1 kHz internally and the latest value is read
 *    every 10 ms, so its timestamp error is below 1 ms. ECG timestamps are taken at the ADC read.
 *  - If one I2C sensor is missing, the others keep streaming and the error is printed every 2 s.
 *
 * Serial speed: 460800 baud (the three streams need about 13 kB/s, 230400 baud would be
 * about 55 % busy). If your USB-serial adapter has trouble, set SERIAL_BAUD to 230400 here and
 * use --baud 230400 in the viewer.
 *
 * Board: "ESP32 Dev Module".  PlatformIO env: all.
 * Close the Serial Monitor before running the Python viewer.
 */

#include <Arduino.h>
#include <Wire.h>
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// ---------------- configuration ----------------
static const uint32_t SERIAL_BAUD      = 460800;
static const int      QUEUE_LEN        = 768;       // ~1.7 s of buffering at 450 messages/s

// ECG (AD8232)
static const int      PIN_ECG_OUT      = 34;
static const int      PIN_LO_PLUS      = 32;
static const int      PIN_LO_MINUS     = 33;
static const uint32_t ECG_RATE_HZ      = 250;

// I2C bus
static const int      PIN_SDA          = 21;
static const int      PIN_SCL          = 22;
static const uint32_t I2C_HZ           = 400000;
static const uint32_t POLL_PERIOD_MS   = 3;         // I2C task period (PPG FIFO poll)

// IMU (MPU6050)
static const uint8_t  MPU_ADDR         = 0x68;
static const uint32_t IMU_RATE_HZ      = 100;       // output rate (the chip itself runs at 1 kHz)
static const uint8_t  ACCEL_FS_SEL     = 1;         // 0:+-2g  1:+-4g   2:+-8g    3:+-16g
static const uint8_t  GYRO_FS_SEL      = 1;         // 0:+-250 1:+-500  2:+-1000  3:+-2000 deg/s
static const uint8_t  DLPF_CFG         = 3;         // ~44 Hz accel / ~42 Hz gyro, 1 kHz base

// PPG (MAX30102)   output rate = SAMPLE_RATE_SPS / AVERAGING = 100 Hz
static const uint8_t  MAX_ADDR         = 0x57;
static const uint16_t SAMPLE_RATE_SPS  = 400;
static const uint8_t  AVERAGING        = 4;
static const uint8_t  PULSE_WIDTH_CODE = 3;         // 411 us, 18 bit (max 400 sps)
static const uint8_t  ADC_RANGE_CODE   = 1;         // 4096 nA
static const uint8_t  LED_RED_PA       = 0x24;      // value * 0.2 mA  (0x24 = 7.2 mA)
static const uint8_t  LED_IR_PA        = 0x24;
// -------------------------------------------------

static_assert(configTICK_RATE_HZ >= 1000, "needs a 1 ms FreeRTOS tick (default for Arduino-ESP32)");
static_assert(SAMPLE_RATE_SPS % AVERAGING == 0, "AVERAGING must divide SAMPLE_RATE_SPS");
static_assert(1000 % IMU_RATE_HZ == 0, "IMU_RATE_HZ must divide 1000");

static const uint32_t PPG_RATE_HZ        = SAMPLE_RATE_SPS / AVERAGING;
static const int64_t  PPG_PERIOD_US      = 1000000LL / PPG_RATE_HZ;
static const int64_t  IMU_PERIOD_US      = 1000000LL / IMU_RATE_HZ;
static const uint32_t ECG_PERIOD_US      = 1000000UL / ECG_RATE_HZ;
static const int      ACCEL_RANGE_G      = 2 << ACCEL_FS_SEL;
static const int      GYRO_RANGE_DPS     = 250 << GYRO_FS_SEL;

// MPU6050 registers
static const uint8_t MPU_REG_SMPLRT_DIV   = 0x19;
static const uint8_t MPU_REG_CONFIG       = 0x1A;
static const uint8_t MPU_REG_GYRO_CONFIG  = 0x1B;
static const uint8_t MPU_REG_ACCEL_CONFIG = 0x1C;
static const uint8_t MPU_REG_ACCEL_XOUT_H = 0x3B;
static const uint8_t MPU_REG_PWR_MGMT_1   = 0x6B;
static const uint8_t MPU_REG_WHO_AM_I     = 0x75;

// MAX30102 registers
static const uint8_t MAX_REG_INT_STATUS1 = 0x00;
static const uint8_t MAX_REG_INT_STATUS2 = 0x01;
static const uint8_t MAX_REG_FIFO_WR_PTR = 0x04;
static const uint8_t MAX_REG_FIFO_OVF    = 0x05;
static const uint8_t MAX_REG_FIFO_RD_PTR = 0x06;
static const uint8_t MAX_REG_FIFO_DATA   = 0x07;
static const uint8_t MAX_REG_FIFO_CFG    = 0x08;
static const uint8_t MAX_REG_MODE_CFG    = 0x09;
static const uint8_t MAX_REG_SPO2_CFG    = 0x0A;
static const uint8_t MAX_REG_LED1_PA     = 0x0C;   // RED
static const uint8_t MAX_REG_LED2_PA     = 0x0D;   // IR
static const uint8_t MAX_REG_PART_ID     = 0xFF;   // 0x15 for MAX30102

enum Kind : uint8_t { KIND_ECG = 0, KIND_IMU = 1, KIND_PPG = 2 };

struct Msg {
  int64_t t_us;
  int32_t v[6];     // ECG: adc, lo+, lo-   IMU: ax ay az gx gy gz   PPG: red, ir
  uint8_t kind;
};

static QueueHandle_t msgQueue = nullptr;
static volatile uint32_t droppedCount[3] = {0, 0, 0};   // queue-full events per kind
static volatile uint32_t imuI2cErrors    = 0;
static volatile uint32_t ppgI2cErrors    = 0;
static volatile uint32_t ppgFifoOverflow = 0;
static volatile bool     imuReady        = false;
static volatile bool     ppgReady        = false;
static volatile uint8_t  whoAmI          = 0;
static volatile uint8_t  partId          = 0;

static inline void pushMsg(const Msg &m) {
  if (xQueueSend(msgQueue, &m, 0) != pdTRUE) droppedCount[m.kind]++;
}

// ---------------- I2C helpers (only the I2C task uses these after setup) ----------------
static bool writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool readRegs(uint8_t addr, uint8_t reg, uint8_t *buf, size_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;    // repeated start
  size_t got = Wire.requestFrom((int)addr, (int)len);
  if (got != len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)Wire.read();
  return true;
}

// ---------------- ECG ----------------
// Runs in the esp_timer task every ECG_PERIOD_US. No I2C here, so the sample time is steady.
static void onEcgTimer(void *arg) {
  Msg m;
  m.kind = KIND_ECG;
  m.t_us = esp_timer_get_time();
  m.v[0] = analogRead(PIN_ECG_OUT);
  m.v[1] = digitalRead(PIN_LO_PLUS);
  m.v[2] = digitalRead(PIN_LO_MINUS);
  m.v[3] = m.v[4] = m.v[5] = 0;
  pushMsg(m);
}

// ---------------- MPU6050 ----------------
static bool initMpu() {
  uint8_t id = 0;
  if (!readRegs(MPU_ADDR, MPU_REG_WHO_AM_I, &id, 1)) return false;   // nothing answering
  whoAmI = id;
  bool ok = true;
  ok &= writeReg(MPU_ADDR, MPU_REG_PWR_MGMT_1, 0x80);                // device reset
  delay(100);
  ok &= writeReg(MPU_ADDR, MPU_REG_PWR_MGMT_1, 0x01);                // sleep off, clock = gyro X PLL
  delay(50);
  ok &= writeReg(MPU_ADDR, MPU_REG_CONFIG,       DLPF_CFG);
  ok &= writeReg(MPU_ADDR, MPU_REG_SMPLRT_DIV,   0);                 // chip runs at 1 kHz, we read the latest value at IMU_RATE_HZ
  ok &= writeReg(MPU_ADDR, MPU_REG_GYRO_CONFIG,  (uint8_t)(GYRO_FS_SEL  << 3));
  ok &= writeReg(MPU_ADDR, MPU_REG_ACCEL_CONFIG, (uint8_t)(ACCEL_FS_SEL << 3));
  return ok;
}

static void readImu(int64_t t) {
  uint8_t b[14];
  if (!readRegs(MPU_ADDR, MPU_REG_ACCEL_XOUT_H, b, sizeof(b))) {
    imuI2cErrors++;
    return;
  }
  Msg m;
  m.kind = KIND_IMU;
  m.t_us = t;
  m.v[0] = (int16_t)((b[0]  << 8) | b[1]);    // ax
  m.v[1] = (int16_t)((b[2]  << 8) | b[3]);    // ay
  m.v[2] = (int16_t)((b[4]  << 8) | b[5]);    // az
  m.v[3] = (int16_t)((b[8]  << 8) | b[9]);    // gx   (b[6], b[7] = temperature, not logged)
  m.v[4] = (int16_t)((b[10] << 8) | b[11]);   // gy
  m.v[5] = (int16_t)((b[12] << 8) | b[13]);   // gz
  pushMsg(m);
}

// ---------------- MAX30102 ----------------
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

static bool initMax() {
  uint8_t srCode = 0, aveCode = 0;
  if (!sampleRateCode(SAMPLE_RATE_SPS, srCode) || !averagingCode(AVERAGING, aveCode)) return false;
  uint8_t id = 0;
  if (!readRegs(MAX_ADDR, MAX_REG_PART_ID, &id, 1)) return false;    // nothing answering
  partId = id;

  bool ok = true;
  ok &= writeReg(MAX_ADDR, MAX_REG_MODE_CFG, 0x40);                  // soft reset (bit clears itself)
  for (int i = 0; i < 50; i++) {
    delay(2);
    uint8_t m = 0xFF;
    if (readRegs(MAX_ADDR, MAX_REG_MODE_CFG, &m, 1) && (m & 0x40) == 0) break;
  }
  ok &= writeReg(MAX_ADDR, MAX_REG_FIFO_WR_PTR, 0);
  ok &= writeReg(MAX_ADDR, MAX_REG_FIFO_OVF, 0);
  ok &= writeReg(MAX_ADDR, MAX_REG_FIFO_RD_PTR, 0);
  ok &= writeReg(MAX_ADDR, MAX_REG_FIFO_CFG, (uint8_t)((aveCode << 5) | (1 << 4) | 0x0F));  // averaging, rollover on
  ok &= writeReg(MAX_ADDR, MAX_REG_SPO2_CFG, (uint8_t)((ADC_RANGE_CODE << 5) | (srCode << 2) | PULSE_WIDTH_CODE));
  ok &= writeReg(MAX_ADDR, MAX_REG_LED1_PA, LED_RED_PA);
  ok &= writeReg(MAX_ADDR, MAX_REG_LED2_PA, LED_IR_PA);
  ok &= writeReg(MAX_ADDR, MAX_REG_MODE_CFG, 0x03);                  // SpO2 mode: RED then IR per sample
  uint8_t dummy[2];
  readRegs(MAX_ADDR, MAX_REG_INT_STATUS1, dummy, 1);                 // clear interrupt flags
  readRegs(MAX_ADDR, MAX_REG_INT_STATUS2, dummy + 1, 1);
  return ok;
}

static void pollPpg(int64_t t) {
  static uint32_t pollCount = 0;
  uint8_t wr = 0, rd = 0;
  if (!readRegs(MAX_ADDR, MAX_REG_FIFO_WR_PTR, &wr, 1) || !readRegs(MAX_ADDR, MAX_REG_FIFO_RD_PTR, &rd, 1)) {
    ppgI2cErrors++;
    return;
  }
  int n = (int)((wr - rd) & 0x1F);

  if ((pollCount++ & 63) == 0) {                           // statistics only, every ~190 ms
    uint8_t ovf = 0;
    if (readRegs(MAX_ADDR, MAX_REG_FIFO_OVF, &ovf, 1)) ppgFifoOverflow = ovf & 0x1F;
  }
  if (n == 0) return;

  uint8_t b[6 * 16];
  int done = 0;
  while (done < n) {
    int chunk = (n - done < 16) ? (n - done) : 16;   // <= 16 samples = 96 bytes (Wire buffer is 128)
    if (!readRegs(MAX_ADDR, MAX_REG_FIFO_DATA, b, (size_t)chunk * 6)) {
      ppgI2cErrors++;
      return;
    }
    for (int i = 0; i < chunk; i++) {
      const uint8_t *p = b + i * 6;
      Msg m;
      m.kind = KIND_PPG;
      m.v[0] = (int32_t)((((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2]) & 0x3FFFF);   // red
      m.v[1] = (int32_t)((((uint32_t)p[3] << 16) | ((uint32_t)p[4] << 8) | p[5]) & 0x3FFFF);   // ir
      m.v[2] = m.v[3] = m.v[4] = m.v[5] = 0;
      m.t_us = t - (int64_t)(n - 1 - (done + i)) * PPG_PERIOD_US;
      pushMsg(m);
    }
    done += chunk;
  }
}

// ---------------- I2C task: the only code that touches the bus after setup ----------------
static void i2cTask(void *arg) {
  TickType_t lastWake = xTaskGetTickCount();
  int64_t nextImuUs = 0, nextImuTryUs = 0, nextPpgTryUs = 0;
  for (;;) {
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(POLL_PERIOD_MS));
    int64_t t = esp_timer_get_time();

    // (Re)try to find and configure each sensor until it answers; the other one keeps running.
    if (!ppgReady && t >= nextPpgTryUs) {
      nextPpgTryUs = t + 2000000;
      if (initMax()) ppgReady = true;
    }
    if (!imuReady && t >= nextImuTryUs) {
      nextImuTryUs = t + 2000000;
      if (initMpu()) {
        imuReady = true;
        nextImuUs = 0;
      }
      t = esp_timer_get_time();
    }

    if (ppgReady) pollPpg(t);

    if (imuReady && t >= nextImuUs) {
      readImu(t);
      nextImuUs += IMU_PERIOD_US;
      if (nextImuUs <= t) nextImuUs = t + IMU_PERIOD_US;    // first sample, or we fell behind: resync
    }
  }
}

// ---------------- status messages (printed from loop() only) ----------------
static void printMeta(uint32_t queueMax) {
  Serial.printf("# ECG_META fs_hz=%lu adc_bits=12 dropped=%lu\n",
                (unsigned long)ECG_RATE_HZ, (unsigned long)droppedCount[KIND_ECG]);
  if (imuReady) {
    Serial.printf("# IMU_META fs_hz=%lu accel_range_g=%d gyro_range_dps=%d who_am_i=0x%02X "
                  "i2c_errors=%lu dropped=%lu\n",
                  (unsigned long)IMU_RATE_HZ, ACCEL_RANGE_G, GYRO_RANGE_DPS, (unsigned)whoAmI,
                  (unsigned long)imuI2cErrors, (unsigned long)droppedCount[KIND_IMU]);
    if (whoAmI != 0x68) {
      Serial.printf("# WARNING who_am_i=0x%02X (MPU6050 should be 0x68). Clone or other chip?\n", (unsigned)whoAmI);
    }
  } else {
    Serial.println("# ERROR no MPU6050 answering at 0x68. Check VCC/GND/SDA(21)/SCL(22)/AD0.");
  }
  if (ppgReady) {
    Serial.printf("# PPG_META fs_hz=%lu adc_bits=18 led_red_ma=%.1f led_ir_ma=%.1f adc_range_na=%d "
                  "part_id=0x%02X i2c_errors=%lu fifo_ovf=%lu dropped=%lu\n",
                  (unsigned long)PPG_RATE_HZ, LED_RED_PA * 0.2, LED_IR_PA * 0.2,
                  2048 << ADC_RANGE_CODE, (unsigned)partId,
                  (unsigned long)ppgI2cErrors, (unsigned long)ppgFifoOverflow,
                  (unsigned long)droppedCount[KIND_PPG]);
    if (partId != 0x15) {
      Serial.printf("# WARNING part_id=0x%02X (MAX30102 should be 0x15). Other chip or clone?\n", (unsigned)partId);
    }
  } else {
    Serial.println("# ERROR no MAX30102 answering at 0x57. Check VIN/GND/SDA(21)/SCL(22).");
  }
  Serial.printf("# ALL_META baud=%lu queue_len=%d queue_max=%lu\n",
                (unsigned long)SERIAL_BAUD, QUEUE_LEN, (unsigned long)queueMax);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);
  Serial.println("# ALL_LOGGER v1");
  Serial.println("# lines: E,timestamp_us,adc,lo_plus,lo_minus | I,timestamp_us,ax,ay,az,gx,gy,gz | P,timestamp_us,red,ir");

  analogReadResolution(12);                       // 0..4095
  analogSetPinAttenuation(PIN_ECG_OUT, ADC_11db); // ~0..3.3 V range
  pinMode(PIN_LO_PLUS, INPUT);
  pinMode(PIN_LO_MINUS, INPUT);

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(I2C_HZ);

  msgQueue = xQueueCreate(QUEUE_LEN, sizeof(Msg));

  esp_timer_handle_t ecgTimer = nullptr;
  esp_timer_create_args_t args = {};
  args.callback        = &onEcgTimer;
  args.arg             = nullptr;
  args.dispatch_method = ESP_TIMER_TASK;
  args.name            = "ecg_sample";
  esp_timer_create(&args, &ecgTimer);
  esp_timer_start_periodic(ecgTimer, ECG_PERIOD_US);

  xTaskCreatePinnedToCore(i2cTask, "i2c_sensors", 6144, nullptr, 3, nullptr, 1);
}

void loop() {
  static uint32_t lastMetaMs = 0;
  static uint32_t queueMax   = 0;
  Msg  m;
  char line[96];

  // Drain everything that is waiting; block up to 20 ms when empty.
  for (;;) {
    uint32_t waiting = (uint32_t)uxQueueMessagesWaiting(msgQueue);
    if (waiting > queueMax) queueMax = waiting;
    if (xQueueReceive(msgQueue, &m, pdMS_TO_TICKS(20)) != pdTRUE) break;
    int n = 0;
    switch (m.kind) {
      case KIND_ECG:
        n = snprintf(line, sizeof(line), "E,%lld,%ld,%ld,%ld\n",
                     (long long)m.t_us, (long)m.v[0], (long)m.v[1], (long)m.v[2]);
        break;
      case KIND_IMU:
        n = snprintf(line, sizeof(line), "I,%lld,%ld,%ld,%ld,%ld,%ld,%ld\n",
                     (long long)m.t_us, (long)m.v[0], (long)m.v[1], (long)m.v[2],
                     (long)m.v[3], (long)m.v[4], (long)m.v[5]);
        break;
      default:
        n = snprintf(line, sizeof(line), "P,%lld,%ld,%ld\n",
                     (long long)m.t_us, (long)m.v[0], (long)m.v[1]);
        break;
    }
    if (n > 0) Serial.write((const uint8_t *)line, (size_t)n);
  }

  uint32_t now = millis();
  if (now - lastMetaMs >= 2000) {
    lastMetaMs = now;
    printMeta(queueMax);
  }
}