/*
 * ecg_logger.ino
 * ---------------------------------------------------------------
 * ESP32 + AD8232 single-lead ECG  ->  USB serial (CSV lines)
 *
 * Wiring (3.3 V only, never 5 V on the AD8232):
 *   AD8232 3.3V   -> ESP32 3V3
 *   AD8232 GND    -> ESP32 GND
 *   AD8232 OUTPUT -> GPIO34   (ADC1, input-only pin)
 *   AD8232 LO+    -> GPIO32
 *   AD8232 LO-    -> GPIO33
 *   AD8232 SDN    -> not connected
 *
 * Output (one line per sample, 250 Hz):
 *   timestamp_us,adc,lo_plus,lo_minus
 *   e.g.  1234567,2048,0,0
 *
 * Lines starting with '#' are comments / status messages. The Python
 * viewer prints them and otherwise ignores them.
 *
 * Design notes
 *  - Sampling is driven by an esp_timer periodic timer, so the sample
 *    instant does not depend on loop() or on Serial speed.
 *  - The timer callback only reads the ADC and pushes into a FreeRTOS
 *    queue. loop() drains the queue and prints. Serial.print never runs
 *    inside the timer callback.
 *  - The timestamp is taken at the start of the callback (esp_timer_get_time,
 *    64-bit microseconds since boot). Gaps in the timestamps reveal any
 *    dropped samples.
 *  - Works with Arduino-ESP32 core 2.x and 3.x (no hw-timer API used).
 *
 * Board: "ESP32 Dev Module" (or your exact ESP32 board).
 * Serial Monitor must be CLOSED when the Python viewer is running.
 */

#include <Arduino.h>
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

// ---------------- configuration ----------------
static const int      PIN_ECG_OUT      = 34;   // AD8232 OUTPUT
static const int      PIN_LO_PLUS      = 32;   // AD8232 LO+
static const int      PIN_LO_MINUS     = 33;   // AD8232 LO-

static const uint32_t SAMPLE_RATE_HZ   = 250;  // 250-500 Hz is fine for ECG
static const uint32_t SERIAL_BAUD      = 230400;
static const int      QUEUE_LEN        = 512;  // ~2 s of buffering at 250 Hz
// -------------------------------------------------

static const uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

struct Sample {
  int64_t  t_us;
  uint16_t adc;
  uint8_t  lo_plus;
  uint8_t  lo_minus;
};

static QueueHandle_t       sampleQueue   = nullptr;
static esp_timer_handle_t  sampleTimer   = nullptr;
static volatile uint32_t   droppedCount  = 0;   // queue-full events

// Runs in the esp_timer task every SAMPLE_PERIOD_US.
static void onSampleTimer(void *arg) {
  Sample s;
  s.t_us     = esp_timer_get_time();
  s.adc      = (uint16_t)analogRead(PIN_ECG_OUT);
  s.lo_plus  = (uint8_t)digitalRead(PIN_LO_PLUS);
  s.lo_minus = (uint8_t)digitalRead(PIN_LO_MINUS);
  if (xQueueSend(sampleQueue, &s, 0) != pdTRUE) {
    droppedCount++;
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);

  analogReadResolution(12);                       // 0..4095
  analogSetPinAttenuation(PIN_ECG_OUT, ADC_11db); // ~0..3.3 V range
  pinMode(PIN_LO_PLUS, INPUT);
  pinMode(PIN_LO_MINUS, INPUT);

  sampleQueue = xQueueCreate(QUEUE_LEN, sizeof(Sample));

  Serial.println("# ECG_LOGGER v1");
  Serial.printf("# fs_hz=%lu baud=%lu adc_bits=12 columns=timestamp_us,adc,lo_plus,lo_minus\n",
                (unsigned long)SAMPLE_RATE_HZ, (unsigned long)SERIAL_BAUD);

  esp_timer_create_args_t args = {};
  args.callback        = &onSampleTimer;
  args.arg             = nullptr;
  args.dispatch_method = ESP_TIMER_TASK;
  args.name            = "ecg_sample";
  esp_timer_create(&args, &sampleTimer);
  esp_timer_start_periodic(sampleTimer, SAMPLE_PERIOD_US);
}

void loop() {
  Sample s;
  char   line[48];

  // Drain everything that is waiting; block up to 20 ms when empty.
  while (xQueueReceive(sampleQueue, &s, pdMS_TO_TICKS(20)) == pdTRUE) {
    int n = snprintf(line, sizeof(line), "%lld,%u,%u,%u\n",
                     (long long)s.t_us,
                     (unsigned)s.adc,
                     (unsigned)s.lo_plus,
                     (unsigned)s.lo_minus);
    Serial.write((const uint8_t *)line, n);
  }

  // Report queue overflows (means Serial could not keep up).
  static uint32_t lastReportMs = 0;
  uint32_t now = millis();
  if (now - lastReportMs >= 5000) {
    lastReportMs = now;
    if (droppedCount > 0) {
      Serial.printf("# dropped=%lu\n", (unsigned long)droppedCount);
    }
  }
}