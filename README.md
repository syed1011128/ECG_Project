# Non-Invasive Cardiovascular Monitoring: ECG + IMU + PPG Acquisition and Live Viewers

ESP32 firmware and Python desktop viewers for three sensors, built one at a time:

| Stage | Sensor | Measures | Firmware env | Viewer |
|---|---|---|---|---|
| 1 | AD8232 | single-lead ECG | `env:ecg` | `desktop/viewer.py` |
| 2 | MPU6050 | motion (accelerometer + gyroscope) | `env:imu` | `desktop/imu_viewer.py` |
| 3 | MAX30102 | PPG (red + infrared) | `env:ppg` | `desktop/ppg_viewer.py` |
| 4 | all three | everything on one clock, pulse arrival time (PAT), motion-aware beat validity | `env:all` | `desktop/fusion_viewer.py` |

This is the data-acquisition part of the capstone **"Non-Invasive Cardiovascular Health Monitoring via
Fused PPG-ECG Sensing and Edge-Computed Anomaly Detection"** (MAX30102 + AD8232 + MPU6050 + ESP32).

> Research / learning prototype. **Not a medical device.** Do not use it for diagnosis or treatment decisions.

```
Sensor -> ESP32 (timer-driven sampling, microsecond timestamps) -> USB serial, 230400 baud, CSV lines
       -> Python viewer (filter, peak detection, BPM, quality label, plot, CSV logging)
```

Current status:

- ECG viewer: working on real hardware.
- IMU viewer: working on real hardware.
- PPG viewer: firmware flashed and streaming on a real MAX30102 board, viewer confirmed working.
  Detection thresholds were chosen with synthetic data and still need tuning on real recordings.
- Fusion (stage 4): combined firmware and fusion viewer are built and tested with synthetic sensors and a
  virtual serial port. They have **not yet been run on the real board**, so the first real run of `env:all` is the next check.
- Not done yet: feature extraction, machine learning, TinyML (see the roadmap).

---

## Contents

1. [Project structure](#1-project-structure)
2. [Safety](#2-safety)
3. [Hardware and wiring](#3-hardware-and-wiring)
4. [One-time Ubuntu setup](#4-one-time-ubuntu-setup)
5. [Firmware: build and flash](#5-firmware-build-and-flash)
6. [Python environment](#6-python-environment)
7. [Stage 1: ECG (AD8232)](#7-stage-1-ecg-ad8232)
8. [Stage 2: IMU (MPU6050)](#8-stage-2-imu-mpu6050)
9. [Stage 3: PPG (MAX30102)](#9-stage-3-ppg-max30102)
10. [Stage 4: ECG + IMU + PPG together (fusion)](#10-stage-4-ecg--imu--ppg-together-fusion)
11. [Serial data formats](#11-serial-data-formats)
12. [Recorded data](#12-recorded-data)
13. [Design notes (how it works)](#13-design-notes-how-it-works)
14. [Troubleshooting](#14-troubleshooting)
15. [Known limitations](#15-known-limitations)
16. [Roadmap](#16-roadmap)

---

## 1. Project structure

```
ECG_Project/
├── platformio.ini              four environments: ecg, imu, ppg, all (default_envs = ecg)
├── README.md
├── .gitignore
│
├── src/
│   ├── ecg/main.cpp            AD8232 firmware   (250 Hz)
│   ├── imu/main.cpp            MPU6050 firmware  (100 Hz)
│   ├── ppg/main.cpp            MAX30102 firmware (100 Hz, RED + IR)
│   └── all/main.cpp            ECG + IMU + PPG together, one tagged stream, one clock
├── include/  lib/  test/       created by PlatformIO, left empty
├── .pio/                       build output (auto-generated)
│
├── desktop/                    Python side
│   ├── requirements.txt        numpy, scipy, pyserial, pyqtgraph, PySide6
│   ├── processing.py           ECG filters, R-peaks, BPM, HRV, StreamingFilter (shared)
│   ├── sources.py              BaseSource, serial reader, ECG simulator
│   ├── viewer.py               live ECG GUI
│   ├── test_processing.py      ECG tests
│   ├── imu_processing.py       unit conversion, gyro offset, motion flag
│   ├── imu_sources.py          IMU serial reader + simulator
│   ├── imu_viewer.py           live IMU GUI
│   ├── test_imu_processing.py  IMU tests
│   ├── ppg_processing.py       finger detect, pulse peaks, BPM, perfusion index, R / SpO2 (experimental)
│   ├── ppg_sources.py          PPG serial reader + simulator
│   ├── ppg_viewer.py           live PPG GUI
│   ├── test_ppg_processing.py  PPG tests
│   ├── fusion_viewer.py        live fusion GUI (ECG + PPG + IMU + PAT)
│   ├── fusion_processing.py    PAT, beat validity (motion-aware), heart-rate cross-check
│   ├── fusion_sources.py       tagged-stream serial reader + synthetic fusion simulator
│   ├── fusion_recorder.py      four-file CSV logging
│   ├── test_fusion_processing.py  fusion tests
│   └── .venv/                  your virtual environment (not committed)
│
├── data/                       recorded sessions (CSV, created automatically)
└── .vscode/launch.json         optional F5 run configurations
```

Important: all `*_viewer.py`, `*_processing.py` and `*_sources.py` files, plus `processing.py` and `sources.py`,
must live in the same `desktop/` folder. The IMU and PPG modules import from `processing.py` / `sources.py`.

---

## 2. Safety

- ECG electrodes touch your body. **Run the laptop on battery with the charger unplugged**
  (or use a USB isolator) while electrodes are attached.
- Power the AD8232 from **3.3 V only**. Never connect it to 5 V.
- Stop immediately if you feel discomfort or skin irritation.
- Wire the sensors with the ESP32 **unplugged** from USB.

---

## 3. Hardware and wiring

Board: ESP32 dev board (`esp32dev`). MPU6050 and MAX30102 share the same I2C bus
(addresses 0x68 and 0x57, no conflict). The AD8232 uses separate pins.

| Sensor pin | ESP32 pin |
|---|---|
| **AD8232** 3.3V | 3V3 |
| AD8232 GND | GND |
| AD8232 OUTPUT | GPIO34 (ADC1, input only) |
| AD8232 LO+ | GPIO32 |
| AD8232 LO- | GPIO33 |
| AD8232 SDN | not connected |
| **MPU6050** VCC | 3V3 |
| MPU6050 GND | GND |
| MPU6050 SDA | GPIO21 |
| MPU6050 SCL | GPIO22 |
| MPU6050 AD0 | GND (address 0x68) |
| MPU6050 INT | not connected |
| **MAX30102** VIN | 3V3 |
| MAX30102 GND | GND |
| MAX30102 SDA | GPIO21 |
| MAX30102 SCL | GPIO22 |
| MAX30102 INT | not connected (firmware polls the FIFO) |

ECG electrodes (disposable Ag/AgCl gel pads, clean dry skin):

| Lead | Position |
|---|---|
| RA | right chest, below the collarbone |
| LA | left chest, below the collarbone |
| RL | right lower abdomen |

Check the colours on your cable, they differ between manufacturers. Sit still, do not talk, wait 10-20 s.

---

## 4. One-time Ubuntu setup

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libxcb-cursor0

# allow your user to open serial ports, then LOG OUT AND LOG IN (or reboot)
sudo usermod -aG dialout $USER

# brltty (braille service) steals CH340/CP210x ports on Ubuntu
sudo apt remove -y brltty
```

After logging back in:

```bash
groups                              # must include: dialout
ls /dev/ttyUSB* /dev/ttyACM*        # your ESP32 port, usually /dev/ttyUSB0
```

No port? Use a data cable (not charge-only) and check `dmesg | tail`.

VS Code extensions: **Python** (Microsoft), **PlatformIO IDE** (first start downloads toolchains, 10-30 minutes,
and the project may open slowly the first time). Optional: Rainbow CSV.

If upload gives a permission error, install PlatformIO's udev rules:

```bash
curl -fsSL https://raw.githubusercontent.com/platformio/platformio-core/develop/platformio/assets/system/99-platformio-udev.rules | sudo tee /etc/udev/rules.d/99-platformio-udev.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 5. Firmware: build and flash

`platformio.ini`:

```ini
[platformio]
default_envs = ecg

[env]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 230400
upload_speed = 921600

[env:ecg]
build_src_filter = -<*> +<ecg/>

[env:imu]
build_src_filter = -<*> +<imu/>

[env:ppg]
build_src_filter = -<*> +<ppg/>

[env:all]
build_src_filter = -<*> +<all/>
monitor_speed = 460800
```

Each environment builds only its own folder. One board runs **one firmware at a time**, so flash the
environment that matches the viewer you want to use.

**Choose the environment first.** `default_envs = ecg`, so an upload without choosing an environment
flashes the ECG firmware. This is the most common mistake (see troubleshooting).

Ways to flash:

1. VS Code status bar: click the `env:...` item, pick `ecg`, `imu` or `ppg`, then tick = Build, arrow = Upload.
   Check that the build log says `Processing <env>`.
2. PlatformIO sidebar (alien icon) -> Project Tasks -> `<env>` -> General -> Upload.
3. Terminal (works even when the extension misbehaves). `pio` lives in PlatformIO's own environment,
   not in your `.venv`:

   ```bash
   cd ~/Documents/PlatformIO/Projects/ECG_Project
   ~/.platformio/penv/bin/pio run -e ppg -t upload       # or -e ecg / -e imu / -e all
   ```

If upload hangs at `Connecting....`, hold the ESP32 **BOOT** button.

If you began with a single `src/main.cpp` (the first ECG firmware), move it:

```bash
mkdir -p src/ecg src/imu src/ppg src/all
mv src/main.cpp src/ecg/main.cpp
```

A loose `src/main.cpp` is ignored by every environment, so `env:ecg` would otherwise say "Nothing to build".

After flashing, open the Serial Monitor (230400) to check the output, then **close it**. Only one program
can use the serial port, and the viewer needs it.

Quick check of what the board is actually running (viewer closed):

```bash
cd desktop && source .venv/bin/activate
python - <<'EOF'
import serial
s = serial.Serial('/dev/ttyUSB0', 230400, timeout=1)
for _ in range(8):
    print(s.readline())
EOF
```

| Output | Firmware on the board |
|---|---|
| `ts,adc,lo+,lo-` (4 fields, 4 ms apart) | ECG |
| `ts,ax,ay,az,gx,gy,gz` (7 fields) | IMU |
| `ts,red,ir` (3 fields) plus `# PPG_META ...` | PPG |
| `E,ts,...` / `I,ts,...` / `P,ts,...` tagged lines (460800 baud) | combined (env:all) |
| `# ERROR no ... answering` | right firmware, wiring problem |
| unreadable bytes | baud mismatch |

---

## 6. Python environment

```bash
cd desktop
python3 -m venv .venv
source .venv/bin/activate          # prompt shows (.venv)
pip install -r requirements.txt
python test_processing.py          # ALL TESTS PASSED
python test_imu_processing.py      # ALL TESTS PASSED
python test_ppg_processing.py      # ALL TESTS PASSED
python test_fusion_processing.py   # ALL TESTS PASSED
```

In VS Code: `Ctrl+Shift+P` -> **Python: Select Interpreter** -> choose the `.venv` one.

Run all viewers from inside `desktop/` (`cd desktop` first). The viewers open their own Qt window, not a
VS Code panel. Check the taskbar if you cannot see it.

Optional F5 launchers are in `.vscode/launch.json` (ECG, IMU and PPG, each with "real sensor" and "simulate").

On Linux the viewers force `QT_QPA_PLATFORMTHEME=generic` to avoid a Qt/GNOME crash (see troubleshooting).
Opt out with `ECG_KEEP_QT_THEME=1`.

Every viewer has a `--simulate` mode: test the desktop side before connecting hardware.

---

## 7. Stage 1: ECG (AD8232)

### Principle

The heart's electrical activity produces a tiny voltage (about 1 mV) between electrodes. The AD8232 amplifies it
(instrumentation amplifier, filtering, right-leg drive), the ESP32 ADC samples the output. The R wave (the tall
spike) is the landmark for heart rate and for R-R interval variability (HRV).

### Firmware (`src/ecg/main.cpp`)

- `esp_timer` fires every 4 ms (250 Hz), reads GPIO34 (12-bit, 11 dB) and the leads-off pins LO+ / LO-.
- Sample is stamped with `esp_timer_get_time()` and pushed into a FreeRTOS queue. `loop()` prints it.
- Output line: `timestamp_us,adc,lo_plus,lo_minus`. Banner `# ECG_LOGGER v1`.

### Run

```bash
cd desktop && source .venv/bin/activate
python viewer.py --simulate                                      # no hardware
python viewer.py --list-ports
python viewer.py --port /dev/ttyUSB0 --subject S01 --activity rest
```

If `--port` is omitted, the viewer tries to auto-detect a CP210x / CH340 port.

### What the window shows

- **BPM**: median of the last few R-R intervals.
- **Signal**: `OK`, `SETTLING`, `LEADS OFF`, `SATURATED`, `FLAT`.
- **HRV**: SDNN, RMSSD, pNN50 from a short window (indicative only).
- Status line: source, measured sampling rate, sample count, dropped-sample estimate, bad serial lines.
- Middle plot: filtered ECG (0.5-40 Hz band-pass + 50 Hz notch, causal) with red R-peak markers.
- Bottom plot: raw ADC (0-4095). Stuck at 0 or 4095 means wiring or contact problems.
- R-peak markers appear about 0.4 s after the beat (the detector waits until the beat is finished).
- The **Simulate leads-off** button tests the warning.

### Processing

Causal band-pass 0.5-40 Hz + 50 Hz mains notch (Bangladesh mains is 50 Hz) -> Pan-Tompkins-style R-peak
detection (5-15 Hz band, derivative, squaring, 100 ms integration, adaptive threshold, refractory period)
-> R-R intervals -> BPM, SDNN, RMSSD, pNN50. Leads-off resets the filter and hides BPM.

### Options (`viewer.py`)

| Option | Default | Meaning |
|---|---|---|
| `--port` | auto | serial port |
| `--baud` | 230400 | must match the firmware |
| `--fs` | 250 | sampling rate set in the firmware (Hz) |
| `--window` | 8 | seconds shown on screen |
| `--notch` | 50 | mains notch in Hz, `0` disables |
| `--subject` | S01 | non-identifying subject code stored in the CSV |
| `--activity` | rest | condition label stored in the CSV |
| `--data-dir` | `../data` | where session CSVs go |
| `--no-log` | off | do not write CSV files |
| `--simulate` | off | built-in synthetic ECG |
| `--list-ports` | | list serial ports and exit |

---

## 8. Stage 2: IMU (MPU6050)

### Principle

The MPU6050 contains a MEMS accelerometer (measures proper acceleration, so about 1 g from gravity when still)
and a MEMS gyroscope (angular rate). Movement corrupts both PPG and ECG, so the IMU gives the "is this window
trustworthy?" context for the other two sensors.

### Firmware (`src/imu/main.cpp`)

- I2C 400 kHz, address 0x68. The chip starts asleep, so the firmware resets and wakes it (`PWR_MGMT_1`).
- Accel +-4 g (8192 LSB/g), gyro +-500 dps (65.5 LSB/dps), digital low-pass filter on, 100 Hz output.
- Reads the 14-byte burst from 0x3B (accel, temperature, gyro) in the timer task, retries init every 2 s.
- Output line: `timestamp_us,ax,ay,az,gx,gy,gz` (raw int16). Banner `# IMU_LOGGER v1` and a
  `# IMU_META fs_hz=100 accel_range_g=4 gyro_range_dps=500 who_am_i=0x68 i2c_errors=.. dropped=..` line
  repeated every 2 s (the viewer opens the port after reset, so a one-time banner would be lost).

### Run

```bash
cd desktop && source .venv/bin/activate
python imu_viewer.py --simulate
python imu_viewer.py --port /dev/ttyUSB0 --subject S01 --activity rest
```

The first 3 s show `CALIBRATING`: keep the sensor **still** (gyro offset is measured then).
Use **Recalibrate gyro** if it was moving.

### What you see

- Accel XYZ (g) and gyro XYZ (deg/s), fixed to the sensor range so clipping is obvious.
- `|a|`: about **1 g when still**, moves away from 1 g when shaken.
- Motion index (rolling std of `|a|`) with a threshold line, and a big `STILL` / `MOTION` label.
  `SATURATED` means raw values hit full scale.
- Sampling rate and ranges come from `IMU_META`, so conversion stays correct if you change them in firmware.

### Motion rule (starting values, tune them)

```
accel_idx = rolling std of |a| over 1 s             MOTION if > 0.05 g    (--motion-thr)
gyro_idx  = rolling mean of |gyro| over 1 s         MOTION if > 15 deg/s  (--gyro-thr)
```

Defaults were chosen with a synthetic signal. Record your own data (sitting still, tilting, shaking, walking)
and adjust. `MOTION` means the window is probably less reliable, not that the physiological signals are invalid.

### Quick test

1. Flat on the table: `|a|` about 1.00 g, gyro about 0, `STILL`.
2. Flip it over: the axis that read +1 g reads -1 g, `|a|` stays about 1 g.
3. Tilt slowly: XYZ change, `|a|` about 1 g, no MOTION.
4. Shake: `|a|` jumps, motion index crosses the line, `MOTION`.
5. Stop: back to `STILL` about 1 s later.

### Options (`imu_viewer.py`)

| Option | Default | Meaning |
|---|---|---|
| `--port`, `--baud` | auto, 230400 | serial port and baud |
| `--fs` | 100 | sampling rate (firmware `IMU_META` overrides) |
| `--accel-range` | 4 | g (firmware `IMU_META` overrides) |
| `--gyro-range` | 500 | deg/s (firmware `IMU_META` overrides) |
| `--window` | 10 | seconds shown |
| `--window-s` | 1.0 | motion-index window (s) |
| `--motion-thr` | 0.05 | accel threshold (g) |
| `--gyro-thr` | 15 | gyro threshold (deg/s) |
| `--calib-s` | 3 | gyro calibration time (s) |
| `--subject`, `--activity` | S01, rest | labels stored in the CSV |
| `--no-log`, `--simulate`, `--list-ports`, `--data-dir` | | as in the ECG viewer |

---

## 9. Stage 3: PPG (MAX30102)

### Principle

A red LED and an infrared LED shine into the fingertip, a photodiode measures the light that comes back.
Blood absorbs light, so with every heartbeat (more blood in the tissue) **less** light returns: the raw counts
dip at each systole. The viewer plots the **inverted** filtered IR so the pulse points up.
The ratio of the pulsatile (AC) to constant (DC) part of red vs. infrared, `R`, relates to oxygen saturation.

### Firmware (`src/ppg/main.cpp`)

- I2C address 0x57, `PART_ID` register 0xFF must read 0x15. SpO2 mode (RED + IR), 18-bit.
- 400 samples/s with 4x on-chip averaging = **100 Hz** output, 411 us pulse width, ADC range 4096 nA,
  LED current 0x24 = 7.2 mA (value x 0.2 mA).
- The sensor samples by itself into a 32-sample FIFO. The ESP32 polls the FIFO pointers every 3 ms in an
  `esp_timer` task, reads up to 16 samples per burst and pushes them into a queue.
- Sample times are **reconstructed** (newest = poll time, older samples 10 ms apart), accurate to about +-3 ms.
- Output line: `timestamp_us,red,ir`. A `# PPG_META fs_hz=100 adc_bits=18 led_red_ma=.. led_ir_ma=..
  adc_range_na=.. part_id=.. i2c_errors=.. fifo_ovf=.. dropped=..` line repeats every 2 s.
  `# ERROR no MAX30102 answering at 0x57...` means wiring.

### Run

```bash
cd desktop && source .venv/bin/activate
python test_ppg_processing.py
python ppg_viewer.py --simulate
python ppg_viewer.py --port /dev/ttyUSB0 --subject S01 --activity rest
```

### Finger placement (decides the signal quality)

- Cover the sensor window with the fingertip pad, resting on it. Light, steady pressure: pressing hard
  squeezes the blood out and flattens the pulse, resting too lightly lets ambient light leak in.
- Keep the hand still, at heart level. A warm finger gives a stronger pulse. Avoid bright light on the sensor.
- Wait about 4 s after placing the finger: the label shows `SETTLING`, then `OK`.
- Without a finger, IR is only a few thousand counts. With a finger it is typically tens of thousands up to
  about 200k.

### What you see

- **BPM**: median of the last few pulse intervals, shown as `NN?` in grey when the signal is `NOISY` / `LOW SIGNAL`.
- **Signal**: `NO DATA`, `NO FINGER`, `SETTLING`, `OK`, `LOW SIGNAL` (perfusion index below 0.1 %),
  `NOISY` (irregular pulse intervals), `SATURATED` (counts at full scale), plus a hint line.
- **PI** (perfusion index, about peak-to-peak / DC of IR, %), **R**, and **experimental SpO2** (`110 - 25 R`,
  shown only when the signal is `OK`).
- Top plot: pulse waveform (`-(causal 0.5-8 Hz band-pass of IR)`), red dots = detected systolic peaks.
  The dots sit at the true pulse time (zero-phase detection), the drawn trace is causally filtered and lags
  by a few tens of ms.
- Middle / bottom plot: raw IR and raw RED. The dashed line on the IR plot is the finger threshold.
- Simulator buttons: **Simulate finger off**, **Simulate motion**.

### Processing

Finger detection with hysteresis (on at the threshold, off at 80 % of it) -> causal 0.5-8 Hz band-pass for display
-> zero-phase band-pass on an 8 s window for peak detection (`find_peaks`, prominence rule, 0.33 s minimum
spacing) -> inter-beat intervals -> BPM. R uses its own zero-phase 0.7-4 Hz band so breathing drift
does not bias it.

### Firmware settings (constants at the top of `src/ppg/main.cpp`)

| Setting | Default | Notes |
|---|---|---|
| `SAMPLE_RATE_SPS` / `AVERAGING` | 400 / 4 | output rate = 400 / 4 = 100 Hz |
| `PULSE_WIDTH_CODE` | 3 (411 us, 18 bit) | 18 bit allows at most 400 sps |
| `ADC_RANGE_CODE` | 1 (4096 nA) | higher range = fewer saturations, coarser steps |
| `LED_RED_PA`, `LED_IR_PA` | 0x24 (7.2 mA) | raise if the pulse is weak, lower if `SATURATED` |

### Options (`ppg_viewer.py`)

| Option | Default | Meaning |
|---|---|---|
| `--port`, `--baud` | auto, 230400 | serial port and baud |
| `--fs` | 100 | sampling rate (firmware `PPG_META` overrides) |
| `--window` | 8 | seconds shown |
| `--finger-thr` | 30000 | IR counts that mean a finger is present |
| `--subject`, `--activity` | S01, rest | labels stored in the CSV |
| `--no-log`, `--simulate`, `--list-ports`, `--data-dir` | | as in the other viewers |

---

## 10. Stage 4: ECG + IMU + PPG together (fusion)

One ESP32 firmware reads all three sensors and the fusion viewer shows them on **one time axis**. Because
ECG, PPG and IMU samples carry the same ESP32 clock, the delay between an R peak and the pulse that follows
it (pulse arrival time, PAT) can be measured, and the IMU can say which beats are not trustworthy.

### Wiring

The combined wiring table in section 3 is all you need. MPU6050 and MAX30102 share SDA (GPIO21) / SCL (GPIO22),
the AD8232 uses GPIO34/32/33. Wire everything with USB unplugged. Check the safety notes in section 2 first.

### Flash and run

1. VS Code status bar: select `env:all`, **Build**, **Upload** (terminal alternative:
   `~/.platformio/penv/bin/pio run -e all -t upload`).
2. Serial Monitor at **460800** baud. Expected:
   ```
   # ALL_LOGGER v1
   E,147276072,1936,0,0
   I,147276080,120,-45,8210,15,-3,7
   P,147276081,81234,102345
   # ECG_META fs_hz=250 adc_bits=12 dropped=0
   # IMU_META fs_hz=100 accel_range_g=4 gyro_range_dps=500 who_am_i=0x68 i2c_errors=0 dropped=0
   # PPG_META fs_hz=100 adc_bits=18 ... part_id=0x15 ...
   # ALL_META baud=460800 queue_len=768 queue_max=..
   ```
   `# ERROR no MPU6050 ...` or `# ERROR no MAX30102 ...` repeats every 2 s if a sensor is missing. The other
   sensors keep streaming in that case.
3. **Close the Serial Monitor**, then:

```bash
cd desktop
source .venv/bin/activate
python test_fusion_processing.py                   # ALL TESTS PASSED
python fusion_viewer.py --simulate                 # no hardware
python fusion_viewer.py --port /dev/ttyUSB0 --subject S01 --activity rest
```

Procedure for a first real session: electrodes on (laptop on battery), fingertip on the MAX30102, MPU6050
fixed on the same hand or forearm, sit still. The IMU needs about 3 s of stillness for its gyro calibration, the
ECG and PPG labels need a few seconds to settle. Then breathe normally and watch PAT.

### What you see

- **Two big numbers**: ECG BPM (red) and PPG BPM (blue, with `?` when the PPG is noisy).
- **Status line**: `ECG <q> | PPG <q> | IMU <q>`, each in its own colour.
- **Heart rate check**: ECG and PPG rates compared (they must agree within 5 BPM while the PPG is clean). A
  disagreement means one of the two sensors is probably wrong right now.
- **PAT line**: median and spread of the last 30 beats, and how many of them were valid.
- **Plot 1**: filtered ECG with R peaks (yellow dots).
- **Plot 2**: PPG pulse with systolic peaks (red dots) and the R peaks as small yellow triangles at the bottom:
  the horizontal distance from a triangle to the next red dot is the PAT.
- **Plot 3**: IMU motion index with its threshold line.
- **Red shaded bands** on plots 1-3: the IMU says motion.
- **Plot 4**: PAT per beat for the last 60 s. Green = valid beat (R to peak), blue diamonds = R to steepest
  upstroke of the same beat, grey = beat rejected. The dotted line is the median.
- Buttons: **Pause display**, **Recalibrate gyro**. With `--simulate` also **Simulate motion / finger off /
  leads off** to see how the flags react.

### How a beat is judged

For every ECG R peak the viewer looks for the PPG systolic peak 100-600 ms later (each pulse is used once). The
beat is **valid** only if all of this holds:

| Check | Meaning when it fails (reason in the CSV) |
|---|---|
| ECG ok | `ecg`: leads off near the R peak, or ECG label not `OK` |
| PPG ok | `ppg`: no pulse found in 100-600 ms, no finger, or PPG label not `OK` (noisy, low signal, saturated) |
| IMU ok | `imu`: gyro not calibrated yet or no IMU samples covering the beat |
| no motion | `motion`: IMU motion flag anywhere from 0.5 s before the R peak to 0.5 s after the pulse |

Two PAT numbers are reported, both from the zero-phase filtered PPG with sub-sample (parabolic) interpolation:
`pat_peak_ms` (R peak to systolic peak) and `pat_slope_ms` (R peak to the steepest upstroke of the pulse).
PAT is R peak to pulse, so it contains the pre-ejection period of the heart. It is **not** pure pulse transit time.

### Options (`fusion_viewer.py`)

| Option | Default | Meaning |
|---|---|---|
| `--port` | auto | serial port |
| `--baud` | 460800 | must match the firmware (`SERIAL_BAUD` in `src/all/main.cpp`) |
| `--ecg-fs` / `--imu-fs` / `--ppg-fs` | 250 / 100 / 100 | nominal rates (IMU and PPG take the firmware's `*_META`) |
| `--accel-range` / `--gyro-range` | 4 / 500 | firmware `IMU_META` overrides |
| `--window` | 8 | seconds shown on the signal plots |
| `--pat-window` | 60 | seconds shown on the PAT trend |
| `--notch` | 50 | mains notch in Hz, `0` disables |
| `--finger-thr` | 30000 | IR counts that mean a finger is present |
| `--motion-thr` / `--gyro-thr` | 0.05 g / 15 deg/s | IMU motion rule |
| `--subject`, `--activity` | S01, rest | labels stored in the CSVs |
| `--no-log`, `--simulate`, `--list-ports`, `--data-dir` | | as in the other viewers |

### Firmware design (`src/all/main.cpp`)

- ECG: an `esp_timer` callback every 4 ms reads the ADC. It does **no I2C**, so a slow I2C transfer cannot
  delay an ECG sample.
- PPG and IMU: one FreeRTOS task (pinned to core 1, 3 ms period) polls the MAX30102 FIFO and reads the MPU6050
  every 10 ms. It is the only code that uses the I2C bus after setup.
- The MPU6050 runs at 1 kHz internally (DLPF on) and the latest value is read every 10 ms, so IMU timestamps are
  accurate to under 1 ms. PPG timestamps are reconstructed from the FIFO read time (about +-3 ms).
- All samples go into one FreeRTOS queue (768 entries). `loop()` prints them as tagged lines. A sensor that fails
  to initialise is retried every 2 s without stopping the others.
- Data rate is about 13 kB/s, which is why the baud rate is 460800 (230400 would be about 55 % busy). If your
  USB-serial adapter has problems at 460800, set `SERIAL_BAUD` to 230400 in the firmware and use `--baud 230400`.

## 11. Serial data formats

All streams: 230400 baud, one CSV line per sample, lines starting with `#` are comments / metadata.
`timestamp_us` is `esp_timer_get_time()`, microseconds since boot. Because every firmware uses the same
clock type, timestamps can be aligned across sensors once they run together.

| Sensor | Rate | Line |
|---|---|---|
| ECG | 250 Hz | `timestamp_us,adc,lo_plus,lo_minus` |
| IMU | 100 Hz | `timestamp_us,ax,ay,az,gx,gy,gz` (raw int16) |
| PPG | 100 Hz | `timestamp_us,red,ir` (18-bit counts, 0..262143) |

The combined firmware (`env:all`) sends the same fields with a one-letter tag in front and 460800 baud:

| Tag | Line |
|---|---|
| `E` | `E,timestamp_us,adc,lo_plus,lo_minus` (250 Hz) |
| `I` | `I,timestamp_us,ax,ay,az,gx,gy,gz` (100 Hz) |
| `P` | `P,timestamp_us,red,ir` (100 Hz) |

Lines of different tags are not strictly ordered in time (PPG samples are stamped with their reconstructed time),
each stream is checked for going forward in time on its own.

The viewers drop and count lines that are malformed, out of range or go backwards in time
(`bad lines` in the status line).

---

## 12. Recorded data

Each viewer writes CSV files to `data/` (git-ignored). Header lines starting with `#` store subject, activity,
source, rates and settings. Load with pandas:

```python
import pandas as pd
df = pd.read_csv("data/ppg_session_20261030_1200.csv", comment="#")
df = df[df["quality"] == "OK"]          # filter by quality before analysis
```

| File | Columns |
|---|---|
| `session_*.csv` (ECG) | `timestamp_us, raw_adc, filtered, bpm, leads_off` |
| `session_*_beats.csv` | `r_peak_timestamp_us, rr_s, bpm_instant` |
| `imu_session_*.csv` | `timestamp_us, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, a_mag_g, accel_idx_g, gyro_idx_dps, motion, calibrated` |
| `ppg_session_*.csv` | `timestamp_us, red, ir, ppg_filt, bpm, pi_pct, r_ratio, spo2_est_experimental, finger, quality` |
| `ppg_session_*_beats.csv` | `peak_timestamp_us, ibi_s, bpm_instant, amp_counts` |
| `fusion_*_ecg.csv`, `_imu.csv`, `_ppg.csv` | same columns as the single-sensor files (one session, same clock) |
| `fusion_*_beats.csv` | `r_timestamp_us, ppg_peak_timestamp_us, pat_peak_ms, pat_slope_ms, ecg_bpm_inst, ppg_bpm_inst, ecg_ok, ppg_ok, imu_ok, motion, valid, reason` |

Use non-identifying subject codes (`S01`, ...). For offline analysis prefer zero-phase filtering
(`processing.filter_offline`) over the causal live filters.

---

## 13. Design notes (how it works)

**Firmware pattern (all three).** An `esp_timer` periodic callback (running in the esp_timer task) samples the
sensor, stamps the sample with `esp_timer_get_time()` and pushes it into a FreeRTOS queue. `loop()` drains the
queue and prints. `Serial.print` is never called inside the timer callback, so sampling timing does not depend on
serial speed. All I2C access stays in that one task, so the shared I2C bus is never used from two places at once.

**Viewer pattern.** A reader thread parses serial lines into a deque. The GUI thread (PySide6 + pyqtgraph) feeds
samples to a pipeline object (`ECGPipeline`, IMU pipeline, `PPGPipeline`), which is plain Python with no GUI
code, so it can be unit-tested with synthetic signals and re-used for offline analysis and later fusion.

**Live vs offline filters.** Live display uses causal filters with steady-state initialisation. Peak detection
uses zero-phase filtering on a window so peak times are not delayed.

---

## 14. Troubleshooting

### Setup and environment

| Symptom | Likely cause / fix |
|---|---|
| `can't open file 'viewer.py'` / `No such file` | You are in the project root. `cd desktop` first. |
| `python: command not found` | Virtual environment not active. `source .venv/bin/activate`, or use `python3`. |
| `ModuleNotFoundError: No module named 'processing'` | `processing.py` / `sources.py` missing from `desktop/`. Keep all viewer files together. |
| `venv` creation fails | `sudo apt install python3-venv` |
| `Permission denied: /dev/ttyUSB0` | `dialout` group missing or no logout/login yet. Check `groups`. |
| Port appears then vanishes | `sudo apt remove brltty`, replug the ESP32. |
| No `/dev/ttyUSB*` | Charge-only cable; check `dmesg \| tail`. |
| Port busy | Serial Monitor or another viewer has the port. Close it. |
| Qt `xcb` plugin error | `sudo apt install libxcb-cursor0` |
| Crash: `GLib-GIO-ERROR ... 'org.gnome.settings-daemon.plugins.xsettings' does not contain a key named 'antialiasing'` | Qt/GNOME settings mismatch. The viewers force `QT_QPA_PLATFORMTHEME=generic` automatically. Make sure your `viewer.py` is the current file (`grep -n QT_QPA viewer.py` must print a line). Also try `QT_QPA_PLATFORM=xcb python viewer.py ...`. |
| `TypeError: '_thread._ThreadHandle' object is not callable` (Python 3.13) | Old `sources.py` with a method called `_handle`. Current version uses `parse_line`. |

### PlatformIO

| Symptom | Likely cause / fix |
|---|---|
| `Nothing to build` on `env:ecg` | `src/ecg/main.cpp` missing. `mkdir -p src/ecg && mv src/main.cpp src/ecg/main.cpp`. |
| `firmware.elf Error 1` | Look at the first `error:` line higher up in the build log, not the last lines. |
| `pio: command not found` in terminal | Use `~/.platformio/penv/bin/pio ...` (PlatformIO's own environment). |
| `Cannot read properties of undefined (reading 'id')` in a VS Code popup | PlatformIO extension bug after `platformio.ini` changed. Upload from the terminal with `~/.platformio/penv/bin/pio run -e <env> -t upload`, or `Ctrl+Shift+P` -> **Developer: Reload Window**, then Project Tasks -> Refresh. |
| `connecting...` forever / upload hangs | Hold the BOOT button during `Connecting....`. |
| Viewer shows `samples 0` and thousands of `bad lines` | **The board runs a different firmware than the viewer expects** (for example ECG firmware with the PPG viewer). Use the raw-read check in section 5, then flash the right env. |
| `PlatformIO project opens very slowly` | First start downloads toolchains (10-30 min). Wait. |

### ECG

| Symptom | Likely cause / fix |
|---|---|
| Raw trace stuck at 4095 / `LEADS OFF` | Electrode not touching, wrong pad, cable not fully inserted. |
| Raw trace stuck at 0 | OUTPUT not on GPIO34, or GND missing. |
| Strong 50 Hz hum | Unplug laptop charger, keep phone/charger away, fresh electrodes, stay still. |
| Wandering baseline | Wait 10-20 s, clean skin, do not tug cables. |
| Wrong BPM | Check that the R wave is the largest spike, reposition electrodes, stay still. |
| Occasional garbled line (`L)R...`) in a raw dump | Timing mismatch of the serial read. Counted as a bad line and dropped, harmless. |

### IMU

| Symptom | Likely cause / fix |
|---|---|
| `# ERROR no MPU6050 answering at 0x68` | VCC/GND/SDA/SCL wrong or loose, SDA/SCL swapped, AD0 floating (connect to GND). |
| All values 0 | Chip still asleep (the firmware wakes it), so check power / wiring. |
| `WARNING who_am_i=0x70` | Probably an MPU6500 clone. Usually still works. |
| `\|a\|` is 2 g instead of 1 g | Range mismatch. The viewer takes ranges from `IMU_META`. |
| `SATURATED` | Movement beyond the range. Raise `ACCEL_FS_SEL` / `GYRO_FS_SEL` in `src/imu/main.cpp`. |
| Gyro not 0 when still | Calibration done while moving. Press **Recalibrate gyro** while still. |

### PPG

| Symptom | Likely cause / fix |
|---|---|
| `# ERROR no MAX30102 answering at 0x57` | VIN/GND/SDA/SCL wrong or loose. Some cheap breakouts have odd pull-ups: try 4.7 k pull-ups to 3V3. |
| `WARNING part_id=0x..` | Not the genuine ID 0x15. May still work. |
| `NO FINGER` although a finger is on it | IR below `--finger-thr`. Press a bit more, raise `LED_IR_PA`, or lower `--finger-thr`. |
| `SATURATED` | LEDs too bright or ambient light. Lower `LED_*_PA` / raise `ADC_RANGE_CODE`, shade the sensor. |
| `LOW SIGNAL` | Weak pulse: adjust pressure/position, warm the finger, raise LED current. |
| `NOISY` / BPM with `?` | Moving finger or poor contact. Hold still. |
| BPM about double or half | Dicrotic notch detected or beats missed. Check the pulse plot, improve contact. |

### Fusion (env:all)

| Symptom | Likely cause / fix |
|---|---|
| Viewer shows `samples 0`, many `bad lines` | Board runs another firmware (single-sensor ones have no `E,`/`I,`/`P,` tags), or baud mismatch (460800 here). Use the raw-read check in section 5 with baud 460800. |
| `# ERROR no MPU6050` / `no MAX30102` every 2 s | That sensor's wiring. The other streams still work, the fusion viewer marks beats `imu` / `ppg` invalid. |
| ECG fine, IMU or PPG `NO DATA` | Same as above, or I2C wires too long. Check `i2c errors` in the status line. |
| `dropped ~N` or `fw queue max` close to 768 | Serial too slow or USB hub problem. Use a direct USB port, or check the baud setting. |
| All beats `invalid`, reason `imu` | Keep the IMU still for 3 s so the gyro calibration finishes. Press **Recalibrate gyro** if needed. |
| All beats `invalid`, reason `ppg` | No finger, weak or noisy pulse. Fix the finger placement (section 9). |
| PAT jumps around by tens of ms | Motion or loose finger. Use only `valid == 1` beats. PAT also changes with breathing and blood pressure, a few tens of ms is normal. |
| No R-peak triangles on the PPG plot | ECG leads off or not settled yet. |

### All streams

| Symptom | Likely cause / fix |
|---|---|
| `i2c errors` grows | Long/loose wires, noisy bus. Shorten wires. |
| `dropped ~N` keeps growing | Baud too low or USB hub problem. Keep 230400, use a direct USB port. |

---

## 15. Known limitations

- **SpO2 is not calibrated and not validated.** It is the textbook formula `110 - 25 R`. The project proposal says
  SpO2 should only be reported after proper validation. Never use it for a health decision.
- **PPG alone cannot reliably separate motion artifacts from real pulses.** A motion burst can still show `OK`
  with a wrong BPM. The IMU motion flag is the tool for this, so use the `quality` column and the IMU flag together.
- **PPG sample times are reconstructed (+-3 ms).** Fine for BPM. For ECG-to-PPG pulse arrival time (PAT),
  use the MAX30102 interrupt pin for exact data-ready times in a later stage.
- Default thresholds (PPG finger 30000 counts, IMU motion 0.05 g / 15 dps, LED current) were chosen with
  synthetic data. Tune them on real recordings.
- The ESP32 ADC is non-linear. Absolute ECG voltage is not trustworthy, waveform shape and R-peak timing are what matter.
- Live filters are causal and distort the waveform slightly. Use zero-phase filtering offline.
- The R-peak and pulse-peak detectors are simple. Expect errors with motion, muscle noise or poor contact.
- Short-window HRV is unstable, indicative only. Single-lead ECG is not a clinical 12-lead ECG.
- Signal-quality labels are simple heuristics, not a validated signal-quality index.
- The PlatformIO toolchain could not be downloaded in the authoring sandbox, so firmware was syntax-checked
  against stub headers there. The ECG, IMU and PPG firmware have since been built and flashed on a real board by the
  user. Unit tests, simulators, GUI and CSV output were verified with synthetic signals.
- One board runs one firmware at a time. Use `env:all` to see all three sensors together.
- **The combined firmware and fusion viewer have only been tested with synthetic data** (including a virtual serial
  port) and a stub-header syntax check. The first run on the real board may need small fixes.
- PAT is R peak to pulse (it includes pre-ejection time). With the reconstructed PPG timestamps (+-3 ms), a constant
  PPG-sensor delay of a few ms and 10 ms sample spacing (interpolated), absolute PAT is only trustworthy to a few ms.
  Relative changes over time are more reliable than the absolute value. The firmware does not use the MAX30102
  interrupt pin yet.
- Beat validity flags use the **live** quality labels at the time the beat is resolved (about 1 s after the R peak), not
  a per-beat quality index. Treat `valid` as a conservative filter, not a validated signal-quality index.
- The IMU measures motion at the sensor, not at the chest or fingertip. Mount it on the same hand/forearm as the finger sensor.

---

## 16. Roadmap

Done:

1. ECG: AD8232 acquisition, R-peak detection, BPM, HRV, viewer, CSV logging.
2. IMU: MPU6050 acquisition, calibration, motion flag, viewer, CSV logging.
3. PPG: MAX30102 acquisition, pulse-peak detection, BPM, perfusion index, R, viewer, CSV logging.

4. Fusion (built, tested in simulation, to be verified on the real board): combined firmware, one clock, fusion
   viewer, pulse arrival time (R peak to pulse), motion-aware beat validity, four-file CSV logging.

Next:

5. Verify `env:all` on the real board; tune thresholds on real recordings (finger, motion, PAT range).
6. Interrupt-based PPG timestamps (MAX30102 INT pin) for tighter PAT, and a per-beat signal-quality index.
7. Feature extraction: R-R, SDNN, RMSSD, pulse rate, pulse intervals, waveform morphology, motion intensity, PAT.
8. Collect labelled data (rest / activity, several subjects) with non-identifying codes.
9. scikit-learn comparison: Random Forest, SVM, KNN, One-Class SVM (precision, recall, F1, confusion matrices,
   false-alert control).
10. TinyML on the ESP32: model size, RAM, quantization, latency, power, local alerts and quality warnings.