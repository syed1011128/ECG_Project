# Live ECG Viewer: ESP32 + AD8232 -> Desktop (Ubuntu + VS Code + PlatformIO)

Streams single-lead ECG from an AD8232 through an ESP32 over USB serial to a
Python desktop viewer: live trace, R-peak markers, BPM, short-window HRV,
signal-quality label, leads-off warning and CSV logging.

> Research / learning prototype. **Not a medical device.** Do not use it for
> diagnosis or treatment decisions.

```
Electrodes -> AD8232 -> ESP32 ADC (250 Hz, timer-driven) -> USB serial (230400 baud)
           -> Python viewer (filter, R-peaks, BPM, plot, CSV)
```

This project is the first stage of a larger fused PPG + ECG + IMU
cardiovascular monitoring research project (MAX30102 + AD8232 + MPU6050 + ESP32).

---

## 1. Project structure

```
ECG_Project/
├── platformio.ini            PlatformIO config (ESP32 board, serial speed)
├── README.md
├── .gitignore
│
├── src/
│   └── main.cpp              ESP32 firmware (250 Hz sampling -> serial CSV)
├── include/  lib/  test/     created by PlatformIO, left empty
├── .pio/                     PlatformIO build output (auto-generated)
│
├── desktop/                  Python viewer
│   ├── viewer.py             live GUI (run this)
│   ├── processing.py         filters, R-peak detection, BPM, HRV, quality
│   ├── sources.py            serial reader + built-in ECG simulator
│   ├── test_processing.py    tests with synthetic ECG (no hardware)
│   ├── requirements.txt
│   └── .venv/                your virtual environment (not committed)
│
├── data/                     recorded sessions (CSV, created automatically)
└── .vscode/                  optional launch.json for F5 run
```

---

## 2. Safety

- The electrodes touch your body. **Run the laptop on battery with the charger
  unplugged** (or use a USB isolator) while electrodes are attached.
- Power the AD8232 from **3.3 V only**. Never connect it to 5 V.
- Stop immediately if you feel discomfort or skin irritation.

---

## 3. Hardware

### Wiring

| AD8232 | ESP32 |
|---|---|
| 3.3V | 3V3 |
| GND | GND |
| OUTPUT | GPIO34 (ADC1, input only) |
| LO+ | GPIO32 |
| LO- | GPIO33 |
| SDN | not connected |

### Electrodes (disposable Ag/AgCl gel pads, clean dry skin)

| Lead | Position |
|---|---|
| RA | right chest, below the collarbone |
| LA | left chest, below the collarbone |
| RL | right lower abdomen |

Check the labels on your cable. Colours differ between manufacturers.
Sit still, do not talk, and wait 10-20 s after attaching.

---

## 4. One-time system setup (Ubuntu)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libxcb-cursor0

# allow your user to open serial ports, then LOG OUT AND LOG IN (or reboot)
sudo usermod -aG dialout $USER

# brltty (braille service) steals CH340/CP210x ports on Ubuntu
sudo apt remove -y brltty
```

After logging back in, check:

```bash
groups                              # must include: dialout
ls /dev/ttyUSB* /dev/ttyACM*        # your ESP32 port, e.g. /dev/ttyUSB0
```

If no port appears: use a data cable (not charge-only) and check `dmesg | tail`.

### VS Code extensions

- **Python** (Microsoft)
- **PlatformIO IDE** (first start downloads toolchains, can take 10-30 minutes)
- Optional: Rainbow CSV

---

## 5. Flash the firmware (PlatformIO)

`platformio.ini`:

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 230400
upload_speed = 921600
```

1. Put the firmware in `src/main.cpp`.
2. Status bar: tick = Build, arrow = Upload, plug = Serial Monitor.
3. Build first, then Upload. If it hangs at `Connecting....`, hold the ESP32 **BOOT** button.
4. Open Serial Monitor (230400). You should see lines like:
   ```
   1234567,2048,0,0        # timestamp_us, adc, lo_plus, lo_minus
   ```
   A line starting with `# ECG_LOGGER v1` is the firmware banner.
5. **Close the Serial Monitor** (`Ctrl+C` in its terminal). Only one program can use the port.

If upload gives a permission error, install PlatformIO's udev rules:

```bash
curl -fsSL https://raw.githubusercontent.com/platformio/platformio-core/develop/platformio/assets/system/99-platformio-udev.rules | sudo tee /etc/udev/rules.d/99-platformio-udev.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 6. Python environment

```bash
cd desktop
python3 -m venv .venv
source .venv/bin/activate          # prompt shows (.venv)
pip install -r requirements.txt
python test_processing.py          # must end with: ALL TESTS PASSED
```

In VS Code: `Ctrl+Shift+P` -> **Python: Select Interpreter** -> pick the `.venv` one.

---

## 7. Run

Test the viewer without hardware first:

```bash
python viewer.py --simulate
```

A moving synthetic ECG at about 72 BPM means the desktop side works.
The **Simulate leads-off** button tests the warning.

With the real sensor (Serial Monitor closed, electrodes attached):

```bash
python viewer.py --list-ports
python viewer.py --port /dev/ttyUSB0 --subject S01 --activity rest
```

If `--port` is omitted the viewer tries to auto-detect a CP210x / CH340 port.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--port` | auto | serial port, e.g. `/dev/ttyUSB0` |
| `--baud` | 230400 | must match the firmware |
| `--fs` | 250 | sampling rate set in the firmware (Hz) |
| `--window` | 8 | seconds shown on screen |
| `--notch` | 50 | mains notch in Hz, `0` disables (Bangladesh = 50 Hz) |
| `--subject` | S01 | non-identifying subject code stored in the CSV |
| `--activity` | rest | condition label stored in the CSV |
| `--data-dir` | `../data` | where session CSVs go |
| `--no-log` | off | do not write CSV files |
| `--simulate` | off | built-in synthetic ECG |
| `--list-ports` | | list serial ports and exit |

### Optional: F5 in VS Code

`.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "ECG Viewer (real sensor)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/desktop/viewer.py",
      "args": ["--port", "/dev/ttyUSB0", "--subject", "S01", "--activity", "rest"],
      "cwd": "${workspaceFolder}/desktop",
      "console": "integratedTerminal"
    },
    {
      "name": "ECG Viewer (simulate)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/desktop/viewer.py",
      "args": ["--simulate"],
      "cwd": "${workspaceFolder}/desktop",
      "console": "integratedTerminal"
    }
  ]
}
```

The viewer opens its own window (Qt), not a panel inside VS Code. Check the taskbar if you cannot see it.

---

## 8. What the window shows

- **BPM** (top left): median of the last few R-R intervals.
- **Signal**: `OK`, `SETTLING` (first ~3 s and after leads-off), `LEADS OFF`, `SATURATED`, `FLAT`.
- **HRV**: SDNN, RMSSD, pNN50 from a short window. Indicative only.
- **Status line**: source, measured sampling rate, sample count, dropped-sample estimate, bad serial lines.
- **Middle plot**: filtered ECG (0.5-40 Hz band-pass + 50 Hz notch, causal) with red R-peak markers.
- **Bottom plot**: raw ADC (0-4095). Stuck at 0 or 4095 means wiring or contact problems.
- R-peak markers appear about 0.4 s after the beat. The detector waits until the beat is finished.

---

## 9. Recorded data (`data/`)

`session_YYYYMMDD_HHMM.csv`

```
# subject=S01
# activity=rest
# source=/dev/ttyUSB0 @ 230400
# fs_nominal_hz=250.0
# notch_hz=50.0
# started=2026-10-28T06:10:00
timestamp_us,raw_adc,filtered,bpm,leads_off
```

`session_YYYYMMDD_HHMM_beats.csv`: `r_peak_timestamp_us,rr_s,bpm_instant`

Load in pandas with `pd.read_csv(path, comment="#")`.
`timestamp_us` is the ESP32 clock (microseconds since boot), kept so PPG and IMU
streams can be aligned later. Use non-identifying subject codes. `data/*.csv` is git-ignored.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'processing'` | `processing.py` (or `sources.py`) is missing from `desktop/`. All of `viewer.py`, `processing.py`, `sources.py` must be in the same folder. |
| `python: command not found` | Virtual environment not activated. `source .venv/bin/activate`, or use `python3`. |
| `venv` creation fails | `sudo apt install python3-venv` |
| `Permission denied: /dev/ttyUSB0` | `dialout` group missing or you have not logged out/in yet. Check `groups`. |
| Port appears then vanishes | `sudo apt remove brltty`, replug the ESP32. |
| No `/dev/ttyUSB*` at all | Charge-only cable, or check `dmesg \| tail`. |
| Port busy | PlatformIO Serial Monitor or another program has the port. Close it. |
| `connecting...` forever / no samples | Wrong port or baud, firmware not flashed. Verify in Serial Monitor at 230400. |
| Qt `xcb` plugin error | `sudo apt install libxcb-cursor0` |
| Crash: `GLib-GIO-ERROR ... 'org.gnome.settings-daemon.plugins.xsettings' does not contain a key named 'antialiasing'` | Qt/GNOME settings mismatch, not a bug in this project. Try in order (until one works): `QT_QPA_PLATFORMTHEME= python viewer.py ...`, then `QT_QPA_PLATFORMTHEME=generic ...`, then `QT_QPA_PLATFORM=xcb ...` or `QT_QPA_PLATFORM=wayland ...`, then `pip install "PySide6==6.8.3"`. |
| Raw trace stuck at 4095 / `LEADS OFF` | Electrode not touching, wrong pad, cable not fully inserted. |
| Raw trace stuck at 0 | OUTPUT not on GPIO34, or GND missing. |
| Strong 50 Hz hum | Unplug laptop charger, keep phone/charger away, fresh electrodes, stay still. |
| Wandering baseline | Wait 10-20 s, clean skin, do not tug cables. |
| `dropped ~N` keeps growing | Baud too low or USB hub problem. Keep 230400, use a direct USB port. |
| Wrong BPM | Make sure the R wave is the largest spike; reposition electrodes; stay still. |

---

## 11. How it works (short)

**Firmware.** An `esp_timer` fires every 4 ms (250 Hz). The callback reads the
ADC (12-bit, 11 dB attenuation) and the two leads-off pins, stamps the sample
with `esp_timer_get_time()` and pushes it into a FreeRTOS queue. `loop()` drains
the queue and prints `timestamp_us,adc,lo_plus,lo_minus`. `Serial.print` never
runs inside the timer callback, so sampling timing does not depend on serial speed.

**Viewer.** A reader thread parses serial lines into a queue (bad or out-of-order
lines are counted and dropped). The GUI thread feeds samples to `ECGPipeline`:
causal band-pass + 50 Hz notch -> Pan-Tompkins-style R-peak detection (5-15 Hz band,
derivative, squaring, 100 ms integration, adaptive threshold, refractory period) ->
R-R intervals -> BPM and HRV. Leads-off resets the filter and hides BPM.

---

## 12. Known limitations

- The ESP32 ADC is non-linear. Absolute voltage is not trustworthy. Waveform shape and R-peak timing are what matter.
- Live filters are causal, so waveform shape is slightly distorted. Use `processing.filter_offline` (zero-phase) for offline analysis.
- The R-peak detector is simple. Expect errors with motion, muscle noise or poor contact.
- Short-window HRV is unstable. Treat it as indicative only.
- Single-lead ECG is not a clinical 12-lead ECG.
- The signal-quality label is a simple heuristic, not a validated SQI.
- Verified in development: unit tests with synthetic ECG, the GUI in simulator mode, and the serial reader through a virtual serial port. Real-hardware behaviour must be checked on your own setup.

---

## 13. Roadmap (project context)

1. Single-lead ECG viewer and logger (this repo)
2. MAX30102 PPG acquisition and pulse-peak detection
3. MPU6050 motion data, shared timestamps across sensors
4. Pulse arrival time (ECG R-peak to PPG pulse) with careful synchronization
5. Feature extraction, scikit-learn models, TinyML on the ESP32