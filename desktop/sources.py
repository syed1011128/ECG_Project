"""
sources.py
==========
Data sources for viewer.py. Each source runs in its own thread and pushes
parsed samples into a deque, so the GUI thread never blocks on I/O.

Sample tuple: (timestamp_us, adc, lo_plus, lo_minus)

SerialSource : the real ESP32 + AD8232 over USB serial
SimSource    : built-in synthetic ECG (no hardware needed, for testing the viewer)
"""
from __future__ import annotations

import collections
import threading
import time

import numpy as np


# ----------------------------------------------------------------------
# Serial port helpers
# ----------------------------------------------------------------------
def list_serial_ports():
    """Return [(device, description), ...] for all serial ports."""
    from serial.tools import list_ports
    return [(p.device, p.description) for p in list_ports.comports()]


def find_serial_port():
    """Pick the most likely ESP32 port (CP210x / CH340 / FTDI / generic USB serial)."""
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    prefer = ("cp210", "ch340", "ch910", "wch", "silicon labs", "ftdi",
              "usb serial", "usb-serial", "esp32", "uart")
    for p in ports:
        desc = f"{p.description} {p.manufacturer or ''}".lower()
        if any(k in desc for k in prefer):
            return p.device
    return ports[0].device if ports else None


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------
class BaseSource(threading.Thread):
    description = "source"

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = collections.deque()
        self.messages = collections.deque(maxlen=100)   # '#' lines from the device
        self.bad_lines = 0
        self.connected = False
        self.error = None
        self._stop_evt = threading.Event()   # NB: do not name this _stop (Thread uses it)

    def stop(self):
        self._stop_evt.set()


# ----------------------------------------------------------------------
# Real hardware
# ----------------------------------------------------------------------
class SerialSource(BaseSource):
    def __init__(self, port, baud=230400):
        super().__init__()
        self.port = port
        self.baud = baud
        self.description = f"{port} @ {baud}"
        self._last_t = None

    def run(self):
        try:
            import serial
            with serial.Serial(self.port, self.baud, timeout=0.2) as ser:
                time.sleep(1.5)              # opening the port resets most ESP32 boards
                ser.reset_input_buffer()
                self.connected = True
                while not self._stop_evt.is_set():
                    line = ser.readline()
                    if line:
                        self.parse_line(line)
        except Exception as exc:             # port busy, unplugged, permission, ...
            self.error = f"{type(exc).__name__}: {exc}"
        self.connected = False

    def parse_line(self, raw_line):
        # NB: do not call this _handle - threading.Thread has its own _handle in Python 3.13
        text = raw_line.decode("ascii", errors="ignore").strip()
        if not text:
            return
        if text[0] == "#":
            self.messages.append(text)
            return
        parts = text.split(",")
        if len(parts) != 4:
            self.bad_lines += 1
            return
        try:
            t, adc, lp, lm = (int(p) for p in parts)
        except ValueError:
            self.bad_lines += 1
            return
        if not (0 <= adc <= 4095) or lp not in (0, 1) or lm not in (0, 1):
            self.bad_lines += 1
            return
        if self._last_t is not None and t <= self._last_t:   # corrupted / truncated timestamp
            self.bad_lines += 1
            return
        self._last_t = t
        self.samples.append((t, adc, lp, lm))


# ----------------------------------------------------------------------
# Synthetic ECG
# ----------------------------------------------------------------------
class SyntheticECG:
    """Streaming synthetic ECG as ESP32-like ADC counts (12 bit).

    Each beat = sum of Gaussians for P, Q, R, S, T. Adds baseline wander,
    50 Hz mains hum and white noise, so the filters have something to remove.
    """
    # (offset from R in s, relative amplitude, width in s)
    WAVES = [(-0.18, 0.12, 0.025),   # P
             (-0.035, -0.10, 0.009),  # Q
             (0.0, 1.00, 0.011),      # R
             (0.03, -0.22, 0.010),    # S
             (0.27, 0.30, 0.045)]     # T

    def __init__(self, fs=250.0, bpm=72.0, hrv_ms=25.0, r_amp=600.0, base_adc=2048,
                 baseline_amp=80.0, mains_amp=30.0, mains_hz=50.0, noise=10.0, seed=None):
        self.fs = float(fs)
        self.bpm = float(bpm)
        self.hrv_s = hrv_ms / 1000.0
        self.r_amp = r_amp
        self.base_adc = base_adc
        self.baseline_amp = baseline_amp
        self.mains_amp = mains_amp
        self.mains_hz = mains_hz
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.beats = [0.6]        # true R-peak times (s); also used by the unit tests

    def _extend(self, t_end):
        while self.beats[-1] < t_end + 1.5:
            rr = 60.0 / self.bpm + self.rng.normal(0.0, self.hrv_s)
            self.beats.append(self.beats[-1] + max(0.35, rr))

    def read(self, k):
        t = (self.n + np.arange(k)) / self.fs
        self._extend(t[-1])
        b = np.asarray(self.beats)
        near = b[(b > t[0] - 1.0) & (b < t[-1] + 1.0)]
        sig = np.zeros(k)
        for tb in near:
            for off, amp, w in self.WAVES:
                sig += amp * np.exp(-0.5 * ((t - tb - off) / w) ** 2)
        sig *= self.r_amp
        sig += self.baseline_amp * np.sin(2 * np.pi * 0.2 * t)
        sig += self.mains_amp * np.sin(2 * np.pi * self.mains_hz * t)
        sig += self.rng.normal(0.0, self.noise, k)
        self.n += k
        return np.clip(np.rint(self.base_adc + sig), 0, 4095).astype(int)


class SimSource(BaseSource):
    """Real-time paced synthetic source. Set .leads_off = True to simulate unplugged electrodes."""

    def __init__(self, fs=250.0, bpm=72.0):
        super().__init__()
        self.fs = float(fs)
        self.gen = SyntheticECG(fs=fs, bpm=bpm)
        self.description = f"SIMULATOR ({bpm:.0f} bpm synthetic ECG)"
        self.leads_off = False

    def run(self):
        self.connected = True
        t0 = time.perf_counter()
        n = 0
        origin_us = 5_000_000
        while not self._stop_evt.is_set():
            target = int((time.perf_counter() - t0) * self.fs)
            if target > n:
                k = target - n
                adc = self.gen.read(k)
                lo = 1 if self.leads_off else 0
                for i in range(k):
                    a = 4095 if lo else int(adc[i])      # real AD8232 rails when leads are off
                    self.samples.append((origin_us + int((n + i) * 1e6 / self.fs), a, lo, lo))
                n = target
            time.sleep(0.005)
        self.connected = False