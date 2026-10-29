"""
ppg_sources.py
==============
Data sources for ppg_viewer.py (same thread + deque pattern as sources.py).

Sample tuple: (timestamp_us, red, ir)    raw 18-bit counts

PPGSerialSource : real ESP32 + MAX30102 over USB serial
PPGSimSource    : synthetic reflective PPG (finger on / off, motion), no hardware needed
"""
from __future__ import annotations

import re
import time

import numpy as np

from sources import BaseSource

_KV = re.compile(r"(\w+)=(\S+)")
ADC_MAX = 262143


class PPGSerialSource(BaseSource):
    def __init__(self, port, baud=230400):
        super().__init__()
        self.port = port
        self.baud = baud
        self.description = f"{port} @ {baud}"
        self.meta = {}                 # filled from '# PPG_META key=value ...' lines
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
            if "PPG_META" in text:
                self.meta = dict(_KV.findall(text))
            return
        parts = text.split(",")
        if len(parts) != 3:
            self.bad_lines += 1
            return
        try:
            t, red, ir = (int(p) for p in parts)
        except ValueError:
            self.bad_lines += 1
            return
        if not (0 <= red <= ADC_MAX) or not (0 <= ir <= ADC_MAX):
            self.bad_lines += 1
            return
        if self._last_t is not None and t <= self._last_t:   # corrupted / truncated timestamp
            self.bad_lines += 1
            return
        self._last_t = t
        self.samples.append((t, red, ir))


# ----------------------------------------------------------------------
# Synthetic PPG
# ----------------------------------------------------------------------
class SyntheticPPG:
    """Reflective PPG as raw MAX30102 counts.

    Raw counts DIP at systole (more blood absorbs more light), so
        raw = DC - AC * pulse(t)
    pulse = systolic Gaussian + smaller diastolic bump (dicrotic wave), per beat.

    The red/IR AC-DC ratio is set by r_ratio (R of the ratio-of-ratios), so the
    experimental SpO2 formula can be tested against a known value.

    Switches: .finger_off (counts fall to a no-finger level), .motion (large slow
    artifacts on top of the pulse).
    """
    # (offset from beat start in s, relative amplitude, width in s)
    WAVES = [(0.16, 1.00, 0.055),    # systolic peak
             (0.42, 0.32, 0.090)]    # diastolic / reflected wave

    def __init__(self, fs=100.0, bpm=70.0, hrv_ms=20.0, dc_ir=100000.0, dc_red=80000.0,
                 ac_ir_frac=0.012, r_ratio=0.5, wander_frac=0.015, noise=25.0, seed=None):
        self.fs = float(fs)
        self.bpm = float(bpm)
        self.hrv_s = hrv_ms / 1000.0
        self.dc = {"ir": dc_ir, "red": dc_red}
        self.ac_frac = {"ir": ac_ir_frac, "red": ac_ir_frac * r_ratio}
        self.wander_frac = wander_frac
        self.noise = noise
        self.finger_off = False
        self.motion = False
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.beats = [0.4]            # beat START times (s); also used by the unit tests
        self._motion_env = 0.0

    def systolic_peak_times(self):
        """True systolic peak times (s) of the beats generated so far."""
        return np.asarray(self.beats) + self.WAVES[0][0]

    def _extend(self, t_end):
        while self.beats[-1] < t_end + 1.5:
            rr = 60.0 / self.bpm + self.rng.normal(0.0, self.hrv_s)
            self.beats.append(self.beats[-1] + max(0.4, rr))

    def read(self, k):
        t = (self.n + np.arange(k)) / self.fs
        self._extend(t[-1])
        b = np.asarray(self.beats)
        near = b[(b > t[0] - 1.5) & (b < t[-1] + 1.5)]
        pulse = np.zeros(k)
        for tb in near:
            for off, amp, w in self.WAVES:
                pulse += amp * np.exp(-0.5 * ((t - tb - off) / w) ** 2)

        target = 1.0 if self.motion else 0.0
        env = target + (self._motion_env - target) * np.exp(-(np.arange(1, k + 1) / self.fs) / 0.3)
        self._motion_env = float(env[-1])

        out = {}
        for ch in ("red", "ir"):
            dc = self.dc[ch]
            raw = dc - dc * self.ac_frac[ch] * pulse
            raw += dc * self.wander_frac * np.sin(2 * np.pi * 0.25 * t)          # breathing-like drift
            raw += env * dc * 0.10 * (np.sin(2 * np.pi * 1.1 * t) + 0.6 * np.sin(2 * np.pi * 2.7 * t + 1.0))
            raw += self.rng.normal(0.0, self.noise, k)
            if self.finger_off:
                raw = 2500.0 + self.rng.normal(0.0, 30.0, k)                      # ambient light only
            out[ch] = np.clip(np.rint(raw), 0, ADC_MAX).astype(int)
        self.n += k
        return out["red"], out["ir"]


class PPGSimSource(BaseSource):
    """Real-time paced synthetic source. Set .finger_off / .motion to change the scene."""

    def __init__(self, fs=100.0, bpm=70.0, r_ratio=0.5):
        super().__init__()
        self.fs = float(fs)
        self.gen = SyntheticPPG(fs=fs, bpm=bpm, r_ratio=r_ratio)
        self.description = f"SIMULATOR (synthetic MAX30102: {bpm:.0f} bpm, R = {r_ratio})"
        self.meta = {"fs_hz": str(int(fs)), "adc_bits": "18", "led_red_ma": "7.2",
                     "led_ir_ma": "7.2", "adc_range_na": "4096", "part_id": "0x15"}

    @property
    def finger_off(self):
        return self.gen.finger_off

    @finger_off.setter
    def finger_off(self, on):
        self.gen.finger_off = bool(on)

    @property
    def motion(self):
        return self.gen.motion

    @motion.setter
    def motion(self, on):
        self.gen.motion = bool(on)

    def run(self):
        self.connected = True
        t0 = time.perf_counter()
        n = 0
        origin_us = 5_000_000
        while not self._stop_evt.is_set():
            target = int((time.perf_counter() - t0) * self.fs)
            if target > n:
                k = target - n
                red, ir = self.gen.read(k)
                for i in range(k):
                    t_us = origin_us + int((n + i) * 1e6 / self.fs)
                    self.samples.append((t_us, int(red[i]), int(ir[i])))
                n = target
            time.sleep(0.005)
        self.connected = False