"""
fusion_sources.py
=================
Data sources for fusion_viewer.py: ECG + IMU + PPG on ONE serial stream / ONE clock.

Serial line format (src/all/main.cpp):
    E,timestamp_us,adc,lo_plus,lo_minus
    I,timestamp_us,ax,ay,az,gx,gy,gz
    P,timestamp_us,red,ir
    # ... comments / meta lines (ECG_META, IMU_META, PPG_META, ALL_META)

Both sources expose the same attributes, so the viewer does not care which one it has:
    ecg_samples  deque of (t_us, adc, lo_plus, lo_minus)
    imu_samples  deque of (t_us, ax, ay, az, gx, gy, gz)   raw int16
    ppg_samples  deque of (t_us, red, ir)                   raw 18-bit counts
    ecg_meta / imu_meta / ppg_meta / all_meta   dicts from the firmware's '# *_META' lines
    messages     deque of other '#' lines;   pop_messages() returns all of them
    total_bad_lines(), connected, error, description

FusionSerialSource reuses the three single-sensor parsers (same range checks, same
"time must go forward" rule), so a line is accepted exactly as in the single-sensor viewers.
FusionSimSource / SyntheticFusion: three synthetic sensors that share one heartbeat, with a
known ECG-to-PPG delay (PAT), so the whole chain can be tested without hardware.
"""
from __future__ import annotations

import collections
import re
import time

import numpy as np

from imu_sources import ImuSerialSource, SyntheticIMU
from ppg_sources import PPGSerialSource, SyntheticPPG
from sources import BaseSource, SerialSource, SyntheticECG

_KV = re.compile(r"(\w+)=(\S+)")


# ----------------------------------------------------------------------
# Real hardware
# ----------------------------------------------------------------------
class FusionSerialSource(BaseSource):
    def __init__(self, port, baud=460800):
        super().__init__()
        self.port = port
        self.baud = baud
        self.description = f"{port} @ {baud}"
        # parser objects only; their threads are never started
        self._ecg = SerialSource(port, baud)
        self._imu = ImuSerialSource(port, baud)
        self._ppg = PPGSerialSource(port, baud)
        self.ecg_meta = {}
        self.all_meta = {}
        self.bad_lines = 0                      # lines with an unknown tag / wrong shape

    # ---- views on the three sub-parsers ---------------------------------
    @property
    def ecg_samples(self):
        return self._ecg.samples

    @property
    def imu_samples(self):
        return self._imu.samples

    @property
    def ppg_samples(self):
        return self._ppg.samples

    @property
    def imu_meta(self):
        return self._imu.meta

    @property
    def ppg_meta(self):
        return self._ppg.meta

    def total_bad_lines(self):
        return self.bad_lines + self._ecg.bad_lines + self._imu.bad_lines + self._ppg.bad_lines

    def bad_lines_by_stream(self):
        return {"ecg": self._ecg.bad_lines, "imu": self._imu.bad_lines,
                "ppg": self._ppg.bad_lines, "other": self.bad_lines}

    def pop_messages(self):
        out = []
        for q in (self.messages, self._ecg.messages, self._imu.messages, self._ppg.messages):
            while q:
                out.append(q.popleft())
        return out

    # ---- thread ---------------------------------------------------------
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
        text = raw_line.strip()
        if not text:
            return
        if text[:1] == b"#":
            s = text.decode("ascii", errors="ignore")
            if "IMU_META" in s:
                self._imu.parse_line(text)
            elif "PPG_META" in s:
                self._ppg.parse_line(text)
            elif "ECG_META" in s:
                self.ecg_meta = dict(_KV.findall(s))
                self.messages.append(s)
            elif "ALL_META" in s:
                self.all_meta = dict(_KV.findall(s))
                self.messages.append(s)
            else:
                self.messages.append(s)
            return
        if len(text) > 2 and text[1:2] == b",":
            tag, body = text[:1], text[2:]
            if tag == b"E":
                self._ecg.parse_line(body)
                return
            if tag == b"I":
                self._imu.parse_line(body)
                return
            if tag == b"P":
                self._ppg.parse_line(body)
                return
        self.bad_lines += 1


# ----------------------------------------------------------------------
# Synthetic fusion: three sensors, one heart
# ----------------------------------------------------------------------
class SyntheticFusion:
    """ECG, IMU and PPG generators that share the same heartbeat.

    The PPG systolic peak is placed exactly `pat_s` after each ECG R peak (plus a small random
    jitter), so a correct PAT measurement must recover pat_s. Switches (all take effect at once):
        .motion      body movement: IMU shakes, PPG gets large artifacts, ECG gets muscle/baseline noise
        .finger_off  PPG sensor not covered
        .leads_off   ECG electrodes not attached (set by the caller as the leads-off flag)
    """

    def __init__(self, bpm=72.0, pat_s=0.25, ecg_fs=250.0, imu_fs=100.0, ppg_fs=100.0,
                 hrv_ms=20.0, pat_jitter_ms=4.0, seed=None):
        s = (lambda k: None if seed is None else seed + k)
        self.ecg_fs, self.imu_fs, self.ppg_fs = float(ecg_fs), float(imu_fs), float(ppg_fs)
        self.pat_s = float(pat_s)
        self.pat_jitter_s = pat_jitter_ms / 1000.0
        self.ecg = SyntheticECG(fs=ecg_fs, bpm=bpm, hrv_ms=hrv_ms, seed=s(0))
        self.ppg = SyntheticPPG(fs=ppg_fs, bpm=bpm, hrv_ms=hrv_ms, seed=s(1))
        self.imu = SyntheticIMU(fs=imu_fs, auto=False, seed=s(2))
        self._rng = np.random.default_rng(s(3))
        self._jitter = []
        self._motion = False
        self.leads_off = False

    # ---- switches --------------------------------------------------------
    @property
    def motion(self):
        return self._motion

    @motion.setter
    def motion(self, on):
        self._motion = bool(on)
        self.imu.force_motion = self._motion
        self.ppg.motion = self._motion

    @property
    def finger_off(self):
        return self.ppg.finger_off

    @finger_off.setter
    def finger_off(self, on):
        self.ppg.finger_off = bool(on)

    # ---- true values for tests -------------------------------------------
    def true_r_times(self):
        return np.asarray(self.ecg.beats)

    # ---- generators --------------------------------------------------------
    def read_ecg(self, k):
        adc = self.ecg.read(k).astype(float)
        if self._motion:
            t = (self.ecg.n - k + np.arange(k)) / self.ecg_fs
            adc += self._rng.normal(0.0, 140.0, k) + 220.0 * np.sin(2 * np.pi * 1.3 * t)
        adc = np.clip(np.rint(adc), 0, 4095).astype(int)
        if self.leads_off:
            adc = np.full(k, 4095)               # the real AD8232 rails when the leads are off
        return adc

    def read_imu(self, k):
        return self.imu.read(k)

    def read_ppg(self, k):
        t_end = (self.ppg.n + k) / self.ppg_fs
        self.ecg._extend(t_end + 1.0)             # make sure R peaks exist beyond the PPG time
        r = np.asarray(self.ecg.beats)
        while len(self._jitter) < r.size:
            self._jitter.append(float(self._rng.normal(0.0, self.pat_jitter_s)))
        lag = self.pat_s - SyntheticPPG.WAVES[0][0]        # beat start = R + PAT - systolic offset
        self.ppg.beats = list(r + lag + np.asarray(self._jitter[:r.size]))
        return self.ppg.read(k)


class FusionSimSource(BaseSource):
    """Real-time paced synthetic source with the same attributes as FusionSerialSource."""

    def __init__(self, bpm=72.0, pat_s=0.25, ecg_fs=250.0, imu_fs=100.0, ppg_fs=100.0, seed=None):
        super().__init__()
        self.gen = SyntheticFusion(bpm=bpm, pat_s=pat_s, ecg_fs=ecg_fs, imu_fs=imu_fs,
                                   ppg_fs=ppg_fs, seed=seed)
        self.description = (f"SIMULATOR (synthetic ECG + IMU + PPG, {bpm:.0f} bpm, "
                            f"true PAT {pat_s * 1000:.0f} ms)")
        self.ecg_samples = collections.deque()
        self.imu_samples = collections.deque()
        self.ppg_samples = collections.deque()
        self.ecg_meta = {"fs_hz": str(int(ecg_fs))}
        self.imu_meta = {"fs_hz": str(int(imu_fs)), "accel_range_g": "4", "gyro_range_dps": "500",
                         "who_am_i": "0x68"}
        self.ppg_meta = {"fs_hz": str(int(ppg_fs)), "adc_bits": "18", "led_red_ma": "7.2",
                         "led_ir_ma": "7.2", "adc_range_na": "4096", "part_id": "0x15"}
        self.all_meta = {}

    def total_bad_lines(self):
        return 0

    def bad_lines_by_stream(self):
        return {"ecg": 0, "imu": 0, "ppg": 0, "other": 0}

    def pop_messages(self):
        out = list(self.messages)
        self.messages.clear()
        return out

    def run(self):
        self.connected = True
        g = self.gen
        t0 = time.perf_counter()
        n = {"ecg": 0, "imu": 0, "ppg": 0}
        origin_us = 5_000_000
        while not self._stop_evt.is_set():
            now = time.perf_counter() - t0

            k = int(now * g.ecg_fs) - n["ecg"]
            if k > 0:
                adc = g.read_ecg(k)
                lo = 1 if g.leads_off else 0
                for i in range(k):
                    self.ecg_samples.append((origin_us + int((n["ecg"] + i) * 1e6 / g.ecg_fs), int(adc[i]), lo, lo))
                n["ecg"] += k

            k = int(now * g.imu_fs) - n["imu"]
            if k > 0:
                raw = g.read_imu(k)
                for i in range(k):
                    self.imu_samples.append((origin_us + int((n["imu"] + i) * 1e6 / g.imu_fs),
                                             *[int(v) for v in raw[i]]))
                n["imu"] += k

            k = int(now * g.ppg_fs) - n["ppg"]
            if k > 0:
                red, ir = g.read_ppg(k)
                for i in range(k):
                    self.ppg_samples.append((origin_us + int((n["ppg"] + i) * 1e6 / g.ppg_fs),
                                             int(red[i]), int(ir[i])))
                n["ppg"] += k

            time.sleep(0.004)
        self.connected = False