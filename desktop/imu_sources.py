"""
imu_sources.py
==============
Data sources for imu_viewer.py (same thread + deque pattern as sources.py).

Sample tuple: (timestamp_us, ax, ay, az, gx, gy, gz)   raw int16 values

ImuSerialSource : real ESP32 + MPU6050 over USB serial
ImuSimSource    : synthetic IMU (still / shaking cycles), no hardware needed
"""
from __future__ import annotations

import re
import time

import numpy as np

from sources import BaseSource
from imu_processing import ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS

_KV = re.compile(r"(\w+)=(\S+)")
INT16_MIN, INT16_MAX = -32768, 32767


class ImuSerialSource(BaseSource):
    def __init__(self, port, baud=230400):
        super().__init__()
        self.port = port
        self.baud = baud
        self.description = f"{port} @ {baud}"
        self.meta = {}                 # filled from '# IMU_META key=value ...' lines
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
            if "IMU_META" in text:
                self.meta = dict(_KV.findall(text))
            return
        parts = text.split(",")
        if len(parts) != 7:
            self.bad_lines += 1
            return
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            self.bad_lines += 1
            return
        t = vals[0]
        if any(not (INT16_MIN <= v <= INT16_MAX) for v in vals[1:]):
            self.bad_lines += 1
            return
        if self._last_t is not None and t <= self._last_t:   # corrupted / truncated timestamp
            self.bad_lines += 1
            return
        self._last_t = t
        self.samples.append(tuple(vals))


# ----------------------------------------------------------------------
# Synthetic IMU
# ----------------------------------------------------------------------
class SyntheticIMU:
    """Sensor lying flat (gravity on +Z) with small noise and a gyro bias.

    Shaking: sinusoidal accel (2-3 Hz) and gyro motion, smoothly faded in/out.
    auto=True shakes during seconds [6, 9) of every 12 s cycle.
    Set .force_motion = True to shake continuously.
    """

    def __init__(self, fs=100.0, accel_range_g=4, gyro_range_dps=500,
                 gyro_bias=(3.0, -2.0, 1.5), acc_noise_g=0.004, gyro_noise_dps=0.15,
                 auto=True, seed=None):
        self.fs = float(fs)
        self.a_lsb = ACCEL_LSB_PER_G[accel_range_g]
        self.g_lsb = GYRO_LSB_PER_DPS[gyro_range_dps]
        self.bias = np.asarray(gyro_bias, dtype=float)
        self.acc_noise = acc_noise_g
        self.gyro_noise = gyro_noise_dps
        self.auto = auto
        self.force_motion = False
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self._env = 0.0

    def _target(self, t0):
        if self.force_motion:
            return 1.0
        return 1.0 if (self.auto and 6.0 <= (t0 % 12.0) < 9.0) else 0.0

    def read(self, k):
        t = (self.n + np.arange(k)) / self.fs
        target = self._target(t[0])
        env = target + (self._env - target) * np.exp(-(np.arange(1, k + 1) / self.fs) / 0.15)
        self._env = float(env[-1])

        acc = np.zeros((k, 3))
        acc[:, 2] = 1.0
        acc[:, 0] += env * 0.60 * np.sin(2 * np.pi * 3.0 * t)
        acc[:, 1] += env * 0.40 * np.sin(2 * np.pi * 2.3 * t + 1.0)
        acc[:, 2] += env * 0.30 * np.sin(2 * np.pi * 1.7 * t + 2.0)
        acc += self.rng.normal(0.0, self.acc_noise, (k, 3))

        gyr = np.tile(self.bias, (k, 1))
        gyr[:, 0] += env * 120.0 * np.sin(2 * np.pi * 2.5 * t)
        gyr[:, 1] += env * 80.0 * np.sin(2 * np.pi * 1.9 * t + 0.5)
        gyr += self.rng.normal(0.0, self.gyro_noise, (k, 3))

        self.n += k
        raw = np.hstack([acc * self.a_lsb, gyr * self.g_lsb])
        return np.clip(np.rint(raw), INT16_MIN, INT16_MAX).astype(int)


class ImuSimSource(BaseSource):
    """Real-time paced synthetic source. Set .force_motion = True to shake continuously."""

    def __init__(self, fs=100.0, accel_range_g=4, gyro_range_dps=500):
        super().__init__()
        self.fs = float(fs)
        self.gen = SyntheticIMU(fs=fs, accel_range_g=accel_range_g, gyro_range_dps=gyro_range_dps)
        self.description = "SIMULATOR (synthetic MPU6050: still, then shaking every 12 s)"
        self.meta = {"fs_hz": str(int(fs)), "accel_range_g": str(accel_range_g),
                     "gyro_range_dps": str(gyro_range_dps), "who_am_i": "0x68"}

    @property
    def force_motion(self):
        return self.gen.force_motion

    @force_motion.setter
    def force_motion(self, on):
        self.gen.force_motion = bool(on)

    def run(self):
        self.connected = True
        t0 = time.perf_counter()
        n = 0
        origin_us = 5_000_000
        while not self._stop_evt.is_set():
            target = int((time.perf_counter() - t0) * self.fs)
            if target > n:
                k = target - n
                raw = self.gen.read(k)
                for i in range(k):
                    t_us = origin_us + int((n + i) * 1e6 / self.fs)
                    self.samples.append((t_us, *[int(v) for v in raw[i]]))
                n = target
            time.sleep(0.005)
        self.connected = False