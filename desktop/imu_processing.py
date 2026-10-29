"""
imu_processing.py
=================
MPU6050 processing for the live IMU viewer (GUI-free, unit-testable).

raw int16  ->  g and deg/s  ->  |a|, |gyro|  ->  motion index  ->  STILL / MOTION

Motion rule (starting values, tune them on your own recordings):
    accel_idx = rolling std of |a| (g) over `window_s`
    gyro_idx  = rolling mean of |gyro| (deg/s) over `window_s`
    MOTION    = accel_idx > motion_thr_g  OR  gyro_idx > gyro_thr_dps

IMPORTANT: a MOTION flag means "this window is likely unreliable for PPG/ECG".
It does NOT prove the physiological signals are invalid. Validate the thresholds
against reviewed recordings before relying on them.

Research/learning implementation, not a medical algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# MPU6050 datasheet sensitivities
ACCEL_LSB_PER_G = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
GYRO_LSB_PER_DPS = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
SATURATION_RAW = 32760          # raw int16 this close to +-32768 means full-scale clipping


def raw_to_units(raw, accel_range_g, gyro_range_dps):
    """raw (n,6) int16-like -> (accel_g (n,3), gyro_dps (n,3))."""
    raw = np.asarray(raw, dtype=float).reshape(-1, 6)
    acc = raw[:, :3] / ACCEL_LSB_PER_G[int(accel_range_g)]
    gyr = raw[:, 3:] / GYRO_LSB_PER_DPS[int(gyro_range_dps)]
    return acc, gyr


def rolling_std(x, w):
    """Std over the trailing window of w samples (expanding window at the start)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()
    w = max(2, int(w))
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    i = np.arange(1, n + 1)
    start = np.maximum(0, i - w)
    cnt = i - start
    s1 = c1[i] - c1[start]
    s2 = c2[i] - c2[start]
    var = s2 / cnt - (s1 / cnt) ** 2
    return np.sqrt(np.maximum(var, 0.0))


def rolling_mean(x, w):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()
    w = max(1, int(w))
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(1, n + 1)
    start = np.maximum(0, i - w)
    return (c1[i] - c1[start]) / (i - start)


@dataclass
class IMUPushResult:
    acc_g: np.ndarray            # (n,3)
    gyro_dps: np.ndarray         # (n,3), gyro offset removed
    a_mag: np.ndarray            # (n,) g
    g_mag: np.ndarray            # (n,) deg/s
    accel_idx: np.ndarray        # (n,) g
    gyro_idx: np.ndarray         # (n,) deg/s
    motion: np.ndarray           # (n,) bool
    calibrated: np.ndarray       # (n,) bool
    extra: dict = field(default_factory=dict)


class IMUPipeline:
    """Rolling buffers + conversion + gyro-offset calibration + motion flag."""

    def __init__(self, fs=100.0, accel_range_g=4, gyro_range_dps=500, window_s=1.0,
                 motion_thr_g=0.05, gyro_thr_dps=15.0, buffer_s=12.0, calib_s=3.0):
        self.window_s = float(window_s)
        self.buffer_s = float(buffer_s)
        self.calib_s = float(calib_s)
        self.motion_thr_g = float(motion_thr_g)
        self.gyro_thr_dps = float(gyro_thr_dps)
        self.configure(fs=fs, accel_range_g=accel_range_g, gyro_range_dps=gyro_range_dps)

        self.t_us = np.zeros(0, dtype=np.int64)
        self.acc = np.zeros((0, 3))
        self.gyr = np.zeros((0, 3))
        self.a_mag = np.zeros(0)
        self.g_mag = np.zeros(0)
        self.sat = np.zeros(0, dtype=bool)
        self.cal = np.zeros(0, dtype=bool)
        # derived series over the whole buffer (recomputed on every push)
        self.accel_idx = np.zeros(0)
        self.gyro_idx = np.zeros(0)
        self.motion = np.zeros(0, dtype=bool)

        self.total_samples = 0
        self.dropped_est = 0
        self.recalibrate()

    # ---- configuration ------------------------------------------------
    def configure(self, fs=None, accel_range_g=None, gyro_range_dps=None):
        if fs is not None:
            self.fs = float(fs)
        if accel_range_g is not None:
            if int(accel_range_g) not in ACCEL_LSB_PER_G:
                raise ValueError(f"unsupported accel range {accel_range_g}")
            self.accel_range_g = int(accel_range_g)
        if gyro_range_dps is not None:
            if int(gyro_range_dps) not in GYRO_LSB_PER_DPS:
                raise ValueError(f"unsupported gyro range {gyro_range_dps}")
            self.gyro_range_dps = int(gyro_range_dps)
        self.buffer_len = int(self.buffer_s * self.fs)
        self.window = max(2, int(self.window_s * self.fs))
        self.calib_n = max(1, int(self.calib_s * self.fs))

    def recalibrate(self):
        """Restart gyro-offset estimation. Keep the sensor still for calib_s seconds."""
        self.calibrated = False
        self._calib_count = 0
        self._calib_sum = np.zeros(3)
        self.gyro_offset = np.zeros(3)

    # ---- public API ---------------------------------------------------
    def push(self, t_us, raw):
        t_us = np.asarray(t_us, dtype=np.int64)
        raw = np.asarray(raw, dtype=float).reshape(-1, 6)
        n = t_us.size
        if n == 0:
            return self._empty_result()

        self._count_drops(t_us)
        acc, gyr = raw_to_units(raw, self.accel_range_g, self.gyro_range_dps)
        sat = np.any(np.abs(raw) >= SATURATION_RAW, axis=1)

        # gyro offset: running mean while calibrating, then frozen
        cal_flags = np.zeros(n, dtype=bool)
        if not self.calibrated:
            self._calib_sum += gyr.sum(axis=0)
            self._calib_count += n
            self.gyro_offset = self._calib_sum / self._calib_count
            if self._calib_count >= self.calib_n:
                self.calibrated = True
        cal_flags[:] = self.calibrated
        gyr = gyr - self.gyro_offset

        a_mag = np.linalg.norm(acc, axis=1)
        g_mag = np.linalg.norm(gyr, axis=1)

        self._append(t_us, acc, gyr, a_mag, g_mag, sat, cal_flags)
        self.total_samples += n
        self._recompute()

        return IMUPushResult(acc, gyr, a_mag, g_mag,
                             self.accel_idx[-n:], self.gyro_idx[-n:],
                             self.motion[-n:], cal_flags)

    @property
    def status(self):
        if self.total_samples == 0:
            return "NO DATA"
        if not self.calibrated:
            return "CALIBRATING"
        recent = max(1, int(self.fs))
        if self.sat[-recent:].any():
            return "SATURATED"
        return "MOTION" if self.motion[-1] else "STILL"

    @property
    def motion_reason(self):
        """Which rule triggered the flag right now: '', 'accel', 'gyro' or 'accel+gyro'."""
        if self.motion.size == 0 or not self.motion[-1]:
            return ""
        parts = []
        if self.accel_idx[-1] > self.motion_thr_g:
            parts.append("accel")
        if self.gyro_idx[-1] > self.gyro_thr_dps:
            parts.append("gyro")
        return "+".join(parts)

    # ---- internals ----------------------------------------------------
    def _empty_result(self):
        e1, e3 = np.zeros(0), np.zeros((0, 3))
        return IMUPushResult(e3, e3, e1, e1, e1, e1, np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))

    def _count_drops(self, t_us):
        expected = 1e6 / self.fs
        tt = t_us if self.t_us.size == 0 else np.concatenate(([self.t_us[-1]], t_us))
        d = np.diff(tt)
        big = d > 1.5 * expected
        if big.any():
            self.dropped_est += int(np.sum(np.round(d[big] / expected) - 1))

    def _append(self, t_us, acc, gyr, a_mag, g_mag, sat, cal):
        n = self.buffer_len
        self.t_us = np.concatenate([self.t_us, t_us])[-n:]
        self.acc = np.concatenate([self.acc, acc])[-n:]
        self.gyr = np.concatenate([self.gyr, gyr])[-n:]
        self.a_mag = np.concatenate([self.a_mag, a_mag])[-n:]
        self.g_mag = np.concatenate([self.g_mag, g_mag])[-n:]
        self.sat = np.concatenate([self.sat, sat])[-n:]
        self.cal = np.concatenate([self.cal, cal])[-n:]

    def _recompute(self):
        self.accel_idx = rolling_std(self.a_mag, self.window)
        self.gyro_idx = rolling_mean(self.g_mag, self.window)
        flag = (self.accel_idx > self.motion_thr_g) | (self.gyro_idx > self.gyro_thr_dps)
        self.motion = flag & self.cal      # never flag while the gyro offset is still unknown