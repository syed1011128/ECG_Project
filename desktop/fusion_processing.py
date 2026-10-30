"""
fusion_processing.py
====================
Sensor fusion for the combined ECG + IMU + PPG stream (GUI-free, unit-testable).

It owns the three single-sensor pipelines (ECGPipeline, IMUPipeline, PPGPipeline) and adds:

  * Pulse arrival time (PAT): for every ECG R peak, the delay until the pulse reaches the
    fingertip. Two definitions are reported
        pat_peak_ms   R peak -> systolic PEAK of the PPG pulse
        pat_slope_ms  R peak -> point of steepest UPSTROKE of the PPG pulse (before the peak)
    Both use the zero-phase filtered PPG with parabolic sub-sample interpolation, so the filter
    adds no delay. PAT is "R peak to pulse" and therefore includes the pre-ejection period of the
    heart, it is NOT pure pulse transit time.
  * Motion-aware validity: every beat gets flags for ECG contact, PPG contact/quality, IMU
    availability and MOTION (IMU flag anywhere between the R peak and the pulse, plus a margin).
    A beat is `valid` only if all are fine. Reason strings say why not.
  * Heart-rate agreement between ECG-derived and PPG-derived BPM (a cheap cross-check).

Synchronisation assumptions (be aware of them when you interpret PAT):
  * all three streams use the ESP32 clock (microseconds since boot), so no offset estimation is
    needed between them;
  * PPG sample times are reconstructed from the FIFO read time (error about +-3 ms) and the
    sensor's own averaging adds a roughly constant delay of a few ms. A constant offset shifts
    every PAT by the same amount (so PAT CHANGES are still meaningful), the +-3 ms jitter adds noise;
  * the ECG R-peak time is accurate to about one sample (4 ms at 250 Hz).

Research/learning implementation, not a medical algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from imu_processing import IMUPipeline
from ppg_processing import PPGPipeline, design_ppg_filter
from processing import ECGPipeline


@dataclass
class BeatRecord:
    r_t_us: int                 # ECG R-peak time
    ppg_t_us: float             # refined PPG systolic-peak time (nan if none)
    pat_peak_ms: float          # R -> systolic peak (nan if none)
    pat_slope_ms: float         # R -> steepest upstroke (nan if none)
    ecg_bpm_inst: float         # from the R-R interval (nan for the first beat)
    ppg_bpm_inst: float         # from the PPG peak-to-peak interval (nan if unknown)
    ecg_ok: bool
    ppg_ok: bool
    imu_ok: bool
    motion: bool
    valid: bool
    reason: str                 # "" when valid, otherwise: ecg / ppg / imu / motion / pat range


def _parabolic(y, i):
    """Sub-sample offset (in samples, -0.5..0.5) of the extremum near index i."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2.0 * b + c
    if den == 0.0:
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


class FusionPipeline:
    def __init__(self, ecg_fs=250.0, imu_fs=100.0, ppg_fs=100.0, notch_hz=50.0, finger_thr=30000.0,
                 accel_range_g=4, gyro_range_dps=500, motion_thr_g=0.05, gyro_thr_dps=15.0,
                 pat_min_s=0.10, pat_max_s=0.60, give_up_s=2.0, motion_pad_s=0.5,
                 hr_agree_bpm=5.0, max_pending=40):
        self.ecg = ECGPipeline(fs=ecg_fs, notch_hz=(notch_hz or None))
        self.imu = IMUPipeline(fs=imu_fs, accel_range_g=accel_range_g, gyro_range_dps=gyro_range_dps,
                               motion_thr_g=motion_thr_g, gyro_thr_dps=gyro_thr_dps)
        self.ppg = PPGPipeline(fs=ppg_fs, finger_thr=finger_thr)
        self.pat_min_us = int(pat_min_s * 1e6)
        self.pat_max_us = int(pat_max_s * 1e6)
        self.give_up_us = int(give_up_s * 1e6)
        self.motion_pad_us = int(motion_pad_s * 1e6)
        self.hr_agree_bpm = float(hr_agree_bpm)
        self.max_pending = int(max_pending)

        self.beats: list[BeatRecord] = []
        self._pending: list[tuple[int, float, float]] = []   # (r_t_us, rr_s, bpm_inst)
        self._last_ppg_used = -1

    # ---- push per sensor ---------------------------------------------------
    def push_ecg(self, t_us, raw, leads_off):
        res = self.ecg.push(t_us, raw, leads_off)
        for r_t, rr, bpm in res.new_beats:
            self._pending.append((int(r_t), float(rr), float(bpm)))
        if len(self._pending) > self.max_pending:
            self._pending = self._pending[-self.max_pending:]
        return res

    def push_imu(self, t_us, raw):
        return self.imu.push(t_us, raw)

    def push_ppg(self, t_us, red, ir):
        return self.ppg.push(t_us, red, ir)

    # ---- pairing -----------------------------------------------------------
    def resolve(self):
        """Pair pending R peaks with PPG pulses. Returns the BeatRecords finished by this call."""
        out = []
        if not self._pending:
            return out
        ppg, ecg = self.ppg, self.ecg
        ppg_end = int(ppg.t_us[-1]) if ppg.t_us.size else None
        ecg_end = int(ecg.t_us[-1]) if ecg.t_us.size else None
        peaks = list(ppg.peak_t_us)

        zp = None          # zero-phase filtered PPG, computed once and only if needed
        while self._pending:
            r_t, rr, bpm_e = self._pending[0]
            lo, hi = r_t + self.pat_min_us, r_t + self.pat_max_us
            found = None
            for p in peaks:
                if p > self._last_ppg_used and lo <= p <= hi:
                    found = p
                    break
            if found is None:
                waited_ppg = ppg_end is not None and ppg_end >= r_t + self.give_up_us
                waited_ecg = ecg_end is not None and ecg_end >= r_t + self.give_up_us + 1_000_000
                if not (waited_ppg or waited_ecg):
                    break                       # the pulse may still arrive: wait
                if self._imu_behind(r_t + self.pat_max_us + self.motion_pad_us, r_t, ecg_end):
                    break                       # IMU samples for the beat window have not arrived yet
                rec = self._make_record(r_t, bpm_e, None, None, None, peaks)
            else:
                if self._imu_behind(found + self.motion_pad_us, r_t, ecg_end):
                    break                       # the motion check needs IMU data up to pulse + margin
                if zp is None:
                    zp = self._zero_phase_ppg()
                t_peak, t_slope = self._refine(zp, found)
                rec = self._make_record(r_t, bpm_e, found, t_peak, t_slope, peaks)
                self._last_ppg_used = found
            self._pending.pop(0)
            self.beats.append(rec)
            out.append(rec)
        if len(self.beats) > 2000:
            self.beats = self.beats[-2000:]
        return out

    def _imu_behind(self, need_us, r_t, ecg_end):
        """True while the IMU buffer has not reached `need_us` yet, but may still get there.
        Gives up (returns False) when the IMU is absent / uncalibrated or the beat is old."""
        imu = self.imu
        if imu.t_us.size == 0 or not imu.calibrated:
            return False
        if imu.t_us[-1] >= need_us:
            return False
        return not (ecg_end is not None and ecg_end >= r_t + self.give_up_us + 1_000_000)

    # ---- PPG timing refinement ---------------------------------------------
    def _zero_phase_ppg(self):
        ppg = self.ppg
        n = int(min(ppg.ir.size, 12.0 * ppg.fs))
        if n < int(3 * ppg.fs):
            return None
        x = ppg.ir[-n:]
        t = ppg.t_us[-n:].astype(float)
        sos = design_ppg_filter(ppg.fs)
        y = -signal.sosfiltfilt(sos, x - np.mean(x))
        dt = np.gradient(t) / 1e6
        dy = np.gradient(y) / np.where(dt > 0, dt, 1.0 / ppg.fs)
        return t, y, dy

    def _refine(self, zp, peak_t_us):
        """Sub-sample systolic-peak time and steepest-upstroke time (microseconds, float)."""
        if zp is None:
            return float(peak_t_us), float("nan")
        t, y, dy = zp
        i = int(np.argmin(np.abs(t - peak_t_us)))
        if i <= 1 or i >= len(t) - 2:
            return float(peak_t_us), float("nan")
        # local maximum of y near i
        j = i + int(np.argmax(y[i - 1:i + 2])) - 1
        t_peak = t[j] + _parabolic(y, j) * 0.5 * (t[min(j + 1, len(t) - 1)] - t[max(j - 1, 0)])
        # steepest upstroke: maximum of dy in the 0.35 s before the peak
        w = int(0.35 * self.ppg.fs)
        a = j - w
        if a < 1:
            return float(t_peak), float("nan")
        k = a + int(np.argmax(dy[a:j + 1]))
        if k <= 0 or k >= len(t) - 1:
            return float(t_peak), float("nan")
        t_slope = t[k] + _parabolic(dy, k) * 0.5 * (t[k + 1] - t[k - 1])
        return float(t_peak), float(t_slope)

    # ---- flags ---------------------------------------------------------------
    def _make_record(self, r_t, bpm_e, found, t_peak, t_slope, peaks):
        ecg, ppg, imu = self.ecg, self.ppg, self.imu
        nan = float("nan")

        # ECG: electrodes on around the R peak, and the live quality label is fine
        m = (ecg.t_us >= r_t - 200_000) & (ecg.t_us <= r_t + 200_000)
        ecg_ok = bool(m.any() and not ecg.lo[m].any() and ecg.quality() == "OK")

        # PPG: finger on at the pulse and a good live quality label
        ppg_ok = False
        if found is not None and ppg.t_us.size:
            i = int(np.argmin(np.abs(ppg.t_us - found)))
            ppg_ok = bool(ppg.fin[i] and ppg.quality() == "OK")

        # IMU: calibrated, and the buffer covers the whole beat window
        start = r_t - self.motion_pad_us
        end = (found if found is not None else r_t + self.pat_max_us) + self.motion_pad_us
        imu_ok, motion = False, False
        if imu.t_us.size > 1 and imu.calibrated and imu.t_us[0] <= start and imu.t_us[-1] >= end:
            imu_ok = True
            mm = (imu.t_us >= start) & (imu.t_us <= end)
            motion = bool(imu.motion[mm].any())

        if found is None:
            pat_peak = pat_slope = ppg_t = ppg_bpm = nan
        else:
            ppg_t = t_peak
            pat_peak = (t_peak - r_t) / 1000.0
            pat_slope = (t_slope - r_t) / 1000.0 if t_slope == t_slope else nan
            idx = peaks.index(found) if found in peaks else -1
            ppg_bpm = nan
            if idx > 0:
                ibi = (found - peaks[idx - 1]) / 1e6
                if 0.3 <= ibi <= 2.0:
                    ppg_bpm = 60.0 / ibi

        reason = ""
        if not ecg_ok:
            reason = "ecg"
        elif found is None or not ppg_ok:
            reason = "ppg"
        elif not imu_ok:
            reason = "imu"
        elif motion:
            reason = "motion"
        valid = reason == ""
        return BeatRecord(r_t_us=int(r_t), ppg_t_us=ppg_t, pat_peak_ms=pat_peak, pat_slope_ms=pat_slope,
                          ecg_bpm_inst=bpm_e, ppg_bpm_inst=ppg_bpm, ecg_ok=ecg_ok, ppg_ok=ppg_ok,
                          imu_ok=imu_ok, motion=motion, valid=valid, reason=reason)

    # ---- summaries -------------------------------------------------------------
    def summary(self, last_n=30):
        """Statistics over the last `last_n` finished beats: valid count and PAT median / std (ms)."""
        recent = self.beats[-last_n:]
        good = [b for b in recent if b.valid]
        out = {"n_total": len(recent), "n_valid": len(good),
               "pat_peak_med": None, "pat_peak_std": None, "pat_slope_med": None, "pat_slope_std": None}
        if len(good) >= 3:
            pk = np.array([b.pat_peak_ms for b in good])
            out["pat_peak_med"], out["pat_peak_std"] = float(np.median(pk)), float(np.std(pk))
            sl = np.array([b.pat_slope_ms for b in good if b.pat_slope_ms == b.pat_slope_ms])
            if sl.size >= 3:
                out["pat_slope_med"], out["pat_slope_std"] = float(np.median(sl)), float(np.std(sl))
        return out

    def hr_agreement(self):
        """(ecg_bpm, ppg_bpm, delta or None, agree or None). agree needs both rates and a good PPG."""
        e, p = self.ecg.bpm, self.ppg.bpm
        if e is None or p is None:
            return e, p, None, None
        d = p - e
        agree = bool(abs(d) <= self.hr_agree_bpm) if self.ppg.quality() == "OK" else None
        return e, p, d, agree

    def overall_status(self):
        """Short text for the display, e.g. 'ECG OK | PPG OK | IMU STILL'."""
        return f"ECG {self.ecg.quality()} | PPG {self.ppg.quality()} | IMU {self.imu.status}"