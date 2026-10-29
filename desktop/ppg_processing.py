"""
ppg_processing.py
=================
MAX30102 PPG processing for the live PPG viewer (GUI-free, unit-testable).

raw RED / IR counts (18 bit)
    -> finger detection (IR level)
    -> causal 0.5-8 Hz band-pass (display + AC amplitude), inverted so the pulse points UP
    -> pulse peaks (zero-phase band-pass on a window, so peak TIMES are not delayed)
    -> inter-beat intervals, BPM
    -> perfusion index (AC/DC of IR) and the red/IR "ratio of ratios" R
       (R uses its own zero-phase 0.7-4 Hz band so breathing drift does not bias it)

Why inverted? A reflective PPG sensor sees LESS returned light when more blood is in
the tissue (blood absorbs light), so the raw counts DIP at every systole. Plotting
-(filtered IR) gives the familiar upward pulse.

SpO2 here is only  110 - 25*R  (a textbook example formula). It is NOT calibrated
and NOT validated. Treat it as experimental. The proposal says SpO2 should only be
reported after proper validation.

Research/learning implementation, not a medical algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from processing import StreamingFilter, bpm_from_rr, rr_from_peaks

ADC_MAX = 262143          # 18-bit
SAT_LEVEL = 262000        # counts this close to full scale are treated as clipped


def design_ppg_filter(fs, low=0.5, high=8.0, order=2):
    """Band-pass as SOS. Initial passband from the proposal: 0.5-8 Hz."""
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def detect_pulse_peaks(ir_raw, fs, min_rr_s=0.33, low=0.5, high=8.0):
    """Systolic pulse peaks in a window of RAW IR counts.

    Returns (indices, amplitudes). Amplitude is the zero-phase filtered peak height
    (counts, pulse pointing up). Uses a zero-phase filter on purpose: a causal filter
    would delay the peaks by tens of ms, which matters later for ECG->PPG timing.
    """
    x = np.asarray(ir_raw, dtype=float)
    empty = (np.zeros(0, dtype=int), np.zeros(0))
    if x.size < int(2 * fs):
        return empty
    sos = design_ppg_filter(fs, low, high)
    y = -signal.sosfiltfilt(sos, x - np.mean(x))          # invert: blood volume up = pulse up
    rng = np.percentile(y, 95) - np.percentile(y, 5)
    if rng < 10.0:                                        # no pulsatile content
        return empty

    d = max(1, int(min_rr_s * fs))
    cand, props = signal.find_peaks(y, distance=d, prominence=0.15 * rng)
    if cand.size == 0:
        return empty
    prom = props["prominences"]
    keep = prom >= 0.5 * np.median(prom)                  # drop small diastolic / noise bumps
    return cand[keep], y[cand[keep]]


@dataclass
class PPGPushResult:
    ppg: np.ndarray                                  # filtered, inverted IR (pulse up), counts
    finger: np.ndarray                               # (n,) bool
    new_beats: list = field(default_factory=list)    # (t_us, ibi_s or nan, bpm_inst or nan, amp_counts)


class PPGPipeline:
    """Rolling buffers, finger detection, filtering, pulse peaks, BPM, PI and R."""

    def __init__(self, fs=100.0, buffer_s=14.0, finger_thr=30000.0, detect_window_s=8.0,
                 settle_s=4.0, detect_every_s=0.25, min_rr_s=0.33, max_rr_s=2.0,
                 min_pi_pct=0.1, stats_window_s=5.0):
        self.finger_thr = float(finger_thr)
        self.buffer_s = float(buffer_s)
        self.detect_window_s = float(detect_window_s)
        self.settle_s = float(settle_s)
        self.detect_every_s = float(detect_every_s)
        self.stats_window_s = float(stats_window_s)
        self.min_rr = float(min_rr_s)
        self.max_rr = float(max_rr_s)
        self.min_pi_pct = float(min_pi_pct)
        self.configure(fs)

        self.t_us = np.zeros(0, dtype=np.int64)
        self.red = np.zeros(0)
        self.ir = np.zeros(0)
        self.ppg = np.zeros(0)         # -(causal band-passed IR)
        self.ppg_red = np.zeros(0)     # -(causal band-passed RED)
        self.fin = np.zeros(0, dtype=bool)

        self.finger = False
        self.clean_run = 0
        self.total_samples = 0
        self.dropped_est = 0
        self.peak_t_us: list[int] = []
        self._since_detect = 0
        self._reset_measurements()

    # ---- configuration ------------------------------------------------
    def configure(self, fs):
        self.fs = float(fs)
        self.buffer_len = int(self.buffer_s * self.fs)
        self.detect_window = int(self.detect_window_s * self.fs)
        self.settle = int(self.settle_s * self.fs)
        self.detect_every = max(1, int(self.detect_every_s * self.fs))
        sos = design_ppg_filter(self.fs)
        self.filt_ir = StreamingFilter(sos)
        self.filt_red = StreamingFilter(sos)
        # narrower zero-phase band for the AC amplitudes used in R: breathing drift (~0.2-0.3 Hz)
        # would otherwise add the SAME relative amount to red and IR and pull R toward 1
        self._sos_r = design_ppg_filter(self.fs, low=0.7, high=4.0)

    def _reset_measurements(self):
        self.pi_pct = None
        self.r_ratio = None
        self.spo2_est = None

    # ---- public API ---------------------------------------------------
    def push(self, t_us, red, ir):
        t_us = np.asarray(t_us, dtype=np.int64)
        red = np.asarray(red, dtype=float)
        ir = np.asarray(ir, dtype=float)
        n = t_us.size
        if n == 0:
            return PPGPushResult(np.zeros(0), np.zeros(0, dtype=bool), [])

        self._count_drops(t_us)

        # finger detection with hysteresis (on at thr, off at 0.8 * thr)
        thr = self.finger_thr * (0.8 if self.finger else 1.0)
        finger_now = float(np.mean(ir)) >= thr
        if finger_now and not self.finger:
            self._on_finger()
        elif not finger_now and self.finger:
            self._on_finger_lost()
        self.finger = finger_now

        if finger_now:
            ppg = -self.filt_ir.process(ir)
            ppg_red = -self.filt_red.process(red)
            self.clean_run += n
        else:
            ppg = np.zeros(n)
            ppg_red = np.zeros(n)
            self.clean_run = 0

        self._append(t_us, red, ir, ppg, ppg_red, np.full(n, finger_now))
        self.total_samples += n
        self._since_detect += n

        new_beats = []
        if self._since_detect >= self.detect_every:
            self._since_detect = 0
            new_beats = self._detect()
            self._update_stats()
        return PPGPushResult(ppg, np.full(n, finger_now), new_beats)

    @property
    def bpm(self):
        if not self.finger or not self.peak_t_us or self.t_us.size == 0:
            return None
        if (self.t_us[-1] - self.peak_t_us[-1]) / 1e6 > 3.0:          # no recent pulse
            return None
        pts = np.asarray(self.peak_t_us[-7:], dtype=float) / 1e6
        return bpm_from_rr(rr_from_peaks(pts, self.min_rr, self.max_rr))

    def quality(self):
        """Simple label for the display. Not a validated signal-quality index."""
        if self.total_samples == 0:
            return "NO DATA"
        if not self.finger:
            return "NO FINGER"
        k = max(1, int(self.fs))
        if np.mean(self.ir[-k:] >= SAT_LEVEL) > 0.02 or np.mean(self.red[-k:] >= SAT_LEVEL) > 0.02:
            return "SATURATED"
        if self.clean_run < self.settle:
            return "SETTLING"
        if self.pi_pct is not None and self.pi_pct < self.min_pi_pct:
            return "LOW SIGNAL"
        rr = self._recent_rr()
        if rr.size >= 4 and np.std(rr) / np.mean(rr) > 0.25:
            return "NOISY"
        return "OK"

    # ---- internals ----------------------------------------------------
    def _on_finger(self):
        self.filt_ir.reset()
        self.filt_red.reset()
        self.peak_t_us = []
        self._reset_measurements()

    def _on_finger_lost(self):
        self.filt_ir.reset()
        self.filt_red.reset()
        self.peak_t_us = []
        self._reset_measurements()

    def _recent_rr(self):
        pts = np.asarray(self.peak_t_us[-8:], dtype=float) / 1e6
        return rr_from_peaks(pts, self.min_rr, self.max_rr)

    def _count_drops(self, t_us):
        expected = 1e6 / self.fs
        tt = t_us if self.t_us.size == 0 else np.concatenate(([self.t_us[-1]], t_us))
        d = np.diff(tt)
        big = d > 1.5 * expected
        if big.any():
            self.dropped_est += int(np.sum(np.round(d[big] / expected) - 1))

    def _append(self, t_us, red, ir, ppg, ppg_red, fin):
        n = self.buffer_len
        self.t_us = np.concatenate([self.t_us, t_us])[-n:]
        self.red = np.concatenate([self.red, red])[-n:]
        self.ir = np.concatenate([self.ir, ir])[-n:]
        self.ppg = np.concatenate([self.ppg, ppg])[-n:]
        self.ppg_red = np.concatenate([self.ppg_red, ppg_red])[-n:]
        self.fin = np.concatenate([self.fin, fin])[-n:]

    def _update_stats(self):
        """Perfusion index and R over the last stats_window_s of clean signal."""
        k = int(self.stats_window_s * self.fs)
        if not self.finger or self.clean_run < self.settle + int(2 * self.fs) or self.ir.size < k:
            self._reset_measurements()
            return
        ir, red = self.ir[-k:], self.red[-k:]
        p_ir = self.ppg[-k:]
        dc_ir, dc_red = float(np.mean(ir)), float(np.mean(red))
        # AC for R: zero-phase 0.7-4 Hz band-pass on the raw window (same shape in both channels)
        ac_ir = float(np.std(signal.sosfiltfilt(self._sos_r, ir - dc_ir)))
        ac_red = float(np.std(signal.sosfiltfilt(self._sos_r, red - dc_red)))
        if dc_ir <= 0 or dc_red <= 0 or ac_ir <= 0:
            self._reset_measurements()
            return
        pp = float(np.percentile(p_ir, 95) - np.percentile(p_ir, 5))
        self.pi_pct = 100.0 * pp / dc_ir                       # approx peak-to-peak / DC, in %
        r = (ac_red / dc_red) / (ac_ir / dc_ir)
        self.r_ratio = r
        # experimental, uncalibrated; only for plausible R values
        self.spo2_est = float(np.clip(110.0 - 25.0 * r, 0.0, 100.0)) if 0.2 <= r <= 1.5 else None

    def _detect(self):
        usable = self.clean_run - self.settle
        if not self.finger or usable < int(3 * self.fs):
            return []
        nwin = int(min(usable, self.detect_window, self.ir.size))
        x = self.ir[-nwin:]
        t = self.t_us[-nwin:]

        idx, amp = detect_pulse_peaks(x, self.fs, self.min_rr)
        lead = int(0.5 * self.fs)      # filter edge effects at the start of the window
        guard = int(0.4 * self.fs)     # newest pulse is not "finished" yet
        sel = (idx >= lead) & (idx < nwin - guard)
        idx, amp = idx[sel], amp[sel]

        new = []
        for i, a in zip(idx, amp):
            tp = int(t[i])
            if self.peak_t_us and tp <= self.peak_t_us[-1] + int(self.min_rr * 1e6):
                continue
            if self.peak_t_us:
                ibi = (tp - self.peak_t_us[-1]) / 1e6
                if not (self.min_rr <= ibi <= self.max_rr):
                    ibi = float("nan")
            else:
                ibi = float("nan")
            bpm_inst = 60.0 / ibi if ibi == ibi else float("nan")
            self.peak_t_us.append(tp)
            new.append((tp, ibi, bpm_inst, float(a)))
        if len(self.peak_t_us) > 200:
            self.peak_t_us = self.peak_t_us[-200:]
        return new