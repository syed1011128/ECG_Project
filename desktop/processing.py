"""
processing.py
=============
ECG signal-processing helpers for the live viewer (and later, offline analysis).

Contents
--------
design_ecg_filter   : 0.5-40 Hz band-pass (+ optional 50 Hz mains notch) as SOS
filter_offline      : zero-phase version for offline analysis (NOT for real time)
StreamingFilter     : causal, chunk-by-chunk filtering that keeps its state
detect_r_peaks      : Pan-Tompkins-style R-peak detector for a window of ECG
rr_from_peaks, bpm_from_rr, hrv_metrics : RR, heart rate, SDNN/RMSSD/pNN50
ECGPipeline         : GUI-independent live pipeline (buffers, filter, peaks, BPM, quality)

Everything here is a research/learning implementation, not a medical algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

ADC_MAX = 4095  # ESP32 12-bit ADC


# ----------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------
def design_ecg_filter(fs, low=0.5, high=40.0, notch_hz=50.0, notch_q=30.0, order=2):
    """Return SOS for band-pass [low, high] Hz, optionally followed by a mains notch.

    Bangladesh mains = 50 Hz. Set notch_hz=None to disable the notch.
    """
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    if notch_hz and notch_hz < fs / 2:
        b, a = signal.iirnotch(notch_hz, notch_q, fs=fs)
        sos = np.vstack([sos, signal.tf2sos(b, a)])
    return sos


def filter_offline(x, fs, **kwargs):
    """Zero-phase (filtfilt) version. Good for offline analysis, not for live use."""
    sos = design_ecg_filter(fs, **kwargs)
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=float))


class StreamingFilter:
    """Causal SOS filter that can be fed chunk by chunk.

    On the first chunk the filter state is initialised to the steady state for
    the first sample, so the big start-up transient (raw ADC sits near 2000,
    not near 0) does not swamp the display.
    """

    def __init__(self, sos):
        self.sos = sos
        self._zi_unit = signal.sosfilt_zi(sos)
        self.zi = None

    def reset(self):
        self.zi = None

    def process(self, x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x
        if self.zi is None:
            self.zi = self._zi_unit * x[0]
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


# ----------------------------------------------------------------------
# R-peak detection, RR, BPM, HRV
# ----------------------------------------------------------------------
def detect_r_peaks(ecg, fs, min_rr_s=0.30, refine_s=0.08):
    """Detect R peaks in a window of (already band-passed) ECG.

    Steps (Pan-Tompkins style):
      1. 5-15 Hz zero-phase band-pass to emphasise the QRS complex
      2. derivative -> square -> moving-window integration (100 ms)
      3. find_peaks with a refractory distance, adaptive threshold
      4. refine each peak to the max |QRS-band signal| within +-refine_s
    Returns integer sample indices into `ecg`. Works for upright or inverted R.
    """
    x = np.asarray(ecg, dtype=float)
    empty = np.zeros(0, dtype=int)
    if x.size < int(2 * fs):
        return empty

    sos_q = signal.butter(2, [5.0, 15.0], btype="bandpass", fs=fs, output="sos")
    q = signal.sosfiltfilt(sos_q, x)

    d = np.gradient(q)
    win = max(1, int(0.10 * fs))
    energy = np.convolve(d * d, np.ones(win) / win, mode="same")

    cand, _ = signal.find_peaks(energy, distance=max(1, int(min_rr_s * fs)))
    if cand.size == 0:
        return empty

    heights = energy[cand]
    ref = np.percentile(heights, 80)          # typical beat height
    floor = 8.0 * np.median(energy)           # reject pure-noise windows
    keep = cand[heights >= max(0.4 * ref, floor)]
    if keep.size == 0:
        return empty

    # refine to the actual R location
    r = max(1, int(refine_s * fs))
    aq = np.abs(q)
    refined = []
    for c in keep:
        lo, hi = max(0, c - r), min(x.size, c + r + 1)
        refined.append(lo + int(np.argmax(aq[lo:hi])))
    refined = np.unique(refined)

    # enforce the refractory period again after refinement (keep the stronger one)
    min_dist = int(min_rr_s * fs)
    out = []
    for p in refined:
        if out and p - out[-1] < min_dist:
            if aq[p] > aq[out[-1]]:
                out[-1] = p
        else:
            out.append(int(p))
    return np.asarray(out, dtype=int)


def rr_from_peaks(peak_times_s, min_rr=0.30, max_rr=2.0):
    """RR intervals (s) between consecutive peaks, keeping only physiological values."""
    t = np.asarray(peak_times_s, dtype=float)
    if t.size < 2:
        return np.zeros(0)
    rr = np.diff(t)
    return rr[(rr >= min_rr) & (rr <= max_rr)]


def bpm_from_rr(rr, n=5):
    """Heart rate from the median of the last n RR intervals (robust to one bad beat)."""
    rr = np.asarray(rr, dtype=float)
    if rr.size == 0:
        return None
    return float(60.0 / np.median(rr[-n:]))


def hrv_metrics(peak_times_s, min_rr=0.30, max_rr=2.0):
    """SDNN, RMSSD (ms) and pNN50 (%) from R-peak times in seconds.

    Short windows give unstable estimates; treat the values as indicative only.
    """
    nan = {"sdnn_ms": float("nan"), "rmssd_ms": float("nan"), "pnn50": float("nan")}
    t = np.asarray(peak_times_s, dtype=float)
    if t.size < 4:
        return nan
    rr = np.diff(t)
    valid = (rr >= min_rr) & (rr <= max_rr)
    pair = valid[:-1] & valid[1:]               # successive differences need two good RRs
    drr = np.diff(rr)[pair]
    rr_ok = rr[valid]
    if rr_ok.size < 3 or drr.size < 2:
        return nan
    return {
        "sdnn_ms": float(np.std(rr_ok, ddof=1) * 1000.0),
        "rmssd_ms": float(np.sqrt(np.mean(drr ** 2)) * 1000.0),
        "pnn50": float(100.0 * np.mean(np.abs(drr) > 0.050)),
    }


# ----------------------------------------------------------------------
# Live pipeline (no GUI code in here, so it can be unit-tested)
# ----------------------------------------------------------------------
@dataclass
class PushResult:
    filtered: np.ndarray
    new_beats: list = field(default_factory=list)  # (t_us, rr_s or nan, bpm_inst or nan)


class ECGPipeline:
    """Keeps rolling buffers, filters samples, detects beats, estimates BPM.

    Call push(t_us, raw, leads_off) with arrays of new samples as they arrive.
    """

    def __init__(self, fs=250.0, buffer_s=12.0, notch_hz=50.0, detect_window_s=6.0,
                 settle_s=3.0, detect_every_s=0.25, min_rr_s=0.30, max_rr_s=2.0):
        self.fs = float(fs)
        self.buffer_len = int(buffer_s * fs)
        self.detect_window = int(detect_window_s * fs)
        self.settle = int(settle_s * fs)
        self.detect_every = int(detect_every_s * fs)
        self.min_rr = float(min_rr_s)
        self.max_rr = float(max_rr_s)

        self.filter = StreamingFilter(design_ecg_filter(fs, notch_hz=notch_hz))

        self.t_us = np.zeros(0, dtype=np.int64)
        self.raw = np.zeros(0)
        self.filt = np.zeros(0)
        self.lo = np.zeros(0, dtype=bool)

        self.clean_run = 0          # consecutive clean samples since start / last leads-off
        self.leads_off = False
        self.total_samples = 0
        self.dropped_est = 0
        self.peak_t_us: list[int] = []
        self._since_detect = 0

    # ---- public API --------------------------------------------------
    def push(self, t_us, raw, leads_off):
        t_us = np.asarray(t_us, dtype=np.int64)
        raw = np.asarray(raw, dtype=float)
        lo = np.asarray(leads_off).astype(bool)
        n = t_us.size
        if n == 0:
            return PushResult(np.zeros(0), [])

        self._count_drops(t_us)

        if lo.any():
            # electrodes off: ADC value is meaningless -> flat line, restart filter later
            filt = np.zeros(n)
            self.filter.reset()
            self.clean_run = 0
            self.leads_off = True
        else:
            filt = self.filter.process(raw)
            self.clean_run += n
            self.leads_off = False

        self._append(t_us, raw, filt, lo)
        self.total_samples += n
        self._since_detect += n

        new_beats = []
        if self._since_detect >= self.detect_every:
            self._since_detect = 0
            new_beats = self._detect()
        return PushResult(filt, new_beats)

    @property
    def bpm(self):
        if self.leads_off or not self.peak_t_us or self.t_us.size == 0:
            return None
        if (self.t_us[-1] - self.peak_t_us[-1]) / 1e6 > 3.0:   # no recent beat
            return None
        pts = np.asarray(self.peak_t_us[-7:], dtype=float) / 1e6
        return bpm_from_rr(rr_from_peaks(pts, self.min_rr, self.max_rr))

    def hrv(self):
        pts = np.asarray(self.peak_t_us[-31:], dtype=float) / 1e6
        return hrv_metrics(pts, self.min_rr, self.max_rr)

    def quality(self):
        """Very simple quality label for the display. Not a validated SQI."""
        if self.total_samples == 0:
            return "NO DATA"
        if self.leads_off:
            return "LEADS OFF"
        r = self.raw[-int(self.fs):]
        if np.mean((r <= 5) | (r >= ADC_MAX - 5)) > 0.05:
            return "SATURATED"
        if np.std(r) < 2.0:
            return "FLAT"
        if self.clean_run < self.settle:
            return "SETTLING"
        return "OK"

    # ---- internals ---------------------------------------------------
    def _count_drops(self, t_us):
        expected = 1e6 / self.fs
        tt = t_us if self.t_us.size == 0 else np.concatenate(([self.t_us[-1]], t_us))
        d = np.diff(tt)
        big = d > 1.5 * expected
        if big.any():
            self.dropped_est += int(np.sum(np.round(d[big] / expected) - 1))

    def _append(self, t_us, raw, filt, lo):
        n = self.buffer_len
        self.t_us = np.concatenate([self.t_us, t_us])[-n:]
        self.raw = np.concatenate([self.raw, raw])[-n:]
        self.filt = np.concatenate([self.filt, filt])[-n:]
        self.lo = np.concatenate([self.lo, lo])[-n:]

    def _detect(self):
        usable = self.clean_run - self.settle
        if usable < int(3 * self.fs):
            return []
        nwin = int(min(usable, self.detect_window, self.filt.size))
        x = self.filt[-nwin:]
        t = self.t_us[-nwin:]

        idx = detect_r_peaks(x, self.fs, self.min_rr)
        lead = int(0.10 * self.fs)    # ignore filter edge effects at the window start
        guard = int(0.15 * self.fs)   # newest beats are not "finished" yet
        idx = idx[(idx >= lead) & (idx < nwin - guard)]

        new = []
        for i in idx:
            tp = int(t[i])
            if self.peak_t_us and tp <= self.peak_t_us[-1] + int(self.min_rr * 1e6):
                continue
            if self.peak_t_us:
                rr = (tp - self.peak_t_us[-1]) / 1e6
                if not (self.min_rr <= rr <= self.max_rr):
                    rr = float("nan")
            else:
                rr = float("nan")
            bpm_inst = 60.0 / rr if rr == rr else float("nan")
            self.peak_t_us.append(tp)
            new.append((tp, rr, bpm_inst))
        if len(self.peak_t_us) > 200:
            self.peak_t_us = self.peak_t_us[-200:]
        return new