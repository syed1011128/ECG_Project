#!/usr/bin/env python3
"""
fusion_viewer.py - live ECG + PPG + IMU viewer for the combined ESP32 firmware (env:all)

    python fusion_viewer.py --list-ports
    python fusion_viewer.py --simulate                       # no hardware needed
    python fusion_viewer.py --port /dev/ttyUSB0 --subject S01 --activity rest

Shows ONE shared time axis for: filtered ECG with R peaks, the PPG pulse with its peaks (and the R
peaks as small triangles, so the ECG-to-pulse delay is visible), the IMU motion index, and a
trend of the pulse arrival time (PAT) per beat. Time ranges where the IMU flags motion are
shaded on all three signal plots. Every beat gets validity flags (ECG ok, PPG ok, IMU ok, motion);
only valid beats go into the PAT statistics.

Everything is logged to ../data/fusion_YYYYMMDD_HHMM_{ecg,imu,ppg,beats}.csv (one clock).

Research/learning prototype only - not a medical device. SpO2 is NOT shown here on purpose
(it is uncalibrated, see ppg_viewer.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# Linux/GNOME fix: force Qt's generic theme (avoids the 'antialiasing' gsettings crash).
# Must run BEFORE Qt is imported. Opt out with ECG_KEEP_QT_THEME=1
if sys.platform.startswith("linux") and not os.environ.get("ECG_KEEP_QT_THEME"):
    os.environ["QT_QPA_PLATFORMTHEME"] = "generic"

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from fusion_processing import FusionPipeline
from fusion_recorder import FusionRecorder
from fusion_sources import FusionSerialSource, FusionSimSource
from sources import find_serial_port, list_serial_ports

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COLORS = {"OK": "#3ddc84", "STILL": "#3ddc84", "SETTLING": "#f5c542", "CALIBRATING": "#f5c542",
          "NO DATA": "#8a949e", "NO FINGER": "#ff9f43", "LOW SIGNAL": "#ff9f43", "NOISY": "#ff9f43",
          "FLAT": "#ff9f43", "MOTION": "#ff5a5f", "SATURATED": "#ff5a5f", "LEADS OFF": "#ff5a5f"}
REASON_TEXT = {"": "valid", "ecg": "ECG not clean (leads off / settling / saturated)",
               "ppg": "PPG not clean (no finger / noisy / no pulse found)",
               "imu": "IMU not ready (calibrating or no data)", "motion": "motion detected by the IMU"}
MAX_REGIONS = 6


# ----------------------------------------------------------------------
class FusionWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.paused = False
        self._frame = 0
        self._err_printed = False
        self._last_meta = (None, None, None)
        self._ppg_ymin = 0.0

        self.pipe = FusionPipeline(ecg_fs=args.ecg_fs, imu_fs=args.imu_fs, ppg_fs=args.ppg_fs,
                                   notch_hz=args.notch, finger_thr=args.finger_thr,
                                   accel_range_g=args.accel_range, gyro_range_dps=args.gyro_range,
                                   motion_thr_g=args.motion_thr, gyro_thr_dps=args.gyro_thr)
        self.source = self._start_source()
        self.recorder = None
        if not args.no_log:
            self.recorder = FusionRecorder(
                args.data_dir, args.subject, args.activity, self.source.description,
                {"ecg_fs_nominal_hz": args.ecg_fs, "imu_fs_nominal_hz": args.imu_fs,
                 "ppg_fs_nominal_hz": args.ppg_fs, "notch_hz": args.notch,
                 "finger_thr_counts": args.finger_thr, "motion_thr_g": args.motion_thr,
                 "gyro_thr_dps": args.gyro_thr})

        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(30)

    # ---- setup ---------------------------------------------------------
    def _start_source(self):
        a = self.args
        if a.simulate:
            src = FusionSimSource(ecg_fs=a.ecg_fs, imu_fs=a.imu_fs, ppg_fs=a.ppg_fs)
        else:
            port = a.port or find_serial_port()
            if not port:
                raise RuntimeError("No serial port found. Plug in the ESP32 (check the USB cable "
                                   "carries data) or run with --simulate.")
            src = FusionSerialSource(port, a.baud)
        src.start()
        return src

    def _make_plot(self, row, ylabel, bottom=False, xrange=None):
        p = self.glw.addPlot(row=row, col=0)
        p.setLabel("left", ylabel)
        if bottom:
            p.setLabel("bottom", "Time (s, 0 = now)")
        p.showGrid(x=True, y=True, alpha=0.25)
        x = xrange if xrange else self.args.window
        p.setXRange(-x, 0, padding=0)
        p.setMouseEnabled(x=False, y=False)
        p.hideButtons()
        return p

    def _build_ui(self):
        self.setWindowTitle("ESP32 live ECG + PPG + IMU fusion")
        self.resize(1250, 960)
        self.setStyleSheet(
            "QMainWindow, QWidget { background-color: #101418; color: #d0d6dc; }"
            "QPushButton { background-color: #1f2933; border: 1px solid #3a4652;"
            "              border-radius: 4px; padding: 6px 14px; }"
            "QPushButton:hover { background-color: #2a3744; }"
            "QPushButton:checked { background-color: #3d566e; border-color: #5b7c99; }"
        )
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        self.ecg_bpm_label = self._big("--", "#ff5a5f")
        self.ppg_bpm_label = self._big("--", "#4cc9f0")
        for big, caption in ((self.ecg_bpm_label, "ECG BPM"), (self.ppg_bpm_label, "PPG BPM")):
            col = QtWidgets.QVBoxLayout()
            col.addWidget(big)
            c = QtWidgets.QLabel(caption)
            c.setStyleSheet("font-size: 12px; color: #9aa5b1;")
            c.setAlignment(QtCore.Qt.AlignHCenter)
            col.addWidget(c)
            top.addLayout(col)
            top.addSpacing(14)
        top.addSpacing(10)

        info = QtWidgets.QVBoxLayout()
        self.quality_label = QtWidgets.QLabel("")
        self.quality_label.setTextFormat(QtCore.Qt.RichText)
        self.quality_label.setStyleSheet("font-size: 17px; font-weight: bold;")
        self.agree_label = QtWidgets.QLabel("")
        self.agree_label.setStyleSheet("font-size: 14px; color: #c3ccd5;")
        self.pat_label = QtWidgets.QLabel("")
        self.pat_label.setStyleSheet("font-size: 14px; color: #c3ccd5;")
        self.hint_label = QtWidgets.QLabel("")
        self.hint_label.setStyleSheet("font-size: 12px; color: #9aa5b1;")
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("font-size: 11px; color: #8a949e;")
        for w in (self.quality_label, self.agree_label, self.pat_label, self.hint_label, self.status_label):
            info.addWidget(w)
        top.addLayout(info, 1)
        vbox.addLayout(top)

        self.glw = pg.GraphicsLayoutWidget()
        vbox.addWidget(self.glw, 1)
        self.ecg_plot = self._make_plot(0, "ECG (ADC)")
        self.ppg_plot = self._make_plot(1, "PPG pulse (counts)")
        self.imu_plot = self._make_plot(2, "Motion index (g)")
        self.pat_plot = self._make_plot(3, "PAT (ms)", bottom=True, xrange=self.args.pat_window)
        for row, stretch in enumerate((4, 4, 2, 3)):
            self.glw.ci.layout.setRowStretchFactor(row, stretch)
        self.ppg_plot.setXLink(self.ecg_plot)
        self.imu_plot.setXLink(self.ecg_plot)

        self.ecg_curve = self.ecg_plot.plot(pen=pg.mkPen("#ff6b6b", width=1.4))
        self.r_scatter = pg.ScatterPlotItem(size=9, pen=None, brush=pg.mkBrush(255, 214, 10, 230))
        self.ecg_plot.addItem(self.r_scatter)

        self.ppg_curve = self.ppg_plot.plot(pen=pg.mkPen("#4cc9f0", width=1.8))
        self.pk_scatter = pg.ScatterPlotItem(size=10, pen=None, brush=pg.mkBrush(255, 90, 95, 220))
        self.ppg_plot.addItem(self.pk_scatter)
        self.rmark_scatter = pg.ScatterPlotItem(size=9, pen=None, symbol="t1",
                                                brush=pg.mkBrush(255, 214, 10, 230))
        self.ppg_plot.addItem(self.rmark_scatter)

        self.imu_curve = self.imu_plot.plot(pen=pg.mkPen("#ff9f43", width=1.6))
        self.imu_plot.addItem(pg.InfiniteLine(pos=self.args.motion_thr, angle=0,
                                              pen=pg.mkPen("#6c7a89", style=QtCore.Qt.DashLine)))

        self.pat_bad = pg.ScatterPlotItem(size=7, pen=None, brush=pg.mkBrush(138, 148, 158, 160))
        self.pat_good = pg.ScatterPlotItem(size=9, pen=None, brush=pg.mkBrush(61, 220, 132, 230))
        self.pat_slope = pg.ScatterPlotItem(size=7, pen=None, symbol="d", brush=pg.mkBrush(76, 201, 240, 220))
        for it in (self.pat_bad, self.pat_slope, self.pat_good):
            self.pat_plot.addItem(it)
        self.pat_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#3ddc84", style=QtCore.Qt.DotLine))
        self.pat_plot.addItem(self.pat_line)
        self.pat_line.hide()

        # shaded "IMU says motion" regions on the three time-aligned plots
        self.regions = {}
        for key, plot in (("ecg", self.ecg_plot), ("ppg", self.ppg_plot), ("imu", self.imu_plot)):
            pool = []
            for _ in range(MAX_REGIONS):
                r = pg.LinearRegionItem(values=(0, 0), movable=False, brush=pg.mkBrush(255, 90, 95, 45),
                                        pen=pg.mkPen(None))
                r.setZValue(-10)
                r.hide()
                plot.addItem(r)
                pool.append(r)
            self.regions[key] = pool

        bottom = QtWidgets.QHBoxLayout()
        self.pause_btn = QtWidgets.QPushButton("Pause display")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._on_pause)
        bottom.addWidget(self.pause_btn)
        recal = QtWidgets.QPushButton("Recalibrate gyro")
        recal.clicked.connect(self.pipe.imu.recalibrate)
        bottom.addWidget(recal)
        if self.args.simulate:
            g = self.source.gen
            for text, attr in (("Simulate motion", "motion"), ("Simulate finger off", "finger_off"),
                               ("Simulate leads off", "leads_off")):
                b = QtWidgets.QPushButton(text)
                b.setCheckable(True)
                b.toggled.connect(lambda on, a=attr: setattr(g, a, bool(on)))
                bottom.addWidget(b)
        rec_text = (f"Recording to: {self.recorder.base}_*.csv in {self.args.data_dir}"
                    if self.recorder else "Recording: OFF (--no-log)")
        self.rec_label = QtWidgets.QLabel(rec_text)
        self.rec_label.setStyleSheet("font-size: 11px; color: #8a949e;")
        bottom.addWidget(self.rec_label, 1)
        vbox.addLayout(bottom)

    @staticmethod
    def _big(text, color):
        lab = QtWidgets.QLabel(text)
        lab.setStyleSheet(f"font-size: 54px; font-weight: bold; color: {color};")
        lab.setAlignment(QtCore.Qt.AlignHCenter)
        lab.setMinimumWidth(110)
        return lab

    def _on_pause(self, on):
        self.paused = bool(on)
        self.pause_btn.setText("Resume display" if on else "Pause display")

    def _sync_meta(self):
        """Take rates and ranges from the firmware's '# *_META' lines when they differ from ours."""
        s = self.source
        meta = (dict(s.imu_meta), dict(s.ppg_meta), dict(s.ecg_meta))
        if meta == self._last_meta:
            return
        self._last_meta = meta
        imu, ppg, _ = meta
        try:
            fs = float(imu["fs_hz"]) if "fs_hz" in imu else None
            ar = int(float(imu["accel_range_g"])) if "accel_range_g" in imu else None
            gr = int(float(imu["gyro_range_dps"])) if "gyro_range_dps" in imu else None
            self.pipe.imu.configure(fs=fs, accel_range_g=ar, gyro_range_dps=gr)
        except (ValueError, KeyError):
            pass
        try:
            if "fs_hz" in ppg:
                fs = float(ppg["fs_hz"])
                if fs > 0 and abs(fs - self.pipe.ppg.fs) > 1e-6:
                    self.pipe.ppg.configure(fs)
        except ValueError:
            pass

    # ---- per-frame work -----------------------------------------------
    def _on_timer(self):
        try:
            self._update()
        except Exception:
            if not self._err_printed:       # print once, keep the app alive
                traceback.print_exc()
                self._err_printed = True

    @staticmethod
    def _drain(dq):
        n = len(dq)
        if not n:
            return None
        return np.array([dq.popleft() for _ in range(n)], dtype=np.int64)

    def _update(self):
        src, pipe, rec = self.source, self.pipe, self.recorder
        for m in src.pop_messages():
            if "_META" not in m:
                print("[device]", m)
        self._sync_meta()

        e = self._drain(src.ecg_samples)
        if e is not None:
            res = pipe.push_ecg(e[:, 0], e[:, 1], e[:, 2] | e[:, 3])
            if rec:
                rec.write_ecg(e[:, 0], e[:, 1], res, pipe.ecg.bpm, e[:, 2] | e[:, 3])
        i = self._drain(src.imu_samples)
        if i is not None:
            res = pipe.push_imu(i[:, 0], i[:, 1:7])
            if rec:
                rec.write_imu(i[:, 0], res)
        p = self._drain(src.ppg_samples)
        if p is not None:
            res = pipe.push_ppg(p[:, 0], p[:, 1], p[:, 2])
            if rec:
                rec.write_ppg(p[:, 0], p[:, 1], p[:, 2], res, pipe.ppg, pipe.ppg.quality())
        new = pipe.resolve()
        if rec and new:
            rec.write_beats(new)

        self._frame += 1
        self._update_labels()
        if not self.paused:
            self._redraw()

    # ---- labels -----------------------------------------------------------
    def _update_labels(self):
        pipe = self.pipe
        eq, pq, iq = pipe.ecg.quality(), pipe.ppg.quality(), pipe.imu.status
        span = lambda name, q: f'{name} <span style="color:{COLORS.get(q, "#ff5a5f")}">{q}</span>'
        self.quality_label.setText("&nbsp;&nbsp;|&nbsp;&nbsp;".join(
            (span("ECG", eq), span("PPG", pq), span("IMU", iq))))

        e_bpm, p_bpm, d, agree = pipe.hr_agreement()
        self.ecg_bpm_label.setText("--" if e_bpm is None else f"{e_bpm:.0f}")
        if p_bpm is None:
            self.ppg_bpm_label.setText("--")
        else:
            flag = "?" if pq in ("NOISY", "LOW SIGNAL") else ""
            self.ppg_bpm_label.setText(f"{p_bpm:.0f}{flag}")
        if d is None:
            self.agree_label.setText("Heart rate check: need both ECG and PPG rates")
        elif agree is None:
            self.agree_label.setText(f"Heart rate check: ECG {e_bpm:.0f} vs PPG {p_bpm:.0f} (PPG not clean, not compared)")
        elif agree:
            self.agree_label.setText(f"Heart rate check: ECG {e_bpm:.0f} vs PPG {p_bpm:.0f}  (diff {d:+.1f})  AGREE")
        else:
            self.agree_label.setText(f"Heart rate check: ECG {e_bpm:.0f} vs PPG {p_bpm:.0f}  (diff {d:+.1f})  DISAGREE: "
                                     "one sensor is probably unreliable right now")

        s = pipe.summary(30)
        if s["pat_peak_med"] is None:
            txt = f"PAT: waiting for valid beats ({s['n_valid']}/{s['n_total']} valid so far, need 3)"
        else:
            sl = "" if s["pat_slope_med"] is None else f"    R to upstroke {s['pat_slope_med']:.0f} ms"
            txt = (f"PAT  R to peak {s['pat_peak_med']:.0f} +- {s['pat_peak_std']:.0f} ms{sl}"
                   f"    ({s['n_valid']}/{s['n_total']} of the last beats valid)")
        self.pat_label.setText(txt)

        hint = ""
        if pipe.beats:
            b = pipe.beats[-1]
            hint = f"Last beat: {REASON_TEXT.get(b.reason, b.reason)}"
        if pipe.imu.status == "CALIBRATING":
            hint = "Gyro calibration (3 s): keep the sensor still. " + hint
        self.hint_label.setText(hint)

        if self._frame % 10 == 0:
            self.status_label.setText(self._status_text())

    def _status_text(self):
        s, pipe = self.source, self.pipe
        if s.error:
            self.status_label.setStyleSheet("font-size: 12px; color: #ff5a5f;")
            return f"ERROR: {s.error}"
        parts = [s.description]

        def rate(t):
            if t.size > 20:
                dt = np.median(np.diff(t[-300:]))
                if dt > 0:
                    return f"{1e6 / dt:.0f}"
            return "-"
        parts.append(f"fs ECG {rate(pipe.ecg.t_us)} / IMU {rate(pipe.imu.t_us)} / PPG {rate(pipe.ppg.t_us)} Hz")
        parts.append(f"samples {pipe.ecg.total_samples}/{pipe.imu.total_samples}/{pipe.ppg.total_samples}")
        parts.append(f"dropped ~{pipe.ecg.dropped_est}/{pipe.imu.dropped_est}/{pipe.ppg.dropped_est}")
        bl = s.bad_lines_by_stream()
        parts.append(f"bad lines E{bl['ecg']} I{bl['imu']} P{bl['ppg']} other{bl['other']}")
        m = s.all_meta
        if m:
            parts.append(f"fw queue max {m.get('queue_max', '?')}/{m.get('queue_len', '?')}")
        parts.append(f"i2c errors IMU {s.imu_meta.get('i2c_errors', '?')} PPG {s.ppg_meta.get('i2c_errors', '?')}")
        if not s.connected and pipe.ecg.total_samples == 0:
            parts.append("connecting...")
        return " | ".join(parts)

    # ---- drawing ------------------------------------------------------------
    @staticmethod
    def _fit_y(plot, y, lo_pct=0.5, hi_pct=99.5, pad=0.15, min_span=50.0):
        if y.size < 10:
            return None
        lo, hi = np.percentile(y, [lo_pct, hi_pct])
        span = max(hi - lo, min_span)
        mid = 0.5 * (hi + lo)
        lo_y, hi_y = mid - (0.5 + pad) * span, mid + (0.5 + pad) * span
        plot.setYRange(lo_y, hi_y, padding=0)
        return lo_y

    def _motion_segments(self, t_ref):
        imu = self.pipe.imu
        if imu.t_us.size < 2:
            return []
        t = (imu.t_us - t_ref) / 1e6
        m = imu.motion & (t >= -self.args.window - 1.0)
        if not m.any():
            return []
        d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
        return [(float(t[a]), float(t[b])) for a, b in zip(starts, ends)][-MAX_REGIONS:]

    def _redraw(self):
        pipe = self.pipe
        ecg, ppg, imu = pipe.ecg, pipe.ppg, pipe.imu
        ends = [a.t_us[-1] for a in (ecg, ppg, imu) if a.t_us.size]
        if not ends:
            return
        t_ref = max(ends)                    # all three streams use the ESP32 clock
        w = self.args.window

        # ECG
        if ecg.t_us.size > 2:
            tr = (ecg.t_us - t_ref) / 1e6
            m = tr >= -w
            self.ecg_curve.setData(tr[m], ecg.filt[m], skipFiniteCheck=True)
            if ecg.peak_t_us:
                pt = (np.asarray(ecg.peak_t_us, dtype=np.int64) - t_ref) / 1e6
                pt = pt[pt >= -w]
                self.r_scatter.setData(pt, np.interp(pt, tr, ecg.filt)) if pt.size else self.r_scatter.setData([], [])
            else:
                self.r_scatter.setData([], [])
            if self._frame % 15 == 0:
                if ecg.leads_off:
                    self.ecg_plot.setYRange(-300, 300, padding=0)
                else:
                    self._fit_y(self.ecg_plot, ecg.filt[m][-int(4 * ecg.fs):], min_span=200.0)

        # PPG
        if ppg.t_us.size > 2:
            tr = (ppg.t_us - t_ref) / 1e6
            m = tr >= -w
            self.ppg_curve.setData(tr[m], ppg.ppg[m], skipFiniteCheck=True)
            if ppg.peak_t_us:
                pt = (np.asarray(ppg.peak_t_us, dtype=np.int64) - t_ref) / 1e6
                pt = pt[pt >= -w]
                self.pk_scatter.setData(pt, np.interp(pt, tr, ppg.ppg)) if pt.size else self.pk_scatter.setData([], [])
            else:
                self.pk_scatter.setData([], [])
            if self._frame % 15 == 0:
                if ppg.finger:
                    y0 = self._fit_y(self.ppg_plot, ppg.ppg[m][-int(4 * ppg.fs):], min_span=40.0)
                    if y0 is not None:
                        self._ppg_ymin = y0
                else:
                    self.ppg_plot.setYRange(-100, 100, padding=0)
                    self._ppg_ymin = -90.0
            if ecg.peak_t_us:      # R peaks as triangles at the bottom of the PPG plot
                rt = (np.asarray(ecg.peak_t_us, dtype=np.int64) - t_ref) / 1e6
                rt = rt[rt >= -w]
                self.rmark_scatter.setData(rt, np.full(rt.size, self._ppg_ymin + 0.06 * abs(self._ppg_ymin) + 2.0))
            else:
                self.rmark_scatter.setData([], [])

        # IMU motion index
        if imu.t_us.size > 2:
            tr = (imu.t_us - t_ref) / 1e6
            m = tr >= -w
            self.imu_curve.setData(tr[m], imu.accel_idx[m], skipFiniteCheck=True)
            if self._frame % 15 == 0:
                top = float(max(np.max(imu.accel_idx[m]) * 1.2, self.args.motion_thr * 2.0, 0.02))
                self.imu_plot.setYRange(0, top, padding=0)

        # shaded motion regions
        segs = self._motion_segments(t_ref)
        for pool in self.regions.values():
            for k, r in enumerate(pool):
                if k < len(segs):
                    r.setRegion(segs[k])
                    r.show()
                else:
                    r.hide()

        # PAT trend
        self._redraw_pat(t_ref)

    def _redraw_pat(self, t_ref):
        beats = self.pipe.beats
        pw = self.args.pat_window
        good_x, good_y, bad_x, bad_y, sl_x, sl_y = [], [], [], [], [], []
        for b in beats[-200:]:
            x = (b.r_t_us - t_ref) / 1e6
            if x < -pw or b.pat_peak_ms != b.pat_peak_ms:
                continue
            if b.valid:
                good_x.append(x)
                good_y.append(b.pat_peak_ms)
                if b.pat_slope_ms == b.pat_slope_ms:
                    sl_x.append(x)
                    sl_y.append(b.pat_slope_ms)
            else:
                bad_x.append(x)
                bad_y.append(b.pat_peak_ms)
        self.pat_good.setData(good_x, good_y)
        self.pat_bad.setData(bad_x, bad_y)
        self.pat_slope.setData(sl_x, sl_y)
        if self._frame % 15 == 0:
            ys = good_y + sl_y
            if len(ys) >= 3:
                lo, hi = min(ys), max(ys)
                span = max(hi - lo, 40.0)
                mid = 0.5 * (lo + hi)
                self.pat_plot.setYRange(mid - 0.75 * span, mid + 0.75 * span, padding=0)
            elif not ys:
                self.pat_plot.setYRange(0, 500, padding=0)
            s = self.pipe.summary(30)
            if s["pat_peak_med"] is not None:
                self.pat_line.setValue(s["pat_peak_med"])
                self.pat_line.show()
            else:
                self.pat_line.hide()

    # ---- shutdown -------------------------------------------------------------
    def closeEvent(self, event):
        self.timer.stop()
        self.source.stop()
        if self.recorder:
            self.recorder.close()
            for p in self.recorder.paths.values():
                print(f"Saved: {p}")
        event.accept()


# ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Live ECG + PPG + IMU fusion viewer for the combined ESP32 firmware")
    ap.add_argument("--port", help="serial port, e.g. /dev/ttyUSB0 or COM5 (auto-detect if omitted)")
    ap.add_argument("--baud", type=int, default=460800, help="must match the firmware (default 460800)")
    ap.add_argument("--ecg-fs", type=float, default=250.0, help="ECG sampling rate in the firmware (Hz)")
    ap.add_argument("--imu-fs", type=float, default=100.0, help="IMU rate (the firmware's IMU_META overrides it)")
    ap.add_argument("--ppg-fs", type=float, default=100.0, help="PPG rate (the firmware's PPG_META overrides it)")
    ap.add_argument("--accel-range", type=int, default=4, help="accel range in g (IMU_META overrides)")
    ap.add_argument("--gyro-range", type=int, default=500, help="gyro range in deg/s (IMU_META overrides)")
    ap.add_argument("--window", type=float, default=8.0, help="seconds shown on the signal plots")
    ap.add_argument("--pat-window", type=float, default=60.0, help="seconds shown on the PAT trend")
    ap.add_argument("--notch", type=float, default=50.0, help="mains notch in Hz, 0 disables (Bangladesh = 50)")
    ap.add_argument("--finger-thr", type=float, default=30000.0, help="IR counts that mean a finger is present")
    ap.add_argument("--motion-thr", type=float, default=0.05, help="accel motion threshold (g)")
    ap.add_argument("--gyro-thr", type=float, default=15.0, help="gyro motion threshold (deg/s)")
    ap.add_argument("--subject", default="S01", help="non-identifying subject code")
    ap.add_argument("--activity", default="rest", help="activity/condition label stored in the CSV")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="where session CSV files go")
    ap.add_argument("--no-log", action="store_true", help="do not write CSV files")
    ap.add_argument("--simulate", action="store_true", help="use built-in synthetic sensors, no hardware")
    ap.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list_ports:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports found.")
        for dev, desc in ports:
            print(f"{dev}\t{desc}")
        return 0

    pg.setConfigOptions(antialias=False, background="#101418", foreground="#d0d6dc")
    pg.mkQApp("Fusion viewer")
    try:
        win = FusionWindow(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    win.show()
    pg.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())