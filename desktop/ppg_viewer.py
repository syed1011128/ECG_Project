#!/usr/bin/env python3
"""
ppg_viewer.py - live MAX30102 PPG viewer for ESP32

    python ppg_viewer.py --list-ports
    python ppg_viewer.py --simulate                       # no hardware needed
    python ppg_viewer.py --port /dev/ttyUSB0 --subject S01 --activity rest

Shows: pulse waveform (filtered IR, pulse pointing up) with peak markers, raw IR and
raw RED counts, BPM, perfusion index, finger / signal-quality label and an
EXPERIMENTAL, uncalibrated SpO2 estimate.
Everything is logged to ../data/ppg_session_YYYYMMDD_HHMM.csv (+ _beats.csv).

Research/learning prototype only - not a medical device. The SpO2 number is NOT
calibrated or validated and must not be used for any health decision.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Linux/GNOME fix: force Qt's generic theme (avoids the 'antialiasing' gsettings crash).
# Must run BEFORE Qt is imported. Opt out with ECG_KEEP_QT_THEME=1
if sys.platform.startswith("linux") and not os.environ.get("ECG_KEEP_QT_THEME"):
    os.environ["QT_QPA_PLATFORMTHEME"] = "generic"

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ppg_processing import ADC_MAX, PPGPipeline
from ppg_sources import PPGSerialSource, PPGSimSource
from sources import find_serial_port, list_serial_ports

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

QUALITY_COLORS = {"OK": "#3ddc84", "SETTLING": "#f5c542", "NO DATA": "#8a949e",
                  "NO FINGER": "#ff9f43", "LOW SIGNAL": "#ff9f43", "NOISY": "#ff9f43",
                  "SATURATED": "#ff5a5f"}
HINTS = {
    "NO FINGER": "Place a fingertip gently on the sensor (IR level is below the finger threshold).",
    "SETTLING": "Keep the finger still and relaxed for a few seconds.",
    "LOW SIGNAL": "Pulse is very weak: adjust pressure/position, warm the finger, or raise the LED current in the firmware.",
    "NOISY": "Pulse intervals are irregular: hold still, check contact pressure (not too hard, not loose).",
    "SATURATED": "Counts hit full scale: lower the LED current / ADC range in the firmware, shield from bright light.",
}


# ----------------------------------------------------------------------
# CSV recording
# ----------------------------------------------------------------------
class PpgRecorder:
    HEADER = ["timestamp_us", "red", "ir", "ppg_filt", "bpm", "pi_pct", "r_ratio",
              "spo2_est_experimental", "finger", "quality"]

    def __init__(self, folder, args, source_desc):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        base, k = f"ppg_session_{stamp}", 1
        while (folder / f"{base}.csv").exists():
            k += 1
            base = f"ppg_session_{stamp}_{k}"
        self.path = folder / f"{base}.csv"
        self.beats_path = folder / f"{base}_beats.csv"

        self._f = open(self.path, "w", newline="")
        for key, val in (("subject", args.subject), ("activity", args.activity),
                         ("source", source_desc), ("fs_nominal_hz", args.fs),
                         ("finger_thr_counts", args.finger_thr),
                         ("note", "ppg_filt = -(causal 0.5-8 Hz band-pass of IR); spo2_est is uncalibrated "
                                  "and only filled while quality == OK; filter rows on the quality column"),
                         ("started", datetime.now().isoformat(timespec="seconds"))):
            self._f.write(f"# {key}={val}\n")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)

        self._fb = open(self.beats_path, "w", newline="")
        self._wb = csv.writer(self._fb)
        self._wb.writerow(["peak_timestamp_us", "ibi_s", "bpm_instant", "amp_counts"])
        self._last_flush = time.time()

    @staticmethod
    def _fmt(v, spec):
        return "" if v is None else format(v, spec)

    def write(self, t_us, red, ir, res, pipe, quality):
        bpm = self._fmt(pipe.bpm, ".1f")
        pi = self._fmt(pipe.pi_pct, ".3f")
        r = self._fmt(pipe.r_ratio, ".4f")
        sp = self._fmt(pipe.spo2_est if quality == "OK" else None, ".1f")   # only when the signal is OK
        for i in range(len(t_us)):
            self._w.writerow((int(t_us[i]), int(red[i]), int(ir[i]), f"{res.ppg[i]:.2f}",
                              bpm, pi, r, sp, int(res.finger[i]), quality))
        if time.time() - self._last_flush > 1.0:
            self._f.flush()
            self._fb.flush()
            self._last_flush = time.time()

    def write_beats(self, beats):
        for t_us, ibi, bpm, amp in beats:
            self._wb.writerow((int(t_us), "" if ibi != ibi else f"{ibi:.4f}",
                               "" if bpm != bpm else f"{bpm:.1f}", f"{amp:.1f}"))

    def close(self):
        for f in (self._f, self._fb):
            try:
                f.close()
            except Exception:
                pass


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
class PpgWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.paused = False
        self._frame = 0
        self._err_printed = False
        self._last_meta = {}

        self.pipe = PPGPipeline(fs=args.fs, finger_thr=args.finger_thr)
        self.source = self._start_source()
        self.recorder = None
        if not args.no_log:
            self.recorder = PpgRecorder(args.data_dir, args, self.source.description)

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(30)

    # ---- setup ---------------------------------------------------------
    def _start_source(self):
        a = self.args
        if a.simulate:
            src = PPGSimSource(fs=a.fs)
        else:
            port = a.port or find_serial_port()
            if not port:
                raise RuntimeError("No serial port found. Plug in the ESP32 (check the USB cable "
                                   "carries data) or run with --simulate.")
            src = PPGSerialSource(port, a.baud)
        src.start()
        return src

    def _make_plot(self, row, ylabel, bottom=False):
        p = self.glw.addPlot(row=row, col=0)
        p.setLabel("left", ylabel)
        if bottom:
            p.setLabel("bottom", "Time (s)")
        p.showGrid(x=True, y=True, alpha=0.25)
        p.setXRange(-self.args.window, 0, padding=0)
        p.setMouseEnabled(x=False, y=False)
        p.hideButtons()
        return p

    def _build_ui(self):
        self.setWindowTitle("ESP32 + MAX30102 live PPG")
        self.resize(1150, 860)
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
        self.bpm_label = QtWidgets.QLabel("--")
        self.bpm_label.setStyleSheet("font-size: 64px; font-weight: bold; color: #ff5a5f;")
        unit = QtWidgets.QLabel("BPM")
        unit.setStyleSheet("font-size: 20px; color: #9aa5b1;")
        top.addWidget(self.bpm_label)
        top.addWidget(unit)
        top.addSpacing(30)

        info = QtWidgets.QVBoxLayout()
        self.quality_label = QtWidgets.QLabel("Signal: --")
        self.quality_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.values_label = QtWidgets.QLabel("")
        self.values_label.setStyleSheet("font-size: 14px; color: #c3ccd5;")
        self.hint_label = QtWidgets.QLabel("")
        self.hint_label.setStyleSheet("font-size: 12px; color: #9aa5b1;")
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("font-size: 12px; color: #8a949e;")
        for w in (self.quality_label, self.values_label, self.hint_label, self.status_label):
            info.addWidget(w)
        top.addLayout(info)
        top.addStretch(1)
        vbox.addLayout(top)

        self.glw = pg.GraphicsLayoutWidget()
        vbox.addWidget(self.glw, 1)

        self.pulse_plot = self._make_plot(0, "Pulse (counts)")
        self.ir_plot = self._make_plot(1, "Raw IR (counts)")
        self.red_plot = self._make_plot(2, "Raw RED (counts)", bottom=True)
        self.pulse_curve = self.pulse_plot.plot(pen=pg.mkPen("#4cc9f0", width=1.8))
        self.peak_scatter = pg.ScatterPlotItem(size=10, pen=None, brush=pg.mkBrush(255, 90, 95, 220))
        self.pulse_plot.addItem(self.peak_scatter)
        self.ir_curve = self.ir_plot.plot(pen=pg.mkPen("#c084fc", width=1.2))
        self.red_curve = self.red_plot.plot(pen=pg.mkPen("#ff6b6b", width=1.2))
        self.ir_plot.addItem(pg.InfiniteLine(pos=self.args.finger_thr, angle=0,
                             pen=pg.mkPen("#6c7a89", style=QtCore.Qt.DashLine)))
        for row, stretch in enumerate((4, 2, 2)):
            self.glw.ci.layout.setRowStretchFactor(row, stretch)

        bottom = QtWidgets.QHBoxLayout()
        self.pause_btn = QtWidgets.QPushButton("Pause display")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._on_pause)
        bottom.addWidget(self.pause_btn)
        if self.args.simulate:
            self.off_btn = QtWidgets.QPushButton("Simulate finger off")
            self.off_btn.setCheckable(True)
            self.off_btn.toggled.connect(lambda on: setattr(self.source, "finger_off", bool(on)))
            bottom.addWidget(self.off_btn)
            self.motion_btn = QtWidgets.QPushButton("Simulate motion")
            self.motion_btn.setCheckable(True)
            self.motion_btn.toggled.connect(lambda on: setattr(self.source, "motion", bool(on)))
            bottom.addWidget(self.motion_btn)
        rec_text = f"Recording to: {self.recorder.path}" if self.recorder else "Recording: OFF (--no-log)"
        self.rec_label = QtWidgets.QLabel(rec_text)
        self.rec_label.setStyleSheet("font-size: 11px; color: #8a949e;")
        bottom.addWidget(self.rec_label, 1)
        vbox.addLayout(bottom)

    def _on_pause(self, on):
        self.paused = bool(on)
        self.pause_btn.setText("Resume display" if on else "Pause display")

    def _sync_meta(self):
        """Take fs from the firmware's '# PPG_META' line when it differs from our setting."""
        meta = dict(self.source.meta)
        if meta == self._last_meta:
            return
        self._last_meta = meta
        try:
            if "fs_hz" in meta:
                fs = float(meta["fs_hz"])
                if fs > 0 and abs(fs - self.pipe.fs) > 1e-6:
                    self.pipe.configure(fs)
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

    def _update(self):
        src = self.source
        while src.messages:
            print("[device]", src.messages.popleft())
        self._sync_meta()

        n = len(src.samples)
        if n:
            rows = np.array([src.samples.popleft() for _ in range(n)], dtype=np.int64)
            res = self.pipe.push(rows[:, 0], rows[:, 1], rows[:, 2])
            if self.recorder:
                self.recorder.write(rows[:, 0], rows[:, 1], rows[:, 2], res, self.pipe, self.pipe.quality())
                if res.new_beats:
                    self.recorder.write_beats(res.new_beats)

        self._frame += 1
        self._update_labels()
        if not self.paused:
            self._redraw()

    def _update_labels(self):
        p = self.pipe
        bpm = p.bpm
        q = p.quality()
        color = QUALITY_COLORS.get(q, "#ff5a5f")
        if bpm is None:
            self.bpm_label.setText("--")
            self.bpm_label.setStyleSheet("font-size: 64px; font-weight: bold; color: #ff5a5f;")
        elif q in ("NOISY", "LOW SIGNAL"):          # shown, but flagged as unreliable
            self.bpm_label.setText(f"{bpm:.0f}?")
            self.bpm_label.setStyleSheet("font-size: 64px; font-weight: bold; color: #8a949e;")
        else:
            self.bpm_label.setText(f"{bpm:.0f}")
            self.bpm_label.setStyleSheet("font-size: 64px; font-weight: bold; color: #ff5a5f;")
        self.quality_label.setText("Signal: " + q)
        self.quality_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
        self.hint_label.setText(HINTS.get(q, ""))

        pi = "--" if p.pi_pct is None else f"{p.pi_pct:.2f} %"
        r = "--" if p.r_ratio is None else f"{p.r_ratio:.2f}"
        sp = "--" if (p.spo2_est is None or q != "OK") else f"{p.spo2_est:.0f} %"
        self.values_label.setText(f"PI {pi}    R {r}    SpO2 {sp}  (EXPERIMENTAL, uncalibrated: do not use for health decisions)")

        if self._frame % 10 == 0:
            fs_est = ""
            if p.t_us.size > 20:
                dt = np.median(np.diff(p.t_us[-300:]))
                if dt > 0:
                    fs_est = f" | fs ~ {1e6 / dt:.1f} Hz"
            s = self.source
            m = s.meta
            led = f"LED R/IR {m.get('led_red_ma', '?')}/{m.get('led_ir_ma', '?')} mA"
            status = (f"{s.description}{fs_est} | {led} | part_id {m.get('part_id', '?')} | "
                      f"samples {p.total_samples} | dropped ~{p.dropped_est} | bad lines {s.bad_lines} | "
                      f"i2c errors {m.get('i2c_errors', '?')} | fifo ovf {m.get('fifo_ovf', '?')}")
            if s.error:
                status = f"ERROR: {s.error}"
                self.status_label.setStyleSheet("font-size: 13px; color: #ff5a5f;")
            elif not s.connected and p.total_samples == 0:
                status += " | connecting..."
            self.status_label.setText(status)

    @staticmethod
    def _fit_y(plot, y, lo_pct=0.5, hi_pct=99.5, pad=0.15, min_span=50.0):
        lo, hi = np.percentile(y, [lo_pct, hi_pct])
        span = max(hi - lo, min_span)
        mid = 0.5 * (hi + lo)
        plot.setYRange(mid - (0.5 + pad) * span, mid + (0.5 + pad) * span, padding=0)

    def _redraw(self):
        p = self.pipe
        if p.t_us.size < 2:
            return
        t_end = p.t_us[-1]
        t_rel = (p.t_us - t_end) / 1e6
        m = t_rel >= -self.args.window
        tr, pu, ir, red = t_rel[m], p.ppg[m], p.ir[m], p.red[m]
        self.pulse_curve.setData(tr, pu, skipFiniteCheck=True)
        self.ir_curve.setData(tr, ir, skipFiniteCheck=True)
        self.red_curve.setData(tr, red, skipFiniteCheck=True)

        if p.peak_t_us:
            pt = (np.asarray(p.peak_t_us, dtype=np.int64) - t_end) / 1e6
            pt = pt[pt >= -self.args.window]
            if pt.size:
                self.peak_scatter.setData(pt, np.interp(pt, tr, pu))
            else:
                self.peak_scatter.setData([], [])
        else:
            self.peak_scatter.setData([], [])

        if self._frame % 15 == 0 and tr.size > 100:
            if p.finger:
                self._fit_y(self.pulse_plot, pu[-int(4 * p.fs):], min_span=40.0)
            else:
                self.pulse_plot.setYRange(-100, 100, padding=0)
            hi = float(max(np.max(ir), self.args.finger_thr * 1.2))
            self.ir_plot.setYRange(0, min(hi * 1.1, ADC_MAX * 1.02), padding=0)
            self._fit_y(self.red_plot, red, 0.0, 100.0, pad=0.1, min_span=2000.0)

    # ---- shutdown ------------------------------------------------------
    def closeEvent(self, event):
        self.timer.stop()
        self.source.stop()
        if self.recorder:
            self.recorder.close()
            print(f"Saved: {self.recorder.path}")
            print(f"Saved: {self.recorder.beats_path}")
        event.accept()


# ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Live MAX30102 PPG viewer for ESP32")
    ap.add_argument("--port", help="serial port, e.g. /dev/ttyUSB0 or COM5 (auto-detect if omitted)")
    ap.add_argument("--baud", type=int, default=230400, help="must match the firmware (default 230400)")
    ap.add_argument("--fs", type=float, default=100.0, help="sampling rate set in the firmware (Hz); "
                    "the firmware's PPG_META line overrides it")
    ap.add_argument("--window", type=float, default=8.0, help="seconds shown on screen")
    ap.add_argument("--finger-thr", type=float, default=30000.0,
                    help="IR counts above this mean a finger is on the sensor (default 30000, tune it)")
    ap.add_argument("--subject", default="S01", help="non-identifying subject code")
    ap.add_argument("--activity", default="rest", help="activity/condition label stored in the CSV")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="where session CSV files go")
    ap.add_argument("--no-log", action="store_true", help="do not write CSV files")
    ap.add_argument("--simulate", action="store_true", help="use built-in synthetic PPG, no hardware")
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
    pg.mkQApp("PPG viewer")
    try:
        win = PpgWindow(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    win.show()
    pg.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())