#!/usr/bin/env python3
"""
viewer.py - live ECG viewer for ESP32 + AD8232

    python viewer.py --list-ports
    python viewer.py --simulate                       # no hardware needed
    python viewer.py --port COM5 --subject S01 --activity rest

Shows: raw ADC trace, filtered ECG (0.5-40 Hz + 50 Hz notch, causal), R-peak
markers, BPM, short-window HRV, signal-quality label and leads-off warning.
Everything is logged to ../data/session_YYYYMMDD_HHMM.csv (+ _beats.csv).

Research/learning prototype only - not a medical device.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from processing import ECGPipeline
from sources import SerialSource, SimSource, find_serial_port, list_serial_ports

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ----------------------------------------------------------------------
# CSV recording
# ----------------------------------------------------------------------
class Recorder:
    """Writes every sample (raw + filtered + BPM + leads-off) and every detected beat."""

    def __init__(self, folder, args, source_desc):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        base, k = f"session_{stamp}", 1
        while (folder / f"{base}.csv").exists():
            k += 1
            base = f"session_{stamp}_{k}"
        self.path = folder / f"{base}.csv"
        self.beats_path = folder / f"{base}_beats.csv"

        self._f = open(self.path, "w", newline="")
        self._f.write(f"# subject={args.subject}\n")
        self._f.write(f"# activity={args.activity}\n")
        self._f.write(f"# source={source_desc}\n")
        self._f.write(f"# fs_nominal_hz={args.fs}\n")
        self._f.write(f"# notch_hz={args.notch}\n")
        self._f.write(f"# started={datetime.now().isoformat(timespec='seconds')}\n")
        self._w = csv.writer(self._f)
        self._w.writerow(["timestamp_us", "raw_adc", "filtered", "bpm", "leads_off"])

        self._fb = open(self.beats_path, "w", newline="")
        self._wb = csv.writer(self._fb)
        self._wb.writerow(["r_peak_timestamp_us", "rr_s", "bpm_instant"])
        self._last_flush = time.time()

    def write(self, t_us, raw, filt, bpm, leads_off):
        bpm_s = "" if bpm is None else f"{bpm:.1f}"
        for i in range(len(t_us)):
            self._w.writerow((int(t_us[i]), int(raw[i]), f"{filt[i]:.2f}", bpm_s, int(leads_off[i])))
        if time.time() - self._last_flush > 1.0:
            self._f.flush()
            self._fb.flush()
            self._last_flush = time.time()

    def write_beats(self, beats):
        for t_us, rr, bpm in beats:
            self._wb.writerow((int(t_us),
                               "" if rr != rr else f"{rr:.4f}",
                               "" if bpm != bpm else f"{bpm:.1f}"))

    def close(self):
        for f in (self._f, self._fb):
            try:
                f.close()
            except Exception:
                pass


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
QUALITY_COLORS = {"OK": "#3ddc84", "SETTLING": "#f5c542", "NO DATA": "#8a949e"}


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.paused = False
        self._frame = 0
        self._err_printed = False

        self.pipe = ECGPipeline(fs=args.fs, notch_hz=(args.notch or None))
        self.source = self._start_source()
        self.recorder = None
        if not args.no_log:
            self.recorder = Recorder(args.data_dir, args, self.source.description)

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(30)

    # ---- setup ---------------------------------------------------------
    def _start_source(self):
        a = self.args
        if a.simulate:
            src = SimSource(fs=a.fs)
        else:
            port = a.port or find_serial_port()
            if not port:
                raise RuntimeError("No serial port found. Plug in the ESP32 (check the USB cable "
                                   "carries data) or run with --simulate.")
            src = SerialSource(port, a.baud)
        src.start()
        return src

    def _build_ui(self):
        self.setWindowTitle("ESP32 + AD8232 live ECG")
        self.resize(1150, 760)
        # fixed dark theme so the labels stay readable on any OS theme
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

        # top row: BPM + labels
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
        self.hrv_label = QtWidgets.QLabel("HRV (short window): --")
        self.hrv_label.setStyleSheet("font-size: 14px; color: #c3ccd5;")
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("font-size: 12px; color: #8a949e;")
        for w in (self.quality_label, self.hrv_label, self.status_label):
            info.addWidget(w)
        top.addLayout(info)
        top.addStretch(1)
        vbox.addLayout(top)

        # plots
        self.glw = pg.GraphicsLayoutWidget()
        vbox.addWidget(self.glw, 1)

        self.ecg_plot = self.glw.addPlot(row=0, col=0)
        self.ecg_plot.setLabel("left", "Filtered ECG (ADC counts)")
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.25)
        self.ecg_plot.setXRange(-self.args.window, 0, padding=0)
        self.ecg_plot.setMouseEnabled(x=False, y=False)
        self.ecg_plot.hideButtons()
        self.ecg_curve = self.ecg_plot.plot(pen=pg.mkPen("#4cc9f0", width=1.6))
        self.peak_scatter = pg.ScatterPlotItem(size=10, pen=None, brush=pg.mkBrush(255, 90, 95, 220))
        self.ecg_plot.addItem(self.peak_scatter)

        self.raw_plot = self.glw.addPlot(row=1, col=0)
        self.raw_plot.setLabel("left", "Raw ADC (0-4095)")
        self.raw_plot.setLabel("bottom", "Time (s)")
        self.raw_plot.showGrid(x=True, y=True, alpha=0.2)
        self.raw_plot.setXRange(-self.args.window, 0, padding=0)
        self.raw_plot.setYRange(0, 4095, padding=0.02)
        self.raw_plot.setMouseEnabled(x=False, y=False)
        self.raw_plot.hideButtons()
        self.raw_curve = self.raw_plot.plot(pen=pg.mkPen("#a0a8b0", width=1.0))
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 1)

        # bottom row: buttons + recording info
        bottom = QtWidgets.QHBoxLayout()
        self.pause_btn = QtWidgets.QPushButton("Pause display")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._on_pause)
        bottom.addWidget(self.pause_btn)
        if self.args.simulate:
            self.sim_btn = QtWidgets.QPushButton("Simulate leads-off")
            self.sim_btn.setCheckable(True)
            self.sim_btn.toggled.connect(lambda on: setattr(self.source, "leads_off", bool(on)))
            bottom.addWidget(self.sim_btn)
        rec_text = f"Recording to: {self.recorder.path}" if self.recorder else "Recording: OFF (--no-log)"
        self.rec_label = QtWidgets.QLabel(rec_text)
        self.rec_label.setStyleSheet("font-size: 11px; color: #8a949e;")
        bottom.addWidget(self.rec_label, 1)
        vbox.addLayout(bottom)

    def _on_pause(self, on):
        self.paused = bool(on)
        self.pause_btn.setText("Resume display" if on else "Pause display")

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

        n = len(src.samples)
        if n:
            rows = np.array([src.samples.popleft() for _ in range(n)], dtype=np.int64)
            leads_off = (rows[:, 2] | rows[:, 3]) != 0
            res = self.pipe.push(rows[:, 0], rows[:, 1], leads_off)
            if self.recorder:
                self.recorder.write(rows[:, 0], rows[:, 1], res.filtered, self.pipe.bpm, leads_off)
                if res.new_beats:
                    self.recorder.write_beats(res.new_beats)

        self._frame += 1
        self._update_labels()
        if not self.paused:
            self._redraw()

    def _update_labels(self):
        p = self.pipe
        bpm = p.bpm
        self.bpm_label.setText("--" if bpm is None else f"{bpm:.0f}")

        q = p.quality()
        color = QUALITY_COLORS.get(q, "#ff5a5f")
        text = "Signal: " + q
        if q == "LEADS OFF":
            text += "  - electrode disconnected, check RA / LA / RL"
        self.quality_label.setText(text)
        self.quality_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")

        if self._frame % 10 == 0:
            h = p.hrv()
            if h["sdnn_ms"] == h["sdnn_ms"]:
                self.hrv_label.setText(f"HRV (short window, indicative only): SDNN {h['sdnn_ms']:.0f} ms | "
                                       f"RMSSD {h['rmssd_ms']:.0f} ms | pNN50 {h['pnn50']:.0f} %")
            else:
                self.hrv_label.setText("HRV (short window): collecting beats...")

            fs_est = ""
            if p.t_us.size > 50:
                dt = np.median(np.diff(p.t_us[-500:]))
                if dt > 0:
                    fs_est = f" | fs ~ {1e6 / dt:.1f} Hz"
            s = self.source
            status = f"{s.description}{fs_est} | samples {p.total_samples} | dropped ~{p.dropped_est} | bad lines {s.bad_lines}"
            if s.error:
                status = f"ERROR: {s.error}"
                self.status_label.setStyleSheet("font-size: 13px; color: #ff5a5f;")
            elif not s.connected and p.total_samples == 0:
                status += " | connecting..."
            self.status_label.setText(status)

    def _redraw(self):
        p = self.pipe
        if p.t_us.size < 2:
            return
        t_end = p.t_us[-1]
        t_rel = (p.t_us - t_end) / 1e6
        m = t_rel >= -self.args.window
        tr, fl, rw = t_rel[m], p.filt[m], p.raw[m]
        self.ecg_curve.setData(tr, fl, skipFiniteCheck=True)
        self.raw_curve.setData(tr, rw, skipFiniteCheck=True)

        if p.peak_t_us:
            pt = (np.asarray(p.peak_t_us, dtype=np.int64) - t_end) / 1e6
            pt = pt[pt >= -self.args.window]
            if pt.size:
                self.peak_scatter.setData(pt, np.interp(pt, tr, fl))
            else:
                self.peak_scatter.setData([], [])
        else:
            self.peak_scatter.setData([], [])

        if self._frame % 15 == 0 and fl.size > 100:
            lo, hi = np.percentile(fl, [0.2, 99.8])
            span = max(hi - lo, 80.0)
            mid = 0.5 * (hi + lo)
            self.ecg_plot.setYRange(mid - 0.65 * span, mid + 0.65 * span, padding=0)

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
    ap = argparse.ArgumentParser(description="Live ECG viewer for ESP32 + AD8232")
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyUSB0 (auto-detect if omitted)")
    ap.add_argument("--baud", type=int, default=230400, help="must match the firmware (default 230400)")
    ap.add_argument("--fs", type=float, default=250.0, help="sampling rate set in the firmware (Hz)")
    ap.add_argument("--window", type=float, default=8.0, help="seconds shown on screen")
    ap.add_argument("--notch", type=float, default=50.0, help="mains notch in Hz, 0 disables (BD = 50)")
    ap.add_argument("--subject", default="S01", help="non-identifying subject code")
    ap.add_argument("--activity", default="rest", help="activity/condition label stored in the CSV")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="where session CSV files go")
    ap.add_argument("--no-log", action="store_true", help="do not write CSV files")
    ap.add_argument("--simulate", action="store_true", help="use built-in synthetic ECG, no hardware")
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
    pg.mkQApp("ECG viewer")
    try:
        win = ViewerWindow(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    win.show()
    pg.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())