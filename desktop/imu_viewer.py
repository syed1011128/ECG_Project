#!/usr/bin/env python3
"""
imu_viewer.py - live MPU6050 viewer for ESP32

    python imu_viewer.py --list-ports
    python imu_viewer.py --simulate                       # no hardware needed
    python imu_viewer.py --port /dev/ttyUSB0 --subject S01 --activity rest

Shows: accelerometer XYZ (g), gyroscope XYZ (deg/s), |a| (about 1 g when still),
motion index with threshold, and a STILL / MOTION label.
Everything is logged to ../data/imu_session_YYYYMMDD_HHMM.csv.

Research/learning prototype only. A MOTION flag means "signals in this window are
probably less reliable", not that they are invalid.
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

from imu_processing import ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS, IMUPipeline
from imu_sources import ImuSerialSource, ImuSimSource
from sources import find_serial_port, list_serial_ports

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COLORS = {"X": "#ff6b6b", "Y": "#4cc9f0", "Z": "#a3e635"}
STATUS_COLORS = {"STILL": "#3ddc84", "MOTION": "#ff9f43", "CALIBRATING": "#f5c542",
                 "SATURATED": "#ff5a5f", "NO DATA": "#8a949e"}


# ----------------------------------------------------------------------
# CSV recording
# ----------------------------------------------------------------------
class ImuRecorder:
    HEADER = ["timestamp_us", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps",
              "a_mag_g", "accel_idx_g", "gyro_idx_dps", "motion", "calibrated"]

    def __init__(self, folder, args, source_desc, pipe):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        base, k = f"imu_session_{stamp}", 1
        while (folder / f"{base}.csv").exists():
            k += 1
            base = f"imu_session_{stamp}_{k}"
        self.path = folder / f"{base}.csv"
        self._f = open(self.path, "w", newline="")
        for key, val in (("subject", args.subject), ("activity", args.activity),
                         ("source", source_desc), ("fs_nominal_hz", args.fs),
                         ("accel_range_g_initial", pipe.accel_range_g),
                         ("gyro_range_dps_initial", pipe.gyro_range_dps),
                         ("motion_thr_g", args.motion_thr), ("gyro_thr_dps", args.gyro_thr),
                         ("window_s", args.window_s),
                         ("started", datetime.now().isoformat(timespec="seconds"))):
            self._f.write(f"# {key}={val}\n")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)
        self._last_flush = time.time()

    def write(self, t_us, res):
        for i in range(len(t_us)):
            a, g = res.acc_g[i], res.gyro_dps[i]
            self._w.writerow((int(t_us[i]),
                              f"{a[0]:.5f}", f"{a[1]:.5f}", f"{a[2]:.5f}",
                              f"{g[0]:.3f}", f"{g[1]:.3f}", f"{g[2]:.3f}",
                              f"{res.a_mag[i]:.5f}", f"{res.accel_idx[i]:.5f}",
                              f"{res.gyro_idx[i]:.3f}",
                              int(res.motion[i]), int(res.calibrated[i])))
        if time.time() - self._last_flush > 1.0:
            self._f.flush()
            self._last_flush = time.time()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
class ImuWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.paused = False
        self._frame = 0
        self._err_printed = False
        self._last_meta = {}

        self.pipe = IMUPipeline(fs=args.fs, accel_range_g=args.accel_range,
                                gyro_range_dps=args.gyro_range, window_s=args.window_s,
                                motion_thr_g=args.motion_thr, gyro_thr_dps=args.gyro_thr,
                                calib_s=args.calib_s)
        self.source = self._start_source()
        self.recorder = None
        if not args.no_log:
            self.recorder = ImuRecorder(args.data_dir, args, self.source.description, self.pipe)

        self._build_ui()
        self._apply_ranges()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(30)

    # ---- setup ---------------------------------------------------------
    def _start_source(self):
        a = self.args
        if a.simulate:
            src = ImuSimSource(fs=a.fs, accel_range_g=a.accel_range, gyro_range_dps=a.gyro_range)
        else:
            port = a.port or find_serial_port()
            if not port:
                raise RuntimeError("No serial port found. Plug in the ESP32 (check the USB cable "
                                   "carries data) or run with --simulate.")
            src = ImuSerialSource(port, a.baud)
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
        self.setWindowTitle("ESP32 + MPU6050 live IMU")
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
        self.status_big = QtWidgets.QLabel("--")
        self.status_big.setStyleSheet("font-size: 46px; font-weight: bold;")
        top.addWidget(self.status_big)
        top.addSpacing(30)
        info = QtWidgets.QVBoxLayout()
        self.values_label = QtWidgets.QLabel("")
        self.values_label.setStyleSheet("font-size: 15px; color: #c3ccd5;")
        self.hint_label = QtWidgets.QLabel("")
        self.hint_label.setStyleSheet("font-size: 12px; color: #9aa5b1;")
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setStyleSheet("font-size: 12px; color: #8a949e;")
        for w in (self.values_label, self.hint_label, self.info_label):
            info.addWidget(w)
        top.addLayout(info)
        top.addStretch(1)
        vbox.addLayout(top)

        self.glw = pg.GraphicsLayoutWidget()
        vbox.addWidget(self.glw, 1)

        self.acc_plot = self._make_plot(0, "Accel (g)")
        self.gyr_plot = self._make_plot(1, "Gyro (deg/s)")
        self.mag_plot = self._make_plot(2, "|a| (g)")
        self.idx_plot = self._make_plot(3, "Motion idx (g)", bottom=True)
        self.acc_plot.addLegend(offset=(10, 5))
        self.gyr_plot.addLegend(offset=(10, 5))

        self.acc_curves, self.gyr_curves = [], []
        for axis_name in "XYZ":
            pen = pg.mkPen(COLORS[axis_name], width=1.4)
            self.acc_curves.append(self.acc_plot.plot(pen=pen, name=axis_name))
            self.gyr_curves.append(self.gyr_plot.plot(pen=pen, name=axis_name))
        self.mag_curve = self.mag_plot.plot(pen=pg.mkPen("#e0e6ec", width=1.4))
        self.mag_plot.addItem(pg.InfiniteLine(pos=1.0, angle=0,
                              pen=pg.mkPen("#6c7a89", style=QtCore.Qt.DashLine)))
        self.idx_curve = self.idx_plot.plot(pen=pg.mkPen("#ff9f43", width=1.6))
        self.idx_plot.addItem(pg.InfiniteLine(pos=self.args.motion_thr, angle=0,
                              pen=pg.mkPen("#f5c542", style=QtCore.Qt.DashLine)))
        for row, stretch in enumerate((3, 3, 2, 2)):
            self.glw.ci.layout.setRowStretchFactor(row, stretch)

        bottom = QtWidgets.QHBoxLayout()
        self.pause_btn = QtWidgets.QPushButton("Pause display")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._on_pause)
        bottom.addWidget(self.pause_btn)
        self.cal_btn = QtWidgets.QPushButton("Recalibrate gyro (keep still)")
        self.cal_btn.clicked.connect(self.pipe.recalibrate)
        bottom.addWidget(self.cal_btn)
        if self.args.simulate:
            self.sim_btn = QtWidgets.QPushButton("Simulate shake")
            self.sim_btn.setCheckable(True)
            self.sim_btn.toggled.connect(lambda on: setattr(self.source, "force_motion", bool(on)))
            bottom.addWidget(self.sim_btn)
        rec_text = f"Recording to: {self.recorder.path}" if self.recorder else "Recording: OFF (--no-log)"
        self.rec_label = QtWidgets.QLabel(rec_text)
        self.rec_label.setStyleSheet("font-size: 11px; color: #8a949e;")
        bottom.addWidget(self.rec_label, 1)
        vbox.addLayout(bottom)

    def _on_pause(self, on):
        self.paused = bool(on)
        self.pause_btn.setText("Resume display" if on else "Pause display")

    def _apply_ranges(self):
        ar, gr = self.pipe.accel_range_g, self.pipe.gyro_range_dps
        self.acc_plot.setYRange(-ar * 1.05, ar * 1.05, padding=0)
        self.gyr_plot.setYRange(-gr * 1.05, gr * 1.05, padding=0)
        self.mag_plot.setYRange(0, min(ar, 3.0) * 1.05, padding=0)

    def _sync_meta(self):
        """Take fs / ranges from the firmware's '# IMU_META' line when it differs from our settings."""
        meta = dict(self.source.meta)
        if meta == self._last_meta:
            return
        self._last_meta = meta
        try:
            fs = float(meta["fs_hz"]) if "fs_hz" in meta else None
            ar = int(meta["accel_range_g"]) if "accel_range_g" in meta else None
            gr = int(meta["gyro_range_dps"]) if "gyro_range_dps" in meta else None
            if ar is not None and ar not in ACCEL_LSB_PER_G:
                ar = None
            if gr is not None and gr not in GYRO_LSB_PER_DPS:
                gr = None
            self.pipe.configure(fs=fs, accel_range_g=ar, gyro_range_dps=gr)
            self._apply_ranges()
        except (ValueError, KeyError):
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
            res = self.pipe.push(rows[:, 0], rows[:, 1:7])
            if self.recorder:
                self.recorder.write(rows[:, 0], res)

        self._frame += 1
        self._update_labels()
        if not self.paused:
            self._redraw()

    def _update_labels(self):
        p = self.pipe
        st = p.status
        color = STATUS_COLORS.get(st, "#ff5a5f")
        self.status_big.setText(st)
        self.status_big.setStyleSheet(f"font-size: 46px; font-weight: bold; color: {color};")

        if p.total_samples:
            self.values_label.setText(
                f"|a| = {p.a_mag[-1]:.3f} g    accel idx = {p.accel_idx[-1]:.3f} g "
                f"(thr {p.motion_thr_g:g})    gyro idx = {p.gyro_idx[-1]:.1f} deg/s (thr {p.gyro_thr_dps:g})")
        else:
            self.values_label.setText("waiting for data...")

        hint = ""
        if st == "CALIBRATING":
            hint = f"Keep the sensor still for {p.calib_s:g} s (gyro offset is being measured)"
        elif st == "MOTION":
            hint = f"Motion triggered by: {p.motion_reason}"
        elif st == "SATURATED":
            hint = "Raw value at full scale: increase the range in the firmware (ACCEL_FS_SEL / GYRO_FS_SEL)"
        self.hint_label.setText(hint)

        if self._frame % 10 == 0:
            fs_est = ""
            if p.t_us.size > 20:
                dt = np.median(np.diff(p.t_us[-300:]))
                if dt > 0:
                    fs_est = f" | fs ~ {1e6 / dt:.1f} Hz"
            s = self.source
            who = s.meta.get("who_am_i", "?")
            i2c = s.meta.get("i2c_errors", "?")
            info = (f"{s.description}{fs_est} | range +-{p.accel_range_g} g, +-{p.gyro_range_dps} deg/s | "
                    f"WHO_AM_I {who} | samples {p.total_samples} | dropped ~{p.dropped_est} | "
                    f"bad lines {s.bad_lines} | i2c errors {i2c}")
            if s.error:
                info = f"ERROR: {s.error}"
                self.info_label.setStyleSheet("font-size: 13px; color: #ff5a5f;")
            elif not s.connected and p.total_samples == 0:
                info += " | connecting..."
            self.info_label.setText(info)

    def _redraw(self):
        p = self.pipe
        if p.t_us.size < 2:
            return
        t_rel = (p.t_us - p.t_us[-1]) / 1e6
        m = t_rel >= -self.args.window
        tr = t_rel[m]
        for i in range(3):
            self.acc_curves[i].setData(tr, p.acc[m, i], skipFiniteCheck=True)
            self.gyr_curves[i].setData(tr, p.gyr[m, i], skipFiniteCheck=True)
        self.mag_curve.setData(tr, p.a_mag[m], skipFiniteCheck=True)
        self.idx_curve.setData(tr, p.accel_idx[m], skipFiniteCheck=True)

        if self._frame % 15 == 0 and tr.size > 20:
            top = max(float(np.percentile(p.accel_idx[m], 99.5)) * 1.2, p.motion_thr_g * 2.0, 0.05)
            self.idx_plot.setYRange(0, top, padding=0)

    # ---- shutdown ------------------------------------------------------
    def closeEvent(self, event):
        self.timer.stop()
        self.source.stop()
        if self.recorder:
            self.recorder.close()
            print(f"Saved: {self.recorder.path}")
        event.accept()


# ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Live MPU6050 viewer for ESP32")
    ap.add_argument("--port", help="serial port, e.g. /dev/ttyUSB0 or COM5 (auto-detect if omitted)")
    ap.add_argument("--baud", type=int, default=230400, help="must match the firmware (default 230400)")
    ap.add_argument("--fs", type=float, default=100.0, help="sampling rate set in the firmware (Hz); "
                    "the firmware's IMU_META line overrides it")
    ap.add_argument("--accel-range", type=int, default=4, choices=sorted(ACCEL_LSB_PER_G),
                    help="accelerometer range in g (default 4; firmware IMU_META overrides it)")
    ap.add_argument("--gyro-range", type=int, default=500, choices=sorted(GYRO_LSB_PER_DPS),
                    help="gyro range in deg/s (default 500; firmware IMU_META overrides it)")
    ap.add_argument("--window", type=float, default=10.0, help="seconds shown on screen")
    ap.add_argument("--window-s", type=float, default=1.0, help="motion-index window length (s)")
    ap.add_argument("--motion-thr", type=float, default=0.05,
                    help="MOTION if rolling std of |a| exceeds this many g (default 0.05, tune it)")
    ap.add_argument("--gyro-thr", type=float, default=15.0,
                    help="MOTION if rolling mean |gyro| exceeds this many deg/s (default 15, tune it)")
    ap.add_argument("--calib-s", type=float, default=3.0, help="gyro offset calibration time (s)")
    ap.add_argument("--subject", default="S01", help="non-identifying subject code")
    ap.add_argument("--activity", default="rest", help="activity/condition label stored in the CSV")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="where session CSV files go")
    ap.add_argument("--no-log", action="store_true", help="do not write a CSV file")
    ap.add_argument("--simulate", action="store_true", help="use built-in synthetic IMU, no hardware")
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
    pg.mkQApp("IMU viewer")
    try:
        win = ImuWindow(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    win.show()
    pg.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())