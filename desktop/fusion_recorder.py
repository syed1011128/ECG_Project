"""
fusion_recorder.py
==================
CSV logging for the fusion viewer (no GUI code, so it can be tested). One session = four files
in data/, all on the ESP32 clock (timestamp_us), so they can be merged later with pandas:

    fusion_YYYYMMDD_HHMM_ecg.csv    timestamp_us, raw_adc, filtered, bpm, leads_off
    fusion_YYYYMMDD_HHMM_imu.csv    timestamp_us, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, a_mag_g,
                                    accel_idx_g, gyro_idx_dps, motion, calibrated
    fusion_YYYYMMDD_HHMM_ppg.csv    timestamp_us, red, ir, ppg_filt, bpm, pi_pct, r_ratio,
                                    spo2_est_experimental, finger, quality
    fusion_YYYYMMDD_HHMM_beats.csv  one row per ECG R peak: PAT, validity flags and reason

Header lines start with '#'. Load with pd.read_csv(path, comment="#").
"""
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path


def _fmt(v, spec):
    return "" if v is None or v != v else format(v, spec)


class FusionRecorder:
    ECG_HEADER = ["timestamp_us", "raw_adc", "filtered", "bpm", "leads_off"]
    IMU_HEADER = ["timestamp_us", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps", "a_mag_g",
                  "accel_idx_g", "gyro_idx_dps", "motion", "calibrated"]
    PPG_HEADER = ["timestamp_us", "red", "ir", "ppg_filt", "bpm", "pi_pct", "r_ratio",
                  "spo2_est_experimental", "finger", "quality"]
    BEAT_HEADER = ["r_timestamp_us", "ppg_peak_timestamp_us", "pat_peak_ms", "pat_slope_ms",
                   "ecg_bpm_inst", "ppg_bpm_inst", "ecg_ok", "ppg_ok", "imu_ok", "motion", "valid", "reason"]

    def __init__(self, folder, subject, activity, source_desc, settings=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        base, k = f"fusion_{stamp}", 1
        while (folder / f"{base}_ecg.csv").exists():
            k += 1
            base = f"fusion_{stamp}_{k}"
        self.base = base
        self.paths = {n: folder / f"{base}_{n}.csv" for n in ("ecg", "imu", "ppg", "beats")}
        header = [("subject", subject), ("activity", activity), ("source", source_desc),
                  ("started", datetime.now().isoformat(timespec="seconds"))]
        header += list((settings or {}).items())
        self._files, self._w = {}, {}
        notes = {
            "ecg": "filtered = causal 0.5-40 Hz band-pass + mains notch",
            "imu": "gyro offset removed after the 3 s calibration; motion = accel or gyro index over threshold",
            "ppg": "ppg_filt = -(causal 0.5-8 Hz band-pass of IR); spo2 is uncalibrated and only filled while quality == OK",
            "beats": "PAT = ECG R peak -> PPG pulse (zero-phase, sub-sample); valid = ECG ok, PPG ok, IMU ok and no motion",
        }
        heads = {"ecg": self.ECG_HEADER, "imu": self.IMU_HEADER, "ppg": self.PPG_HEADER, "beats": self.BEAT_HEADER}
        for name, path in self.paths.items():
            f = open(path, "w", newline="")
            for key, val in header + [("note", notes[name])]:
                f.write(f"# {key}={val}\n")
            w = csv.writer(f)
            w.writerow(heads[name])
            self._files[name], self._w[name] = f, w
        self._last_flush = time.time()

    # ---- writers -------------------------------------------------------------
    def write_ecg(self, t_us, raw, res, bpm, lo):
        w = self._w["ecg"]
        b = _fmt(bpm, ".1f")
        for i in range(len(t_us)):
            w.writerow((int(t_us[i]), int(raw[i]), f"{res.filtered[i]:.2f}", b, int(lo[i])))
        self._maybe_flush()

    def write_imu(self, t_us, res, calibrated=None):
        w = self._w["imu"]
        for i in range(len(t_us)):
            a, g = res.acc_g[i], res.gyro_dps[i]
            w.writerow((int(t_us[i]), f"{a[0]:.4f}", f"{a[1]:.4f}", f"{a[2]:.4f}",
                        f"{g[0]:.3f}", f"{g[1]:.3f}", f"{g[2]:.3f}", f"{res.a_mag[i]:.4f}",
                        f"{res.accel_idx[i]:.4f}", f"{res.gyro_idx[i]:.3f}",
                        int(res.motion[i]), int(res.calibrated[i])))
        self._maybe_flush()

    def write_ppg(self, t_us, red, ir, res, pipe, quality):
        w = self._w["ppg"]
        bpm, pi, r = _fmt(pipe.bpm, ".1f"), _fmt(pipe.pi_pct, ".3f"), _fmt(pipe.r_ratio, ".4f")
        sp = _fmt(pipe.spo2_est if quality == "OK" else None, ".1f")
        for i in range(len(t_us)):
            w.writerow((int(t_us[i]), int(red[i]), int(ir[i]), f"{res.ppg[i]:.2f}", bpm, pi, r, sp,
                        int(res.finger[i]), quality))
        self._maybe_flush()

    def write_beats(self, records):
        w = self._w["beats"]
        for b in records:
            w.writerow((b.r_t_us, _fmt(b.ppg_t_us, ".0f"), _fmt(b.pat_peak_ms, ".2f"),
                        _fmt(b.pat_slope_ms, ".2f"), _fmt(b.ecg_bpm_inst, ".1f"),
                        _fmt(b.ppg_bpm_inst, ".1f"), int(b.ecg_ok), int(b.ppg_ok), int(b.imu_ok),
                        int(b.motion), int(b.valid), b.reason))
        self._maybe_flush()

    def _maybe_flush(self):
        if time.time() - self._last_flush > 1.0:
            for f in self._files.values():
                f.flush()
            self._last_flush = time.time()

    def close(self):
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass