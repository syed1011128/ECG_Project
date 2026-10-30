"""
test_fusion_processing.py - tests for fusion_processing.py / fusion_sources.py / fusion_recorder.py

Run:  python test_fusion_processing.py
No hardware, no GUI needed. Uses three synthetic sensors that share one heartbeat with a known
ECG-to-PPG delay (true PAT = 250 ms, R peak -> systolic peak).
"""
import csv
import tempfile
from pathlib import Path

import numpy as np

from fusion_processing import FusionPipeline
from fusion_recorder import FusionRecorder
from fusion_sources import FusionSerialSource, SyntheticFusion

ORIGIN = 1_000_000
ECG_FS, IMU_FS, PPG_FS = 250.0, 100.0, 100.0
TRUE_PAT_MS = 250.0


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def run(seconds, schedule=None, pipe=None, gen=None, frame_s=0.03, ppg_shift_us=0, skip_imu=False,
        recorder=None, seed=1, **gen_kw):
    """Feed all three sensors in ~30 ms frames, like the GUI does.

    schedule: list of (t_start, t_end, flag) with flag in {'motion', 'finger_off', 'leads_off'}.
    ppg_shift_us: add a constant clock error to the PPG timestamps (synchronisation test).
    """
    gen = gen or SyntheticFusion(bpm=72, pat_s=TRUE_PAT_MS / 1000.0, seed=seed, **gen_kw)
    pipe = pipe or FusionPipeline(ecg_fs=ECG_FS, imu_fs=IMU_FS, ppg_fs=PPG_FS)
    done = {"ecg": 0, "imu": 0, "ppg": 0}
    nframes = int(round(seconds / frame_s))
    for f in range(nframes):
        now = f * frame_s
        for (a, b, flag) in (schedule or []):
            on = a <= now < b
            if flag == "motion":
                gen.motion = on
            elif flag == "finger_off":
                gen.finger_off = on
            elif flag == "leads_off":
                gen.leads_off = on
        t_end = (f + 1) * frame_s
        k = int(t_end * ECG_FS) - done["ecg"]
        if k > 0:
            adc = gen.read_ecg(k)
            t = ORIGIN + ((done["ecg"] + np.arange(k)) * 1e6 / ECG_FS).astype(np.int64)
            lo = np.full(k, 1 if gen.leads_off else 0)
            res = pipe.push_ecg(t, adc, lo)
            if recorder:
                recorder.write_ecg(t, adc, res, pipe.ecg.bpm, lo)
            done["ecg"] += k
        k = int(t_end * IMU_FS) - done["imu"]
        if k > 0 and not skip_imu:
            raw = gen.read_imu(k)
            t = ORIGIN + ((done["imu"] + np.arange(k)) * 1e6 / IMU_FS).astype(np.int64)
            res = pipe.push_imu(t, raw)
            if recorder:
                recorder.write_imu(t, res)
            done["imu"] += k
        elif k > 0:
            done["imu"] += k
        k = int(t_end * PPG_FS) - done["ppg"]
        if k > 0:
            red, ir = gen.read_ppg(k)
            t = ORIGIN + ppg_shift_us + ((done["ppg"] + np.arange(k)) * 1e6 / PPG_FS).astype(np.int64)
            res = pipe.push_ppg(t, red, ir)
            if recorder:
                recorder.write_ppg(t, red, ir, res, pipe.ppg, pipe.ppg.quality())
            done["ppg"] += k
        new = pipe.resolve()
        if recorder and new:
            recorder.write_beats(new)
    return pipe, gen


def beats_in(pipe, t0, t1):
    return [b for b in pipe.beats if t0 <= (b.r_t_us - ORIGIN) / 1e6 < t1]


# ----------------------------------------------------------------------
def test_pat_recovery():
    pipe, gen = run(70)
    good = [b for b in beats_in(pipe, 15, 62) if b.valid]
    total = beats_in(pipe, 15, 62)
    if len(good) < 10:
        return check("PAT recovery", False, f"only {len(good)} valid beats of {len(total)}")
    pk = np.array([b.pat_peak_ms for b in good])
    sl = np.array([b.pat_slope_ms for b in good if b.pat_slope_ms == b.pat_slope_ms])
    ok1 = check("most beats are valid in clean data", len(good) >= 0.9 * len(total), f"{len(good)}/{len(total)}")
    ok2 = check("R -> peak PAT recovered (250 ms +-10)", abs(np.median(pk) - TRUE_PAT_MS) <= 10.0,
                f"median {np.median(pk):.1f} ms, std {np.std(pk):.1f} ms")
    ok3 = check("PAT spread small (std < 12 ms)", np.std(pk) < 12.0, f"std {np.std(pk):.1f} ms")
    # steepest upstroke of a Gaussian pulse (sigma 55 ms) is about one sigma before the peak
    ok4 = check("R -> upstroke PAT earlier than the peak PAT (40-90 ms)",
                sl.size >= 10 and 40.0 <= np.median(pk) - np.median(sl) <= 90.0,
                f"peak {np.median(pk):.1f} ms, slope {np.median(sl):.1f} ms")
    s = pipe.summary(30)
    ok5 = check("summary has PAT median", s["pat_peak_med"] is not None and s["n_valid"] >= 20,
                f"{s['n_valid']}/{s['n_total']} valid, median {s['pat_peak_med']}")
    return ok1 and ok2 and ok3 and ok4 and ok5


def test_clock_offset_shows_up_in_pat():
    pipe, _ = run(50, ppg_shift_us=30_000)
    good = [b.pat_peak_ms for b in beats_in(pipe, 15, 48) if b.valid]
    ok = len(good) >= 10 and abs(np.median(good) - (TRUE_PAT_MS + 30.0)) <= 10.0
    return check("a 30 ms PPG clock error shifts PAT by 30 ms (so sync matters)", ok,
                 f"median {np.median(good) if good else float('nan'):.1f} ms")


def test_motion_invalidates():
    pipe, _ = run(70, schedule=[(30.0, 40.0, "motion")])
    inside = beats_in(pipe, 32.0, 38.0)
    outside = [b for b in beats_in(pipe, 15.0, 27.0)] + beats_in(pipe, 45.0, 62.0)
    bad_inside = sum(1 for b in inside if not b.valid)
    reasons = {b.reason for b in inside if not b.valid}
    ok1 = check("beats during motion are not valid", inside and bad_inside >= 0.9 * len(inside),
                f"{bad_inside}/{len(inside)} invalid, reasons {sorted(reasons)}")
    ok2 = check("motion beats carry the motion flag", sum(1 for b in inside if b.motion) >= 0.7 * len(inside),
                f"{sum(1 for b in inside if b.motion)}/{len(inside)}")
    good_out = sum(1 for b in outside if b.valid)
    ok3 = check("beats away from the motion are valid again", good_out >= 0.8 * len(outside),
                f"{good_out}/{len(outside)}")
    return ok1 and ok2 and ok3


def test_finger_off():
    pipe, _ = run(60, schedule=[(30.0, 40.0, "finger_off")])
    inside = beats_in(pipe, 32.0, 38.0)
    ok1 = check("no finger -> beats invalid, reason ppg",
                inside and all((not b.valid) and b.reason == "ppg" for b in inside),
                f"{[b.reason for b in inside]}")
    ok2 = check("no finger -> no PAT numbers", all(b.pat_peak_ms != b.pat_peak_ms for b in inside))
    return ok1 and ok2


def test_leads_off():
    pipe, _ = run(60, schedule=[(30.0, 40.0, "leads_off")])
    inside = beats_in(pipe, 32.0, 38.0)
    # leads off hides ECG beats entirely: there must be no valid beat in that period
    ok = all(not b.valid for b in inside)
    return check("leads off -> no valid beats", ok, f"{len(inside)} beats recorded in the period")


def test_no_imu():
    pipe, _ = run(40, skip_imu=True)
    late = beats_in(pipe, 15, 36)
    ok = late and all((not b.valid) and b.reason == "imu" for b in late)
    return check("no IMU data -> beats marked invalid, reason imu", ok,
                 f"{sorted({b.reason for b in late})}")


def test_hr_agreement():
    pipe, _ = run(40)
    e, p, d, agree = pipe.hr_agreement()
    ok = e is not None and p is not None and abs(d) <= 3.0 and agree is True
    return check("ECG and PPG heart rate agree on clean data", ok, f"ecg {e}, ppg {p}")


def test_serial_demux():
    s = FusionSerialSource("X")
    lines = [b"# ALL_LOGGER v1\n",
             b"# ECG_META fs_hz=250 adc_bits=12 dropped=0\n",
             b"# IMU_META fs_hz=100 accel_range_g=4 gyro_range_dps=500 who_am_i=0x68 i2c_errors=0 dropped=0\n",
             b"# PPG_META fs_hz=100 adc_bits=18 led_red_ma=7.2 led_ir_ma=7.2 part_id=0x15 i2c_errors=0\n",
             b"# ALL_META baud=460800 queue_len=768 queue_max=12\n",
             b"# ERROR no MPU6050 answering at 0x68.\n",
             b"E,1000000,2048,0,0\r\n",
             b"E,1004000,2050,0,0\n",
             b"I,1000000,10,-5,8200,1,2,3\n",
             b"I,1010000,11,-5,8199,1,2,3\n",
             b"P,1000000,81234,102345\n",
             b"P,1010000,81300,102400\n",
             b"X,1,2,3\n",                    # unknown tag
             b"garbage without tag\n",
             b"E,1008000,5000,0,0\n",          # ADC out of range -> ECG parser rejects
             b"P,999,1,2\n",                   # PPG time goes backwards
             b"I,1020000,1,2\n",               # IMU too few fields
             b"\xff\xfe junk\n",
             b"\n"]
    for l in lines:
        s.parse_line(l)
    by = s.bad_lines_by_stream()
    ok = (len(s.ecg_samples) == 2 and len(s.imu_samples) == 2 and len(s.ppg_samples) == 2
          and by["ecg"] == 1 and by["ppg"] == 1 and by["imu"] == 1 and by["other"] == 3
          and s.imu_meta.get("accel_range_g") == "4" and s.ppg_meta.get("part_id") == "0x15"
          and s.ecg_meta.get("fs_hz") == "250" and s.all_meta.get("queue_max") == "12")
    msgs = s.pop_messages()
    ok2 = any("ERROR no MPU6050" in m for m in msgs)
    return check("serial demux: three streams, per-stream validation, meta lines", ok and ok2,
                 f"samples {len(s.ecg_samples)}/{len(s.imu_samples)}/{len(s.ppg_samples)} bad {by}")


def test_recorder():
    with tempfile.TemporaryDirectory() as d:
        rec = FusionRecorder(d, "S01", "rest", "test", {"fs": 250})
        pipe, _ = run(25, recorder=rec)
        rec.close()
        files = {n: p for n, p in rec.paths.items()}

        def rows(path):
            with open(path) as f:
                return list(csv.reader(l for l in f if not l.startswith("#")))
        r = {n: rows(p) for n, p in files.items()}
        ok1 = all(len(v) > 1 for v in r.values())
        ok2 = (r["ecg"][0] == FusionRecorder.ECG_HEADER and r["imu"][0] == FusionRecorder.IMU_HEADER
               and r["ppg"][0] == FusionRecorder.PPG_HEADER and r["beats"][0] == FusionRecorder.BEAT_HEADER)
        ok3 = len(r["beats"]) - 1 == len(pipe.beats) and all(len(x) == 12 for x in r["beats"])
        ok4 = (len(r["ecg"]) - 1) > 5 * 240 and (len(r["imu"]) - 1) > 5 * 95
        return check("recorder writes four CSV files with the documented columns", ok1 and ok2 and ok3 and ok4,
                     f"ecg {len(r['ecg']) - 1} imu {len(r['imu']) - 1} ppg {len(r['ppg']) - 1} beats {len(r['beats']) - 1}")


if __name__ == "__main__":
    results = [test_pat_recovery(), test_clock_offset_shows_up_in_pat(), test_motion_invalidates(),
               test_finger_off(), test_leads_off(), test_no_imu(), test_hr_agreement(),
               test_serial_demux(), test_recorder()]
    print()
    print("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED")
    raise SystemExit(0 if all(results) else 1)