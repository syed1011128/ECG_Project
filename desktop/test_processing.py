"""
test_processing.py - sanity tests for processing.py using synthetic ECG.

Run:  python test_processing.py
No hardware, no GUI needed.
"""
import numpy as np  # type: ignore[reportMissingImports]
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from processing import ECGPipeline, design_ecg_filter, StreamingFilter  # type: ignore[reportMissingImports]
from sources import SyntheticECG

FS = 250.0


def run(gen, seconds, chunk=8, leads_off_between=None, pipe=None):
    """Feed `seconds` of generator output into a pipeline in small chunks. Returns (pipe, t0_us)."""
    pipe = pipe or ECGPipeline(fs=FS)
    total = int(seconds * FS)
    origin = 1_000_000
    done = 0
    bpm_trace = []
    while done < total:
        k = min(chunk, total - done)
        adc = gen.read(k)
        t_us = origin + ((done + np.arange(k)) * 1e6 / FS).astype(np.int64)
        lo = np.zeros(k, dtype=bool)
        if leads_off_between and leads_off_between[0] <= done / FS < leads_off_between[1]:
            lo[:] = True
            adc = np.full(k, 4095)
        pipe.push(t_us, adc, lo)
        done += k
        bpm_trace.append((done / FS, pipe.bpm, pipe.leads_off))
    return pipe, origin, bpm_trace


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def test_filter_response():
    ok = True
    sos = design_ecg_filter(FS, notch_hz=50.0)
    t = np.arange(0, 20, 1 / FS)

    def gain(f):
        flt = StreamingFilter(sos)
        y = flt.process(np.sin(2 * np.pi * f * t))
        return np.std(y[int(10 * FS):]) / np.std(np.sin(2 * np.pi * f * t[int(10 * FS):]))

    g10, g50, g01, g100 = gain(10), gain(50), gain(0.1), gain(100)
    ok &= check("10 Hz passes (>0.9)", g10 > 0.9, f"gain={g10:.3f}")
    ok &= check("50 Hz mains removed (<0.05)", g50 < 0.05, f"gain={g50:.3f}")
    ok &= check("0.1 Hz drift removed (<0.3)", g01 < 0.3, f"gain={g01:.3f}")
    ok &= check("100 Hz attenuated (<0.1)", g100 < 0.1, f"gain={g100:.3f}")
    return ok


def test_bpm(bpm):
    gen = SyntheticECG(fs=FS, bpm=bpm, hrv_ms=15, seed=bpm)
    pipe, origin, _ = run(gen, 40)
    est = pipe.bpm
    ok = est is not None and abs(est - bpm) <= max(2.0, 0.03 * bpm)
    ok &= check(f"BPM {bpm}", ok, f"estimated={None if est is None else round(est, 1)}")
    return ok


def test_peak_timing():
    gen = SyntheticECG(fs=FS, bpm=75, hrv_ms=30, seed=7)
    pipe, origin, _ = run(gen, 40)
    det = (np.asarray(pipe.peak_t_us) - origin) / 1e6
    true = np.asarray([b for b in gen.beats if 8.0 < b < 38.0])
    det = det[(det > 8.0) & (det < 38.0)]
    hits = sum(np.min(np.abs(det - b)) < 0.030 for b in true) if det.size else 0
    false = sum(np.min(np.abs(true - d)) > 0.030 for d in det)
    ok = hits >= 0.97 * len(true) and false <= 1
    return check("R-peak timing (within 30 ms)", ok, f"hit {hits}/{len(true)}, false {false}")


def test_noise_only():
    rng = np.random.default_rng(1)
    pipe = ECGPipeline(fs=FS)
    done, total = 0, int(30 * FS)
    while done < total:
        t_us = 1_000_000 + ((done + np.arange(8)) * 1e6 / FS).astype(np.int64)
        pipe.push(t_us, 2048 + rng.normal(0, 15, 8), np.zeros(8, dtype=bool))
        done += 8
    return check("noise only -> (almost) no beats", len(pipe.peak_t_us) <= 2, f"beats={len(pipe.peak_t_us)}")


def test_leads_off_recovery():
    gen = SyntheticECG(fs=FS, bpm=70, hrv_ms=15, seed=3)
    pipe, origin, trace = run(gen, 45, leads_off_between=(20.0, 23.0))
    during = [b for (t, b, lo) in trace if 21.0 < t < 22.9]
    ok1 = all(b is None for b in during)
    rec = [t for (t, b, lo) in trace if t > 23.0 and b is not None]
    ok2 = bool(rec) and (rec[0] - 23.0) < 9.0
    ok3 = pipe.bpm is not None and abs(pipe.bpm - 70) < 4
    check("BPM hidden while leads off", ok1)
    check("BPM recovers < 9 s after reconnect", ok2, f"after {rec[0] - 23.0:.1f}s" if rec else "never")
    check("BPM correct after recovery", ok3, f"{pipe.bpm}")
    return ok1 and ok2 and ok3


def test_hrv_runs():
    gen = SyntheticECG(fs=FS, bpm=70, hrv_ms=40, seed=11)
    pipe, _, _ = run(gen, 60)
    h = pipe.hrv()
    ok = h["sdnn_ms"] == h["sdnn_ms"] and 10 < h["sdnn_ms"] < 120 and h["rmssd_ms"] > 5
    return check("HRV values plausible", ok, f"{ {k: round(v, 1) for k, v in h.items()} }")


def test_quality_labels():
    pipe = ECGPipeline(fs=FS)
    r1 = pipe.quality() == "NO DATA"
    t = 1_000_000 + (np.arange(300) * 4000).astype(np.int64)
    pipe.push(t, np.full(300, 4095.0), np.zeros(300, dtype=bool))
    r2 = pipe.quality() == "SATURATED"
    pipe2 = ECGPipeline(fs=FS)
    pipe2.push(t, np.full(300, 2000.0), np.zeros(300, dtype=bool))
    r3 = pipe2.quality() == "FLAT"
    pipe3 = ECGPipeline(fs=FS)
    pipe3.push(t, np.full(300, 4095.0), np.ones(300, dtype=bool))
    r4 = pipe3.quality() == "LEADS OFF"
    return check("quality labels", r1 and r2 and r3 and r4, f"{[r1, r2, r3, r4]}")


if __name__ == "__main__":
    results = [test_filter_response()]
    for b in (50, 72, 100, 140):
        results.append(test_bpm(b))
    results += [test_peak_timing(), test_noise_only(), test_leads_off_recovery(),
                test_hrv_runs(), test_quality_labels()]
    print()
    print("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED")
    raise SystemExit(0 if all(results) else 1)