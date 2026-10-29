"""
test_ppg_processing.py - sanity tests for ppg_processing.py / ppg_sources.py with synthetic PPG.

Run:  python test_ppg_processing.py
No hardware, no GUI needed.
"""
import numpy as np

from ppg_processing import PPGPipeline, design_ppg_filter, ADC_MAX
from ppg_sources import PPGSerialSource, SyntheticPPG
from processing import StreamingFilter

FS = 100.0
ORIGIN = 1_000_000


def feed(gen, seconds, pipe=None, chunk=3, start=0, finger_off_between=None, motion_between=None):
    """Feed `seconds` of generator output into a pipeline in small chunks. Returns (pipe, trace)."""
    pipe = pipe or PPGPipeline(fs=FS)
    total = int(seconds * FS)
    done = 0
    trace = []
    while done < total:
        k = min(chunk, total - done)
        now_s = (start + done) / FS
        if finger_off_between:
            gen.finger_off = finger_off_between[0] <= now_s < finger_off_between[1]
        if motion_between:
            gen.motion = motion_between[0] <= now_s < motion_between[1]
        red, ir = gen.read(k)
        t_us = ORIGIN + ((start + done + np.arange(k)) * 1e6 / FS).astype(np.int64)
        pipe.push(t_us, red, ir)
        done += k
        trace.append((now_s + k / FS, pipe.bpm, pipe.quality()))
    return pipe, trace


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def test_filter_response():
    sos = design_ppg_filter(FS)
    t = np.arange(0, 40, 1 / FS)

    def gain(f):
        flt = StreamingFilter(sos)
        x = np.sin(2 * np.pi * f * t)
        y = flt.process(x)
        s = slice(int(20 * FS), None)
        return np.std(y[s]) / np.std(x[s])

    g12, g01, g20, g40 = gain(1.2), gain(0.1), gain(20.0), gain(40.0)
    ok = check("1.2 Hz pulse passes (>0.9)", g12 > 0.9, f"gain={g12:.3f}")
    ok &= check("0.1 Hz drift removed (<0.1)", g01 < 0.1, f"gain={g01:.3f}")
    # 2nd-order Butterworth band-pass at 8 Hz: about -19 dB (0.11) at 20 Hz is what the design gives
    ok &= check("20 Hz attenuated (<0.15, about -16 dB or better)", g20 < 0.15, f"gain={g20:.3f}")
    ok &= check("40 Hz attenuated (<0.05)", g40 < 0.05, f"gain={g40:.3f}")
    return ok


def test_bpm(bpm):
    gen = SyntheticPPG(fs=FS, bpm=bpm, hrv_ms=15, seed=bpm)
    pipe, _ = feed(gen, 45)
    est = pipe.bpm
    ok = est is not None and abs(est - bpm) <= max(2.0, 0.03 * bpm)
    return check(f"BPM {bpm}", ok, f"estimated={None if est is None else round(est, 1)} quality={pipe.quality()}")


def test_peak_timing():
    gen = SyntheticPPG(fs=FS, bpm=75, hrv_ms=30, seed=7)
    pipe, _ = feed(gen, 45)
    det = (np.asarray(pipe.peak_t_us) - ORIGIN) / 1e6
    true = gen.systolic_peak_times()
    true = true[(true > 10.0) & (true < 42.0)]
    det = det[(det > 10.0) & (det < 42.0)]
    hits = sum(np.min(np.abs(det - b)) < 0.040 for b in true) if det.size else 0
    false = sum(np.min(np.abs(true - d)) > 0.040 for d in det)
    ok = hits >= 0.95 * len(true) and false <= 1
    med = float(np.median([np.min(np.abs(det - b)) for b in true])) * 1000 if det.size else float("nan")
    return check("pulse-peak timing (within 40 ms)", ok, f"hit {hits}/{len(true)}, false {false}, median error {med:.1f} ms")


def test_no_finger():
    gen = SyntheticPPG(fs=FS, seed=1)
    gen.finger_off = True
    pipe, _ = feed(gen, 15)
    ok = pipe.quality() == "NO FINGER" and pipe.bpm is None and len(pipe.peak_t_us) == 0
    return check("no finger -> NO FINGER, no BPM, no beats", ok, f"q={pipe.quality()} beats={len(pipe.peak_t_us)}")


def test_finger_off_recovery():
    gen = SyntheticPPG(fs=FS, bpm=70, hrv_ms=15, seed=3)
    pipe, trace = feed(gen, 50, finger_off_between=(20.0, 24.0))
    during = [(b, q) for (t, b, q) in trace if 21.0 < t < 23.9]
    ok1 = all(b is None and q == "NO FINGER" for b, q in during)
    rec = [t for (t, b, q) in trace if t > 24.0 and b is not None]
    ok2 = bool(rec) and (rec[0] - 24.0) < 12.0
    ok3 = pipe.bpm is not None and abs(pipe.bpm - 70) < 4
    check("BPM hidden while finger is off", ok1)
    check("BPM back < 12 s after finger returns", ok2, f"after {rec[0] - 24.0:.1f}s" if rec else "never")
    check("BPM correct after recovery", ok3, f"{pipe.bpm}")
    return ok1 and ok2 and ok3


def test_spo2_ratio():
    # clean pulse (no breathing drift): R should match the generator's r_ratio
    gen = SyntheticPPG(fs=FS, bpm=70, hrv_ms=10, r_ratio=0.5, wander_frac=0.0, seed=5)
    pipe, _ = feed(gen, 40)
    r, s = pipe.r_ratio, pipe.spo2_est
    ok = r is not None and 0.45 <= r <= 0.55 and s is not None and 96.0 <= s <= 99.0
    check("R ratio recovers generator value (no drift)", ok, f"R={None if r is None else round(r, 3)} SpO2_est={None if s is None else round(s, 1)}")

    # breathing-like baseline drift (1.5 % at 0.25 Hz, bigger than the pulse itself) must not bias R
    gen2 = SyntheticPPG(fs=FS, bpm=70, hrv_ms=10, r_ratio=0.5, wander_frac=0.015, seed=5)
    pipe2, _ = feed(gen2, 40)
    r2 = pipe2.r_ratio
    ok2 = r2 is not None and 0.42 <= r2 <= 0.58
    check("R ratio with strong baseline drift stays near truth", ok2, f"R={None if r2 is None else round(r2, 3)} (true 0.5)")

    ok3 = pipe.pi_pct is not None and 0.8 <= pipe.pi_pct <= 1.5
    check("perfusion index ~1.2 % for 1.2 % AC/DC", ok3, f"PI={None if pipe.pi_pct is None else round(pipe.pi_pct, 2)} %")
    return ok and ok2 and ok3


def test_saturation_and_low_signal():
    pipe = PPGPipeline(fs=FS)
    t = ORIGIN + (np.arange(500) * 10000).astype(np.int64)
    pipe.push(t, np.full(500, ADC_MAX), np.full(500, ADC_MAX))
    ok1 = pipe.quality() == "SATURATED"
    check("full-scale counts -> SATURATED", ok1, f"q={pipe.quality()}")

    gen = SyntheticPPG(fs=FS, bpm=70, ac_ir_frac=0.0003, wander_frac=0.0, seed=2)
    pipe2, _ = feed(gen, 30)
    ok2 = pipe2.quality() == "LOW SIGNAL"
    check("tiny pulse -> LOW SIGNAL", ok2, f"q={pipe2.quality()} PI={pipe2.pi_pct}")
    return ok1 and ok2


def test_motion_survives():
    gen = SyntheticPPG(fs=FS, bpm=70, hrv_ms=15, seed=9)
    pipe, trace = feed(gen, 60, motion_between=(20.0, 28.0))
    ok = pipe.bpm is not None and abs(pipe.bpm - 70) < 4
    return check("pipeline survives a motion burst and recovers", ok, f"bpm after={None if pipe.bpm is None else round(pipe.bpm, 1)}")


def test_drops():
    pipe = PPGPipeline(fs=FS)
    gen = SyntheticPPG(fs=FS, seed=4)
    t = ORIGIN + (np.arange(100) * 10000).astype(np.int64)
    red, ir = gen.read(100)
    keep = np.ones(100, dtype=bool)
    keep[40:45] = False                       # 5 missing samples
    for i in np.flatnonzero(keep):
        pipe.push(t[i:i + 1], red[i:i + 1], ir[i:i + 1])
    return check("dropped-sample estimate (5 missing)", pipe.dropped_est == 5, f"est={pipe.dropped_est}")


def test_serial_parser():
    s = PPGSerialSource("X")
    lines = [b"# PPG_LOGGER v1\n",
             b"# PPG_META fs_hz=100 adc_bits=18 led_red_ma=7.2 led_ir_ma=7.2 adc_range_na=4096 part_id=0x15\n",
             b"\xff\xfe garbage\n",
             b"1000000,81234,102345\r\n",
             b"1010000,81300,102400\n",
             b"1020000,81300\n",                  # too few fields
             b"1030000,999999,102400\n",           # out of 18-bit range
             b"1030000,-5,102400\n",               # negative
             b"500,81300,102400\n",                # time goes backwards
             b"1040000,262143,0\n",
             b"1050000,12,x\n",                    # not a number
             b"\n"]
    for l in lines:
        s.parse_line(l)
    good = list(s.samples)
    ok = (len(good) == 3 and s.bad_lines == 6 and s.meta.get("fs_hz") == "100"
          and s.meta.get("led_red_ma") == "7.2" and len(s.messages) == 2)
    return check("serial parser: bad lines dropped, meta parsed", ok,
                 f"good={len(good)} bad={s.bad_lines} meta_keys={sorted(s.meta)}")


if __name__ == "__main__":
    results = [test_filter_response()]
    for b in (50, 72, 100, 140):
        results.append(test_bpm(b))
    results += [test_peak_timing(), test_no_finger(), test_finger_off_recovery(), test_spo2_ratio(),
                test_saturation_and_low_signal(), test_motion_survives(), test_drops(), test_serial_parser()]
    print()
    print("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED")
    raise SystemExit(0 if all(results) else 1)