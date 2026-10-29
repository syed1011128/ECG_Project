"""
test_imu_processing.py - sanity tests for imu_processing.py / imu_sources.py.

Run:  python test_imu_processing.py
No hardware, no GUI needed.
"""
import numpy as np

from imu_processing import (ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS, IMUPipeline,
                            raw_to_units, rolling_mean, rolling_std)
from imu_sources import ImuSerialSource, SyntheticIMU

FS = 100.0
BIAS = (3.0, -2.0, 1.5)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return bool(ok)


def run(seconds, chunk=3, force_motion=False, pipe=None, gen=None, seed=1):
    """Feed `seconds` of synthetic IMU into a pipeline. Returns (pipe, trace) with
    trace = [(t_s, status, motion_flag), ...] after every chunk."""
    pipe = pipe or IMUPipeline(fs=FS)
    gen = gen or SyntheticIMU(fs=FS, gyro_bias=BIAS, seed=seed)
    gen.force_motion = force_motion
    total, done, trace = int(seconds * FS), 0, []
    while done < total:
        k = min(chunk, total - done)
        raw = gen.read(k)
        t_us = 1_000_000 + ((done + np.arange(k)) * 1e6 / FS).astype(np.int64)
        pipe.push(t_us, raw)
        done += k
        trace.append((done / FS, pipe.status, bool(pipe.motion[-1])))
    return pipe, trace


def test_conversion():
    raw = np.array([[8192, 0, -8192, 655, 0, -655]])
    acc, gyr = raw_to_units(raw, 4, 500)
    ok = np.allclose(acc, [[1.0, 0.0, -1.0]]) and np.allclose(gyr, [[10.0, 0.0, -10.0]])
    ok &= ACCEL_LSB_PER_G[2] == 16384 and GYRO_LSB_PER_DPS[250] == 131
    return check("raw -> g / deg/s (4 g: 8192 LSB/g, 500 dps: 65.5 LSB/dps)", ok, f"acc={acc[0]}, gyr={gyr[0]}")


def test_rolling():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    w = 50
    s = rolling_std(x, w)
    m = rolling_mean(x, w)
    ok = abs(s[-1] - np.std(x[-w:])) < 1e-9 and abs(m[-1] - np.mean(x[-w:])) < 1e-9
    ok &= abs(s[9] - np.std(x[:10])) < 1e-9          # expanding window at the start
    return check("rolling_std / rolling_mean match numpy", ok)


def test_still():
    # auto=False: no scheduled shaking, the sensor just lies still for 10 s
    pipe, trace = run(10, gen=SyntheticIMU(fs=FS, gyro_bias=BIAS, auto=False, seed=1))
    a = pipe.a_mag[-500:]
    ok1 = abs(a.mean() - 1.0) < 0.02
    ok2 = pipe.gyr[-500:].std() < 1.0 and np.linalg.norm(pipe.gyr[-300:], axis=1).mean() < 0.6
    ok3 = np.allclose(pipe.gyro_offset, BIAS, atol=0.3)
    ok4 = pipe.status == "STILL" and not pipe.motion[-300:].any()
    check("still: |a| about 1 g", ok1, f"mean={a.mean():.4f}")
    check("still: gyro offset removed", ok2, f"mean|g|={np.linalg.norm(pipe.gyr[-300:], axis=1).mean():.3f} dps")
    check("still: offset estimate matches bias", ok3, f"{np.round(pipe.gyro_offset, 2)}")
    check("still: STILL, no false MOTION", ok4, f"status={pipe.status}")
    return ok1 and ok2 and ok3 and ok4


def test_calibrating_label():
    pipe, trace = run(1.5)
    ok = pipe.status == "CALIBRATING" and not pipe.motion.any()
    return check("status CALIBRATING and no flag in first 3 s", ok, f"status={pipe.status}")


def test_motion_timing():
    # synthetic IMU shakes during t in [6, 9) of each 12 s cycle (auto mode)
    pipe, trace = run(14, force_motion=False)
    # note: run() sets force_motion=False each call; auto schedule still applies
    during = [m for (t, s, m) in trace if 7.0 < t < 8.9]
    before = [m for (t, s, m) in trace if 4.0 < t < 5.9]
    after = [m for (t, s, m) in trace if 10.5 < t < 11.9]
    ok1, ok2, ok3 = all(during), not any(before), not any(after)
    check("shake: MOTION during shaking", ok1, f"{sum(during)}/{len(during)}")
    check("shake: STILL before shaking", ok2)
    check("shake: back to STILL within ~1.5 s after shaking", ok3)
    return ok1 and ok2 and ok3


def test_force_motion():
    pipe, trace = run(8, force_motion=True)
    ok = pipe.status == "MOTION" and pipe.motion_reason in ("accel", "gyro", "accel+gyro")
    return check("forced shake -> MOTION + reason", ok, f"status={pipe.status}, reason={pipe.motion_reason}")


def test_saturation():
    pipe, _ = run(5)
    raw = np.array([[32767, 0, 0, 0, 0, 0]])
    pipe.push(np.array([10_000_000]), raw)
    return check("raw at full scale -> SATURATED", pipe.status == "SATURATED", f"status={pipe.status}")


def test_drops_and_chunk1():
    pipe = IMUPipeline(fs=FS)
    gen = SyntheticIMU(fs=FS, seed=3)
    t = 1_000_000
    for i in range(300):
        if i == 100:
            t += 50_000                       # 5 samples missing (50 ms at 100 Hz)
        pipe.push(np.array([t]), gen.read(1))
        t += 10_000
    return check("dropped-sample estimate (5 missing, chunk size 1)", pipe.dropped_est == 5, f"est={pipe.dropped_est}")


def test_ranges_change():
    pipe = IMUPipeline(fs=FS, accel_range_g=4)
    pipe.configure(accel_range_g=8)
    pipe.push(np.array([1_000_000]), np.array([[4096, 0, 0, 0, 0, 0]]))
    ok = abs(pipe.acc[-1, 0] - 1.0) < 1e-9
    try:
        pipe.configure(accel_range_g=5)
        ok = False
    except ValueError:
        pass
    return check("range change applies, invalid range rejected", ok)


def test_serial_parser():
    s = ImuSerialSource("X")
    lines = [b"# IMU_LOGGER v1\n",
             b"# IMU_META fs_hz=100 accel_range_g=4 gyro_range_dps=500 who_am_i=0x68 i2c_errors=0 dropped=0\n",
             b"\xff\xfe garbage\n",
             b"1000000,120,-45,8210,15,-3,7\r\n",
             b"1010000,121,-44,8211,16,-3,6\n",
             b"1020000,1,2\n",                       # too few fields
             b"1030000,99999,0,0,0,0,0\n",           # outside int16
             b"500,1,2,3,4,5,6\n",                   # timestamp goes backwards
             b"1040000,1,2,3,4,5,x\n",               # not an int
             b"1050000,-32768,0,0,0,0,32767\n",      # extreme but valid
             b"\n"]
    for line in lines:
        s.parse_line(line)
    good = list(s.samples)
    ok = (len(good) == 3 and good[0] == (1000000, 120, -45, 8210, 15, -3, 7)
          and s.bad_lines == 5 and s.meta.get("accel_range_g") == "4"
          and s.meta.get("who_am_i") == "0x68" and len(s.messages) == 2)
    return check("serial parser: good lines kept, 5 bad lines dropped, meta parsed", ok,
                 f"good={len(good)}, bad={s.bad_lines}, meta_keys={sorted(s.meta)}")


if __name__ == "__main__":
    results = [test_conversion(), test_rolling(), test_still(), test_calibrating_label(),
               test_motion_timing(), test_force_motion(), test_saturation(),
               test_drops_and_chunk1(), test_ranges_change(), test_serial_parser()]
    print()
    print("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED")
    raise SystemExit(0 if all(results) else 1)