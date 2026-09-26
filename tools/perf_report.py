"""Physics step timing on Metal for the release notes. Informational only: nothing here fails a release.

Usage: python tools/perf_report.py MODEL.xml[+floor] [--worlds 1024 4096] [--calls 100]

Steps the model with MuJoCo Warp on metal:0 from a state reached after a few steps with random controls,
replaying one captured step graph and synchronizing after every call, with the solver settings of mjlab's G1
velocity task (10 iterations run in full, since Metal evaluates graph loop conditions on the host; 20 line
search iterations; 300 constraint rows per world). Prints the GPU utilization
the system reported before the measurement: other GPU users (a browser, a simulator) slow the numbers down, so
compare releases measured under similar load.
"""

import argparse
import os
import statistics
import subprocess
import sys
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics_parity import load


def gpu_utilization():
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    key = '"Device Utilization %"='
    i = out.find(key)
    return out[i + len(key) :].split(",")[0].split("}")[0] + "%" if i >= 0 else "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--worlds", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--calls", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    wp.config.quiet = True

    mjm, digest = load(args.model)
    mjd = mujoco.MjData(mjm)
    if mjm.nkey:
        mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    name = os.path.join(os.path.basename(os.path.dirname(args.model)), os.path.basename(args.model))
    lines = [f"{name} [{digest}], nv {mjm.nv}, warp {wp.config.version}, device {wp.get_device('metal:0').name}"]
    lines.append(f"GPU utilization before the run: {gpu_utilization()}")
    rng = np.random.default_rng(0)
    with wp.ScopedDevice("metal:0"):
        for worlds in args.worlds:
            m = mjw.put_model(mjm)
            if hasattr(m.opt, "warn_overflow"):
                m.opt.warn_overflow = 0
            m.opt.graph_conditional = False
            m.opt.iterations, m.opt.ls_iterations = 10, 20
            d = mjw.put_data(mjm, mjd, nworld=worlds, njmax=300)
            if mjm.nu:
                d.ctrl.assign(rng.uniform(-0.3, 0.3, (worlds, mjm.nu)).astype(np.float32))
            for _ in range(20):  # leave the initial state, reach contacts
                mjw.step(m, d)
            with wp.ScopedCapture() as capture:
                mjw.step(m, d)
            wp.capture_launch(capture.graph)
            wp.synchronize()
            medians = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                for _ in range(args.calls):
                    wp.capture_launch(capture.graph)
                    wp.synchronize()
                medians.append(1e3 * (time.perf_counter() - start) / args.calls)
            step = statistics.median(medians)
            lines.append(
                f"{worlds:5d} worlds: {step:6.2f} ms per step (median of {args.repeats} x {args.calls} synchronized "
                f"graph replays), {worlds / step * 1e3:9.0f} world-steps/s"
            )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
