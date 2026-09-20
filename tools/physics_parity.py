"""Release gate: the same MuJoCo Warp model stepped on the CPU and on Metal must stay together.

Usage (in an environment that has the wheel under test, ``mujoco-warp`` and ``mujoco``):

    python tools/physics_parity.py [--steps N] MODEL.xml [MODEL.xml+floor ...]

``+floor`` after a path adds a ground plane just below the model, so that a robot description without a
scene still exercises contacts. Pick models on both sides of the solver's size thresholds; a run that does
not include one model with more than 32 and one with more than 64 degrees of freedom fails, because a
factorization bug that only showed above 32 degrees of freedom once shipped in five wheels.

"It trains" is not evidence that the physics is right. This script is.
"""

import argparse
import os
import sys

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

QPOS_TOL = 1e-3  # absolute, after --steps steps; contact-rich models drift chaotically beyond that
QVEL_TOL = 2e-2  # relative to the largest velocity


def load(path):
    if path.endswith("+floor"):
        spec = mujoco.MjSpec.from_file(path[: -len("+floor")])
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], pos=[0, 0, -0.03])
        return spec.compile()
    return mujoco.MjModel.from_xml_path(path)


def simulate(mjm, device, steps):
    mjd = mujoco.MjData(mjm)
    if mjm.nkey:
        mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    with wp.ScopedDevice(device):
        m = mjw.put_model(mjm)
        if hasattr(m.opt, "warn_overflow"):
            m.opt.warn_overflow = 0  # solver iteration notices would drown the report
        d = mjw.put_data(mjm, mjd, nworld=2, nconmax=max(256, 8 * mjm.ngeom), njmax=max(512, 12 * mjm.nv))
        nefc_max = 0
        for _ in range(steps):
            mjw.step(m, d)
            nefc_max = max(nefc_max, int(d.nefc.numpy().max()))
        wp.synchronize()
        return d.qpos.numpy()[0].copy(), d.qvel.numpy()[0].copy(), nefc_max


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    wp.config.quiet = True
    if not wp.is_metal_available():
        sys.exit("error: no Metal device; is the warp-metal overlay active?")

    failures, sizes, contacts = 0, [], False
    for path in args.models:
        name = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
        mjm = load(path)
        q_cpu, v_cpu, nefc_cpu = simulate(mjm, "cpu", args.steps)
        q_gpu, v_gpu, nefc_gpu = simulate(mjm, "metal:0", args.steps)
        dq = float(np.abs(q_cpu - q_gpu).max())
        dv = float(np.abs(v_cpu - v_gpu).max() / (1e-6 + np.abs(v_cpu).max()))
        ok = bool(np.isfinite(q_gpu).all() and np.isfinite(v_gpu).all() and dq < QPOS_TOL and dv < QVEL_TOL)
        failures += not ok
        sizes.append(mjm.nv)
        contacts |= nefc_cpu > 0
        print(
            f"{'ok      ' if ok else 'MISMATCH'} {name}: nv {mjm.nv}, max nefc cpu/metal {nefc_cpu}/{nefc_gpu}, "
            f"max|dqpos| {dq:.1e}, rel dqvel {dv:.1e}"
        )

    if not any(32 < nv <= 64 for nv in sizes) or not any(nv > 64 for nv in sizes):
        sys.exit(f"error: model sizes {sorted(sizes)} do not cover both 32 < nv <= 64 and nv > 64")
    if not contacts:
        sys.exit("error: no model had an active constraint; add a scene with contacts or a +floor model")
    if failures:
        sys.exit(f"{failures} model(s) differ between CPU and Metal")
    print(f"parity holds for {len(sizes)} models over {args.steps} steps")


if __name__ == "__main__":
    main()
