"""Release gate: MuJoCo Warp on Metal must agree with the CPU and with MuJoCo itself.

Usage (in an environment that has the wheel under test, ``mujoco-warp`` and ``mujoco``):

    python tools/physics_parity.py [--steps N] [--report FILE] MODEL.xml [MODEL.xml+floor ...]

``+floor`` after a path adds a ground plane just below the model, so that a robot description without a
scene still exercises contacts.

Three checks per model:

1. The acceleration of the first step on the CPU and on Metal against ``mj_forward`` of MuJoCo (C). The CPU
   and Metal runs share the overlay's code, so only an outside reference can catch a defect they have in
   common.
2. Positions and velocities after ``--steps`` steps, Metal against the CPU.
3. The largest number of active constraints seen, which must be equal.

A run fails unless the models include one with more than 32 and one with more than 64 degrees of freedom and
at least one active constraint: a factorization bug that only showed above 32 degrees of freedom once shipped
in five wheels, while the test suites passed and the robot "trained".
"""

import argparse
import hashlib
import os
import sys

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

QACC_RTOL, QACC_ATOL = 1e-3, 1e-4  # first step against MuJoCo (C); both devices agree to about 5e-5
QPOS_ATOL = 1e-3  # after --steps steps; contact-rich models drift chaotically beyond that
QVEL_RTOL, QVEL_ATOL = 2e-2, 1e-4  # the absolute floor keeps a model at rest from failing on noise


def load(path):
    source = path.removesuffix("+floor")
    with open(source, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]
    if path.endswith("+floor"):
        spec = mujoco.MjSpec.from_file(source)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], pos=[0, 0, -0.03])
        return spec.compile(), digest
    return mujoco.MjModel.from_xml_path(source), digest


def simulate(mjm, mjd, device, steps):
    with wp.ScopedDevice(device):
        m = mjw.put_model(mjm)
        if hasattr(m.opt, "warn_overflow"):
            m.opt.warn_overflow = 0  # solver iteration notices would drown the report
        d = mjw.put_data(mjm, mjd, nworld=2, nconmax=max(256, 8 * mjm.ngeom), njmax=max(512, 12 * mjm.nv))
        nefc_max, first_qacc = 0, None
        for step in range(steps):
            mjw.step(m, d)
            nefc_max = max(nefc_max, int(d.nefc.numpy().max()))
            if step == 0:
                first_qacc = d.qacc.numpy()[0].copy()  # acceleration at the initial state
        wp.synchronize()
        return d.qpos.numpy()[0].copy(), d.qvel.numpy()[0].copy(), first_qacc, nefc_max


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--report", help="also write the report to this file")
    args = ap.parse_args()
    wp.config.quiet = True
    if not wp.is_metal_available():
        sys.exit("error: no Metal device; is the warp-metal overlay active?")

    lines = [
        f"warp {wp.config.version}, mujoco {mujoco.__version__}, mujoco_warp {getattr(mjw, '__version__', '?')}, "
        f"device {wp.get_device('metal:0').name}, {args.steps} steps"
    ]
    failures, sizes, contacts = 0, [], False
    for path in args.models:
        name = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
        mjm, digest = load(path)
        mjd = mujoco.MjData(mjm)
        if mjm.nkey:
            mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
        mujoco.mj_forward(mjm, mjd)
        qacc_ref = mjd.qacc.copy()

        q_cpu, v_cpu, a_cpu, nefc_cpu = simulate(mjm, mjd, "cpu", args.steps)
        q_gpu, v_gpu, a_gpu, nefc_gpu = simulate(mjm, mjd, "metal:0", args.steps)

        qacc_limit = QACC_ATOL + QACC_RTOL * float(np.abs(qacc_ref).max())
        da_cpu = float(np.abs(a_cpu - qacc_ref).max())
        da_gpu = float(np.abs(a_gpu - qacc_ref).max())
        dq = float(np.abs(q_cpu - q_gpu).max())
        dv = float(np.abs(v_cpu - v_gpu).max())
        qvel_limit = QVEL_ATOL + QVEL_RTOL * float(np.abs(v_cpu).max())
        problems = []
        if not (np.isfinite(q_gpu).all() and np.isfinite(v_gpu).all()):
            problems.append("non-finite state on Metal")
        if da_cpu > qacc_limit:
            problems.append(f"cpu qacc differs from MuJoCo by {da_cpu:.1e} (limit {qacc_limit:.1e})")
        if da_gpu > qacc_limit:
            problems.append(f"metal qacc differs from MuJoCo by {da_gpu:.1e} (limit {qacc_limit:.1e})")
        if not dq < QPOS_ATOL:
            problems.append(f"qpos differs by {dq:.1e}")
        if not dv < qvel_limit:
            problems.append(f"qvel differs by {dv:.1e} (limit {qvel_limit:.1e})")
        if nefc_cpu != nefc_gpu:
            problems.append(f"constraint counts differ: cpu {nefc_cpu}, metal {nefc_gpu}")

        failures += bool(problems)
        sizes.append(mjm.nv)
        contacts |= nefc_cpu > 0
        lines.append(
            f"{'ok      ' if not problems else 'MISMATCH'} {name} [{digest}]: nv {mjm.nv}, nefc {nefc_cpu}/{nefc_gpu}, "
            f"qacc vs MuJoCo cpu {da_cpu:.1e} metal {da_gpu:.1e}, dqpos {dq:.1e}, dqvel {dv:.1e}"
            + "".join(f"\n           - {p}" for p in problems)
        )
        print(lines[-1], flush=True)

    verdict = None
    if not any(32 < nv <= 64 for nv in sizes) or not any(nv > 64 for nv in sizes):
        verdict = f"error: model sizes {sorted(sizes)} do not cover both 32 < nv <= 64 and nv > 64"
    elif not contacts:
        verdict = "error: no model had an active constraint; add a scene with contacts or a +floor model"
    elif failures:
        verdict = f"FAILED: {failures} of {len(sizes)} models"
    lines.append(verdict or f"parity holds for {len(sizes)} models")
    if args.report:
        with open(args.report, "w") as f:
            f.write("\n".join([lines[0], *lines[1:]]) + "\n")
    if verdict:
        sys.exit(verdict)
    print(lines[-1])


if __name__ == "__main__":
    main()
