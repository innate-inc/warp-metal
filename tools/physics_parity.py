"""Release gate: MuJoCo Warp on Metal must agree with the CPU and with MuJoCo itself.

Usage (in an environment that has the wheel under test, ``mujoco-warp`` and ``mujoco``):

    python tools/physics_parity.py [--steps N] [--report FILE] MODEL.xml [MODEL.xml+floor ...]

``+floor`` after a path adds a ground plane just below the model, so that a robot description without a
scene still exercises contacts.

Four checks per model:

1. The acceleration of the first step on the CPU and on Metal against ``mj_forward`` of MuJoCo (C). The CPU
   and Metal runs share the overlay's code, so only an outside reference can catch a defect they have in
   common.
2. The acceleration of one Metal forward at 8 random states (perturbed positions, random velocities and
   controls) against ``mj_forward``, within 5e-3 relative. Not chaotic, so it is the sharp check: the
   32 < nv <= 40 Cholesky bug of 1.17.0.2 gives errors of about 20 on the G1.
3. Positions and velocities after ``--steps`` steps, Metal against the CPU. Contact-rich models are chaotic,
   so the tolerance is the larger of the fixed one and 3 times the spread of CPU runs whose velocities were
   perturbed by about 1e-7 (the "band" in the report).
4. The largest number of active constraints seen, which must be equal.

The report also lists which kernels Warp's Metal code generator compiled with the fused register Cholesky
(Adjoint._match_metal_fused_cholesky), and the run fails if a Newton model with at most 40 degrees of freedom
ran but MuJoCo Warp's solver Cholesky kernel no longer took that path, or a model with a gathered mass-matrix
block of at most 40 degrees of freedom ran but the block factorization kernel no longer took it. The run uses a
fresh kernel cache so every module is generated and counted.

A run fails unless the models include one with more than 32 and one with more than 64 degrees of freedom and
at least one active constraint: a factorization bug that only showed above 32 degrees of freedom once shipped
in five wheels, while the test suites passed and the robot "trained".
"""

import argparse
import hashlib
import os
import sys
import tempfile

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

QACC_RTOL, QACC_ATOL = 1e-3, 1e-4  # first step against MuJoCo (C); both devices agree to about 5e-5
QPOS_ATOL = 1e-3  # after --steps steps; contact-rich models drift chaotically beyond that
QVEL_RTOL, QVEL_ATOL = 2e-2, 1e-4  # the absolute floor keeps a model at rest from failing on noise
CHAOS_RUNS, CHAOS_NOISE, CHAOS_FACTOR = 4, 1e-7, 3.0  # perturbed CPU runs that measure a model's sensitivity
RANDOM_WORLDS, RANDOM_QACC_RTOL = 8, 5e-3  # random states, one forward each, against mj_forward; good builds <= 2.2e-3


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


def random_state_qacc_error(mjm, rng):
    """Worst relative qacc error of one Metal forward at random states against MuJoCo's mj_forward.

    Unlike the stepped comparison this is not chaotic: every world is one forward from a known state, so a
    factorization or solver defect shows at full size (a 32 < nv <= 40 Cholesky bug gave errors of about 20).
    """
    mjd = mujoco.MjData(mjm)
    if mjm.nkey:
        mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    qpos = np.tile(mjd.qpos, (RANDOM_WORLDS, 1))
    qvel = rng.normal(0.0, 0.5, (RANDOM_WORLDS, mjm.nv))
    ctrl = rng.uniform(-0.3, 0.3, (RANDOM_WORLDS, mjm.nu))
    for w in range(RANDOM_WORLDS):
        mujoco.mj_integratePos(mjm, qpos[w], rng.normal(0.0, 0.05, mjm.nv), 1.0)
    ref = np.zeros((RANDOM_WORLDS, mjm.nv))
    for w in range(RANDOM_WORLDS):
        mjd.qpos[:], mjd.qvel[:], mjd.ctrl[:] = qpos[w], qvel[w], ctrl[w]
        mujoco.mj_forward(mjm, mjd)
        ref[w] = mjd.qacc
    with wp.ScopedDevice("metal:0"):
        m = mjw.put_model(mjm)
        if hasattr(m.opt, "warn_overflow"):
            m.opt.warn_overflow = 0
        d = mjw.put_data(mjm, mjd, nworld=RANDOM_WORLDS, nconmax=max(256, 8 * mjm.ngeom), njmax=max(512, 12 * mjm.nv))
        d.qpos.assign(qpos.astype(np.float32))
        d.qvel.assign(qvel.astype(np.float32))
        d.ctrl.assign(ctrl.astype(np.float32))
        mjw.forward(m, d)
        wp.synchronize()
        got = d.qacc.numpy()
    if not np.isfinite(got).all():
        return float("inf")
    return float((np.abs(got - ref).max(axis=1) / (1.0 + np.abs(ref).max(axis=1))).max())


def chaos_band(mjm, mjd, steps, q_cpu, v_cpu, rng):
    """How far CPU runs from velocities perturbed by about 1e-7 end up after ``steps`` steps (qpos, qvel)."""
    band_q = band_v = 0.0
    for _ in range(CHAOS_RUNS):
        perturbed = mujoco.MjData(mjm)
        perturbed.qpos[:], perturbed.qvel[:], perturbed.ctrl[:] = mjd.qpos, mjd.qvel, mjd.ctrl
        perturbed.qvel[:] += CHAOS_NOISE * (1.0 + np.abs(mjd.qvel)) * rng.standard_normal(mjm.nv)
        mujoco.mj_forward(mjm, perturbed)
        q, v, _, _ = simulate(mjm, perturbed, "cpu", steps)
        band_q = max(band_q, float(np.abs(q - q_cpu).max()))
        band_v = max(band_v, float(np.abs(v - v_cpu).max()))
    return band_q, band_v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--report", help="also write the report to this file")
    ap.add_argument("--kernel-cache", help="generate kernels into this (empty) directory instead of a temporary one")
    args = ap.parse_args()
    wp.config.quiet = True
    # generate every module afresh, so the code generator's Metal rewrites below are all seen and counted
    wp.config.kernel_cache_dir = args.kernel_cache or tempfile.mkdtemp(prefix="warp-parity-")
    if not wp.is_metal_available():
        sys.exit("error: no Metal device; is the warp-metal overlay active?")

    header = (
        f"warp {wp.config.version}, mujoco {mujoco.__version__}, mujoco_warp {getattr(mjw, '__version__', '?')}, "
        f"device {wp.get_device('metal:0').name}, {args.steps} steps"
    )
    lines = [header]
    failures, sizes, contacts = 0, [], False
    expect_fused = False  # a Newton model small enough for the register Cholesky ran on Metal
    expect_block_fused = False  # a model with a gathered mass-matrix block the register Cholesky can take
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
        rng = np.random.default_rng(int(digest, 16) % (2**32))  # the same draws for the same model every run
        band_q, band_v = chaos_band(mjm, mjd, args.steps, q_cpu, v_cpu, rng)
        random_error = random_state_qacc_error(mjm, rng)

        qacc_limit = QACC_ATOL + QACC_RTOL * float(np.abs(qacc_ref).max())
        da_cpu = float(np.abs(a_cpu - qacc_ref).max())
        da_gpu = float(np.abs(a_gpu - qacc_ref).max())
        dq = float(np.abs(q_cpu - q_gpu).max())
        dv = float(np.abs(v_cpu - v_gpu).max())
        # a contact-rich model's trajectory depends on the order of floating-point sums; the tolerance after
        # several steps is the larger of the fixed one and CHAOS_FACTOR times the spread of perturbed CPU runs
        qpos_limit = max(QPOS_ATOL, CHAOS_FACTOR * band_q)
        qvel_limit = max(QVEL_ATOL + QVEL_RTOL * float(np.abs(v_cpu).max()), CHAOS_FACTOR * band_v)
        problems = []
        if not (np.isfinite(q_gpu).all() and np.isfinite(v_gpu).all()):
            problems.append("non-finite state on Metal")
        if da_cpu > qacc_limit:
            problems.append(f"cpu qacc differs from MuJoCo by {da_cpu:.1e} (limit {qacc_limit:.1e})")
        if da_gpu > qacc_limit:
            problems.append(f"metal qacc differs from MuJoCo by {da_gpu:.1e} (limit {qacc_limit:.1e})")
        if not dq < qpos_limit:
            problems.append(f"qpos differs by {dq:.1e} (limit {qpos_limit:.1e})")
        if not dv < qvel_limit:
            problems.append(f"qvel differs by {dv:.1e} (limit {qvel_limit:.1e})")
        if not random_error < RANDOM_QACC_RTOL:
            problems.append(f"qacc at random states differs from MuJoCo by {random_error:.1e} (relative)")
        if nefc_cpu != nefc_gpu:
            problems.append(f"constraint counts differ: cpu {nefc_cpu}, metal {nefc_gpu}")

        failures += bool(problems)
        sizes.append(mjm.nv)
        expect_fused |= mjm.nv <= 40 and mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
        with wp.ScopedDevice("cpu"):
            tiles = getattr(mjw.put_model(mjm), "M_tiles", ())
        expect_block_fused |= any(t.size <= 40 and t.elemid.size > 0 for t in tiles)
        contacts |= nefc_cpu > 0
        lines.append(
            f"{'ok      ' if not problems else 'MISMATCH'} {name} [{digest}]: nv {mjm.nv}, nefc {nefc_cpu}/{nefc_gpu}, "
            f"qacc vs MuJoCo cpu {da_cpu:.1e} metal {da_gpu:.1e}, random-state qacc {random_error:.1e}, "
            f"dqpos {dq:.1e} (band {band_q:.1e}), dqvel {dv:.1e} (band {band_v:.1e})"
            + "".join(f"\n           - {p}" for p in problems)
        )
        print(lines[-1], flush=True)

    # Warp's Metal code generator runs MuJoCo Warp's dense solver Cholesky from registers when the kernel has the
    # exact load / factor / solve shape it matches (codegen.py, Adjoint._match_metal_fused_cholesky). A MuJoCo
    # Warp change to that kernel would silently lose the speedup, so the gate checks that it still matched.
    rewrites = getattr(getattr(wp._src.codegen, "Adjoint", None), "metal_fused_cholesky_rewrites", None)
    solver_fused = block_fused = 0
    if rewrites is None:
        lines.append("metal fused Cholesky: not in this Warp build")
    else:
        solver_fused = sum(n for key, n in rewrites.items() if key.startswith("_update_gradient_cholesky."))
        block_fused = sum(n for key, n in rewrites.items() if key.startswith("_tile_cholesky_factorize_solve_block."))
        lines.append("metal fused Cholesky: " + (", ".join(f"{k} x{n}" for k, n in sorted(rewrites.items())) or "none"))
    print(lines[-1], flush=True)

    verdict = None
    if not any(32 < nv <= 64 for nv in sizes) or not any(nv > 64 for nv in sizes):
        verdict = f"error: model sizes {sorted(sizes)} do not cover both 32 < nv <= 64 and nv > 64"
    elif not contacts:
        verdict = "error: no model had an active constraint; add a scene with contacts or a +floor model"
    elif failures:
        verdict = f"FAILED: {failures} of {len(sizes)} models"
    elif rewrites is not None and expect_fused and not solver_fused:
        verdict = (
            "FAILED: MuJoCo Warp's solver Cholesky kernel (_update_gradient_cholesky) no longer matches Warp's Metal "
            "fused-Cholesky rewrite, so it would run without it; see Adjoint._match_metal_fused_cholesky"
        )
    elif rewrites is not None and expect_block_fused and not block_fused:
        verdict = (
            "FAILED: MuJoCo Warp's block factorization kernel (_tile_cholesky_factorize_solve_block) no longer "
            "matches Warp's Metal fused-Cholesky rewrite, so it would run without it; see "
            "Adjoint._match_metal_fused_cholesky"
        )
    lines.append(verdict or f"parity holds for {len(sizes)} models")
    if args.report:
        with open(args.report, "w") as f:
            f.write("\n".join(lines) + "\n")
    if verdict:
        sys.exit(verdict)
    print(lines[-1])


if __name__ == "__main__":
    main()
