"""Everything to run before merging a Metal backend change or releasing a wheel, with one summary.

Usage:
    python tools/check.py FORK --python PYTHON MODEL [MODEL ...]

FORK is the checkout of https://github.com/innate-inc/warp the wheel was built from. PYTHON is a clean
environment with that wheel, ``mujoco-warp``, ``mujoco`` and ``pytest`` installed (tools/release.py makes one;
see the README for doing it by hand). MODEL... are the physics gate's models (tools/physics_parity.py).

Checks, each reported as PASS, FAIL or SKIP:

1. stub: FORK/tools/check_metal_stub.py, so builds without Metal keep every symbol Warp binds.
2. smoke: tools/smoke_test.py.
3. warp metal tests: FORK/warp/tests/test_metal.py against the installed wheel (includes the Cholesky
   threshold sweeps, the fused-Cholesky and 64-bit division guards when the fork has them).
4. mujoco_warp solver tests and size tests: the installed MuJoCo Warp's solver_test.py, and the
   test_solve_m_single_tree / test_newton_search_direction size tests (SKIP if that MuJoCo Warp lacks them).
5. physics gate: tools/physics_parity.py with a fresh kernel cache.
6. ir audit: every Metal module the gate generated, compiled with the Metal compiler, must be free of 64-bit
   division by a runtime value (SKIP without xcrun metal).

Exits with status 1 if any check fails. ``--require NAME`` makes a SKIP of that check a failure; tools/release.py
requires the stub check, the size tests and the IR audit unless given ``--allow-skip``. Timing is not checked
here (tools/perf_report.py is informational).
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIVISION = re.compile(r"= (?:udiv|urem|sdiv|srem) (?:exact )?i64 [^,]+, (%[\w.]+)")


def run(cmd, cwd=None, env=None, log=None):
    """Run cmd; return (exit code, output). Output also goes to the log file if given."""
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    if log:
        with open(log, "a") as f:
            f.write(f"$ {' '.join(cmd)}\n{output}\n")
    return result.returncode, output


def last_match(pattern, text, default=""):
    found = re.findall(pattern, text, re.MULTILINE)
    return found[-1] if found else default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fork")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--python", required=True, help="python of the clean environment with the wheel")
    ap.add_argument("--report", help="also write the summary to this file")
    ap.add_argument("--parity-report", help="write the physics gate's report to this file")
    ap.add_argument("--log", help="write every command's output to this file (default: a temporary file)")
    ap.add_argument(
        "--require",
        action="append",
        default=[],
        choices=["stub", "mujoco_warp size tests", "ir audit"],
        help="treat a SKIP of this check as a failure (tools/release.py requires all three)",
    )
    args = ap.parse_args()
    fork, python = os.path.abspath(args.fork), args.python
    models = [
        os.path.abspath(m.removesuffix("+floor")) + ("+floor" if m.endswith("+floor") else "") for m in args.models
    ]
    tmp = tempfile.mkdtemp(prefix="warp-metal-check-")
    log = args.log or os.path.join(tmp, "check.log")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "WARP_METAL_DISABLE")}
    env["WARP_CACHE_PATH"] = os.path.join(tmp, "kernels-tests")
    results = []

    def record(name, status, detail, start):
        if status == "SKIP" and name in args.require:
            status, detail = "FAIL", f"skipped, but required: {detail}"
        results.append((name, status, detail, time.time() - start))
        print(f"{status:4s}  {name}: {detail}", flush=True)

    # 1. stub
    start = time.time()
    tool = os.path.join(fork, "tools", "check_metal_stub.py")
    if os.path.exists(tool):
        code, out = run([sys.executable, tool, fork], log=log)
        record("stub", "PASS" if code == 0 else "FAIL", out.strip().splitlines()[-1] if out.strip() else "", start)
    else:
        record("stub", "SKIP", "the fork has no tools/check_metal_stub.py", start)

    # 2. smoke
    start = time.time()
    code, out = run([python, os.path.join(ROOT, "tools", "smoke_test.py")], cwd=tmp, env=env, log=log)
    record("smoke", "PASS" if code == 0 else "FAIL", last_match(r"^ok: .*", out) or f"exit {code}", start)

    # 3. warp metal tests, from the fork against the installed wheel
    start = time.time()
    tests = os.path.join(tmp, "warp_tests")
    os.makedirs(tests)
    shutil.copy(os.path.join(fork, "warp", "tests", "test_metal.py"), tests)
    code, out = run([python, "-m", "unittest", "test_metal"], cwd=tests, env=env, log=log)
    ran, verdict = last_match(r"^Ran \d+ tests?", out), last_match(r"^(OK.*|FAILED.*)$", out)
    summary = f"{ran} {verdict}".strip()
    record("warp metal tests", "PASS" if code == 0 else "FAIL", summary, start)

    # 4. MuJoCo Warp tests from the installed package
    code, out = run([python, "-c", "import mujoco_warp, os; print(os.path.dirname(mujoco_warp.__file__))"], env=env)
    package = out.strip().splitlines()[-1] if code == 0 else None
    for name, files, selection in (
        ("mujoco_warp solver tests", ["solver_test.py"], None),
        (
            "mujoco_warp size tests",
            ["smooth_test.py", "solver_test.py"],
            "test_solve_m_single_tree or test_newton_search_direction",
        ),
    ):
        start = time.time()
        if package is None:
            record(name, "FAIL", "mujoco_warp is not importable in the environment", start)
            continue
        cmd = [python, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
        cmd += [os.path.join(package, "_src", f) for f in files] + (["-k", selection] if selection else [])
        code, out = run(cmd, cwd=tmp, env=env, log=log)
        summary = last_match(r"^.*\d+ (?:passed|failed|deselected).*$", out) or f"exit {code}"
        status = "SKIP" if code == 5 else ("PASS" if code == 0 else "FAIL")  # 5: no test selected
        record(name, status, "not in this MuJoCo Warp" if code == 5 else summary, start)

    # 5. physics gate, generating every kernel into a cache the IR audit reads
    start = time.time()
    kernels = os.path.join(tmp, "kernels-gate")
    os.makedirs(kernels)
    parity_report = args.parity_report or os.path.join(tmp, "physics_parity.txt")
    cmd = [python, os.path.join(ROOT, "tools", "physics_parity.py"), "--kernel-cache", kernels]
    code, out = run([*cmd, "--report", parity_report, *models], cwd=tmp, env=env, log=log)
    record(
        "physics gate", "PASS" if code == 0 else "FAIL", last_match(r"^(parity holds.*|FAILED.*|error.*)$", out), start
    )

    # 6. IR audit of the gate's Metal modules
    start = time.time()
    if subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, check=False).returncode != 0:
        record("ir audit", "SKIP", "the Metal compiler (xcrun metal, from Xcode) is not installed", start)
    else:
        sources = sorted(glob.glob(os.path.join(kernels, "**", "*.metal"), recursive=True))
        offenders, failed = [], []
        for source in sources:
            ir = os.path.join(tmp, "module.ll")
            code, _ = run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2", "-O2", "-fno-fast-math", "-S",
                           "-emit-llvm", source, "-o", ir])  # fmt: skip
            if code != 0:
                failed.append(os.path.basename(source))
                continue
            with open(ir) as f:
                if any(DIVISION.search(line) for line in f):
                    offenders.append(os.path.basename(source))
        detail = f"{len(sources)} modules, {len(offenders)} with 64-bit division by a runtime value"
        if failed:
            detail += f", {len(failed)} did not compile: {', '.join(failed[:3])}"
        if offenders:
            detail += ": " + ", ".join(offenders[:5])
        record("ir audit", "PASS" if not offenders and not failed and sources else "FAIL", detail, start)

    failures = [r for r in results if r[1] == "FAIL"]
    lines = [f"{status:4s}  {name:28s} {detail}  ({seconds:.0f} s)" for name, status, detail, seconds in results]
    lines.append(f"{'FAILED' if failures else 'all checks passed'}: {len(results) - len(failures)} of {len(results)} ok"
                 f" (log: {log})")  # fmt: skip
    print("\n" + "\n".join(lines))
    if args.report:
        with open(args.report, "w") as f:
            f.write("\n".join(lines[:-1]) + "\n" + lines[-1].split(" (log:")[0] + "\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
