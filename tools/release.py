"""Build a warp-metal wheel from the Warp fork and test it in a clean environment.

Usage:
    python tools/release.py <fork checkout> --revision NN [--skip-build] [--publish]

Steps: download the stock ``warp-lang`` wheel of the fork's version, build the core library in the
fork, generate the overlay (``tools/generate.py``), build the wheel, install it next to the stock
package in a fresh virtual environment and run ``tools/smoke_test.py`` and, with ``--parity MODEL...``,
``tools/check.py`` (the fork's stub check and Metal tests, MuJoCo Warp's solver and size tests, the physics
gate and the IR audit; a failure stops the release) and the informational ``tools/perf_report.py``. The fork's
``tools/check_metal_stub.py`` also runs before anything is built. With ``--publish`` the wheel
is then attached to a GitHub prerelease named after its version and tagged on ``HEAD``; that needs the
release commit (the version and pin that ``generate.py`` writes to ``pyproject.toml``) to exist already,
so run once without ``--publish``, commit, and run again with it. Needs macOS on Apple Silicon, ``uv``
and, for ``--publish``, ``gh``.
"""

import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NVIDIA_INDEX = "https://pypi.nvidia.com"  # nightlies; releases come from PyPI


def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "fork",
        help="checkout of https://github.com/innate-inc/warp at the commit to release",
    )
    ap.add_argument(
        "--revision",
        required=True,
        help="two digits; start at 01 for each Warp version",
    )
    ap.add_argument(
        "--parity",
        nargs="+",
        metavar="MODEL",
        default=[],
        help="MuJoCo models for tools/physics_parity.py (CPU against Metal); required with --publish",
    )
    ap.add_argument(
        "--mujoco-warp",
        default="mujoco-warp @ git+https://github.com/DavidDobas/mujoco_warp@metal-3.11",
        help="MuJoCo Warp requirement installed for the parity check",
    )
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="use the fork's existing warp/bin/libwarp.dylib",
    )
    ap.add_argument(
        "--publish",
        action="store_true",
        help="create the GitHub prerelease and upload the wheel",
    )
    args = ap.parse_args()
    if args.publish and not args.parity:
        sys.exit("error: --publish needs --parity MODEL...; a wheel is not released without the physics check")
    fork = os.path.abspath(args.fork)
    # the check runs from a temporary directory, so make the model paths absolute
    parity_models = [
        os.path.abspath(m[: -len("+floor")]) + "+floor" if m.endswith("+floor") else os.path.abspath(m)
        for m in args.parity
    ]
    with open(os.path.join(fork, "VERSION.md")) as f:
        warp_version = f.read().strip()

    # A stable Warp release has to resolve from PyPI alone, because that is what `pip install warp-metal`
    # gives users; only nightlies need NVIDIA's index and pre-release resolution.
    nightly = ".dev" in warp_version
    download_index = ["--extra-index-url", NVIDIA_INDEX] if nightly else []
    install_index = (
        ["--prerelease", "allow", "--index-strategy", "unsafe-best-match", "--extra-index-url", NVIDIA_INDEX]
        if nightly
        else []
    )

    # builds without Metal must keep every symbol Warp binds; stop before building anything if they would not
    stub_check = os.path.join(fork, "tools", "check_metal_stub.py")
    if os.path.exists(stub_check):
        run([sys.executable, stub_check, fork])

    with tempfile.TemporaryDirectory() as tmp:
        # stock wheel: what the overlay is compared against and what users will have installed
        run(
            ["uvx", "pip", "download", "--no-deps", "--only-binary=:all:", "--platform", "macosx_11_0_arm64",
             "--python-version", "3.12", *download_index, f"warp-lang=={warp_version}", "-d", tmp]
        )  # fmt: skip
        stock = os.path.join(tmp, "stock")
        with zipfile.ZipFile(glob.glob(os.path.join(tmp, "warp_lang-*.whl"))[0]) as wheel:
            wheel.extractall(stock)

        if not args.skip_build:
            # only the core library is shipped; the LLVM helper comes from the stock wheel
            run(["uv", "run", "build_lib.py", "--no-standalone"], cwd=fork)
        run([sys.executable, os.path.join(ROOT, "tools", "generate.py"), fork, "--stock", stock,
             "--revision", args.revision])  # fmt: skip

        if args.publish:
            require_release_commit()

        shutil.rmtree(os.path.join(ROOT, "dist"), ignore_errors=True)
        run(["uv", "build", "--wheel"], cwd=ROOT)
        (wheel_path,) = glob.glob(os.path.join(ROOT, "dist", "warp_metal-*.whl"))

        # clean environment, run from outside both source trees
        env = os.path.join(tmp, "env")
        run(["uv", "venv", "--python", "3.12", env])
        python = os.path.join(env, "bin", "python")
        run(["uv", "pip", "install", "--python", python, *install_index, wheel_path])
        run([python, os.path.join(ROOT, "tools", "smoke_test.py")], cwd=tmp)
        if parity_models:
            # every check, including the physics gate, in this clean environment; see tools/check.py
            run(["uv", "pip", "install", "--python", python, *install_index, args.mujoco_warp, "pytest"])
            run(
                [sys.executable, os.path.join(ROOT, "tools", "check.py"), fork, *parity_models, "--python", python,
                 "--report", os.path.join(ROOT, "dist", "check.txt"),
                 "--parity-report", os.path.join(ROOT, "dist", "physics_parity.txt")],
                cwd=tmp,
            )  # fmt: skip
            write_perf_report(python, parity_models, os.path.join(ROOT, "dist", "perf.txt"), tmp)

    print(f"\nbuilt and tested {wheel_path}")
    if args.publish:
        version = os.path.basename(wheel_path).split("-")[1]
        committed = subprocess.check_output(["git", "show", "HEAD:pyproject.toml"], cwd=ROOT, text=True)
        if f'version = "{version}"' not in committed:
            sys.exit(f"error: HEAD does not declare version {version}; commit pyproject.toml before publishing")
        target = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        with open(wheel_path, "rb") as f:
            wheel_sha256 = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(ROOT, "dist", "physics_parity.txt")) as f:
            parity_report = f.read()
        with open(os.path.join(ROOT, "dist", "check.txt")) as f:
            check_report = f.read()
        with open(os.path.join(ROOT, "dist", "perf.txt")) as f:
            perf_report = f.read()
        notes = (
            f"Overlays warp-lang {warp_version}. Built from the Warp fork at "
            f"https://github.com/innate-inc/warp/commit/{fork_commit(fork)}.\n\n"
            f"SHA-256 of `{os.path.basename(wheel_path)}`, which the file published to PyPI must match:\n\n"
            f"```\n{wheel_sha256}\n```\n\n"
            "Physics parity against the CPU and against MuJoCo (tools/physics_parity.py), run on this wheel in a "
            f"clean environment:\n\n```\n{parity_report}```\n\n"
            f"All checks (tools/check.py) on this wheel:\n\n```\n{check_report}```\n\n"
            "Physics step timing (tools/perf_report.py; informational, depends on other GPU load):\n\n"
            f"```\n{perf_report}```\n"
        )
        run(
            ["gh", "release", "create", f"v{version}", wheel_path, "--prerelease", "--target", target,
             "--title", f"warp-metal {version}", "--notes", notes],
            cwd=ROOT,
        )  # fmt: skip


def write_perf_report(python, models, path, cwd):
    """The G1 timing for the release notes; never stops a release."""
    model = next((m for m in models if "g1" in os.path.basename(m.removesuffix("+floor")).lower()), None)
    if model is None:
        text = "no G1 model among the parity models; nothing measured\n"
    else:
        result = subprocess.run(
            [python, os.path.join(ROOT, "tools", "perf_report.py"), model],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [line for line in result.stdout.splitlines() if not line.startswith(("Module ", "Warp "))]
        text = "\n".join(lines) + "\n" if result.returncode == 0 else f"perf report failed:\n{result.stderr[-2000:]}\n"
    with open(path, "w") as f:
        f.write(text)
    print(text)


def require_release_commit():
    """Stop unless HEAD already carries the version and pin that generate.py has just written.

    The release tag is put on HEAD, so the tagged tree has to describe the wheel that is attached to it.
    """
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        sys.exit(
            "error: --publish needs the release commit first. generate.py has updated the version or the "
            "warp-lang pin (or the tree has other changes):\n" + dirty + "\n"
            "Commit them, then run this command again."
        )


def fork_commit(fork):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=fork, text=True).strip()


if __name__ == "__main__":
    main()
