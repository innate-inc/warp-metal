# warp-metal

Run [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels on Apple Silicon GPUs.

`warp-metal` adds a `metal:0` device to the stock `warp-lang` package. Projects built on Warp,
such as [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) and
[mjlab](https://github.com/mujocolab/mjlab), keep their normal dependency on `warp-lang`; on a Mac
you add this one package and simulate on the GPU.

This is an independent open-source project, developed at https://github.com/innate-inc. It is not
affiliated with or endorsed by NVIDIA; Warp is NVIDIA's project and trademark.

```
uv add warp-metal        # or: pip install warp-metal
```

This installs `warp-lang 1.17.0` from PyPI next to it; nothing else is needed.

```python
import warp as wp

@wp.kernel
def scale(a: wp.array[float], s: float):
    i = wp.tid()
    a[i] = a[i] * s

a = wp.array([1.0, 2.0, 3.0], dtype=float, device="metal:0")
wp.launch(scale, dim=3, inputs=[a, 2.0], device="metal:0")
print(a.numpy())  # [2. 4. 6.]
```

Nothing has to be imported: the package installs a small startup hook so that `import warp`
already knows about Metal. `warp_metal.status()` tells you whether the hook is active.

## What works

Kernels, tiles (cooperative, with a threadgroup arena), backward kernels and tapes, graph capture
and replay including `capture_if`/`capture_while`, meshes, BVHs, hash grids, NanoVDB volumes,
textures, half and 64-bit atomics, references, printing and assertions from kernels, DLPack and
Torch interop over unified memory, and MuJoCo Warp's full test suite. Warp's own test suite runs on
Metal with the exceptions listed below.

Arrays on `metal:0` live in unified memory: their host pointer is valid, `wp.to_torch` returns a
zero-copy CPU tensor, and NumPy arrays or Torch CPU tensors can be passed to kernels directly.

## What does not work

- **float64**: Apple GPUs have no double type. Kernels that use it raise at launch.
- **Some atomics are not atomic**: Metal only has 32-bit atomics, so 8-bit operations, most 16-bit and 64-bit
  ones, and `&=`/`|=`/`^=` on those types are plain read-modify-write.
- **Spinlocks across threads**: Apple GPUs do not guarantee forward progress between SIMD lanes.
- `wp.fixedarray`, fabric arrays, deterministic scatter mode, saveable (APIC) captures.
- Native snippets must be valid Metal: pointer casts need `WP_THREAD`/`WP_DEVICE`, no `long long`.
- Transcendental functions may differ from NumPy by 1 ulp.
- Apple's GPU compiler can miscompile kernels with several inlined, fully unrolled products of 4x4 or larger
  matrices. Warp's own matrix product avoids it; check hand-written loop nests of that size against the CPU.

The full list, including which atomic operations are not atomic on Metal, is in the fork's user guide:
https://github.com/innate-inc/warp/blob/main/docs/user_guide/metal.rst

## Versions

A release is named after the `warp-lang` version it overlays plus a revision: `warp-metal 1.17.0.2`
overlays `warp-lang 1.17.0`, and the last number counts the `warp-metal` builds for that Warp version. Preview builds for Warp nightlies are attached to the
GitHub releases of this repository (the revision is then two digits appended to the nightly's date,
and the nightly itself comes from NVIDIA's package index, `https://pypi.nvidia.com`).

The package overlays modules of one exact `warp-lang` version and pins it as a dependency
(see `pyproject.toml`). A new `warp-lang` release needs a matching `warp-metal` release; until it
ships, stay on the previous Warp. If the versions do not match, Warp loads unmodified and a
message on stderr says why. `WARP_METAL_DISABLE=1` (or `true`) turns the overlay off.

Requires macOS 15 or newer on Apple Silicon (Metal 3.2).

## Using it with mjlab and MuJoCo Warp

mjlab and MuJoCo Warp need two small, generic changes to run on a non-CUDA GPU device. They are
carried in branches of David Dobas's forks and have not been proposed upstream yet
([MuJoCo Warp](https://github.com/DavidDobas/mujoco_warp/pull/1),
[mjlab](https://github.com/DavidDobas/mjlab/pull/1)). Until they are upstream, pin those branches:

```toml
[tool.uv.sources]
mjlab       = { git = "https://github.com/DavidDobas/mjlab", branch = "metal" }
mujoco-warp = { git = "https://github.com/DavidDobas/mujoco_warp", branch = "metal-3.11" }
```

(`metal-3.11` is the version mjlab pins with the patch applied; `metal` carries the same patch on
MuJoCo Warp's main branch and is what the pull request is based on.)

For [microduck_rl](https://github.com/DavidDobas/microduck_rl/tree/metal) the `metal` branch
already carries these sources: clone it and `uv run train ...` works on a Mac as is.

Then train as usual, with the agent on Torch's MPS device:

```
MJLAB_AGENT_DEVICE=mps uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 1024
```

On an M5 Pro, one training iteration of that task at 1024 environments takes about 1.0 s
(RTX 5090: 0.3 s). The physics kernels are limited by threadgroup memory on Apple GPUs, so the gap
widens with more environments.

## How it is built

This repository holds no copy of Warp. It contains the import hook (`src/warp_metal/_bootstrap.py`),
the tools that build a wheel, and this README. The backend itself is developed and tested in a fork
of Warp, https://github.com/innate-inc/warp, whose `main` branch is NVIDIA's `main` plus the Metal backend.
Releases for a stable Warp version are built from a branch of that fork based on NVIDIA's release tag
(`metal-1.17` for `warp-lang 1.17.0`).

A release is built from a checkout of that fork:

```
python tools/release.py <fork checkout> --revision 01
```

1. downloads the stock `warp-lang` wheel of the fork's version and checks that the fork contains the
   upstream commit that wheel was built from,
2. builds the core library in the fork,
3. copies into the wheel every `warp._src` module that differs from the stock package, the native
   headers kernels compile against, and the library (`tools/generate.py`; the generated files are
   ignored by git),
4. builds the wheel, installs it next to the stock package in a fresh environment and runs
   `tools/smoke_test.py` on the GPU.
5. with `--parity MODEL...`, steps MuJoCo Warp models on the CPU and on Metal and compares the state
   (`tools/physics_parity.py`). The models must include more than 32 and more than 64 degrees of
   freedom and active contacts, and `--publish` refuses to run without this check: a factorization bug
   that only showed above 32 degrees of freedom was once found just before the first PyPI upload.

Publishing to PyPI is a separate, manual step: the `Publish to PyPI` workflow uploads the wheel attached to a
GitHub release after checking its SHA-256, through PyPI trusted publishing, so no token is stored.

The wheel records the fork commit it was built from (`warp_metal.FORK_COMMIT`); that commit is the
readable source of everything the wheel overlays. The LLVM helper library used for CPU kernels comes
from the stock `warp-lang` package.

A source checkout of this repository is not usable as a package on its own: without the generated
overlay, `warp_metal.status()` reports that it is inactive and Warp loads unmodified.

## License

Apache 2.0, like Warp. The overlay modules and headers are derived from NVIDIA Warp; see NOTICE.
