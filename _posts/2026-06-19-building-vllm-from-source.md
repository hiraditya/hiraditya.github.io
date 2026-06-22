---
title: "Building vLLM from Source: A Field Guide (with all the pitfalls)"
date: 2026-06-19 08:00:00 -0700
categories: [Systems, Compilers, Engineering]
tags: [vllm, build, cuda, python, ubuntu]
description: "A step-by-step field guide to building vLLM from source on Ubuntu 26.04, covering Python 3.14 compatibility, CUDA driver issues, and toolchain pitfalls."
---

Building vLLM[^1] from source sounds like a `pip install -e .` away. In practice, on a
fresh machine with a recent OS and a recent Python, you hit a chain of version-skew,
driver, and toolchain issues that each fail with a cryptic message. This post walks
through a real end-to-end build on an **AWS g5 instance (NVIDIA A10G)** running
**Ubuntu 26.04 + Python 3.14**, documenting every error encountered and the fix.

The target was a CUDA build of a vLLM fork. The same playbook applies to a stock
`vllm-project/vllm` checkout.

______________________________________________________________________

## TL;DR — the working recipe

```bash
# 1. Confirm you actually have a GPU (see "Pitfall 1" — easy to get wrong)
lspci | grep -i nvidia        # hardware present?
nvidia-smi                    # driver working?

# 2. Driver (if nvidia-smi fails but lspci shows the GPU)
sudo apt-get install -y nvidia-driver-575-open nvidia-modprobe dkms
sudo modprobe -r nouveau && sudo modprobe nvidia   # or reboot

# 3. Virtual env
python3 -m venv ~/go/venv && source ~/go/venv/bin/activate
pip install --upgrade pip

# 4. CUDA torch + a CONSISTENT pip CUDA toolkit (critical: one minor version)
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0   # default index = CUDA build
pip install "cuda-toolkit[nvcc]==13.3.0" "nvidia-cuda-runtime==13.3.29" \
            "nvidia-cuda-nvrtc==13.3.33" "nvidia-cublas==13.3.0.5"

# 5. Assemble CUDA_HOME from the pip layout
export CUDA_HOME=$VIRTUAL_ENV/lib/python3.*/site-packages/nvidia/cu13
ln -sfn $CUDA_HOME/lib $CUDA_HOME/lib64
( cd $CUDA_HOME/lib && for f in lib*.so.*; do ln -sf "$f" "${f%%.so.*}.so"; done )
mkdir -p $CUDA_HOME/lib/stubs
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so $CUDA_HOME/lib/stubs/libcuda.so

# 6. Build (scope arch to YOUR GPU — A10G is sm_86)
export PATH=$CUDA_HOME/bin:$PATH CUDACXX=$CUDA_HOME/bin/nvcc
export VLLM_TARGET_DEVICE=cuda TORCH_CUDA_ARCH_LIST="8.6+PTX"
export MAX_JOBS=12 NVCC_THREADS=2
export CMAKE_ARGS="-DCUDAToolkit_ROOT=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"
pip install -v -e . --no-build-isolation
```

Read on for *why* each line is there and what breaks without it.

______________________________________________________________________

## Prerequisites & how to check them

Before anything else, take an inventory. Getting this wrong wastes the most time —
including the most embarrassing pitfall of all.

| Requirement | How to check | Notes |
|---|---|---|
| **A GPU (and which one)** | `lspci \| grep -i nvidia` | Determines CUDA vs CPU build. **Don't trust `nvidia-smi` alone** — see Pitfall 1. |
| GPU driver loaded | `nvidia-smi` | If it fails but `lspci` shows a GPU, the driver isn't installed/loaded. |
| Compute capability | `nvidia-smi --query-gpu=compute_cap --format=csv` | A10G = `8.6`. You build kernels for this. |
| CPU flags (CPU build only) | `lscpu \| grep -oE 'avx512f\|avx2'` | vLLM CPU wants AVX512; AVX2 works with limited features. |
| Compiler | `gcc --version` | vLLM recommends gcc 12–13; newer (15) mostly works but watch nvcc host-compiler limits. |
| Python | `python3 --version` | Check the repo's `requires-python` in `pyproject.toml`. |
| RAM / cores | `nproc; free -h` | CUDA compiles are RAM-hungry (~2–3 GB per parallel job). |
| build tools | `cmake --version; ninja --version` | vLLM needs cmake ≥ 3.26. |

### Pitfall 1: "There's no GPU here" — when there definitely is

This one cost us a whole CPU build. The very first check was:

```bash
nvidia-smi   # → command not found
```

Conclusion drawn: *no GPU, do a CPU build.* **Wrong.** `nvidia-smi` missing only means
the **driver/userspace tools aren't installed** — it says nothing about the hardware.
The actual hardware check is:

```bash
$ lspci | grep -i nvidia
00:1e.0 3D controller: NVIDIA Corporation GA102GL [A10G] (rev a1)
```

The A10G was there the whole time; it just had no driver. **Always check `lspci` (or
`/proc/driver/nvidia`, `ls /dev/nvidia*`) before concluding "no GPU."** On cloud
instances that aren't "Deep Learning AMIs," a bare GPU with no driver is the norm, not
the exception.

> **Lesson:** `lspci` detects hardware. `nvidia-smi` detects a *working driver*. They
> answer different questions. Decide CPU-vs-GPU from `lspci`.

______________________________________________________________________

## Step 2: Install and load the NVIDIA driver

`lspci` shows the GPU, `nvidia-smi` is missing → install the driver.

```bash
sudo apt-get update
sudo apt-get install -y dkms build-essential \
     linux-headers-$(uname -r) \
     nvidia-driver-575-open
```

We used the **open-kernel** variant (`-open`), which is NVIDIA's recommendation for
Ampere and newer (A10G is Ampere). The `575` metapackage pulled driver `580.159.03`.

### Pitfall 2: `modprobe nvidia` → "No such device" (nouveau owns the GPU)

```text
$ sudo modprobe nvidia
modprobe: ERROR: could not insert 'nvidia': No such device

$ dmesg | grep NVRM
NVRM: GPU 0000:00:1e.0 is already bound to nouveau.
```

The open-source **nouveau** driver grabs the GPU at boot. The NVIDIA module can't bind
while nouveau holds it. Fix — blacklist, unbind, and load:

```bash
echo -e "blacklist nouveau\noptions nouveau modeset=0" | \
    sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo -n "0000:00:1e.0" | sudo tee /sys/bus/pci/drivers/nouveau/unbind
sudo rmmod nouveau
sudo modprobe nvidia
sudo update-initramfs -u    # make the blacklist survive reboots
```

If `rmmod nouveau` complains it's in use (e.g. a display manager), a reboot after the
blacklist + initramfs update achieves the same thing cleanly.

### Pitfall 3: `nvidia-smi` works but CUDA returns error 999 ("unknown error")

This is the subtle one. After loading the module:

```text
$ nvidia-smi          # works, shows the A10G
$ python -c "import torch; print(torch.cuda.is_available())"
RuntimeError: CUDA unknown error ...    # False
```

A direct driver-API probe confirmed the runtime was broken even
though `nvidia-smi` was fine:

```python
import ctypes
ctypes.CDLL("libcuda.so.1").cuInit(0)   # → 999 (CUDA_ERROR_UNKNOWN)
```

Two distinct causes, both worth knowing:

1. **Stale/incorrect UVM device nodes.** `nvidia-smi` uses `/dev/nvidia0` +
   `/dev/nvidiactl` (major 195). CUDA additionally needs `/dev/nvidia-uvm`. After a
   manual driver bring-up those nodes can be missing or have the wrong major. Recreate
   them against `/proc/devices`:

   ```bash
   sudo modprobe nvidia_uvm
   UVM_MAJOR=$(grep nvidia-uvm /proc/devices | awk '{print $1}')
   sudo rm -f /dev/nvidia-uvm /dev/nvidia-uvm-tools
   sudo mknod -m 666 /dev/nvidia-uvm        c $UVM_MAJOR 0
   sudo mknod -m 666 /dev/nvidia-uvm-tools  c $UVM_MAJOR 1
   ```

1. **`nvidia-modprobe` is not installed.** This setuid helper is what the CUDA runtime
   shells out to in order to create/initialize device nodes for non-root processes.
   Without it, raw `cuInit` may pass but **torch's runtime init throws 999**. This was
   the actual fix for us:

   ```bash
   sudo apt-get install -y nvidia-modprobe
   sudo nvidia-modprobe -c 0 -u
   ```

   After this: `torch.cuda.is_available() → True`. A reboot also installs the proper
   udev rules and avoids the manual `mknod` dance — but if you can't reboot, the two
   steps above get you there.

> **Lesson:** `nvidia-smi` working ≠ CUDA working. They use different device nodes.
> If `cuInit` returns 999, look at `/dev/nvidia-uvm` and make sure `nvidia-modprobe`
> exists.

______________________________________________________________________

## Step 3: The virtual environment

Nothing exotic here, but keep it isolated from system Python:

```bash
python3 -m venv ~/go/venv
source ~/go/venv/bin/activate
pip install --upgrade pip
```

We used Python **3.14**. Check the repo supports it:

```bash
grep requires-python pyproject.toml
# requires-python = ">=3.10,<3.15"   ✅ 3.14 allowed
```

It built fine — `torch==2.11.0` and every dependency had `cp314` wheels. But see
Pitfall 6: a *bundled submodule* had its own narrower Python check.

______________________________________________________________________

## Step 4: CUDA torch + a *consistent* CUDA toolkit

vLLM compiles `.cu` kernels, so it needs `nvcc` — which PyTorch wheels do **not**
bundle (they ship runtime libraries only). You have two options:

- Install the full CUDA toolkit to `/usr/local/cuda` via NVIDIA's apt repo, or
- Assemble a toolkit entirely from pip wheels.

We went pip-only (no apt repo for Ubuntu 26.04 yet, and it keeps everything in the
venv). First, the CUDA build of torch:

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
python -c "import torch; print(torch.version.cuda)"   # → 13.0  (wheel tag: 2.11.0+cu130)
```

Then nvcc and the dev components via the modern unified meta package:

```bash
pip install "cuda-toolkit[nvcc]==13.3.0"
```

### Pitfall 4: the `nvidia-cuda-nvcc-cu13` package is a stub

The old naming is a trap:

```text
$ pip install nvidia-cuda-nvcc-cu13
ERROR: ... (from versions: 0.0.0a0, 0.0.1)   # placeholder only!
```

The real compiler ships via the **`cuda-toolkit[nvcc]`** extra (which pulls
`nvidia-cuda-nvcc`, `nvidia-nvvm`, `nvidia-cuda-crt`). Use the meta package's extras,
not the `*-cu13` standalone names.

### Pitfall 5: CUDA toolkit version skew (three separate failures)

This was the single biggest time sink. The pip CUDA ecosystem is split across many
packages (`nvidia-cuda-nvcc`, `nvidia-nvvm`, `nvidia-cuda-crt`, `nvidia-cuda-cccl`,
`nvidia-cuda-runtime`, `nvidia-cublas`, …) and pip will happily install **mismatched
minor versions**. Each mismatch fails differently:

**5a. ptxas can't assemble newer PTX:**

```text
ptxas fatal : Unsupported .version 9.3; current version is '9.0'
```

nvcc front-end was 13.3 (emits PTX 9.3) but `ptxas` was 13.0 (≤ PTX 9.0). → align them.

**5b. CMake refuses on nvcc-vs-headers mismatch** (PyTorch's `cuda.cmake`):

```text
CMake Error: FindCUDA says CUDA version is 13.3 (from nvcc), but
the CUDA headers say the version is 13.0.
```

**5c. flashinfer's bundled cccl refuses at *runtime*** (its JIT compiler):

```text
cccl/.../cuda_toolkit.h:41: error: "CUDA compiler and CUDA
toolkit headers are incompatible, please check your include paths"
```

The cccl check requires `CUDART_VERSION`'s minor to **exactly equal** nvcc's minor.

**The fix for all three:** pin the *entire* CUDA userspace to one minor version.

> **Why 13.3 and not 13.0 (to match torch's `cu130`)?** Because **CUDA 13.0 headers
> don't compile on glibc 2.43** (Ubuntu 26.04):
>
> ```text
> /usr/include/.../mathcalls.h:206: error: exception specification is incompatible
> with that of previous function "rsqrt"
> ```
>
> CUDA 13.1+ headers fixed this. So we align *up* to 13.3. torch built for `cu130`
> still runs on a 13.3 runtime thanks to **CUDA 13 minor-version compatibility** (any
> 13.x toolkit runs on an R580+ driver).

```bash
pip install "cuda-toolkit==13.3.0" "nvidia-cuda-runtime==13.3.29" \
            "nvidia-cuda-nvcc==13.3.33" "nvidia-nvvm==13.3.33" \
            "nvidia-cuda-crt==13.3.33"  "nvidia-cuda-cccl==13.3.3.3.1" \
            "nvidia-cuda-nvrtc==13.3.33" "nvidia-cublas==13.3.0.5"

# verify nvcc and headers agree:
nvcc --version | grep release                               # 13.3
grep CUDART_VERSION $CUDA_HOME/include/cuda_runtime_api.h   # 13030  (= 13.3)
```

`pip` prints a dependency-conflict *warning* (torch pins `cuda-toolkit==13.0.2`) — it's
cosmetic; torch runs fine via minor-version compat. **But beware:** reinstalling vLLM
later re-pulls its `requirements/cuda.txt` and **silently downgrades the runtime back
to 13.0**, breaking flashinfer's JIT again. Re-run the 13.3 pins after any reinstall.

______________________________________________________________________

## Step 5: Assemble a working `CUDA_HOME`

The pip wheels lay CUDA out under `.../site-packages/nvidia/cu13/{bin,include,lib}`,
which is *almost* what CMake and downstream linkers expect — but missing three things:

```bash
export CUDA_HOME=$VIRTUAL_ENV/lib/python3.14/site-packages/nvidia/cu13

# (a) unversioned dev symlinks: wheels ship libcudart.so.13, linkers want libcudart.so
( cd $CUDA_HOME/lib && for f in lib*.so.*; do ln -sf "$f" "${f%%.so.*}.so"; done )

# (b) lib64 alias: some tools (flashinfer JIT) hardcode $CUDA_HOME/lib64
ln -sfn $CUDA_HOME/lib $CUDA_HOME/lib64

# (c) a libcuda stub for driver-API linking (pip ships no stubs/)
mkdir -p $CUDA_HOME/lib/stubs
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so $CUDA_HOME/lib/stubs/libcuda.so
```

Sanity check before the big build:

```bash
cat > /tmp/t.cu <<'EOF'
#include <cuda_runtime.h>
__global__ void k(){}
int main(){k<<<1,1>>>();return cudaDeviceSynchronize();}
EOF
$CUDA_HOME/bin/nvcc -arch=sm_86 -I$CUDA_HOME/include -L$CUDA_HOME/lib -lcudart /tmp/t.cu -o /tmp/t.out
# Also confirm CMake finds it:
cmake -P <(echo 'find_package(CUDAToolkit REQUIRED); message("CTK ${CUDAToolkit_VERSION}")') 2>&1
```

______________________________________________________________________

## Step 6: Build vLLM

Set the build environment and go. The most important variable is
**`TORCH_CUDA_ARCH_LIST`** — scope it to *your* GPU or you'll compile every
architecture and wait 5–10× longer.

```bash
cd ~/go/vllm
export PATH=$CUDA_HOME/bin:$PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST="8.6+PTX"     # A10G = sm_86
export MAX_JOBS=12          # ~2-3 GB RAM per job; tune to your box
export NVCC_THREADS=2
export CMAKE_ARGS="-DCUDAToolkit_ROOT=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"
pip install -v -e . --no-build-isolation
```

A few notes:

- `--no-build-isolation` is required so the build sees the torch/CUDA you installed.
- `enforce_eager`-style arch warnings like `DeepGEMM/FlashMLA will not compile: unsupported CUDA architecture 8.6` are **expected** on Ampere — those kernels target
  Hopper (sm_90+) and are simply skipped.
- On 16 cores / 62 GB this took ~30–40 min and produced `_C.abi3.so` (~117 MB),
  `_moe_C.abi3.so`, etc.

### Pitfall 6: a bundled submodule rejects your Python

Even though the top-level `pyproject.toml` allowed Python 3.14, the vendored
flash-attention CMake had its own allow-list:

```text
CMake Error at .deps/vllm-flash-attn-src/cmake/utils.cmake:20:
  Python version (3.14) is not one of the supported versions: 3.9;3.10;3.11;3.12;3.13.
```

Fix — add your version to the macro (vLLM points `FETCHCONTENT_BASE_DIR` at `.deps`, so
edits there persist; just don't `rm -rf .deps` before rebuilding):

```cmake
# .deps/vllm-flash-attn-src/cmake/utils.cmake
set(_SUPPORTED_VERSIONS_LIST ${SUPPORTED_VERSIONS} ${ARGN} "3.14")
```

> **This patch is not permanent.** flash-attn is pulled via CMake FetchContent at a
> pinned `GIT_TAG`. The moment you `git pull`/update vLLM and that tag changes (or you
> `rm -rf .deps`), FetchContent re-clones a *fresh* copy and your edit is gone — the
> 3.14 check fails again at the next configure. Re-apply the one-liner after any update
> that bumps the flash-attn tag.

### Pitfall 7: dependency-resolver deadlock (`ResolutionImpossible`)

On a recent `main`, `pip install -e .` can die *before compiling anything* with:

```text
ERROR: Cannot install cuda-tile[tileiras]==1.4.0, cuda-toolkit==13.0.2 and vllm
because these package versions have conflicting dependencies.
  torch 2.11.0 depends on cuda-toolkit==13.0.2
  cuda-tile[tileiras] 1.4.0 depends on cuda-toolkit>=13.2,<13.4
ERROR: ResolutionImpossible
```

Two of vLLM's own dependencies pin **incompatible** CUDA-toolkit ranges (torch wants
exactly 13.0.2; a newer kernel package wants ≥13.2). pip's strict resolver refuses to
proceed. This is an upstream packaging conflict, not something you caused — and it's
exactly why we aligned the toolkit to **13.3** earlier (it satisfies the ≥13.2 side,
and torch runs fine against it via minor-version compat).

The fix is to build the package **without re-resolving the whole graph**, since you've
already curated a working CUDA stack:

```bash
pip install -v -e . --no-build-isolation --no-deps
```

`--no-deps` compiles and installs vLLM using the environment you've assembled, instead
of letting pip try (and fail) to reconcile every transitive pin. Afterwards, install
any genuinely-missing runtime deps individually and re-run the smoke test. (Upstream's
own docs use `uv`, whose override/resolution model sidesteps this; with plain pip,
`--no-deps` is the escape hatch.)

### Pitfall 8: `MAX_JOBS` and parallelism

`MAX_JOBS` controls ninja's parallel compile jobs. CUDA compiles use ~2–3 GB each, so
`MAX_JOBS × 3 GB` should fit in RAM. On 62 GB you can run 16; we used 12 as a safe
default. You'll notice ninja drops to fewer jobs near the end (`[267/340]`) — that's
dependency ordering on the final heavy template units and the `.so` link, not a
misconfiguration. `NVCC_THREADS` parallelizes within a single nvcc invocation.

______________________________________________________________________

## Step 7: Verify — and the runtime-only pitfalls

A successful build does **not** mean inference works. vLLM's runtime JIT-compiles more
kernels on first use, which surfaces a fresh set of issues.

```python
from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m", enforce_eager=True,
          gpu_memory_utilization=0.5, max_model_len=512)
print(llm.generate(["The capital of France is"],
                   SamplingParams(temperature=0, max_tokens=20))[0].outputs[0].text)
```

### Pitfall 9: `Could not find nvcc and default cuda_home='/usr/local/cuda'`

**flashinfer JIT-compiles sampling kernels at runtime** and needs `nvcc` — but at
runtime nobody set `CUDA_HOME`, so it falls back to the nonexistent `/usr/local/cuda`.
Because our toolkit lives in the venv, export it (and bake it into `activate` so it's
always present):

```bash
cat >> $VIRTUAL_ENV/bin/activate <<'EOF'
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.14/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
EOF
```

This is also where Pitfalls 5c (cccl version check) and the `lib64` symlink
(`cannot find -lcudart`) bite — they're runtime-JIT failures, not build failures, so
they only appear here. With the 13.3 alignment + the `lib64` symlink in place, the JIT
compile succeeds and you get:

```text
PROMPT: 'The capital of France is'
OUTPUT: ' the capital of the French Republic...'
```

🎉

______________________________________________________________________

## Step 8: Run the GPU test suite

A `generate()` proves the happy path; the kernel tests prove the build broadly. The
suite that most directly exercises what you just compiled is `tests/kernels/`. Run it
with `CUDA_HOME` on `PATH` (the tests JIT-compile too):

```bash
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.14/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
python -m pytest tests/kernels/core tests/kernels/attention -q
```

On an A10G a focused subset (activation, layernorm, rotary/positional encoding, paged
attention, cache) runs in ~1 hr and lands at **2402 passed, 583 skipped, 36 failed**.
The 583 skips are arch-gated kernels (Hopper/Blackwell sm_90+) correctly opting out.
The 36 failures are **all the same issue** — see Pitfall 10.

### Pitfall 10: FP8 KV-cache tests *fail* (not skip) on SM < 89

Every one of those 36 failures is `test_reshape_and_cache_flash[...fp8...]` with:

```text
FP8 KV cache needs native fp8e4nv (SM89+). Use --kv-cache-dtype bfloat16 ...
```

The A10G is **sm_86**; native FP8 (`fp8e4nv`) needs **sm_89+** (Ada/Hopper). This is a
hardware limit, not a broken build — but unlike the cleanly arch-gated kernels, this
Triton path `assert`s on unsupported hardware instead of `skip`ping, so it counts as a
failure. Deselect the FP8 cases to get a fully green run:

```bash
python -m pytest tests/kernels/attention/test_cache.py -k "not fp8" -q
# 335 passed, 403 skipped, 477 deselected, 0 failed
```

Takeaway: on pre-Ada GPUs, treat FP8 KV-cache test failures as expected, and gate them
out with `-k "not fp8"` rather than chasing them.

______________________________________________________________________

## Appendix: every error → one-line fix

| Error | Root cause | Fix |
|---|---|---|
| `nvidia-smi: command not found` (assumed no GPU) | driver not installed; hardware was there | `lspci \| grep nvidia` to detect hardware |
| `modprobe nvidia: No such device` | nouveau owns the GPU | blacklist + unbind + `rmmod nouveau` |
| `CUDA unknown error` / `cuInit → 999` | missing/stale UVM nodes; no `nvidia-modprobe` | `apt install nvidia-modprobe`; recreate `/dev/nvidia-uvm` |
| `nvidia-cuda-nvcc-cu13` has no real version | wrong package name | use `cuda-toolkit[nvcc]` |
| `ptxas Unsupported .version 9.3` | nvcc/ptxas minor mismatch | pin all CUDA pkgs to one minor |
| CMake: `nvcc says 13.3 but headers say 13.0` | runtime headers ≠ nvcc | align headers to nvcc version |
| `mathcalls.h: rsqrt ... incompatible` | CUDA 13.0 headers vs glibc 2.43 | use CUDA ≥ 13.1 headers |
| flash-attn CMake: Python 3.14 not supported | submodule allow-list | patch `utils.cmake` (re-apply after any update that bumps its tag) |
| `ResolutionImpossible` (cuda-toolkit 13.0.2 vs ≥13.2) | conflicting CUDA pins across vLLM deps | build with `pip install -e . --no-deps` |
| `cccl: compiler and toolkit headers incompatible` | runtime downgraded after vLLM reinstall | re-pin CUDA runtime to nvcc's minor |
| `cannot find -lcudart` (JIT link) | wheels use `lib/`, tool wants `lib64/` | `ln -sfn $CUDA_HOME/lib $CUDA_HOME/lib64` |
| `Could not find nvcc ... /usr/local/cuda` | `CUDA_HOME` unset at runtime | export `CUDA_HOME` (bake into `activate`) |
| `FP8 KV cache needs native fp8e4nv (SM89+)` (test fails) | A10G is sm_86; FP8 path asserts instead of skipping | not a build bug — deselect with `-k "not fp8"` |

## Updating an existing checkout

Pulling a newer vLLM isn't just `git pull` — an editable source build has moving parts
that a pull invalidates. The sequence that works:

```bash
git fetch upstream && git reset --hard upstream/main   # or your target commit
rm -rf build .deps && find vllm -name '*.abi3.so' -delete   # force a clean rebuild
# re-apply the flash-attn 3.14 patch (the tag changed → .deps was re-fetched)
pip install -v -e . --no-build-isolation --no-deps     # --no-deps dodges resolver conflicts
# re-pin the CUDA toolkit to 13.3 if anything got downgraded, then re-run the smoke test
```

Before pulling, check the gap with `git diff --name-only HEAD..upstream/main | grep -E '\.cu|CMakeLists|requirements/'` — if native/build files changed (they usually have),
budget for a full recompile (~30–40 min) and re-verification. Also confirm the
`torch==` pin and `requires-python` in `pyproject.toml` didn't move; if torch's version
changed, you're re-doing the whole CUDA/toolkit alignment, not just a rebuild.

## Key takeaways

1. **Detect hardware with `lspci`, not `nvidia-smi`.** Don't build for CPU because a
   tool is missing.
1. **`nvidia-smi` working ≠ CUDA working.** UVM nodes + `nvidia-modprobe` matter.
1. **Pin the entire CUDA pip toolkit to one minor version.** Skew fails three
   different ways at three different stages.
1. **Pick the CUDA minor that's compatible with your glibc/compiler**, then rely on
   CUDA minor-version compatibility for the driver/torch.
1. **A green build isn't done** — runtime JIT (flashinfer) needs `CUDA_HOME` and a
   couple of symlinks. Verify with a real `generate()`.
1. **Scope `TORCH_CUDA_ARCH_LIST` to your GPU** to keep build times sane.
1. **Some test failures are hardware limits, not build bugs.** On pre-Ada GPUs the FP8
   KV-cache tests `assert` instead of `skip` — deselect them with `-k "not fp8"`.

## References

[^1]: vLLM Project. *vLLM Repository*. ([Link](https://github.com/vllm-project/vllm))

*Disclaimer: This article was generated using the Gemini 3.1 Pro model.*
