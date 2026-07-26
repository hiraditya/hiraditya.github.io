---
title: "XLA Up Close: What It Optimizes, and What It Won't"
date: 2026-07-25 12:00:00 -0700
categories: [Compilers, ML-Systems]
tags: [jax, xla, hlo, stablehlo, compilers, tpu]
---

Here is a small function and the entire program XLA compiled it into:

```python
def simp(x):
    y = (x + 0.0) * 1.0
    return jnp.transpose(jnp.transpose(y))
```

```text
ENTRY %main (x: f32[3,3]) -> f32[3,3] {
  %x = f32[3,3]{1,0} parameter(0)
  ROOT %copy = f32[3,3]{1,0} copy(%x)
}
```

The add-zero, the multiply-one, and both transposes are gone. XLA proved the whole thing is the identity and emitted a single copy. This is not a toy result; it is the same algebraic-simplification machinery that lets XLA take a page of dense linear algebra and hand back a fused kernel that runs at the memory-bandwidth limit of a TPU. XLA is a genuine optimizing compiler, and a very good one.[^1]

It is also rigid in ways that will bite you, and the interesting thing — the thing worth a long post — is that its strengths and its limitations are the *same design decision*. XLA freezes your shapes, sees the whole graph at once, allocates every buffer statically, and fuses against a global cost model you do not control. That is exactly why it is fast, and exactly why it is inflexible. This is a review and a critique of that bargain. I will show you what XLA actually does, using real HLO dumps, sourcing each claim to the OpenXLA source and docs, and then I will tell you where the bargain stops paying off.

If you have not met the layer above XLA: it is the compiler behind JAX, behind TensorFlow, and behind PyTorch through PyTorch/XLA.[^1] JAX traces your Python into a small typed IR and hands it to XLA; what XLA does with it from there is the subject here. The tracing half — how Python becomes that IR — is a topic of its own, and I leave it aside.

## Where XLA sits, and what it eats

XLA is an ahead-of-time compiler for array programs. "Ahead of time" is relative to *execution*, not to your process: it compiles the first time a given shape signature appears, caches the result, and reuses it.[^7] Its input is a graph of high-level tensor operations, and the modern portable front door to that graph is **StableHLO**, an MLIR dialect that JAX emits and that XLA ingests.[^3] The full path:

```
jaxpr ──▶ StableHLO ──▶ HLO ──▶ [ optimization passes ] ──▶ optimized HLO ──▶ backend
                                                                              ├─ LLVM  (CPU)
                                                                              ├─ LLVM/PTX + cuBLAS/cuDNN/Triton (GPU)
                                                                              └─ XLA:TPU (closed backend)
```

The naming here is a genuine mess and worth untangling once. There is the classic **HLO** — a hand-written C++ IR with its own protobuf and text format that predates MLIR and is explicitly *not* based on it.[^4] There is the MLIR world layered on later: `MHLO`, and then **StableHLO**, which is built on MHLO and adds serialization plus backward/forward compatibility guarantees, so that it can serve as a stable portability contract between frameworks and compilers.[^3] JAX lowers to StableHLO; XLA converts StableHLO into its internal HLO and optimizes that.[^2][^7] Even the acronym is unsettled — XLA's own architecture page glosses HLO as "high level operations" while its terminology page says "High Level Optimizer."[^2][^4] I will use "HLO" for the thing XLA optimizes and "StableHLO" for what it is handed, because that is how the dumps read.

Here is what a `jit`-ed `relu(x @ W + b)` looks like when JAX hands it over, straight from `jax.jit(f).lower(...).as_text()`:

```text
func.func public @main(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>,
                       %arg2: tensor<16xf32>) -> tensor<4x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
         (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
  %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<4x16xf32>
  %3 = stablehlo.add %0, %2 : tensor<4x16xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x16xf32>
  %5 = stablehlo.maximum %3, %4 : tensor<4x16xf32>
  return %5 : tensor<4x16xf32>
}
```

Every value has SSA-style single-assignment semantics (it is MLIR underneath), every tensor is typed, and broadcasting is explicit — the `+ b` is a real `broadcast_in_dim` followed by an `add`, because XLA models it as an operation with a cost, not as sugar. The op vocabulary is small: XLA's opcode enum holds roughly 150 opcodes and is statically asserted to stay under 256.[^6] It covers contraction (`dot_general`), `convolution`, `reduce`, `dynamic-slice`, the collectives (`all-reduce`, `all-gather`), and the escape hatch `custom-call`.[^5] That small, typed, functional vocabulary is what keeps the optimizer tractable. One caveat to hold: classic HLO shapes are static, but StableHLO can express *bounded* dynamism (`tensor<?x2xf32>`), a distinction that matters a great deal in the critique.[^3]

## The optimizer, pass by pass

Dump the pipeline (`XLA_FLAGS=--xla_dump_to=...`) and XLA writes out every intermediate module — on a trivial function I counted 39 stages. The architecture doc frames it as three phases: target-independent optimization, target-dependent optimization, and target-specific codegen.[^2] Cleaned of noise, the spine is:

```
simplification (algsimp, CSE, DCE) ─▶ layout assignment ─▶ fusion
  ─▶ copy insertion ─▶ scheduling ─▶ backend codegen (LLVM IR ─▶ object / PTX)
```

Each stage is where one of XLA's strengths — and one of its sharp edges — lives.

### Simplification

The front of the pipeline is ordinary optimizing-compiler work: algebraic simplification,[^10] common-subexpression elimination, and dead-code elimination.[^10] The `simp` example at the top is this stage doing its job — `(x+0)*1` folds away, `transpose(transpose(x))` cancels, and the whole computation collapses to a copy. This is real and it is good. But notice the ceiling. These are *local, syntactic* rewrites over the graph you handed it. XLA will cancel a double transpose; it will not discover that two differently-written subgraphs compute the same mathematical function, and it will not reassociate a reduction to expose parallelism it was not given. The algebra it knows is a fixed rule set in `algebraic_simplifier.cc`, not a solver.

### Layout assignment

This is the first thing that surprises people, because it is invisible in your source. Every array in optimized HLO carries a **layout**: the `{1,0}` in `f32[4,16]{1,0}` is the minor-to-major dimension order — here, row-major; `{0,1}` would be column-major.[^8] XLA chooses these physical layouts to make consuming operations efficient, and on TPU the story is richer still, with tiled layouts because the hardware wants data in tiles, not flat rows.[^8]

The catch is that layout is a constraint-propagation problem, and when constraints collide, XLA inserts copies. The `layout_assignment` pass carries explicit machinery for exactly this — `SetOperandLayout`, `CreateCopyWithNewLayout`, `CopyOperandIfLayoutsDiffer`, and constraint propagation for custom calls.[^9] Watch it happen when a computation feeds an operation that demands a particular layout — an eigendecomposition, which lowers to a LAPACK routine that insists on column-major input:

```text
%eigh = (f32[16,16]{0,1}, f32[16]{0}, s32[]) custom-call(%multiply_copy_fusion),
        custom_call_target="lapack_ssyevd_ffi",
        operand_layout_constraints={f32[16,16]{0,1}}
```

The data arrives row-major (`{1,0}`) and the routine wants column-major (`{0,1}`), so XLA quietly materializes a transposing copy — the `multiply_copy_fusion` feeding the call. That copy never appears in your Python. It is a real cost, it is layout-driven, and the only way to see it is to read the HLO. A "mysterious" transpose in a profile is very often layout assignment reconciling a constraint you could not see.

### Fusion

Fusion is where XLA earns its reputation, and where the contrast with a framework compiler like PyTorch's TorchInductor is sharpest. The goal is the same one I covered in the [torch.compile post]({% post_url 2026-07-22-how-torch-compile-works %}): keep intermediate values in registers instead of streaming them to and from memory. But XLA does it at the granularity of HLO ops, folds the result into explicit `fusion` nodes whose bodies are their own computations, and — critically — distinguishes fusion *kinds*. The `FusionKind` enum has exactly four: `kLoop`, `kInput`, `kOutput`, `kCustom`.[^11]

Take `tanh(exp(x) * 2 + 1)`. XLA fuses the entire chain into one loop kernel:

```text
%fused_computation (param_0: f32[1024]) -> f32[1024] {
  %exp.0  = f32[1024] exponential(%param_0)
  %mul.0  = f32[1024] multiply(%exp.0, broadcast(constant(2)))
  %add.0  = f32[1024] add(%mul.0, broadcast(constant(1)))
  ROOT %tanh.0 = f32[1024] tanh(%add.0)
}
ENTRY %main (x: f32[1024]) -> f32[1024] {
  ROOT %fusion = f32[1024] fusion(%x), kind=kLoop, calls=%fused_computation
}
```

Four elementwise ops, one `kind=kLoop` fusion, one kernel, one pass over memory. The base pass fuses "vertically," producers into consumers.[^12] And `relu(x @ W + b)` splits by kind: the matmul becomes a `kind=kCustom` fusion routed to a tuned kernel, while the bias-add and ReLU merge into a `kind=kLoop` fusion on top. On GPU the main fusion pass is `priority_fusion`, which is explicit about being cost-model-driven: it "assigns a priority to each producer instruction based on the estimated performance benefit of fusing it into its consumers," where `priority = time_unfused - time_fused`.[^12]

That cost model is the strength and the problem. It is global and automatic, which is why XLA fuses well across a whole graph without you lifting a finger. It is also internal: the decision comes from the model in `priority_fusion`, not from anything you can annotate. When the model under-fuses, or when the pattern you want cannot be expressed at HLO-op granularity — flash attention is the canonical example — you are out of levers inside XLA. That is precisely why Pallas exists.

### Buffer assignment

Here is a defining property and a genuine strength: XLA does **whole-program static memory allocation**. Because it sees the entire graph and every shape is known, the `buffer_assignment` pass computes buffer liveness and assigns every value a concrete offset in a preplanned allocation before the program runs; a single allocation may hold multiple values "with disjoint liveness" — buffer reuse, computed statically.[^13] The dump is explicit:

```text
BufferAssignment:
allocation 0: size 1024, parameter 0, shape |f32[16,16]|
allocation 1: size 64, output, maybe-live-out:
   value: eigh{1}                 (size=64, offset=0)
   value: broadcast_select_fusion (size=64, offset=0)   <- same buffer, reused
allocation 2: size 1092, preallocated-temp:
   value: multiply_copy_fusion    (size=1024, offset=64)
   value: eigh{}                  (size=24,   offset=0)
```

`allocation 1` is shared by two values whose lifetimes do not overlap; `allocation 2` is a single temp arena holding several temporaries at fixed offsets. No `malloc` on the execution path, peak memory known at compile time. This pairs with **donation**, where an input buffer is aliased to an output so an update happens in place — configured through `HloInputOutputAliasConfig` and essential for keeping optimizer state and KV caches from doubling your memory.[^13] For static-shaped programs this is close to optimal. It also depends *entirely* on static shapes; the moment a buffer's size is unknown at compile time, the guarantee weakens. Hold that thought.

### Copy insertion and scheduling

Two shorter passes close out the target-independent work. `copy_insertion` is a legalization pass that "inserts copies (kCopy instructions) to eliminate several kinds of problems" — chiefly values that are simultaneously live while sharing a buffer, and enforcing the aliasing that donation promised.[^14] Then the scheduler orders instructions; XLA's `latency_hiding_scheduler` is "oriented to hiding latencies of operations that can run in parallel," and it carries explicit knobs for overlapping collectives with compute (`all_reduce_overlap_limit`, `all_gather_overlap_limit`, and friends) — a detail that matters in the sharding post.[^15]

### Codegen and the backends

After scheduling, XLA hands off to a backend. On CPU and GPU it lowers to LLVM IR — the dump even drops the `.ll` and `.o` files next to the HLO — and then to native code or, on GPU, PTX via the LLVM NVPTX backend.[^2][^16] GPU additionally routes dense matmuls to cuBLAS and convolutions to cuDNN as custom calls (`__cublas$gemm`, `__cudnn$convForward`), rewritten in by dedicated passes.[^16] And for "more advanced fusions which include matrix multiplication or softmax, XLA:GPU uses Triton as a code-generation layer," through a real Triton emitter in the source tree — a direct through-line to the [Triton compiler deep-dive]({% post_url 2026-07-21-triton-compiler-deep-dive %}).[^17] TPU goes to the closed-source XLA:TPU backend; a maintainer states plainly that "the TPU backend is non-OSS, so you can't build it from the GH repo."[^19] The whole StableHLO/HLO layering is one more instance of the MLIR dialect-stack pattern I wrote about in [the MLIR tour]({% post_url 2026-06-23-mlir-dialect-stack-for-ml %}).

This is not hand-waving; you can watch it on real hardware. Lowering a `softmax` on an A100-SXM4-80GB, XLA:GPU emits precisely a Triton fusion:

```text
ROOT %triton_softmax.2 = f32[2048,2048]{1,0} fusion(%x.1), kind=kCustom,
  calls=%triton_softmax_computation.2,
  backend_config={"fusion_backend_config":{"kind":"__triton",
    "block_level_fusion_config":{"num_warps":"4","output_tiles":[{"sizes":["1","2048"]}]}}}
```

The `"kind":"__triton"` is XLA choosing Triton, and the `block_level_fusion_config` is XLA choosing the launch parameters for you — four warps, an output tile of `1x2048`.[^17] A matmul, by contrast, lowers by default to a `kCustom` `gemm_fusion` (also Triton); disable that path with `--xla_gpu_enable_triton_gemm=false` and the same matmul falls back to a cuBLAS custom-call, `custom_call_target="__cublas$lt$matmul"`.[^16] Which of the two you get is decided by autotuning, not by you.

### Autotuning

One backend behavior deserves its own note because it dominates the compile-time story later. On GPU, XLA **autotunes**: for a given fusion it generates candidate implementations (Triton configs, cuBLASLt, cuDNN), profiles them on the real device, and keeps the fastest — the autotuner core profiles all candidates and selects the best.[^18] It is effective and it is expensive; OpenXLA's own persisted-autotuning doc admits the tuning "can take a long time if there are many fusions," which is the entire reason a cache exists.[^18] You can feel it directly: on the A100, compiling a single 4096x4096 matmul took about **2 seconds**, and an 8192x8192 one about **2.8 seconds** — for one fused operation, almost all of it the autotuning search. The open-source autotuner is GPU-only; the equivalent for TPU lives in the closed backend.[^19]

## The critique

Everything above is XLA at its best: a whole-program optimizer that fuses aggressively, allocates statically, and produces genuinely fast code. Now the other side. None of what follows says XLA is bad — it says the design has a shape, and the shape has edges you will hit.

### Static shapes are not a preference, they are the whole deal

Every strength in the last section — static allocation, global fusion, layout planning, in-place donation — is downstream of one assumption: the shapes are known at compile time. When they are, XLA is excellent. When they are not, you are working against the grain of the entire system. StableHLO can represent *bounded* dynamism, but OpenXLA is candid that there is "limited framework support," and specifically that "JAX does not currently trace operations which lead to data dependent dynamism."[^21]

And modern ML is full of shapes that are not static. LLM inference has a sequence length that grows token by token and a KV cache that grows with it; batched serving has ragged batches. These are the mainstream, not the exotic. XLA's answer is bounded dynamic shapes plus padding and bucketing into a few static shapes compiled separately — JAX's own issue tracker states that "JAX currently supports dynamic shape input through padding or bucket padding," and the JAX scaling book notes that "prefills are padded to the longest sequence and we waste a lot of compute."[^21][^23] Truly data-dependent shapes it refuses outright. Ask for the indices of the nonzero elements of a vector inside `jit`:

```python
jit(lambda x: jnp.nonzero(x))(jnp.array([0., 1., 0., 2.]))
# ConcretizationTypeError: Abstract tracer value encountered where a
# concrete value is expected: traced array with shape int32[]
```

JAX's own `nonzero` docstring is explicit: "Because the size of the output of `nonzero` is data-dependent, the function is not compatible with JIT," and it must be given a static `size=` to be usable under transformation.[^22] This is the correct behavior given the design, and it is also a wall — the same tension I wrote about in [the VLIW post]({% post_url 2026-06-16-why-vliw-architecture-is-popular-again %}): systems that win by assuming static, predictable structure pay for it precisely when the workload turns dynamic, and the workloads are turning dynamic.

### The compile-time tax is real, but be honest about where it comes from

You will read that XLA "takes minutes to compile." Sometimes it does, and it is worth being precise, because the naive version of the complaint is wrong. A 24-layer, 1024-wide dense tower — 360 HLO instructions after optimization — compiled on CPU in my measurements in about **0.03 seconds**. XLA is not slow at compiling graphs. Move to the GPU, though, and a *single* 4096x4096 matmul took roughly **2 seconds** to compile on an A100 — dozens of times the whole CPU tower, for one operation — because the backend was autotuning it.

The minutes come from three places. First, **autotuning**: as above, XLA benchmarks kernel configs on real hardware at compile time, and that search dominates — which is why the persisted-autotuning cache exists.[^18] Second, **graph size**: a full model with the backward pass and an optimizer step is not 360 instructions but tens of thousands, and several passes are super-linear; real long-compile reports exist independent of autotuning.[^28] Third — the one that actually hurts in deployment — **per-shape specialization**: every new shape signature recompiles, so a serving workload with varying sequence lengths spends real wall-clock recompiling, a long-standing complaint on the JAX tracker.[^21] The mitigations are a persistent on-disk compilation cache and compiling a repeated block instead of the whole model, and JAX documents both.[^20] They also tell you the shape of the problem: XLA wants to compile once, for one shape, and amortize. When your deployment cannot promise that, the model fights you.

### Fusion is automatic, which means you cannot argue with it

The flip side of a global cost model is that it is *the* cost model. XLA's fusion decision comes from the internal priority model, not from anything you can annotate.[^12] For most code the automatic decision is fine to good. But if the heuristics decide not to fuse two operations you know should fuse, or to lay out a tensor in a way that forces a copy, your recourse is to restructure the source and hope the model reads it differently, or to leave XLA entirely via `custom_call`. There is no user-facing "fuse these, not those." For the last slice of performance on a kernel that matters, the absence of a manual override is a real ceiling — and it is why performance-critical teams reach for Pallas.

### `custom_call` is an escape hatch and an optimization barrier

When XLA cannot express or generate what you need, you drop to `custom_call` — a node pointing at an external function. The StableHLO spec describes it exactly: it is an "escape hatch" whose "semantics ... are implementation-defined and are not analyzed by the compiler."[^24] That is the whole problem in one sentence. Look again at the `eigh` example: XLA knows nothing about what happens inside `lapack_ssyevd_ffi`; it cannot fuse across it, cannot reorder around it, and must satisfy its layout constraints with copies. JAX's FFI docs make the runtime consequence concrete — an FFI call is "an opaque black box: JAX can't look inside it," and a sharded input may force XLA to "gather the whole array onto every device with an all-gather" and run the call redundantly.[^24] This is the same property I flagged for NCCL collectives in the torch.compile post: an opaque region that fragments optimization on either side. The more of your hot path lives in custom calls, the less of it XLA is actually optimizing.

### Opacity and the closed backend

XLA is a black box in daily use. When performance is wrong, there is no gentle story — you dump HLO, learn to read fusion kinds and layout annotations, and reverse-engineer a cost model that is not documented as a contract. The tooling (`--xla_dump_to`, the HLO-as-HTML renderer, the pass dumps) is genuinely good once you know it exists, but the learning curve is steep and self-taught. And the highest-value backend, XLA:TPU, is closed: a maintainer notes that "the TPU dependent optimizations are mostly not open source, essentially just those parts that are shared with GPU and CPU."[^19] For a system this central to modern ML, the debuggability floor is higher than it should be.

### Portability of the IR is not portability of performance

StableHLO's pitch is a stable, portable IR — write once, run on any backend that speaks it — and as a *portability* contract it delivers.[^3] But portable code is not portable performance. Fast on TPU means the right tiled layouts and XLA:TPU's fusion choices; fast on GPU means cuBLAS, Triton, and GPU autotuning; fast on CPU means something else again.[^8][^16][^17] The same StableHLO lowered to two backends gives you two *correct* programs, one or both of which may need backend-specific coaxing to be *fast*. StableHLO resolves the portability half and leaves the performance half where it was — the trilemma every heterogeneous system meets.

### Placement and topology live outside the IR

The last one is the one this series has been circling, and the pipeline names it without meaning to: among the passes are a `pre-spmd-partitioner`, a `sharding-removal`, and an `async-collective-replacer` — a whole sub-machine for turning sharding annotations into partitioned, communicating programs. That machine is a *separate* one, bolted on the side, and the source makes the distinction sharp. An array's type is its dtype and its shape. Its layout is physical and first-class. But *where the array lives* — which device, which slice of a mesh — is not part of the type: sharding is a nullable attribute hung off the instruction (`has_sharding()`, `set_sharding()`), backed by the `OpSharding` proto, not a field of `Shape`.[^25] A separate partitioner — GSPMD, and now the MLIR-based **Shardy**, which "incorporates the best of GSPMD and PartIR" — consumes those annotations and adds "the necessary data movement/formatting and collective operations."[^26][^27] You can literally watch the annotation get stripped by a pass.

That is the seam. XLA models the *physical layout* of bytes within a device beautifully and models the *placement* of bytes across devices as an annotation that a side-machine consumes. The distinction is not academic: it is why XLA can fuse a matmul perfectly and still have no idea that one of its operands is about to require a cross-device transfer.

## XLA as a software project

Everything above is about the compiler. This section is about the codebase, because XLA is a long-lived lineage — the TensorFlow compiler, extracted into its own repository in August 2022 — and old projects accumulate a second set of properties that have nothing to do with their design and everything to do with how they are maintained. If you are choosing XLA for something you have to live with for years, these matter as much as the fusion model.

Start with the backends. I read the commit history for both the GPU and CPU paths, and both are actively developed. Over the last year on `main`, `xla/backends/gpu` took 2,186 commits and `xla/service/gpu` another 1,679, with `xla/backends/gpu/codegen/triton` alone accounting for 595 — which makes GPU a very active area of the repository. `xla/backends/cpu` took 668 and `xla/service/cpu` 374; smaller in absolute terms, but per source file the churn is comparable to GPU's. The vendor-library integration is current rather than vestigial: `cudnn.cc` and `cublaslt.cc` were both modified four days before I wrote this.[^29]

That is worth pausing on. Google is a TPU shop. It would be entirely rational for it to treat GPU and CPU as compatibility targets — kept correct, kept building, and otherwise left alone. It does not. The GPU path is not a port; it is a first-class backend with its own Triton emitter, its own autotuner, its own cuBLAS and cuDNN integration, and the heaviest commit traffic in the tree. Sustaining that for hardware you do not sell, at that level of investment, is a commendable engineering effort and it deserves saying plainly.

Nor is it a closed effort. Over the last year, 1,304 commits reached `main` as exported external pull requests, among them 355 from NVIDIA, 304 from AMD, and 148 from Intel.[^31] A TPU company taking patches at that volume from three accelerator vendors — all of whom compete with the TPU — says more about a real commitment to targets beyond its own silicon than any architecture document could.

So the interesting question is not how much attention the open backends get. It is *where the work happens*.

### Everything lands internally first, wherever it was reviewed

I checked the last 2,000 commits on `main`. All 2,000 carry a `PiperOrigin-RevId:` trailer.[^30] Not most — all of them, including all 385 in that window authored from something other than a `@google.com` address. Whoever writes a change, it is committed inside Google's monorepo and reaches GitHub as a Copybara export.

Where the *review* happens is a separate question, and there the repository is doing real work. Outside contributions are opened, reviewed, and iterated in public. Across fourteen recent vendor pull requests I sampled, every one carried GitHub reviews — 79 reviews and 143 review-and-discussion comments in total, and one of them went through 13 reviews and 22 commits before landing.[^31] That is not a rubber stamp on a decision already taken elsewhere; it is ordinary open-source code review, done by Google engineers, in the open. Google's own changes are the mirror-image case: written and reviewed internally, they surface on GitHub as already-approved exports opened by `copybara-service[bot]`, typically with no GitHub review activity at all.

So the arrangement is two review paths and one commit path. That is a more interesting structure than "a mirror," and it is not a complaint about corporate stewardship — XLA is a Google compiler and nobody pretended otherwise. But the single commit path has consequences you will feel. For the roughly 88% of changes written at Google, the review that gated them happened somewhere you cannot read, and the tests that decided they were safe ran on hardware you cannot see — if they were TPU tests, against a backend whose source you do not have. When one of those breaks you, what you get is a squashed summary of a discussion that was never public. And because the internal commit is the authoritative one, even a pull request approved on GitHub is imported and re-landed rather than merged; a friction the project has discussed on its own tracker.[^32] Roughly 12% of last year's commits are mechanical exports as well: "Automated Code Change," license-header sweeps, LLVM integrations.

### The public CI tests less than its job names suggest

The `CI` workflow runs on `pull_request` and on pushes to `main`, so it is a genuine presubmit gate for the outside contributions above — an external patch does have to get public CI green. What that gate covers is the question. The matrix is mostly CPU: Linux x86, Windows x86, ARM64, a Bzlmod variant, plus JAX and TensorFlow build jobs to catch downstream breakage. XLA's own GPU test coverage in it is one job on a `linux-x86-g2-16-l4-1gpu` pool — a single L4, an inference-class card. And the jobs named "XLA Linux x86 GPU Hermetic ROCm" and "XLA Linux X86 GPU ONEAPI" are assigned `linux-x86-n2-16`, which are CPU pools: they compile, they cannot execute a GPU test.[^33]

Notice which silicon that is. The only NVIDIA hardware anywhere in the public CI configuration is an L4 and an 8×H100 node — an inference-class card and a 2022 datacenter part. The source tree, meanwhile, carries Blackwell support: `sm_100`, `sm_103`, and `sm_120` code paths, and a `CudaComputeCapability::Blackwell()` predicate that gates real codegen decisions.[^39] Nothing in public CI can execute any of it. So whether XLA is correct and fast on the NVIDIA GPUs people are actually buying in 2026 is either something you take on faith — Google and NVIDIA are certainly testing it somewhere, just nowhere you can see — or something you go measure yourself.

AMD's arrangement is the interesting counterexample, and it cuts against the usual assumption about which vendor is better served here. AMD self-hosts: `rocm_ci.yml` dispatches to the runner label `amd-do-linux-xla-gpu-gfx950-1`, and gfx950 is CDNA4 — the MI350 series, current generation rather than a part two cycles back. It also triggers on `pull_request`, so every pull request gets a verdict from real Instinct hardware. It is not spotless: across 398 completed runs in a four-day window it went 213 success to 89 failure, a 70.5% pass rate, which is its own kind of noisy.[^39] But a 70% presubmit signal that contributors actually see on their PR is a different proposition from a 48% nightly that blocks nothing, and it runs on hardware you can buy today. The lesson generalizes past XLA: a vendor that shows up with current silicon and wires it into presubmit gets a better-tested backend than one whose hardware reaches the public tree only as a cron job.

Multi-GPU lives in a separate workflow, `Multi-Device CI`, on 8×H100 — the only public job that exercises the collectives and partitioning machinery I spent a section on above. Its triggers are `workflow_dispatch` and a midnight cron. It does not run on `pull_request` at all.[^34] So the one public job covering distributed execution gates nothing: no pull request is ever blocked by it.

The results follow from that. Over its last 100 completed runs on `main` (2026-04-19 through 2026-07-26) it recorded 52 failures against 48 successes, including seven consecutive daily failures from July 16 to July 22.[^34] That is not flaky with a tail; that is a coin flip, sustained for months.

Be careful what you conclude, though. A red nightly does not mean XLA's H100 support is broken — the internal equivalent presumably is not, because Google runs this code on vastly more than eight H100s, and `main` arrives pre-gated by tests you cannot see. That is the structural point. For code coming down the Copybara pipe, the public nightly is not a gate but a replica, and a replica that blocks nothing can stay red for a week without anything stopping. What you lose is not Google's confidence in the code; it is *your* ability to bisect. The signal that would tell you whether `main` is good on multi-GPU today is unreliable, so you end up building and testing XLA yourself — which brings us to the build.

### The build configuration is archaeology

Two lines tell the story. `.bazelrc` line 32 is `import %workspace%/tensorflow.bazelrc` — XLA's build still inherits a TensorFlow configuration file, four years after the repository split. Inside that file, under the comment `# By default, build TF in C++ 17 mode.`, sit the flags that pin the language standard: `-std=c++17` for Linux, macOS, Windows, and SYCL.[^35] In mid-2026, with `.bazelversion` at 8.7.0 and LLVM tracked near head, XLA builds as C++17 — two standards behind, under a comment that still says "TF." C++20 appears in the history only reactively, as repairs for downstream breakage ("Fix C++20 ambiguity in IntType comparison operators," July 2026): consumers compile XLA under newer standards and the project patches the fallout without moving its own baseline.

Second, `.bazelrc` line 2: `common --noenable_bzlmod --enable_workspace`. XLA defaults to the legacy `WORKSPACE` dependency system while pinned to Bazel 8, which deprecates it — with Bzlmod available as an opt-in `--config` carrying its own separate CI job. A migration underway, unfinished, and parked in the default-off position.

Neither is dangerous. Both are the tax of an old build, and together they are why "just build XLA from source" is an afternoon rather than a command.

### The issue tracker is where all of this surfaces

The repository carries 955 open items: 719 pull requests and 236 issues.[^36] The PR queue moves, if slowly — median open age 45 days, 59% older than a month. The issue tracker does not move. Median open-issue age is 364 days; half are over a year old, 27% over two. The oldest open issue is #1, "CMake build support," filed the day the repository was created and still open 1,447 days later.

And the long threads cluster in one place: build and toolchain. The most-commented open issue in the repository is #16866, "Hermetic CUDA no longer respects `TF_DOWNLOAD_CLANG`" — 88 comments, opened September 2024, last active November 2025, still open.[^37] Sort the tracker by comment count and build failures dominate the top. That is the signature of a project whose compiler is excellent and whose *build* is what outsiders actually fight.

### The CPU backend is not neglected; it is perpetually mid-migration

One more pattern worth naming, because it is easy to misread as neglect. The CPU backend is not sitting still — it is being rewritten, again. At HEAD, `xla/backends/cpu` carries an elemental emitter, a fusion-emitter path, a `tiled` lowering, a oneDNN emitter, a newer `ynn` emitter, and an in-progress `xtile` stack that drew 126 commits in the last year. The recent log reads like a migration in flight: "Remove SymbolicTileAnalysis-based tiling from CPU tiled fusion emitter," "Deprecate `xla_cpu_use_fusion_emitters` debug option." The tip of tree the day I checked was `[REVERT] Enable complex data types (C64, C128) in CPU block fusion codegen (xtile)`.[^38]

Several codegen generations coexisting behind debug flags is what sustained investment looks like in a codebase this old — but it is also a real cost to you. Which emitter compiles your fusion depends on flags whose defaults move between releases, so CPU performance is less reproducible across XLA versions than the polish of the GPU path would lead you to expect.

The summary I would offer is that XLA's engineering health and its openness are two different axes, and it is easy to read one as a proxy for the other. The compiler is actively and heavily maintained, on every backend it ships. The *project* around it is downstream of an internal one, with a nightly multi-GPU job that gates nothing and shows it, a build inherited from TensorFlow, and an issue tracker where outside reports go to age. If you depend on XLA, budget for pinning versions, running your own tests, and reading HLO rather than filing issues.

## The bargain that comes with using XLA

XLA is one of the best static-shape, whole-program array compilers ever built. It fuses across an entire model, allocates every byte before the program runs, plans layouts down to tile order, and produces code that saturates real hardware. Every one of those wins comes from the same four commitments: freeze the shapes, see the whole graph, allocate statically, fuse with a global model. And every limitation — the recompiles, the rejected dynamic shapes, the un-overridable fusion, the `custom_call` walls, the placement-as-annotation — is the bill for those same four commitments. You do not get XLA's strengths and dodge its rigidities. They are one decision.

The project-level friction is a separate bill, owed to history rather than to design — the inherited build, the drifted public CI, the aging tracker — but you pay it at the same desk, and it belongs in the estimate.

That is worth internalizing before you reach for it. XLA shows how much performance is available once you assume a static graph on a single device. What it does not model — dynamism, and where the data lives — is not an oversight; it is the frontier. And the placement machine XLA strips out is where I would look next: sharding, like layout before it, is something you find out about at trace time rather than something the type system ever promised you.

## References

[^1]: **XLA README and Architecture.** "The XLA compiler takes models from popular ML frameworks such as PyTorch, TensorFlow, and JAX, and optimizes them for high-performance execution." ([Repo](https://github.com/openxla/xla), [Architecture](https://openxla.org/xla/architecture))

[^2]: **XLA Architecture — the compilation pipeline.** Target-independent optimization (CSE, fusion, buffer analysis), target-dependent optimization, and LLVM-based codegen for CPU/GPU. ([Link](https://openxla.org/xla/architecture))

[^3]: **StableHLO.** Portability layer between frameworks and compilers; "based on the MHLO dialect" with serialization and versioning; supports bounded dynamic shapes. ([Repo](https://github.com/openxla/stablehlo), [Spec](https://openxla.org/stablehlo/spec))

[^4]: **XLA Terminology — HLO.** "HLO … is an internal graph representation (IR) for the XLA compiler … It is not based on MLIR, and has its own textual syntax and binary (protobuf based) representation." (Note: this page expands HLO as "High Level Optimizer," while the architecture page uses "high level operations.") ([Link](https://openxla.org/xla/terminology))

[^5]: **XLA Operation Semantics.** Definitions of `dot_general`, `convolution`, `reduce`, `dynamic-slice`, `all-reduce`/`all-gather`, and `custom-call` ("Call a user-provided function within a computation"). ([Link](https://openxla.org/xla/operation_semantics))

[^6]: **`hlo_opcode.h`.** The HLO opcode enum (`kDot`, `kConvolution`, `kReduce`, `kDynamicSlice`, `kAllReduce`, `kAllGather`, `kCustomCall`, …); ~150 opcodes, statically asserted under 256. ([Source](https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_opcode.h))

[^7]: **JAX: Ahead-of-time lowering and compilation.** `.lower()` produces StableHLO; `.compile()` produces the optimized executable; compiled functions are "specialized to a particular set of argument 'types,' such as arrays with a specific shape and element type." ([Link](https://docs.jax.dev/en/latest/aot.html))

[^8]: **XLA: Shapes and Layout.** Minor-to-major layout order; "`{1,0}`" is row-major, "`{0,1}`" column-major; tiled layouts for accelerators. ([Link](https://openxla.org/xla/shapes), [Tiled layout](https://openxla.org/xla/tiled_layout))

[^9]: **`layout_assignment.cc`.** Assigns layouts "while satisfying all necessary invariants and minimizing cost," with `SetOperandLayout`, `CreateCopyWithNewLayout`, `CopyOperandIfLayoutsDiffer`, and custom-call constraint propagation. ([Source](https://github.com/openxla/xla/blob/main/xla/service/layout_assignment.cc))

[^10]: **Simplification passes.** `algebraic_simplifier` ("mostly arithmetic simplifications"), `hlo_dce` (removes dead instructions/computations), `hlo_cse` (common-subexpression elimination). ([algsimp](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc), [dce](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/hlo_dce.cc), [cse](https://github.com/openxla/xla/blob/main/xla/service/hlo_cse.h))

[^11]: **`FusionKind`.** `enum class FusionKind { kLoop, kInput, kOutput, kCustom };`. ([Source](https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_instruction.h))

[^12]: **Fusion passes.** `instruction_fusion` fuses "vertically, meaning producing instructions are fused into their consumers"; `priority_fusion` (XLA:GPU) assigns priority from a cost model, "`priority = time_unfused - time_fused`." ([instruction_fusion](https://github.com/openxla/xla/blob/main/xla/service/instruction_fusion.h), [priority_fusion](https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/priority_fusion.h))

[^13]: **`buffer_assignment.cc` and input/output aliasing.** Assigns LogicalBuffers to BufferAllocations at fixed offsets; "A single BufferAllocation may hold LogicalBuffers with disjoint liveness" (reuse). Donation is configured via `HloInputOutputAliasConfig`. ([buffer_assignment](https://github.com/openxla/xla/blob/main/xla/service/buffer_assignment.h), [alias config](https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_input_output_alias_config.h))

[^14]: **`copy_insertion.cc`.** "A legalization HLO pass which inserts copies (kCopy instructions) to eliminate several kinds of problems in the HLO module." ([Source](https://github.com/openxla/xla/blob/main/xla/service/copy_insertion.cc))

[^15]: **`latency_hiding_scheduler.cc`.** "A scheduler oriented to hiding latencies of operations that can run in parallel with other operations," with collective-overlap limits. ([Source](https://github.com/openxla/xla/blob/main/xla/service/latency_hiding_scheduler.h))

[^16]: **GPU libraries and rewriters.** cuBLAS/cuDNN custom-call targets (`__cublas$gemm`, `__cudnn$convForward`) in `cublas_cudnn.h`; `gemm_rewriter`/`conv_rewriter` lower matmul/convolution to them. ([targets](https://github.com/openxla/xla/blob/main/xla/service/gpu/cublas_cudnn.h), [GPU arch](https://openxla.org/xla/gpu_architecture))

[^17]: **XLA:GPU and Triton.** "For more advanced fusions which include matrix multiplication or softmax, XLA:GPU uses Triton as a code-generation layer"; Triton emitter under `xla/backends/gpu/codegen/triton/`. ([GPU arch](https://openxla.org/xla/gpu_architecture), [Triton emitter](https://github.com/openxla/xla/blob/main/xla/backends/gpu/codegen/triton/fusion.h))

[^18]: **Autotuning.** The autotuner profiles all candidate implementations and selects the fastest; OpenXLA notes tuning "can take a long time if there are many fusions," motivating a persisted cache. ([autotuner](https://github.com/openxla/xla/blob/main/xla/backends/autotuner/autotuner.cc), [persisted autotuning](https://openxla.org/xla/persisted_autotuning))

[^19]: **XLA:TPU is closed-source.** Maintainer: "the TPU backend is non-OSS, so you can't build it from the GH repo"; and "the TPU dependent optimizations are mostly not open source." ([#11599](https://github.com/openxla/xla/issues/11599), [#25613](https://github.com/openxla/xla/issues/25613))

[^20]: **JAX: caching and persistent compilation cache.** "JAX will store copies of compiled programs on disk"; jit "will get compiled, and the resulting XLA code will get cached," reused on subsequent calls with the same shapes/dtypes. ([persistent cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html), [jit caching](https://docs.jax.dev/en/latest/jit-compilation.html))

[^21]: **Dynamic shapes are limited.** StableHLO: "limited framework support … JAX does not currently trace operations which lead to data dependent dynamism." JAX supports dynamic input "through padding or bucket padding"; variable-length arrays "trigger recompilation." ([StableHLO dynamism](https://openxla.org/stablehlo/dynamism), [JAX #26265](https://github.com/jax-ml/jax/issues/26265), [JAX #2521](https://github.com/jax-ml/jax/issues/2521))

[^22]: **`jax.numpy.nonzero`.** "Because the size of the output of `nonzero` is data-dependent, the function is not compatible with JIT"; requires a statically-specified `size=`. ([Link](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nonzero.html))

[^23]: **KV-cache / prefill padding.** "Prefills are padded to the longest sequence and we waste a lot of compute." ([JAX scaling book, inference](https://github.com/jax-ml/scaling-book/blob/main/inference.md))

[^24]: **`custom_call` is opaque to the optimizer.** StableHLO spec: an escape hatch whose "semantics … are implementation-defined and are not analyzed by the compiler." JAX FFI: "an opaque black box: JAX can't look inside it." ([StableHLO custom_call](https://openxla.org/stablehlo/spec#custom_call), [JAX FFI](https://docs.jax.dev/en/latest/ffi.html))

[^25]: **Sharding is an annotation, not part of the type.** `HloInstruction` carries an optional, nullable sharding (`has_sharding()`, `set_sharding()`) separate from its `Shape`; backed by the `OpSharding` proto. ([hlo_instruction.h](https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_instruction.h), [hlo_sharding.h](https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_sharding.h))

[^26]: **SPMD partitioner (GSPMD).** `SpmdPartitioner` with `SPMDCollectiveOpsCreator`, "a set of functions that create the cross-partition collective ops"; GSPMD "consumes HLO with sharding annotations … and produces a sharded HLO … with proper collectives." ([spmd_partitioner](https://github.com/openxla/xla/blob/main/xla/service/spmd/spmd_partitioner.h), [GSPMD paper](https://arxiv.org/abs/2105.04663))

[^27]: **Shardy.** "An MLIR-based tensor partitioning system … Built from the collaboration of both the GSPMD and PartIR teams, it incorporates the best of both"; the partitioner adds "the necessary data movement/formatting and collective operations." ([Repo](https://github.com/openxla/shardy), [Overview](https://openxla.org/shardy/overview))

[^28]: **Real-world long-compile reports.** E.g. XLA compilation orders of magnitude slower on newer GPU targets, and recurring compile-time spikes on the JAX tracker. ([openxla/xla #44517](https://github.com/openxla/xla/issues/44517), [jax #30185](https://github.com/jax-ml/jax/issues/30185))

[^29]: **Per-directory commit activity.** Measured on a clone of `openxla/xla` at `5617432` (2026-07-26) with `git rev-list --count --since=2025-07-26 HEAD -- <path>`: `xla/backends/gpu` 2186, `xla/service/gpu` 1679, `xla/backends/gpu/codegen/triton` 595, `xla/backends/cpu` 668, `xla/service/cpu` 374, `xla/stream_executor/cuda` 325, `xla/stream_executor/rocm` 135. Normalized by `.cc`/`.h` file count, CPU churn is comparable to GPU. Last modification dates for `xla/stream_executor/cuda/cudnn.cc` and `cublaslt.cc` were both 2026-07-22. ([Repo](https://github.com/openxla/xla))

[^30]: **Copybara export.** Every one of the last 2,000 commits on `main` (verified individually, not sampled) contains a `PiperOrigin-RevId:` trailer, indicating export from Google's internal monorepo rather than development on GitHub. ([Commit history](https://github.com/openxla/xla/commits/main))

[^31]: **Author domains and external contributions.** `git log --since=2025-07-26 --format='%ae'` over 10,610 commits: google.com 9328 (88%), nvidia.com 360, amd.com 305, openxla.org 204, intel.com 148, github noreply 140, gmail.com 51, amazon.com 23. Of those, 1,304 commits carry the `PR #NNNNN:` subject prefix Copybara applies to imported pull requests, including 355 of NVIDIA's 360, 304 of AMD's 305, and all 148 of Intel's. In the same 2,000-commit window used for the Copybara check, 385 commits were authored from non-`@google.com` addresses and all 385 also carried a `PiperOrigin-RevId:` trailer. Mechanical commits ("Automated Code Change," license-header sweeps, LLVM integrations) account for 1,289 of the yearly set. **Review activity on external PRs:** sampling the 14 most recent vendor-authored pull requests referenced by those commits (`#41903`, `#44560`, `#44796`, `#45296`, `#45707`, `#45938`, `#45950`, `#46026`, `#46077`, `#46085`, `#46095`, `#46122`, `#46126`, `#46138`), all 14 had at least one GitHub review; totals were 79 reviews and 143 review-plus-issue comments, with [#41903](https://github.com/openxla/xla/pull/41903) at 13 reviews and 22 commits. By contrast, recently merged `copybara-service[bot]` PRs — internal changes exported as pull requests — showed 0 reviews and 0 comments across the 18 sampled.

[^32]: **Copybara attribution friction.** "Better Link Copybara-generated PRs to the Googler who authored them" — 31 comments discussing how internally-authored exports and external contributions appear in the public history. ([openxla/xla #448](https://github.com/openxla/xla/issues/448))

[^33]: **CI job matrix.** `.github/workflows/ci.yml` defines the matrix: XLA Linux/Windows x86 CPU, ARM64 CPU, a Bzlmod variant, plus JAX and TensorFlow jobs. XLA's only GPU-executing job uses pool `linux-x86-g2-16-l4-1gpu` (one L4); the "XLA Linux x86 GPU Hermetic ROCm" and "XLA Linux X86 GPU ONEAPI" jobs are assigned pool `linux-x86-n2-16` (CPU). ([ci.yml](https://github.com/openxla/xla/blob/main/.github/workflows/ci.yml))

[^34]: **Multi-Device CI: triggers and failure rate.** `.github/workflows/ci_multi_device.yml`, pool `linux-x86-a3-8g-h100-8gpu`. Its `on:` block contains only `workflow_dispatch` and `schedule: - cron: "0 0 * * *"` — there is no `pull_request` trigger, unlike `ci.yml`, which has both `pull_request` and `push: branches: [main]`. Of the workflow's last 100 completed runs on `main` via the Actions API (spanning 2026-04-19 to 2026-07-26): 52 `failure`, 48 `success`; consecutive daily failures 2026-07-16 through 2026-07-22. ([ci_multi_device.yml](https://github.com/openxla/xla/blob/main/.github/workflows/ci_multi_device.yml), [workflow runs](https://github.com/openxla/xla/actions/workflows/ci_multi_device.yml))

[^35]: **C++17 and the inherited TensorFlow build config.** `.bazelrc` line 32: `import %workspace%/tensorflow.bazelrc`. In `tensorflow.bazelrc`, line 384 reads `# By default, build TF in C++ 17 mode.`, followed by `common:linux --cxxopt=-std=c++17` / `--host_cxxopt=-std=c++17` (lines 385–386), the same for macOS (387–388), `/std:c++17` for Windows (389–390), and c++17 for SYCL (307–308). `.bazelversion` is `8.7.0`; `.bazelrc` line 2 is `common --noenable_bzlmod --enable_workspace`. ([.bazelrc](https://github.com/openxla/xla/blob/main/.bazelrc), [tensorflow.bazelrc](https://github.com/openxla/xla/blob/main/tensorflow.bazelrc))

[^36]: **Open issue and PR ages.** GitHub API, 2026-07-26: `open_issues_count` 955, comprising 719 open pull requests and 236 open issues. Open PRs: median age 45 days, 59% older than 30 days. Open issues: median age 364 days, 50% older than one year, 27% older than two. Oldest open issue is #1, "CMake build support," created 2022-08-09 (1,447 days). ([Issue #1](https://github.com/openxla/xla/issues/1))

[^37]: **Most-commented open issue.** "Hermetic CUDA no longer respects TF_DOWNLOAD_CLANG" — 88 comments, opened 2024-09-06, last updated 2025-11-23, still open. Sorting the tracker by comment count returns predominantly build failures. ([openxla/xla #16866](https://github.com/openxla/xla/issues/16866))

[^39]: **GPU hardware coverage in public CI vs. supported targets.** Every `pool:` label across `.github/workflows/` resolves to CPU (`linux-x86-n2-16`, `linux-x86-n2-128`, `linux-arm64-c4a-16`, `linux-arm64-t2a-48`, `windows-x86-n2-16`) except two NVIDIA pools: `linux-x86-g2-16-l4-1gpu` (L4) and `linux-x86-a3-8g-h100-8gpu` (8×H100). No Blackwell-class runner appears. XLA source nonetheless targets it: `git grep` over `xla/` returns `sm_100` (16), `sm_103` (6), `sm_120` (1) plus `a`-suffixed variants, and `xla/stream_executor/device_description.h` defines `CudaComputeCapability::Blackwell()` and `BlackwellFamily()` alongside `Volta()`, `Ampere()`, and `Hopper()`. **ROCm:** `.github/workflows/rocm_ci.yml` triggers on `pull_request` and defaults `runner_label` to `amd-do-linux-xla-gpu-gfx950-1` (gfx950 = CDNA4, MI350 series), a vendor-hosted runner. Across 398 completed runs returned by the Actions API (2026-07-24 to 2026-07-27, all branches): 213 `success`, 89 `failure`, 86 `cancelled`, 10 `action_required` — a 70.5% pass rate among decided runs. Note the short window: this is a high-volume presubmit workflow, so 398 runs span only four days and the rate should not be read as a long-run average. ([rocm_ci.yml](https://github.com/openxla/xla/blob/main/.github/workflows/rocm_ci.yml), [device_description.h](https://github.com/openxla/xla/blob/main/xla/stream_executor/device_description.h))

[^38]: **CPU codegen generations.** At `5617432`, `xla/backends/cpu` contains `codegen/elemental`, `codegen/emitters`, `codegen/tiled`, `onednn_emitter.cc`, `ynn_emitter.cc`, and an in-progress `xtile` stack (126 commits mentioning "xtile" in the last year). Recent subjects include "Remove SymbolicTileAnalysis-based tiling from CPU tiled fusion emitter" and "Deprecate `xla_cpu_use_fusion_emitters` debug option"; HEAD on 2026-07-26 was `[REVERT] Enable complex data types (C64, C128) in CPU block fusion codegen (xtile).`

---

*Disclaimer: This article was generated using the Claude Opus 4.8 model.*
