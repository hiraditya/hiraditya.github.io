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

## The bargain, stated plainly

XLA is one of the best static-shape, whole-program array compilers ever built, and I do not say that as a hedge before the criticism — I say it because the criticism only makes sense against that baseline. It fuses across an entire model, allocates every byte before the program runs, plans layouts down to tile order, and produces code that saturates real hardware. Every one of those wins comes from the same four commitments: freeze the shapes, see the whole graph, allocate statically, fuse with a global model. And every limitation — the recompiles, the rejected dynamic shapes, the un-overridable fusion, the `custom_call` walls, the placement-as-annotation — is the bill for those same four commitments. You do not get XLA's strengths and dodge its rigidities. They are one decision.

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

---

*Disclaimer: This article was generated using the Claude Opus 4.8 model.*
