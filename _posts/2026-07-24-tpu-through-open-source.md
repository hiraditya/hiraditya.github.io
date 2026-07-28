---
title: "Programming the TPU: What Its Open-Source Compiler Already Tells You"
date: 2026-07-27 00:00:00 -0700
categories: [Compilers, ML-Systems]
tags: [tpu, xla, mosaic, jax, hardware]
---

The TPU has less public hardware documentation than a GPU. There is no vendor ISA manual, no die-level memory-model spec; you get peak FLOPs, HBM capacity, and a block diagram. But if you write JAX or Pallas kernels for TPUs, you are not actually flying blind, because Google ships — openly, Apache-2.0 licensed — the compilers that target the chip, and a compiler has to name the parts of the machine it generates code for. The memory spaces, the register geometry, the instruction set, the synchronization primitives: all of it is spelled out in code you can read. Two sources carry most of it: the XLA:TPU **C API** in `openxla/xla`, and, far more usefully, the **Mosaic TPU dialect** — the MLIR dialect that Pallas kernels lower to — shipped inside `jax`.[^1][^2]

This post is a reading of that material for practitioners. For each part of the machine the compiler describes, I will point at where it is documented and then say plainly how knowing it helps you write faster, more correct TPU code. If you have ever wondered why a TPU kernel wants shapes that are multiples of 128, what "keep it in VMEM" actually means, or whether you can `printf` from a TPU kernel, this is where the answers already live.

I noted in [my review of XLA]({% post_url 2026-07-23-xla-review-and-critique %}) that XLA:TPU is the one backend that is closed — a maintainer's own words are that "the TPU dependent optimizations are mostly not open source." That is true of the *optimizations*. It is much less true of the *machine model*, and this post is about the difference.

One honest note on scope. The compiler keeps the exact *numbers* — precise lane counts, memory sizes, latencies — in the closed `libtpu.so`, and the open dialect is deliberately parameterized rather than hardcoded: the layout-inference entry point takes a `hardware_generation` and a `target_shape` as arguments.[^3] That is a useful signal in itself. It tells you which quantities are stable enough to design around (the *structure*: spaces, tilings, op set) and which you should let the compiler pick (the *constants*). The structure is documented; the constants are supplied per generation.

## Eight memory spaces, and why your `Ref`'s space is the first thing to get right

Start with memory, because on a TPU it is the thing you manage explicitly and the thing that most determines performance. Mosaic's `MemorySpace` enum names eight:[^4]

```text
kVmem          (0)            vector memory   — on-chip scratchpad for the vector unit
kSmem          (1)            scalar memory   — on-chip, for the scalar core
kHbm           (2)            high-bandwidth memory — off-chip DRAM
kCmem          (3)            "cmem"          — a further on-chip space
kSemaphoreMem  (4)            semaphore memory — where sync primitives live
kVmemShared    (5)            VMEM shared across subcores
kHost          (6)            host DRAM
kAny           (4294967295)   unconstrained / compiler's choice
```

Compared with CUDA's three that matter (global, shared, local), the TPU exposes a richer, explicitly *typed* hierarchy: distinct scalar and vector on-chip memories, a dedicated semaphore space, a shared-VMEM tier for cross-subcore communication, and host DRAM as a named space.

**How this helps you write better TPU code.** In a Pallas TPU kernel every buffer is a `Ref` tagged with one of these spaces, and that tag is the single most consequential choice you make. VMEM is the fast on-chip scratchpad your vector unit reads at speed; HBM is off-chip and only reached by DMA. The core performance idiom of every good TPU kernel follows directly: stage the tile you are about to compute on from HBM into VMEM, work on it there, and write it back — never compute directly against HBM. Scalars and loop bounds belong in SMEM, not VMEM. Getting the space wrong is not a micro-optimization; it is the difference between a kernel that runs at bandwidth and one that stalls or fails to lower.

VMEM is also finite and you will run out of it, which is why Mosaic exposes a `vmem_limit_bytes` kernel parameter — and its docstring carries a detail worth knowing before you spend an afternoon on it: raising it "must be used in conjunction with the `--xla_tpu_scoped_vmem_limit_kib=N` flag with `N*1kib > vmem_limit_bytes`."[^10] Two knobs, and the compiler flag has to move with the kernel parameter.

## Two kinds of core, and they do not have the same register shape

The `CoreType` enum names three things:[^4]

```text
kTc                (0)  TensorCore        — the dense matmul/vector engine
kScScalarSubcore   (1)  SparseCore, scalar subcore
kScVectorSubcore   (2)  SparseCore, vector subcore
```

TensorCore is what people mean by "TPU." **SparseCore** is the other engine, and the open XLA C API tells you what it is for: its entry points are `SparseCore_GetMaxIdsAndUniques` and `TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice`, with a dedicated `CompileTimeSparseCoreAllocationFailure` error code.[^5] "Ids and uniques" is embedding-lookup vocabulary — SparseCore is the hardware embedding engine for recommendation and ranking models, the gather/scatter-over-huge-tables workloads that are not dense matmul. It is itself split into separately addressable scalar and vector subcores.

The more interesting detail is that the two engines do not share a register geometry. The comment on the layout-inference pass is explicit that `target_shape` is shaped differently per core:[^3]

```text
// The `target_shape` can either be
// * 1D -- (lane count) SparseCore tiling; or
// * 2D -- (sublane count, lane count) TensorCore tiling.
```

TensorCore tiles data over two dimensions. SparseCore tiles over one. That is a genuine architectural difference, not a compiler convenience, and it fits what each engine is for: dense math wants a 2D tile to feed a systolic array, while embedding gathers want a flat vector of independent lookups.

**How this helps you write better TPU code.** It tells you which workloads map to which silicon, and therefore where to spend effort. Dense transformer math is a TensorCore problem — think in matmuls and vector ops. Large embedding tables are a SparseCore problem, and the framework paths that target it are the ones to reach for rather than trying to force gather-heavy lookups through the TensorCore. It also means intuitions do not transfer between the two: the "make everything a multiple of 8 and 128" advice below is a *TensorCore* rule, derived from a 2D tiling that SparseCore does not have. If your model is recsys-shaped, the existence and capacity of SparseCore on your generation is a real capacity-planning input, which is exactly why `MaybeAvailableSparseCoresPerLogicalDevice` is a queryable fact.

## Two-dimensional vector registers — why 8 and 128 are magic numbers

The TensorCore's vector unit does not use flat SIMD registers. Mosaic's `VectorLayout` describes values as tiled over **two** dimensions of a vector register (a "vreg"): **sublanes** and **lanes**, with reductions and shuffles defined along one or the other (`Direction::kSublanes`, `kLanes`).[^6] Sub-32-bit types are *packed*: the layout appends an implicit `(32/bitwidth, 1)` tiling, so a vreg holds `target_shape[0] * packing` sublanes, where `packing = 32/bitwidth`.[^7] bf16 packs 2 elements per 32-bit slot; int8 packs 4. The exact dimensions come from `target_shape` per generation, but 128 lanes is pervasive (tilings like `(1, 128)` and `(8, 128)` recur throughout) and 8 sublanes is the mainline configuration — so the canonical vreg is an `8 × 128` tile of 32-bit slots.

Packing adds a third axis, and the verifier says so: `"Vreg rank should be 2 or 3"`.[^8] A vreg is 2D for 32-bit data and 3D once sub-32-bit elements are packed into each slot. There are two ways to arrange that packing, named by `PackFormat`: `compressed` and `interleaved`.[^4] And the conversions are not arbitrary — the verifier restricts them to `"Only packing/unpacking f32 <-> bf16 and integer <-> integer"`.[^8]

**How this helps you write better TPU code.** This is the source of the "shapes should be multiples of 128 (and 8)" folklore, now with a reason attached. When your last dimension is a multiple of 128 and your second-to-last is a multiple of 8, your data fills vregs exactly, with no wasted lanes and no padding; when it is not, the compiler pads to the tile and you pay for lanes that do no work. It also explains why *relayouts* — changing how a value is tiled across vregs — show up in profiles and cost real time: prefer to keep a value in one layout through a kernel rather than forcing transposes between sublane- and lane-major. And it makes the low-precision story concrete: bf16 packing two-per-slot is why bf16 throughput is what it is, and why sub-optimal shapes hurt *more* in low precision, since a padded lane now wastes two elements instead of one. Design your tiles around `(8, 128)` and you are working with the hardware instead of against it.

## The grid is typed, and the type tells the compiler what it may reorder

This is the piece I see missed most often, and it is the cheapest performance in Pallas. A Mosaic kernel's grid dimensions each carry a `DimensionSemantics`, and there are four:[^4]

```text
parallel           (0)   may execute in any order
arbitrary          (1)   must execute sequentially
core_parallel      (2)   parallel across cores
subcore_parallel   (3)   parallel across subcores
```

These are exposed directly in Pallas as `pltpu.PARALLEL`, `pltpu.ARBITRARY`, `pltpu.CORE_PARALLEL`, and `pltpu.SUBCORE_PARALLEL`, passed as `dimension_semantics` on the kernel's compiler params.[^10] The docstring is blunt about the contract: "parallel" for dimensions that can execute in any order, "arbitrary" for dimensions that must be executed sequentially.

There is a related knob in `RevisitMode`, which is `immediate` or `any` — whether a grid block, once left, may be returned to immediately or at any later point.[^4]

**How this helps you write better TPU code.** The default is the conservative one. A dimension you leave unannotated is treated as order-dependent, which forecloses parallelization and, just as importantly, forecloses the pipelining that makes TPU kernels fast — if the compiler cannot prove two iterations are independent, it cannot overlap their DMAs. Marking a genuinely independent grid axis `PARALLEL` is often a one-line change with a large effect, and marking a reduction axis `ARBITRARY` is how you stay correct. `CORE_PARALLEL` and `SUBCORE_PARALLEL` are the multi-core equivalents, and their existence tells you the compiler is willing to spread a grid axis across cores if you promise it may. This is a case where the annotation is not documentation, it is a permission slip, and the compiler will not guess on your behalf.

## The vector ISA — a menu of what is cheap in hardware

`tpu_ops.td` defines **90 operations**, and it is the closest thing to a public TPU vector ISA. Reading it tells you which primitives the hardware does natively.[^8] Grouped:

- **Loads/stores**: `vector_load`/`store`, `strided_load`/`store`, `shuffled_load`/`store`, indexed `vector_load_idx`/`store_idx`, plus `gather` and `dynamic_gather` (gather/scatter within VMEM).
- **Data movement**: `dynamic_rotate`, `sublane_shuffle`, `broadcast_in_sublanes`, `iota`, `repeat`, `concatenate`, `transpose`, `roll_vectors`/`unroll_vectors`, `relayout`.
- **Packing/precision**: `pack_subelements`/`unpack_subelements`, `pack_elementwise`/`unpack_elementwise`, `ext_f`/`trunc_f`, int↔float conversions, `stochastic_convert` (stochastic rounding as a hardware primitive), `bitcast`/`bitcast_vreg`.
- **Reductions**: `all_reduce`, `reduce_index`, `scan`, `scan_count`, `sort`.
- **Masks**: `create_mask`, `create_subelement_mask`, `pack_mask`, `mask_cast`.
- **Math**: `reciprocal`, `matmul`, and `prng_seed32`/`prng_random_bits` (an on-core RNG).

Two enums sharpen the picture. `ReductionKind` is `sum`, `max`, `min`, `arg_max`, `arg_min`, and `find_first_set` — the last being a bit-scan, which is a hardware reduction you would not guess was there.[^4] And `RoundingMode` has exactly two members: `towards_zero` and `to_nearest_even`.[^4] Not a full IEEE menu — two modes, which is what the datapath provides.

Low precision gets a notably general treatment. Beyond the standard types, the dialect defines `Float8EXMY`, described as an "EXMY type in a 8 bit container" and parameterized by an underlying float type, with meaningful bits aligned to the LSB and higher bits ignored — citing arXiv 2405.13938 for the format family.[^11] That is a *container* for a range of 8-bit float layouts rather than a commitment to one, which tells you the hardware and compiler are set up to move between fp8 variants.

**How this helps you write better TPU code.** This is a catalog of cheap operations. When a computation can be expressed as a sublane shuffle, a rotate, or an in-VMEM indexed gather, it is a single hardware primitive rather than something you should hand-roll with loops. Several are worth calling out. `stochastic_convert` means stochastic rounding is a datapath operation, so low-precision training that relies on it is not paying a software tax. `prng_random_bits` being on-core is the hardware reason JAX threads explicit PRNG keys — randomness is generated where the compute is, deterministically. `find_first_set` as a reduction kind is the kind of thing that turns a masked-search loop into one op. And the two-member `RoundingMode` is a portability warning: if your numerics depend on a rounding mode outside those two, you are going to emulate it.

## The matrix unit is a FIFO you feed

The MXU (the systolic array) is exposed not as one `matmul` op but as a *pipeline* — which tells you how it is actually driven:[^8]

```text
matmul_push_rhs     push the RHS (weights) into the array
matmul_acc_lhs      stream the LHS (activations), accumulating
matmul_pop          pop an accumulated result
matmul_lhs_fifo     LHS FIFO staging
matmul_pop_fifo     pop, with an `mxu_index` attribute
```

Three facts fall out. The interface is weight-stationary: you load the stationary operand once, then stream the moving operand through. `mxu_index` means there is more than one MXU per core. And the accumulator is fixed-width regardless of your inputs — the verifier requires `"Expected matmul acc to be 32-bit"`, while `ContractPrecision` lets the contraction itself be `bf16` or `fp32`.[^4][^8]

**How this helps you write better TPU code.** Weight-stationary is a scheduling hint you can act on: once weights are pushed, streaming many activation rows through amortizes that load, which is the hardware reason larger batch (or longer sequence) dimensions improve MXU utilization, and why tiny matmuls waste it. Multiple MXUs (`mxu_index`) means there is parallelism across matrix units to keep fed — structure work so more than one can run. And the 32-bit accumulator is the most actionable of the three: "bf16 inputs, fp32 accumulation" is not a vague mixed-precision gesture, it is the only shape the unit offers. You do not get a bf16 accumulator, so you do not need to fear one.

## Data moves by DMA, and synchronizes by semaphore

There is no cache coherence to lean on. Data moves between spaces by explicit asynchronous DMA, and correctness comes from semaphores:[^8]

```text
enqueue_dma / enqueue_indirect_dma       start an async copy
wait_dma / wait_dma2 / wait_indirect_dma block until it lands
semaphore_read / _wait / _signal         the sync primitives
alloca_semaphore                         allocate one
get_barrier_semaphore                    the barrier's semaphore
barrier, fetch_and_add_sync              coordination
delay                                    stall for a given number of nanos
core_id / subcore_id / device_id         who and where am I
```

Mosaic also names the overlap directly: `PipelineMode` is `synchronous` or `double_buffered`.[^4]

**How this helps you write better TPU code.** This *is* the TPU kernel programming model, and it is where kernel performance is won or lost. The pattern is: `enqueue_dma` a tile from HBM into VMEM, `wait_dma` on its semaphore, compute, DMA the result back — and, crucially, **double-buffer**: issue the next tile's DMA before computing on the current one, so data movement overlaps arithmetic instead of serializing with it. A TPU kernel that waits for each DMA before starting the next is leaving most of its performance on the table; one that pipelines DMA against compute runs near bandwidth. In Pallas this is exactly what the grid and the input/output specs are arranging for you — and it is why the `DimensionSemantics` annotation above matters so much, since the compiler can only build that pipeline across iterations it is allowed to reorder.

Note also `enqueue_indirect_dma` and `wait_indirect_dma`: DMA where the addresses themselves come from data. That is gather/scatter at the memory-movement level rather than inside the vector unit, which is the right tool for sparse access patterns that would otherwise serialize.

## The chip boundary is visible, and the compiler analyzes it

`device_id` sitting next to `core_id` and `subcore_id` is a hint, and the dialect header confirms it. Mosaic exposes an analysis with this signature:[^3]

```cpp
std::pair<bool, bool> mightCommunicateBetweenChips(Operation* op);
```

Its implementation runs an `analyzeCrossChipCommunication` walk and returns whether the op has cross-chip communication and whether it uses a custom barrier.[^12] So the compiler knows, per operation, whether it crosses a chip boundary.

**How this helps you write better TPU code.** It tells you the boundary is a first-class concept at the kernel level, not just at the XLA and GSPMD level above it — a Pallas kernel can address its own core, subcore, and device, and the compiler tracks when an operation reaches off-chip. Practically, this is the level at which "is this kernel secretly doing a cross-chip transfer?" is a decidable question rather than a guess, and it is why kernels that stay on-chip are structurally different animals from kernels that do not.

## You can `printf` from a TPU kernel, and time regions of it

This is the section I wish someone had shown me first, because the ops are right there and they are easy to miss. Debugging and profiling are dialect operations:[^8]

```text
log            a tag string + variadic values, with a `formatted` flag
log_buffer     dump a whole buffer
trace          a traced region
trace_start / trace_stop / trace_value
delay          stall for N nanoseconds
```

`tpu.log` takes a `StrAttr` tag, a variadic list of inputs, and a `formatted` boolean.[^8] That is `printf`, on device, from inside a kernel.

One more is worth knowing because its name gives nothing away. `tpu.weird` takes an F32 and returns an I1, and the verifier enforces exactly that.[^8] What is it? JAX's lowering answers: `lax.is_finite` lowers to the negation of it.[^13]

```python
@register_lowering_rule(lax.is_finite_p)
def _is_finite_lowering_rule(ctx: LoweringRuleContext, x):
  out_aval, = ctx.avals_out
  out_type = ctx.aval_to_ir_type(out_aval)
  return _not_lowering_rule(ctx, tpu.weird(out_type, x))
```

So the TPU has a hardware predicate for "this float is weird" — NaN or infinity — and `is_finite` is its complement.

**How this helps you write better TPU code.** Kernel debugging on an accelerator is usually the worst part of the job, and the tooling here is better than its reputation: you can log tagged values from inside a kernel, dump buffers, and bracket regions with trace markers that show up in a profile rather than guessing from wall-clock deltas. `delay` is a genuine instrument for testing pipelining hypotheses — insert a known stall and see whether your DMA overlap actually absorbs it. And knowing `is_finite` is one hardware op means NaN-hunting inside a kernel is cheap; you do not need to round-trip to the host to find where your numerics went bad.

## Hints you are allowed to hand the compiler

A handful of ops exist purely so you can tell the compiler something it cannot prove:[^8]

```text
assume_multiple       "this value is a multiple of N"
assume_layout         assert a layout rather than infer it
erase_layout          drop a layout annotation
reinterpret_cast      retype a memref
memref_slice / memref_squeeze / memref_reshape / memref_bitcast
get_internal_scratch  ask for scratch memory
get_iteration_bound   the current grid bound
```

`assume_multiple`'s summary states it plainly: "Assumes that a value is a multiple of a given integer."[^8]

**How this helps you write better TPU code.** These are the escape valves for the padding problem. When a dimension is dynamic, the compiler must assume the worst and emit masking and padding around every access; `assume_multiple` on a bound you know is a multiple of 128 or 8 lets it drop that. That is often the difference between a kernel that hits the vreg geometry described above and one that pays for a partial tile on every iteration. The memref-reshaping family is how you re-view a buffer without a copy, and `get_internal_scratch` is the sanctioned way to ask for working space instead of smuggling in another input. As with `dimension_semantics`, the theme is that the compiler defaults to conservative and gives you named ways to relax it — but only if you know they exist.

## Generations, and writing code that survives them

The open C API enumerates the lineage — `TpuVersionEnum` runs `kTpuV2, kTpuV3, kTpuV4, kTpuV5`[^9] — and the layout machinery takes a `hardware_generation` and a `target_shape` rather than baking in constants.[^3]

There is a second stability mechanism worth noting: the Mosaic dialect is *versioned and serialized*. The serde pass reads and writes a `stable_mosaic.version` attribute and tracks a highest supported version.[^14] The op list itself carries evidence of that versioning in the open — `wait_dma` and `wait_dma2` coexist, which is what an evolving interface with a compatibility story looks like.

**How this helps you write better TPU code.** It is a portability instruction. Because the exact vreg geometry and memory sizes are supplied per generation, kernels that hardcode a specific tile size or buffer capacity are brittle across v4/v5/newer; kernels written in terms of the compiler's parameters — let Pallas and Mosaic pick the tiling for the target, express sizes relative to the block, avoid magic numbers — move forward without a rewrite. The parameterization is the compiler telling you which knobs are its job, not yours. And the versioned serialization means a compiled Mosaic kernel is a versioned artifact, not an opaque blob tied to one toolchain build, which is the property you want if you are shipping kernels rather than just running them.

## What this is good for

None of the above requires a leaked document — it is Google's own public code, describing Google's own hardware, because a compiler that emits TPU code has to. Read as intended, it is a practical manual:

- the **memory spaces** are the menu for your `Ref` tags, and `vmem_limit_bytes` is the knob when you exhaust VMEM;
- the **two core types** have different register geometries, so TensorCore intuitions do not transfer to SparseCore;
- the **vreg geometry** is why your shapes should be multiples of 8 and 128, and why it matters more in bf16;
- the **grid semantics** are a permission slip for pipelining, and the default is conservative;
- the **ISA** is the list of cheap primitives, including a few (`find_first_set`, `stochastic_convert`) you would not expect;
- the **MXU FIFO** is why weight reuse and batch size matter, and why fp32 accumulation is the only option on offer;
- the **DMA/semaphore model** is why double-buffering is the whole game;
- and the **log and trace ops** mean you can debug in place instead of bisecting blind.

There is a broader observation I will keep brief. The TPU has a genuinely rich, explicit structure — a typed memory hierarchy, heterogeneous cores with different register shapes, a 2D register geometry, an async data-movement model, a visible chip boundary — and the Mosaic compiler tracks all of it as MLIR enums and attributes on ops, checked by its verifier. That structure is real and worth programming to. That it lives *beside* the type system, as annotations rather than as types, is a separate and interesting question: nothing in a Pallas kernel's Python signature records that a `Ref` must be in VMEM, or that a grid axis is parallel, and so those facts are discovered at lowering time rather than guaranteed at authoring time. For the working TPU programmer, though, the immediately useful thing is that the structure is documented, and you can design to it.

## Sourcing

Everything here is from public, Apache-2.0 code, read at `jax-ml/jax` commit `6079305` and `openxla/xla` commit `5617432`. The Mosaic TPU dialect (`jaxlib/mosaic/dialect/tpu/`) in `jax-ml/jax` is the primary source; the XLA:TPU C API (`xla/tpu/`) in `openxla/xla` supplies the generation and SparseCore details. Where I quote a constraint, it is a verifier message or an enum case, not an inference.

## References

[^1]: **OpenXLA / XLA — the TPU C API.** `xla/tpu/` contains the interface to `libtpu` (`libtftpu.h`, `tpu_executor_c_api.h`, `c_api_decl.h`). ([Link](https://github.com/openxla/xla/tree/main/xla/tpu))

[^2]: **Mosaic TPU dialect.** The MLIR dialect Pallas TPU kernels lower to. ([Link](https://github.com/jax-ml/jax/tree/main/jaxlib/mosaic/dialect/tpu))

[^3]: **`tpu_dialect.h` — hardware parameterization, per-core tiling, and the cross-chip analysis.** `createInferMemRefLayoutPass` takes an `int hardware_generation` and an `absl::Span<const int64_t> target_shape`; the accompanying comment states that `target_shape` is "1D -- (lane count) SparseCore tiling; or 2D -- (sublane count, lane count) TensorCore tiling." The same header declares `std::pair<bool, bool> mightCommunicateBetweenChips(Operation* op)`. ([Link](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/tpu_dialect.h))

[^4]: **`tpu_enums.td` — memory spaces, core types, modes.** `MemorySpace` (vmem 0, smem 1, hbm 2, cmem 3, semaphore_mem 4, vmem_shared 5, host 6, any 4294967295); `CoreType` (tc / sc_scalar_subcore / sc_vector_subcore); `DimensionSemantics` (parallel 0, arbitrary 1, core_parallel 2, subcore_parallel 3); `RevisitMode` (immediate / any); `PipelineMode` (synchronous / double_buffered); `ContractPrecision` (bf16 / fp32); `PackFormat` (compressed / interleaved); `ReductionKind` (sum / max / min / arg_max / arg_min / find_first_set); `RoundingMode` (towards_zero / to_nearest_even); `VectorLayoutDim` (tiled / lanes / sublanes). ([Link](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/tpu_enums.td))

[^5]: **SparseCore as an embedding engine.** `SparseCore_GetMaxIdsAndUniques`, `TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice` (`xla/tpu/tpu_ops_c_api.h`), and `CompileTimeSparseCoreAllocationFailure` (`xla/error/error_codes.h`). ([Link](https://github.com/openxla/xla/blob/main/xla/tpu/tpu_ops_c_api.h))

[^6]: **`layout.h` — the 2D vreg (sublanes × lanes) model.** `VectorLayout` maps values into vregs tiled over sublanes and lanes; `Direction` is `kSublanes`/`kLanes`/`kSubelements`. ([Link](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/layout.h))

[^7]: **`layout.cc` — packing of sub-32-bit types.** `tilesPerVreg` returns `{target_shape[0] * packing, target_shape[1]}` with `packing = 32/bitwidth`. ([Link](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/layout.cc))

[^8]: **`tpu_ops.td` and `tpu_ops.cc` — the 90-op set and its verifiers.** `tpu_ops.td` defines 90 `TPU_*Op` records: vector loads/stores (incl. strided/shuffled/indexed), `gather`/`dynamic_gather`, `dynamic_rotate`, `sublane_shuffle`, pack/unpack (subelement and elementwise), `stochastic_convert`, `prng_seed32`/`prng_random_bits`, `reciprocal`, reductions (`all_reduce`, `reduce_index`, `scan`, `scan_count`, `sort`); the MXU FIFO ops (`matmul_push_rhs`, `matmul_acc_lhs`, `matmul_pop`, `matmul_lhs_fifo`, `matmul_pop_fifo` with `mxu_index`); DMA/semaphore ops (`enqueue_dma`, `enqueue_indirect_dma`, `wait_dma`, `wait_dma2`, `wait_indirect_dma`, `semaphore_read`/`_wait`/`_signal`, `alloca_semaphore`, `get_barrier_semaphore`, `barrier`, `fetch_and_add_sync`, `delay`); identity ops (`device_id`, `core_id`, `subcore_id`); debug and profiling ops (`log` — `StrAttr:$tag`, `Variadic<AnyType>:$inputs`, `DefaultValuedAttr<BoolAttr, "false">:$formatted` — plus `log_buffer`, `trace`, `trace_start`, `trace_stop`, `trace_value`); hint ops (`assume_multiple`, summarized "Assumes that a value is a multiple of a given integer", `assume_layout`, `erase_layout`, `reinterpret_cast`, `get_internal_scratch`, `get_iteration_bound`, the `memref_*` family); and `weird` (`AnyType:$input` to `AnyType:$output`, verified F32 to I1). Verifier messages quoted here: "Vreg rank should be 2 or 3", "Expected matmul acc to be 32-bit", "Only packing/unpacking f32 <-> bf16 and integer <-> integer is", "Not implemented: masked load with non-32-bit element type". ([tpu_ops.td](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/tpu_ops.td), [tpu_ops.cc](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/tpu_ops.cc))

[^9]: **`c_api_decl.h` — TPU generations.** `TpuVersionEnum { kUnknownTpuVersion, kTpuV2, kTpuV3, kTpuV4, kTpuV5 }` and `TpuCoreTypeEnum`. ([Link](https://github.com/openxla/xla/blob/main/xla/tpu/c_api_decl.h))

[^10]: **Pallas Mosaic compiler params.** `jax/_src/pallas/mosaic/core.py` defines `GridDimensionSemantics` with `PARALLEL`, `CORE_PARALLEL`, `SUBCORE_PARALLEL`, `ARBITRARY`, re-exported as module-level constants and passed via `dimension_semantics`. The compiler-params docstring documents `dimension_semantics` ("Either 'parallel' for dimensions that can execute in any order, or 'arbitrary' for dimensions that must be executed sequentially"), `allow_input_fusion`, and `vmem_limit_bytes` — the latter noting it "must be used in conjunction with the `--xla_tpu_scoped_vmem_limit_kib=N` flag with `N*1kib > vmem_limit_bytes`." ([Link](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/core.py))

[^11]: **`tpu_types.td` — the parameterized 8-bit float container.** `TPU_Float8EXMYType`, "EXMY type in a 8 bit container. Meaningful bits are aligned to LSB, and bits higher than the underlying exmy type in the container are considered as ignored," parameterized by an underlying `FloatType` and citing [arXiv:2405.13938](https://arxiv.org/abs/2405.13938). ([Link](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/tpu_types.td))

[^12]: **`tpu_dialect.cc` — cross-chip communication analysis.** `mightCommunicateBetweenChips` constructs a `CommsAnalysisState`, calls `analyzeCrossChipCommunication(op, &state)`, and returns `{state.has_communication, state.has_custom_barrier}`. ([Link](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/tpu_dialect.cc))

[^13]: **`is_finite` lowers to `not weird`.** `jax/_src/pallas/mosaic/lowering.py` registers the lowering rule for `lax.is_finite_p` as `_not_lowering_rule(ctx, tpu.weird(out_type, x))`. ([Link](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/lowering.py))

[^14]: **Mosaic dialect versioning.** `jaxlib/mosaic/dialect/tpu/transforms/serde.cc` defines `kVersionAttrName = "stable_mosaic.version"` and serializes against a `kVersion` / `highest_version`. ([Link](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/transforms/serde.cc))

---

*Disclaimer: This article was generated using Claude Opus 4.8 and Claude Opus 5 models. All cited source is public, Apache-2.0 licensed code.*
