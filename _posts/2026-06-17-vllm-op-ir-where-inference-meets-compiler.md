---
title: "vLLM's op IR, or: where the inference engine meets the compiler"
date: 2026-06-17 08:00:00 -0700
categories: [Systems, Compilers]
tags: [architecture, compiler, pytorch, vllm]
mermaid: true
---

If you work on ML frameworks, compilers, or kernel performance, vLLM[^1] is worth understanding not as "the thing that serves Llama fast" but as a case study in a specific tension: **an inference engine has to be a compiler target and a hand-tuned-kernel dispatcher at the same time.** Recently vLLM grew a small op-level IR to resolve that tension explicitly. This post walks through where vLLM sits in the stack, then digs into why that IR exists and how it's built — because the design decisions in `vllm/ir` are the kind of thing you'll recognize if you've ever fought AOTAutograd's decomposition table or tried to keep a custom kernel alive through `torch.compile`[^2].

```text
            ┌─────────────────┐
    ┌──────▶│   vLLM Engine   │─────────┐
    │       └────────┬────────┘         │
    │                │                  │
    │ lowers into    │ calls            │ dispatches to
    │ (op IR)        │ (torch.compile)  │
    │                ▼                  ▼
    │       ┌─────────────────┐ ┌────────────────┐
    └───────┤     PyTorch     │ │   Hand-Tuned   │
            │   (Inductor)    │ │    Kernels     │
            └─────────────────┘ └────────────────┘
```

## The Two-Faced Inference Engine

To understand the problem, we first have to look at the landscape. At its core, vLLM is a **model-serving engine**[^3]. It does not train. It takes trained weights and runs forward passes for inference at the highest throughput and lowest cost it can manage. Its founding idea was PagedAttention[^4] — managing the KV cache the way an OS manages virtual memory, with non-contiguous pages and an indirection table — which enables continuous batching without significant memory fragmentation.

In a data center it occupies the layer between the runtime and the orchestration tier (like Ray Serve or llm-d[^5]):

```text
[Applications] [Agents] [RAG]                     ← your product
[API Gateway / Router (LiteLLM, Envoy)]           ← auth, multi-model, billing
[Cluster Orchestration (K8s, Ray Serve, llm-d)]   ← N replicas, KV-aware routing, P/D split
─────────────────────────────────────────────────────────────────────────────────────────────
>>> vLLM engine <<<                               ← scheduler, KV cache, model exec, sampling
[PyTorch] [CUDA/ROCm] [NCCL] [Triton] [FlashAttn] ← runtime
[GPUs/TPUs] [NVLink] [RDMA]                       ← hardware
```

One vLLM instance owns one model across one node's GPUs (tensor/expert parallel within a node, pipeline parallel across nodes). Everything above it — autoscaling, prefix-cache-aware routing, prefill/decode disaggregation — is orchestration around many vLLM replicas. The mental model that fits best is the application server: it's the engine that actually executes the workload efficiently, while gateways and schedulers layer around it.

The part that matters for the rest of this post: vLLM's model-execution path is migrating to be **`torch.compile`-centric**. During the autoregressive decode phase, where the engine generates one token at a time and operations are fast, memory-bound matrix-vector multiplications, eager-mode PyTorch's Python overhead and kernel launch latencies become significant bottlenecks. Capturing the model graph via `torch.compile` and CUDA graphs is mandatory to squash that overhead. However, once you commit to ahead-of-time compilation, you inherit every problem a compiler frontend has — and that's where the IR comes from.

## When the Compiler Meets Reality

This duality—acting as a compiler target while relying on opaque kernels—creates immediate friction in practice. Consider a standard operation like RMSNorm. In vLLM, it doesn't just exist as a single mathematical formula. It has to exist in at least these forms simultaneously:

- a pure-PyTorch reference (correct, slow, runs anywhere),
- one or more hand-written CUDA kernels,
- ROCm and other backend variants,
- a *fused* variant (`fused_add_rms_norm`) that folds the residual add in and writes in place,
- quantized variants.

Now layer on three requirements that pull in different directions:

1. **You must retain graph-level fusion capabilities.** Inductor's whole value is fusing pointwise/reduction chains. If you inject a completely opaque custom op[^6] into the middle of the graph, Inductor's generic code generator cannot see inside it, forcing it to route around the op and break fusion at the boundary. Therefore, you have to explicitly orchestrate fusion yourself before lowering.
2. **You must be able to swap implementations by hardware and by argument shape/dtype** — at runtime, on the hot path, cheaply.
3. **Every fast path must be checkable against the reference**, within numerical tolerance, or you'll ship a kernel that's subtly wrong at one shape out of millions.

Without a unifying abstraction, all of this lives as `if current_platform.is_cuda() and dtype == ...` branches sprinkled through model code, invisible to the compiler and impossible to test uniformly. The IR is the seam that pulls those three concerns out of the model and into one object.

## A Dialect in Disguise

To resolve this tug-of-war without compromising either side, the engine needs an intermediate representation. But the terminology here might trip up traditional compiler engineers: `vllm/ir` is **not** an AST or an LLVM/MLIR-style IR with its own blocks, regions, or SSA values. Rather, it is an **op registry built on PyTorch's `torch.library` custom-op machinery**[^7]. It acts as an "IR" only in the sense that it strictly defines the node semantics that populate the [FX graph](https://pytorch.org/docs/stable/fx.html) (PyTorch's Python-level intermediate representation for program transformations). If you think in compiler terms, it's a *dialect*: a set of named, stable, non-decomposed ops with reference semantics, plus the metadata required to lower and verify them.

Here's the registration (`vllm/ir/op.py`[^8]):

```python
lib = vllm_ir_torch_lib  # Library("vllm_ir", "FRAGMENT")
lib.define(self.name + self._schema_str)
# CompositeExplicitAutograd is not decomposed
# by ATen IR normalization in AOTAutograd
lib.impl(self.name, self._inner_call, dispatch_key="CompositeExplicitAutograd")
lib._register_fake(self.name, self._fake_call)
```

That comment is the whole point. The `CompositeExplicitAutograd` dispatch key[^9] is chosen specifically so AOTAutograd **does not decompose the op** during ATen IR normalization. The reference implementation in `vllm/ir/ops/layernorm.py`[^10] is written in plain PyTorch — `x.pow(2).mean(...)`, `torch.rsqrt(...)` — and if it were composite-decomposed, Inductor would see the constituent pointwise/reduction ops and the *identity* "this is an RMSNorm" would be gone. By registering it explicitly, the op survives into the FX graph as a single opaque node. Because Inductor cannot fuse this opaque node automatically, vLLM runs custom compilation passes[^11] prior to lowering. These passes pattern-match the high-level semantic nodes (e.g., `rms_norm` followed by `add`) and rewrite them into `fused_add_rms_norm`. This shifts **fusion to a graph rewrite over the dialect, rather than relying on the backend or hand-coding it in the model.**

So each op has a dual personality:
- **To the compiler:** one opaque, schema-typed node with a fake/meta function for shape and dtype propagation.
- **To the runtime:** a dispatcher over N implementations.

## Traffic Control on the Hot Path

Once the graph is captured and compiled, the execution drops into the runtime where every microsecond matters. Here, an op holds `impls: dict[str, IrOpImpl]`. `"native"` and `"unfused"` are reserved; everything else is a named **provider** — a CUDA kernel, a Triton kernel, a fused variant. Selection is two-tiered, and the design notes which tier is allowed to be slow:

```python
def dispatch(self, *args, **kwargs) -> "IrOpImpl":
    """This function is on the hot path (op dispatch), must be fast."""
    for impl in self._priority_impls:
        if impl.supports_args(*args, **kwargs):
            return impl
    ...
```

Two distinct predicates, deliberately separated:

- **`supported`** — a *static* property: does this platform/library even have the kernel? Evaluated once. The docs explicitly forbid putting env-var or global-state logic here.
- **`supports_args`** — a *dynamic* check against the actual tensors: dtype, shape, alignment. This runs per call.

And `supports_all_args` (when `supports_args is None`) lets `_filter_priority_impls` short-circuit: once it hits an impl that handles everything, it stops building the list and never appends the native fallback. The invariant the dispatcher enforces — the last impl in any priority list must accept all args — is exactly the "total function at the bottom of the lattice" discipline you'd want in a lowering pipeline. Priority is set per-process (`set_default`) or scoped (`set_priority` context manager), which is where the "is this kernel enabled?" policy lives — kept out of `supported`/`supports_args` on purpose.

There's also an escape hatch compiler folks will appreciate: `enable_torch_wrap`[^12]. When wrapping is off, `__call__` skips the `torch.ops` dispatch entirely and calls `_inner_call` directly. The motivation here is simple: to avoid torch dispatch overhead in eager mode, and to avoid forcing a lowering step on platforms that don't use Inductor. The abstraction is free when you opt out.

## Battle Scars: Strict Schemas and Hidden Mutations

Beyond the hot path, building a compiler integration at this scale reveals subtle edge cases. There are two specific details in the IR design that address common production issues.

**Every provider must have byte-identical schema to the reference.** `IrOpImpl.__init__` re-infers the schema from the impl function and rejects any mismatch — names, types, *and* defaults. It even validates that a `supports_args` predicate has the same parameter names and defaults as the native signature, so the dispatch hot path can forward `*args` positionally without rebinding. This is the registry refusing to let two implementations of "the same op" silently diverge in interface.

**Inplace is modeled as a separate overload, with forced functionalization on the default path.** Fused kernels want to reuse activation memory; the functional graph the compiler reasons about must not have hidden mutation. The IR splits this: `IrOpInplace` creates a `.maybe_inplace` overload whose schema is inferred with `mutates_args=op.activations`. The default overload stays functional by *cloning* activation inputs before calling an inplace impl:

```python
def func_impl_fn(self, *args, **kwargs):
    if not self.inplace:
        return self.impl_fn(*args, **kwargs)
    new_args = list(args)
    for i in self.op.activation_indices:   # clone activations
        new_args[i] = args[i].clone()
    ...
```

So callers on the functional path get correct functional semantics; the compiler, after it has done its functionalization/buffer-reuse analysis, can lower to the `maybe_inplace` overload and reclaim the memory. This is precisely the functional-vs-mutating-overload dance you'd implement in any serious lowering stack — here it's surfaced as two `torch.library` overloads of one op.

## The Unglamorous Necessities: Correctness and Caching

Finally, to make this robust enough for a production serving stack, two more pieces close the loop, addressing infrastructure requirements often omitted in prototypes.

**Reference-differential testing is built into the op.** Each op can register an input generator and per-dtype tolerances[^13]:

```python
rms_norm.override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)
```

The native PyTorch impl *is* the executable spec; every provider is checked against it at generated inputs within tolerance. The `fp16` tolerance bump exists because reductions accumulate rounding error at $32768 \times 16384$; this expected behavior is explicitly encoded in the op definition.

**Implementations carry a content hash for cache invalidation.** `IrOpImpl.uuid` hashes the source file of the impl function and feeds it into both the vLLM compile cache and the AOTAutograd/Inductor cache keys[^14]. Change a kernel's source, its uuid changes, the lowering pass uuid changes, stale compiled artifacts get invalidated. This is the unglamorous but essential part of making a compile cache correct across code changes — and it's wired directly into the op object.

## A Blueprint for Modern Serving Stacks

When you step back and look at the whole system, the design addresses a core question: "how do I keep hand-written kernels and a frontend compiler from fighting each other?"

- **Don't decompose** (explicit dispatch key) → the compiler can still pattern-match and fuse at op granularity.
- **One node, N providers** with static + dynamic support predicates → hardware and shape specialization without polluting model code or the graph.
- **Schema-locked overloads + forced functionalization** → mutation is available for performance but invisible to the functional analysis.
- **Reference impl as spec + tolerances** → every fast path is differentially testable by construction.
- **Source-hashed uuids** → the compile cache stays correct.

It's a small directory — a few hundred lines, and only `layernorm` ported so far, so it's clearly early in a larger V1/compilation migration. But it addresses a problem common to teams building a `torch.compile`-based serving stack: you need a canonical set of named ops that are simultaneously a compiler dialect, a multi-backend dispatch table, and a correctness oracle. vLLM's answer is to make that one object and let the dispatch key, the schema checker, and the uuid do the load-bearing work.

If you're building anything that has to host both a compiler and a kernel zoo, `vllm/ir` is a compact, opinionated reference for how the seam can look.

## References

[^1]: vLLM Project. *vLLM Repository*. ([Link](https://github.com/vllm-project/vllm))
[^2]: PyTorch. *`torch.compiler` / TorchInductor Overview*. ([Link](https://docs.pytorch.org/docs/stable/torch.compiler.html))
[^3]: vLLM Project. *vLLM Documentation*. ([Link](https://docs.vllm.ai))
[^4]: Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. ([Link](https://arxiv.org/abs/2309.06180))
[^5]: llm-d. *Kubernetes-native distributed serving stack built around model servers like vLLM*. ([Link](https://github.com/llm-d/llm-d))
[^6]: PyTorch. *PyTorch Custom Operators Landing Page*. (Details Python and C++/CUDA registration, fake/meta kernels). ([Link](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html))
[^7]: PyTorch. *`torch.library` API Reference*. ([Link](https://docs.pytorch.org/docs/stable/library.html))
[^8]: vLLM Source. `vllm/ir/op.py`. (Implements `register_op`, the `IrOp` / `IrOpImpl` / `IrOpInplace` classes, dispatch, schema enforcement, and the `torch.library` registration).
[^9]: PyTorch. *Registering a Dispatched Operator in C++*. (Walk-through of the PyTorch dispatcher and dispatch keys). ([Link](https://docs.pytorch.org/tutorials/advanced/dispatcher.html))
[^10]: vLLM Source. `vllm/ir/ops/layernorm.py`. (The `rms_norm` and `fused_add_rms_norm` ops; example of input generators and tolerance overrides).
[^11]: vLLM Source. `vllm/compilation/`. (The `torch.compile`-based graph capture and fusion passes that consume these ops).
[^12]: vLLM Source. `vllm/ir/__init__.py`. (Public surface including `register_op`, `enable_torch_wrap`, `set_default_torch_wrap`).
[^13]: vLLM Source. `vllm/ir/tolerances.py`. (Default per-dtype numerical tolerances).
[^14]: vLLM Source. `vllm/ir/util.py`. (`hash_source` / `weak_cache`, used for impl uuids and cache invalidation).

*Disclaimer: This article was generated using the Gemini 3.1 Pro model.*
