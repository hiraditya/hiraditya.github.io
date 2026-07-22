---
title: "What torch.compile Sees, and What It's Blind To"
date: 2026-07-21 12:00:00 -0700
categories: [Compilers, ML-Systems]
tags: [pytorch, torch-compile, dynamo, inductor, compilers]
---

The first time I put `torch.compile` in front of an LLM inference server, it spent the better part of a minute compiling and handed back a 6% speedup. I nearly wrote the feature off. The next morning, on a different model, it compiled in twelve seconds and ran more than twice as fast. Same GPU, same driver, same PyTorch build. The difference had nothing to do with FLOPs or memory bandwidth. It was graph breaks — the points where the compiler gave up on my Python and dumped execution back into the interpreter.

That is the whole game, and it is worth stating plainly before we go anywhere near the internals: `torch.compile` is not a compiler in the sense that `gcc` or `nvcc` is. It does not read your source and emit a self-contained binary. It watches your Python run, lifts the stretches it can handle into a graph, compiles those, and stitches the compiled regions back together with eager fallbacks for everything it could not capture. Your speedup is a function of one ratio: how much of the hot path landed in compiled regions versus how much leaked back into the interpreter.

Three components do the work.

```
[Python Bytecode]
     │
     ▼
[TorchDynamo: Graph Capture] ──(graph break)──▶ [Eager CPython]
     │
     ▼
[FX Graph (ATen ops)]
     │
     ▼
[AOTAutograd: Joint Tracing] ──▶ [Forward Graph] + [Backward Graph]
     │
     ▼
[TorchInductor: Fusion & Codegen]
     │
     ├──▶ [Triton Kernels] (GPU)
     ├──▶ [C++/OpenMP] (CPU)
     └──▶ [cuBLAS / cuDNN] (vendor libs)
```

TorchDynamo captures the graph. AOTAutograd traces the backward pass ahead of time so it can be compiled alongside the forward. TorchInductor generates code. Inductor is the default backend, but Dynamo is deliberately pluggable — pass `backend="openxla"` to target TPUs, or a vendor backend like IPEX for Intel. Most people never change it, and they shouldn't.

One detail trips up everyone at least once: nothing compiles when you call `torch.compile(model)`. Compilation is lazy. It fires on the first forward pass that reaches a given code path, and the artifacts are cached on disk. So the "why is my first step so slow" question has a boring answer, and the "why did step 900 suddenly stall" question is the interesting one. We will get to that.

## Dynamo is the part that decides whether any of this works

I will make an argument up front: Dynamo is the only piece of this stack you need to understand deeply. Everything downstream is competent and mostly invisible when it works. Dynamo is where your code either compiles or doesn't, and it tends to fail quietly — no exception, just a slow model and a log line you weren't watching.

Older approaches like TorchScript tried to solve capture by forcing you into a restricted Python subset: no data-dependent control flow, no third-party libraries, no exotic objects. It failed in production for the obvious reason. Real model code is full of exactly those things. Dynamo took the opposite bet. Instead of constraining the language, it hooks CPython's frame evaluation API — PEP 523, the extension point that lets an external tool replace the default bytecode evaluator — and intercepts frames as they execute.[^1]

At the C layer, Dynamo installs its handler with `PyInterpreterState_SetEvalFrameFunc`. When CPython is about to run a Python function, it has already built a `PyFrameObject` holding the bytecode (`f_code`), the locals (`f_locals`), and the value stack. Rather than let `_PyEval_EvalFrameDefault` run that frame, Dynamo takes it and walks the bytecode itself, one instruction at a time, through a symbolic interpreter called `InstructionTranslator`.

```
  CPython Interpreter                     TorchDynamo
┌───────────────────────┐            ┌──────────────────────────────────┐
│  PyFrameObject        │            │ InstructionTranslator            │
│  - f_code (bytecode)  │──intercept─▶│  - symbolic value stack          │
│  - f_locals           │ (PEP 523)  │  - VariableTracker mapping        │
└───────────────────────┘            │  - FX graph builder              │
                                     └──────────────────────────────────┘
```

The mental model that matters is this: as Dynamo walks the bytecode, it is running your function symbolically rather than concretely. `LOAD_FAST` and `STORE_FAST` shuffle references on a symbolic stack. When it hits a binary op whose operands are tensors, it does not multiply anything — it appends a node to an FX graph (PyTorch's IR for a DAG of operations) and keeps going. Tensors are tracked as `TensorVariable` handles carrying shape, dtype, and device. Dynamic dimensions are tracked as symbolic integers backed by `sympy` expressions, so a size can stay `s0 * 2 + 16` instead of collapsing to a concrete `1024` at trace time. That symbolic-shape machinery is the difference between a graph that generalizes across batch sizes and one that recompiles every time an input changes.

Where it gets interesting is control flow. A branch on a static Python value — a config flag, a constant — is resolved at trace time; Dynamo follows the live branch and never records the dead one. But a branch on a *runtime tensor value*, the classic `if x.sum() > 0:`, cannot be resolved symbolically. Dynamo does not know which way it goes. So it stops.

### Graph breaks, and why I have spent more time on them than anything else

When Dynamo hits something it cannot trace through — a data-dependent branch, a `print`, a call into a C extension it can't see inside — it takes a graph break. It compiles everything accumulated so far into one subgraph, falls back to the eager interpreter for the offending instruction, and tries to resume tracing on the other side.

```python
@torch.compile
def f(x):
    y = x * 2          # 1. traced into subgraph 0
    z = y + 1          # 2. still subgraph 0
    print(z.shape)     # 3. graph break: a side effect Dynamo won't capture
    w = z.relu()       # 4. traced into subgraph 1
    return w
```

Two subgraphs, one `print` between them. Each subgraph compiles and optimizes in isolation. You have lost every fusion opportunity that would have crossed the boundary, and you pay an extra handoff between compiled code and the interpreter on every call. `torch._dynamo.explain(f)(x)` will lay it out for you:

```text
Graph Break 1:
  Reason: builtin: print
  User code: print(z.shape)
Graphs generated: 2
```

One `print` in a toy function is harmless. Dozens of breaks scattered through a real forward pass is the difference between 2x and no speedup at all, and they accumulate silently as a model grows. The single most useful command I know here is:

```bash
TORCH_LOGS="graph_breaks,recompiles" python train.py
```

It prints every break and every recompilation with the bytecode that caused it. Be warned that it reports CPython bytecodes, not your source lines, so reading it fluently takes practice. The usual culprits are boring: logging and `print`, data-dependent branches, calls into C extensions Dynamo can't trace, and a few pathological `__torch_function__` overrides. Fix the boring ones first and most of your breaks disappear.

### Guards, recompilation, and the trap that burns people in production

For every compiled graph, Dynamo emits a **guard function** — a fast check, run in C, that verifies the assumptions it made while tracing still hold. Shapes, dtypes, device, the identity of the Python objects involved. You can watch them:

```bash
TORCH_LOGS="guards" python train.py
```

```text
[guards] TENSOR_MATCH: L['x'].shape == (32, 768)
[guards] TENSOR_MATCH: L['x'].dtype == torch.float32
[guards] TENSOR_MATCH: L['x'].device == device(type='cuda', index=0)
```

Guards form a tree. On each call Dynamo runs the check; if it passes, the cached graph runs and you never touch the tracer. If it fails, Dynamo throws away the fast path, re-evaluates the frame, and compiles a new specialized variant.

Here is where people get hurt. If your batch size oscillates between 32 and 64 across iterations, the shape guard fails on every other call, and Dynamo cheerfully compiles a fresh graph each time. I have seen a server spend more wall-clock time recompiling than running tensor math, with a CPU-bound profile that looks nothing like a model that is supposedly "compiled." The `recompiles` log above is how you catch it.

The fix is to stop specializing on the dimension that varies:

```python
# Tell Dynamo this dimension is dynamic; trace it as a symbol, not a constant.
torch._dynamo.mark_dynamic(x, 0)     # batch dim varies at runtime
```

`mark_dynamic` (or the blunter `dynamic=True`) makes Dynamo inject a symbolic size `s0` instead of baking in `32`. That kills the thrashing. It is not free — you give up the optimizations that lean on a known static shape, like full loop unrolling and some static allocation. As with most things in this stack, the right answer is to measure both and keep the one that wins on your shapes, not to reach for the knob reflexively.

## AOTAutograd: compiling the backward pass, not just the forward

In eager PyTorch the backward pass builds itself as the forward runs. Every op records a backward function onto an autograd tape allocated on the heap, and `loss.backward()` walks that tape. That is flexible and it is also invisible to a compiler — there is no backward graph to optimize until it's already executing.

AOTAutograd traces both passes ahead of time through `__torch_dispatch__` and produces a single joint graph, then splits it into a forward graph and a backward graph that each go to Inductor.[^2]

```
       Eager PyTorch                              AOTAutograd
┌──────────────────────────┐               ┌───────────────────────┐
│ Forward: run ops,        │               │ Trace joint fwd+bwd    │
│ build heap tape          │               │ (__torch_dispatch__)   │
└────────────┬─────────────┘               └───────────┬───────────┘
             │                                         ▼
┌────────────▼─────────────┐               ┌───────────────────────┐
│ Backward: walk tape,     │               │ Decompose +            │
│ free allocations         │               │ functionalize          │
└──────────────────────────┘               └───────────┬───────────┘
                                                       ▼
                                           ┌───────────────────────┐
                                           │ Min-cut partitioner    │
                                           └───────────┬───────────┘
                                            ┌──────────┴──────────┐
                                            ▼                     ▼
                                      Forward Graph        Backward Graph
```

Three transforms run on that joint graph, and only one of them is likely to ruin your week.

**Decomposition** rewrites high-level ops into primitives. `nn.LayerNorm` becomes `mean`, `sub`, `pow`, `sqrt`, `div`; `nn.Linear` becomes `mm` plus `add`. This shrinks the operator surface a backend has to support from thousands of ATen ops (ATen is PyTorch's C++ kernel library) down to a few hundred. It is plumbing, and it works.

**Functionalization** removes in-place mutation. Model code is full of it — `x.add_(1)`, `y[:, 0] = 3`, views that alias storage. A compiler cannot safely reorder operations that write to shared memory, so functionalization rewrites every mutation into a pure, out-of-place form, tracking aliasing through `FunctionalTensorWrapper` and version counters.[^3]

```python
# What you wrote:
x.add_(1)                              # 1. mutates x in place
y = x.view(-1)                         # 2. y aliases x's storage

# What AOTAutograd traces:
x_1 = torch.ops.aten.add.Tensor(x, 1)  # 3. pure: new tensor, no mutation
y   = torch.ops.aten.view.default(x_1, [-1])  # 4. view of the new tensor
```

This transform is also the source of the nastiest bug class in the entire stack, and it is worth being pedantic about. Functionalization trusts operator schemas. If a custom C++ op mutates a tensor in place but its schema forgets to declare that argument as `Tensor!`, functionalization never learns about the write. It doesn't bump the version counter, so the compiler is free to reorder a later read *ahead of* the write. The result is silent numerical corruption — no exception, no warning, just wrong numbers that look plausible enough to ship. I lost real time to exactly this on a custom kernel, and I wrote up the ABI and schema discipline that prevents it in [the vLLM custom-ops post]({% post_url 2026-06-22-vllm-custom-kernels-and-abi %}). If you register custom ops, get the mutation annotations right before you trust a single compiled number.

**Partitioning** is the clever one. The joint graph holds forward activations that the backward pass will need for gradients. Storing all of them costs memory; recomputing them costs FLOPs. AOTAutograd runs a min-cut algorithm (`min_cut_rematerialization_partition`) over a graph whose edges are weighted by save-cost versus recompute-cost, and cuts it where the total is cheapest.[^5] In practice, cheap pointwise ops like `relu` get thrown away and recomputed in the backward pass, while expensive `mm` outputs are saved. This is activation checkpointing that nobody had to annotate by hand, and on a memory-bound training run it is quietly one of the biggest wins in the pipeline.

## Inductor: where the memory traffic gets cut

Inductor takes the forward and backward graphs and turns them into code. Its actual job — the thing to keep in your head — is minimizing memory traffic, not maximizing arithmetic. On a modern GPU, a pointwise or reduction op is bound by how fast you can move bytes to and from HBM, and compute pipelines sit idle waiting on memory long before they saturate. Every optimization Inductor makes is in service of keeping intermediate values off HBM.

Internally it lowers FX nodes into a loop-grid IR built from `SchedulerNode`s, each describing a computation over an N-dimensional index space — pointwise nodes where output index equals input index, reduction nodes that collapse an axis (`sum`, `softmax`), and template buffers that wrap an external kernel like a cuBLAS GEMM. Then it looks at the index expressions of adjacent nodes to decide what can fuse.

Two kinds of fusion do most of the work. **Vertical fusion** merges a producer and its consumer into one loop: `mm → add → relu → dropout` becomes a single kernel, and the intermediates live in registers instead of round-tripping through HBM. **Horizontal fusion** merges independent nodes that read the same input, so the input is loaded once and shared. The first is where the speedups come from.

The payoff is easiest to see in the generated code. Take a fused bias-add, ReLU, and scale:

```python
@torch.compile
def fused_op(x, bias, scale):
    return torch.relu(x + bias) * scale
```

With `TORCH_LOGS="output_code"`, Inductor shows you the Triton it emitted. Lightly cleaned up:

```python
@triton.jit
def triton_poi_fused_add_mul_relu_0(in_ptr0, in_ptr1, in_ptr2,
                                     out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # x is [1024, 256]; bias and scale are [256], broadcast over the rows.
    xnumel = 262144                              # 1. total elements: 1024 * 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    col = xindex % 256                           # 2. column index for the broadcasts
    tmp0 = tl.load(in_ptr0 + xindex, xmask)      # 3. x: full read from HBM
    tmp1 = tl.load(in_ptr1 + col, xmask)         # 4. bias: one value reused per column
    tmp3 = tl.load(in_ptr2 + col, xmask)         # 5. scale: same
    tmp2 = tmp0 + tmp1                           # 6. add, in registers
    tmp4 = triton_helpers.maximum(0, tmp2)       # 7. relu, fused, no HBM round-trip
    tmp5 = tmp4 * tmp3                           # 8. scale, fused
    tl.store(out_ptr0 + xindex, tmp5, xmask)     # 9. single write back to HBM
```

Eager PyTorch would launch three kernels here and write two full intermediate tensors to HBM and read them back. Inductor's fused kernel reads each input once, does all three operations in registers, and writes the result once. On a bandwidth-bound shape that is most of the win, and it is why fusion — not clever math — is the lever that matters. For how Triton lowers this source down to a GPU binary, I went through the whole pipeline in [the Triton deep-dive]({% post_url 2026-07-21-triton-compiler-deep-dive %}).

Inductor does not send everything through Triton, and this is a deliberate, sensible choice rather than a limitation. Dense square GEMMs go straight to cuBLAS; convolutions go to cuDNN; those libraries are hand-tuned and Triton rarely beats them. But for the tall-and-skinny GEMMs that dominate LLM decoding, cuBLAS tile sizes are tuned for the wrong regime, and Inductor's autotuner will generate candidate Triton GEMMs, benchmark them against the cuBLAS call on your actual hardware, and keep whichever is faster. Pointwise ops, reductions, and GEMM-plus-epilogue fusions go through Triton. The dispatch is per-shape and empirical, which is the right way to make this decision.

One last pass worth naming: after kernel boundaries are fixed, Inductor plans static memory allocation. It computes lifetime intervals for every intermediate buffer, and if buffer A dies before buffer B is born, B reuses A's offset in a preallocated arena. That removes `cudaMalloc` calls from the hot path and pulls down peak VRAM — which, on a model you are trying to fit on the GPU you actually have, is the difference between running and OOM.

## FlexAttention: the whole pipeline paying off at once

Attention is where all of this stops being abstract. FlashAttention is a hand-written CUDA kernel, and for years any variation on it — a sliding window, ALiBi bias, a document-boundary mask — meant writing or forking CUDA. PyTorch 2.5's `flex_attention` lets you express the score modification in plain PyTorch and folds it into a compiled FlashAttention kernel with no graph break.[^6]

```python
from torch.nn.attention.flex_attention import flex_attention

# A causal mask written as ordinary PyTorch, not CUDA.
def causal_mask(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float("-inf"))

compiled_flex = torch.compile(flex_attention)
out = compiled_flex(query, key, value, score_mod=causal_mask)
```

Dynamo captures `causal_mask` into an FX subgraph, and Inductor injects that index arithmetic directly into the inner loop of a templated FlashAttention Triton kernel. You get a custom attention variant running at the memory-bandwidth limit of the hardware without leaving Python or touching a `.cu` file. This is the clearest argument for the whole design: capture in Python, lower to Triton, specialize per shape.

## Compilation time is the real tax, and how to isolate a failure

The cold-start cost deserves its own paragraph because it is the practical obstacle people actually hit. A large model takes 30 to 120 seconds to compile on first run, split across Dynamo's tracing, AOTAutograd's graph work, and Inductor's codegen — which itself includes Triton's JIT. For a long-lived inference server you compile once and cache and never think about it again. For an interactive development loop the cold start is a genuine cost, and the standard mitigation is regional compilation: wrap the transformer block, not the whole model, so you compile one block and reuse it.

When something goes wrong, the mistake is to read the Python traceback and try to reason about the whole stack at once. Don't. The single most valuable debugging move is to cut the pipeline in half:

```python
# Runs Dynamo capture + AOTAutograd, skips Inductor/Triton codegen entirely.
torch.compile(model, backend="eager")
```

If the bug still reproduces under `backend="eager"`, it lives in graph capture or functionalization. If it vanishes, it lives in Inductor fusion or Triton codegen. That one bisection saves more time than any amount of staring. From there the logging flags each expose one layer: `TORCH_LOGS="graph_breaks,recompiles"` for capture problems, `TORCH_LOGS="output_code"` to read the generated kernels, and `TORCH_COMPILE_DEBUG=1` to dump the IR at every intermediate pass when you need to see exactly where a graph went sideways.

## What the compiler cannot see

Everything above operates inside a boundary that is easy to miss until it costs you. The pipeline sees op types, tensor shapes, dtypes, strides, device tags, and the full dataflow graph including the backward pass. That is enough to fuse, eliminate dead code, reorder memory access, and pick kernels. It is a well-scoped, well-solved problem.

What it does not see is *where computation runs and where data lives*. A `cuda:0` tensor and a `cuda:1` tensor are indistinguishable in the IR. Host DRAM, device HBM, and pinned memory are all just tensors with a device tag; there is no type-level model of the memory hierarchy. An accidental `.cpu()` in a hot loop is a silent performance cliff, not a compile error. And NCCL collectives — `all_reduce`, `all_gather` — are opaque barriers to Inductor. It cannot fuse across them or reason about their cost.

This is intentional, and I think it is the correct call for what `torch.compile` set out to do. Compiling a single-device graph is a bounded problem with a clean cost model. Placement and routing across a cluster is a different kind of problem: the constraints are topological, the costs depend on the interconnect, and correctness cannot be read off a tensor's shape. So that layer still lives in hand-written, uncompiled Python — FSDP's sharding, DeepSpeed's pipeline scheduler, Megatron's tensor-parallel annotations — validated by running the model and watching whether the loss goes down.[^7] No compiler verifies any of it.

That gap is exactly the interesting one. `torch.compile` shows how much you can win once the problem is scoped to a single device with a static-enough graph. The open question — the one I keep circling back to — is what a compiler that reasoned about placement and memory topology as first-class, type-level concerns would look like. That is a subject for another post.

## References

[^1]: **PEP 523 — Adding a frame evaluation API to CPython.** The extension point Dynamo uses to intercept and replace bytecode evaluation. ([Link](https://peps.python.org/pep-0523/))

[^2]: **AOTAutograd: Ahead-of-Time Tracing for PyTorch.** How the forward and backward passes are traced jointly for the compiler. ([Link](https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html))

[^3]: **Functionalization in PyTorch: Everything You Wanted to Know.** Yang, E. (ezyang). How in-place mutation and aliasing are rewritten into pure form for compiler safety. ([Link](https://dev-discuss.pytorch.org/t/functionalization-in-pytorch-everything-you-wanted-to-know/965))

[^4]: **TorchInductor: A PyTorch-Native Compiler Backend.** IR lowering, fusion, and Triton codegen. ([Link](https://pytorch.org/docs/stable/torch.compiler_inductor_user_guide.html))

[^5]: **Min-Cut Rematerialization Partitioning.** Inductor's automatic activation-checkpointing strategy for cutting peak memory. ([Link](https://pytorch.org/docs/stable/torch.compiler_aot_autograd.html))

[^6]: **FlexAttention: Fused Custom Attention in PyTorch.** Expressing score modifications in Python and lowering them into a compiled FlashAttention kernel. ([Link](https://pytorch.org/blog/flex-attention/))

[^7]: **PyTorch Distributed and torch.compile.** How FSDP, DDP, and the compilation stack interact, and where the compiler's scope ends. ([Link](https://pytorch.org/docs/stable/torch.compiler_faq.html))

---

*Disclaimer: This article was generated using the Gemini 3.1 Pro and Claude Opus 4.8 models.*
