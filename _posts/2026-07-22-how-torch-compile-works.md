---
title: "How torch.compile Actually Works"
date: 2026-07-21 12:00:00 -0700
categories: [Compilers, ML-Systems]
tags: [pytorch, torch-compile, dynamo, inductor, compilers]
---

Most PyTorch users think `torch.compile` is a compiler. It is not. Not in the way `gcc` or `nvcc` is a compiler.

It is a system that intercepts Python bytecode at runtime and extracts the parts it can handle. It compiles those parts through a pipeline of three internal tools — TorchDynamo, AOTAutograd, and TorchInductor — and falls back to eager Python for everything else.

The interesting engineering is not in the compilation. It is in the boundaries: where the compiler gives up and hands control back to Python. Those boundaries decide whether you get a 2x speedup or none at all.

## The Three Components

Before walking through the pipeline, a word on what these tools are:

- **TorchDynamo** is the graph capture frontend. It hooks into CPython's bytecode evaluator, watches your Python run, and records the PyTorch operations it sees into a graph. It is the part that decides what can be compiled and what cannot.

- **AOTAutograd** is the autograd compiler. Standard PyTorch builds the backward pass on the fly during forward execution. AOTAutograd traces both passes ahead of time, so the backward pass can be compiled too.

- **TorchInductor** is the code generator. It takes the graphs from AOTAutograd and turns them into fast code: Triton kernels for GPU, C++ for CPU, or calls to vendor libraries like cuBLAS. Inductor is the default backend, but Dynamo's design is pluggable — you can swap it for other backends like OpenXLA (for TPUs) or IPEX (for Intel GPUs) by passing `torch.compile(backend='openxla')`. In practice, most users stick with Inductor.

## The Pipeline

```
[Python Bytecode]
     │
     ▼
[TorchDynamo: Graph Capture] ──(graph break)──▶ [Eager Python]
     │
     ▼
[FX Graph (ATen ops)]
     │
     ▼
[AOTAutograd: Tracing] ──▶ [Forward Graph] + [Backward Graph]
     │
     ▼
[TorchInductor: Fusion & Code Gen]
     │
     ├──▶ [Triton Kernels] (GPU)
     ├──▶ [C++ Code] (CPU)
     └──▶ [cuBLAS / cuDNN] (vendor libs)
```

When you call `torch.compile(model)`, nothing compiles yet. Compilation happens lazily on the first forward pass. Each stage feeds the next. The final output — Triton kernels, C++ code, or cuBLAS calls — gets cached on disk so the second run skips the whole pipeline.

## Stage 1: TorchDynamo — Capturing a Graph from Live Python

Dynamo does not parse Python source. It does not need type annotations or a restricted language subset. It hooks into CPython's frame evaluation API — PEP 523 (a CPython extension point that lets external tools replace the default bytecode evaluator) — and intercepts bytecode as it runs.[^1]

What that means in practice: when Python is about to execute a frame (a function call), Dynamo steps in before CPython's evaluator runs. It walks the bytecode one instruction at a time, executing it symbolically. PyTorch operations — `torch.add`, `nn.Linear`, tensor indexing — get recorded into an FX graph (PyTorch's Python-level IR for representing a DAG of operations). Plain Python — list appends, `print` calls, C extension calls — gets traced through if possible. If not, Dynamo gives up.

"Giving up" is a **graph break**. Dynamo stops tracing, compiles what it has so far into a subgraph, and falls back to CPython for the unsupported instruction. Then it tries to resume tracing after the break.

```python
@torch.compile
def f(x):
    y = x * 2          # 1. Dynamo traces this
    z = y + 1           # 2. Dynamo traces this
    print(z.shape)      # 3. Graph break: side effect
    w = z.relu()        # 4. Dynamo starts a new subgraph
    return w
```

This function compiles into two subgraphs with a Python `print` call between them. Each subgraph gets compiled on its own. You get fusion within each region, but you lose cross-region fusion and pay two kernel-launch boundaries instead of one.

### Guards

Python is dynamic. A tensor's shape might change on the next call. A module might get swapped out. A global might flip.

So Dynamo generates a **guard function** alongside each compiled graph. It is a fast boolean check that tests the assumptions the compilation made: tensor shapes, dtypes, device, and the identity of Python objects used during tracing. On each call, Dynamo runs the guard first. If it passes, the cached code runs. If it fails, Dynamo recompiles.[^2]

Guard failures are a common problem. A model that sees different input shapes on every call recompiles every time. Compilation is slow — 10–60 seconds for a large model. This is the cold start problem and the main reason `torch.compile` sometimes makes things slower.

### Why not TorchScript?

PyTorch used to ship TorchScript (`torch.jit.script`, `torch.jit.trace`) for ahead-of-time compilation. It required rewriting code into a restricted Python subset. It could not handle data-dependent control flow, dynamic types, or most third-party libraries. Few people used it because the rewriting cost was high and the error messages were bad.

Dynamo takes the other approach: compile what you can, fall back for what you cannot. No rewriting. The tradeoff is partial compilation instead of whole-program, and you have to think about graph breaks. But the entry cost is one decorator. And the performance ceiling is higher — TorchScript could not fuse across Python control flow or trace through third-party libraries, so its compiled graphs were often small and fragmented in ways that limited the speedup.

## Stage 2: AOTAutograd — Tracing the Backward Pass

In normal PyTorch, the backward pass builds itself during the forward pass. Each forward op records a backward function onto the autograd tape. The tape runs once during `loss.backward()`.

AOTAutograd changes this.[^3] It traces both the forward and backward passes ahead of time and produces a joint graph. Then it splits that graph into a separate forward and backward graph. Each one goes to Inductor for compilation.

Three things happen in this stage:

### Decomposition

High-level PyTorch ops get broken into primitive ATen ops ("A Tensor Library" — PyTorch's C++ operator kernel library). `nn.Linear` becomes `mm` plus `add`. `nn.LayerNorm` becomes `mean`, `sub`, `pow`, `sqrt`, `div`. This cuts the number of ops Inductor needs to handle from thousands to a few hundred.

### Functionalization

PyTorch code is full of in-place mutations: `x.add_(1)`, `y[:, 0] = 3`, `buffer.copy_(new_data)`. A compiler cannot safely reorder or fuse ops that change shared state. Functionalization rewrites every mutation into a pure, out-of-place form.[^4]

```python
# Before functionalization:
x.add_(1)       # 1. mutates x in place

# After functionalization:
x_new = x + 1   # 2. pure: creates a new tensor
# 3. all later uses of x now point to x_new
```

This step is not optional. If the op schema lies about what it mutates (a missing `Tensor!` annotation), functionalization produces a wrong graph. The compiled model then silently computes wrong results. I covered this failure mode in the [vLLM custom ops post]({% post_url 2026-06-22-vllm-custom-kernels-and-abi %}).

### Partitioning

The joint forward-backward graph gets split by a min-cut partitioner (a graph algorithm that finds the cheapest set of edges to cut so that saved-activation memory is minimized). It picks which forward activations to save for the backward pass and which to throw away and recompute later. This is activation checkpointing built into the compiler. It runs without any user annotation. The goal: minimize peak memory while keeping recomputation cost reasonable.[^5]

## Stage 3: TorchInductor — Fusion and Code Generation

Inductor is the backend compiler. It takes the forward and backward FX graphs and produces runnable code.

### The fusion algorithm

Inductor's main job is cutting memory traffic. On a modern GPU, reading from and writing to HBM takes far longer than doing arithmetic. Fusing ops so that intermediate values stay in registers instead of hitting HBM is the biggest win Inductor can deliver.

Two strategies:

**Vertical fusion** chains dependent ops. If op B consumes the output of op A and nothing else reads it, they fuse into one kernel. The intermediate tensor never touches HBM. A chain like `matmul → add → relu → dropout` becomes one kernel.

**Horizontal fusion** groups independent ops that read the same data. If two ops both read tensor X, Inductor fuses them into one kernel that reads X once.

The fusion pass builds a dependency graph, groups fusible nodes, and emits one kernel per group. Output is Triton code for GPU, C++ for CPU.

### Triton vs. cuBLAS: who gets what

Not every op goes through Triton. Inductor routes by op type:

| Op pattern | Backend | Why |
|:---|:---|:---|
| Pointwise (add, mul, relu, ...) | Triton | Fusion is the win; Triton fuses these well |
| Reductions (sum, mean, softmax) | Triton | Block-level reductions map naturally |
| Dense GEMM (matmul, linear) | cuBLAS | NVIDIA tunes cuBLAS per GPU; hard to beat |
| Convolutions | cuDNN | Same story |
| Fused GEMM + epilogue (matmul + relu) | Triton template | When epilogue fusion saves enough bandwidth |

A note on GEMMs. Inductor does not always call cuBLAS for matrix multiplies. It has a template system that benchmarks cuBLAS against Triton GEMM templates for each shape and picks the faster one. For small or odd shapes (common in LLM inference), Triton sometimes wins. For large square GEMMs, cuBLAS almost always wins.

For more on how Triton turns source into a GPU binary, see the [Triton deep-dive]({% post_url 2026-07-21-triton-compiler-deep-dive %}).

### Memory planning

After fusion, Inductor looks at tensor lifetimes and reuses buffers. If tensor A dies before tensor B is born and they are the same size, B takes A's memory. This cuts peak usage, which matters when every gigabyte of HBM counts.

## Where It Breaks

### Graph breaks

Graph breaks are the most common reason `torch.compile` does not help. Each break splits the graph into smaller pieces. Smaller pieces mean less fusion and more kernel launches. Common causes:

- **`print`, logging, or any side effect.** Dynamo cannot prove these do not affect tensor math.
- **Data-dependent control flow.** `if x.sum() > 0:` branches on a runtime value. Dynamo does not know which path to trace.
- **C extensions.** Calls into C libraries that Dynamo cannot see through.
- **Unsupported Python constructs.** Some `torch.autograd` functions, `__torch_function__` overrides, or unusual Python patterns.

To debug, set `TORCH_LOGS='graph_breaks'`. The output tells you which bytecode caused the break. It talks in CPython bytecodes, not your source lines, so it takes some reading. Two other tricks: `torch.compile(backend='eager')` runs Dynamo's tracing without actually compiling, so you can isolate whether a bug is in the tracing or the codegen. And `torch.profiler` can visualize which regions of a forward pass are compiled versus eager, which makes it easier to spot where graph breaks cost you the most.

### Dynamic shapes

By default, Dynamo locks in input shapes. A model compiled with batch size 32 recompiles for batch size 64. `torch.compile(dynamic=True)` tells Dynamo to use symbolic shapes. This avoids recompilation but limits some optimizations — the compiler cannot fold shape-dependent constants.

Even with `dynamic=True`, shapes that change every call thrash the cache. `torch._dynamo.mark_dynamic()` lets you tell Dynamo which dimensions vary. But you need to know your shapes ahead of time, which is not always the case.

### Custom ops

C++ or CUDA extensions that Dynamo cannot trace cause graph breaks. If your model calls a custom kernel — PagedAttention, quantized GEMM, fused normalization — Dynamo stops at the boundary.

The fix is `torch.library`. You register the op with a schema, a meta function for shape inference, and optional decomposition rules.[^6] This makes the op visible to the compiler. I covered the registration details in the [vLLM ABI post]({% post_url 2026-06-22-vllm-custom-kernels-and-abi %}). The key: the schema must tell the truth about what gets mutated.

### Compilation time

Compilation is slow. A large LLM takes 30–120 seconds to compile on first run. The time splits roughly between Dynamo's tracing, AOTAutograd's graph work, and Inductor's codegen plus Triton's JIT.

For production inference, you compile once and cache. For training, the backward graph adds time. For development, the cold start hurts. The practical fix is regional compilation: compile just the transformer block, not the whole model.

## What torch.compile Sees and What It Does Not

The pipeline sees op types, tensor shapes, dtypes, strides, device tags, and the full dataflow graph including the backward pass. That is enough to fuse ops, kill dead code, reorder memory accesses, and pick the best kernel.

What it does not see:

- **Which physical device a tensor lives on.** `cuda:0` and `cuda:1` tensors look the same to the compiler. Whether moving data between them is correct or efficient is not its concern.
- **Memory space differences.** Host memory, device memory, pinned memory, unified memory — all just tensors with a device tag. There is no type-level wall between HBM and host DRAM. An accidental `.cpu()` is a runtime error or a silent performance hit, not a compile error.
- **Placement.** The compiler does not decide where ops run. It compiles the graph as given. If a matmul should run on device 1 for locality, that is the user's job. Or the framework's — FSDP, pipeline parallelism, and tensor parallelism all manage placement in Python.
- **Cross-device communication.** NCCL collectives work with `torch.compile` but act as opaque barriers. The compiler cannot fuse across a collective or reason about its cost.

This is a deliberate scope choice, not a missing feature. Compiling a single-device graph is a well-defined problem. Placement and routing across a cluster is a different problem. The constraints are topological, the costs depend on hardware, and correctness cannot be read off tensor shapes.[^7]

Today, that outer layer lives in uncompiled Python. FSDP's sharding logic, DeepSpeed's pipeline scheduler, Megatron's tensor-parallel annotations — all written by hand, tested by running the model, and checked by watching whether the loss goes down. No compiler verifies them.

## References

[^1]: **PEP 523 — Adding a frame evaluation API to CPython.** The Python Enhancement Proposal that lets tools intercept frame evaluation. TorchDynamo uses this to capture FX graphs from running Python. ([Link](https://peps.python.org/pep-0523/))

[^2]: **TorchDynamo: An Experiment in Dynamic Python Acceleration.** Ansel, J. et al. PyTorch team, 2022. Describes the guard system and graph break strategy. ([Link](https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html))

[^3]: **AOTAutograd: Ahead-of-Time Tracing for PyTorch.** Functorch / PyTorch team. Traces forward and backward passes ahead of time for compiler use. ([Link](https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html))

[^4]: **Functionalization in PyTorch.** Yang, E. (ezyang). How in-place mutations get rewritten into pure form for compiler safety. ([Link](https://dev-discuss.pytorch.org/t/functionalization-in-pytorch-everything-you-wanted-to-know/965))

[^5]: **Min-Cut Rematerialization Partitioning.** PyTorch Inductor's activation checkpointing that splits the joint graph to cut peak memory. ([Link](https://pytorch.org/docs/stable/torch.compiler_aot_autograd.html))

[^6]: **torch.library: Custom Operators for torch.compile.** How to register custom ops with schemas and meta functions so the compiler can see them. ([Link](https://pytorch.org/docs/stable/library.html))

[^7]: **PyTorch Distributed and torch.compile.** PyTorch team. How FSDP, DDP, and the compilation stack interact. ([Link](https://pytorch.org/docs/stable/torch.compiler_faq.html))

---

*Disclaimer: This article was generated using the Gemini 3.1 Pro and Claude Opus 4.8 models.*
