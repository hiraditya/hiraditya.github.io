---
title: "How torch.compile Actually Works"
date: 2026-07-22 12:00:00 -0700
categories: [Compilers, ML-Systems]
tags: [pytorch, torch-compile, dynamo, inductor, compilers]
---

The first time I ran `torch.compile` on a model, it took 47 seconds to compile and ran 6% faster. The second time, on a different model, it took 12 seconds and ran 2.1x faster. The difference had nothing to do with the models. It had everything to do with graph breaks — places where the compiler gave up and fell back to Python.

That experience is common, and it points at the central thing to understand about `torch.compile`: it is not a compiler in the traditional sense. It does not take your program and produce a binary. It watches your Python execute, captures the parts it can handle into a graph, compiles those parts, and stitches the compiled regions back together with eager fallbacks for everything else. The quality of the result depends almost entirely on how much of your code ends up in the compiled regions versus the fallbacks.

Three internal components do the work:

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

**TorchDynamo** captures the graph. **AOTAutograd** traces the backward pass ahead of time so it can be compiled too. **TorchInductor** generates code — Triton kernels for GPU, C++ for CPU, or calls into cuBLAS. Inductor is the default backend, but Dynamo is pluggable — you can target OpenXLA for TPUs or IPEX for Intel by passing `torch.compile(backend='openxla')`. Most people stick with Inductor.

Nothing compiles when you call `torch.compile(model)`. Compilation is lazy — it triggers on the first forward pass and caches the result on disk.

## Dynamo: The Unusual Part

Dynamo is the part worth understanding deeply, because it is where `torch.compile` either succeeds or fails for your code.

It does not parse Python source. It does not need type annotations. It hooks into CPython's frame evaluation API — PEP 523 (a CPython extension point that lets external tools replace the default bytecode evaluator) — and intercepts bytecode as your code runs.[^1]

When Python is about to execute a function, Dynamo steps in before CPython's evaluator. It walks the bytecode one instruction at a time, executing symbolically. PyTorch ops get recorded into an FX graph (PyTorch's internal IR for representing a DAG of operations). Plain Python — list appends, `print` calls, C extension calls — gets traced through when possible. When it is not possible, Dynamo hits a **graph break**: it compiles everything accumulated so far into one subgraph, falls back to CPython for the unsupported instruction, and tries to resume tracing afterward.

```python
@torch.compile
def f(x):
    y = x * 2          # 1. Dynamo traces this
    z = y + 1           # 2. Dynamo traces this
    print(z.shape)      # 3. Graph break: side effect
    w = z.relu()        # 4. Dynamo starts a new subgraph
    return w
```

Two subgraphs, one `print` in between. Each subgraph compiles and optimizes on its own. You lose cross-region fusion and pay an extra kernel-launch boundary. In a model with dozens of graph breaks, this is the difference between 2x faster and no faster.

I have spent more time hunting graph breaks than any other `torch.compile` issue. The common causes: `print` or logging, data-dependent branches (`if x.sum() > 0`), calls into C extensions Dynamo cannot see through, and exotic Python patterns like certain `__torch_function__` overrides. Setting `TORCH_LOGS='graph_breaks'` prints which bytecode caused each break. The output refers to CPython bytecodes rather than your source lines, so it takes some practice to read.

Two debugging tricks I wish I had known earlier: `torch.compile(backend='eager')` runs Dynamo's tracing without compiling, which isolates whether a problem is in the capture or the codegen. And `torch.profiler` can show which parts of a forward pass are compiled versus eager, so you can see where the breaks cost you most.

### Guards and recompilation

Dynamo generates a **guard function** alongside each compiled graph — a fast check that the assumptions from tracing still hold (tensor shapes, dtypes, device, identity of Python objects). If the guard passes, the cached code runs. If it fails, Dynamo recompiles.[^2]

This is where people get burned. A model that sees a different batch size on every call recompiles on every call. Compilation takes 10–60 seconds for a large model. `torch.compile(dynamic=True)` switches to symbolic shapes, which avoids recompilation but limits constant folding. `torch._dynamo.mark_dynamic()` lets you flag specific varying dimensions. Neither option is free.

### What happened to TorchScript?

PyTorch used to ship TorchScript for this job. It required rewriting your code into a restricted Python subset — no data-dependent control flow, no dynamic types, no third-party libraries. Few people used it because the effort was high and the error messages were unhelpful. Dynamo's approach is the opposite: compile what you can, fall back for what you cannot. The performance ceiling is also higher. TorchScript could not fuse across Python control flow, so its compiled graphs were often small and fragmented.

## AOTAutograd: Compiling the Backward Pass

In standard PyTorch, the backward pass builds itself during forward execution. Each op records a backward function on the autograd tape. The tape runs once during `loss.backward()`.

AOTAutograd traces both passes ahead of time and produces a joint graph.[^3] Then it splits that graph into separate forward and backward graphs that each go to Inductor. Three things happen here.

**Decomposition** breaks high-level ops into primitives. `nn.Linear` becomes `mm` plus `add`. `nn.LayerNorm` becomes `mean`, `sub`, `pow`, `sqrt`, `div`. This shrinks the op surface from thousands to a few hundred ATen ops ("A Tensor Library" — PyTorch's C++ operator kernel library).

**Functionalization** removes in-place mutations. PyTorch code is full of them — `x.add_(1)`, `y[:, 0] = 3`. A compiler cannot safely reorder ops that mutate shared state, so functionalization rewrites every mutation into a pure out-of-place form.[^4]

```python
# Before functionalization:
x.add_(1)       # 1. mutates x in place

# After functionalization:
x_new = x + 1   # 2. pure: creates a new tensor
# 3. all later uses of x now point to x_new
```

This step catches a class of bugs I find genuinely nasty. If a custom op's schema lies about what it mutates (a missing `Tensor!` annotation), functionalization builds a wrong graph. The compiled model silently computes wrong results — no error, no warning. I covered this in the [vLLM custom ops post]({% post_url 2026-06-22-vllm-custom-kernels-and-abi %}).

**Partitioning** splits the joint graph with a min-cut algorithm (finds the cheapest edges to cut so that saved-activation memory is minimized). It picks which forward activations to save for backward and which to recompute. This is activation checkpointing without user annotations. The goal: minimize peak memory while keeping recomputation cost sane.[^5]

## Inductor: Where Fusion Happens

Inductor takes the forward and backward FX graphs and produces code. The core job is cutting memory traffic — on a modern GPU, reading from and writing to HBM costs far more than doing arithmetic. Fusing ops so intermediate values stay in registers instead of bouncing through HBM is the main lever.

**Vertical fusion** chains dependent ops into one kernel. `matmul → add → relu → dropout` becomes a single kernel. The intermediate tensors never hit HBM.

**Horizontal fusion** groups independent ops that read the same input, so the input gets loaded once.

Inductor does not send everything through Triton. Dense GEMMs go to cuBLAS. Convolutions go to cuDNN. For GEMMs, Inductor benchmarks cuBLAS against Triton templates for each shape and picks the winner. For small or irregular shapes — common in LLM inference — Triton sometimes beats cuBLAS because cuBLAS's tile sizes are tuned for large matrices. For large square GEMMs, cuBLAS almost always wins. Pointwise ops, reductions, and fused GEMM-plus-epilogue patterns go through Triton.

For how Triton turns that source into a GPU binary, see the [Triton deep-dive]({% post_url 2026-07-21-triton-compiler-deep-dive %}).

After fusion, Inductor reuses memory buffers across tensors with non-overlapping lifetimes. If tensor A dies before tensor B is born and they are the same size, B gets A's allocation. This matters when every gigabyte of HBM counts.

## Compilation Time

This deserves its own section because it is the main practical obstacle.

A large LLM takes 30–120 seconds to compile on first run. The time splits between Dynamo's tracing, AOTAutograd's graph work, and Inductor's codegen (which includes Triton's own JIT). For production inference, you compile once and cache. For development, the cold start is real. The workaround is regional compilation: compile just the transformer block, not the whole model.

Custom C++ or CUDA ops cause graph breaks unless registered with `torch.library` (schema + meta function for shape inference + mutation annotations).[^6] I covered the details in the [vLLM ABI post]({% post_url 2026-06-22-vllm-custom-kernels-and-abi %}).

## What the Compiler Cannot See

The compilation pipeline sees op types, tensor shapes, dtypes, strides, device tags, and the full dataflow graph including the backward pass. That is enough to fuse ops, kill dead code, reorder memory access, and pick kernels.

What it does not see is where things run and where data lives. A `cuda:0` tensor and a `cuda:1` tensor look the same to the compiler. Host memory and device memory are both just tensors with a device tag. There is no type-level distinction between HBM and host DRAM. An accidental `.cpu()` in a hot path is a silent performance cliff, not a compile error.

The compiler also does not decide placement. It compiles the graph as given. If a matmul should run on device 1 for locality, that is the user's job — or the framework's. NCCL collectives work with `torch.compile` but act as opaque barriers. The compiler cannot fuse across them or reason about communication cost.

This is a deliberate scope choice. Compiling a single-device graph is a well-defined problem. Placement and routing across a cluster is different. The constraints are topological, the costs depend on hardware, and correctness cannot be read off tensor shapes.[^7]

Today, that outer layer lives in uncompiled Python. FSDP's sharding logic, DeepSpeed's pipeline scheduler, Megatron's tensor-parallel annotations — all hand-written, tested by running the model, checked by watching whether the loss goes down. No compiler verifies them.

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
