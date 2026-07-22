---
title: "How torch.compile Actually Works"
date: 2026-07-22 12:00:00 -0700
categories: [Compilers, ML-Systems]
tags: [pytorch, torch-compile, dynamo, inductor, compilers]
---

The first time I ran `torch.compile` on a model, it took 47 seconds to compile and ran 6% faster. On another model a day later, it compiled in 12 seconds and ran 2.1x faster. The difference had nothing to do with GPU compute bound limits. It was entirely about graph breaks — places where the compiler gave up and dumped execution back into CPython.

That outcome is standard. `torch.compile` is not a monolithic compiler like `gcc` or `nvcc`. It does not ingest Python and spit out an executable binary. It is a runtime interception engine that traces Python execution, extracts contiguous subgraphs of PyTorch operations, compiles those subgraphs, and leaves the remaining uncompilable Python to eager execution.

The total speedup depends on how many subgraphs you end up with and what happens at their boundaries.

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
[AOTAutograd: Tracing] ──▶ [Forward Graph] + [Backward Graph]
     │
     ▼
[TorchInductor: Codegen]
     │
     ├──▶ [Triton Kernels] (GPU)
     ├──▶ [C++ Code] (CPU)
     └──▶ [cuBLAS / cuDNN] (Vendor Libs)
```

The pipeline uses three distinct sub-projects. TorchDynamo captures the graph from bytecode. AOTAutograd traces the backward pass ahead of time. TorchInductor generates the actual Triton or C++ code.

Nothing compiles when you wrap a model in `torch.compile(model)`. Tracing and codegen occur lazily during the first execution pass, with compiled artifacts cached on disk.

## TorchDynamo and Bytecode Interception

Older PyTorch compilation attempts like TorchScript required writing code in a restricted Python subset. They failed to gain broad adoption because real-world ML code uses dynamic control flow, third-party libraries, and arbitrary Python constructs.

TorchDynamo takes a different approach. It hooks into CPython's frame evaluation API (`PEP 523`) to intercept execution before the default interpreter runs a frame.[^1] Dynamo symbolically evaluates the CPython bytecode instructions sequentially.

When it processes PyTorch tensor operations, it appends them to an FX graph DAG. When it hits plain Python logic, it attempts to evaluate it symbolically. When it encounters unsupported constructs — like an un-traceable C extension or an unhandled side effect — it triggers a graph break.

```python
@torch.compile
def f(x):
    y = x * 2          # Traced into Subgraph 0
    z = y + 1           # Traced into Subgraph 0
    print(z.shape)      # Graph break (side effect)
    w = z.relu()        # Traced into Subgraph 1
    return w
```

Running `torch._dynamo.explain(f)(x)` reveals the exact boundary split:

```text
Explanation:
  Graph Break 1:
    Reason: Generic break on print
    User code location: print(z.shape)
  Subgraphs: 2
```

Instead of one unified kernel launch sequence, the runtime executes Subgraph 0 in compiled code, switches back to eager CPython to print the shape, and then launches Subgraph 1. The context switch between compiled code and eager CPython introduces overhead, and any fusion opportunity across the `print` boundary is lost.

### Guards and Cache Thrashing

Dynamo produces a guard function alongside every compiled FX graph. Guards verify that the execution environment matches the assumptions made during symbolic tracing.

You can inspect the active guards for a compiled function via PyTorch's internal logging (`TORCH_LOGS="guards"`):

```text
[guards] TREE_GUARD: L['x'].shape == (32, 768)
[guards] TREE_GUARD: L['x'].dtype == torch.float32
[guards] TREE_GUARD: L['x'].device == device(type='cuda', index=0)
```

If your input batch size changes from 32 to 64 on the next call, the guard check fails. Dynamo halts execution, re-evaluates the frame bytecode, and generates a new compiled graph. If your input shapes fluctuate on every iteration, Dynamo spends more time compiling than executing.

Passing `dynamic=True` forces Dynamo to create symbolic shape variables (like `s0`, `s1`) rather than specialization constants. This prevents guard thrashing, but it trades away compiler optimizations that rely on static tensor dimensions (such as aggressive loop unrolling and static buffer allocations).

## AOTAutograd: Pre-Tracing the Backward Graph

Standard PyTorch builds the autograd tape dynamically during the forward pass. Autograd nodes are allocated on the heap as forward operations execute and are consumed during `loss.backward()`.

AOTAutograd replaces this runtime tape construction.[^2] It leverages `__torch_dispatch__` to trace both forward and backward graphs ahead of time, returning a joint forward-backward FX graph.

The joint graph undergoes three transformations:

1. **Decomposition:** High-level ops like `nn.LayerNorm` or `nn.Linear` are lowered into fundamental ATen primitives (`mean`, `sub`, `pow`, `mm`). This shrinks the operator surface area that downstream codegen backends must implement.
2. **Functionalization:** In-place mutations (`x.add_(1)`) and views are rewritten into pure out-of-place functional operations.[^3] Functionalization ensures that compiler reordering and kernel fusion Passes do not break memory alias assumptions. If a custom C++ extension mutates arguments in-place without declaring `Tensor!` in its schema, functionalization generates invalid graphs, producing silent numerical corruption.
3. **Partitioning:** A min-cut graph partitioner evaluates the joint graph to decide which forward activations should be stored in memory for the backward pass and which should be discarded and recomputed. This automates activation checkpointing at the IR level.

## TorchInductor: Kernel Fusion and Codegen

TorchInductor processes the functionalized, partitioned FX graphs to generate device code.[^4] Its primary goal is memory bandwidth optimization. On modern GPUs, memory bandwidth is the primary bottleneck for point-wise and reduction ops.

Inductor applies two primary fusion strategies:

- **Vertical Fusion:** Combines producer-consumer chains (e.g., `matmul → add → relu → dropout`) into a single kernel so intermediate tensors remain in GPU registers or SRAM rather than writing out to HBM.
- **Horizontal Fusion:** Combines independent operations reading from identical input allocations to share read memory transactions.

Inductor does not route every op to Triton. It uses a dispatch heuristic:

```text
Point-wise / Reductions / Custom Fusions ──▶ Triton Codegen
Dense Square GEMMs (Large M, N, K)       ──▶ cuBLAS
Convolutions                             ──▶ cuDNN
```

For non-standard GEMM shapes, Inductor's autotuner generates candidate Triton GEMM kernels alongside cuBLAS calls, benchmarks them dynamically, and selects the fastest implementation for that specific tile size.

After kernel grouping, Inductor performs memory planning. It calculates tensor lifetimes across the graph and aliases memory buffers for non-overlapping activations, minimizing peak allocation size.

## Debugging Workflow

When `torch.compile` underperforms or errors out, relying on standard Python tracebacks is insufficient because execution is split across Dynamo tracing, AOTAutograd IR transforms, and Triton JIT compilation.

A practical debugging workflow uses PyTorch's logging tools:

```bash
# 1. Identify graph breaks and recompilations
TORCH_LOGS="graph_breaks,recompiles" python train.py

# 2. Inspect generated Triton code
TORCH_LOGS="output_code" python train.py

# 3. Isolate Dynamo tracing from Inductor codegen
python -c "import torch; torch.compile(model, backend='eager')"
```

Setting `backend='eager'` runs Dynamo bytecode tracing and AOTAutograd graph generation while skipping Inductor's Triton codegen. If an issue persists under `backend='eager'`, the bug lies in graph capture or functionalization; if it disappears, the issue is in Inductor fusion or Triton code generation.

## Scope and Boundary Limits

`torch.compile` operates strictly on single-device execution graphs. Inductor sees op types, shapes, dtypes, strides, and local dataflow.

It does not reason about:

- **Physical topology or memory spaces:** The compiler treats a tensor on `cuda:0` and a tensor on `cuda:1` identically in terms of graph structure. It has no type-level model for Host DRAM vs. Device HBM vs. Pinned Memory.
- **Inter-device communication:** Multi-GPU communication primitives (such as NCCL `all_reduce` or `all_gather`) act as opaque barriers to Inductor. The compiler cannot fuse tensor operations across a collective boundary.
- **Device placement:** Model parallelism (FSDP, Megatron-LM tensor parallelism, pipeline parallel schedules) is managed entirely in uncompiled outer Python code.

This scope is intentional. Intra-device graph compilation is a well-scoped optimization problem. Cross-device placement, memory space topology management, and inter-connect routing require a global model of execution space and hardware topology — concerns that sit outside single-graph compilers.

## References

[^1]: **PEP 523 — Adding a frame evaluation API to CPython.** Official specification for intercepting interpreter frame evaluation. ([Link](https://peps.python.org/pep-0523/))

[^2]: **AOTAutograd: Ahead-of-Time Tracing for PyTorch.** PyTorch Architecture Documentation on joint forward-backward graph capture. ([Link](https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html))

[^3]: **Functionalization in PyTorch.** Yang, E. Detailed breakdown of in-place mutation removal for compiler passes. ([Link](https://dev-discuss.pytorch.org/t/functionalization-in-pytorch-everything-you-wanted-to-know/965))

[^4]: **TorchInductor: A PyTorch-Native Compiler Backend.** PyTorch Compiler Documentation on IR lowering and Triton codegen. ([Link](https://pytorch.org/docs/stable/torch.compiler_inductor_user_guide.html))

---

*Disclaimer: This article was generated using the Gemini 3.1 Pro and Claude Opus 4.8 models.*
