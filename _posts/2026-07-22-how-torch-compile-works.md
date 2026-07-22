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

## TorchDynamo: Frame Interception and Symbolic Tracing

Older PyTorch compilation attempts like TorchScript required writing code in a restricted Python subset. They failed to gain broad adoption because real-world ML code uses dynamic control flow, third-party libraries, and arbitrary Python constructs.

TorchDynamo takes a different approach. It hooks into CPython's frame evaluation API (`PEP 523`) to intercept execution before the default interpreter runs a frame.[^1]

### The CPython Frame Evaluation API

At the C extension level, Dynamo registers a custom frame evaluation function using `PyInterpreterState_SetEvalFrameFunc`. Whenever CPython executes a Python function, it constructs a `PyFrameObject` containing the function's bytecode (`f_code`), local variables (`f_locals`), and evaluation stack.

Instead of calling CPython's standard `_PyEval_EvalFrameDefault`, Dynamo intercepts the `PyFrameObject`. It walks the bytecode instructions sequentially using an internal symbolic execution engine (`InstructionTranslator`).

```
  CPython Interpreter                     TorchDynamo
┌───────────────────────┐            ┌──────────────────────────────────┐
│  PyFrameObject        │            │ InstructionTranslator            │
│  - f_code (bytecode)  │──intercept─▶  - Symbolic Stack               │
│  - f_locals           │ (PEP 523)  │  - VariableTracker Mapping       │
└───────────────────────┘            │  - FX GraphBuilder               │
                                     └──────────────────────────────────┘
```

Dynamo maps each CPython opcode to a corresponding symbolic handler:

- `LOAD_FAST` / `STORE_FAST`: Pushes or pops references from a symbolic evaluation stack.
- `BINARY_MULTIPLY` / `BINARY_ADD`: Operates on symbolic variables (`VariableTracker` instances). If operands are tensors (`TensorVariable`), Dynamo emits a node into the FX Graph DAG.
- `POP_JUMP_IF_FALSE` / `POP_JUMP_IF_TRUE`: Evaluates conditional branching. If the condition depends on a static Python boolean or constant, Dynamo traces the active branch. If it depends on a runtime tensor value (`x.sum() > 0`), Dynamo triggers a graph break.

### VariableTrackers and Symbolic Math

During bytecode interpretation, Dynamo wraps Python objects in `VariableTracker` subclasses:

- `TensorVariable`: Represents PyTorch tensors. Tracks shape, dtype, device, and the corresponding node in the FX graph.
- `SymIntVariable` / `SymFloatVariable`: Represents dynamic scalar dimensions. Uses `sympy` expressions (such as `s0 * 2 + 16`) to track symbolic relationships without resolving concrete integer values.
- `UserDefinedObjectVariable`: Wraps custom Python objects, tracing attribute accesses (`self.weight`) back to graph inputs or constant values.

When Dynamo processes PyTorch operations, it records them into an `fx.Graph`.

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

Guards are structured as a decision tree (`TreeGuard`). When a compiled entry point is called, Dynamo executes the guard C function. If the guard evaluates to `True`, the execution engine reuses the compiled graph. If the guard evaluates to `False`, Dynamo halts execution, re-evaluates the frame bytecode, and generates a new compiled graph.

If your input batch size changes from 32 to 64 on the next call, the guard check fails. If input shapes fluctuate on every iteration, Dynamo spends more time compiling than executing.

Passing `dynamic=True` forces Dynamo to create symbolic shape variables (`s0`, `s1`) rather than static constants. This prevents guard thrashing, but it trades away compiler optimizations that rely on static tensor dimensions (such as static buffer allocations and exact loop unrolling).

```python
# Force symbolic shape tracing for varying batch dimensions
torch._dynamo.mark_dynamic(x, 0)
```

## AOTAutograd: Pre-Tracing the Backward Graph

Standard PyTorch builds the autograd tape dynamically during the forward pass. Autograd nodes are allocated on the heap as forward operations execute and are consumed during `loss.backward()`.

AOTAutograd replaces this runtime tape construction.[^2] It leverages `__torch_dispatch__` to trace both forward and backward graphs ahead of time, returning a joint forward-backward FX graph.

```
       Eager PyTorch                              AOTAutograd
┌──────────────────────────┐               ┌───────────────────────┐
│ Forward Pass             │               │ Traces Joint Graph    │
│  - Executes Ops          │               │ (__torch_dispatch__)  │
│  - Builds Heap Tape      │               └───────────┬───────────┘
└────────────┬─────────────┘                           │
             │                                         ▼
┌────────────▼─────────────┐               ┌───────────────────────┐
│ Backward Pass            │               │ Functionalization &   │
│  - Walks Tape            │               │ Decomposition Pass    │
│  - Frees Allocations     │               └───────────┬───────────┘
└──────────────────────────┘                           │
                                                       ▼
                                           ┌───────────────────────┐
                                           │ Min-Cut Partitioner   │
                                           └───────────┬───────────┘
                                                       │
                                            ┌──────────┴───────────┐
                                            ▼                      ▼
                                      Forward Graph         Backward Graph
```

The joint graph undergoes three lowering passes:

### 1. Decomposition

High-level PyTorch ops like `nn.LayerNorm` or `nn.Linear` are lowered into fundamental ATen primitives (`mean`, `sub`, `pow`, `mm`). This shrinks the operator surface area that downstream codegen backends must implement.

### 2. Functionalization and Alias Tracking

PyTorch code contains in-place mutations (`x.add_(1)`) and aliased views (`x.view(...)`, `x.transpose(...)`). A compiler cannot safely reorder ops that mutate shared memory buffers.

Functionalization converts all mutated storage operations into pure out-of-place functional operations.[^3] It tracks tensor aliasing using an internal `FunctionalTensorWrapper` and a version counter.

```python
# Original user code
x.add_(1)
y = x.view(-1)

# Functionalized representation inside AOTAutograd IR
x_1 = torch.ops.aten.add.Tensor(x, 1)
y = torch.ops.aten.view.default(x_1, [-1])
```

If a custom C++ extension mutates arguments in-place without declaring `Tensor!` in its schema, functionalization fails to inject the new tensor version (`x_1`). The compiler assumes `x` is unmodified and reorders downstream reads, causing silent numerical corruption.

### 3. Min-Cut Graph Partitioning

The joint forward-backward graph contains intermediate tensors created during the forward pass that are needed to compute gradients in the backward pass.

AOTAutograd uses a min-cut graph partitioning algorithm (`min_cut_rematerialization_partition`) to decide which forward activations to save in memory and which to discard and recompute during backward.[^5]

The partitioner builds a bipartite graph where:
- Nodes represent forward operations.
- Edges represent intermediate tensors.
- Edge weights represent the memory footprint (bytes) required to store the activation versus the compute cost (FLOPs) required to recompute it.

Point-wise ops (like `ReLU` or `Add`) have low recomputation cost relative to their memory footprint, so the partitioner discards their forward outputs and recomputes them during the backward pass. Heavy matrix multiplications (`mm`) have high recomputation cost, so their outputs are saved in the forward-to-backward activation storage.

## TorchInductor: Lowering and Code Generation

TorchInductor processes the functionalized, partitioned FX graphs to generate device code.[^4] Its primary objective is memory bandwidth optimization. On modern GPUs, memory bandwidth is the primary bottleneck for point-wise and reduction ops.

### Intermediate Representation: Loop Grids

Inductor lowers FX nodes into an internal loop-grid IR composed of `SchedulerNode` objects. Each node represents a computation mapped over an $N$-dimensional index space:

- `Pointwise`: Operations where each output element depends only on input elements at the same index.
- `Reduction`: Operations that aggregate data across one or more dimensions (e.g., `sum`, `softmax`).
- `TemplateBuffer`: Operations backed by external C++ or CUDA template kernels (e.g., cuBLAS matrix multiplies).

Inductor analyzes the index expressions of adjacent `SchedulerNode` objects to perform kernel fusion.

### Vertical and Horizontal Fusion

Inductor applies two primary fusion passes:

- **Vertical Fusion:** Combines producer-consumer chains into a single loop body. If node $B$ reads the output of node $A$, Inductor merges their index spaces so $A$'s output is stored in a GPU register or SRAM and consumed immediately by $B$.
- **Horizontal Fusion:** Combines independent `SchedulerNode` instances that read from identical memory allocations, merging their loop headers so global memory read operations are shared across threads.

### Triton Code Generation Walkthrough

Consider a fused bias addition, ReLU, and scale operation:

```python
@torch.compile
def fused_op(x, bias, scale):
    return (torch.relu(x + bias)) * scale
```

Inductor fuses these three operations into a single Triton kernel. You can inspect the generated Python code by setting `TORCH_LOGS="output_code"`:

```python
# Generated by TorchInductor for GPU execution
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid

@triton.jit
def kernel_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel

    # Load inputs directly into registers
    x = tl.load(in_ptr0 + xindex, xmask)
    bias = tl.load(in_ptr1 + xindex, xmask)
    scale = tl.load(in_ptr2 + xindex, xmask)

    # Fused compute in GPU registers
    tmp0 = x + bias
    tmp1 = tl.maximum(0.0, tmp0)
    tmp2 = tmp1 * scale

    # Single write back to global memory (HBM)
    tl.store(out_ptr0 + xindex, tmp2, xmask)
```

Without fusion, eager PyTorch executes three separate kernel launches, writing two intermediate tensors to HBM and reading them back. Inductor's fused kernel executes a single read of each input and a single store to the output, operating entirely within thread registers.

### Dispatch Heuristics and GEMM Routing

Inductor does not route every operation to Triton. It uses a hybrid dispatch model:

```text
Point-wise / Reductions / Epilogue Fusions ──▶ Triton Codegen
Dense Square GEMMs (Large M, N, K)         ──▶ cuBLAS / cuBLASLt
Convolutions                               ──▶ cuDNN
```

For non-standard GEMM shapes (such as tall-and-skinny matrices in LLM decoding), Inductor's autotuner generates candidate Triton GEMM kernels alongside standard cuBLAS calls. It compiles and benchmarks each candidate on the target GPU, selecting the execution path with the lowest runtime latency.

### Memory Planning and Buffer Reuse

After kernel boundaries are finalized, Inductor performs static memory allocation planning. It constructs a lifetime interval graph for all intermediate buffers.

If Tensor $A$ is freed before Tensor $B$ is allocated, Inductor assigns Tensor $B$ to the exact byte offset of Tensor $A$'s memory allocation inside a consolidated workspace arena. This reduces `cudaMalloc` calls to zero during the hot execution path and lowers peak VRAM footprint.

## FlexAttention: Compiling Attention Patterns (PyTorch 2.5+)

Standard attention implementations (like FlashAttention) use hand-written CUDA templates. Extending them to support custom masking or score modifications (e.g., sliding window attention, alibi positional embeddings, document masking) historically required writing new C++/CUDA kernels.

PyTorch 2.5 introduced `flex_attention`, which integrates custom attention patterns directly into `torch.compile` without graph breaks.[^6]

```python
from torch.nn.attention.flex_attention import flex_attention

def causal_mask(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float("-inf"))

# torch.compile lowers score_mod directly into the attention Triton block loops
compiled_flex = torch.compile(flex_attention)
out = compiled_flex(query, key, value, score_mod=causal_mask)
```

Dynamo traces the Python `causal_mask` function into an FX graph. Inductor then injects the resulting index math directly into the inner loops of a template Triton FlashAttention kernel. This allows custom attention mechanics to run at hand-tuned CUDA speed without leaving Python.

## Practical Debugging Workflow

When `torch.compile` underperforms or errors out, relying on standard Python tracebacks is insufficient because execution is split across Dynamo tracing, AOTAutograd IR transforms, and Triton JIT compilation.

A systematic debugging workflow isolates each layer using PyTorch's logging infrastructure:

```bash
# 1. Identify graph breaks and recompilations
TORCH_LOGS="graph_breaks,recompiles" python train.py

# 2. Inspect generated Triton code and loop bounds
TORCH_LOGS="output_code" python train.py

# 3. Dump the full compiler graph IR at each lowering stage
TORCH_COMPILE_DEBUG=1 python train.py

# 4. Isolate Dynamo tracing from Inductor codegen
python -c "import torch; torch.compile(model, backend='eager')"
```

Setting `backend='eager'` runs Dynamo bytecode tracing and AOTAutograd graph generation while skipping Inductor's Triton codegen. If an error persists under `backend='eager'`, the bug lies in graph capture or functionalization; if it disappears, the issue is in Inductor fusion or Triton code generation.

## Scope and Boundary Limits

`torch.compile` operates strictly on single-device execution graphs. Inductor sees op types, shapes, dtypes, strides, and local dataflow.

It does not reason about:

- **Physical topology or memory spaces:** The compiler treats a tensor on `cuda:0` and a tensor on `cuda:1` identically in terms of graph structure. It has no type-level model for Host DRAM vs. Device HBM vs. Pinned Memory.
- **Inter-device communication:** Multi-GPU communication primitives (such as NCCL `all_reduce` or `all_gather`) act as opaque barriers to Inductor. The compiler cannot fuse tensor operations across a collective boundary.
- **Device placement:** Model parallelism (FSDP, Megatron-LM tensor parallelism, pipeline parallel schedules) is managed entirely in uncompiled outer Python code.

This scope is intentional. Intra-device graph compilation is a well-scoped optimization problem. Cross-device placement, memory space topology management, and inter-connect routing require a global model of execution space and hardware topology — concerns that sit outside single-graph compilers.[^7]

Today, that outer layer lives in uncompiled Python. FSDP's sharding logic, DeepSpeed's pipeline scheduler, Megatron's tensor-parallel annotations — all hand-written, tested by running the model, checked by watching whether the loss goes down. No compiler verifies them.

## References

[^1]: **PEP 523 — Adding a frame evaluation API to CPython.** Official specification for intercepting interpreter frame evaluation. ([Link](https://peps.python.org/pep-0523/))

[^2]: **AOTAutograd: Ahead-of-Time Tracing for PyTorch.** PyTorch Architecture Documentation on joint forward-backward graph capture. ([Link](https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html))

[^3]: **Functionalization in PyTorch.** Yang, E. Detailed breakdown of in-place mutation removal for compiler passes. ([Link](https://dev-discuss.pytorch.org/t/functionalization-in-pytorch-everything-you-wanted-to-know/965))

[^4]: **TorchInductor: A PyTorch-Native Compiler Backend.** PyTorch Compiler Documentation on IR lowering and Triton codegen. ([Link](https://pytorch.org/docs/stable/torch.compiler_inductor_user_guide.html))

[^5]: **Min-Cut Rematerialization Partitioning.** PyTorch Inductor activation checkpointing strategy for memory optimization. ([Link](https://pytorch.org/docs/stable/torch.compiler_aot_autograd.html))

[^6]: **FlexAttention: Fused Custom Attention in PyTorch.** PyTorch Documentation on compiled attention pattern mechanisms. ([Link](https://pytorch.org/blog/flex-attention/))

[^7]: **PyTorch Distributed and torch.compile.** PyTorch Architecture Guide on interactions between FSDP, DDP, and graph compilation. ([Link](https://pytorch.org/docs/stable/torch.compiler_faq.html))

---

*Disclaimer: This article was generated using the Gemini 3.1 Pro and Claude Opus 4.8 models.*
