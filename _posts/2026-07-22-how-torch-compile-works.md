---
title: "How torch.compile Actually Works"
date: 2026-07-22 12:00:00 -0700
categories: [Compilers, ML-Systems]
tags: [pytorch, torch-compile, dynamo, inductor, compilers]
---

I spent three days last month debugging a silent numerical corruption in a multimodal inference pipeline. The loss would randomly explode halfway through an epoch. The culprit was a custom CUDA extension that did an in-place mutation on a tensor buffer, but the C++ schema didn't include the `Tensor!` annotation. Eager PyTorch didn't care. But when we wrapped the model in `torch.compile`, the compiler assumed the buffer was immutable, reordered a read operation past the mutation, and quietly fed garbage to our linear layers.

That bug forced me to finally dig into what `torch.compile` actually does. Most people treat it like a magical `gcc` for PyTorch. It isn't. It's a massive, brittle, brilliant hack that intercepts your Python execution at runtime, rips out the tensor operations, and tries to staple them together into a GPU binary before handing control back to CPython.

The speedup you get has almost nothing to do with the compiler's optimizations. It depends entirely on how many times the compiler gives up and dumps you back into eager Python.

If we were drawing this on a whiteboard, I'd sketch a big box for your Python bytecode at the top. Below that, a funnel called TorchDynamo that tries to catch every instruction. When Dynamo hits something it understands, it pipes it into a PyTorch-native graph. When it hits something weird—like a `print` statement or a weird C extension—it breaks the pipe, spits out the graph, and hands the weird instruction back to CPython. Everything in the graph goes through AOTAutograd to trace the backward pass, and then into TorchInductor to get squashed down into a single Triton kernel.

### The Interception Game

TorchScript failed because it asked us to write a restricted dialect of Python. Nobody wants to do that. Real training scripts are full of dynamic control flow and weird dictionary unpacks.

TorchDynamo doesn't ask you to change your code. Instead, it uses a CPython extension API (PEP 523) to hijack the interpreter. Before CPython evaluates a frame, Dynamo intercepts the `PyFrameObject` and symbolically evaluates the bytecode.

If it sees a tensor operation, it records it in an FX graph. If it sees dynamic dimensions, it spawns symbolic `sympy` variables (like `s0 * 2`) to track the math instead of hardcoding the shape. But when it hits a wall—say, branching on the value of a tensor `if x.sum() > 0:`—it triggers a graph break.

I've learned to hate graph breaks. A single `print(x.shape)` in the middle of a transformer block splits your computation into two separate graphs. Instead of one fused kernel that keeps your activations in GPU SRAM, you get two kernel launches and a round-trip to global memory (HBM). Setting `TORCH_LOGS="graph_breaks"` is the only way to find these, and deciphering the CPython bytecode output is an acquired taste.

To make matters worse, Dynamo caches these graphs behind guards. A guard is a fast C check that runs before the graph executes, verifying that the input shapes and dtypes match the assumptions made during compilation. If your dataloader spits out a batch of size 31 instead of 32, the guard fails. Dynamo halts, intercepts the frame again, and recompiles. This is why a model with variable sequence lengths will sometimes spend more time recompiling than actually training. You can force symbolic shapes with `torch._dynamo.mark_dynamic()`, but it trades away compiler optimizations.

### Tracing the Tape Ahead of Time

Standard PyTorch builds the autograd tape on the fly. Nodes are allocated on the heap as your forward pass runs.

You can't compile a dynamic heap allocation. So before code generation happens, AOTAutograd steps in and uses `__torch_dispatch__` to run the forward and backward passes ahead of time. This gives the compiler a static, joint forward-backward graph to optimize.

This is where my numerical corruption bug happened. AOTAutograd runs a functionalization pass that rewrites all in-place mutations into out-of-place functional equivalents. It does this by tracking tensor aliases. If you lie to it in your custom op schema, functionalization builds a valid-looking graph that computes complete garbage.

Once functionalized, a min-cut graph partitioner chops the joint graph into a forward and backward half. The partitioner does the math on every intermediate tensor: is it cheaper (in terms of memory footprint) to save this activation for the backward pass, or is it cheaper (in terms of compute) to just recalculate it later? It automates activation checkpointing at the IR level.

### Squashing the Graph

TorchInductor is the backend. Its entire existence is dedicated to minimizing HBM traffic. Compute is virtually free on a Hopper GPU; memory bandwidth is everything.

Inductor looks at the FX graph and aggressively fuses operations. If a linear layer's output is immediately fed into an activation function and a dropout layer, Inductor fuses them vertically. The intermediate tensors never leave the GPU registers. If two independent operations read from the same embedding matrix, Inductor fuses them horizontally so the matrix is only loaded from HBM once.

The output is usually Triton code. Inductor emits a Python file containing a `@triton.jit` kernel where all your operations are squashed into a single loop. It's surprisingly readable. You can dump it with `TORCH_LOGS="output_code"` and see exactly how it maps your tensor dimensions to Triton block grids.

It doesn't use Triton for everything, though. Dense, square matrix multiplications still go straight to cuBLAS. But for the weird, tall-and-skinny matrices we see in LLM decoding, Inductor will actually compile a custom Triton GEMM template, benchmark it against cuBLAS dynamically, and use whichever is faster.

### The Illusion of Control

The compilation pipeline is incredibly smart, but it operates with blinders on. It sees ops, shapes, and strides.

It does not see physical topology. A tensor on `cuda:0` and a tensor on `cuda:1` look exactly the same to the compiler. It has no concept of the difference between Host DRAM and Device HBM. An accidental `.cpu()` call in your training loop isn't a compiler error; it's just a silent performance cliff.

Multi-GPU primitives like NCCL `all_reduce` act as opaque walls. The compiler can't fuse across them or reason about the network cost.

This is why `torch.compile` feels incomplete when you're writing distributed code. The intra-device math is perfectly optimized, but the cross-device placement, sharding, and memory space management are still entirely manual. We're still writing FSDP wrappers and pipeline schedules by hand, hoping the loss goes down, because the compiler has no idea the network even exists.

## References

[^1]: **PEP 523 — Adding a frame evaluation API to CPython.** Official specification for intercepting interpreter frame evaluation. ([Link](https://peps.python.org/pep-0523/))

---

*Disclaimer: This article was generated using the Gemini 3.1 Pro and Claude Opus 4.8 models.*
