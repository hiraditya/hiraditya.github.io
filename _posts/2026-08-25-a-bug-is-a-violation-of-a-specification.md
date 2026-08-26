---
title: "A Bug Is a Violation of a Specification"
date: 2026-08-25 00:00:00 -0700
categories: [Systems, Inference]
tags: [inference, disaggregation, determinism, numerics, vllm, sglang, llm-serving]
mermaid: true
---

Prefix caching can cause the same prompt to produce different logits on a cache hit versus a cache miss. In many operational contexts, this is quickly categorized as a defect. However, a bug strictly implies the violation of a specification. In this domain, no such specification exists.

As established previously, a key-value (KV) cache carries the numeric signatures of the specific kernel configuration that generated it. This becomes a systemic issue when prefill and decode phases execute under differing configurations or across disparate vendor implementations. Examining the engineering effort required to strictly control this variance reveals fundamental structural properties of modern inference engines.

## The Engineering Cost of Determinism

To understand the mechanics, we can look at the active efforts to enforce determinism within major serving engines. SGLang #10278, tracking "deterministic inference with Batch Invariant Ops," illustrates the breadth of the requirement. At the time of writing, it lists 28 requirements across attention backends, quantization schemes, expert parallelism, and speculative decoding drafters. A significant portion remains unresolved.[^1]

Similarly, vLLM #34046 introduces an opt-in `--deterministic-prefix-caching` flag. This forces a cache-miss prefill to split at the last block boundary, ensuring the suffix GEMM operates with the identical `M` dimension regardless of the cache state. 

The mechanism driving the variance is straightforward:

| Request | Cache State | Tokens Computed | GEMM M Dimension |
|---|---|---|---|
| Run 1 | Miss | All N tokens in one pass | M = N |
| Run 2+ | Hit | Only uncached suffix | M = N % block_size |

GEMM backends, such as cuBLAS or Tensile, select optimal tile configurations dynamically based on the `M` dimension. Different tiling choices dictate different K-dimension reduction partitions. Because floating-point addition is non-associative, altering the accumulation order produces results that can diverge by 1 Unit in the Last Place (ULP), even when accumulating in FP32.[^2]

Crucially, this is not a synchronization issue or an atomic race condition. The underlying kernels are entirely deterministic given fixed dimensions. The variance emerges strictly from the library optimizing tile shapes for the specific problem size presented.

## Measuring Amplification

A single ULP difference at the start of a network is rarely terminal on its own, but deep transformer architectures amplify these perturbations. The vLLM pull request provides empirical validation on a standard model across intermediate layers:

| | Elements Differing | Max Difference |
|---|---|---|
| Layer 0 | 1 of 92,160 | 0.008 (1 ULP) |
| Layer 14 | ~13,500 | 0.5 |
| Layer 27 | ~14,100 | 8.0 |
| Logits | 129,895 of 151,936 | Argmax flips |

A 1 ULP deviation at layer 0 propagates into substantial divergence by layer 27, ultimately flipping the argmax decision at the logits level for a majority of the vocabulary. 

While non-zero sampling temperatures dilute the immediate impact of an argmax flip, they do not resolve the underlying reproducibility failure. Divergent logits guarantee that a run cannot be reliably replayed, regardless of the downstream sampling strategy or fixed random seeds.

## Feature vs. Defect

The engineer who identified this discrepancy originally filed it as a bug: **`[Bug][ROCm]: Prefix caching produces different output on first request (cache miss) vs subsequent requests (cache hit)`**.[^3] 

However, the same engineer subsequently implemented the fix and correctly reclassified it as a **`[Feature]`**, explicitly noting in the PR:

> All methods are identical in accuracy --- the GEMM is working correctly. The 12 differing elements between M=31 and M=15 paths are a consequence of different tile-level K-reduction ordering, **not a precision bug.**

This distinction is critical. Designating a behavior as a bug asserts a deviation from a required contract. In this case, neither vLLM nor the underlying GEMM libraries have ever guaranteed bitwise equivalence across varying matrix dimensions. Optimizing tile selection by shape is the precise mechanism by which these libraries achieve high utilization. 

The introduction of `--deterministic-prefix-caching` does not restore broken behavior; it establishes a new, stricter operational guarantee that trades scheduling flexibility for numeric consistency.

## The Determinism Hierarchy

"Determinism" in ML systems is an overloaded term. It represents a hierarchy of invariants, each requiring distinct engineering tradeoffs:

| Level | Output is invariant to | Status |
|---|---|---|
| 0 | Repeating the identical call | Largely solved (fixed kernel, fixed shape, no atomics). |
| 1 | Batch size | Active development (e.g., SGLang batch-invariant ops). |
| 2 | Cache hit vs miss | Addressed via scheduling constraints (e.g., vLLM #34046). |
| 3 | KV block size | Inherently follows from Level 2. |
| 4 | Parallelism degree (TP/DP/EP) | Partially solved (deterministic all-reduce available, but edge cases remain). |
| 5 | Quantization scheme | Unsolved; dynamic range constraints make scale granularity inherently variable. |
| 6 | Engine choice (e.g., vLLM vs SGLang) | Unattempted. |
| 7 | Vendor hardware (e.g., TPU vs GPU) | Currently inexpressible. |

When hardware vendors claim deterministic execution, they generally refer to Level 0. When inference engineers discuss nondeterminism, they are typically debugging failures at Level 1 or 2.[^4]

## Evaluating the Engineering Cost

Enforcing these invariants is generally assumed to incur steep performance penalties, but the reality is more nuanced.

**Levels 2 and 3** are inexpensive. The vLLM implementation reports negligible overhead: −0.4% at a 0% cache hit rate, and −0.02% at 95%. The only cost is a minor scheduling constraint for unaligned cache-miss prefills. Given the negligible penalty, this behavior warrants being a default rather than an opt-in flag.

**Level 1**, however, requires invasive kernel modifications. Normalization, matmul, and attention operations must adopt a unified reduction strategy regardless of the batch dimension. This intentionally bypasses shape-specialized optimizations, reliably degrading peak performance.

**Level 4** compromises collective communication efficiency. A strictly deterministic `all_reduce` is constrained from utilizing dynamic or topology-optimized reduction trees.

**Level 5** remains unaddressed because quantization scale granularity is tightly coupled to dynamic range. Enforcing reproducibility across quantization boundaries requires accepting suboptimal clipping profiles.

The general trend is clear: lower-rung determinism typically requires cheap scheduling constraints, while higher-rung determinism demands expensive arithmetic compromises.

## The Limits of Internal Implementation

Levels 0 through 5 share a defining characteristic: they can be resolved internally by a single engineering organization controlling the engine. 

Levels 6 and 7 represent a fundamental shift from technical implementation to industry coordination. Requiring two distinct engines to produce identical logits (Level 6) necessitates strict, maintained agreements on reduction order, accumulator precision, and scale granularity.

Level 7 requires this agreement across disparate hardware vendors and closed-source kernels. There is currently no specification or contract interface where such an agreement could be encoded. A KV cache tensor carries shape, dtype, and layout, but lacks metadata defining the numeric contract of its generation. Consequently, consumers of the cache cannot verify the bitwise compatibility of the incoming state.

## Defining a Numeric Contract

To achieve Level 7 reproducibility in a disaggregated architecture, a rigorous numeric specification for KV caches must be established. At a minimum, this contract must define:

- **Reduction Order:** A guarantee of shape-independent reduction ordering.
- **Accumulator Precision:** Explicit separation from storage precision (e.g., isolating FP32 accumulation into BF16 storage versus pure BF16 pipelines).
- **Tile-Selection Policy:** Strict tile-invariance guarantees.
- **Quantization Granularity:** Explicit placement and scaling logic, which is physically embedded in the cache layout.
- **Collective Algorithms:** Deterministic reduction orders for tensor-parallel environments.

These parameters are not theoretical; the generating compiler holds all of these facts internally. The deficiency is the lack of a standardized protocol to communicate them downstream.

Until such a specification is defined and adopted, cross-vendor bitwise reproducibility remains an intractable problem. When engineers correctly diagnose deep divergence cascading from a 1 ULP shift and refuse to label it a bug, they are acknowledging this reality: there is no bug because there is no specification to violate.

---

## References

[^1]: **SGLang #10278, "[Feature] Support deterministic inference with Batch Invariant Ops."** Tracking issue covering attention backends, deterministic all-reduce for tensor parallelism, radix cache support, model coverage, quantization, parallelism and speculative decoding. ([sgl-project/sglang#10278](https://github.com/sgl-project/sglang/issues/10278))

[^2]: **vLLM #34046, "[Feature][Scheduler] Add split prefix caching feature to eliminate bf16 GEMM tiling divergence across cache-hit/miss paths."** Identifies the M-dimension tiling divergence and provides empirical layer-by-layer amplification measurements. ([vllm-project/vllm#34046](https://github.com/vllm-project/vllm/pull/34046))

[^3]: **vLLM #33123, "[Bug][ROCm]: Prefix caching produces different output on first request (cache miss) vs subsequent requests (cache hit)."** The originating report for the prefix caching numeric divergence. ([vllm-project/vllm#33123](https://github.com/vllm-project/vllm/issues/33123))

[^4]: **Defeating Nondeterminism in LLM Inference.** Thinking Machines Lab. Details batch-invariance failures and the required kernel rewrites for normalization, matmul, and attention. ([Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/))

---

*Disclaimer: Researched and drafted with AI assistance (Gemini 3.1 Pro, Claude Opus 4.8). Direction, technical judgment, and final edits are mine; every claim is traceable to the sources cited above.*
