---
title: "A Bug Is a Violation of a Specification"
date: 2026-08-24 00:00:00 -0700
categories: [Systems, Inference]
tags: [inference, disaggregation, determinism, numerics, vllm, sglang, llm-serving]
mermaid: true
---

Prefix caching makes the same prompt produce different logits on a cache hit versus a cache miss. Is that a bug?

A bug is a violation of a specification. No specification exists that this behaviour violates.

[The last post]({% post_url 2026-08-24-a-cache-its-own-prefill-would-never-have-produced %}) argued that a KV cache carries the numerics of whichever kernel produced it, and that this becomes a problem when prefill and decode belong to different vendors. This numeric variation is frequently categorized as a simple defect. But examining the engineering work required to patch it reveals a structural property of inference engines.

## The work of determinism

SGLang #10278 is a tracking issue titled *"Support deterministic inference with Batch Invariant Ops"* — the work of implementing the batch-invariant kernel approach across a serving engine. 17 of 28 items are checked. The remaining 11 unchecked include FlashInfer radix cache support, prefill-with-and-without-radix-cache equivalence, linear attention models, all four quantization line items, DP attention, expert parallelism, speculative decoding drafters, and an open entry reading "Not deterministic on Blackwell for TP4." The issue is closed and labelled inactive.[^1]

vLLM #34046 adds an opt-in flag, `--deterministic-prefix-caching`, that forces cache-miss prefills to split at the last block boundary. The suffix GEMM then runs with the same M dimension regardless of cache state. It has been open since February.

Its problem statement describes the mechanism:

> With prefix caching enabled, the first request for a given prefix (cache miss) and subsequent identical requests (cache hit) take fundamentally different computation paths.

| Request | Cache State | Tokens Computed | GEMM M Dimension |
|---|---|---|---|
| Run 1 | Miss | All N tokens in one pass | M = N |
| Run 2+ | Hit | Only uncached suffix | M = N % block_size |

> The GEMM backend (Tensile on ROCm, cuBLAS on CUDA) selects tile configurations based on the M dimension. Different tiles partition the K-dimension reduction differently. Although accumulation uses fp32 internally, fp32 addition is non-associative --- different accumulation orders produce results that can differ by 1 ULP.[^2]

Atomics are absent from that account. The nondeterminism arises from a vendor BLAS library choosing tile shapes by problem size. The kernels involved are deterministic. Run the same shape twice and you get the same answer. Run a different shape that computes the same *mathematical* quantity and you do not.

## The amplification, measured

The PR contains empirical data on why 1 ULP matters, isolated on a real model across layers 0/14/27:

| | elements differing | max difference |
|---|---|---|
| Layer 0 | 1 of 92,160 | 0.008 (1 ULP) |
| Layer 14 | ~13,500 | 0.5 |
| Layer 27 | ~14,100 | 8.0 |
| Logits | 129,895 of 151,936 | argmax flips |

One element, one unit in the last place, at layer 0. By the logits, the majority of the vocabulary has moved and the greedy choice changes. This is the entire mechanism by which a rounding difference becomes a different answer.

That last row deserves a qualification, because argmax is a claim about greedy decoding. At temperature zero the flip *is* the emitted token, and a 1 ULP difference decides the output whenever it exceeds the gap between the top two logits. At higher temperatures the perturbation is diluted by the entropy of the distribution: it shifts the sampling boundaries slightly, and even with a fixed seed it changes the drawn token only when the draw lands inside that shifted window. What temperature does not fix is reproducibility. Different logits from identical inputs mean the run cannot be replayed, whatever sampling strategy sits downstream of them.

## Specifications and features

The engineer who found this discrepancy filed it as **`[Bug][ROCm]: Prefix caching produces different output on first request (cache miss) vs subsequent requests (cache hit)`**. That issue is closed and marked stale.[^3]

The same engineer then wrote the fix, and labelled it **`[Feature]`**. The PR text states:

> All methods are identical in accuracy --- the GEMM is working correctly. The 12 differing elements between M=31 and M=15 paths are a consequence of different tile-level K-reduction ordering, **not a precision bug.**

Filed as a bug, implemented as a feature. The author states in writing that nothing is broken. This is the correct position.

Call a behaviour a bug, and you assert it differs from required behaviour. This reduces to a prior question: what did anyone promise?

Nothing in vLLM promises that a cache hit and a cache miss produce identical logits. Nothing in cuBLAS promises that a GEMM with M = 31 and a GEMM with M = 15 use the same K-reduction order. Selecting tiles by shape is how the library earns its performance. Nothing in the CUDA programming model promises that mathematically equivalent computations are bitwise equivalent.

The flag in that PR offers a *new* guarantee that nobody previously made. It does not restore promised behaviour.

## Determinism is not one property

"Deterministic" names at least seven different properties. Here is the ladder, with what each rung is invariant to and where the ecosystem currently stands.

| Level | Output is invariant to | Status |
|---|---|---|
| 0 | repeating the identical call | largely solved: fixed kernel, fixed shape, no atomics |
| 1 | batch size | the batch-invariant kernel work; SGLang 17 of 28 items |
| 2 | cache hit vs miss | vLLM #34046, open since February |
| 3 | KV block size | follows from 2 |
| 4 | TP / DP / EP degree | deterministic all-reduce landed; Blackwell TP4 still failing |
| 5 | quantization scheme | all four SGLang line items unchecked |
| 6 | which engine (vLLM ≡ SGLang) | not attempted by anyone |
| 7 | which vendor (Trainium ≡ Cerebras) | not expressible |

The batch-invariance result from the previous post lives at level 1.[^4] Saying "GPUs are deterministic" usually refers to level 0. Saying "inference is nondeterministic" usually refers to level 1 or 2.

## What each rung costs

Determinism is often assumed to trade directly against performance. The data does not entirely support that.

**Level 2 is nearly free.** The vLLM PR reports zero impact on cache hits, decode steps, and block-aligned prompts. The only cost is one extra scheduling step for non-block-aligned cache-miss prefills. End-to-end, that comes to **−0.4% at a 0% cache hit rate, −0.08% at a typical 80%, and −0.02% at 95%.** At those numbers, the feature costs nothing measurable. Level 2 and its consequence level 3 are cheap, identified, and should be defaults rather than flags.

**Level 1 costs a kernel rewrite.** Normalisation, matmul, and attention all move to a single reduction strategy regardless of shape. This abandons the shape-specialisation that made them fast. SGLang's tracker includes a line item for accelerating the batch-invariant Triton kernels afterwards, indicating the first version was slow.

**Level 4 costs collective performance.** A deterministic all-reduce cannot use whatever reduction tree the topology makes fastest today.

**Level 5 is unstarted.** The reason is visible in the shape of the problem: quantization scale granularity tracks dynamic range. Fixing it for reproducibility means accepting worse clipping somewhere.

The pattern is not a smooth curve. The cheap rungs constrain *scheduling*, and the expensive rungs constrain *arithmetic*.

## Where the ladder stops being an engineering problem

Levels 0 through 5 share a property that levels 6 and 7 lack: one team owns all the code.

Implementing these invariants requires a team to change their own kernels, scheduler, and collectives, and then test that their engine agrees with itself. That is hard, multi-quarter work, but it is ordinary work with a clear owner.

Level 6 asks two engines to produce identical logits. No one has attempted it. It is a coordination problem before it is a technical one. vLLM and SGLang would have to agree on reduction order, tile policy, accumulator precision, and scale granularity, holding that agreement across releases.

Level 7 asks two vendors to do the same with unpublished kernels. Here the difficulty is not effort or willingness. **There is no artefact in which the agreement could be written down.** The KV connector carries shape, dtype, and layout. NIXL's descriptor carries an address, a length, and a device. The router prices workers in blocks. A vendor pair that wanted to promise bitwise compatibility has nowhere to put the promise, and a customer that wanted to verify it has nothing to test against.

Inside one engine, a team can decide to make the invariant part of the contract, as vLLM is doing with a flag. Across two vendors, they cannot decide, because there is no contract to amend.

## What a criterion would have to say

If an engineer wanted to specify level 7, the fields are enumerable. A numeric contract for a KV cache would have to fix, at minimum:

**Reduction order**, or an explicit statement that the producer guarantees a shape-independent one. **Accumulator precision**, separately from storage precision, because fp32-accumulate into bf16-store is a different result from bf16 throughout. **Tile-selection policy**, or a guarantee of tile-invariance. **Quantization scale granularity and placement**, which part two showed is physically embedded in the cache layout. **Collective algorithm and reduction order** where tensor parallelism is involved. And **scheduler split policy**, since vLLM #34046 demonstrates that where you chunk a prefill changes the arithmetic.

Systems would also need a way to declare which level of the ladder is claimed, discovering at configuration time that one promises level 5 and the other promises level 1.

None of this is exotic. The producing implementation already knows every item. These are facts a compiler holds about its own machine, with no channel to communicate them to anyone else's.

## The limits of internal fixes

Fixing every issue on both trackers produces an engine that agrees with itself, achieving levels 0 through 5 for one implementation. But disaggregated deployments need two implementations from two companies to agree. No amount of work inside a single repository moves that line.

When the engineer who found the divergence, diagnosed it to 1 ULP, and wrote the fix declines to call it a bug, they make an accurate observation. No specification was violated, because none exists.

Thanks to [Micah Villmow](https://www.linkedin.com/in/micah-villmow-1542534/), whose comment on the previous post argued that this was a defect in vLLM and SGLang rather than a structural property, and pointed to SGLang #10278 and vLLM #34046. The data in those two issues isolates the mechanics of numeric divergence more precisely than anything else I have found on it.

---

## References

[^1]: **SGLang #10278, "[Feature] Support deterministic inference with Batch Invariant Ops."** Tracking issue covering attention backends, deterministic all-reduce for tensor parallelism, radix cache support, model coverage, quantization, parallelism and speculative decoding. Seventeen items checked, eleven unchecked at time of writing, including all four quantization entries, DP attention, expert parallelism, speculative decoding drafters, the prefill-with-versus-without-radix-cache equivalence, and "Not deterministic on Blackwell for TP4." Closed, labelled inactive. ([sgl-project/sglang#10278](https://github.com/sgl-project/sglang/issues/10278))

[^2]: **vLLM #34046, "[Feature][Scheduler] Add split prefix caching feature to eliminate bf16 GEMM tiling divergence across cache-hit/miss paths."** Source of the cache-miss versus cache-hit M-dimension table, the statement that Tensile and cuBLAS select tile configurations by M and partition the K-dimension reduction differently, the layer-by-layer amplification measurements from 1 ULP at layer 0 to argmax flips at the logits, the assertion that "the GEMM is working correctly … not a precision bug," and the overhead figures of −0.4% at 0% cache hit rate, −0.08% at 80% and −0.02% at 95%. Opt-in behind `--deterministic-prefix-caching`; open since February 2026. ([vllm-project/vllm#34046](https://github.com/vllm-project/vllm/pull/34046))

[^3]: **vLLM #33123, "[Bug][ROCm]: Prefix caching produces different output on first request (cache miss) vs subsequent requests (cache hit)."** The originating report, filed as a bug by the same engineer who later implemented the fix as a feature. Closed, labelled stale. ([vllm-project/vllm#33123](https://github.com/vllm-project/vllm/issues/33123))

[^4]: **Defeating Nondeterminism in LLM Inference.** Thinking Machines Lab. The batch-invariance result discussed in the previous post: 1,000 identical requests at temperature zero producing 80 unique completions, first diverging at token 103, resolved by rewriting normalisation, matmul and attention to use a reduction order fixed independently of batch size. ([Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/))

---

*Disclaimer: Researched and drafted with AI assistance (Claude Opus 5). Direction, technical judgment, and final edits are mine; every claim is traceable to the sources cited above. The measurements quoted here are from the linked vLLM pull request and its author's own testing on ROCm, not mine; the SGLang item counts were read from that tracking issue on 2026-08-25 and will move as it is updated. This post exists because of a correction offered on the previous one by Micah Villmow, and the substance of that correction is his rather than mine.*
