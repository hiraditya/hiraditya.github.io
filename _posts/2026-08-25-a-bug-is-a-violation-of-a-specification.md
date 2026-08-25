---
title: "A Bug Is a Violation of a Specification"
date: 2026-08-25 00:00:00 -0700
categories: [Systems, Inference]
tags: [inference, disaggregation, determinism, numerics, vllm, sglang, llm-serving]
mermaid: true
---

[The last post]({% post_url 2026-08-24-a-cache-its-own-prefill-would-never-have-produced %}) argued that a KV cache carries the numerics of whichever kernel produced it, and that this becomes a problem when prefill and decode belong to different vendors.

The sharpest pushback was that I am describing bugs. Not a structural property, just defects in two serving engines, with issues already open against them. Fix those, the argument goes, and the caches reproduce.

That deserves a real answer, because the person making it is more right than my post allowed, and the specific issues cited turn out to be the most interesting artefacts in this whole area.

## What the two issues actually say

The first is SGLang #10278. It is not a bug report; it is a tracking issue titled *"Support deterministic inference with Batch Invariant Ops"* — the work of implementing the batch-invariant kernel approach across a serving engine. Seventeen of its twenty-eight items are checked. The remaining eleven include FlashInfer radix cache support, the prefill-with-and-without-radix-cache equivalence that was quoted at me, linear attention models, all four quantization line items, DP attention, expert parallelism, speculative decoding drafters, and an open entry reading "Not deterministic on Blackwell for TP4." The issue is closed and labelled inactive.[^1]

The second is vLLM #34046, which is more interesting. It adds an opt-in flag, `--deterministic-prefix-caching`, that forces cache-miss prefills to split at the last block boundary so that the suffix GEMM always runs with the same M dimension regardless of cache state. It has been open since February.

Its problem statement is worth quoting, because it is a better description of the mechanism than mine was:

> With prefix caching enabled, the first request for a given prefix (cache miss) and subsequent identical requests (cache hit) take fundamentally different computation paths.

| Request | Cache State | Tokens Computed | GEMM M Dimension |
|---|---|---|---|
| Run 1 | Miss | All N tokens in one pass | M = N |
| Run 2+ | Hit | Only uncached suffix | M = N % block_size |

> The GEMM backend (Tensile on ROCm, cuBLAS on CUDA) selects tile configurations based on the M dimension. Different tiles partition the K-dimension reduction differently. Although accumulation uses fp32 internally, fp32 addition is non-associative --- different accumulation orders produce results that can differ by 1 ULP.[^2]

Note what is absent from that account: atomics. The nondeterminism arises from a vendor BLAS library choosing tile shapes by problem size. Every kernel involved is deterministic. Run the same shape twice and you get the same answer. Run a different shape that computes the same *mathematical* quantity and you do not.

## The amplification, measured

The same PR contains the best empirical data I have seen on why one ULP matters, isolated on a real model:

| | elements differing | max difference |
|---|---|---|
| Layer 0 | 1 of 92,160 | 0.008 (1 ULP) |
| Layer 14 | ~13,500 | 0.5 |
| Layer 27 | ~14,100 | 8.0 |
| Logits | 129,895 of 151,936 | argmax flips |

One element, one unit in the last place, at layer zero. By the logits, the majority of the vocabulary has moved and the selected token changes. That is the entire mechanism by which a rounding difference becomes a different answer, and it was measured by the person who then wrote the fix.

## The tell

Here is what makes this more than a disagreement about severity.

The engineer who found this filed it as **`[Bug][ROCm]: Prefix caching produces different output on first request (cache miss) vs subsequent requests (cache hit)`**. That issue is closed and marked stale.[^3]

The same engineer then wrote the fix, and labelled it **`[Feature]`**. And in the PR text:

> All methods are identical in accuracy --- the GEMM is working correctly. The 12 differing elements between M=31 and M=15 paths are a consequence of different tile-level K-reduction ordering, **not a precision bug.**

So: filed as a bug, fixed as a feature, and the fix's author states in writing that nothing is broken.

That is not confusion. It is the correct position, and it points at the thing I actually want to argue.

**A bug is a violation of a specification.** Call something a bug and you are asserting that observed behaviour differs from required behaviour. So the question "is prefix-cache-dependent output a bug?" reduces to a prior question: what did anyone promise?

Nothing in vLLM promises that a cache hit and a cache miss produce identical logits. Nothing in cuBLAS promises that a GEMM with M=31 and a GEMM with M=15 use the same K-reduction order — quite the opposite; selecting tiles by shape is how the library earns its performance. Nothing in the CUDA programming model promises that mathematically equivalent computations are bitwise equivalent.

Which means the flag in that PR is exactly the right shape. It is not restoring promised behaviour. It is offering a *new* guarantee that nobody previously made, at a price, opt-in.

## Determinism is not one property

The reason this argument keeps going in circles is that "deterministic" names at least seven different properties, and people asserting it usually mean different ones.

Here is the ladder, with what each rung is invariant to and where the ecosystem currently stands.

| Level | Output is invariant to | Status |
|---|---|---|
| 0 | repeating the identical call | largely solved: fixed kernel, fixed shape, no atomics |
| 1 | batch size | the batch-invariant kernel work; SGLang 17 of 28 items |
| 2 | cache hit vs miss | vLLM #34046, open since February |
| 3 | KV block size | follows from 2, per the pushback |
| 4 | TP / DP / EP degree | deterministic all-reduce landed; Blackwell TP4 still failing |
| 5 | quantization scheme | all four SGLang line items unchecked |
| 6 | which engine (vLLM ≡ SGLang) | not attempted by anyone |
| 7 | which vendor (Trainium ≡ Cerebras) | not expressible |

The batch-invariance result from the previous post lives at level 1.[^4] Someone saying "GPUs are deterministic" usually means level 0, and they are right. Someone saying "inference is nondeterministic" usually means level 1 or 2, and they are also right. The disagreement is almost never about facts.

## What each rung costs

I expected to write that determinism is bought with performance at a worsening exchange rate. The data does not entirely support that, and the correction is worth making.

**Level 2 is nearly free.** The vLLM PR reports zero impact on cache hits, decode steps, and block-aligned prompts. The only cost is one extra scheduling step for non-block-aligned cache-miss prefills, and end-to-end that comes to **−0.4% at a 0% cache hit rate, −0.08% at a typical 80%, and −0.02% at 95%.** At those numbers the argument for shipping it is simply that it is correct-by-a-useful-definition and costs nothing measurable.

So on level 2 and its consequence level 3, the pushback is right and I was wrong to lump them in with the rest. These are cheap, identified, and should be defaults rather than flags.

**Level 1 costs a kernel rewrite.** Normalisation, matmul and attention all move to a single reduction strategy regardless of shape, which is precisely giving up the shape-specialisation that made them fast. SGLang's own tracker has a line item for accelerating the batch-invariant Triton kernels afterwards, which tells you the first version was slow enough to need it.

**Level 4 costs collective performance,** since a deterministic all-reduce cannot use whatever reduction tree the topology makes fastest today.

**Level 5 is unstarted,** and the reason is visible in the shape of the problem: quantization scale granularity is chosen to track dynamic range, and fixing it for reproducibility means accepting worse clipping somewhere.

The pattern is not a smooth curve. It is that the cheap rungs are cheap because they constrain *scheduling*, and the expensive rungs are expensive because they constrain *arithmetic*.

## Where the ladder stops being an engineering problem

Levels 0 through 5 have something in common that levels 6 and 7 do not: one team owns all the code.

Every fix above is somebody changing their own kernels, their own scheduler, their own collectives, and then testing that their own engine agrees with itself. That is hard, multi-quarter work — the SGLang tracker is the evidence — but it is ordinary work with a clear owner.

Level 6 asks two engines to produce identical logits. No one has attempted it, and it is a coordination problem before it is a technical one: vLLM and SGLang would have to agree on reduction order, tile policy, accumulator precision and scale granularity, and then hold that agreement across releases.

Level 7 asks two vendors to do the same with kernels that are not published. And here the difficulty is not effort or willingness. It is that **there is no artefact in which the agreement could be written down.** The KV connector carries shape, dtype and layout. NIXL's descriptor carries an address, a length and a device. The router prices workers in blocks. A vendor pair that wanted to promise bitwise compatibility has nowhere to put the promise, and a customer that wanted to verify it has nothing to test against.

Which is where the "is it a bug?" question resolves. Inside one engine, you can at least *decide* to make the invariant part of the contract, as vLLM is doing with a flag. Across two vendors you cannot decide, because there is no contract to amend.

## What a criterion would have to say

If someone wanted to specify level 7 rather than argue about it, the fields are enumerable. A numeric contract for a KV cache would have to fix, at minimum:

**Reduction order**, or an explicit statement that the producer guarantees a shape-independent one. **Accumulator precision**, separately from storage precision, because fp32-accumulate into bf16-store is a different result from bf16 throughout. **Tile-selection policy**, or a guarantee of tile-invariance, which is the thing cuBLAS and Tensile deliberately do not offer. **Quantization scale granularity and placement**, which part two showed is already physically embedded in the cache layout. **Collective algorithm and reduction order** where tensor parallelism is involved. And **scheduler split policy**, since vLLM #34046 demonstrates that where you chunk a prefill changes the arithmetic.

Then, on top of those, a way to *declare which level of the ladder is being claimed*, so that two systems can discover at configuration time that one of them promises level 5 and the other promises level 1.

None of that is exotic. Every item is something the producing implementation already knows. It is the same observation this series keeps arriving at from different directions: these are facts a compiler holds about its own machine, with no channel to communicate them to anyone else's.

## Giving the pushback its due

The correction stands on two of the three points. Prefix-cache and block-size divergence are distinct from batch-size divergence, they are identified, and at least one of them costs so little that shipping it is close to free. Anyone reading the previous post as saying that nondeterminism is mysterious or unfixable should read it as saying less than that.

Where I would still disagree is the inference. Fixing every issue on both trackers produces an engine that agrees with itself, which is levels 0 through 5 for one implementation. The disaggregated deployments described earlier in this series need two implementations from two companies to agree, and no amount of work inside either repository moves that.

And the deepest version of the point is the one the PR author made without meaning to. When the engineer who found the divergence, diagnosed it to a ULP, and wrote the fix declines to call it a bug, that is not evasion. It is the accurate observation that no specification was violated, because none exists.

---

## References

[^1]: **SGLang #10278, "[Feature] Support deterministic inference with Batch Invariant Ops."** Tracking issue covering attention backends, deterministic all-reduce for tensor parallelism, radix cache support, model coverage, quantization, parallelism and speculative decoding. Seventeen items checked, eleven unchecked at time of writing, including all four quantization entries, DP attention, expert parallelism, speculative decoding drafters, the prefill-with-versus-without-radix-cache equivalence, and "Not deterministic on Blackwell for TP4." Closed, labelled inactive. ([sgl-project/sglang#10278](https://github.com/sgl-project/sglang/issues/10278))

[^2]: **vLLM #34046, "[Feature][Scheduler] Add split prefix caching feature to eliminate bf16 GEMM tiling divergence across cache-hit/miss paths."** Source of the cache-miss versus cache-hit M-dimension table, the statement that Tensile and cuBLAS select tile configurations by M and partition the K-dimension reduction differently, the layer-by-layer amplification measurements from 1 ULP at layer 0 to argmax flips at the logits, the assertion that "the GEMM is working correctly … not a precision bug," and the overhead figures of −0.4% at 0% cache hit rate, −0.08% at 80% and −0.02% at 95%. Opt-in behind `--deterministic-prefix-caching`; open since February 2026. ([vllm-project/vllm#34046](https://github.com/vllm-project/vllm/pull/34046))

[^3]: **vLLM #33123, "[Bug][ROCm]: Prefix caching produces different output on first request (cache miss) vs subsequent requests (cache hit)."** The originating report, filed as a bug by the same engineer who later implemented the fix as a feature. Closed, labelled stale. ([vllm-project/vllm#33123](https://github.com/vllm-project/vllm/issues/33123))

[^4]: **Defeating Nondeterminism in LLM Inference.** Thinking Machines Lab. The batch-invariance result discussed in the previous post: 1,000 identical requests at temperature zero producing 80 unique completions, first diverging at token 103, resolved by rewriting normalisation, matmul and attention to use a reduction order fixed independently of batch size. ([Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/))

---

*Disclaimer: Researched and drafted with AI assistance (Claude Opus 5). Direction, technical judgment, and final edits are mine; every claim is traceable to the sources cited above. The measurements quoted here are from the linked vLLM pull request and its author's own testing on ROCm, not mine; the SGLang item counts were read from that tracking issue on 2026-08-25 and will move as it is updated. This post exists because of a correction offered on the previous one, and the substance of that correction is the reader's rather than mine.*
