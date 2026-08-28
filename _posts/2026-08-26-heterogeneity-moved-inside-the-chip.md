---
title: "Heterogeneity Moved Inside the Chip"
date: 2026-08-26 06:00:00 -0700
categories: [Systems, Inference]
tags: [inference, disaggregation, accelerators, speculative-decoding, hardware, compilers, llm-serving]
mermaid: true
---

At Hot Chips this week, OpenAI presented Jalapeño, an inference ASIC co-designed with Broadcom.[^1] The architecture slide circulating since the presentation frames a request pipeline split three ways: prefill, a draft model, and speculative verification.[^2] While the traditional split is just prefill and decode, replacing standard decode with a tight draft-and-verify speculation loop leads to a re-evaluation of hardware bottlenecks.

It is worth inspecting the design closely. It rejects the industry trend of physical disaggregation entirely, opting instead for a unified silicon architecture that dynamically reallocates its own internal bottlenecks.

## The Three-Phase Pipeline

A single request traverses three distinct operational regimes, each saturating a completely different dimension of the hardware architecture.

**Prefill** encodes the context. It is heavily compute-bound, demanding maximum arithmetic density for attention and projection operations. Memory bandwidth requirements are relatively low, and interconnect communication is predictable and easy to schedule.

**The Draft Model** runs speculative generations. This model is deliberately small, operating at a batch size close to one. It moves negligible data but is hyper-sensitive to network latency. It is entirely latency-bound. Every microsecond spent moving a tensor or waiting on an interconnect directly degrades the token generation rate.

**Speculative Verification** replaces the decode phase. It is fundamentally bottlenecked by HBM bandwidth to load the primary model weights for attention calculations. For Mixture-of-Experts (MoE) topologies, this phase also demands extreme, burst-tolerant interconnect bandwidth for expert routing.

A textbook systems engineering response to three distinct bottleneck profiles is physical specialization. You design a compute-heavy prefill ASIC, a latency-optimized SRAM-heavy draft ASIC, and a bandwidth-optimized verify ASIC. You then route the request over a fabric through all three. 

OpenAI put that exact topology on a slide and explicitly rejected it. 

Their conclusion was simple: heterogeneity must move inside the chip, not across a network. Their design philosophy dictates keeping the KV cache local and dynamically activating the necessary silicon blocks as the phase changes.

## The Cost of the Boundary

The decision to reject physical disaggregation rests on three harsh realities of production serving. 

First, the ratio of work across these three phases is never fixed. The distribution shifts wildly based on the model architecture, token efficiency, dynamic context lengths, attention algorithms, speculative acceptance rates, and the required latency-versus-throughput service level agreement (SLA). If you build a fleet comprising a fixed ratio of specialized prefill and decode servers, your cluster is optimized for exactly one workload mix. The moment your workload drifts—say, users start submitting 100K-token prompts instead of 4K—entire racks of specialized decode silicon sit idle, gated by the overwhelmed prefill tier. 

Second, the KV cache is enormous. Prefill generates a KV cache that the decode phase requires immediately. Disaggregating the phases means this cache must cross a network boundary on every request. As I covered previously, the network transport cost for a large KV cache destroys the latency budget of the request. The cache must stay local to the compute elements that will consume it.

Third, speculative decoding is a tight loop, not a linear pipeline. The draft model proposes a handful of tokens. The verify model accepts a prefix and rejects the rest. The draft model resumes from the last accepted token. This round-trip occurs constantly. Inserting a network hop between the draft and verify stages means paying fabric latency on every single speculation cycle. The entire mechanism of speculative decoding is designed to buy latency; spending that latency on a network boundary defeats the architecture.

By consolidating the phases onto a single die, OpenAI avoided these penalties. One chip, one memory hierarchy, one scheduler, and one set of unified kernels. The question of how to efficiently cross vendor lines or network boundaries disappears when you simply refuse to build the boundary.

## Dynamic Activation and Dark Silicon

OpenAI justified this unified architecture with a stark economic claim: dark silicon is cheaper than idle accelerators. 

A dedicated accelerator incurs capital and operational costs for packaging, HBM, I/O transceivers, network fabric, and cooling, regardless of its utilization. A unified chip can selectively power-gate its unused arithmetic or memory blocks while keeping the KV cache resident and the interconnect active.

Consolidation settles the physical location of the computation, but it leaves the scheduling problem completely unresolved. 

"Activate the right ratio by phase; unused units go dark" is inherently a runtime decision. During prefill, the chip must activate its dense matrix-multiply blocks and gate its memory controllers. During verification, it must reverse this, saturating memory bandwidth while activating burst-tolerant network links for MoE routing. The draft phase requires gating almost everything except a low-latency path to the verification units. 

The hardware must reconfigure its resource mix hundreds of times per second. The inputs driving this reconfiguration—context length, instantaneous speculative acceptance rate, multi-tenant QoS targets—are strictly dynamic. They cannot be known at compile time. 

At a network boundary, this is a protocol negotiation problem. On a unified die, it becomes a compiler and runtime scheduling problem. This is a superior domain for the problem to exist. A compiler has deep visibility into the phase structure of the computation, and a unified runtime has solitary ownership over the hardware execution.

## The Missing Compiler Primitives

Yet, we currently lack the programming model to express these hardware demands. A modern compiler intermediate representation (IR)—whether it is MLIR's `linalg` dialect, Triton IR, or PTX—is designed to express tensor shapes, memory layouts, and data flow. It completely lacks the vocabulary to express temporal traffic patterns or phase-level hardware transitions. 

An IR does not declare that a specific sequence of operations will generate 800 GB/s of all-to-all traffic in 5-microsecond bursts. It cannot express that a computation is compute-heavy while its immediate consumer is strictly memory-bandwidth bound. 

Without these annotations, the hardware runtime cannot schedule efficiently. To power-gate a silicon block, the runtime must preemptively wake it up *before* the computation arrives. Waking dormant systolic arrays or spinning up SERDES links takes measurable time. If the runtime is strictly reactive, the latency cost of waking the dark silicon destroys the latency budget you gained by avoiding the network boundary in the first place. 

If a compiler cannot name a computational phase, the runtime cannot preemptively allocate the hardware. We are treating a dynamic, multi-modal hardware reconfiguration problem with static, single-mode compiler tools.

## The Capacity Wall

The assumption underlying the "keep KV local" strategy is that the hardware possesses sufficient capacity to actually hold it. Capacity is strictly a function of context length, and this is where the unified architecture encounters severe friction.

A Jalapeño package pairs a compute die with six HBM4 stacks, delivering 216 GiB of capacity at 15.4 TB/s bandwidth within a 700 W envelope. A full 128-chip rack holds 27.5 TB.[^5] 

Whether 216 GiB is expansive or suffocating depends entirely on the KV layout and the sequence length. 

Consider a standard Grouped-Query Attention (GQA) layout (e.g., 80 layers, 8 KV heads, 128-wide head dimension) at FP8 precision. A 1M-token sequence requires roughly 153 GiB of KV cache. A single session consumes over 70% of a Jalapeño package's total HBM capacity. If that cache is stored in BF16, it requires 305 GiB, which physically exceeds the package limits. 

Conversely, a heavily compressed latent design, like Multi-Head Latent Attention (MLA), requires roughly 656 bytes per token. A 1M-token session consumes just 0.6 GiB, allowing a single package to host over 350 concurrent sessions.

While OpenAI does not publish its exact serving configurations, the capacity delta between GQA and MLA dictates the viability of the architecture. A compressed layout easily sustains local KV retention. A conventional layout at 1M tokens destroys it.

When context length scales, batch size collapses. Decode throughput is entirely dependent on batching multiple sequences to amortize the memory bandwidth cost of loading the model weights. If a chip can only hold one 1M-token session, batch size drops to one. The decode phase collapses into a purely memory-bound GEMV operation. 

Furthermore, agentic workloads introduce severe temporal pressures. Agents spend significant time idling—waiting on API calls, tool execution, or human input. During these gaps, the session's massive KV cache remains resident in HBM, the most expensive storage medium in the entire cluster. 

The traditional solution to agentic idling is to page the KV cache down the memory hierarchy to host DRAM or a pooled flash tier. But moving data off-package directly contradicts the core architectural premise of keeping the KV cache local.

The dark-silicon argument also falters under persistent agentic workloads. Power-gating unused blocks saves operational expenditure, but it does not reclaim die area. An agentic workload heavily skews toward long prefill and short decode. This leaves the memory and network blocks dark on silicon that required massive capital expenditure to fabricate. Power-gating is a band-aid for operational costs; it does not solve the capital inefficiency of deploying unified silicon for heavily skewed workloads.

## The NVIDIA's Counter-Bet: Physical Disaggregation

The unified architecture is a compelling engineering argument, but it is contested by the rest of the industry. 

NVIDIA recently removed Rubin CPX—a part specifically optimized for compute-bound prefill using GDDR7—from its roadmap at GTC 2026. The production slot was instead allocated to a 256-chip, SRAM-based Groq 3 LPX rack, acquired through a massive licensing arrangement.[^6] 

This is a structural rejection of the unified die approach. By pulling Groq's SRAM-based architecture into the roadmap to serve the latency-bound draft model phase, NVIDIA is doubling down on physical disaggregation. They are asserting that a unified HBM pool cannot simultaneously serve the deterministic, ultra-low latency demands of a batch-size-of-1 draft model *and* the massive capacity/throughput demands of batched prefill and verification without extreme, unacceptable compromise. 

SRAM provides entirely deterministic access times with massive internal bandwidth, perfectly matching the draft model's profile. HBM provides the capacity required for KV caches and model weights, but its access latency is fundamentally too high for optimal speculative generation. NVIDIA’s bet is that the physical limitations of memory hierarchy physics—SRAM vs. HBM—dictate that specialized silicon pools connected by an ultra-fast fabric will mathematically outperform a unified die that tries to compromise between the two.

The most sophisticated engineering organizations in the industry are actively funding directly opposed architectural philosophies. One is betting that the latency penalty of crossing a network boundary is fatal to speculative decoding, meaning heterogeneity must move on-die. The other is betting that workload ratios will skew so heavily, and memory access profiles are so distinct, that maintaining specialized, disaggregated silicon pools is the only mathematically viable path to cluster efficiency.

Both bets rely on the exact same underlying premise: prefill, draft, and verification require fundamentally different hardware. Jalapeño is a forceful argument that they require different *configurations* of the same computer, reallocated dynamically while the request is in flight.

Procuring different computers for different phases is a solved supply-chain problem. Designing a runtime and compiler stack capable of dynamically reallocating compute, memory, and interconnect across a unified die based on unpredictable runtime metrics is a language problem. The hardware has arrived; the software stack required to actually exploit it has not.

---

## References

[^1]: **OpenAI Jalapeño custom AI ASIC, Hot Chips 2026.** Presented on day two of the conference, 25 August 2026, by Richard Ho, Ravi Narayanaswami and Chris Leary; the chip is built in collaboration with Broadcom. ([ServeTheHome](https://www.servethehome.com/openai-jalapeno-asic-at-hot-chips-2026/))

[^2]: **Slide photographs from the talk.** Four slides — "A request spans three distinct hardware regimes," "Key Takeaways," "Changing ratios → inefficiency in heterogeneous fleets," and "System choice: KV moves or the active silicon mix varies." All slide text quoted in this post is transcribed from those photographs. ([@beffjezos](https://x.com/beffjezos/status/2092416851737518190), 26 August 2026)

[^3]: **SemiAnalysis on Jalapeño.** Reports that the performance figures were supplied by OpenAI, that the published runs cover an 8k input / 1k output workload rather than agentic traces, and that the speculative decoding configuration differs between Jalapeño and the systems it is compared against. Also states that OpenAI chose not to disaggregate prefill and decode across separate chip pools, and that the draft model shares chips and fabric with the main model. ([SemiAnalysis](https://newsletter.semianalysis.com/p/openai-jalapeno-better-than-nvidia))

[^4]: **The five-part series.** [Prefill and decode want different computers]({% post_url 2026-08-16-prefill-and-decode-want-different-computers %}), [the KV cache has no ABI]({% post_url 2026-08-18-the-kv-cache-has-no-abi %}), [there is no address]({% post_url 2026-08-19-there-is-no-address %}), [two schedulers, one SLO]({% post_url 2026-08-20-two-schedulers-one-slo %}), and [a cache its own prefill would never have produced]({% post_url 2026-08-24-a-cache-its-own-prefill-would-never-have-produced %}).

[^5]: **Jalapeño configuration.** Each package pairs the compute die with six HBM4 stacks for 216 GiB at 15.4 TB/s and 13.4 PFLOP/s of MXFP4 matrix compute in a 700 W envelope; a rack is 128 chips holding 27.5 TB, and a pod is 2,048 ASICs. HBM4 reported as supplied by Samsung. ([The Register](https://www.theregister.com/systems/2026/08/25/openais-upcoming-jalapeno-chip-looks-like-itll-be-an-inference-beast/5292052), [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/openai-says-its-jalapeno-chip-beats-nvidias-gb300-in-first-published-benchmarks))

[^6]: **Rubin CPX removed from the NVIDIA roadmap.** Announced at the AI Infra Summit in September 2025 as a prefill-specialised part using GDDR7 rather than HBM, and dropped at GTC 2026. The slot was taken by a 256-chip SRAM-based Groq 3 LPX rack acquired through a licensing arrangement reported at roughly $20B. ([Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-removes-rubin-cpx-accelerators-from-its-roadmap-groq-3-lpus-take-center-stage-as-cpx-is-removed))

---

*Disclaimer: Researched and drafted with AI assistance (Claude Opus 5, Gemini 3.1 Pro). Direction, technical judgment, and final edits are mine. The slide text quoted here is transcribed from photographs of the talk rather than from OpenAI-published material, and I did not attend; the talk metadata and the benchmark reporting come from the two secondary sources cited above. The argument about speculative decoding across a network boundary is mine, reasoned from the phase structure the slide describes, not a measurement.*
