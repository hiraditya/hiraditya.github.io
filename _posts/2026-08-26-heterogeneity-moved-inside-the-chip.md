---
title: "Heterogeneity Moved Inside the Chip"
date: 2026-08-26 06:00:00 -0700
categories: [Systems, Inference]
tags: [inference, disaggregation, accelerators, speculative-decoding, hardware, compilers, llm-serving]
mermaid: true
---

At Hot Chips this week, OpenAI presented Jalapeño, an inference ASIC built with Broadcom.[^1] The slide that has been circulating since shows a request split three ways: prefill, a draft model, and speculative verification.[^2]. Typically requests are split between prefill and decode but using a draft model and speculative verication to replace the typical 'decode' makes it worthwhile to inspect their architecture further.

## One request split three ways

The first slide sets up the problem in the same terms this series has been using. A single request passes through three phases, and each one saturates a different part of the machine.

| Phase | Job | Bottleneck | Profile |
|---|---|---|---|
| Prefill | Encode context | FLOPs + attention | Compute high, memory bandwidth low, comms smooth |
| Draft model | Speculate | Network latency | Small model, ultra-low batch, latency bound |
| Spec-verify | Decode | Attention + HBM bandwidth | Attention compute-bound, MoE bandwidth-hungry, comms bursty |

The prefill column reads as expected: "attention-heavy and primarily compute-bound. Low memory-BW demand; communication is easier to schedule smoothly." The verify column is the mirror image, with the added detail that mixture-of-experts routing makes its communication arrive in bursts rather than a steady stream.

The middle column is the one that did not exist when [part one]({% post_url 2026-08-16-prefill-and-decode-want-different-computers %}) of this series argued that prefill and decode want different computers. A draft model is small, runs at a batch size close to one, and moves very little data. It requires "low network bandwidth, but extreme latency sensitivity." TLDR: It is impatient!

Their last slide concludes thier line of thought on designing Jalapeño: "What matters is requests/second/watt at the required SLA latency. Each phase hits a different bottleneck; efficiency only counts if the complete request remains within its end-to-end latency target."

That is the same objective [part four]({% post_url 2026-08-20-two-schedulers-one-slo %}) argued no disaggregated system can actually optimize, because no component owns the end-to-end budget.

## The system choice

Three phases with three bottleneck profiles is a textbook argument for specialization. Build a prefill chip, a draft chip, a verify chip, and route the request through all three.

OpenAI put that option on a slide and rejected it. The title is "System choice: KV moves or the active silicon mix varies."

```mermaid
graph TB
    subgraph L["Disaggregated specialization"]
        P["Prefill<br/>compute"] -->|KV| D["Draft<br/>latency"]
        D -->|KV| V["Verify<br/>memory + compute"]
    end
    subgraph R["Unified, internally heterogeneous"]
        C["compute"] --- M["memory"] --- N["network"]
        C --- K["KV stays local"]
    end
    L --> LX["network + synchronization<br/>at every phase boundary"]
    R --> RX["activate the right ratio by phase<br/>unused units go dark"]
    style LX fill:#7f1d1d,color:#fff
    style RX fill:#1e3a5f,color:#fff
```

The right-hand column won. In OpenAI's words: "Heterogeneity moves inside the chip—not across a KV-moving network." The summary line is "Locality is king: keep KV local and activate the right resources."

A separate slide gives the reasoning in one sentence: "dark silicon is cheaper than idle accelerators." Separate accelerators still pay for package, HBM, I/O, network and cooling power whether or not they are doing anything. Blocks on a single die can be gated off while the KV cache stays where it is.

## Why the boundary was refused

Three arguments carry that decision, and each maps onto something this series worked through from the software side.

**The ratio is not fixed.** The slide shows three workload mixes with visibly different prefill/draft/verify proportions, and lists what moves them: model architecture, token efficiency, context length, attention algorithms, speculative acceptance rate, and the latency-versus-throughput target. A fleet built as a fixed split of specialized pools is correct for exactly one of those mixes. For the rest, an entire chip idles because it belongs to the wrong pool.

**Locality.** Prefill produces a KV cache that the next phase needs immediately. Disaggregating means that cache crosses a network at every phase boundary. [Part two]({% post_url 2026-08-18-the-kv-cache-has-no-abi %}) covered what that cache costs to move and [part three]({% post_url 2026-08-19-there-is-no-address %}) covered why you cannot say where it should land.

**Speculation is a tight loop.** This one is specific to the new middle phase. Draft and verify are not a pipeline; they are a loop. The draft model proposes k tokens, verify accepts some prefix and rejects the rest, and the draft resumes from whatever survived. That round trip happens every few tokens. Put a network between them and each speculation round pays a network latency, against a phase the slide itself labels latency-bound. The mechanism that speculative decoding uses to buy latency is the first thing a boundary would spend.

## The diagnosis, arrived at from the other side

The five posts before this one argued that a programming model for heterogeneous inference would have to express five things that are inexpressible today: where a stage runs, what the bytes mean, where a tensor resides and what moving it costs, who preempts whom, and what the values are permitted to be.[^4]

OpenAI's design makes all five expressible by the only method currently available. It deletes the boundary. One vendor, one chip, one memory hierarchy, one scheduler, one set of kernels. Every question this series asked about crossing a vendor line stops being a question when there is no line.

This is the strongest confirmation the argument has had. Presented with three phases that genuinely want different hardware, an organization with its own silicon team looked at the cost of the boundary and declined to pay it.

The benchmark numbers on the closing slide are being quoted widely and I am setting them aside. SemiAnalysis, which saw the runs in person, reports that the figures came from OpenAI, cover a single 8k/1k workload, and are not iso-configuration — the comparisons involve different speculative decoding settings on either side.[^3] The architectural argument does not depend on them.

## What deleting the boundary does not delete

Consolidation settles who owns the machine and leaves open what to do with it.

"Activate the right ratio by phase; unused units go dark" is a runtime decision. Prefill wants the compute blocks and can leave memory bandwidth mostly idle. Verify wants the opposite, plus a burst-tolerant interconnect for expert routing. Draft wants very little of anything except a short path to the verify units. Something has to set that mix, per phase, and reset it a few hundred times a second.

The inputs to that decision are the same list from the changing-ratios slide, and every item on it is dynamic. Context length varies per request. Speculative acceptance rate varies per request and drifts with the model. The latency-versus-throughput target varies by customer tier. None of these are known when the kernel is compiled.

At the die boundary the question changes jurisdiction. Across a network it was a protocol problem, and the answer was that no protocol exists. On a single chip it is a compiler and runtime problem, which is a better place for it to be — a compiler has visibility into the phase structure of the computation, and there is finally a single owner who could act on the answer.

What is missing is a way to say it. A kernel today declares the shapes it consumes and the memory it touches. It does not declare that this phase is compute-heavy and bandwidth-light while the next inverts that, that the interconnect should expect bursts rather than a steady stream, or that the resource mix should be re-derived when the acceptance rate moves. Those are properties of a phase, and a phase is not something the interface currently names.

## The assumption underneath

Keeping the KV cache local is a claim about capacity, and capacity is a function of context length.

A Jalapeño package pairs its compute die with six HBM4 stacks: 216 GiB at 15.4 TB/s in a 700 W envelope, with a 128-chip rack holding 27.5 TB.[^5] Whether that is generous or tight depends on how much KV a session carries, and the spread across attention designs is wide enough to change the answer.

| KV layout | Per token | One 1M-token session | Such sessions per 216 GiB |
|---|---|---|---|
| MLA, the 656-byte entry from part two | 656 B | 0.6 GiB | ~350 |
| GQA: 80 layers, 8 KV heads, 128-wide, fp8 | 160 KiB | ~153 GiB | 1.4 |
| the same at bf16 | 320 KiB | ~305 GiB | does not fit |

OpenAI does not publish its serving configurations, so the two GQA rows are a bracket rather than a measurement of anything it runs. The point is the ratio between them. A compressed-latent design keeps a million tokens local without difficulty. A conventional grouped-query layout at the same length puts one session on seventy percent of a package.

Batch size binds before capacity does. Decode throughput comes from batching sequences together, and long context is what destroys batch size. At 8k a chip holds hundreds of sessions and batches across them. At 1M with the layout above it holds one, and decode collapses into the memory-bound GEMV described in part one, with nothing left to amortise the weight traffic against. Across a full rack, 27.5 TB works out to roughly 168 concurrent million-token sessions.

The dark-silicon argument also narrows here. Gating an unused block recovers power. It does not recover die area, and an agentic mix — long input, short output — skews toward prefill for long stretches, leaving the memory and network blocks dark on silicon that was already bought. Gating is an operating-cost answer to what becomes a capital-cost question once the skew is persistent.

Long sessions add a third pressure unrelated to peak load. Agentic work waits: on a tool call, on a retrieval, on a person. Through those gaps the session's KV sits resident in HBM, which is the most expensive place in the system to hold cold state. The usual remedy is to move it down a tier, to host memory or a pooled store, which is disaggregating memory inside a design whose argument was that disaggregation costs too much.

None of this is disqualifying, and the defence is strong. "Local" can mean rack-local across a coherent fabric rather than resident on one die, and 27.5 TB is real headroom. Agentic sessions also share unusually large prefixes — system prompts, tool schemas, repository context — so if prefix cache hit rates are high, prefill work collapses and the phase mix swings back toward decode, which is the regime a balanced chip is built for.

What is missing is evidence. The published figures cover an 8k input, 1k output workload. SemiAnalysis, which watched the runs in person, reports there are no agentic traces yet, and that the components under the most pressure in that regime are the routers and the prefix caching machinery.[^3] Those are the parts that decide the question.

## Two live bets, not a consensus

The rest of the industry has not converged on this answer.

NVIDIA removed Rubin CPX from its roadmap at GTC 2026 — the part built specifically for compute-bound prefill, with GDDR7 standing in for HBM. The slot went to a 256-chip SRAM-based Groq 3 LPX rack, through a licensing deal reported at around twenty billion dollars.[^6] One specialist was exchanged for another aimed at the opposite end of the request, while Rubin and Rubin Ultra keep growing. Both bets are running inside the same roadmap.

The vendor pairings from part one point the same way. Each puts a specialised part on one side of a phase boundary and treats the boundary as a cost worth paying.

Two organisations with comparable information and comparable incentives are answering this in opposite directions, which is a fair sign the question is still open.

## Different configurations of the same computer

Part one of this series claimed that prefill and decode want different computers. Jalapeño is a reasonable argument that they want different *configurations* of the same computer, and that the configuration should change while the request is in flight.

That is a more demanding claim than the one I made, not a softer one. Different computers is a procurement problem, and the industry knows how to solve procurement problems. A machine that reallocates its own compute, memory and interconnect between three phases of a single request, driven by properties that are only known at runtime, is a language problem. It is the one I have been circling for five posts, and the hardware just arrived first.

---

## References

[^1]: **OpenAI Jalapeño custom AI ASIC, Hot Chips 2026.** Presented on day two of the conference, 25 August 2026, by Richard Ho, Ravi Narayanaswami and Chris Leary; the chip is built in collaboration with Broadcom. ([ServeTheHome](https://www.servethehome.com/openai-jalapeno-asic-at-hot-chips-2026/))

[^2]: **Slide photographs from the talk.** Four slides — "A request spans three distinct hardware regimes," "Key Takeaways," "Changing ratios → inefficiency in heterogeneous fleets," and "System choice: KV moves or the active silicon mix varies." All slide text quoted in this post is transcribed from those photographs. ([@beffjezos](https://x.com/beffjezos/status/2092416851737518190), 26 August 2026)

[^3]: **SemiAnalysis on Jalapeño.** Reports that the performance figures were supplied by OpenAI, that the published runs cover an 8k input / 1k output workload rather than agentic traces, and that the speculative decoding configuration differs between Jalapeño and the systems it is compared against. Also states that OpenAI chose not to disaggregate prefill and decode across separate chip pools, and that the draft model shares chips and fabric with the main model. ([SemiAnalysis](https://newsletter.semianalysis.com/p/openai-jalapeno-better-than-nvidia))

[^4]: **The five-part series.** [Prefill and decode want different computers]({% post_url 2026-08-16-prefill-and-decode-want-different-computers %}), [the KV cache has no ABI]({% post_url 2026-08-18-the-kv-cache-has-no-abi %}), [there is no address]({% post_url 2026-08-19-there-is-no-address %}), [two schedulers, one SLO]({% post_url 2026-08-20-two-schedulers-one-slo %}), and [a cache its own prefill would never have produced]({% post_url 2026-08-24-a-cache-its-own-prefill-would-never-have-produced %}).

[^5]: **Jalapeño configuration.** Each package pairs the compute die with six HBM4 stacks for 216 GiB at 15.4 TB/s and 13.4 PFLOP/s of MXFP4 matrix compute in a 700 W envelope; a rack is 128 chips holding 27.5 TB, and a pod is 2,048 ASICs. HBM4 reported as supplied by Samsung. ([The Register](https://www.theregister.com/systems/2026/08/25/openais-upcoming-jalapeno-chip-looks-like-itll-be-an-inference-beast/5292052), [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/openai-says-its-jalapeno-chip-beats-nvidias-gb300-in-first-published-benchmarks))

[^6]: **Rubin CPX removed from the NVIDIA roadmap.** Announced at the AI Infra Summit in September 2025 as a prefill-specialised part using GDDR7 rather than HBM, and dropped at GTC 2026. The slot was taken by a 256-chip SRAM-based Groq 3 LPX rack acquired through a licensing arrangement reported at roughly $20B. ([Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-removes-rubin-cpx-accelerators-from-its-roadmap-groq-3-lpus-take-center-stage-as-cpx-is-removed))

---

*Disclaimer: Researched and drafted with AI assistance (Claude Opus 5). Direction, technical judgment, and final edits are mine. The slide text quoted here is transcribed from photographs of the talk rather than from OpenAI-published material, and I did not attend; the talk metadata and the benchmark reporting come from the two secondary sources cited above. The argument about speculative decoding across a network boundary is mine, reasoned from the phase structure the slide describes, not a measurement.*
