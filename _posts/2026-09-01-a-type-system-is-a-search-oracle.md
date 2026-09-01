---
title: "A Type System Is a Search Oracle"
date: 2026-09-01 06:00:00 -0700
categories: [Systems, Compilers]
tags: [types, llm, rust, lean, cpp, formal-methods, code-generation, compilers]
mermaid: true
---

Working with models on real code, I keep noticing the same thing. The Rust I get back is better than the C++ I get back, and the Lean is better than either. Not marginally. The Rust compiles and does roughly what I asked; the C++ compiles and does something adjacent to what I asked, and I find out which on a Tuesday three weeks later.

The published benchmarks say the opposite, and clearly. Rust is a low-resource language as far as a pretrained model is concerned, underrepresented relative to Python by a wide margin, and it scores well below Python on the standard multilingual code benchmarks.[^1] Lean barely registers. If you ranked languages by how much of them a model has read, my experience runs backwards down the list.

Both observations are correct. Reconciling them is the interesting part, and the answer changes what I think a type system is for.

## What one-shot accuracy leaves out

A pass@1 number measures whether the first draft is right. Nobody ships the first draft. The number I care about is different: of the programs that reach production, how many are wrong, and how much work did it take to get there.

The cleanest experiment I have found on this used Idris, which is a reasonable stand-in for the far end of the type-strength axis. Li and Krishnamachari gave GPT-5 fifty-six Exercism problems in Idris and measured it zero-shot against the same model on other languages.[^2]

| Language | Zero-shot | With compiler errors fed back |
|---|---|---|
| Python | 45 / 50 | — |
| Erlang | 35 / 47 | — |
| Idris | 22 / 56 | **54 / 56** |

Zero-shot, the result is what the benchmarks predict. Idris solves 39% where Python solves 90%. The model has not read much Idris and it shows.

Then the compiler goes in the loop, and Idris finishes at 96% — above where Python started. The ablation is the part worth sitting with. They also tried feeding the model documentation, and feeding it a guide to classifying Idris errors. Neither worked as well as handing back the local compilation errors. The compiler's own message, generated from the model's own broken code, was the most valuable signal available.

That is the reconciliation. Training data determines where the first draft lands. The type system determines whether the loop that follows converges on something correct or just on something that runs.

## C++ is the control group

If the thesis were "static typing helps," C++ would be fine. It is statically typed, aggressively so, with a type system elaborate enough to be Turing-complete at compile time. My experience with generated C++ is nonetheless closer to Python than to Rust, and I think that difference is the most informative data point I have.

The distinction is not static versus dynamic. It is how much a successful compile promises.

In Rust, a program that compiles has been checked for a specific and useful set of properties: no use-after-free, no data races between threads, no aliasing a mutable reference, every enum match exhausted, every error path acknowledged at the call site. The checker cannot be talked out of these. Defeating it requires writing `unsafe`, which is a lexically visible, greppable admission that the guarantee stops here.

In C++, a program that compiles has been checked that the names resolve and the overloads pick out. It has not been checked for use-after-free, iterator invalidation, data races, out-of-bounds access, or signed overflow. An implicit conversion can quietly change the meaning of a call. A `reinterpret_cast` will convert any pointer into any other pointer with no ceremony at all. Undefined behaviour is not a diagnostic; it is a licence for the optimiser to assume the case never happens.

So "it compiles" carries far less information in C++ than in Rust, and it is the information content of that signal that determines how useful the compiler is as a reviewer. A checker that can be defeated silently is a checker whose approval means less.

```mermaid
graph LR
    A["generated program"] --> B{"checker"}
    B -->|"rejected"| C["error text<br/>fed back to the model"]
    C --> A
    B -->|"accepted"| D["reaches runtime"]
    D --> E["wrong in ways<br/>the checker cannot see"]
    style C fill:#1e3a5f,color:#fff
    style E fill:#7f1d1d,color:#fff
```

Every type system is this diagram. What differs between languages is how much of the wrongness flows down the right-hand edge instead of the left. Python sends nearly everything right. C++ sends a great deal right, including most of the memory-safety and concurrency errors that matter. Rust pushes a large, well-defined class of it left. Lean pushes almost all of it left.

## Lean is the limit case

Lean is where the argument stops being about ergonomics and becomes about algorithms.

A Lean proof is checked by a kernel. It checks or it does not, and the answer is a decision rather than an opinion. That single property changes what you are allowed to do with an unreliable generator, because it makes generation searchable.

Sampling thirty-two candidate proofs and keeping the one that checks is a sound procedure. You get a correct proof or you get nothing, and you always know which. Delta Prover reaches 95.9% on miniF2F this way, using a general-purpose model with no fine-tuning at all, driving Lean 4 through decomposition and iterative repair against compiler feedback.[^3] Recent systems report figures in that range with the same basic shape: propose, check, repair, repeat.

Now try that in Python. Sample thirty-two implementations, run the tests, keep the ones that pass. What you have is thirty-two programs that agree with your test suite, which is a much weaker statement than thirty-two correct programs, and you have no way to distinguish the two. The oracle is partial, so the search is unsound. You are back to reading the code.

A verifier you can call cheaply and trust completely turns a mediocre generator into a good one, because brute force becomes admissible. That is why models are unreasonably good at Lean given how little Lean exists to have been trained on. The scarcity is real. The oracle compensates.

## What the checker still cannot see

Rust that compiles can still be wrong. It can compute the wrong thing correctly, with excellent memory safety, forever. Types constrain the shape of a computation, not its intent, and outside a dependently typed setting they only encode the part of the specification you chose to write down.

The self-repair literature is consistent about where the remaining difficulty lives: syntactic and type errors turn out to be far more tractable for a model to fix from feedback than logical or algorithmic ones.[^4] That finding is usually reported as a limitation. I read it as the mechanism. Moving an error from the second category into the first is exactly what a stronger type system does, and it is the whole of the benefit. An off-by-one in an index becomes a compile error when the index is a distinct type. A forgotten case becomes a compile error when the match must be exhaustive. A stale pointer becomes a compile error when lifetimes are tracked.

None of that makes the model smarter. It relocates a class of mistakes from the expensive category to the cheap one.

## Why this matters more for generated code than for mine

When I write a function, the code is a partial record of a model I hold in my head. The invariants I did not write down are still real, because I know them and I will maintain them. Review works reasonably well against that background, since a reviewer can ask what I was thinking and get a coherent answer.

A generated function comes with no such model. It is locally plausible text, which is precisely the failure mode that human review is worst at catching. Reviewers are good at spotting code that looks wrong. Generated code that is wrong usually looks right — that is the same property that makes it useful when it happens to be correct.

The mental model that used to carry the unwritten invariants is gone, and a machine checker is the only cheap way to put constraints back. It is indifferent to plausibility. It does not get tired at four in the afternoon, and it does not extend the benefit of the doubt to code that reads confidently.

This is the argument I made from a different direction in [an earlier post on provenance]({% post_url 2026-08-14-provenance-is-not-correctness %}): knowing where code came from tells you nothing about whether it is correct, and a signature on a program is not a proof about its behaviour. The conclusion is the same from both sides. As more code is generated, the value of properties a machine can check rises, and the value of properties resting on an author's understanding falls, because there is no author holding the understanding.

## The part I have not resolved

I would like this to be a measurement rather than an impression, and it is not one yet. My evidence is one controlled experiment on a language nobody deploys, a body of theorem-proving results whose success depends on a total oracle that ordinary programming does not have, and my own experience, which is uncontrolled and which knows what conclusion it prefers.

What I would want is the number nobody publishes: defects per thousand lines of generated code that reached production, cut by language, controlled for the same task and the same reviewer discipline. Pass@1 on programming puzzles is a poor proxy, and it is the wrong end of the pipeline.

The mechanism is clear enough to act on regardless. A generator with a nonzero error rate needs a verifier, the strength of the type system sets how much of the specification the verifier can decide, and everything it cannot decide falls to a reviewer whose weakest moment is confident, plausible, wrong code. Choosing a language for a codebase that will be substantially machine-written is now partly a decision about how much of your specification you want the compiler to hold.

---

## References

[^1]: **Low-resource programming languages in code LLMs.** Rust is consistently classified among the low-resource languages for code generation, substantially underrepresented in pretraining data relative to Python, with correspondingly lower pass rates on multilingual benchmarks such as MultiPL-E. Reported figures vary between studies, so I have avoided quoting a single number. ([Survey: LLM-based code generation for low-resource and domain-specific languages](https://arxiv.org/pdf/2410.03981), [MultiPL-E](https://arxiv.org/pdf/2208.08227))

[^2]: **Compiler-Guided Inference-Time Adaptation: Improving GPT-5 Programming Performance in Idris.** Minda Li and Bhaskar Krishnamachari, February 2026. Zero-shot, GPT-5 solves 22 of 56 Exercism problems in Idris, against 45 of 50 in Python and 35 of 47 in Erlang. Of the refinement strategies tested — platform feedback, added documentation, error-classification guides, and local compilation errors — feeding back local compilation errors performed best, raising Idris to 54 of 56. ([arXiv:2602.11481](https://arxiv.org/abs/2602.11481))

[^3]: **Solving Formal Math Problems by Decomposition and Iterative Reflection.** Delta Prover drives Lean 4 from a general-purpose LLM with no model specialization, using reflective decomposition and iterative proof repair against compiler feedback, and reports 95.9% on miniF2F. ([arXiv:2507.15225](https://arxiv.org/abs/2507.15225))

[^4]: **Iterative feedback loops for LLM code correction.** Across iterative-refinement studies, models improve markedly when given compiler errors and failing test cases, and the residual difficulty concentrates in logical and algorithmic errors rather than syntactic or type errors. ([Unlocking LLM Code Correction with Iterative Feedback Loops](https://arxiv.org/pdf/2606.17514))

---

*Disclaimer: Researched and drafted with AI assistance (Claude Opus 5). Direction, technical judgment, and final edits are mine. The Idris and Lean figures are quoted from the linked papers rather than reproduced by me. The comparison between generated Rust, C++ and Lean that opens this post is my own working experience and is not a controlled measurement; I have tried to be explicit about which claims rest on it.*
