---
title: "Tight, Reproducible, and Wrong: Performance Analysis on an Apple M4"
date: 2026-08-06 09:00:00 -0700
categories: [Systems, Performance]
tags: [apple-silicon, benchmarking, scheduling, rust, rayon]
mermaid: true
---

I spent two days explaining a result that did not exist.

The setup: a compiler frontend I am building interns types two different ways, and I wanted to know which one parallelises better. Deferred interning gives every worker its own arena and reconciles at a barrier. Content-addressed interning hashes the structure so identity is final the moment a type is minted, and no reconciliation is needed at all. Both emit byte-identical MLIR. The only question was throughput.

The benchmark said this:

| threads | deferred | content | ms deferred / content |
|---|---|---|---|
| 1 | 1.00 | 1.00 | 44.0 / 45.6 |
| 2 | **1.01** | 1.59 | **43.8** / 28.7 |
| 3 | 1.80 | 1.93 | 24.5 / 23.6 |
| 4 | 2.19 | 2.34 | 20.1 / 19.5 |

Deferred interning gained nothing from a second core. Content gained 1.59x. And at one thread deferred was *faster*, so this was not a constant-factor loss hiding in the arena allocation. It was specifically a failure to convert the second core into throughput, which is exactly the kind of thing a barrier-based design is supposed to be vulnerable to. The story wrote itself.

I had a mechanism, too. Deferred hands each worker private slow-path arenas, so it allocates per-function `Vec`s that content never creates, and malloc contention is thread-count sensitive. Plausible. Testable. Wrong.

Every number in that table is real. I can reproduce all of them. The finding is still false.

## The part that should have stopped me

Look at what I did to defend against noise. Fifteen repetitions per cell. Medians, not means. Three separate whole-run repeats. The interquartile range inside every cell was under 2%: one cell ran 30.00 to 30.33 ms, another 49.54 to 50.92 ms. By every convention I had absorbed over ten years of benchmarking on x86 Linux, that is a clean measurement.

It was a clean measurement of the wrong thing.

And I had the evidence to catch it, because the pipeline is instrumented per phase rather than end to end. Every run accumulates wall time under fifteen named phases:

```rust
pub const PHASES: [&str; 15] = [
    "parse",
    "macro_expand",
    "name_resolution",
    "registry_freeze",   // serial: the spine the parallel region waits on
    "sig_clone",
    "env_build",
    "return_prov",
    "type_check",
    "dedup_barrier",     // deferred interning reconciles here, and only here
    "stream_extract",
    "simd_patch",
    "codegen",
    "  codegen:setup",
    "  codegen:emit",
    "teardown",
];
```

These are wall times for the whole region, not summed per-thread CPU time, so each number is that phase's share of the *critical path* — which is the right question when you are asking where a parallel speedup went.

So the two candidate stories were separable by measurement, and had been all along. If deferred interning fails to parallelise, the cost must appear in `dedup_barrier`: that is the phase where deferred does its reconciliation, and it is the entire theoretical objection to the design. It measured **0.3 ms**, against a gap of 35 ms.

The gap sat in `codegen` instead — 24.7 ms at one thread, 32.3 ms at two. A phase that runs after the type stream has been extracted and does not touch the interner at all. It got *slower with more threads*, which no interning design can explain and which, on its own, should have sent me looking at the machine.

I wrote a note to myself that this was "odd, worth measuring before believing," and then went on believing it.

When your proposed mechanism cannot reach the phase where the cost lands, you do not have a mechanism. You have a coincidence with a story attached.

## What the machine actually is

The host is an Apple M4. Not a symmetric machine, and `sysctl` will tell you so:

```console
$ sysctl -n hw.nperflevels hw.ncpu
2
10

$ sysctl -n hw.perflevel0.name hw.perflevel0.physicalcpu \
         hw.perflevel1.name hw.perflevel1.physicalcpu
Performance
4
Efficiency
6
```

Four performance cores. Six efficiency cores. Ten "CPUs" that are not interchangeable in any sense that matters to a speedup table.

The caches are asymmetric too, and by more than people expect:

```console
$ sysctl -n hw.perflevel0.l1icachesize hw.perflevel0.l1dcachesize hw.perflevel0.l2cachesize
196608      # 192 KB L1i
131072      # 128 KB L1d
16777216    #  16 MB L2, shared across the P cluster

$ sysctl -n hw.perflevel1.l1icachesize hw.perflevel1.l1dcachesize hw.perflevel1.l2cachesize
131072      # 128 KB L1i
65536       #  64 KB L1d
4194304     #   4 MB L2, shared across the E cluster
```

A thread that lands on an E-core gets a quarter of the L2 and half the L1d. If your working set was tuned against the P-cluster's 16 MB, it does not merely run slower on an E-core, it runs a different algorithm's worth of cache misses.

And note `hw.cachelinesize` is **128 bytes** on Apple Silicon, not the 64 you have padded against your whole career. Every `#[repr(align(64))]` you wrote to kill false sharing is half a line short here. That one is not a measurement problem, it is a correctness-of-your-optimisation problem, and it is silent.

## Quantifying the asymmetry

Running the compiler's thread ladder from 1 to 10 gives the shape of the machine directly:

| threads | speedup | gain from the added thread |
|---|---|---|
| 1 | 1.00 | — |
| 2 | 1.47 | +0.47 |
| 3 | 1.88 | +0.41 |
| 4 | 2.24 | +0.36 |
| 5 | 2.39 | +0.15 |
| 6 | 2.54 | +0.15 |
| 8 | 2.84 | +0.075 |
| 10 | 2.92 | +0.04 |

The knee lands exactly on 4. That is `hw.perflevel0.physicalcpu`, and the coincidence is not a coincidence. Each of the first four threads is worth about 0.4 of a speedup unit. Each of the next two is worth 0.15. So on this workload an E-core contributes roughly **0.37 of a P-core**, and by threads nine and ten it is under 0.15 — at that point you are paying full scheduling and memory-traffic cost for a rounding error.

This is the number that makes the bug comprehensible. If a two-thread pool draws one P-core and one E-core instead of two P-cores, the pool's throughput is not 2 units but roughly 1.37, and the wall clock stretches by about 1.7x. Which is very close to the gap I had attributed to a barrier.

## The trap: bimodal across runs, tight within them

The decisive experiment was to stop comparing the two modes to each other and measure each one alone, one mode per process. Same binary, same corpus, same configuration.

`--modes content`, three processes:

| process | t=1 | t=2 | t=4 |
|---|---|---|---|
| A | 44.12 | **30.13** | 20.06 |
| B | 45.40 | **50.03** | 21.24 |
| C | 46.03 | **50.66** | 20.23 |

And `--modes deferred`, four processes, at t=2: 57.46, 32.89, 31.16, 51.74.

Two clusters. In both modes. The bimodality follows the *process*, not the mode.

I reproduced the same shape outside the compiler, in about sixty lines of C, to convince myself it was the machine and not my code. Fixed total work split N ways across persistent worker threads parked on a condvar barrier between repetitions — the structure a `rayon` pool actually has. Ten separate processes at two threads:

```
proc1   median 172.7 ms   IQR  1.6%
proc2   median 174.0 ms   IQR  1.5%
proc3   median 175.5 ms   IQR  0.7%
proc4   median 176.8 ms   IQR  1.1%
proc5   median 184.6 ms   IQR 15.8%
proc6   median 187.2 ms   IQR  4.7%
proc7   median 186.8 ms   IQR  2.8%
proc8   median 185.0 ms   IQR  0.5%
proc9   median 184.9 ms   IQR  0.4%
proc10  median 185.9 ms   IQR  0.6%
```

Two clusters again, with sub-2% dispersion inside most of them. The magnitude is only 7% rather than 70%, because a pure integer-ALU dependency chain with no memory traffic gives the scheduler almost nothing to punish. But the *shape* is the whole lesson: **tight within a process, clustered across processes.**

That shape defeats every noise defence in the standard toolkit. Repetitions do not help, because the nuisance variable is constant for the life of the pool. Medians do not help, for the same reason. Discarding outliers actively hurts — there are no outliers, there are two populations, and trimming makes each one look even more trustworthy. Reporting a confidence interval is the worst option of all, because you will publish a tight interval around a confounded estimate and the tightness will read as rigour.

A confidence interval describes dispersion around your estimate. It says nothing whatsoever about whether your estimate is of the quantity you think it is.

That C reproduction has a limit. When I created fresh threads for every repetition instead of reusing a pool, the bimodality vanished — 8 processes, every median within 3%. macOS does migrate threads; it is not pinning anything. What it does not appear to do is aggressively re-cluster a persistent pool that stays busy. So "placement is fixed when the pool is built" is too strong as a statement about the kernel. "Placement is sticky, and a long-lived pool tends to keep whatever cluster it settled into" is what I can actually defend, and it is sufficient to produce the bug.

## Why you cannot just pin the threads

On Linux this entire post would end here, with `taskset -c 0-3`. On Apple Silicon that door is bolted.

The Mach thread affinity API still exists, still compiles, still links. It just refuses:

```c
// Ask for the affinity API that works fine on an Intel Mac.
thread_affinity_policy_data_t pol = { .affinity_tag = 1 };
kern_return_t kr = thread_policy_set(mach_thread_self(),
                                     THREAD_AFFINITY_POLICY,
                                     (thread_policy_t)&pol,
                                     THREAD_AFFINITY_POLICY_COUNT);
```

```console
thread_policy_set(THREAD_AFFINITY_POLICY) = 46 ((os/kern) service not supported)
```

46 is `KERN_NOT_SUPPORTED`. XNU is candid about why. `thread_policy_set` gates the affinity case on a predicate[^1]:

```c
if (!thread_affinity_is_supported()) {
        result = KERN_NOT_SUPPORTED;
        break;
}
```

and that predicate is a one-liner[^2]:

```c
boolean_t
thread_affinity_is_supported(void)
{
	return ml_get_max_affinity_sets() != 0;
}
```

Affinity sets are an Intel-era mechanism for grouping threads onto a shared L2. Apple Silicon returns zero. There is no supported way to say "run this thread on core 3," and there will not be one, because the whole design intent is that the scheduler owns placement so it can manage power.

What you get instead is **Quality of Service**: a declaration of intent, from which the OS infers placement. On an asymmetric machine the OS uses the energy-efficiency information in a QoS class to influence whether a thread lands on a P or an E core[^3][^4]. Note the verb. Influence.

## QoS is a hint, and something above you can override it

I measured what QoS actually buys on an otherwise idle M4. Same workload on a fresh thread per class, verifying with `qos_class_self()` that the request took effect:

```console
$ ./probe2
USER_INTERACTIVE  rc=0 requested=33 effective=33     310.9 ms
DEFAULT           rc=0 requested=21 effective=21     294.8 ms
UTILITY           rc=0 requested=17 effective=17     298.6 ms
BACKGROUND        rc=0 requested=9  effective=9      300.8 ms
```

The QoS class was set successfully in every case, and it changed nothing. On an idle machine the scheduler has four P-cores sitting there and hands you one regardless of how modestly you asked. So you cannot use QoS to *force* a thread onto an E-core for testing, and more importantly you cannot infer from a passing `pthread_set_qos_class_self_np` that you got the placement you asked for.

Now run the identical binary underneath a background task policy:

```console
$ taskpolicy -b ./probe2
USER_INTERACTIVE  rc=0 requested=33 effective=33    1054.9 ms
DEFAULT           rc=0 requested=21 effective=21    1094.4 ms
UTILITY           rc=0 requested=17 effective=17    1119.1 ms
BACKGROUND        rc=0 requested=9  effective=9     1115.9 ms
```

Between 3.4x and 3.7x slower. `taskpolicy -b` applies `PRIO_DARWIN_BG`, which maps to the Background QoS class and confines the process to efficiency cores[^5]. Read the first row again: a thread that explicitly requested `QOS_CLASS_USER_INTERACTIVE`, and whose request succeeded, still ran at E-core speed. **A thread's QoS is clamped by the task policy it inherits, and the thread cannot see the clamp.** `qos_class_self()` cheerfully reports 33.

This is the practical hazard, and it has nothing to do with my compiler. Anything that launches your benchmark can apply that policy: a CI agent, an IDE's build task, a `launchd` job with a low-priority key, a terminal session inherited from something that was itself demoted. Your program has no reliable way to report that it happened. You get numbers that are internally consistent, reproducible, tight — and 3.6x off, with the offset silently attached to the *environment*, which is precisely the variable nobody records in the results table.

If you take one operational thing from this post: when a macOS benchmark number moves and your code did not, check whether the launching context changed before you check anything else.

## The second-order headaches

**`available_parallelism()` counts cores that are not equal.** Rust returns 10 on this machine[^6], matching `hw.ncpu`, and `rayon` sizes its global pool from exactly that[^7]. So the default pool on an M4 is ten threads: four good ones and six worth about 0.37 each, with the last two contributing under 0.15. The default is not merely suboptimal, it is a pool whose composition changes the answer your benchmark gives, chosen by a function that has no way to know the difference.

**The first repetition is a frequency ramp, not a measurement.** In my thread-count sweep the first rep at one thread came in at 110.2 ms against a steady state of 82.6 ms — 33% high, purely from DVFS spinning up. Discard warmup reps. On a machine with this much clock range, one warmup is not enough for a short kernel.

**Single-thread turbo is not a valid baseline.** The P-cluster clocks higher when one core is busy than when four are. Your `speedup = t1 / tn` therefore divides by a number measured at a clock the parallel run never sees, and the parallel arm is penalised for a frequency change you did not attribute. This is not unique to Apple, but the range is wide here.

**The timebase is coarser than you think.** `mach_absolute_time` on Apple Silicon is not nanoseconds:

```console
mach_timebase_info: numer=125 denom=3  -> 41.6667 ns/tick
```

A 24 MHz counter[^8]. Fine for millisecond phases, useless for timing an individual function call the way you would with `rdtsc`. Convert through `mach_timebase_info`, never assume the ratio is 1.

**There is no `perf`.** No `perf stat`, no `perf record`, no PMU counters from a normal process. Instruments and the `kperf` interface behind it are the sanctioned route, `powermetrics` (as root) will show you per-cluster residency and frequency, and that is roughly the extent of it. The specific counter you want in order to distinguish "this thread stalled on memory" from "this thread ran on a small core" is not conveniently available, which is a large part of why this class of bug survives so long.

## The fix is experimental design, not systems programming

The bug was never in the compiler. It was in the harness, and specifically in the order it ran things.

```mermaid
graph TB
    subgraph BAD["Confounded: each mode builds its own pool"]
        direction TB
        A1["deferred: build pool at t=2"] --> A2["drew P + E cluster"]
        A2 --> A3["43.8 ms"]
        B1["content: build pool at t=2"] --> B2["drew P + P"]
        B2 --> B3["28.7 ms"]
        A3 --> C1["conclusion: deferred does not scale"]
        B3 --> C1
        C1 --> C2["FALSE - measured placement, not mode"]
    end

    subgraph GOOD["Paired: one pool per cell, both modes inside it"]
        direction TB
        D1["build pool at t=2"] --> D2["whatever placement it drew"]
        D2 --> D3["deferred rep 1"]
        D2 --> D4["content rep 1"]
        D3 --> D5["alternate, rep by rep"]
        D4 --> D5
        D5 --> D6["placement cancels in the ratio"]
    end

    style C2 fill:#7f1d1d,color:#fff
    style D6 fill:#14532d,color:#fff
```

The old harness ran each mode's entire thread ladder end to end. Two ladders, two pools, two independent draws from the placement lottery. Whichever mode happened to draw the efficiency cores at `t=2` looked like it could not use a second core.

The fix is blocking, and it is nearly a hundred years old. Build **one pool per cell** and run every mode inside it, alternating repetition by repetition. Whatever placement that pool drew, both modes drew it, so placement cancels in the ratio. The nuisance variable is still there and still large — I just stopped letting it vary between the things I was comparing.

With pairing:

| threads | deferred | content | speedup |
|---|---|---|---|
| seq | 99.5 | 99.1 | 1.00 |
| 1 | 99.7 | 99.3 | 1.00 |
| 2 | 69.0 | 68.9 | 1.44 |
| 4 | 43.1 | 43.2 | 2.30 |
| 8 | 33.9 | 33.4 | 2.95 |

Within 1% at every thread count. There is no anti-scaling. There never was. The two designs perform identically, and the 1.59-versus-1.01 table was a picture of which cluster each pool happened to land on.

Two guards now sit in the harness output so this cannot recur quietly:

- **A row slower than a row with fewer threads is flagged `SUSPECT`.** More workers cannot lengthen the same work. If the table says otherwise, the table is measuring the machine, and it should say so in its own output rather than wait for someone to notice.
- **`--modes` accepts a single mode**, so one process measures one mode, and passing the two modes in reversed order is a second run. Between them, "does this effect follow the mode or its position in the run" becomes decidable instead of assumed.

That second guard is the one I would push hardest. Any A/B harness that runs A's full sweep and then B's full sweep has silently confounded the comparison with *time* — thermal state, background daemons, page-cache warmth, and on this machine, cluster placement. Interleave, or you are comparing two experiments rather than two treatments.

## The same harness lied a second time, and not about Apple

Placement is the Apple-specific failure. The other one in this harness is not, and it is the more interesting of the two.

An earlier revision of the benchmark's README carried a published table of scaling results. Those numbers are withdrawn, because they were taken against a frontend that built a `TransferCostGraph` per *function* and ran an all-pairs shortest-path precompute twice for each one. A profile eventually put that at the top of the entire compiler by self time. Deleting it cut wall clock by 2.35x at one thread.

So far this is an ordinary optimisation story. Here is the part that matters:

**Removing that work made the measured parallel scaling worse.** Across 8 threads the curve fell from 2.9x to 2.67x.

Nothing regressed. The compiler got 2.35x faster in absolute terms. But the deleted work sat *inside* the parallel region and scaled beautifully — all-pairs shortest path over a per-function graph is embarrassingly parallel and touches no shared state. It was inflating the numerator of every speedup ratio while contributing nothing a user wanted.

A scaling curve measured over avoidable work overstates how parallel your compiler is. Amdahl's law is a statement about the serial *fraction*, and you can always improve the fraction by padding the parallel part with garbage. Every unnecessary parallel-friendly computation in your pipeline is buying you scaling numbers you have not earned, and the better your profiler-guided cleanup gets, the worse your speedup graph looks. If you optimise and your scaling curve improves, check whether you deleted serial work or merely added parallel work.

Two smaller traps from the same file, both of which produced wrong numbers before they were caught:

**The instrument was inside the experiment.** The parse phase opened its per-module closure with a `println!`, inside a rayon parallel-for. `println!` takes the global stdout lock, so every worker serialised on it once per module — 512 lock acquisitions per repetition at the largest cell, in the exact region whose scaling was being measured. Profile that run and you will find contention, correctly reported, on stdout rather than on the thing you are studying. Progress chatter is now behind an environment gate for any timed run.

**A cached input outlived the generator that produced it.** The corpus is written to a directory named by a digest of its parameters, and reused when the digest matches. Change the generator without bumping its version constant and every cached corpus still looks current, so the next run compiles the *old* programs while faithfully reporting the *new* flags. That is not a hypothetical; it happened during development, and the version constant exists because of it.

## What it cost

The claim I had to retract was not small. I had been arguing that content-addressed identity was worth its complexity partly because it parallelised better. That performance argument was mine, and it was wrong.

The design argument survives intact and never needed the speed claim: with content addressing, identity is final at mint time, so there is no barrier, no patch pass, and determinism holds across any scheduling, partitioning, worker count, or process boundary. Reproducible builds are the point. "And it is faster" was a bonus I invented from a scheduling artifact.

Those two failures have different fixes. Publishing a wrong number is a harness bug, and the harness is fixed. Building an architectural argument on top of a number I had already noticed did not fit its own mechanism is a judgement failure, and the only fix for that one is to treat "the cost landed in a phase my explanation cannot reach" as a stop condition rather than a footnote.

Apple Silicon is a fine machine to develop on. It is a hostile machine to *measure* on, in a specific way: it removes the control you would use to eliminate the largest nuisance variable, then hides that variable behind numbers with excellent-looking dispersion. The defence is not a better timer or more repetitions. It is assuming the machine is heterogeneous and adversarial, and designing the comparison so that whatever it does to you, it does to both arms at once.

Which leaves a rule I now apply to anything measured on this laptop:

**Ratios between two modes on one machine survive core heterogeneity. Absolute scaling curves do not.**

A paired A/B inside a single pool cancels placement, so "deferred versus content" is answerable here. "This compiler achieves 2.92x on ten threads" is not, and no amount of care in the harness makes it so — the last six of those threads are cores worth roughly a third of the first four, and 4→8 threads gaining 31% is a description of what happens when you add four weak cores, not a description of how parallel the code is. That number needs homogeneous hardware before anyone quotes it, mine included.

---

## References

[^1]: **XNU `thread_policy_set`, `THREAD_AFFINITY_POLICY` case.** Returns `KERN_NOT_SUPPORTED` when `thread_affinity_is_supported()` is false. ([`osfmk/kern/thread_policy.c`](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/thread_policy.c))

[^2]: **XNU `thread_affinity_is_supported()`.** Defined as `ml_get_max_affinity_sets() != 0`; the file notes affinity sets are not used on platforms with a single processor set. ([`osfmk/kern/affinity.c`](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/affinity.c))

[^3]: **Prioritize Work with Quality of Service Classes.** Apple's definition of the user-interactive, user-initiated, utility and background classes, and the statement that the system adjusts scheduling, CPU and I/O throughput based on them. Apple does not document a QoS-to-core mapping here. ([Link](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/EnergyGuide-iOS/PrioritizeWorkWithQoS.html))

[^4]: **Optimize for Apple silicon with performance and efficiency cores.** Apple's developer note on asymmetric multiprocessing, describing QoS as the mechanism by which the OS places threads on P or E cores. ([Link](https://developer.apple.com/news/?id=vk3m204o))

[^5]: **`PRIO_DARWIN_BG` and efficiency-core confinement.** Discussed in the LLVM patch adding `ThreadPriority::Low` and QoS class Utility on macOS, which distinguishes Utility from the Background class precisely because Background confines the thread to efficiency cores. ([D124715](https://reviews.llvm.org/D124715), and the parallel Ruby discussion at [Feature #17566](https://bugs.ruby-lang.org/issues/17566))

[^6]: **`std::thread::available_parallelism`.** Documented as the amount of parallelism available, not the amount of *useful* parallelism; it makes no distinction between core types. Returns 10 on this M4. ([Link](https://doc.rust-lang.org/std/thread/fn.available_parallelism.html))

[^7]: **`rayon::ThreadPoolBuilder::num_threads`.** When unset (or 0), rayon derives the global pool size from the available parallelism. ([Link](https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html))

[^8]: **`mach_absolute_time` and `mach_timebase_info`.** Apple's technical note on converting the raw tick count to nanoseconds via the numerator/denominator pair, which is 125/3 on Apple Silicon. ([QA1398](https://developer.apple.com/library/archive/qa/qa1398/_index.html))

---

*Disclaimer: Researched and drafted with AI assistance (Claude Opus 5). Direction, technical judgment, and final edits are mine; every claim is traceable to the sources cited above. The `sysctl`, QoS, `taskpolicy`, affinity and timebase measurements in this post were run on the M4 described; the compiler thread ladder and the paired results come from my own benchmark harness.*
