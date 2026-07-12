---
title: "Hardware Hazards on the B200: Ground-Truth Testing for Instruction Schedulers"
date: 2026-07-12 12:00:00 -0700
categories: [Systems, GPU Architecture]
tags: [gpu, assembly, ptxsass, b200]
---

When developing instruction schedulers and assemblers for modern GPUs like the Nvidia B200, static analysis and coverage metrics are necessary but insufficient. A scheduler can report 100% coverage on Read-After-Write (RAW) dependency tracking, yet still emit code that fails catastrophically on actual silicon. 

This happens because the true hardware pipeline is the final arbiter of correctness. When an assembler under-stalls a dependency—allowing a consumer instruction to issue before the producer's result is firmly committed—the hardware will dutifully execute the schedule, reading stale register state and producing incorrect results. These are not defects in the silicon; they are schedule violations where the hardware exposes our incorrect assumptions. 

In compiler backends, we generally adhere to a strict rule: over-stalling is a performance bug, but under-stalling is a silent correctness bug. To catch these, we must maintain a registry of hardware hazards backed by minimal, reproducible on-silicon tests. Let us examine the mechanics of these hazards on the B200 architecture and how we can systematically trigger them to validate scheduler correctness.

## 1. The Predicate-Consumer Under-Stall (H1)

The most insidious bugs are those that slip through rigorous static checks. Recently, a scheduler shipped with a bug involving predicate evaluation, despite static metrics claiming full RAW coverage.

The pattern involves an integer set-predicate instruction (`ISETP`) that computes a condition and writes it to a predicate register, which is subsequently read by a branch instruction. Consider a classic back-edge branch:

```assembly
// 1. Produce the predicate P1 based on some condition.
ISETP.GE.AND P1, PT, R0, R1, PT;

// 2. Consume P1 as the branch target condition, guarded by P0.
@!P0 BRA P1, target;
```

The bug originated in the `def_use` analysis pass. The analyzer correctly recorded the guard predicate `P0` as a use, but it dropped the branch condition operand `P1`. Consequently, the dependency graph missed the `ISETP` $\rightarrow$ `BRA` RAW dependency. The scheduler, unaware of the dependency, failed to insert the required predicate-latency stall.

The branch issued roughly 4 cycles after the `ISETP`, well before the predicate's actual latency (modeled at 13 cycles) had elapsed. The branch instruction read a stale value for `P1`, took the wrong execution edge, and resulted in silent miscomputation. 

To prevent this, the scheduler must be gated by a static test asserting that `def_use(@!P0 BRA P1)` yields uses `{0,1}`. However, the true defense is an on-silicon probe: forcing a stall on the `ISETP` and validating that the branch reads the correct predicate on hardware.

## 2. Fixed-Latency RAW Under-Stalls (H2)

Fixed-latency arithmetic instructions (like `FFMA` and `DFMA`) require precise cycle delays before their destination registers can be read by subsequent instructions. If a scheduler emits a `stall` instruction with a cycle count strictly below the hardware's fixed latency, the consumer will read the producer's destination register before the writeback stage completes.

Through direct hardware probing on the B200, we can measure the exact latency floors where execution transitions from incorrect to correct:
- **FFMA (Single-precision Fused Multiply-Add):** Requires a 4-cycle stall. A 3-cycle stall yields the wrong result, while a 4-cycle stall computes correctly.
- **DFMA (Double-precision Fused Multiply-Add):** Requires an 8-cycle stall. A 7-cycle stall fails.

To validate the scheduler against these latencies, we employ probe kernels that intentionally under-stall and over-stall these dependencies. A correct scheduler must target the cycle floor exactly. We use tools like `cuobjdump -sass` to verify the inserted stalls and run the resulting binaries directly on the GPU to confirm the absence of data corruption.

## 3. Variable Latency and Uncovered Scoreboards (H3)

Not all operations have predictable, fixed latencies. Memory operations (`LDG`, `LDS`), atomic operations (`ATOM`), and complex math functions (`MUFU`) have variable execution times depending on cache state, memory subsystem contention, and structural hazards.

These operations require the scheduler to manage dependencies via scoreboard barriers rather than fixed cycle stalls. The producer instruction allocates a scoreboard slot, and the consumer instruction must wait on that specific scoreboard barrier before issuing.

If an assembler strips these control barriers (for instance, in a minimal `strip` mode that forces every instruction to `stall 1`), the pipeline falls apart. Consumers read destination registers before the variable-latency memory result has landed. This invariably leads to wrong results and often triggers a `CUDA_ERROR_ILLEGAL_ADDRESS` if the stale data is used in a subsequent memory access. 

Validation here requires bulk testing: generating stripped binaries for various kernels and asserting that they either compute the wrong result or crash, proving that the scoreboard barriers in the fully assembled binaries are the sole mechanism guaranteeing correctness.

## 4. Crash-Amplified Load-Use Hazards (H4)

While fixed-latency under-stalls (H2) cause silent data corruption by reading a nearby valid—but stale—value, load-use hazards can be amplified to provide a deterministic, loud failure. This is highly useful in CI environments where binary pass/fail signals are preferred over heuristic output checking.

A load-use hazard occurs when a value intended to be used as a memory address is read before the load producing it has landed. To amplify this into a guaranteed crash, we can poison the index register with a wild constant (e.g., `0x40000000`, which offsets by +4 GiB). We maintain this poison value via a runtime-unknown guard.

```c
// 1. Poison the address register with a wild offset
uint32_t addr_reg = 0x40000000; 

// 2. Load the actual address (variable latency)
// If the scheduler misses the scoreboard wait, the next instruction
// will use the poisoned addr_reg instead of the loaded value.
addr_reg = load_actual_address(); 

// 3. Consume the address register
// A correct schedule waits for the load. An under-covered schedule reads poison.
execute_memory_operation(addr_reg); 
```

If the schedule is correct, the load lands, the valid pointer is used, and execution succeeds. If the schedule under-covers the latency, the hardware reads the poisoned register, resulting in a deterministic MMU fault (`CUDA_ERROR_ILLEGAL_ADDRESS`). This fault is safely contained by the MMU and the CUDA driver—it does not damage the hardware, and the CUDA context can simply be restarted.

## Conclusion

Building reliable compilers requires treating the hardware as the ultimate source of truth. Relying solely on static analysis or internal metrics is insufficient, as the predicate-consumer bug demonstrated. By building a registry of targeted, minimal hardware hazards and executing them on actual silicon, compiler engineers can ensure that their scheduling logic remains sound against the reality of the pipeline.

## References

[^1]: **B200 Instruction Scheduling:** Findings based on internal B200 hardware probes and `schedule.py` validation.

*Disclaimer: This article was generated using the Gemini 3.1 Pro and Claude Opus 4.8 models.*
