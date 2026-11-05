---
title: "Hacking on XLA: Building It, and Teaching It a New Pass"
date: 2026-11-04 22:00:00 -0800
categories: [Compilers, ML-Systems]
tags: [jax, xla, hlo, compilers, bazel, llvm]
---

The [XLA review]({% post_url 2026-07-23-xla-review-and-critique %}) said that when you disagree with XLA's fusion decisions, your recourse is to restructure your source or leave through `custom_call`. That is true of fusion. It isn't true of the pipeline as a whole. XLA is a C++ compiler with an ordinary pass manager, and you can build it from source and add a pass. The pass itself isn't the interesting part. Writing one forces you to learn how the pipeline decides it is done, and that matters more than any single rewrite.

Here is where this ends up. A new pass that rewrites `add(x, x)` into `multiply(x, 2)` runs inside XLA's CPU simplification pipeline:

```text
$ hlo-opt --platform=cpu --stage=hlo add.hlo     # input: ROOT a = add(x, x)
%fused_computation (param_0: f32[8,8]) -> f32[8,8] {
  %param_0 = f32[8,8]{1,0} parameter(0)
  %constant.1 = f32[] constant(2)
  %broadcast.1 = f32[8,8]{1,0} broadcast(%constant.1), dimensions={}
  ROOT %multiply.1 = f32[8,8]{1,0} multiply(%param_0, %broadcast.1)
}

ENTRY %e (x: f32[8,8]) -> f32[8,8] {
  %x = f32[8,8]{1,0} parameter(0)
  ROOT %broadcast_multiply_fusion = f32[8,8]{1,0} fusion(%x), kind=kLoop, calls=%fused_computation
}
```

The stock compiler leaves that `add` alone and emits a `wrapped_add` fusion. The rewrite itself is useless. Getting it there means meeting a migrated interface, two pass registries, and a fixed-point loop that fails quietly when two passes disagree.

Everything below was produced with XLA built from source at commit `907e39b` (2026-11-04),[^xla] plus jax 0.11.0 where JAX is involved.

## The build

XLA builds with Bazel, and only Bazel:

```bash
git clone https://github.com/openxla/xla.git && cd xla
git checkout 907e39b762e0f97937e72b62f37486e82ee01cac
python3 configure.py --backend=CPU --host_compiler=CLANG
bazelisk build //xla/tools:hlo-opt //xla/tools:run_hlo_module
```

`.bazelrc` sets `USE_HERMETIC_CC_TOOLCHAIN=1`, so Bazel downloads a pinned clang and sysroot and doesn't use the system compiler.[^bazelrc] That matters more than it looks. The build doesn't depend on your distribution's compiler, so it won't fail because your clang is too new or too old. The price is that the first build compiles LLVM and MLIR as part of the graph. On a 16-vCPU x86 machine that was also running other builds, with Bazel limited to 10 jobs at low priority, it reported:

```text
INFO: Elapsed time: 3050.074s, Critical Path: 74.76s
INFO: 13653 processes: 3801 internal, 9852 local.
INFO: Build completed successfully, 13653 total actions
```

The critical path is 75 seconds out of 51 minutes. The build is limited by throughput, not by dependency depth, so more cores help almost linearly. The two binaries are 234 MB (`hlo-opt`) and 237 MB (`run_hlo_module`). `tensorflow.bazelrc` sets `-c opt`, so these are optimized builds with `NDEBUG` defined, and debug-only checks are compiled out. Both are `testonly` Bazel targets, meant as tools. `hlo-opt` also gets GPU passes only when the build is configured for a GPU backend.[^hlo-opt-build]

## `hlo-opt` is the workbench

`hlo-opt` reads an HLO module, runs either a named list of passes or a whole compiler stage, and prints the result.[^opt-main] The flags that matter:

- `--platform`: `cpu`, `gpu`, or the default `transforms`, which has only the hardware-independent passes. At `907e39b` the CPU platform has 125 named passes, and `transforms` has 87.
- `--passes=a,b,c` runs those passes once each, in order.
- `--stage=hlo` runs the backend's whole HLO pipeline. The CPU backend also offers `hlo-backend`, `html` and `llvm-before-optimizations`.

Feed it the identity in disguise from the review post:

```text
HloModule id

ENTRY e {
  x = f32[128,128]{1,0} parameter(0)
  zero = f32[] constant(0)
  bz = f32[128,128]{1,0} broadcast(zero), dimensions={}
  add = f32[128,128]{1,0} add(x, bz)
  t1 = f32[128,128]{1,0} transpose(add), dimensions={1,0}
  t2 = f32[128,128]{1,0} transpose(t1), dimensions={1,0}
  ROOT r = f32[128,128]{1,0} multiply(t2, t2)
}
```

`--passes=algsimp` runs only the algebraic simplifier:

```text
$ hlo-opt --platform=cpu --passes=algsimp id.hlo
HloModule id, entry_computation_layout={(f32[128,128]{1,0})->f32[128,128]{1,0}}

ENTRY %e (x: f32[128,128]) -> f32[128,128] {
  %x = f32[128,128]{1,0} parameter(0)
  ROOT %r = f32[128,128]{1,0} multiply(%x, %x)
}
```

`--stage=hlo` gives the same arithmetic, wrapped in the loop fusion that the CPU emitter will turn into one kernel:

```text
$ hlo-opt --platform=cpu --stage=hlo id.hlo
%wrapped_multiply_computation (param_0: f32[128,128]) -> f32[128,128] {
  %param_0 = f32[128,128]{1,0} parameter(0)
  ROOT %r.1 = f32[128,128]{1,0} multiply(%param_0, %param_0)
}

ENTRY %e (x: f32[128,128]) -> f32[128,128] {
  %x = f32[128,128]{1,0} parameter(0)
  ROOT %wrapped_multiply = f32[128,128]{1,0} fusion(%x), kind=kLoop, calls=%wrapped_multiply_computation
}
```

Being able to run one pass and see exactly what it did is the most useful property of the tool. It answers "which pass introduced this?" without bisecting the pipeline by hand. For the full pipeline, `XLA_FLAGS="--xla_dump_to=DIR --xla_dump_hlo_pass_re=.*"` writes a snapshot only when a pass changed the module. On a small module that is 20 files, named by pipeline and pass like `0005.simplification.after_pipeline-start.before_algsimp.txt`.

### From JAX, the identity never reaches XLA

The same function in JAX gives a different result:

```python
import jax, jax.numpy as jnp

def f(x):
    # 1. (x + 0) transposed twice is x, so this is x * x in disguise.
    y = (x + 0.0).T.T
    return y * y
```

The jaxpr keeps all four operations, and so does the StableHLO that `jax.jit(f).lower(x)` emits. But the HLO that XLA receives, dumped with `--xla_dump_to`, is already:

```text
ENTRY %main.1 (x.1: f32[128,128]) -> f32[128,128] {
  %x.1 = f32[128,128]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %mul.1 = f32[128,128]{1,0} multiply(%x.1, %x.1), metadata={op_name="jit(f)/mul" stack_frame_id=2}
}
```

Converting StableHLO to HLO runs StableHLO's own target-independent simplifier first,[^shlo-import] and that removed the identity before `algsimp` saw it. So if you want to watch `algsimp` work from JAX, you need an input that the StableHLO simplifier leaves alone. An outer product written as a matmul is one:

```python
@jax.jit
def outer(a, b):
    # 1. A matmul whose contracting dimension has size 1 is an outer product.
    return a @ b
```

`XLA_FLAGS=--xla_disable_hlo_passes=algsimp` changes what the CPU backend emits for it, on an Apple-silicon Mac and on x86 alike:

```text
# default
ROOT %broadcast_multiply_fusion = f32[256,256]{1,0} fusion(%b.1, %a.1), kind=kLoop, ...
# --xla_disable_hlo_passes=algsimp
ROOT %ynn_fusion = f32[256,256]{1,0} fusion(%a.1, %b.1), kind=kCustom, ...
#   wrapping %dot_general.0 = f32[256,256]{1,0} dot(%param_0, %param_1), ...
```

With `algsimp`, the dot is first stripped of its size-1 contracting dimension and then rewritten into an elementwise multiply of two broadcasts once no contraction is left.[^algsimp-dot] The loop emitter fuses that. Without `algsimp`, the dot survives and the CPU backend hands it to the YNNPACK library as a custom fusion. That's a different kernel from a different library, chosen because one pass didn't run. At `907e39b` the flag also accepts `algsimp:2` (the third invocation), `simplification/algsimp` (the one inside a named pipeline) and `@42` (a raw pass id).[^disable-flag] It doesn't warn about names it doesn't recognize, which is now openxla/xla#52413.[^i52413]

## Writing the pass the way the tutorials do

The CPU provider in `xla/tools/hlo_opt/cpu_opt.cc` registers its passes with `RegisterPass<T>(args...)`, keyed by whatever `T(args...).name()` returns.[^cpu-opt] Adding a pass to `hlo-opt` means writing an `HloModulePass` subclass and adding one `RegisterPass` line. This pass rewrites `add(x, x)` into `multiply(x, 2)`. I picked it because XLA has no rule for it in either direction. `algsimp` leaves both `add(x, x)` and `multiply(x, broadcast(2))` alone, so if the rewrite shows up, it came from this pass.

```cpp
// add-to-mul2: rewrite add(x, x) into multiply(x, 2), f32 only.
class MulByTwoFromAdd : public HloModulePass {
 public:
  absl::string_view name() const override { return "add-to-mul2"; }
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override {
    bool changed = false;
    for (HloComputation* comp : module->computations(execution_threads)) {
      // 1. Collect matches first. Rewriting while iterating instructions()
      //    would invalidate the iteration.
      std::vector<HloInstruction*> adds;
      for (HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kAdd &&
            instr->operand(0) == instr->operand(1) &&
            instr->shape().element_type() == F32) {
          adds.push_back(instr);
        }
      }
      // 2. Build constant(2), broadcast it to the add's shape, multiply.
      for (HloInstruction* add : adds) {
        HloInstruction* x = add->mutable_operand(0);
        HloInstruction* two = comp->AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
        HloInstruction* two_b = comp->AddInstruction(
            HloInstruction::CreateBroadcast(add->shape(), two, {}));
        HloInstruction* mul = comp->AddInstruction(
            HloInstruction::CreateBinary(add->shape(), HloOpcode::kMultiply,
                                         x, two_b));
        // 3. Rewire every user of the add to the multiply, then delete it.
        ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(add, mul));
        changed = true;
      }
    }
    // 4. Report whether the module changed. A pipeline wrapped in HloPassFix
    //    uses this to decide whether to run again.
    return changed;
  }
};
```

This is the shape that most tutorials and many older passes have: a public `Run(...) override`. At `907e39b` it doesn't compile:

```text
xla/tools/hlo_opt/cpu_opt.cc:92:7: error: only virtual member functions can be marked 'override'
   92 |       override {
      |       ^~~~~~~~
In file included from xla/tools/hlo_opt/cpu_opt.cc:44:
./xla/hlo/tools/hlo_opt/opt_lib.h:86:27: error: allocating an object of abstract class type 'xla::(anonymous namespace)::MulByTwoFromAdd'
   86 |     std::string pass_name(T(args...).name());
      |                           ^
xla/tools/hlo_opt/cpu_opt.cc:238:5: note: in instantiation of function template specialization 'xla::OptProvider::RegisterPass<xla::(anonymous namespace)::MulByTwoFromAdd>' requested here
  238 |     RegisterPass<MulByTwoFromAdd>();
      |     ^
./xla/hlo/pass/hlo_pass_interface.h:119:32: note: unimplemented pure virtual method 'RunImpl' in 'MulByTwoFromAdd'
  119 |   virtual absl::StatusOr<bool> RunImpl(
      |                                ^
```

The two errors are one problem seen twice. `Run` isn't virtual, so `override` is rejected. And `RunImpl` is pure virtual and still unimplemented, so the class is abstract. `RegisterPass` then tries to construct one just to ask for its name.

## The wall: `Run` stopped being virtual a year ago

`HloPassInterface::Run` isn't virtual anymore. Commit `edcb8f08a040`, on 2025-11-04, made `Run` a public non-virtual entry point and moved the overridable part into a protected pure-virtual `RunImpl`. It did that across 483 files in one change, so every pass in the tree moved at once.[^nvi] The commit message gives the reason: "This allows us to add pre&post operations around the overridden behavior." Here is what `Run` does at `907e39b`:[^pass-interface]

```cpp
// xla/hlo/pass/hlo_pass_interface.cc
absl::StatusOr<bool> HloPassInterface::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // 1. Honor --xla_run_hlo_passes_starting_from: skip passes until the
  //    named one has started.
  if (!ShouldRunPass(module, name(), IsPassPipeline())) {
    return false;
  }

  // 2. Label the thread with the module and pass being run.
  auto* env = tsl::Env::Default();
  std::unique_ptr<tsl::ThreadNote> thread_note;
  thread_note = env->AddThreadNote(absl::StrCat("Running HLO pass on module ",
                                                module->name(), ": ", name()));

  // 3. Only now the pass's own logic.
  return RunImpl(module, execution_threads);
}
```

The first step is the payoff. `--xla_run_hlo_passes_starting_from` skips every pass before a named one. That works for every pass in the tree, including yours, because no subclass can bypass the base class anymore. It is the standard argument for the non-virtual interface pattern: the base class owns the protocol, and subclasses own only their step.

The documentation hasn't fully caught up. At `907e39b`, `xla/hlo/pass/README.md` says the interface "provides core methods like `Run`, and`RunOnChangedComputations` that passes must implement." A few paragraphs later the same file says to "implement the required virtual methods (e.g., `RunImpl`)."[^readme] That's reported as openxla/xla#52414.[^i52414] The header also has a comment that reads like a migration note: "Eventually all passes should override this method instead of Run()." That comment is on `RunOnChangedComputations`, the incremental hook that `HloPassFix` uses, and it describes a different migration that hasn't finished.

The fix is one name and one access specifier. Override `RunImpl`, and put it under `protected:`. After that, an incremental rebuild of `hlo-opt` recompiles one file and relinks:

```text
INFO: Elapsed time: 19.408s, Critical Path: 18.09s

$ hlo-opt --platform=cpu --passes=add-to-mul2 add.hlo
ENTRY %e (x: f32[8,8]) -> f32[8,8] {
  %x = f32[8,8]{1,0} parameter(0)
  %constant = f32[] constant(2)
  %broadcast = f32[8,8]{1,0} broadcast(%constant), dimensions={}
  ROOT %multiply = f32[8,8]{1,0} multiply(%x, %broadcast)
}
```

That is 19 seconds from edit to working binary, against 51 minutes for the first build. `touch`ing the file without changing it and rebuilding takes 0.9 seconds, because Bazel keys actions on content digests, not timestamps. Running `--passes=add-to-mul2,algsimp` gives the same output, so `algsimp` really has nothing to say about `multiply(x, 2)`.

## Doing it properly

The inline class works, but XLA's own passes don't look like it. A pass in the tree lives in its own library with a header. It uses `DfsHloRewriteVisitor` and the pattern matcher, and it comes with a FileCheck test. The visitor version is shorter and handles more types:

```cpp
// xla/hlo/transforms/simplifiers/add_to_mul2.cc
class AddToMul2Visitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleAdd(HloInstruction* add) override {
    HloInstruction *lhs, *rhs;
    // 1. Match add(x, x): the same instruction on both sides.
    if (!Match(add, m::Add(m::Op(&lhs), m::Op(&rhs))) || lhs != rhs) {
      return absl::OkStatus();
    }
    // 2. Floating point only. x + x and x * 2 both compute exactly 2x before
    //    rounding, so they round to the same value.
    if (!primitive_util::IsFloatingPointType(add->shape().element_type())) {
      return absl::OkStatus();
    }
    // 3. MakeScalarLike builds constant(2) in x's element type, broadcast to
    //    x's shape, so the pass works for f16, bf16, f32 and f64 alike.
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * mul,
        MakeBinaryHlo(HloOpcode::kMultiply, lhs, MakeScalarLike(lhs, 2)));
    // 4. ReplaceInstruction rewires the users and records that we changed
    //    something, which RunOnModule returns to the pipeline.
    return ReplaceInstruction(add, mul);
  }
};

absl::StatusOr<bool> AddToMul2::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return AddToMul2Visitor().RunOnModule(module, execution_threads);
}

// 5. Self-register by name in the global HloPassRegistry.
const HloPassFactory* const kAddToMul2Registration =
    RegisterHloPassFactory<DefaultHloPassFactory<AddToMul2>>("add-to-mul2");
```

The rewrite preserves values exactly. Doubling a float is exact unless it overflows, and when it overflows both forms give the same infinity. Signed zeros keep their sign in both forms, and NaNs stay NaNs. Integers are left alone only so that the test can show what "no change" looks like. The test uses the FileCheck helper from `HloHardwareIndependentTestBase`:

```cpp
TEST_F(AddToMul2Test, RewritesBf16Doubling) {
  const char* hlo = R"(
HloModule m

ENTRY e {
  x = bf16[4,8]{1,0} parameter(0)
  ROOT a = bf16[4,8]{1,0} add(x, x)
}
)";
  // 1. The constant must come out in bf16, not f32.
  RunAndFilecheckHloRewrite(hlo, AddToMul2{}, R"(
// CHECK: [[x:%[^ ]+]] = bf16[4,8]{1,0} parameter(0)
// CHECK: [[two:%[^ ]+]] = bf16[] constant(2)
// CHECK: [[b:%[^ ]+]] = bf16[4,8]{1,0} broadcast([[two]]), dimensions={}
// CHECK: ROOT [[m:%[^ ]+]] = bf16[4,8]{1,0} multiply([[x]], [[b]])
)");
}

TEST_F(AddToMul2Test, LeavesIntegersAlone) {
  const char* hlo = R"(
HloModule m

ENTRY e {
  x = s32[16]{0} parameter(0)
  ROOT a = s32[16]{0} add(x, x)
}
)";
  // 2. std::nullopt as the expectation means "the pass must not change it".
  RunAndFilecheckHloRewrite(hlo, AddToMul2{}, std::nullopt);
}

TEST_F(AddToMul2Test, IsInTheGlobalRegistry) {
  // 3. alwayslink keeps the static registration from being dropped by the
  //    linker, so the name resolves without naming the class.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloPassInterface> pass,
                       HloPassRegistry::Global().Create("add-to-mul2"));
  EXPECT_EQ(pass->name(), "add-to-mul2");
}
```

```text
$ bazelisk test //xla/hlo/transforms/simplifiers:add_to_mul2_test --test_output=all
[ RUN      ] AddToMul2Test.RewritesBf16Doubling
[       OK ] AddToMul2Test.RewritesBf16Doubling (4 ms)
[ RUN      ] AddToMul2Test.IsInTheGlobalRegistry
[       OK ] AddToMul2Test.IsInTheGlobalRegistry (0 ms)
[ RUN      ] AddToMul2Test.LeavesIntegersAlone
[       OK ] AddToMul2Test.LeavesIntegersAlone (0 ms)
[  PASSED  ] 3 tests.
INFO: Elapsed time: 20.258s, Critical Path: 11.81s
//xla/hlo/transforms/simplifiers:add_to_mul2_test                        PASSED in 0.0s
```

The FileCheck patterns bind names with `[[x:%[^ ]+]]` and reuse them with `[[x]]`. So the test checks that the multiply consumes the parameter and the broadcast, not just that some multiply exists. The BUILD rule is ordinary: a `cc_library` with the pass's dependencies, plus `alwayslink = True`, for the reason below.

The third test covers the newest part of the pass framework. `HloPassRegistry`, added on 2026-10-27, maps pass names to factories, and 29 passes in the tree registered themselves by `907e39b`.[^registry] Registration is a static initializer, so the library needs `alwayslink = True`. Without it, nothing references the object file and the linker drops it, registration included. `hlo-opt` doesn't use the global registry yet. It keeps its own name-to-lambda map, filled by `RegisterPass` calls. Once the inline pass is gone and the library is linked in, `hlo-opt` can't find the name:

```text
$ hlo-opt --platform=cpu --passes=add-to-mul2 add.hlo
INVALID_ARGUMENT: Pass add-to-mul2 not found.
=== Source Location Trace: ===
xla/hlo/tools/hlo_opt/opt_lib.cc:191
```

Two registries now exist side by side, and a pass in one is invisible to the other. That is the normal state of a migration in progress. Expect `hlo-opt` to read the global registry eventually, but don't assume it does today.

## Putting it in the real pipeline

`hlo-opt --passes` runs each pass once. The real CPU compiler doesn't. Its simplification pipeline is an `HloPassFix<HloPassPipeline>` that reruns until a whole iteration changes nothing. The cap is 25 iterations, or 4 with the fast-compile preset:[^cpu-pipeline]

```cpp
// xla/service/cpu/cpu_compiler.cc, CreateSimplificationPipeline
const int iteration_limit = fast_compile ? 4 : 25;

auto pipeline =
    HloPassFix<HloPassPipeline>::Create(iteration_limit, std::string(name));
...
pipeline->AddPass<AlgebraicSimplifier>(options);
pipeline->AddPass<AddToMul2>();        // 1. The new pass, right after algsimp.
pipeline->AddPass<SortSimplifier>();
pipeline->AddPass<HloDCE>();
```

The CPU compiler builds this pipeline twice, as `simplification` and as `post_scatter_expansion_simplification`, with the scatter expanders in between. A pass's return value is how `HloPassFix` decides whether to go around again. `TF_CPP_VMODULE=hlo_pass_fix=3` logs every iteration. For the outer product on the stock build, the first iteration changes something and the second confirms that nothing else will:

```text
hlo_pass_fix.h:99] simplification iteration 0 changed_this_iteration: true
hlo_pass_fix.h:99] simplification iteration 1 changed_this_iteration: false
```

So even a single rewrite costs one extra trip through the whole pipeline, just to prove it was the last one. With the new pass inserted after `algsimp`, a module that negates `add(x, x)` behaves the same way. The rewrite happens in iteration 0, and iteration 1 confirms the fixed point. Dumping around the pass with `--xla_dump_hlo_pass_re=add-to-mul2` shows it running in every iteration of both simplification pipelines:

```text
module_0000.fight.0000.simplification.after_algsimp.before_add-to-mul2.txt
module_0000.fight.0001.simplification.after_add-to-mul2.before_simplify-sorts.txt
module_0000.fight.0002.simplification.after_algsimp.before_add-to-mul2.txt
module_0000.fight.0003.simplification.after_add-to-mul2.before_simplify-sorts.txt
module_0000.fight.0004.post_scatter_expansion_simplification.after_algsimp.before_add-to-mul2.txt
module_0000.fight.0005.post_scatter_expansion_simplification.after_add-to-mul2.before_simplify-sorts.txt
```

`run_hlo_module` runs the module on the CPU backend and on the interpreter and compares the results. The interpreter doesn't use the CPU pipeline, so it is an independent check that the rewrite preserved the answer:

```text
$ run_hlo_module --platform=cpu --reference_platform=interpreter fight.hlo
... compiled and ran in 0.01736s.
... compiled and ran in 0.0012462s.
** Results on cpu and interpreter are close enough. **
0/1 runs failed.
```

## When two passes disagree

Returning `true` is a promise that the module made progress. Here is what happens when the promise is false. Add a second pass, `Mul2ToAdd`, that undoes the first:

```cpp
// 1. The inverse rewrite. It exists only to make the fixed point unreachable.
absl::Status HandleMultiply(HloInstruction* mul) override {
  HloInstruction* x;
  if (!Match(mul, m::MultiplyAnyOrder(
                      m::Op(&x), m::Broadcast(m::ConstantScalar(2))))) {
    return absl::OkStatus();
  }
  ABSL_ASSIGN_OR_RETURN(HloInstruction * add,
                        MakeBinaryHlo(HloOpcode::kAdd, x, x));
  return ReplaceInstruction(mul, add);
}
```

Put both passes in the simplification pipeline, and every iteration ends with exactly the module it started with.

By default the compile succeeds and prints nothing. The output is correct, and its one visible scar is a name. The final `add` is called `%add.50`, because it was rebuilt 50 times:

```text
$ hlo-opt --platform=cpu --stage=hlo fight.hlo
%fused_computation (param_0.1: f32[8,8]) -> f32[8,8] {
  %param_0.1 = f32[8,8]{1,0} parameter(0)
  %add.50 = f32[8,8]{1,0} add(%param_0.1, %param_0.1)
  ROOT %n.1 = f32[8,8]{1,0} negate(%add.50)
}
```

`TF_CPP_VMODULE=hlo_pass_fix=3` shows where the 50 came from. Both simplification pipelines ran into the cap:

```text
hlo_pass_fix.h:99] simplification iteration 0 changed_this_iteration: true
hlo_pass_fix.h:99] simplification iteration 1 changed_this_iteration: true
...
hlo_pass_fix.h:99] simplification iteration 24 changed_this_iteration: true
hlo_pass_fix.h:111] Unexpectedly high number of iterations in HLO passes 'simplification' for module 'fight'. Exiting fixed point loop.
hlo_pass_fix.h:99] post_scatter_expansion_simplification iteration 0 changed_this_iteration: true
...
```

The code that handles the cap explains the silence:[^pass-fix]

```cpp
// xla/hlo/pass/hlo_pass_fix.h, RunToFixPoint
run_state->IncrementIteration();
if (run_state->iteration == iteration_limit_) {
  const DebugOptions& debug_options = module->config().debug_options();
  // 1. Opt-in: abort instead of giving up.
  if (debug_options
          .xla_unsupported_crash_on_hlo_pass_fix_max_iterations()) {
    LOG(FATAL) << "Unexpectedly high number of iterations "
               << iteration_limit_ << " in HLO pass '" << Pass::name()
               << "' for module '" << module->name() << "'";
  }
  // 2. Default: a verbose log line, invisible unless VLOG >= 1 is enabled.
  VLOG(1) << "Unexpectedly high number of iterations in HLO passes '"
          << Pass::name() << "' for module '" << module->name()
          << "'. Exiting fixed point loop.";
  // 3. Report "no change" upward, so an enclosing fixed point stops too.
  if (!debug_options
           .xla_unsupported_crash_on_hlo_pass_silent_hlo_change()) {
    // Clear changed and abort in case this is fixed point is nested.
    run_state->changed.clear();
  }
  break;
}
```

The third step is the interesting one. After 25 iterations that each changed the module, the loop reports that it changed nothing. That's a deliberate choice. If this loop sits inside another fixed-point loop, reporting "changed" would make the outer loop rerun it 25 more times, and so on up the nest. It does mean that by default, a pass returning `true` forever is indistinguishable from a pipeline that converged. Two debug options make it visible:

```text
$ XLA_FLAGS=--xla_hlo_pass_fix_detect_cycles=true hlo-opt --platform=cpu --stage=hlo fight.hlo
W hlo_pass_fix.h:90] Cycle detected when running simplification on iteration 1; hash: 10622735922843943528
W hlo_pass_fix.h:90] Cycle detected when running simplification on iteration 2; hash: 10622735922843943528
...

$ XLA_FLAGS=--xla_unsupported_crash_on_hlo_pass_fix_max_iterations=true hlo-opt --platform=cpu --stage=hlo fight.hlo
F hlo_pass_fix.h:107] Unexpectedly high number of iterations 25 in HLO pass 'simplification' for module 'fight'
```

Cycle detection hashes the module at the start of each iteration. The hash ignores instruction names, so the module after a round trip hashes the same as before it. The warning fired 48 times, 24 for each pipeline, from the second iteration on. The cost is real but small on toy inputs. On a chain of 200 doublings, the median of five `--stage=hlo` runs was 18 ms for the stock compiler, 20 ms with `AddToMul2` alone, and 70 ms with both passes. The machine was shared, so treat the ratio as indicative. It's a 3.5x compile-time penalty with no warning at the default log level and a correct answer at the end. Nothing prompts anyone to look for it.

## Pipelines as data

The newest part of the framework moves pipelines out of C++. `HloPipelineProto`, added on 2026-10-26, describes a pipeline as a tree of nodes. Each node is either a pass named by its key in the registry or a sequence of nodes, and every node can carry its own fixed-point limit:[^pipeline-proto]

```protobuf
message HloPipelineNodeProto {
  // Pass: key in the pass registry.
  // Node sequence: label used in HLO dumps and pass metadata.
  string name = 1;
  // Set iff the node is a node sequence rather than a pass. May be empty.
  HloPipelineNodeSequenceProto sub_nodes = 2;
  // Pass options. Must be empty for a node sequence.
  map<string, HloPassOptionValue> options = 3;
  // N <= 1: no fixed-point behavior, i.e. run once. N > 1: rerun until the
  // module stops changing, at most N times.
  int32 max_fixed_point_iterations = 4;
}
```

`HloPassFix<HloPassPipeline>` with a limit of 25 becomes a sequence node with `max_fixed_point_iterations: 25`. The registry supplies the passes by name, and `BuildHloPassPipeline` turns the proto into a runnable pipeline. At `907e39b` the GPU compiler is the only user. On 2026-10-30 it got a default proto pipeline, built with an `HloPipelineEditor`, that covers host-compute annotation and the passes before SPMD partitioning.[^gpu-pipeline] The CPU compiler and `hlo-opt` still build their pipelines in C++.

I think this is the right direction, and the experiment above is the reason. A fixed-point group whose members undo each other is a property of the pipeline, not of any one pass. When the pipeline is C++, you find that with a debugger. When the pipeline is data, a tool can check it. For example, it can run each fixed-point group on a corpus with cycle detection on and flag the groups that never settle. Data pipelines would also give `add-to-mul2` a way into a backend without editing `cpu_compiler.cc`. None of that exists yet. 29 passes are registered, and `hlo-opt` lists 125 for the CPU platform alone.

## The review's claims, from the inside

With a working `hlo-opt`, three claims from the review post become things you can watch.

**Sharding is an annotation that a separate pass consumes.** Here is a `dot` whose contracting dimension is split across two partitions, run through only the partitioner:

```text
$ hlo-opt --platform=cpu --passes=spmd-partitioning spmd.hlo
ENTRY %e_spmd (param: f32[256,256], param.1: f32[256,1024]) -> f32[256,1024] {
  %param = f32[256,256]{1,0} parameter(0)
  %param.1 = f32[256,1024]{1,0} parameter(1)
  %dot = f32[256,1024]{1,0} dot(%param, %param.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %all-reduce = f32[256,1024]{1,0} all-reduce(%dot), channel_id=1, replica_groups=mesh['axis_0'=2] {'axis_0'}, use_global_device_ids=true, to_apply=%add.clone, frontend_attributes={is_spmd_generated="true"}
}
```

The pass halved K from 512 to 256 per partition, created a reduction computation, and inserted an `all-reduce` that it labels `is_spmd_generated="true"`. The replica groups now use the mesh-axis notation rather than an explicit device list. The sharding started as an attribute on the parameters. One pass read it and rewrote the program.

**Dynamic shapes are static shapes with a wrapper.** A bounded dynamic dimension goes through the full CPU pipeline like this:

```text
$ hlo-opt --platform=cpu --stage=hlo dyn.hlo     # input: negate(f32[<=128,8])
  %p.padded = (f32[128,8]{1,0}, s32[], s32[]) custom-call(%p), custom_call_target="PadToStatic"
  %wrapped_negate = f32[128,8]{1,0} fusion(%p.padded#0), kind=kLoop, calls=%wrapped_negate_computation
  %wrapped_clamp = s32[] fusion(%constant, %p.padded#1, %constant.1), kind=kLoop, calls=%wrapped_clamp_computation
  ROOT %custom-call = f32[<=128,8]{1,0} custom-call(%wrapped_negate, %wrapped_clamp, %constant.2), custom_call_target="SliceToDynamic"
```

The tensor is padded to its static bound, the work runs on `f32[128,8]`, and the result is sliced back. The runtime size is clamped to `[0, 128]` on the way out.

**`custom_call` is a fusion barrier.** A custom call between two elementwise regions:

```text
$ hlo-opt --platform=cpu --stage=hlo cc.hlo
  %broadcast_multiply_fusion = f32[1024]{0} fusion(%x), kind=kLoop, calls=%fused_computation.1
  %c = f32[1024]{0} custom-call(%broadcast_multiply_fusion), custom_call_target="MyOpaqueThing"
  ROOT %multiply_add_fusion = f32[1024]{0} fusion(%x, %c), kind=kLoop, calls=%fused_computation
```

There are two fusions, one on each side, and nothing fuses through the call.

## What this tells you about XLA

XLA's pass framework is easy to extend. A rewrite is a visitor method, the pattern matcher and `MakeBinaryHlo` handle the plumbing, and the FileCheck helper turns a test into a dozen lines. The incremental loop of edit, rebuild and run takes under a minute once the first build is done. If XLA is missing a rewrite you need, the architecture isn't what stops you.

It's also a codebase in motion, and the documentation trails the code. `Run` stopped being the override point a year ago, and the README still says both things. The global registry is eight days old at `907e39b`, and the tool you'd use to try your pass doesn't read it yet. Copying an existing pass as a template is the right instinct, as long as the pass you copy was written after the migration you're relying on.

The most important part of a pass is the `bool` it returns. `HloPassFix` trusts it completely. A pass that reports progress it didn't make, or two passes that undo each other, costs 25 iterations per pipeline and produces no output at the default log level. If I were adding a pass to a fixed-point group, I would run the test suite with `--xla_hlo_pass_fix_detect_cycles=true` and `--xla_unsupported_crash_on_hlo_pass_fix_max_iterations=true`. The first flag warns and the second fails loudly, and both cost nothing in production.

The review's structural claims hold up from the inside. Sharding is an annotation that a pass turns into collectives. Dynamic shapes are padded to static ones. `custom_call` is a wall. None of that is a flaw a 40-line pass can fix, because it is the shape of the design. What a from-source build buys you is the ability to stop taking any of it on faith. The [companion post]({% post_url 2026-11-04-ten-ways-to-break-xla %}) uses the same build to find where that shape breaks.

## References

[^xla]: **openxla/xla at `907e39b`.** ([Link](https://github.com/openxla/xla/tree/907e39b762e0f97937e72b62f37486e82ee01cac))

[^bazelrc]: **`.bazelrc`.** `common --repo_env USE_HERMETIC_CC_TOOLCHAIN=1`. ([Link](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/.bazelrc#L15))

[^hlo-opt-build]: **`//xla/tools:hlo-opt`.** A `testonly` `xla_cc_binary` whose GPU provider is added only under `if_gpu_is_configured`. ([Link](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/tools/BUILD#L91-L105))

[^opt-main]: **`xla/hlo/tools/hlo_opt/opt_main.cc`.** The `hlo-opt` flags and their defaults. ([Link](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/hlo/tools/hlo_opt/opt_main.cc#L63-L76))

[^shlo-import]: **`xla/hlo/translate/stablehlo.cc`.** The StableHLO-to-HLO conversion and the StableHLO simplification it runs first. ([Link](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/hlo/translate/stablehlo.cc))

[^algsimp-dot]: **`AlgebraicSimplifierVisitor::HandleDot`.** `RemoveDegenerateDimensionFromDot` and the rule that rewrites a dot with no contracting dimensions as `mul(broadcast(transpose(x)), broadcast(transpose(y)))`. ([Link](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4305-L4312))

[^disable-flag]: **`xla_disable_hlo_passes`.** The flag's help text in `xla/debug_options_flags.cc`. ([Link](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/debug_options_flags.cc#L1532-L1540))

[^cpu-opt]: **`xla/tools/hlo_opt/cpu_opt.cc`.** The CPU `OptProvider` and its `RegisterProviderPasses` list. ([Link](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/tools/hlo_opt/cpu_opt.cc#L111-L192))

[^nvi]: **Commit `edcb8f08a040`.** "Implement non-virtual interface (NVI) pattern for HloPassInterface's Run & RunOnModuleGroup functions. This allows us to add pre&post operations around the overridden behavior." 2025-11-04, 483 files. ([Link](https://github.com/openxla/xla/commit/edcb8f08a0409dbc14fcbcc0433bad661a715c80))

[^pass-interface]: **`xla/hlo/pass/hlo_pass_interface.cc` and `.h`.** The non-virtual `Run`, `ShouldRunPass`, and the protected `RunImpl`. ([cc](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/hlo/pass/hlo_pass_interface.cc#L33-L68), [h](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/hlo/pass/hlo_pass_interface.h#L76-L122))

[^readme]: **`xla/hlo/pass/README.md`.** Line 16 says passes implement `Run`; line 51 says `RunImpl`. ([Link](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/hlo/pass/README.md?plain=1#L16-L51))

[^i52413]: **openxla/xla#52413.** "`--xla_disable_hlo_passes` silently ignores entries that match no pass." Filed 2026-11-05. ([Link](https://github.com/openxla/xla/issues/52413))

[^i52414]: **openxla/xla#52414.** "hlo/pass/README.md says passes implement Run; since the NVI change they implement RunImpl." Filed 2026-11-05. ([Link](https://github.com/openxla/xla/issues/52414))

[^registry]: **`HloPassRegistry`.** Added in `ae2e068d` ("Add HloPassRegistry to create HLO passes by name"), 2026-10-27. ([Commit](https://github.com/openxla/xla/commit/ae2e068d642b3822aed0542bfc70e702bd1c1c92), [header](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/hlo/pass/hlo_pass_registry.h))

[^cpu-pipeline]: **`CreateSimplificationPipeline` in `xla/service/cpu/cpu_compiler.cc`.** ([Link](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/service/cpu/cpu_compiler.cc#L374-L428))

[^pass-fix]: **`xla/hlo/pass/hlo_pass_fix.h`.** `RunToFixPoint`, the iteration cap and `kDefaultIterationLimit = 25`. ([Link](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/hlo/pass/hlo_pass_fix.h#L78-L151))

[^pipeline-proto]: **`HloPipelineProto`.** Added in `bd2b8778` ("Add HloPipelineProto for declarative HLO pass pipelines"), 2026-10-26. ([Commit](https://github.com/openxla/xla/commit/bd2b8778306b2b3555d8cb96264cafc8fdd9504b), [proto](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/hlo/pass/hlo_pipeline.proto#L28-L42))

[^gpu-pipeline]: **`GetDefaultCompilationPipeline`.** Added in `8bc0dd33` ("Add GetDefaultCompilationPipeline in gpu_pipeline"), 2026-10-30, and run through `RunHloPassPipeline` in the GPU compiler. ([Commit](https://github.com/openxla/xla/commit/8bc0dd33091c6e3b7a6ead7f076872678d88aaf9), [gpu_pipeline.cc](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/service/gpu/gpu_pipeline.cc#L51-L59), [gpu_compiler.cc](https://github.com/openxla/xla/blob/907e39b762e0f97937e72b62f37486e82ee01cac/xla/service/gpu/gpu_compiler.cc#L701-L711))

---

*Disclaimer: Researched and drafted with AI assistance (Gemini 3.1 Pro, Claude Opus 4.8, Claude Opus 5.5, Gemini 4 Argon). Direction, technical judgment, and final edits are mine; every claim is traceable to the sources cited above.*
