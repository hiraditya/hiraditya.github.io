---
title: "How JAX Shards a Computation Across a Mesh"
date: 2026-07-30 00:00:00 -0700
categories: [Compilers, ML-Systems]
tags: [jax, xla, gspmd, shardy, sharding, distributed]
---

I have eight devices. I want a matmul whose contracting dimension is split across some of them. So I write the matmul normally, say where the two operands live, and compile:

```python
mesh = Mesh(np.array(jax.devices()).reshape(4, 2), ('data', 'model'))
xs = jax.device_put(x, NamedSharding(mesh, P('data', 'model')))   # K split on 'model'
Ws = jax.device_put(W, NamedSharding(mesh, P('model', None)))     # K split on 'model'
compiled = jax.jit(lambda a, b: jnp.tanh(a @ b)).lower(xs, Ws).compile()
```

I never wrote a collective. The compiled program has one anyway:

{% raw %}
```text
%all-reduce = f32[64,1024]{1,0} all-reduce(%ynn_fusion), channel_id=1,
    replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true,
    to_apply=%add.clone, frontend_attributes={is_spmd_generated="true"}
```
{% endraw %}

Because I sharded the contracting dimension, each device holds a *partial* product, and the result is only correct after summing across the shard. The compiler worked that out, grouped the eight devices into the four `model`-pairs — {% raw %}`{{0,1},{2,3},{4,5},{6,7}}`{% endraw %} — inserted the all-reduce, and tagged it `is_spmd_generated="true"` so you can tell its work from yours. That is the whole value proposition: you say *where the data is*, and the compiler derives *what communication is required*.

This is the third post on the JAX/XLA stack. The [first]({% post_url 2026-07-23-jax-tracing-and-jaxpr %}) covered how Python becomes a jaxpr; the tour of [XLA]({% post_url 2026-07-24-a-tour-of-xla-where-mlir-lives %}) and its [rigid price]({% post_url 2026-07-23-xla-review-and-critique %}) covered what happens to that jaxpr on one device. This one is the step neither covered.

I drafted it expecting to end on a complaint — that placement in JAX is an annotation hanging off a value rather than part of its type, so placement mistakes get caught by a propagator or a runtime instead of a type checker. That was true when I started paying attention. It is not true now, and finding out how thoroughly it stopped being true was the most interesting thing about writing this. Everything below runs on JAX 0.11.0 with eight CPU devices.[^1]

## The vocabulary, and the one part that changed

Three types carry the model.

A **`Mesh`** is a named grid of devices. Eight devices arranged 4×2, with the axes named `data` and `model`. From there on you never name a physical device; you name mesh axes. That indirection is the good idea, because it decouples the logical parallelism from the count of chips you happen to have.

A **`PartitionSpec`** — written `P(...)` — maps each dimension of an array to a mesh axis, or to `None` for replicated. `P('data', 'model')` says split dimension 0 across `data`, dimension 1 across `model`. `P('model', None)` says split dimension 0 across `model` and replicate dimension 1.

A **`NamedSharding`** is a `Mesh` plus a `PartitionSpec`: concretely how this array is laid out on this grid.

The part that changed is a fourth thing, and it is easy to miss because it lives in a keyword argument. Every mesh axis has an **axis type** — `Auto`, `Explicit`, or `Manual`.[^2] The axis type decides whether placement shows up in the type of a traced value, and the two mesh constructors do not agree on the default:

```python
Mesh(np.array(jax.devices()).reshape(4, 2), ('data', 'model')).axis_types
# (Auto, Auto)

jax.make_mesh((4, 2), ('data', 'model')).axis_types
# (Explicit, Explicit)
```

`Mesh` is the old low-level constructor and it gives you `Auto`. `jax.make_mesh` is the one the docs now steer you to, and it gives you `Explicit`. Those two lines produce grids of the same eight devices that behave differently under tracing. Nearly every sharding tutorial written before about a year ago uses the first one.

## Auto: GSPMD infers, and the type stays quiet

Start with `Auto`, because it is the classical behaviour and the baseline for everything else. Shard the inputs and look at what the tracer knows:

```python
mesh = Mesh(np.array(jax.devices()).reshape(4, 2), ('data', 'model'))
xs = jax.device_put(jnp.ones((256, 512)), NamedSharding(mesh, P('data', 'model')))
print(jax.typeof(xs))
```

```text
float32[256,512]
```

Dtype and shape. The array is physically distributed over eight devices right now, and its type says nothing about that. Placement is real but invisible to the front end — which is exactly the situation I expected to be writing about.

The system that makes it work anyway is **GSPMD**, and it runs inside XLA as part of `jit`.[^3] It does two jobs. **Propagation**: given shardings on some values, infer a consistent sharding for every other value, the way a type inferencer propagates types. **Partitioning**: rewrite the single logical program into the per-device SPMD program, inserting collectives so the sharded computation equals the original.

Both are visible in the compiled HLO. Here is the whole entry computation for `tanh(x @ W)`, eight partitions:

{% raw %}
```text
ENTRY %main.0_spmd (param: f32[64,256], param.1: f32[256,1024]) -> f32[64,1024] {
  %param = f32[64,256]{1,0} parameter(0), sharding={devices=[4,2]<=[8]}
  %param.1 = f32[256,1024]{1,0} parameter(1),
      sharding={devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}
  %ynn_fusion = f32[64,1024]{1,0} fusion(%param, %param.1), kind=kCustom, ...
  %all-reduce = f32[64,1024]{1,0} all-reduce(%ynn_fusion), channel_id=1,
      replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true,
      to_apply=%add.clone, frontend_attributes={is_spmd_generated="true"}
  ROOT %wrapped_tanh = f32[64,1024]{1,0} fusion(%all-reduce), kind=kLoop, ...
}
```
{% endraw %}

The shapes are per-device. Logical `x` was 256×512; each device gets `f32[64,256]` — the batch split four ways over `data`, the contracting dimension split two ways over `model`. The entry computation is even renamed `main.0_spmd` to mark that it is post-partitioning.

Then read the second parameter's sharding out loud: `{devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}`. Two tiles along dimension 0, one along dimension 1, four replicas; device order obtained by transposing the 4×2 grid; trailing dimension means replication. It is correct, it is complete, and it is a puzzle. Which brings us to the second thing that changed.

## Shardy is not the future; it is the default

The known complaint about GSPMD is that propagation is powerful and its decisions are opaque. When it shards something in a way you did not intend, working out *why* is genuinely hard, and the notation above is a large part of the reason.

The answer is **Shardy**, an MLIR-based partitioner (dialect `sdy`) built jointly by the GSPMD and PartIR teams.[^4] I had it filed as the direction of travel. It is already the default: `jax_use_shardy_partitioner` is declared with `default=True` and `upgrade=True`, pointing at a migration guide.[^5]

The difference is worth seeing directly, because it is not a refactor. Same program, same shardings, lowered both ways:

```python
jax.config.update('jax_use_shardy_partitioner', True)
print(jax.jit(lambda a, b: jnp.tanh(a @ b)).lower(xs, Ws).as_text())
```

```text
module @jit__lambda attributes {mhlo.num_partitions = 8 : i32, ...} {
  sdy.mesh @mesh = <["data"=4, "model"=2]>
  func.func public @main(
      %arg0: tensor<256x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>},
      %arg1: tensor<512x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}]>}
  ) -> ... {
```

And with Shardy off:

```text
module @jit__lambda attributes {mhlo.num_partitions = 8 : i32, ...} {
  func.func public @main(
      %arg0: tensor<256x512xf32> {mhlo.sharding = "{devices=[4,2]<=[8]}"},
      %arg1: tensor<512x1024xf32> {mhlo.sharding = "{devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}"}
  ) -> ... {
```

The old sharding is a **string**. `mhlo.sharding = "{devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}"` is an opaque attribute that MLIR carries around and cannot inspect; every tool that wants to know what it means has to parse it. The Shardy version declares the mesh once as a named symbol, `sdy.mesh @mesh = <["data"=4, "model"=2]>`, and each sharding is a structured attribute referring to it by name, with the axis names written out: `[{"data"}, {"model"}]`.

That is the entire "why Shardy" argument, and it is a compiler-engineering argument rather than a features argument. Sharding stopped being a comment the compiler passes through and became part of the IR the compiler is made of. Nothing about the model changed — mesh axes, dimension-to-axis mappings, propagation, collective insertion. It got representable.

## What `Auto` will not tell you

Propagation always succeeds. That is the feature, and it is also the failure mode: there is no sharding you can hand GSPMD that it will refuse, only shardings that make it insert more communication than you wanted. Same `tanh(x @ W)`, four different placements, counting the collectives in the compiled program:

| `x` | `W` | collectives inserted |
|---|---|---|
| `P('data', None)` | `P(None, None)` | none |
| `P('data', None)` | `P(None, 'model')` | none |
| `P('data', 'model')` | `P('model', None)` | `all-reduce` |
| `P('model', 'data')` | `P(None, None)` | `all-gather` |

The first two are free. The third is tensor parallelism working as intended — you split the contraction, so you pay one all-reduce to sum the partials. The fourth is a bug, and it is the interesting one:

{% raw %}
```text
%param   = f32[128,128]{1,0} parameter(0), sharding={devices=[2,4]<=[4,2]T(1,0)}
%param.1 = f32[512,1024]{1,0} parameter(1), sharding={replicated}
%all-gather = f32[128,512]{0,1} all-gather(%wrapped_copy), channel_id=1,
    replica_groups=[2,4]<=[4,2]T(1,0), dimensions={1},
    use_global_device_ids=true, frontend_attributes={is_spmd_generated="true"}
```
{% endraw %}

I sharded `x`'s feature dimension over `data` but left `W` replicated. The contraction needs all 512 features against a full copy of `W`, so the partitioner reconstitutes `x` — `f32[128,128]` back up to `f32[128,512]` — before it can do the matmul. The sharding bought nothing and cost an all-gather of the entire activation tensor on every call.

Nothing in that program is an error. It compiles, it produces correct numbers, and the only evidence is an `all-gather` you have to go looking for in HLO you have to know how to dump. This is what people mean when they say GSPMD is hard to debug: not that it makes bad decisions, but that its decisions are correct-by-construction and expensive-by-accident, and it has no way to tell you which of the two you asked for.

## Explicit: placement moves into the type

Now switch constructors. `jax.make_mesh` gives `Explicit` axes, and the same `device_put` produces a different type:

```python
mesh = jax.make_mesh((4, 2), ('data', 'model'))       # Explicit by default
with jax.sharding.set_mesh(mesh):
    xs = jax.device_put(jnp.ones((256, 512)), NamedSharding(mesh, P('data', 'model')))
    Ws = jax.device_put(jnp.ones((512, 1024)), NamedSharding(mesh, P('model', None)))
    print(jax.typeof(xs), '|', jax.typeof(Ws))
```

```text
float32[256@data,512@model] | float32[512@model,1024]
```

`256@data` — 256 rows distributed over mesh axis `data`. Placement is in the type, per axis, and it is there because `ShapedArray` has a `sharding` slot alongside `shape` and `dtype`, which is the thread the [previous post]({% post_url 2026-07-23-jax-tracing-and-jaxpr %}) pulled on.

A type is only interesting if something checks it. Multiply those two arrays:

```python
make_jaxpr(lambda a, b: jnp.tanh(a @ b))(xs, Ws)
```

```text
ShardingTypeError: Contracting dimensions are sharded and it is ambiguous how
the output should be sharded. Please specify the output sharding via the
`out_sharding` parameter.
Got lhs_contracting_spec=('model',) and rhs_contracting_spec=('model',)
```

The shapes are fine. `(256,512) @ (512,1024)` contracts cleanly. What does not check is where the operands live: contracting a `model`-sharded axis against a `model`-sharded axis leaves a result that needs a reduction, and JAX will not pick the output placement for you. `ShardingTypeError` is a real exception class in `core.py`, raised from 53 sites.[^6]

This is the same program that compiled silently under `Auto`. Under `Auto`, GSPMD made the choice; under `Explicit`, you make it, and the type propagates:

```python
def f(a, b):
    return jnp.tanh(jax.lax.dot_general(a, b, (((1,), (0,)), ((), ())),
                                        out_sharding=P('data', None)))
print(make_jaxpr(f)(xs, Ws))
```

```text
{ lambda ; a:f32[256@data,512@model] b:f32[512@model,1024]. let
    c:f32[256@data,1024] = dot_general[dimension_numbers=(([1], [0]), ([], []))] a b
    d:f32[256@data,1024] = tanh c
  in (d,) }
```

Every binder carries its placement and `tanh` propagates it. Compile this and compare it against the `Auto` version: the two modules are identical apart from instruction numbering — the same fusions, the same all-reduce over the same four replica groups. You get the same program either way.

One line of the HLO header does change, and it is the interesting one:[^7]

```text
Auto:     allow_spmd_sharding_propagation_to_parameters={false,false},
          allow_spmd_sharding_propagation_to_output={true}, num_partitions=8
Explicit: allow_spmd_sharding_propagation_to_parameters={false,false},
          num_partitions=8
```

Under `Auto`, JAX grants XLA permission to choose where the result lives. Under `Explicit` it withdraws that permission, because the type already said. The latitude the compiler is given shrinks by exactly the amount the type now carries.

The tradeoff is a real and I do not think `Explicit` is free. `Auto` lets you shard two inputs and walk away; the propagator handles a thousand-line model and usually gets it right. `Explicit` asks you to resolve every ambiguity it finds, and a large model has more of them than you expect. What you buy is that the ambiguities surface where you wrote them, with the mesh axis named.

Be precise about what that does and does not buy you, because it is easy to over-read. Take the accidental all-gather from two sections ago — batch on `model`, features on `data`, weights replicated — and run it on an `Explicit` mesh:

```text
avals: float32[256@model,512@data] | float32[512,1024]
traces cleanly, out aval float32[256@model,1024]
compiled program: 3 all-gathers
```

No error. That sharding is unambiguous, consistently typed, and slow, and a type checker has no opinion about the third one. `ShardingTypeError` fires when a program is *ill-typed* — contracting two axes sharded the same way, broadcasting incompatible specs — not when it is merely expensive. The class of bug that moved from runtime to trace time is wrong-or-undefined placement. The class that did not move is placement that type-checks and costs you an extra collective on every step. Still yours to find, still in the profile.

## Manual: `shard_map`, and a second placement type

There is a third regime. Sometimes you want to write the per-device program yourself and call the collectives by hand, because you know the communication pattern better than a propagator does. That is `shard_map`, and its axes are `Manual`.[^8]

```python
def g(a, b):
    def per_shard(ash, bsh):
        return jax.lax.psum(ash @ bsh, 'model')     # explicit all-reduce
    return jax.shard_map(per_shard, mesh=mesh,
                         in_specs=(P('data', 'model'), P('model', None)),
                         out_specs=P('data', None))(a, b)
```

Look at the jaxpr, specifically the inner one:

```text
{ lambda ; a:f32[256@data,512@model] b:f32[512@model,1024]. let
    c:f32[256@data,1024] = shard_map[
      check_vma=True
      in_specs=(P('data', 'model'), P('model', None))
      jaxpr={ lambda ; d:f32[64,256]{V:(data,model)} e:f32[256,1024]{V:model}. let
          f:f32[256,1024]{V:(data,model)} = pvary[axes=('data',)] e
          g:f32[64,1024]{V:(data,model)} = dot_general[...] d f
          h:f32[64,1024]{V:data} = psum_invariant[axes=('model',)] g
        in (h,) }
      out_specs=(P('data', None),)
    ] a b
  in (c,) }
```

Two things happened to the types crossing that boundary. The shapes became per-device — `f32[64,256]`, not `f32[256,512]` — which is the point of `shard_map`. And a new annotation appeared: `{V:(data,model)}`.

`V` is the set of mesh axes the value **varies** over. It comes from a second placement field on the abstract value, a `ManualAxisType` carrying three frozensets of axis names:[^9]

```python
class ManualAxisType:
  __slots__ = ('varying', 'unreduced', 'reduced', 'unreduced_kind', '__weakref__')
```

Read the inner jaxpr as type inference over that set and it tells a story. `d` varies over both axes. `e` varies only over `model`, so JAX inserts `pvary[axes=('data',)]` to lift it to `{V:(data,model)}` before the contraction — a coercion, generated because the types did not line up. Then `psum_invariant[axes=('model',)]` takes `{V:(data,model)}` to `{V:data}`. The collective's job, at the type level, is to *remove* an axis from the varying set: after summing over `model`, the value no longer varies over `model`, and the type records that.

`unreduced` is the one I find most telling. It marks a value that is a pending partial sum — mathematically incomplete until a reduction happens. Set an `out_specs` that disagrees with the aval's unreduced set and JAX compares them directly:

```text
ValueError: out_specs passed to shard_map should be equal to the unreduced
present on the out_aval. Got out_specs=P('data', None, unreduced={'model'},
unreduced_kind=sum) and out_aval=f32[64,1024]{V:(data,model)}
```

And forget the collective entirely — write `lambda p, q: p @ q` with `out_specs=P('data', None)` — and you do not get wrong numbers:

```text
shard_map applied to the function '<lambda>' was given out_specs which require
replication which can't be statically inferred given the mesh:

The mesh given has shape (4, 2) with corresponding axis names ('data', 'model').

out_specs is P('data', None) which implies that the corresponding output value
is replicated across mesh axis 'model', but could not infer replication over
any axes

Check if these output values are meant to be replicated over those mesh axes.
If not, consider revising the corresponding out_specs entries. If so, consider
disabling the check by passing the check_vma=False argument to `jax.shard_map`.
```

A missing all-reduce in hand-written SPMD code used to be the classic silent-wrong-numbers bug: every device returns its own partial sum, the loss curve looks a bit off, and you spend two days on it. Here it is a trace-time error that names the mesh axis, explains the inference it could not make, and tells you which flag turns the check off. That is what having placement in the type buys.

## What is still not in the type

I would rather end on the honest gap than the good news, so: ask JAX to shard a 257-element dimension across four devices.

```python
jax.device_put(jnp.ones((257, 512)), NamedSharding(mesh, P('data', None)))
```

```text
ValueError: Sharding spec ('data',) implies that array axis 0 is partitioned
4 times, but does not evenly divide the dimension size 257.
Got shape: (257, 512)
```

A plain `ValueError`, raised inside `device_put`.[^10] Not a `ShardingTypeError`, not at trace time, and — I checked — identical under `Auto` and `Explicit`. The type `f32[257@data]` is not one the checker rejects; it is one you cannot build, and you find that out when you try to build it.

The divisibility obligation is a *relationship* between a dimension's extent and the size of the mesh axis it is sharded on. Expressing it as something checkable before the program runs needs shapes and mesh sizes in the same type-level arithmetic — the sort of thing dependent types or a shape-constraint solver handle, and neither is in JAX. So it stays a value-level check at the boundary. If your batch is 1000 and the mesh axis you want to split it over has 16 devices, nothing in the type system will say so; `device_put` will, every time, forever.

That is a much narrower complaint than the one I sat down to write. The interesting part is the direction. A year or so ago the whole of placement was outside the type, `Auto` was the only thing there was, and the notation was an opaque string. Now: sharding is a slot on `ShapedArray`, the recommended constructor makes it `Explicit` by default, mismatches raise from 53 sites in `core.py`, `shard_map` carries a second placement type tracking which axes a value varies over and which reductions it still owes, the partitioner represents all of it as structured MLIR, and what is left outside is divisibility arithmetic.

Nobody announced this as a language-design project. It arrived as a sequence of pragmatic fixes to expensive bugs — wrong numbers from a missing `psum`, an all-gather nobody asked for, a propagation decision no one could explain. Each fix moved one more property of *where data lives* from a comment, to an annotation, to the type. Which is the pattern I keep running into, in XLA layouts and in the [TPU's memory spaces]({% post_url 2026-07-24-tpu-through-open-source %}) too: the line between what a type carries and what rides alongside it is not a principled boundary. It is wherever the last expensive bug pushed it.

The last post in this series goes to the kernel layer, where you stop asking the compiler for placement and start writing it: Pallas, with Mosaic on TPU and Triton on GPU, where the memory spaces become `Ref`s you index by hand.

## References

[^1]: **Reproduction environment.** JAX 0.11.0, CPU backend, `jax.config.update('jax_num_cpu_devices', 8)` for an 8-device mesh. Every jaxpr, HLO excerpt, MLIR excerpt, and error message is copied from a session on that setup; source line numbers refer to the same release. The HLO is from `jax.jit(...).lower(...).compile().as_text()` and the MLIR from `.lower(...).as_text()`. ([Link](https://pypi.org/project/jax/0.11.0/))

[^2]: **JAX: Explicit sharding.** The `Auto` / `Explicit` / `Manual` axis-type distinction — declared as an enum at `jax/_src/mesh.py:111` — and what changes when an axis is `Explicit`. ([Link](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html))

[^3]: **GSPMD: General and Scalable Parallelization for ML Computation Graphs.** The propagation-plus-partitioning design that infers a full sharding from partial annotations and inserts the required collectives. ([Link](https://arxiv.org/abs/2105.04663))

[^4]: **Shardy (`sdy`).** The MLIR-based tensor partitioning system built by the GSPMD and PartIR teams, with sharding as structured IR rather than a string attribute. ([Repo](https://github.com/openxla/shardy), [Overview](https://openxla.org/shardy/overview))

[^5]: **`jax_use_shardy_partitioner` default.** Declared at `jax/_src/config.py:2167` in JAX 0.11.0 with `default=True` and `upgrade=True`; the help text points at the JAX-to-Shardy migration guide. ([Link](https://docs.jax.dev/en/latest/shardy_jax_migration.html))

[^6]: **`ShardingTypeError`.** Declared at `jax/_src/core.py:2197`; `grep -rn "ShardingTypeError("` over the installed 0.11.0 package returns 53 raise sites. ([Link](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py))

[^7]: **The propagation-permission flag.** Comparing `jax.jit(...).lower(...).compile().as_text()` between the two mesh constructors, the compiled modules are identical apart from instruction numbering, except that the `HloModule` header carries `allow_spmd_sharding_propagation_to_output={true}` under `Auto` and omits it under `Explicit` with an `out_sharding` supplied.

[^8]: **JAX: `shard_map`.** Per-shard programming with manual collectives — the explicit counterpart to GSPMD's automatic partitioning. ([Link](https://docs.jax.dev/en/latest/notebooks/shard_map.html))

[^9]: **Varying manual axes (VMA).** `ManualAxisType` is declared at `jax/_src/core.py:2363` with slots `varying`, `unreduced`, `reduced`, and `unreduced_kind`; `ShapedArray` holds one in its `manual_axis_type` slot, and a comment there links the docs section on tracking how values vary over manual mesh axes under `check_vma=True`. ([Link](https://docs.jax.dev/en/latest/notebooks/shard_map.html#tracking-how-values-vary-over-manual-mesh-axes-and-check-vma-true))

[^10]: **The divisibility check.** Raised as a plain `ValueError` from `jax/_src/core.py:2297` — "…implies that array axis 0 is partitioned 4 times, but does not evenly divide the dimension size 257" — during `device_put`, identically under `Auto` and `Explicit` axis types. ([Link](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py))

---

*Disclaimer: Researched and drafted with AI assistance (Gemini 3.1 Pro, Claude Opus 4.8 and Opus 5). Direction, technical judgment, and final edits are mine; every claim is traceable to the sources cited above.*
