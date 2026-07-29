---
title: "What JAX Traces, and What It Refuses"
date: 2026-07-28 12:00:00 -0700
categories: [Compilers, ML-Systems]
tags: [jax, xla, jaxpr, tracing, autodiff, compilers]
---

Coming from PyTorch, the first thing JAX does is offend you. You write an honest little function with an `if` in it, wrap it in `jax.jit`, call it, and instead of running it throws `TracerBoolConversionError`. No graceful fallback, no partial compile — a hard stop on a line of Python that would have run fine a second ago. `torch.compile` would have shrugged, taken a graph break, and moved on. JAX refuses.

That refusal is the most important thing to understand about JAX, and once it clicks the whole system falls into place. JAX is not a framework with a compiler attached. It is a **tracing machine**. Every headline feature — `jit`, `grad`, `vmap`, `shard_map` — is the same move applied over and over: run a pure Python function once with stand-in values, record the sequence of primitive operations into a small typed IR called a *jaxpr*, and then transform or compile that IR. Understanding JAX means understanding the trace boundary — what the machine can see when your function runs, and what it is structurally blind to.

Everything below runs on JAX 0.11.0, CPU backend, and every jaxpr and every number is copied from a session rather than written from memory.[^1] Line numbers refer to that release. This is the front end: how Python becomes a jaxpr, why control flow is special, why every transformation is a rewrite of the same object, and — the part that has changed most in the last year — what the type of a traced value has quietly grown to include.

## The jaxpr is the whole point

Start with the object everything revolves around. `jax.make_jaxpr` traces a function and hands you the IR without compiling it:

```python
import jax.numpy as jnp
from jax import make_jaxpr

def f(x):
    return jnp.maximum(x * 2.0 + 1.0, 0.0)   # relu(2x + 1)

print(make_jaxpr(f)(jnp.float32(3.0)))
```

```text
{ lambda ; a:f32[]. let
    b:f32[] = mul a 2.0:f32[]
    c:f32[] = add b 1.0:f32[]
    d:f32[] = max c 0.0:f32[]
  in (d,) }
```

Read it like a typed lambda in A-normal form.[^2] After the semicolon come the input binders — here one scalar, `a:f32[]`. Then a `let` block of primitive applications, each binding a typed intermediate, and finally the returned tuple. Every value carries its type as a dtype and a shape: `f32[]` is a float32 scalar, `f32[4,8]` a matrix. Notice what is *not* there. The number `3.0` you passed in is gone. The jaxpr is not a trace of one execution; it is the abstract computation, parameterized over the types of its inputs, with all concrete data erased.

Give it real arrays and the primitives get more interesting:

```python
def layer(x, W, b):
    return jnp.maximum(x @ W + b, 0.0)

x = jnp.ones((4, 8)); W = jnp.ones((8, 16)); b = jnp.zeros((16,))
print(make_jaxpr(layer)(x, W, b))
```

```text
{ lambda ; a:f32[4,8] b:f32[8,16] c:f32[16]. let
    d:f32[4,16] = dot_general[
      dimension_numbers=(([1], [0]), ([], []))
      preferred_element_type=float32
    ] a b
    e:f32[1,16] = broadcast_in_dim[broadcast_dimensions=(1,)] c
    f:f32[4,16] = add d e
    g:f32[4,16] = max f 0.0:f32[]
  in (g,) }
```

Your `@` became `dot_general` — the one general contraction primitive that every matmul, batched matmul, and tensor contraction in JAX lowers to. The `dimension_numbers=(([1],[0]),([],[]))` says "contract axis 1 of the left operand with axis 0 of the right, no batch axes," which is an ordinary matrix multiply. Your `+ b` became an explicit `broadcast_in_dim` followed by `add`, because the jaxpr does not hide broadcasting — it is a real operation with a real cost, so it gets a node.

This is the level JAX reasons at. Import `jax.lax`, `jax.nn`, `jax.random`, and `jax.scipy.special`, then sweep the heap for live `core.Primitive` objects, and you find **285** of them.[^3] That count includes plenty of internal machinery you will never type, but the order of magnitude is the point: a few hundred typed operations, and no Python left.

## Tracing is abstract interpretation, not bytecode surgery

Here is where JAX diverges from `torch.compile` at the mechanism level, and the contrast explains everything downstream. Dynamo intercepts CPython at the frame-evaluation layer and symbolically executes your bytecode, opcode by opcode; when it meets something it cannot model, it takes a graph break and falls back to the interpreter.[^4] JAX does none of that. It never looks at your bytecode. It just calls your function — with fake arguments.

When you trace `f`, JAX does not pass in an array. It passes in a `Tracer`: an object carrying an *abstract value*, and recording every primitive applied to it. Put a print inside a traced function and you can see exactly what flows through:

```python
def probe(x):
    print(type(x).__name__, "|", x.aval)
    return x + 1

make_jaxpr(probe)(jnp.ones((2, 3)))
# DynamicJaxprTracer | float32[2,3]
```

The value moving through your function is a `DynamicJaxprTracer` whose `aval` is `float32[2,3]`. Your Python executes once, top to bottom, during tracing, and every `jax.numpy` call it makes appends a node to the jaxpr being built. That is why `x @ W` turns into `dot_general` — the tracer's `__matmul__` records the primitive instead of computing anything.

The abstract value is the entire worldview of the compiler. It knows your array is `float32[2,3]`. It does not know, and never asked, what the numbers are. Hold onto that sentence; the last section of this post is about how much *else* the aval now knows.

## The trace boundary: what JAX refuses to see

Now the offending `if`. A tracer has a type but no concrete value, so the moment your code demands a concrete boolean from it, tracing cannot continue:

```python
from jax import jit

def bad(x):
    if x > 0:          # needs a real bool; x is a tracer
        return x
    return -x

jit(bad)(jnp.float32(1.0))
```

```text
jax.errors.TracerBoolConversionError:
Attempted boolean conversion of traced array with shape bool[].
The error occurred while tracing the function bad for jit. This concrete
value was not available in Python because it depends on the value of the
argument x.
```

`x > 0` produces another tracer — a `bool[]` — and a Python `if` needs to branch *now*, which means collapsing that tracer to `True` or `False`, which JAX cannot do without the data. So it stops. This is the exact spot where `torch.compile` would graph-break, and the two systems make opposite choices. Dynamo chooses "keep running, compile what you can." JAX chooses "the trace stays pure and complete, so make the control flow part of the IR or don't compile at all."

I have come around to preferring JAX's choice, with a caveat. A graph break is a silent performance cliff — your model still runs, just slower, and you find out months later reading a profile. `TracerBoolConversionError` is loud and immediate; it fails in your face on the first call. That honesty is worth a lot. The cost is real: you cannot sprinkle data-dependent Python control flow through your model and expect it to compile.

## Control flow has to become a primitive

The fix is to lift the branch out of Python and into the IR:[^5]

```python
from jax import lax

def good(x):
    # select the branch inside the IR instead of in Python
    return lax.cond(x > 0, lambda v: v, lambda v: -v, x)

print(make_jaxpr(good)(jnp.float32(1.0)))
```

```text
{ lambda ; a:f32[]. let
    b:bool[] = gt a 0.0:f32[]
    c:i32[] = convert_element_type[new_dtype=int32 weak_type=False] b
    d:f32[] = cond[
      branches=(
        { lambda ; e:f32[]. let f:f32[] = neg e in (f,) }
        { lambda ; g:f32[]. let  in (g,) }
      )
    ] c a
  in (d,) }
```

The branch is now a `cond` primitive whose two arms are themselves little jaxprs, and the predicate is a traced `bool[]` rather than a Python bool. Both arms are in the IR because JAX traced both of them, which you can watch directly:

```python
hits = []
def true_fn(v):  hits.append("true");  return v * 10
def false_fn(v): hits.append("false"); return v - 10

make_jaxpr(lambda x: lax.cond(x > 0, true_fn, false_fn, x))(jnp.float32(1.0))
print(hits)     # ['true', 'false']
```

Both branch functions ran at trace time. That is the price of making control flow data-independent, and it has a consequence people trip over: if one arm of your `cond` allocates a 10 GB intermediate, that allocation is in the program whether or not the predicate ever selects it.

Tracing both arms is not the same as *running* both. An unbatched `cond` executes one branch. But batch the predicate — which is what happens the moment you `vmap` a per-example decision — and the primitive disappears:

```python
f = lambda x: lax.cond(x > 0, lambda v: v * 10, lambda v: v - 10, x)
print(make_jaxpr(vmap(f))(jnp.arange(4, dtype=jnp.float32)))
```

```text
{ lambda ; a:f32[4]. let
    b:bool[4] = gt a 0.0:f32[]
    c:i32[4] = convert_element_type[new_dtype=int32 weak_type=False] b
    d:bool[4] = eq c 0:i32[]
    e:f32[4] = stop_gradient a
    f:f32[4] = select_n d e a
    g:f32[4] = sub f 10.0:f32[]
    h:bool[4] = eq c 1:i32[]
    i:f32[4] = stop_gradient a
    j:f32[4] = select_n h i a
    k:f32[4] = mul j 10.0:f32[]
    l:f32[4] = select_n c g k
  in (l,) }
```

No `cond` anywhere. Both the `sub` and the `mul` run on all four lanes and a `select_n` picks per lane, because there is no such thing as a divergent branch across a vector register.[^5] Each lane pays for both arms. When one arm is an attention block and the other is `return x`, that is the whole cost of the expensive arm on every example, and it is invisible unless you read the jaxpr.

Loops split the same way. `lax.while_loop` is for data-dependent iteration counts. `lax.scan` is for a fixed count, and it is the one to internalize, because it traces the body exactly once:

```python
n = []
def body(carry, x):
    n.append(1)
    return carry + x, carry

print(make_jaxpr(lambda c, xs: lax.scan(body, c, xs))(
      jnp.float32(0.), jnp.arange(1000, dtype=jnp.float32)))
print(len(n), "trace(s) of the body")
```

```text
{ lambda ; a:f32[] b:f32[1000]. let
    c:f32[] d:f32[1000] = scan[
      jaxpr={ lambda ; e:f32[] f:f32[]. let g:f32[] = add e f in (g, e) }
      length=1000
      reverse=False
      unroll=1
    ] a b
  in (c, d) }
1 trace(s) of the body
```

One equation, `length=1000`, one traced body. Write the same loop as a Python `for` and JAX will happily trace it — it just unrolls the whole thing into the jaxpr, and you pay for that in the compiler. Here is a 5,000-step recurrence over a 32×32 weight matrix, measured both ways on CPU:

```text
                top-level eqns    trace     compile
python for            10000       0.11 s    36.92 s
lax.scan                  1       0.00 s     0.02 s
```

Thirty-seven seconds versus twenty milliseconds, for a program that computes the same thing. This is the single most common self-inflicted wound in JAX code, and it does not announce itself — nothing errors, your job just sits in XLA for a minute before every run. The rule to internalize: anything that decides the *shape or structure* of the computation from a value must become a primitive, because the trace only knows types, never values.

## Purity is not optional

The trace also assumes your function is pure — output determined entirely by its inputs, no side effects, no hidden state. This is not a style preference; it is load-bearing. Your Python runs while JAX traces it, so any side effect fires only during tracing, once per trace, and not on the cached calls that follow. A `print` inside a jitted function prints while JAX traces it, then goes silent on every cached call after — baffling if you did not expect it. Mutating a global from inside a traced function is worse: the write happens at trace time against whatever the state was then, and the compiled program has no memory of it. (The trace-counting trick further down works precisely because of this.)

The most visible consequence is randomness, and it is the single most complained-about thing in JAX. NumPy's `np.random.randn()` reads and advances a global RNG state — a side effect, invisible to the tracer, impossible to reproduce or parallelize deterministically. JAX cannot trace that, so it refuses to have it. Randomness is instead an explicit, pure function of a key you pass in:[^6]

```python
key = jax.random.key(0)
jax.random.normal(key, (3,))   # [ 1.6226422  2.0252647 -0.4335944 ]
jax.random.normal(key, (3,))   # identical — same key, same draw
```

Same key, same numbers, every time — a draw is a deterministic function of its key, which is what a pure trace demands. To get fresh randomness you advance the state yourself, by splitting a key into new independent keys:

```python
k1, k2 = jax.random.split(key)
jax.random.normal(k1, (3,))    # [ 1.0040143 -0.9063372 -0.7481722 ]
```

People hate this until they need it, and then they stop. Because the randomness is threaded explicitly, a JAX computation is bit-for-bit reproducible no matter how it is parallelized — the key, not a hidden global, determines the result. The generator underneath is Threefry, a counter-based cipher-like construction designed so that the *n*-th draw is computable directly from the counter without stepping through the first *n−1*, which is exactly what makes a split-and-scatter design work across 512 chips.[^7]

And a key is a first-class typed value, not a special case bolted onto the API. Trace a draw and look at the binder:

```python
print(make_jaxpr(lambda k: jax.random.normal(k, (3,)))(key))
```

```text
{ lambda ; o:key<fry>[]. let
    p:f32[3] = jit[
      name=_normal
      jaxpr={ lambda ; o:key<fry>[]. let
          ...
          f:u32[3] = random_bits[bit_width=32 shape=(3,)] a
          g:u32[3] = shift_right_logical f 9:u32[]
          h:u32[3] = or g 1065353216:u32[]
          i:f32[3] = bitcast_convert_type[new_dtype=float32] h
          ...
```

The input type is `key<fry>[]` — an extended dtype naming the Threefry implementation — and the body is the bit-twiddling you would write by hand: draw 32 random bits, shift right by 9 to keep the 23 mantissa bits, `or` in the exponent for 1.0 (`0x3F800000` = 1065353216), bitcast to float, subtract one. A uniform in [0,1) built from integer ops, fully visible in the IR, with no hidden state anywhere. The whole design is a direct consequence of the tracing model: no global state means no global RNG.

## Every transformation is a jaxpr rewrite

This is the part that makes JAX feel less like a library and more like a small compiler you drive from Python. Because tracing produces one uniform IR, every transformation is defined as a rewrite of that IR — and rewrites compose.

**`grad` is a source transformation.** Reverse-mode autodiff traces your function, then produces a *new* jaxpr that computes the gradient.[^8] You can read it:

```python
from jax import grad

def g(x):
    return jnp.sum(jnp.sin(x) ** 2)

print(make_jaxpr(grad(g))(jnp.ones((3,))))
```

```text
{ lambda ; a:f32[3]. let
    b:f32[3] = sin a
    c:f32[3] = cos a
    d:f32[3] = integer_pow[y=2] b
    e:f32[3] = mul 2.0:f32[] b
    _:f32[] = reduce_sum[axes=(0,) out_sharding=None] d
    f:f32[3] = broadcast_in_dim 1.0:f32[]
    g:f32[3] = mul f e
    h:f32[3] = mul g c
  in (h,) }
```

The whole chain rule is sitting in the `let` block. `b = sin a` and `c = cos a` are the residuals the backward pass needs; `e = 2·sin(a)`; `f` is the incoming cotangent of the sum, which is `1`; and the result is `h = f · e · c = 2·sin(a)·cos(a)` — the derivative of `sin²(x)`. Two details. The forward result `d = sin²(a)` and its `reduce_sum` are computed and then *dropped* — bound to `_` — because the gradient does not need them; they are dead code XLA will delete later. And the gradient jaxpr is just another jaxpr, which is why `grad(grad(f))` and `grad(jit(f))` work without special machinery. For how the forward trace gets linearized and transposed into this, I wrote up the mechanics in [the autodiff post]({% post_url 2026-06-24-autodiff-representations %}).

**`vmap` rewrites primitives, not Python.** Vectorization is not a loop over your function. It rewrites each primitive to a batched version by threading a batch axis through the jaxpr:

```python
def dot(a, b):
    return jnp.dot(a, b)                       # vectors -> scalar

make_jaxpr(dot)(jnp.ones(5), jnp.ones(5))
#   dot_general[dimension_numbers=(([0], [0]), ([], []))]

from jax import vmap
make_jaxpr(vmap(dot))(jnp.ones((32, 5)), jnp.ones((32, 5)))
#   dot_general[dimension_numbers=(([1], [1]), ([0], [0]))]
```

Look at what changed. The unbatched dot contracts axis 0 with axis 0 and produces a scalar. Under `vmap`, `dimension_numbers` gained a batch entry — `([0],[0])` in the third slot — so axis 0 of each operand is now a batch dimension and the contraction moved to axis 1. There is no Python loop and no per-example overhead; `vmap` produced a single batched `dot_general` that XLA lowers to one kernel. This is why the advice in JAX is always "write the function for one example and `vmap` it," where the equivalent advice in eager frameworks is "manually batch everything and pray the shapes line up."

The rewrites are table-driven, and the tables are not the same size:

| Registry | Entries |
|---|---|
| MLIR lowering rules (`mlir._lowerings`) | 251 |
| batching rules (`batching.fancy_primitive_batchers`) | 225 |
| JVP rules (`ad.primitive_jvps`) | 200 |
| transpose rules (`ad.primitive_transposes`) | 84 |

A primitive needs a lowering rule to run at all, a batching rule to survive `vmap`, a JVP rule to be differentiated forward, and a transpose rule for the linearized computation to be reversed. Those are four independent obligations, and nobody has discharged all four for every primitive. Reverse-mode `grad` is the narrowest gate: 84 transpose rules. When you hit a primitive with no rule, JAX does not silently degrade — it raises, names the primitive, and tells you which rule is missing. Same philosophy as the `if`.

Because all three transformations are jaxpr-to-jaxpr rewrites over the same IR, they stack in any order. `jit(grad(vmap(f)))` is not a special case; it is three rewrites composed.

## Retracing, and the cache key that decides your compile time

`jit` is the transformation that stages the jaxpr out to XLA and caches the compiled result. The cache key is the thing to memorize, because it is where JAX's compile-time behavior comes from. JAX keys on the *abstract value* of each argument and nothing else:[^9]

```python
count = {"n": 0}

@jit
def f(x):
    count["n"] += 1          # side effect: runs only while tracing
    return x * 2 + 1

f(jnp.ones((4,), jnp.float32));  # count -> 1   (trace + compile)
f(jnp.ones((4,), jnp.float32));  # count -> 1   (cache hit)
f(jnp.ones((8,), jnp.float32));  # count -> 2   (new shape: retrace)
f(jnp.ones((8,), jnp.int32));    # count -> 3   (new dtype: retrace)
```

Same aval, cache hit, no retrace. Change the shape or the dtype and JAX traces and compiles a fresh program. This is the same failure mode as recompilation thrashing in [the torch.compile post]({% post_url 2026-07-22-how-torch-compile-works %}), but the cause is cleaner and more predictable: there is no guard tree to reason about, just the aval. If your batch size varies every step, you recompile every step. The fixes are shape padding, bucketing to a handful of fixed sizes, or symbolic shape polymorphism via `jax.export`, which lowers one shape-generic program at the cost of giving XLA less to work with.[^10]

The flip side is `static_argnums`, which tells `jit` to treat an argument as a compile-time Python constant rather than a traced value. That is mandatory when an argument determines a *shape*, since shapes must be concrete during tracing:

```python
from functools import partial

@partial(jit, static_argnums=0)
def g(n, x):
    return jnp.broadcast_to(x, (n,))   # n must be concrete to be a shape

print(make_jaxpr(g, static_argnums=0)(3, jnp.float32(7.0)))
```

```text
{ lambda ; a:f32[]. let
    b:f32[3] = jit[
      name=g
      jaxpr={ lambda ; a:f32[]. let b:f32[3] = broadcast_in_dim a in (b,) }
    ] a
  in (b,) }
```

The `3` is gone from the argument list and baked into the type of `b`. A different `n` is a different compiled program — static arguments are part of the cache key too. Use them for things that genuinely define structure (a number of layers, a boolean flag) and never for values that vary a lot, or you are back to recompiling constantly.

## PyTrees, briefly

One more front-end mechanism, because it is everywhere. JAX transformations operate on *PyTrees* — arbitrarily nested tuples, lists, and dicts of arrays.[^11] A model's parameters are usually one big nested dict, and JAX treats it as a container it can flatten into a flat list of leaves plus a `treedef` describing the structure:

```python
import jax.tree_util as jtu
params = {"w": jnp.ones((2, 2)), "layers": [jnp.zeros((3,)), jnp.ones((3,))]}
leaves, treedef = jtu.tree_flatten(params)
# 3 leaves; treedef = PyTreeDef({'layers': [*, *], 'w': *})
```

This is why `grad(loss)(params)` returns a gradient with the exact structure of `params`, and why `jit` accepts and returns nested dicts transparently. Flatten to leaves, transform the leaves, unflatten with the saved `treedef`. Unglamorous plumbing doing a lot of quiet work.

## The aval grew places

Go back to that binder: `a:f32[4,8]`. Dtype and shape, and I said the compiler knows nothing else. That was true for most of JAX's life, and it stopped being true recently. Look at what `ShapedArray` actually declares:[^12]

```python
class ShapedArray(AbstractValue):
  __slots__ = ['shape', 'dtype', 'weak_type', 'sharding', 'manual_axis_type',
               'memory_space', '_stripped_weak_type', '__weakref__']
```

`sharding`. `memory_space`. The abstract value — the thing tracing propagates, the thing that keys the compilation cache — now carries where the array lives.

To see it, build a mesh with **explicit** axes. JAX classifies each mesh axis as `Auto`, `Explicit`, or `Manual`; under `Auto`, sharding is a hint the compiler propagates on its own and the type stays quiet, which is what you get by default and why you may never have noticed any of this.[^13] Mark the axes `Explicit` and placement moves into the type:

```python
import jax
from jax.sharding import PartitionSpec as P, AxisType

jax.config.update('jax_num_cpu_devices', 8)
mesh = jax.make_mesh((4, 2), ('x', 'y'),
                     axis_types=(AxisType.Explicit, AxisType.Explicit))

with jax.sharding.set_mesh(mesh):
    a = jax.device_put(jnp.zeros((8, 16)), jax.NamedSharding(mesh, P('x', 'y')))
    b = jax.device_put(jnp.zeros((16, 4)), jax.NamedSharding(mesh, P('y', None)))
    print(jax.typeof(a), jax.typeof(b))
```

```text
float32[8@x,16@y] float32[16@y,4]
```

`f32[8@x,16@y]`: eight rows sharded over mesh axis `x`, sixteen columns over `y`. The `@` is not decoration in a debug print — it comes from the aval's own formatter, which walks the shape and the `PartitionSpec` together and emits one string.[^14] Placement is *in the type*, per axis.

And a type is only interesting if something checks it. Multiply those two arrays:

```python
make_jaxpr(lambda p, q: p @ q)(a, b)
```

```text
jax._src.core.ShardingTypeError: Contracting dimensions are sharded and it
is ambiguous how the output should be sharded. Please specify the output
sharding via the `out_sharding` parameter.
Got lhs_contracting_spec=('y',) and rhs_contracting_spec=('y',)
```

That is a type error about *placement*, raised during tracing, before any device does anything. The shapes are fine — `(8,16) @ (16,4)` contracts cleanly. What does not check is where the operands live: contracting a `y`-sharded axis against a `y`-sharded axis leaves the result needing a reduction whose placement JAX will not guess for you. Elementwise ops are checked the same way:

```text
ShardingTypeError: add got incompatible shardings for broadcasting:
('x', 'y'), ('y', 'x').
```

`ShardingTypeError` is a real exception class in `jax/_src/core.py`, thirteen lines above `MemorySpace` and three hundred above `ShapedArray`, and it is raised from **53 sites** across the codebase.[^15] This is a type checker for data placement, and it has the density of raise-sites of one.

Answer the question it asked and the trace goes through, with the placement of the result written into the binder:

```python
def f(p, q):
    return jax.lax.dot_general(p, q, (((1,), (0,)), ((), ())),
                               out_sharding=P('x', None))

print(make_jaxpr(f)(a, b))
```

```text
{ lambda ; a:f32[8@x,16@y] b:f32[16@y,4]. let
    c:f32[8@x,4] = dot_general[dimension_numbers=(([1], [0]), ([], []))] a b
  in (c,) }
```

Memory space rides along in the same slot. `MemorySpace` is a three-case enum — `Device`, `Host`, `Any` — and the formatter prints anything that is not `Device` as an angle-bracketed prefix on the dtype:

```python
h = jax.device_put(jnp.zeros((8, 16)),
                   jax.NamedSharding(mesh, P('x', 'y'), memory_kind='pinned_host'))
print(jax.typeof(h))
```

```text
float32<host>[8@x,16@y]
```

Read `float32<host>[8@x,16@y]` as one type expression and it tells you the element type, the memory space, the extent of each axis, and the mesh axis each is distributed over. That is a long way from `f32[4,8]`, and the direction of travel is unambiguous: the things a compiler needs to know about *where* data lives are migrating out of side-channel annotations and into the type of the value.

I think that direction is correct, and I do not think it is finished. `sharding` and `memory_space` got in because sharding bugs were expensive enough — silently wrong numbers, or a 100× slowdown from an unplanned all-to-all — that "annotate and hope the partitioner agrees" stopped being tolerable. That argument is not special to sharding. It applies to every property of a value that the hardware treats as load-bearing and the type currently ignores: layout, alignment, which of a chip's several memories a buffer sits in, whether a pointer is valid on the device you are about to dereference it from. Those are still, today, comments and conventions and runtime aborts. There is no principled line separating "in the type" from "beside the type" here — there is a line someone drew, and it has already moved twice.

What that means for the [tour of XLA]({% post_url 2026-07-24-a-tour-of-xla-where-mlir-lives %}) and its [rigid price]({% post_url 2026-07-23-xla-review-and-critique %}) is that the front end is handing the compiler more than it used to. And what it means for the next post is that sharding deserves its own treatment: how a `PartitionSpec` becomes a mesh assignment, where GSPMD ends and the newer Shardy partitioner begins, and what happens on the axes you leave as `Auto` — which is to say, the ones where the type stays quiet and the compiler decides for you.

## References

[^1]: **Reproduction environment.** All jaxprs, error messages, timings, and registry counts in this post are from JAX 0.11.0 on the CPU backend, Python 3.14, with `jax.config.update('jax_num_cpu_devices', 8)` for the mesh examples. Source line references are to the same release. ([Link](https://pypi.org/project/jax/0.11.0/))

[^2]: **JAX: Understanding jaxprs.** The jaxpr grammar, binders, `let`-bound equations, and how tracing with abstract values produces one. ([Link](https://docs.jax.dev/en/latest/jaxpr.html))

[^3]: **Primitive count method.** After `import jax.lax, jax.nn, jax.random, jax.scipy.special`, sweeping `gc.get_objects()` for instances of `jax._src.core.Primitive` yields 285 live objects. This includes primitives that exist only for internal staging (`addupdate`, `accum_grad_in_ref`, Pallas-side ops) and is an upper bound on the user-facing surface, not a count of documented operations.

[^4]: **torch.compile / TorchDynamo — frame-evaluation capture.** The bytecode-interception design and the graph-break fallback that JAX declines to have. ([Link](https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html))

[^5]: **JAX: Control flow and logical operators**, plus the `lax.cond` docstring. The docs cover why data-dependent Python control flow must become `lax.cond`, `lax.scan`, or `lax.while_loop`. The batching behaviour is stated in the source: "In contrast with `jax.lax.select`, using `cond` indicates that only one of the two branches is executed (up to compiler rewrites and optimizations). However, when transformed with `jax.vmap` to operate over a batch of predicates, `cond` is converted to `jax.lax.select`. Both branches will be traced in all cases." (`jax/_src/lax/control_flow/conditionals.py:212`, JAX 0.11.0.) ([Link](https://docs.jax.dev/en/latest/control-flow.html))

[^6]: **JAX: Pseudorandom numbers.** Why global RNG state is incompatible with the traced, functional model, and the explicit key/`split` design that replaces it. ([Link](https://docs.jax.dev/en/latest/random-numbers.html))

[^7]: **Salmon, Moraes, Dror, Shaw — "Parallel Random Numbers: As Easy as 1, 2, 3" (SC '11).** The counter-based Threefry generator underneath JAX's `key<fry>` dtype, designed so the *n*-th output is computable directly from the counter. ([Link](https://dl.acm.org/doi/10.1145/2063384.2063405))

[^8]: **JAX: Automatic differentiation.** How JAX composes forward-mode linearization and transposition to produce reverse-mode gradients as a program transformation over jaxprs. ([Link](https://docs.jax.dev/en/latest/automatic-differentiation.html))

[^9]: **JAX: Just-in-time compilation.** Tracing, caching keyed on abstract shape and dtype, and `static_argnums`. ([Link](https://docs.jax.dev/en/latest/jit-compilation.html))

[^10]: **JAX: Exporting and serialization — shape polymorphism.** Lowering a single shape-generic program with symbolic dimensions, and the constraints that come with it. ([Link](https://docs.jax.dev/en/latest/export/shape_poly.html))

[^11]: **JAX: Working with pytrees.** Flatten/unflatten, `treedef`, and how transformations map over leaves. ([Link](https://docs.jax.dev/en/latest/working-with-pytrees.html))

[^12]: **`ShapedArray.__slots__`.** `jax/_src/core.py:2450` in JAX 0.11.0 — `shape`, `dtype`, `weak_type`, `sharding`, `manual_axis_type`, `memory_space`. ([Link](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py))

[^13]: **JAX: Explicit sharding (sharding in types).** The `Auto` / `Explicit` / `Manual` axis-type distinction, defined at `jax/_src/mesh.py:111`, and what changes when an axis is marked `Explicit`. ([Link](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html))

[^14]: **The aval formatter.** `str_short_aval` at `jax/_src/core.py:2632` composes the dtype, an angle-bracketed memory space for anything other than `MemorySpace.Device`, and the shape interleaved with the `PartitionSpec` by `_get_shape_sharding_str` at `jax/_src/core.py:2611`. ([Link](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py))

[^15]: **`ShardingTypeError`.** Declared at `jax/_src/core.py:2197`; `grep -rn "ShardingTypeError(" jax/` over the installed 0.11.0 package returns 53 raise sites. `MemorySpace` is declared four lines below it at `jax/_src/core.py:2201`. ([Link](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py))

---

*Disclaimer: Researched and drafted with AI assistance (Gemini 3.1 Pro, Claude Opus 4.8 and Opus 5). Direction, technical judgment, and final edits are mine; every claim is traceable to the sources cited above.*
