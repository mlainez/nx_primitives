# nx_primitives

> ### ⚠️ Very early work — built for a workshop, not for production
>
> This package was written for the **Goatmire Elixir workshop** on running
> Nerves on Fairphone 3 hardware. It exists for tinkering and teaching.
>
> It is **not an actively maintained project** (yet). There are no
> stability guarantees, APIs will change without notice, and parts of it
> are wired-but-unproven. Treat it as a starting point to hack on, not as
> a dependency to build a product on.
>
> See [`nerves_ai`](https://github.com/mlainez/nerves_ai) for the full
> stack and the workshop context.

Cross-platform Nx-tensor primitives with a pluggable native backend.

Part of the [`nerves_ai`](https://github.com/mlainez/nerves_ai) edge-AI
stack, but usable standalone — this package defines the generic API and
carries no native code of its own.

## What's here

| Module | What it does | Needs a backend? |
|---|---|---|
| `NxPrimitives.FFT` | Forward / inverse / real-input FFT | yes |
| `NxPrimitives.Embeddings` | L2-normalise, cosine similarity, top-k | no — pure Nx |
| `NxPrimitives.Quantized` | int8 weight-only matmul | yes |
| `NxPrimitives.QuantizedConv` | int8 conv2d | yes |

## Install

```elixir
defp deps do
  [
    {:nx_primitives, github: "mlainez/nx_primitives"},
    # plus a backend — on ARM:
    {:arm_ai, github: "mlainez/arm_ai"}
  ]
end
```

## Backend

`NxPrimitives.FFT` delegates to whichever module is configured as the
backend:

```elixir
config :nx_primitives, backend: ArmAI.NxPrimitivesBackend
```

The canonical implementation today is `ArmAI.NxPrimitivesBackend`
(NEON-tuned, via the `arm_ai` NIF). Alternative backends — CUDA, AVX,
Metal — slot in by implementing the `NxPrimitives.Backend` behaviour and
pointing that config key at them.

If you depend on `nerves_ai`, this wiring happens for you at boot.

## Usage

```elixir
# FFT (backend-dispatched)
spectrum = NxPrimitives.FFT.rfft(samples)

# Embeddings — pure Nx, works on any backend including BinaryBackend
query  = NxPrimitives.Embeddings.l2_normalize(query_vec)
scores = NxPrimitives.Embeddings.cosine_similarity(query, corpus)
{values, indices} = NxPrimitives.Embeddings.top_k(scores, 5)

# int8 weight-only matmul
q = NxPrimitives.Quantized.from_f32(weights)
out = NxPrimitives.Quantized.matmul(q, activations)
```

## Caveats

`Embeddings` is pure Nx and needs no backend at all — it runs anywhere.

`Quantized` and `QuantizedConv` are still coupled to `arm_ai` directly,
because their storage layout is implementation-specific. The
`NxPrimitives.Backend` behaviour does not yet abstract them; see each
module's docs for the path to a backend-pluggable version.

## License

Apache-2.0
