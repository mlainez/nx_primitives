# nx_primitives

> ### ⚠️ Very early work — built for a workshop, not for production
>
> This package was written for the **Goatmire Elixir workshop** on running
> Nerves on Fairphone 3 hardware. It exists for tinkering and teaching.
>
> There are no stability guarantees and APIs will change without notice.
>
> See [`nerves_ai`](https://github.com/mlainez/nerves_ai) for the full
> stack and the workshop context.

Nx-tensor compute primitives for edge inference.

Part of the [`nerves_ai`](https://github.com/mlainez/nerves_ai) edge-AI
stack. This package carries no native code of its own. FFT goes through
the pluggable `NxPrimitives.Backend`; the other modules call the
[`arm_ai`](https://github.com/mlainez/arm_ai) NEON kernels directly.

## What's here

| Module | What it does | Needs |
|---|---|---|
| `NxPrimitives.FFT` | Forward / inverse / real-input FFT | a configured backend |
| `NxPrimitives.Embeddings` | L2-normalise, cosine similarity, top-k | `arm_ai` + `nx_arm` |
| `NxPrimitives.Quantized` | int8 weight-only matmul | `arm_ai` + `nx_arm` |
| `NxPrimitives.QuantizedConv` | int8 conv2d (NHWC) | `arm_ai` |

## Install

```elixir
defp deps do
  [
    {:nx_primitives, github: "mlainez/nx_primitives"},
    # the ARM backend and kernels:
    {:arm_ai, github: "mlainez/arm_ai"},
    {:nx_arm, github: "mlainez/nx_arm"}
  ]
end
```

## Backend

`NxPrimitives.FFT` delegates to whichever module is configured as the
backend:

```elixir
config :nx_primitives, backend: ArmAI.NxPrimitivesBackend
```

The only implementation today is `ArmAI.NxPrimitivesBackend` (rustfft,
via the `arm_ai` NIF). Other backends can implement the
`NxPrimitives.Backend` behaviour and point that config key at them.

If you depend on `nerves_ai`, this wiring happens for you at boot.

## Usage

```elixir
# FFT (backend-dispatched): 1-D f32 in, interleaved [re, im, ...] out
spectrum = NxPrimitives.FFT.rfft(samples)

# Embeddings: corpus is {n, d}, query is {d}
scores  = NxPrimitives.Embeddings.cosine_similarity(query, corpus)
indices = NxPrimitives.Embeddings.top_k(scores, 5)   # list of row indices

# int8 weight-only matmul
q = NxPrimitives.Quantized.from_f32(weights)
out = NxPrimitives.Quantized.matmul(q, activations)
```

## Toolchain

Built and tested with Erlang/OTP 29.1.1 and Elixir 1.20.4, matching the
official Nerves systems (see `.tool-versions`).

## License

Apache-2.0
