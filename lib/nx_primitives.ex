defmodule NxPrimitives do
  @moduledoc """
  Cross-platform Nx-tensor primitives with a pluggable native
  backend.

  * `NxPrimitives.FFT` — forward / inverse / real-input FFT
  * `NxPrimitives.Embeddings` — l2-normalise, cosine sim, top-k
    (pure Nx, no backend needed)
  * `NxPrimitives.Quantized` — int8 weight-only matmul
  * `NxPrimitives.QuantizedConv` — int8 conv2d

  ## Backend

  `NxPrimitives.FFT` delegates to whichever module is configured as
  the backend. Today the canonical implementation is
  `ArmAI.NxPrimitivesBackend` (NEON-tuned via the `arm_ai` NIF);
  alternative backends (CUDA, AVX, etc.) can be slotted in by
  implementing the `NxPrimitives.Backend` behaviour.

      config :nx_primitives, backend: ArmAI.NxPrimitivesBackend

  `Quantized` and `QuantizedConv` are still coupled to `arm_ai`
  directly — their storage layout is implementation-specific. See
  the moduledoc on each for the path to backend-pluggable versions.
  """
end
