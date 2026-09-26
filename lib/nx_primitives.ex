defmodule NxPrimitives do
  @moduledoc """
  Nx-tensor compute primitives for edge inference.

  * `NxPrimitives.FFT` — forward / inverse / real-input FFT, through the
    pluggable `NxPrimitives.Backend`
  * `NxPrimitives.Embeddings` — l2-normalise, cosine similarity, top-k
  * `NxPrimitives.Quantized` — int8 weight-only matmul
  * `NxPrimitives.QuantizedConv` — int8 conv2d (NHWC input, OHWI weights)

  ## Backend

      config :nx_primitives, backend: ArmAI.NxPrimitivesBackend

  `Embeddings`, `Quantized` and `QuantizedConv` call the `arm_ai` NEON
  kernels directly and need `arm_ai` + `nx_arm` in your deps.
  """
end
