defmodule ArmNxPrimitives do
  @moduledoc """
  Cross-domain Nx-tensor primitives built over `arm_ai`'s NIF.

  * `ArmNxPrimitives.FFT` — complex-64 FFT / IFFT via rustfft
  * `ArmNxPrimitives.Embeddings` — l2-normalise, cosine sim, top-k
  * `ArmNxPrimitives.Quantized` — quantized tensor format helpers
  * `ArmNxPrimitives.QuantizedConv` — int8 conv with f32 activations

  Depends on `:nx_arm` for the backend.
  """
end
