defmodule NxPrimitives.FFT do
  @moduledoc """
  Fast Fourier Transform — generic API delegating to a configured
  `NxPrimitives.Backend`.

  All inputs/outputs are Nx tensors.

      # 1024-point complex FFT (input: shape {2N} interleaved re,im,...)
      x = Nx.iota({2048})
      y = NxPrimitives.FFT.fft(x)              # complex → complex

      # Real-input FFT (input: shape {N} real, output: shape {2*(N/2+1)})
      y_real = NxPrimitives.FFT.rfft(x)

  ## Backend selection

  By default reads `Application.get_env(:nx_primitives, :backend)`.
  Override per-call with `backend: SomeModule`.

      NxPrimitives.FFT.fft(x, backend: ArmAI.NxPrimitivesBackend)
  """

  @doc """
  Complex → complex forward FFT. Input is interleaved
  `[re0, im0, re1, im1, ...]` of length `2*N`. Output has the same
  length.
  """
  @spec fft(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def fft(input, opts \\ []), do: NxPrimitives.Backend.resolve(opts).fft(input)

  @doc """
  Complex → complex inverse FFT. Same layout as `fft/1`. Output is
  scaled by `1/N` so a forward+inverse round-trip is the identity.
  """
  @spec ifft(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def ifft(input, opts \\ []), do: NxPrimitives.Backend.resolve(opts).ifft(input)

  @doc """
  Real-input FFT. Input is a length-`N` real signal; output is the
  non-redundant half as interleaved `[re, im, re, im, ...]` of
  length `2*(N/2 + 1)`.
  """
  @spec rfft(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def rfft(input, opts \\ []), do: NxPrimitives.Backend.resolve(opts).rfft(input)
end
