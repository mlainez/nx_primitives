defmodule NxPrimitives.Backend do
  @moduledoc """
  Behaviour for `NxPrimitives` operations that go through a
  pluggable native backend. Today that is the FFT family.

  Each implementation maps the public Nx-tensor API to its own
  compute path. The canonical implementation today is
  `ArmAI.NxPrimitivesBackend` (NEON-tuned via the `arm_ai` NIF);
  alternative implementations (CUDA, x86 AVX, etc.) can plug in
  without changes to consumer code.

  ## Configuring the active backend

      # In your config/runtime.exs (or config.exs):
      config :nx_primitives, backend: ArmAI.NxPrimitivesBackend

  Or override per-call:

      NxPrimitives.FFT.fft(x, backend: ArmAI.NxPrimitivesBackend)

  ## What's NOT in the behaviour

  `NxPrimitives.Embeddings`, `NxPrimitives.Quantized` and
  `NxPrimitives.QuantizedConv` call the `arm_ai` NIF directly, so they
  need `arm_ai` (and `nx_arm`) in your deps.
  """

  @doc "Forward complex FFT. Interleaved `[re, im, ...]` in/out."
  @callback fft(Nx.Tensor.t()) :: Nx.Tensor.t()

  @doc "Inverse complex FFT. Scaled by 1/N so fft+ifft is identity."
  @callback ifft(Nx.Tensor.t()) :: Nx.Tensor.t()

  @doc "Real-input FFT. Length-N real in, `2*(N/2+1)` interleaved out."
  @callback rfft(Nx.Tensor.t()) :: Nx.Tensor.t()

  @doc """
  Return the configured backend module. Reads the `:backend`
  option, falling back to `Application.get_env(:nx_primitives, :backend)`.
  Raises if neither is set.
  """
  @spec resolve(keyword()) :: module()
  def resolve(opts) do
    case Keyword.get(opts, :backend) || Application.get_env(:nx_primitives, :backend) do
      nil ->
        raise """
        No NxPrimitives backend configured. Add one to your config:

            config :nx_primitives, backend: ArmAI.NxPrimitivesBackend

        Or pass `backend:` explicitly to the NxPrimitives.* call.
        """

      backend when is_atom(backend) ->
        backend
    end
  end
end
