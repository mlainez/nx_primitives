defmodule NxPrimitives.QuantizedConv do
  @moduledoc """
  2-D convolution with int8 per-output-channel quantized weights and
  f32 activations, running in the `arm_ai` NIF.

  ## Format

  Weights are quantized per output channel: for each output channel
  `co` (a row of the `[Cout, Kh*Kw*Cin]` weight matrix) we store
  `Kh*Kw*Cin` signed bytes plus one f32 scale. Dequantizing a single
  weight is `scale[co] * weight_i8[co, kh*Kw*Cin + kw*Cin + ci]`.

  ## Building a `QuantizedConv` from an Axon / Bumblebee Conv2d kernel

  Axon stores the kernel in HWIO order, shape `{Kh, Kw, Cin, Cout}`.
  Transpose it to `{Cout, Kh, Kw, Cin}`, then quantize each output
  channel to int8 with its own scale before calling `new/4`.

  ## Convention

  Input and output are NHWC (channels-last), matching Axon and Bumblebee
  defaults. Stride and padding are 2-element / 4-element lists
  respectively: `stride = [sh, sw]`,
  `padding = [pad_top, pad_bottom, pad_left, pad_right]`.
  """

  alias ArmAI.Native

  @enforce_keys [:weights, :scales, :bias, :cout, :kh, :kw, :cin]
  defstruct [:weights, :scales, :bias, :cout, :kh, :kw, :cin]

  @type t :: %__MODULE__{
          weights: binary(),
          scales: binary(),
          bias: binary(),
          cout: non_neg_integer(),
          kh: non_neg_integer(),
          kw: non_neg_integer(),
          cin: non_neg_integer()
        }

  @doc """
  Build a `QuantizedConv` from its component arrays.

    * `weights_int8` — raw int8 bytes, `Cout * Kh * Kw * Cin` long.
    * `scales` — list of `Cout` f32 scales (Elixir floats).
    * `bias` — list of `Cout` f32 biases, or `nil` for no bias.
    * `dims` — `{cout, cin, kh, kw}`.
  """
  @spec new(binary(), [float()], [float()] | nil, {pos_integer(), pos_integer(), pos_integer(), pos_integer()}) :: t()
  def new(weights_int8, scales, bias, {cout, cin, kh, kw})
      when is_binary(weights_int8) and is_list(scales) do
    expected = cout * kh * kw * cin

    if byte_size(weights_int8) != expected do
      raise ArgumentError,
            "weight buffer should be #{expected} bytes (Cout=#{cout} × Kh=#{kh} × Kw=#{kw} × Cin=#{cin}), got #{byte_size(weights_int8)}"
    end

    if length(scales) != cout do
      raise ArgumentError,
            "scales should have #{cout} entries, got #{length(scales)}"
    end

    bias_bin =
      case bias do
        nil ->
          <<>>

        list when is_list(list) and length(list) == cout ->
          for b <- list, into: <<>>, do: <<b::float-little-32>>

        list when is_list(list) ->
          raise ArgumentError,
                "bias should have #{cout} entries, got #{length(list)}"
      end

    scales_bin = for s <- scales, into: <<>>, do: <<s::float-little-32>>

    %__MODULE__{
      weights: weights_int8,
      scales: scales_bin,
      bias: bias_bin,
      cout: cout,
      kh: kh,
      kw: kw,
      cin: cin
    }
  end

  @doc """
  Apply the convolution to an input `Nx.Tensor` (NHWC f32).

  Options:
    * `:strides`  — `{sh, sw}` (default `{1, 1}`)
    * `:padding`  — `:valid` (default), `:same`, or an explicit
      `[{pt, pb}, {pl, pr}]` list (matching Nx's convention)

  Returns an `Nx.Tensor` on `Nx.BinaryBackend`.
  """
  @spec apply(Nx.Tensor.t(), t(), keyword()) :: Nx.Tensor.t()
  def apply(%Nx.Tensor{} = input, %__MODULE__{} = qc, opts \\ []) do
    {n, h_in, w_in, cin} =
      case Nx.shape(input) do
        {n, h, w, c} -> {n, h, w, c}
        other -> raise ArgumentError, "expected NHWC input, got shape #{inspect(other)}"
      end

    if cin != qc.cin do
      raise ArgumentError,
            "input Cin=#{cin} does not match weight Cin=#{qc.cin}"
    end

    {sh, sw} = Keyword.get(opts, :strides, {1, 1})

    {pt, pb, pl, pr} =
      case Keyword.get(opts, :padding, :valid) do
        :valid ->
          {0, 0, 0, 0}

        :same ->
          # Pad to keep H_out = ceil(H / sh), W_out = ceil(W / sw).
          pad_h = max(0, (div(h_in + sh - 1, sh) - 1) * sh + qc.kh - h_in)
          pad_w = max(0, (div(w_in + sw - 1, sw) - 1) * sw + qc.kw - w_in)
          {div(pad_h, 2), pad_h - div(pad_h, 2), div(pad_w, 2), pad_w - div(pad_w, 2)}

        [{pt, pb}, {pl, pr}] ->
          {pt, pb, pl, pr}

        other ->
          raise ArgumentError, "unsupported padding: #{inspect(other)}"
      end

    # Input must be on the host as raw f32 bytes; this is a CPU NIF.
    input_bin =
      input
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.as_type(:f32)
      |> Nx.to_binary()

    out_bin =
      Native.conv2d_int8_op(
        input_bin,
        qc.weights,
        qc.scales,
        qc.bias,
        [n, h_in, w_in, cin, qc.cout, qc.kh, qc.kw],
        [sh, sw],
        [pt, pb, pl, pr]
      )

    h_out = div(h_in + pt + pb - qc.kh, sh) + 1
    w_out = div(w_in + pl + pr - qc.kw, sw) + 1

    Nx.from_binary(out_bin, :f32, backend: Nx.BinaryBackend)
    |> Nx.reshape({n, h_out, w_out, qc.cout})
  end
end
