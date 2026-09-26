defmodule NxPrimitives.QuantizedConvTest do
  use ExUnit.Case, async: true

  alias NxPrimitives.QuantizedConv

  # Reference: dequantize the int8 OHWI weights and run Nx.conv on NHWC.
  defp reference(input, w_i8, scales, bias, {cout, cin, kh, kw}, opts) do
    w =
      w_i8
      |> Nx.from_binary(:s8)
      |> Nx.reshape({cout, kh, kw, cin})
      |> Nx.as_type(:f32)
      |> Nx.multiply(Nx.reshape(Nx.tensor(scales, type: :f32), {cout, 1, 1, 1}))

    out =
      Nx.conv(input, w,
        strides: Tuple.to_list(Keyword.get(opts, :strides, {1, 1})),
        padding: Keyword.get(opts, :padding, :valid),
        input_permutation: [0, 3, 1, 2],
        kernel_permutation: [0, 3, 1, 2],
        output_permutation: [0, 3, 1, 2]
      )

    if bias, do: Nx.add(out, Nx.tensor(bias, type: :f32)), else: out
  end

  defp fixture(dims = {cout, cin, kh, kw}) do
    w_i8 = for i <- 0..(cout * kh * kw * cin - 1), into: <<>>, do: <<rem(i * 37, 255) - 127::signed-8>>
    scales = for o <- 1..cout, do: 0.01 * o
    bias = for o <- 1..cout, do: 0.1 * o
    {w_i8, scales, bias, dims}
  end

  defp input(shape), do: Nx.iota(shape, type: :f32) |> Nx.divide(17) |> Nx.sin()

  for {name, opts} <- [
        {"valid, stride 1", []},
        {"same padding", [padding: :same]},
        {"stride 2, explicit padding", [strides: {2, 2}, padding: [{1, 0}, {0, 1}]]}
      ] do
    test "matches dequantized Nx.conv (#{name})" do
      {w_i8, scales, bias, dims} = fixture({4, 3, 3, 3})
      x = input({2, 7, 6, 3})
      opts = unquote(opts)

      got = QuantizedConv.apply(x, QuantizedConv.new(w_i8, scales, bias, dims), opts)
      want = reference(x, w_i8, scales, bias, dims, opts)

      assert Nx.shape(got) == Nx.shape(want)
      assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(got, want)))) < 1.0e-3
    end
  end

  test "works without bias" do
    {w_i8, scales, _bias, dims} = fixture({2, 1, 1, 1})
    x = input({1, 3, 3, 1})
    got = QuantizedConv.apply(x, QuantizedConv.new(w_i8, scales, nil, dims))
    want = reference(x, w_i8, scales, nil, dims, [])
    assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(got, want)))) < 1.0e-4
  end

  test "rejects a weight buffer of the wrong size" do
    assert_raise ArgumentError, ~r/weight buffer/, fn ->
      QuantizedConv.new(<<0, 0>>, [1.0], nil, {1, 1, 1, 1})
    end
  end
end
