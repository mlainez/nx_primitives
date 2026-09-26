defmodule NxPrimitives.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :nx_primitives,
      version: @version,
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "NxPrimitives",
      description:
        "Nx-tensor primitives for edge inference — FFT (pluggable backend), embeddings (cosine sim / top-k), int8 quantized matmul + conv on the arm_ai NIF.",
      package: package(),
      docs: [main: "readme", extras: ["README.md"]]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp elixirc_paths(:test), do: ["lib"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nx, "~> 0.12.0"},
      # Optional: Embeddings / Quantized / QuantizedConv call the arm_ai
      # NIF directly, and ArmAI.NxPrimitivesBackend is the FFT backend.
      {:arm_ai, github: "mlainez/arm_ai", optional: true},
      {:nx_arm, github: "mlainez/nx_arm", optional: true}
    ]
  end

  defp package do
    [
      name: :nx_primitives,
      licenses: ["Apache-2.0"],
      files: ~w(lib mix.exs README.md LICENSE),
      links: %{"GitHub" => "https://github.com/mlainez/nx_primitives"}
    ]
  end
end
