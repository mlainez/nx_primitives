defmodule ArmNxPrimitives.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :arm_nx_primitives,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "ArmNxPrimitives",
      description:
        "Cross-domain Nx-tensor primitives over arm_ai: FFT, embeddings (cosine sim / top-k), quantized matmul + conv",
      package: package(),
      docs: [main: "readme", extras: ["README.md"]]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:rustler, "~> 0.36", optional: true},
      {:rustler_precompiled, "~> 0.8"},
      {:nx, "~> 0.9"},
      {:arm_ai, path: "../arm_ai"},
      {:nx_arm, path: "../nx_arm"}
    ]
  end

  defp package do
    [
      name: :arm_nx_primitives,
      licenses: ["Apache-2.0"],
      files: ~w(lib mix.exs README.md),
      links: %{"GitHub" => "https://github.com/marclainez/arm_nx_primitives"}
    ]
  end
end
