defmodule NxPrimitives.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :nx_primitives,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "NxPrimitives",
      description:
        "Cross-platform Nx-tensor primitives — FFT, embeddings (cosine sim / top-k), quantized matmul + conv — with a pluggable native backend (see `NxPrimitives.Backend`).",
      package: package(),
      docs: [main: "readme", extras: ["README.md"]]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nx, "~> 0.9"},
      # In tests we use ArmAI's NxPrimitives backend impl.
      # Production users wire their own backend via config.
      {:arm_ai, path: "../arm_ai", only: [:dev, :test]},
      {:nx_arm, path: "../nx_arm", only: [:dev, :test]},
      {:rustler, "~> 0.36", optional: true},
      {:rustler_precompiled, "~> 0.8"}
    ]
  end

  defp package do
    [
      name: :nx_primitives,
      licenses: ["Apache-2.0"],
      files: ~w(lib mix.exs README.md),
      links: %{"GitHub" => "https://github.com/marclainez/nx_primitives"}
    ]
  end
end
