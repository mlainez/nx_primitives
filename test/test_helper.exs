ExUnit.start()

# Wire ArmAI.NxPrimitivesBackend as the FFT/quantized backend
# so the test suite has a concrete impl to drive.
Application.put_env(:nx_primitives, :backend, ArmAI.NxPrimitivesBackend)
