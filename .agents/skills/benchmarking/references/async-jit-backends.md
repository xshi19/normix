# Async and JIT timing

- **JAX:** warm each shape/dtype/static-argument/device signature. Use `jax.block_until_ready` on inputs and warmup results before timing, and on every timed result before stopping.
- **PyTorch CUDA:** after warmup, call `torch.cuda.synchronize(device)` before and after timing on every participating device. Unrelated queued work can inflate this measurement.
- Separate compilation from warm execution, including with `torch.compile`. A first call may include tracing, compilation, allocation, and execution; call it first-call latency unless phases are isolated. Record persistent compilation-cache state and recompilations.
- Per-call barriers measure latency and suppress overlap. For throughput, time representative submissions with barriers at batch boundaries; report batch size and completed work.
- Include transfers for end-to-end claims; exclude and disclose them for device-kernel measurements. Unsynchronized dispatch timings do not establish completed-work speedups.
