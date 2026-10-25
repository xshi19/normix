# GPU performance

normix runs on a GPU whenever the JAX arrays live there: choose the device with
`jax.default_device` or by placing arrays on it. `backend='jax'` always runs on
the JAX device and `backend='cpu'` always runs on the CPU; normix never switches
between them on its own.

## Why small GPU programs are launch-bound

The Bessel quadrature (`log_kv` and its derivatives) finds an integration window
with fixed-length loops: a few doubling steps and tens of bisection steps. Each
step is a tiny GPU kernel, and a GPU launches kernels one after another. The
arithmetic is negligible, so the run time is the cost of launching a few hundred
kernels in sequence. A CPU has no such cost for a loop of scalar operations,
which is why the same code can be faster on the CPU for small inputs.

## Optional: XLA command buffers

XLA can record a sequence of kernels once as a CUDA graph and replay it with a
single launch. Loops (`WHILE`) and branches (`CONDITIONAL`) are not recorded by
default. Enable them before JAX starts:

```bash
export XLA_FLAGS="--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL --xla_gpu_graph_min_graph_size=1"
```

or, in Python, before the first JAX operation (before `import jax` is safest):

```python
import os
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL "
    "--xla_gpu_graph_min_graph_size=1"
)
```

What the flags mean:

- `--xla_gpu_enable_command_buffer` is a comma-separated list of command types
  that XLA records into CUDA graphs. Setting it replaces the default list
  (`FUSION, CUBLAS, CUBLASLT, CUSTOM_CALL, CUDNN, DYNAMIC_SLICE_FUSION` on
  jaxlib 0.9.1). A `+` prefix adds a type to the default list instead.
  An empty value turns command buffers off.
- `--xla_gpu_graph_min_graph_size` is the smallest number of kernels worth
  recording. The default is 5; with 1, short sequences are recorded too.

Measured effect (one RTX 4090 under WSL2, jax 0.9.1, steady-state time of
`jax.jacrev` of `GIG.from_expectation`; compile time is about 2.2 s in every
row):

| Setting | Steady-state time |
|---|---|
| Default | 0.0128 s |
| `+WHILE,+CONDITIONAL` | 0.0100 s |
| `+WHILE,+CONDITIONAL` with `min_graph_size=1` | 0.0088 s |
| Command buffers off | 0.0142 s |

Results are identical to the printed digits. Batch EM fits of NIG and GH on
5,000 points did not get faster (their time is not dominated by Bessel launch
cost) and gave the same final log-likelihood.

Caveats:

- The flag is global: it applies to every jitted program in the process, not
  only to the Bessel code.
- CUDA graphs use extra device memory (about 20 MiB more in the EM test above).
  If you run out of memory while instantiating graphs, clear the flag.
- Only one workload on one GPU model was measured. Treat the flag as
  experimental, and measure your own workload before relying on it.
- It must be set before JAX initializes the GPU backend; changing it later has
  no effect.
