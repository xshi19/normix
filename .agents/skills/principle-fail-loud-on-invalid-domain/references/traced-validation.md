# Validation under tracing

- Ordinary Python exceptions cannot branch on unknown traced values. Use the backend's runtime checks or an explicit failure status with invalid outputs; tracing does not prohibit every runtime error mechanism.
- Ensure failure reaches the caller after compilation and batching.
- Making inactive branch inputs safe for differentiation can preserve the function; flooring the selected answer can change it.
