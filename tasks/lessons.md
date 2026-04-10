# Lessons

## 2026-04-09

- Mistake: Treated MLX `bfloat16` hidden states as directly convertible with `np.asarray(..., dtype=np.float32)`.
  Root cause: The MLX buffer format for `bfloat16` did not match NumPy's requested dtype coercion path.
  Preventative rule: Cast MLX arrays to `mx.float32` before crossing into NumPy or Torch.

- Mistake: Loaded persisted runtime state with bare `torch.load(...)`.
  Root cause: PyTorch 2.6 now defaults `weights_only=True`, which rejects the saved genome payload.
  Preventative rule: For trusted local checkpoints that include non-tensor Python objects, pass `weights_only=False` explicitly.
