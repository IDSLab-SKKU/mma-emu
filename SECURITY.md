# Security Policy

## Scope

This repository is a research fork of
[vLLM](https://github.com/vllm-project/vllm) that emulates MMA accumulation
arithmetic. It is intended for studying the numerical behaviour of quantized
inference, **not for production serving**.

The emulation kernels run on CUDA cores and are far slower than the tensor-core
paths they replace. They are off by default: the kernels decline every layer
unless an accumulation algorithm is configured. The fork inherits whatever
security posture upstream vLLM had at the commit it branched from, and does
not track subsequent fixes.

## Reporting

**Issues in this fork's own code** — the kernels under
`csrc/libtorch_stable/quantization/mma_emu/`, the `mma_emu.py` kernel modules,
or the upstream files this fork modifies (listed in [NOTICE](NOTICE)) — should
be reported privately through
[GitHub's private vulnerability reporting](https://github.com/IDSLab-SKKU/NADPE/security/advisories/new)
for this repository.

**Issues in vLLM itself** should go to
[vLLM's vulnerability submission form](https://github.com/vllm-project/vllm/security/advisories/new),
not here. This fork does not maintain vLLM and cannot ship fixes for it. If a
report turns out to affect upstream rather than this fork, we will say so and
ask you to refile there.

This is a small research project. Expect a slower response than an actively
maintained distribution would give, and do not depend on it where that matters.

## Related policies

- [vLLM Security Guide](https://docs.vllm.ai/en/latest/usage/security.html) —
  the deployment assumptions this fork inherits.
- [PyTorch's Security Policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md)
  — on interacting with models safely. Treat untrusted checkpoints with the
  same care you would untrusted code.
