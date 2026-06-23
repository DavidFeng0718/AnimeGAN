# Colab Runtime Target

The default Colab image is not a pinned project dependency. Treat it as a
moving runtime and verify it at notebook startup:

```bash
python -m sg2ada.env --check --require-cuda
```

This project targets:

- Python 3.10+
- PyTorch 2.7+
- torchvision 0.22+
- CUDA-enabled GPU runtime

The Colab launcher installs `requirements.txt` with pip's default
`only-if-needed` upgrade strategy, so a compatible preinstalled Colab PyTorch
stack is reused instead of being replaced.

Reference points checked during the refactor:

- The public `googlecolab/colabtools` repository says it contains Colab's Python
  libraries, but it does not publish runtime releases.
- PyTorch's current stable install page lists Stable 2.7.0, requires Python
  3.10+, and offers CUDA 12.6/12.8 wheels.
