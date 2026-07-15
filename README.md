# HPC-SNN

Federated adaptive Parameter-free Attention for spiking neural networks.

The implemented centralized path supports official SHD and SSC event files, deterministic 10 ms/140-channel preprocessing, configurable LIF models, equation and public-behavior PfA, bounded smoke runs, tiny-overfit acceptance, and reproducible run artifacts.

## CPU setup and checks

Install CPU PyTorch from the official PyTorch index, then install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -e ".[dev]"
ruff check src tests
pytest
```

## Data and configuration commands

```bash
fedapfa-download-data shd
fedapfa-download-data ssc
fedapfa-inspect-data data/raw/shd/shd_train.h5
fedapfa-validate-config experiments/week01_pfa_reproduction/01_tiny_overfit.yaml
fedapfa-train-centralized experiments/week01_pfa_reproduction/01_tiny_overfit.yaml
fedapfa-train-centralized experiments/week01_pfa_reproduction/02_plain_lif_shd.yaml --device cpu
fedapfa-train-centralized experiments/week01_pfa_reproduction/09_ssc_smoke.yaml --device cpu
fedapfa-evaluate-checkpoint experiments/week01_pfa_reproduction/10_shd_plain_lif_full.yaml runs/<run-id>/checkpoints/best_validation.pt
```

Raw datasets and run outputs remain excluded from Git. Roihu setup and dry-run submission commands are documented in `environment/roihu/README.md`.
