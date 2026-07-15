# HPC-SNN

Reproducible research framework for federated spiking neural networks with precision firing attention, adaptive spike-budget control, and HPC-oriented distributed scheduling.

## Quick start

```bash
uv sync --extra dev
uv run fedapfa-validate-config configs/base/debug.yaml
uv run pytest
```

Raw data and run artifacts are ignored; commit only curated results.
