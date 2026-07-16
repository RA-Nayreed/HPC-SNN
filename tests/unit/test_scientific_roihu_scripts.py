import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
ARRAYS = (
    "heterogeneity_array.sbatch",
    "heterogeneity_context_array.sbatch",
    "published_fedsnn_array.sbatch",
)
WRAPPERS = ("submit_roihu_heterogeneity.sh", "submit_roihu_published_fedsnn.sh")


@pytest.mark.parametrize("name", ARRAYS)
def test_scientific_arrays_use_required_roihu_resources_and_telemetry(name):
    text = (ROOT / "scripts/slurm" / name).read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpumedium" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --gres=gpu:gh200:1" in text
    assert "python-pytorch/2.10" in text
    assert "torch.cuda.is_available()" in text
    assert "nvidia-smi" in text and "sleep 2" in text
    assert "/scratch/${CSC_PROJECT}/${USER}/" in text


def test_training_arrays_use_automatic_resume_and_required_task_counts():
    heterogeneity = (ROOT / "scripts/slurm/heterogeneity_array.sbatch").read_text(encoding="utf-8")
    published = (ROOT / "scripts/slurm/published_fedsnn_array.sbatch").read_text(encoding="utf-8")
    assert "--resume-auto" in heterogeneity and '"9"' in heterogeneity
    assert "--resume-auto" in published and '"3"' in published


@pytest.mark.parametrize("name", (*ARRAYS, *WRAPPERS))
def test_scientific_shell_scripts_parse(name):
    subprocess.run(["bash", "-n", str(ROOT / "scripts/slurm" / name)], check=True)
