from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_centralized_array_is_strict_and_uses_verified_roihu_stack():
    text = (ROOT / "scripts/slurm/roihu_centralized_array.sbatch").read_text()
    lowered = text.lower()
    assert "#SBATCH --partition=gpumedium" in text
    assert "#SBATCH --cpus-per-task=72" in text
    assert "#SBATCH --gres=gpu:gh200:1" in text
    assert "#SBATCH --time=36:00:00" in text
    assert "python-pytorch/2.10" in text
    assert "/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv" in text
    assert "import DCLS" in text
    assert "import dcls" not in text
    assert "--resume-auto" in text
    assert "/data/shd" in text and "/data/ssc" in text
    assert "/runs/centralized" in text
    assert "reduced[_-]?sample|sweep|memorization" in text
    assert "gputest" not in lowered
    assert "dry-run" not in lowered


def test_centralized_submission_wrapper_has_limited_concurrency_and_no_diagnostic_path():
    text = (ROOT / "scripts/slurm/submit_roihu_centralized.sh").read_text()
    lowered = text.lower()
    assert "max_parallel=4" in text
    assert "--max-parallel" in text
    assert '--account="${CSC_PROJECT}"' in text
    assert "task_count" in text and "18" in text
    assert "/scratch/${CSC_PROJECT}/" in text
    assert "/slurm-logs/centralized" in text
    assert "Submitted job ID:" in text
    assert "gputest" not in lowered
    assert "dry-run" not in lowered


def test_centralized_shell_scripts_parse():
    import subprocess

    for name in ("roihu_centralized_array.sbatch", "submit_roihu_centralized.sh"):
        subprocess.run(["bash", "-n", str(ROOT / "scripts/slurm" / name)], check=True)
