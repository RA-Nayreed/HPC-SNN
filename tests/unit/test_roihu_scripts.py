from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_roihu_batch_resources_and_commands():
    test = (ROOT / "scripts/slurm/roihu_week01_gputest.sbatch").read_text()
    medium = (ROOT / "scripts/slurm/roihu_week01_gpumedium.sbatch").read_text()
    for text, partition in [(test, "gputest"), (medium, "gpumedium")]:
        assert f"#SBATCH --partition={partition}" in text
        assert "#SBATCH --nodes=1" in text
        assert "#SBATCH --ntasks=1" in text
        assert "#SBATCH --cpus-per-task=72" in text
        assert "#SBATCH --gres=gpu:gh200:1" in text
        assert "python-pytorch/2.10" in text
        assert "srun python3 -m fedapfa.cli.train_centralized" in text
        assert "--mem" not in text
    assert "#SBATCH --time=00:15:00" in test
    assert "#SBATCH --time=36:00:00" in medium


def test_submission_wrapper_is_dry_run_capable_and_safe():
    text = (ROOT / "scripts/slurm/submit_roihu_week01.sh").read_text()
    assert '--account="$CSC_PROJECT"' in text
    assert "--dry-run" in text
    assert "full training is forbidden on gputest" in text
    assert "smoke runs require --allow-smoke-gpumedium" in text
    assert "((dry_run)) ||" in text
