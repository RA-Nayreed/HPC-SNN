import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SBATCH = ROOT / "scripts/slurm/resource_measurement.sbatch"
SUBMIT = ROOT / "scripts/slurm/submit_roihu_resource_measurement.sh"


def test_resource_allocation_and_project_interpreter_contract():
    text = SBATCH.read_text()
    assert "#SBATCH --partition=gpumedium" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --gres=gpu:gh200:1" in text
    assert "#SBATCH --cpus-per-task=72" in text
    assert "#SBATCH --mem=217086M" in text
    assert "#SBATCH --time=24:00:00" in text
    assert "module load python-pytorch/2.10" in text
    assert 'python_bin="${venv}/bin/python3"' in text
    assert '"${python_bin}" -m torch.distributed.run' in text
    assert not re.search(r"(^|[ /])torchrun([ \"']|$)", text)


def test_calibration_precedes_sequential_six_task_execution():
    text = SBATCH.read_text()
    calibration = text.index("fedapfa.cli.calibrate_resource_measurement")
    iteration = text.index("for ((index = 0; index < task_count; index++))")
    training = text.index("fedapfa.cli.train_resource_measurement")
    assert calibration < iteration < training
    assert "instrumentation_calibration.json" in text
    assert "--resume-auto" in text
    assert "SHD" in text and "SSC" in text
    assert "nvidia-cuda-mps" not in text
    assert "CUDA_MPS_PIPE_DIRECTORY" in text
    assert 'cuda_process_service}" == "none"' in text
    assert "shd_train.h5" in text and "shd_test.h5" in text
    assert "ssc_train.h5" in text and "ssc_valid.h5" in text and "ssc_test.h5" in text
    assert "this job never downloads data" in text


def test_submission_wrapper_has_one_job_and_no_array_controls():
    text = SUBMIT.read_text()
    assert '--account="${CSC_PROJECT}"' in text
    assert "sbatch --parsable" in text
    assert "--array" not in text
    assert "max-parallel" not in text
    assert "scratch_prefix=" in text and "/scratch/" in text
    assert "slurm-logs/resource_measurement" in text
    assert '[[ "${task_count}" == "6" ]]' in text
    assert "printf '%s\\n' \"${job_id}\"" in text


def test_resource_shell_scripts_parse():
    subprocess.run(["bash", "-n", str(SBATCH)], check=True)
    subprocess.run(["bash", "-n", str(SUBMIT)], check=True)
