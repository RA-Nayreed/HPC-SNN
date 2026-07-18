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


def test_federated_array_uses_one_gh200_and_job_level_telemetry():
    text = (ROOT / "scripts/slurm/fedavg_single_gpu.sbatch").read_text()
    assert "#SBATCH --partition=gpumedium" in text
    assert "#SBATCH --gres=gpu:gh200:1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "python-pytorch/2.10" in text
    assert "/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv" in text
    assert "torch.cuda.is_available()" in text
    assert "nvidia-smi" in text and "sampling_interval_seconds=2" in text
    assert "--resume-auto" in text
    assert "/runs/federated" in text and "/telemetry/federated" in text


def test_federated_submission_defaults_to_one_and_limits_six_tasks():
    text = (ROOT / "scripts/slurm/submit_roihu_federated.sh").read_text()
    assert "max_parallel=1" in text
    assert '[[ "${max_parallel}" =~ ^[1-6]$ ]]' in text
    assert '[[ "${task_count}" == "6" ]]' in text
    assert "/slurm-logs/federated" in text
    assert "Submitted job ID:" in text


def test_federated_shell_scripts_parse():
    import subprocess

    for name in ("fedavg_single_gpu.sbatch", "submit_roihu_federated.sh"):
        subprocess.run(["bash", "-n", str(ROOT / "scripts/slurm" / name)], check=True)


def test_distributed_array_uses_manifest_topology_and_mps_cleanup():
    text = (ROOT / "scripts/slurm/federated_multigpu.sbatch").read_text()
    assert "#SBATCH --partition=gpumedium" in text
    assert "#SBATCH --nodes=1" in text and "#SBATCH --ntasks=1" in text
    assert "#SBATCH --gres" not in text
    assert "#SBATCH --time=36:00:00" in text
    assert 'torchrun --standalone --nnodes=1 --nproc-per-node="${process_count}"' in text
    assert "--resume-auto" in text
    assert "srun torchrun" in text
    assert "nvidia-cuda-mps-control -d" in text
    assert "CUDA_MPS_PIPE_DIRECTORY" in text and "CUDA_MPS_LOG_DIRECTORY" in text
    assert "100 / client_processes_per_device" in text
    assert text.index("mps_started=1") < text.index("get_server_list")
    assert "trap stop_execution_services EXIT" in text
    assert "trap 'exit 130' INT" in text and "trap 'exit 143' TERM" in text
    assert "printf 'quit\\n' | nvidia-cuda-mps-control" in text
    assert '${SLURM_TMPDIR}/fedapfa_mps/${job_label}' in text


def test_distributed_submission_groups_24_tasks_by_physical_device_count():
    text = (ROOT / "scripts/slurm/submit_roihu_distributed_evaluation.sh").read_text()
    assert "max_parallel=1" in text
    assert "expected_task_count=24" in text and "expected_task_count=9" in text
    assert "--datasets" in text and "--device-counts" in text and "--collection" in text
    assert "device_capacity_evaluation" in text
    assert '--gres="gpu:gh200:${device_count}"' in text
    assert '--array="${indices[${device_count}]}%${max_parallel}"' in text
    assert '--account="${CSC_PROJECT}"' in text
    assert '/slurm-logs/${collection}' in text
    assert "one_gpu_job_id=" in text
    assert "two_gpu_job_id=" in text
    assert "four_gpu_job_id=" in text


def test_distributed_shell_scripts_parse():
    import subprocess

    for name in ("federated_multigpu.sbatch", "submit_roihu_distributed_evaluation.sh"):
        subprocess.run(["bash", "-n", str(ROOT / "scripts/slurm" / name)], check=True)
