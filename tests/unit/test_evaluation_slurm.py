import re
from pathlib import Path

SCHEDULING_SUBMIT = Path("scripts/slurm/submit_roihu_scheduling_evaluation.sh").read_text()
SCHEDULING_BATCH = Path("scripts/slurm/scheduling_evaluation.sbatch").read_text()
HIERARCHY_SUBMIT = Path("scripts/slurm/submit_roihu_hierarchical_reduction.sh").read_text()
HIERARCHY_BATCH = Path("scripts/slurm/hierarchical_reduction.sbatch").read_text()


def _assert_current_nccl_error_handling(text):
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING=1" in text
    assert re.search(r"export[^\n]*\bNCCL_ASYNC_ERROR_HANDLING=", text) is None


def test_scheduling_launch_is_one_node_four_gpu_project_python_and_unthrottled_by_default():
    assert "#SBATCH --partition=gpumedium" in SCHEDULING_BATCH
    assert "#SBATCH --nodes=1" in SCHEDULING_BATCH
    assert "#SBATCH --gres=gpu:gh200:4" in SCHEDULING_BATCH
    assert '"${python_bin}" -m torch.distributed.run' in SCHEDULING_BATCH
    assert "--nproc-per-node=4" in SCHEDULING_BATCH
    assert "torchrun" not in SCHEDULING_BATCH
    assert "FEDAPFA_ALLOCATED_GPU_UUIDS" in SCHEDULING_BATCH
    assert "exactly four GPU UUIDs" in SCHEDULING_BATCH
    assert 'array="0-5"' in SCHEDULING_SUBMIT
    assert 'array+="%${max_parallel}"' in SCHEDULING_SUBMIT
    assert 'max_parallel=""' in SCHEDULING_SUBMIT
    assert "scheduling_evaluation_job_id=" in SCHEDULING_SUBMIT
    assert "CUDA_MPS" in SCHEDULING_BATCH and "unset CUDA_MPS" in SCHEDULING_BATCH
    _assert_current_nccl_error_handling(SCHEDULING_BATCH)


def test_hierarchy_launch_has_exact_rendezvous_node_ranks_and_failure_propagation():
    assert "#SBATCH --partition=gpumedium" in HIERARCHY_BATCH
    assert "#SBATCH --nodes=2" in HIERARCHY_BATCH
    assert "#SBATCH --gres=gpu:gh200:2" in HIERARCHY_BATCH
    assert 'master_addr="${allocated_nodes[0]}"' in HIERARCHY_BATCH
    assert '--node-rank="${SLURM_PROCID}"' in HIERARCHY_BATCH
    assert "--nnodes=2 --nproc-per-node=2" in HIERARCHY_BATCH
    assert "--kill-on-bad-exit=1" in HIERARCHY_BATCH
    assert '"${EVALUATION_PYTHON}" -m torch.distributed.run' in HIERARCHY_BATCH
    assert "torchrun" not in HIERARCHY_BATCH
    assert "FEDAPFA_GPU_UUID_PREFLIGHT_DIR" in HIERARCHY_BATCH
    assert "exactly four distinct GPU UUIDs" in HIERARCHY_BATCH
    assert "hierarchical_reduction_job_id=" in HIERARCHY_SUBMIT
    assert 'max_parallel=""' in HIERARCHY_SUBMIT
    assert "FEDAPFA_GPU_TELEMETRY_NODE_FILES" in HIERARCHY_BATCH
    assert ".node-0.csv" in HIERARCHY_BATCH and ".node-1.csv" in HIERARCHY_BATCH
    assert "fedapfa.cli.merge_gpu_telemetry" in HIERARCHY_BATCH
    assert "trap stop_telemetry EXIT" in HIERARCHY_BATCH
    _assert_current_nccl_error_handling(HIERARCHY_BATCH)
