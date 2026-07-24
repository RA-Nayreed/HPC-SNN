from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

import pytest

from fedapfa.configuration import (
    COMPARATIVE_SEEDS,
    NON_IID_EXECUTION_ORDERS,
    NON_IID_TREATMENTS,
    SCALING_TOPOLOGIES,
    ConfigurationError,
    comparative_scientific_identity,
    load_comparative_allocations,
    load_comparative_evaluation_manifest,
    validate_comparative_path_disjointness,
    validate_resolved_comparative_manifest,
    validate_resolved_comparative_pair,
)

ROOT = Path(__file__).resolve().parents[2]
SCALING = ROOT / "experiments/system_scaling_energy_evaluation/manifest.yaml"
NON_IID = ROOT / "experiments/non_iid_energy_evaluation/manifest.yaml"
SCALING_SUBMIT = ROOT / "scripts/slurm/submit_roihu_system_scaling_energy.sh"
SCALING_BATCH = ROOT / "scripts/slurm/system_scaling_energy.sbatch"
SCALING_ALLOCATION_INDICES = {
    "one_node_one_gpu": [0, 1, 2, 12, 13, 14],
    "one_node_two_gpu": [3, 4, 5, 15, 16, 17],
    "one_node_four_gpu": [6, 7, 8, 18, 19, 20],
    "two_nodes_four_gpus": [9, 10, 11, 21, 22, 23],
}
SCALING_RESOURCES = {
    "one_node_one_gpu": (1, 1, 1, 72, "217086M"),
    "one_node_two_gpu": (1, 1, 2, 144, "434172M"),
    "one_node_four_gpu": (1, 1, 4, 288, "868344M"),
    "two_nodes_four_gpus": (2, 2, 2, 144, "434172M"),
}


def _write_command(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _run_scaling_submission(tmp_path: Path, max_parallel: int | None = None) -> list[list[str]]:
    command_dir = tmp_path / "commands"
    command_dir.mkdir()
    _write_command(command_dir / "module", "#!/usr/bin/env bash\nexit 0\n")
    _write_command(command_dir / "mkdir", "#!/usr/bin/env bash\nexit 0\n")
    record_path = tmp_path / "sbatch-arguments"
    _write_command(
        command_dir / "sbatch",
        r"""#!/usr/bin/env bash
set -euo pipefail
for argument in "$@"; do
    printf '%s\0' "${argument}" >> "${SBATCH_ARGUMENT_RECORD}"
done
printf '\n' >> "${SBATCH_ARGUMENT_RECORD}"
printf '319835\n'
""",
    )
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _write_command(
        venv / "bin" / "python3",
        r"""#!/usr/bin/env bash
set -euo pipefail
case "$*" in
    *"scientific_manifest allocation-count"*) printf '24\n' ;;
    *"scientific_manifest count"*) printf '24\n' ;;
    *"scientific_manifest validate"*) printf '{}\n' ;;
    -c*) exit 0 ;;
    *) exit 2 ;;
esac
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{command_dir}:{os.environ['PATH']}",
        "CSC_PROJECT": "project_2001234",
        "USER": "scientist",
        "FEDAPFA_VENV": str(venv),
        "SBATCH_ARGUMENT_RECORD": str(record_path),
    }
    command = [
        "bash",
        str(SCALING_SUBMIT),
        "--work-dir",
        "/scratch/project_2001234/scientist/week7",
    ]
    if max_parallel is not None:
        command.extend(("--max-parallel", str(max_parallel)))
    subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        [argument.decode() for argument in line.split(b"\0") if argument]
        for line in record_path.read_bytes().splitlines()
        if line
    ]


def _scaling_exports(invocations: list[list[str]]) -> dict[str, str]:
    exports = {}
    for invocation in invocations:
        export_arguments = [argument for argument in invocation if argument.startswith("--export=")]
        assert len(export_arguments) == 1
        specifications = export_arguments[0].removeprefix("--export=").split(",")
        topology_assignments = [value for value in specifications if value.startswith("SCALING_TOPOLOGY=")]
        index_assignments = [value for value in specifications if value.startswith("SCALING_ALLOCATION_INDICES=")]
        assert len(topology_assignments) == len(index_assignments) == 1
        topology = topology_assignments[0].partition("=")[2]
        exports[topology] = index_assignments[0].partition("=")[2]
    return exports


def _run_scaling_map_validation(transport: str, task_id: str) -> subprocess.CompletedProcess[str]:
    text = SCALING_BATCH.read_text(encoding="utf-8")
    start = text.index('[[ "${SCALING_ALLOCATION_INDICES}" =~')
    end = text.index('allocation="$("${python_bin}"', start)
    script = "set -euo pipefail\n" + text[start:end] + 'printf "%s\\n" "${manifest_index}"\n'
    return subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "SCALING_ALLOCATION_INDICES": transport,
            "SLURM_ARRAY_TASK_ID": task_id,
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def scaling_submissions(tmp_path: Path) -> list[list[str]]:
    return _run_scaling_submission(tmp_path)


def test_scaling_submission_exports_exact_slurm_safe_maps_and_resources(
    scaling_submissions: list[list[str]],
) -> None:
    assert len(scaling_submissions) == 4
    exports = _scaling_exports(scaling_submissions)
    assert set(exports) == set(SCALING_ALLOCATION_INDICES)
    recovered = {}
    for topology, transport in exports.items():
        assert "," not in transport
        values = transport.split(":")
        assert len(values) == 6
        assert all(value.isdecimal() for value in values)
        parsed = [int(value) for value in values]
        assert len(parsed) == len(set(parsed)) == 6
        recovered[topology] = parsed
    assert recovered == SCALING_ALLOCATION_INDICES
    flattened = [index for values in recovered.values() for index in values]
    assert sorted(flattened) == list(range(24))
    assert len(flattened) == len(set(flattened))
    for invocation in scaling_submissions:
        export_argument = next(argument for argument in invocation if argument.startswith("--export="))
        specifications = export_argument.removeprefix("--export=").split(",")
        topology = next(value.partition("=")[2] for value in specifications if value.startswith("SCALING_TOPOLOGY="))
        nodes, tasks, gpus_per_node, cpus, memory = SCALING_RESOURCES[topology]
        expected_arguments = {
            "--array=0-5",
            f"--nodes={nodes}",
            f"--ntasks={tasks}",
            "--ntasks-per-node=1",
            f"--gres=gpu:gh200:{gpus_per_node}",
            f"--cpus-per-task={cpus}",
            f"--mem={memory}",
        }
        assert expected_arguments <= set(invocation)


def test_scaling_submission_supports_optional_positive_throttle(tmp_path: Path) -> None:
    invocations = _run_scaling_submission(tmp_path, max_parallel=3)
    assert len(invocations) == 4
    assert all("--array=0-5%3" in invocation for invocation in invocations)


def test_scaling_array_tasks_resolve_exact_manifest_allocations(
    scaling_submissions: list[list[str]],
) -> None:
    exports = _scaling_exports(scaling_submissions)
    allocations = {allocation.allocation_index: allocation for allocation in load_comparative_allocations(SCALING)}
    expected_seeds = (37, 47, 57)
    for topology, transport in exports.items():
        for task_id, expected_index in enumerate(SCALING_ALLOCATION_INDICES[topology]):
            result = _run_scaling_map_validation(transport, str(task_id))
            assert result.returncode == 0, result.stderr
            assert int(result.stdout) == expected_index
            allocation = allocations[expected_index]
            expected_dataset = "shd" if task_id < 3 else "ssc"
            expected_seed = expected_seeds[task_id % 3]
            assert allocation.dataset == expected_dataset
            assert allocation.seed == expected_seed
            assert list(allocation.execution_order) == [topology]
            assert allocation.tasks[0].config["comparative_evaluation"]["treatment_id"] == topology


@pytest.mark.parametrize(
    ("transport", "task_id"),
    [
        ("0,1,2,12,13,14", "0"),
        ("0:1:2:12:13", "0"),
        ("0:1:2:12:13:14:15", "0"),
        ("0:1::2:12:13:14", "0"),
        ("0:1:two:12:13:14", "0"),
        ("0:1:2:12:13:13", "0"),
        ("0:1:2:12:13:-1", "0"),
        ("0:1:2:12:13:14", "6"),
        ("0:1:2:12:13:14", "-1"),
    ],
)
def test_scaling_batch_rejects_malformed_index_transport(transport: str, task_id: str) -> None:
    result = _run_scaling_map_validation(transport, task_id)
    assert result.returncode == 2
    assert "scaling allocation-index map is incompatible" in result.stderr


def test_comparative_manifests_have_exact_disjoint_matrices() -> None:
    scaling = load_comparative_evaluation_manifest(SCALING)
    non_iid = load_comparative_evaluation_manifest(NON_IID)
    assert len(scaling) == len(non_iid) == 24
    assert {task.seed for task in (*scaling, *non_iid)} == set(COMPARATIVE_SEEDS) == {37, 47, 57}
    assert {task.config["comparative_evaluation"]["treatment_id"] for task in scaling} == set(SCALING_TOPOLOGIES)
    assert {task.config["comparative_evaluation"]["treatment_id"] for task in non_iid} == set(NON_IID_TREATMENTS)
    paths = [(task.config["output_root"], task.experiment, task.seed) for task in (*scaling, *non_iid)]
    assert len(paths) == len(set(paths)) == 48
    validate_comparative_path_disjointness(SCALING, NON_IID)


def test_scaling_topologies_and_fixed_execution_policy_are_exact() -> None:
    tasks = load_comparative_evaluation_manifest(SCALING)
    observed = {}
    for task in tasks:
        treatment = task.config["comparative_evaluation"]["treatment_id"]
        parallel = task.config["parallel_execution"]
        observed[treatment] = (
            parallel["node_count"],
            parallel["devices_per_node"],
            parallel["device_count"],
            parallel["process_count"],
        )
        assert parallel["client_processes_per_device"] == 1
        assert parallel["cuda_process_service"] == "none"
        assert parallel["control_backend"] == "nccl"
        assert task.config["scheduler"]["strategy"] == "example_count_longest_processing_time"
        assert task.config["aggregation_execution"]["topology"] == "flat_ordered"
        assert task.config["federated"]["partition"]["alpha"] == 0.5
    assert observed == SCALING_TOPOLOGIES


def test_non_iid_treatments_and_rotated_orders_are_exact() -> None:
    tasks = load_comparative_evaluation_manifest(NON_IID)
    for task in tasks:
        treatment = task.config["comparative_evaluation"]["treatment_id"]
        partition = task.config["federated"]["partition"]
        assert (partition["method"], partition["alpha"]) == NON_IID_TREATMENTS[treatment]
        parallel = task.config["parallel_execution"]
        assert (
            parallel["node_count"],
            parallel["devices_per_node"],
            parallel["device_count"],
            parallel["process_count"],
        ) == (1, 4, 4, 4)
    allocations = load_comparative_allocations(NON_IID)
    assert len(allocations) == 6
    assert all(allocation.execution_order == NON_IID_EXECUTION_ORDERS[allocation.seed] for allocation in allocations)
    assert all(len(allocation.tasks) == 4 for allocation in allocations)


def test_scaling_has_one_execution_per_allocation() -> None:
    allocations = load_comparative_allocations(SCALING)
    assert len(allocations) == 24
    assert all(len(allocation.tasks) == len(allocation.execution_order) == 1 for allocation in allocations)


def test_resolved_whitelist_rejects_unintended_difference() -> None:
    tasks = load_comparative_evaluation_manifest(SCALING)
    reference = next(
        task.config
        for task in tasks
        if task.dataset == "shd"
        and task.seed == 37
        and task.config["comparative_evaluation"]["treatment_id"] == "one_node_one_gpu"
    )
    candidate = copy.deepcopy(reference)
    candidate["federated"]["learning_rate"] = 0.002
    with pytest.raises(ConfigurationError, match="outside the whitelist"):
        validate_resolved_comparative_pair(reference, candidate, "system_scaling_energy_evaluation")


def test_resolved_manifest_identity_is_checked_after_path_overrides() -> None:
    scaling = validate_resolved_comparative_manifest(
        SCALING,
        data_root="/tmp/comparative-data",
        output_root="/tmp/runs/system_scaling_energy_evaluation",
    )
    non_iid = validate_resolved_comparative_manifest(
        NON_IID,
        data_root="/tmp/comparative-data",
        output_root="/tmp/runs/non_iid_energy_evaluation",
    )
    assert len(scaling) == len(non_iid) == 24
    assert {(value["dataset"], value["seed"]) for value in scaling} == {
        (dataset, seed) for dataset in ("shd", "ssc") for seed in COMPARATIVE_SEEDS
    }


def test_scaling_scientific_identity_is_topology_independent() -> None:
    tasks = load_comparative_evaluation_manifest(SCALING)
    selected = [task for task in tasks if task.dataset == "shd" and task.seed == 37]
    identities = [comparative_scientific_identity(task.config) for task in selected]
    assert all(value == identities[0] for value in identities[1:])
    assert identities[0]["seed_streams"]["client_training"] == selected[0].config["seed_streams"]["client_training"]


def test_non_iid_whitelist_allows_only_distribution_identity() -> None:
    tasks = load_comparative_evaluation_manifest(NON_IID)
    iid = next(
        task.config
        for task in tasks
        if task.dataset == "ssc" and task.seed == 37 and task.config["comparative_evaluation"]["treatment_id"] == "iid"
    )
    alpha = next(
        task.config
        for task in tasks
        if task.dataset == "ssc"
        and task.seed == 37
        and task.config["comparative_evaluation"]["treatment_id"] == "dirichlet_alpha_0_1"
    )
    record = validate_resolved_comparative_pair(iid, alpha, "non_iid_energy_evaluation")
    assert record["whitelist_version"] == "resolved_leaf_paths_v1"
    assert set(record["observed_difference_paths"]) == {
        "name",
        "metadata.experiment",
        "comparative_evaluation.treatment_id",
        "federated.partition.method",
        "federated.partition.alpha",
    }


def test_comparative_slurm_launchers_encode_required_policy() -> None:
    scaling = (ROOT / "scripts/slurm/system_scaling_energy.sbatch").read_text(encoding="utf-8")
    non_iid = (ROOT / "scripts/slurm/non_iid_energy.sbatch").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/slurm/submit_roihu_system_scaling_energy.sh").read_text(encoding="utf-8")
    calibration = (ROOT / "scripts/slurm/comparative_measurement_calibration.sbatch").read_text(encoding="utf-8")
    for text in (scaling, non_iid):
        assert "gpumedium" in text
        assert "set -euo pipefail" in text
        assert '"${python_bin}" -m torch.distributed.run' in text
        assert "--kill-on-bad-exit=1" in text
        assert "TORCH_NCCL_ASYNC_ERROR_HANDLING=1" in text
        assert "export NCCL_ASYNC_ERROR_HANDLING" not in text
        assert "mpirun" not in text and "mpiexec" not in text
        assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=" not in text
        assert "|| true" not in text
    assert "--rdzv-backend=c10d" in scaling
    assert "--nnodes=2 --nproc-per-node=2" in scaling
    non_iid_launch = non_iid.index('srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1')
    non_iid_treatment_end = non_iid.index("record_comparative_allocation_timing end-treatment")
    non_iid_allocation_complete = non_iid.index("record_comparative_allocation_timing complete")
    assert non_iid_launch < non_iid_treatment_end < non_iid_allocation_complete
    assert "217086M 434172M 868344M 434172M" in wrapper
    assert "%1" not in wrapper
    assert "calibrate_comparative_measurement" in calibration
    assert "--rdzv-backend=c10d" in calibration
    assert "gpumedium" in calibration


def _fake_comparative_batch_environment(tmp_path: Path, *, visible_devices: int):
    command_dir = tmp_path / "commands"
    command_dir.mkdir()
    _write_command(command_dir / "module", "#!/usr/bin/env bash\nexit 0\n")
    _write_command(
        command_dir / "nvidia-smi",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%b\\n' \"${FAKE_RAW_UUID_LINES}\"\n",
    )
    _write_command(
        command_dir / "scontrol",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'node-a\\n'\n",
    )
    _write_command(
        command_dir / "srun",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\t%s\n' "${FEDAPFA_TREATMENT_POSITION:-preflight}" "$*" >> "${SRUN_INVOCATION_RECORD}"
if [[ "$*" == *"torch.distributed.run"* ]]; then
    if [[ "${SRUN_FAIL_POSITION}" == "all" || "${FEDAPFA_TREATMENT_POSITION:-}" == "${SRUN_FAIL_POSITION}" ]]; then
        exit 41
    fi
    exit 0
fi
while (($#)) && [[ "$1" == --* ]]; do
    shift
done
export SLURM_PROCID="${SLURM_PROCID:-0}"
exec "$@"
""",
    )
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _write_command(
        venv / "bin" / "python3",
        r"""#!/usr/bin/env bash
set -euo pipefail
case "$*" in
    *"record_comparative_allocation_timing"*) printf '%s\n' "$3" >> "${TIMING_ACTION_RECORD}" ;;
    *"scientific_manifest allocation-task"*) printf '%s\n' "${FAKE_ALLOCATION_TASK}" ;;
    *"scientific_manifest validate"*) exit 0 ;;
    *"allocated_gpu_uuids"*) printf '%s\n' "${FAKE_CANONICAL_UUIDS}" ;;
    *"torch.cuda.device_count"*) printf '%s\n' "${FAKE_VISIBLE_DEVICE_COUNT}" ;;
    -c*) exit 0 ;;
    *) exit 2 ;;
esac
""",
    )
    work_dir = tmp_path / "work"
    data_root = work_dir / "data" / "shd"
    data_root.mkdir(parents=True)
    (data_root / "shd_train.h5").touch()
    (data_root / "shd_test.h5").touch()
    job_tmp = tmp_path / "job-tmp"
    job_tmp.mkdir()
    timing_record = tmp_path / "timing-actions.txt"
    srun_record = tmp_path / "srun-invocations.txt"
    uuids = [f"GPU-{index}" for index in range(visible_devices)]
    environment = {
        **os.environ,
        "PATH": f"{command_dir}:{os.environ['PATH']}",
        "CSC_PROJECT": "project_2001234",
        "USER": "scientist",
        "FEDAPFA_VENV": str(venv),
        "WORK_DIR": str(work_dir),
        "REPO_ROOT": str(ROOT),
        "EVALUATION_MANIFEST": str(SCALING),
        "SLURM_ARRAY_JOB_ID": "900",
        "SLURM_JOB_ID": "901",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_JOB_NODELIST": "node-a",
        "TMPDIR": str(job_tmp),
        "FAKE_VISIBLE_DEVICE_COUNT": str(visible_devices),
        "FAKE_RAW_UUID_LINES": "\n".join(uuids),
        "FAKE_CANONICAL_UUIDS": ",".join(value.removeprefix("GPU-") for value in uuids),
        "TIMING_ACTION_RECORD": str(timing_record),
        "SRUN_INVOCATION_RECORD": str(srun_record),
    }
    return environment, work_dir, timing_record, srun_record


def test_scaling_batch_propagates_failed_srun_and_cannot_report_success(tmp_path: Path) -> None:
    environment, work_dir, timing_record, srun_record = _fake_comparative_batch_environment(
        tmp_path, visible_devices=1
    )
    calibration = (
        work_dir
        / "calibration/system_scaling_energy_evaluation/one_node_one_gpu/instrumentation_calibration.json"
    )
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}", encoding="utf-8")
    environment.update(
        {
            "SLURM_ARRAY_TASK_ID": "0",
            "SCALING_TOPOLOGY": "one_node_one_gpu",
            "SCALING_ALLOCATION_INDICES": "0:1:2:12:13:14",
            "FAKE_ALLOCATION_TASK": "0\tshd\t37\tone_node_one_gpu\tconfig.yaml",
            "SRUN_FAIL_POSITION": "all",
        }
    )

    result = subprocess.run(
        ["bash", str(SCALING_BATCH)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 41
    assert timing_record.read_text(encoding="utf-8").splitlines() == ["start", "begin-treatment"]
    assert "torch.distributed.run" in srun_record.read_text(encoding="utf-8")


def test_non_iid_batch_stops_at_first_unrecoverable_treatment_failure(tmp_path: Path) -> None:
    environment, work_dir, timing_record, srun_record = _fake_comparative_batch_environment(
        tmp_path, visible_devices=4
    )
    calibration = (
        work_dir
        / "calibration/system_scaling_energy_evaluation/one_node_four_gpu/instrumentation_calibration.json"
    )
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}", encoding="utf-8")
    environment.update(
        {
            "EVALUATION_MANIFEST": str(NON_IID),
            "SLURM_ARRAY_TASK_ID": "0",
            "FAKE_ALLOCATION_TASK": (
                "0\tshd\t37\tiid\tiid.yaml\tdirichlet_alpha_1_0\ta1.yaml\t"
                "dirichlet_alpha_0_5\ta05.yaml\tdirichlet_alpha_0_1\ta01.yaml"
            ),
            "SRUN_FAIL_POSITION": "2",
        }
    )

    result = subprocess.run(
        ["bash", str(ROOT / "scripts/slurm/non_iid_energy.sbatch")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 41
    assert timing_record.read_text(encoding="utf-8").splitlines() == [
        "start",
        "begin-treatment",
        "end-treatment",
        "begin-treatment",
    ]
    training_positions = [
        line.partition("\t")[0]
        for line in srun_record.read_text(encoding="utf-8").splitlines()
        if "torch.distributed.run" in line
    ]
    assert training_positions == ["1", "2"]
