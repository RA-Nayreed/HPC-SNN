import os
import platform
import socket
import sys

import torch


def environment_metadata():
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    nccl_version = torch.cuda.nccl.version() if torch.distributed.is_nccl_available() else None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": nccl_version,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "visible_gpu_count": gpu_count,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_count else None,
        "hostname": socket.gethostname(),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"),
        "SLURM_JOB_ACCOUNT": os.environ.get("SLURM_JOB_ACCOUNT"),
        "SLURM_NTASKS": os.environ.get("SLURM_NTASKS"),
        "SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
        "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION"),
        "SLURM_NODELIST": os.environ.get("SLURM_NODELIST"),
    }
