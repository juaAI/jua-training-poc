#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import sys
import atexit
import signal
import subprocess
from typing import Any, List

def main(args):
    # Get the directory where this script is running from
    # This will be the run directory since launch_sbatch.py copies this script there
    run_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Run directory: {run_dir}")
    
    # Config file is in the same directory
    config_path = os.path.join(run_dir, "config.yaml")
    print(f"Config path: {config_path}")
    print(f"Config exists: {os.path.exists(config_path)}")

    docker_run_args = [
        "--init",
        "--rm",
        "-m",
        "940g",
        "--shm-size",
        "500g",
        "--gpus",
        "all",
        "--network",
        "host",
        "--privileged",
        "--ulimit",
        "nofile=1000000",
        "-w",
        "/app",
        "-e",
        "PYTHONPATH=/app",
        "-v",
        "/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu",
        "-v",
        "/opt/hpcx:/opt/hpcx",
        "-v",
        "/dev/infiniband:/dev/infiniband",
        "-v",
        "/mnt/jua-shared-1:/mnt/data",
        "-e",
        "LD_LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib:/opt/hpcx/ucc/lib/ucc:/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib/ucx:/opt/hpcx/ucx/lib:/opt/hpcx/sharp/lib:/opt/hpcx/hcoll/lib:/opt/hpcx/ompi/lib",
        "-v",
        f"{config_path}:/app/config.yaml:ro",
        "-v",
        "/etc/crusoe/nccl_topo/h100-80gb-sxm-ib-cloud-hypervisor.xml:/etc/crusoe/nccl_topo/h100-80gb-sxm-ib-cloud-hypervisor.xml:ro",
        "-e",
        f"WANDB_API_KEY={args.wandb_api_key}",
        "-e",
        "NCCL_TOPO_FILE=/etc/crusoe/nccl_topo/h100-80gb-sxm-ib-cloud-hypervisor.xml",
        "-e",
        "NCCL_SOCKET_NTHREADS=4",
        "-e",
        "NCCL_NSOCKS_PERTHREAD=8",
        "-e",
        "NCCL_IB_MERGE_VFS=0",
        "-e",
        "NCCL_IB_HCA=^mlx5_0:1",
    ]

    base_path = Path.cwd().resolve()
    accelerate_config_path = base_path / "accelerate_config.yaml"
    has_accelerate_config = accelerate_config_path.is_file()
    if has_accelerate_config:
        docker_run_args.extend(
            ["-v", f"{accelerate_config_path}:/accelerate_config.yaml:ro"]
        )

    cmd = [
        "sudo",
        "-n",
        "/usr/bin/docker",
        "run",
        *docker_run_args,
        image_tag,
        "poetry",
        "run",
        "accelerate",
        "launch",
        "--multi_gpu",
        "--mixed_precision",
        "fp16",
        "--num_processes",
        str(args.num_processes),
        "--num_machines",
        str(args.num_machines),
        "--machine_rank",
        machine_rank,
        "--main_process_ip",
        args.main_process_ip,
        "--main_process_port",
        "20671",
        "source/trainer.py",
        "--run_id",
        f"{args.run_id}-{args.attempt}",
        "--config",
        "/app/config.yaml"
    ]

    if args.dry:
        cmd = [c if c else "''" for c in cmd]
        print(" ".join(cmd))
        return
    
    print("Running command:")
    print(" ".join(cmd))

    run_proc(cmd)

def run_proc(cmd: List[str]):
    global proc  # noqa: PLW0603
    proc = subprocess.Popen(cmd)  # noqa: S603
    return_code = proc.wait()
    if return_code:
        sys.exit(return_code)

def kill_proc():
    if proc is None:
        return

    print("Killing proc and all docker containers")
    proc.terminate()
    
    # Check if there are any running containers before trying to kill them
    running_containers = subprocess.run(
        ["sudo", "-n", "docker", "ps", "-q"],
        capture_output=True,
        text=True
    ).stdout.strip()
    
    if running_containers:
        subprocess.run(["sudo", "-n", "docker", "kill", *running_containers.split('\n')])

def handle_signal(signo: int, _frame: Any):
    sys.exit(128 + signo)

def build_docker_image(args):
    cmd = [
        "sudo",
        "-n",
        "/usr/bin/docker",
        "build",
        "-t",
        image_tag,
        "-f",
        args.image,
        "."
    ]
    
    if args.dry:
        cmd = [c if c else "''" for c in cmd]
        print(" ".join(cmd))
        return
    
    print("Building Docker image:")
    print(" ".join(cmd))
    
    run_proc(cmd)

if __name__ == "__main__":
    proc = None
    atexit.register(kill_proc)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--dry", action="store_true", help="Dry run - print commands without executing")
    parser.add_argument("--build", action="store_true", help="Build the Docker image before running")
        
    # Accelerate arguments
    parser.add_argument("--main_process_ip", required=True)
    parser.add_argument("--num_processes", type=int, required=True)
    parser.add_argument("--num_machines", type=int, required=True)
    parser.add_argument("--image", required=True)  # Path to Dockerfile
    parser.add_argument("--wandb_api_key", required=True)

    args = parser.parse_args()
    machine_rank = os.getenv("SLURM_NODEID", "0")

    image_tag = "climax-training:latest"

    if args.build:
        build_docker_image(args)
    
    main(args)