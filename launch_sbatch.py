#!/usr/bin/env python3.11

import argparse
from pathlib import Path
import subprocess
from datetime import datetime
from textwrap import dedent
import sys
import shutil
import yaml

SCRIPT_ROOT = Path(__file__).parent.resolve()
# get current user home directory
HOME = str(Path.home())
# create training root at "home/$USER/runs"
TRAINING_ROOT = Path(f"{HOME}/runs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument(
        "--nodes",
        "-N",
        default=1,
        type=int,
        help="Number of nodes to use for training (default: %(default)s)",
    )
    parser.add_argument(
        "--partition",
        "-p",
        default="batch",
        help="Slurm partition to use (default: %(default)s)",
    )
    parser.add_argument(
        "--gpus_per_node",
        "-G",
        type=int,
        default=8,
        help="Number of GPUs to use per node (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="Path to the config file to use for this training run.",
    )
    parser.add_argument(
        "--accelerate_config",
        "-ac",
        help="Path to the (optional) Accelerate config file to use for distributed training",
        type=Path,
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Run without submitting job (default: %(default)s)",
    )
    parser.add_argument(
        "--run_id",
        required=True,
        type=str,
        help="The run ID to use for this training run.",
    )
    parser.add_argument(
        "--just-testing",
        "-jt",
        action="store_true",
        help="Set a 15-minute time limit for test runs",
    )
    

    args = parser.parse_args()

    total_num_gpus = args.nodes * args.gpus_per_node
    date = datetime.utcnow().strftime("%Y-%m-%d")
    result = subprocess.run(["petname"], capture_output=True, text=True, check=True)
    pet_name = result.stdout.strip()
    run_name = f"{date}-{pet_name}"
    run_path = TRAINING_ROOT / run_name

    run_args = [
        f"--image={SCRIPT_ROOT}/Dockerfile",
        f"--num_machines={args.nodes}",
        f"--num_processes={total_num_gpus}",
        "--main_process_ip=$(hostname -I | cut -d' ' -f1)",
        "--wandb_api_key=$WANDB_API_KEY",
        f"--run_id={run_name}",
        "--attempt=$ATTEMPT",
    ]

    srun_command_parts = [
        "srun",
        "--label",
        "--kill-on-bad-exit",
        f"--chdir={run_path}",
        f"--output={run_path}/attempt-$ATTEMPT.log",
        "./docker_runner.py",
        *run_args,
    ]
    srun_command = " ".join(srun_command_parts)

    # Add this before the sbatch submission
    build_script = dedent(
        """
        #!/bin/bash
        set -e  # Exit on any error
        
        echo "Current directory: $(pwd)"
        echo "Directory contents:"
        ls -la
        
        echo "Building Docker image..."
        DOCKER_BUILDKIT=1 sudo -n docker build -t climax-training:latest . || {
            echo "Docker build failed"
            exit 1
        }
        
        echo "Verifying image..."
        sudo -n docker images | grep climax-training || {
            echo "Image not found after build"
            exit 1
        }
        
        echo "Build successful"
        """
    ).strip()
    
    # Prepare the run's working directory
    run_path.mkdir(parents=True)

    # Copy all necessary files for Docker build
    worker_file = run_path / "docker_runner.py"
    shutil.copy2(SCRIPT_ROOT / "docker_runner.py", worker_file)
    shutil.copy2(SCRIPT_ROOT / "Dockerfile", run_path / "Dockerfile")
    
    # Copy source code and requirements
    if (SCRIPT_ROOT / "source").exists():
        shutil.copytree(SCRIPT_ROOT / "source", run_path / "source", dirs_exist_ok=True)
    if (SCRIPT_ROOT / "requirements.txt").exists():
        shutil.copy2(SCRIPT_ROOT / "requirements.txt", run_path / "requirements.txt")
    if (SCRIPT_ROOT / "pyproject.toml").exists():
        shutil.copy2(SCRIPT_ROOT / "pyproject.toml", run_path / "pyproject.toml")
    if (SCRIPT_ROOT / "poetry.lock").exists():
        shutil.copy2(SCRIPT_ROOT / "poetry.lock", run_path / "poetry.lock")
    
    # Write the build script
    build_file = run_path / "build.sh"
    build_file.write_text(build_script)
    build_file.chmod(0o755)
    
    # Add this to your SLURM script before the main command
    herefile = dedent(
        f"""
        #!/bin/bash
        #SBATCH -N {args.nodes}
        #SBATCH -J {run_name}
        #SBATCH --partition {args.partition}
        #SBATCH --chdir "{run_path}"
        #SBATCH -o "batch.log"
        {('#SBATCH --time "00:15:00"' if args.just_testing else '')}

        # Build the image on each node
        srun --ntasks={args.nodes} ./build.sh
        
        # Check if build was successful
        if [ $? -ne 0 ]; then
            echo "Docker build failed"
            exit 1
        fi

        # Verify image exists
        if ! docker images | grep climax-training > /dev/null; then
            echo "Image climax-training not found after build"
            exit 1
        fi

        touch ${{SLURM_JOB_ID}}.job-id

        for ATTEMPT in {{1..{1 if args.just_testing else 20}}}
        do
            echo "Starting attempt $ATTEMPT"
            {srun_command} && s=0 && break
            s=$?
            echo "Attempt $ATTEMPT failed with exit code $s"
            sleep 15
        done

        if [ "$s" -ne 0 ]
        then
            echo "FAILED WITH EXIT CODE $s"
            exit $s
        fi

        echo "SUCCESS"
        """
    ).strip()

    cmd = ["sbatch", "batch.sh"]

    if args.dry:
        print("Dry run. Not submitting job.")
        print(" ".join(cmd))
        print("With input:")
        print(herefile)
        sys.exit(0)

    worker_file.chmod(0o755)

    # Write the batch script
    batch_file = run_path / "batch.sh"
    batch_file.write_text(herefile)
    batch_file.chmod(0o555)

    target_config_file = run_path / "config.yaml"
    shutil.copy(args.config, target_config_file)
    target_config_file.chmod(0o444)

    if args.accelerate_config:
        target_accelerate_config_file = run_path / "accelerate_config.yaml"
        shutil.copy(args.accelerate_config, target_accelerate_config_file)
        target_accelerate_config_file.chmod(0o444)

    subprocess.run(cmd, cwd=run_path, check=True)

    print()
    print("Some useful commands:")
    print(f"cd {run_path}")
    print(f"tail -f {run_path}/batch.log")
    print(f"tail -f {run_path}/attempt-1.log")