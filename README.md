# Description
This is a repository setup for evaluating training performance on various hardware.

### Structure

- `trainer.py`: The main training script.
- `model.py`: The model definition.
- `data.py`: The data loading and preprocessing.
- `utils.py`: Utility functions.
- `Dockerfile`: The Dockerfile for the training container.
- `pyproject.toml`: The poetry configuration.

## Local Development
Install [poetry](https://python-poetry.org/docs/):
```bash
curl -sSL https://install.python-poetry.org | python3 -
# Add poetry to your profile
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc 
```
If you encounter issues with SSL certificates, try to follow the steps described [here](https://stackoverflow.com/a/73270162).

Install initial dependencies:
```bash
poetry install
```

To run the training script, use the following command:
```bash
poetry run accelerate launch source/trainer.py --config configs/tiny_model_local.yaml
```

## Running in Slurm

Build the Docker image:
```bash
docker build -t climax-training:latest -f Dockerfile .
```

Test training with the docker image:
```bash
docker run \
  --init \
  --rm \
  -m 940g \
  --shm-size=500g \
  --gpus all \
  --network host \
  --privileged \
  --ulimit nofile=1000000 \
  -w /app \
  -e PYTHONPATH=/app \
  -v "/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu" \
  -v "/opt/hpcx:/opt/hpcx" \
  -v "/dev/infiniband:/dev/infiniband" \
  -v "/mnt/jua-shared-1:/mnt/data" \
  -v "$(pwd)/configs/tiny_model_local.yaml:/app/config.yaml:ro" \
  -v "/etc/crusoe/nccl_topo/h100-80gb-sxm-ib-cloud-hypervisor.xml:/etc/crusoe/nccl_topo/h100-80gb-sxm-ib-cloud-hypervisor.xml:ro" \
  -e LD_LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib:/opt/hpcx/ucc/lib/ucc:/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib/ucx:/opt/hpcx/ucx/lib:/opt/hpcx/sharp/lib:/opt/hpcx/hcoll/lib:/opt/hpcx/ompi/lib \
  -e NCCL_TOPO_FILE=/etc/crusoe/nccl_topo/h100-80gb-sxm-ib-cloud-hypervisor.xml \
  -e NCCL_SOCKET_NTHREADS=4 \
  -e NCCL_NSOCKS_PERTHREAD=8 \
  -e NCCL_IB_MERGE_VFS=0 \
  -e NCCL_IB_HCA=^mlx5_0:1 \
  climax-training:latest \
  poetry run accelerate launch \
    --multi_gpu \
    --mixed_precision fp16 \
    --num_processes 8 \
    source/trainer.py \
    --run_id test-run-1 \
    --config /app/configs/tiny_model.yaml
```

To run the training script in Slurm, use the following command:
```bash
poetry run python launch_sbatch.py --nodes 2 --gpus_per_node 8 --config configs/medium_model.yaml --run_id test-run
```

## Notes
1. You will need to update the data mount path in `launch_sbatch.py`.
2. You will need to add docker as a sudoer to each compute node. For example:
```bash
# Open sudoers file safely with visudo
sudo visudo

# Add the following line
%docker ALL=(ALL) NOPASSWD: /usr/bin/docker
```

## Expected results

### Tiny model - Total parameters: 0.009 B
1 GPU - 3650/140227 [02:20<1:24:26, 26.96timestep/s]
8 GPU's - 4200/17529 [00:38<01:57, 113.13timestep/s]

### Small model - Total parameters: 0.093 B
1 GPU - 1748/140227 [01:24<1:46:44, 21.62timestep/s]
8 GPU's - 4824/17529 [00:52<02:10, 97.09timestep/s

### Medium model - Total parameters: 1.067 B
1 GPU - 1776/140227 [02:00<5:56:49,  6.51timestep/s]
8 GPU's - 2240/17529 [00:54<05:51, 43.51timestep/s]
16 GPU's - 3408/8765 [00:41<01:02, 85.68timestep/s]