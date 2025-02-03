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

Add the `poetry-exec-plugin` plugin:
```bash
poetry self add poetry-exec-plugin
```

To run the training script, use the following command:
```bash
poetry run accelerate launch source/trainer.py --config configs/tiny_model.yaml
```

## Docker Testing

Build the Docker image:
```bash
docker build -t climax-training .
```

Run training in Docker:
```bash
docker run --shm-size=8g --gpus all -v <path_to_data>:data climax-training --config configs/tiny_model.yaml
docker run --shm-size=8g --gpus all -v /mnt/jua-shared-1:/mnt/data/ climax-training --config configs/tiny_model.yaml
```

## Running in Slurm

To run the training script in Slurm, use the following command:
```bash
sbatch scripts/run_training.sh
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