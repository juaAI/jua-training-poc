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
poetry run accelerate launch source/trainer.py --config configs/small_model.yaml --wandb_group test
```
