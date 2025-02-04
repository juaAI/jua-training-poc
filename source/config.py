import os
from collections import defaultdict
from datetime import datetime
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field, model_validator


class Variable(BaseModel):
    name: str
    input: bool
    output: bool

class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"

class TrainingConfig(BaseModel):
    # TRAINER
    dry: bool = False  # run without submitting job
    num_epochs: int = Field(default=2, ge=1)
    num_warmup_epochs: int = 5  # not used in ept2
    warmup_steps: int = 2000  # used in ept2
    enable_compilation: bool = False
    skip_training: bool = False
    log_to_wandb: bool = False
    log_validation_metrics: bool = True
    optimizer_type: OptimizerType
    learning_rate: float
    min_learning_rate: float
    optimizer_betas: tuple[float, float]
    optimizer_epsilon: float
    optimizer_weight_decay: float
    optimizer_fused: bool
    gradient_accumulation_steps: int = 1
    use_lat_weights: bool = False
    report_sample_interval: int = 500

    # MODEL
    decoder_depth: int
    img_size: tuple[int, int]
    patch_size: int
    embed_dim: int
    depth: int
    decoder_depth: int
    num_heads: int
    mlp_ratio: float
    lead_time_set: list[int]
    drop_path: float
    drop_rate: float

    # DATALOADER
    batch_size: int = 1
    num_workers: int = 8  # used by the standard PyTorch DataLoader
    shuffle: bool = True  # used by the standard PyTorch DataLoader
    pin_memory: bool = True  # used by the standard PyTorch DataLoader
    prefetch_factor: int = 2  # used by the standard PyTorch DataLoader
    persistent_workers: bool = True  # used by the standard PyTorch DataLoader
    multiprocessing_context: str = "spawn"  # used by the standard PyTorch DataLoader
    datasets_weight: dict[str, float] | None = None

    # DATA
    stats_path: str = (
        "/mnt/jua-shared-1/jua-silver-layer/all_variables_stats_together.json"
    )
    zarr_path: str = "/mnt/jua-shared-1/jua-silver-layer/erax-norm.zarr"
    start_date: str = "1979-01-01"
    end_date: str = "2019-12-31"
    levels: list[int] = []
    input_variable_names: list[str] = []
    output_variable_names: list[str] = []

    # VARIABLES
    variables: list[Variable] = []

    @cached_property
    def lead_time_set_max(self) -> int:
        return max(self.lead_time_set)

    @cached_property
    def output_variable_map(self) -> dict[str, int]:
        return {variable: idx for idx, variable in enumerate(self.output_names)}

    @classmethod
    def from_yaml(cls: type[Self], file_path: str) -> Self:
        # Convert string path to Path object
        file_path = Path(file_path)
        with file_path.open() as f:
            return cls(**yaml.safe_load(f))