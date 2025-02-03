import argparse
import logging
import os

import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast
import wandb

import numpy as np
import torch
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    FullyShardedDataParallelPlugin,
    InitProcessGroupKwargs,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from source.config import TrainingConfig
from source.model import ClimaX
from source.data import ToyEra5Dataset

from source.modules.metrics.mse import lat_weighted_loss
from source.modules.optimizers.adam import get_optimizer
from source.modules.schedulers.cosine_annealing import CosineAnnealingWithWarmup

def get_accelerator(gradient_accumulation_steps: int) -> Accelerator:
    distributed_data_parallel_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True
    )
    init_process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=3600),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[distributed_data_parallel_kwargs, init_process_group_kwargs],
        step_scheduler_with_optimizer=False,
    )

    if accelerator.distributed_type == "FSDP":
        gpus_per_node = 8
        if (
            accelerator.num_processes // gpus_per_node > 1
            and accelerator.state.fsdp_plugin.state_dict_type
            != StateDictType.FULL_STATE_DICT
        ):
            raise ValueError(
                "FSDP on multi node is only supported with FULL_STATE_DICT state_dict_type"
            )
        if (
            accelerator.state.fsdp_plugin.state_dict_type
            == StateDictType.FULL_STATE_DICT
        ):
            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=False
                ),
                optim_state_dict_config=FullOptimStateDictConfig(
                    offload_to_cpu=True, rank0_only=False
                ),
            )
        accelerator.state.fsdp_plugin = fsdp_plugin
    return accelerator


def get_scheduler(
    optimizer: Optimizer,
    num_epochs: int,
    steps_per_epoch: int,
    min_learning_rate: float,
    num_warmup_epochs: int,
) -> LRScheduler:
    max_steps = num_epochs * steps_per_epoch
    warmup_steps = num_warmup_epochs * steps_per_epoch

    return CosineAnnealingWithWarmup(
        optimizer,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        eta_min=min_learning_rate,
    )

class Trainer:
    def __init__(  # noqa: PLR0915
        self,
        *,
        accelerator: Accelerator,
        config: TrainingConfig,
        run_id: str,
    ):
        self.accelerator = accelerator

        self.config = config
        self.run_id = run_id
        # self.output_vars = self.config.output_variable_map
        self.lats = (
            torch.from_numpy(
                np.linspace(-90, 90, self.config.usable_latitudes, endpoint=True)
            )
            if self.config.use_lat_weights
            else None
        )

        dataset = ToyEra5Dataset(
            zarr_path=config.zarr_path,
            stats_path=config.stats_path,
            start_date=config.start_date,
            end_date=config.end_date,
            lead_time_set=config.lead_time_set,
            input_variable_names=config.input_variable_names,
            output_variable_names=config.output_variable_names,
            levels=config.levels,
        )

        model = ClimaX(
            input_vars=dataset.get_input_variable_names(),
            output_vars=dataset.get_output_variable_names(),
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            decoder_depth=config.decoder_depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            drop_path=config.drop_path,
            drop_rate=config.drop_rate,
        )

        if accelerator.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            logging.info("Total parameters: %.3f B", total_params / 1e9)

        training_dataloader = TorchDataLoader(
            dataset=dataset, 
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            multiprocessing_context=config.multiprocessing_context,
        )

        optimizer = get_optimizer(
            model,
            optimizer_type=config.optimizer_type,
            learning_rate=config.learning_rate,
            betas=config.optimizer_betas,
            epsilon=config.optimizer_epsilon,
            weight_decay=config.optimizer_weight_decay,
            fused=config.optimizer_fused,
        )
        self.effective_batch_size = config.batch_size * self.accelerator.num_processes
        self.steps_per_epoch = (
            len(training_dataloader) // self.accelerator.num_processes
        )
        scheduler = get_scheduler(
            optimizer,
            config.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
            min_learning_rate=config.min_learning_rate,
            num_warmup_epochs=config.num_warmup_epochs,
        )
        (
            optimizer,
            scheduler,
            model,
            training_dataloader,
        ) = self.accelerator.prepare(
            optimizer,
            scheduler,
            model,
            training_dataloader,
        )

        self.model = model
        self.optimizer = cast(Optimizer, optimizer)
        self.scheduler = cast(LRScheduler, scheduler)
        self.training_dataloader = training_dataloader
        self.dataset = dataset

    @property
    def should_log_to_wandb(self):
        return self.accelerator.is_main_process and self.config.log_to_wandb

    def train(self):
        # Enable gradient descent
        self.model.train()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.train_current_epoch()
            self.accelerator.wait_for_everyone()

            

    def train_current_epoch(self):
        progress_bar = None
        self.last_reported = 0

        epoch_starting_step = self.epoch * self.steps_per_epoch

        for step, batch in enumerate(
            self.training_dataloader, start=epoch_starting_step
        ):
            if self.accelerator.is_local_main_process and progress_bar is None:
                progress_bar = tqdm(
                    total=len(self.training_dataloader),
                    smoothing=0.1,
                    unit="timestep",
                    file=sys.stdout,
                )
            if not self.config.skip_training:
                with self.accelerator.accumulate(self.model):
                    self.execute_training_step(step=step, batch=batch)

            if progress_bar is not None:
                progress_bar.update(self.effective_batch_size)

    def execute_training_step(
        self,
        *,
        step: int,
        batch: dict[str, torch.Tensor],
    ):
        # Execute the forward and backward pass through the model
        prediction_data: torch.Tensor = self.model(
            x=batch["input"],
            lead_times=batch["lead_time"]
        )

        all_losses = lat_weighted_loss(
            pred=prediction_data,
            y=batch["output"],
            output_vars=self.dataset.get_output_variable_map(),
            loss_multipliers={},
            lat=self.lats,
        )

        avg_loss = all_losses["loss"]

        self.accelerator.backward(avg_loss)
        # clip the gradients to avoid exploding gradients.
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        stats: dict[str, Any] = {}
        sample = step * self.effective_batch_size
        report = sample // self.config.report_sample_interval

        if self.should_log_to_wandb:
            # denormalize lead times
            stats.update(
                {f"training/loss/{k}": v.item() for k, v in all_losses.items()}
            )
            stats["training/learning_rate"] = self.scheduler.get_last_lr()[0]

            stats["training/step"] = step
            stats["training/sample"] = sample

            report = sample // self.config.report_sample_interval
            if report > self.last_reported:
                self.last_reported = report
                stats.update(
                    self.generate_images(
                        output_variables=self.config.output_variable_names,
                        target_data=batch["output"],
                        prediction_data=prediction_data,
                    )
                )
            wandb.log(stats)

    def generate_images(
        self,
        output_variables: list[str],
        target_data: torch.Tensor,
        prediction_data: torch.Tensor,
    ) -> dict[str, wandb.Image]:
        images = {}
        for idx, var_name in enumerate(output_variables):
            images[f"training/target/{var_name}"] = wandb.Image(
                target_data[0, idx].detach().to(device="cpu", dtype=torch.float32),
            )
            images[f"training/prediction/{var_name}"] = wandb.Image(
                prediction_data[0, idx].detach().to(device="cpu", dtype=torch.float32),
            )
        return images

def initialize_wandb_run(
    *, run_id: str, config: TrainingConfig, group: str, num_processes: int
) -> None:
    config_dict = config.model_dump(mode="json")
    config_dict["docker_image"] = os.getenv("DOCKER_IMAGE_TAG", "NO_DOCKER")
    config_dict["gpus"] = num_processes

    wandb_id = f"climax-{run_id}"

    wandb_notes = None

    wandb.init(
        project="prod",
        group=group,
        id=wandb_id,
        name=wandb_id,
        notes=wandb_notes,
        config=config_dict,
        reinit=True,
        resume="never",
    )
    wandb.define_metric("training/step")
    wandb.define_metric("training/sample")
    wandb.define_metric("training/epoch")
    wandb.define_metric("training/*", step_metric="training/sample")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_id",
        "-r",
        default=None,
        type=str,
        help="The run ID to use for this training run.",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="The path to the config file to use for this training run.",
    )
    parser.add_argument(
        "--wandb_group",
        required=True,
        type=str,
        help="The wandb group - use the experiment code like 'exp-123'",
    )

    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)
    accelerator = get_accelerator(config.gradient_accumulation_steps)

    run_id = args.run_id
    if not run_id:
        date = datetime.utcnow().strftime("%Y-%m-%d")
        run_id = f"{date}-manual-manatee"
        logging.info('No run_id specified, using "%s"', run_id)
        logging.warning(
            "Note that two manual runs on the same day will overwrite each other. "
            "To avoid this, please specify a unique run ID with --run_id",
        )

    if accelerator.is_main_process and config.log_to_wandb:
        initialize_wandb_run(
            run_id=run_id,
            config=config,
            group=args.wandb_group,
            num_processes=accelerator.num_processes,
        )

    trainer = Trainer(
        accelerator=accelerator,
        config=config,
        run_id=run_id,
    )

    trainer.train()

if __name__ == "__main__":
    main()
