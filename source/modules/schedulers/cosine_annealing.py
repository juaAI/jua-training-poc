import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Learning rate scheduler that increases linearly from eta_min to LR
    configured in the optimiser during the warmup period.
    After the warmup period the LR decreases according to the cosine annealing
    schedule.
    CosineAnnealing logic copied from torch.optim.lr_scheduler.CosineAnnealingLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        warmup_steps: int,
        eta_min: float = 0.00002,
        *,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
        max_steps : int
            The number of steps to train for
        warmup_steps : int
            The number of steps to warmup for
        eta_min : float
            The minimum learning rate to use.
        """
        self.warmup_steps = min(warmup_steps, max_steps)
        self.cosine_steps = max_steps - self.warmup_steps
        self.eta_min = eta_min

        # This calls self.step() so all setup needs to be done first
        super().__init__(
            optimizer=optimizer,
            verbose=verbose,
        )

        if warmup_steps > 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = eta_min
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    @property
    def last_step(self) -> int:
        """
        Return the number of times `self.step()` was invoked.
        The super class updates `last_epoch` on every step, which can be confusing
        when working with a "step" interval.
        """
        return self.last_epoch

    def get_lr(self) -> list[float]:  # type: ignore[override]
        # FIXME: mypy thinks get_lr should return a single float but no idea why
        if self.warmup_steps and self.last_step <= self.warmup_steps:
            # if in warmup increase warmup from min to max at a linear rate:
            return [
                (group["initial_lr"] / self.warmup_steps) * self.last_step
                for group in self.optimizer.param_groups
            ]
        # else reduce from current place at cosine towards min:
        cosine_step = self.last_step - self.warmup_steps
        # Skip computation at the height of the cosine
        if cosine_step == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            (1 + math.cos(math.pi * cosine_step / self.cosine_steps))
            / (1 + math.cos(math.pi * (cosine_step - 1) / self.cosine_steps))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
