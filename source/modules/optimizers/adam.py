from typing import Any
import torch

from source.config import OptimizerType

def get_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_type: OptimizerType,
    learning_rate: float,
    betas: tuple[float, float],
    epsilon: float,
    weight_decay: float,
    fused: bool,
) -> torch.optim.Optimizer:
    """
    Return a configured instance of the AdamW optimizer.

    Note: weight_decay is an L2 regularization applied to weights, usually ignored for biases
    and only usually applied to matrices.
    pos_embed and var_embed layers are bias vectors and thus not regularized.
    """
    no_decay = []
    decay = []
    for name, m in model.named_parameters():
        if "var_embed" in name or "pos_embed" in name:
            no_decay.append(m)
        else:
            decay.append(m)

    optimizer: Any
    if optimizer_type == OptimizerType.ADAM:
        optimizer = torch.optim.Adam
    elif optimizer_type == OptimizerType.ADAMW:
        optimizer = torch.optim.AdamW
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer(
        [
            {
                "params": decay,
                "weight_decay": weight_decay,
                "fused": fused,
            },
            {
                "params": no_decay,
                "weight_decay": 0,
                "fused": fused,
            },
        ],
        eps=epsilon,
        lr=learning_rate,
        betas=betas,
    )
