import torch

from source.modules.metrics.weighting import (
    get_latitude_weights,
)

def calculate_weighted_loss(
    loss_multipliers: dict[str, float],
    output_var_count: int,
    loss_dict: dict[str, torch.Tensor],
):
    unmasked_var_count = output_var_count - len(loss_multipliers)
    weighted_loss_ratio = unmasked_var_count + sum(loss_multipliers.values())
    total_error = torch.stack([v for v in loss_dict.values()]).sum()
    loss_dict["loss"] = total_error / weighted_loss_ratio
    return loss_dict


def lat_weighted_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    loss_multipliers: dict[str, float],
    output_vars: dict[str, int],
    lat: torch.Tensor | None = None,
    *,
    loss_order: int = 2,  # MAE: loss_order=1, MSE: loss_order=2
) -> dict[str, torch.Tensor]:
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        loss_multipliers: a dictionary of loss multipliers for each variable
        vars: dict of variable name to index in y and pred
        lat: H

    Returns:
        dict of losses for each variable and the total loss
        example:
        {
            "var1": Tensor(0.1),
            "var2": Tensor(0.2),
            "loss": Tensor(0.15)
            ...
        }
    """

    error = torch.abs(pred - y) ** loss_order  # [N, C, H, W]

    w_lat = get_latitude_weights(lat, error) if lat is not None else 1.0

    loss_dict = {}
    for var_name, var_index in output_vars.items():
        loss_dict[var_name] = (error[:, var_index] * w_lat).mean()
        if var_name in loss_multipliers:
            loss_dict[var_name] *= loss_multipliers[var_name]

    return calculate_weighted_loss(loss_multipliers, len(output_vars), loss_dict)

