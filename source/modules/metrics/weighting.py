import torch

def get_latitude_weights(lat: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    w_lat = torch.cos(torch.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = (
        w_lat.unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)
    )  # (1, H, 1)

    return w_lat