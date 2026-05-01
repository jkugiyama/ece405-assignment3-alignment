import torch

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    mask = mask.bool()
    return torch.sum(
        torch.where(mask, tensor, torch.zeros_like(tensor)),
        dim=dim
    ) / normalize_constant