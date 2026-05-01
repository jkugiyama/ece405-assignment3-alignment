import torch

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:

    mask = mask.to(dtype=tensor.dtype)

    masked = tensor * mask
    summed = masked.sum(dim=dim)
    counts = mask.sum(dim=dim)

    return summed / counts