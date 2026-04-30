import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # Compute log probabilities in a numerically stable way
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    
    # Convert to probabilities
    probs = torch.exp(log_probs)
    
    # Entropy: -sum(p * log p)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return entropy