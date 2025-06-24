import torch

def kld_loss(log_var, mu):
    # first, compute sum loss on feat dim (=(1,2,...)), then compute mean loss on BATCH dim
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

