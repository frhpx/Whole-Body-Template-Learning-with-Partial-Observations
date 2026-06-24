import torch


def masked_sdf_loss(pred, gt, mask):
    diff = (pred - gt) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1)

def deformation_loss(delta_x, x):
    u, v, w = delta_x[:, 0], delta_x[:, 1], delta_x[:, 2]
    ones = torch.ones_like(u)
    grad_u = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    grad_v = torch.autograd.grad(v, x, ones, create_graph=True)[0]
    grad_w = torch.autograd.grad(w, x, ones, create_graph=True)[0]
    return torch.stack([grad_u, grad_v, grad_w], dim=1).norm(dim=-1).mean()

def latent_loss(z, sigma=0.01):
    return torch.mean(z ** 2) / (sigma ** 2)
