import torch
import torch.nn.functional as F

def _sinkhorn_1d_batch(mu, nu, eps=0.1, n_iter=10):
    B, N = mu.shape
    _, M = nu.shape

    eps = max(eps, 1e-6)
    
    i = torch.arange(N, device=mu.device, dtype=torch.float32).view(1, -1, 1)
    j = torch.arange(M, device=mu.device, dtype=torch.float32).view(1, 1, -1)
    C = (i - j).abs()
    K = torch.exp(-C / eps).expand(B, -1, -1)

    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    for _ in range(n_iter):
        u_new = mu / (torch.clamp((K @ v.unsqueeze(-1)).squeeze(-1), min=1e-8))
        v_new = nu / (torch.clamp((K.transpose(1, 2) @ u_new.unsqueeze(-1)).squeeze(-1), min=1e-8))
        
        u = torch.clamp(u_new, min=1e-8, max=1e8)
        v = torch.clamp(v_new, min=1e-8, max=1e8)

    transport = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    dist = torch.sum(transport * C, dim=(1, 2))
    return torch.clamp(dist, min=0.0, max=1e6)

def wasserstein_features_fast(f1, f2, eps=0.1, n_iter=10):
    """
    f1, f2: Tensor (B, C, H, W)
    -> Global Avg Pool -> (B, C) -> Normalize -> Sinkhorn
    """
    B, C, H, W = f1.shape

    f1_pooled = F.adaptive_avg_pool2d(f1, (1, 1)).view(B, C)
    f2_pooled = F.adaptive_avg_pool2d(f2, (1, 1)).view(B, C)

    mu = F.relu(f1_pooled) + 1e-6
    nu = F.relu(f2_pooled) + 1e-6
    mu = mu / mu.sum(dim=1, keepdim=True)
    nu = nu / nu.sum(dim=1, keepdim=True)

    return _sinkhorn_1d_batch(mu, nu, eps=eps, n_iter=n_iter)  # (B,)