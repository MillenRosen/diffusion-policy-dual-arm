import torch


class DiffusionScheduler:
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_steps = num_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        shape = [x0.shape[0]] + [1] * (x0.dim() - 1)
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[t]).view(shape)
        sqrt_one_minus = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(shape)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    @torch.no_grad()
    def sample(self, model, obs_cond, action_shape):
        batch_size = obs_cond.shape[0]
        x = torch.randn((batch_size,) + tuple(action_shape), device=obs_cond.device)
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((batch_size,), t, device=obs_cond.device, dtype=torch.long)
            pred_noise = model(x, t_tensor, obs_cond)
            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            beta = self.betas[t]
            mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
            if t > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x

