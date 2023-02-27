import torch
from params import params

class Diffusion:
    def __init__(self):
        self.beta = self.compute_beta()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def compute_beta(self):
        return torch.linspace(params.beta_start, params.beta_end, params.noise_steps)

    def apply_noise(awld, x, t):
        sqrt_alpha_hat = torch.sqrt(awld.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - awld.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, nsamples):
        return torch.randint(0, params.noise_steps, (nsamples,))

    @torch.no_grad()
    def sample(self, model, nsamples):
        model.eval()

        x = torch.randn(nsamples, 3, params.img_size, params.img_size).to(model.device)
        for i in reversed(range(params.noise_steps)):
            t = torch.full((nsamples,), i, dtype=torch.long).to(model.device)
            noise_hat = model(x, t)

            alpha = self.alpha[t][:, None, None, None].to(model.device)
            alpha_hat = self.alpha_hat[t][:, None, None, None].to(model.device)
            beta = self.beta[t][:, None, None, None].to(model.device)

            if i == 0:
                noise = torch.zeros_like(noise_hat).to(model.device)
            else:
                noise = torch.randn_like(noise_hat).to(model.device)

            print(x.device, noise_hat.device, noise.device, alpha.device, alpha_hat.device, beta.device)

            x = 1/torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise_hat) + noise

        x = torch.clamp(x, -1, 1)
        model.train()
        return x



