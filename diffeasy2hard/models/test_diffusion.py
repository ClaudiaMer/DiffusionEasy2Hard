import torch
from torch.nn import Linear
from lineardiffusion.stats.Gaussian import Gaussian
from lineardiffusion.models import Diffusion
from networks import TinyModel,AffineLinear
from betas import linear_scaled
torch.set_default_dtype(torch.float64)
device = torch.device("cuda")
dim = 3
model = TinyModel(dim).to(device)

diff = Diffusion(model, dim, device)

from lineardiffusion.plotting import set_nice_params, dimcolors
import matplotlib.pyplot as plt
set_nice_params()

fig, axs = plt.subplots(2,2)
axs = axs.flatten()
cols = dimcolors(len(axs))
labels = [r'$\beta_t$',r'$\alpha_t$', r'$\bar{\alpha}_t$',r'$\sigma_t$'  ]
for ax, val, col, label in zip(axs, 
                        [diff.beta, diff.alpha, 
                        diff.alpha_bar, diff.sigma], 
                        cols, labels):
    ax.plot(diff.ts, val.to("cpu"), color=col, marker=".", label=label)
    ax.legend(frameon=False)
    ax.set_xlabel("t")
plt.tight_layout()
plt.savefig('schedule_linear.svg')

fig, axs = plt.subplots(2,2)
axs = axs.flatten()
cols = dimcolors(len(axs))
labels = [r'$\beta_t$',r'$\alpha_t$', r'$\bar{\alpha}_t$',r'$\sigma_t$'  ]

for T in [10,20,100,200]: 
    betas = linear_scaled(T=T)
    diff_t = Diffusion(model, dim, device,T=T, beta=betas)
    
    for ax, val, col, label in zip(axs, 
                            [diff_t.beta, diff_t.alpha, 
                            diff_t.alpha_bar, diff_t.sigma], 
                            cols, labels):
        ax.plot(diff_t.ts, val.to("cpu"), color=col, marker=".", label=label)
        ax.legend(frameon=False)
        ax.set_xlabel("t")
    plt.tight_layout()
    plt.savefig('schedule_linear_scaled_at_T=%i.svg'%T)

dim = 3
covariance = torch.tensor([[2, 1, 0.2],
                           [1, 2, -0.1],
                           [0.2, -0.1, 3]]).to(device)
mean = torch.tensor([1,4.0, -2.0]).to(device)

g = Gaussian(dim=dim, device=device, mean=mean, covariance=covariance)

N = 10000
data = g.sample(N)

for t in range(0, diff.T,10): 
    t_vec = (torch.ones(N)*t).long()
    noised_data, noise = diff.add_forward_noise(data, t_vec)
    mean_noised = torch.mean(noised_data, dim=0)
    cov_noised = torch.cov(noised_data.T)
    cov_pred = torch.eye(dim).to(device)*(1 - diff.alpha_bar[t]) \
               + diff.alpha_bar[t]* covariance
    assert torch.allclose(mean_noised, diff.sqrt_alpha_bar[t]*mean, atol=1e-1)
    assert torch.allclose(cov_noised, cov_pred,
                        atol=1e-1)
    
for t in range(0, diff.T,10): 
    t_vec = (torch.ones(N)*t).long()#
    noise = diff.sample_noise(data, t_vec)
    mean_noised = torch.mean(noise, dim=0)
    cov_noised = torch.cov(noise.T)
    cov_pred = torch.eye(dim).to(device)*diff.sigma[t]**2
    assert torch.allclose(mean_noised,
                          torch.zeros_like(mean_noised), atol=1e-1)
    assert torch.allclose(cov_noised, cov_pred,
                        rtol=1e-1, atol=2e-3)

assert diff.model(data,0).shape == data.shape
assert diff.mu_theta(data,t_vec).shape == data.shape

samples = diff.sample(10, style="Song")
assert samples.shape == (10,dim)
diff.loss(data)


from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=100)
diff.train(dataloader=dataloader, n_epochs=1, track_steps=1)
num_train_steps = N//100
assert len(diff.tracker.tracks["steps"]) == num_train_steps + 1 

# turn on regularization
diff.gamma += 0.1
diff.train(dataloader=dataloader, n_epochs=1, track_steps=1)

assert len(diff.tracker.tracks["loss"]) == 2*num_train_steps + 2
assert diff.tracker.tracks["steps"][-1] == num_train_steps


# test sampling : 

diff = Diffusion(None,dim, device, T=100)
model = AffineLinear(diff.dim,device=diff.device,
                      sqrt_alpha_bars=diff.sqrt_alpha_bar,
                      init_scaling=0.01)
diff.model = model
#diff.sigma = torch.zeros(diff.T).to(diff.device)
latent = torch.randn(10,dim).to(diff.device)
sample = diff.sample(10, latents=latent)
print(1.0/torch.sqrt(diff.alpha))
print("latent",latent/sample)

diff = Diffusion(None,dim, device, T=10)
model = AffineLinear(diff.dim,device=diff.device,
                      sqrt_alpha_bars=diff.sqrt_alpha_bar,
                      init_scaling=0.01, fullweighttensor=False)


from lineardiffusion.theory.weights import set_opt_params, set_opt_params_plus_corr
diff.model = model
diff = set_opt_params(diff, torch.eye(dim).to(diff.device), torch.zeros(dim).to(diff.device), fullweighttensor=False)
w1 = diff.model.get_weight(torch.tensor([0,0]).long().to(diff.device))
diff = set_opt_params_plus_corr(diff,torch.eye(dim).to(diff.device), torch.zeros(dim).to(diff.device), 3, fullweighttensor=False, )
w2 = diff.model.get_weight(torch.tensor([0,0]).long().to(diff.device))
print(w1-w2)
latent = torch.randn(10,dim).to(diff.device, )
sample = diff.sample(10, latents=latent, sample_batch=2)

print("all diffusion tests successful")