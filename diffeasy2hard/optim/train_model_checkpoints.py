

import argparse
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from lineardiffusion.models.diffusion import Diffusion
from lineardiffusion.models.u_net import UNet
from lineardiffusion.scripts.CelebA.utils import get_subset, steps2epochs
from lineardiffusion.stats.set_seed import set_seed

def save_checkpoint(model, optimizer, epoch, loss, path, generated, indices):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'generated': generated, 
        'indices': indices

    }, path)

def reduce_train_set_to_N(data_tensor, N): 
    
    num_samples = data_tensor.shape[0]
    if num_samples >= N: 
        indices = torch.randint(0,num_samples,(N,))
    data_tensor = data_tensor[indices]
    return data_tensor, indices



def run_experiment(load_data, model_save_folder):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--c", type=float, default=0.0,
                         help="regularization strength")
    parser.add_argument("--steps", type=int, default=120000,
                         help="training steps")
    parser.add_argument("--N", type=int, default=120000,
                         help="number of samples")
    parser.add_argument("--weight_decay", type=float,
                         default=0.0, help="weight decay")
    parser.add_argument("--num_checkpoints", type=int,
                         default=100, help="number of checkpoints")
    parser.add_argument("--adamW", help="whether to use adamW",
                         action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=100, type=int)
    
    # cosine learning rate scheduling
    parser.add_argument("--cosine_lr", help="use cosine annealing learning rate scheduling",
                         action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    # Learning Parameters
    c = args.c
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_timesteps = 1000
    base_learning_rate = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay

    num_steps = args.steps
    end = torch.log10(torch.tensor([1.0 * args.steps])).item()
    steps_to_print = list(torch.logspace(0, end, steps=args.num_checkpoints))
    print("checkpoints: ", steps_to_print)

    # Load data
    data_tensor = load_data(args)
    data_tensor, indices = reduce_train_set_to_N(data_tensor, args.N)
    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dim = data_tensor.shape[-1] ** 2
    dim_1d = data_tensor.shape[-1]

    diffmodel = Diffusion(None, dim=dim, device=device)
    model = UNet(T=diffmodel.T, dim=dim, betas=diffmodel.beta, in_channels=1).to(device)
    diffmodel.model = model

    # Optimizer
    if args.adamW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay)

    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        print("Using cosine learning rate schedule.")
    else:
        scheduler = None

    num_epochs = steps2epochs(num_steps, batch_size, data_tensor.shape[0])
    global_step = 0
    losses = []

    # Training loop
    for epoch in range(num_epochs):
        print(f"epoch {epoch} out of {num_epochs}")
        model.train()

        for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)

            noisy, noise = diffmodel.add_forward_noise(batch, timesteps)
            noise_pred = model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)

            # Regularization
            if c > 0:
                sqrt_alpha_bar_t = diffmodel.sqrt_alpha_bar[timesteps].reshape(-1, 1, 1, 1)
                noise_pred_scaled = sqrt_alpha_bar_t * noise_pred
                loss += c * torch.mean(noise_pred_scaled ** 2)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step() 

            losses.append(loss.detach().item())
            global_step += 1
            print(global_step)

            # Save checkpoint
            if len(steps_to_print) > 0 and (global_step >= steps_to_print[0].item() or global_step == 0):
                folder = model_save_folder(args)

                name = f"{global_step}_{base_learning_rate:.4f}_{batch_size}_{weight_decay:.4f}_{c:.4f}_oracle.pt"
                path = folder + name

                model.eval()
                generated = diffmodel.sample(10, shape=(10, 1, dim_1d, dim_1d)).cpu()
                save_checkpoint(model, optimizer, epoch, loss, path, generated, indices)
                model.train()
                steps_to_print.pop(0)
    print("done")
    return args