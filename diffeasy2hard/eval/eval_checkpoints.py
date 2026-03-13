import os
import glob
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data import TensorDataset
import numpy as np

from diffeasy2hard.stats.Gaussian import Gaussian
from diffeasy2hard.models.u_net import UNet
from diffeasy2hard.models.diffusion import Diffusion
from diffeasy2hard.utils.general import copy_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_BATCHES = 100
BATCH_SIZE = 50
WORK_DIR = os.environ.get("WORK")

def load_checkpoint(path, model, optimizer=None, map_location=None):
    """
    Load a checkpoint for a model and optionally an optimizer.
    
    Args:
        path (str): Path to the saved checkpoint.
        model (torch.nn.Module): Model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer instance to load state into.
        map_location (str or torch.device, optional): Device to map the checkpoint to.

    Returns:
        dict: Loaded checkpoint containing epoch, loss, and generated data.
    """
    checkpoint = torch.load(path, map_location=map_location)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded successfully from {path}")
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'generated': checkpoint['generated'], 
        'indices': checkpoint['indices']
    }

def get_steps_paths(folder, interval, weight_decay, c):
    all_model_paths = glob.glob(folder+"*.pt")
    print(all_model_paths)
    steps = []
    model_paths = []
    for model_path in all_model_paths:
        file_name = model_path[len(folder):]
        step, lr, bs, weight_decay_, c_, _ = \
              file_name.split("_")
        print(float(weight_decay))
        if float(weight_decay_) == weight_decay \
            and float(c_) == c:
            steps += [int(step)]
            model_paths += [model_path]

    
    sorting = np.argsort(steps)
    model_paths = [model_paths[i] for 
                i in sorting]
    steps = [steps[i] for i in sorting]
    steps = steps[::interval]
    model_paths = model_paths[::interval]
    return steps, model_paths



def eval_checkpoints(data_folder, model_folder, save_folder, interval, weight_decay, c, t="all", dim1d=32): 
    """Evaluate loss of clone datasets on checkpoints.

    Args:
        data_folder (string): folder containing training and test data
        model_folder (string): checkpoint location
        save_folder (string): where to save results
        interval (int): use to subsample the checkpoints, if larger 1 evaluate every interval-th checkpoint
        weight_decay (float): weight decay of checkpoints
        c (float): regularization of checkpoints
        t (str, optional): Diffusion time at which to evaluate loss. Defaults to "all".
    """
    # look for trained models
    steps, model_paths = get_steps_paths(model_folder, interval, weight_decay, c)

    # load train and test set
    test_set = torch.load(data_folder+"/test.pt")[0]
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
    
    
    # load stats
    d = torch.load(data_folder+"/test_stats.pt")
    mean = d["mean"].to(device)
    cov = d["cov"].to(device)
    dim = len(mean)

    # Generate clone test sets
    mean_only_model = Gaussian(dim=dim,
                               device=torch.device("cpu"),
                               mean=mean)
    data_mean_only_model = mean_only_model.sample(NUM_BATCHES*BATCH_SIZE).reshape(NUM_BATCHES*BATCH_SIZE,1,dim1d,dim1d)
    dataset_mean_only = TensorDataset(data_mean_only_model)
    mean_only_loader = DataLoader(dataset_mean_only,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    cov_model = Gaussian(dim=dim, device=torch.device("cpu"),
                         mean=mean, covariance=cov)
    data_cov_model = cov_model.sample(NUM_BATCHES*BATCH_SIZE).reshape(NUM_BATCHES*BATCH_SIZE,1,dim1d,dim1d)
    dataset_cov = TensorDataset(data_cov_model)
    cov_loader = DataLoader(dataset_cov, batch_size=BATCH_SIZE,shuffle=True)
    
    # data containers
    test_losses = np.zeros(len(steps))
    train_losses = np.zeros(len(steps))
    test_losses_std = np.zeros(len(steps))
    train_losses_std = np.zeros(len(steps))
    mean_losses = np.zeros(len(steps))
    cov_losses = np.zeros(len(steps))
    mean_losses_std = np.zeros(len(steps))
    cov_losses_std = np.zeros(len(steps))

    # iterate over trained models
    for model_path, step, s in zip(model_paths, steps, range(len(steps))): 
        num_timesteps = 1000
        diffmodel = Diffusion(None, dim=dim, device=device)
        model = UNet(T=diffmodel.T, dim=dim, betas=diffmodel.beta, in_channels=1,).to(device)
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), )
        res_dict = load_checkpoint(model_path,model,optimizer)
        train_set = torch.load(data_folder+"/train.pt")[0][res_dict['indices']]
        if train_set.shape[0] < NUM_BATCHES*BATCH_SIZE: 
            train_set = copy_data(NUM_BATCHES*BATCH_SIZE, train_set.shape[0], train_set)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        train_loss = np.zeros(NUM_BATCHES)
        test_loss = np.zeros(NUM_BATCHES)
        mean_loss = np.zeros(NUM_BATCHES)
        cov_loss = np.zeros(NUM_BATCHES)
        
        for train_data, test_data, n in zip(train_loader,test_loader,range(NUM_BATCHES)):
            for batch, container in zip([train_data, test_data],
                                        [train_loss, test_loss]):
                batch = batch.to(device)
                noise = torch.randn(batch.shape).to(device)
                if t=="all":
                    timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)
                                            ).long().to(device)
                else: 
                    timesteps = (torch.ones(batch.shape[0])*t).long().to(device)
                noisy, noise = diffmodel.add_forward_noise(batch, timesteps)
                noise_pred = model(noisy, timesteps)
                loss = F.mse_loss(noise_pred, noise)
                container[n] += loss.detach().cpu().numpy() 
        for mean_data, cov_data, n in zip( mean_only_loader,cov_loader,range(NUM_BATCHES)):
            for batch, container in zip([mean_data, cov_data],
                                        [mean_loss, cov_loss]):
                batch = batch[0].to(device)
                if t=="all":
                    timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)
                                            ).long().to(device)
                else: 
                    timesteps = (torch.ones(batch.shape[0])*t).long().to(device)
                noisy, noise = diffmodel.add_forward_noise(batch, timesteps)
                noise_pred = model(noisy, timesteps)
                loss = F.mse_loss(noise_pred, noise)
                container[n] += loss.detach().cpu().numpy()

        test_losses[s] = np.mean(test_loss)
        train_losses[s] += np.mean(train_loss)
        test_losses_std[s] = np.std(test_loss)
        train_losses_std[s] = np.std(train_loss)
        cov_losses[s] = np.mean(cov_loss)
        mean_losses[s] = np.mean(mean_loss)
        cov_losses_std[s] = np.std(cov_loss)
        mean_losses_std[s] = np.std(mean_loss)
        all_losses = np.vstack([test_losses,test_losses_std,train_losses,train_losses_std])
    
        os.makedirs(save_folder, exist_ok=True)
        np.save(save_folder+f"weight_decay{weight_decay}_c{c}_trainandTest_{t}.npy", all_losses)
        all_losses = np.vstack([mean_losses,mean_losses_std,cov_losses,cov_losses_std])
        np.save(save_folder+f"weight_decay{weight_decay}_c{c}_meanandcov_{t}.npy", all_losses)
        np.save(save_folder+f"weight_decay{weight_decay}_c{c}_steps_{t}.npy", np.array(steps))


