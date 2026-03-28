
import os

import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from diffeasy2hard.load_and_save import make_filename_from_args, save_experiment_results
from diffeasy2hard.plotting.plot_diffusion_results import plot_result, plot_samples
from diffeasy2hard.models.u_net import UNet
from diffeasy2hard.models.diffusion import Diffusion
from diffeasy2hard.stats.Gaussian import Gaussian

def get_data_directory(DATASET,
                        PATCH_SIZE,
                        CLASS):

    work_dir = os.environ.get("WORK")

    if DATASET == "alot" or DATASET=="ALOT":
        base = Path(work_dir) / "ICA" / "ALOT" / "alot_patches"

    elif DATASET == "celebA" or DATASET=="CelebA80_full":
        base = Path(work_dir) / "ICA" / "CelebA80_full"

    else:
        raise ValueError(f"Unknown dataset {DATASET}")

    size_dir = base / f"size_{PATCH_SIZE}" / CLASS
    return size_dir

def get_data_path(DATASET,
                  PATCH_SIZE,
                  WHITEN,
                  CLASS,
                  WHITENING_METHOD="cov"):

    size_dir = get_data_directory(DATASET, PATCH_SIZE, CLASS)

    if WHITEN:
        if WHITENING_METHOD == "cov":
            return size_dir / "white.npy"
        elif WHITENING_METHOD == "amplitude":
            return size_dir / "white_2.npy"
        else:
            raise ValueError("Unknown whitening method")
    else:
        return size_dir / "spatial.npy"


def train_test_split_indices(N, train_fraction, rng):
    indices = np.arange(N)
    rng.shuffle(indices)

    n_train = int(train_fraction * N)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return train_idx, test_idx


def get_data(DATASET,
             PATCH_SIZE,
             WHITEN, 
             CLASS_1,
             SPLIT_SEED,
             TRAIN_FRAC = 0.8,
             VERBOSE=True, 
             WHITENING_METHOD="cov"): 
    

    DATA_PATH_1 = get_data_path(DATASET, PATCH_SIZE, WHITEN, CLASS_1, WHITENING_METHOD=WHITENING_METHOD)
    DATA_DIR = get_data_directory(DATASET, PATCH_SIZE, CLASS_1)

    if VERBOSE:
        print(f"loading class one from {DATA_PATH_1}")
    X = np.load(DATA_PATH_1).astype(np.float32)
    
    N = len(X)
    X = X.reshape(N,1, PATCH_SIZE, PATCH_SIZE)
    split_rng = np.random.default_rng(SPLIT_SEED)
    train_idx, test_idx = train_test_split_indices(N, TRAIN_FRAC, split_rng)

    X_train, X_test = X[train_idx], X[test_idx]

    
    X_test = X_test[:10000]
    
    cov = np.load(DATA_DIR / "cov_spatial.npy").astype(np.float32)
    mean = np.load(DATA_DIR / "mean_spatial.npy").astype(np.float32)

    mean = torch.tensor(mean)
    cov = torch.tensor(cov)
    dim = PATCH_SIZE**2
    mean_only_model = Gaussian(dim=dim ,
                               device=torch.device("cpu"),
                               mean=mean)
    
    X_mean = mean_only_model.sample(10000).reshape(10000,1,PATCH_SIZE,PATCH_SIZE)


    cov_model = Gaussian(dim=dim, device=torch.device("cpu"),
                         mean=mean, covariance=cov)
    X_cov = cov_model.sample(10000).reshape(10000,1,PATCH_SIZE,PATCH_SIZE)
    
    return (
    X_train,
    X_test,
    X_mean,
    X_cov
    )

def diffusion_experiment(DATASET="CelebA80_full",
                        PATCH_SIZE = 8,
                        WHITEN = False, 
                        CLASS_1 = "full",
                        LR = 1e-4,
                        EPOCHS = 5,
                        TRAIN_FRAC = 0.8,  # fraction of data for training
                        RECORD_INTERVAL = "log", 
                        MAX_NUM_STEPS=100000, 
                        BATCH_SIZE=1,
                        SEED=0,
                        VERBOSE=True, 
                        WHITENING_METHOD="cov",
                        RECORD_STEPS=50, 
                        OPTIMIZER="AdamW",):


    # ----------------------------
    # Load dataset
    # ----------------------------
    # work_dir = os.environ.get("WORK")
    # directory = work_dir+"/ICA/ALOT/"
    SPLIT_SEED = 31*SEED + SEED**2 +13

    (
    X_train,
    X_test,
    X_mean,
    X_cov
    ) = get_data(DATASET,
                 PATCH_SIZE,
                WHITEN, 
                CLASS_1,
                SPLIT_SEED,
                TRAIN_FRAC = 0.8,
                VERBOSE=True,
                WHITENING_METHOD=WHITENING_METHOD)

    # Convert to torch tensors
    X_train = torch.Tensor(X_train)
    X_train_for_record = X_train[:10000]
    X_test = torch.Tensor(X_test)
    X_mean = torch.Tensor(X_mean)
    X_cov = torch.Tensor(X_cov)

    if MAX_NUM_STEPS is None: 
        MAX_NUM_STEPS = len(X_train)*EPOCHS//BATCH_SIZE

    # dataset + dataloader
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_dataset_for_record = TensorDataset(X_train_for_record)
    train_loader_for_record = DataLoader(train_dataset_for_record, batch_size=BATCH_SIZE, shuffle=True)
    # test dataset + dataloader
    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_mean_dataset = TensorDataset(X_mean)
    test_mean_loader = DataLoader(test_mean_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_cov_dataset = TensorDataset(X_cov)
    test_cov_loader = DataLoader(test_cov_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # data containers
    ts = ["all", 100, 500, 800]

    train_losses = {}
    test_losses = {}
    mean_clone_losses = {}
    cov_clone_losses = {}

    for t in ts: 
        for dict_ in [train_losses, test_losses, mean_clone_losses, cov_clone_losses]:
            dict_[t] = []


    models = []
    generated = []
    # objects for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusionmodel = Diffusion(None, PATCH_SIZE**2, device)
    
    model = UNet(diffusionmodel.T, PATCH_SIZE**2, diffusionmodel.beta,
                    in_channels=1, time_emb_dim=100).to(diffusionmodel.device)
    diffusionmodel.model = model

    if OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=LR)
    elif OPTIMIZER == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LR)
    elif OPTIMIZER == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LR)
    else:
        raise ValueError(f"Unknown optimizer {OPTIMIZER}")


    # Setup recording    
    if RECORD_INTERVAL=="log": 
        record_steps = np.logspace(0,np.log10(MAX_NUM_STEPS),RECORD_STEPS).astype(int)
    else:
        record_steps = np.arange(1, MAX_NUM_STEPS+1, RECORD_INTERVAL)
    record_steps = list(np.unique(record_steps))
    
    record_steps_models = list(np.copy(record_steps)[1::10])
   
    if not record_steps[-1] == MAX_NUM_STEPS: 
        record_steps += [MAX_NUM_STEPS]
    record_steps_ = np.copy(record_steps)

    if not record_steps_models[-1] == MAX_NUM_STEPS: 
        record_steps_models += [MAX_NUM_STEPS]
    record_steps_models_ = np.copy(record_steps_models)

    # place to record to: 
    work_dir = os.environ.get("WORK")
    directory = work_dir+f"/Diffeasy2hard_sub/{DATASET}/train_experiments_diff/"
    os.makedirs(directory, exist_ok=True)
    
    def record(diffusionmodel, ts = ["all", 100, 500, 800], record_model=False): 
        # Compute test loss for this step
        diffusionmodel.model.eval()
        with torch.no_grad():
            for t in ts:
                if not t == "all":
                    ts = lambda x: (torch.ones(len(x))*t).to(diffusionmodel.device).long()
                    loss_func = lambda x: diffusionmodel.loss(x, ts=ts(x))
                else: 
                    loss_func = lambda x: diffusionmodel.loss(x)  

                train_losses[t].append(
                    diffusionmodel.eval_loss_dataset(train_loader_for_record, loss_func=loss_func).item())
                
                test_losses[t].append(
                    diffusionmodel.eval_loss_dataset(test_loader, loss_func=loss_func).item())

                mean_clone_losses[t].append(
                    diffusionmodel.eval_loss_dataset(test_mean_loader, loss_func=loss_func).item())
                cov_clone_losses[t].append(
                    diffusionmodel.eval_loss_dataset(test_cov_loader, loss_func=loss_func).item())

            if record_model:
                models.append(diffusionmodel.model.state_dict())
                """
                savename = make_filename_from_args(directory=directory, 
                                        PATCH_SIZE = PATCH_SIZE,
                                            WHITEN = WHITEN, 
                                            CLASS_1 = CLASS_1,
                                            LR = LR,
                                            EPOCHS = EPOCHS,
                                            TRAIN_FRAC = TRAIN_FRAC,  # fraction of data for training
                                            RECORD_INTERVAL = RECORD_INTERVAL, 
                                            MAX_NUM_STEPS = MAX_NUM_STEPS,  
                                            BATCH_SIZE=BATCH_SIZE,
                                            SEED=SEED, 
                                            WHITENING_METHOD=WHITENING_METHOD, 
                                            step=num_steps)
                
                samples = diffusionmodel.sample(16, shape=(16,1,PATCH_SIZE,PATCH_SIZE)).detach().cpu().numpy()
                generated.append(samples)
                plot_samples(samples, savename, PATCH_SIZE)
                """
        diffusionmodel.model.train()
    # ----------------------------
    # Training loop with online SGD

    num_steps = 0
    for epoch in range(EPOCHS):
        for x_batch in train_loader:
            x_batch = x_batch[0].to(diffusionmodel.device)
            print(x_batch.shape)
            optimizer.zero_grad()
            
            # Forward pass
            loss = diffusionmodel.loss(x_batch)
            loss.backward()
            optimizer.step()
            
            # record if necessary        
            if num_steps == record_steps[0]: 
                record_model = (num_steps == record_steps_models[0])
                record(diffusionmodel, record_model=record_model)
                record_steps.pop(0)
                if record_model: 
                    record_steps_models.pop(0)
            if len(record_steps) == 0: 
                break
            num_steps += 1
        if len(record_steps) == 0: 
                break
            
    
    # create savename
    savename = make_filename_from_args(directory=directory, 
                                       PATCH_SIZE = PATCH_SIZE,
                                        WHITEN = WHITEN, 
                                        CLASS_1 = CLASS_1,
                                        LR = LR,
                                        EPOCHS = EPOCHS,
                                        TRAIN_FRAC = TRAIN_FRAC,  # fraction of data for training
                                        RECORD_INTERVAL = RECORD_INTERVAL, 
                                        MAX_NUM_STEPS = MAX_NUM_STEPS,  
                                        BATCH_SIZE=BATCH_SIZE,
                                        SEED=SEED, 
                                        WHITENING_METHOD=WHITENING_METHOD, 
                                        OPTIMIZER=OPTIMIZER,)
    results_all = {
        "model": model,
        "train_loss": train_losses,
        "test_loss": test_losses,
        "loss_on_mean_clone": mean_clone_losses,
        "loss_on_cov_clone": cov_clone_losses,
        "record_steps": record_steps_,
        "record_steps_models":record_steps_models_,
        "models": models,
        "generated": generated,
        "SPLIT_SEED": SPLIT_SEED
    }

    save_experiment_results(savename+".pkl", results_all)
    plot_result(
        train_losses["all"],
        test_losses["all"],
        mean_clone_losses["all"],
        cov_clone_losses["all"],
        record_steps_,
        savename=savename + ".pdf",
        PATCH_SIZE=PATCH_SIZE, 
            )

    
    return results_all

