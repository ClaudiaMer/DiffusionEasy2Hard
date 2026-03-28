import argparse
import torch
import numpy as np
import os
from diffeasy2hard.optim.train_model_checkpoints import \
    run_experiment
from diffeasy2hard.eval.eval_checkpoints import \
    eval_checkpoints
from pathlib import Path


WORK_DIR = os.environ.get("WORK")
DATA_DIR = WORK_DIR + "/CelebAdata/data80x80/"

def get_data_directory(DATASET,
                       PATCH_SIZE,
                       CLASS):

    work_dir = os.environ.get("WORK")

    if DATASET == "celebA" or DATASET=="CelebA80_full":
        base = Path(work_dir) / "ICA" / "CelebA80_full"

    else:
        raise ValueError(f"Unknown dataset {DATASET}")

    size_dir = base / f"size_{PATCH_SIZE}" / CLASS
    return size_dir

def load_data(args): 
    if args.dim_1d == 80:
        data_tensor = torch.load(DATA_DIR+"train.pt")[0]
    else: 
        dir = get_data_directory("CelebA80_full", args.dim_1d, "full")
        data_path = dir / "spatial.npy"
        data_numpy = np.load(data_path).astype(np.float32)
        data_tensor = data_numpy.reshape(-1, 1, args.dim_1d, args.dim_1d)

    if data_tensor.min() >= 0.0 and data_tensor.max() <= 1.0: 
        print("Data seems to be in [0,1], normalizing to [-1,1]")
        data_tensor = 2.0 * data_tensor - 1.0   # map [0,1] → [-1,1]
    return data_tensor

def model_save_folder(args): 

    work_dir = os.environ.get("WORK")+"/CelebAdata/"
    folder = work_dir+f"trained/checkpoints_seed{args.seed}_N{args.N}"
    if args.adamW: 
        folder += "_adamW"
    if args.cosine_lr:
        folder += "_cosine"
        
    folder += "/"
    os.makedirs(folder, exist_ok=True)
    return folder

if __name__ == "__main__": 
    # train models and save checkpoints
    # use argparse commands to specify hyperparameters such as seed, N, weight decay, c, etc.
    args = run_experiment(load_data, model_save_folder)

    ts = [100,500,800]
    
    seeds = list(range(1,3))
    N = args.N
    c = args.c 
    weight_decay = args.weight_decay
    seed = args.seed

    data_folder = DATA_DIR
    model_folder = model_save_folder(args)
    save_folder = f"error/checkpoints_seed{seed}_N{N}_PATCH_SIZE{args.dim_1d}/"

    # evaluate checkpoints and save losses
    eval_checkpoints(data_folder, model_folder, save_folder, 1, weight_decay,c, dim1d=80)

    # evaluate checkpoints at different timesteps and save losses
    for t in ts:
        eval_checkpoints(data_folder, model_folder, save_folder, 1, weight_decay,c,t=t, dim1d=80)