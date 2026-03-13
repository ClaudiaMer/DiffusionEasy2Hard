import argparse
import torch
import os
from diffeasy2hard.optim.train_model_checkpoints import \
    run_experiment
from diffeasy2hard.eval.eval_checkpoints import \
    eval_checkpoints

WORK_DIR = os.environ.get("WORK")
DATA_DIR = WORK_DIR + "/CelebAdata/data80x80/"

def load_data(args): 
    data_tensor = torch.load(DATA_DIR+"train.pt")[0]
        
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
    save_folder = f"error/checkpoints_seed{seed}_N{N}/"

    # evaluate checkpoints and save losses
    eval_checkpoints(data_folder, model_folder, save_folder, 1, weight_decay,c, dim1d=80)

    # evaluate checkpoints at different timesteps and save losses
    for t in ts:
        eval_checkpoints(data_folder, model_folder, save_folder, 1, weight_decay,c,t=t, dim1d=80)