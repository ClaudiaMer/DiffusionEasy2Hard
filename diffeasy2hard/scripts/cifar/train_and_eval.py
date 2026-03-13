import torch
import os
from diffeasy2hard.optim.train_model_checkpoints import \
      run_experiment
from diffeasy2hard.eval.eval_checkpoints import \
      eval_checkpoints


DATA_PATH = "cifar10_splits/"
WORK_DIR = os.environ.get("WORK")
DATA_DIR = WORK_DIR + "/" + DATA_PATH

def load_data(args): 
    # load training data
    data_tensor = torch.load(DATA_DIR + "train.pt")[0]
    return data_tensor


def model_save_folder(args): 

    work_dir = os.environ.get("WORK")+"/cifar10_"
    folder = work_dir+f"trained/checkpoints_seed{args.seed}_N{args.N}"
    if args.adamW: 
        folder += "_adamW"
    
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
    
    weight_decays = [0.0]

    data_folder = DATA_DIR
    model_folder = model_save_folder(args)
    save_folder = f"error/checkpoints_seed{seed}_N{N}/"

    # evaluate checkpoints and save losses
    eval_checkpoints(data_folder, model_folder, save_folder, 1, weight_decay,c)

    # evaluate checkpoints at different timesteps and save losses
    for t in ts:
        eval_checkpoints(data_folder, model_folder, save_folder, 1, weight_decay,c,t=t)

