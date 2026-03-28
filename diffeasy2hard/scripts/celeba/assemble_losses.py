import os
import torch
import pickle

work_dir = os.environ.get("WORK")

# PATCH_SIZE = 16

losses_train = []
losses_test = []
loss_on_cov_clone = []
loss_on_mean_clone = []

for seed in [0,1,2,3,4, 10,11,12,13,14]: 

    path = f"{work_dir}/Diffeasy2hard_sub/CelebA80_full/train_experiments_diff/exp_PATCH_SIZE16_WHITEN0_CLASS_1full_LR0.0001_EPOCHS1_TRAIN_FRAC0.8_RECORD_INTERVALlog_MAX_NUM_STEPS1620_BATCH_SIZE100_SEED{seed}_WHITENING_METHODcov.pkl"
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = data["results"]  # extract the "results" dict from the loaded data
    for key in ["train_loss", "test_loss", "loss_on_cov_clone", "loss_on_mean_clone"]:
        if key == "train_loss":
            losses_train.append(data[key])
        elif key == "test_loss":
            losses_test.append(data[key])
        elif key == "loss_on_cov_clone":
            loss_on_cov_clone.append(data[key])
        elif key == "loss_on_mean_clone": 
            loss_on_mean_clone.append(data[key])


results = {
    "losses_train": losses_train,
    "losses_test": losses_test,
    "loss_on_cov_clone": loss_on_cov_clone, 
    "loss_on_mean_clone": loss_on_mean_clone,
    "steps": data["record_steps"]
}

print(results)
torch.save(results, f"error/assembled_losses_PatchSize16.pt")

# PATCH_SIZE = 32

losses_train = []
losses_test = []
loss_on_cov_clone = []
loss_on_mean_clone = []

for seed in [0,1,2,3, 10,11,12,13,14]: 

    path = f"{work_dir}/Diffeasy2hard_sub/CelebA80_full/train_experiments_diff/exp_PATCH_SIZE32_WHITEN0_CLASS_1full_LR0.0001_EPOCHS4_TRAIN_FRAC0.8_RECORD_INTERVALlog_MAX_NUM_STEPS6483_BATCH_SIZE100_SEED{seed}_WHITENING_METHODcov.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = data["results"]  # extract the "results" dict from the loaded data

    for key in ["train_loss", "test_loss", "loss_on_cov_clone", "loss_on_mean_clone"]:
        if key == "train_loss":
            losses_train.append(data[key])
        elif key == "test_loss":
            losses_test.append(data[key])
        elif key == "loss_on_cov_clone":
            loss_on_cov_clone.append(data[key])
        elif key == "loss_on_mean_clone": 
            loss_on_mean_clone.append(data[key])

results = {
    "losses_train": losses_train,
    "losses_test": losses_test,
    "loss_on_cov_clone": loss_on_cov_clone, 
    "loss_on_mean_clone": loss_on_mean_clone, 
    "steps": data["record_steps"]
}

torch.save(results, f"error/assembled_losses_PatchSize32.pt")


# PATCH_SIZE = 50

losses_train = []
losses_test = []
loss_on_cov_clone = []
loss_on_mean_clone = []

for seed in [0,1,2,3,4,10,11,12,13,14]: 

    path = f"{work_dir}/Diffeasy2hard_sub/CelebA80_full/train_experiments_diff/exp_PATCH_SIZE50_WHITEN0_CLASS_1full_LR0.0001_EPOCHS6_TRAIN_FRAC0.8_RECORD_INTERVALlog_MAX_NUM_STEPS9724_BATCH_SIZE100_SEED{seed}_WHITENING_METHODcov.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = data["results"]  # extract the "results" dict from the loaded data
    for key in ["train_loss", "test_loss", "loss_on_cov_clone", "loss_on_mean_clone"]:
        if key == "train_loss":
            losses_train.append(data[key])
        elif key == "test_loss":
            losses_test.append(data[key])
        elif key == "loss_on_cov_clone":
            loss_on_cov_clone.append(data[key])
        elif key == "loss_on_mean_clone": 
            loss_on_mean_clone.append(data[key])

results = {
    "losses_train": losses_train,
    "losses_test": losses_test,
    "loss_on_cov_clone": loss_on_cov_clone, 
    "loss_on_mean_clone": loss_on_mean_clone,
    "steps": data["record_steps"]
}

torch.save(results, f"error/assembled_losses_PatchSize50.pt")