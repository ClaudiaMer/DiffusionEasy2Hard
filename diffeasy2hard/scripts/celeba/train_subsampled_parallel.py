import os
from diffeasy2hard.optim.subsampled_data_train_routine import diffusion_experiment

CLASSES = ["full"]
batchsize = 100
RECORD_STEPS = 50

# Build the full list of configurations
configs = []
for patchsize, epochs in zip([16,32,],[8,8,8]):#[8, 16, 32],[10, 10, 10]):
    for class_1 in CLASSES:
        for LR in [0.001]:
            for SEED in [0,1,2]:
                configs.append({
                    "CLASS_1": class_1,
                    "PATCH_SIZE": patchsize,
                    "EPOCHS": epochs,
                    "LR": LR, 
                    "SEED": SEED, 
                })

# Get task index from SLURM
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

cfg = configs[task_id]

print(f"Running config {task_id}: {cfg}")

diffusion_experiment(
    DATASET="CelebA80_full",
    CLASS_1=cfg["CLASS_1"],
    PATCH_SIZE=cfg["PATCH_SIZE"],
    VERBOSE=True,
    LR=cfg["LR"],
    BATCH_SIZE=batchsize,
    EPOCHS=cfg["EPOCHS"],
    MAX_NUM_STEPS=None,
    RECORD_STEPS=RECORD_STEPS, 
    SEED=cfg["SEED"], 
    OPTIMIZER="SGD"
)
            