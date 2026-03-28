import numpy as np
import torch
import matplotlib.pyplot as plt

from icastats.plotting.utils import set_nice_params, get_next_panel_label


def plot_result(
        train_losses,
        test_losses,
        mean_clone_losses,
        cov_clone_losses,
        record_steps,
        savename="test.pdf",
        PATCH_SIZE=8, 
):
    # ----------------------------
    # Plot loss & accuracy curves
    # ----------------------------
    set_nice_params()

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    steps = record_steps[:len(train_losses)]

    # ==========================
    # Loss panel
    # ==========================
    ax.plot(steps, train_losses, label="Train loss", color="pink")
    ax.plot(steps, test_losses, label="Test loss", color="red")
    ax.plot(steps, mean_clone_losses, label="mean clone loss", color="blue")
    ax.plot(steps, cov_clone_losses, label="cov clone loss", color="black")

    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(savename + ".pdf")
    plt.close()


def plot_samples( samples, savename, PATCH_SIZE): 

    fig, axs = plt.subplots(4,4, figsize=(8,8))
    
    axs = axs.flatten()

    for sample, ax in zip(samples, axs): 
        ax.imshow(sample.reshape(PATCH_SIZE, PATCH_SIZE))
        ax.axis("off")


    plt.tight_layout()
    plt.savefig(savename + "samples.pdf")
    plt.show()
    plt.close()