
import torch
import numpy as np

def steps2epochs(steps,batch_size,dataset_size): 
    """Convert number of steps to number of epochs."""
    steps_in_epoch = dataset_size//batch_size
    n_epochs = steps//steps_in_epoch
    return n_epochs


def copy_data(num_latent, N, train_data): 
    """Copy the training data to create a larger dataset if num_latent is greater than N."""
    if N >= num_latent: 
        return train_data[:num_latent]
    else: 
        num_times = int(np.ceil(num_latent/N))
        stacked_data = torch.vstack([train_data,]*num_times)
        return stacked_data[:num_latent]
    
def cosine_sim_img_batch(img, batch): 
    """Compute the cosine similarity between a single image and a batch of images."""
    norm_batch = torch.sqrt(torch.einsum("ncij ->n", batch**2))
    norm_img = torch.sqrt(torch.sum(img**2))
    cosine_sims = torch.einsum("ncij,cij, n -> n", batch, img, 1.0/norm_batch)/norm_img
    return cosine_sims

def find_closest(img, batch): 
    """Find the closest image in the batch to the given image using cosine similarity."""
    cosine_sims = cosine_sim_img_batch(img, batch)
    max_sim_idx = torch.argmax(cosine_sims)
    max_sim = cosine_sims[max_sim_idx]
    closest_img_in_batch = batch[max_sim_idx]
    return max_sim, closest_img_in_batch
