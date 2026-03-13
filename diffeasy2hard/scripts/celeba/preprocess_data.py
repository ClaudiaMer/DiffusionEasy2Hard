import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import os

# ----------------------------
# Parameters
# ----------------------------
work_dir = os.environ.get("WORK")
DATA_DIR = work_dir+"/CelebAdata/data80x80"
OUTPUT_DIR = work_dir+"/CelebAdata/data80x80"
BATCH_SIZE = 128
SEED = 42


# ----------------------------
# Load datasets
# ----------------------------
def load_split(name):
    images = torch.load(f"{DATA_DIR}/{name}.pt")[0]
    print(f"{name} split loaded: images {images.shape}")
    images = 2*images -1
    return images

train_images = load_split("train")
val_images = load_split("validate")
test_images = load_split("test")

# ----------------------------
# Function to compute mean and covariance
# ----------------------------
def compute_mean_cov(images):
    """
    images: tensor of shape (N, 1, H, W)
    Returns: mean (D,), covariance (D, D) where D = H*W
    """
    N = images.shape[0]
    # Flatten to shape (N, D)
    X = images.view(N, -1)

    # Compute mean
    mean = X.mean(dim=0)

    # Center data
    X_centered = X - mean

    # Covariance = (X^T * X) / (N - 1)
    cov = (X_centered.T @ X_centered) / (N - 1)

    return mean, cov

# ----------------------------
# Compute statistics
# ----------------------------
print("\nComputing statistics...")

train_mean, train_cov = compute_mean_cov(train_images)
val_mean, val_cov = compute_mean_cov(val_images)
test_mean, test_cov = compute_mean_cov(test_images)

# ----------------------------
# Print results
# ----------------------------
print("\n--- Results ---")
print(f"Train Mean shape: {train_mean.shape}, Cov shape: {train_cov.shape}")
print(f"Val Mean shape:   {val_mean.shape}, Cov shape: {val_cov.shape}")
print(f"Test Mean shape:  {test_mean.shape}, Cov shape: {test_cov.shape}")

# Example: first 10 mean values
print("\nFirst 10 mean pixel values (Train):")
print(train_mean[:10])

# ----------------------------
# Save statistics (optional)
# ----------------------------
torch.save({"mean": train_mean, "cov": train_cov}, f"{DATA_DIR}/train_stats.pt")
torch.save({"mean": val_mean, "cov": val_cov}, f"{DATA_DIR}/val_stats.pt")
torch.save({"mean": test_mean, "cov": test_cov}, f"{DATA_DIR}/test_stats.pt")

print("\nStatistics saved successfully!")