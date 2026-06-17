# DiffusionEasy2Hard
Experiments on distributional simplicity bias in learning diffusion models

Use diffeasy2hard/MCM_twolayer_minimal to reproduce MCM experiments

Use diffeasy2hard/scrips/{celeba,cifar} to reproduce sequential learning experiments

## Structure of the project: 

- models: Diffusion model class for training and sampling from diffusion models and utilities (u_net model, noising beta schedule, loss tracking function)
- optim: contains main training routine (train model checkpoints) as well as an optional additional routine to train on datasets with sub-sampled images (lower resolution images)
- eval: contains eval_checkpoints that evaluates trained models on data clones (Gaussian data with mean and covariance matched to image data)
- stats: contains class Gaussian used to define and sample from data clones
- plotting: visualization utilities
- utils: image comparison and saving and loading utilities
- scripts: preprocessing, execution and evaluation pipeline for these datasets
