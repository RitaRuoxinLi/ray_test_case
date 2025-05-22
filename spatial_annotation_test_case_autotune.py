"""
Example script: Training a scVI model with Ray Tune for hyperparameter optimization

This script demonstrates how to:
1. Load a spatial transcriptomics dataset (AnnData).
2. Perform a stratified train/val/test split.
3. Use Ray Tune to optimize scVI training hyperparameters (batch size in this case).
4. Evaluate and save the best model.
5. Extract the latent embedding from the best model.

Author: Ruoxin
"""

# ================================
# Library Imports
# ================================
import anndata as ad
import squidpy as sq
import cellcharter as cc
import pandas as pd
import scanpy as sc
import scvi
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
import os
import gc
import ray
from ray import tune, train
from datetime import datetime
import argparse
import json
import torch
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

# ================================
# Reproducibility Settings
# ================================
seed_everything(12345)
scvi.settings.seed = 12345

# ================================
# Argument Parsing for Manual Run
# ================================
parser = argparse.ArgumentParser(description="SCVI training hyperparameters")

# Model architecture
parser.add_argument("--n_layers", type=int, default=2, help="Number of hidden layers in encoder/decoder")
parser.add_argument("--n_latent", type=int, default=48, help="Dimensionality of the latent space")
parser.add_argument("--n_hidden", type=int, default=128, help="Number of hidden units per layer")
parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--gene_likelihood", type=str, default="zinb", help="Gene likelihood model (e.g., zinb, nb)")

# Training configuration
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

args = parser.parse_args()

# ================================
# Load Data
# ================================
adata_ref_sp = sc.read_h5ad("/home/workspace/private/adata_ref_sp.h5ad")

# ================================
# Training Function (Ray Compatible)
# ================================
def train_scvi_with_ray(config, adata_ref_sp, train_idx, val_idx, test_idx, return_model=False):
    """
    Train a scVI model with a given configuration and data split.
    
    Args:
        config (dict): Ray Tune hyperparameter dictionary.
        adata_ref_sp (AnnData): Reference annotated data.
        train_idx, val_idx, test_idx (np.ndarray): Indices for data split.
        return_model (bool): Whether to return the model (for final training).
        
    Returns:
        SCVI model object if return_model is True.
    """
    scvi.model.SCVI.setup_anndata(adata_ref_sp, layer="counts", batch_key='sample')
    
    model = scvi.model.SCVI(
        adata_ref_sp,
        n_layers=config["n_layers"],
        n_latent=config["n_latent"],
        n_hidden=config["n_hidden"],
        dropout_rate=config["dropout_rate"],
        dispersion="gene",
        gene_likelihood=config["gene_likelihood"]
    )

    model.train(
        early_stopping=True,
        enable_progress_bar=False,
        max_epochs=config.get("max_epochs", 1000),
        batch_size=config["batch_size"],
        plan_kwargs={
            "lr": config["lr"],
            "n_epochs_kl_warmup": config.get("n_epochs_kl_warmup", 400),
            "reduce_lr_on_plateau": True
        },
        datasplitter_kwargs={
            "external_indexing": [train_idx, val_idx, test_idx]
        },
        check_val_every_n_epoch=1
    )

    # Report validation ELBO
    val_loss = np.array(model.history["elbo_validation"].values, dtype=float).flatten()
    if return_model:
        return model
    else:
        train.report({"val_elbo": val_loss[-1]})

# ================================
# Step 1: Train/Validation/Test Split
# ================================
train_val_idx, test_idx = train_test_split(
    np.arange(adata_ref_sp.shape[0]),
    test_size=0.1,
    random_state=42
)
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.05,
    random_state=42
)

# ================================
# Step 2: Define Hyperparameter Search Space
# ================================
search_space = {
    "n_layers": args.n_layers,
    "n_latent": args.n_latent,
    "n_hidden": args.n_hidden,
    "dropout_rate": args.dropout_rate,
    "gene_likelihood": args.gene_likelihood,
    "batch_size": tune.choice([64, 128, 512, 1024]),
    "lr": args.lr
}

# ================================
# Step 3: Run Ray Tune
# ================================
analysis = tune.run(
    tune.with_parameters(
        train_scvi_with_ray,
        adata_ref_sp=adata_ref_sp,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx
    ),
    config=search_space,
    num_samples=1,
    metric="val_elbo",
    mode="min",
    resources_per_trial={"cpu": 0, "gpu": 4},
    name="scvi_ray_tune"
)

# ================================
# Step 4: Retrieve and Apply Best Config
# ================================
best_trial = analysis.get_best_trial(metric="val_elbo", mode="min", scope="all")
print("Best trial config:", best_trial.config)

# ================================
# Step 5: Retrain Final Model with Best Config
# ================================
model = train_scvi_with_ray(best_trial.config, adata_ref_sp, train_idx, val_idx, test_idx, return_model=True)

# Save model
dir_path = "/home/workspace/private/best_scvi_model"
os.makedirs(dir_path, exist_ok=True)
model.save(dir_path, overwrite=True)

# ================================
# Step 6: Generate Latent Embeddings
# ================================
model = scvi.model.SCVI.load(dir_path, adata=adata_ref_sp)
adata_ref_sp.obsm['X_scVI'] = model.get_latent_representation(adata_ref_sp).astype(np.float32)
