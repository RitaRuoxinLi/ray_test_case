# ray_test_case
Generate test case of ray for the HISE team

## The first test case: parallel_computing_in_preprocessing.py is  Ray-Powered Scrublet Test Pipeline

This repository contains a minimal test case demonstrating how to parallelize [Scrublet](https://github.com/AllonKleinLab/scrublet) — a tool for detecting doublets in single-cell RNA-seq data — using [Ray](https://www.ray.io/).

## 🧪 Purpose

To test and validate parallel task scheduling with Ray by running Scrublet independently on multiple biological samples stored within a merged AnnData object (`colectomy_adata_baysor.h5ad`).

This script is designed to:
- Evaluate Ray’s handling of memory and CPU allocation
- Serve as a blueprint for scaling biological preprocessing pipelines
- Act as a reproducible performance test case in constrained environments (e.g. Google Cloud IDEs)

## 🧬 Pipeline Overview

1. **Load** a preprocessed AnnData object (`.h5ad`) containing multiple samples.
2. **Split** the data by sample ID using `adata.obs['sample']`.
3. **Run Scrublet** per sample to:
   - Simulate doublets
   - Predict doublet scores
   - Annotate the sample-level AnnData
4. **Parallelize** step 3 across samples using `ray.remote`, with each task limited to `num_cpus=1` to avoid resource contention.
5. **Collect and verify** all annotated outputs.

## ⚙️ Key Features

- ✅ Ray-powered parallel task execution
- ✅ Sample-level CPU isolation (`num_cpus=1`)
- ✅ Fault-tolerant: failed samples are logged and skipped
- ✅ Works in cloud IDEs with limited shared memory (e.g. `/dev/shm`)
- ✅ Can be extended to batch save or merge results

## 🔍 Notes on Environment

- If running in Docker or constrained environments, `/dev/shm` may be too small.
- Ray will fall back to `/tmp` for object store, which reduces performance.
- To avoid oversubscription, limit active workers or set `num_cpus` explicitly.

## 🧩 Usage

Make sure you have the following in your environment:
```bash
pip install ray scanpy anndata scrublet
```
## The Second Test Case: scVI Hyperparameter Optimization

This repository demonstrates how to use [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to perform scalable hyperparameter search for training [scVI](https://scvi-tools.org/) models on the latent space of single-cell and spatial transcriptomics data. This test case also illustrates how to leverage Hise HPC’s GPU environment for efficient model training.

[scVI](https://scvi-tools.org/) is based on a **variational autoencoder (VAE)**, a deep generative model that learns a compressed latent representation of gene expression while modeling technical noise such as dropout and batch effects—making it well-suited for denoising and embedding high-dimensional single-cell data.

---

### 🚀 Key Features

- Train/validation/test splitting using `train_test_split`
- Custom `train_scvi_with_ray` function compatible with Ray Tune
- Configurable search space for hyperparameters (e.g., batch size)

> 💡 *Note: This example only tunes `batch_size` for simplicity. In practical use, you are encouraged to tune additional parameters such as `n_layers`, `n_hidden`, and `lr` via `tune.choice([...])` or `tune.loguniform(...)`.*

- Retrieval and reuse of the best trial configuration
- Saving and reloading of the best trained scVI model
- Extraction of latent embeddings (`X_scVI`)

---

### 🧰 Use Cases & Extensions

This repository serves as a **template for integrating Ray Tune with scVI**, and can be extended to support:

- Expanded search spaces (e.g., network depth, learning rate)
- Checkpointing and recovery of intermediate training states
- Multi-node training with Ray on HPC clusters

---

### 📂 Input Data

Path to input `.h5ad` file: /home/workspace/private/adata_ref_sp.h5ad

---

### 🧪 Output

- Best trained model is saved to:
- /home/workspace/private/best_scvi_model

- Latent representation is stored in:
```python
adata_ref_sp.obsm['X_scVI']


