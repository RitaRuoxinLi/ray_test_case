# ray_test_case
Generate test case of ray for the HISE team

# The first test case: parallel_computing_in_preprocessing.py is  Ray-Powered Scrublet Test Pipeline

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
