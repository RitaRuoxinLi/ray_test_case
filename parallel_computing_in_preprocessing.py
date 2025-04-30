import ray
import numpy as np
import scanpy as sc
import logging
import scrublet as scr

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Start Ray
ray.init(
    _memory=int(4e9),  # ~4 GB total memory
    object_store_memory=int(1e9),  # 1 GB for object store (uses /tmp instead of /dev/shm)
    ignore_reinit_error=True
)
ray.init(num_cpus=12, ignore_reinit_error=True)


@ray.remote
def run_scrublet_on_sample(sample_adata, sample_id):
    """
    Run scrublet on an individual sample's adata and return annotated object.
    """
    try:
        X = sample_adata.X
        counts_dense = X.toarray() if not isinstance(X, np.ndarray) else X
        scrub = scr.Scrublet(counts_dense)
        doublet_scores, predicted_doublets = scrub.scrub_doublets()
        sample_adata.obs['doublet_score'] = doublet_scores
        sample_adata.obs['predicted_doublet'] = predicted_doublets
        sample_adata.obs['sample'] = sample_id
        logging.info(f"✅ Scrublet done for sample: {sample_id}")
        return sample_adata
    except Exception as e:
        logging.error(f"❌ Failed on sample {sample_id}: {e}")
        return None

# Load preprocessed merged AnnData
adata_path = "private/colectomy_adata_baysor.h5ad"  # update path
adata_cpm = sc.read_h5ad(adata_path)

# Split by sample
sample_ids = adata_cpm.obs['sample'].unique()
adata_per_sample = {
    sid: adata_cpm[adata_cpm.obs['sample'] == sid].copy() for sid in sample_ids
}

futures = [
    run_scrublet_on_sample.options(num_cpus=1).remote(adata, sid)
    for sid, adata in adata_per_sample.items()
]


# Collect results
annotated_adatas = ray.get(futures)
annotated_adatas = [a for a in annotated_adatas if a is not None]

print(f"🎉 Scrublet completed for {len(annotated_adatas)} samples.")