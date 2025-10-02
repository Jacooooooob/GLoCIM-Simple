# ✨GLORY: Global Graph-Enhanced Personalized News Recommendations
Code for our paper [_Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations_](https://arxiv.org/pdf/2307.06576.pdf) published at RecSys 2023. 

<p align="center">
  <img src="glory.jpg" alt="Glory Model Illustration" width="600" />
  <br>
  Glory Model Illustration
</p>


### Environment
> Python 3.8.10
> pytorch 1.13.1+cu117
```shell
cd GLORY

apt install unzip python3.8-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```shell
# dataset download
bash scripts/data_download.sh

# Run
python3 src/main.py model=GLORY dataset=MINDsmall reprocess=True
```

## Speed-up: Fast Validation Controls

This repo adds validation-time controls. For fair, high-fidelity comparison with GLORY, defaults here favor accuracy over speed.

- `val_mode` (bool or string): turn validation on/off (`true`/`false`, or `'off'/'0'` to disable).
- `val_steps` (int): validate every N training steps.
- `val_start_step` (int): do not run validation before this step.
- `val_embed_mode` (`cache` | `recompute`):
  - `recompute` (default for fidelity): re-encode on every validation to reflect current weights.
  - `cache`: encode once and reuse cached `news_emb_cache.pt` (faster, may lag behind model updates).
- `val_sample_ratio` (float 0–1): sample a fraction of validation to trade accuracy for speed.

Where they live:
- Small config: `configs/small.yaml`
- Large config: `configs/large.yaml` (more aggressive defaults)
- Global defaults: `configs/default.yaml`

### Recommended Settings

- High-fidelity (to align with GLORY):
  - `val_embed_mode: recompute`, `val_amp: false`, `val_sample_ratio: 1.0`.
  - Use this when you want validation to track current model weights exactly.

- Speed-focused (override via CLI when needed):
  - `+val_embed_mode=cache +val_amp=true +val_sample_ratio=0.1` (plus adjust `val_steps`/`val_start_step`).

### CLI Overrides (Hydra)

You can override any config on the command line, for example:

```bash
# Disable validation entirely
python3 src/main.py +val_mode=false

# Start validating later and less frequently
python3 src/main.py +val_start_step=20000 +val_steps=20000

# Use cached embeddings and sample only 10% of validation for speed
python3 src/main.py +val_embed_mode=cache +val_sample_ratio=0.1 +val_amp=true

# Force recompute embeddings (slow, most faithful; default here)
python3 src/main.py +val_embed_mode=recompute +val_amp=false
```

### Caching Details

- Cache path: `<dataset_dir>/val/news_emb_cache.pt` (Torch tensor on CPU).
- To force rebuild: set `val_embed_mode=recompute` or delete the cache file.
- Neighbor vectors for long-chain modeling are also generated once via `prepare_neighbor_vec_list` and saved under the dataset directory (controlled by `reprocess_neighbors`).

### Notes

- For multi-GPU (DDP), only rank 0 writes the cache and a barrier is used to avoid read-after-create races.
- If you change the news encoder weights during training and want validation embeddings to reflect that, set `val_embed_mode=recompute`.

### Validation Neighbor Injection

- Control whether validation injects offline neighbor vectors into the long-chain encoder via `val_use_offline_neighbors`.
  - `false` (default): disable injection for GLORY-aligned fidelity.
  - `true`: enable injection to leverage precomputed neighbors.

Examples:

```bash
# Disable offline neighbor injection during validation (GLORY-aligned)
python3 src/main.py +val_use_offline_neighbors=false

# Enable offline neighbor injection during validation
python3 src/main.py +val_use_offline_neighbors=true
```

### Smoke Test (Non-Graph Validation)

We include a tiny smoke test to verify the non-graph validation DataLoader and collate pathway with pinned memory:

```bash
python3 scripts/smoke_non_graph_val.py
```

This script builds a minimal ValidDataset from the first two lines of `data/MINDsmall/val/behaviors_np4_0.tsv`, enables `pin_memory=True`, and checks that the collated outputs are CPU tensors (preferably pinned) ready for fast H2D copies.


### Bibliography

```shell
@misc{yang2023going,
      title={Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations}, 
      author={Boming Yang and Dairui Liu and Toyotaro Suzumura and Ruihai Dong and Irene Li},
      year={2023},
      publisher ={RecSys},
}
```


```shell
cd /home/mist/.virtualenvs/bin
source activate
python3 ~/wwh/GLORY/src/main.py model=GLORY dataset=MINDsmall reprocess=True
}
```



