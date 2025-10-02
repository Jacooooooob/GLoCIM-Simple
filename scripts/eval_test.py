#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on the test set and print AUC/MRR/nDCG@5/@10.

How to run (examples):
  # Auto-pick best checkpoint; explicit Hydra config name
  PYTHONPATH=src WANDB_MODE=offline python scripts/eval_test.py --config-name small

  # Specify checkpoint via env var (avoid Hydra arg conflicts)
  EVAL_CKPT=checkpoint/GLORY_MINDsmall_bots_10pct_default_auc0.6757.pth \
  PYTHONPATH=src WANDB_MODE=offline python scripts/eval_test.py --config-name small

Notes:
  - By default, picks the highest-AUC checkpoint under `checkpoint/`.
  - Reuses the existing validation pipeline but redirects `val_dir -> test_dir`.
  - Runs single-GPU DDP to satisfy distributed barriers in validation.
"""
import json
import os
from pathlib import Path
import re
from typing import Optional

import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig

# Local imports with fallbacks for PYTHONPATH layouts
try:
    from utils.common import seed_everything, get_root, load_model
except Exception:  # pragma: no cover
    from src.utils.common import seed_everything, get_root, load_model
try:
    from dataload.data_preprocess import prepare_preprocessed_data
except Exception:  # pragma: no cover
    from src.dataload.data_preprocess import prepare_preprocessed_data
try:
    from src.main import val  # when PYTHONPATH points to project root
except Exception:  # pragma: no cover
    from main import val      # when PYTHONPATH includes src


def _pick_best_ckpt(ckp_dir: Path) -> Optional[Path]:
    """Pick checkpoint with the highest AUC embedded in filename.

    Filename pattern example:
      GLORY_MINDsmall_default_auc0.6757506728172302.pth
      GLORY_MINDsmall_bots_10pct_default_auc0.6718.pth
    """
    if not ckp_dir.exists():
        return None
    best_path, best_auc = None, float('-inf')
    pat = re.compile(r"auc([0-9]*\.[0-9]+)")
    for p in ckp_dir.glob("*.pth"):
        m = pat.search(p.name)
        if not m:
            continue
        try:
            auc = float(m.group(1))
        except Exception:
            continue
        if auc > best_auc:
            best_auc, best_path = auc, p
    return best_path


def _init_ddp():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=1,
        rank=0,
    )


def _load_ckpt_into_model(model: torch.nn.Module, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)


def main_inner():
    # Ensure project root is discoverable by config (path.default uses PROJECT_ROOT)
    root = get_root()
    os.environ.setdefault("PROJECT_ROOT", str(root))

    # W&B offline to avoid any networking during eval
    os.environ.setdefault("WANDB_MODE", "offline")

    @hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="small")
    def _runner(cfg: DictConfig):
        # Single-GPU eval
        cfg.gpu_num = 1
        seed_everything(cfg.seed)

        # Make sure test preprocessing exists (no-op if already prepared)
        prepare_preprocessed_data(cfg)

        # Redirect validation directory to test directory for reuse of val() pipeline
        cfg.dataset.val_dir = cfg.dataset.test_dir

        # Init DDP (val() uses dist.barrier and reduce)
        _init_ddp()

        try:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

            # Build model and wrap DDP
            model = load_model(cfg).to(device)
            ddp_kwargs = {"device_ids": [0]} if device.type == 'cuda' else {}
            model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

            # Pick or use provided checkpoint (env: EVAL_CKPT or CKPT_PATH)
            ckpt_env = os.environ.get("EVAL_CKPT") or os.environ.get("CKPT_PATH")
            ckpt_path = Path(ckpt_env) if ckpt_env else _pick_best_ckpt(Path(cfg.path.ckp_dir))
            if not ckpt_path or not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found. Env EVAL_CKPT={ckpt_env!r}; auto-picked={ckpt_path}")

            _load_ckpt_into_model(model.module, ckpt_path)

            # Run evaluation on test (via val())
            res = val(model=model, local_rank=0, cfg=cfg)

            # Pretty print + save json
            print("--------------------测试集指标-----------------------")
            for k in ["auc", "mrr", "ndcg5", "ndcg10"]:
                print(f"{k}\t{res.get(k)}")
            print("--------------------------------------------------")

            out_dir = root / "outputs" / "test_eval"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "metrics.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"checkpoint": str(ckpt_path), **res}, f, ensure_ascii=False, indent=2)
            print(f"Saved: {out_path}")
        finally:
            # Best-effort cleanup
            if dist.is_available() and dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass

    _runner()


if __name__ == "__main__":
    # No argparse to avoid conflicts with Hydra; use env var EVAL_CKPT if needed
    main_inner()
