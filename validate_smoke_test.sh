#!/bin/bash

# 设置环境变量
export PYTHONPATH=src
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29551

# 执行内联 Python 脚本
python - <<'PY'
import os
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig

from utils.common import seed_everything, load_model
from dataload.data_preprocess import prepare_preprocessed_data
from src.main import val

@hydra.main(version_base="1.2", config_path="configs", config_name="small")
def main(cfg: DictConfig):
    cfg.gpu_num = 1
    seed_everything(cfg.seed)
    prepare_preprocessed_data(cfg)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=1,
        rank=0,
    )

    model = load_model(cfg).to("cuda:0")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])

    res = val(model=model, local_rank=0, cfg=cfg)
    print("--- Validation Metrics ---")
    print(res)
    print("------------------------")

if __name__ == "__main__":
    main()
PY
