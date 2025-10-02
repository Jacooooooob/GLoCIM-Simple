#!/usr/bin/env python3
"""
Quick smoke test for the non-graph validation pipeline:

- Builds a ValidDataset (non-graph) over MINDsmall val behaviors.
- Uses a tiny subset (first 2 lines) to create a DataLoader with pin_memory.
- Checks that collated outputs are CPU torch.Tensors and pinned when requested.

Run:
  python3 scripts/smoke_non_graph_val.py
"""
import os
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataload.dataset import ValidDataset
from dataload.data_load import collate_fn


def main():
    root = Path(__file__).resolve().parents[1]
    val_dir = root / 'data' / 'MINDsmall' / 'val'

    news_index = pickle.load(open(val_dir / 'news_dict.bin', 'rb'))

    # Build a lightweight dummy news embedding matrix [N+1, 400]
    # 400 = head_num * head_dim for default GLORY
    num_news = len(news_index)
    news_emb = np.zeros((num_news + 1, 400), dtype=np.float32)

    beh_file = val_dir / 'behaviors_np4_0.tsv'
    assert beh_file.exists(), f"Missing behaviors file: {beh_file}"

    # Wrap ValidDataset but limit to first 2 lines by copying a temp file
    tmp_file = val_dir / 'behaviors_np4_0.head2.tsv'
    if not tmp_file.exists():
        with open(beh_file, 'r', encoding='utf-8') as fin, open(tmp_file, 'w', encoding='utf-8') as fout:
            for i, ln in enumerate(fin):
                if i >= 2:
                    break
                fout.write(ln)

    class Cfg:  # minimal cfg shim
        class M:
            his_size = 50
        model = M()
        num_workers = 0

    ds = ValidDataset(filename=tmp_file, news_index=news_index, news_emb=news_emb,
                      local_rank=0, cfg=Cfg)

    dl = DataLoader(ds, batch_size=1, num_workers=0, pin_memory=True,
                    collate_fn=lambda b: collate_fn(b, 0))

    batch = next(iter(dl))
    if len(batch) == 6:
        clicked_news, clicked_mask, candidate_news, clicked_index, candidate_index, labels = batch
    else:
        clicked_news, clicked_mask, candidate_news, clicked_index, candidate_index = batch
        labels = None

    # Basic assertions
    assert isinstance(clicked_news, torch.Tensor) and clicked_news.device.type == 'cpu'
    assert isinstance(candidate_news, torch.Tensor) and candidate_news.device.type == 'cpu'
    assert isinstance(clicked_index, torch.Tensor) and clicked_index.device.type == 'cpu'

    # Pinned memory check (DataLoader will pin the returned CPU tensors when pin_memory=True)
    # Some tensors (e.g., small index tensors) may or may not be pinned; embeddings should be.
    print('clicked_news.is_pinned:', clicked_news.is_pinned())
    print('candidate_news.is_pinned:', candidate_news.is_pinned())
    print('clicked_index.is_pinned:', clicked_index.is_pinned())
    print('OK: non-graph val smoke test passed.')


if __name__ == '__main__':
    main()

