import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import os
from torch_geometric.utils import to_undirected
#将有向图转化为无向图
from tqdm import tqdm
import pickle
#序列化和反序列化
from dataload.dataset import *
from dataload.data_preprocess import prepare_distributed_data


def load_data(cfg, mode='train', model=None, local_rank=0):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    # ------------- load news.tsv-------------
    news_index = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))

    news_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    # ------------- load behaviors_np{X}.tsv --------------
    if mode == 'train':
        target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_{local_rank}.tsv"
        # Fallback: auto-generate behaviors_np{npratio}_{rank}.tsv when missing
        if not target_file.exists():
            try:
                lr = int(os.environ.get('LOCAL_RANK', local_rank))
            except Exception:
                lr = local_rank
            if lr == 0:
                print(f"[train] Missing {target_file.name}; generating split files via prepare_distributed_data().")
                prepare_distributed_data(cfg, 'train')
            if dist.is_available() and dist.is_initialized():
                # let non-zero ranks wait for rank-0 to finish generation
                dist.barrier()
            # re-check
            if not target_file.exists():
                raise FileNotFoundError(f"Required file not found after generation attempt: {target_file}")
        if cfg.model.use_graph:
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            if cfg.model.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
            print(f"[{mode}] News Graph Info: {news_graph}")

            news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))

            if cfg.model.use_entity:
                entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            dataset = TrainGraphDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
                neighbor_dict=news_neighbors_dict,
                news_graph=news_graph,
                entity_neighbors=entity_neighbors
            )
            dataloader = DataLoader(dataset, batch_size=None)

        else:
            dataset = TrainDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
            )

            dataloader = DataLoader(dataset,
                                    batch_size=int(cfg.batch_size / cfg.gpu_num),
                                    pin_memory=True)
        return dataloader
    elif mode in ['val', 'test']:
        # convert the news to embeddings (with optional caching)
        cache_mode = getattr(cfg, 'val_embed_mode', 'cache')  # 'recompute' | 'cache'
        cache_path = Path(data_dir[mode]) / 'news_emb_cache.pt'

        news_emb = None
        if cache_mode != 'recompute' and cache_path.exists():
            # Fast path: reuse cached embeddings
            loaded = torch.load(cache_path, map_location='cpu')
            if isinstance(loaded, torch.Tensor):
                news_emb = loaded.numpy()
            else:
                news_emb = np.asarray(loaded)
        else:
            # Compute embeddings once, then optionally save for reuse
            news_dataset = NewsDataset(news_input)
            pw = bool(getattr(cfg, 'num_workers', 0) and cfg.num_workers > 0)
            news_dataloader = DataLoader(
                news_dataset,
                batch_size=int(cfg.batch_size * cfg.gpu_num),
                num_workers=cfg.num_workers,
                pin_memory=True,
                persistent_workers=pw,
            )

            stacked_news = []
            with torch.no_grad():
                for news_batch in tqdm(news_dataloader, desc=f"[{local_rank}] Processing validation News Embedding"):
                    # current pipeline uses the same local encoder regardless of use_graph flag here
                    batch_emb = model.module.local_news_encoder(
                        news_batch.long().unsqueeze(0).to(local_rank)
                    ).squeeze(0).detach()
                    stacked_news.append(batch_emb)
            news_emb_tensor = torch.cat(stacked_news, dim=0).cpu()
            news_emb = news_emb_tensor.numpy()

            # Save cache from rank 0 only to avoid contention
            try:
                if cache_mode != 'recompute' and (int(os.environ.get('LOCAL_RANK', local_rank)) == 0):
                    torch.save(news_emb_tensor, cache_path)
                # Best-effort barrier to let others see the file if DDP is active
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            except Exception:
                pass

        if cfg.model.use_graph:
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))

            if cfg.model.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
            print(f"[{mode}] News Graph Info: {news_graph}")

            if cfg.model.use_entity:
                # entity_graph = torch.load(Path(data_dir[mode]) / "entity_graph.pt")
                entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            if mode in ['val', 'test']:
                news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
                # Use original behaviors.tsv so impressions keep labels (Nxxxx-0/1)
                dataset = ValidGraphDataset(
                    filename=Path(data_dir[mode]) / "behaviors.tsv",
                    news_index=news_index,
                    news_input=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    neighbor_dict=news_neighbors_dict,
                    news_graph=news_graph,
                    news_entity=news_input[:, -8:-3],
                    entity_neighbors=entity_neighbors,
                    news_dict = news_dict
                )

            dataloader = DataLoader(dataset, batch_size=None)

        else:
            if mode == 'val':
                # Use original behaviors.tsv for non-graph validation
                dataset = ValidDataset(
                    filename=Path(data_dir[mode]) / "behaviors.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                )
            else:
                dataset = ValidDataset(
                    filename=Path(data_dir[mode]) / f"behaviors.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                )

            # CPU collate, move-to-device in consumer; enable pinned memory for faster H2D
            _pw = bool(getattr(cfg, 'num_workers', 0) and cfg.num_workers > 0)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=cfg.num_workers,
                pin_memory=True,
                persistent_workers=_pw,
                collate_fn=lambda b: collate_fn(b, local_rank),
            )
        return dataloader


def collate_fn(tuple_list, local_rank):
    """Non-graph validation collate: return CPU torch.Tensors so that
    DataLoader(pin_memory=True) can fully leverage faster H2D copies.

    This function keeps outputs on CPU; the consumer should call
    `.to(device, non_blocking=True)`.
    """
    # unpack
    clicked_news = [x[0] for x in tuple_list]
    clicked_mask = [x[1] for x in tuple_list]
    candidate_news = [x[2] for x in tuple_list]
    clicked_index = [x[3] for x in tuple_list]
    candidate_index = [x[4] for x in tuple_list]
    has_label = (len(tuple_list[0]) == 6)
    labels = [x[5] for x in tuple_list] if has_label else None

    # convert to CPU tensors (float for embeddings/masks, long for indices/labels)
    def _to_float_tensor(items):
        if len(items) == 1:
            return torch.as_tensor(items[0], dtype=torch.float32)
        return torch.stack([torch.as_tensor(it, dtype=torch.float32) for it in items], dim=0)

    def _to_long_tensor(items):
        if len(items) == 1:
            return torch.as_tensor(items[0], dtype=torch.long)
        return torch.stack([torch.as_tensor(it, dtype=torch.long) for it in items], dim=0)

    clicked_news = _to_float_tensor(clicked_news)
    clicked_mask = _to_float_tensor(clicked_mask)
    candidate_news = _to_float_tensor(candidate_news)
    clicked_index = _to_long_tensor(clicked_index)
    candidate_index = _to_long_tensor(candidate_index)
    if has_label:
        labels = _to_long_tensor(labels)

    return (clicked_news, clicked_mask, candidate_news,
            clicked_index, candidate_index, labels) if has_label else (
            clicked_news, clicked_mask, candidate_news, clicked_index, candidate_index)
