"""
Common utils and tools.
"""
import pickle
import random

import pandas as pd
import torch
import numpy as np
import pyrootutils
from pathlib import Path
import hashlib
import torch.distributed as dist

import importlib
from omegaconf import DictConfig, ListConfig


def seed_everything(seed):
    torch.manual_seed(seed)#一些参数随机性被记录：模型初始权重值，dropout设置种子，样本的随机采样
    torch.cuda.manual_seed_all(seed) #GPU上的随机权重初始化都是相同的，确保不同GPU上的计算都使用相同的随机种子，以确保一致性。
    torch.backends.cudnn.deterministic = True #用于加速卷积的一个深度学习库
    torch.backends.cudnn.benchmark = False#这一行代码关闭了CuDNN的自动优化，以确保运行时不会尝试通过使用不同的算法来优化性能，从而保持可重复性
    random.seed(seed) #Python 标准库的生成器
    np.random.seed(seed) #设置了 NumPy 库中的随机数生成器的种子

def load_model(cfg):
    framework = getattr(importlib.import_module(f"models.{cfg.model.model_name}"), cfg.model.model_name)

    if cfg.model.use_entity:
        entity_dict = pickle.load(open(Path(cfg.dataset.val_dir) / "entity_dict.bin", "rb"))
        entity_emb_path = Path(cfg.dataset.val_dir) / "combined_entity_embedding.vec"
        entity_emb = load_pretrain_emb(entity_emb_path, entity_dict, 100)
    else:
        entity_emb = None

    if cfg.dataset.dataset_lang == 'english':
        word_dict = pickle.load(open(Path(cfg.dataset.train_dir) / "word_dict.bin", "rb"))
        glove_emb = load_pretrain_emb(cfg.path.glove_path, word_dict, cfg.model.word_emb_dim)
    else:
        word_dict = pickle.load(open(Path(cfg.dataset.train_dir) / "word_dict.bin", "rb"))
        glove_emb = len(word_dict)
    model = framework(cfg, glove_emb=glove_emb, entity_emb=entity_emb)

    return model


def save_model(cfg, model, optimizer=None, mark=None):
    file_path = Path(f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_{mark}.pth")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        },
        file_path)
    print(f"Model Saved. Path = {file_path}")


def load_pretrain_emb(embedding_file_path, target_dict, target_dim):
    """Load pre-trained embeddings with persistent on-disk caching.

    - Builds an embedding matrix of shape (|dict|+1, dim), index-aligned with target_dict values.
    - Caches per (embedding basename, dim, |dict|, dict-key-hash) under outputs/emb_cache.
    """
    vocab_size = len(target_dict)
    # stable hash for dict keys (order-independent)
    try:
        keys_sorted = sorted(target_dict.keys())
    except Exception:
        keys_sorted = list(target_dict.keys())
    h = hashlib.md5("|".join(map(str, keys_sorted)).encode("utf-8")).hexdigest()[:8]

    root = get_root()
    cache_dir = Path(root) / "outputs" / "emb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = Path(embedding_file_path).name if embedding_file_path is not None else "none"
    cache_path = cache_dir / f"{base}.dim{target_dim}.v{vocab_size}.{h}.npy"

    if cache_path.exists():
        embedding_matrix = np.load(cache_path)
        # quick stats from cached matrix
        nonzeros = int(np.count_nonzero(np.linalg.norm(embedding_matrix[1:], axis=1))) if embedding_matrix.shape[0] > 1 else 0
        miss_rate = (vocab_size - nonzeros) / vocab_size if vocab_size != 0 else 0
        print('-----------------------------------------------------')
        print(f'Dict length: {vocab_size}')
        print(f'Have words: {nonzeros} (cached)')
        print(f'Missing rate: {miss_rate}')
        return embedding_matrix

    embedding_matrix = np.zeros(shape=(vocab_size + 1, target_dim))
    have_item = []
    if embedding_file_path is not None:
        with open(embedding_file_path, 'rb') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                itme = line[0].decode(errors='ignore')
                if itme in target_dict:
                    index = target_dict[itme]
                    try:
                        tp = [float(x) for x in line[1:]]
                    except Exception:
                        continue
                    if len(tp) >= target_dim:
                        embedding_matrix[index] = np.array(tp[:target_dim])
                    else:
                        # pad short vectors if any (robustness)
                        vec = np.zeros((target_dim,), dtype=float)
                        vec[:len(tp)] = np.array(tp)
                        embedding_matrix[index] = vec
                    have_item.append(itme)
    # save cache
    try:
        np.save(cache_path, embedding_matrix)
    except Exception:
        pass
    print('-----------------------------------------------------')
    print(f'Dict length: {vocab_size}')
    print(f'Have words: {len(have_item)}')
    miss_rate = (vocab_size - len(have_item)) / vocab_size if vocab_size != 0 else 0
    print(f'Missing rate: {miss_rate}')
    return embedding_matrix


def reduce_mean(result, nprocs):
    rt = result.detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def pretty_print(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key)+ '\t' + str(value))


def get_root():
    return pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "README.md"],
        pythonpath=True,
        dotenv=True,
    )

def print_model_memory_usage(model):
    print("Model's state_dict:")
    total_params = 0
    for param_tensor in model.state_dict():
        # 获取每个参数的大小（元素总数）
        num_params = model.state_dict()[param_tensor].numel()
        # 每个元素占用的内存（以float32为例，每个元素占4字节）
        param_size = num_params * 4
        total_params += param_size
        print(f"{param_tensor} has {num_params} params: {param_size} bytes")
    print(f"Total memory for model parameters: {total_params} bytes")



class EarlyStopping:
    """Early Stopping class (robust to NaN).

    - Ignores NaN scores (does not increase patience counter).
    - Uses -inf as initial best score for 'max' mode semantics.
    """

    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = -float("inf")

    def __call__(self, score):
        """The greater score, the better result. Be careful the symbol."""
        # Ignore NaN scores: neither early stop nor get_better; don't count patience
        try:
            if score is None or (isinstance(score, float) and np.isnan(score)):
                return False, False
        except Exception:
            pass

        if score > self.best_score:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_score = score
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better
