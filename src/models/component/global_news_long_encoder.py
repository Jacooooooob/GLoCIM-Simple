import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential, GCNConv
from pathlib import Path
import torch.nn.functional as F
import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
import pickle



class GlobalNewsLongEncoder(nn.Module):
    def __init__(self, cfg, glove_emb=None):
        super().__init__()
        self.cfg = cfg
        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        self.his_size = int(getattr(cfg.model, 'his_size', 50))
        self.long_chain_len = int(getattr(cfg.model, 'long_chain_len', 6))

        # cache for neighbors tensor (built from trimmed_news_neighbors_dict)
        self._nbr_cache_key = None
        self._nbr_cache_cpu = None
        self._nbr_cache_tensor = None
        self._nbr_cache_device = None

        # Attention over long-chain expanded items
        self.mha = MultiHeadAttention(self.news_dim, self.news_dim, self.news_dim,
                                      cfg.model.head_num, cfg.model.head_dim)
        self.dropout = nn.Dropout(p=cfg.dropout_probability)
        self.ln_after_flat = nn.LayerNorm(self.news_dim)
        self.pool = AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim)
        self.ln_out = nn.LayerNorm(self.news_dim)

    # 一个索引对应的映射news_input,然后才能注意力是吧,但是有一个文件了其实，那么实际上就是把那个文件加载进来，然后相乘不就行了吗？还是说不行呢？还是作为一个整体呢？okk，那么我选择最简化方法了！先采取最简单的方法，后面不行再加复杂度
    # 先选10个邻居，然后trans合并（注意填充的0向量不参与筛选）
    # mapping_idx不行的哦，需要的是点击新闻的索引哈。
    #######################################neighbor_dict索引差了1！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    def _get_neighbors_tensor(self, trimmed_news_neighbors_dict, N, device):
        key = id(trimmed_news_neighbors_dict)
        if self._nbr_cache_key != key:
            # Build CPU tensor once per dict
            nbr = torch.zeros((N, 3), dtype=torch.long)
            # keys are 1-based; -1 denotes padding, map to 0 here
            for k, lst in trimmed_news_neighbors_dict.items():
                if not lst:
                    continue
                # ensure valid index range
                i = int(k)
                if 1 <= i <= N:
                    vals = [v if v > 0 else 0 for v in lst[:3]]
                    # pad to length 3
                    vals += [0] * (3 - len(vals))
                    nbr[i - 1] = torch.tensor(vals, dtype=torch.long)
            self._nbr_cache_cpu = nbr
            self._nbr_cache_key = key
            self._nbr_cache_tensor = None
            self._nbr_cache_device = None
        if self._nbr_cache_device != device or self._nbr_cache_tensor is None:
            self._nbr_cache_tensor = self._nbr_cache_cpu.to(device)
            self._nbr_cache_device = device
        return self._nbr_cache_tensor

    def forward(self, news_input, click_history, outputs_dict, trimmed_news_neighbors_dict, mask=None):
        """
        news_input        : [B, his, D]
        click_history     : [B, his] (1-based news ids)
        outputs_dict      : [N, 3, D] (neighbor embeddings per news id)
        neighbors mapping : dict[id]->list[3] (1-based neighbor ids, -1 for pad)
        """
        device = news_input.device
        B, his, D = news_input.shape
        L = self.long_chain_len

        # Build or fetch neighbors id tensor [N, 3] (1-based ids; 0 means invalid)
        N = outputs_dict.shape[0]
        neighbors = self._get_neighbors_tensor(trimmed_news_neighbors_dict, N, device)

        # Prepare containers
        results = torch.zeros((B, his, L, D), device=device, dtype=news_input.dtype)

        # Initial indices (1-based ids)
        idx = click_history.to(device=device, dtype=torch.long)
        # Broadcast current vectors
        cur = news_input  # [B, his, D]

        for t in range(L):
            # valid positions have positive ids; others treated as zero vectors and no progress
            valid = idx > 0  # [B, his]
            # clamp to [1, N] then convert to 0-based for indexing outputs_dict/neighbors
            idx0 = idx.clamp(min=1, max=N) - 1  # [B, his]
            # neighbor candidate embeddings: [B, his, 3, D]
            cand = outputs_dict[idx0]
            # zero-out invalid positions to avoid accidental use of id=1 row
            cand = torch.where(valid.unsqueeze(-1).unsqueeze(-1), cand, torch.zeros_like(cand))
            # similarity with current vector: [B, his, 3]
            scores = (cand * cur.unsqueeze(2)).sum(dim=-1)
            max_vals, max_idx = scores.max(dim=-1)  # [B, his]

            # select vectors per (B, his)
            sel = torch.gather(cand, 2, max_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D)).squeeze(2)

            # zero-out when all candidates are zero (max score == 0)
            nz_mask = (max_vals != 0)
            if sel.dtype.is_floating_point:
                sel = torch.where(nz_mask.unsqueeze(-1), sel, torch.zeros_like(sel))

            results[:, :, t, :] = sel

            # next hop indices from neighbor id table [N,3]
            nbr_group = neighbors[idx0]  # [B, his, 3]
            next_idx = torch.gather(nbr_group, 2, max_idx.unsqueeze(-1)).squeeze(-1)  # [B, his], 1-based or 0
            # when nz_mask is False, keep current idx (no progress)
            idx = torch.where(nz_mask, next_idx, idx)

        # pad/truncate to his_size for downstream static reshape
        if his < self.his_size:
            pad_h = self.his_size - his
            # Pad along the 'his' dimension only: (D_l, D_r, L_l, L_r, his_l, his_r, B_l, B_r)
            results = F.pad(results, (0, 0, 0, 0, 0, pad_h, 0, 0), value=0)
            his_eff = self.his_size
        else:
            # if his > his_size, keep only the last his_size
            if his > self.his_size:
                results = results[:, -self.his_size:, :, :]
                his_eff = self.his_size
            else:
                his_eff = his

        # flatten to [B, his_eff*L, D]
        x = results.reshape(B, his_eff * L, D)

        # attention stack (Dropout -> MHA -> reshape to pool per clicked -> LN/Dropout -> Pool -> LN)
        x = self.dropout(x)
        x = self.mha(x, x, x, mask)
        x = x.view(B * his_eff, L, D)
        x = self.ln_after_flat(x)
        x = self.dropout(x)
        # AttentionPooling expects positional attn_mask; avoid unsupported kwarg name
        x = self.pool(x)
        x = self.ln_out(x)

        # back to [B, his_eff, D]
        final_output = x.view(B, his_eff, D)
        return final_output


