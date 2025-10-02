import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import torch
import torch.nn.functional as F

# Suppress sklearn AUC warnings for single-class y_true; we handle it safely below.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def top1_accuracy(y_true, y_hat):
    """Compute top-1 accuracy over candidate set.

    y_true: LongTensor [...], containing the index of the positive item
    y_hat:  FloatTensor [..., num_candidates], model scores per candidate
    """
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot

# Backward-compat alias (historically misnamed as AUC during training logs)
def area_under_curve(y_true, y_hat):
    return top1_accuracy(y_true, y_hat)


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2.0 ** y_true - 1.0
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return float(np.sum(gains / discounts))


def ndcg_score(y_true, y_score, k=10):
    """Safe nDCG@k. Returns 0.0 when ideal DCG is 0 to avoid NaN."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.size == 0 or y_true.sum() == 0:
        return 0.0
    best = dcg_score(y_true, y_true, k)
    if best <= 0.0 or not np.isfinite(best):
        return 0.0
    actual = dcg_score(y_true, y_score, k)
    return float(actual / best)


def mrr_score(y_true, y_score):
    """Safe MRR. When no positives, returns 0.0 to avoid division by zero.

    This follows the original implementation's convention:
    average of reciprocals of ranks for all positives.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order)
    denom = int(np.sum(y_true_sorted))
    if denom <= 0:
        return 0.0
    rr_score = y_true_sorted / (np.arange(len(y_true_sorted)) + 1)
    return float(np.sum(rr_score) / denom)


def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)


def safe_auc(y_true, y_score):
    """Safe ROC-AUC. Returns NaN when y_true has a single class."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.size == 0 or y_true.min() == y_true.max():
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def cal_metric(pair):
    """Returns a tuple (auc, mrr, ndcg5, ndcg10) with safe computations."""
    y_true, y_score = pair
    auc = safe_auc(y_true, y_score)
    mrr = mrr_score(y_true, y_score)
    ndcg5 = ndcg_score(y_true, y_score, 5)
    ndcg10 = ndcg_score(y_true, y_score, 10)
    return auc, mrr, ndcg5, ndcg10


# Diversity Metrics

def ILAD(vecs):
    # similarity = torch.mm(vecs, vecs.T)
    # similarity = cosine_similarity(X=vecs)
    similarity = F.cosine_similarity(vecs.unsqueeze(dim=0), vecs.unsqueeze(dim=1))
    distance = (1 - similarity)/2
    score = distance.mean()-1/distance.shape[0]
    return score.item()


def ILMD(vecs):
    # similarity = torch.mm(vecs, vecs.T)
    # similarity = cosine_similarity(X=vecs)
    similarity = F.cosine_similarity(vecs.unsqueeze(dim=0), vecs.unsqueeze(dim=1))
    distance = (1 - similarity) / 2
    score = distance.min()
    return score.item()

def density_ILxD(scores, news_emb, top_k=5):
    """
    Args:
        scores: [batch_size, y_pred_score]
        news_emb: [batch_size, news_num, news_emb_size]
        top_k: integer, n=5, n=10
    """
    top_ids = torch.argsort(scores)[-top_k:]
    news_emb =  news_emb / torch.sqrt(torch.square(news_emb).sum(dim=-1)).reshape((len(news_emb), 1))
    # nv: (top_k, news_emb_size)
    nv = news_emb[top_ids]
    ilad = ILAD(nv)
    ilmd = ILMD(nv)

    return ilad, ilmd




