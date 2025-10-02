import os.path
import logging
from pathlib import Path

import hydra
import math
import wandb
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import pickle

from dataload.data_load import load_data
from dataload.data_preprocess import prepare_preprocessed_data, prepare_neighbor_vec_list
from utils.metrics import *
from utils.common import *

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"
# 不强行改写可见设备，避免误屏蔽 GPU；如需指定，请在外部设置

def train(model, optimizer, scaler, scheduler, dataloader, local_rank, cfg, early_stopping):
    model.train()
    torch.set_grad_enabled(True)
    mode = "train"
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    first = True
    updated = False


    # print("开始预处理阶段")
    # glory_state_dict = model.module.local_news_encoder.state_dict()
    # data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    # torch.save(glory_state_dict, Path(data_dir["train"]) / "news_local_news_encoder.pth")
    # print("预处理完成")

    #加载
    with open(Path(data_dir[mode]) / "news_outputs_dict.pt", 'rb') as bf:
        outputs_dict = torch.load(bf, map_location=torch.device(f'cuda:{local_rank}'))

    with open(Path(data_dir[mode]) / "trimmed_news_neighbors_dict.bin", 'rb') as file:
        trimmed_news_neighbors_dict = pickle.load(file)

    sum_loss = torch.zeros(1).to(local_rank)
    sum_acc = torch.zeros(1).to(local_rank)



    for cnt, (subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, labels) \
            in enumerate(tqdm(dataloader,
                              total=int(cfg.num_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)), #为了先测试成功，把batch设置为1，原语句是total=int(cfg.num_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)
                              desc=f"[{local_rank}] Training"), start=1):

        #测一下处理1000次的时间和内存
        # if cnt == 31:
        #     print("处理完成，检查时间和内存占用")
        #     break;
        if cnt > int(cfg.num_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)):
            print("完成{}个epoch的训练了".format(cfg.num_epochs))
            break
        subgraph = subgraph.to(local_rank, non_blocking=True) #将子图部署到特定的GPU上
        mapping_idx = mapping_idx.to(local_rank, non_blocking=True)
        candidate_news = candidate_news.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)
        candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
        entity_mask = entity_mask.to(local_rank, non_blocking=True)

        if(updated == True):
            with open(Path(data_dir[mode]) / "news_outputs_dict.pt", 'rb') as bf:
                outputs_dict = torch.load(bf, map_location=torch.device(f'cuda:{local_rank}'))

            file_path = Path(data_dir[mode]) / "trimmed_news_neighbors_dict.bin"
            with open(file_path, 'rb') as file:
                trimmed_news_neighbors_dict = pickle.load(file)
            updated = False

        with torch.amp.autocast('cuda'):#自动混合精度训练的一部分，可以提高训练速度和效率。它会自动将某些操作从单精度（float32）转换为半精度（float16），这样做的好处是可以加快计算速度，减少内存使用
            bz_loss, y_hat = model(subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask,outputs_dict,trimmed_news_neighbors_dict, labels)


        # Accumulate the gradients，不知道咋加速的，黑盒呗。
        scaler.scale(bz_loss).backward()

        if cnt % cfg.accumulation_steps == 0 or cnt == int(cfg.dataset.pos_count / cfg.batch_size):
            # Update the parameters
            scaler.step(optimizer)
            old_scaler = scaler.get_scale()
            scaler.update()
            new_scaler = scaler.get_scale()
            if new_scaler >= old_scaler:
                scheduler.step()
                ## https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/164814
            optimizer.zero_grad(set_to_none=True)



        sum_loss += bz_loss.data.float()
        sum_acc += top1_accuracy(labels, y_hat)
        if cnt == 10:
            print(torch.cuda.get_device_properties(0))  # 显示第一个GPU的属性
            print(torch.cuda.memory_allocated(0))  # 显示第一个GPU的已分配内存
            print(torch.cuda.memory_cached(0))  # 显示第一个GPU的缓存内存

        # ----------------重新训练筛选邻居节点（默认关闭，仅当 reprocess_neighbors=True 才执行）----------------------
        if (cnt == 3000 or cnt == 6000 or cnt == 9000) and getattr(cfg, "reprocess_neighbors", False):
            print("又开始预处理咯，祝我好运")
            glory_state_dict = model.module.local_news_encoder.state_dict()
            data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
            torch.save(glory_state_dict, Path(data_dir["train"]) / "news_local_news_encoder.pth")
            prepare_neighbor_vec_list(cfg, 'train')
            updated = True



        # -------------------训练集日志---------------------
        if cnt % cfg.log_steps == 0: #输出训练数据      总共36930次，5个epoch的话
            if local_rank == 0:
                wandb.log({"train_loss": sum_loss.item() / cfg.log_steps, "train_acc": sum_acc.item() / cfg.log_steps})
            print('[{}] Ed: {}, average_loss: {:.5f}, average_acc: {:.5f}'.format(
                 local_rank, cnt * cfg.batch_size, sum_loss.item() / cfg.log_steps, sum_acc.item() / cfg.log_steps))
            #if best_acc < sum_acc.item()/ cfg.log_steps:
                    #beyond = True
                    #best_acc = sum_acc.item() / cfg.log_steps
                    #best_loss = sum_loss.item() / cfg.log_steps
            if math.isnan(sum_loss.item() / cfg.log_steps):
                print("检测到NaN损失值，终止训练")
                break

            sum_loss.zero_()
            sum_acc.zero_()

        # #保存训练集的东东
        # x = cfg.dataset.pos_count // cfg.batch_size + 1
        # if cnt > x and beyond:
        #     print("------------------------------------------------")
        #     print(f"Better Result!")
        #     print("best_auc为：{}，best_loss为：{}".format(best_auc,best_loss))
        #     print("------------------------------------------------")
        #     if local_rank == 0:
        #         save_model(cfg, model, optimizer, f"{cfg.ml_label}_auc{best_auc}")
        #     beyond = False

        #if cnt > int(cfg.val_skip_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)) and cnt % cfg.val_steps == 0:

        # 测试/验证触发（可配置开关与起始步）
        # 允许 bool 或字符串（'off'/'false' 关闭）
        _vm = getattr(cfg, 'val_mode', True)
        if isinstance(_vm, str):
            val_enabled = _vm.strip().lower() not in ('off', 'false', '0', 'no')
        else:
            val_enabled = bool(_vm)
        val_start = int(getattr(cfg, 'val_start_step', 8000))
        if val_enabled and (cnt % cfg.val_steps == 0) and (cnt > val_start):
            if first is True:
                # 仅在需要时生成验证集邻居向量
                val_out = Path(data_dir["val"]) / "news_outputs_dict.pt"
                val_trim = Path(data_dir["val"]) / "trimmed_news_neighbors_dict.bin"
                if getattr(cfg, "reprocess_neighbors", False) or (not val_out.exists()) or (not val_trim.exists()):
                    prepare_neighbor_vec_list(cfg, 'val')
                first = False
            res = val(model, local_rank, cfg)
            model.train()
            print("--------------------验证结果-----------------------")
            if local_rank == 0:
                pretty_print(res)
                # 全 NaN 时跳过 wandb.log，避免污染曲线
                vals = [res.get('auc'), res.get('mrr'), res.get('ndcg5'), res.get('ndcg10')]
                try:
                    has_finite = any(np.isfinite(v) for v in vals)
                except Exception:
                    has_finite = True
                if has_finite:
                    wandb.log(res)
                else:
                    print("[val] all metrics NaN; skip wandb.log", flush=True)
            print("--------------------模型验证检查点--------------------")
            early_stop, get_better = early_stopping(res['auc'])

            if early_stop:
                print("Early Stop.")
                break
            elif get_better:
                print(f"Better Result!")
                if local_rank == 0:
                    save_model(cfg, model, optimizer, f"{cfg.ml_label}_auc{res['auc']}")
                    wandb.run.summary.update({"best_auc": res["auc"], "best_mrr": res['mrr'],
                                              "best_ndcg5": res['ndcg5'], "best_ndcg10": res['ndcg10']})



def val(model, local_rank, cfg):

    model.eval()

    mode = "val"
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    # 是否在验证阶段使用离线邻居注入
    use_offline = bool(getattr(cfg, 'val_use_offline_neighbors', False))
    outputs_dict = None
    trimmed_news_neighbors_dict = None
    if use_offline:
        # 仅当启用时加载离线邻居向量
        with open(Path(data_dir[mode]) / "news_outputs_dict.pt", 'rb') as bf:
            outputs_dict = torch.load(bf, map_location=torch.device(f'cuda:{local_rank}'))
        file_path = Path(data_dir[mode]) / "trimmed_news_neighbors_dict.bin"
        with open(file_path, 'rb') as file:
            trimmed_news_neighbors_dict = pickle.load(file)


    dataloader = load_data(cfg, mode='val', model=model, local_rank=local_rank)


    #sample_ratio = 0.005
    tasks = []
    skipped_zero_pos = 0
    with torch.no_grad():
        # for cnt, (subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, labels, click_history) \
        #         in enumerate(tqdm(dataloader,
        #                           total=int(cfg.dataset.val_len / cfg.gpu_num),
        #                           desc=f"[{local_rank}] Validating")):
        #     # cnt += 1
        # zzy
        sample_ratio = float(getattr(cfg, 'val_sample_ratio', 1.0))
        for cnt, batch in enumerate(tqdm(dataloader,
                                         total=int(cfg.dataset.val_len / cfg.gpu_num),
                                         desc=f"[{local_rank}] Validating")):
            # ---- 最小兜底：batch 结构异常或为空就跳过，避免 “expected 8, got 0” 崩溃 ----
            try:
                blen = len(batch)
            except Exception:
                blen = "NA"
            if not isinstance(batch, (list, tuple)) or blen != 8:
                print(f"[VAL-SKIP] unexpected batch at #{cnt}: "
                      f"type={type(batch).__name__}, len={blen}", flush=True)
                continue

            # 采样验证以加速（可配置，默认 1.0 表示不过滤）
            if sample_ratio < 1.0:
                try:
                    if np.random.random() > sample_ratio:
                        continue
                except Exception:
                    pass

            subgraph, mappings, clicked_entity, candidate_input, \
            candidate_entity, entity_mask, labels, click_history = batch

            # 避免额外拷贝：直接从 numpy 构造，再转 float
            if isinstance(candidate_input, np.ndarray):
                candidate_emb = torch.from_numpy(candidate_input).float().to(local_rank, non_blocking=True)
            else:
                candidate_emb = torch.as_tensor(candidate_input, dtype=torch.float32).to(local_rank, non_blocking=True)
            candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
            entity_mask = entity_mask.to(local_rank, non_blocking=True)
            clicked_entity = clicked_entity.to(local_rank, non_blocking=True)

            # 可配置 AMP 验证（默认开启）
            _use_amp = bool(getattr(cfg, 'val_amp', True))
            if _use_amp:
                with torch.amp.autocast('cuda'):
                    scores = model.module.validation_process(
                        subgraph, mappings, clicked_entity, candidate_emb,
                        candidate_entity, entity_mask, outputs_dict,
                        trimmed_news_neighbors_dict, click_history)
            else:
                scores = model.module.validation_process(
                    subgraph, mappings, clicked_entity, candidate_emb,
                    candidate_entity, entity_mask, outputs_dict,
                    trimmed_news_neighbors_dict, click_history)
            if torch.is_tensor(scores):
                scores = scores.detach().float().cpu().numpy()
            else:
                scores = np.asarray(scores, dtype=float)

            if torch.is_tensor(labels):
                labels_arr = labels.detach().cpu().numpy().astype(int)
            else:
                labels_arr = np.asarray(labels, dtype=int)

            # 跳过单类别（全0或全1）的验证单元，避免 AUC/MRR/nDCG 出现 NaN
            pos_cnt = int(np.sum(labels_arr))
            if pos_cnt == 0 or pos_cnt == len(labels_arr):
                skipped_zero_pos += 1
                continue

            tasks.append((labels_arr, scores))

    # 开启线程池，把计算任务分发
    if len(tasks) > 0:
        with mp.Pool(processes=cfg.num_workers) as pool:
            results = pool.map(cal_metric, tasks)
        if len(results) > 0:
            val_auc, val_mrr, val_ndcg5, val_ndcg10 = np.array(results, dtype=float).T
        else:
            val_auc = val_mrr = val_ndcg5 = val_ndcg10 = np.array([])
    else:
        val_auc = val_mrr = val_ndcg5 = val_ndcg10 = np.array([])

    # barrier
    torch.distributed.barrier()#同步所有的计算节点

    # 平均值（全 NaN 时返回 NaN；让 EarlyStopping 忽略该 epoch）
    def _nanmean_or_nan(values, name):
        try:
            m = float(np.nanmean(values)) if len(values) else float("nan")
        except Exception:
            m = float("nan")
        if not np.isfinite(m):
            logging.warning(f"[val] {name} is NaN this epoch (no valid groups)")
            return float("nan")
        return m

    auc_mean = _nanmean_or_nan(val_auc, "auc")
    mrr_mean = _nanmean_or_nan(val_mrr, "mrr")
    ndcg5_mean = _nanmean_or_nan(val_ndcg5, "ndcg5")
    ndcg10_mean = _nanmean_or_nan(val_ndcg10, "ndcg10")

    reduced_auc = reduce_mean(torch.tensor(auc_mean).float().to(local_rank), cfg.gpu_num)
    reduced_mrr = reduce_mean(torch.tensor(mrr_mean).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg5 = reduce_mean(torch.tensor(ndcg5_mean).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg10 = reduce_mean(torch.tensor(ndcg10_mean).float().to(local_rank), cfg.gpu_num)

    logging.info(f"[val] skipped groups (single-class): {skipped_zero_pos}")

    res = {
        "auc": float(reduced_auc.item()),
        "mrr": float(reduced_mrr.item()),
        "ndcg5": float(reduced_ndcg5.item()),
        "ndcg10": float(reduced_ndcg10.item()),
    }

    return res

def cleanup_dist():
    """Best-effort DDP cleanup to avoid NCCL warnings on early exit."""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass

#有助于多进程任务，使用多个gpu同时训练时会遇到
def main_worker(local_rank, cfg):
    # -----------------------------------------Environment Initial
    seed_everything(cfg.seed)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:42658',
                            world_size=cfg.gpu_num,
                            rank=local_rank)
    try:
        # -----------------------------------------Dataset & Model Load
        num_training_steps = int(cfg.num_epochs * cfg.dataset.pos_count / (cfg.batch_size * cfg.accumulation_steps))#accumulation的作用：积累梯度，如果=2，那么说明每两个batch更新一次梯度
        num_warmup_steps = int(num_training_steps * cfg.warmup_ratio + 1)#3x236344/32x1 = 22157     #2215
        train_dataloader = load_data(cfg, mode='train', local_rank=local_rank)
        #dataloader加载完毕
        model: object = load_model(cfg).to(local_rank) #模型上显卡
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

        lr_lambda = lambda step: 0.2 if step > num_warmup_steps else step / num_warmup_steps * 0.2 #学习率进行线性增加 ???
        scheduler = LambdaLR(optimizer, lr_lambda)#调度器，根据上述的学习率增加逻辑

        # ------------------------------------------Load Checkpoint & optimizer
        if cfg.load_checkpoint:
            file_path = Path(f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_default_auc0.6790186762809753.pth")
            print(file_path)
            print("--------------------------")
            checkpoint = torch.load(file_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])  # After Distributed strict取消是因为消融实验呢
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) #分布式

        ##########
        optimizer.zero_grad(set_to_none=True) #梯度清0
        scaler = torch.cuda.amp.GradScaler()  #利用混和精度减少GPU计算的手段

        # ------------------------------------------Main Start
        early_stopping = EarlyStopping(cfg.early_stop_patience) ##模型的性能在5个连续的周期内都没改进，则训练就会停止

        #用于跟踪机器学习实验，记录指标，输出，模型权重等
        if local_rank == 0:
            wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                       project=cfg.logger.exp_name, name=cfg.logger.run_name)
            print(model)

        # for _ in tqdm(range(1, cfg.num_epochs + 1), desc="Epoch"):
        train(model, optimizer, scaler, scheduler, train_dataloader, local_rank, cfg, early_stopping)
        #scaler用于梯度缩放
        #scheduler用于学习率调节

        if local_rank == 0:
            wandb.finish()
    finally:
        cleanup_dist()


@hydra.main(version_base="1.2", config_path=os.path.join(get_root(), "configs"), config_name="small")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)#固定随机种子
    cfg.gpu_num = torch.cuda.device_count()
    prepare_preprocessed_data(cfg)
    print("开始训练！")
    mp.spawn(main_worker, nprocs=cfg.gpu_num, args=(cfg,)) #mp.spawn主要用于多进程处理


if __name__ == "__main__":
    main()

