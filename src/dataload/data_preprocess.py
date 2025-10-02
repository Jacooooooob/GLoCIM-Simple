import collections
import os
from pathlib import Path
from nltk.tokenize import word_tokenize
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from utils.common import load_pretrain_emb
import torch.nn.functional as F
from tqdm import tqdm
import random
import pickle
from collections import Counter
import numpy as np
import torch
import json
import itertools
import logging
import warnings
from collections.abc import Mapping

from models.GLORY import GLORY
from utils.common import load_model
from models.component.news_encoder import *
from models.component.pre_news_encoder import *

logger = logging.getLogger(__name__)




def update_dict(target_dict, key, value=None):
    """
    Function for updating dict with key / key+value

    Args:
        target_dict(dict): target dict
        key(string): target key
        value(Any, optional): if None, equals len(dict+1)
    """

    if key not in target_dict:
        if value is None:
            target_dict[key] = len(target_dict) + 1
        else:
            target_dict[key] = value


def get_sample(all_elements, num_sample):
    if num_sample > len(all_elements):
        return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
    else:
        return random.sample(all_elements, num_sample)


def prepare_distributed_data(cfg, mode="train"): #重新组织排列数据行，并且分发数据到各个gpu上
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    # check
    target_file = os.path.join(data_dir[mode], f"behaviors_np{cfg.npratio}_0.tsv")#npratio == (negative/positive)ratio
    if os.path.exists(target_file) and not cfg.reprocess:
        return 0
    print(f'Target_file does not exist. New behavior file in {target_file}')

    behaviors = []
    behavior_file_path = os.path.join(data_dir[mode], 'behaviors.tsv')

    if mode == 'train':
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                iid, uid, time, history, imp = line.strip().split('\t') #制表符：顾名思义，就是制表，对齐上下
                impressions = [x.split('-') for x in imp.split(' ')]
                pos, neg = [], []
                for news_ID, label in impressions:
                    if label == '0':
                        neg.append(news_ID)
                    elif label == '1':
                        pos.append(news_ID)
                if len(pos) == 0 or len(neg) == 0:
                    continue
                for pos_id in pos:
                    neg_candidate = get_sample(neg, cfg.npratio) #4个负样本 npratio -> negative to positive ratio
                    neg_str = ' '.join(neg_candidate)
                    new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n' # 1 u222 1-1-1 n11 n22 n23 n1111 (n12 n21 n33 n21 n21) ......
                    behaviors.append(new_line)
        random.shuffle(behaviors)

        behaviors_per_file = [[] for _ in range(cfg.gpu_num)] #[[] for _ in range()]
        for i, line in enumerate(behaviors):
            behaviors_per_file[i % cfg.gpu_num].append(line)#把behavior分成了好几份，为接下来分布式训练做准备 1应该换成 cfg.gpu_num

    elif mode in ['val', 'test']:
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                iid, uid, time, history, imp = line.strip().split('\t')  # 制表符：顾名思义，就是制表，对齐上下
                impressions = [x.split('-') for x in imp.split(' ')]
                pos, neg = [], []
                for news_ID, label in impressions:
                    if label == '0':
                        neg.append(news_ID)
                    elif label == '1':
                        pos.append(news_ID)
                if len(pos) == 0 or len(neg) == 0:
                    continue
                for pos_id in pos:
                    neg_candidate = get_sample(neg, cfg.npratio)  # 4个负样本 npratio -> negative to positive ratio
                    neg_str = ' '.join(neg_candidate)
                    new_line = '\t'.join([iid, uid, time, history, pos_id,
                                          neg_str]) + '\n'  # 1 u222 1-1-1 n11 n22 n23 n1111 (n12 n21 n33 n21 n21) ......
                    behaviors.append(new_line)
        random.shuffle(behaviors)

        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]  # [[] for _ in range()]
        for i, line in enumerate(behaviors):
            behaviors_per_file[i % cfg.gpu_num].append(line)  # 把behavior分成了好几份，为接下来分布式训练做准备 1应该换成 cfg.gpu_num

    print(f'[{mode}]Writing files...')
    for i in range(cfg.gpu_num):
        processed_file_path = os.path.join(data_dir[mode], f'behaviors_np{cfg.npratio}_{i}.tsv')
        with open(processed_file_path, 'w') as f:
            f.writelines(behaviors_per_file[i])

    return len(behaviors) #返回行为文件的条数


def read_raw_news(cfg, file_path, mode='train'):
    """
    Function for reading the raw news file, news.tsv

    Args:
        cfg:
        file_path(Path):                path of news.tsv
        mode(string, optional):        train or test


    Returns:
        tuple:     (news, news_index, category_dict, subcategory_dict, word_dict)

    """
    import nltk
    #nltk.download('punkt')
    #1.文本处理和清洁（分词，句子分割，词干提取）
    #2.用于训练和应用各种NLP相关的机器学习模型
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    if mode in ['val', 'test']:
        news_dict = pickle.load(open(Path(data_dir["train"]) / "news_dict.bin", "rb"))
        entity_dict = pickle.load(open(Path(data_dir["train"]) / "entity_dict.bin", "rb"))
        news = pickle.load(open(Path(data_dir["train"]) / "nltk_news.bin", "rb"))
    else:
        news = {}
        news_dict = {}
        entity_dict = {}

    category_dict = {}
    subcategory_dict = {}
    word_cnt = Counter()  # Counter is a subclass of the dictionary dict.

    num_line = len(open(file_path, encoding='utf-8').readlines())
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_line, desc=f"[{mode}]Processing raw news"):
            # split one line
            split_line = line.strip('\n').split('\t')
            news_id, category, subcategory, title, abstract, url, t_entity_str, _ = split_line
            update_dict(target_dict=news_dict, key=news_id) #创建news索引

            # Entity
            if t_entity_str:
                entity_ids = [obj["WikidataId"] for obj in json.loads(t_entity_str)] #loads先将json数据转化成字典
                for entity_id in entity_ids:
                    update_dict(target_dict=entity_dict, key=entity_id)#原来的列表生成式没用啊，所以改成现在这样吧，创建实体索引？？？？？？？？？？？？？？？？？？？？？？？？？？？？？真的没用吗为什么要改呢？？？？？？？？？？？？？？？？？？？？？？？
            else:
                entity_ids = t_entity_str
            
            tokens = word_tokenize(title.lower(), language=cfg.dataset.dataset_lang)

            update_dict(target_dict=news, key=news_id, value=[tokens, category, subcategory, entity_ids,
                                                                news_dict[news_id]])

            if mode == 'train':
                update_dict(target_dict=category_dict, key=category)
                update_dict(target_dict=subcategory_dict, key=subcategory)
                word_cnt.update(tokens)

        if mode == 'train':
            word = [k for k, v in word_cnt.items() if v > cfg.model.word_filter_num] #why set like this?
            word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
            return news, news_dict, category_dict, subcategory_dict, entity_dict, word_dict
        #news:[tokens, category, subcategory, entity_ids,news_dict[news_id]]
        #news_dict:[news_id]
        #news_entity:[entity]
        else:  # val, test
            return news, news_dict, None, None, entity_dict, None


def read_parsed_news(cfg, news, news_dict,
                     category_dict=None, subcategory_dict=None, entity_dict=None,
                     word_dict=None):
    news_num = len(news) + 1
    news_category, news_subcategory, news_index = [np.zeros((news_num, 1), dtype='int32') for _ in range(3)]
    news_entity = np.zeros((news_num, 5), dtype='int32')

    news_title = np.zeros((news_num, cfg.model.title_size), dtype='int32')

    for _news_id in tqdm(news, total=len(news), desc="Processing parsed news"):
        _title, _category, _subcategory, _entity_ids, _news_index = news[_news_id]

        news_category[_news_index, 0] = category_dict[_category] if _category in category_dict else 0
        news_subcategory[_news_index, 0] = subcategory_dict[_subcategory] if _subcategory in subcategory_dict else 0
        news_index[_news_index, 0] = news_dict[_news_id]

        # entity
        entity_index = [entity_dict[entity_id] if entity_id in entity_dict else 0 for entity_id in _entity_ids]
        news_entity[_news_index, :min(cfg.model.entity_size, len(_entity_ids))] = entity_index[:cfg.model.entity_size]
        ##########################3改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        for _word_id in range(min(cfg.model.title_size, len(_title))):
            if _title[_word_id] in word_dict:
                news_title[_news_index, _word_id] = word_dict[_title[_word_id]]

    return news_title, news_entity, news_category, news_subcategory, news_index
#news_input(新闻题目（30），新闻实体（5），新闻种类（1），新闻子种类（1），新闻索引（1）)


def prepare_preprocess_bin(cfg, mode):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    if cfg.reprocess is True:
        # Glove
        nltk_news, nltk_news_dict, category_dict, subcategory_dict, entity_dict, word_dict = read_raw_news(
            file_path=Path(data_dir[mode]) / "news.tsv",
            cfg=cfg,
            mode=mode,
        )
        #print("N55528的索引是：")
        #print(nltk_news_dict["N55528"])
        #return news, news_dict, category_dict, subcategory_dict, entity_dict, word_dict
        if mode == "train":
            pickle.dump(category_dict, open(Path(data_dir[mode]) / "category_dict.bin", "wb"))#
            pickle.dump(subcategory_dict, open(Path(data_dir[mode]) / "subcategory_dict.bin", "wb"))
            pickle.dump(word_dict, open(Path(data_dir[mode]) / "word_dict.bin", "wb"))
        else:
            category_dict = pickle.load(open(Path(data_dir["train"]) / "category_dict.bin", "rb"))
            subcategory_dict = pickle.load(open(Path(data_dir["train"]) / "subcategory_dict.bin", "rb"))
            word_dict = pickle.load(open(Path(data_dir["train"]) / "word_dict.bin", "rb"))

        pickle.dump(entity_dict, open(Path(data_dir[mode]) / "entity_dict.bin", "wb"))
        pickle.dump(nltk_news, open(Path(data_dir[mode]) / "nltk_news.bin", "wb"))
        pickle.dump(nltk_news_dict, open(Path(data_dir[mode]) / "news_dict.bin", "wb"))
        nltk_news_features = read_parsed_news(cfg, nltk_news, nltk_news_dict,
                                              category_dict, subcategory_dict, entity_dict,
                                              word_dict)#word_dict是分词
        news_input = np.concatenate([x for x in nltk_news_features], axis=1)
        pickle.dump(news_input, open(Path(data_dir[mode]) / "nltk_token_news.bin", "wb"))
        print("Glove token preprocess finish.")
    else:
        print(f'[{mode}] All preprocessed files exist.')


def prepare_news_graph(cfg, mode='train'):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}



    nltk_target_path = Path(data_dir[mode]) / "nltk_news_graph.pt"

    reprocess_flag = False
    if nltk_target_path.exists() is False:
        reprocess_flag = True
        
    if (reprocess_flag == False) and (cfg.reprocess == False):
        print(f"[{mode}] All graphs exist !")
        return
    
    # -----------------------------------------News Graph------------------------------------------------
    behavior_path = Path(data_dir['train']) / "behaviors.tsv"
    origin_graph_path = Path(data_dir['train']) / "nltk_news_graph.pt"

    news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    nltk_token_news = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    print("nltk_token_news dimensions:")
    print(nltk_token_news.shape)


    
    # ------------------- Build Graph -------------------------------
    if mode == 'train':
        edge_list, user_set = [], set()
        num_line = len(open(behavior_path, encoding='utf-8').readlines())
        with open(behavior_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=num_line, desc=f"[{mode}] Processing behaviors news to News Graph"): #tqdm是一个进度条
                line = line.strip().split('\t')

                # check duplicate user
                used_id = line[1]
                if used_id in user_set:
                    continue
                else:
                    user_set.add(used_id)

                # record cnt & read path
                history = line[3].split()
                if len(history) > 1:
                    long_edge = [news_dict[news_id] for news_id in history]
                    edge_list.append(long_edge)#邻居的边的索引的集合

        # edge count
        node_feat = nltk_token_news
        target_path = nltk_target_path
        num_nodes = len(news_dict) + 1

        short_edges = []
        for edge in tqdm(edge_list, total=len(edge_list), desc=f"Processing news edge list"):
            # Trajectory Graph
            if cfg.model.use_graph_type == 0:#时序点击图
                for i in range(len(edge) - 1):
                    short_edges.append((edge[i], edge[i + 1]))
                    # short_edges.append((edge[i + 1], edge[i]))
            elif cfg.model.use_graph_type == 1:#全连接图
                # Co-occurence Graph
                for i in range(len(edge) - 1):
                    for j in range(i+1, len(edge)):
                        short_edges.append((edge[i], edge[j]))
                        short_edges.append((edge[j], edge[i]))
            else:
                assert False, "Wrong"

        edge_weights = Counter(short_edges) #用于确定边的数量，以便计算频率
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)

        data = Data(x=torch.from_numpy(node_feat),#节点已经包含了很多信息，有title，entity，category和subcategory，index等
                edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=num_nodes)# Data是PyTorch Geometric 中的类，专门用于处理图数据
    
        torch.save(data, target_path)#存到了nltk_news_graph.pt中去了
        print(data)
        print(f"[{mode}] Finish News Graph Construction, \nGraph Path: {target_path} \nGraph Info: {data}")
    
    elif mode in ['test', 'val']:
        origin_graph = torch.load(origin_graph_path, map_location="cpu", weights_only=False)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr
        node_feat = nltk_token_news

        data = Data(x=torch.from_numpy(node_feat),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(news_dict) + 1)
        
        torch.save(data, nltk_target_path)
        print(f"[{mode}] Finish nltk News Graph Construction, \nGraph Path: {nltk_target_path}\nGraph Info: {data}")


def prepare_neighbor_list(cfg, mode='train', target='news'):

    #--------------------------------Neighbors List-------------------------------------------
    print(f"[{mode}] Start to process neighbors list")

    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    neighbor_dict_path = Path(data_dir[mode]) / f"{target}_neighbor_dict.bin"
    weights_dict_path = Path(data_dir[mode]) / f"{target}_weights_dict.bin"
    category_dict_path = Path(data_dir[mode]) /f"{target}_category_dict.bin"



    if target == 'news':
        target_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path, map_location="cpu", weights_only=False)
        category_info = graph_data.x[:, -3]
        news_index = graph_data.x[:, -1]
        #将tensor转化为list，方便利用pickle转化为二进制数据
        news_index = news_index.tolist()
        category_info = category_info.tolist()
        category_dict = dict(zip(news_index, category_info))

        pickle.dump(category_dict, open(category_dict_path, "wb"))
        reprocess_flag = False
        for file_path in [neighbor_dict_path, weights_dict_path, category_dict_path]:
            if file_path.exists() is False:
                reprocess_flag = True

        if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
            print(f"[{mode}] All {target} Neighbor dict exist !")
            return

    elif target == 'entity':
        target_graph_path = Path(data_dir[mode]) / "entity_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path, map_location="cpu", weights_only=False)

        reprocess_flag = False
        for file_path in [neighbor_dict_path, weights_dict_path]:
            if file_path.exists() is False:
                reprocess_flag = True

        if (reprocess_flag == False) and (cfg.reprocess == False):
            print(f"[{mode}] All {target} Neighbor dict exist !")
            return

    else:
        assert False, f"[{mode}] Wrong target {target} "
    #在处理大量数据或对性能有较高要求的情况下，选择二进制格式通常更有效。这就是为什么在机器学习模型的存储、大数据处理和高性能计算中，二进制格式非常普遍的原因。
    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr
    # print("前10个的图数据为：")
    # print(graph_data.x[0,0])
    # print("图维度是多少呢")
    # print(graph_data.size)
    # print("类别信息为：")
    # print(category_info)
    # print("新闻索引信息为：")
    # print(news_index)
    # print("新闻类别信息为：")
    # print(category_info)
    # print(category_info.size)


    # tensor = torch.load('/home/luoyf/GLORY/category_info.pt')
    # print(tensor)


    # print(category_dict)



    #print(category_dict[0])
    #print(category_dict[1])

    if cfg.model.directed is False:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr) #有向图 -> 无向图

    neighbor_dict = collections.defaultdict(list)
    neighbor_weights_dict = collections.defaultdict(list)
    
    # for each node (except 0)
    for i in tqdm(range(1, len(target_dict) + 1)):
        #print(i)# Using tqdm for progress tracking
        dst_edges = torch.where(edge_index[1] == i)[0]          # i as destination,找出所有以i为目标节点的初始节点
        neighbor_weights = edge_attr[dst_edges]
        neighbor_nodes = edge_index[0][dst_edges]               # neighbors as src
        sorted_weights, indices = torch.sort(neighbor_weights, descending=True)
        neighbor_dict[i] = neighbor_nodes[indices].tolist()#将邻居按权重降序排列
        neighbor_weights_dict[i] = sorted_weights.tolist()#将权重对应排号

    '''edge_index是一个在图数据结构中常用来表示图中所有边的张量（Tensor）。在这段代码中，edge_index是一个维度为2xN的张量，其中N是图中边的数量。
    这个张量的两行分别代表边的源节点和目标节点。每一列的两个元素分别表示一条边的起点（源节点）和终点（目标节点）。'''

    print("Processing completed")
    pickle.dump(neighbor_dict, open(neighbor_dict_path, "wb"))
    print("neighbor_dict的维度是{}".format(len(neighbor_dict)))
    print("neighbor_dict[2]的值是{}".format(neighbor_dict[2]))
    print("neighbor_dict[3]的值是{}".format(neighbor_dict[3]))

    pickle.dump(neighbor_weights_dict, open(weights_dict_path, "wb"))
    print("neighbor_weights_dict的维度是{}".format(len(neighbor_weights_dict)))


    if target == 'news':
        print(f"[{mode}] Finish {target} Neighbor dict \nDict Path: {neighbor_dict_path}, \nWeight Dict: {weights_dict_path},\nCategory Dict: {category_dict_path}")
    if target == 'entity':
        print(
            f"[{mode}] Finish {target} Neighbor dict \nDict Path: {neighbor_dict_path}, \nWeight Dict: {weights_dict_path}")

def prepare_entity_graph(cfg, mode='train'):

    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}


    target_path = Path(data_dir[mode]) / "entity_graph.pt"
    reprocess_flag = False
    if target_path.exists() is False:
        reprocess_flag = True
    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] Entity graph exists!")
        return

    entity_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
    origin_graph_path = Path(data_dir['train']) / "entity_graph.pt"

    if mode == 'train':
        target_news_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        news_graph = torch.load(target_news_graph_path, map_location="cpu", weights_only=False)
        print("news_graph,", news_graph)
        entity_indices = news_graph.x[:, -8:-3].numpy()
        print("entity_indices, ", entity_indices.shape)

        entity_edge_index = []
        # -------- Inter-news -----------------
        # for entity_idx in entity_indices:
        #     entity_idx = entity_idx[entity_idx > 0]
        #     edges = list(itertools.combinations(entity_idx, r=2))
        #     entity_edge_index.extend(edges)

        news_edge_src, news_edge_dest = news_graph.edge_index
        edge_weights = news_graph.edge_attr.long().tolist()
        for i in range(news_edge_src.shape[0]):
            src_entities = entity_indices[news_edge_src[i]]
            dest_entities = entity_indices[news_edge_dest[i]]
            src_entities_mask = src_entities > 0
            dest_entities_mask = dest_entities > 0
            src_entities = src_entities[src_entities_mask]
            dest_entities = dest_entities[dest_entities_mask]
            edges = list(itertools.product(src_entities, dest_entities)) * edge_weights[i]
            entity_edge_index.extend(edges)

        edge_weights = Counter(entity_edge_index)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)

        # --- Entity Graph Undirected
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        data = Data(x=torch.arange(len(entity_dict) + 1),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(entity_dict) + 1)
            
        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")
    elif mode in ['val', 'test']:
        origin_graph = torch.load(origin_graph_path, map_location="cpu", weights_only=False)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr

        data = Data(x=torch.arange(len(entity_dict) + 1),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(entity_dict) + 1)
        
        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")

def prepare_neighbor_vec_list(cfg, mode = "train"):#一个问题是：切分前三个，但是有的邻居不满足三个邻居节点，这个想一下哈；
    #
    with torch.no_grad():
        # 若已存在且未要求重算，则直接跳过
        data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
        out_vec_path = Path(data_dir[mode]) / "news_outputs_dict.pt"
        trimmed_path = Path(data_dir[mode]) / "trimmed_news_neighbors_dict.bin"
        if (out_vec_path.exists() and trimmed_path.exists()) and (not getattr(cfg, "reprocess_neighbors", False)):
            print(f"[{mode}] Neighbor vectors already exist; skip. (set reprocess_neighbors=True to rebuild)")
            return
        #处理邻居字典，准备数据输入
        data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

        news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))
        trimmed_news_neighbors_dict = {}
        #print("news_neighbors_dict的维度是：{}".format(len(news_neighbors_dict)))
        #print("news_neighbors_dict[1]的值是：{}".format(news_neighbors_dict[1]))
        for key, neighbors_list in news_neighbors_dict.items():
            # 前3元素字典
            trimmed_neighbors = neighbors_list[:3]
            # 将裁剪后的列表存储在新的字典中
            trimmed_news_neighbors_dict[key] = trimmed_neighbors


        with open(Path(data_dir[mode]) / "trimmed_news_neighbors_dict.bin", 'wb') as file:
            pickle.dump(trimmed_news_neighbors_dict, file)


        news_input_np = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
        news_input = torch.from_numpy(news_input_np)  # dtype will be converted inside model

        # 引入模型（带缓存的 GloVe 加载）
        word_dict = pickle.load(open(Path(cfg.dataset.train_dir) / "word_dict.bin", "rb"))
        glove_emb = load_pretrain_emb(cfg.path.glove_path, word_dict, cfg.model.word_emb_dim)
        pre_local_news_encoder = PreNewsEncoder(cfg, glove_emb)

        enc_path = Path(data_dir["train"]) / "news_local_news_encoder.pth"
        state_dict = None
        if enc_path.exists():
            try:
                state_dict = torch.load(enc_path, map_location="cpu", weights_only=True)  # 仅用于 state_dict
            except TypeError:
                state_dict = torch.load(enc_path, map_location="cpu")
            logger.info(f"[neighbor_vec] Loaded local encoder from {enc_path}")
        else:
            logger.warning(f"[neighbor_vec] Missing encoder checkpoint: {enc_path}. "
                           "Proceed with randomly-initialized PreNewsEncoder (GloVe-only). "
                           "Neighbor vectors may be lower quality.")

        # 仅当确有权重时再加载；否则保持当前初始化
        if state_dict is not None and not isinstance(state_dict, Mapping):
            try:
                state_dict = state_dict.state_dict()
                logger.info("[neighbor_vec] Converted loaded model object to state_dict()")
            except Exception as e:
                raise TypeError(f"Loaded object is not a state_dict and has no .state_dict(): {type(state_dict)}") from e

        if state_dict is not None:
            missing, unexpected = pre_local_news_encoder.load_state_dict(state_dict, strict=False)
            logger.info(f"[neighbor_vec] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        pre_local_news_encoder.eval()

        # 优化：一次性收集唯一邻居，分批在 GPU/CPU 上前向
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pre_local_news_encoder.to(device)

        # 补齐长度为 3
        for key, neighbors_list in news_neighbors_dict.items():
            trimmed_neighbors = neighbors_list[:3] + [-1] * (3 - len(neighbors_list[:3]))
            trimmed_news_neighbors_dict[key] = trimmed_neighbors

        unique_ids = sorted({idx for lst in trimmed_news_neighbors_dict.values() for idx in lst if idx != -1})
        nid2emb = {}
        zeros400 = torch.zeros(400)

        bs = 4096
        for i in tqdm(range(0, len(unique_ids), bs), desc='批量编码邻居向量'):
            batch_ids = unique_ids[i:i+bs]
            feats = news_input[batch_ids].to(device)
            with torch.no_grad():
                out = pre_local_news_encoder(feats)  # [B, 400]
            out_cpu = out.detach().cpu()
            for j, nid in enumerate(batch_ids):
                nid2emb[nid] = out_cpu[j]

        # 组装为 [N, 3, 400]
        tensors_list = []
        for key, neighbors in trimmed_news_neighbors_dict.items():
            vecs = []
            for nid in neighbors:
                if nid == -1:
                    vecs.append(zeros400)
                else:
                    vecs.append(nid2emb.get(nid, zeros400))
            tensors_list.append(torch.stack(vecs))

        if tensors_list:
            stacked_tensor = torch.stack(tensors_list)
        else:
            print("No data available for stacking")

        # 保存整个字典为一个单一的文件
        torch.save(stacked_tensor, Path(data_dir[mode]) / "news_outputs_dict.pt")
        print("Processing completed successfully")
def prepare_preprocessed_data(cfg):
    '''
    1.分布式数据
    2.二进制文件（类别字典，子类别字典，word字典，实体字典，nltk news，新闻字典，nltk_token_news）
    3.准备新闻图和邻居列表
    4.准备实体图和邻居列表
    '''
    prepare_distributed_data(cfg, "train")#准备分布式的数据
    prepare_distributed_data(cfg, "val")

    prepare_preprocess_bin(cfg, "train")#准备预处理的二进制文件
    prepare_preprocess_bin(cfg, "val")
    prepare_preprocess_bin(cfg, "test")

    prepare_news_graph(cfg, 'train')#新闻图的构建
    prepare_news_graph(cfg, 'val')
    prepare_news_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'news')
    prepare_neighbor_list(cfg, 'val', 'news')
    prepare_neighbor_list(cfg, 'test', 'news')

    # 准备需要的邻居向量：仅在缺失或显式要求重算时执行
    train_vec = Path(cfg.dataset.train_dir) / "news_outputs_dict.pt"
    train_trim = Path(cfg.dataset.train_dir) / "trimmed_news_neighbors_dict.bin"
    val_vec = Path(cfg.dataset.val_dir) / "news_outputs_dict.pt"
    val_trim = Path(cfg.dataset.val_dir) / "trimmed_news_neighbors_dict.bin"

    need_rebuild = getattr(cfg, "reprocess_neighbors", False)
    if need_rebuild or (not train_vec.exists()) or (not train_trim.exists()):
        prepare_neighbor_vec_list(cfg, 'train')
    else:
        print("[train] Neighbor vectors exist; skip.")

    if need_rebuild or (not val_vec.exists()) or (not val_trim.exists()):
        prepare_neighbor_vec_list(cfg, 'val')
    else:
        print("[val] Neighbor vectors exist; skip.")



    prepare_entity_graph(cfg, 'train')
    prepare_entity_graph(cfg, 'val')
    prepare_entity_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'entity')
    prepare_neighbor_list(cfg, 'val', 'entity')
    prepare_neighbor_list(cfg, 'test', 'entity')

    # entity端的筛选向量，还没弄呢，先占个位置
    # prepare_neighbor_list(cfg, 'train', 'entity')
    # prepare_neighbor_list(cfg, 'val', 'entity')
    # prepare_neighbor_list(cfg, 'test', 'entity')

    # Entity vec process
    data_dir = {"train":cfg.dataset.train_dir, "val":cfg.dataset.val_dir, "test":cfg.dataset.test_dir}
    train_entity_emb_path = Path(data_dir['train']) / "entity_embedding.vec"
    val_entity_emb_path = Path(data_dir['val']) / "entity_embedding.vec"
    test_entity_emb_path = Path(data_dir['test']) / "entity_embedding.vec"

    val_combined_path = Path(data_dir['val']) / "combined_entity_embedding.vec"
    test_combined_path = Path(data_dir['test']) / "combined_entity_embedding.vec"

    os.system("cat " + f"{train_entity_emb_path} {val_entity_emb_path}" + f" > {val_combined_path}")
    os.system("cat " + f"{train_entity_emb_path} {test_entity_emb_path}" + f" > {test_combined_path}")

