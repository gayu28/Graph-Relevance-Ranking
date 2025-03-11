import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb

from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_geometric.nn import GATConv, HeteroConv
from transformers import AutoModel

from src.config.retriever import load_yaml
from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import GraphRetriever
from src.setup import set_seed, prepare_sample
from src.reranker import AdaptiveLLMReranker

@torch.no_grad()
def eval_epoch(config, device, data_loader, model, reranker):
    model.eval()
    reranker.eval()
    metric_dict = defaultdict(list)
    
    for sample in tqdm(data_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
        num_non_text_entities, relation_embs, topic_entity_one_hot,
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        # Graph-based retrieval
        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot).reshape(-1)
        
        reranked_scores = reranker(pred_triple_logits, q_emb)
        sorted_triple_ids_pred = torch.argsort(reranked_scores, descending=True).cpu()
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(len(triple_ranks_pred))
        
        target_triple_ids = target_triple_probs.nonzero().squeeze(-1)
        num_target_triples = len(target_triple_ids)
        if num_target_triples == 0:
            continue
        
        for k in config['eval']['k_list']:
            recall_k_sample = (triple_ranks_pred[target_triple_ids] < k).sum().item()
            metric_dict[f'triple_recall@{k}'].append(recall_k_sample / num_target_triples)
    
    return {key: np.mean(val) for key, val in metric_dict.items()}

def train_epoch(device, train_loader, model, reranker, optimizer, rerank_optimizer):
    model.train()
    reranker.train()
    epoch_loss = 0
    
    for sample in tqdm(train_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
        num_non_text_entities, relation_embs, topic_entity_one_hot,
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
        
        if len(h_id_tensor) == 0:
            continue
        
        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot)
        
        reranked_scores = reranker(pred_triple_logits, q_emb)
        loss = F.binary_cross_entropy_with_logits(reranked_scores, target_triple_probs.to(device).unsqueeze(-1))
        
        optimizer.zero_grad()
        rerank_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rerank_optimizer.step()
        
        epoch_loss += loss.item()
    
    return {'loss': epoch_loss / len(train_loader)}

def main(args):
    config = load_yaml(f'configs/retriever/{args.dataset}.yaml')
    device = torch.device('cuda:0')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])
    
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    exp_name = f"{config['train']['save_prefix']}_{ts}"
    wandb.init(project=args.dataset, name=exp_name, config=config)
    os.makedirs(exp_name, exist_ok=True)
    
    train_set = RetrieverDataset(config=config, split='train')
    val_set = RetrieverDataset(config=config, split='val')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_retriever)
    
    emb_size = train_set[0]['q_emb'].shape[-1]
    model = GraphRetriever(emb_size, **config['retriever']).to(device)
    reranker = AdaptiveLLMReranker(emb_size).to(device)
    optimizer = Adam(model.parameters(), **config['optimizer'])
    rerank_optimizer = Adam(reranker.parameters(), **config['optimizer'])
    
    best_val_metric = 0
    for epoch in range(config['train']['num_epochs']):
        val_eval_dict = eval_epoch(config, device, val_loader, model, reranker)
        target_val_metric = val_eval_dict['triple_recall@100']
        
        if target_val_metric > best_val_metric:
            best_val_metric = target_val_metric
            best_state_dict = {
                'config': config,
                'model_state_dict': model.state_dict(),
                'reranker_state_dict': reranker.state_dict()
            }
            torch.save(best_state_dict, os.path.join(exp_name, 'best_model.pth'))
        
        train_log_dict = train_epoch(device, train_loader, model, reranker, optimizer, rerank_optimizer)
        wandb.log(train_log_dict)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['webqsp', 'cwq'], help='Dataset name')
    args = parser.parse_args()
    main(args)
