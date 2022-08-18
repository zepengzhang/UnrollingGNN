import numpy as np
import os
import torch
import random
import importlib
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, to_networkx

def load_data(args):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', args.dataset)
    # you can modify the hyperparameters for different datasets as suggested in the Appendix
    args.epochs = 1000
    args.patience = 100
    args.dim_hidden = 64
    args.num_layers = 5
    args.lr = 0.005
    args.weight_decay = 0
    args.dropout = 0.8
    if args.dataset in ['ogbn-arxiv']:
        data = PygNodePropPredDataset(args.dataset)
        split_idx = data.get_idx_split()
        args.num_classes = data.num_classes      
        data = data[0]
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)        
        data.train_mask = split_idx['train']
        data.val_mask = split_idx['valid']
        data.test_mask = split_idx['test']
        
    elif args.dataset in ['Cora', 'Citeseer', 'Pubmed']:      
        data = Planetoid(path, args.dataset, split='public', transform=T.NormalizeFeatures())
        args.num_classes = data.num_classes
        data = data[0]
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=data.x.size(0))
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index
            
    elif args.dataset in ['CoauthorCS', 'CoauthorPhysics']:
        data = Coauthor(path, args.dataset[8:], transform=T.NormalizeFeatures())[0]
        args.num_classes = int(max(data.y) + 1)
        indices = []
        for i in range(args.num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
        train_index = torch.cat([i[:20] for i in indices], dim=0)
        val_index = torch.cat([i[20:50] for i in indices], dim=0)
        rest_index = torch.cat([i[50:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        def index_to_mask(index, size):
            mask = torch.zeros(size, dtype=torch.bool, device=index.device)
            mask[index] = 1
            return mask        
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    
    elif args.dataset in ['AmazonComputers', 'AmazonPhoto']:
        data = Amazon(path, args.dataset[6:], transform=T.NormalizeFeatures())[0]
        args.num_classes = int(max(data.y) + 1)               
        data_nx = to_networkx(data)
        data_nx = data_nx.to_undirected()
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        lcc_mask = list(data_nx.nodes)
        indices = []
        for i in range(args.num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
        train_index = torch.cat([i[:20] for i in indices], dim=0)
        val_index = torch.cat([i[20:50] for i in indices], dim=0)
        rest_index = torch.cat([i[50:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        def index_to_mask(index, size):
            mask = torch.zeros(size, dtype=torch.bool, device=index.device)
            mask[index] = 1
            return mask        
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    
    return args, data

def set_seed(repetition):
    ## fix random seed for reproducibility
    seeds_init = [9, 47, 112, 356125, 33324716, 4653645, 74441235, 712345, 7456987, 9090993] 
    seed = seeds_init[repetition]   
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed
    
def get_model(args):
    Model = getattr(importlib.import_module('models'), args.type_model)
    model = Model(args)
    return model