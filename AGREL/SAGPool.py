import dgl
import torch
import torch.nn as nn
import torch.nn.functional as f
import dgl
from dgl.nn import GatedGraphConv, GraphConv, AvgPooling, MaxPooling,  RelGraphConv
import dgl.function as fn
import random
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def update_graphid_to_nodeids_after_pooling_optimized(original_graph, graphid_to_nodeids, mask):
    
    pooled_graph = dgl.node_subgraph(original_graph, mask, store_ids=True)
    old_to_new = pooled_graph.ndata[dgl.NID].to(device)
    old_to_new_dict = {old.item(): new for new, old in enumerate(old_to_new)}
    
    graphid_to_nodeids_pooled = {}
    for gid, node_ids in graphid_to_nodeids.items():
        node_ids = node_ids.to(device)
        kept_mask = mask[node_ids]
        kept_original_ids = node_ids[kept_mask]
        new_node_ids = torch.tensor([old_to_new_dict[old.item()] for old in kept_original_ids], device=device)
        graphid_to_nodeids_pooled[gid] = new_node_ids
        
    
    return pooled_graph, graphid_to_nodeids_pooled
class KeepRatioPredictor(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  
        )
    
    def forward(self, num_nodes, avg_score, density):
        features = torch.tensor(
            [num_nodes / 600.0, avg_score, density],  
            dtype=torch.float32,
            device=device  
        )
        return self.mlp(features).squeeze()


def top_k_learned(scores, graph, graphid_to_nodeids,predictor, device='cuda'):
    mask = torch.zeros_like(scores, dtype=torch.bool, device=device)
    for gid, node_ids in graphid_to_nodeids.items():
        node_ids = node_ids.to(device)
        num_nodes = len(node_ids)
        if num_nodes == 0 :
            continue
        
       
        subgraph = graph.subgraph(node_ids)
        density = subgraph.num_edges() / (num_nodes * (num_nodes - 1))
        avg_score = scores[node_ids].mean().item()
       
        keep_ratio = predictor(num_nodes, avg_score, density)
        
        
        keep_num = max(2, int(num_nodes * keep_ratio))
        
        _, topk_idx = torch.topk(scores[node_ids], k=keep_num)
        mask[node_ids[topk_idx]] = True
    
    return mask
class SAGPoolOptimized(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.9, non_linearity=torch.tanh):
        super(SAGPoolOptimized, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.non_linearity = non_linearity
        self.ratio_predictor = KeepRatioPredictor()
       
        self.use_lightweight_attention = True
        
        if self.use_lightweight_attention:
           
            self.score_layer = nn.Sequential(
                nn.Linear(in_channels, in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(in_channels // 4, 1)
            )
        else:
           
            self.score_layer = RelGraphConv(
                in_feat=in_channels, 
                out_feat=1, 
                num_rels=9, 
                regularizer='basis', 
                num_bases=4,  
                activation=f.relu,
                dropout=0.1
           

            )
    
    def forward(self, g, x, e, graphid_to_nodeids):
        
        
        
        if self.use_lightweight_attention:
           
            score = self.score_layer(x).squeeze(-1)
        else:
           
            e = e.to(device)
            score = self.score_layer(g, x, etypes=e).squeeze() 
        
        
        mask = top_k_learned(score, g, graphid_to_nodeids, self.ratio_predictor)
        
       
        score_selected = score[mask]
        x_pooled = x[mask] * self.non_linearity(score_selected).view(-1, 1)
        
       
        pooled_graph, graphid_to_nodeids_pooled = update_graphid_to_nodeids_after_pooling_optimized(
            g, graphid_to_nodeids, mask
        )
        
        return pooled_graph.to(device), x_pooled.to(device), graphid_to_nodeids_pooled

