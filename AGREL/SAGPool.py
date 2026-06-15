import dgl
import torch
import torch.nn as nn
import torch.nn.functional as f
import dgl
from dgl.nn import GatedGraphConv, GraphConv, AvgPooling, MaxPooling,  RelGraphConv
import dgl.function as fn
import random
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn
import torch.nn.functional as F


class KeepRatioPredictor(nn.Module):
    """根据 (节点数, 平均分, 密度) 预测 keep_ratio ∈ (0, 1)。"""
    def __init__(self, hidden_dim=64, min_ratio=0.1, max_ratio=0.95):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, num_nodes, avg_score, density):
      
        device = avg_score.device
        n = torch.tensor(num_nodes / 600.0, dtype=torch.float32, device=device)
        d = torch.tensor(float(density), dtype=torch.float32, device=device)
        feats = torch.stack([n, avg_score, d])             
        raw = torch.sigmoid(self.mlp(feats).squeeze())
     
        return self.min_ratio + (self.max_ratio - self.min_ratio) * raw


def differentiable_top_k(scores, keep_ratio, min_keep=2, temperature=0.5):
    """
    返回:
      hard_mask  : bool,  前向真正使用的选择
      soft_gate  : float, 仅在 hard_mask 位置上有意义的 sigmoid 门控,反向带梯度
    """
    n = scores.shape[0]
    device = scores.device

  
    k_cont = torch.clamp(n * keep_ratio, min=float(min_keep), max=float(n))
    k = int(k_cont.detach().item())
    k = max(min_keep, min(k, n))

   
    sorted_s, _ = torch.sort(scores, descending=True)
    pos = torch.clamp(k_cont - 1.0, 0.0, float(n - 1) - 1e-3)
    pos_floor = int(pos.detach().item())
    pos_ceil = min(pos_floor + 1, n - 1)
    frac = pos - pos_floor                               
    threshold = sorted_s[pos_floor] * (1.0 - frac) + sorted_s[pos_ceil] * frac

    
    _, topk_idx = torch.topk(scores, k=k)
    hard_mask = torch.zeros(n, dtype=torch.bool, device=device)
    hard_mask[topk_idx] = True


    soft_gate = torch.sigmoid((scores - threshold) / temperature)
    return hard_mask, soft_gate


def top_k_learned(scores, graph, graphid_to_nodeids, predictor, device='cuda',
                  min_keep=2, temperature=0.5):
    """
    返回:
      hard_mask  : [N] bool
      soft_gate  : [N] float,与 hard_mask 同形状
      keep_ratios: [num_graphs] tensor,供外部做正则
    """
    N = scores.shape[0]
    hard_mask = torch.zeros(N, dtype=torch.bool, device=device)
    soft_gate = torch.zeros(N, dtype=scores.dtype, device=device)
    keep_ratios = []

    for gid, node_ids in graphid_to_nodeids.items():
        node_ids = node_ids.to(device)
        num_nodes = node_ids.numel()
        if num_nodes == 0:
            continue
        if num_nodes <= min_keep:
          
            hard_mask[node_ids] = True
            soft_gate[node_ids] = 1.0
            continue

        sub_scores = scores[node_ids]

     
        subgraph = graph.subgraph(node_ids)
        denom = max(num_nodes * (num_nodes - 1), 1)
        density = subgraph.num_edges() / denom
        avg_score = sub_scores.mean()                    

        keep_ratio = predictor(num_nodes, avg_score, density)
        keep_ratios.append(keep_ratio)

        sub_hard, sub_soft = differentiable_top_k(
            sub_scores, keep_ratio,
            min_keep=min_keep, temperature=temperature,
        )

        hard_mask[node_ids] = sub_hard
        soft_gate[node_ids] = sub_soft

    keep_ratios = (torch.stack(keep_ratios)
                   if len(keep_ratios) > 0
                   else torch.zeros(0, device=device))
    return hard_mask, soft_gate, keep_ratios


class SAGPoolOptimized(nn.Module):
    def __init__(self, in_channels, ratio=0.9, non_linearity=torch.tanh,
                 temperature=0.5, min_keep=2,
                 ratio_reg_weight=0.01, ratio_reg_target=0.6):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio                                 
        self.non_linearity = non_linearity
        self.temperature = temperature
        self.min_keep = min_keep
        self.ratio_reg_weight = ratio_reg_weight
        self.ratio_reg_target = ratio_reg_target

        self.ratio_predictor = KeepRatioPredictor()
        self.use_lightweight_attention = True

        if self.use_lightweight_attention:
            self.score_layer = nn.Sequential(
                nn.Linear(in_channels, in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(in_channels // 4, 1),
            )
        else:
            from dgl.nn import RelGraphConv
            self.score_layer = RelGraphConv(
                in_feat=in_channels, out_feat=1, num_rels=9,
                regularizer='basis', num_bases=4,
                activation=F.relu, dropout=0.1,
            )

      
        self.last_ratio_reg = torch.tensor(0.0)

    def forward(self, g, x, e, graphid_to_nodeids):
        device = x.device

       
        if self.use_lightweight_attention:
            score = self.score_layer(x).squeeze(-1)
        else:
            e = e.to(device)
            score = self.score_layer(g, x, etypes=e).squeeze(-1)

      
        hard_mask, soft_gate, keep_ratios = top_k_learned(
            score, g, graphid_to_nodeids, self.ratio_predictor,
            device=device, min_keep=self.min_keep,
            temperature=self.temperature,
        )

     
        gate_sel = soft_gate[hard_mask]
        ste_gate = 1.0 + gate_sel - gate_sel.detach()      

        score_sel = score[hard_mask]
        x_pooled = (x[hard_mask]
                    * self.non_linearity(score_sel).view(-1, 1)
                    * ste_gate.view(-1, 1))

       
        pooled_graph, graphid_to_nodeids_pooled = \
            update_graphid_to_nodeids_after_pooling_optimized(
                g, graphid_to_nodeids, hard_mask
            )

        
        if keep_ratios.numel() > 0:
            self.last_ratio_reg = self.ratio_reg_weight * \
                ((keep_ratios - self.ratio_reg_target) ** 2).mean()
        else:
            self.last_ratio_reg = torch.tensor(0.0, device=device)

        return pooled_graph.to(device), x_pooled.to(device), graphid_to_nodeids_pooled
