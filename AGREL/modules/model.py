import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from dgl.nn import GatedGraphConv, GraphConv, AvgPooling, MaxPooling,  RelGraphConv
import dgl
from convlayers import GTLayer
from SAGPool import SAGPool
from mlp_readout import MLPReadout
from SAGPool import SAGPool,SAGPoolOptimized
from utils import global_avg_pool, global_max_pool
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class AGRELg(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(AGRELg, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.gcn = GraphConv(in_feats=input_dim, out_feats=output_dim)
        n_layers = 3
        num_head = 10
        self.n_layers = n_layers
        self.gtn =  nn.ModuleList([GTLayer(input_dim, output_dim, num_heads = num_head,
                                              dropout = 0.2,
                                         max_edge_types = max_edge_types, layer_norm= False,
                                         batch_norm= True, residual= True)
                                   for _ in range (n_layers - 1)])
        self.MPL_layer = MLPReadout(output_dim*2, 2)
        self.sigmoid = nn.Sigmoid()
        
        
        ffn_ratio = 2
        self.concat_dim = output_dim
        self.sagpool = SAGPoolOptimized(self.out_dim)
        

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to(device)
        features = features.to(device)
        edge_types = edge_types.to(device)
        gid= batch.graphid_to_nodeids
        for conv in self.gtn:
            features = conv(graph, features, edge_types, gid)
        
        new_graph,new_features,new_graphid = self.sagpool(graph, features,edge_types,batch.graphid_to_nodeids)
        batch.graphid_to_nodeids = new_graphid
        batch.graphs = new_graph
        outputs = torch.cat([global_avg_pool(new_features, batch.graphid_to_nodeids), global_max_pool(new_features, batch.graphid_to_nodeids)], dim=1)
        outputs = self.MPL_layer(outputs)
        outputs = nn.Softmax(dim=1)(outputs)
        return outputs

class AGRELh(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(AGRELh, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.gcn = GraphConv(in_feats=input_dim, out_feats=output_dim)
        n_layers = 3
        num_head = 10
        self.n_layers = n_layers
        self.gtn1 =  GTLayer(input_dim, output_dim, num_heads = num_head,
                                              dropout = 0.5,
                                         max_edge_types = max_edge_types, layer_norm= False,
                                         batch_norm= True, residual= True)
        self.gtn2 =  GTLayer(output_dim, output_dim, num_heads = num_head,
                                              dropout = 0.5,
                                         max_edge_types = max_edge_types, layer_norm= False,
                                         batch_norm= True, residual= True)
        self.gtn3 =  GTLayer(output_dim, output_dim, num_heads = num_head,
                                              dropout = 0.5,
                                         max_edge_types = max_edge_types, layer_norm= False,
                                         batch_norm= True, residual= True)
        
        self.MPL_layer = MLPReadout(output_dim*2, 2)
        self.sigmoid = nn.Sigmoid()
        
        
        ffn_ratio = 2
        self.concat_dim = output_dim
        
        self.sagpool1 = SAGPoolOptimized(self.out_dim)
        self.sagpool2 = SAGPoolOptimized(self.out_dim)
        
        self.sagpool3 = SAGPoolOptimized(self.out_dim)

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to(device)
        features = features.to(device)
        edge_types = edge_types.to(device)
        gid= batch.graphid_to_nodeids
        
        x = self.gtn1(graph, features, edge_types, gid)
        graph,x,gid=self.sagpool1(graph, x, edge_types, gid)
        x1=torch.cat([global_avg_pool(x, gid), global_max_pool(x, gid)], dim=1)
        edge_types=graph.edata['etype'].to(device)

        x = self.gtn2(graph, x, edge_types, gid)
        graph,x,gid=self.sagpool2(graph, x, edge_types, gid)
        x2 = torch.cat([global_avg_pool(x, gid), global_max_pool(x, gid)], dim=1)
        edge_types=graph.edata['etype'].to(device)

        x = self.gtn3(graph, x, edge_types, gid)
        graph,x,gid=self.sagpool3(graph, x, edge_types, gid)
        x3 = torch.cat([global_avg_pool(x, gid), global_max_pool(x, gid)], dim=1)
        
        outputs = x1+x2+x3
        outputs = F.relu(outputs)
        outputs = self.MPL_layer(outputs)
        outputs = nn.Softmax(dim=1)(outputs)
        return outputs

