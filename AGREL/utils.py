import numpy as np
import torch_scatter
import torch
from data_loader import n_identifier, g_identifier, l_identifier
import inspect
from datetime import datetime
import logging
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def load_default_identifiers(n, g, l):
    if n is None:
        n = n_identifier
    if g is None:
        g = g_identifier
    if l is None:
        l = l_identifier
    return n, g, l


def initialize_batch(entries, batch_size, shuffle=False):
    total = len(entries)
    print(str(total)+'k'*35)
    indices = np.arange(0, total , 1)
    if shuffle:
        np.random.shuffle(indices)
    batch_indices = []
    start = 0
    end = len(indices)
    curr = start
    while curr < end:
        c_end = curr + batch_size
        if c_end > end:
            c_end = end
        batch_indices.append(indices[curr:c_end])
        curr = c_end
    return batch_indices[::-1]


def tally_param(model):
    total = 0
    for param in model.parameters():
        total += param.data.nelement()
    return total


def debug(*msg, sep='\t'):
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    print('[' + str(time) + '] File \"' + file_name + '\", line ' + str(ln) + '  ', end='\t')
    for m in msg:
        print(m, end=sep)
    print('')

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="w", encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        #logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
def getindicator_from_graphid(graphid_to_nodeids_pooled, device=device):
    graph_indicator = []
    for gid, node_ids in graphid_to_nodeids_pooled.items():
        num_nodes = len(node_ids)
        graph_indicator.append(torch.full((num_nodes,), gid, dtype=torch.long, device=node_ids.device))
    graph_indicator = torch.cat(graph_indicator, dim=0)
    if device is not None:
        graph_indicator = graph_indicator.to(device)
    return graph_indicator
def global_max_pool(x, graphid_to_nodeids):
    graph_indicator = getindicator_from_graphid(graphid_to_nodeids)

    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]

def global_avg_pool(x, graphid_to_nodeids):
    graph_indicator = getindicator_from_graphid(graphid_to_nodeids)
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)