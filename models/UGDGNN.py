from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul

class UGDGNN(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, args, cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(UGDGNN, self).__init__(**kwargs)
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.K = self.num_layers
        self.cached = self.transductive
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        
        self.input_trans = torch.nn.Linear(self.num_features, self.dim_hidden)
        self.output_trans = torch.nn.Linear(self.dim_hidden, self.num_classes)
        
        self.layers1 = torch.nn.ModuleList([])
        self.weight1 = torch.nn.Parameter(torch.ones(self.K+1)/self.K)
        self.weight2 = torch.nn.Parameter(torch.ones(self.K+1)/self.K)
        for _ in range(self.K+1):
            self.layers1.append(torch.nn.Linear(self.num_classes, self.num_classes))
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        
    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None   

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_trans(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.output_trans(x)
        
        x = h
        x_final = self.weight1[self.K] * self.layers1[self.K](h) + (1 - self.weight1[self.K]) * h
        x_final = self.weight2[self.K] * x_final
        for k in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    edge_index = edge_index.set_value(value, layout='coo')

            x_prop = self.propagate(edge_index, x=x, edge_weight=edge_weight,size=None)
            x = self.weight1[k] * self.layers1[k](x_prop) + (1 - self.weight1[k]) * x_prop
            x_final += self.weight2[k] * x
        
        return F.log_softmax(x_final, dim=1)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)
