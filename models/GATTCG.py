# -*- coding: utf-8 -*-

import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.nn.norm import BatchNorm
from typing import Optional, Tuple, Union
from torch_geometric.nn.inits import glorot, zeros
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch.nn import Conv2d

lag = 10
device = torch.device("cuda:1")

class time_att(nn.Module):
    def __init__(self, n_hidden_1):
        super(time_att, self).__init__()
        self.W = nn.Parameter(torch.zeros(lag, n_hidden_1))
        nn.init.xavier_normal_(self.W.data)

    def forward(self, ht):
        ht_W = ht.mul(self.W)
        ht_W = torch.sum(ht_W, dim=2)
        att = F.softmax(ht_W, dim=1)
        return att

#
class ATT_LSTM(nn.Module):
    def __init__(self, in_dim=12, n_hidden_1=64, out_dim=64):
        super(ATT_LSTM, self).__init__()
        self.LSTM = nn.LSTM(
            in_dim, n_hidden_1, 1, batch_first=True, bidirectional=False
        )
        self.time_att = time_att(n_hidden_1)
        self.fc = nn.Sequential(nn.Linear(n_hidden_1, out_dim), nn.ReLU(True))

    def forward(self, x):
        ht, (hn, cn) = self.LSTM(x)

        t_att = self.time_att(ht).unsqueeze(dim=1)
        att_ht = torch.bmm(t_att, ht)

        att_ht = self.fc(att_ht)

        return att_ht


class Macro_Encoder(nn.Module):
    def __init__(self, in_dim=1, hidden_size=8, num_layers=1, output_size=3):
        super(Macro_Encoder, self).__init__()
        self.in_dim = in_dim
        self.RNN = nn.RNN(
            input_size=in_dim, hidden_size=hidden_size, num_layers=1, batch_first=True
        )
        self.time_att = time_att(hidden_size)
        self.fc = nn.Sequential(nn.Linear(hidden_size, output_size), nn.ReLU(True))

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        x = x.reshape([-1, 10, self.in_dim])
        out, h = self.RNN(x)
        att = self.time_att(out).unsqueeze(dim=1)
        att_ht = torch.bmm(att, out)
        att_ht = self.fc(att_ht)

        return att_ht

        """ MLP version
        x = x.reshape([-1, self.seq_len, self.in_dim])
        out, h = self.RNN(x)
        # out = self.cross_mlp(x)
        att = self.time_att(out).unsqueeze(dim=1)
        att_ht = torch.bmm(att, out)
        att_ht = self.fc(att_ht)
        return att_ht
        """
    
        
class Modified_GATConv(GATConv):
    def __init__(self, macro_out_dim: int = 3, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.lin_macro = Linear(
            macro_out_dim,
            self.heads * self.out_channels,
            bias=False,
            weight_initializer="glorot",
        ).to(device)
        self.att_macro = Parameter(torch.Tensor(1, self.heads, self.out_channels))
        self.lin_macro.reset_parameters()
        glorot(self.att_macro)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        macro: OptTensor = None,
        size: Size = None,
        return_attention_weights=None,
    ):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        if self.add_self_loops:
            tmp = macro[0, :]
            loop = tmp.repeat([x_src.shape[0], 1])
            macro = torch.cat([macro, loop], dim=0)
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(
            edge_index, alpha=alpha, edge_attr=edge_attr, macro=macro
        )
        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def edge_updater(self, edge_index: Adj, **kwargs):
        for hook in self._edge_update_forward_pre_hooks.values():
            res = hook(self, (edge_index, kwargs))
            if res is not None:
                edge_index, kwargs = res

        size = self.__check_input__(edge_index, size=None)

        coll_dict = self.__collect__(self.__edge_user_args__, edge_index, size, kwargs)
        edge_kwargs = self.inspector.distribute("edge_update", coll_dict)
        out = self.edge_update(**edge_kwargs)

        for hook in self._edge_update_forward_hooks.values():
            res = hook(self, (edge_index, kwargs), out)
            if res is not None:
                out = res

        return out

    def edge_update(
        self,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
        macro: Tensor,
    ) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        if macro is not None:
            if macro.dim() == 1:
                macro = macro.view(-1, 1)
            macro = self.lin_macro(macro.to(device))
            macro = macro.view(-1, self.heads, self.out_channels)
            alpha_macro = (macro * self.att_macro).sum(dim=-1)
            alpha = alpha + alpha_macro
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


class GATTCG(torch.nn.Module):
    def __init__(
        self,
        in_dim=12,
        hid_dim=64,
        macro_in_dim=1,
        macro_hid_dim=8,
        in_channels=64,
        out_channels=32,
        heads=1,
        num_classes=3,
        edge_dim=9,
        out_features=9,
    ):
        super(GATTCG, self).__init__()
        self.dropout = 0.7
        self.training = True
        self.att_lstm = ATT_LSTM(in_dim, hid_dim, hid_dim)
        self.norm_1 = BatchNorm(in_channels, eps=1e-5)  # in_channels
        self.macro_encode = Macro_Encoder(
            in_dim=macro_in_dim, hidden_size=macro_hid_dim
        )

        self.conv1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.out = nn.Linear(2 * 2 * 16, edge_dim)
        self.gatconv_1 = Modified_GATConv(
            in_channels=in_channels,  
            out_channels=out_channels,
            heads=heads,
            edge_dim=edge_dim,
            macro_dim=macro_in_dim,
            dropout=self.dropout,
        )
        self.norm_2 = BatchNorm(out_channels * heads, eps=1e-5)
        self.gatconv_2 = Modified_GATConv(
            in_channels=out_channels * heads + in_dim,
            out_channels=out_channels,  
            heads=heads,
            edge_dim=edge_dim,
            macro_dim=macro_in_dim,
            dropout=self.dropout,
        )
        self.linear = nn.Linear(out_channels * heads, num_classes)
        self.act = nn.ReLU()
        self.fc = nn.Linear(out_features, 1)

    def forward(self, data):
        x0, edge_index, edge_attr, macro_seq = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.macro_seq,
        )

        x = self.att_lstm(x0)
        x = torch.squeeze(x)

        macro = self.macro_encode(macro_seq)
        macro = torch.squeeze(macro)

        x = self.norm_1(x)
        x = F.dropout(x, self.dropout, training=self.training)

        edge_attr = torch.reshape(edge_attr, (-1, 3, 9, 9))
        edge_attr = self.conv1(edge_attr)
        edge_attr = torch.reshape(edge_attr, (edge_attr.shape[0], -1))
        edge_attr = self.out(edge_attr)

        x = self.gatconv_1(x, edge_index, edge_attr, macro)

        x = self.norm_2(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gatconv_2(
            torch.cat((x, x0[:, -1, :]), dim=1), edge_index, edge_attr, macro
        )

        x = self.act(self.linear(x))
        return F.log_softmax(x, dim=1)
