import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import torch
from torch_geometric.nn import (
    GlobalAttention,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
import math
from modules.gnn_module import GNNNodeEmbedding
from modules.masked_transformer_encoder import MaskedOnlyTransformerEncoder
from modules.transformer_encoder import TransformerNodeEncoder
from modules.utils import pad_batch, unpad_batch
from .base_model import BaseModel

from modules.ccp import MultiHeadSOP
class SOGNNTransformer(BaseModel):
    @staticmethod
    def get_emb_dim(args):
        return args.gnn_emb_dim

    @staticmethod
    def add_args(parser):
        TransformerNodeEncoder.add_args(parser)
        MaskedOnlyTransformerEncoder.add_args(parser)
        group = parser.add_argument_group("GNNTransformer - Training Config")
        group.add_argument("--pos_encoder", default=False, action='store_true')
        group.add_argument("--pretrained_gnn", type=str, default=None, help="pretrained gnn_node node embedding path")
        group.add_argument("--reduction_list", nargs="+", type=int)
        group.add_argument("--sr", type=float, default=1)
        group.add_argument("--cheads", type=int, default=1)
        group.add_argument("--is_triu", action="store_true", default=False)
        group.add_argument("--use_bn", action="store_true", default=False)
        group.add_argument("--last_bn", action="store_true", default=False)
        group.add_argument("--cfeat_drop", type=float, default=0.3)
        group.add_argument("--mlp_dropout", type=float, default=0)
        group.add_argument("--final_dropout", type=float, default=0.6)
        group.add_argument("--fix_epochs", type=int, default=0)
        group.add_argument("--freeze_type", type=str,default="all")
        group.add_argument("--num_ccp", type=int, default=1)

        group.add_argument("--sqrt_d", action="store_true", default=False)
        group.add_argument("--fuse_type", type=str)
        group.add_argument("--fuse_alpha", type=float, default=1)
        
    @staticmethod
    def name(args):
        name = f"{args.model_type}-pooling={args.graph_pooling}"
        name += "-norm_input" if args.transformer_norm_input else ""
        name += f"+{args.gnn_type}"
        name += "-virtual" if args.gnn_virtual_node else ""
        name += f"-JK={args.gnn_JK}"
        name += f"-enc_layer={args.num_encoder_layers}"
        name += f"-enc_layer_masked={args.num_encoder_layers_masked}"
        name += f"-d={args.d_model}"
        name += f"-act={args.transformer_activation}"
        name += f"-tdrop={args.transformer_dropout}"
        name += f"-gdrop={args.gnn_dropout}"
        name += "-pretrained_gnn" if args.pretrained_gnn else ""
        name += f"-freeze_gnn={args.freeze_gnn}" if args.freeze_gnn is not None else ""
        name += "-prenorm" if args.transformer_prenorm else "-postnorm"
        return name

    def __init__(self, num_tasks, node_encoder, edge_encoder_cls, args):
        super().__init__()
        self.gnn_node = GNNNodeEmbedding(
            args.gnn_virtual_node,
            args.gnn_num_layer,
            args.gnn_emb_dim,
            node_encoder,
            edge_encoder_cls,
            JK=args.gnn_JK,
            drop_ratio=args.gnn_dropout,
            residual=args.gnn_residual,
            gnn_type=args.gnn_type,
            
        )
        gnn_emb_dim = 2 * args.gnn_emb_dim if args.gnn_JK == "cat" else args.gnn_emb_dim
        self.gnn2transformer = nn.Linear(gnn_emb_dim, args.d_model)
        self.pos_encoder = PositionalEncoding(args.d_model, dropout=0) if args.pos_encoder else None
        self.transformer_encoder = TransformerNodeEncoder(args)
        self.masked_transformer_encoder = MaskedOnlyTransformerEncoder(args)
        self.num_encoder_layers = args.num_encoder_layers
        self.num_encoder_layers_masked = args.num_encoder_layers_masked
        self.fix_epochs = args.fix_epochs
        self.freeze_type = args.freeze_type
        self.num_tasks = num_tasks
        self.pooling = args.graph_pooling
        self.graph_pred_linear_list = torch.nn.ModuleList()
        self.max_seq_len = args.max_seq_len
        self.pretrained_gnn = args.pretrained_gnn
        self.fuse_type = args.fuse_type
        self.fuse_alpha = args.fuse_alpha
        self.sqrd_d = args.sqrt_d
        if self.pooling == "ccp":
            self.pools = nn.Sequential()
            for i in range(args.num_ccp):
                pool = MultiHeadSOP(
                    hidden_channels=args.d_model, 
                    reduction_list= args.reduction_list,
                    is_triu=args.is_triu,
                    cheads=args.cheads,
                    use_bn=args.use_bn,
                    last_bn=args.last_bn,
                    cfeat_drop=args.cfeat_drop,
                    args=args,
                    mlp_dropout=args.mlp_dropout,
                    num_classes=self.num_tasks,
                )
                self.pools.append(pool)
            d = pool.d_h
            output_dim = d*d if not args.is_triu else d * (d+1)//2
            output_dim = output_dim * args.cheads
        if args.max_seq_len is None:
            self.graph_pred_linear = torch.nn.Linear(output_dim, self.num_tasks)
        else:
            for i in range(args.max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(output_dim, self.num_tasks))
        # need to be done
        if args.pretrained_gnn:
            # logger.info(self.gnn_node)
            state_dict = torch.load(args.pretrained_gnn)['model']
            
            # self.gnn_node.load_state_dict(state_dict)
            new_state_dict = dict()
            for k,v in state_dict.items():
                if k in self.state_dict() and self.state_dict()[k].shape == v.shape:
                    new_state_dict[k] = v
            logger.info("Load GNN state from: {}", new_state_dict.keys())
            own_state_dict = self.state_dict()
            own_state_dict.update(new_state_dict)
            self.load_state_dict(own_state_dict)
            self.freeze(args.freeze_type) 

    # 这个还没检查，学习率问题
    def freeze(self, con="gnn"):
        # gnn, trans, all
        logger.info(f"Freezing {con}")
        for name, parameter in self.named_parameters():
            if con in ["gnn", "all"]:
                if "gnn_node" in name:          
                    parameter.requires_grad = False
            if con in ["trans", "all"]:
                if "transformer" in name:          
                    parameter.requires_grad = False
                    

    def open_grad(self):
        # 开启参数更新
        for name, parameter in self.named_parameters():
            parameter.requires_grad = True

    def forward(self, batched_data, perturb=None):
        h_node = self.gnn_node(batched_data, perturb)
        h_node = self.gnn2transformer(h_node)  # [s, b, d_model]
        # padded_h_node [max_num_nodes, batch_size, d_model]
        # src_paddiing_mask [batch_size, max_num_nodes],False is dummy 
        # num_nodes: [batch_size], 每个元素都是个 tensor 值
        # max_num_nodes: value
        padded_h_node, src_padding_mask, num_nodes, mask, max_num_nodes = pad_batch(
            h_node, batched_data.batch, self.transformer_encoder.max_input_len, get_mask=True
        )  # Pad in the front

        # TODO(paras): implement mask
        transformer_out = padded_h_node
        # 可以修改顺序 然后加入LPE
        if self.pos_encoder is not None:
            transformer_out = self.pos_encoder(transformer_out)
        if self.num_encoder_layers_masked > 0:
            adj_list = batched_data.adj_list
            padded_adj_list = torch.zeros((len(adj_list), max_num_nodes, max_num_nodes), device=h_node.device)
            for idx, adj_list_item in enumerate(adj_list):
                N, _ = adj_list_item.shape
                padded_adj_list[idx, 0:N, 0:N] = torch.from_numpy(adj_list_item)
            transformer_out = self.masked_transformer_encoder(
                transformer_out.transpose(0, 1), attn_mask=padded_adj_list, valid_input_mask=src_padding_mask
            ).transpose(0, 1)
        if self.num_encoder_layers > 0:
            transformer_out, _ = self.transformer_encoder(transformer_out, src_padding_mask)  # [s, b, h], [b, s]

        if self.pooling in ["last", "cls"]:
            h_graph = transformer_out[-1]
        elif self.pooling == "mean":
            gnn_unpad = unpad_batch(padded_h_node, h_node, num_nodes, mask, max_num_nodes)
            h_graph = global_mean_pool(gnn_unpad, batched_data.batch)
        elif self.pooling == "ccp":
            gnn_unpad = unpad_batch(padded_h_node, h_node, num_nodes, mask, max_num_nodes)
            h_graph = 0
            for pool in self.pools:
                h_graph += pool(gnn_unpad, batched_data.batch)
            if self.fuse_type is not None:
                return h_graph
        else:
            raise NotImplementedError

            
        if self.max_seq_len is None:
            return self.graph_pred_linear(h_graph)
        pred_list = []
        # code暂时不管
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))

        return pred_list

    def epoch_callback(self, epoch):
        # TODO: maybe unfreeze the gnn at the end.
        if epoch > self.fix_epochs and self.pretrained_gnn is not None:
            print("open_grad")
            self.open_grad()
            self.fix_epochs = 1000000000000

    def _gnn_node_state(self, state_dict):
        module_name = "gnn_node"
        new_state_dict = dict()
        for k, v in state_dict.items():
            if module_name in k:
                new_key = k.split(".")
                module_index = new_key.index(module_name)
                new_key = ".".join(new_key[module_index + 1 :])
                new_state_dict[new_key] = v
        return new_state_dict


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)