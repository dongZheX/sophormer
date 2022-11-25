from .sop.mlp import MLP
from .sop.MPNCOV import Triuvec, Sqrtm
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_softmax, scatter_mean, scatter_add
class MultiHeadSOP(nn.Module):
    def __init__(self,
                hidden_channels, 
                reduction_list,
                is_triu=True,
                cheads=2,
                use_bn=False,
                last_bn=False,
                cfeat_drop=0.3,
                mlp_dropout=0.5,
                final_dropout=0.6,
                num_classes=1,
                args=None):
        super(MultiHeadSOP, self).__init__()
        self.input_dim = hidden_channels
        self.hidden_channels = hidden_channels
        self.reduction_list = reduction_list
        self.is_triu = is_triu
        self.cheads = cheads
        self.use_bn = use_bn
        self.last_bn = last_bn
        self.mlp_dropout = mlp_dropout
        self.final_dropout = final_dropout
        if self.reduction_list is not None:
            self.reduction_layer = MLP(self.hidden_channels, self.reduction_list, use_bn=self.use_bn, last_bn=self.last_bn, dropout=self.mlp_dropout)
            self.hidden_channels = self.reduction_list[-1]
        else:
            self.reduction_layer = None 
        self.cfeat_drop = cfeat_drop
        self.d_h = self.hidden_channels // self.cheads 
        self.sqrt_d = args.sqrt_d
        self.fuse_type = args.fuse_type
        self.fuse_alpha = args.fuse_alpha
        self.tmp_dim = self.d_h ** 2 if not self.is_triu else self.d_h * (self.d_h + 1) // 2
        if self.fuse_type == "concat":      
            self.visual_fc = nn.Linear(self.tmp_dim * self.cheads, num_classes)
        elif self.fuse_type == "sum_fc":
            self.class_fc = nn.Linear(self.input_dim, num_classes)
            self.visual_fc = nn.Linear(self.tmp_dim * self.cheads, num_classes)
        elif self.fuse_type == "aggr":
            self.visual_fc = nn.Linear(self.tmp_dim * self.cheads, num_classes)
        
            

    def name(self):
        return "MULTIHEADSOP"

    def add_args(self, args):
        pass


    def aggregate(self, x, cls_token, batch):
        bz = batch[-1].item() + 1
        new_batch = batch.new_zeros(batch.shape[0] + bz)
        new_x = x.new_zeros(x.shape[0] + bz, x.shape[1])
        num_nodes = scatter_add(torch.ones_like(batch), batch)
        cumsum_num_nodes = num_nodes.cumsum(0)
        for i in range(bz):
            start_id = 0 if i == 0 else cumsum_num_nodes[i-1]
            end_id = cumsum_num_nodes[i]
            new_batch[start_id + i: end_id + i] = batch[start_id: end_id]
            new_batch[end_id + i] = new_batch[start_id + i]
            new_x[start_id + i: end_id + i, :] = x[start_id: end_id, :]
            new_x[end_id + i,:] = cls_token[i,:]
        return new_x, new_batch


    # checked
    def forward(self, x, batch, edge_index=None, deg=None, cls_token=None):
        if self.fuse_type == "aggr" and cls_token is not None:
            x, batch = self.aggregate(x, cls_token, batch)
        # No weight version
        h = F.dropout(x, p=self.cfeat_drop, training=self.training)
        if self.reduction_layer is not None:
            x = self.reduction_layer(h)
        # deg 和 edge_index暂时不管
        bz = batch[-1].item() + 1
        M = x.size(0)
        embed_dim = x.size(1)
        x = x.view(M, self.cheads, self.d_h).permute(1, 0, 2) # [H, M, D]
        wj = torch.einsum("hik,hkj->hkij", x.permute(0, 2, 1), x) #[H , M ,D, D] 
        triu_d = self.d_h * (self.d_h+1) // 2
        # 默认全使用对角线
        if self.is_triu:
            wj = Triuvec.apply(wj.reshape(-1, self.d_h, self.d_h)).view(self.cheads, M, triu_d)
        else:
            wj = wj.view(self.cheads, M, -1)
        # if self.use_weight:
        #     flatten_wj = wj.permute(1, 0, 2).contiguous().view(M, -1)
        #     imp = self.attention(flatten_wj) # [M， cheads]
        #     imp = scatter_softmax(imp * self.sr, batch, dim=0).transpose(1, 0).unsqueeze(-1) # [cheads, M, -1] 
        #     if self.cattn_drop > 0:
        #         imp = F.dropout(imp, p=self.cattn_drop, training=self.training)
        #     output = scatter_mean((wj * imp), batch, dim=1).permute(1, 0, 2).contiguous().view(bz, -1)   # [cheads, N, -1]
        # else:
        output = scatter_mean(wj.permute(1,0,2).contiguous().view(M, -1), batch, dim=0)
        if self.sqrt_d:
            output = output * (self.d_h ** -0.5) 
        
        # isqrt 暂时抛弃
        s_h =  F.dropout(output, p=self.final_dropout, training=self.training)    
        if self.fuse_type and cls_token:
            if self.fuse_type == "concat":
                return self.visual_fc(torch.cat([cls_token, s_h],dim=1))
            elif self.fuse_type == "sum_fc":
                return self.fuse_alpha *  self.visual_fc(s_h) + (1-self.fuse_alpha) * self.class_fc(cls_token)         
            elif self.fuse_type == "aggr":
                return self.visual_fc(s_h)
            else:
                raise NotImplementedError       
        else:
            return s_h



# def aggregate(x, cls_token, batch):
#     bz = batch[-1].item() + 1
#     new_batch = batch.new_zeros(batch.shape[0] + bz)
#     new_x = x.new_zeros(x.shape[0] + bz, x.shape[1])
#     num_nodes = scatter_add(torch.ones_like(batch), batch)
#     cumsum_num_nodes = num_nodes.cumsum(0)
#     for i in range(bz):
#         start_id = 0 if i == 0 else cumsum_num_nodes[i-1]
#         end_id = cumsum_num_nodes[i]
#         new_batch[start_id + i: end_id + i] = batch[start_id: end_id]
#         new_batch[end_id + i] = new_batch[start_id + i]
#         new_x[start_id + i: end_id + i, :] = x[start_id: end_id, :]
#         new_x[end_id + i,:] = cls_token[i,:]
#     return new_x, new_batch
# import numpy as np
# x = torch.randn([12, 36])
#    = torch.from_numpy(np.asarray([0,0,0,1,1,2,2,2,2,3,3,3]))
# cls_token = torch.randn([36]).repeat(4,1)
# a ,b = aggregate(x, cls_token, batch)
# print(a)
# print(b)