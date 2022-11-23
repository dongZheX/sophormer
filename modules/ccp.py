from .sop.mlp import MLP
from .sop.MPNCOV import Triuvec, Sqrtm
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_softmax, scatter_mean
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
                args=None):
        super(MultiHeadSOP, self).__init__()
        self.hidden_channels = hidden_channels
        self.reduction_list = reduction_list
        self.is_triu = is_triu
        self.cheads = cheads
        self.use_bn = use_bn
        self.last_bn = last_bn
        self.mlp_dropout = mlp_dropout
        self.final_dropout = final_dropout
        self.reduction_layer = MLP(self.hidden_channels, self.reduction_list, use_bn=self.use_bn, last_bn=self.last_bn, dropout=self.mlp_dropout)
        self.hidden_channels = self.reduction_list[-1]
        self.cfeat_drop = cfeat_drop
        self.d_h = self.hidden_channels // self.cheads 
        

    def name(self):
        return "MULTIHEADSOP"

    def add_args(self, args):
        pass

    # checked
    def forward(self, x, batch, edge_index=None, deg=None):
        # No weight version
        h = F.dropout(x, p=self.cfeat_drop, training=self.training)
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
        # isqrt 暂时抛弃
        return F.dropout(output, p=self.final_dropout, training=self.training)    
            


    