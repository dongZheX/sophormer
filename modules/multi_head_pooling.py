import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from .sot.covariance import Covariance
# need to implement
# from .covariance import Covariance
from .normalization import svPN, MPN, ePN, scaleNorm,Triuvec
# x = torch.randn(12, 32)
# y = torch.randn(4, 32)
# batch = torch.from_numpy(np.asarray([0,0,0,1,1,2,2,3,3,4,4,4]))
#  Classifier(3, 32, representationConfig)
class MultiHeadPooling(nn.Module):
    def __init__(self, dim, num_heads=8, wr_dim=16,remove_mean=False, qkv_bias=False, normalization={}):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.wr = nn.Linear(dim, wr_dim*num_heads*2, bias=qkv_bias)
        self.wr_dim = wr_dim
        self.scale = head_dim ** -0.5
        self.wr = nn.Linear(dim, wr_dim * num_heads * 2, bias=qkv_bias)
        self.wr_dim = wr_dim
        self.cov = Covariance(remove_mean=remove_mean)
        norm_type = normalization['type']
        normalization.pop('type')
        if norm_type == 'svPN':
            self.norm = svPN(**normalization)
        elif norm_type == 'sqrt_d':
            self.norm = scaleNorm(factor=self.wr_dim**-0.5)
        elif norm_type == 'sqrt_t':
            self.norm = scaleNorm(factor=self.wr_dim**-0.5, is_learnable=True)
        elif norm_type == 'ePN':
            self.norm = ePN()
        elif norm_type == 'LN':
            self.norm = nn.LayerNorm(self.wr_dim)
        elif norm_type == "None":
            self.norm = nn.Identity()
        self.output_dim = num_heads * (wr_dim ** 2) if normalization['vec'] != "triu" else num_heads * (wr_dim * (wr_dim+1))//2
    def forward(self, x, batch):
        M, C = x.shape
        bz = batch[-1].item() + 1
        qk = self.wr(x).reshape(M, 2, self.num_heads, self.wr_dim).permute(1, 2, 0, 3)
        q, k = qk[0], qk[1]
        # cov
        x = self.cov(batch, q, k).reshape(self.num_heads * bz, self.wr_dim, self.wr_dim)
        x = self.norm(x)
        x = x.view(self.num_heads, bz, -1).permute(1, 0, 2).contiguous().view(bz, -1)
        return x

# 这里周四要再看看如何设置比较合理
# 通过None的方式设置吧

class Classifier(nn.Module):
    def __init__(self, num_classes=1, input_dim=300, representationConfig={}, args=None):
        super(Classifier, self).__init__()
        self.fusion_type = representationConfig['fusion_type']
        self.re_type = representationConfig['type']
        self.remove_mean = representationConfig['args']['remove_mean']
        self.cfeat_drop = args.cfeat_drop
        self.final_dropout = args.final_dropout
        if self.re_type == 'MGCrP':
            self.representation = MultiHeadPooling(**representationConfig['args'])
            output_dim = self.representation.output_dim
            if self.fusion_type == 'concat':
                output_dim = input_dim + output_dim 
            self.visual_fc = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.re_type == 'GCP':
            cov_dim = int(pow(2 * representationConfig['args']['num_heads'] * representationConfig['args']['dim'] ** 2, 0.5))
            self.dr = nn.Linear(input_dim, cov_dim)
            self.representation = Covariance(remove_mean=self.remove_mean)
            self.normalization = MPN(input_dim=cov_dim)
            if self.fusion_type == 'concat':
                output_dim = input_dim + int(cov_dim * (cov_dim + 1) / 2)
            else:
                output_dim = int(cov_dim * (cov_dim + 1) / 2)
            self.visual_fc = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.re_type == 'dropGCP':
            cov_dim = int(pow(2 * representationConfig['args']['num_heads'] * representationConfig['args']['dim'] ** 2, 0.5))
            self.dr = nn.Linear(input_dim, cov_dim)
            self.representation = Covariance(remove_mean=self.remove_mean)
            if self.fusion_type == 'concat':
                output_dim = input_dim + int(cov_dim * (cov_dim + 1) / 2)
            else:
                output_dim = int(cov_dim * (cov_dim + 1) / 2)
            self.visual_fc = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.cls_fc = nn.Linear(input_dim, num_classes)

    @staticmethod
    def add_args(parser):
        pass

    def forward(self, cls_token, node_token, batch):
        # 暂时不支持aggregate all
        
        if self.re_type is not None:
            if self.re_type == 'GCP':
                x = node_token
                cls_head = self.cls_fc(cls_token)
                x = self.dr(x)
                x = self.representation(batch, x.unsqueeze(0)).squeeze(0)
                x = self.normalization(x)
                x = x.view(x.size(0), -1)

            if self.re_type == "dropGCP":
                node_token = F.dropout(node_token, p=self.cfeat_drop,training=self.training)
                x = node_token
                cls_head = self.cls_fc(cls_token)
                x = self.dr(x)
                x = self.representation(batch, x.unsqueeze(0)).squeeze(0)
                x = Triuvec.apply(x)
                x = x.view(x.size(0), -1)
                x = F.dropout(x, p=self.final_dropout, training=self.training)
            if self.re_type == 'MGCrP':
                node_token = F.dropout(node_token, p=self.cfeat_drop,training=self.training)
                x = node_token
                cls_head = self.cls_fc(cls_token)
                x = self.representation(x, batch)
                x = x.view(x.size(0), -1)
                x = F.dropout(x, p=self.final_dropout, training=self.training)

            if self.fusion_type == 'concat':
                x = torch.cat([cls_token, x], dim=1)
                head = self.visual_fc(x)
                return head
            elif self.fusion_type == 'sum_fc':
                vis_head = self.visual_fc(x)
                return vis_head + cls_head
            return x
        else:
            head = self.cls_fc(cls_token)
            return head

class OnlyNodeClassifier(nn.Module):
    def __init__(self, num_classes=1, input_dim=300, representationConfig={}, args=None):
        super(OnluNodeClassifier, self).__init__()
        self.fusion_type = representationConfig['fusion_type'] # unuse argument
        self.re_type = representationConfig['type']
        self.cfeat_drop = args.cfeat_drop
        self.final_dropout = args.final_dropout
        if self.re_type == 'MGCrP':
            self.representation = MultiHeadPooling(**representationConfig['args'])
            output_dim = self.representation.output_dim
            self.visual_fc = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.re_type == 'GCP':
            cov_dim = int(pow(2 * representationConfig['args']['num_heads'] * representationConfig['args']['dim'] ** 2, 0.5))
            self.dr = nn.Linear(input_dim, cov_dim)
            self.representation = Covariance(remove_mean=True)
            self.normalization = MPN(input_dim=cov_dim)
            output_dim = int(cov_dim * (cov_dim + 1) / 2)
            self.visual_fc = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    @staticmethod
    def add_args(parser):
        pass
    def forward(self, node_token, batch):  

        node_token = F.dropout(node_token, p=self.cfeat_drop,training=self.training)
        # 暂时不支持aggregate all
        if self.re_type == 'GCP':
            x = node_token
            x = self.dr(x)
            x = self.representation(batch,x)
            x = self.normalization(x)
            x = x.view(x.size(0), -1)

        if self.re_type == 'MGCrP':
            x = node_token
            x = self.representation(x, batch)
            x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.final_dropout, training=self.training)
        output = self.visual_fc(x)
        return output        


# representationConfig = dict(
#     type='GCP',
#     fusion_type='concat',
    
#     args=dict(
#         dim=32,
#         num_heads=4,
#         wr_dim=16,
#         remove_mean=False,
#         normalization=dict(
#             type='LN',
#             alpha=0.5,
#             iterNum=1,
#             svNum=1,
#             regular=nn.Dropout(0.5),
#             input_dim=16,
#             vec="full"
#         )
#     ),
# )
# x = torch.randn(12, 32)
# y = torch.randn(5, 32)
# batch = torch.from_numpy(np.asarray([0,0,0,1,1,2,2,3,3,4,4,4]))
# c = Classifier(3, 32, representationConfig)
# c(y, x, batch)