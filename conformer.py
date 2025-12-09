import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_

conformer_small_patch16_modules = [
    'nodeA',
    'nodeB',
    'nodeC',
    'nodeD',
    'nodeE',
    'nodeF_step1_2',
    'nodeF_trans_steps_2',
    'nodeH_fusion_2',
    'nodeF_step1_3',
    'nodeF_trans_steps_3',
    'nodeH_fusion_3',
    'nodeF_step1_4',
    'nodeF_trans_steps_4',
    'nodeH_fusion_4',
    'nodeF_step1_5',
    'nodeF_trans_steps_5',
    'nodeH_fusion_5',
    'nodeF_step1_6',
    'nodeF_trans_steps_6',
    'nodeH_fusion_6',
    'nodeF_step1_7',
    'nodeF_trans_steps_7',
    'nodeH_fusion_7',
    'nodeF_step1_8',
    'nodeF_trans_steps_8',
    'nodeH_fusion_8',
    'nodeF_step1_9',
    'nodeF_trans_steps_9',
    'nodeH_fusion_9',
    'nodeF_step1_10',
    'nodeF_trans_steps_10',
    'nodeH_fusion_10',
    'nodeF_step1_11',
    'nodeF_trans_steps_11',
    'nodeH_fusion_11',
    'nodeI_trans_head',
    'nodeI_conv_head'
]

conformer_small_patch16_dependencies = {
    'nodeA': {'prev':[], 'next': ['nodeE']},
    'nodeB': {'prev':[], 'next': ['nodeC', 'nodeD']},
    'nodeC': {'prev':['nodeB'], 'next': ['nodeF_step1_2']},
    'nodeD': {'prev':['nodeB'], 'next': ['nodeE']},
    'nodeE': {'prev':['nodeA', 'nodeD'], 'next': ['nodeF_trans_steps_2']},
    
    'nodeF_step1_2': {'prev':['nodeC'], 'next': ['nodeF_trans_steps_2, nodeH_fusion_2']},
    'nodeF_trans_steps_2': {'prev':['nodeE', 'nodeF_step1_2'], 'next': ['nodeF_trans_steps_3', 'nodeH_fusion_2']},
    'nodeH_fusion_2': {'prev':['nodeF_step1_2', 'nodeF_trans_steps_2'], 'next': ['nodeF_step1_3', 'nodeF_trans_steps_3']},
    
    'nodeF_step1_3': {'prev':['nodeH_fusion_2'], 'next': ['nodeF_trans_steps_3, nodeH_fusion_3']},
    'nodeF_trans_steps_3': {'prev':['nodeF_trans_steps_2', 'nodeF_step1_3'], 'next': ['nodeF_trans_steps_4', 'nodeH_fusion_3']},
    'nodeH_fusion_3': {'prev':['nodeF_step1_3', 'nodeF_trans_steps_3'], 'next': ['nodeF_step1_4']},
    
    'nodeF_step1_4': {'prev':['nodeH_fusion_3'], 'next': ['nodeF_trans_steps_4, nodeH_fusion_4']},
    'nodeF_trans_steps_4': {'prev':['nodeF_trans_steps_3', 'nodeF_step1_4'], 'next': ['nodeF_trans_steps_5', 'nodeH_fusion_4']},
    'nodeH_fusion_4': {'prev':['nodeF_step1_4', 'nodeF_trans_steps_4'], 'next': ['nodeF_step1_5']},
    
    'nodeF_step1_5': {'prev':['nodeH_fusion_4'], 'next': ['nodeF_trans_steps_5, nodeH_fusion_5']},
    'nodeF_trans_steps_5': {'prev':['nodeF_trans_steps_4', 'nodeF_step1_5'], 'next': ['nodeF_trans_steps_6', 'nodeH_fusion_5']},
    'nodeH_fusion_5': {'prev':['nodeF_step1_5', 'nodeF_trans_steps_5'], 'next': ['nodeF_step1_6']},
    
    'nodeF_step1_6': {'prev':['nodeH_fusion_5'], 'next': ['nodeF_trans_steps_6', 'nodeH_fusion_6']},
    'nodeF_trans_steps_6': {'prev':['nodeF_trans_steps_5', 'nodeF_step1_6'], 'next': ['nodeF_trans_steps_7', 'nodeH_fusion_6']},
    'nodeH_fusion_6': {'prev':['nodeF_step1_6', 'nodeF_trans_steps_6'], 'next': ['nodeF_step1_7']},
    
    'nodeF_step1_7': {'prev':['nodeH_fusion_6'], 'next': ['nodeF_trans_steps_7', 'nodeH_fusion_7']},
    'nodeF_trans_steps_7': {'prev':['nodeF_trans_steps_6', 'nodeF_step1_7'], 'next': ['nodeF_trans_steps_8', 'nodeH_fusion_7']},
    'nodeH_fusion_7': {'prev':['nodeF_step1_7', 'nodeF_trans_steps_7'], 'next': ['nodeF_step1_8']},
    
    'nodeF_step1_8': {'prev':['nodeH_fusion_7'], 'next': ['nodeF_trans_steps_8']},
    'nodeF_trans_steps_8': {'prev':['nodeF_trans_steps_7', 'nodeF_step1_8'], 'next': ['nodeF_trans_steps_9', 'nodeH_fusion_8']},
    'nodeH_fusion_8': {'prev':['nodeF_step1_8', 'nodeF_trans_steps_8'], 'next': ['nodeF_step1_9', 'nodeH_fusion_8']},
    
    'nodeF_step1_9': {'prev':['nodeH_fusion_8'], 'next': ['nodeF_trans_steps_9', 'nodeH_fusion_9']},
    'nodeF_trans_steps_9': {'prev':['nodeF_trans_steps_8', 'nodeF_step1_9'], 'next': ['nodeF_trans_steps_10', 'nodeH_fusion_9']},
    'nodeH_fusion_9': {'prev':['nodeF_step1_9', 'nodeF_trans_steps_9'], 'next': ['nodeF_step1_10']},
    
    'nodeF_step1_10': {'prev':['nodeH_fusion_9'], 'next': ['nodeF_trans_steps_10', 'nodeH_fusion_10']},
    'nodeF_trans_steps_10': {'prev':['nodeF_trans_steps_9', 'nodeF_step1_10'], 'next': ['nodeF_trans_steps_11', 'nodeH_fusion_10']},
    'nodeH_fusion_10': {'prev':['nodeF_step1_10', 'nodeF_trans_steps_10'], 'next': ['nodeF_step1_11']},
    
    'nodeF_step1_11': {'prev':['nodeH_fusion_10'], 'next': ['nodeF_trans_steps_11', 'nodeH_fusion_11']},
    'nodeF_trans_steps_11': {'prev':['nodeF_trans_steps_10', 'nodeF_step1_11'], 'next': ['nodeI_trans_head', 'nodeH_fusion_11']},
    'nodeH_fusion_11': {'prev':['nodeF_step1_11', 'nodeF_trans_steps_11'], 'next': ['nodeI_conv_head']},
    
    'nodeI_trans_head': {'prev':['nodeF_trans_steps_11'], 'next': ['']},
    'nodeI_conv_head': {'prev':['nodeI_conv_head'], 'next': ['']},
    }

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


# WE WILL BREAK IT!
'''
class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t
'''   
    
class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x) # step1

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t) # step2_transformer

        x_t = self.trans_block(x_st + x_t) # step3_transformer

        if self.num_med_block > 0:
            for m in self.med_block: #step2_cnn
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride) # step4_transformer
        x = self.fusion_block(x, x_t_r, return_x_2=False) # step_fusion

        return x, x_t
    

class TransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    UPDATED: now it only has the tranformer block.
    """

    def __init__(self, outplanes, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        super(TransBlock, self).__init__()
        expansion = 4

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim

    def forward(self, x2, x_t):
        _, _, H, W = x2.shape
        x_st = self.squeeze_block(x2, x_t) # step2_transformer

        x_t = self.trans_block(x_st + x_t) # step3_transformer

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride) # step4_transformer

        return x_t, x_t_r


class Conformer(nn.Module):

    def __init__(self, mygraph=[], patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0
        self.num_med_block = num_med_block
        self.mygraph = mygraph
        if 'nodeA' in self.mygraph:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=.02)

        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        if 'nodeI_trans_head' in self.mygraph:
            self.trans_norm = nn.LayerNorm(embed_dim)
            self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        if 'nodeI_conv_head' in self.mygraph:
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        # start nodeA
        if 'nodeB' in self.mygraph:
            self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
            self.bn1 = nn.BatchNorm2d(64)
            self.act1 = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]
        # end nodeA --> x_base

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        if 'nodeC' in self.mygraph:
            self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        
        trans_dw_stride = patch_size // 4
        if 'nodeD' in self.mygraph:
            self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        
        if 'nodeE' in self.mygraph:
            self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            if f'nodeF_step1_{i}' in self.mygraph:
                self.add_module('conv_trans_step1_' + str(i),
                    ConvBlock(inplanes=stage_1_channel, outplanes=stage_1_channel, res_conv=False, stride=1, groups=1))
            if f'nodeF_trans_steps_{i}' in self.mygraph:
                self.add_module('conv_trans_steps_' + str(i),
                    TransBlock(stage_1_channel, dw_stride=trans_dw_stride, embed_dim=embed_dim, 
                            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                            qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
                            drop_path_rate=self.trans_dpr[i-1]
                            )
            )

            if num_med_block > 0:
                for j in range(num_med_block):
                    if f'nodeG_medblock_{j}_{i}' in self.mygraph:
                        self.add_module('conv_trans_medblock_' + str(j) + '_' + str(i),
                        Med_ConvBlock(inplanes=stage_1_channel, groups=1)
                        )

            if f'nodeH_fusion_{i}' in self.mygraph:
                self.add_module('conv_trans_fusion_' + str(i),
                        ConvBlock(inplanes=stage_1_channel, outplanes=stage_1_channel, groups=1)
                )


        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage # 5
        fin_stage = fin_stage + depth // 3 # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            if f'nodeF_step1_{i}' in self.mygraph:
                self.add_module('conv_trans_step1_' + str(i),
                    ConvBlock(inplanes=in_channel, outplanes=stage_2_channel, res_conv=res_conv, stride=s, groups=1))
            if f'nodeF_trans_steps_{i}' in self.mygraph:
                self.add_module('conv_trans_steps_' + str(i),
                    TransBlock(stage_2_channel, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim, 
                            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                            qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
                            drop_path_rate=self.trans_dpr[i-1]
                            )
            )

            if num_med_block > 0:
                for j in range(num_med_block):
                    if f'nodeG_medblock_{j}_{i}' in self.mygraph:
                        self.add_module('conv_trans_medblock_' + str(j) + '_' + str(i),
                        Med_ConvBlock(inplanes=stage_2_channel, groups=1)
                        )

            if f'nodeH_fusion_{i}' in self.mygraph:
                self.add_module('conv_trans_fusion_' + str(i),
                        ConvBlock(inplanes=stage_2_channel, outplanes=stage_2_channel, groups=1)
                )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            if f'nodeF_step1_{i}' in self.mygraph:
                self.add_module('conv_trans_step1_' + str(i),
                    ConvBlock(inplanes=in_channel, outplanes=stage_3_channel, res_conv=res_conv, stride=s, groups=1))
            if f'nodeF_trans_steps_{i}' in self.mygraph:
                self.add_module('conv_trans_steps_' + str(i),
                    TransBlock(stage_3_channel, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim, 
                            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                            qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
                            drop_path_rate=self.trans_dpr[i-1]
                            )
            )

            if num_med_block > 0:
                for j in range(num_med_block):
                    if f'nodeG_medblock_{j}_{i}' in self.mygraph:
                        self.add_module('conv_trans_medblock_' + str(j) + '_' + str(i),
                        Med_ConvBlock(inplanes=stage_3_channel, groups=1)
                        )
            
            if f'nodeH_fusion_{i}' in self.mygraph:
                if last_fusion:
                    self.add_module('conv_trans_fusion_' + str(i),
                        ConvBlock(inplanes=in_channel, outplanes=stage_3_channel, res_conv=res_conv, stride=s, groups=1)
                    )
                else:
                    self.add_module('conv_trans_fusion_' + str(i),
                        ConvBlock(inplanes=stage_3_channel, outplanes=stage_3_channel, groups=1)
                    )
            
        self.fin_stage = fin_stage

        

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}


    def forward(self, input_list):
        
        return_list = {}
        if 'nodeA' in self.mygraph:
            B = input_list['x'].shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1) #nodeA
            return_list['cls_tokens'] = cls_tokens
        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        if 'nodeB' in self.mygraph:
            x_base = self.maxpool(self.act1(self.bn1(self.conv1(input_list['x'])))) #nodeB
            return_list['x_base'] = x_base
        # 1 stage
        if 'nodeC' in self.mygraph:
            x = self.conv_1(input_list['x_base'], return_x_2=False) #nodeC
            return_list['x'] = x
        if 'nodeD' in self.mygraph:
            x_t = self.trans_patch_conv(input_list['x_base']).flatten(2).transpose(1, 2) #nodeD
            return_list['x_t'] = x_t
        if 'nodeE' in self.mygraph:
            x_t = torch.cat([input_list['cls_tokens'], input_list['x_t']], dim=1)
            x_t = self.trans_1(x_t) 
            return_list['x_t'] = x_t
        # 2 ~ final 
        for i in range(2, self.fin_stage):
            # x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)
            if f'nodeF_step1_{i}' in self.mygraph:
                x, x2 = eval('self.conv_trans_step1_' + str(i))(input_list['x'])
                return_list['x'] = x
                return_list['x2'] = x2
            if f'nodeF_trans_steps_{i}' in self.mygraph:
                x_t, x_t_r = eval('self.conv_trans_steps_' + str(i))(input_list['x2'], input_list['x_t'])
                return_list['x_t'] = x_t
                return_list['x_t_r'] = x_t_r
            if self.num_med_block  > 0:
                for j in range(self.num_med_block):
                    if f'nodeG_medblock_{j}_{i}' in self.mygraph:
                        x = eval('self.conv_trans_medblock_' + str(j) + '_' + str(i))(input_list['x'])
                        return_list['x'] = x  
            if f'nodeH_fusion_{i}' in self.mygraph:
                x = eval('self.conv_trans_fusion_' + str(i))(input_list['x'], input_list['x_t_r'], return_x_2=False)
                return_list['x'] = x

        # conv classification
        if 'nodeI_conv_head' in self.mygraph:
            x_p = self.pooling(input_list['x']).flatten(1)
            conv_cls = self.conv_cls_head(x_p)
            return_list['conv_cls'] = conv_cls

        # trans classification
        if 'nodeI_trans_head' in self.mygraph:
            x_t = self.trans_norm(input_list['x_t'])
            tran_cls = self.trans_cls_head(x_t[:, 0])
            return_list['tran_cls'] = tran_cls
        
        # [conv_cls, tran_cls]
        return return_list
