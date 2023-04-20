import math
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, get_norm
from detectron2.config import configurable
from ..layers.deform_conv import DFConv2d
import copy
# from mmcv.cnn import ConvModule, bias_init_with_prob
# from mmcv.ops import CornerPool, batched_nms
# from mmcv.runner import BaseModule
# from ..layers.CC import CC_module # TODO CCA

__all__ = ["HumanLikerHead"]

# TODO corner_pooling
# class BiCornerPool(BaseModule):
#     """Bidirectional Corner Pooling Module (TopLeft, BottomRight, etc.)
#
#     Args:
#         in_channels (int): Input channels of module.
#         out_channels (int): Output channels of module.
#         feat_channels (int): Feature channels of module.
#         directions (list[str]): Directions of two CornerPools.
#         norm_cfg (dict): Dictionary to construct and config norm layer.
#         init_cfg (dict or list[dict], optional): Initialization config dict.
#             Default: None
#     """
#
#     def __init__(self,
#                  in_channels,
#                  directions,
#                  feat_channels=128,
#                  out_channels=128,
#                  norm_cfg=dict(type='BN', requires_grad=True),
#                  # norm_cfg=dict(type='GN', num_groups=25),
#                  init_cfg=None):
#         super(BiCornerPool, self).__init__(init_cfg)
#         self.direction1_conv = ConvModule(
#             in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)
#         self.direction2_conv = ConvModule(
#             in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)
#
#         self.aftpool_conv = ConvModule(
#             feat_channels,
#             out_channels,
#             3,
#             padding=1,
#             norm_cfg=norm_cfg,
#             act_cfg=None)
#
#         self.conv1 = ConvModule(
#             in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
#         self.conv2 = ConvModule(
#             in_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg)
#
#         self.direction1_pool = CornerPool(directions[0])
#         self.direction2_pool = CornerPool(directions[1])
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         """Forward features from the upstream network.
#
#         Args:
#             x (tensor): Input feature of BiCornerPool.
#
#         Returns:
#             conv2 (tensor): Output feature of BiCornerPool.
#         """
#         direction1_conv = self.direction1_conv(x)
#         direction2_conv = self.direction2_conv(x)
#         direction1_feat = self.direction1_pool(direction1_conv)
#         direction2_feat = self.direction2_pool(direction2_conv)
#         aftpool_conv = self.aftpool_conv(direction1_feat + direction2_feat)
#         conv1 = self.conv1(x)
#         relu = self.relu(aftpool_conv + conv1)
#         conv2 = self.conv2(relu)
#         return conv2



class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class HumanLikerHead(nn.Module):
    @configurable
    def __init__(self, 
        # input_shape: List[ShapeSpec],
        in_channels,
        num_levels,
        *,
        num_classes=80,
        norm='GN',
        num_cls_convs=4,
        num_box_convs=4,
        num_share_convs=0,
        use_deformable=False,
        prior_prob=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.out_kernel = 3

        head_configs = {
            "cls": (0, use_deformable),
            "bbox": (num_box_convs, use_deformable),
            "share": (num_share_convs, use_deformable)}

        channels = {
            'cls': in_channels,
            'bbox': in_channels,
            'share': in_channels,
        }
        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            channel = channels[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d
                tower.append(conv_func(
                        in_channels if i == 0 else channel,
                        channel, 
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                ))
                if norm == 'GN' and channel % 32 != 0:
                    tower.append(nn.GroupNorm(25, channel))
                elif norm != '':
                    tower.append(get_norm(norm, channel))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        # self.cc_att = CC_module(in_channels)
        # self.tl_pool = BiCornerPool(in_channels, ['top', 'left'], out_channels=in_channels)
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )
        self.offset = nn.Conv2d(
            in_channels, 4, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(num_levels)])

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower,
            self.bbox_pred,
            self.offset,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        
        torch.nn.init.constant_(self.bbox_pred.bias, 8.)
        torch.nn.init.constant_(self.offset.bias, 1.)
        prior_prob = prior_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        self.tl_angle_hm = nn.Conv2d(
            in_channels, 1, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )
        torch.nn.init.constant_(self.tl_angle_hm.bias, bias_value)
        torch.nn.init.normal_(self.tl_angle_hm.weight, std=0.01)


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            # 'input_shape': input_shape,
            'in_channels': [s.channels for s in input_shape][0],
            'num_levels': len(input_shape),
            'num_classes': cfg.MODEL.HUMANLIKER.NUM_CLASSES,
            'norm': cfg.MODEL.HUMANLIKER.NORM,
            'num_cls_convs': cfg.MODEL.HUMANLIKER.NUM_CLS_CONVS,
            'num_box_convs': cfg.MODEL.HUMANLIKER.NUM_BOX_CONVS,
            'num_share_convs': cfg.MODEL.HUMANLIKER.NUM_SHARE_CONVS,
            'use_deformable': cfg.MODEL.HUMANLIKER.USE_DEFORMABLE,
            'prior_prob': cfg.MODEL.HUMANLIKER.PRIOR_PROB,
        }
        return ret

    def forward(self, x):
        clss = []
        bbox_reg = []
        tl_angle_hms = []
        offsets = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            clss.append(None)
            # print('shape:', bbox_tower.size())
            tl_angle_hms.append(self.tl_angle_hm(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            reg = self.scales[l](reg)
            offset = self.offset(bbox_tower)
            offset = self.scales[l](offset)

            bbox_reg.append(F.relu(reg))
            offsets.append(offset)
        
        return clss, bbox_reg, tl_angle_hms, offsets