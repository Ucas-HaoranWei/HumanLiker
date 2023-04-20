from .modeling.meta_arch.humanliker_detector import HumanLikerDetector
from .modeling.dense_heads.humanliker import HumanLiker
from .modeling.roi_heads.custom_roi_heads import CustomROIHeads, CustomCascadeROIHeads

from .modeling.backbone.fpn_p5 import build_p67_resnet_fpn_backbone
from .modeling.backbone.res2net import build_p67_res2net_fpn_backbone
from .modeling.backbone.swin_transformer import build_swint_fpn_backbone

from .data.datasets.coco import _PREDEFINED_SPLITS_COCO

