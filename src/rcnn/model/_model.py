"""All models are defined here."""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from rcnn import cfg
from rcnn.model._proposal import ProposalBlock
from rcnn.model._roi import RoiBlock
from rcnn.model._suppress import SuppressBlock

__all__ = [
    "get_rpn_model",
]


def get_rpn_model() -> Model:
    layer_in = Input(shape=(cfg.H, cfg.W, cfg.C))
    vgg_block = VGG16(include_top=False)
    rpn_block = ProposalBlock()
    roi_block = RoiBlock()
    sup_block = SuppressBlock(cfg.N_SUPP_SCORE, cfg.N_SUPP_NMS, cfg.NMS_TH)
    feature_map = vgg_block(layer_in)
    rpn_del, rpn_cls = rpn_block(feature_map)
    roi_cls, roi_del, roi_box = roi_block(rpn_cls, rpn_del)
    sup_box = sup_block(roi_cls, roi_box)
    mdl = Model(inputs=layer_in, outputs=[sup_box, roi_cls, roi_del, roi_box])
    return mdl
