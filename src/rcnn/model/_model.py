"""All models are defined here."""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from src.rcnn import cfg
from src.rcnn.model._proposal import ProposalBlock
from src.rcnn.model._roi import RoiBlock

__all__ = [
    "get_rpn_model",
]


def get_rpn_model() -> Model:
    n_score = 1000
    n_nms = 10
    nms_th = 0.7
    layer_in = Input(shape=(cfg.H, cfg.W, cfg.C))
    vgg = VGG16(include_top=False)
    rpn = ProposalBlock()
    feature_map = vgg(layer_in)
    rpn_box, rpn_cls = rpn(feature_map)
    roi = RoiBlock(n_score, n_nms, nms_th)
    roi_box = roi(rpn_cls, rpn_box)
    mdl = Model(inputs=layer_in, outputs=[roi_box, rpn_box, rpn_cls])
    return mdl
