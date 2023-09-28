"""All models are defined here."""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from rcnn import cfg
from rcnn.model._roi import roi
from rcnn.model._rpn import rpn
from rcnn.model._suppress import suppress

__all__ = [
    "get_rpn_model",
]

vgg = VGG16(include_top=False)


def get_rpn_model() -> Model:
    layer_in = Input(shape=(cfg.H, cfg.W, cfg.C))
    feature_map = vgg(layer_in)
    rpn_dlt, rpn_log = rpn(feature_map, cfg.H_FM, cfg.W_FM)
    roi_cls, roi_dlt, roi_box = roi(rpn_log, rpn_dlt)
    sup_box = suppress(roi_cls, roi_box, cfg.N_SUPP_SCORE, cfg.N_SUPP_NMS,
                       cfg.NMS_TH)
    mdl = Model(inputs=layer_in, outputs=[sup_box, roi_cls, roi_dlt, roi_box])
    return mdl
