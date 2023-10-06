"""All models are defined here."""
import tensorflow as tf

from rcnn import cfg
from rcnn.model._base import get_vgg16
from rcnn.model._roi import roi
from rcnn.model._rpn import rpn
from rcnn.model._suppress import suppress

__all__ = [
    "get_rpn_model",
]

vgg16 = get_vgg16(cfg.H, cfg.W)


def get_rpn_model() -> tf.keras.Model:
    dlt, log = rpn(vgg16.output, cfg.H_FM, cfg.W_FM)
    bbx = roi(dlt)
    sup_box = suppress(bbx, log, cfg.N_SUPP_SCORE, cfg.N_SUPP_NMS, cfg.NMS_TH)
    mdl = tf.keras.Model(inputs=vgg16.input, outputs=[sup_box, log, dlt, bbx])
    return mdl
