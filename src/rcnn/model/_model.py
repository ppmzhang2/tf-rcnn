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
    """Get the RPN model.

    Returns:
        tf.keras.Model: The RPN model which outputs the following:
            - dlt: The delta of the RPN (B, N_VAL_AC, 4).
            - log: The logit of the RPN (B, N_VAL_AC, 1).
            - sup_box: The suppressed bounding boxes (B, N_SUPP_NMS, 4).

    """
    dlt, log = rpn(vgg16.output, cfg.H_FM, cfg.W_FM)
    bbx = roi(dlt)
    sup_box = suppress(bbx, log, cfg.N_SUPP_SCORE, cfg.N_SUPP_NMS, cfg.NMS_TH)
    mdl = tf.keras.Model(inputs=vgg16.input, outputs=[dlt, log, sup_box])
    return mdl
