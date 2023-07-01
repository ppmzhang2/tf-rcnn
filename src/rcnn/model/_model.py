"""All models are defined here."""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from src.rcnn import cfg
from src.rcnn.model._rpn import RPN

__all__ = [
    "mdl_rpn",
]


def get_rpn_model() -> Model:
    layer_in = Input(shape=(cfg.H, cfg.W, 3))
    vgg = VGG16(include_top=False)
    rpn = RPN()
    vgg_out = vgg(layer_in)
    rpn_out = rpn(vgg_out)
    mdl = Model(inputs=layer_in, outputs=rpn_out)
    return mdl


mdl_rpn = get_rpn_model()
