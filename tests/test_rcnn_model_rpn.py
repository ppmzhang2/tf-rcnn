"""Test the RPN model."""
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from rcnn.model import RPN


def setup_model() -> Model:
    """Set up the model."""
    # model = VGG16 + RPN
    input_layer = Input(shape=(416, 416, 3))
    vgg = VGG16(include_top=False)
    rpn = RPN(9)
    vgg_out = vgg(input_layer)
    rpn_out = rpn(vgg_out)
    model = Model(inputs=input_layer, outputs=rpn_out)
    model.compile(optimizer="adam", loss="mse")
    return model


def setup_io() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Set up the inputs / outputs."""
    b = 16  # batch size
    inputs = tf.random.uniform((b, 416, 416, 3))  # (B, H, W, C)
    boxes = tf.random.uniform((b, 13, 13, 36))  # (B, H_feat, W_feat, 9 * 4)
    labels = tf.random.uniform((b, 13, 13, 9))  # (B, H_feat, W_feat, 9)
    return (
        inputs,
        tf.reshape(boxes, (b, -1, 4)),
        tf.reshape(labels, (b, -1, 1)),
    )


def test_rpt_training() -> None:
    """Test the RPN model training."""
    model = setup_model()
    inputs, boxes, labels = setup_io()
    history = model.fit(inputs, [boxes, labels], epochs=1)
    assert history.history["loss"][0] > 0


def test_rpt_predict() -> None:
    """Test the RPN model output."""
    model = setup_model()
    inputs, boxes, labels = setup_io()
    output = model.predict(inputs)
    assert output[0].shape == boxes.shape
    assert output[1].shape == labels.shape
