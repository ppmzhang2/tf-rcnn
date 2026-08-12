"""Test the RPN model."""
import pytest
import tensorflow as tf

from rcnn import anchor
from rcnn.model._suppress import nms_pad
from rcnn.model._suppress import suppress

BS = 2  # batch size
EPS = 1e-6  # epsilon


@pytest.mark.parametrize(
    ("n_dim", "seed"),
    [
        (10, 0),
        (20, 1),
        (30, 2),
    ],
)
def test_suppress_nms_pad(n_dim: int, seed: int) -> None:
    """Test `nms_pad`."""
    n_nms = 5
    iou_th = 0.5
    bx = tf.random.uniform((BS, n_dim, 4), minval=0., maxval=1., seed=seed)
    scores = tf.random.uniform((BS, n_dim), minval=0., maxval=1., seed=seed)
    idx_nms = nms_pad(bx, scores, n_nms, iou_th)  # -1 padded indices
    idx_res = idx_nms * tf.where(idx_nms < 0, 0, 1)
    idx_exp, _ = tf.image.non_max_suppression_padded(
        bx,
        scores,
        n_nms,
        iou_th,
        pad_to_max_output_size=True,
    )
    assert tf.reduce_all(tf.equal(idx_res, idx_exp))


def test_suppress_infer() -> None:
    """Test the shape inference of the Suppress layer."""
    input1 = tf.keras.layers.Input(shape=(anchor.N_RPNAC, 4))
    input2 = tf.keras.layers.Input(shape=(anchor.N_RPNAC, 1))
    n_score = 300
    n_nms = 20
    iou_th = 0.7
    outputs = suppress(input1, input2, n_score, n_nms, iou_th)
    assert outputs.shape == (None, n_nms, 4)


def test_suppress_predict() -> None:
    """Test the suppress layer output."""
    input1 = tf.keras.layers.Input(shape=(anchor.N_RPNAC, 4))
    input2 = tf.keras.layers.Input(shape=(anchor.N_RPNAC, 1))
    n_score = 300
    n_nms = 20
    iou_th = 0.7
    outputs = suppress(input1, input2, n_score, n_nms, iou_th)
    mdl = tf.keras.Model(inputs=[input1, input2], outputs=outputs)
    bbx = tf.random.uniform((BS, anchor.N_RPNAC, 4))
    scores = tf.random.uniform((BS, anchor.N_RPNAC, 1))
    rois = mdl([bbx, scores])
    assert rois.shape == (BS, n_nms, 4)
