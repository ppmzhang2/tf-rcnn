"""Test data.rpn module."""
from dataclasses import dataclass

import pytest
import tensorflow as tf

from rcnn.data import rpn


@dataclass
class Data:
    """Data for testing."""
    mask: tf.Tensor
    num: int


def rand_mask(n_total: int, n_one: int) -> tf.Tensor:
    """Generate a random mask."""
    idx_one = tf.random.shuffle(tf.range(
        n_total,
        dtype=tf.int32,
    ))[:n_one][:, tf.newaxis]  # (n_one, 1) of values in [0, n_total)
    return tf.scatter_nd(
        idx_one,
        tf.ones((n_one, ), dtype=tf.float32),
        (n_total, ),
    )


@pytest.mark.parametrize("data", [
    Data(mask=rand_mask(100, 5), num=5),
    Data(mask=rand_mask(1000, 20), num=20),
])
def test_data_rpn_sample_mask(data: Data) -> None:
    """Test risk.align_mask."""
    mask_orig = rpn._sample_mask(data.mask, data.num)
    assert tf.reduce_all(tf.equal(mask_orig, data.mask))
    mask_samp = rpn._sample_mask(data.mask, data.num - 1)
    assert tf.reduce_sum(mask_samp) == data.num - 1


def test_data_rpn_get_gt_box() -> None:
    """Test rpn._get_gt_box.

    TODO: Add more test cases.
    """
    # Test for positive case
    bx_gt = tf.constant([[10, 10, 50, 50]], dtype=tf.float32)
    bx_ac = tf.constant(
        [[15, 15, 45, 45], [20, 20, 35, 35]],
        dtype=tf.float32,
    )
    exp = tf.constant([[10, 10, 50, 50], [-1, -1, -1, -1]], dtype=tf.float32)

    bx_tgt = rpn._get_gt_box(bx_ac, bx_gt)
    assert tf.reduce_all(tf.equal(bx_tgt, exp))
