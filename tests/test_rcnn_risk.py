"""Test Faster R-CNN risk functions."""
from dataclasses import dataclass

import pytest
import tensorflow as tf

from rcnn import risk


@dataclass
class Data:
    """Data for testing."""
    mask_obj: tf.Tensor
    mask_bkg: tf.Tensor


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
    Data(mask_obj=rand_mask(100, 5), mask_bkg=rand_mask(100, 20)),
    Data(mask_obj=rand_mask(1000, 10), mask_bkg=rand_mask(1000, 40)),
])
def test_risk_align_mask(data: Data) -> None:
    """Test risk.align_mask."""
    mask_bkg_ = risk.align_mask(data.mask_obj, data.mask_bkg)
    assert tf.reduce_all(tf.equal(mask_bkg_ * data.mask_bkg, mask_bkg_))
    assert tf.reduce_sum(mask_bkg_) == tf.reduce_sum(data.mask_obj)
