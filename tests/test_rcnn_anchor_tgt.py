"""Test `anchor._tgt` module."""
from dataclasses import dataclass

import pytest
import tensorflow as tf

from rcnn.anchor import get_gt_box
from rcnn.anchor._tgt import _sample_mask

EPS = 1e-4


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
def test_anchor_sample_mask(data: Data) -> None:
    """Test `anchor._get_gt_box`."""
    mask_orig = _sample_mask(data.mask, data.num)
    assert tf.reduce_all(tf.equal(mask_orig, data.mask))
    mask_samp = _sample_mask(data.mask, data.num - 1)
    assert tf.reduce_sum(mask_samp) == data.num - 1


def test_anchor_get_gt_box() -> None:
    """Test `anchor.get_gt_box`.

    TODO: Add more test cases.
    """
    ac = tf.constant(
        [
            [0.00536, 0.04197, 0.18214, 0.39553],
            [0.10447, 0.17770, 0.45803, 0.88480],
            [0.15625, 0.28125, 0.65625, 0.78125],
            [0.44286, 0.16697, 0.61964, 0.52053],
            [0.47947, 0.00536, 0.83303, 0.18214],
            [0.65625, 0.09375, 0.90625, 0.34375],
        ],
        dtype=tf.float32,
    )
    bx_ac = tf.stack([ac, ac], axis=0)

    bx_gt = tf.constant(
        [
            [
                [0.12214, 0.13633, 0.48070, 0.72344],
                [0.36673, 0.66955, 0.84451, 0.74650],
                [0.18636, 0.99865, 0.66122, 0.60031],
                [0.44270, 0.42613, 0.94372, 0.93281],
                [0.88695, 0.12983, 0.66826, 0.60736],
                [0.27063, 0.64531, 0.42411, 0.94157],
                [0.00732, 0.83720, 0.30758, 0.80999],
                [0.41412, 0.34623, 0.56122, 0.10012],
                [0.20587, 0.12901, 0.89896, 0.82086],
                [0.07143, 0.63325, 0.36864, 0.93287],
                [0.34293, 0.40313, 0.63463, 0.18864],
            ],
            [
                [0.38862, 0.69712, 0.58034, 0.82260],
                [0.11028, 0.60969, 0.30264, 0.98425],
                [0.98968, 0.35481, 0.93476, 0.12806],
                [0.39919, 0.32615, 0.22796, 0.75751],
                [0.01344, 0.15832, 0.25190, 0.84047],
                [0.28441, 0.59504, 0.08894, 0.91351],
                [0.41829, 0.42219, 0.17677, 0.20744],
                [0.75593, 0.74215, 0.28857, 0.85431],
                [0.62988, 0.10834, 0.38291, 0.72056],
                [0.90610, 0.86687, 0.22143, 0.52384],
                [0.86183, 0.91191, 0.18030, 0.70272],
            ],
        ],
        dtype=tf.float32,
    )

    bx_exp = tf.constant(
        [
            [
                [0.34293, 0.40313, 0.63463, 0.18864],
                [0.12214, 0.13633, 0.4807, 0.72344],
                [0.20587, 0.12901, 0.89896, 0.82086],
                [0.34293, 0.40313, 0.63463, 0.18864],
                [0.34293, 0.40313, 0.63463, 0.18864],
                [0.34293, 0.40313, 0.63463, 0.18864],
            ],
            [
                [-1., -1., -1., -1.],
                [0.01344, 0.15832, 0.2519, 0.84047],
                [0.38862, 0.69712, 0.58034, 0.8226],
                [-1., -1., -1., -1.],
                [-1., -1., -1., -1.],
                [0.28441, 0.59504, 0.08894, 0.91351],
            ],
        ],
        dtype=tf.float32,
    )

    bx_tgt = get_gt_box(bx_ac, bx_gt)
    assert tf.reduce_sum(tf.abs(bx_tgt - bx_exp)) < EPS
