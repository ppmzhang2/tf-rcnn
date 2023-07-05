"""Region Proposal Block."""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from rcnn import cfg


class ProposalBlock(tf.keras.layers.Layer):
    """Region Proposal Block.

    It accepts a feature map from a backbone network as input and outputs
    bounding box regression predictions and classification predictions.
    """

    def __init__(self, **kwargs):
        """Initialize the RPN.

        Args:
            kwargs: Other keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

        _init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        _regu = tf.keras.regularizers.l2(0.0005)

        # R-CNN paper uses 512 filters for the VGG backbone
        self._conv2d = Conv2D(
            512,
            (3, 3),
            kernel_initializer=_init,
            padding="same",
            activation="relu",
            name="rpn_conv",
        )
        # Regression layer for the bounding box coordinates
        self.reg_layer = Conv2D(cfg.N_ANCHOR * 4, (1, 1),
                                kernel_initializer=_init,
                                kernel_regularizer=_regu,
                                activation=None,
                                name="rpn_bbox")
        # Classification layer to predict the foreground or background
        self.cls_layer = Conv2D(cfg.N_ANCHOR, (1, 1),
                                kernel_initializer=_init,
                                kernel_regularizer=_regu,
                                activation=None,
                                name="rpn_cls")

    def call(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Forward pass.

        Args:
            inputs (tf.Tensor): Feature map from the backbone network.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Bounding box regression predictions
            and classification predictions.
            The first tensor has shape [n_batch, N, 4] and the second tensor
            has shape [n_batch, N, 1], where `N` is the number of all anchors
            in the feature map.
            e.g. for a 1024 by 768 image of stride 32,
            `N = 1024 / 32 * 768 / 32 * 9 = 6912`.
        """
        n_batch = tf.shape(inputs)[0]
        shared = self._conv2d(inputs)  # Shared convolutional base
        box_reg = self.reg_layer(shared)  # coordinate regression
        lbl_cls = self.cls_layer(shared)  # label classification
        return (
            tf.reshape(box_reg, (n_batch, -1, 4)),  # [n_batch, N, 4]
            tf.reshape(lbl_cls, (n_batch, -1, 1)),  # [n_batch, N, 1]
        )
