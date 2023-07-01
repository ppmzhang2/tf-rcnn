"""Region Proposal Network (RPN) for Faster R-CNN."""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D


class RPN(tf.keras.Model):
    """Region Proposal Network (RPN) for Faster R-CNN.

    It accepts a feature map from a backbone network as input and outputs
    bounding box regression predictions and classification predictions.
    """

    def __init__(
        self,
        n_anchor: int,
        **kwargs,
    ):
        """Initialize the RPN.

        Args:
            n_anchor (int): Number of anchors per pixel location, usually 3*3.
            kwargs: Other keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._n_anchor = n_anchor

        # R-CNN paper uses 512 filters for the VGG backbone
        self._conv2d = Conv2D(
            512,
            (3, 3),
            padding="same",
            activation="relu",
            name="rpn_conv",
        )
        # Regression layer for the bounding box coordinates
        self.reg_layer = Conv2D(self._n_anchor * 4, (1, 1),
                                activation="linear",
                                name="rpn_bbox")
        # Classification layer to predict the foreground or background
        self.cls_layer = Conv2D(self._n_anchor, (1, 1),
                                activation="sigmoid",
                                name="rpn_cls")

    def call(self, inputs: tf.Tensor) -> list[tf.Tensor]:
        """Forward pass.

        Args:
            inputs (tf.Tensor): Feature map from the backbone network.

        Returns:
            list[tf.Tensor]: Bounding box regression predictions and
            classification predictions.
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
        return [
            tf.reshape(box_reg, (n_batch, -1, 4)),  # [n_batch, N, 4]
            tf.reshape(lbl_cls, (n_batch, -1, 1)),  # [n_batch, N, 1]
        ]
