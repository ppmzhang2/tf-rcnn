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
        shared = self._conv2d(inputs)  # Shared convolutional base
        box_reg = self.reg_layer(shared)  # coordinate regression
        lbl_cls = self.cls_layer(shared)  # label classification

        return [box_reg, lbl_cls]
