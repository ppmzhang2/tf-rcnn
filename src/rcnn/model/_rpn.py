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
        anchors_per_location: int,
        anchor_stride: int,
        **kwargs: tf.Tensor,
    ):
        super().__init__(**kwargs)

        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride

        # R-CNN paper uses 512 filters for the VGG backbone
        self._conv2d = Conv2D(
            512,
            (3, 3),
            padding="same",
            activation="relu",
            name="rpn_conv",
        )

        # Regression layer for the bounding box coordinates
        self.regression_layers = Conv2D(anchors_per_location * 4, (1, 1),
                                        activation="linear",
                                        name="rpn_bbox")

        # Classification layer to predict the foreground or background
        self.classification_layers = Conv2D(anchors_per_location, (1, 1),
                                            activation="sigmoid",
                                            name="rpn_cls")

    def call(self, inputs: tf.Tensor) -> list[tf.Tensor]:
        # Shared convolutional base
        shared = self._conv2d(inputs)

        # Regression layer
        box_regression = self.regression_layers(shared)

        # Classification layer
        box_classification = self.classification_layers(shared)

        return [box_regression, box_classification]
