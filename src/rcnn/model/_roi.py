"""Region of Interest (RoI) Block."""
import tensorflow as tf

from rcnn import box


def get_roi(rpn_del: tf.Tensor) -> tf.Tensor:
    """Get proposed Region of Interests (RoIs) of a single image.

    Args:
        rpn_del (tf.Tensor): RPN bounding box delta predictions.
            Shape [N_val_ac, 4].

    Returns:
        tf.Tensor: proposed Region of Interests (RoIs).
    """
    # Buffer to clip the RoIs. Defaults to 1e-1.
    buffer = 1e-1
    ac_val = tf.constant(box.val_anchors, dtype=tf.float32)  # (N_val_ac, 4)
    rois = box.delta2bbox(ac_val, rpn_del)  # (N_val_ac, 4)
    rois = tf.clip_by_value(rois, -buffer, 1. + buffer)  # (B, n_val, 4)
    return rois


class RoiBlock(tf.keras.layers.Layer):
    """Region of Interest (RoI) Block.

    It receives the RPN class and bounding box offset predictions and produces
    transformed results based on valid anchor masks.
    """

    def __init__(self, **kwargs):
        """Initialize the Block."""
        super().__init__(**kwargs)

    def call(
        self,
        rpn_cls: tf.Tensor,
        rpn_del: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get Region of Interests (RoIs) of a batch of images.

        Args:
            rpn_cls (tf.Tensor): RPN classification predictions.
                Shape [B, N_ac, 1].
            rpn_del (tf.Tensor): RPN bounding box delta predictions.
                Shape [B, N_ac, 4].

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: RPN classification,
                bounding box delta, and RoIs. All are filtered by valid anchor
                masks.
        """
        rpn_cls_val = tf.map_fn(
            lambda x: x[box.val_anchor_mask == 1],
            rpn_cls,
            fn_output_signature=tf.TensorSpec([None, 1], tf.float32),
        )  # (B, n_val_ac, 1)
        rpn_del_val = tf.map_fn(
            lambda x: x[box.val_anchor_mask == 1],
            rpn_del,
            fn_output_signature=tf.TensorSpec([None, 4], tf.float32),
        )  # (B, n_val_ac, 4)
        rois = tf.map_fn(
            get_roi,
            rpn_del_val,
            fn_output_signature=tf.TensorSpec([None, 4], tf.float32),
        )  # (B, n_val_ac, 4)
        return rpn_cls_val, rpn_del_val, rois
