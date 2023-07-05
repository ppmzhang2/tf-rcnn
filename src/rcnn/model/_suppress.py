"""Suppression Block, including score thresholding and NMS."""
import tensorflow as tf


def suppress(
    rpn_prob: tf.Tensor,
    rpn_roi: tf.Tensor,
    n_score: int,
    n_nms: int,
    nms_th: float,
) -> tf.Tensor:
    """Suppress Region of Interests (RoIs) based on score thresholding and NMS.

    Args:
        rpn_prob (tf.Tensor): RPN classification predictions.
            Shape [N_val_ac, 1].
        rpn_roi (tf.Tensor): Region of Interests (RoIs) bounding box.
            Shape [N_val_ac, 4].
        n_score (int): number of top scores to keep.
        n_nms (int): number of RoIs to keep after NMS.
        nms_th (float): NMS threshold.

    Returns:
        tf.Tensor: proposed Region of Interests (RoIs).
    """
    # score thresholding
    scores = tf.squeeze(rpn_prob, axis=-1)  # (N_val_ac,)
    top_score_idx = tf.math.top_k(scores, k=n_score).indices
    top_score = tf.gather(scores, top_score_idx)  # (n_score,)
    top_roi = tf.gather(rpn_roi, top_score_idx)  # (n_score, 4)
    # nms
    nms_idx = tf.image.non_max_suppression(top_roi, top_score, n_nms,
                                           nms_th)  # (n_nms,)
    rois = tf.gather(top_roi, nms_idx)  # (n_nms, 4)
    return rois


class SuppressBlock(tf.keras.layers.Layer):
    """Suppression Block, including score thresholding and NMS.

    It receives the RPN logits and RoIs and produces the suppressed Region of
    Interests (RoI).
    """

    def __init__(self, n_score: int, n_nms: int, nms_th: float, **kwargs):
        """Initialize the Block.

        Args:
            n_score (int): number of top scores to keep.
            n_nms (int): number of RoIs to keep after NMS.
            nms_th (float): NMS threshold.
            kwargs: Other keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._n_score = n_score
        self._n_nms = n_nms
        self._nms_th = nms_th

    def call(self, rpn_prob: tf.Tensor, rpn_roi: tf.Tensor) -> tf.Tensor:
        """Get proposed Region of Interests (RoIs) of a batch of images.

        Args:
            rpn_prob (tf.Tensor): RPN classification predictions.
                Shape [B, N_val_ac, 1].
            rpn_roi (tf.Tensor): Region of Interests (RoIs) bounding box.
                Shape [B, N_val_ac, 4].

        Returns:
            tf.Tensor: proposed Region of Interests (RoIs) [B, n_nms, 4]
        """
        rois = tf.map_fn(
            lambda x: suppress(
                x[0],
                x[1],
                self._n_score,
                self._n_nms,
                self._nms_th,
            ),
            (rpn_prob, rpn_roi),
            fn_output_signature=tf.TensorSpec([None, 4], tf.float32),
        )
        return rois
