"""Region of Interest (RoI) Block."""
import tensorflow as tf

from src.rcnn import box


class RoiBlock(tf.keras.Model):
    """RoI Block.

    It receives the RPN class and bounding box predictions and produces Region
    of Interests (RoI).
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

    def get_roi(self, rpn_prob: tf.Tensor, rpn_del: tf.Tensor) -> tf.Tensor:
        """Get proposed Region of Interests (RoIs) of a single image.

        Args:
            rpn_prob (tf.Tensor): RPN classification predictions.
                Shape [N_ac, 1].
            rpn_del (tf.Tensor): RPN bounding box delta predictions.
                Shape [N_ac, 4].

        Returns:
            tf.Tensor: proposed Region of Interests (RoIs).
        """
        scores = tf.squeeze(rpn_prob, axis=-1)  # (N_ac,)
        top_score_idx = tf.math.top_k(scores, k=self._n_score).indices
        top_score = tf.gather(scores, top_score_idx)  # (n_score,)
        top_delta = tf.gather(rpn_del, top_score_idx)  # (n_score, 4)
        top_ac = tf.gather(box.anchors, top_score_idx)  # (n_score, 4)
        rois_all = box.bbox2delta(top_ac, top_delta)  # (n_score, 4)
        # clip
        rois = box.bbox.clip(rois_all, 1., 1.)  # (n_score, 4)
        # nms
        nms_idx = tf.image.non_max_suppression(rois, top_score, self._n_nms,
                                               self._nms_th)  # (n_nms,)
        rois = tf.gather(rois, nms_idx)  # (n_nms, 4)
        return rois

    def call(self, rpn_prob: tf.Tensor, rpn_del: tf.Tensor) -> tf.Tensor:
        """Get proposed Region of Interests (RoIs) of a batch of images.

        Args:
            rpn_prob (tf.Tensor): RPN classification predictions.
                Shape [B, N_ac, 1].
            rpn_del (tf.Tensor): RPN bounding box delta predictions.
                Shape [B, N_ac, 4].

        Returns:
            tf.Tensor: proposed Region of Interests (RoIs) [B, n_nms, 4]
        """
        rois = tf.map_fn(self.get_roi, (rpn_prob, rpn_del), dtype=tf.float32)
        return rois
