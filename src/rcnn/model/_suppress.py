"""Suppression layer, including score thresholding and NMS."""
import tensorflow as tf


def nms(inputs: tuple[tf.Tensor, tf.Tensor, int, float]) -> tf.Tensor:
    """Non-Maximum Suppression (NMS) index for a batch of images.

    Args:
        inputs (tuple[tf.Tensor, tf.Tensor, int, float]):
            - boxes (tf.Tensor): boxes to perform NMS on. Shape [B, N, 4].
            - scores (tf.Tensor): scores to perform NMS on. Shape [B, N].
            - n_nms (int): number of boxes to keep after NMS.
            - nms_th (float): NMS threshold.

    Returns:
        tf.Tensor: indices of the boxes to keep after NMS. Shape [B, M] where
            M <= n_nms.
    """
    boxes, scores, n_nms, nms_th = inputs

    def _nms(args: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """NMS index for a single image.

        Args:
            args (tuple[tf.Tensor, tf.Tensor]): boxes and scores for a single
                image. Shape [N, 4] and [N, ].

        Returns:
            tf.Tensor: indices of the boxes to keep.
                Shape [M, ] where M <= n_nms.
        """
        boxes = args[0]
        scores = args[1]
        return tf.image.non_max_suppression(
            boxes,
            scores,
            n_nms,
            nms_th,
        )

    return tf.map_fn(
        fn=_nms,
        elems=(boxes, scores),
        fn_output_signature=tf.TensorSpec(shape=(None, ), dtype=tf.int32),
    )


def suppress(
    rpn_prob: tf.Tensor,
    rpn_roi: tf.Tensor,
    n_score: int,
    n_nms: int,
    nms_th: float,
) -> tf.Tensor:
    """Suppression Block, including score thresholding and NMS.

    It receives the RPN logits and RoIs and produces the suppressed Region of
    Interests (RoI).

    Args:
        rpn_prob (tf.Tensor): RPN classification predictions.
            Shape [B, N_val_ac, 1].
        rpn_roi (tf.Tensor): Region of Interests (RoIs) bounding box.
            Shape [B, N_val_ac, 4].
        n_score (int): number of top scores to keep.
        n_nms (int): number of RoIs to keep after NMS.
        nms_th (float): NMS threshold.

    Returns:
        tf.Tensor: proposed Region of Interests (RoIs) [B, n_nms, 4]
    """
    # score thresholding
    scores = tf.squeeze(rpn_prob, axis=-1)  # (B, N_val_ac)
    top_score_idx = tf.math.top_k(scores, k=n_score).indices
    top_score = tf.gather(scores, top_score_idx, batch_dims=1)  # B, n_score
    top_roi = tf.gather(rpn_roi, top_score_idx, batch_dims=1)  # B, n_score, 4
    # nms: wrap the `tf.image.non_max_suppression` function as dynamic shape
    # compatibility is not supported.
    # TODO: handle varying numbers of RoIs per image with padding
    nms_idx = tf.keras.layers.Lambda(nms)((top_roi, top_score, n_nms, nms_th))
    # 2. gather top nms
    nms_roi = tf.gather(top_roi, nms_idx, batch_dims=1)  # (B, n_nms, 4)
    return nms_roi
