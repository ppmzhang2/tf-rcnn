"""Suppression layer, including score thresholding and NMS."""
import tensorflow as tf


def nms_(inputs: tuple[tf.Tensor, tf.Tensor, int, float]) -> tf.Tensor:
    """Non-Maximum Suppression (NMS).
    
    Wrap the `tf.image.non_max_suppression` function as dynamic shape
    compatibility is not supported.
    """
    boxes, scores, n_nms, nms_th = inputs

    def single_image_nms(args: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """NMS for a single image."""
        boxes = args[0]
        scores = args[1]
        idx = tf.image.non_max_suppression(
            boxes,
            scores,
            n_nms,
            nms_th,
        )
        return idx

    return tf.map_fn(
        fn=single_image_nms,
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
    # nms
    # 1. sort by score
    nms_idx = tf.keras.layers.Lambda(nms_)((top_roi, top_score, n_nms, nms_th))
    # 2. gather top nms
    nms_roi = tf.gather(top_roi, nms_idx, batch_dims=1)  # (B, n_nms, 4)
    return nms_roi
