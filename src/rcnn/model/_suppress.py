"""Suppression layer, including score thresholding and NMS."""
import tensorflow as tf

MIN_SCORE = -9999.0


def nms_pad(
    bx: tf.Tensor,
    scores: tf.Tensor,
    n_nms: int,
    nms_th: float,
) -> tf.Tensor:
    """Get NMS index for a batch of images with -1 padding.

    CANNOT use `tf.image.non_max_suppression` as it outputs a tensor with
    dynamic shape

    Args:
        bx (tf.Tensor): boxes to perform NMS on. Shape [B, N, 4].
        scores (tf.Tensor): scores to perform NMS on. Shape [B, N].
        n_nms (int): number of boxes to keep after NMS.
        nms_th (float): NMS threshold.

    Returns:
        tf.Tensor: indices of the boxes to keep after NMS. Shape [B, n_nms]
    """
    bx_dummy = tf.zeros_like(bx[:, 0:1, :])
    bx = tf.concat([bx_dummy, bx], axis=1)
    scores_dummy = tf.ones_like(scores[:, 0:1]) * MIN_SCORE
    scores = tf.concat([scores_dummy, scores], axis=1)
    selected_indices, _ = tf.image.non_max_suppression_padded(
        bx,
        scores,
        n_nms,
        iou_threshold=nms_th,
        pad_to_max_output_size=True,
    )
    return selected_indices - 1


def suppress(
    bx: tf.Tensor,
    log: tf.Tensor,
    n_score: int,
    n_nms: int,
    nms_th: float,
) -> tf.Tensor:
    """Suppression Block, including score thresholding and NMS.

    It receives the RPN logits and RoIs and produces the suppressed Region of
    Interests (RoI).

    Args:
        bx (tf.Tensor): RoI bounding box. Shape [B, N_val_ac, 4].
        log (tf.Tensor): RoI logits. Shape [B, N_val_ac, 1].
        n_score (int): number of top scores to keep.
        n_nms (int): number of RoIs to keep after NMS.
        nms_th (float): NMS threshold.

    Returns:
        tf.Tensor: proposed Region of Interests (RoIs) [B, n_nms, 4]
    """
    # score thresholding
    scores = tf.squeeze(log, axis=-1)  # (B, N_val_ac)
    idx_topk = tf.math.top_k(scores, k=n_score).indices
    score_topk = tf.gather(scores, idx_topk, batch_dims=1)  # B, n_score
    roi_topk = tf.gather(bx, idx_topk, batch_dims=1)  # B, n_score, 4
    # non-maximum suppression
    idx_nms = nms_pad(roi_topk, score_topk, n_nms, nms_th)  # (B, n_nms)
    # fetch the RoIs; -1 will result in (0., 0., 0., 0.)
    roi_nms = tf.gather(roi_topk, idx_nms, batch_dims=1)  # (B, n_nms, 4)
    # for shape inference
    return tf.reshape(roi_nms, (-1, n_nms, 4))
