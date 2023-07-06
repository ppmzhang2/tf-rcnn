"""Generate RPN targets for training."""
import tensorflow as tf

from rcnn import box

NEG_TH_ACGT = 0.30  # lower threshold for anchor-GT highest IoU
POS_TH_ACGT = 0.70  # upper threshold for anchor-GT highest IoU
NEG_TH_GTAC = 0.01  # lower threshold for GT-anchor highest IoU

NUM_POS_RPN = 128  # number of positive anchors
NUM_NEG_RPN = 128  # number of negative anchors

__all__ = [
    "get_gt_box",
    "get_gt_mask",
]


def num_pos_mask(mask: tf.Tensor) -> int:
    """Get number of positive anchors.

    Args:
        mask (tf.Tensor): 0/1 mask of anchors (N_ac,)

    Returns:
        int: number of positive anchors
    """
    return tf.get_static_value(tf.reduce_sum(tf.cast(mask, tf.int32)))


def _sample_mask(mask: tf.Tensor, num: int) -> tf.Tensor:
    """Sample `num` anchors from `mask`.

    Args:
        mask (tf.Tensor): 0/1 mask of anchors (N_ac,)
        num (int): number of positive anchors to sample

    Returns:
        tf.Tensor: 0/1 mask of anchors (N_ac,)
    """
    # return if number of positive anchors is less than `num`
    pos_num = num_pos_mask(mask)
    if pos_num <= num:
        return mask
    # 1. get indices of positive
    idx_pos = tf.where(mask > 0)  # (n_pos, 1), tf.int64
    # 2. randomly select `num` positive indices
    idx_pos = tf.random.shuffle(idx_pos)[:num]  # (n_obj, 1), tf.int64
    # 3. update mask with selected indices
    mask_ = tf.tensor_scatter_nd_update(
        tf.zeros_like(mask, dtype=mask.dtype),  # (N_ac,)
        idx_pos,  # (n_obj, 1)
        tf.ones((num, ), dtype=mask.dtype),  # (n_obj,)
    )  # (N_ac,)
    return mask_


def sample_mask(mask: tf.Tensor, num: int) -> tf.Tensor:
    """Sample `num` anchors from `mask` for a batch of images.

    Args:
        mask (tf.Tensor): 0/1 mask of anchors (B, N_ac)
        num (int): number of positive anchors to sample

    Returns:
        tf.Tensor: 0/1 mask of anchors (B, N_ac)
    """
    return tf.map_fn(
        lambda x: _sample_mask(x, num),  # (N_ac,) -> (N_ac,)
        mask,  # (B, N_ac)
        fn_output_signature=tf.TensorSpec(shape=(None, ), dtype=mask.dtype),
    )  # (B, N_ac)


def _get_gt_box(bx_ac: tf.Tensor, bx_gt: tf.Tensor) -> tf.Tensor:
    """Get ground truth boxes based on IoU for each anchor for a single image.

    Args:
        bx_ac (tf.Tensor): anchor tensor (N_ac, 4)
        bx_gt (tf.Tensor): ground truth tensor (N_gt, 4)

    Returns:
        tf.Tensor: ground truth boxes (N_ac, 4) for each anchor of tf.float32
          - [-1.0, -1.0, -1.0, -1.0]: background
          - [0.0, 0.0, 0.0, 0.0]: ignore
          - otherwise: foreground
    """
    # (N_ac, N_gt)
    # - *_ac_gt is longer than iou_gt_ac as it contains all anchors
    # - iou_ac_gt must be compared with thresholds
    ious = box.bbox.iou_mat(bx_ac, bx_gt)
    idx_ac_gt = tf.argmax(ious, axis=1)  # (N_ac,)
    iou_ac_gt = tf.reduce_max(ious, axis=1)  # (N_ac,)
    idx_gt_ac = tf.argmax(ious, axis=0)  # (N_gt,)
    iou_gt_ac = tf.reduce_max(ious, axis=0)  # (N_gt,)

    # target boxes
    # 1. GT's best AC Match With threshold
    #    - initialize with -1
    bx_tgt_gtac = -tf.ones_like(bx_ac)  # (N_ac, 4)
    #    - update with GT-anchor matches, where `idx_gt_ac` indicates indices
    #      of the best mached anchors for each GT boxes [0, N_gt)
    cond = tf.where(iou_gt_ac > NEG_TH_GTAC)  # (<N_gt, 1)
    bx_tgt_gtac = tf.tensor_scatter_nd_update(
        bx_tgt_gtac,  # (N_ac, 4)
        tf.gather(idx_gt_ac, cond),  # (<N_gt, 1) where value in [0, N_ac)
        tf.gather_nd(bx_gt, cond),  # (<N_gt, 4)
    )  # (N_ac, 4)
    # 2. AC's best GT Match
    #    - initialize with zeros
    bx_tgt_acgt = tf.zeros_like(bx_ac, dtype=tf.float32)  # (N_ac, 4)
    #    - update with AC-GT matches, where `idx_ac_gt` indicates indices of
    #      the best mached GT boxes for each anchors [0, N_ac).
    #      IoU lower than `NEG_TH_ACGT` tagged as background (-1.0);
    #      IoU higher than `POS_TH_ACGT` tagged as foreground (positive);
    #      any other IoU will be tagged as ignore (0.0)
    bx_tgt_acgt = tf.where(
        tf.repeat(iou_ac_gt[:, tf.newaxis], 4, axis=-1) < NEG_TH_ACGT,
        tf.constant([[-1.0]]),
        bx_tgt_acgt,
    )  # (N_ac, 4)
    bx_tgt_acgt = tf.where(
        tf.repeat(iou_ac_gt[:, tf.newaxis], 4, axis=-1) > POS_TH_ACGT,
        tf.gather(bx_gt, idx_ac_gt),  # (N_ac, 4)
        bx_tgt_acgt,
    )  # (N_ac, 4)
    # 3. combine 1 and 2
    bx_tgt = tf.where(bx_tgt_gtac >= 0, bx_tgt_gtac, bx_tgt_acgt)
    return bx_tgt


def get_gt_box(bx_gt: tf.Tensor) -> tf.Tensor:
    """Get ground truth boxes based on IoU for each anchor for RPN training.

    Args:
        bx_gt (tf.Tensor): ground truth tensor (B, N_gt, 4)

    Returns:
        tf.Tensor: ground truth boxes (B, N_ac, 4) for each anchor (tf.float32)
          - [-1.0, -1.0, -1.0, -1.0]: background
          - [0.0, 0.0, 0.0, 0.0]: ignore
          - otherwise: foreground
    """
    bx_ac = tf.constant(box.val_anchors, dtype=tf.float32)  # (N_ac, 4)
    return tf.map_fn(
        lambda x: _get_gt_box(bx_ac, x),  # (N_gt, 4) -> (N_ac, 4)
        bx_gt,  # (B, N_gt, 4)
        fn_output_signature=tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    )  # (B, N_ac, 4)


def get_gt_mask(bx_tgt: tf.Tensor, *, bkg: bool = False) -> tf.Tensor:
    """Get target mask for each anchor based on target boxes for RPN training.

    Args:
        bx_tgt (tf.Tensor): target ground truth boxes (B, N_ac, 4)
        bkg (bool, optional): whether to indicate background. Defaults False.

    Returns:
        tf.Tensor: 0/1 mask for each box (B, N_ac) for each anchor
    """
    # 0. coordinate sum of target boxes (B, N_ac):
    #    - positive: foreground
    #    - -4.0: background
    #    - 0.0: ignore
    _coor_sum = tf.reduce_sum(bx_tgt, axis=-1)
    # init with tf.float32 zeros
    mask = tf.zeros_like(_coor_sum, dtype=tf.float32)  # (B, N_ac)
    if bkg:
        mask = tf.where(_coor_sum < 0, 1.0, mask)
        return sample_mask(mask, NUM_NEG_RPN)
    mask = tf.where(_coor_sum > 0, 1.0, mask)
    return sample_mask(mask, NUM_POS_RPN)
