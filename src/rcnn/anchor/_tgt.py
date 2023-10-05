"""Generate RPN targets for training."""
import tensorflow as tf

from rcnn.anchor import _box

IOU_SCALE = 10000  # scale for IoU for converting to int
NEG_TH_ACGT = int(IOU_SCALE * 0.30)  # lower bound for anchor-GT highest IoU
POS_TH_ACGT = int(IOU_SCALE * 0.70)  # upper bound for anchor-GT highest IoU
NEG_TH_GTAC = int(IOU_SCALE * 0.01)  # lower bound for GT-anchor highest IoU

NUM_POS_RPN = 128  # number of positive anchors
NUM_NEG_RPN = 128  # number of negative anchors

__all__ = [
    "get_gt_box",
    "get_gt_mask",
]


def sample_mask(mask: tf.Tensor, num: int) -> tf.Tensor:
    """Sample `num` anchors from `mask` for a batch of images.

    Args:
        mask (tf.Tensor): 0/1 mask of anchors (B, N_ac)
        num (int): number of positive anchors to sample

    Returns:
        tf.Tensor: 0/1 mask of anchors (B, N_ac)
    """
    th = num / (tf.reduce_sum(mask, axis=-1, keepdims=True) + 1e-6)
    rand = tf.random.uniform(
        tf.shape(mask),
        minval=0,
        maxval=1,
        dtype=tf.float32,
    )
    return tf.cast(rand < th, tf.float32) * mask


def filter_on_max(x: tf.Tensor) -> tf.Tensor:
    """Filter rows based on the maximum value in the second column.

    Given a 2D tensor, the function filters the rows based on the following
    condition:
    For rows having the same value in the first column (key), retain only the
    row that has the maximum value in the second column (score).

    Args:
        x (tf.Tensor): 2D tensor (M, 2) of format (key, score)

    Returns:
        tf.Tensor: 1D boolean mask (M,) indicating which rows to keep
    """
    # Get unique values and their indices from the first column
    k_unique, k_idx = tf.unique(x[:, 0])

    # Find the maximum values in the second column for each unique element in
    # the first column
    max_indices = tf.math.unsorted_segment_max(data=x[:, 1],
                                               segment_ids=k_idx,
                                               num_segments=tf.size(k_unique))

    # Create a boolean mask where each max value is True, others are False
    return tf.reduce_any(x[:, 1][:, tf.newaxis] == max_indices, axis=-1)


def get_gt_gtac(
    bx_ac: tf.Tensor,
    bx_gt: tf.Tensor,
    idx: tf.Tensor,
    ious: tf.Tensor,
) -> tf.Tensor:
    """Get GT boxes based on GT's best AC (GT-AC) Match above threshold.

    - get positions `coord_gt` of GT boxes with IoU higher than `POS_TH_ACGT`;
      (using `flag_gtac` is also fine)
    - corresponding anchor positions `coord_ac` will also be selected
    - remove duplicates **anchors** (by batch index and anchor index) and keep
      the one with the highest IoU
    - update target boxes with selected GT boxes, and keep the rest as -1.0

    Args:
        bx_ac (tf.Tensor): anchor tensor (B, N_ac, 4)
        bx_gt (tf.Tensor): ground truth tensor (B, N_gt, 4)
        idx (tf.Tensor): pre-computed indices matrix (B, N_gt) where value in
            [0, N_ac), representing indices of the best mached anchor for each
            GT box
        ious (tf.Tensor): pre-computed IoU matrix (B, N_gt) where value in
            [0, 10000], representing the best IoU for each GT box

    Returns:
        tf.Tensor: best IoU matched GT boxes (B, N_ac, 4) for anchors
          - [-1.0, -1.0, -1.0, -1.0]: background
          - positive: foreground
    """
    # T/F matrix (B, N_gt) indicating whether the best IoU of each GT box is
    # above threshold
    flag_gtac = ious > NEG_TH_GTAC  # (B, N_gt)
    # coordinate pairs (M, 2) of ground truth boxes where `M <= B*N_gt`
    # of format (batch index, GT index), indicating the best matched GT boxes
    # no duplicates as `tf.where` here returns indices of non-zero elements,
    # which are unique
    coord_gt = tf.cast(tf.where(flag_gtac), tf.int32)
    # vector of best matched anchors' indices (M,) where value in [0, N_ac)
    # may have duplicates as multiple GT boxes may have the same best matched
    # anchor
    idx_ac = tf.gather_nd(idx, coord_gt)
    # coordinate pairs (M, 2) of anchors where `M <= B*N_gt` of format
    # (batch index, anchor index), indicating the best matched anchor
    # may have duplicates as `idx_ac` may have duplicates
    coord_ac = tf.stack([coord_gt[:, 0], idx_ac], axis=-1)

    # filtering: one anchor may have multiple matches with GT boxes, which can
    # lead to overwriting for the same anchor.
    # We only keep for one anchor the GT box with the highest IoU.
    # `arr` is a 2D tensor of format (M, 2) where `M <= B*N_gt`:
    # - the first column is the hash value of the coordinate pairs
    # - the second column is the best IoU of the corresponding coordinate pairs
    arr = tf.stack(
        [
            coord_ac[:, 0] * 10000 + coord_ac[:, 1],
            tf.boolean_mask(ious, flag_gtac, axis=0),
        ],
        axis=-1,
    )
    mask = filter_on_max(arr)  # (M,) with M1 True values where M1 <= M

    # update target boxes (B, N_ac, 4) with ground truth boxes
    # - indices: indicates the anchor positions, which have the best IoU
    # (against GT boxes) above threshold, to be updated
    # - updates: indicates the best matched GT boxes
    # corresponding to the anchor positions above
    return tf.tensor_scatter_nd_update(
        -tf.ones_like(bx_ac),  # init with -1, (B, N_ac, 4)
        tf.boolean_mask(coord_ac, mask),  # (M1, 2)
        tf.boolean_mask(tf.boolean_mask(bx_gt, flag_gtac), mask),  # (M1, 4)
    )


def get_gt_acgt(
    bx_ac: tf.Tensor,
    bx_gt: tf.Tensor,
    idx: tf.Tensor,
    ious: tf.Tensor,
) -> tf.Tensor:
    """Get GT boxes based on AC's best GT Match (AC-GT) above threshold.

    - detect background
      - anchors with IoU lower than `NEG_TH_ACGT` will be set to -1.0
    - detect AC-GT foreground
      - get positions `coord_ac` of anchors with IoU higher than `POS_TH_ACGT`
      - corresponding GT boxes will be selected as well
      - update target boxes `bx_tgt_acgt` on `pos` with selected GT boxes

    Args:
        bx_ac (tf.Tensor): anchor tensor (B, N_ac, 4)
        bx_gt (tf.Tensor): ground truth tensor (B, N_gt, 4)
        idx (tf.Tensor): pre-computed indices matrix (B, N_ac) where
            value in [0, N_gt), representing indices of the best mached GT box
            for each anchor
        ious (tf.Tensor): pre-computed IoU matrix (B, N_ac) where value in
            [0, 10000], representing the best IoU for each anchor

    Returns:
        tf.Tensor: GT boxes (B, N_ac, 4) for each anchor of tf.float32
          - [-1.0, -1.0, -1.0, -1.0]: background
          - [0.0, 0.0, 0.0, 0.0]: ignore
          - otherwise: foreground
    """
    # idx_acgt = tf.argmax(ious, axis=2, output_type=tf.int32)
    # iou_acgt = tf.reduce_max(ious, axis=2)

    # initialize with zeros
    bx_tgt_acgt = tf.zeros_like(bx_ac, dtype=tf.float32)  # (B, N_ac, 4)
    # detect background
    bx_tgt_acgt = tf.where(
        tf.repeat(ious[..., tf.newaxis], 4, axis=-1) < NEG_TH_ACGT,
        tf.constant([[-1.0]]),
        bx_tgt_acgt,
    )  # (N_ac, 4)

    # T/F matrix (B, N_ac) indicating whether the best IoU of each anchor is
    # above threshold
    flag_acgt = ious > POS_TH_ACGT
    # coordinate pairs (M, 2) of anchors where `M <= B*N_ac`
    # of format (batch index, AC index), indicating the best matched anchors
    # no duplicates as `tf.where` here returns indices of non-zero elements,
    # which are unique
    coord_ac = tf.cast(tf.where(flag_acgt), tf.int32)
    # vector of best matched GT boxes' indices (M,) where value in [0, N_gt)
    # may have duplicates as one anchor may have multiple matches with GT boxes
    values = tf.gather_nd(idx, coord_ac)
    # coordinate pairs (M, 2) of GT boxes where `M <= B*N_ac` of format
    # (batch index, GT index), indicating the best matched GT boxes
    # may have duplicates as `values` may have duplicates
    coord_gt = tf.stack([coord_ac[:, 0], values], axis=-1)

    # no need to filter as we only concern about the duplicates in `coord_ac`,
    # which can lead to overwriting of target boxes.
    return tf.tensor_scatter_nd_update(
        bx_tgt_acgt,  # (B, N_ac, 4)
        coord_ac,  # (M, 2)
        tf.gather_nd(bx_gt, coord_gt),  # (M, 4)
    )  # (B, N_ac, 4)


def get_gt_box(bx_ac: tf.Tensor, bx_gt: tf.Tensor) -> tf.Tensor:
    """Get ground truth boxes based on IoU for each anchor for RPN training.

    Args:
        bx_ac (tf.Tensor): anchor tensor (B, N_ac, 4)
        bx_gt (tf.Tensor): ground truth tensor (B, N_gt, 4)

    Returns:
        tf.Tensor: ground truth boxes (B, N_ac, 4) for each anchor (tf.float32)
          - [-1.0, -1.0, -1.0, -1.0]: background
          - [0.0, 0.0, 0.0, 0.0]: ignore
          - otherwise: foreground
    """
    ious = tf.cast(IOU_SCALE * _box.iou_batch(bx_ac, bx_gt), tf.int32)
    idx_gtac = tf.argmax(ious, axis=1, output_type=tf.int32)
    iou_gtac = tf.reduce_max(ious, axis=1)
    idx_acgt = tf.argmax(ious, axis=2, output_type=tf.int32)
    iou_acgt = tf.reduce_max(ious, axis=2)
    bx_tgt_gtac = get_gt_gtac(bx_ac, bx_gt, idx_gtac, iou_gtac)
    bx_tgt_acgt = get_gt_acgt(bx_ac, bx_gt, idx_acgt, iou_acgt)
    return tf.where(bx_tgt_gtac >= 0, bx_tgt_gtac, bx_tgt_acgt)


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
