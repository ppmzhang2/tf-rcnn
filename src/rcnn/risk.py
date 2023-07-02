"""Risk functions."""
import tensorflow as tf

from src.rcnn import box
from src.rcnn import cfg

NEG_TH_ACGT = 0.30  # lower threshold for anchor-GT highest IoU
POS_TH_ACGT = 0.70  # upper threshold for anchor-GT highest IoU
NEG_TH_GTAC = 0.15  # lower threshold for GT-anchor highest IoU


def valid_ac_mask(ac: tf.Tensor) -> tf.Tensor:
    """Get valid anchors mask based on image size.

    Args:
        ac (tf.Tensor): anchor tensor (N, 4) in relative coordinates
        h (float): height of the image
        w (float): width of the image

    Returns:
        tf.Tensor: 0/1 mask of valid anchors (N,) of type tf.float32
    """
    return tf.where(
        (box.bbox.xmin(ac) >= 0) & (box.bbox.ymin(ac) >= 0)
        & (box.bbox.xmax(ac) >= box.bbox.xmin(ac))
        & (box.bbox.ymax(ac) >= box.bbox.ymin(ac))
        & (box.bbox.xmax(ac) <= 1) & (box.bbox.ymax(ac) <= 1),
        1.0,
        0.0,
    )


def align_mask(mask_obj: tf.Tensor, mask_bkg: tf.Tensor) -> tf.Tensor:
    """Align background mask with object mask.

    Args:
        mask_obj (tf.Tensor): 0/1 mask of object anchors (N_ac,)
        mask_bkg (tf.Tensor): 0/1 mask of background anchors (N_ac,)

    Returns:
        tf.Tensor: randomly selected background mask (N_ac,) with same number
            of object anchors i.e. `sum(mask_obj) == sum(mask_bkg)`.
    """
    # 1. get indices of object anchors
    idx_obj = tf.where(mask_obj > 0)  # (n_obj, 1)
    # 2. get indices of background anchors
    idx_bkg = tf.where(mask_bkg > 0)  # (n_bkg, 1)
    # 3. randomly select background anchors
    n_obj = tf.shape(idx_obj)[0]
    idx_bkg = tf.random.shuffle(idx_bkg)[:n_obj]  # (n_obj, 1)
    # 4. update background mask with selected ones
    mask_ = tf.tensor_scatter_nd_update(
        tf.zeros_like(mask_bkg, dtype=tf.float32),  # (N_ac,)
        idx_bkg,  # (n_obj, 1)
        tf.ones((n_obj, ), dtype=tf.float32),  # (n_obj,)
    )  # (N_ac,)
    return mask_


def rpn_target_iou(
    bx_ac: tf.Tensor,
    bx_gt: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Get RPN targets of a single image for training based on IoU.

    Args:
        bx_ac (tf.Tensor): anchor tensor (N_ac, 4)
        bx_gt (tf.Tensor): ground truth tensor (N_gt, 4)

    Returns:
        tuple[tf.Tensor, tf.Tensor]: tags (N_ac,), target boxes (N_ac, 4), both
            of type tf.float32
            - tags are flags for each anchor, with values:
              - +1: foreground
              - -1: background
              - 0: ignore
            - target boxes are ground truth boxes for each anchor
              - [-2.0, -2.0, -2.0, -2.0]: background
              - [-1.0, -1.0, -1.0, -1.0]: ignore
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
    bx_tgt_acgt = -tf.ones_like(bx_ac, dtype=tf.float32)  # (N_ac, 4)
    #    - update with AC-GT matches, where `idx_ac_gt` indicates indices of
    #      the best mached GT boxes for each anchors [0, N_ac).
    #      IoU lower than `NEG_TH_ACGT` tagged as background (-2.0);
    #      IoU higher than `POS_TH_ACGT` tagged as foreground (positive);
    #      any other IoU will be tagged as ignore (-1.0)
    bx_tgt_acgt = tf.where(
        tf.repeat(iou_ac_gt[:, tf.newaxis], 4, axis=-1) < NEG_TH_ACGT,
        tf.constant([[-2.0]]),
        bx_tgt_acgt,
    )  # (N_ac, 4)
    bx_tgt_acgt = tf.where(
        tf.repeat(iou_ac_gt[:, tf.newaxis], 4, axis=-1) > POS_TH_ACGT,
        tf.gather(bx_gt, idx_ac_gt),  # (N_ac, 4)
        bx_tgt_acgt,
    )  # (N_ac, 4)
    # 3. combine 1 and 2
    bx_tgt = tf.where(bx_tgt_gtac >= 0, bx_tgt_gtac, bx_tgt_acgt)

    # tags
    # - flag for each anchor
    # - values
    #   - +1: foreground
    #   - -1: background
    #   - 0: ignore
    # 0. init with zeros
    tags = tf.zeros_like(iou_ac_gt, dtype=tf.float32)  # (N_ac,)
    # 1. coordinate sum of target boxes:
    #    - positive: foreground
    #    - -8.0: background
    #    - -4.0: ignore
    _bx_tgt_sum = tf.reduce_sum(bx_tgt, axis=-1)
    _sum_neg = -8.0  # -2.0 * 4
    # 2. update with foreground
    tags = tf.where(_bx_tgt_sum > 0, 1.0, tags)
    # 3. update with background
    tags = tf.where(_bx_tgt_sum == _sum_neg, -1.0, tags)
    return tags, bx_tgt


def rpn_target(bx_gt: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Get RPN target tensors for training of a single image.

    Args:
        bx_gt (tf.Tensor): ground truth boxes (N_gt, 4)

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: object mask (N_ac,),
            background mask (N_ac,), and target deltas (N_ac, 4)
    """
    tags, bx_tgt = rpn_target_iou(box.anchors, bx_gt)
    # print(f"{bx_tgt[200:205]=}")  # -2.0 or -1.0
    # print(f"{box.anchors[200:205]=}")  # normal
    delta_tgt = box.bbox2delta(bx_tgt, box.anchors)
    mask_valid = valid_ac_mask(box.anchors)
    mask_obj = tf.cast(tags > 0, tf.float32) * mask_valid
    mask_bkg_all = tf.cast(tags < 0, tf.float32) * mask_valid
    # randomly select background anchors with same number of objects
    mask_bkg = align_mask(mask_obj, mask_bkg_all)
    return mask_obj, mask_bkg, delta_tgt


def risk_rpn_reg(
    bx_del: tf.Tensor,
    bx_tgt: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """Delta regression risk for RPN with smooth L1 loss.

    `bx` values cannot be `inf` or `-inf` otherwise the loss will be `nan`.

    Args:
        bx_del (tf.Tensor): predicted deltas (N_ac, 4)
        bx_tgt (tf.Tensor): target deltas (N_ac, 4)
        mask (tf.Tensor): 0/1 object mask (N_ac,)

    Returns:
        tf.Tensor: risk (scalar)
    """
    fn_huber = tf.keras.losses.Huber(reduction="none", )
    loss = fn_huber(bx_tgt, bx_del)  # (N_ac,)
    # print(bx_del[:5])  # normal
    # print(bx_tgt[:5])  # dw, dh = -inf
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + cfg.EPS)


def risk_rpn_obj(logits: tf.Tensor, mask_obj: tf.Tensor) -> tf.Tensor:
    """Objectness classification risk for RPN with binary cross entropy loss.

    Args:
        logits (tf.Tensor): predicted logits (N_ac, 1)
        mask_obj (tf.Tensor): 0/1 object mask (N_ac,)

    Returns:
        tf.Tensor: risk (scalar)
    """
    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction="none",
    )
    loss = bce(mask_obj[:, tf.newaxis], logits)  # (N_ac,)
    return tf.reduce_sum(loss * mask_obj) / (tf.reduce_sum(mask_obj) + cfg.EPS)


def risk_rpn_bkg(logits: tf.Tensor, mask_bkg: tf.Tensor) -> tf.Tensor:
    """Background classification risk for RPN with binary cross entropy loss.

    Args:
        logits (tf.Tensor): predicted logits (N_ac, 1)
        mask_bkg (tf.Tensor): 0/1 background mask (N_ac,)

    Returns:
        tf.Tensor: risk (scalar)
    """
    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction="none",
    )
    loss = bce(1.0 - mask_bkg[:, tf.newaxis], logits)  # (N_ac,)
    return tf.reduce_sum(loss * mask_bkg) / (tf.reduce_sum(mask_bkg) + cfg.EPS)


def risk_rpn(
    bx_del: tf.Tensor,
    logits: tf.Tensor,
    bx_gt: tf.Tensor,
) -> tf.Tensor:
    """RPN risk of a single image.

    Args:
        bx_del (tf.Tensor): predicted deltas (N_ac, 4)
        logits (tf.Tensor): predicted logits (N_ac, 1)
        bx_gt (tf.Tensor): ground truth boxes (N_gt, 4)

    Returns:
        tf.Tensor: risk (scalar)
    """
    mask_obj, mask_bkg, bx_tgt = rpn_target(bx_gt)
    # print(tf.reduce_sum(mask_obj), tf.reduce_sum(mask_bkg))  # > 0
    risk_reg = risk_rpn_reg(bx_del, bx_tgt, mask_obj)
    risk_obj = risk_rpn_obj(logits, mask_obj)
    risk_bkg = risk_rpn_bkg(logits, mask_bkg)
    return risk_reg + risk_obj + risk_bkg


def risk_rpn_batch(
    bx_del: tf.Tensor,
    logits: tf.Tensor,
    bx_gt: tf.Tensor,
) -> tf.Tensor:
    """RPN risk of a batch of images.

    Args:
        bx_del (tf.Tensor): predicted deltas (N, N_ac, 4)
        logits (tf.Tensor): predicted logits (N, N_ac, 1)
        bx_gt (tf.Tensor): ground truth boxes (N, N_gt, 4)

    Returns:
        tf.Tensor: risk (scalar)
    """
    risk = tf.map_fn(
        lambda x: risk_rpn(x[0], x[1], x[2]),
        (bx_del, logits, bx_gt),
        dtype=tf.float32,
    )
    return tf.reduce_mean(risk)
