import tensorflow as tf

__all__ = [
    "roi_align",
]


def get_abs_roi(fm: tf.Tensor, rois: tf.Tensor) -> tf.Tensor:
    h_fm, w_fm = tf.shape(fm)[1], tf.shape(fm)[2]
    return tf.stack(
        [
            rois[..., 0] * tf.cast(h_fm, tf.float32),
            rois[..., 1] * tf.cast(w_fm, tf.float32),
            rois[..., 2] * tf.cast(h_fm, tf.float32),
            rois[..., 3] * tf.cast(w_fm, tf.float32),
        ],
        axis=-1,
    )


def get_idx(rois: tf.Tensor) -> tf.Tensor:
    """Get indices for RoI alignment.

    Args:
        rois (tf.Tensor): RoIs, (bsize, n_roi, 4)

    Returns:
        tf.Tensor: (bsize, n_roi, 2) tensor of indices (batch_idx, roi_idx)
    """
    batch_size, num_rois, _ = tf.shape(rois)
    return tf.stack(
        tf.meshgrid(tf.range(batch_size), tf.range(num_rois), indexing="ij"),
        axis=-1,
    )


def roi_align(
    fm: tf.Tensor,
    rois: tf.Tensor,
    pool_size: int,
) -> tf.Tensor:
    """RoI Alignment.

    Args:
        fm (tf.Tensor): feature map, (bsize, h, w, c)
        rois (tf.Tensor): RoIs, (bsize, n_roi, 4)
        pool_size (int): output size

    Returns:
        tf.Tensor: (bsize, n_roi, pool_size, pool_size, c) tensor
    """

    def align_roi(idx: tf.Tensor) -> tf.Tensor:
        """Align a single ROI.

        Args:
            idx (tf.Tensor): (2,) tensor of indices (batch_idx, roi_idx)

        Returns:
            tf.Tensor: (pool_size, pool_size, c) tensor
        """
        b, r = idx[0], idx[1]
        y1, x1, y2, x2 = rois_abs[b, r, :]
        y1 = tf.cast(tf.math.floor(y1), tf.int32)
        x1 = tf.cast(tf.math.floor(x1), tf.int32)
        y2 = tf.maximum(tf.cast(tf.math.ceil(y2), tf.int32), y1 + 1)
        x2 = tf.maximum(tf.cast(tf.math.ceil(x2), tf.int32), x1 + 1)

        roi = fm[b, y1:y2, x1:x2, :]

        return tf.image.resize(roi, (pool_size, pool_size), method="bilinear")

    def align_rois(idx: tf.Tensor) -> tf.Tensor:
        """Align the RoIs of a single image.

        Args:
            idx (tf.Tensor): (n_roi, 2) tensor of indices (batch_idx, roi_idx)

        Returns:
            tf.Tensor: (n_roi, pool_size, pool_size, c) tensor
        """
        return tf.map_fn(align_roi, idx, dtype=tf.float32)

    rois_abs = tf.keras.layers.Lambda(
        lambda x: get_abs_roi(x[0], x[1]),
        name="get_abs_roi",
    )([fm, rois])  # (bsize, n_roi, 4)

    indices = tf.keras.layers.Lambda(
        get_idx,
        name="get_idx",
    )(rois)  # (bsize, n_roi, 2)

    # (bsize, n_roi, pool_size, pool_size, c)
    output = tf.map_fn(align_rois, indices, dtype=tf.float32)

    return output
