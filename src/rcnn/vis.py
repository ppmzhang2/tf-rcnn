"""Visualization."""
import os

import cv2

from rcnn import cfg
from rcnn import data


def show_gt(n_sample: int) -> None:
    """Show ground truth."""
    # Load dataset
    ds_tr, ds_te, ds_info = data.load_test(cfg.DS, n_sample)
    names = ds_info.features["objects"]["label"].names

    img, (bx, lb) = next(iter(ds_te))
    for i in range(n_sample):
        pic = data.draw_pred(img[i], bx[i], lb[i], names)
        cv2.imwrite(
            os.path.join(cfg.DATADIR, f"{cfg.DS_PREFIX}_test_gt_{i:04d}.jpg"),
            pic * cfg.IMGNET_STD + cfg.IMGNET_MEAN,
        )
