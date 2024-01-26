"""Model training."""
import logging
import os

import cv2
import tensorflow as tf

from rcnn import cfg
from rcnn import data
from rcnn.model import get_rpn_model
from rcnn.model import suppress

PATH_CKPT_RPN = os.path.join(cfg.MODELDIR, "rpn", "ckpt")
LOGGER = logging.getLogger(__name__)

# optimizer = tf.keras.optimizers.Adam(
#     learning_rate=0.0001,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-08,
# )
optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.LR_INIT, momentum=0.9)
cb_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_meanap",
    patience=6,
    mode="max",
    restore_best_weights=True,
)
cb_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_meanap",
    factor=0.5,
    patience=3,
    min_lr=0.0001,
)


def get_cb_ckpt(path: str) -> tf.keras.callbacks.ModelCheckpoint:
    """Get callback for saving model weights."""
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        monitor="val_meanap",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
    )


def train_rpn(epochs: int, batch_size: int) -> None:
    """Train RPN.

    Args:
        epochs (int): number of epochs.
        batch_size (int): batch size.
    """
    ds_tr, ds_va, _ = data.load_train_valid(
        cfg.DS,
        batch_size,
        16,  # TODO: customize
    )
    model = get_rpn_model(freeze_backbone=False, freeze_rpn=False)
    model.compile(optimizer=optimizer)
    model.fit(
        ds_tr,
        epochs=epochs,
        validation_data=ds_va,
        callbacks=[get_cb_ckpt(PATH_CKPT_RPN), cb_earlystop, cb_lr],
    )


def predict_rpn(n_sample: int) -> None:
    """Predict RPN."""
    # Load model
    model = get_rpn_model()
    model.load_weights(PATH_CKPT_RPN)
    # Load dataset
    ds_te, _ = data.load_test(cfg.DS, n_sample)
    # Predict
    img, (bx, lb) = next(iter(ds_te))
    _, log, bbx = model(img, training=False)  # (B, N_roi, 4)
    bbx_sup = suppress(bbx, log, cfg.N_SUPP_SCORE, cfg.N_SUPP_NMS, cfg.NMS_TH)
    for i in range(n_sample):
        pic = data.draw_rois(img[i], bbx_sup[i])
        cv2.imwrite(
            os.path.join(cfg.DATADIR, f"{cfg.DS_PREFIX}_test_rpn_{i:04d}.jpg"),
            pic * cfg.IMGNET_STD + cfg.IMGNET_MEAN,
        )
