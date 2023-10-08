"""Model training."""
import logging
import os

import cv2
import tensorflow as tf

from rcnn import cfg
from rcnn import data
from rcnn.model import get_rpn_model

RPN_CKPTS_PATH = os.path.join(cfg.MODELDIR, "rpn_ckpts")
LOGGER = logging.getLogger(__name__)

# optim = tf.keras.optimizers.Adam(
#     learning_rate=0.0001,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-08,
# )
optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.LR_INIT, momentum=0.9)
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=RPN_CKPTS_PATH,
    monitor="val_meanap",
    save_best_only=True,
    save_weights_only=True,
    mode="max",
)
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


def train_rpn(epochs: int) -> None:
    """Train RPN.

    Args:
        epochs (int): number of epochs.
    """
    ds_tr, ds_va, _ = data.load_train_valid(
        cfg.DS,
        cfg.BATCH_SIZE_TR,
        cfg.BATCH_SIZE_TE,
    )
    model = get_rpn_model()
    model.compile(optimizer=optimizer)
    model.fit(
        ds_tr,
        epochs=epochs,
        validation_data=ds_va,
        callbacks=[cb_checkpoint, cb_earlystop, cb_lr],
    )


def predict_rpn(n_sample: int) -> None:
    """Predict RPN."""
    # Load model
    model = get_rpn_model()
    model.load_weights(RPN_CKPTS_PATH)
    # Load dataset
    ds_te, _ = data.load_test(cfg.DS, n_sample)
    # Predict
    img, (bx, lb) = next(iter(ds_te))
    _, _, sup_box = model(img, training=False)  # (B, N_roi, 4)
    for i in range(n_sample):
        pic = data.draw_rois(img[i], sup_box[i])
        cv2.imwrite(
            os.path.join(cfg.DATADIR, f"{cfg.DS_PREFIX}_test_rpn_{i:04d}.jpg"),
            pic * cfg.IMGNET_STD + cfg.IMGNET_MEAN,
        )
