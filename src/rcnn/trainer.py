"""Model training."""
import logging
import os

import cv2
import tensorflow as tf

from rcnn import cfg
from rcnn import data
from rcnn import vis
from rcnn.model import get_rpn_model
from rcnn.risk import mean_ap_rpn
from rcnn.risk import risk_rpn

RPN_CKPTS_DIR = os.path.join(cfg.MODELDIR, "rpn_ckpts")
LOGGER = logging.getLogger(__name__)

# general optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
)


def load_rpn_model() -> tuple[tf.keras.Model, tf.train.CheckpointManager]:
    """Load the RPN model and the latest checkpoint manager.

    If no checkpoint is found, the model and the checkpoint manager are
    initialized from scratch.

    Returns:
        tuple[tf.keras.Model, tf.train.CheckpointManager]: RPN model and
            checkpoint manager.
    """
    rpn_model = get_rpn_model()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=rpn_model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=RPN_CKPTS_DIR,
        max_to_keep=10,
    )

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        LOGGER.info(f"Restored from {manager.latest_checkpoint}")
    else:
        LOGGER.info("Initializing from scratch.")

    return rpn_model, manager


def train_rpn_step(
    model: tf.keras.Model,
    images: tf.Tensor,
    bx_gt: tf.Tensor,
) -> float:
    """Train RPN for one step.

    Args:
        model (tf.keras.Model): RPN model.
        images (tf.Tensor): images (B, H, W, 3)
        bx_gt (tf.Tensor): ground truth boxes (B, N_gt, 4)
    """
    with tf.GradientTape() as tape:
        _, logits, _, roi_box = model(images, training=True)
        bx_tgt = data.rpn.get_gt_box(bx_gt)
        mask_obj = data.rpn.get_gt_mask(bx_tgt, bkg=False)
        mask_bkg = data.rpn.get_gt_mask(bx_tgt, bkg=True)
        loss = risk_rpn(roi_box, bx_tgt, logits, mask_obj, mask_bkg)
    grads = tape.gradient(loss, model.trainable_variables)
    # check NaN
    for grad in grads:
        if tf.math.reduce_any(tf.math.is_nan(grad)):
            msg = "NaN gradient detected."
            raise ValueError(msg)
    # clip gradient
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(
        zip(  # noqa: B905
            grads,
            model.trainable_variables,
        ))
    return loss


def train_rpn(epochs: int, save_intv: int, batch: int) -> None:
    """Train RPN.

    Args:
        epochs (int): number of epochs.
        save_intv (int): interval to save the model.
        batch (int): batch size.
    """
    batch_va = 16
    # initialize metrics
    loss_tr = tf.keras.metrics.Mean(name="train_loss")
    mean_ap = tf.keras.metrics.Mean(name="mean_ap")
    # Load model and checkpoint manager
    model, manager = load_rpn_model()
    # Load dataset
    ds_tr, ds_va, _ = data.load_train_valid(cfg.DS, batch, batch_va)

    for ep in range(epochs):
        # Training loop
        LOGGER.info(f"EPOCH {ep + 1:02d}")
        loss_tr.reset_states()  # reset metrics after each epoch
        for i, (img, bx_gt, _) in enumerate(ds_tr):
            loss_tr(train_rpn_step(model, img, bx_gt))
            LOGGER.info(f"-- Ep/Batch {ep + 1:02d}/{i + 1:03d} "
                        f"Training Loss {loss_tr.result():.4f}")

            # Save model every 'save_interval' batches
            if i % save_intv == save_intv - 1:
                LOGGER.info(f"-- Ep/Batch {ep + 1:02d}/{i + 1:03d} "
                            "Saving model")
                manager.save(checkpoint_number=i)

        # Validation loop
        LOGGER.info(f"EPOCH {ep + 1:02d} (Validation)")
        mean_ap.reset_states()  # reset metrics after each epoch
        for i, (img, bx_gt, _) in enumerate(ds_va):
            bx_sup, _, _, _ = model(img, training=False)
            mean_ap(mean_ap_rpn(bx_sup, bx_gt, iou_threshold=0.5))
            LOGGER.info(f"-- Ep/Batch {ep + 1:02d}/{i + 1:03d} (VA) "
                        f"mAP {mean_ap.result():.4f}")
        LOGGER.info(f"EPOCH {ep + 1:02d} "
                    f"Training Loss {loss_tr.result():.4f} "
                    f"mAP {mean_ap.result():.4f}")


def predict_rpn(n_sample: int) -> None:
    """Predict RPN."""
    # Load model and checkpoint manager
    model, manager = load_rpn_model()
    # Load dataset
    ds_te, ds_info = data.load_test(cfg.DS, n_sample, shuffle=False)

    # Predict
    img, bx, lb = next(iter(ds_te))
    sup_box, _, _, _ = model(img, training=False)  # (B, N_roi, 4)
    for i in range(n_sample):
        pic = vis.draw_rois(img[i], sup_box[i])
        cv2.imwrite(
            os.path.join(cfg.DATADIR, f"{cfg.DS_PREFIX}_test_rpn_{i:04d}.jpg"),
            pic * 255.0,
        )


def show_gt(n_sample: int) -> None:
    """Predict RPN."""
    # Load dataset
    ds_te, ds_info = data.load_test(cfg.DS, n_sample, shuffle=False)
    names = ds_info.features["objects"]["label"].names

    img, bx, lb = next(iter(ds_te))
    for i in range(n_sample):
        pic = vis.draw_pred(img[i], bx[i], lb[i], names)
        cv2.imwrite(
            os.path.join(cfg.DATADIR, f"{cfg.DS_PREFIX}_test_gt_{i:04d}.jpg"),
            pic * 255.0,
        )
