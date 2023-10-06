"""Model training."""
import logging
import os

import cv2
import tensorflow as tf

from rcnn import anchor
from rcnn import cfg
from rcnn import data
from rcnn import vis
from rcnn.model import get_rpn_model
from rcnn.risk import mean_ap_rpn
from rcnn.risk import risk_rpn

RPN_CKPTS_DIR = os.path.join(cfg.MODELDIR, "rpn_ckpts")
LOGGER = logging.getLogger(__name__)

ac = tf.constant(anchor.RPNAC, dtype=tf.float32)  # (N_ac, 4)

# general optimizer
# TODO:
# - use different learning rate
# - learning rate scheduler
optim = tf.keras.optimizers.Adam(
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
    checkpoint = tf.train.Checkpoint(optimizer=optim, model=rpn_model)
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


@tf.function
def train_rpn_step(
    mdl: tf.keras.Model,
    images: tf.Tensor,
    bx_ac: tf.Tensor,
    bx_gt: tf.Tensor,
) -> tf.Tensor:
    """Train RPN for one step.

    - The `tf.function` decorator is necessary to avoid the `tf.function`
      retracing issue:
      no `tf.function` introduces the re-tracing issue, just as abusing does.
    - To make sure `tf.function` works, functions inside it should be succinct,
      and ideally contain only `tf` or `tf.math` operations, and no `np` or
      `python` operations.

    Args:
        mdl (tf.keras.Model): RPN model.
        images (tf.Tensor): images (B, H, W, 3)
        bx_ac (tf.Tensor): anchor boxes (B, N_ac, 4)
        bx_gt (tf.Tensor): ground truth boxes (B, N_gt, 4)

    Returns:
        tf.Tensor: loss value.
    """
    with tf.GradientTape() as tape:
        dlt, log, _ = mdl(images, training=True)
        bx_tgt = anchor.get_gt_box(bx_ac, bx_gt)
        dlt_tgt = anchor.bbox2delta(bx_tgt, bx_ac)
        mask_obj = anchor.get_gt_mask(bx_tgt, bkg=False)
        mask_bkg = anchor.get_gt_mask(bx_tgt, bkg=True)
        loss = risk_rpn(dlt, dlt_tgt, log, mask_obj, mask_bkg)
    grads = tape.gradient(loss, mdl.trainable_variables)
    # check NaN using assertions; it works both in
    # - Graph Construction Phase / Defining operations (blueprint)
    # - Session Execution Phase / Running operations (actual computation)
    grads = [
        tf.debugging.assert_all_finite(g, message="NaN/Inf gradient detected.")
        for g in grads
    ]
    # clip gradient
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optim.apply_gradients(zip(grads, mdl.trainable_variables))  # noqa: B905
    return loss


def train_rpn(epochs: int, save_intv: int, batch: int) -> None:
    """Train RPN.

    Args:
        epochs (int): number of epochs.
        save_intv (int): interval to save the model.
        batch (int): batch size.

    Raises:
        ValueError: if NaN gradient is detected.
    """
    batch_va = 16
    # initialize metrics
    loss_tr = tf.keras.metrics.Mean(name="train_loss")
    mean_ap = tf.keras.metrics.Mean(name="mean_ap")
    # Load model and checkpoint manager
    mdl, manager = load_rpn_model()
    # Load dataset
    ds_tr, ds_va, _ = data.load_train_valid(cfg.DS, batch, batch_va)

    for ep in range(epochs):
        # Training loop
        LOGGER.info(f"EPOCH {ep + 1:02d}")
        loss_tr.reset_states()  # reset metrics after each epoch
        for i, (img, bx_gt, _) in enumerate(ds_tr):
            # Get anchor boxes
            bx_ac = tf.repeat(ac[tf.newaxis, ...], img.shape[0], axis=0)
            loss_tr(train_rpn_step(mdl, img, bx_ac, bx_gt))
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
            _, _, bx_sup = mdl(img, training=False)
            mean_ap(mean_ap_rpn(bx_sup, bx_gt, iou_th=0.5))
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
