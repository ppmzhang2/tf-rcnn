"""Model training."""
import logging
import os

import cv2
import tensorflow as tf

from rcnn import cfg
from rcnn import vis
from rcnn.data import load_test
from rcnn.data import load_train_valid
from rcnn.model import get_rpn_model
from rcnn.risk import risk_rpn_batch

RPN_CKPTS_DIR = os.path.join(cfg.MODELDIR, "rpn_ckpts")
LOGGER = logging.getLogger(__name__)

# general optimizer
optimizer = tf.keras.optimizers.Adam()
# M1 optimizer
# optimizer = tf.keras.optimizers.legacy.Adam(
#     learning_rate=0.0001,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-08,
# )
loss_tr = tf.keras.metrics.Mean(name="train_loss")


@tf.function
def train_rpn_step(
    model: tf.keras.Model,
    images: tf.Tensor,
    bx_gt: tf.Tensor,
) -> None:
    """Train RPN for one step.

    Args:
        model (tf.keras.Model): RPN model.
        images (tf.Tensor): images (B, H, W, 3)
        bx_gt (tf.Tensor): ground truth boxes (B, N_gt, 4)
    """
    with tf.GradientTape() as tape:
        _, logits, _, roi_box = model(images, training=True)
        loss = risk_rpn_batch(logits, roi_box, bx_gt)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(  # noqa: B905
            grads,
            model.trainable_variables,
        ))
    loss_tr(loss)


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


def train_rpn(epochs: int, save_intv: int, batch: int) -> None:
    """Train RPN.

    Args:
        epochs (int): number of epochs.
        save_intv (int): interval to save the model.
        batch (int): batch size.
    """
    # Load model and checkpoint manager
    model, manager = load_rpn_model()
    # Load dataset
    ds_tr, ds_va, ds_info = load_train_valid(cfg.DS, batch, batch)

    # Training loop
    for ep in range(epochs):
        # Reset the metrics at the start of the next epoch
        loss_tr.reset_states()
        for i, (img, bx, _) in enumerate(ds_tr):
            train_rpn_step(model, img, bx)
            LOGGER.info(f"Epoch {ep + 1:02d} Batch {i + 1:03d} "
                        f"Training Loss {loss_tr.result():.4f}")

            # Save model every 'save_interval' batches
            if i % save_intv == 0:
                LOGGER.info(f"Saving checkpoint for epoch {ep + 1} "
                            f"at batch {i + 1}")
                manager.save(checkpoint_number=i)


def predict_rpn(n_sample: int) -> None:
    """Predict RPN."""
    # Load model and checkpoint manager
    model, manager = load_rpn_model()
    # Load dataset
    ds_te, ds_info = load_test(cfg.DS, n_sample, shuffle=False)

    # Predict
    img, bx, lb = next(iter(ds_te))
    sup_box, _, _, _ = model(img, training=False)  # (B, N_roi, 4)
    for i in range(n_sample):
        pic = vis.draw_rois(img[i], sup_box[i])
        cv2.imwrite(
            os.path.join(cfg.DATADIR, f"{cfg.DS_PREFIX}_test_{i:04d}.jpg"),
            pic * 255.0,
        )
