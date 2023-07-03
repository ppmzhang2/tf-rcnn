"""Model training."""
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from src.rcnn import cfg
from src.rcnn.data import process_data
from src.rcnn.model import get_rpn_model
from src.rcnn.risk import risk_rpn_batch

rpn_ckpts_dir = os.path.join(cfg.MODELDIR, "rpn_ckpts")

# general optimizer
# optimizer = tf.keras.optimizers.Adam()
# M1 optimizer
optimizer = tf.keras.optimizers.legacy.Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
)
train_loss = tf.keras.metrics.Mean(name="train_loss")

(ds_train, ds_test), ds_info = tfds.load(
    "voc/2007",
    split=["train", "validation"],
    shuffle_files=True,
    with_info=True,
)

ds_train = ds_train.map(
    process_data,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
).batch(4).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(
    process_data,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
).batch(4).prefetch(tf.data.experimental.AUTOTUNE)


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
        bx_roi, bx_del, logits = model(images, training=True)
        loss = risk_rpn_batch(bx_del, logits, bx_roi, bx_gt)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(  # noqa: B905
            grads,
            model.trainable_variables,
        ))
    train_loss(loss)


def train_rpn(
    ds_train: tf.data.Dataset,
    ds_test: tf.data.Dataset,
    epochs: int,
    save_interval: int,
) -> None:
    """Train RPN.

    Args:
        ds_train (tf.data.Dataset): training dataset.
        ds_test (tf.data.Dataset): testing dataset.
        epochs (int): number of epochs.
        save_interval (int): number of batches after which to save the model.
    """
    model = get_rpn_model()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=rpn_ckpts_dir,
        max_to_keep=10,
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    for ep in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        for i, (img, bx, _) in enumerate(ds_train):
            train_rpn_step(model, img, bx)
            print(f"Epoch {ep + 1} Batch {i + 1} "
                  f"Training Loss {train_loss.result():.4f}")

            # Save model every 'save_interval' batches
            if i % save_interval == 0:
                print(f"Saving checkpoint for epoch {ep + 1} at batch {i + 1}")
                manager.save(checkpoint_number=i)


def load_rpn_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    """Load RPN and ROI models."""
    rpn_model = get_rpn_model()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=rpn_model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=rpn_ckpts_dir,
        max_to_keep=10,
    )
    checkpoint.restore(manager.latest_checkpoint)
    return rpn_model


if __name__ == "__main__":
    train_rpn(ds_train, ds_test, epochs=1, save_interval=10)
