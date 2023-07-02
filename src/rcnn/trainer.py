"""Model training."""
import tensorflow as tf
import tensorflow_datasets as tfds

from src.rcnn.data import process_data
from src.rcnn.model import mdl_rpn
from src.rcnn.risk import risk_rpn_batch

optimizer = tf.keras.optimizers.Adam()
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
)


@tf.function
def train_rpn_step(images: tf.Tensor, bx_gt: tf.Tensor) -> None:
    """Train RPN for one step.

    Args:
        images (tf.Tensor): images (B, H, W, 3)
        bx_gt (tf.Tensor): ground truth boxes (B, N_gt, 4)
    """
    with tf.GradientTape() as tape:
        bx_del, logits = mdl_rpn(images, training=True)
        loss = risk_rpn_batch(bx_del, logits, bx_gt)
    grads = tape.gradient(loss, mdl_rpn.trainable_variables)
    optimizer.apply_gradients(
        zip(  # noqa: B905
            grads,
            mdl_rpn.trainable_variables,
        ))
    train_loss(loss)


if __name__ == "__main__":
    for epoch in range(5):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        for img, bx, _ in ds_train:
            train_rpn_step(img, bx)
            print(f"training loss: {train_loss.result()}")
