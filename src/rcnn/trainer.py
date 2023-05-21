"""Model training."""
import tensorflow as tf
import tensorflow_datasets as tfds

from rcnn.ds_handler import ds_handler
from rcnn.model import model  # TODO: revision
from rcnn.risk import risk_total

(ds_train, ds_test), ds_info = tfds.load(
    "voc/2007",
    split=["train", "validation"],
    shuffle_files=True,
    with_info=True,
)

ds_train = ds_train.map(
    ds_handler,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
ds_test = ds_test.map(
    ds_handler,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)

model.compile(optimizer="adam", loss=risk_total, metrics=["accuracy"])

batch_size = 32

model.fit(
    ds_train.batch(batch_size),
    epochs=10,
    validation_data=ds_test.batch(batch_size),
    validation_freq=1,
    verbose=1,
)
