"""All models are defined here."""
import tensorflow as tf

from rcnn import anchor
from rcnn import cfg
from rcnn import risk
from rcnn.model._roi import AC_VAL
from rcnn.model._roi import roi
from rcnn.model._rpn import rpn

__all__ = [
    "ModelRPN",
]


class ModelRPN(tf.keras.Model):
    """RPN Model with custom training / testing steps."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)
        self.mean_loss = tf.keras.metrics.Mean(name="loss")
        self.mean_ap = tf.keras.metrics.AUC(name="meanap", curve="PR")

    def train_step(
        self,
        data: tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
    ) -> dict[str, tf.Tensor]:
        """The logic for one training step."""
        x, (bx_gt, _) = data

        bsize = tf.shape(x)[0]
        # create anchors with matching batch size (B, N_ac, 4)
        # NOTE: cannot use a constant batch size as the last batch may have a
        # different size
        ac_ = tf.repeat(AC_VAL[tf.newaxis, ...], bsize, axis=0)

        with tf.GradientTape() as tape:
            # In TF2, the `training` flag affects, during both training and
            # inference, behavior of layers such as normalization (e.g. BN)
            # and dropout.
            dlt_prd, log_prd, bbx_prd = self(x, training=True)
            # NOTE: cannot use broadcasting for performance
            bx_tgt = anchor.get_gt_box(ac_, bx_gt)
            # NOTE: cannot use broadcasting for performance
            dlt_tgt = anchor.bbox2delta(bx_tgt, ac_)
            mask_obj = anchor.get_gt_mask(bx_tgt, bkg=False)
            mask_bkg = anchor.get_gt_mask(bx_tgt, bkg=True)
            loss = risk.risk_rpn(dlt_prd, dlt_tgt, log_prd, mask_obj, mask_bkg)

        trainable_vars = self.trainable_variables

        grads = tape.gradient(loss, trainable_vars)
        # check NaN using assertions; it works both in
        # - Graph Construction Phase / Defining operations (blueprint)
        # - Session Execution Phase / Running operations (actual computation)
        grads = [
            tf.debugging.assert_all_finite(
                g, message="NaN/Inf gradient detected.") for g in grads
        ]
        # clip gradient
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(
            zip(  # noqa: B905
                grads,
                trainable_vars,
            ))

        self.mean_loss.update_state(loss)
        label = risk.tf_label(bbx_prd, bx_tgt, iou_th=0.5)
        self.mean_ap.update_state(label, log_prd)

        return {
            "loss": self.mean_loss.result(),
            "meanap": self.mean_ap.result(),
        }

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """List of the model's metrics.

        We list our `Metric` objects here so that `reset_states()` can be
        called automatically at the start of each epoch
        or at the start of `evaluate()`.
        If you don't implement this property, you have to call
        `reset_states()` yourself at the time of your choosing.
        """
        return [self.mean_loss, self.mean_ap]

    def test_step(
        self,
        data: tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
    ) -> dict[str, tf.Tensor]:
        """Logic for one evaluation step."""
        x, (bx_gt, _) = data

        dlt_prd, log_prd, bbx_prd = self(x, training=False)
        label = risk.tf_label(bbx_prd, bx_gt, iou_th=0.5)
        self.mean_ap.update_state(label, log_prd)

        return {
            "meanap": self.mean_ap.result(),
        }


def get_rpn_model(
    *,
    freeze_backbone: bool = False,
    freeze_rpn: bool = False,
) -> ModelRPN:
    """Create a RPN model for training or prediction."""
    # Backbone
    bb = tf.keras.applications.resnet50.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(cfg.H, cfg.W, 3),
    )

    # Add RPN layers on top of the backbone
    tmp_dlt, tmp_log, rpn_layers = rpn(bb.output, cfg.H_FM, cfg.W_FM)
    bbx = roi(tmp_dlt)

    # Freeze all layers in the backbone for training
    if freeze_backbone:
        for layer in bb.layers:
            layer.trainable = False
    # Freeze all layers in the RPN for training
    if freeze_rpn:
        for layer in rpn_layers:
            layer.trainable = False

    # Create the ModelRPN instance
    model = ModelRPN(
        inputs=bb.input,
        outputs=[tmp_dlt, tmp_log, bbx],
    )

    return model
