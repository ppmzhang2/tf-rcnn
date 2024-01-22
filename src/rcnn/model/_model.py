"""All models are defined here."""
import tensorflow as tf

from rcnn import anchor
from rcnn import cfg
from rcnn import risk
from rcnn.model._base import get_vgg16
from rcnn.model._roi import AC_VAL
from rcnn.model._roi import roi
from rcnn.model._rpn import rpn
from rcnn.model._suppress import suppress

__all__ = [
    "ModelRPN",
]


class ModelRPN(tf.keras.Model):
    """RPN Model with custom training / testing steps."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)
        self.mean_loss = tf.keras.metrics.Mean(name="loss")
        self.mean_ap = tf.keras.metrics.Mean(name="meanap")

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
            dlt, log, bx_sup = self(x, training=True)
            # NOTE: cannot use broadcasting for performance
            bx_tgt = anchor.get_gt_box(ac_, bx_gt)
            # NOTE: cannot use broadcasting for performance
            dlt_tgt = anchor.bbox2delta(bx_tgt, ac_)
            mask_obj = anchor.get_gt_mask(bx_tgt, bkg=False)
            mask_bkg = anchor.get_gt_mask(bx_tgt, bkg=True)
            loss = risk.risk_rpn(dlt, dlt_tgt, log, mask_obj, mask_bkg)

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
        self.mean_ap.update_state(risk.mean_ap_rpn(bx_sup, bx_gt, iou_th=0.5))

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

        _, _, bx_sup = self(x, training=False)
        self.mean_ap.update_state(risk.mean_ap_rpn(bx_sup, bx_gt, iou_th=0.5))

        return {"meanap": self.mean_ap.result()}


vgg16 = get_vgg16(cfg.H, cfg.W)


def get_rpn_model() -> tf.keras.Model:
    """Get the RPN model.

    Returns:
        tf.keras.Model: The RPN model which outputs the following:
            - dlt: The delta of the RPN (B, N_VAL_AC, 4).
            - log: The logit of the RPN (B, N_VAL_AC, 1).
            - sup_box: The suppressed bounding boxes (B, N_SUPP_NMS, 4).

    """
    dlt, log = rpn(vgg16.output, cfg.H_FM, cfg.W_FM)
    bbx = roi(dlt)
    sup_box = suppress(bbx, log, cfg.N_SUPP_SCORE, cfg.N_SUPP_NMS, cfg.NMS_TH)
    mdl = ModelRPN(inputs=vgg16.input, outputs=[dlt, log, sup_box])
    return mdl
