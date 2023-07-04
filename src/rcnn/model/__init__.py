"""Model package."""
from rcnn.model._model import get_rpn_model
from rcnn.model._proposal import ProposalBlock

__all__ = [
    "get_rpn_model",
    "ProposalBlock",
]
