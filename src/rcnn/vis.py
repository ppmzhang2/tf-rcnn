"""Visualization utilities for object detection."""
import cv2
import numpy as np
import tensorflow as tf

COLOR_BOX = (0, 255, 0)  # Green color for bounding box
COLOR_TXT = (0, 0, 255)  # Red color for class tag
THICKNESS_BOX = 2  # Line thickness
THICKNESS_TXT = 1  # Text thickness
SIZE_FONT = 0.5  # Font size


def draw_pred(
    img: tf.Tensor,
    bboxes: tf.Tensor,
    labels: tf.Tensor,
    names: list[str],
) -> np.ndarray:
    """Draw predicted bounding boxes and class tags on the image.

    Args:
        img (tf.Tensor): The input image tensor (H, W, C).
        bboxes (tf.Tensor): Bounding box tensor (N, 4).
        labels (tf.Tensor): Class label tensor (N,).
        names (list[str]): List of class names.

    Returns:
        np.ndarray: The image with bounding boxes and class tags.
    """
    # Convert image tensor to numpy array
    img = img.numpy()
    # Convert image from RGB to BGR as OpenCV uses BGR format
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Get image dimensions
    height, width, _ = img.shape

    # Draw bounding boxes
    for bbx, lbl in zip(bboxes, labels, strict=True):
        ymin, xmin, ymax, xmax = tf.get_static_value(bbx)
        tag = names[int(tf.get_static_value(lbl))]
        pt_tl = (int(xmin * width), int(ymin * height))  # top left point
        pt_br = (int(xmax * width), int(ymax * height))  # bottom right point
        img = cv2.rectangle(img, pt_tl, pt_br, COLOR_BOX, THICKNESS_BOX)
        # Put class tag
        img = cv2.putText(img, tag, pt_tl, cv2.FONT_HERSHEY_SIMPLEX, SIZE_FONT,
                          COLOR_TXT, THICKNESS_TXT, cv2.LINE_AA)
    return img


def draw_rois(img: tf.Tensor, rois: tf.Tensor) -> np.ndarray:
    """Draw predicted Region of Interests (ROIs) on the image.

    Args:
        img (tf.Tensor): The input image tensor (H, W, C).
        rois (tf.Tensor): RoI tensor (N_roi, 4).

    Returns:
        np.ndarray: The image with ROIs.
    """
    # Convert image tensor to numpy array
    img = img.numpy()
    # Convert image from RGB to BGR as OpenCV uses BGR format
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Get image dimensions
    height, width, _ = img.shape

    # Draw bounding boxes
    for roi in rois:
        ymin, xmin, ymax, xmax = tf.get_static_value(roi)
        pt_tl = (int(xmin * width), int(ymin * height))  # top left point
        pt_br = (int(xmax * width), int(ymax * height))  # bottom right point
        img = cv2.rectangle(img, pt_tl, pt_br, COLOR_BOX, THICKNESS_BOX)
    return img


def show_image(img: np.ndarray) -> None:
    """Display the image.

    Args:
        img (np.ndarray): The input image.
    """
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
