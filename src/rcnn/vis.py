"""Demo."""
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds

COLOR_BOX = (0, 255, 0)  # Green color for bounding box
COLOR_TXT = (0, 0, 255)  # Red color for class tag
THICKNESS_BOX = 2  # Line thickness
THICKNESS_TXT = 1  # Text thickness
SIZE_FONT = 0.5  # Font size


def show_pred(
    image: tf.Tensor,
    bboxes: tf.Tensor,
    labels: tf.Tensor,
    names: list[str],
) -> None:
    """Visualize the predicted bounding boxes and class tags on the image.

    Args:
        image (tf.Tensor): The input image tensor (H, W, C).
        bboxes (tf.Tensor): Bounding box tensor (N, 4).
        labels (tf.Tensor): Class label tensor (N,).
        names (list[str]): List of class names.
    """
    # Convert image tensor to numpy array
    image = image.numpy()
    # Convert image from RGB to BGR as OpenCV uses BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Get image dimensions
    height, width, _ = image.shape

    # Draw bounding boxes
    for bbx, lbl in zip(bboxes, labels, strict=True):
        ymin, xmin, ymax, xmax = tf.get_static_value(bbx)
        tag = names[int(tf.get_static_value(lbl))]
        beg_point = (int(xmin * width), int(ymin * height))
        end_point = (int(xmax * width), int(ymax * height))
        thickness = THICKNESS_BOX
        image = cv2.rectangle(
            image,
            beg_point,
            end_point,
            COLOR_BOX,
            thickness,
        )
        # Put class tag
        tag_position = beg_point
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = SIZE_FONT
        font_thickness = THICKNESS_TXT
        image = cv2.putText(image, tag, tag_position, font, font_scale,
                            COLOR_TXT, font_thickness, cv2.LINE_AA)

    # Display image
    cv2.imshow("Image with Bounding Boxes", image)
    # Wait for a key press and close the window afterwards
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_rois(image: tf.Tensor, rois: tf.Tensor) -> None:
    """Visualize the predicted Region of Interests (ROIs) on the image.

    Args:
        image (tf.Tensor): The input image tensor (H, W, C).
        rois (tf.Tensor): RoI tensor (N, 4).
    """
    # Convert image tensor to numpy array
    image = image.numpy()
    # Convert image from RGB to BGR as OpenCV uses BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Get image dimensions
    height, width, _ = image.shape

    # Draw bounding boxes
    for roi in rois:
        ymin, xmin, ymax, xmax = tf.get_static_value(roi)
        beg_point = (int(xmin * width), int(ymin * height))
        end_point = (int(xmax * width), int(ymax * height))
        thickness = THICKNESS_BOX
        image = cv2.rectangle(
            image,
            beg_point,
            end_point,
            COLOR_BOX,
            thickness,
        )

    # Display image
    cv2.imshow("Image with Bounding Boxes", image)
    # Wait for a key press and close the window afterwards
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    (ds_train, ds_test), ds_info = tfds.load(
        "voc/2007",
        split=["train", "validation"],
        shuffle_files=True,
        with_info=True,
    )

    names = ds_info.features["objects"]["label"].names

    for example in ds_train:
        show_rois(example["image"], example["objects"]["bbox"])
        break
