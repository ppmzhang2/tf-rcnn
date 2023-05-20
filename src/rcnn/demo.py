"""Demo."""
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds


def show_image_bbox(
    image: tf.Tensor,
    bboxes: list[tuple[float, float, float, float]],
) -> None:
    """Visualize an image with bounding boxes using OpenCV.

    Args:
        image (tf.Tensor): The input image tensor.
        bboxes (List[Tuple[float, float, float, float]]): A list of bounding
          box coordinates in the format (ymin, xmin, ymax, xmax).
    """
    # Convert image tensor to numpy array
    image = image.numpy()

    # Convert image from RGB to BGR as OpenCV uses BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get image dimensions
    height, width, _ = image.shape

    # Draw bounding boxes
    for bbox in bboxes:
        ymin, xmin, ymax, xmax = bbox
        start_point = (int(xmin * width), int(ymin * height))
        end_point = (int(xmax * width), int(ymax * height))
        color = (0, 255, 0)  # Green color in BGR
        thickness = 2  # Line thickness of 2 pixels
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    # Display image
    cv2.imshow("Image with Bounding Boxes", image)

    # Wait for a key press and close the window afterwards
    cv2.waitKey(0)
    cv2.destroyAllWindows()


(ds_train, ds_test), ds_info = tfds.load(
    "voc/2007",
    split=["train", "validation"],
    shuffle_files=True,
    with_info=True,
)

for example in ds_train:
    # print(example["image"])
    # print(example["objects"])
    # print(example["labels"])
    show_image_bbox(example["image"], example["objects"]["bbox"])
    break
