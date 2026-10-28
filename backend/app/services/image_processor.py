
import cv2
import numpy as np


MAX_DIMENSION = 1280


def analyze_image_bytes(image_bytes: bytes) -> dict:
    """
    Decode an uploaded image, optionally resize it,
    and calculate basic image statistics.
    """

    # Convert raw bytes into a NumPy array.
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the array into an OpenCV color image.
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Reject invalid or unsupported image data.
    if image is None:
        raise ValueError(
            "The uploaded file is not a valid or supported image."
        )

    original_height, original_width = image.shape[:2]
    channels = image.shape[2]

    processed_image = image
    was_resized = False

    # Resize large images while preserving their aspect ratio.
    longest_side = max(original_width, original_height)

    if longest_side > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest_side

        new_width = max(1, round(original_width * scale))
        new_height = max(1, round(original_height * scale))

        processed_image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        was_resized = True

    # Convert the processed image to grayscale.
    grayscale = cv2.cvtColor(
        processed_image,
        cv2.COLOR_BGR2GRAY,
    )

    # Calculate average pixel brightness from 0 to 255.
    mean_brightness = float(np.mean(grayscale))

    processed_height, processed_width = processed_image.shape[:2]

    return {
        "original_width": original_width,
        "original_height": original_height,
        "channels": channels,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "was_resized": was_resized,
        "mean_brightness": round(mean_brightness, 2),
    }
