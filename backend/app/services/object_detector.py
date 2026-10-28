
import base64
from functools import lru_cache

import cv2
import numpy as np
from ultralytics import YOLO


MAX_DIMENSION = 1280
CONFIDENCE_THRESHOLD = 0.25


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    """
    Load the model once and reuse it for subsequent requests.

    The pretrained weights are downloaded automatically
    the first time they are needed, if not already available.
    """
    return YOLO("yolov8n.pt")


def detect_objects(image_bytes: bytes) -> dict:
    """
    Detect objects in uploaded image bytes and return
    detection details plus an annotated JPEG data URL.
    """

    # Decode uploaded bytes into an OpenCV image.
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode the uploaded image.")

    original_height, original_width = image.shape[:2]

    # Keep large images manageable for CPU inference.
    longest_side = max(original_width, original_height)

    if longest_side > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest_side

        new_width = max(1, round(original_width * scale))
        new_height = max(1, round(original_height * scale))

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    processed_height, processed_width = image.shape[:2]

    # Run pretrained YOLO inference.
    model = get_model()

    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        verbose=False,
    )

    result = results[0]
    detections = []

    # Extract each detected object's label, confidence,
    # and bounding-box coordinates.
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = [
                int(value)
                for value in box.xyxy[0].tolist()
            ]

            detections.append(
                {
                    "class_id": class_id,
                    "label": result.names[class_id],
                    "confidence": round(confidence, 4),
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                }
            )

    # YOLO draws its detected bounding boxes and labels.
    annotated_image = result.plot()

    success, encoded_image = cv2.imencode(
        ".jpg",
        annotated_image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )

    if not success:
        raise ValueError("Could not encode the annotated image.")

    # Return a data URL that a browser can display directly.
    image_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")

    return {
        "model": "YOLOv8n",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "detection_count": len(detections),
        "detections": detections,
        "annotated_image_data_url": (
            f"data:image/jpeg;base64,{image_base64}"
        ),
    }
