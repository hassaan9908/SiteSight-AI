
import json
import os

from dotenv import load_dotenv
from google import genai


# Load environment variables from backend/.env.
load_dotenv()


def generate_inspection_report(analysis: dict) -> str:
    """
    Generate a report grounded in the image statistics
    and the objects detected by the YOLO model.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "paste_your_actual_api_key_here":
        raise ValueError(
            "Gemini API key is missing. Add GEMINI_API_KEY "
            "to backend/.env and restart the server."
        )

    image_info = analysis.get("image", {})
    detection_info = analysis.get("object_detection", {})
    detections = detection_info.get("detections", [])

    # Limit the amount of detection data sent to the LLM.
    simplified_detections = [
        {
            "label": str(item.get("label", "unknown")),
            "confidence": round(
                float(item.get("confidence", 0)), 3
            ),
        }
        for item in detections[:60]
    ]

    report_data = {
        "image_dimensions": {
            "width": image_info.get("original_width"),
            "height": image_info.get("original_height"),
        },
        "detected_object_count": detection_info.get(
            "detection_count", len(detections)
        ),
        "detections": simplified_detections,
    }

    prompt = f"""
You are SiteSight AI, an assistant that helps a human
review the results of a computer-vision image analysis.

The following JSON contains image metadata and detections
from a pretrained, general-purpose YOLO object-detection model.

ANALYSIS DATA:
{json.dumps(report_data, indent=2)}

Create a concise report with these sections:

1. SUMMARY
Briefly summarize what the model detected.

2. DETECTED OBJECTS
List the detected object categories and their confidence
scores. Mention when no objects were detected.

3. SUGGESTED FOLLOW-UP CHECKS
Give reasonable, general suggestions for what a human
reviewer could inspect more closely, based on the
available detections. Do not imply that a hazard exists.

4. LIMITATIONS
Explain that the results depend on the model's supported
classes, image quality, and detection confidence.

Important rules:
- Use only the information supplied in the JSON.
- Do not claim to have visually inspected the original image.
- Do not infer structural damage, missing safety equipment,
  safety violations, or building-code compliance from
  generic object labels.
- A detection confidence score is not a probability that
  an object or a safety condition is definitively correct.
- No detections does not mean that a scene is empty or safe.
- Keep the report clear, professional, and concise.
"""

    client = genai.Client(api_key=api_key)

    response = client.interactions.create(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        input=prompt,
    )

    report = response.output_text

    if not report or not report.strip():
        raise RuntimeError("Gemini returned an empty report.")

    return report.strip()
