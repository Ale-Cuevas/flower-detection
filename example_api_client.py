"""Example client for the Flower Detection API."""

import base64
import json
from pathlib import Path

import requests
from PIL import Image
from io import BytesIO

API_URL = "http://localhost:8000"


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    img = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def check_health():
    """Check API health status."""
    response = requests.get(f"{API_URL}/health")
    print("Health check:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def predict_flowers(image_path: str, detection_threshold: float = 0.5):
    """Detect and classify flowers in an image."""
    print(f"\nPredicting flowers in: {image_path}")

    # Encode image
    print("Encoding image to base64...")
    b64_image = encode_image_to_base64(image_path)

    # Create request payload
    payload = {
        "image": b64_image,
        "detection_threshold": detection_threshold,
    }

    # Send request
    print("Sending prediction request...")
    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"\nResults:")
        print(f"  Flowers detected: {result['num_flowers']}")
        print(f"  Image size: {result['image_width']}x{result['image_height']}")

        for i, flower in enumerate(result["flowers"], 1):
            box = flower["box"]
            clf = flower["classification"]
            print(f"\n  Flower {i}:")
            print(f"    Name: {clf['flower_name']}")
            print(f"    Classification confidence: {clf['classification_confidence']:.2%}")
            print(f"    Detection confidence: {box['detection_score']:.2%}")
            print(f"    Bounding box: ({box['x1']:.0f}, {box['y1']:.0f}) "
                  f"to ({box['x2']:.0f}, {box['y2']:.0f})")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    import sys

    # Check health
    if not check_health():
        print("\nAPI is not ready. Make sure to:")
        print("  1. Run: python app.py")
        print("  2. Ensure model weights exist:")
        print("     - flower_classifier_resnet18.pt")
        print("     - flower_detector_fasterrcnn.pt")
        sys.exit(1)

    # Predict on provided image or default test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "sunflower.jpg"

    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Usage: python example_api_client.py <image_path>")
        sys.exit(1)

    predict_flowers(image_path)
