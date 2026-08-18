"""Pytest configuration and shared fixtures."""

import base64
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from flower_pipeline import FlowerDetector, FlowerClassifier, FlowerPipeline


@pytest.fixture
def sample_image_rgb():
    """Create a simple RGB test image."""
    img = Image.new("RGB", (224, 224), color="red")
    return img


@pytest.fixture
def sample_image_with_flower_like_features():
    """Create an image with flower-like features (circular blob)."""
    img = Image.new("RGB", (256, 256), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    # Draw a yellow circle (flower-like)
    draw.ellipse([50, 50, 150, 150], fill=(255, 255, 0), outline=(255, 200, 0), width=3)
    # Draw green stem-like lines
    draw.rectangle([120, 150, 130, 200], fill=(0, 128, 0))
    return img


@pytest.fixture
def base64_image(sample_image_rgb):
    """Convert image to base64 string for API testing."""
    buffer = BytesIO()
    sample_image_rgb.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def test_image_file(tmp_path, sample_image_rgb):
    """Save test image to a temporary file."""
    img_path = tmp_path / "test_image.jpg"
    sample_image_rgb.save(img_path)
    return str(img_path)


@pytest.fixture(scope="session")
def dummy_detector_weights(tmp_path_factory):
    """Create a dummy detector weights file for testing (avoids model download)."""
    tmp_dir = tmp_path_factory.mktemp("weights")
    weights_file = tmp_dir / "flower_detector_fasterrcnn.pt"
    # We won't actually create a valid model file here;
    # tests that need real weights should be marked as integration tests
    return str(weights_file)


@pytest.fixture(scope="session")
def dummy_classifier_weights(tmp_path_factory):
    """Create a dummy classifier weights file for testing."""
    tmp_dir = tmp_path_factory.mktemp("weights")
    weights_file = tmp_dir / "flower_classifier_resnet18.pt"
    # We won't actually create a valid model file here
    return str(weights_file)
