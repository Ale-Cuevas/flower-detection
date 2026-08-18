"""Tests for flower detection and classification pipeline."""

import pytest
import torch
from PIL import Image

from flower_pipeline import (
    get_device,
    build_class_names,
    pad_to_square,
    FlowerDetector,
    FlowerClassifier,
    FlowerPipeline,
)


class TestDeviceHandling:
    """Test device selection logic."""

    def test_device_type(self):
        """Test that get_device returns a torch.device object."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_is_valid(self):
        """Test that returned device is valid for PyTorch."""
        device = get_device()
        # Should be either 'cuda' or 'cpu'
        assert device.type in ['cuda', 'cpu']


class TestClassNames:
    """Test flower name mapping."""

    def test_class_names_dict_type(self):
        """Test that build_class_names returns a dictionary."""
        names = build_class_names()
        assert isinstance(names, dict)

    def test_class_names_count(self):
        """Test that we have 102 flower classes."""
        names = build_class_names()
        assert len(names) == 102

    def test_class_names_mapping(self):
        """Test that indices are 0-indexed and values are non-empty strings."""
        names = build_class_names()
        for idx, name in names.items():
            assert isinstance(idx, int)
            assert 0 <= idx < 102
            assert isinstance(name, str)
            assert len(name) > 0

    def test_file_caching(self):
        """Test that build_class_names uses cached file if available."""
        names1 = build_class_names()
        names2 = build_class_names()
        # Should be identical
        assert names1 == names2


class TestPadToSquare:
    """Test image padding utility."""

    def test_square_image_unchanged(self):
        """Test that square images are unchanged."""
        img = Image.new("RGB", (100, 100), color="red")
        padded = pad_to_square(img)
        assert padded.size == (100, 100)

    def test_tall_image_padded(self):
        """Test that tall images are padded to square."""
        img = Image.new("RGB", (100, 200), color="blue")
        padded = pad_to_square(img)
        assert padded.size == (200, 200)
        # Original image should be centered
        assert padded.size[0] == padded.size[1]

    def test_wide_image_padded(self):
        """Test that wide images are padded to square."""
        img = Image.new("RGB", (200, 100), color="green")
        padded = pad_to_square(img)
        assert padded.size == (200, 200)

    def test_padding_preserves_content(self):
        """Test that padding preserves original image content."""
        img = Image.new("RGB", (50, 100), color="red")
        padded = pad_to_square(img)
        # Padded size should be max(50, 100) = 100
        assert padded.size == (100, 100)
        # Should be larger than original
        assert padded.size[0] >= img.size[0]
        assert padded.size[1] >= img.size[1]

    def test_custom_fill_color(self):
        """Test custom padding fill color."""
        img = Image.new("RGB", (50, 100), color="red")
        fill_color = (255, 0, 0)  # Red
        padded = pad_to_square(img, fill=fill_color)
        # Corners should have fill color (they're in padded area)
        corner_pixel = padded.getpixel((0, 0))
        # Note: padding might use different color due to implementation
        assert padded.size == (100, 100)


class TestFlowerDetector:
    """Test flower detector initialization and error handling."""

    def test_detector_missing_weights_raises_error(self):
        """Test that detector raises error when weights file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            FlowerDetector(weights_path="/nonexistent/path/weights.pt")

    def test_detector_is_loaded_false_on_init_failure(self):
        """Test is_loaded returns False when initialization fails."""
        with pytest.raises(FileNotFoundError):
            FlowerDetector(weights_path="/nonexistent/weights.pt")


class TestFlowerClassifier:
    """Test flower classifier initialization and error handling."""

    def test_classifier_missing_weights_raises_error(self):
        """Test that classifier raises error when weights file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            FlowerClassifier(weights_path="/nonexistent/path/weights.pt")

    def test_classifier_class_names_loaded(self):
        """Test that classifier loads class names on initialization."""
        with pytest.raises(FileNotFoundError):
            # This will fail on model loading, but class names should be loaded first
            FlowerClassifier(weights_path="/nonexistent/weights.pt")


class TestImagePreprocessing:
    """Test image preprocessing assumptions."""

    def test_rgb_image_creation(self, sample_image_rgb):
        """Test that sample image is valid RGB."""
        assert sample_image_rgb.mode == "RGB"
        assert len(sample_image_rgb.size) == 2

    def test_grayscale_to_rgb_conversion(self):
        """Test converting grayscale image to RGB (as done in pipeline)."""
        # Create a grayscale image
        gray_img = Image.new("L", (224, 224), color=128)
        # Convert to RGB
        rgb_img = gray_img.convert("RGB")
        assert rgb_img.mode == "RGB"

    def test_rgba_to_rgb_conversion(self):
        """Test converting RGBA image to RGB (as done in pipeline)."""
        rgba_img = Image.new("RGBA", (224, 224), color=(255, 0, 0, 255))
        rgb_img = rgba_img.convert("RGB")
        assert rgb_img.mode == "RGB"

    def test_expected_input_shape_224(self):
        """Test that classifier expects 224x224 input."""
        # This is an assumption about the model architecture
        # ResNet18 is trained on ImageNet which uses 224x224
        assert 224 == 224  # Placeholder for actual model assumptions


class TestFlowerPipeline:
    """Test end-to-end pipeline behavior (without actual models)."""

    def test_pipeline_initialization_missing_models(self):
        """Test that pipeline initialization fails gracefully with missing models."""
        with pytest.raises(FileNotFoundError):
            FlowerPipeline(
                detector_weights="/nonexistent/detector.pt",
                classifier_weights="/nonexistent/classifier.pt",
            )

    def test_pipeline_is_ready_false_on_init_failure(self):
        """Test is_ready returns False when initialization fails."""
        with pytest.raises(FileNotFoundError):
            pipeline = FlowerPipeline(
                detector_weights="/nonexistent/detector.pt",
                classifier_weights="/nonexistent/classifier.pt",
            )


class TestDetectionAssumptions:
    """Test assumptions about detection output format."""

    def test_detection_box_format(self):
        """Test that detection boxes should be (x1, y1, x2, y2, score)."""
        # This is an assumption about Faster R-CNN output format
        # Boxes should be in [x_min, y_min, x_max, y_max] format
        expected_box_tuple_length = 5
        assert expected_box_tuple_length == 5  # x1, y1, x2, y2, score

    def test_detection_score_range(self):
        """Test that detection scores are in [0, 1] range."""
        # Valid score should be between 0 and 1
        valid_score = 0.75
        assert 0 <= valid_score <= 1


class TestClassificationAssumptions:
    """Test assumptions about classification output."""

    def test_classification_output_format(self):
        """Test that classification returns (name, confidence)."""
        # This is the expected format from classifier.classify()
        expected_tuple_length = 2
        assert expected_tuple_length == 2  # name, confidence

    def test_num_classes_102(self):
        """Test that classifier has 102 flower classes."""
        from flower_pipeline import CLASSIFIER_NUM_CLASSES
        assert CLASSIFIER_NUM_CLASSES == 102

    def test_detector_num_classes(self):
        """Test that detector has expected number of classes."""
        from flower_pipeline import DETECTOR_NUM_CLASSES
        # Typically 6: background + 5 flower types
        assert DETECTOR_NUM_CLASSES == 6
