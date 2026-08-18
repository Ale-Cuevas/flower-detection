"""Tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError

from schemas import (
    DetectionBox,
    FlowerClassification,
    FlowerDetectionResult,
    PredictRequest,
    PredictResponse,
    HealthResponse,
)


class TestDetectionBox:
    """Test DetectionBox schema validation."""

    def test_valid_box(self):
        """Test creating a valid detection box."""
        box = DetectionBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0, detection_score=0.95)
        assert box.x1 == 10.0
        assert box.detection_score == 0.95

    def test_negative_coordinates_rejected(self):
        """Test that negative coordinates are rejected."""
        with pytest.raises(ValidationError):
            DetectionBox(x1=-5.0, y1=20.0, x2=100.0, y2=150.0, detection_score=0.95)

    def test_invalid_score_too_high(self):
        """Test that confidence score > 1 is rejected."""
        with pytest.raises(ValidationError):
            DetectionBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0, detection_score=1.5)

    def test_invalid_score_negative(self):
        """Test that negative confidence score is rejected."""
        with pytest.raises(ValidationError):
            DetectionBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0, detection_score=-0.1)

    def test_boundary_scores(self):
        """Test that boundary values (0 and 1) are accepted."""
        box_low = DetectionBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0, detection_score=0.0)
        box_high = DetectionBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0, detection_score=1.0)
        assert box_low.detection_score == 0.0
        assert box_high.detection_score == 1.0


class TestFlowerClassification:
    """Test FlowerClassification schema validation."""

    def test_valid_classification(self):
        """Test creating a valid classification."""
        clf = FlowerClassification(flower_name="Sunflower", classification_confidence=0.87)
        assert clf.flower_name == "Sunflower"
        assert clf.classification_confidence == 0.87

    def test_confidence_bounds(self):
        """Test that confidence must be in [0, 1]."""
        with pytest.raises(ValidationError):
            FlowerClassification(flower_name="Rose", classification_confidence=1.5)

    def test_missing_flower_name(self):
        """Test that flower_name is required."""
        with pytest.raises(ValidationError):
            FlowerClassification(classification_confidence=0.9)


class TestPredictRequest:
    """Test PredictRequest schema validation."""

    def test_valid_request(self, base64_image):
        """Test creating a valid prediction request."""
        req = PredictRequest(image=base64_image, detection_threshold=0.5)
        assert req.detection_threshold == 0.5

    def test_default_threshold(self, base64_image):
        """Test that detection_threshold defaults to 0.5."""
        req = PredictRequest(image=base64_image)
        assert req.detection_threshold == 0.5

    def test_invalid_threshold_too_high(self, base64_image):
        """Test that threshold > 1 is rejected."""
        with pytest.raises(ValidationError):
            PredictRequest(image=base64_image, detection_threshold=1.5)

    def test_invalid_threshold_negative(self, base64_image):
        """Test that negative threshold is rejected."""
        with pytest.raises(ValidationError):
            PredictRequest(image=base64_image, detection_threshold=-0.1)

    def test_missing_image(self):
        """Test that image field is required."""
        with pytest.raises(ValidationError):
            PredictRequest(detection_threshold=0.5)


class TestPredictResponse:
    """Test PredictResponse schema validation."""

    def test_empty_response(self):
        """Test a valid response with no flowers detected."""
        resp = PredictResponse(
            num_flowers=0,
            flowers=[],
            image_height=256,
            image_width=256,
        )
        assert resp.num_flowers == 0
        assert len(resp.flowers) == 0

    def test_response_with_flowers(self):
        """Test a response with detected flowers."""
        box = DetectionBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0, detection_score=0.95)
        clf = FlowerClassification(flower_name="Rose", classification_confidence=0.88)
        flower = FlowerDetectionResult(box=box, classification=clf)

        resp = PredictResponse(
            num_flowers=1,
            flowers=[flower],
            image_height=256,
            image_width=256,
        )
        assert resp.num_flowers == 1
        assert resp.flowers[0].classification.flower_name == "Rose"

    def test_invalid_dimensions_zero(self):
        """Test that zero or negative dimensions are rejected."""
        with pytest.raises(ValidationError):
            PredictResponse(
                num_flowers=0,
                flowers=[],
                image_height=0,
                image_width=256,
            )

    def test_metadata_optional(self):
        """Test that metadata field is optional."""
        resp = PredictResponse(
            num_flowers=0,
            flowers=[],
            image_height=256,
            image_width=256,
        )
        assert resp.metadata == {}

    def test_num_flowers_matches_list(self):
        """Test that num_flowers should match the length of flowers list (client responsibility)."""
        box = DetectionBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0, detection_score=0.95)
        clf = FlowerClassification(flower_name="Tulip", classification_confidence=0.92)
        flower = FlowerDetectionResult(box=box, classification=clf)

        # This is allowed by schema (client can set num_flowers independently)
        resp = PredictResponse(
            num_flowers=5,
            flowers=[flower],
            image_height=256,
            image_width=256,
        )
        assert resp.num_flowers == 5
        assert len(resp.flowers) == 1


class TestHealthResponse:
    """Test HealthResponse schema validation."""

    def test_all_ready(self):
        """Test health response when models are ready."""
        health = HealthResponse(
            status="ready",
            detector_loaded=True,
            classifier_loaded=True,
        )
        assert health.status == "ready"
        assert health.detector_loaded is True
        assert health.classifier_loaded is True

    def test_partial_ready(self):
        """Test health response when only one model is loaded."""
        health = HealthResponse(
            status="partial",
            detector_loaded=True,
            classifier_loaded=False,
        )
        assert health.status == "partial"

    def test_not_ready(self):
        """Test health response when models are loading."""
        health = HealthResponse(
            status="models_loading",
            detector_loaded=False,
            classifier_loaded=False,
        )
        assert health.detector_loaded is False
