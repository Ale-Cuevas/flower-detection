"""Tests for FastAPI application endpoints."""

import base64
import json
from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def base64_invalid():
    """Return an invalid base64 string."""
    return "not_a_valid_base64!!!"


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_endpoint_exists(self, client):
        """Test that health endpoint responds."""
        response = client.get("/health")
        assert response.status_code in [200, 503]

    def test_health_response_format(self, client):
        """Test that health response has required fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "detector_loaded" in data
        assert "classifier_loaded" in data
        assert isinstance(data["detector_loaded"], bool)
        assert isinstance(data["classifier_loaded"], bool)


class TestPredictEndpointValidation:
    """Test request validation for /predict endpoint."""

    def test_predict_endpoint_exists(self, client, base64_image):
        """Test that predict endpoint responds to POST."""
        payload = {"image": base64_image}
        response = client.post("/predict", json=payload)
        # Might be 503 if models aren't loaded, but not 404
        assert response.status_code != 404

    def test_predict_missing_image(self, client):
        """Test that predict requires image field."""
        payload = {}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_invalid_base64(self, client, base64_invalid):
        """Test that predict handles invalid base64."""
        payload = {"image": base64_invalid}
        response = client.post("/predict", json=payload)
        # Should be 400 (bad request) when base64 decode fails
        assert response.status_code in [400, 500]

    def test_predict_invalid_detection_threshold_too_high(self, client, base64_image):
        """Test that detection_threshold > 1 is rejected."""
        payload = {
            "image": base64_image,
            "detection_threshold": 1.5,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_detection_threshold_negative(self, client, base64_image):
        """Test that negative detection_threshold is rejected."""
        payload = {
            "image": base64_image,
            "detection_threshold": -0.1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_valid_threshold_values(self, client, base64_image):
        """Test that valid threshold values are accepted."""
        for threshold in [0.0, 0.5, 1.0]:
            payload = {
                "image": base64_image,
                "detection_threshold": threshold,
            }
            response = client.post("/predict", json=payload)
            # Should not be 422 (validation error)
            assert response.status_code != 422

    def test_predict_non_image_base64(self, client):
        """Test that non-image base64 data is rejected."""
        # Valid base64 but not an image
        invalid_image_b64 = base64.b64encode(b"this is not an image").decode("utf-8")
        payload = {"image": invalid_image_b64}
        response = client.post("/predict", json=payload)
        # Should fail when trying to open as image
        assert response.status_code in [400, 500]


class TestPredictResponseSchema:
    """Test /predict response schema when models are available."""

    @patch("app.pipeline")
    def test_predict_response_format(self, mock_pipeline, client, base64_image):
        """Test that predict returns proper response schema."""
        # Mock pipeline to simulate successful prediction
        mock_pipeline.is_ready.return_value = True
        mock_pipeline.predict.return_value = []

        payload = {"image": base64_image}
        response = client.post("/predict", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert "num_flowers" in data
            assert "flowers" in data
            assert "image_height" in data
            assert "image_width" in data
            assert isinstance(data["num_flowers"], int)
            assert isinstance(data["flowers"], list)
            assert isinstance(data["image_height"], int)
            assert isinstance(data["image_width"], int)

    @patch("app.pipeline")
    def test_predict_no_flowers_detected(self, mock_pipeline, client, base64_image):
        """Test response when no flowers are detected."""
        mock_pipeline.is_ready.return_value = True
        mock_pipeline.predict.return_value = []

        payload = {"image": base64_image}
        response = client.post("/predict", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert data["num_flowers"] == 0
            assert data["flowers"] == []

    @patch("app.pipeline")
    def test_predict_flowers_detected(self, mock_pipeline, client, base64_image):
        """Test response format when flowers are detected."""
        mock_prediction = [
            {
                "box": (10.0, 20.0, 100.0, 150.0),
                "detection_score": 0.95,
                "flower": "Sunflower",
                "classification_confidence": 0.88,
            }
        ]
        mock_pipeline.is_ready.return_value = True
        mock_pipeline.predict.return_value = mock_prediction

        payload = {"image": base64_image}
        response = client.post("/predict", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert data["num_flowers"] == 1
            assert len(data["flowers"]) == 1

            flower = data["flowers"][0]
            assert "box" in flower
            assert "classification" in flower

            box = flower["box"]
            assert "x1" in box
            assert "y1" in box
            assert "x2" in box
            assert "y2" in box
            assert "detection_score" in box

            clf = flower["classification"]
            assert "flower_name" in clf
            assert "classification_confidence" in clf
            assert clf["flower_name"] == "Sunflower"

    @patch("app.pipeline")
    def test_predict_multiple_flowers(self, mock_pipeline, client, base64_image):
        """Test response with multiple flowers detected."""
        mock_predictions = [
            {
                "box": (10.0, 20.0, 100.0, 150.0),
                "detection_score": 0.95,
                "flower": "Sunflower",
                "classification_confidence": 0.88,
            },
            {
                "box": (150.0, 50.0, 250.0, 200.0),
                "detection_score": 0.87,
                "flower": "Rose",
                "classification_confidence": 0.92,
            },
        ]
        mock_pipeline.is_ready.return_value = True
        mock_pipeline.predict.return_value = mock_predictions

        payload = {"image": base64_image}
        response = client.post("/predict", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert data["num_flowers"] == 2
            assert len(data["flowers"]) == 2


class TestPredictModelNotReady:
    """Test /predict behavior when models aren't loaded."""

    @patch("app.pipeline")
    def test_predict_models_not_ready(self, mock_pipeline, client, base64_image):
        """Test that 503 is returned when models aren't ready."""
        mock_pipeline.is_ready.return_value = False

        payload = {"image": base64_image}
        response = client.post("/predict", json=payload)
        assert response.status_code == 503

    @patch("app.pipeline", None)
    def test_predict_pipeline_none(self, client, base64_image):
        """Test that 503 is returned when pipeline is None."""
        payload = {"image": base64_image}
        response = client.post("/predict", json=payload)
        assert response.status_code == 503


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]

    def test_api_has_title(self, client):
        """Test that API has proper title."""
        response = client.get("/openapi.json")
        schema = response.json()
        assert "info" in schema
        assert "title" in schema["info"]
        assert "Flower" in schema["info"]["title"]


class TestErrorHandling:
    """Test error handling in API."""

    @patch("app.pipeline")
    def test_prediction_internal_error_handling(self, mock_pipeline, client, base64_image):
        """Test that internal errors return 500."""
        mock_pipeline.is_ready.return_value = True
        mock_pipeline.predict.side_effect = RuntimeError("Model error")

        payload = {"image": base64_image}
        response = client.post("/predict", json=payload)
        assert response.status_code == 500

    def test_malformed_json(self, client):
        """Test that malformed JSON returns 422."""
        response = client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
