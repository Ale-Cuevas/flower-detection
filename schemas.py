"""Pydantic schemas for flower detection and classification API."""

from pydantic import BaseModel, Field
from typing import List


class DetectionBox(BaseModel):
    """A single flower detection bounding box."""
    x1: float = Field(..., ge=0, description="Left edge of bounding box (pixels)")
    y1: float = Field(..., ge=0, description="Top edge of bounding box (pixels)")
    x2: float = Field(..., ge=0, description="Right edge of bounding box (pixels)")
    y2: float = Field(..., ge=0, description="Bottom edge of bounding box (pixels)")
    detection_score: float = Field(..., ge=0, le=1, description="Detection confidence [0, 1]")


class FlowerClassification(BaseModel):
    """Classification result for a single detected flower."""
    flower_name: str = Field(..., description="Predicted flower species name")
    classification_confidence: float = Field(..., ge=0, le=1, description="Classification confidence [0, 1]")


class FlowerDetectionResult(BaseModel):
    """A single flower with detection box and classification."""
    box: DetectionBox = Field(..., description="Bounding box coordinates and detection score")
    classification: FlowerClassification = Field(..., description="Flower species and confidence")


class PredictRequest(BaseModel):
    """Request to detect and classify flowers in an image."""
    image: str = Field(..., description="Image as base64-encoded string (PNG, JPG, etc.)")
    detection_threshold: float = Field(default=0.5, ge=0, le=1, description="Minimum detection confidence [0, 1]")


class PredictResponse(BaseModel):
    """Response containing detected and classified flowers."""
    num_flowers: int = Field(..., ge=0, description="Number of flowers detected above threshold")
    flowers: List[FlowerDetectionResult] = Field(default_factory=list, description="List of detected flowers")
    image_height: int = Field(..., gt=0, description="Original image height in pixels")
    image_width: int = Field(..., gt=0, description="Original image width in pixels")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    detector_loaded: bool
    classifier_loaded: bool
