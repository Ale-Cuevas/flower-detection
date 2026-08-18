"""FastAPI application for flower detection and classification."""

import base64
import logging
from io import BytesIO

from fastapi import FastAPI, HTTPException
from PIL import Image

from flower_pipeline import FlowerPipeline
from schemas import PredictRequest, PredictResponse, DetectionBox, FlowerClassification, FlowerDetectionResult, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Flower Detection API",
    description="Two-stage pipeline to detect flowers and classify their species",
    version="1.0.0",
)

# Initialize pipeline at startup
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global pipeline
    try:
        logger.info("Loading flower detection and classification models...")
        pipeline = FlowerPipeline()
        if pipeline.is_ready():
            logger.info("Models loaded successfully")
        else:
            logger.warning("One or both models failed to load")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if pipeline is None:
        return HealthResponse(
            status="models_loading",
            detector_loaded=False,
            classifier_loaded=False,
        )

    return HealthResponse(
        status="ready" if pipeline.is_ready() else "partial",
        detector_loaded=pipeline.detector.is_loaded(),
        classifier_loaded=pipeline.classifier.is_loaded(),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Detect flowers and classify each one.

    Takes a base64-encoded image and returns bounding boxes with flower classifications.
    """
    if pipeline is None or not pipeline.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check /health endpoint.",
        )

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 image: {str(e)}",
        )

    try:
        # Run pipeline
        results = pipeline.predict(image, request.detection_threshold)

        # Format response
        flowers = []
        for result in results:
            x1, y1, x2, y2 = result["box"]
            flower = FlowerDetectionResult(
                box=DetectionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    detection_score=result["detection_score"],
                ),
                classification=FlowerClassification(
                    flower_name=result["flower"],
                    classification_confidence=result["classification_confidence"],
                ),
            )
            flowers.append(flower)

        return PredictResponse(
            num_flowers=len(flowers),
            flowers=flowers,
            image_height=image.height,
            image_width=image.width,
            metadata={"detection_threshold": request.detection_threshold},
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
