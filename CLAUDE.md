# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **two-stage flower detection and classification system** using fine-tuned PyTorch models:

1. **Detector** (Faster R-CNN ResNet50): Locates flowers in an image (bounding boxes)
2. **Classifier** (ResNet18): Identifies what species each detected flower is (102 classes from Flowers102 dataset)

The pipeline can run both stages together for end-to-end inference, or each independently.

## Architecture & Key Components

### Models
- **Classifier**: ResNet18 fine-tuned on [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset (102 flower species)
  - Base: ImageNet-pretrained ResNet18
  - Training: Two-phase (head-only → layer4 + head fine-tuning)
  - Weights file: `flower_classifier_resnet18.pt`

- **Detector**: Faster R-CNN ResNet50 FPN v2 fine-tuned on custom dataset from Roboflow
  - Base: Pre-trained on COCO (91 classes)
  - Training: Head replaced for custom flower detection (typically 6 classes: background + 5 flower types)
  - Weights file: `flower_detector_fasterrcnn.pt`
  - Data format: COCO JSON annotations with folder structure (`dataset/train/`, `dataset/valid/`)

### Data Preprocessing
- **Classifier**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Detector**: Tensor format without normalization at loading stage
- **Special handling**: Detected crops are padded to square (not cropped) to preserve tall/narrow flowers

### Key Parameters
- Classifier: 224×224 input, 102 output classes
- Detector: Variable input size, ~6 output classes (adjust `NUM_CLASSES` in training script)
- Detection threshold: 0.5 (configurable in inference)

## File Organization

**Core modules**:
- `flower_pipeline.py` — Core pipeline classes (FlowerDetector, FlowerClassifier, FlowerPipeline)
- `app.py` — FastAPI REST application
- `schemas.py` — Pydantic request/response models

**Training scripts** (numbered sequentially):
- `01_train_classifier.py` — Train ResNet18 classifier on Flowers102
- `02_train_detector.py` — Train Faster R-CNN detector on COCO-format dataset
- `03_predict.py` — Unified inference script (both models)

**Inference scripts**:
- `04_flower_classify.py` — Classification only (single image)
- `05_detect_and_classify.py` — Two-stage pipeline (detection + classification)

**Testing**:
- `tests/` — Comprehensive test suite
  - `conftest.py` — Pytest fixtures and configuration
  - `test_schemas.py` — Pydantic schema validation tests
  - `test_flower_pipeline.py` — Core pipeline unit tests
  - `test_app.py` — FastAPI endpoint tests

**Supporting files**:
- `cat_to_name.json` — Flower name mapping for 102 classes (downloaded on first run)
- `pyproject.toml` — Dependency management via UV

## Development Setup

### Install dependencies
```bash
uv sync
```

### Install dev dependencies (for testing)
```bash
uv sync --extra dev
```

### Verify GPU availability
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Common Commands

### Running the REST API

**Start the server**:
```bash
python app.py
```
- Runs on `http://localhost:8000`
- Swagger docs at `http://localhost:8000/docs`
- ReDoc at `http://localhost:8000/redoc`

**Production deployment** (with Uvicorn):
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**With auto-reload for development**:
```bash
uvicorn app:app --reload
```

### Testing

**Run all tests**:
```bash
pytest
```

**Run specific test file**:
```bash
pytest tests/test_schemas.py
pytest tests/test_flower_pipeline.py
pytest tests/test_app.py
```

**Run with coverage**:
```bash
pytest --cov=. --cov-report=html
```

**Run only unit tests** (excludes integration tests):
```bash
pytest -m "not integration"
```

**Run with verbose output**:
```bash
pytest -v
```

### Training

**Train classifier** (requires Flowers102 dataset — auto-downloaded):
```bash
python 01_train_classifier.py
```
- Downloads ~370MB on first run
- Output: `flower_classifier_resnet18.pt`
- Runtime: ~10–15 min on GPU, ~2 hours on CPU

**Train detector** (requires COCO-format dataset from Roboflow):
```bash
# First, export a flower-detection dataset from Roboflow in COCO format
# Place at ./dataset/train/ and ./dataset/valid/ with _annotations.coco.json in each
python 02_train_detector.py
```
- Expects `./dataset/train/_annotations.coco.json`, `./dataset/valid/_annotations.coco.json`
- Output: `flower_detector_fasterrcnn.pt`
- Adjust `NUM_CLASSES` if your dataset has a different number of flower types

### Inference

**Classification only** (use trained classifier):
```bash
python 04_flower_classify.py path/to/image.jpg
```
- Output: Flower name + top-3 predictions with confidence scores

**Detection only** (use trained detector):
```bash
python 03_predict.py path/to/image.jpg
```
- Runs both classification and detection if both weight files exist

**Full two-stage pipeline** (detection + classification):
```bash
python 05_detect_and_classify.py path/to/image.jpg
```
- Finds flowers, crops them, classifies each → outputs annotated image + JSON results

## REST API

### Endpoints

#### `GET /health`
Health check endpoint. Returns model loading status.

**Response**:
```json
{
  "status": "ready",
  "detector_loaded": true,
  "classifier_loaded": true
}
```

#### `POST /predict`
Detect and classify flowers in an image (two-stage pipeline).

**Request**:
```json
{
  "image": "base64_encoded_image_string",
  "detection_threshold": 0.5
}
```

**Response**:
```json
{
  "num_flowers": 2,
  "flowers": [
    {
      "box": {
        "x1": 10.0,
        "y1": 20.0,
        "x2": 100.0,
        "y2": 150.0,
        "detection_score": 0.95
      },
      "classification": {
        "flower_name": "Sunflower",
        "classification_confidence": 0.88
      }
    }
  ],
  "image_height": 256,
  "image_width": 256,
  "metadata": {
    "detection_threshold": 0.5
  }
}
```

### Request/Response Schemas

Pydantic models in `schemas.py` validate all inputs and outputs:
- **PredictRequest**: base64 image + detection_threshold (0–1)
- **PredictResponse**: num_flowers, flowers list, dimensions, metadata
- **DetectionBox**: x1, y1, x2, y2, detection_score
- **FlowerClassification**: flower_name, classification_confidence
- **FlowerDetectionResult**: Combines box + classification
- **HealthResponse**: status, detector_loaded, classifier_loaded

## Testing Strategy

### Schema Validation Tests (`test_schemas.py`)
- Validate Pydantic model constraints (value ranges, required fields)
- Test boundary values (0, 1 for confidences)
- Ensure type checking and field requirements

### Unit Tests (`test_flower_pipeline.py`)
- Device handling and availability
- Image preprocessing (padding, format conversion)
- Class name mapping and counts
- Error handling for missing model weights
- Model assumptions (input shape 224×224, 102 classes for classifier)

### Integration Tests (`test_app.py`)
- FastAPI endpoint availability
- Request validation (invalid base64, out-of-range thresholds)
- Response schema compliance
- Error scenarios (missing image, models not ready)
- Mocked pipeline for isolated endpoint testing

### Test Fixtures (`conftest.py`)
- `sample_image_rgb`: Simple 224×224 RGB test image
- `sample_image_with_flower_like_features`: Image with circular blob (flower-like)
- `base64_image`: Base64-encoded test image for API testing
- `test_image_file`: Temporary file for image testing

## Training Details

### Classifier Training (01_train_classifier.py)
Two-phase approach:
1. **Phase 1** (5 epochs): Freeze backbone, train new head only (LR=1e-3)
2. **Phase 2** (5 epochs): Unfreeze layer4, fine-tune with lower LR (backbone=1e-5, head=1e-4)

Dataset split: 1020 train, 1020 val (102 classes × 10 images each)

### Detector Training (02_train_detector.py)
- Batch size: 4 (due to variable-length targets in Faster R-CNN)
- Optimizer: SGD (momentum=0.9, weight_decay=0.0005)
- Custom `FlowerDetectionDataset` wraps COCO JSON format
- Custom `collate_fn` handles variable box counts per image

## Important Notes

### Image Preprocessing Consistency
- Classifier always uses ImageNet normalization — failure to normalize = poor predictions
- Detector handles normalization internally (ToTensor only)
- Inference transforms must match training transforms

### COCO Dataset Format
Expected folder structure for detector training:
```
./dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── image2.jpg
└── valid/
    ├── _annotations.coco.json
    └── image1.jpg
```

COCO bbox format in JSON: [x_min, y_min, width, height] → converted to [x_min, y_min, x_max, y_max] for Faster R-CNN

### Model Architecture Details
- **ResNet18 classifier**: Final FC layer changed from 1000→NUM_CLASSES (102)
- **Faster R-CNN detector**: Box predictor head changed from 91→NUM_CLASSES (match your dataset)
- Both load `weights=None` at inference to avoid downloading ImageNet weights again

### Flower Name Mapping
- Classifier uses `cat_to_name.json` (1-indexed in JSON, 0-indexed when accessed)
- Detector just outputs integer class IDs (5 flower types + background)
- To add class names to detector output, build a similar mapping from Roboflow annotations

### Base64 Image Encoding for API
Images should be encoded as base64 PNG/JPG before sending to `/predict`:
```python
import base64
from PIL import Image
from io import BytesIO

img = Image.open("photo.jpg")
buffer = BytesIO()
img.save(buffer, format="PNG")
buffer.seek(0)
b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
```

## Debugging Tips

- **Classification low confidence**: Check input normalization, image resolution, or retrain with longer epochs
- **Detection missing flowers**: Lower `DETECTION_SCORE_THRESHOLD` (currently 0.5) or retrain detector with more epochs
- **CUDA out of memory**: Reduce batch size (currently 4 for detector, 32 for classifier) or use CPU (`device = "cpu"`)
- **FileNotFoundError for weights**: Run training script first or ensure `.pt` files are in repo root
- **COCO loading errors**: Verify annotation JSON path and structure; check that category IDs match in bboxes
- **API won't start**: Ensure both model weight files exist; check /health endpoint for model status
- **Tests failing on missing weights**: Run `pytest -m "not integration"` to skip tests requiring actual model files
