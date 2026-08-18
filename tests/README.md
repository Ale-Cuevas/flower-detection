# Flower Detection Testing Suite

This directory contains comprehensive tests for the flower detection and classification pipeline.

## Test Structure

### `conftest.py`
Pytest configuration and shared fixtures:
- `sample_image_rgb`: Simple 224×224 RGB test image
- `sample_image_with_flower_like_features`: Image with circular blob simulating a flower
- `base64_image`: Base64-encoded version of sample image for API testing
- `test_image_file`: Temporary file path for image I/O testing
- `dummy_detector_weights` / `dummy_classifier_weights`: Placeholder paths for testing

### `test_schemas.py`
**Purpose**: Validate Pydantic model constraints and API contract

**Tests**:
- `TestDetectionBox`: Bounding box coordinate and score validation
  - Negative coordinates rejected
  - Scores must be in [0, 1]
  - Boundary values accepted

- `TestFlowerClassification`: Classification result validation
  - Valid name + confidence combinations
  - Confidence bounds enforcement
  - Required field validation

- `TestPredictRequest`: API request schema validation
  - Image field required
  - Detection threshold defaults to 0.5
  - Threshold bounds [0, 1]

- `TestPredictResponse`: API response schema validation
  - Response structure with num_flowers, flowers list, dimensions
  - Empty response handling (no flowers)
  - Multiple flowers in response
  - Optional metadata field

- `TestHealthResponse`: Health check response validation
  - Status field and boolean flags
  - All combinations of model states

**Run**:
```bash
pytest tests/test_schemas.py
pytest tests/test_schemas.py::TestDetectionBox::test_negative_coordinates_rejected -v
```

### `test_flower_pipeline.py`
**Purpose**: Unit tests for core pipeline components

**Tests**:
- `TestDeviceHandling`: PyTorch device selection
  - Valid device types (cuda/cpu)
  - Device availability

- `TestClassNames`: Flower name mapping
  - Dictionary structure and 102 classes
  - Index/name pairing validation
  - File caching behavior

- `TestPadToSquare`: Image padding utility
  - Square images unchanged
  - Tall/wide images padded correctly
  - Content preservation

- `TestFlowerDetector`: Detector initialization
  - Missing weights handled gracefully
  - Error messages helpful

- `TestFlowerClassifier`: Classifier initialization
  - Missing weights handled gracefully
  - Class names loaded on init

- `TestImagePreprocessing`: Image format handling assumptions
  - RGB image creation
  - Grayscale→RGB conversion
  - RGBA→RGB conversion

- `TestFlowerPipeline`: End-to-end pipeline assumptions
  - Graceful failure on missing models

- `TestDetectionAssumptions`: Detection output format validation
  - Box format: (x1, y1, x2, y2, score)
  - Score range [0, 1]

- `TestClassificationAssumptions`: Classification output validation
  - Output format: (name, confidence)
  - 102 flower classes
  - Detector has 6 classes (background + 5 flowers)

**Run**:
```bash
pytest tests/test_flower_pipeline.py
pytest tests/test_flower_pipeline.py::TestPadToSquare -v
```

### `test_app.py`
**Purpose**: FastAPI endpoint testing and integration

**Tests**:
- `TestHealthEndpoint`: Health check endpoint
  - Endpoint availability
  - Response format validation

- `TestPredictEndpointValidation`: Request validation
  - Missing required fields rejected (422)
  - Invalid base64 rejected (400)
  - Detection threshold validation
  - Non-image data rejected

- `TestPredictResponseSchema`: Response format with mocked models
  - Correct schema structure
  - Empty response (no flowers)
  - Single/multiple flowers
  - All required fields present

- `TestPredictModelNotReady`: Error handling when models unavailable
  - 503 returned when models not ready
  - 503 returned when pipeline is None

- `TestAPIDocumentation`: OpenAPI schema
  - Schema available at /openapi.json
  - Endpoints documented
  - Title and description present

- `TestErrorHandling`: Exception handling
  - Internal errors return 500
  - Malformed JSON returns 422

**Run**:
```bash
pytest tests/test_app.py
pytest tests/test_app.py::TestPredictResponseSchema -v
```

## Running Tests

### All tests
```bash
pytest
```

### With coverage report
```bash
pytest --cov=. --cov-report=html
# Open htmlcov/index.html to view results
```

### Specific test file
```bash
pytest tests/test_schemas.py -v
```

### Specific test class
```bash
pytest tests/test_app.py::TestHealthEndpoint -v
```

### Specific test function
```bash
pytest tests/test_schemas.py::TestDetectionBox::test_negative_coordinates_rejected -v
```

### Run with marker filtering
```bash
# Run only unit tests (not integration tests requiring model weights)
pytest -m "unit"

# Run only API tests
pytest -m "api"

# Run everything except integration tests
pytest -m "not integration"
```

### With multiple verbosity levels
```bash
pytest -v          # Verbose (show all test names)
pytest -vv         # Very verbose (show test details)
pytest -q          # Quiet (minimal output)
```

## Test Categories

### Schema Tests
Validate Pydantic model constraints without needing model weights.
- Fast execution
- No GPU required
- Test API contract

**Run**: `pytest tests/test_schemas.py -v`

### Pipeline Unit Tests
Test core pipeline components:
- Device handling
- Image preprocessing
- Error handling
- Model assumptions

These DON'T require actual model weights—they test logic and error cases.

**Run**: `pytest tests/test_flower_pipeline.py -v`

### API Tests
Test FastAPI endpoints:
- Request validation
- Response format
- Error handling
- Health checks

Uses mocks to avoid requiring model weights.

**Run**: `pytest tests/test_app.py -v`

## Integration Tests (Requires Model Weights)

To run tests that require actual trained models:

1. Ensure model weights exist:
   ```bash
   python 01_train_classifier.py    # Creates flower_classifier_resnet18.pt
   python 02_train_detector.py      # Creates flower_detector_fasterrcnn.pt
   ```

2. Run integration tests:
   ```bash
   pytest tests/ -m "integration"
   ```

Currently, most tests are unit/API tests that don't require weights. Add integration tests by:
1. Marking with `@pytest.mark.integration`
2. Using actual models instead of mocks

Example:
```python
@pytest.mark.integration
def test_end_to_end_pipeline_with_real_models(sample_image_rgb):
    """Test full pipeline with actual trained models."""
    pipeline = FlowerPipeline()  # Requires weights to exist
    results = pipeline.predict(sample_image_rgb)
    assert isinstance(results, list)
```

## Test Coverage

To measure test coverage:

```bash
pytest --cov=. --cov-report=html
```

This generates an HTML report showing:
- Lines covered by tests
- Lines not covered
- Coverage percentage by file

View the report:
```bash
open htmlcov/index.html
```

## Common Test Patterns

### Testing with Fixtures
```python
def test_something(sample_image_rgb):
    """sample_image_rgb is provided by conftest.py"""
    assert sample_image_rgb.size == (224, 224)
```

### Testing Validation Errors
```python
def test_invalid_score():
    with pytest.raises(ValidationError):
        DetectionBox(..., detection_score=1.5)  # > 1
```

### Mocking External Dependencies
```python
@patch("app.pipeline")
def test_predict_with_mocked_pipeline(mock_pipeline, client):
    mock_pipeline.is_ready.return_value = True
    mock_pipeline.predict.return_value = []
```

### Testing API Responses
```python
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
```

## Troubleshooting

### Tests fail with "No module named 'flower_pipeline'"
- Make sure you're running pytest from the repo root
- Or install the package: `pip install -e .`

### Tests timeout
- Tests should complete within 10 seconds (no actual model inference)
- If timing out, check for blocking I/O or infinite loops

### "FileNotFoundError: flower_detector_fasterrcnn.pt"
- These tests require actual model weights
- Mark them with `@pytest.mark.integration`
- Or skip them: `pytest -m "not integration"`

### "CUDA is not available"
- Tests should work on CPU too
- GPU is optional; CPU tests just run slower

## Best Practices

1. **Keep tests isolated**: Each test should be independent
2. **Use fixtures**: Reuse common setup via conftest.py
3. **Mock external deps**: Don't require real models for unit tests
4. **Clear assertions**: Make it obvious what's being tested
5. **Meaningful names**: Test name should describe what's being tested
6. **Fast execution**: Aim for <100ms per unit test

## Adding New Tests

1. Create test file in `tests/` with name `test_*.py`
2. Import fixtures from conftest.py
3. Use descriptive class/function names
4. Add docstrings explaining what's tested
5. Mark with appropriate pytest markers (`@pytest.mark.integration`, etc.)

Example:
```python
class TestNewFeature:
    """Test suite for new feature."""

    def test_basic_functionality(self, sample_image_rgb):
        """Test that feature does X."""
        result = feature_function(sample_image_rgb)
        assert result is not None

    def test_error_handling(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError):
            feature_function(None)
```
