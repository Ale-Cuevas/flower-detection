"""Core flower detection and classification pipeline."""

import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DETECTOR_WEIGHTS = "weights/flower_detector_fasterrcnn.pt"
DETECTOR_NUM_CLASSES = 6

CLASSIFIER_WEIGHTS = "weights/flower_classifier_resnet18.pt"
CLASSIFIER_NUM_CLASSES = 102

CAT_TO_NAME_URL = "https://raw.githubusercontent.com/udacity/pytorch_challenge/master/cat_to_name.json"
CAT_TO_NAME_LOCAL = "cat_to_name.json"


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_class_names() -> Dict[int, str]:
    """Download or load flower name mapping (102 Flowers102 classes)."""
    try:
        with open(CAT_TO_NAME_LOCAL) as f:
            cat_to_name = json.load(f)
    except FileNotFoundError:
        urllib.request.urlretrieve(CAT_TO_NAME_URL, CAT_TO_NAME_LOCAL)
        with open(CAT_TO_NAME_LOCAL) as f:
            cat_to_name = json.load(f)
    return {idx: cat_to_name[str(idx + 1)] for idx in range(CLASSIFIER_NUM_CLASSES)}


class FlowerDetector:
    """Faster R-CNN detector for flower localization."""

    def __init__(self, weights_path: str = DETECTOR_WEIGHTS, num_classes: int = DETECTOR_NUM_CLASSES):
        self.device = get_device()
        self.num_classes = num_classes
        self.weights_path = weights_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load pretrained Faster R-CNN and replace head for custom classes."""
        try:
            model = fasterrcnn_resnet50_fpn_v2(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            model.to(self.device).eval()
            self.model = model
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Detector weights not found at {self.weights_path}. "
                f"Run 02_train_detector.py to train the model first."
            )

    def detect(self, image: Image.Image, score_threshold: float = 0.5) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect flowers in an image.

        Args:
            image: PIL Image (RGB)
            score_threshold: Minimum detection confidence [0, 1]

        Returns:
            List of (x1, y1, x2, y2, score) tuples for each detection above threshold
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Check that weights file exists.")

        x = transforms.ToTensor()(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(x)[0]

        results = []
        for box, score in zip(prediction["boxes"], prediction["scores"]):
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append((x1, y1, x2, y2, score.item()))
        return results

    def is_loaded(self) -> bool:
        """Check if model is successfully loaded."""
        return self.model is not None


class FlowerClassifier:
    """ResNet18 classifier for flower species identification."""

    def __init__(self, weights_path: str = CLASSIFIER_WEIGHTS, num_classes: int = CLASSIFIER_NUM_CLASSES):
        self.device = get_device()
        self.num_classes = num_classes
        self.weights_path = weights_path
        self.model = None
        self.class_names = build_class_names()
        self._load_model()

    def _load_model(self):
        """Load fine-tuned ResNet18 classifier."""
        try:
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
            model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            model.to(self.device).eval()
            self.model = model
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Classifier weights not found at {self.weights_path}. "
                f"Run 01_train_classifier.py to train the model first."
            )

    def classify(self, image: Image.Image) -> Tuple[str, float]:
        """
        Classify a flower image.

        Args:
            image: PIL Image (RGB)

        Returns:
            Tuple of (flower_name, confidence)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Check that weights file exists.")

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        x = preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        name = self.class_names.get(idx.item(), f"class_{idx.item()}")
        return name, conf.item()

    def is_loaded(self) -> bool:
        """Check if model is successfully loaded."""
        return self.model is not None


def pad_to_square(image: Image.Image, fill: Tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    """
    Pad image to square without cropping (preserves all content).

    Args:
        image: PIL Image
        fill: RGB fill color for padding

    Returns:
        Square PIL Image
    """
    w, h = image.size
    side = max(w, h)
    padded = Image.new("RGB", (side, side), fill)
    paste_x = (side - w) // 2
    paste_y = (side - h) // 2
    padded.paste(image, (paste_x, paste_y))
    return padded


class FlowerPipeline:
    """Two-stage pipeline: detect flowers, then classify each."""

    def __init__(
        self,
        detector_weights: str = DETECTOR_WEIGHTS,
        classifier_weights: str = CLASSIFIER_WEIGHTS,
        detector_num_classes: int = DETECTOR_NUM_CLASSES,
        classifier_num_classes: int = CLASSIFIER_NUM_CLASSES,
    ):
        self.detector = FlowerDetector(detector_weights, detector_num_classes)
        self.classifier = FlowerClassifier(classifier_weights, classifier_num_classes)

    def predict(self, image: Image.Image, detection_threshold: float = 0.5) -> List[Dict]:
        """
        Detect and classify flowers in an image.

        Args:
            image: PIL Image (RGB)
            detection_threshold: Minimum detection confidence [0, 1]

        Returns:
            List of dicts with keys: box, detection_score, flower, classification_confidence
        """
        boxes = self.detector.detect(image, detection_threshold)
        results = []

        for box in boxes:
            x1, y1, x2, y2, det_score = box
            crop = image.crop((x1, y1, x2, y2))
            crop = pad_to_square(crop)

            flower_name, class_conf = self.classifier.classify(crop)
            results.append({
                "box": (x1, y1, x2, y2),
                "detection_score": det_score,
                "flower": flower_name,
                "classification_confidence": class_conf,
            })

        return results

    def is_ready(self) -> bool:
        """Check if both models are loaded."""
        return self.detector.is_loaded() and self.classifier.is_loaded()
