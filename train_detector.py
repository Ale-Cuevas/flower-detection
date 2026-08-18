"""
STEP 2 & 3 (detection version): fine-tune a pretrained Faster R-CNN.

This assumes you've exported a flower-detection dataset in COCO format
from Roboflow Universe (https://universe.roboflow.com -- search "flower
detection"). After export you'll have a folder like:

    dataset/
        train/
            _annotations.coco.json
            image1.jpg
            image2.jpg
            ...
        valid/
            _annotations.coco.json
            ...

Install extra dependency for reading COCO json:
    pip install pycocotools

Run:
    python 02_train_detector.py
"""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------------------
# 1. DATASET -- wraps a COCO-format folder into a PyTorch Dataset
# -----------------------------------------------------------------------
class FlowerDetectionDataset(Dataset):
    def __init__(self, root, ann_file):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        img = Image.open(os.path.join(self.root, img_info["file_name"])).convert("RGB")

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO format: [x_min, y_min, width, height]
            boxes.append([x, y, x + w, y + h])  # Faster R-CNN wants [x_min, y_min, x_max, y_max]
            labels.append(ann["category_id"])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        return self.transform(img), target


def collate_fn(batch):
    # Detection targets have variable-length boxes per image, so we
    # can't stack them into one tensor like a normal batch -- keep as a list.
    return tuple(zip(*batch))


# -----------------------------------------------------------------------
# 2. DATA LOADERS (edit these paths to match your downloaded dataset)
# -----------------------------------------------------------------------
DATA_ROOT = "./dataset"
NUM_CLASSES = 14  # 5 flower types + 1 background class (Faster R-CNN always needs a background class)

train_set = FlowerDetectionDataset(
    root=f"{DATA_ROOT}/train",
    ann_file=f"{DATA_ROOT}/train/_annotations.coco.json",
)
val_set = FlowerDetectionDataset(
    root=f"{DATA_ROOT}/valid",
    ann_file=f"{DATA_ROOT}/valid/_annotations.coco.json",
)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=collate_fn)

# -----------------------------------------------------------------------
# 3. MODEL: load pretrained Faster R-CNN, replace the box-prediction head
# -----------------------------------------------------------------------
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

# The pretrained model predicts 91 COCO classes -- swap the head for our NUM_CLASSES
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

model = model.to(device)

# -----------------------------------------------------------------------
# 4. FINE-TUNE
# -----------------------------------------------------------------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # In train mode, Faster R-CNN returns a dict of losses directly
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | total_loss={epoch_loss/len(train_loader):.3f}")

# -----------------------------------------------------------------------
# 5. SAVE
# -----------------------------------------------------------------------
save_path = "weights/flower_detector_fasterrcnn.pt"
torch.save(model.state_dict(), save_path)
print(f"\nSaved weights to {save_path}")
