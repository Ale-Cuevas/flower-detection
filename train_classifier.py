"""
STEP 2 & 3: Load a pretrained ResNet18 and fine-tune it on Flowers102.

Run:
    pip install torch torchvision matplotlib
    python 01_train_classifier.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import Flowers102

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------
# 1. DATA
# -----------------------------------------------------------------------
# ImageNet mean/std -- required because we're using ImageNet-pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Downloads automatically the first time you run this
train_set = Flowers102(root="./data", split="train", download=True, transform=train_transform)
val_set = Flowers102(root="./data", split="val", download=True, transform=val_transform)

NUM_CLASSES = 102

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)

# -----------------------------------------------------------------------
# 2. MODEL: load pretrained ResNet18, replace the final layer
# -----------------------------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all pretrained layers first -- we only train the new head initially.
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully-connected layer (was 1000 ImageNet classes -> 102 flower classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)  # this new layer is trainable by default

model = model.to(device)

# -----------------------------------------------------------------------
# 3. TRAIN (Phase 1: train only the new head, backbone frozen)
# -----------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(is_train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


PHASE1_EPOCHS = 5
print("\n--- Phase 1: training only the new classifier head ---")
for epoch in range(PHASE1_EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{PHASE1_EPOCHS} | "
          f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
          f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

# -----------------------------------------------------------------------
# 4. FINE-TUNE (Phase 2: unfreeze the last conv block, train with a lower LR)
# -----------------------------------------------------------------------
# This is the "fine-tuning" step proper: now that the new head has learned
# something sensible, we unfreeze part of the backbone and adapt its
# high-level features to flowers specifically, using a small learning rate
# so we don't destroy the pretrained weights.
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    [
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4},
    ]
)

PHASE2_EPOCHS = 5
print("\n--- Phase 2: fine-tuning layer4 + head ---")
for epoch in range(PHASE2_EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{PHASE2_EPOCHS} | "
          f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
          f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

# -----------------------------------------------------------------------
# 5. SAVE
# -----------------------------------------------------------------------
save_path = "weights/flower_classifier_resnet18.pt"
torch.save(model.state_dict(), save_path)
print(f"\nSaved weights to {save_path}")
