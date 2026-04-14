import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ─── CONFIG ────────────────────────────────────────────────────────────────────
JSON_PATH = "data_trainning.json"
IMAGES_DIR = "data/images/"
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-5
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EARLY_STOPPING_PATIENCE = 5
BEST_MODEL_PATH = "best_model_aurora.pth"

# ─── SEED ──────────────────────────────────────────────────────────────────────
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── CLASSES ───────────────────────────────────────────────────────────────────
CLASSES = [
    "Clairs sans aurores",
    "Nuageux sans aurores",
    "Clairs avec aurores légères",
    "Nuageux avec aurores légères",
    "Clairs avec aurores fortes",
    "Nuageux avec aurores fortes",
    "Obstrués"
]
label2idx = {label: i for i, label in enumerate(CLASSES)}

# ─── 1. LECTURE DU JSON ────────────────────────────────────────────────────────
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = []
for item in data:
    try:
        image_path = Path(item["file_upload"])
        label = item["annotations"][0]["result"][0]["value"]["choices"][0]
        full_path = Path(IMAGES_DIR) / image_path.name

        if full_path.exists() and label in label2idx:
            samples.append((str(full_path), label2idx[label]))
    except (IndexError, KeyError, TypeError):
        continue

print(f"✅ {len(samples)} images valides trouvées\n")

print("Répartition globale :")
for i, cls in enumerate(CLASSES):
    count = sum(1 for _, l in samples if l == i)
    pct = (count / len(samples)) * 100 if len(samples) > 0 else 0
    print(f"  {cls}: {count} ({pct:.2f}%)")

if len(samples) < 10:
    raise ValueError("Pas assez d'images valides pour entraîner un modèle.")

# ─── 2. SPLIT STRATIFIÉ TRAIN / VAL / TEST ─────────────────────────────────────
paths = [s[0] for s in samples]
labels = [s[1] for s in samples]

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=(1 - TRAIN_RATIO),
    stratify=labels,
    random_state=SEED
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=SEED
)

train_samples = list(zip(train_paths, train_labels))
val_samples = list(zip(val_paths, val_labels))
test_samples = list(zip(test_paths, test_labels))

print("\nRépartition après split :")
print(f"  Train: {len(train_samples)}")
print(f"  Val  : {len(val_samples)}")
print(f"  Test : {len(test_samples)}")

# ─── 3. DATASET ────────────────────────────────────────────────────────────────
class AuroraDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ─── 4. TRANSFORMS ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.90, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = AuroraDataset(train_samples, transform=train_transform)
val_dataset = AuroraDataset(val_samples, transform=eval_transform)
test_dataset = AuroraDataset(test_samples, transform=eval_transform)

# ─── 5. WEIGHTED RANDOM SAMPLER ────────────────────────────────────────────────
train_labels_only = [label for _, label in train_samples]
class_counts = [train_labels_only.count(i) for i in range(len(CLASSES))]
class_sample_weights = [1.0 / count if count > 0 else 0.0 for count in class_counts]
sample_weights = [class_sample_weights[label] for label in train_labels_only]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ─── 6. DEVICE ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Entraînement sur : {device}")

# ─── 7. MODÈLE ─────────────────────────────────────────────────────────────────
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

# Fine-tuning partiel : layer3 + layer4 + fc
for param in model.parameters():
    param.requires_grad = False

for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Paramètres entraînables : {trainable_params} / {total_params}")

# ─── 8. LOSS / OPTIMIZER / SCHEDULER ───────────────────────────────────────────
train_class_counts = [sum(1 for _, l in train_samples if l == i) for i in range(len(CLASSES))]
class_weights = []
for count in train_class_counts:
    if count == 0:
        class_weights.append(0.0)
    else:
        class_weights.append(len(train_samples) / count)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# ─── 9. EVALUATION ─────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = 100 * correct / total if total > 0 else 0.0
    return avg_loss, acc, all_labels, all_preds

# ─── 10. ENTRAÎNEMENT + EARLY STOPPING ─────────────────────────────────────────
best_val_acc = 0.0
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total if total > 0 else 0.0
    train_acc = 100 * correct / total if total > 0 else 0.0

    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"LR: {current_lr:.6f} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  ✅ Nouveau meilleur modèle sauvegardé")
    else:
        epochs_without_improvement += 1
        print(f"  ⏳ Pas d'amélioration depuis {epochs_without_improvement} epoch(s)")

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\n🛑 Early stopping déclenché à l'epoch {epoch+1}")
        break

print(f"\n✅ Meilleure accuracy validation : {best_val_acc:.2f}%")
print(f"✅ Modèle chargé depuis : {BEST_MODEL_PATH}")

# ─── 11. TEST FINAL ────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

print("\n📊 Résultats finaux sur TEST")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc : {test_acc:.2f}%")

print("\n📌 Matrice de confusion :")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\n📌 Classification report :")
print(classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    zero_division=0
))