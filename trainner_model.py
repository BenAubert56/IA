import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ─── CONFIG ────────────────────────────────────────────────────────────────────
JSON_PATH   = "data_trainning.json"
IMAGES_DIR  = "data/images/"   # dossier contenant les .jpg
BATCH_SIZE  = 16
EPOCHS      = 50
LR          = 1e-3
SEED        = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
# TRAIN + VAL + TEST = 1.0
# ───────────────────────────────────────────────────────────────────────────────

# ─── SEED / REPRODUCTIBILITÉ ───────────────────────────────────────────────────
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
idx2label = {i: label for label, i in label2idx.items()}

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
    print(f"  {cls}: {count}")

if len(samples) < 10:
    raise ValueError("Pas assez d'images valides pour entraîner un modèle.")

# ─── 2. SPLIT STRATIFIÉ TRAIN / VAL / TEST ─────────────────────────────────────
paths = [s[0] for s in samples]
labels = [s[1] for s in samples]

# D'abord on sépare TRAIN du reste
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=(1 - TRAIN_RATIO),
    stratify=labels,
    random_state=SEED
)

# Puis on sépare le reste en VAL / TEST
# Comme temp = 30%, on coupe en deux pour faire 15% / 15%
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=SEED
)

train_samples = list(zip(train_paths, train_labels))
val_samples   = list(zip(val_paths, val_labels))
test_samples  = list(zip(test_paths, test_labels))

print("\nRépartition après split :")
print(f"  Train: {len(train_samples)}")
print(f"  Val  : {len(val_samples)}")
print(f"  Test : {len(test_samples)}")

# ─── 3. DATASET PERSONNALISÉ ───────────────────────────────────────────────────
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

# ─── 4. TRANSFORMS ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
val_dataset   = AuroraDataset(val_samples, transform=eval_transform)
test_dataset  = AuroraDataset(test_samples, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ─── 5. DEVICE ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Entraînement sur : {device}")

# ─── 6. MODÈLE ──────────────────────────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# On gèle le backbone pour un premier POC
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(device)

# ─── 7. LOSS / OPTIMIZER ────────────────────────────────────────────────────────
# Pondération simple des classes selon leur fréquence dans le train
train_class_counts = [sum(1 for _, l in train_samples if l == i) for i in range(len(CLASSES))]
class_weights = []
for count in train_class_counts:
    if count == 0:
        class_weights.append(0.0)
    else:
        class_weights.append(len(train_samples) / count)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# ─── 8. FONCTIONS D'ÉVALUATION ─────────────────────────────────────────────────
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

    avg_loss = total_loss / total if total > 0 else 0
    acc = 100 * correct / total if total > 0 else 0
    return avg_loss, acc, all_labels, all_preds

# ─── 9. ENTRAÎNEMENT ────────────────────────────────────────────────────────────
best_val_acc = 0.0
best_model_path = "best_model_aurora.pth"

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

    train_loss = total_loss / total if total > 0 else 0
    train_acc = 100 * correct / total if total > 0 else 0

    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

print(f"\n✅ Meilleur modèle sauvegardé : {best_model_path}")
print(f"✅ Meilleure accuracy validation : {best_val_acc:.2f}%")

# ─── 10. TEST FINAL ─────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(best_model_path, map_location=device))

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