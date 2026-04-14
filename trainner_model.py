import os
import json
import copy
import random
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# =========================================================
# CONFIG
# =========================================================
JSON_PATH = "data_trainning.json"
IMAGES_DIR = "data/images"   # à adapter

IMG_SIZE = 260  # taille native B2 = 260
BATCH_SIZE = 16
NUM_WORKERS = 0
SEED = 42

NUM_EPOCHS_PHASE1 = 8
NUM_EPOCHS_PHASE2 = 12
NUM_EPOCHS_PHASE3 = 12
EARLY_STOPPING_PATIENCE = 5

BEST_MODEL_PATH = "best_model_aurora_effb2_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MixUp
MIXUP_ALPHA = 0.4
MIXUP_PROB = 0.4

# TTA
TTA_N = 3


# =========================================================
# REPRODUCTIBILITE
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================================================
# CLASSES
# =========================================================
CLASS_NAMES = [
    "Clairs sans aurores",
    "Nuageux sans aurores",
    "Clairs avec aurores légères",
    "Nuageux avec aurores légères",
    "Clairs avec aurores fortes",
    "Nuageux avec aurores fortes",
    "Obstrués",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


# =========================================================
# TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(12),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.02
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(
        p=0.15,
        scale=(0.02, 0.08),
        ratio=(0.3, 3.3)
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

tta_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.92, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================================================
# LECTURE JSON LABEL STUDIO
# =========================================================
def extract_filename(task):
    file_upload = task.get("file_upload")
    if file_upload:
        return os.path.basename(file_upload)

    data = task.get("data", {})
    image_path = data.get("image")
    if image_path:
        return os.path.basename(image_path)

    return None


def extract_label(task):
    annotations = task.get("annotations", [])
    if not annotations:
        return None

    for ann in annotations:
        results = ann.get("result", [])
        if not results:
            continue

        for res in results:
            value = res.get("value", {})
            choices = value.get("choices", [])
            if choices:
                return choices[0]

    return None


def load_samples_from_labelstudio(json_path, images_dir, class_to_idx):
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    samples = []
    skipped_no_label = 0
    skipped_bad_label = 0
    skipped_missing_file = 0

    for task in tasks:
        filename = extract_filename(task)
        label_name = extract_label(task)

        if label_name is None:
            skipped_no_label += 1
            continue

        if label_name not in class_to_idx:
            skipped_bad_label += 1
            continue

        if filename is None:
            skipped_missing_file += 1
            continue

        image_path = os.path.join(images_dir, filename)

        if not os.path.exists(image_path):
            skipped_missing_file += 1
            continue

        samples.append({
            "image_path": image_path,
            "label_name": label_name,
            "label_idx": class_to_idx[label_name],
            "filename": filename,
        })

    print(f"✅ {len(samples)} images valides trouvées")
    print(f"⚠️ Sans label : {skipped_no_label}")
    print(f"⚠️ Labels inconnus : {skipped_bad_label}")
    print(f"⚠️ Fichiers absents : {skipped_missing_file}")

    return samples


# =========================================================
# DATASET
# =========================================================
class AuroraDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        label = item["label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================================================
# MIXUP
# =========================================================
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a = y
    y_b = y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =========================================================
# PREPARATION DONNEES
# =========================================================
all_samples = load_samples_from_labelstudio(JSON_PATH, IMAGES_DIR, CLASS_TO_IDX)

if len(all_samples) == 0:
    raise ValueError("Aucune image exploitable trouvée.")

all_labels = np.array([s["label_idx"] for s in all_samples])

print("\nRépartition globale :")
global_counts = Counter(all_labels)
for class_idx in range(NUM_CLASSES):
    count = global_counts.get(class_idx, 0)
    pct = 100.0 * count / len(all_samples)
    print(f"  {CLASS_NAMES[class_idx]}: {count} ({pct:.2f}%)")

indices = np.arange(len(all_samples))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    stratify=all_labels,
    random_state=SEED
)

temp_labels = all_labels[temp_idx]

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SEED
)

train_samples = [all_samples[i] for i in train_idx]
val_samples = [all_samples[i] for i in val_idx]
test_samples = [all_samples[i] for i in test_idx]

print("\nRépartition après split :")
print(f"  Train: {len(train_samples)}")
print(f"  Val  : {len(val_samples)}")
print(f"  Test : {len(test_samples)}")

train_dataset = AuroraDataset(train_samples, transform=train_transform)
val_dataset = AuroraDataset(val_samples, transform=eval_transform)
test_dataset = AuroraDataset(test_samples, transform=eval_transform)


# =========================================================
# SAMPLER PONDERE + CLASS WEIGHTS
# =========================================================
train_labels = np.array([s["label_idx"] for s in train_samples])
train_counts = Counter(train_labels)

sample_weights = np.array([
    len(train_labels) / train_counts[label]
    for label in train_labels
], dtype=np.float64)

train_sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

class_weights = torch.tensor(
    [len(train_labels) / train_counts.get(i, 1) for i in range(NUM_CLASSES)],
    dtype=torch.float32
).to(DEVICE)

print("\nPoids de classes :")
for i, w in enumerate(class_weights):
    print(f"  {CLASS_NAMES[i]}: {w.item():.4f}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# =========================================================
# MODELE
# =========================================================
def build_model(num_classes):
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(256, num_classes)
    )

    return model


model = build_model(NUM_CLASSES).to(DEVICE)


def count_params(m):
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    return trainable, total


trainable, total = count_params(model)
print(f"\n🚀 Device : {DEVICE}")
print(f"Paramètres entraînables : {trainable:,} / {total:,}")


# =========================================================
# LOSS
# =========================================================
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.10
)


# =========================================================
# TRAIN / EVAL
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, use_mixup=False):
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        if use_mixup and random.random() < MIXUP_PROB:
            images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100.0 * np.mean(np.array(y_true) == np.array(y_pred))
    epoch_f1 = f1_score(y_true, y_pred, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100.0 * np.mean(np.array(y_true) == np.array(y_pred))
    epoch_f1 = f1_score(y_true, y_pred, average="macro")

    return epoch_loss, epoch_acc, epoch_f1, y_true, y_pred


def evaluate_tta(model, dataset, n_aug=3):
    model.eval()
    all_probs = []

    for _ in range(n_aug):
        tta_dataset = AuroraDataset(dataset.samples, transform=tta_transform)
        tta_loader = DataLoader(
            tta_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        probs_run = []
        with torch.no_grad():
            for images, _ in tta_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                probs_run.append(probs.cpu().numpy())

        all_probs.append(np.concatenate(probs_run, axis=0))

    mean_probs = np.mean(all_probs, axis=0)
    y_pred = np.argmax(mean_probs, axis=1)
    y_true = [s["label_idx"] for s in dataset.samples]

    acc = 100.0 * np.mean(np.array(y_true) == y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return acc, f1, y_true, y_pred


def run_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    phase_name,
    best_val_f1,
    best_model_wts,
    use_mixup=False,
    patience=EARLY_STOPPING_PATIENCE
):
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, use_mixup=use_mixup
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)

        scheduler.step()

        print(
            f"[{phase_name}] Epoch {epoch+1}/{num_epochs} | LR: {lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            epochs_no_improve = 0
            print(f"  ✅ Nouveau meilleur modèle (Val F1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ Pas d'amélioration depuis {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"\n🛑 Early stopping {phase_name} à l'epoch {epoch+1}")
            break

    return best_val_f1, best_model_wts


# =========================================================
# PHASE 1 : CLASSIFIER SEUL
# =========================================================
print("\n===== PHASE 1 : classifier uniquement =====")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=4,
    T_mult=1,
    eta_min=1e-5
)

best_val_f1 = -1.0
best_model_wts = copy.deepcopy(model.state_dict())

best_val_f1, best_model_wts = run_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    NUM_EPOCHS_PHASE1,
    "Phase1",
    best_val_f1,
    best_model_wts,
    use_mixup=False
)


# =========================================================
# PHASE 2 : DERNIERS BLOCS
# =========================================================
print("\n===== PHASE 2 : dégel blocs finaux =====")

for name, param in model.named_parameters():
    if any(f"features.{i}" in name for i in [5, 6, 7]):
        param.requires_grad = True

trainable, total = count_params(model)
print(f"Paramètres entraînables : {trainable:,} / {total:,}")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=6,
    T_mult=1,
    eta_min=1e-6
)

best_val_f1, best_model_wts = run_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    NUM_EPOCHS_PHASE2,
    "Phase2",
    best_val_f1,
    best_model_wts,
    use_mixup=True
)


# =========================================================
# PHASE 3 : OUVERTURE PARTIELLE PLUS LARGE
# =========================================================
print("\n===== PHASE 3 : fine-tuning plus large =====")

for name, param in model.named_parameters():
    if any(f"features.{i}" in name for i in [3, 4, 5, 6, 7]):
        param.requires_grad = True

trainable, total = count_params(model)
print(f"Paramètres entraînables : {trainable:,} / {total:,}")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=8,
    T_mult=1,
    eta_min=1e-7
)

best_val_f1, best_model_wts = run_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    NUM_EPOCHS_PHASE3,
    "Phase3",
    best_val_f1,
    best_model_wts,
    use_mixup=True,
    patience=6
)


# =========================================================
# CHARGEMENT MEILLEUR MODELE
# =========================================================
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
print(f"\n✅ Meilleur modèle chargé — Val F1 : {best_val_f1:.4f}")


# =========================================================
# TEST FINAL
# =========================================================
print("\n📊 Évaluation standard sur TEST :")
test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion)
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Acc : {test_acc:.2f}%")
print(f"  Test F1  : {test_f1:.4f}")

print(f"\n📊 Évaluation TTA sur TEST (n={TTA_N}) :")
tta_acc, tta_f1, y_true_tta, y_pred_tta = evaluate_tta(model, test_dataset, n_aug=TTA_N)
print(f"  TTA Acc: {tta_acc:.2f}%")
print(f"  TTA F1 : {tta_f1:.4f}")

print("\n📌 Matrice de confusion (TTA) :")
print(confusion_matrix(y_true_tta, y_pred_tta))

print("\n📌 Classification report (TTA) :")
print(classification_report(
    y_true_tta,
    y_pred_tta,
    target_names=CLASS_NAMES,
    digits=3
))