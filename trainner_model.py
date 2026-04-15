import os
import copy
import random
from collections import Counter
from multiprocessing import freeze_support

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
# Dossier parent contenant les sous-dossiers par classe
IMAGES_DIR = "data/image_test"

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0   # Windows safe
SEED = 42

NUM_EPOCHS_PHASE1 = 8
NUM_EPOCHS_PHASE2 = 10
EARLY_STOPPING_PATIENCE = 4

PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.10

BEST_MODEL_PATH = "best_model_resnet18_flat_final2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# UTILS
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# =========================================================
# TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.90, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(
        brightness=0.12,
        contrast=0.12,
        saturation=0.06,
        hue=0.015
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
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


# =========================================================
# FOLDER PARSING
# =========================================================
def load_samples_from_folders(images_dir, class_to_idx):
    """
    Parcourt les sous-dossiers correspondant aux classes pour extraire
    les chemins d'images et leurs labels.
    """
    samples = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    for class_name, idx in class_to_idx.items():
        class_dir = os.path.join(images_dir, class_name)
        
        if not os.path.exists(class_dir) or not os.path.isdir(class_dir):
            print(f"⚠️ Dossier introuvable pour la classe : '{class_name}' (Chemin: {class_dir})")
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(class_dir, filename)
                samples.append({
                    "image_path": image_path,
                    "label_name": class_name,
                    "label_idx": idx,
                    "filename": filename,
                })

    print(f"✅ {len(samples)} images valides trouvées dans les dossiers.")
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
# MODEL
# =========================================================
def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze du backbone
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(in_features, num_classes)
    )

    return model


# =========================================================
# TRAIN / EVAL
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
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


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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
    device,
    patience=4,
):
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

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
# MAIN
# =========================================================
def main():
    set_seed(SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    print(f"Dossier source : {IMAGES_DIR}")
    
    # Chargement depuis les dossiers
    all_samples = load_samples_from_folders(IMAGES_DIR, CLASS_TO_IDX)
    
    if len(all_samples) == 0:
        raise ValueError("Aucune image exploitable trouvée dans les dossiers.")

    all_labels = np.array([s["label_idx"] for s in all_samples])

    print("\nRépartition globale :")
    global_counts = Counter(all_labels)
    for class_idx in range(NUM_CLASSES):
        count = global_counts.get(class_idx, 0)
        pct = 100.0 * count / len(all_samples) if len(all_samples) > 0 else 0
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

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )

    model = build_model(NUM_CLASSES).to(DEVICE)
    trainable, total = count_params(model)

    print(f"\n🚀 Device : {DEVICE}")
    print(f"Paramètres entraînables au départ : {trainable:,} / {total:,}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    # =====================================================
    # PHASE 1
    # =====================================================
    print("\n===== PHASE 1 : classifier uniquement =====")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR,
        weight_decay=WEIGHT_DECAY
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
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS_PHASE1,
        phase_name="Phase1",
        best_val_f1=best_val_f1,
        best_model_wts=best_model_wts,
        device=DEVICE,
        patience=EARLY_STOPPING_PATIENCE,
    )

    # =====================================================
    # PHASE 2
    # =====================================================
    print("\n===== PHASE 2 : fine-tuning layer4 =====")

    for param in model.layer4.parameters():
        param.requires_grad = True

    trainable, total = count_params(model)
    print(f"Paramètres entraînables phase 2 : {trainable:,} / {total:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE2_LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=6,
        T_mult=1,
        eta_min=1e-6
    )

    best_val_f1, best_model_wts = run_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS_PHASE2,
        phase_name="Phase2",
        best_val_f1=best_val_f1,
        best_model_wts=best_model_wts,
        device=DEVICE,
        patience=EARLY_STOPPING_PATIENCE,
    )

    # =====================================================
    # TEST FINAL
    # =====================================================
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    print(f"\n✅ Meilleur modèle chargé — Val F1 : {best_val_f1:.4f}")

    print("\n📊 Évaluation finale sur TEST :")
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.2f}%")
    print(f"Test F1  : {test_f1:.4f}")

    print("\n📌 Matrice de confusion :")
    print(confusion_matrix(y_true, y_pred))

    print("\n📌 Classification report :")
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=3
    ))


if __name__ == "__main__":
    freeze_support()
    main()