import os
import copy
import random
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# =========================================================
# CONFIG
# =========================================================
IMAGES_DIR = "data/pseudo_labels_dataset"

IMG_SIZE = 224
BATCH_SIZE = 16   # batch size PAR processus
NUM_WORKERS = 4
SEED = 42

NUM_EPOCHS_PHASE1 = 8
NUM_EPOCHS_PHASE2 = 10
EARLY_STOPPING_PATIENCE = 4

PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.10

BEST_MODEL_PATH = "best_model_resnet18_ddp.pth"

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
# UTILS & DDP
# =========================================================
def setup_ddp():
    """Initialise le process group DDP sur CPU."""
    dist.init_process_group(backend="gloo")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return local_rank, global_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


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
    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.015),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =========================================================
# FOLDER PARSING
# =========================================================
def load_samples_from_folders(images_dir, class_to_idx, global_rank):
    samples = []
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

    for class_name, idx in class_to_idx.items():
        class_dir = os.path.join(images_dir, class_name)

        if not os.path.exists(class_dir) or not os.path.isdir(class_dir):
            if global_rank == 0:
                print(f"⚠️ Dossier introuvable : '{class_name}'")
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(valid_extensions):
                samples.append({
                    "image_path": os.path.join(class_dir, filename),
                    "label_name": class_name,
                    "label_idx": idx,
                })

    if global_rank == 0:
        print(f"✅ {len(samples)} images trouvées.")
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
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to("cpu")
        labels = labels.to("cpu")

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
    epoch_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


def evaluate(model, loader, criterion):
    """
    Important :
    - si model est wrappé en DDP
    - et que seule la machine/rank 0 fait l'évaluation
    alors il faut utiliser model.module et NON model(...)
    """
    eval_model = model.module if isinstance(model, DDP) else model
    eval_model.eval()

    running_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to("cpu")
            labels = labels.to("cpu")

            outputs = eval_model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100.0 * np.mean(np.array(y_true) == np.array(y_pred))
    epoch_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1, y_true, y_pred


def run_phase(
        model,
        train_loader,
        val_loader,
        train_sampler,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        phase_name,
        best_val_f1,
        global_rank,
        patience=4,
        epoch_offset=0,
):
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Important en DDP : même si chaque rank voit sa partition,
        # il faut reseeder le sampler à chaque epoch
        train_sampler.set_epoch(epoch + epoch_offset)

        lr = optimizer.param_groups[0]["lr"]

        # 1. Train sur tous les ranks
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        scheduler.step()

        # 2. Barrière : tout le monde termine le train avant de passer à la suite
        dist.barrier()

        # 3. Validation uniquement sur rank 0, avec model.module dans evaluate()
        if global_rank == 0:
            val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)

            print(
                f"[{phase_name}] Epoch {epoch + 1}/{num_epochs} | "
                f"LR: {lr:.6f} | "
                f"Train F1: {train_f1:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.module.state_dict(), BEST_MODEL_PATH)
                epochs_no_improve = 0
                print(f"  ✅ Nouveau meilleur modèle (Val F1={best_val_f1:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  ⏳ Pas d'amélioration depuis {epochs_no_improve} epoch(s)")

            stop_signal = torch.tensor(
                [float(epochs_no_improve), float(best_val_f1)],
                dtype=torch.float32
            ).to("cpu")
        else:
            stop_signal = torch.tensor([0.0, 0.0], dtype=torch.float32).to("cpu")

        # 4. Broadcast du signal depuis rank 0 vers tous les ranks
        dist.broadcast(stop_signal, src=0)

        epochs_no_improve = int(stop_signal[0].item())
        best_val_f1 = float(stop_signal[1].item())

        if epochs_no_improve >= patience:
            if global_rank == 0:
                print(f"\n🛑 Early stopping {phase_name} à l'epoch {epoch + 1}")
            break

        # 5. Barrière avant epoch suivante
        dist.barrier()

    return best_val_f1


# =========================================================
# MAIN
# =========================================================
def main():
    local_rank, global_rank, world_size = setup_ddp()
    set_seed(SEED + global_rank)

    try:
        if global_rank == 0:
            print(f"🚀 Début de l'entraînement DDP (CPU) sur {world_size} processus.")

        # =====================================================
        # 1. Chargement des données
        # =====================================================
        all_samples = load_samples_from_folders(IMAGES_DIR, CLASS_TO_IDX, global_rank)

        if len(all_samples) == 0:
            raise ValueError("Aucune image exploitable trouvée.")

        all_labels = np.array([s["label_idx"] for s in all_samples])

        # Split identique sur tous les ranks
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

        train_dataset = AuroraDataset(train_samples, transform=train_transform)
        val_dataset = AuroraDataset(val_samples, transform=eval_transform)
        test_dataset = AuroraDataset(test_samples, transform=eval_transform)

        # =====================================================
        # 2. Class weights
        # =====================================================
        train_labels = np.array([s["label_idx"] for s in train_samples])
        train_counts = Counter(train_labels)

        class_weights = torch.tensor(
            [len(train_labels) / train_counts.get(i, 1) for i in range(NUM_CLASSES)],
            dtype=torch.float32
        ).to("cpu")

        # =====================================================
        # 3. Sampler / Loaders
        # =====================================================
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            drop_last=False,
        )

        loader_kwargs = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": False,
        }

        train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        # =====================================================
        # 4. Modèle phase 1
        # =====================================================
        base_model = build_model(NUM_CLASSES).to("cpu")
        trainable, total = count_params(base_model)

        if global_rank == 0:
            print(f"📦 Paramètres entraînables phase 1 : {trainable} / {total}")

        model = DDP(base_model)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING
        )

        # =====================================================
        # PHASE 1
        # =====================================================
        if global_rank == 0:
            print("\n===== PHASE 1 : classifier uniquement =====")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=PHASE1_LR,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=4, T_mult=1, eta_min=1e-5
        )

        best_val_f1 = -1.0

        best_val_f1 = run_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=NUM_EPOCHS_PHASE1,
            phase_name="Phase1",
            best_val_f1=best_val_f1,
            global_rank=global_rank,
            patience=EARLY_STOPPING_PATIENCE,
            epoch_offset=0,
        )

        # =====================================================
        # PHASE 2 : rewrap propre DDP
        # =====================================================
        if global_rank == 0:
            print("\n===== PHASE 2 : fine-tuning layer4 =====")

        # Important : on synchronise tous les ranks avant de modifier le modèle
        dist.barrier()

        # Déwrap propre
        base_model = model.module
        del model

        # Défreeze layer4 AVANT de rewrapper en DDP
        for param in base_model.layer4.parameters():
            param.requires_grad = True

        trainable, total = count_params(base_model)
        if global_rank == 0:
            print(f"📦 Paramètres entraînables phase 2 : {trainable} / {total}")

        # Rewrap DDP
        model = DDP(base_model)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=PHASE2_LR,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=6, T_mult=1, eta_min=1e-6
        )

        best_val_f1 = run_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=NUM_EPOCHS_PHASE2,
            phase_name="Phase2",
            best_val_f1=best_val_f1,
            global_rank=global_rank,
            patience=EARLY_STOPPING_PATIENCE,
            epoch_offset=NUM_EPOCHS_PHASE1,
        )

        # =====================================================
        # TEST FINAL uniquement sur rank 0
        # =====================================================
        dist.barrier()

        if global_rank == 0:
            model.module.load_state_dict(
                torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True)
            )

            print(f"\n✅ Meilleur modèle chargé — Val F1 : {best_val_f1:.4f}")

            test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
                model, test_loader, criterion
            )

            print("\n📊 Évaluation finale sur TEST :")
            print(f"Test Loss: {test_loss:.4f} | Test Acc : {test_acc:.2f}% | Test F1 : {test_f1:.4f}")
            print("\n📌 Matrice de confusion :\n", confusion_matrix(y_true, y_pred))
            print(
                "\n📌 Classification report :\n",
                classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
            )

        dist.barrier()

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
