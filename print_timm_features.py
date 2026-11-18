import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm

# ---- IMPORT YOUR DATASET ----
from megatron.core.datasets.utils import Split
from megatron.core.datasets.vision_dataset import MegatronVisionDataset  # <-- change this

# =========================
# CONFIG
# =========================
DATA_PATH = "/home/hatef.4/datasets/imagenet-1-image" #imagenet-1k/ILSVRC_subset_128/train"  # root dir
MODEL_NAME = "vit_large_patch16_rope_224"
BATCH_SIZE = 1
IMAGE_SIZE = 224
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# =========================
# SIMPLE CONFIG OBJECT
# =========================
class Config:
    def __init__(self):
        self.random_seed = SEED
        self.image_size = IMAGE_SIZE
        self.batch_size = BATCH_SIZE
        self.split = None
        self.split_matrix = None


config = Config()

# =========================
# BUILD DATASET
# =========================
low_level_dataset = MegatronVisionDataset.build_low_level_dataset(
    DATA_PATH,
    config,
)

indices = np.arange(len(low_level_dataset), dtype=np.int32)
rng = np.random.default_rng(SEED)
rng.shuffle(indices)

n = len(indices)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_idx = indices #[:train_end]
valid_idx = indices # [train_end:val_end]
test_idx  = indices[val_end:]

# pad to batch size
def pad(x):
    return np.pad(x, (0, (-len(x)) % BATCH_SIZE), mode="wrap")

train_idx = pad(train_idx)
valid_idx = pad(valid_idx)
test_idx  = pad(test_idx)

train_ds = MegatronVisionDataset(
    low_level_dataset, DATA_PATH, train_idx, None, Split.train, config
)

valid_ds = MegatronVisionDataset(
    low_level_dataset, DATA_PATH, valid_idx, None, Split.valid, config
)

test_ds = MegatronVisionDataset(
    low_level_dataset, DATA_PATH, test_idx, None, Split.test, config
)

# =========================
# DATALOADER
# =========================
valid_loader = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# =========================
# MODEL
# =========================
model = timm.create_model(MODEL_NAME, pretrained=True)
model = model.to(DEVICE)
model.eval()

'''for name, param in model.named_parameters():
        if param is None:
            print(f"{name} is none")
        flat = param.data.view(-1)  # flatten tensor
        print(f"{name}: {flat[:5]}")'''

# =========================
# EVAL LOOP
# =========================
def evaluate(model, dataloader):
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(DEVICE).float()
            labels = batch["labels"].to(DEVICE).long()

            logits = model(images)

            loss = F.cross_entropy(logits, labels)

            # ---- Top-1 ----
            preds = torch.argmax(logits, dim=1)
            top1_correct = (preds == labels).sum().item()

            # ---- Top-5 ----
            top5_preds = torch.topk(logits, k=5, dim=1).indices
            # check if label is in top-5 predictions
            top5_correct = top5_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            total_loss += loss.item() * images.size(0)
            total_top1 += top1_correct
            total_top5 += top5_correct
            total_samples += images.size(0)
            print(f"Samples: {total_samples}/{len(dataloader) * BATCH_SIZE}")

    avg_loss = total_loss / total_samples
    top1_acc = total_top1 / total_samples
    top5_acc = total_top5 / total_samples

    return avg_loss, top1_acc, top5_acc


# =========================
# RUN EVAL
# =========================
val_loss, val_top1, val_top5 = evaluate(model, valid_loader)

print(f"Validation Loss: {val_loss:.4f}")
print(f"Top-1 Acc: {val_top1:.4f}")
print(f"Top-5 Acc: {val_top5:.4f}")