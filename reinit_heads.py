import torch

INPUT_CKPT = "checkpoints-warmup-224/iter_0002817/mp_rank_00/model_optim_rng.pt"
OUTPUT_CKPT = "checkpoints-warmup-500/iter_0000000/mp_rank_00/model_optim_rng.pt"

NEW_NUM_CLASSES = 500

print("Loading checkpoint...")
ckpt = torch.load(INPUT_CKPT, map_location="cpu", weights_only=False)

model = ckpt["model"]

# ----------------------------------------
# Inspect old head
# ----------------------------------------

old_weight = model["head.weight"]
old_bias = model["head.bias"]

print("Old head weight:", old_weight.shape)
print("Old head bias:", old_bias.shape)

hidden_size = old_weight.shape[1]

# ----------------------------------------
# Create new head
# ----------------------------------------

new_weight = torch.empty(NEW_NUM_CLASSES, hidden_size)
new_bias = torch.zeros(NEW_NUM_CLASSES)

torch.nn.init.trunc_normal_(new_weight, std=0.02)
torch.nn.init.zeros_(new_bias)

# ----------------------------------------
# Replace checkpoint tensors
# ----------------------------------------

model["head.weight"] = new_weight
model["head.bias"] = new_bias

print("New head weight:", model["head.weight"].shape)
print("New head bias:", model["head.bias"].shape)

# ----------------------------------------
# Save updated checkpoint
# ----------------------------------------

import os

os.makedirs(os.path.dirname(OUTPUT_CKPT), exist_ok=True)

torch.save(ckpt, OUTPUT_CKPT)

print(f"Saved modified checkpoint to: {OUTPUT_CKPT}")
