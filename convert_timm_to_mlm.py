import torch
import timm


TIMM_MODEL = "hf_hub:naver-ai/rope_axial_deit_large_patch16_LS"
ckpt_dir = "checkpoints/iter_0000000"
OUTPUT=f"{ckpt_dir}/mp_rank_00/model_optim_rng.pt"
COPY_HEAD = True

print("Loading timm model...")

model = timm.create_model("vit_large_patch16_rope_224", pretrained=True)

state = model.state_dict()


megatron_state = {}
copied = set()


def copy(dst, src):

    if src not in state:
        print(f"WARNING: missing source key: {src}")
        return

    megatron_state[dst] = state[src]
    copied.add(src)


############################################
# Patch embedding
############################################

print("Converting patch embedding")

copy("patch_embed.weight", "patch_embed.proj.weight")
copy("patch_embed.bias", "patch_embed.proj.bias")


############################################
# CLS token
############################################

if "cls_token" in state:
    print("Copying CLS token")
    copy("cls_token", "cls_token")


############################################
# Absolute positional embeddings
############################################

if "pos_embed" in state:
    print("Copying positional embeddings")
    copy("pos_embed", "pos_embed")


############################################
# Transformer layers
############################################

num_layers = len(model.blocks)

print(f"Converting {num_layers} transformer layers")

for i in range(num_layers):

    timm_prefix = f"blocks.{i}"
    meg_prefix = f"encoder.layers.{i}"

    print(f"Layer {i}")


    ########################################
    # LayerNorm 1
    ########################################

    copy(
        f"{meg_prefix}.self_attention.linear_qkv.layer_norm_weight",
        f"{timm_prefix}.norm1.weight",
    )

    copy(
        f"{meg_prefix}.self_attention.linear_qkv.layer_norm_bias",
        f"{timm_prefix}.norm1.bias",
    )


    ########################################
    # LayerNorm 2
    ########################################

    copy(
        f"{meg_prefix}.mlp.linear_fc1.layer_norm_weight",
        f"{timm_prefix}.norm2.weight",
    )

    copy(
        f"{meg_prefix}.mlp.linear_fc1.layer_norm_bias",
        f"{timm_prefix}.norm2.bias",
    )


    ########################################
    # QKV projection
    ########################################

    copy(
        f"{meg_prefix}.self_attention.linear_qkv.weight",
        f"{timm_prefix}.attn.qkv.weight",
    )

    copy(
        f"{meg_prefix}.self_attention.linear_qkv.bias",
        f"{timm_prefix}.attn.qkv.bias",
    )


    ########################################
    # Attention output projection
    ########################################

    copy(
        f"{meg_prefix}.self_attention.linear_proj.weight",
        f"{timm_prefix}.attn.proj.weight",
    )

    copy(
        f"{meg_prefix}.self_attention.linear_proj.bias",
        f"{timm_prefix}.attn.proj.bias",
    )


    ########################################
    # MLP first layer
    ########################################

    copy(
        f"{meg_prefix}.mlp.linear_fc1.weight",
        f"{timm_prefix}.mlp.fc1.weight",
    )

    copy(
        f"{meg_prefix}.mlp.linear_fc1.bias",
        f"{timm_prefix}.mlp.fc1.bias",
    )


    ########################################
    # MLP second layer
    ########################################

    copy(
        f"{meg_prefix}.mlp.linear_fc2.weight",
        f"{timm_prefix}.mlp.fc2.weight",
    )

    copy(
        f"{meg_prefix}.mlp.linear_fc2.bias",
        f"{timm_prefix}.mlp.fc2.bias",
    )


############################################
# Final LayerNorm
############################################

if "norm.weight" in state:

    print("Copying final LayerNorm")

    copy("encoder.final_layernorm.weight", "norm.weight")
    copy("encoder.final_layernorm.bias", "norm.bias")


############################################
# classifier head (ALWAYS INIT NEW)
############################################

import torch.nn as nn

print("Initializing Megatron classification head")

if "head.weight" in state and COPY_HEAD:
    print("copying head")
    
    copy("head.weight", "head.weight")
    copy("head.bias", "head.bias")
    print(f"Using checkpoint head")
else:
    # infer target num classes (fallback to timm head if mismatch-safe)
    target_num_classes = None

    if hasattr(model, "head") and model.head is not None:
        if hasattr(model.head, "weight"):
            target_num_classes = model.head.weight.shape[0]

    # fallback (safe default for ViT pretrained on ImageNet)
    if target_num_classes is None:
        target_num_classes = 1000

    hidden_size = model.embed_dim  # timm ViT attribute

    # initialize fresh head weights (Megatron format)
    target_num_classes = 1000
    head_weight = torch.empty(target_num_classes, hidden_size)
    head_bias = torch.zeros(target_num_classes)

    torch.nn.init.trunc_normal_(head_weight, std=0.02)
    torch.nn.init.zeros_(head_bias)
    print(f"Created new head: {target_num_classes} x {hidden_size}")

    megatron_state["head.weight"] = head_weight
    megatron_state["head.bias"] = head_bias

############################################
# Detect skipped parameters
############################################

unused = sorted(set(state.keys()) - copied)

print("\n====================================")
print("UNUSED TIMM PARAMETERS")
print("====================================")

for k in unused:
    print(k)


############################################
# Highlight important skipped features
############################################

important_patterns = [
    "relative_position",
    "rel_pos",
    "bias_table",
    "rope",
    "gamma",
    "ls",
    "dist",
    "register",
]

important_unused = [
    k for k in unused
    if any(p in k.lower() for p in important_patterns)
]

if unused:

    print("\n⚠️ IMPORTANT FEATURES NOT IMPORTED")

    for k in unused:
        print(k)


############################################
# Summary stats
############################################

print("\n====================================")
print("SUMMARY")
print("====================================")

print("Total timm params:", len(state))
print("Copied params:", len(copied))
print("Unused params:", len(unused))


############################################
# Save checkpoint
############################################

checkpoint = {
    "checkpoint_version": 3.0,
    "iteration": 0,
    "model": megatron_state,
    "optimizer": None,
    "opt_param_scheduler": None,
    "rng_state": None,
    "args": None,
    "num_floating_point_operations_so_far": 0,
}

torch.save(
    checkpoint,
    OUTPUT
)

print("\nSaved Megatron checkpoint →", OUTPUT)


