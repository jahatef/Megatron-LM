import torch
from transformers import AutoModel


HF_MODEL = "google/vit-base-patch16-224"
OUTPUT = "vit_base_patch16_megatron.pt"


print("Loading HuggingFace ViT...")
hf_model = AutoModel.from_pretrained(HF_MODEL)
hf_state = hf_model.state_dict()

megatron_state = {}


def copy(dst, src):
    megatron_state[dst] = hf_state[src]


############################################
# Patch embedding
############################################

print("Converting patch embedding")

copy(
    "patch_embed.weight",
    "embeddings.patch_embeddings.projection.weight",
)

copy(
    "patch_embed.bias",
    "embeddings.patch_embeddings.projection.bias",
)


############################################
# CLS token
############################################

print("Copying CLS token")

copy(
    "cls_token",
    "embeddings.cls_token",
)


############################################
# Absolute position embeddings
############################################

if "embeddings.position_embeddings" in hf_state:
    print("Copying absolute positional embeddings")

    copy(
        "pos_embed",
        "embeddings.position_embeddings",
    )


############################################
# Transformer layers
############################################

num_layers = hf_model.config.num_hidden_layers

print(f"Converting {num_layers} transformer layers")

for i in range(num_layers):

    hf_prefix = f"encoder.layer.{i}"
    meg_prefix = f"encoder.layers.{i}"

    print(f"Layer {i}")


    ########################################
    # LayerNorm 1
    ########################################

    copy(
        f"{meg_prefix}.input_layernorm.weight",
        f"{hf_prefix}.layernorm_before.weight",
    )

    copy(
        f"{meg_prefix}.input_layernorm.bias",
        f"{hf_prefix}.layernorm_before.bias",
    )


    ########################################
    # LayerNorm 2
    ########################################

    copy(
        f"{meg_prefix}.post_attention_layernorm.weight",
        f"{hf_prefix}.layernorm_after.weight",
    )

    copy(
        f"{meg_prefix}.post_attention_layernorm.bias",
        f"{hf_prefix}.layernorm_after.bias",
    )


    ########################################
    # QKV merge (critical step)
    ########################################

    q = hf_state[f"{hf_prefix}.attention.attention.query.weight"]
    k = hf_state[f"{hf_prefix}.attention.attention.key.weight"]
    v = hf_state[f"{hf_prefix}.attention.attention.value.weight"]

    qb = hf_state[f"{hf_prefix}.attention.attention.query.bias"]
    kb = hf_state[f"{hf_prefix}.attention.attention.key.bias"]
    vb = hf_state[f"{hf_prefix}.attention.attention.value.bias"]


    megatron_state[
        f"{meg_prefix}.self_attention.query_key_value.weight"
    ] = torch.cat([q, k, v], dim=0)

    megatron_state[
        f"{meg_prefix}.self_attention.query_key_value.bias"
    ] = torch.cat([qb, kb, vb], dim=0)


    ########################################
    # Attention output projection
    ########################################

    copy(
        f"{meg_prefix}.self_attention.dense.weight",
        f"{hf_prefix}.attention.output.dense.weight",
    )

    copy(
        f"{meg_prefix}.self_attention.dense.bias",
        f"{hf_prefix}.attention.output.dense.bias",
    )


    ########################################
    # MLP first layer
    ########################################

    copy(
        f"{meg_prefix}.mlp.dense_h_to_4h.weight",
        f"{hf_prefix}.intermediate.dense.weight",
    )

    copy(
        f"{meg_prefix}.mlp.dense_h_to_4h.bias",
        f"{hf_prefix}.intermediate.dense.bias",
    )


    ########################################
    # MLP second layer
    ########################################

    copy(
        f"{meg_prefix}.mlp.dense_4h_to_h.weight",
        f"{hf_prefix}.output.dense.weight",
    )

    copy(
        f"{meg_prefix}.mlp.dense_4h_to_h.bias",
        f"{hf_prefix}.output.dense.bias",
    )


############################################
# Final LayerNorm
############################################

if "layernorm.weight" in hf_state:

    print("Copying final LayerNorm")

    copy(
        "encoder.final_layernorm.weight",
        "layernorm.weight",
    )

    copy(
        "encoder.final_layernorm.bias",
        "layernorm.bias",
    )

############################################
# classifier head
############################################

if "classifier.weight" in hf_state:
    copy(
        "head.weight",
         "classifier.weight",
         )

    copy(
        "head.bias",
        "classifier.bias",
    )

############################################
# Save checkpoint
############################################

torch.save(megatron_state, OUTPUT)

print("Saved Megatron checkpoint →", OUTPUT)