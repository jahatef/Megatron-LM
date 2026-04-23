import torch
from transformers import AutoModel


HF_MODEL = "naver-ai/rope_axial_deit_large_patch16_LS"
OUTPUT = "rope_axial_deit_large.pt"


print("Loading HuggingFace ViT...")
hf_model = AutoModel.from_pretrained(HF_MODEL)
hf_state = hf_model.state_dict()

megatron_state = {}

copied_keys = set()
missing_source_keys = []
shape_mismatch_keys = []


def copy(dst, src):

    if src not in hf_state:
        print(f"⚠️ Missing HF key: {src}")
        missing_source_keys.append(src)
        return

    megatron_state[dst] = hf_state[src]

    copied_keys.add(src)


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
    # QKV merge
    ########################################

    qk = f"{hf_prefix}.attention.attention.query.weight"
    kk = f"{hf_prefix}.attention.attention.key.weight"
    vk = f"{hf_prefix}.attention.attention.value.weight"

    qb = f"{hf_prefix}.attention.attention.query.bias"
    kb = f"{hf_prefix}.attention.attention.key.bias"
    vb = f"{hf_prefix}.attention.attention.value.bias"

    if all(k in hf_state for k in [qk, kk, vk]):

        q = hf_state[qk]
        k = hf_state[kk]
        v = hf_state[vk]

        megatron_state[
            f"{meg_prefix}.self_attention.query_key_value.weight"
        ] = torch.cat([q, k, v], dim=0)

        copied_keys.update([qk, kk, vk])

    else:
        print(f"⚠️ Missing QKV weights in layer {i}")

    if all(k in hf_state for k in [qb, kb, vb]):

        q = hf_state[qb]
        k = hf_state[kb]
        v = hf_state[vb]

        megatron_state[
            f"{meg_prefix}.self_attention.query_key_value.bias"
        ] = torch.cat([q, k, v], dim=0)

        copied_keys.update([qb, kb, vb])

    else:
        print(f"⚠️ Missing QKV bias in layer {i}")


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
# REPORT UNUSED HF PARAMETERS
############################################

unused_keys = sorted(set(hf_state.keys()) - copied_keys)

print("\n==============================")
print("Conversion Summary")
print("==============================")

if missing_source_keys:
    print(f"\n⚠️ Missing expected HF keys ({len(missing_source_keys)}):")
    for k in missing_source_keys:
        print("  ", k)

if unused_keys:

    print(f"\n⚠️ Unused HF checkpoint parameters ({len(unused_keys)}):")

    for k in unused_keys:
        print("  ", k)

    print("\nThese likely correspond to:")
    print(" • RoPE buffers")
    print(" • relative position tables")
    print(" • distillation tokens")
    print(" • extra classifier heads")
    print(" • layer scale parameters (LS)")
else:
    print("\n✅ All HF weights were consumed")


############################################
# Save checkpoint
############################################

torch.save(megatron_state, OUTPUT)

print("\nSaved Megatron checkpoint →", OUTPUT)