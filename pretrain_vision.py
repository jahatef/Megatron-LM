# Copyright (c) 2024, NVIDIA CORPORATION.
"""Pretrain Vision Transformer using Megatron Core."""
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_local_spec,
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.models.common.embeddings import RotaryEmbeddingViT

from megatron.training import (
    get_args,
    get_timers,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args

from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder
    )
from megatron.core.datasets.blended_megatron_dataset_config import (
    BlendedMegatronDatasetConfig,
)
from megatron.core.datasets.vision_dataset import (
    MegatronVisionDataset,
)
from megatron.core.datasets.utils import Split



# -----------------------
# ViT Model
# -----------------------

class MegatronViT(GraphableMegatronModule, MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        args = get_args()
        config.num_prefix_tokens = 1
        self.config = config
        self.pos_embed_type = args.position_embedding_type


        self.hidden_size = config.hidden_size
        self.patch_size = args.patch_dim
        self.img_h = args.img_size
        self.img_w = args.img_size
        self.rotary_base = args.vit_rotary_base
        self.rope_impl = args.vit_rope_impl
        self.num_patches = (self.img_h // self.patch_size) * (self.img_w // self.patch_size)

        # Patch embedding
        self.patch_embed = torch.nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        # CLS token
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        # Position embedding
        if self.pos_embed_type == "learned_absolute":
            self.pos_embed = torch.nn.Parameter(
                torch.zeros(1, self.num_patches + 1, self.hidden_size)
            )
        else:
            self.pos_embed = None

        # Transformer encoder
        self.encoder = TransformerBlock(
            config=config,
            spec=config.layer_spec,
            pre_process=True,
            post_process=True,
        )

        # Classification head (optional; can be removed for MAE-style pretrain)
        self.head = torch.nn.Linear(self.hidden_size, args.num_classes)

        self._init_weights()
        
        # Precompute RoPE cache
        if self.pos_embed_type == "rope":
            self.rotary_emb = RotaryEmbeddingViT(
                dim=self.hidden_size // config.num_attention_heads,
                num_heads=config.num_attention_heads,
                rotary_base = self.rotary_base,
                rope_impl = self.rope_impl,
            )
        else:
            self.rotary_emb = None


            
            
    def _init_weights(self):
        if self.pos_embed is not None:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, images, labels=None):
        B = images.shape[0]

        # Patchify
        x = self.patch_embed(images)               # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)           # [B, N, D]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)      # [B, N+1, D]

        # Add positional embeddings
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Megatron expects [seq, batch, hidden]
        x = x.transpose(0, 1)

        # Transformer
        seq_len, batch_size, H = x.shape

        rotary_pos_emb = None

        if self.rotary_emb is not None:
            rotary_pos_emb = self.rotary_emb(
                H=self.img_h // self.patch_size,
                W=self.img_w // self.patch_size,
                device=x.device,
            )

        # Vision has no masking → allow all attention
        attention_mask = torch.ones(
            (batch_size, 1, seq_len, seq_len),
            device=x.device,
            dtype=torch.bool,
        )
        #print_rank_0(f"DEBUG: x shape before encoder: {x.size()}")
        x = self.encoder(
            x,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
        #print_rank_0(f"DEBUG: x shape after encoder: {x.size()}")
        #if x.size()[1] > 2:
        #    print_rank_0(f"DEBUG: x across batch: {(x[:, 0] - x[:, 1]).abs().mean()}")
        cls_out = x[0]

        logits = self.head(cls_out)

        if labels is None:
            return logits
        labels = labels.long()
        loss = F.cross_entropy(logits, labels)
        return  loss

# -----------------------
# Model provider
# -----------------------

def model_provider(pre_process=True, post_process=True,config=None,pg_collection=None,wrap_with_ddp=False):
    args = get_args()
    print_rank_0("Building ViT model ...")

    if not config:
        config = core_transformer_config_from_args(args)
        config.num_layers = args.num_layers

        if args.transformer_impl == "transformer_engine":
            config.layer_spec = get_vit_layer_with_transformer_engine_spec()
        else:
            config.layer_spec = get_vit_layer_with_local_spec()

    model = MegatronViT(config)
    return model

# -----------------------
# Dataset (dummy ImageNet-style)
# -----------------------

'''def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()
    print(f"DEBUG: train_val_test_num_samples: {train_val_test_num_samples}\n split: {args.split}")
    print_rank_0("Building ViT datasets with BlendedMegatronDatasetBuilder ...")

    dataset_config = BlendedMegatronDatasetConfig(
        random_seed=args.seed,
        sequence_length=0,
        split=args.split,   # IMPORTANT
        tokenizer=None,
        image_size=args.img_size,
        batch_size=args.global_batch_size,
        blend=None,  # single dataset root
    )

    sizes = [
        train_val_test_num_samples[0],
        train_val_test_num_samples[1],
        train_val_test_num_samples[2],
    ]

    def is_built_on_rank():
        return (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

    builder = BlendedMegatronDatasetBuilder(
        cls=MegatronVisionDataset,
        sizes=sizes,
        is_built_on_rank=is_built_on_rank,
        config=dataset_config,
    )
    print(f"DEBUG: sizes: {sizes}, ")

    train_ds, valid_ds, test_ds = builder.build()

    print_rank_0(
        f"Dataset sizes: train={len(train_ds)}, "
        f"valid={len(valid_ds)}, test={len(test_ds)}"
    )

    return train_ds, valid_ds, test_ds'''

def train_valid_test_datasets_provider(_):
    args = get_args()

    print_rank_0("Building SIMPLE ViT datasets (no Megatron builder)")

    root = args.data_path[0]

    dataset_config = BlendedMegatronDatasetConfig(
        random_seed=args.seed,
        sequence_length=0,
        split=args.split,
        image_size=args.img_size,
        tokenizer=None,
        batch_size=args.global_batch_size,
        blend=None,
        allow_ambiguous_pad_tokens=True,
    )

    # ---- build actual dataset (important fix) ----
    low_level_dataset = MegatronVisionDataset.build_low_level_dataset(
        root,
        dataset_config,
    )

    # correct indices (over samples, NOT directories)
    indices = np.arange(len(low_level_dataset), dtype=np.int32)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)   

    n = len(indices)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_idx = indices[:train_end]
    valid_idx = indices[train_end:val_end]
    test_idx  = indices[val_end:]

    pad = lambda x: np.pad(
        x,
        (0, (-len(x)) % args.global_batch_size),
        mode="wrap"
    )

    train_idx = pad(train_idx)
    valid_idx = pad(valid_idx)
    test_idx  = pad(test_idx)

    train_ds = MegatronVisionDataset(
        low_level_dataset, root, train_idx, None, Split.train, dataset_config
    )

    valid_ds = MegatronVisionDataset(
        low_level_dataset, root, valid_idx, None, Split.valid, dataset_config
    )

    test_ds = MegatronVisionDataset(
        low_level_dataset, root, test_idx, None, Split.test, dataset_config
    )
    print_rank_0(
        f"Dataset sizes: train={len(train_ds)}, "
        f"valid={len(valid_ds)}, test={len(test_ds)}"
    )

    return train_ds, valid_ds, test_ds
# -----------------------
# Batch
# -----------------------

def get_batch(data_iterator):
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data = tensor_parallel.broadcast_data(
        ["images", "labels"],
        data,
        datatype=torch.float32,
    )

    images = data["images"]
    labels = data["labels"].long()

    return images, labels

# -----------------------
# Forward step
# -----------------------

def forward_step(data_iterator, model):
    timers = get_timers()
    timers("batch").start()
    images, labels = get_batch(data_iterator)
    timers("batch").stop()

    logits = model(images)
    #logits1 = model(images[:1])
    #logits2 = model(images[1:2])
    #print(f"DEBUG: logits difference: {(logits1 - logits2).abs().mean()}")

    def loss_func(output_tensor):
        loss = F.cross_entropy(output_tensor, labels)

        with torch.no_grad():
            # ---- Top-1 accuracy ----
            _, preds = torch.max(output_tensor, dim=-1)
            #print(f"DEBUG: output_tensor: {output_tensor}")
            #print(f"DEBUG: output_tensor.size(): {output_tensor.size()}")
            #print(f"DEBUG: predictions: {preds}")
            #print(f"DEBUG: labels: {labels}")
            top1_acc = (preds == labels).float().mean()

            # ---- Top-5 accuracy ----
            top5_preds = output_tensor.topk(5, dim=-1).indices
            top5_acc = (
                top5_preds.eq(labels.unsqueeze(-1))
                .any(dim=-1)
                .float()
                .mean()
            )

        loss_dict = {
            "loss": loss,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
        }

        return loss, loss_dict

    return logits, loss_func

# -----------------------
# Extra args
# -----------------------

def add_vit_args(parser):
    group = parser.add_argument_group("ViT")

    return parser

# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_vit_args,
    )
