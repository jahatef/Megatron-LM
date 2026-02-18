# Copyright (c) 2024, NVIDIA CORPORATION.
"""Pretrain Vision Transformer using Megatron Core."""
import os
import math
import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_local_spec,
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.models.common.embeddings import RotaryEmbeddingDinoV3

from megatron.training import (
    get_args,
    get_timers,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args

from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.utils import Split
from megatron.core.datasets.blended_megatron_dataset_config import (
    BlendedMegatronDatasetConfig,
)



# -----------------------
# ViT Model
# -----------------------

class MegatronViT(GraphableMegatronModule, MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        args = get_args()
        self.config = config
        self.pos_embed_type = args.position_embedding_type


        self.hidden_size = config.hidden_size
        self.patch_size = args.patch_dim
        self.img_h = args.img_size
        self.img_w = args.img_size
        print(f"\n\nheight:{self.img_h}, width:{self.img_w}\n\n")
        self.num_patches = (self.img_h // self.patch_size) * (self.img_w // self.patch_size)
        print(f"\n\npatch size: {self.patch_size}\n\n")

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
        if self.pos_embed_type == "absolute":
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
        #print(f"\n\n\n number of classes: {args.num_classes}\n\n\n")
        self.head = torch.nn.Linear(self.hidden_size, args.num_classes)

        self._init_weights()
        
        # Precompute RoPE cache
        if self.pos_embed_type == "rope":
            self.rotary_emb = RotaryEmbeddingDinoV3(
                dim=self.hidden_size // config.num_attention_heads
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
        print(f"\n\nx size: {x.size()}\n\n")
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
            head_dim = self.hidden_size // self.config.num_attention_heads

            rotary_pos_emb = self.rotary_emb(
                seq_len=seq_len,
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
        if rotary_pos_emb is not None:
            pass #rotary_pos_emb = rotary_pos_emb[1:]  # remove CLS position
        print(f"\n\npos_emb: {rotary_pos_emb.size()}, x: {x.size()}\n\n")
        x = self.encoder(
            x,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )

        #print(f"\n\n\n x before head: {x.shape} \n\n\n")
        cls_out = x[0]

        logits = self.head(cls_out)
        #print(f"logits shape: {logits.shape}")

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
'''
def train_valid_test_datasets_provider(train_val_test_num_samples):
    from megatron.core.datasets.vision_dataset import (
        MegatronVisionDataset,
    )

    args = get_args()
    print_rank_0("Building ViT datasets via Megatron dataset builder ...")

    # Build dataset config
    dataset_config = BlendedMegatronDatasetConfig(
        random_seed=args.seed,
        sequence_length=0,  # unused for vision, but required
        split=args.split,
        tokenizer=None,     # vision does not use tokenizer
        allow_ambiguous_pad_tokens=True,
    )

    builder = BlendedMegatronDatasetBuilder(
        cls=MegatronVisionDataset,

        config=dataset_config,
    )

    train_ds, valid_ds, test_ds = builder.build_datasets(
        splits=(Split.train, Split.valid, Split.test),
        num_samples=train_val_test_num_samples,
    )

    return train_ds, valid_ds, test_ds
    '''

def train_valid_test_datasets_provider(train_val_test_num_samples=[20, 1, 1]):
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.blended_megatron_dataset_config import (
        BlendedMegatronDatasetConfig,
    )
    from megatron.core.datasets.utils import Split
    from megatron.core.parallel_state import is_pipeline_first_stage
    from megatron.core.datasets.vision_dataset import (
        MegatronVisionDataset,
    )

    args = get_args()
    print_rank_0("Building ViT datasets with BlendedMegatronDatasetBuilder ...")

    # --------------------------------------------------
    # Dataset config
    # --------------------------------------------------
    dataset_root = args.data_path[0]

    dataset_config = BlendedMegatronDatasetConfig(
        random_seed=args.seed,
        sequence_length=0,
        #split=args.split,
        tokenizer=None,
        image_size=args.img_size,
        blend_per_split=[
            ([os.path.join(dataset_root, "training")], None),
            ([os.path.join(dataset_root, "testing")],   None), None
            #([os.path.join(dataset_root, "testing")],  None),
        ],
    )


    # --------------------------------------------------
    # Sample counts per split
    # Order must match Split enum
    # --------------------------------------------------
    sizes = [
        train_val_test_num_samples[0],  # train
        train_val_test_num_samples[1],  # valid
        train_val_test_num_samples[2],  # test
    ]

    # --------------------------------------------------
    # Only one rank actually builds datasets
    # (Megatron standard pattern)
    # --------------------------------------------------
    def is_built_on_rank():
        return (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

    # --------------------------------------------------
    # Builder
    # --------------------------------------------------
    builder = BlendedMegatronDatasetBuilder(
        cls=MegatronVisionDataset,
        sizes=sizes,
        is_built_on_rank=is_built_on_rank,
        config=dataset_config,
    )

    train_ds, valid_ds, test_ds = builder.build()

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
    def loss_func(output_tensor):
        loss = F.cross_entropy(output_tensor, labels)

        loss_dict = {"loss": loss}
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
