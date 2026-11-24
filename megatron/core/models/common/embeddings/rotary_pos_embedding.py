# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.inference.contexts import BaseInferenceContext
    from megatron.core.packed_seq_params import PackedSeqParams

import logging
import math
from functools import lru_cache

import torch
from torch import Tensor, nn

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import (  # for backward compatibility; pylint: disable=unused-import
    _apply_rotary_pos_emb_bshd,
    _apply_rotary_pos_emb_thd,
    _rotate_half,
    apply_rotary_pos_emb,
    get_pos_emb_on_this_cp_rank,
)
from megatron.core.utils import deprecate_inference_params, internal_api

logger = logging.getLogger(__name__)


__all__ = ['RotaryEmbedding', 'MultimodalRotaryEmbedding','RotaryEmbeddingAxial',"RotaryEmbeddingViT", "RotaryEmbeddingMixedAxis","RotaryEmbeddingHilbert"]


class RotaryEmbedding(nn.Module):
    """Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position
            embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE
            for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
        rope_scaling (bool, optional): Apply rope scaling as used in llama 3.x.
        rope_scaling_factor (float, optional): rope scaling factor in llama 3.x. Defaults to 8.
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly
            on the GPU. Defaults to False
        cp_group (torch.distributed.ProcessGroup, optional): Process group for context parallel.
            Defaults to None.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        use_cpu_initialization: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        device = 'cpu' if use_cpu_initialization else torch.cuda.current_device()
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        if rope_scaling:
            self.inv_freq = self._apply_scaling(self.inv_freq, factor=rope_scaling_factor)

        self.cp_group = (
            cp_group
            if cp_group is not None
            else parallel_state.get_context_parallel_group(check_initialized=False)
        )

    def _apply_scaling(
        self,
        freqs,
        factor=8,
        low_freq_factor=1,
        high_freq_factor=4,
        original_max_position_embeddings=8192,
    ):
        # This implementation is adapted from:
        # https://github.com/huggingface/transformers/blob/2a5a6ad18aa22e98429bb5ecb880660328030ea0/src/transformers/modeling_rope_utils.py#L303-L343

        factor = factor  # `8` in the original implementation
        low_freq_factor = low_freq_factor  # `1` in the original implementation
        high_freq_factor = high_freq_factor  # `4` in the original implementation
        old_context_len = original_max_position_embeddings  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / freqs
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama

    def get_freqs_non_repeated(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Generates matrix of frequencies based on positions in the sequence,
        used to create positional encodings"""
        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.outer(seq, self.inv_freq)  # [seq len, dim]

        return freqs

    def get_cos_sin(self, max_seq_len: int, offset: int = 0) -> (Tensor, Tensor):
        """Cosine and sine values for RoPE are precomputed for all positions up to the maximum
        sequence length"""
        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    def get_emb(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Forward pass of RoPE embedding before CP sharding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        if self.inv_freq.device.type == 'cpu':
            # move `inv_freq` to GPU once at the first micro-batch forward pass
            self.inv_freq = self.inv_freq.to(device=torch.cuda.current_device())
        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        return emb

    @lru_cache(maxsize=32)
    @internal_api
    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Tensor:
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.
            packed_seq (bool, optional): Whether to use packed sequence. Defaults to False.
            cp_group (torch.distributed.ProcessGroup, optional): Context parallel group.
                Defaults to None.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        emb = self.get_emb(max_seq_len, offset)
        if cp_group is None:
            cp_group = self.cp_group
        if cp_group is not None and cp_group.size() > 1 and not packed_seq:
            # slice rotary_pos_emb along sequence dimension
            # and select the parition of the current CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0, cp_group)
        return emb

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f'{prefix}inv_freq', None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_rotary_seq_len(
        self,
        inference_context: BaseInferenceContext,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
        packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> int:
        """Function to get the rotary sequence length.

        Args:
            inference_context : Used during Inference time
            transformer (TransformerBlock): The transformer block (decoder/encoder) used
                by the model
            transformer_input (Tensor): Input tensor to the transformer
            transformer_config (TransformerConfig): Transformer config used by the model
            packed_seq_params (PackedSeqParams): Packed sequence params

        Returns:
            int: The rotary sequence length
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if packed_seq_params is not None:
            # max_seqlen are the max sequence length in the packed sequence before being divived
            # by the tp and cp size.
            return max(packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv)
        elif inference_context is not None:
            rotary_seq_len = inference_context.max_sequence_length
        else:
            if transformer is not None and transformer.input_tensor is not None:
                rotary_seq_len = transformer.input_tensor.size(0)
            else:
                rotary_seq_len = transformer_input.size(0)

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

        rotary_seq_len *= transformer_config.context_parallel_size

        return rotary_seq_len


class MultimodalRotaryEmbedding(nn.Module):
    """Multimodal Rotary Embedding for language model.
    Based on https://github.com/alibaba/Pai-Megatron-Patch/blob/
    efa5a752e845267936db9ae7df1b6aba92e9ff9a/megatron_patch/model/qwen2_vl/rotary_pos_embedding.py
    Copyright (c) 2025 alibaba/Pai-Megatron-Patch. Apache 2.0 license.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position
            embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE
            for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: int = 10000,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device())
                / dim
            )
        )
        self.cp_group = (
            cp_group
            if cp_group is not None
            else parallel_state.get_context_parallel_group(check_initialized=False)
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        mrope_section: List[int],
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal,
                height and width in rope calculation.
            cp_group (torch.distributed.ProcessGroup, optional): Context parallel group.
                Defaults to None.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)  # shape (3, bs, seq_length, 2 * dim)
        else:
            bs = freqs.shape[1]
            emb = torch.stack((freqs.view(3, bs, -1, 1), freqs.view(3, bs, -1, 1)), dim=-1).view(
                3, bs, freqs.shape[0], -1
            )

        # generate freqs with mrope_section
        # shape (bs, seq_length, 2 * dim)
        mrope_section = mrope_section * 2
        emb = torch.cat([m[i % 3] for i, m in enumerate(emb.split(mrope_section, dim=-1))], dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        if cp_group is None:
            cp_group = self.cp_group
        if cp_group is not None and cp_group.size() > 1:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current
            # CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0, cp_group)
        return emb

def _init_rotary_base(module, rotary_base, num_heads, axes=2):
    """
    Initializes rotary_base consistently across RoPE variants.

    axes=1  → axial / polar
    axes=2  → mixed-axis variants

    Supports:

        rotary_base=None
        len=1
        len=num_heads
        len=axes
        len=num_heads*axes
    """

    if rotary_base is None:

        module.rotary_base = nn.Parameter(
            torch.ones(num_heads, axes)
        )

        module.freq_mode = "learned"

        return

    rotary_base = torch.tensor(rotary_base, dtype=torch.float32)

    ########################################################
    # shared scalar
    ########################################################

    if rotary_base.numel() == 1:

        module.register_buffer(
            "rotary_base",
            rotary_base.repeat(num_heads, axes),
            persistent=False,
        )

        module.freq_mode = "shared"

        return

    ########################################################
    # shared per-axis
    ########################################################

    if rotary_base.numel() == axes:

        module.register_buffer(
            "rotary_base",
            rotary_base.repeat(num_heads, 1),
            persistent=False,
        )

        module.freq_mode = "shared_axis"

        return

    ########################################################
    # per-head scalar
    ########################################################

    if rotary_base.numel() == num_heads:
        module.register_buffer(
            "rotary_base",
            rotary_base[:, None].repeat(1, axes),
            persistent=False,
        )

        module.freq_mode = "handpicked"

        return

    ########################################################
    # per-head per-axis
    ########################################################

    if rotary_base.numel() == num_heads * axes:
        module.register_buffer(
            "rotary_base",
            rotary_base.view(num_heads, axes),
            persistent=False,
        )
        
        module.freq_mode = "handpicked_axis"

        return

    raise ValueError(
        "Invalid rotary_base length for given axes/head config"
    )

def _rotary_dim_from_fraction(dim, rope_fraction, multiple):
    rotary_dim = int(dim * rope_fraction)
    rotary_dim = (rotary_dim // multiple) * multiple
    rotary_dim = max(multiple, min(rotary_dim, dim))
    if rotary_dim != dim * rope_fraction:
        logger.warning(...)            # e.g. 0.3 * 64 = 19.2 -> 16
    return rotary_dim

def _get_periods(rotary_dim, rotary_base, multiple):
        # n frequencies per axis, from 1 (fastest) down to exactly 1/base (slowest)
        n = rotary_dim // multiple
        exponents = torch.linspace(0, 1, n, dtype=torch.float32)
        return 1 / (rotary_base ** exponents)

class RotaryEmbeddingAxial(nn.Module):
    """Axial RoPE numerics matching DINOv3 for Vision Transformers.

    Produces RoPE angle embeddings for 2D patch grids.
    Does not rotate tensors directly (Megatron-compatible).

    Args:
        dim: Head dimension.
        num_heads: Number of attention heads.
        rotary_base: RoPE base(s). If None, learn one per head.
        normalize_coords: Reserved for future use.
        rotate_half: Match rotate_half or interleaved RoPE layout.
        rope_fraction: Fraction of channels that receive RoPE.
                       Must be in (0, 1].
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rotary_base=None,
        normalize_coords: str = "separate",
        rotate_half: bool = True,
        rope_fraction: float = 1.0,
    ) -> None:
        super().__init__()

        assert 0.0 < rope_fraction <= 1.0

        self.dim = dim
        self.rotate_half = rotate_half
        self.axes = 1
        self.normalize_coords = normalize_coords

        #axial rope divides the head dimension by 4
        self.multiple = 4

        # Number of channels that receive RoPE.
        # Rounded down to nearest multiple of 4 so that
        # the frequency construction remains valid.
        self.rotary_dim = _rotary_dim_from_fraction(dim, rope_fraction, 4)

        self.rotary_base = self._init_rotary_base(
            num_heads=num_heads,
            rotary_base=rotary_base,
            axes=self.axes,
        )

    def _init_rotary_base(self, num_heads, rotary_base, axes):
        if rotary_base is None:
            return nn.Parameter(torch.rand(num_heads, axes) * 1000)

        rotary_base = torch.tensor(rotary_base, dtype=torch.float32)

        assert (
            rotary_base.numel() == 1
            or len(rotary_base) == num_heads
        ), (
            f"num_heads and rotary_base must have compatible sizes. "
            f"Got rotary_base size {len(rotary_base)}."
        )

        if rotary_base.numel() == 1:
            return rotary_base.repeat(num_heads, axes)

        return rotary_base[:, None].repeat(1, axes)

    def _build_coords(self, H: int, W: int, device) -> Tensor:
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        coords = torch.stack([x, y], dim=-1)
        return coords.view(-1, 2)

    def get_embed(self, H: int, W: int, device) -> Tensor:
        coords = self._build_coords(H, W, device)
        periods = _get_periods(self.rotary_dim, self.rotary_base, self.multiple).to(device)

        coords = coords[:, :, None, None]

        angles = coords * periods[None, None, :, :]
        angles = angles.permute(0, 2, 1, 3)
        angles = angles.flatten(2, 3)

        if self.rotate_half:
            angles = angles.tile(2)
        else:
            angles = angles.repeat_interleave(2, dim=-1)

        return angles[:, None, :, :]

    def forward(
        self,
        H: int,
        W: int,
        device=None,
    ) -> Tensor:
        if device is None:
            device = self.rotary_base.device

        return self.get_embed(H, W, device)

class RotaryEmbeddingMixedAxis(nn.Module):
    """
    Megatron-LM compatible mixed-axis 2D RoPE.

    Produces angle tensors (not complex rotations).

    Output shape:
        [seq_len, 1, num_heads, dim]

    Args:
        dim: Head dimension.
        num_heads: Number of attention heads.
        rotary_base: RoPE base(s). If None, learn one per head.
        rotate_half: Match rotate_half or interleaved layout.
        rope_fraction: Fraction of channels that receive RoPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rotary_base: float = 10.0,
        rotate_half: bool = True,
        rope_fraction: float = 1.0,
    ):
        super().__init__()

        assert 0.0 < rope_fraction <= 1.0

        self.dim = dim
        self.num_heads = num_heads
        self.rotate_half = rotate_half
        self.axes = 1

        #mixed axis rope divides the head dimension by 4
        self.multiple = 4

        # Number of output channels that receive RoPE.
        # Must be divisible by 4 because mixed-axis RoPE
        # builds frequencies in dim/4 chunks.
        self.rotary_dim = _rotary_dim_from_fraction(dim, rope_fraction, self.multiple)

        learned_thetas = rotary_base is None

        self.rotary_base = self._init_rotary_base(
            num_heads,
            rotary_base,
            self.axes,
        )
        
        mag = _get_periods(self.rotary_dim, self.rotary_base, self.multiple)

        freqs = []

        for i in range(num_heads):
            alpha = torch.rand(1) * 2 * math.pi

            fx = torch.cat(
                [
                    mag[i, :] * torch.cos(alpha),
                    mag[i, :] * torch.cos(math.pi / 2 + alpha),
                ]
            )

            fy = torch.cat(
                [
                    mag[i, :] * torch.sin(alpha),
                    mag[i, :] * torch.sin(math.pi / 2 + alpha),
                ]
            )

            freqs.append(torch.stack([fx, fy], dim=0))

        freqs = torch.stack(freqs, dim=0)

        if learned_thetas:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def _init_rotary_base(self, num_heads, rotary_base, axes):
        if rotary_base is None:
            return nn.Parameter(
                torch.rand(num_heads, axes) * 200
            )

        rotary_base = torch.tensor(rotary_base, dtype=torch.float32)

        assert (
            rotary_base.numel() == 1
            or len(rotary_base) == num_heads
        ), (
            f"num_heads and rotary_base must have compatible sizes. "
            f"Got rotary_base size {len(rotary_base)}."
        )

        if rotary_base.numel() == 1:
            return rotary_base.repeat(num_heads, axes)

        return rotary_base[:, None].repeat(1, axes)

    def build_coords(self, H, W, device):
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        return x.flatten().float(), y.flatten().float()

    def get_embed(self, H, W, device):
        x, y = self.build_coords(H, W, device)

        fx = self.freqs[:, 0, :]
        fy = self.freqs[:, 1, :]

        angles_x = torch.einsum("n,hd->nhd", x, fx)
        angles_y = torch.einsum("n,hd->nhd", y, fy)

        angles = angles_x + angles_y

        if self.rotate_half:
            angles = torch.cat([angles, angles], dim=-1)
        else:
            angles = angles.repeat_interleave(2, dim=-1)

        return angles[:, None, :, :]

    def forward(self, H, W, device=None):
        if device is None:
            device = self.freqs.device

        return self.get_embed(H, W, device)

import torch
import torch.nn as nn
from torch import Tensor


class RotaryEmbeddingPolar(nn.Module):
    """
    Polar-coordinate RoPE for Vision Transformers.

    Uses:
        (radius r, angle theta)

    instead of:
        (x, y)

    Produces Megatron-compatible angle tensors:

        [seq_len, 1, num_heads, dim]

    Args:
        dim: Head dimension.
        num_heads: Number of attention heads.
        rotary_base: RoPE base(s). If None, learn one per head.
        normalize_coords: Reserved for future use.
        rotate_half: Match rotate_half or interleaved layout.
        rope_fraction: Fraction of channels that receive RoPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rotary_base: float = 100.0,
        normalize_coords: str = "separate",
        rotate_half: bool = True,
        rope_fraction: float = 1.0,
    ) -> None:
        super().__init__()

        assert 0.0 < rope_fraction <= 1.0

        self.dim = dim
        self.rotate_half = rotate_half
        self.axes = 1
        self.normalize_coords = normalize_coords

        # Number of channels that receive RoPE.
        # Must be divisible by 4 because the frequency
        # construction generates dim/4 unique frequencies.
        self.multiple = 4 

        self.rotary_dim = _rotary_dim_from_fraction(dim, rope_fraction, self.multiple)

        self.rotary_base = self._init_rotary_base(
            num_heads,
            rotary_base,
            self.axes,
        )

    def _init_rotary_base(self, num_heads, rotary_base, axes):
        if rotary_base is None:
            return nn.Parameter(
                torch.rand(num_heads, axes) * 200
            )

        rotary_base = torch.tensor(rotary_base, dtype=torch.float32)

        assert (
            rotary_base.numel() == 1
            or len(rotary_base) == num_heads
        ), (
            f"num_heads and rotary_base must have compatible sizes. "
            f"Got rotary_base size {len(rotary_base)}."
        )

        if rotary_base.numel() == 1:
            return rotary_base.repeat(num_heads, axes)

        return rotary_base[:, None].repeat(1, axes)

    def _build_coords(self, H: int, W: int, device) -> Tensor:
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        x = x.float()
        y = y.float()

        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)

        coords = torch.stack([r, theta], dim=-1)

        return coords.view(-1, 2)

    def get_embed(self, H: int, W: int, device) -> Tensor:
        coords = self._build_coords(H, W, device)
        periods = _get_periods(self.rotary_dim, self.rotary_base, self.multiple).to(device)

        coords = coords[:, :, None, None]

        angles = coords * periods[None, None, :, :]
        angles = angles.permute(0, 2, 1, 3)
        angles = angles.flatten(2, 3)

        if self.rotate_half:
            angles = angles.tile(2)
        else:
            angles = angles.repeat_interleave(2, dim=-1)

        # Keep only rotary_dim channels
        angles = angles[..., : self.rotary_dim]

        return angles[:, None, :, :]

    def forward(
        self,
        H: int,
        W: int,
        device=None,
    ) -> Tensor:
        if device is None:
            device = self.rotary_base.device

        return self.get_embed(H, W, device)
    
class RotaryEmbeddingPolarMixedAxis(nn.Module):
    """
    Mixed-axis Polar RoPE for Vision Transformers.

    Uses per-head mixed directions in polar space:

        angle = r * fr(head) + theta * ftheta(head)

    Produces Megatron-compatible angle tensors:

        [seq_len, 1, num_heads, dim]

    Args:
        dim: Head dimension.
        num_heads: Number of attention heads.
        rotary_base: RoPE base(s). If None, learn one per head.
        normalize_coords: Reserved for future use.
        rotate_half: Match rotate_half or interleaved layout.
        rope_fraction: Fraction of channels that receive RoPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rotary_base: float = 100.0,
        normalize_coords: str = "separate",
        rotate_half: bool = True,
        rope_fraction: float = 1.0,
    ) -> None:
        super().__init__()

        assert 0.0 < rope_fraction <= 1.0

        self.dim = dim
        self.num_heads = num_heads
        self.axes = 1
        self.normalize_coords = normalize_coords
        self.rotate_half = rotate_half

        # Number of output channels that receive RoPE.
        # Rounded down to nearest multiple of 4 since the
        # mixed-axis construction uses dim/4 frequency bands.
        self.multiple = 4 

        self.rotary_dim = _rotary_dim_from_fraction(dim, rope_fraction, self.multiple)

        learned_thetas = rotary_base is None

        self.rotary_base = self._init_rotary_base(
            num_heads,
            rotary_base,
            self.axes,
        )

        base = (
            torch.arange(0, self.rotary_dim, 4).float()
            / self.rotary_dim
        )

        mag = _get_periods(self.rotary_dim, self.rotary_base, self.multiple)

        freqs = []

        for i in range(num_heads):
            alpha = torch.rand(1) * 2 * math.pi

            fr = torch.cat(
                [
                    mag[i] * torch.cos(alpha),
                    mag[i] * torch.cos(math.pi / 2 + alpha),
                ]
            )

            ftheta = torch.cat(
                [
                    mag[i] * torch.sin(alpha),
                    mag[i] * torch.sin(math.pi / 2 + alpha),
                ]
            )

            freqs.append(torch.stack([fr, ftheta], dim=0))

        freqs = torch.stack(freqs, dim=0)

        if learned_thetas:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def _init_rotary_base(self, num_heads, rotary_base, axes):
        if rotary_base is None:
            return nn.Parameter(
                torch.rand(num_heads, axes) * 200
            )

        rotary_base = torch.tensor(rotary_base, dtype=torch.float32)

        assert (
            rotary_base.numel() == 1
            or len(rotary_base) == num_heads
        ), (
            f"num_heads and rotary_base must have compatible sizes. "
            f"Got rotary_base size {len(rotary_base)}."
        )

        if rotary_base.numel() == 1:
            return rotary_base.repeat(num_heads, axes)

        return rotary_base[:, None].repeat(1, axes)

    def _build_coords(self, H: int, W: int, device):
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        x = x.float()
        y = y.float()

        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)

        return r.flatten(), theta.flatten()

    def get_embed(self, H: int, W: int, device) -> Tensor:
        r, theta = self._build_coords(H, W, device)

        fr = self.freqs[:, 0, :]
        ftheta = self.freqs[:, 1, :]

        angles_r = torch.einsum("n,hd->nhd", r, fr)
        angles_theta = torch.einsum("n,hd->nhd", theta, ftheta)

        angles = angles_r + angles_theta

        if self.rotate_half:
            angles = torch.cat([angles, angles], dim=-1)
        else:
            angles = angles.repeat_interleave(2, dim=-1)

        return angles[:, None, :, :]

    def forward(
        self,
        H: int,
        W: int,
        device=None,
    ) -> Tensor:
        if device is None:
            device = self.freqs.device

        return self.get_embed(H, W, device)

class RotaryEmbeddingHilbert(nn.Module):
    """
    Megatron-LM compatible Hilbert-curve RoPE.

    Algorithm:

    1) Build 1D RoPE sequence
    2) Generate Hilbert traversal
    3) Scatter sequence into 2D grid
    4) Flatten grid back to ViT order

    Produces:
        [seq_len, 1, num_heads, dim]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rotary_base: float = None,
        rotate_half: bool = True,
        rope_fraction: float = 1.0,
    ):
        super().__init__()

        assert 0.0 < rope_fraction <= 1.0

        self.dim = dim
        self.num_heads = num_heads
        self.rotate_half = rotate_half
        self.axes = 1

        # Number of channels receiving RoPE.
        # Hilbert uses dim/2 frequencies, so round to a multiple of 2.
        self.multiple = 2

        self.rotary_dim = _rotary_dim_from_fraction(dim, rope_fraction, self.multiple)

        self.rotary_base = self._init_rotary_base(
            num_heads,
            rotary_base,
            self.axes,
        )

        # Match original implementation:
        # supplied bases are assumed sqrt(base)
        self.rotary_base = self.rotary_base ** 2

    def _init_rotary_base(self, num_heads, rotary_base, axes):
        if rotary_base is None:
            return nn.Parameter(
                torch.rand(num_heads, axes) * 200
            )

        rotary_base = torch.tensor(rotary_base, dtype=torch.float32)

        assert (
            rotary_base.numel() == 1
            or len(rotary_base) == num_heads
        ), (
            f"num_heads and rotary_base must have compatible sizes. "
            f"Got rotary_base size {len(rotary_base)}."
        )

        if rotary_base.numel() == 1:
            return rotary_base.repeat(num_heads, axes)

        return rotary_base[:, None].repeat(1, axes)

    def _rot(self, s, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y

            x, y = y, x

        return x, y

    def _hilbert_d2xy(self, n, d):
        x = 0
        y = 0

        t = d
        s = 1

        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)

            x, y = self._rot(s, x, y, rx, ry)

            x += s * rx
            y += s * ry

            t //= 4
            s *= 2

        return x, y

    def build_hilbert_grid(self, H, W, device):
        assert H == W, "Hilbert RoPE requires square grids"

        N = H * W

        n = 1 << (H - 1).bit_length()

        coords = []

        for d in range(n * n):
            x, y = self._hilbert_d2xy(n, d)

            if x < W and y < H:
                coords.append((x, y))

                if len(coords) == N:
                    break

        return coords

    def get_embed(self, H, W, device):
        coords = self.build_hilbert_grid(H, W, device)
        freq = _get_periods(self.rotary_dim, self.rotary_base, self.multiple).to(device)

        N = H * W

        seq_positions = torch.arange(
            N,
            device=device,
            dtype=torch.float32,
        )

        angles_1d = (
            seq_positions[:, None, None, None]
            * freq[None, None, :, :]
        )

        angle_grid = torch.zeros(
            H,
            W,
            1,
            self.num_heads,
            self.rotary_dim // 2,
            device=device,
        )

        for idx, (x, y) in enumerate(coords):
            angle_grid[y, x] = angles_1d[idx]

        angles = angle_grid.flatten(0, 1)

        if self.rotate_half:
            angles = torch.cat([angles, angles], dim=-1)
        else:
            angles = angles.repeat_interleave(2, dim=-1)

        return angles

    def forward(self, H, W, device=None):
        if device is None:
            device = self.rotary_base.device

        return self.get_embed(H, W, device)

class RotaryEmbeddingViT(nn.Module):
    """
    Unified interface for all RoPE variants.

    Supported rope_impl:

        "axial"
        "mixed_axis"
        "hilbert"
        "polar"
        "mixed_polar"
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = None,
        rotary_base: float = None,
        rotate_half: bool = True,
        rope_impl: str = "axial",
        rope_fraction: float = 1.0,
    ):
        super().__init__()

        self.rope_impl = rope_impl

        if rope_impl == "axial":
            self.impl = RotaryEmbeddingAxial(
                dim=dim,
                num_heads=num_heads,
                rotary_base=rotary_base,
                rotate_half=rotate_half,
                rope_fraction=rope_fraction,
            )

        elif rope_impl == "mixed_axis":

            assert num_heads is not None, (
                "mixed_axis RoPE requires num_heads"
            )

            self.impl = RotaryEmbeddingMixedAxis(
                dim=dim,
                num_heads=num_heads,
                rotary_base=rotary_base,
                rotate_half=rotate_half,
                rope_fraction=rope_fraction,
            )

        elif rope_impl == "hilbert":

            self.impl = RotaryEmbeddingHilbert(
                dim=dim,
                num_heads=num_heads,
                rotary_base=rotary_base,
                rotate_half=rotate_half,
                rope_fraction=rope_fraction,
            )

        elif rope_impl == "polar":

            self.impl = RotaryEmbeddingPolar(
                dim=dim,
                num_heads=num_heads,
                rotary_base=rotary_base,
                rotate_half=rotate_half,
                rope_fraction=rope_fraction,
            )

        elif rope_impl == "mixed_polar":

            self.impl = RotaryEmbeddingPolarMixedAxis(
                dim=dim,
                num_heads=num_heads,
                rotary_base=rotary_base,
                rotate_half=rotate_half,
                rope_fraction=rope_fraction,
            )

        else:
            raise ValueError(
                f"Unknown rope_impl '{rope_impl}'. "
                "Supported: axial, mixed_axis, hilbert, "
                "polar, mixed_polar"
            )

    def forward(self, H, W, device=None):
        return self.impl(H, W, device=device)