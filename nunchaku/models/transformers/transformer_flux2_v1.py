"""
Nunchaku FLUX.2 Transformer with C++ backend (v1 pattern).

Delegates transformer block execution to :class:`QuantizedFlux2Model` (C++) for
performance, while keeping embeddings, modulations, rotary embeddings, and output
normalization in Python.
"""

import json
import logging
import os
from pathlib import Path

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from huggingface_hub import utils
from torch import nn

from ..._C import QuantizedFlux2Model
from ..._C import utils as cutils
from ...utils import get_precision, get_precision_from_quantization_config, pad_tensor
from ..embeddings import pack_rotemb
from .utils import NunchakuModelLoaderMixin

logger = logging.getLogger(__name__)


def _pack_flux2_rotary_emb(freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Pack FLUX.2 (cos, sin) rotary embeddings into the format expected by C++ kernels."""
    cos, sin = freqs_cis
    rotemb = torch.stack([sin[:, 0::2], cos[:, 0::2]], dim=-1).unsqueeze(0).unsqueeze(-2).contiguous()
    return pack_rotemb(pad_tensor(rotemb, 256, 1))


def load_quantized_flux2_module(
    state_dict: dict[str, torch.Tensor],
    *,
    num_layers: int,
    num_single_layers: int,
    dim: int,
    num_attention_heads: int,
    attention_head_dim: int,
    mlp_ratio: int,
    device: str | torch.device = "cuda",
    use_fp4: bool = False,
    offload: bool = False,
    bf16: bool = True,
) -> QuantizedFlux2Model:
    """Create and load a :class:`QuantizedFlux2Model` from a state dict."""
    device = torch.device(device)
    assert device.type == "cuda"
    m = QuantizedFlux2Model()
    cutils.disable_memory_auto_release()
    m.init(
        num_layers,
        num_single_layers,
        dim,
        num_attention_heads,
        attention_head_dim,
        mlp_ratio,
        use_fp4,
        offload,
        bf16,
        0 if device.index is None else device.index,
    )
    m.loadDict(state_dict, True)
    return m


def _contiguous_to_interleaved(mod: torch.Tensor, n_chunks: int) -> torch.Tensor:
    """Convert modulation from contiguous chunk layout to interleaved layout.

    Python Flux2Modulation produces contiguous chunks: [shift0..., scale0..., gate0..., shift1..., ...]
    C++ split_mod expects interleaved: [shift0_0, scale0_0, gate0_0, shift0_1, scale0_1, gate0_1, ...]
    """
    B = mod.shape[0]
    dim = mod.shape[-1] // n_chunks
    # [B, n_chunks, dim] -> transpose last two -> [B, dim, n_chunks] -> flatten
    return mod.view(B, n_chunks, dim).transpose(1, 2).contiguous().view(B, -1)


class NunchakuFlux2TransformerBlocks(nn.Module):
    """Wraps :class:`QuantizedFlux2Model` and manages tensor conversion for the C++ backend."""

    def __init__(self, m: QuantizedFlux2Model, device: str | torch.device):
        super().__init__()
        self.m = m
        self.dtype = torch.bfloat16 if m.isBF16() else torch.float16
        self.device = device

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        mod_img: torch.Tensor,
        mod_txt: torch.Tensor,
        mod_single: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        txt_tokens = encoder_hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(self.device)

        # Convert modulation from contiguous (torch.chunk) to interleaved (C++ split_mod) layout
        mod_img = _contiguous_to_interleaved(mod_img, 6).to(self.dtype).to(self.device)
        mod_txt = _contiguous_to_interleaved(mod_txt, 6).to(self.dtype).to(self.device)
        mod_single = _contiguous_to_interleaved(mod_single, 3).to(self.dtype).to(self.device)
        rotary_emb_img = rotary_emb_img.to(self.device)
        rotary_emb_txt = rotary_emb_txt.to(self.device)
        rotary_emb_single = rotary_emb_single.to(self.device)

        # C++ forward: returns concatenated [txt, img] hidden states
        hidden_states = self.m.forward(
            hidden_states,
            encoder_hidden_states,
            mod_img,
            mod_txt,
            mod_single,
            rotary_emb_img,
            rotary_emb_txt,
            rotary_emb_single,
        )

        hidden_states = hidden_states.to(original_dtype).to(original_device)

        # Split back into encoder (txt) and image hidden states
        encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
        hidden_states = hidden_states[:, txt_tokens:, ...]

        return encoder_hidden_states, hidden_states


class NunchakuFlux2Transformer2DModelV1(Flux2Transformer2DModel, NunchakuModelLoaderMixin):
    """
    Nunchaku FLUX.2 Transformer with C++ backend.

    Embeddings, modulations, and output projection run in Python.
    Transformer blocks (double-stream + single-stream) run in C++.
    """

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        device = kwargs.get("device", "cuda")
        if isinstance(device, str):
            device = torch.device(device)
        offload = kwargs.get("offload", False)
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        if not (
            pretrained_model_name_or_path.is_file()
            or pretrained_model_name_or_path.name.endswith((".safetensors", ".sft"))
        ):
            raise AssertionError("Only safetensors are supported")

        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        config = json.loads(metadata.get("config", "{}"))

        # Determine precision from checkpoint metadata
        if quantization_config:
            precision = get_precision_from_quantization_config(quantization_config)
        else:
            precision = get_precision(device=device)

        # Split state dict: quantized (blocks) vs unquantized (embeddings, norms)
        quantized_part_sd = {}
        unquantized_part_sd = {}
        for k, v in model_state_dict.items():
            if k.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                quantized_part_sd[k] = v
            else:
                unquantized_part_sd[k] = v

        # Extract model config
        num_layers = config.get("num_layers", 5)
        num_single_layers = config.get("num_single_layers", 20)
        num_attention_heads = config.get("num_attention_heads", 24)
        attention_head_dim = config.get("attention_head_dim", 128)
        dim = num_attention_heads * attention_head_dim
        mlp_ratio = int(config.get("mlp_ratio", 3.0))

        # Create and load C++ model
        m = load_quantized_flux2_module(
            quantized_part_sd,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim * num_attention_heads,
            mlp_ratio=mlp_ratio,
            device=device,
            use_fp4=precision == "fp4",
            offload=offload,
            bf16=torch_dtype == torch.bfloat16,
        )

        # Inject C++ module into the Python model
        transformer.transformer_blocks = nn.ModuleList([NunchakuFlux2TransformerBlocks(m, device)])
        transformer.single_transformer_blocks = nn.ModuleList([])

        # Load unquantized parts (embeddings, modulations, norms, projections)
        transformer.to_empty(device=device)
        transformer.load_state_dict(unquantized_part_sd, strict=False)

        if kwargs.get("return_metadata", False):
            return transformer, metadata
        return transformer

    def set_attention_impl(self, impl: str, attn_func=None):
        """Set the attention implementation for the C++ backend."""
        block = self.transformer_blocks[0]
        if isinstance(block, NunchakuFlux2TransformerBlocks):
            if attn_func is None:
                attn_func = lambda x: x  # noqa: E731 — dummy, unused for non-custom impls
            block.m.setAttentionImpl(impl, attn_func)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> torch.Tensor | Transformer2DModelOutput:
        # Embeddings
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Time + guidance embedding → modulations
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        # Rotary embeddings
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        rotary_emb_img = _pack_flux2_rotary_emb(image_rotary_emb)
        rotary_emb_txt = _pack_flux2_rotary_emb(text_rotary_emb)
        rotary_emb_single = _pack_flux2_rotary_emb(
            (
                torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
                torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
            )
        )

        # Delegate to C++ backend
        nunchaku_block = self.transformer_blocks[0]
        encoder_hidden_states, hidden_states = nunchaku_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            mod_img=double_stream_mod_img,
            mod_txt=double_stream_mod_txt,
            mod_single=single_stream_mod,
            rotary_emb_img=rotary_emb_img,
            rotary_emb_txt=rotary_emb_txt,
            rotary_emb_single=rotary_emb_single,
        )

        # Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
