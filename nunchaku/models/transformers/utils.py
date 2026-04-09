"""
Utilities for Nunchaku transformer model loading.
"""

import json
import logging
import os
from pathlib import Path

import torch
from diffusers import __version__
from huggingface_hub import constants, hf_hub_download
from torch import nn

from ...utils import load_state_dict_in_safetensors
from ..linear import SVDQW4A4Linear

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuModelLoaderMixin:
    """
    Mixin for standardized model loading in Nunchaku transformer models.
    """

    @classmethod
    def _build_model(
        cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs
    ) -> tuple[nn.Module, dict[str, torch.Tensor], dict[str, str]]:
        """
        Build a transformer model from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file.
        **kwargs
            Additional keyword arguments (e.g., ``torch_dtype``).

        Returns
        -------
        tuple
            (transformer, state_dict, metadata)
        """
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        state_dict, metadata = load_state_dict_in_safetensors(pretrained_model_name_or_path, return_metadata=True)

        config = json.loads(metadata["config"])

        with torch.device("meta"):
            transformer = cls.from_config(config).to(kwargs.get("torch_dtype", torch.bfloat16))

        return transformer, state_dict, metadata

    @classmethod
    def _build_model_legacy(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[nn.Module, str, str]:
        """
        Build a transformer model from a legacy folder structure.

        .. warning::
            This method is deprecated and will be removed in December 2025.
            Please use :meth:`_build_model` instead.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the folder containing model weights.
        **kwargs
            Additional keyword arguments for HuggingFace Hub download and config loading.

        Returns
        -------
        tuple
            (transformer, unquantized_part_path, transformer_block_path)
        """
        logger.warning(
            "Loading models from a folder will be deprecated in December 2025. "
            "Please download the latest safetensors model, or use one of the following tools to "
            "merge your model into a single file: the CLI utility `python -m nunchaku.merge_safetensors` "
            "or the ComfyUI workflow `merge_safetensors.json`."
        )
        subfolder = kwargs.get("subfolder", None)
        if os.path.exists(pretrained_model_name_or_path):
            dirname = (
                pretrained_model_name_or_path
                if subfolder is None
                else os.path.join(pretrained_model_name_or_path, subfolder)
            )
            unquantized_part_path = os.path.join(dirname, "unquantized_layers.safetensors")
            transformer_block_path = os.path.join(dirname, "transformer_blocks.safetensors")
        else:
            download_kwargs = {
                "subfolder": subfolder,
                "repo_type": "model",
                "revision": kwargs.get("revision", None),
                "cache_dir": kwargs.get("cache_dir", None),
                "local_dir": kwargs.get("local_dir", None),
                "user_agent": kwargs.get("user_agent", None),
                "force_download": kwargs.get("force_download", False),
                "proxies": kwargs.get("proxies", None),
                "etag_timeout": kwargs.get("etag_timeout", constants.DEFAULT_ETAG_TIMEOUT),
                "token": kwargs.get("token", None),
                "local_files_only": kwargs.get("local_files_only", None),
                "headers": kwargs.get("headers", None),
                "endpoint": kwargs.get("endpoint", None),
                "resume_download": kwargs.get("resume_download", None),
                "force_filename": kwargs.get("force_filename", None),
                "local_dir_use_symlinks": kwargs.get("local_dir_use_symlinks", "auto"),
            }
            unquantized_part_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path), filename="unquantized_layers.safetensors", **download_kwargs
            )
            transformer_block_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path), filename="transformer_blocks.safetensors", **download_kwargs
            )

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        config, _, _ = cls.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            user_agent={"diffusers": __version__, "file_type": "model", "framework": "pytorch"},
            **kwargs,
        )

        with torch.device("meta"):
            transformer = cls.from_config(config).to(kwargs.get("torch_dtype", torch.bfloat16))
        return transformer, unquantized_part_path, transformer_block_path


def patch_scale_key(transformer_from_config: nn.Module, state_dict_from_checkpoint: dict):
    """
    Modify scale parameters so that the state dict from the checkpoint file can be loaded to the transformer model created from the config.

    Parameters
    ----------
    transformer_from_config : nn.Module
        The transformer model created from the `config.json`
    state_dict_from_checkpoint : dict
        The state dict loaded from the checkpoint file (typically .safetensors)
    """
    state_dict = transformer_from_config.state_dict()
    for k in state_dict.keys():
        if k not in state_dict_from_checkpoint:
            assert ".wcscales" in k
            state_dict_from_checkpoint[k] = torch.ones_like(state_dict[k])

    for n, m in transformer_from_config.named_modules():
        if isinstance(m, SVDQW4A4Linear):
            if m.wtscale is not None:
                m.wtscale = state_dict_from_checkpoint.pop(f"{n}.wtscale", 1.0)


def convert_flux_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert a v1-style (C++ backend) state dict to diffusers-native key names
    expected by :class:`~nunchaku.models.transformers.transformer_flux_v2.NunchakuFluxTransformer2DModelV2`.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        The original state dict.

    Returns
    -------
    dict[str, torch.Tensor]
        The converted state dict.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if "single_transformer_blocks." in k:
            if ".qkv_proj." in k:
                new_k = k.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".out_proj." in k:
                new_k = k.replace(".out_proj.", ".attn.to_out.")
            elif ".norm_q." in k or ".norm_k." in k:
                new_k = k.replace(".norm_k.", ".attn.norm_k.")
                new_k = new_k.replace(".norm_q.", ".attn.norm_q.")
            else:
                new_k = k
            new_k = new_k.replace(".lora_down", ".proj_down")
            new_k = new_k.replace(".lora_up", ".proj_up")
            if ".smooth_orig" in k:
                new_k = new_k.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in k:
                new_k = new_k.replace(".smooth", ".smooth_factor")
            new_state_dict[new_k] = v
        elif "transformer_blocks." in k:
            if ".mlp_context_fc1" in k:
                new_k = k.replace(".mlp_context_fc1.", ".ff_context.net.0.proj.")
            elif ".mlp_context_fc2" in k:
                new_k = k.replace(".mlp_context_fc2.", ".ff_context.net.2.")
            elif ".mlp_fc1" in k:
                new_k = k.replace(".mlp_fc1.", ".ff.net.0.proj.")
            elif ".mlp_fc2" in k:
                new_k = k.replace(".mlp_fc2.", ".ff.net.2.")
            elif ".qkv_proj_context." in k:
                new_k = k.replace(".qkv_proj_context.", ".attn.add_qkv_proj.")
            elif ".qkv_proj." in k:
                new_k = k.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".norm_q." in k or ".norm_k." in k:
                new_k = k.replace(".norm_k.", ".attn.norm_k.")
                new_k = new_k.replace(".norm_q.", ".attn.norm_q.")
            elif ".norm_added_q." in k or ".norm_added_k." in k:
                new_k = k.replace(".norm_added_k.", ".attn.norm_added_k.")
                new_k = new_k.replace(".norm_added_q.", ".attn.norm_added_q.")
            elif ".out_proj." in k:
                new_k = k.replace(".out_proj.", ".attn.to_out.0.")
            elif ".out_proj_context." in k:
                new_k = k.replace(".out_proj_context.", ".attn.to_add_out.")
            else:
                new_k = k
            new_k = new_k.replace(".lora_down", ".proj_down")
            new_k = new_k.replace(".lora_up", ".proj_up")
            if ".smooth_orig" in k:
                new_k = new_k.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in k:
                new_k = new_k.replace(".smooth", ".smooth_factor")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def convert_flux2_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert a v1-style (C++ backend) state dict to diffusers-native key names
    expected by :class:`~nunchaku.models.transformers.transformer_flux2.NunchakuFlux2Transformer2DModel`.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        The original state dict from a quantized safetensors checkpoint.

    Returns
    -------
    dict[str, torch.Tensor]
        The converted state dict.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if "single_transformer_blocks." in k:
            if ".qkv_proj." in k:
                new_k = k.replace(".qkv_proj.", ".attn.qkv_proj.")
            elif ".out_proj." in k:
                new_k = k.replace(".out_proj.", ".attn.out_proj.")
            elif ".mlp_fc1." in k:
                new_k = k.replace(".mlp_fc1.", ".attn.mlp_fc1.")
            elif ".mlp_fc2." in k:
                new_k = k.replace(".mlp_fc2.", ".attn.mlp_fc2.")
            elif ".norm_q." in k or ".norm_k." in k:
                new_k = k.replace(".norm_k.", ".attn.norm_k.")
                new_k = new_k.replace(".norm_q.", ".attn.norm_q.")
            else:
                new_k = k
        elif "transformer_blocks." in k:
            if ".mlp_context_fc1." in k:
                new_k = k.replace(".mlp_context_fc1.", ".ff_context.linear_in.")
            elif ".mlp_context_fc2." in k:
                new_k = k.replace(".mlp_context_fc2.", ".ff_context.linear_out.")
            elif ".mlp_fc1." in k:
                new_k = k.replace(".mlp_fc1.", ".ff.linear_in.")
            elif ".mlp_fc2." in k:
                new_k = k.replace(".mlp_fc2.", ".ff.linear_out.")
            elif ".qkv_proj_context." in k:
                new_k = k.replace(".qkv_proj_context.", ".attn.to_added_qkv.")
            elif ".qkv_proj." in k:
                new_k = k.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".norm_added_q." in k or ".norm_added_k." in k:
                new_k = k.replace(".norm_added_k.", ".attn.norm_added_k.")
                new_k = new_k.replace(".norm_added_q.", ".attn.norm_added_q.")
            elif ".norm_q." in k or ".norm_k." in k:
                new_k = k.replace(".norm_k.", ".attn.norm_k.")
                new_k = new_k.replace(".norm_q.", ".attn.norm_q.")
            elif ".out_proj_context." in k:
                new_k = k.replace(".out_proj_context.", ".attn.to_add_out.")
            elif ".out_proj." in k:
                new_k = k.replace(".out_proj.", ".attn.to_out.0.")
            else:
                new_k = k
        else:
            new_k = k

        # Common renames: SVD projections and smooth factors
        new_k = new_k.replace(".lora_down", ".proj_down")
        new_k = new_k.replace(".lora_up", ".proj_up")
        if ".smooth_orig" in new_k:
            new_k = new_k.replace(".smooth_orig", ".smooth_factor_orig")
        elif ".smooth" in new_k:
            new_k = new_k.replace(".smooth", ".smooth_factor")

        new_state_dict[new_k] = v

    return new_state_dict


def convert_fp16(transformer_from_config: nn.Module, state_dict_from_checkpoint: dict):
    state_dict = transformer_from_config.state_dict()
    for k in state_dict.keys():
        if state_dict[k].dtype != state_dict_from_checkpoint[k].dtype:
            assert (
                state_dict[k].dtype == torch.float16 and state_dict_from_checkpoint[k].dtype == torch.bfloat16
            ), f"Unexpected dtype difference for key: {k}, model dtype: {state_dict[k].dtype}, \
                checkpoint dtype: {state_dict_from_checkpoint[k].dtype}"
            state_dict_from_checkpoint[k] = torch.nan_to_num(
                state_dict_from_checkpoint[k].to(torch.float16), nan=0.0, posinf=65504, neginf=-65504
            )
