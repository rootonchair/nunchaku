New Model Integration Guide
===========================

This guide documents the step-by-step process for adding a new transformer model
to Nunchaku using the pure-Python (v2) pattern. It uses FLUX.2 as a worked example.

Prerequisites
-------------

Before starting, you need:

1. **A diffusers base class** — the upstream model must exist in ``diffusers.models.transformers``
   (e.g., ``Flux2Transformer2DModel``). Nunchaku wrappers inherit from these.

2. **Quantized weights** — a ``.safetensors`` checkpoint containing SVDQuant W4A4 parameters
   (``qweight``, ``wscales``, ``smooth_factor``, ``proj_down``, ``proj_up``) plus a
   ``quantization_config`` entry in the file metadata.

3. **Understanding of the model architecture** — specifically which layers are linear
   projections (candidates for quantization) and which are norms/embeddings (kept in
   full precision).

File Structure
--------------

Each model integration creates one new file and touches three ``__init__.py`` files:

.. code-block:: text

   nunchaku/
   +-- models/
   |   +-- transformers/
   |   |   +-- transformer_<name>.py     # NEW: model runtime
   |   |   +-- __init__.py               # EDIT: add export
   |   +-- __init__.py                   # EDIT: add export
   +-- __init__.py                       # EDIT: add export

Naming convention: ``transformer_<model_family>.py`` →
class ``Nunchaku<ModelFamily>Transformer2DModel``.

Step 1: Wrap Sub-Blocks
-----------------------

For each block type in the diffusers model, create a Nunchaku wrapper class that:

- Inherits from the diffusers block class (e.g., ``Flux2TransformerBlock``)
- Calls ``super(DiffusersBlockClass, self).__init__()`` to skip the parent's
  ``__init__`` (we replace all submodules manually)
- Replaces linear layers with ``SVDQW4A4Linear.from_linear(original, **kwargs)``
- Keeps norms and non-linear layers as-is

**Attention wrapper pattern:**

.. code-block:: python

   from ..linear import SVDQW4A4Linear
   from ..utils import fuse_linears

   class NunchakuMyAttention(DiffusersAttention):
       def __init__(self, other: DiffusersAttention, **kwargs):
           super(DiffusersAttention, self).__init__()  # skip parent init
           # Copy non-quantized attributes
           self.heads = other.heads
           self.head_dim = other.head_dim
           self.norm_q = other.norm_q
           self.norm_k = other.norm_k

           # Fuse Q/K/V into single quantized projection
           with torch.device("meta"):
               to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
           self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)

           # Quantize output projection
           self.to_out = other.to_out
           self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

``fuse_linears()`` (from ``nunchaku.models.utils``) creates a single ``nn.Linear``
with concatenated output features. It operates on ``meta`` device — no weights are
copied; the actual values come from the state dict later.

``SVDQW4A4Linear.from_linear()`` (from ``nunchaku.models.linear``) creates a
shape-only quantized linear module. Pass ``precision`` (``"int4"`` or ``"nvfp4"``)
and ``rank`` (SVD rank, typically 32) via ``**kwargs``.

**FeedForward wrapper pattern:**

.. code-block:: python

   class NunchakuMyFeedForward(DiffusersFeedForward):
       def __init__(self, other: DiffusersFeedForward, **kwargs):
           super(DiffusersFeedForward, self).__init__()
           self.linear_in = SVDQW4A4Linear.from_linear(other.linear_in, **kwargs)
           self.act_fn = other.act_fn
           self.linear_out = SVDQW4A4Linear.from_linear(other.linear_out, **kwargs)

For SwiGLU-style down-projections where ShiftedLinear is not applied during
quantization, set ``self.linear_out.act_unsigned = False`` to keep the signed
activation path.

**Transformer block wrapper pattern:**

.. code-block:: python

   class NunchakuMyTransformerBlock(DiffusersTransformerBlock):
       def __init__(self, block: DiffusersTransformerBlock, **kwargs):
           super(DiffusersTransformerBlock, self).__init__()
           self.norm1 = block.norm1              # keep norms
           self.attn = NunchakuMyAttention(block.attn, **kwargs)  # wrap attention
           self.norm2 = block.norm2
           self.ff = NunchakuMyFeedForward(block.ff, **kwargs)    # wrap MLP

Step 2: Optimized Forward Paths
-------------------------------

For performance, Nunchaku provides fused CUDA kernels. Use them when the input
is on CUDA and rotary embeddings are in packed format:

- ``fused_qkv_norm_rottary()`` (from ``nunchaku.ops.fused``) — fuses QKV projection
  + RMSNorm + rotary embedding application into one kernel call
- ``attention_fp16()`` (from ``nunchaku._C.ops``) — fused FP16 attention kernel

The forward method should check conditions and fall back to the unfused path
(standard PyTorch ops) when fused kernels can't be used (e.g., KV-cache mode,
non-CUDA device).

**Rotary embedding packing:**

FLUX.2 uses a different rotary format than FLUX.1. Each model may need a custom
packing function. The key requirement is that packed embeddings must be padded
to a multiple of 256 along the sequence dimension:

.. code-block:: python

   from ..embeddings import pack_rotemb
   from ...utils import pad_tensor

   rotemb = pack_rotemb(pad_tensor(rotemb, 256, 1))

Step 3: Main Model Class
-------------------------

The main model class inherits from both the diffusers model and
``NunchakuModelLoaderMixin``:

.. code-block:: python

   from .utils import NunchakuModelLoaderMixin, patch_scale_key

   class NunchakuMyTransformer2DModel(DiffusersTransformer2DModel, NunchakuModelLoaderMixin):
       def _patch_model(self, **kwargs):
           for i, block in enumerate(self.transformer_blocks):
               self.transformer_blocks[i] = NunchakuMyTransformerBlock(block, **kwargs)
           for i, block in enumerate(self.single_transformer_blocks):
               self.single_transformer_blocks[i] = NunchakuMySingleBlock(block, **kwargs)
           return self

**``from_pretrained()`` pattern:**

.. code-block:: python

   @classmethod
   def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
       device = kwargs.get("device", "cpu")
       torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

       # 1. Load safetensors → state_dict + metadata
       transformer, model_state_dict, metadata = cls._build_model(
           pretrained_model_name_or_path, **kwargs
       )

       # 2. Extract quantization config
       quantization_config = json.loads(metadata.get("quantization_config", "{}"))
       rank = int(quantization_config.get("rank", 32))

       # 3. Determine precision
       if quantization_config:
           precision = get_precision_from_quantization_config(quantization_config)
           if torch.device(device).type == "cuda":
               check_hardware_compatibility(quantization_config, device)
       else:
           precision = get_precision(device=device)
           if precision == "fp4":
               precision = "nvfp4"

       # 4. Patch model with quantized blocks
       transformer = transformer.to(torch_dtype)
       transformer._patch_model(precision=precision, rank=rank, torch_dtype=torch_dtype)
       transformer = transformer.to_empty(device=device)

       # 5. Handle state dict
       patch_scale_key(transformer, model_state_dict)  # adds wcscales for nvfp4
       transformer.load_state_dict(model_state_dict)

       return transformer

``_build_model()`` (from ``NunchakuModelLoaderMixin``) handles downloading from
HuggingFace Hub, loading the safetensors file, extracting the config, and creating
the diffusers model on ``meta`` device.

``patch_scale_key()`` (from ``nunchaku.models.transformers.utils``) creates missing
``.wcscales`` parameters (needed for nvfp4 precision) and extracts ``wtscale`` values.

**State dict key conversion:**

If your quantized checkpoint uses different key names than the diffusers model
(e.g., ``qkv_proj`` vs ``attn.to_qkv``), write a ``convert_<model>_state_dict()``
function to remap keys before ``load_state_dict()``. See
``convert_flux_state_dict()`` in ``transformer_flux_v2.py`` for an example.

Common key mappings:

- ``.lora_down`` → ``.proj_down`` (SVD low-rank projection)
- ``.lora_up`` → ``.proj_up``
- ``.smooth_orig`` → ``.smooth_factor_orig``
- ``.smooth`` → ``.smooth_factor``

Step 4: Register Exports
------------------------

Add the new class to three ``__init__.py`` files:

1. ``nunchaku/models/transformers/__init__.py``:

   .. code-block:: text

      from .transformer_mymodel import NunchakuMyModelTransformer2DModel

      __all__ = [
          ...,
          "NunchakuMyModelTransformer2DModel",
      ]

2. ``nunchaku/models/__init__.py`` — same import from ``.transformers``

3. ``nunchaku/__init__.py`` -- same import from ``.models``

Step 5: CPU Offload Support (Optional)
--------------------------------------

For models that need to run on low-VRAM GPUs, implement ``set_offload()`` using
``CPUOffloadManager`` (from ``nunchaku.models.utils``):

.. code-block:: python

   from ..utils import CPUOffloadManager

   def set_offload(self, offload: bool, **kwargs):
       if offload:
           self.transformer_block_offload_manager = CPUOffloadManager(
               self.transformer_blocks,
               use_pin_memory=kwargs.get("use_pin_memory", True),
               on_gpu_modules=[self.embedder, self.norm_out, self.proj_out],
               num_blocks_on_gpu=kwargs.get("num_blocks_on_gpu", 1),
           )

The forward method then uses ``get_block()`` / ``step()`` instead of direct
iteration. See ``NunchakuFlux2Transformer2DModel.forward()`` for the full pattern.

Step 6: Testing
---------------

1. **Import test**: Verify the new class is importable from the top-level package.

2. **Inference test**: Create an example script in ``examples/`` and a corresponding
   test in ``tests/``. Use LPIPS comparison against reference images.

3. **Backward compatibility**: Ensure existing model imports and tests still pass.

Checklist
---------

.. list-table::
   :widths: 5 50
   :header-rows: 0

   * - [ ]
     - Wrapper classes for all block types (Attention, FeedForward, TransformerBlock, SingleBlock)
   * - [ ]
     - Main model class with ``_patch_model()``, ``from_pretrained()``, ``forward()``
   * - [ ]
     - Fused kernel paths in forward methods (with fallback)
   * - [ ]
     - State dict key conversion function (if needed)
   * - [ ]
     - Export in all three ``__init__.py`` files
   * - [ ]
     - ``act_unsigned = False`` set on down-projections where ShiftedLinear is not applied
   * - [ ]
     - Rotary embeddings padded to 256 along sequence dim
   * - [ ]
     - ``patch_scale_key()`` called before ``load_state_dict()``
   * - [ ]
     - Hardware compatibility check in ``from_pretrained()``
   * - [ ]
     - Example script in ``examples/``
   * - [ ]
     - Test in ``tests/``
   * - [ ]
     - Pre-commit checks pass
