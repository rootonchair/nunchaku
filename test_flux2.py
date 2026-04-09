import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image

from nunchaku.models.transformers.transformer_flux2 import NunchakuFlux2Transformer2DModel
from nunchaku.utils import get_precision

REPO = "black-forest-labs/FLUX.2-klein-4B"  # or local absolute path

transformer = NunchakuFlux2Transformer2DModel.from_pretrained(
    "/mnt/disks/workspace/nunchaku/nunchaku_flux2_klein_4b_int4.safetensors",
    torch_dtype=torch.bfloat16,
)
pipe = Flux2KleinPipeline.from_pretrained(
    REPO, torch_dtype=torch.bfloat16, transformer=transformer
)

pipe.to("cuda")
# transformer.set_offload(
#     True, use_pin_memory=False, num_blocks_on_gpu=1
# )
# pipeline._exclude_from_cpu_offload.append("transformer")
# pipeline.enable_sequential_cpu_offload()

# ref = load_image("https://example.com/your_ref.png").convert("RGB")
image = pipe(
    prompt="A cat holding a sign that says 'Hello, World!' in a park during sunset",
    guidance_scale=1.0,  # matches official Klein examples; tune if needed
    num_inference_steps=4,  # common for the distilled model; see Diffusers docs otherwise
    generator=torch.Generator("cpu").manual_seed(1),
).images[0]
image.save("flux2_klein_nunchaku.png")
