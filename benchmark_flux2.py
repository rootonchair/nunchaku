"""Benchmark Nunchaku (INT4) vs original (BF16) Flux2 Klein 4B inference."""

import argparse
import gc
import os

import torch
from diffusers import Flux2KleinPipeline

from nunchaku.models.transformers.transformer_flux2 import NunchakuFlux2Transformer2DModel

REPO = "black-forest-labs/FLUX.2-klein-4B"
NUM_STEPS = 4
GUIDANCE_SCALE = 1.0
SEED = 42

PROMPTS = [
    "A cat holding a sign that says 'Hello, World!' in a park during sunset",
    "A futuristic city skyline at night with neon lights reflecting on water",
    "A watercolor painting of a cozy cabin in the mountains during autumn",
    "An astronaut riding a horse on the surface of Mars, photorealistic",
    "A bowl of ramen with steam rising, top-down view, food photography",
    "A medieval knight standing in a field of sunflowers, cinematic lighting",
    "A corgi wearing a tiny crown sitting on a velvet throne, studio photo",
    "An oil painting of a stormy sea with a lighthouse in the distance",
    "A robot reading a book in a library, soft ambient light, 3D render",
    "A macro photo of a dewdrop on a leaf reflecting a mountain landscape",
]


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_pipeline(pipe, *, output_dir: str, label: str, num_warmup: int) -> tuple[list[float], float]:
    """Run warmup, then generate one image per prompt with timing. Save outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # Warmup with first prompt
    for _ in range(num_warmup):
        pipe(
            prompt=PROMPTS[0],
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_STEPS,
            generator=torch.Generator("cpu").manual_seed(SEED),
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed runs — one per prompt, same seed each time
    timings = []
    for i, prompt in enumerate(PROMPTS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = pipe(
            prompt=prompt,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_STEPS,
            generator=torch.Generator("cpu").manual_seed(SEED),
        )
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

        path = os.path.join(output_dir, f"{label}_{i:02d}.png")
        result.images[0].save(path)
        print(f"  [{label}] prompt {i:02d}: {timings[-1]:.1f} ms -> {path}")

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    return timings, peak_mem


def report(label: str, timings: list[float], peak_mem: float):
    avg = sum(timings) / len(timings)
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Images:      {len(timings)}")
    print(f"  Avg latency: {avg:.1f} ms")
    print(f"  Min latency: {min(timings):.1f} ms")
    print(f"  Max latency: {max(timings):.1f} ms")
    print(f"  Peak VRAM:   {peak_mem:.2f} GiB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Flux2 Klein 4B: Nunchaku vs Original")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument(
        "--weights",
        type=str,
        default="/mnt/disks/workspace/nunchaku/nunchaku_flux2_klein_4b_int4.safetensors",
        help="Path to Nunchaku INT4 safetensors",
    )
    parser.add_argument("--output-dir", type=str, default="benchmark_outputs", help="Directory to save images")
    parser.add_argument("--skip-original", action="store_true", help="Skip original BF16 benchmark")
    parser.add_argument("--skip-nunchaku", action="store_true", help="Skip Nunchaku INT4 benchmark")
    args = parser.parse_args()

    nunchaku_timings = None
    original_timings = None

    # --- Nunchaku (INT4) ---
    if not args.skip_nunchaku:
        print("\n>>> Loading Nunchaku (INT4) pipeline...")
        flush()
        transformer = NunchakuFlux2Transformer2DModel.from_pretrained(
            args.weights, torch_dtype=torch.bfloat16
        )
        pipe = Flux2KleinPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16, transformer=transformer)
        pipe.to("cuda")

        print(f">>> Generating {len(PROMPTS)} images with Nunchaku ({args.warmup} warmup)...")
        nunchaku_timings, nunchaku_mem = run_pipeline(
            pipe, output_dir=args.output_dir, label="nunchaku", num_warmup=args.warmup
        )
        report("Nunchaku INT4", nunchaku_timings, nunchaku_mem)

        del pipe, transformer
        flush()

    # --- Original (BF16) ---
    if not args.skip_original:
        print("\n>>> Loading original (BF16) pipeline...")
        flush()
        pipe = Flux2KleinPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
        pipe.to("cuda")

        print(f">>> Generating {len(PROMPTS)} images with Original ({args.warmup} warmup)...")
        original_timings, original_mem = run_pipeline(
            pipe, output_dir=args.output_dir, label="original", num_warmup=args.warmup
        )
        report("Original BF16", original_timings, original_mem)

        del pipe
        flush()

    # --- Summary ---
    if nunchaku_timings and original_timings:
        nunchaku_avg = sum(nunchaku_timings) / len(nunchaku_timings)
        original_avg = sum(original_timings) / len(original_timings)
        print(f"\n{'=' * 50}")
        print(f"  Summary")
        print(f"{'=' * 50}")
        print(f"  Speedup:     {original_avg / nunchaku_avg:.2f}x")
        print(f"  VRAM saving: {original_mem - nunchaku_mem:.2f} GiB ({original_mem:.2f} -> {nunchaku_mem:.2f})")
        print(f"\n  Images saved to: {args.output_dir}/")
        print(f"  Compare pairs: nunchaku_XX.png vs original_XX.png (same seed={SEED})")


if __name__ == "__main__":
    main()
