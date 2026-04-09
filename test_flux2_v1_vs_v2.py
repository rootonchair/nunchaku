"""Benchmark FLUX.2 Klein 4B: v1 (C++ backend) vs v2 (pure Python) speed and quality."""

import argparse
import gc
import os

import torch
from diffusers import Flux2KleinPipeline

REPO = "black-forest-labs/FLUX.2-klein-4B"
WEIGHTS = "/mnt/disks/workspace/nunchaku/nunchaku_flux2_klein_4b_int4.safetensors"
OUTPUT_DIR = "flux2_v1_vs_v2"
SEED = 42
NUM_STEPS = 4
GUIDANCE_SCALE = 1.0

PROMPTS = [
    "A cat holding a sign that says 'Hello, World!' in a park during sunset",
    "A futuristic city skyline at night with neon lights reflecting on water",
    "An astronaut riding a horse on the surface of Mars, photorealistic",
]


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def benchmark(pipe, *, label: str, num_warmup: int, num_runs: int, save_images: bool):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gen_kwargs = dict(
        prompt=PROMPTS[0],
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
        generator=torch.Generator("cpu").manual_seed(SEED),
    )

    # Warmup
    for _ in range(num_warmup):
        pipe(**gen_kwargs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed runs
    timings = []
    for i in range(num_runs):
        kwargs = dict(
            prompt=PROMPTS[i % len(PROMPTS)],
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_STEPS,
            generator=torch.Generator("cpu").manual_seed(SEED),
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = pipe(**kwargs)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        timings.append(ms)

        if save_images and i < len(PROMPTS):
            path = os.path.join(OUTPUT_DIR, f"{label}_{i:02d}.png")
            result.images[0].save(path)
            print(f"  [{label}] run {i}: {ms:.1f} ms -> {path}")
        else:
            print(f"  [{label}] run {i}: {ms:.1f} ms")

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    return timings, peak_mem


def report(label: str, timings: list[float], peak_mem: float):
    avg = sum(timings) / len(timings)
    print(f"\n  {label}")
    print(f"  {'─' * 40}")
    print(f"  Runs:        {len(timings)}")
    print(f"  Avg latency: {avg:.1f} ms  ({1000/avg:.2f} it/s)")
    print(f"  Min latency: {min(timings):.1f} ms")
    print(f"  Max latency: {max(timings):.1f} ms")
    print(f"  Peak VRAM:   {peak_mem:.2f} GiB")
    return avg


def main():
    parser = argparse.ArgumentParser(description="Benchmark Flux2 Klein 4B: V1 (C++) vs V2 (Python)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--save-images", action="store_true", help="Save output images for comparison")
    parser.add_argument("--skip-v1", action="store_true")
    parser.add_argument("--skip-v2", action="store_true")
    args = parser.parse_args()

    v2_avg = v1_avg = None
    v2_mem = v1_mem = 0

    # --- V2 (pure Python) ---
    if not args.skip_v2:
        print("\n>>> Loading V2 (pure Python) pipeline...")
        from nunchaku.models.transformers.transformer_flux2 import NunchakuFlux2Transformer2DModel

        flush()
        transformer = NunchakuFlux2Transformer2DModel.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
        pipe = Flux2KleinPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16, transformer=transformer)
        pipe.to("cuda")

        print(f">>> Benchmarking V2: {args.warmup} warmup + {args.runs} timed runs")
        v2_timings, v2_mem = benchmark(
            pipe, label="v2", num_warmup=args.warmup, num_runs=args.runs, save_images=args.save_images
        )
        v2_avg = report("V2 (pure Python)", v2_timings, v2_mem)

        del pipe, transformer
        flush()

    # --- V1 (C++ backend) ---
    if not args.skip_v1:
        print("\n>>> Loading V1 (C++ backend) pipeline...")
        from nunchaku.models.transformers.transformer_flux2_v1 import NunchakuFlux2Transformer2DModelV1

        flush()
        torch.cuda.reset_peak_memory_stats()
        transformer = NunchakuFlux2Transformer2DModelV1.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
        pipe = Flux2KleinPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16, transformer=transformer)
        pipe.to("cuda")
        transformer.set_attention_impl("nunchaku-fp16")

        print(f">>> Benchmarking V1: {args.warmup} warmup + {args.runs} timed runs")
        v1_timings, v1_mem = benchmark(
            pipe, label="v1", num_warmup=args.warmup, num_runs=args.runs, save_images=args.save_images
        )
        v1_avg = report("V1 (C++ backend)", v1_timings, v1_mem)

        del pipe, transformer
        flush()

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print(f"  Summary")
    print(f"{'=' * 50}")
    if v2_avg:
        print(f"  V2 avg: {v2_avg:.1f} ms | VRAM: {v2_mem:.2f} GiB")
    if v1_avg:
        print(f"  V1 avg: {v1_avg:.1f} ms | VRAM: {v1_mem:.2f} GiB")
    if v1_avg and v2_avg:
        print(f"  Speedup:     {v2_avg / v1_avg:.2f}x")
        print(f"  VRAM saving: {v2_mem - v1_mem:.2f} GiB")


if __name__ == "__main__":
    main()
