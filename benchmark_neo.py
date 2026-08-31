"""Benchmark for sd-webui-forge-neo.

Runs the same generation several times and reports timings, so that the effect of
the optimizations in *Settings -> Neo Optimizations* can be measured on real
hardware instead of guessed at.

Usage::

    python benchmark_neo.py --steps 20 --width 1024 --height 1024 --runs 3
    python benchmark_neo.py --ab neo_cudnn_benchmark          # on vs. off
    python benchmark_neo.py --ab neo_fast_sampling_path --steps 8 --runs 5
    python benchmark_neo.py --ab neo_cuda_graph --runs 5       # CUDA graph replay

For the TeaCache threshold and TorchInductor compilation, which are not booleans,
set the option in the UI (or config.json) and run without --ab; the console then
reports the achieved skip rate / replay count for each generation.

Notes:

* the first run of every configuration is a warm-up and is not counted
* use ``--no-save`` (the default) so that disk speed does not pollute the numbers
* everything is measured with the normal webui pipeline, so whatever samplers,
  LoRAs and ControlNets are configured are used as-is
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark sd-webui-forge-neo")

    p.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars, high quality")
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--batch-count", type=int, default=1)
    p.add_argument("--sampler", type=str, default=None, help="defaults to the sampler selected in the UI")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--cfg-scale", type=float, default=7.0)

    p.add_argument("--runs", type=int, default=3, help="timed runs per configuration")
    p.add_argument("--warmup", type=int, default=1, help="untimed runs before the timed ones")
    p.add_argument(
        "--ab",
        type=str,
        default=None,
        help="name of a boolean setting to A/B test, e.g. neo_cudnn_benchmark",
    )
    p.add_argument("--save", action="store_true", help="save the generated images (off by default)")
    p.add_argument("--json", type=str, default=None, help="also write the results to this JSON file")

    return p.parse_args()


def boot():
    from modules import initialize
    from modules_forge.initialization import initialize_forge

    initialize_forge()
    initialize.imports()
    initialize.initialize()

    return initialize


def measure(args, *, label: str):
    from modules import processing, shared
    from modules.processing import StableDiffusionProcessingTxt2Img

    results = []

    total_runs = args.warmup + args.runs

    for index in range(total_runs):
        warm = index < args.warmup

        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            steps=args.steps,
            width=args.width,
            height=args.height,
            batch_size=args.batch_size,
            n_iter=args.batch_count,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            sampler_name=args.sampler,
            do_not_save_samples=not args.save,
            do_not_save_grid=not args.save,
        )

        with contextlib.closing(p):
            shared.state.begin(job="benchmark")
            shared.state.textinfo = "benchmark"

            try:
                started = time.perf_counter()
                processed = processing.process_images(p)
                elapsed = time.perf_counter() - started
            finally:
                shared.state.end()

        images = len(getattr(processed, "images", []) or [])
        steps_done = args.steps * args.batch_count

        record = {
            "elapsed": elapsed,
            "images": images,
            "steps": steps_done,
            "s_per_image": elapsed / images if images else 0.0,
            "it_per_s": steps_done / elapsed if elapsed > 0 else 0.0,
            "warmup": warm,
        }

        results.append(record)

        tag = "warm-up" if warm else "run"
        print(
            f"  [{label}] {tag} {index + 1}/{total_runs}: "
            f"{elapsed:.2f}s  ({record['s_per_image']:.2f}s/image, {record['it_per_s']:.2f} it/s)"
        )

    timed = [r for r in results if not r["warmup"]]
    return {
        "label": label,
        "runs": len(timed),
        "mean_s": statistics.fmean(r["elapsed"] for r in timed),
        "min_s": min(r["elapsed"] for r in timed),
        "stdev_s": statistics.stdev([r["elapsed"] for r in timed]) if len(timed) > 1 else 0.0,
        "mean_it_per_s": statistics.fmean(r["it_per_s"] for r in timed),
        "mean_s_per_image": statistics.fmean(r["s_per_image"] for r in timed),
        "details": timed,
    }


def vram_peak_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def main():
    args = parse_args()

    boot()

    from modules import shared

    print(f"\nModel:   {shared.opts.sd_model_checkpoint}")
    print(f"Sampler: {args.sampler or '(the one selected in the UI)'}")
    print(f"Size:    {args.width}x{args.height}, {args.steps} steps, batch {args.batch_size} x {args.batch_count}\n")

    try:
        from modules import neo_tuning

        print("Backend: " + neo_tuning.describe() + "\n")
    except Exception:
        pass

    summary = []

    if args.ab:
        if not hasattr(shared.opts, args.ab):
            raise SystemExit(f"unknown setting: {args.ab}")

        for value in (False, True):
            shared.opts.set(args.ab, value)
            print(f"--- {args.ab} = {value} ---")
            summary.append(measure(args, label=f"{args.ab}={value}"))
            print()
    else:
        print("--- default ---")
        summary.append(measure(args, label="default"))
        print()

    print("=" * 72)
    for s in summary:
        print(
            f"{s['label']:<28} {s['mean_s']:7.2f}s  min {s['min_s']:7.2f}s  "
            f"±{s['stdev_s']:5.2f}s   {s['mean_it_per_s']:6.2f} it/s   {s['mean_s_per_image']:6.2f}s/image"
        )
    print("=" * 72)

    if len(summary) == 2:
        a, b = summary
        if a["mean_s"] > 0:
            delta = (a["mean_s"] - b["mean_s"]) / a["mean_s"] * 100.0
            print(f"{b['label']} is {delta:+.1f}% faster than {a['label']}")

    peak = vram_peak_mb()
    if peak:
        print(f"peak VRAM (torch allocations): {peak:.0f} MB")

    if args.json:
        with open(args.json, "w", encoding="utf8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
