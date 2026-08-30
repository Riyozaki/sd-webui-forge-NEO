"""Central place for the low-level PyTorch knobs used while generating.

Everything here is strictly opt-out: the settings live in
*Settings -> Neo Optimizations*, and every knob is restored to the value it had
before the sampling call, so nothing leaks into the rest of the UI.

The three knobs are:

``cudnn.benchmark``
    Lets cuDNN auto-tune every convolution algorithm for the *actual* input
    shapes instead of using its built-in heuristics.  Sampling uses the very
    same shapes for every step of a job, so the (one-off) tuning cost is
    amortised over the whole generation and the UNet - which is dominated by
    convolutions - gets measurably faster.  It is deliberately *only* enabled
    while sampling: outside of it shapes change all the time (upscalers, VAE
    tiling, preprocessors) and tuning would then be a pure loss.

``matmul.allow_tf32`` / ``cudnn.allow_tf32``
    Only affects fp32 math.  Off by default because it is a (tiny) precision
    change; users of ``--all-in-fp32`` may want it.

``matmul.allow_fp16_accumulation``
    Allows fp16 GEMMs to accumulate in fp16.  Same story as the CMD flag
    ``--fast-fp16``: faster, marginally less precise.  Off by default.
"""

from __future__ import annotations

import contextlib

import torch

from modules import shared


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def low_vram_mode() -> bool:
    """Heuristic: autotuning costs extra scratch memory, so keep it off there."""
    try:
        from backend import memory_management

        if memory_management.vram_state in (
            memory_management.VRAMState.NO_VRAM,
            memory_management.VRAMState.LOW_VRAM,
            memory_management.VRAMState.SHARED,
        ):
            return True
        if memory_management.total_vram and memory_management.total_vram < 6 * 1024 * 1024 * 1024:
            return True
    except Exception:
        pass

    return False


def cudnn_benchmark_enabled() -> bool:
    try:
        if not shared.opts.neo_cudnn_benchmark:
            return False
    except Exception:
        return False

    return not low_vram_mode()


@contextlib.contextmanager
def sampling_context():
    """Apply (and afterwards revert) the performance knobs for one sampling call."""

    if not _cuda_available():
        yield
        return

    restore = []

    def _set(getter, setter, value):
        try:
            old = getter()
        except Exception:
            return
        if old == value:
            return
        try:
            setter(value)
        except Exception:
            return
        restore.append((setter, old))

    try:
        from backend.sampling import sampling_function

        sampling_function.FAST_SAMPLING_PATH = bool(getattr(shared.opts, "neo_fast_sampling_path", True))
    except Exception:
        pass

    try:
        _set(
            lambda: torch.backends.cudnn.benchmark,
            lambda v: setattr(torch.backends.cudnn, "benchmark", v),
            cudnn_benchmark_enabled(),
        )

        if getattr(shared.opts, "neo_tf32", False):
            _set(
                lambda: torch.backends.cuda.matmul.allow_tf32,
                lambda v: setattr(torch.backends.cuda.matmul, "allow_tf32", v),
                True,
            )
            _set(
                lambda: torch.backends.cudnn.allow_tf32,
                lambda v: setattr(torch.backends.cudnn, "allow_tf32", v),
                True,
            )

        if getattr(shared.opts, "neo_fp16_accumulation", False):
            _set(
                lambda: torch.backends.cuda.matmul.allow_fp16_accumulation,
                lambda v: setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", v),
                True,
            )
            try:
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
            except Exception:
                pass

        yield
    finally:
        for setter, old in restore:
            try:
                setter(old)
            except Exception:
                pass


def describe() -> str:
    """One-line summary used by the benchmark helper."""
    return (
        f"cudnn.benchmark={cudnn_benchmark_enabled()} "
        f"tf32={bool(getattr(shared.opts, 'neo_tf32', False))} "
        f"fp16_accumulation={bool(getattr(shared.opts, 'neo_fp16_accumulation', False))} "
        f"low_vram={low_vram_mode()}"
    )
