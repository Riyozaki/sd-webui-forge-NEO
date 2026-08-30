"""TeaCache - skip UNet evaluations whose result would barely change.

The idea (from *Timestep Embedding Aware Cache*, Liu et al.) is simple: the UNet
output changes smoothly with the timestep, so for many consecutive steps the
*difference* between the current and the previous output is tiny.  We therefore
remember the residual of the last computed step and, when the time embedding has
barely moved, answer the step with

    out = x + previous_residual

instead of running the whole UNet.

Caveats, in plain words:

* **this is an approximation** - it is disabled (threshold ``0``) by default.
  The higher the threshold, the more steps are skipped and the more the result
  drifts from the un-cached one.
* deciding whether a step can be skipped needs the distance between two
  embeddings as a Python number, which means **one device synchronisation per
  step**.  That is why the numbers below should be measured rather than assumed:
  on a slow GPU the saved UNet evaluations win by a wide margin, on a very fast
  one the sync may eat into the win.
* the cache is reset at the beginning and the end of every sampling call, and
  whenever the shape of the latent changes, so a hires pass or a new batch can
  never reuse a stale residual.

The upstream implementation rescales the measured distance with a per-model
polynomial.  We deliberately do **not** hard-code those coefficients (they are
numbers we could not verify here); a plain relative L1 distance with a
user-tunable threshold is used instead, and the achieved skip rate is reported
so the threshold can be tuned from evidence.
"""

from __future__ import annotations

import torch


class TeaCache:
    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self):
        self.prev_emb = None
        self.prev_residual = None
        self.accumulated = 0.0
        self.shape = None

        self.step = 0
        self.computed = 0
        self.skipped = 0
        self.total_steps = None

        self.step_t = None
        self.step_handled = False

        self.unsupported = False
        """set when the model has no time embedding we can measure"""

    def release(self):
        """Drop the cached tensors so they cannot pin VRAM between jobs."""
        self.prev_emb = None
        self.prev_residual = None
        self.shape = None

    # ------------------------------------------------------------------ lifecycle

    def begin(self, total_steps: int | None = None):
        self.reset()
        self.total_steps = total_steps

    def finish(self):
        summary = self.summary()
        if summary:
            print(summary)

        was_enabled = self.computed + self.skipped
        self.reset()
        return was_enabled

    # ------------------------------------------------------------------ config

    @staticmethod
    def settings() -> tuple[float, float]:
        try:
            from modules import shared

            threshold = float(getattr(shared.opts, "neo_teacache_threshold", 0.0) or 0.0)
            warmup = float(getattr(shared.opts, "neo_teacache_warmup", 0.0) or 0.0)
        except Exception:
            return 0.0, 0.0

        return threshold, max(0.0, min(warmup, 0.9))

    # ------------------------------------------------------------------ step tracking

    def is_repeat_call(self, timestep_value: float | None) -> bool:
        """True when this is *not* the first forward call of the current step.

        ``calc_cond_uncond_batch`` normally batches the conditional and the
        unconditional pass into a single call, but when memory is tight it splits
        them.  In that case the second call has the very same timestep, and
        reusing the residual of the first half would mix both halves up - so
        every call after the first one of a step simply runs normally.
        """
        if timestep_value is None or not self.step_handled:
            return False

        return self.step_t is not None and self.step_t == timestep_value

    def mark_step(self, timestep_value: float | None):
        self.step_t = timestep_value
        self.step_handled = True

    # ------------------------------------------------------------------ decision

    def should_compute(self, emb: torch.Tensor, x: torch.Tensor, total_steps: int | None = None) -> bool:
        """Return ``True`` when the UNet really has to run for this step."""

        if self.unsupported:
            return True

        threshold, warmup = self.settings()
        if threshold <= 0:
            return True

        shape = tuple(x.shape)
        if self.prev_emb is None or self.prev_residual is None or shape != self.shape:
            return True

        steps = int(total_steps or self.total_steps or 0)
        if steps and warmup > 0 and self.step < warmup * steps:
            return True

        if self.prev_emb.shape != emb.shape:
            return True

        try:
            rel = (emb - self.prev_emb).abs().mean() / (self.prev_emb.abs().mean() + 1e-8)
            distance = float(rel.detach().item())
        except Exception:
            return True

        if not (distance == distance):  # NaN
            return True

        self.accumulated += distance

        if self.accumulated < threshold:
            self.skipped += 1
            self.step += 1
            return False

        self.accumulated = 0.0
        return True

    def update(self, emb: torch.Tensor, out: torch.Tensor, x: torch.Tensor):
        """Remember the residual of a step that was actually computed."""

        self.computed += 1
        self.step += 1

        self.prev_emb = emb.detach().clone()
        self.prev_residual = (out - x).detach().clone()
        self.shape = tuple(x.shape)

    def skip(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.prev_residual

    # ------------------------------------------------------------------ reporting

    def summary(self) -> str:
        if self.computed + self.skipped == 0:
            return ""

        total = self.computed + self.skipped
        return f"TeaCache: skipped {self.skipped}/{total} steps ({self.skipped / total * 100:.0f}%)"


teacache = TeaCache()
"""Process wide singleton.  There is one sampler running at a time, and the cache
is reset around every sampling call anyway."""
