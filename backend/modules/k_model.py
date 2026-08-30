import math

import torch

from backend import attention, memory_management
from backend.modules.k_prediction import k_prediction_from_diffusers_scheduler


class KModel(torch.nn.Module):
    def __init__(self, model, diffusers_scheduler, k_predictor=None, config=None):
        super().__init__()

        self.config = config

        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        print(f"K-Model Created: {dict(storage_dtype=self.storage_dtype, computation_dtype=self.computation_dtype)}")

        self.diffusion_model = model

        if k_predictor is None:
            self.predictor = k_prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = k_predictor

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        sigma = t
        xc = self.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat((xc, c_concat), dim=1)

        context = c_crossattn
        dtype = self.computation_dtype

        xc = xc.to(dtype)
        t = self.predictor.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        model_output = self._run_diffusion_model(xc, t, context, control, transformer_options, extra_conds).float()
        return self.predictor.calculate_denoised(sigma, model_output, x)

    def _run_diffusion_model(self, xc, t, context, control, transformer_options, extra_conds):
        """Single entry point for the denoiser, so the cache has exactly one place to hook into."""

        def run():
            return self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)

        try:
            from modules import neo_cache

            cache = neo_cache.teacache
        except Exception:
            return run()

        threshold, _ = cache.settings()
        if threshold <= 0 or cache.unsupported:
            return run()

        timestep_value = self._timestep_value(t)

        # a split step (cond and uncond in two separate calls) must not reuse the
        # residual of its other half
        if cache.is_repeat_call(timestep_value):
            return run()

        emb = self._time_embedding(t, dtype=xc.dtype)
        if emb is None or timestep_value is None:
            cache.unsupported = True
            return run()

        if not cache.should_compute(emb, xc):
            out = cache.skip(xc)
        else:
            out = run()
            cache.update(emb, out, xc)

        cache.mark_step(timestep_value)
        return out

    @staticmethod
    def _timestep_value(t):
        """The current timestep as a Python float; ``None`` if it cannot be read."""
        try:
            return float(t.detach().reshape(-1)[0].item())
        except Exception:
            return None

    def _time_embedding(self, t, dtype=None):
        """The timestep embedding used to decide whether a step can be skipped.

        ``None`` means "this architecture has nothing we can measure", which
        disables the cache for the rest of the sampling run.
        """
        try:
            from backend.nn.unet import timestep_embedding

            model = self.diffusion_model
            channels = getattr(model, "model_channels", None)
            time_embed = getattr(model, "time_embed", None)

            if channels is None or time_embed is None:
                return None

            t_emb = timestep_embedding(t, channels)
            if dtype is not None:
                t_emb = t_emb.to(dtype)

            return time_embed(t_emb)
        except Exception:
            return None

    def memory_required(self, input_shape: list[int]) -> float:
        """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.56/comfy/model_base.py#L350"""
        input_shapes = [input_shape]
        area = sum(map(lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes))

        if memory_management.xformers_enabled():
            return (area * memory_management.dtype_size(self.computation_dtype) * 0.01 * self.config.memory_usage_factor) * (1024 * 1024)
        else:
            return (area * 0.15 * self.config.memory_usage_factor) * (1024 * 1024)
