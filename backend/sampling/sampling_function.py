# Started from some codes from early ComfyUI and then 80% rewritten,
# mainly for supporting different special control methods in Forge
# Copyright Forge 2024


import collections
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.patcher.unet import UnetPatcher

import torch

from backend import memory_management, utils
from backend.args import args, dynamic_args
from backend.operations import cleanup_cache
from backend.sampling.condition import (
    Condition,
    compile_conditions,
    compile_weighted_conditions,
)


def _strength_is_one(strength) -> bool:
    try:
        return math.isclose(float(strength), 1.0)
    except Exception:
        return False


CondObj = collections.namedtuple("CondObj", ["input_x", "mult", "conditioning", "area", "control", "patches"])
_low_vram_warning_shown = False


def get_area_and_mult(conds, x_in, timestep_in):
    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    if "timestep_start" in conds:
        timestep_start = conds["timestep_start"]
        if timestep_in[0] > timestep_start:
            return None
    if "timestep_end" in conds:
        timestep_end = conds["timestep_end"]
        if timestep_in[0] < timestep_end:
            return None
    if "area" in conds:
        area = conds["area"]
    if "strength" in conds:
        strength = conds["strength"]

    input_x = x_in[:, :, area[2] : area[0] + area[2], area[3] : area[1] + area[3]]

    # [NEO] When the condition covers the whole latent, has no mask and uses the default
    # strength, the multiplier is provably `torch.ones_like(input_x)` (the feathering below
    # is a no-op for full-area conditions).  Signalling that with `mult=None` lets the
    # batching code skip six latent-sized allocations and ~8 elementwise kernels per step.
    if "mask" in conds:
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds["mask"]
        assert mask.shape[1] == x_in.shape[2]
        assert mask.shape[2] == x_in.shape[3]
        mask = mask[:, area[2] : area[0] + area[2], area[3] : area[1] + area[3]] * mask_strength
        mask = mask.unsqueeze(1)
        batch_repeats = input_x.shape[0] // mask.shape[0]
        if batch_repeats > 1:
            # One channel is enough: every latent channel shares the same mask,
            # so the four identical copies were pure allocation and bandwidth.
            mask = mask.repeat(batch_repeats, 1, 1, 1)
        mult = mask * strength
    elif (
        _strength_is_one(strength)
        and area[2] == 0
        and area[3] == 0
        and area[0] == x_in.shape[2]
        and area[1] == x_in.shape[3]
    ):
        # [NEO] Fast path: the multiplier is a tensor of ones (the feathering below is a
        # no-op for full-area conditions), so there is nothing to allocate or to apply.
        mult = None
    else:
        # Same single-channel trick as above; broadcasting handles the rest.
        mult = torch.ones_like(input_x[:, :1]) * strength

    if mult is not None and "mask" not in conds:
        rr = 8
        if area[2] != 0:
            for t in range(rr):
                mult[:, :, t : 1 + t, :] *= (1.0 / rr) * (t + 1)
        if (area[0] + area[2]) < x_in.shape[2]:
            for t in range(rr):
                mult[:, :, area[0] - 1 - t : area[0] - t, :] *= (1.0 / rr) * (t + 1)
        if area[3] != 0:
            for t in range(rr):
                mult[:, :, :, t : 1 + t] *= (1.0 / rr) * (t + 1)
        if (area[1] + area[3]) < x_in.shape[3]:
            for t in range(rr):
                mult[:, :, :, area[1] - 1 - t : area[1] - t] *= (1.0 / rr) * (t + 1)

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)

    control = conds.get("control", None)

    patches = None
    return CondObj(input_x, mult, conditioning, area, control, patches)


def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True


def can_concat_cond(c1, c2):
    if c1.input_x.shape != c2.input_x.shape:
        return False
    if c1.control is not c2.control:
        return False
    if c1.patches is not c2.patches:
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)


def cond_cat(c_list):
    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out


FAST_SAMPLING_PATH = True
"""[NEO] Skip the per-step mask accumulators when a single condition (+ a single
unconditional condition) covers the whole latent without a mask.  Mirrors the
"Fast path for the common single-condition case" setting; see modules/neo_tuning.py."""


_COND_MARK_CACHE = {}
_COND_INDICES_CACHE = {}
_COND_CACHE_LIMIT = 128
"""[NEO] `compute_cond_mark` / `compute_cond_indices` only depend on the pattern of
cond/uncond entries and on the length of the sigma tensor, so they are identical for
every step of every generation with the same shape.  They used to be rebuilt - and the
result copied from *pageable CPU memory to the GPU with a blocking copy* - on every
single sampling step, which forced a full device synchronisation per step.
Memoising them removes that synchronisation entirely."""


def _cond_cache_key(cond_or_uncond, sigmas):
    return (tuple(cond_or_uncond), int(sigmas.shape[0]), sigmas.device, sigmas.dtype)


def compute_cond_mark(cond_or_uncond, sigmas):
    key = _cond_cache_key(cond_or_uncond, sigmas)

    cached = _COND_MARK_CACHE.get(key)
    if cached is not None:
        return cached

    cond_or_uncond_size = int(sigmas.shape[0])

    cond_mark = []
    for cx in cond_or_uncond:
        cond_mark += [cx] * cond_or_uncond_size

    cond_mark = sigmas.new_tensor(cond_mark)

    if len(_COND_MARK_CACHE) >= _COND_CACHE_LIMIT:
        _COND_MARK_CACHE.clear()
    _COND_MARK_CACHE[key] = cond_mark
    return cond_mark


def compute_cond_indices(cond_or_uncond, sigmas):
    key = _cond_cache_key(cond_or_uncond, sigmas)

    cached = _COND_INDICES_CACHE.get(key)
    if cached is not None:
        return cached

    cl = int(sigmas.shape[0])

    cond_indices = []
    uncond_indices = []
    for i, cx in enumerate(cond_or_uncond):
        if cx == 0:
            cond_indices += list(range(i * cl, (i + 1) * cl))
        else:
            uncond_indices += list(range(i * cl, (i + 1) * cl))

    result = (cond_indices, uncond_indices)

    if len(_COND_INDICES_CACHE) >= _COND_CACHE_LIMIT:
        _COND_INDICES_CACHE.clear()
    _COND_INDICES_CACHE[key] = result
    return result


def clear_cond_mark_cache():
    _COND_MARK_CACHE.clear()
    _COND_INDICES_CACHE.clear()
    return


def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
    global _low_vram_warning_shown

    out_cond = torch.zeros_like(x_in)
    out_count = torch.full_like(x_in, 1e-37)

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.full_like(x_in, 1e-37)

    COND = 0
    UNCOND = 1

    to_run = []
    for x in cond:
        p = get_area_and_mult(x, x_in, timestep)
        if p is None:
            continue

        to_run += [(p, COND)]
    if uncond is not None:
        for x in uncond:
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            to_run += [(p, UNCOND)]

    # [NEO] The overwhelmingly common case is "one condition + one unconditional
    # condition, both covering the whole latent with no mask".  In that case the mask
    # accumulators below degenerate into a copy, so we can skip allocating (and doing
    # arithmetic on) four extra latent-sized tensors on every sampling step.
    simple = (
        FAST_SAMPLING_PATH
        and len(to_run) <= 2
        and sum(1 for _, flag in to_run if flag == COND) == 1
        and all(p.mult is None for p, _ in to_run)
    )

    if simple:
        out_cond = None
        out_uncond = None
    else:
        out_cond = torch.zeros_like(x_in)
        out_count = torch.ones_like(x_in) * 1e-37

        out_uncond = torch.zeros_like(x_in)
        out_uncond_count = torch.ones_like(x_in) * 1e-37

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        if memory_management.signal_empty_cache:
            memory_management.soft_empty_cache()

        free_memory = memory_management.get_free_memory(x_in.device)

        if not _low_vram_warning_shown and not args.disable_gpu_warning and x_in.device.type == "cuda":
            free_memory_mb = free_memory / (1024.0 * 1024.0)
            safe_memory_mb = 1536.0
            if free_memory_mb < safe_memory_mb:
                _low_vram_warning_shown = True
                print("\n\n----------------------")
                print(f"[Low GPU VRAM Warning] Free memory is {free_memory_mb:.2f} MB, below the safe value of {safe_memory_mb:.2f} MB.")
                print("[Low GPU VRAM Warning] Performance may degrade severely if the driver starts using shared memory.")
                print("[Low GPU VRAM Warning] Lower 'GPU Weights' to leave more memory available for inference.")
                print("[Low GPU VRAM Warning] Add --disable-gpu-warning to the command line flags to silence this message.")
                print("----------------------\n\n")

        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[: len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = input_x[0] if batch_chunks == 1 else torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = timestep if batch_chunks == 1 else torch.cat([timestep] * batch_chunks)

        transformer_options = {}
        if "transformer_options" in model_options:
            transformer_options = model_options["transformer_options"].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        transformer_options["cond_mark"] = compute_cond_mark(cond_or_uncond=cond_or_uncond, sigmas=timestep)
        transformer_options["cond_indices"], transformer_options["uncond_indices"] = compute_cond_indices(cond_or_uncond=cond_or_uncond, sigmas=timestep)

        c["transformer_options"] = transformer_options

        if control is not None:
            p = control
            while p is not None:
                p.transformer_options = transformer_options
                p = p.previous_controlnet
            control_cond = c.copy()  # get_control may change items in this dict, so we need to copy it
            c["control"] = control.get_control(input_x, timestep_, control_cond, len(cond_or_uncond))
            c["control_model"] = control

        if "model_function_wrapper" in model_options:
            output = model_options["model_function_wrapper"](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
        del input_x

        if simple:
            for o in range(batch_chunks):
                if cond_or_uncond[o] == COND:
                    out_cond = output[o]
                else:
                    out_uncond = output[o]
        else:
            for o in range(batch_chunks):
                if cond_or_uncond[o] == COND:
                    out_cond[:, :, area[o][2] : area[o][0] + area[o][2], area[o][3] : area[o][1] + area[o][3]] += output[o] * mult[o]
                    out_count[:, :, area[o][2] : area[o][0] + area[o][2], area[o][3] : area[o][1] + area[o][3]] += mult[o]
                else:
                    out_uncond[:, :, area[o][2] : area[o][0] + area[o][2], area[o][3] : area[o][1] + area[o][3]] += output[o] * mult[o]
                    out_uncond_count[:, :, area[o][2] : area[o][0] + area[o][2], area[o][3] : area[o][1] + area[o][3]] += mult[o]
        del mult

    if simple:
        # Keep the historic contract: a condition that was filtered out (or that was
        # never requested) is reported as zeros rather than as `None`.
        if out_cond is None:
            out_cond = torch.zeros_like(x_in)
        if out_uncond is None:
            out_uncond = torch.zeros_like(x_in)
        return out_cond, out_uncond

    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond


def sampling_function_inner(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None, return_full=False):
    edit_strength = sum((item["strength"] if "strength" in item else 1) for item in cond)

    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    for fn in model_options.get("sampler_pre_cfg_function", []):
        model, cond, uncond_, x, timestep, model_options = fn(model, cond, uncond_, x, timestep, model_options)

    cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)

    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep, "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    elif not math.isclose(edit_strength, 1.0):
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "cond_scale": cond_scale, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred, "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    if return_full:
        return cfg_result, cond_pred, uncond_pred

    return cfg_result


def sampling_function(self, denoiser_params, cond_scale, cond_composition, extra_model_options=None):
    unet_patcher = self.inner_model.inner_model.forge_objects.unet
    model = unet_patcher.model
    control = unet_patcher.controlnet_linked_list
    extra_concat_condition = unet_patcher.extra_concat_condition
    x = denoiser_params.x
    timestep = denoiser_params.sigma
    uncond = compile_conditions(denoiser_params.text_uncond)
    cond = compile_weighted_conditions(denoiser_params.text_cond, cond_composition)
    model_options = (unet_patcher.model_options or {}).copy()
    model_options.update(extra_model_options or {})
    seed = self.p.seeds[0]

    if extra_concat_condition is not None:
        image_cond_in = extra_concat_condition
    else:
        image_cond_in = denoiser_params.image_cond

    if isinstance(image_cond_in, torch.Tensor):
        if image_cond_in.shape[0] == x.shape[0] and image_cond_in.shape[2] == x.shape[2] and image_cond_in.shape[3] == x.shape[3]:
            if uncond is not None:
                for i in range(len(uncond)):
                    uncond[i]["model_conds"]["c_concat"] = Condition(image_cond_in)
            for i in range(len(cond)):
                cond[i]["model_conds"]["c_concat"] = Condition(image_cond_in)

    if control is not None:
        for h in cond:
            h["control"] = control
        if uncond is not None:
            for h in uncond:
                h["control"] = control

    for modifier in model_options.get("conditioning_modifiers", []):
        model, x, timestep, uncond, cond, cond_scale, model_options, seed = modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

    denoised, cond_pred, uncond_pred = sampling_function_inner(model, x, timestep, uncond, cond, cond_scale, model_options, seed, return_full=True)
    return denoised, cond_pred, uncond_pred


def sampling_prepare(unet: "UnetPatcher", x: torch.Tensor, *, is_img2img: bool = False):
    global _low_vram_warning_shown
    _low_vram_warning_shown = False

    if is_img2img and dynamic_args.get("kontext", False):
        unet.set_transformer_option("ref_latents", [x.detach().clone()])
    else:
        unet.set_transformer_option("ref_latents", None)

    shape = list(x.shape)
    mem_shape = [2 * shape[0]] + shape[1:]

    unet_inference_memory = unet.memory_required(mem_shape)
    additional_inference_memory = unet.extra_preserved_memory_during_sampling
    additional_model_patchers = unet.extra_model_patchers_during_sampling

    if unet.controlnet_linked_list is not None:
        additional_inference_memory += unet.controlnet_linked_list.inference_memory_requirements(unet.model_dtype())
        additional_model_patchers += unet.controlnet_linked_list.get_models()

    if unet.has_online_lora():
        lora_memory = utils.nested_compute_size(unet.lora_patches, element_size=utils.dtype_to_element_size(unet.model.computation_dtype))
        additional_inference_memory += lora_memory

    memory_management.load_models_gpu(models=[unet] + additional_model_patchers, memory_required=unet_inference_memory, hard_memory_preservation=additional_inference_memory)

    if unet.has_online_lora():
        utils.nested_move_to_device(unet.lora_patches, device=unet.current_device, dtype=unet.model.computation_dtype)

    real_model = unet.model

    percent_to_timestep_function = lambda p: real_model.predictor.percent_to_sigma(p)

    for cnet in unet.list_controlnets():
        cnet.pre_run(real_model, percent_to_timestep_function)

    return


def sampling_cleanup(unet):
    if unet.has_online_lora():
        utils.nested_move_to_device(unet.lora_patches, device=unet.offload_device)
    for cnet in unet.list_controlnets():
        cnet.cleanup()
    cleanup_cache()
    return
