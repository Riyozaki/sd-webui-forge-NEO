"""[NEO] Optional UNet acceleration: TorchInductor compilation and CUDA graph replay.

Two independent, opt-in speedups for the denoiser, both off by default.

``torch.compile``
    Compiles the UNet forward with TorchInductor.  Dynamo is configured with
    ``suppress_errors = True``, so *any* problem inside the compiler falls back
    to the ordinary eager implementation instead of breaking a generation.
    Compilation happens on the first UNet call rather than at load time, which
    keeps startup fast but makes the first generation slow.

CUDA graphs
    Records one UNet call and replays it for every following step.  Sampling
    calls the denoiser with identical shapes each step, so the launch sequence
    of hundreds of kernels per step collapses into a single replay.  Two things
    make it safe enough to ship:

    * every input is copied into a static buffer before the replay, so nothing
      is baked in except the launch sequence itself;
    * the recording is compared against the eager result before it is used.  If
      the replayed output differs by more than ``GRAPH_RELATIVE_TOLERANCE`` of
      the eager peak, graph replay is switched off for good.

    Recording is skipped when a ControlNet is active (its tensors travel inside
    ``transformer_options`` and change every step), when gradients are enabled,
    or when anything else is already capturing.

Neither is a good default on an 8 GB card: compilation costs minutes on the
first generation and a captured graph keeps a private memory pool reserved.
"""

from __future__ import annotations

import threading

import torch

from modules import shared

GRAPH_RELATIVE_TOLERANCE = 0.02
"""How far a replay may deviate from the eager result before it is rejected.

fp16 rounding differences between two orders of execution land around 1e-3 of
the peak value.  A stale pointer or a frozen input shows up as a relative error
of 10% or more, so 2% separates the two cases with room to spare.
"""

_WARMUP_RUNS = 3
"""Forwards to run before recording, so cuDNN has finished auto-tuning."""

_MAX_RECORDINGS_PER_PASS = 4
"""Upper bound on recordings for one sampling pass.

Recording costs a few forwards, so a model whose state keeps changing (and
which would therefore re-record on every step) is better off without graphs
at all than with a recording per step.
"""

_SAFE_TENSOR_KEYS = frozenset({"sigmas", "cond_mark"})
"""Tensors in ``transformer_options`` that may change between calls.

They are only read by ControlNet, which disables recording anyway, so their
presence must not by itself make a model ineligible.
"""


def _opts(name, default=None):
    try:
        return getattr(shared.opts, name, default)
    except Exception:
        return default


def _command_line_disabled(flag: str) -> bool:
    """Escape hatch for when the UI is unreachable because of the option."""

    try:
        from backend.args import args

        return bool(getattr(args, flag, False))
    except Exception:
        return False


def _cuda_ready() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    except Exception:
        return False


class _CapturedGraph:
    """One recorded UNet call plus the static buffers it reads from."""

    def __init__(self, key, graph, owner, statics, extra_statics, output):
        self.key = key
        self.graph = graph
        self.owner = owner  # keeps the model alive so id() cannot be recycled
        self.statics = statics
        self.extra_statics = extra_statics
        self.output = output
        self.replays = 0

    def replay(self, xc, t, context, extra_conds):
        static_xc, static_t, static_context = self.statics
        static_xc.copy_(xc)
        static_t.copy_(t)
        static_context.copy_(context)
        for name, buffer in self.extra_statics.items():
            buffer.copy_(extra_conds[name])

        self.graph.replay()
        self.replays += 1

        # The buffer is overwritten by the next replay, and TeaCache keeps the
        # result of a step around to reuse - hand out a copy instead.
        return self.output.clone()


class UNetAccelerator:
    """Owns the compiled UNets and the single active CUDA graph."""

    def __init__(self):
        self._lock = threading.RLock()

        self._compiled = {}  # id(unet) -> (unet, original forward)
        self._compiled_failed = False

        self._graph = None
        self._graph_disabled_reason = None

        self.captures = 0
        self.replays = 0
        self.calls = 0

    # ------------------------------------------------------------------ options

    @property
    def cuda_graph_requested(self) -> bool:
        if not bool(_opts("neo_cuda_graph", False)) or self._graph_disabled_reason is not None:
            return False
        return not _command_line_disabled("neo_no_cuda_graph")

    @property
    def compile_mode(self) -> str:
        if _command_line_disabled("neo_no_compile"):
            return "disabled"
        return _opts("neo_unet_compile", "disabled") or "disabled"

    # --------------------------------------------------------------------- API

    def invalidate(self, reason: str = "") -> None:
        """Drop the captured graph and restore every compiled forward."""

        with self._lock:
            self._release_graph()

            for unet, original in list(self._compiled.values()):
                try:
                    object.__setattr__(unet, "forward", original)
                except Exception:
                    pass
            self._compiled.clear()
            self._compiled_failed = False

        if reason:
            print(f"[NEO] UNet acceleration reset ({reason})")

    def run(self, unet, xc, t, context, control, transformer_options, extra_conds):
        """Call ``unet``, using the compiled forward and a graph replay if possible."""

        call = self._resolve(unet)
        self.calls += 1

        if self.cuda_graph_requested:
            out = self._replay_or_record(call, unet, xc, t, context, control, transformer_options, extra_conds)
            if out is not None:
                return out
        elif self._graph is not None:
            # The setting was turned off while a recording was still alive.
            self._release_graph()

        return call(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)

    def _replay_or_record(self, call, unet, xc, t, context, control, transformer_options, extra_conds):
        """Replay the current recording, or make a new one; ``None`` means "run eager"."""

        key = self._graph_key(unet, xc, t, context, control, transformer_options, extra_conds)
        if key is None:
            return None

        graph = self._graph

        if graph is None or graph.key != key:
            if graph is not None:
                # Shapes or weights changed: hand the old pool back before
                # reserving another one.
                self._release_graph()

            graph = self._capture(call, key, unet, xc, t, context, control, transformer_options, extra_conds)
            if graph is None:
                return None
            self._graph = graph

        return self._replay(graph, xc, t, context, extra_conds)

    def release_graphs(self, reason: str = "") -> None:
        """Free the recording, keeping any compiled forwards.

        Called when something else runs out of memory: the pool a recording
        owns is invisible to the allocator, so dropping it is the cheapest way
        to give several hundred megabytes back.
        """

        with self._lock:
            if self._graph is None:
                return
            self._release_graph()
            if reason:
                print(f"[NEO] dropped the CUDA graph recording ({reason})")

    def begin_sampling(self) -> None:
        """Start a new report; called once per sampling pass."""

        self.captures = 0
        self.replays = 0
        self.calls = 0

    def summary(self) -> str:
        """One line for the console, or an empty string when nothing was used."""

        parts = []
        if self.replays:
            parts.append(f"replayed {self.replays}/{self.calls} UNet calls")
        if self.captures:
            parts.append(f"{self.captures} recording(s)")
        if self._compiled:
            parts.append("compiled with TorchInductor")
        if self._graph_disabled_reason:
            parts.append(f"cuda graphs off: {self._graph_disabled_reason}")
        return ", ".join(parts)

    # ----------------------------------------------------------------- compile

    def _resolve(self, unet):
        """Install (once) the compiled forward and return the object to call."""

        mode = self.compile_mode
        key = id(unet)

        if mode == "disabled" or self._compiled_failed:
            if key in self._compiled:
                self._restore_compiled(key)
            return unet

        with self._lock:
            if key in self._compiled:
                return unet

            try:
                import torch._dynamo

                # Without this, a single unsupported op aborts the generation
                # instead of quietly falling back to eager.
                torch._dynamo.config.suppress_errors = True

                original = unet.forward
                compiled = torch.compile(original, backend="inductor", mode="default", dynamic=(mode == "dynamic shapes"))

                # object.__setattr__ keeps the wrapper out of _modules, so the
                # model tree that the memory manager walks stays unchanged.
                object.__setattr__(unet, "forward", compiled)
                self._compiled[key] = (unet, original)
                print("[NEO] UNet handed to TorchInductor - the first generation will take a few minutes.")
            except Exception as error:
                self._compiled_failed = True
                print(f"[NEO] TorchInductor unavailable ({type(error).__name__}: {error}); using the ordinary UNet.")

        return unet

    def _restore_compiled(self, key):
        with self._lock:
            entry = self._compiled.pop(key, None)
            if entry is None:
                return
            unet, original = entry
            try:
                object.__setattr__(unet, "forward", original)
            except Exception:
                pass

    # -------------------------------------------------------------- cuda graph

    def _graph_key(self, unet, xc, t, context, control, transformer_options, extra_conds):
        """Everything a recording depends on, or ``None`` when it cannot be recorded."""

        if not _cuda_ready():
            return None
        if control is not None:
            return None
        if torch.is_grad_enabled():
            return None
        if not (xc.is_cuda and t.is_cuda and context.is_cuda and xc.device == t.device == context.device):
            return None

        try:
            if torch.cuda.is_current_stream_capturing():
                return None
            stream = torch.cuda.current_stream().cuda_stream
        except Exception:
            return None

        for value in extra_conds.values():
            if not torch.is_tensor(value) or not value.is_cuda or value.device != xc.device:
                return None

        for name, value in transformer_options.items():
            if torch.is_tensor(value) and name not in _SAFE_TENSOR_KEYS:
                # An extension is feeding the UNet something that changes every
                # step; freezing that inside a graph would silently be wrong.
                return None

        fingerprint = self._weight_fingerprint(unet)
        if fingerprint is None:
            return None

        return (
            id(unet),
            fingerprint,
            stream,
            tuple(xc.shape),
            str(xc.dtype),
            tuple(t.shape),
            str(t.dtype),
            tuple(context.shape),
            str(context.dtype),
            tuple(sorted((name, tuple(value.shape), str(value.dtype)) for name, value in extra_conds.items())),
            tuple(sorted(str(name) for name in transformer_options)),
        )

    def _weight_fingerprint(self, unet):
        """Cheap signature of the weights, so a LoRA merge invalidates the graph.

        Forge applies LoRAs by replacing parameter objects, so a recording made
        before a merge would keep pointing at the old (possibly recycled)
        storage.  Summing the addresses is far cheaper than hashing the weights
        and catches every replacement.
        """

        try:
            parameters = list(unet.parameters())
        except Exception:
            return None

        if not parameters:
            # Quantised models keep their weights outside of parameters; there
            # is nothing here that would tell us about a weight change.
            return None

        try:
            return sum(parameter.data_ptr() for parameter in parameters) & 0xFFFFFFFF
        except Exception:
            return None

    def _capture(self, call, key, unet, xc, t, context, control, transformer_options, extra_conds):
        with self._lock:
            if self._graph_disabled_reason is not None:
                return None

            if self.captures >= _MAX_RECORDINGS_PER_PASS:
                self._disable_graphs("the recording had to be remade on every step")
                return None

            def forward(x, timestep, cond, extras):
                return call(x, timestep, context=cond, control=control, transformer_options=transformer_options, **extras)

            try:
                torch.cuda.synchronize()

                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    for _ in range(_WARMUP_RUNS):
                        reference = forward(xc, t, context, extra_conds)
                torch.cuda.current_stream().wait_stream(warmup_stream)
                torch.cuda.synchronize()

                reference = reference.detach().clone()

                static_xc = xc.detach().clone()
                static_t = t.detach().clone()
                static_context = context.detach().clone()
                static_extra = {name: value.detach().clone() for name, value in extra_conds.items()}

                static_xc.copy_(xc)
                static_t.copy_(t)
                static_context.copy_(context)
                for name, buffer in static_extra.items():
                    buffer.copy_(extra_conds[name])

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_output = forward(static_xc, static_t, static_context, static_extra)

                self.captures += 1
                captured = _CapturedGraph(key, graph, unet, (static_xc, static_t, static_context), static_extra, static_output)

                torch.cuda.synchronize()
                if not self._verify(reference, static_output):
                    self._disable_graphs("replayed result differs from the eager one")
                    return None

                print(
                    "[NEO] UNet recorded for CUDA graph replay "
                    f"({tuple(xc.shape)}, {str(xc.dtype).replace('torch.', '')})"
                )
                return captured
            except torch.cuda.OutOfMemoryError:
                self._disable_graphs("not enough VRAM to record the UNet")
                return None
            except Exception as error:
                self._disable_graphs(f"{type(error).__name__}: {error}")
                return None

    def _verify(self, reference, candidate) -> bool:
        """Compare the recording against the eager result before trusting it."""

        try:
            reference = reference.detach().float()
            candidate = candidate.detach().float()
            peak = float(reference.abs().max().item())
            worst = float((reference - candidate).abs().max().item())
        except Exception:
            return False

        if worst <= max(peak, 1.0) * GRAPH_RELATIVE_TOLERANCE:
            return True

        print(f"[NEO] CUDA graph check failed: worst difference {worst:.4f} against a peak of {peak:.4f}")
        return False

    def _replay(self, graph, xc, t, context, extra_conds):
        try:
            self.replays += 1
            return graph.replay(xc, t, context, extra_conds)
        except Exception as error:
            self._disable_graphs(f"{type(error).__name__}: {error}")
            return None

    def _disable_graphs(self, reason: str) -> None:
        self._graph_disabled_reason = reason
        self._release_graph()
        print(f"[NEO] CUDA graph replay disabled ({reason}).")

    def _release_graph(self) -> None:
        graph, self._graph = self._graph, None
        if graph is None:
            return
        try:
            del graph
        except Exception:
            pass
        try:
            from backend import memory_management

            # The recording owns a private pool; hand the memory back so the
            # rest of the generation sees an accurate free-memory figure.
            memory_management.soft_empty_cache(True)
        except Exception:
            pass


accelerator = UNetAccelerator()
"""Module level singleton; ``KModel`` calls into it for every denoiser call."""
