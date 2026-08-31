"""Behaviour tests for modules/neo_compile.py.

``neo_compile`` talks to ``torch.cuda`` directly, which needs a GPU.  These
tests install a stand-in ``torch`` module first, so the logic - when a
recording is allowed, how it is verified, and how it is torn down - can be
checked on any machine.
"""

import contextlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
SOURCE_PATH = REPOSITORY_ROOT / "modules" / "neo_compile.py"


# --------------------------------------------------------------------------- stub


class _StubTensor:
    def __init__(self, data, dtype=None, device=None, shape=None):
        self._data = list(data)
        self.dtype = dtype
        self.device = device
        self.shape = shape or (len(self._data),)

    def data_ptr(self):
        return self._ptr

    @property
    def _ptr(self):
        return id(self)

    def is_cuda(self):
        return True

    def detach(self):
        return self

    def clone(self):
        clone = _StubTensor(self._data, self.dtype, self.device, self.shape)
        clone._marker = getattr(self, "_marker", None)
        return clone

    def copy_(self, other):
        self._data = list(other._data)
        return self

    def abs(self):
        return _StubTensor([abs(v) for v in self._data], self.dtype, self.device, self.shape)

    def max(self):
        return _StubTensor([max(self._data)], self.dtype, self.device, (1,))

    def float(self):
        return _StubTensor([float(v) for v in self._data], "float32", self.device, self.shape)

    def item(self):
        return self._data[0]

    def __sub__(self, other):
        return _StubTensor([a - b for a, b in zip(self._data, other._data)], self.dtype, self.device, self.shape)


class _StubStream:
    def __init__(self, ident):
        self.cuda_stream = ident

    def wait_stream(self, _other):
        return None


class _StubGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class _StubCuda:
    def __init__(self):
        self.capturing = False
        self.streams = 0
        self.OutOfMemoryError = _StubTorch.OutOfMemoryError

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def is_current_stream_capturing(self):
        return self.capturing

    def current_stream(self):
        return _StubStream(1)

    def Stream(self):
        self.streams += 1
        return _StubStream(100 + self.streams)

    def synchronize(self):
        return None

    @contextlib.contextmanager
    def stream(self, stream):
        yield stream

    @contextlib.contextmanager
    def graph(self, graph):
        self.capturing = True
        try:
            yield graph
        finally:
            self.capturing = False

    @staticmethod
    def CUDAGraph():
        return _StubGraph()


class _StubTorch(types.ModuleType):
    class OutOfMemoryError(Exception):
        pass

    def __init__(self):
        super().__init__("torch")
        self.float16 = "float16"
        self.float32 = "float32"
        self.Tensor = _StubTensor
        self.cuda = _StubCuda()
        self.grad_enabled = False
        self.compiled = []
        self._dynamo = types.SimpleNamespace(config=types.SimpleNamespace(suppress_errors=False))

    @staticmethod
    def is_tensor(value):
        return isinstance(value, _StubTensor)

    def is_grad_enabled(self):
        return self.grad_enabled

    def compile(self, function, **kwargs):
        self.compiled.append((function, kwargs))

        def wrapper(*args, **kw):
            return function(*args, **kw)

        return wrapper


def _install_stubs():
    torch_stub = _StubTorch()
    sys.modules["torch"] = torch_stub
    sys.modules["torch._dynamo"] = torch_stub._dynamo

    shared = types.SimpleNamespace(opts=types.SimpleNamespace(neo_cuda_graph=False, neo_unet_compile="disabled"))
    modules = types.ModuleType("modules")
    modules.shared = shared
    # Point the stand-in at the real directory: without a __path__ it is not a
    # package, and any later `from modules import ...` in this process - another
    # test module, for instance - would fail with "modules is not a package".
    modules.__path__ = [str(REPOSITORY_ROOT / "modules")]
    sys.modules["modules"] = modules
    sys.modules["modules.shared"] = shared

    specification = importlib.util.spec_from_file_location("modules.neo_compile", SOURCE_PATH)
    neo_compile = importlib.util.module_from_spec(specification)
    sys.modules["modules.neo_compile"] = neo_compile
    specification.loader.exec_module(neo_compile)

    return torch_stub, shared.opts, neo_compile


TORCH, OPTS, neo_compile = _install_stubs()


# ------------------------------------------------------------------------ fixture


class _Parameter:
    def __init__(self, pointer):
        self._pointer = pointer

    def data_ptr(self):
        return self._pointer


class _FakeUNet:
    """Deterministic denoiser; perturbs its output for the recorded buffers."""

    def __init__(self, perturb_recorded=False, raise_on=None):
        self.parameters_list = [_Parameter(1000 + index) for index in range(3)]
        self.calls = 0
        self.perturb_recorded = perturb_recorded
        self.raise_on = raise_on

    def parameters(self):
        return list(self.parameters_list)

    def forward(self, x, timestep, context=None, control=None, transformer_options=None, **extra):
        self.calls += 1

        if self.raise_on is not None and self.calls == self.raise_on:
            raise TORCH.OutOfMemoryError("stub")

        values = [1.0, 2.0, 3.0, 4.0]
        if self.perturb_recorded and getattr(x, "_marker", None) == "recorded":
            values = [value + 5.0 for value in values]

        return TORCH.Tensor(values, TORCH.float16, "cuda", (1, 4))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NeoCompileTests(unittest.TestCase):
    def setUp(self):
        self.accelerator = neo_compile.UNetAccelerator()
        OPTS.neo_cuda_graph = True
        OPTS.neo_unet_compile = "disabled"
        TORCH.grad_enabled = False

    def _inputs(self, shape=(1, 4, 8, 8), marker=None):
        x = TORCH.Tensor([0.5] * 4, TORCH.float16, "cuda", shape)
        timestep = TORCH.Tensor([900.0], TORCH.float32, "cuda", (1,))
        context = TORCH.Tensor([0.25] * 6, TORCH.float16, "cuda", (1, 3, 2))
        for tensor in (x, timestep, context):
            tensor._marker = marker
        return x, timestep, context

    def _call(self, unet, shape=(1, 4, 8, 8), control=None, extra=None, transformer_options=None):
        x, timestep, context = self._inputs(shape)
        return self.accelerator.run(unet, x, timestep, context, control, transformer_options or {}, extra or {})

    def _mark_recorded_buffers(self):
        """Make ``clone()`` label its result, as the static buffers are."""

        original = TORCH.Tensor.clone

        def clone(self):
            clone = original(self)
            clone._marker = getattr(self, "_marker", None) or "recorded"
            return clone

        return original, clone

    def test_records_once_and_replays_every_step(self):
        unet = _FakeUNet()

        self._call(unet)
        self.assertEqual(self.accelerator.captures, 1)
        self.assertEqual(self.accelerator.replays, 1)
        # three warm-up forwards plus the recording itself
        self.assertEqual(unet.calls, 4)

        for _ in range(9):
            self._call(unet)

        self.assertEqual(unet.calls, 4, "a replay must not run the UNet again")
        self.assertEqual(self.accelerator.replays, 10)
        self.assertIn("replayed 10/10", self.accelerator.summary())

    def test_disabled_by_default(self):
        OPTS.neo_cuda_graph = False
        unet = _FakeUNet()

        self._call(unet)

        self.assertEqual(unet.calls, 1)
        self.assertEqual(self.accelerator.captures, 0)
        self.assertEqual(self.accelerator.summary(), "")

    def test_controlnet_and_gradients_are_not_recorded(self):
        unet = _FakeUNet()
        self._call(unet)
        captures = self.accelerator.captures

        self._call(unet, control=object())
        self.assertEqual(self.accelerator.captures, captures, "ControlNet tensors change every step")

        TORCH.grad_enabled = True
        self._call(unet)
        self.assertEqual(self.accelerator.captures, captures)

    def test_tensor_in_transformer_options_blocks_recording(self):
        unet = _FakeUNet()
        self._call(unet, transformer_options={"patches": TORCH.Tensor([1.0])})

        self.assertEqual(self.accelerator.captures, 0)

    def test_new_shape_re_records(self):
        unet = _FakeUNet()

        self._call(unet, shape=(1, 4, 8, 8))
        self._call(unet, shape=(1, 4, 16, 16))

        self.assertEqual(self.accelerator.captures, 2)

    def test_changed_weights_re_record(self):
        unet = _FakeUNet()

        self._call(unet)
        unet.parameters_list[0] = _Parameter(999_999)
        self._call(unet)

        self.assertEqual(self.accelerator.captures, 2, "a LoRA merge replaces the parameters")

    def test_diverging_recording_disables_replay(self):
        original, marker_clone = self._mark_recorded_buffers()
        TORCH.Tensor.clone = marker_clone
        try:
            unet = _FakeUNet(perturb_recorded=True)
            result = self._call(unet)
        finally:
            TORCH.Tensor.clone = original

        self.assertIsNotNone(result, "the caller still gets a result")
        self.assertIsNotNone(self.accelerator._graph_disabled_reason)
        self.assertEqual(self.accelerator.replays, 0)

    def test_out_of_memory_disables_replay(self):
        unet = _FakeUNet(raise_on=2)
        result = self._call(unet)

        self.assertIsNotNone(result)
        self.assertIn("VRAM", self.accelerator._graph_disabled_reason or "")

    def test_repeated_re_recording_gives_up(self):
        unet = _FakeUNet()

        for shape in [(1, 4, 8, 8), (1, 4, 16, 16), (1, 8, 8, 8), (2, 4, 8, 8), (1, 4, 32, 32)]:
            self._call(unet, shape=shape)

        self.assertIsNotNone(self.accelerator._graph_disabled_reason)
        self.assertLessEqual(self.accelerator.captures, neo_compile._MAX_RECORDINGS_PER_PASS)

    def test_compile_is_installed_once_and_restored(self):
        OPTS.neo_unet_compile = "static shapes"
        unet = _FakeUNet()
        original = unet.forward

        self._call(unet)
        self.assertIsNot(unet.forward, original)
        self.assertEqual(len(TORCH.compiled), 1)
        self.assertEqual(TORCH.compiled[0][1]["dynamic"], False)

        self.accelerator.invalidate()

        self.assertEqual(unet.forward, original)
        self.assertFalse(self.accelerator._compiled)

    def test_invalidate_releases_the_recording(self):
        unet = _FakeUNet()

        self._call(unet)
        self.assertIsNotNone(self.accelerator._graph)

        self.accelerator.invalidate("test")

        self.assertIsNone(self.accelerator._graph)


if __name__ == "__main__":
    unittest.main()
