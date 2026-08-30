import ast
import unittest
from functools import wraps
from pathlib import Path


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _State:
    def __init__(self):
        self.end_calls = 0

    def begin(self, job=None):
        pass

    def end(self):
        self.end_calls += 1


class _Shared:
    state = _State()


class _Progress:
    def add_task_to_queue(self, _task):
        pass

    def start_task(self, _task):
        pass

    def record_results(self, *_args):
        pass

    def finish_task(self, _task):
        pass


class CallQueueCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load only the wrapper under test, without importing optional WebUI dependencies."""
        source_path = Path(__file__).parents[1] / "modules" / "call_queue.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        wrapper_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "wrap_gradio_gpu_call"
        )

        cls.shared = _Shared()
        cls.progress = _Progress()
        namespace = {
            "wraps": wraps,
            "queue_lock": _Lock(),
            "shared": cls.shared,
            "progress": cls.progress,
            "wrap_gradio_call": lambda function, **_kwargs: function,
        }
        module = ast.fix_missing_locations(ast.Module(body=[wrapper_node], type_ignores=[]))
        exec(compile(module, str(source_path), "exec"), namespace)
        cls.wrap_gpu_call = staticmethod(namespace["wrap_gradio_gpu_call"])

    def setUp(self):
        self.shared.state.end_calls = 0
        self.progress.finish_task = lambda _task: None

    def test_state_ends_when_generation_raises(self):
        def fail():
            raise RuntimeError("generation failed")

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.wrap_gpu_call(fail)()

        self.assertEqual(self.shared.state.end_calls, 1)

    def test_state_ends_when_progress_finalization_raises(self):
        def fail_finish(_task):
            raise RuntimeError("progress failed")

        self.progress.finish_task = fail_finish

        with self.assertRaisesRegex(RuntimeError, "progress failed"):
            self.wrap_gpu_call(lambda: "result")()

        self.assertEqual(self.shared.state.end_calls, 1)


if __name__ == "__main__":
    unittest.main()
