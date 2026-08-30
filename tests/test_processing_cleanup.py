import ast
import types
import unittest
from pathlib import Path


class _Profiler:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Options:
    data = {}

    @staticmethod
    def get_default(_key):
        return None


class ProcessingCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load process_images alone so this test needs no Torch, PIL, or Gradio."""
        source_path = Path(__file__).parents[1] / "modules" / "processing.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        function_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_images"
        )

        cls.events = []
        cls.raise_while_restoring = False

        def set_config(_values, **kwargs):
            if kwargs.get("save_config") is False and cls.raise_while_restoring:
                raise RuntimeError("restore failed")

        namespace = {
            "StableDiffusionProcessing": object,
            "Processed": object,
            "opts": _Options(),
            "sd_models": types.SimpleNamespace(checkpoint_aliases={}),
            "set_config": set_config,
            "manage_model_and_prompt_cache": lambda _p: None,
            "sd_samplers": types.SimpleNamespace(
                fix_p_invalid_sampler_and_scheduler=lambda _p: None
            ),
            "profiling": types.SimpleNamespace(Profiler=_Profiler),
            "process_images_inner": lambda _p: (_ for _ in ()).throw(
                RuntimeError("generation failed")
            ),
            "extra_networks": types.SimpleNamespace(
                deactivate=lambda _p, _data: cls.events.append("deactivate")
            ),
            "devices": types.SimpleNamespace(
                torch_gc=lambda: cls.events.append("torch_gc")
            ),
        }
        module = ast.fix_missing_locations(ast.Module(body=[function_node], type_ignores=[]))
        exec(compile(module, str(source_path), "exec"), namespace)
        cls.process_images = staticmethod(namespace["process_images"])

    def setUp(self):
        self.events.clear()
        type(self).raise_while_restoring = False
        self.processing = types.SimpleNamespace(
            scripts=None,
            override_settings={},
            override_settings_restore_afterwards=True,
            disable_extra_networks=False,
            extra_network_data={"lora": [object()]},
        )

    def test_generation_failure_deactivates_networks_and_releases_cache(self):
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.process_images(self.processing)

        self.assertEqual(self.events, ["deactivate", "torch_gc"])

    def test_restore_failure_still_deactivates_networks_and_releases_cache(self):
        type(self).raise_while_restoring = True

        with self.assertRaisesRegex(RuntimeError, "restore failed"):
            self.process_images(self.processing)

        self.assertEqual(self.events, ["deactivate", "torch_gc"])


if __name__ == "__main__":
    unittest.main()
