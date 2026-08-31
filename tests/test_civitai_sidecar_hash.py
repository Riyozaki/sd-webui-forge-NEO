"""Tests for the Civitai sidecar reader in extensions-builtin/sd_forge_lora/network.py.

``civitai_auto_v2_hash`` lets a downloaded model skip hashing: the AutoV2 value
Civitai publishes is the same one the built-in hash would compute, so reading it
from the sidecar saves reading gigabytes at every scan.  The function is loaded
from source so that importing the extension (and with it the whole webui) is not
necessary.
"""

import ast
import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
SOURCE_PATH = REPOSITORY_ROOT / "extensions-builtin" / "sd_forge_lora" / "network.py"


def _load_function():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))

    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "civitai_auto_v2_hash"]
    assert len(definitions) == 1, "civitai_auto_v2_hash is gone from network.py"

    # the compiled hash pattern lives next to it at module level
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "_hash_pattern" for target in node.targets)
    ]
    assert len(constants) == 1, "_hash_pattern is gone from network.py"

    namespace = {"os": os, "json": json, "re": re}
    module = ast.fix_missing_locations(ast.Module(body=constants + definitions, type_ignores=[]))
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["civitai_auto_v2_hash"]


civitai_auto_v2_hash = _load_function()

HASH = "a" * 64


class CivitaiSidecarHashTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.weight = self.root / "example.safetensors"
        self.weight.write_bytes(b"weights")
        self.sidecar = self.root / "example.json"
        self.addCleanup(self.directory.cleanup)

    def _write_sidecar(self, payload):
        self.sidecar.write_text(json.dumps(payload), encoding="utf-8")

    def test_reads_the_published_autov2_hash(self):
        self._write_sidecar({"civitai": {"hashes": {"AutoV2": HASH, "SHA256": "b" * 64}}})

        self.assertEqual(civitai_auto_v2_hash(str(self.weight)), HASH)

    def test_accepts_the_uppercase_spelling(self):
        self._write_sidecar({"civitai": {"hashes": {"AUTOV2": HASH}}})

        self.assertEqual(civitai_auto_v2_hash(str(self.weight)), HASH)

    def test_ignores_a_sidecar_older_than_the_weights(self):
        self._write_sidecar({"civitai": {"hashes": {"AutoV2": HASH}}})
        older = time.time() - 3600
        os.utime(self.sidecar, (older, older))
        os.utime(self.weight, (time.time(), time.time()))

        self.assertIsNone(civitai_auto_v2_hash(str(self.weight)), "the file may have been replaced")

    def test_returns_none_without_a_sidecar(self):
        self.assertIsNone(civitai_auto_v2_hash(str(self.weight)))

    def test_returns_none_for_a_broken_or_incomplete_sidecar(self):
        for payload in ("not json", {}, {"civitai": None}, {"civitai": {"hashes": {}}}, {"civitai": {"hashes": {"AutoV2": "nope"}}}):
            with self.subTest(payload=payload):
                self.sidecar.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
                self.assertIsNone(civitai_auto_v2_hash(str(self.weight)))


if __name__ == "__main__":
    unittest.main()
