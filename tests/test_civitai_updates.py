"""Tests for the "Check for updates" rendering in the Civitai browser.

The renderers are loaded from source (importing the extension would pull in
gradio), and the two things that matter are covered: the ``data-running``
marker the polling JavaScript keys off, and that nothing the site sends back
can reach the page unescaped.
"""

import ast
import html
import os
import time
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
SOURCE_PATH = REPOSITORY_ROOT / "extensions-builtin" / "sd_forge_civitai" / "scripts" / "civitai_browser.py"

WANTED = ("esc", "render_updates_status", "render_updates_list")


def _load():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in WANTED]
    names = {node.name for node in nodes}
    assert names == set(WANTED), f"missing from civitai_browser.py: {set(WANTED) - names}"

    namespace = {"html_lib": html, "os": os, "time": time, "UPDATE_STATE": {}}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace


LOADED = _load()
RENDER_STATUS = LOADED["render_updates_status"]
RENDER_LIST = LOADED["render_updates_list"]
STATE = LOADED["UPDATE_STATE"]


def reset(**overrides):
    STATE.clear()
    STATE.update(
        {
            "running": False,
            "checked": 0,
            "total": 0,
            "skipped": 0,
            "items": {},
            "error": "",
            "finished_at": 0.0,
        }
    )
    STATE.update(overrides)


def entry(name="my lora", current="v1.0", new="v2.0", extra=1):
    def version(identifier, label):
        return {"id": identifier, "name": label, "baseModel": "SDXL 1.0", "publishedAt": "2026-07-01T00:00:00Z"}

    newer = [version(200 + i, f"v2.{i}") for i in range(extra)]
    return {
        "model_id": 7,
        "model_name": name,
        "model_type": "LORA",
        "installed": version(100, current),
        "newer": newer,
        "latest": newer[0] if newer else version(100, current),
        "file": "/models/Lora/mine.safetensors",
        "name": name,
    }


class CivitaiUpdatesRendering(unittest.TestCase):
    def test_marks_itself_running_while_the_thread_works(self):
        reset(running=True, checked=3, total=40)

        html_text = RENDER_STATUS()

        self.assertIn('data-running="1"', html_text)
        self.assertIn("3 of 40", html_text)

    def test_finished_state_is_pollable_and_reports_the_result(self):
        reset(total=40, skipped=5, finished_at=time.time(), items={"a" * 64: entry()})

        html_text = RENDER_STATUS()

        self.assertIn('data-running="0"', html_text)
        self.assertIn("40 networks checked", html_text)
        self.assertIn("5 without a hash yet", html_text)

    def test_reports_an_up_to_date_library(self):
        reset(total=12, finished_at=time.time())

        html_text = RENDER_STATUS()

        self.assertIn("up to date", html_text)

    def test_empty_list_before_the_first_check(self):
        reset()

        self.assertIn("Check for updates", RENDER_LIST("civitai.com"))

    def test_lists_every_outdated_network_with_its_versions(self):
        reset(finished_at=time.time(), items={"a" * 64: entry(name="detail lora", current="v1.0", new="v2.0", extra=3)})

        html_text = RENDER_LIST("civitai.red")

        self.assertIn("detail lora", html_text)
        self.assertIn("v1.0", html_text)
        self.assertIn("<b>v2.0</b>", html_text)
        self.assertIn("+2 more version(s)", html_text)
        self.assertIn('data-hash="%s"' % ("a" * 64), html_text)
        self.assertIn("civitai.red/models/7", html_text, "the link must follow the selected site")

    def test_never_lets_the_site_inject_markup(self):
        reset(finished_at=time.time(), items={"b" * 64: entry(name="<img src=x onerror=alert(1)>")})

        html_text = RENDER_LIST("civitai.com")

        self.assertNotIn("<img src=x", html_text)
        self.assertIn("&lt;img src=x", html_text)

    def test_says_so_when_nothing_can_be_checked(self):
        reset(finished_at=time.time())

        self.assertIn("newest version", RENDER_LIST("civitai.com"))


if __name__ == "__main__":
    unittest.main()
