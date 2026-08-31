"""Tests for the ArbuzDiffusion theme.

Gradio is not importable here, so the test installs a stub that records what the
theme asks for.  What matters is not the exact colours but the contract: both
weights build, an unknown theme name falls back instead of crashing, the density
setting reaches the theme's own scales, and a gradio upgrade that renames a
token cannot take the UI down with it.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = str(Path(__file__).parents[1])

# The tokens gradio 4.40 really has; the stub owns exactly these, so an override
# of something that does not exist is visibly dropped.
KNOWN_TOKENS = {
    "font",
    "font_mono",
    "body_background_fill",
    "body_text_color",
    "body_text_color_subdued",
    "background_fill_primary",
    "background_fill_secondary",
    "block_background_fill",
    "block_border_color",
    "block_border_width",
    "block_radius",
    "block_shadow",
    "block_title_background_fill",
    "block_title_text_color",
    "block_title_text_weight",
    "block_label_background_fill",
    "block_label_text_color",
    "block_label_text_weight",
    "border_color_primary",
    "border_color_accent",
    "border_color_accent_subdued",
    "color_accent",
    "color_accent_soft",
    "input_background_fill",
    "input_border_color",
    "input_placeholder_color",
    "input_radius",
    "input_shadow",
    "input_shadow_focus",
    "panel_background_fill",
    "panel_border_color",
    "panel_radius",
    "button_border_width",
    "button_transition",
    "button_primary_background_fill",
    "button_primary_background_fill_hover",
    "button_primary_text_color",
    "button_primary_text_color_hover",
    "button_primary_border_color",
    "button_primary_border_color_hover",
    "button_primary_radius",
    "button_secondary_background_fill",
    "button_secondary_background_fill_hover",
    "button_secondary_text_color",
    "button_secondary_text_color_hover",
    "button_secondary_border_color",
    "button_secondary_border_color_hover",
    "button_secondary_radius",
    "button_cancel_background_fill",
    "button_cancel_background_fill_hover",
    "button_cancel_text_color",
    "button_cancel_border_color",
    "button_cancel_radius",
    "checkbox_label_radius",
    "checkbox_background_color",
    "checkbox_border_color",
    "checkbox_border_color_focus",
    "checkbox_border_color_hover",
    "checkbox_border_color_selected",
    "checkbox_background_color_selected",
    "slider_color",
    "slider_color_dark",
    "loader_color",
    "error_background_fill",
    "error_border_color",
    "error_text_color",
    "stat_background_fill",
    "table_radius",
    "table_even_background_fill",
    "table_odd_background_fill",
    "table_border_color",
}


def _install_stub_gradio():
    gradio = types.ModuleType("gradio")

    class Color:
        def __init__(self, name, **stops):
            self.name = name
            self.stops = stops

    class Size:
        def __init__(self, name):
            self.name = name

    class Base:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for token in KNOWN_TOKENS:
                setattr(self, token, None)

        def set(self, **kwargs):
            for key, value in kwargs.items():
                assert hasattr(self, key), f"unknown token: {key}"
                setattr(self, key, value)

    sizes = types.SimpleNamespace(
        **{name: Size(name) for name in (
            "spacing_sm", "spacing_md", "spacing_lg",
            "text_sm", "text_md", "text_lg",
            "radius_md",
        )}
    )
    utils = types.ModuleType("gradio.themes.utils")
    utils.colors = types.SimpleNamespace(Color=Color)
    utils.sizes = sizes
    themes = types.ModuleType("gradio.themes")
    themes.Base = Base
    themes.utils = utils

    gradio.themes = themes
    sys.modules["gradio"] = gradio
    sys.modules["gradio.themes"] = themes
    sys.modules["gradio.themes.utils"] = utils
    return themes


THEMES = _install_stub_gradio()

# Loaded by path, not as `modules.neo_theme`: another test module installs a
# stand-in `modules` package, and import order inside a suite is not ours to
# rely on.
SPECIFICATION = importlib.util.spec_from_file_location("neo_theme_under_test", Path(REPO_ROOT) / "modules" / "neo_theme.py")
neo_theme = importlib.util.module_from_spec(SPECIFICATION)
sys.modules["neo_theme_under_test"] = neo_theme
SPECIFICATION.loader.exec_module(neo_theme)


class PaletteTests(unittest.TestCase):
    def test_every_ramp_has_the_full_scale(self):
        for ramp in (neo_theme.FLESH, neo_theme.RIND, neo_theme.BARK):
            with self.subTest(ramp=ramp.name):
                self.assertEqual(len(ramp.stops), 11)

    def test_the_mark_is_a_complete_svg(self):
        self.assertTrue(neo_theme.MARK_SVG.startswith("<svg"))
        self.assertTrue(neo_theme.MARK_SVG.endswith("</svg>"))
        self.assertIn("viewBox", neo_theme.MARK_SVG)

    def test_the_favicon_needs_no_file(self):
        href = neo_theme.favicon_href()

        self.assertTrue(href.startswith("data:image/svg+xml"))
        self.assertNotIn("#", href, "an unescaped # would truncate the data URI")


class ThemeBuildingTests(unittest.TestCase):
    def test_both_weights_build(self):
        self.assertIsInstance(neo_theme.build_theme("Arbuz Dark"), neo_theme.ArbuzDark)
        self.assertIsInstance(neo_theme.build_theme("Arbuz Light"), neo_theme.ArbuzLight)

    def test_an_unknown_name_falls_back_to_dark(self):
        self.assertIsInstance(neo_theme.build_theme("gradio/base"), neo_theme.ArbuzDark)
        self.assertIsInstance(neo_theme.build_theme(""), neo_theme.ArbuzDark)
        self.assertIsInstance(neo_theme.build_theme(None), neo_theme.ArbuzDark)

    def test_the_hues_are_ours(self):
        dark = neo_theme.build_theme("Arbuz Dark")

        self.assertIs(dark.kwargs["primary_hue"], neo_theme.FLESH)
        self.assertIs(dark.kwargs["secondary_hue"], neo_theme.RIND)
        self.assertIs(dark.kwargs["neutral_hue"], neo_theme.BARK)

    def test_the_weights_differ_where_it_counts(self):
        dark = neo_theme.build_theme("Arbuz Dark")
        light = neo_theme.build_theme("Arbuz Light")

        self.assertNotEqual(dark.body_background_fill, light.body_background_fill)
        self.assertEqual(dark.button_primary_radius, light.button_primary_radius)

    def test_density_reaches_the_theme_scales(self):
        compact = neo_theme.build_theme("Arbuz Dark", "Compact")
        cozy = neo_theme.build_theme("Arbuz Dark", "Cozy")
        spacious = neo_theme.build_theme("Arbuz Dark", "Spacious")

        self.assertEqual(compact.kwargs["spacing_size"].name, "spacing_sm")
        self.assertEqual(cozy.kwargs["spacing_size"].name, "spacing_md")
        self.assertEqual(spacious.kwargs["spacing_size"].name, "spacing_lg")
        self.assertEqual(compact.kwargs["text_size"].name, "text_sm")

    def test_density_may_be_passed_as_a_custom_value(self):
        # DropdownEditable: whatever the user typed must not break the build
        self.assertEqual(neo_theme.build_theme("Arbuz Dark", "huge").kwargs["spacing_size"].name, "spacing_md")

    def test_every_override_survives_the_stub(self):
        # the stub asserts on unknown tokens, so this also proves the token
        # names match a real gradio's
        theme = neo_theme.build_theme("Arbuz Dark")

        for token in ("body_background_fill", "input_shadow_focus", "slider_color", "error_text_color"):
            with self.subTest(token=token):
                self.assertIsNotNone(getattr(theme, token))


class SafetyTests(unittest.TestCase):
    def test_a_token_gradio_dropped_is_ignored(self):
        theme = neo_theme.build_theme("Arbuz Light")

        neo_theme._apply(theme, body_background_fill="#000000", this_token_does_not_exist="boom")

        self.assertEqual(theme.body_background_fill, "#000000")


class DegradeGracefullyTests(unittest.TestCase):
    """The theme is built at startup, so a gradio that moves its helpers must
    not be able to stop the WebUI from starting."""

    def test_without_the_colour_and_size_helpers(self):
        utils = sys.modules["gradio.themes.utils"]
        colors, sizes = utils.colors, utils.sizes
        try:
            utils.colors = None
            utils.sizes = None

            module = importlib.util.module_from_spec(SPECIFICATION)
            SPECIFICATION.loader.exec_module(module)
        finally:
            utils.colors = colors
            utils.sizes = sizes

        self.assertEqual(module.FLESH, "rose")
        self.assertEqual(module.RIND, "emerald")
        self.assertEqual(module.BARK, "stone")
        self.assertFalse(module.HAS_SIZES)
        self.assertIsInstance(module.build_theme("Arbuz Light", "Compact"), module.ArbuzLight)


class FlagTests(unittest.TestCase):
    def test_the_flags_carry_both_values(self):
        html = neo_theme.flags_html("Compact", "Arbuz Light")

        self.assertIn('data-density="compact"', html)
        self.assertIn('data-weight="light"', html)

    def test_the_flags_default_sensibly(self):
        html = neo_theme.flags_html(None, "something else")

        self.assertIn('data-density="cozy"', html)
        self.assertIn('data-weight="dark"', html)

    def test_the_header_mentions_the_name(self):
        self.assertIn(neo_theme.APP_TITLE, neo_theme.brand_bar_html("a tagline"))
        self.assertIn("a tagline", neo_theme.brand_bar_html("a tagline"))
        self.assertNotIn("tagline", neo_theme.brand_bar_html(""))


if __name__ == "__main__":
    unittest.main()
