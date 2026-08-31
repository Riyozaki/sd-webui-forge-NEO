"""The ArbuzDiffusion look.

A fork that carries its own name should also carry its own face.  Everything
visual lives here: the palette, the two Gradio themes built from it, the mark
and the small pieces of HTML the UI injects (favicon, header).

The palette is a watermelon, read from the outside in:

* **bark**  - the deep green rind.  It is the *neutral* ramp, so the whole
  interface is tinted with it rather than being a flat grey.
* **flesh** - the red-pink of the fruit.  It is the *primary* ramp: the one
  accent, used for the generate button, focus rings and active states.
* **rind**  - the bright green just under the skin.  The *secondary* ramp, for
  success states, the progress bar and anything that should read as "alive".

Both themes are plain :class:`gradio.themes.Base` subclasses, so every
component Gradio ships is recoloured, and every ``var(--primary-…)`` the
hand-written CSS uses resolves to these values.
"""

from __future__ import annotations

from urllib.parse import quote

import gradio as gr

try:
    from gradio.themes.utils import colors, sizes
except ImportError:  # pragma: no cover - an older gradio kept sizes elsewhere
    from gradio.themes.utils import colors  # type: ignore
    from gradio.themes import sizes  # type: ignore
except Exception:  # pragma: no cover - a gradio that moved them must not stop the UI
    colors = None
    sizes = None

APP_TITLE = "ArbuzDiffusion"
"""Window title, and the name the header shows."""

THEME_NAMES = ("Arbuz Dark", "Arbuz Light")
"""Names as they appear in Settings -> User Interface -> Gradio Theme."""

DEFAULT_THEME = "Arbuz Dark"

FONT = [
    "Inter",
    "Segoe UI Variable Text",
    "Segoe UI",
    "ui-sans-serif",
    "system-ui",
    "-apple-system",
    "Helvetica Neue",
    "Arial",
    "sans-serif",
]

FONT_MONO = [
    "JetBrains Mono",
    "IBM Plex Mono",
    "Cascadia Mono",
    "ui-monospace",
    "SFMono-Regular",
    "Consolas",
    "monospace",
]

# --------------------------------------------------------------------- the palette

try:
    FLESH = colors.Color(
        name="flesh",
        c50="#fff1f4",
        c100="#ffe1e7",
        c200="#ffc5d1",
        c300="#ff9db1",
        c400="#ff6b86",
        c500="#ff4d6d",
        c600="#f73659",
        c700="#d32445",
        c800="#b01e3c",
        c900="#8f1d35",
        c950="#51091c",
    )

    RIND = colors.Color(
        name="rind",
        c50="#ecfdf3",
        c100="#d6f7e4",
        c200="#aeefcd",
        c300="#74e0ac",
        c400="#3dcc88",
        c500="#16a96a",
        c600="#0f8a58",
        c700="#106e49",
        c800="#11583c",
        c900="#0e4833",
        c950="#062a1e",
    )

    BARK = colors.Color(
        name="bark",
        c50="#f5f8f6",
        c100="#e7ede9",
        c200="#d1dad4",
        c300="#adbab2",
        c400="#7e9087",
        c500="#5f7167",
        c600="#48584f",
        c700="#36453c",
        c800="#222e28",
        c900="#161e19",
        c950="#0c1210",
    )
except Exception:  # pragma: no cover
    # No Colour class to build ramps with. Name the nearest built-in hues
    # instead, so the WebUI still starts - just without our exact shades.
    FLESH, RIND, BARK = "rose", "emerald", "stone"

# The mark is small enough to be inlined: no binary asset, no request, and it
# inherits the current colour for the parts that should.
MARK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="ArbuzDiffusion">'
    '<path d="M2 36a30 30 0 0 1 60 0Z" fill="#157f4f"/>'
    '<path d="M6 36a26 26 0 0 1 52 0Z" fill="#d9f2e2"/>'
    '<path d="M10 36a22 22 0 0 1 44 0Z" fill="#ff4d6d"/>'
    '<g fill="#2b1b17">'
    '<ellipse cx="22" cy="27" rx="2.1" ry="3.3"/>'
    '<ellipse cx="32" cy="22" rx="2.1" ry="3.3"/>'
    '<ellipse cx="42" cy="27" rx="2.1" ry="3.3"/>'
    "</g>"
    "</svg>"
)


def favicon_href() -> str:
    """The mark as a data URI, so the browser tab needs no extra file."""
    return "data:image/svg+xml;utf8," + quote(MARK_SVG, safe="")


# --------------------------------------------------------------------- the themes


def _apply(theme, **overrides):
    """``theme.set()`` asserts on unknown keys; a gradio upgrade must not break the UI."""
    known = {key: value for key, value in overrides.items() if hasattr(theme, key)}
    theme.set(**known)
    return theme


def _shared(**extra):
    """Tokens that are the same in both weights, plus whatever the caller adds."""
    overrides = {
        "font": FONT,
        "font_mono": FONT_MONO,
        "button_border_width": "1px",
        "button_transition": "background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease",
        "block_title_text_weight": "600",
        "block_label_text_weight": "600",
        "block_radius": "12px",
        "panel_radius": "12px",
        "input_radius": "10px",
        "button_primary_radius": "10px",
        "button_secondary_radius": "10px",
        "button_cancel_radius": "10px",
        "checkbox_label_radius": "8px",
        "table_radius": "10px",
    }
    overrides.update(extra)
    return overrides


def _dark() -> dict:
    return _shared(
        body_background_fill="#0e1412",
        body_text_color="#e6f0e9",
        body_text_color_subdued="#9db3a5",
        background_fill_primary="#161e19",
        background_fill_secondary="#1b2521",
        block_background_fill="#151d1a",
        block_border_color="#26332d",
        block_border_width="1px",
        block_shadow="none",
        block_title_background_fill="transparent",
        block_title_text_color="#e6f0e9",
        block_label_background_fill="#1b2521",
        block_label_text_color="#b9ccbf",
        border_color_primary="#26332d",
        border_color_accent="#ff4d6d",
        border_color_accent_subdued="#3a262c",
        color_accent="#ff4d6d",
        color_accent_soft="rgba(255, 77, 109, 0.25)",
        input_background_fill="#111916",
        input_border_color="#2b3931",
        input_placeholder_color="#6d8075",
        input_shadow="inset 0 1px 0 rgba(0, 0, 0, 0.35)",
        input_shadow_focus="0 0 0 2px rgba(255, 77, 109, 0.35)",
        panel_background_fill="#1b2521",
        panel_border_color="#2b3931",
        button_primary_background_fill="#f73659",
        button_primary_background_fill_hover="#ff4d6d",
        button_primary_text_color="#fff5f7",
        button_primary_text_color_hover="#ffffff",
        button_primary_border_color="#f73659",
        button_primary_border_color_hover="#ff6b86",
        button_secondary_background_fill="#1f2b25",
        button_secondary_background_fill_hover="#27352d",
        button_secondary_text_color="#dcebe1",
        button_secondary_text_color_hover="#ffffff",
        button_secondary_border_color="#2f3d35",
        button_secondary_border_color_hover="#3d4e43",
        button_cancel_background_fill="#2a1c20",
        button_cancel_background_fill_hover="#3a2328",
        button_cancel_text_color="#ffc9d3",
        button_cancel_border_color="#4a2a32",
        checkbox_background_color="#111916",
        checkbox_border_color="#33453c",
        checkbox_border_color_focus="#ff4d6d",
        checkbox_border_color_hover="#3d4e43",
        checkbox_border_color_selected="#ff4d6d",
        checkbox_background_color_selected="#ff4d6d",
        slider_color="#ff4d6d",
        slider_color_dark="#16a96a",
        loader_color="#ff4d6d",
        error_background_fill="#2a1418",
        error_border_color="#6b2130",
        error_text_color="#ffb3c0",
        stat_background_fill="#111916",
        table_even_background_fill="#151d1a",
        table_odd_background_fill="#1a2320",
        table_border_color="#26332d",
    )


def _light() -> dict:
    return _shared(
        body_background_fill="#f4f8f5",
        body_text_color="#16211c",
        body_text_color_subdued="#5c6f63",
        background_fill_primary="#ffffff",
        background_fill_secondary="#edf3ee",
        block_background_fill="#ffffff",
        block_border_color="#d7e3da",
        block_border_width="1px",
        block_shadow="0 1px 2px rgba(22, 33, 28, 0.06)",
        block_title_background_fill="transparent",
        block_title_text_color="#16211c",
        block_label_background_fill="#edf3ee",
        block_label_text_color="#3f5147",
        border_color_primary="#d7e3da",
        border_color_accent="#e11d48",
        border_color_accent_subdued="#f6ccd5",
        color_accent="#e11d48",
        color_accent_soft="rgba(225, 29, 72, 0.18)",
        input_background_fill="#ffffff",
        input_border_color="#cfdcd4",
        input_placeholder_color="#8b9c91",
        input_shadow="inset 0 1px 2px rgba(22, 33, 28, 0.05)",
        input_shadow_focus="0 0 0 3px rgba(225, 29, 72, 0.18)",
        panel_background_fill="#ffffff",
        panel_border_color="#d7e3da",
        button_primary_background_fill="#e11d48",
        button_primary_background_fill_hover="#be123c",
        button_primary_text_color="#fff5f7",
        button_primary_text_color_hover="#ffffff",
        button_primary_border_color="#e11d48",
        button_primary_border_color_hover="#be123c",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#edf3ee",
        button_secondary_text_color="#26372f",
        button_secondary_text_color_hover="#16211c",
        button_secondary_border_color="#cfdcd4",
        button_secondary_border_color_hover="#b6c7bd",
        button_cancel_background_fill="#fdeef1",
        button_cancel_background_fill_hover="#fbdde3",
        button_cancel_text_color="#9f1239",
        button_cancel_border_color="#f6ccd5",
        checkbox_background_color="#ffffff",
        checkbox_border_color="#b6c7bd",
        checkbox_border_color_focus="#e11d48",
        checkbox_border_color_hover="#9fb1a5",
        checkbox_border_color_selected="#e11d48",
        checkbox_background_color_selected="#e11d48",
        slider_color="#e11d48",
        slider_color_dark="#157f4f",
        loader_color="#e11d48",
        error_background_fill="#fdeef1",
        error_border_color="#f6ccd5",
        error_text_color="#9f1239",
        stat_background_fill="#edf3ee",
        table_even_background_fill="#ffffff",
        table_odd_background_fill="#f4f8f5",
        table_border_color="#d7e3da",
    )


class ArbuzDark(gr.themes.Base):
    """Rind at night: near-black green surfaces, flesh-red accent."""

    def __init__(self, **kwargs):
        # defaults, so build_theme() can override the scales for a density
        if HAS_SIZES:
            kwargs.setdefault("spacing_size", sizes.spacing_md)
            kwargs.setdefault("text_size", sizes.text_md)
            kwargs.setdefault("radius_size", sizes.radius_md)

        super().__init__(
            primary_hue=FLESH,
            secondary_hue=RIND,
            neutral_hue=BARK,
            **kwargs,
        )
        _apply(self, **_dark())


class ArbuzLight(gr.themes.Base):
    """Daylight: paper-white with a green cast, the same flesh-red accent."""

    def __init__(self, **kwargs):
        # defaults, so build_theme() can override the scales for a density
        if HAS_SIZES:
            kwargs.setdefault("spacing_size", sizes.spacing_md)
            kwargs.setdefault("text_size", sizes.text_md)
            kwargs.setdefault("radius_size", sizes.radius_md)

        super().__init__(
            primary_hue=FLESH,
            secondary_hue=RIND,
            neutral_hue=BARK,
            **kwargs,
        )
        _apply(self, **_light())


DENSITIES = ("Compact", "Cozy", "Spacious")
"""The three steps of the interface density setting."""


HAS_SIZES = sizes is not None and hasattr(sizes, "radius_md")
"""False on a gradio whose size presets moved; the themes then keep its defaults."""


def _scale(density: str | None):
    """Gradio sizes for a density name.

    Using the theme's own spacing and text scales keeps the change consistent
    everywhere - every component Gradio ships reads them - instead of poking at
    paddings from CSS with ``!important``.
    """
    if not HAS_SIZES:  # pragma: no cover
        return None, None

    step = (density or "Cozy").strip().lower()
    spacing = {"compact": "spacing_sm", "spacious": "spacing_lg"}.get(step, "spacing_md")
    text = {"compact": "text_sm", "spacious": "text_lg"}.get(step, "text_md")
    return getattr(sizes, spacing, sizes.spacing_md), getattr(sizes, text, sizes.text_md)


def build_theme(name: str | None = None, density: str | None = None):
    """The theme for a name from the settings drop-down."""
    spacing, text = _scale(density)
    extra = {}
    if spacing is not None:
        extra["spacing_size"] = spacing
    if text is not None:
        extra["text_size"] = text

    if (name or "").strip().lower() == "arbuz light":
        return ArbuzLight(**extra)
    return ArbuzDark(**extra)


def is_arbuz_theme(name: str | None) -> bool:
    return (name or "").strip().lower() in {"arbuz dark", "arbuz light"}


# --------------------------------------------------------------------- the header


def brand_bar_html(tagline: str = "") -> str:
    """The slim header: mark, wordmark, tagline."""
    return (
        '<div class="arbuz-brand">'
        f'<span class="arbuz-brand-mark">{MARK_SVG}</span>'
        f'<span class="arbuz-brand-name">{APP_TITLE}</span>'
        + (f'<span class="arbuz-brand-tagline">{tagline}</span>' if tagline else "")
        + "</div>"
    )


def flags_html(density: str, theme_name: str) -> str:
    """A hidden element carrying what the stylesheet cannot know on its own.

    Both values are baked in when the UI is built, which is why changing either
    setting asks for a reload.  The page script turns them into classes on the
    Gradio root so the hand written CSS can follow the theme.
    """
    light = "light" if (theme_name or "").strip().lower() == "arbuz light" else "dark"
    return (
        '<div id="arbuz-flags" style="display:none" '
        f'data-density="{(density or "Cozy").strip().lower()}" data-weight="{light}"></div>'
    )
