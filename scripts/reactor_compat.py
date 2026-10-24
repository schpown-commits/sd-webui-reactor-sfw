"""
ReActor Compatibility Layer
============================
Central utility for detecting the active WebUI backend (A1111, Forge, Forge Neo,
SD.Next, reForge) and the Gradio major version (3 vs 4).

Provides cross-platform helpers so that the rest of the extension never has to
scatter try/except blocks around UI or module imports.
"""

import importlib
import gradio as gr


# ---------------------------------------------------------------------------
# WebUI backend detection
# ---------------------------------------------------------------------------

def _has_module(name: str) -> bool:
    """Check whether a Python module exists without importing it."""
    return importlib.util.find_spec(name) is not None


def is_forge() -> bool:
    """True when running inside any Forge variant (Forge, Forge Neo, reForge)."""
    return _has_module("ldm_patched") or _has_module("backend")


def is_sdnext() -> bool:
    """True when running inside SD.Next (vladmandic fork)."""
    try:
        from scripts.reactor_helpers import get_SDNEXT
        return get_SDNEXT()
    except Exception:
        return False


def is_a1111() -> bool:
    """True when running inside vanilla AUTOMATIC1111 WebUI."""
    return not is_forge() and not is_sdnext()


# ---------------------------------------------------------------------------
# Gradio version helpers
# ---------------------------------------------------------------------------

def gradio_major_version() -> int:
    """Return the Gradio major version as an integer (3 or 4)."""
    try:
        return int(gr.__version__.split(".")[0])
    except Exception:
        return 3  # safe fallback


def gr_update(component_class, **kwargs):
    """
    Cross-version wrapper for Gradio component updates.

    In Gradio 3.x components have a `.update()` classmethod; in 4.x the
    idiomatic way is to return a new component instance with the desired
    properties.

    Usage::

        return gr_update(gr.Dropdown, value="x", choices=["x", "y"])
    """
    if gradio_major_version() >= 4:
        return component_class(**kwargs)
    else:
        return component_class.update(**kwargs)


def tool_variant() -> str:
    """
    Return a valid Button variant string for the current Gradio version.

    Gradio 3 supports ``"tool"``; Gradio 4 only supports
    ``"primary" | "secondary" | "stop" | "huggingface"``.
    """
    return "tool" if gradio_major_version() < 4 else "secondary"


# ---------------------------------------------------------------------------
# Safe imports for modules that may not exist on every backend
# ---------------------------------------------------------------------------

def safe_import_scripts_postprocessing():
    """
    Import ``modules.scripts_postprocessing`` if available.

    Returns ``None`` when the module does not exist (e.g. some Forge builds).
    """
    try:
        from modules import scripts_postprocessing
        return scripts_postprocessing
    except ImportError:
        return None


def safe_import_face_restoration_models():
    """
    Import ``codeformer_model`` and ``gfpgan_model`` from whichever location
    the current backend exposes them.

    Returns a ``(codeformer_model, gfpgan_model)`` tuple.  Either or both may
    be ``None`` if the models are unavailable (e.g. Forge Neo minimal).
    """
    try:  # A1111
        from modules import codeformer_model, gfpgan_model
        return codeformer_model, gfpgan_model
    except ImportError:
        pass
    try:  # SD.Next / some Forge variants
        from modules.postprocess import codeformer_model, gfpgan_model
        return codeformer_model, gfpgan_model
    except ImportError:
        pass
    # Forge Neo / minimal — face restoration not available
    return None, None
