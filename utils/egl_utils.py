# utils/egl_utils.py
import os
import pkgutil

import pybullet as p


def load_egl() -> int:
    """
    Load Bullet's eglRenderer plugin (Colab style).
    Call this once right after p.connect(p.DIRECT) and before loading assets.
    """
    os.environ.pop("DISPLAY", None)
    os.environ.setdefault("PYBULLET_EGL", "1")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

    egl = pkgutil.get_loader("eglRenderer")
    if egl:
        pid = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    else:
        # falls back to the default name found in PATH
        pid = p.loadPlugin("eglRendererPlugin", "_eglRendererPlugin")

    assert pid >= 0, "EGL plugin failed to load"
    print(f"Using GPU hardware via EGL (plugin id: {pid})")
    return pid


def is_egl_active() -> bool:
    return p.getPluginId("_eglRendererPlugin") >= 0
