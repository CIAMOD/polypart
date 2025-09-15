"""Top-level package for polypart.

This module exposes lightweight access to submodules and manages the
package version. Submodules are imported lazily on first attribute access
to avoid pulling heavy dependencies during a simple ``import polypart``.

Attributes:
    __version__ (str): package version, prefers the installed distribution
        version when available.
    ftyping, geometry, ppart, io: lazily loaded submodules.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _version
from typing import Any

__all__ = ("ftyping", "geometry", "ppart", "io", "__version__")

# Static fallback version (kept for checkouts). When the package is installed
# the runtime will override this with the installed distribution version.
__version__ = "0.1.0"


def _get_installed_version() -> str | None:
    try:
        return _version("polypart")
    except PackageNotFoundError:
        return None
    except Exception:
        # Be conservative: don't let unexpected errors break imports.
        return None


# Try to override the static version with the installed package version.
_installed = _get_installed_version()
if _installed is not None:
    __version__ = _installed


def __getattr__(name: str) -> Any:
    """Lazy-load submodules as attributes on the package module.

    Accessing ``polypart.geometry`` for the first time triggers importing
    ``polypart.geometry`` and caches it in the package namespace.
    """
    if name in ("ftyping", "geometry", "ppart", "io"):
        module = import_module(f"polypart.{name}")
        globals()[name] = module
        return module
    if name == "__version__":
        return __version__
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(globals().keys()) + ["ftyping", "geometry", "ppart", "io"]
