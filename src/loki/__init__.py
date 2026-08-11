"""
LOKI: Leverage Optimal significance to unveil Keplerian orbIt pulsars.

A high-performance C++ library for pulsar searching with Python bindings.
"""

import os
from importlib import metadata

__version__ = metadata.version(__name__)
# Git revision the container image was built from (set by docker/Dockerfile).
__commit__ = os.environ.get("LOKI_GIT_COMMIT", "unknown")

# CPU backend (always available)
from . import libloki

# GPU backend (conditionally available)
try:
    from . import libculoki

    __all__ = ["libculoki", "libloki"]
except ImportError:
    libculoki = None
    __all__ = ["libloki"]
