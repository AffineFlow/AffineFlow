"""
Classical machine learning module for MLEngine.

Dynamically exposes all components from the `knn_core` C++ extension.
"""

import knn_core
from knn_core import *

__all__ = getattr(knn_core, "__all__", [
    name for name in dir(knn_core) if not name.startswith("_")
])