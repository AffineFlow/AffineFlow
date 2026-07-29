"""
Neural network module for AffineFlow.

Dynamically exposes all components from the `nnengine` package to ensure
seamless integration without requiring new releases when `NNEngine` changes.
"""

import nnengine
from nnengine import *

__all__ = getattr(nnengine, "__all__", [
    name for name in dir(nnengine) if not name.startswith("_")
])