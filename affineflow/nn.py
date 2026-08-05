"""
Neural network module for AffineFlow.
"""

try:
    import affineflow_nn
    from affineflow_nn import *  # type: ignore[wildcard-import]

    __all__ = getattr(affineflow_nn, "__all__", [
        name for name in dir(affineflow_nn) if not name.startswith("_")
    ])
except ImportError:
    __all__ = []
    pass