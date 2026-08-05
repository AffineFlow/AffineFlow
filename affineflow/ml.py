"""
Classical machine learning module for AffineFlow.
"""

try:
    import affineflow_ml
    from affineflow_ml import *  # type: ignore[wildcard-import]
    
    __all__ = getattr(affineflow_ml, "__all__", [
        name for name in dir(affineflow_ml) if not name.startswith("_")
    ])
except ImportError:
    __all__ = []
    pass