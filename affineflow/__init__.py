"""
AffineFlow: A unified, high-performance Machine Learning framework.
"""
import importlib
import pkgutil
from pathlib import Path

__version__ = "0.1.3"

def __getattr__(name):
    """
    Lazily load sub-packages (e.g., ml, nn, compose, sklearn) 
    only when they are explicitly accessed by the user.
    """
    try:
        return importlib.import_module(f".{name}", __name__)
    except ImportError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    """
    Dynamically discover available sub-packages to power IDE autocomplete 
    without hardcoding module names.
    """
    return [
        module.name
        for module in pkgutil.iter_modules([str(Path(__file__).parent)])
        if not module.name.startswith("_")
    ]