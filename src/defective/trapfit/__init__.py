"""trapfit: technique-aware trap spectrum fitting for the DEFECTIVE agent."""
from .router import run, load_config, infer_technique

__all__ = ["run", "load_config", "infer_technique"]
__version__ = "0.1.0"
