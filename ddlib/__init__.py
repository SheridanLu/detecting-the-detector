"""ddlib: small helper package for the 'Detecting the Detector' notebooks (paths, data, text cleaning, metrics, plots)."""
from .paths import get_repo, data_dir
from .textclean import clean, format_gate
from .seeds import set_seed

__all__ = ["get_repo", "data_dir", "clean", "format_gate", "set_seed"]
