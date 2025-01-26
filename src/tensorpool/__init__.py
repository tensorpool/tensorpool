from importlib.metadata import version

try:
    __version__ = version("tensorpool")
except ImportError:
    # Package not installed
    __version__ = "unknown"
