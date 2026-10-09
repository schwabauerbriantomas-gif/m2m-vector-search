"""DEPRECATED: usar m2m.integrations. Este shim existe por compatibilidad."""
import warnings
from m2m.integrations.langchain import M2MVectorStore  # noqa: F401

warnings.warn(
    "integrations.langchain esta deprecado; use m2m.integrations.langchain",
    DeprecationWarning,
    stacklevel=2,
)
__all__ = ["M2MVectorStore"]
