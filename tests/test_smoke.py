from importlib import import_module
from pathlib import Path
import pkgutil
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _import_modules(package: str) -> None:
    pkg = import_module(package)
    search_paths = list(getattr(pkg, "__path__", []))
    if not search_paths and getattr(pkg, "__file__", None):
        search_paths = [Path(pkg.__file__).parent]
    for path in search_paths:
        for mod in pkgutil.iter_modules([str(path)]):
            try:
                import_module(f"{package}.{mod.name}")
            except ModuleNotFoundError as exc:
                if "kaggle" in str(exc):
                    pytest.skip("kaggle dependency missing")
                else:
                    raise


def test_import_skeleton() -> None:
    _import_modules("src")
    _import_modules("scripts")
