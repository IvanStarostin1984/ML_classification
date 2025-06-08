import importlib

MODULES = [
  'src.dataprep',
  'src.features',
  'src.diagnostics',
  'src.preprocessing',
  'src.selection',
  'src.split',
  'src.models.logreg',
  'src.models.cart',
]

def test_imports():
  for module in MODULES:
    assert importlib.import_module(module)
