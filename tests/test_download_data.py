from pathlib import Path
import importlib
import pytest
from scripts import download_data


def test_download_invoked(monkeypatch, tmp_path):
    dest = tmp_path / "raw"
    calls = {}

    class DummyApi:
        def authenticate(self):
            calls["auth"] = True

        def dataset_download_files(self, dataset, path, unzip):
            calls["dataset"] = dataset
            calls["path"] = path
            calls["unzip"] = unzip

    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")
    monkeypatch.setattr(download_data, "DEST_DIR", dest)
    module = importlib.import_module("kaggle.api.kaggle_api_extended")
    monkeypatch.setattr(module, "KaggleApi", lambda: DummyApi())

    download_data.main()

    assert calls["auth"]
    assert calls["dataset"] == download_data.DATASET
    assert Path(calls["path"]) == dest
    assert dest.exists()


def test_skip_if_present(monkeypatch, tmp_path, capsys):
    dest = tmp_path
    dest.mkdir(exist_ok=True)
    (dest / download_data.CSV_NAME).touch()

    monkeypatch.setattr(download_data, "DEST_DIR", dest)

    calls = {}

    class DummyApi:
        def authenticate(self):
            calls["auth"] = True

        def dataset_download_files(self, dataset, path, unzip):
            calls["dataset"] = dataset

    module = importlib.import_module("kaggle.api.kaggle_api_extended")
    monkeypatch.setattr(module, "KaggleApi", lambda: DummyApi())

    download_data.main()
    out = capsys.readouterr().out
    assert "Skipping download" in out
    assert calls == {}


def test_warn_if_src_missing(monkeypatch, capsys):
    import builtins

    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.dataprep":
            raise ImportError
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit):
        importlib.reload(download_data)
    out = capsys.readouterr().out
    assert "pip install -e ." in out
