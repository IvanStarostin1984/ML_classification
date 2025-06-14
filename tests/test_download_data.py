from pathlib import Path
import importlib
import hashlib
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
            (Path(path) / download_data.CSV_NAME).write_text("data")

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
    sha = dest / f"{download_data.CSV_NAME}.sha256"
    digest = hashlib.sha256((dest / download_data.CSV_NAME).read_bytes()).hexdigest()
    assert sha.read_text() == digest


def test_skip_if_present(monkeypatch, tmp_path, capsys):
    dest = tmp_path
    dest.mkdir(exist_ok=True)
    csv = dest / download_data.CSV_NAME
    csv.write_text("data")
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    (dest / f"{download_data.CSV_NAME}.sha256").write_text(digest)

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


def test_redownload_if_sha_mismatch(monkeypatch, tmp_path):
    dest = tmp_path
    dest.mkdir(exist_ok=True)
    csv = dest / download_data.CSV_NAME
    csv.write_text("old")
    (dest / f"{download_data.CSV_NAME}.sha256").write_text("badsha")

    monkeypatch.setattr(download_data, "DEST_DIR", dest)
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")

    calls = {}

    class DummyApi:
        def authenticate(self):
            calls["auth"] = True

        def dataset_download_files(self, dataset, path, unzip):
            calls["dataset"] = dataset
            (Path(path) / download_data.CSV_NAME).write_text("new")

    module = importlib.import_module("kaggle.api.kaggle_api_extended")
    monkeypatch.setattr(module, "KaggleApi", lambda: DummyApi())

    download_data.main()

    assert calls["dataset"] == download_data.DATASET
