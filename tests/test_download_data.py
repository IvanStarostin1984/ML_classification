from pathlib import Path
import importlib
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
    (dest / "dummy.txt").touch()

    monkeypatch.setattr(download_data, "DEST_DIR", dest)

    download_data.main()
    out = capsys.readouterr().out
    assert "Skipping download" in out
