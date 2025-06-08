import pandas as pd
from src.diagnostics import roc_pr_boxplots, fairness_bar
from src.manifest import write_manifest


def test_plots_and_manifest(tmp_path):
    folds = pd.DataFrame(
        {
            "roc_auc": [0.9, 0.92, 0.85, 0.87],
            "pr_auc": [0.7, 0.72, 0.69, 0.71],
            "model": ["LR", "LR", "DT", "DT"],
        }
    )
    ax1 = roc_pr_boxplots(folds)
    ax2 = fairness_bar(pd.DataFrame({"model": ["LR", "DT"], "fairness": [0.9, 0.85]}))
    assert hasattr(ax1, "figure")
    assert hasattr(ax2, "figure")

    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("x")
    f2.write_text("y")
    out = tmp_path / "artefacts" / "SHA256_manifest.txt"
    write_manifest([str(f1), str(f2)], out)
    text = out.read_text()
    assert "python" in text
    assert str(f1) in text
