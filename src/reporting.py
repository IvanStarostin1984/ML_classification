from __future__ import annotations

from pathlib import Path
import json
import logging
import shutil
from typing import Callable, Iterable

import pandas as pd


ROOT = Path(".")
ART = ROOT / "artefacts"
PLOTS = ROOT / "plots"
OUT = ROOT / "report_artifacts"

_file_index: dict[str, list[Path]] = {}


def _build_index() -> None:
    for d in (ROOT, ART, PLOTS):
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file():
                    _file_index.setdefault(p.name, []).append(p)
    for paths in _file_index.values():
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)


_build_index()


__all__ = [
    "find_path",
    "read_latest_glob",
    "write_section",
    "flatten_cv",
    "flatten_metrics",
    "dump_dataset_overview",
    "dump_lr_params",
    "dump_cart_params",
    "dump_feature_counts",
    "dump_corr_top10",
    "dump_dropped_twins",
    "dump_skew_profile",
    "dump_cart_overfit",
    "main",
]


def find_path(name: str) -> Path | None:
    """Return the newest path matching ``name`` in artefact folders."""
    return _file_index.get(name, [None])[0]


def read_latest_glob(
    pattern: str, dirs: Iterable[Path] = (ART, PLOTS, ROOT)
) -> Path | None:
    """Return most recent file matching ``pattern`` from ``dirs``."""
    candidates: list[Path] = []
    for d in dirs:
        candidates.extend(d.glob(pattern))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def write_section(f, title: str, reader_fn: Callable[[object], None]) -> None:
    """Write a section header then call ``reader_fn`` to populate it."""
    underline = "—" * min(len(title), 80)
    logging.info("Writing section: %s", title)
    f.write(f"• {title}\n{underline}\n")
    try:
        reader_fn(f)
    except FileNotFoundError as e:
        msg = f"**WARNING**: {e}"
        f.write(msg + "\n")
        logging.warning(msg)
    except Exception as e:  # pragma: no cover - unexpected
        msg = f"**ERROR**: {e}"
        f.write(msg + "\n")
        logging.error(msg, exc_info=True)
    f.write("\n")


def flatten_cv(path: str | Path) -> pd.DataFrame:
    """Flatten cross-validation CSV with multi-index columns."""
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df = df.dropna(how="all").dropna(how="all", axis=1)
    df.columns = [f"{a}_{b}".strip("_") for a, b in df.columns]
    return df.round(4).loc[["DT", "LR"]]


def flatten_metrics(md: dict) -> dict:
    """Flatten nested metric dictionary."""
    out: dict[str, float | int] = {}
    for k, v in md.items():
        if k == "bootstrap_iters":
            out[k] = int(v)
        elif isinstance(v, list) and len(v) == 2:
            out[f"{k}_low"] = round(v[0], 4)
            out[f"{k}_high"] = round(v[1], 4)
        elif isinstance(v, dict):
            for subk, subv in v.items():
                if isinstance(subv, float):
                    out[f"{k}_{subk}"] = round(subv, 4)
                else:
                    out[f"{k}_{subk}"] = subv
        elif isinstance(v, float):
            out[k] = int(v) if v.is_integer() else round(v, 4)
        else:
            out[k] = v
    return out


# reader helper functions ----------------------------------------------------


def dump_dataset_overview(ff) -> None:
    info = find_path("split_info.json")
    if info:
        data = json.load(info.open())
        total = data["n_total"]
        counts = {int(k): v for k, v in data["class_counts"].items()}
        approved, rejected = counts.get(1, 0), counts.get(0, 0)
        missing = False
    else:
        csv = find_path("loan_approval_dataset.csv")
        if not csv:
            raise FileNotFoundError("no split_info.json or raw CSV")
        df_full = pd.read_csv(csv)
        total = len(df_full)
        mapping = {"approved": 1, "rejected": 0, "y": 1, "n": 0}
        status = df_full["loan_status"].astype(str).str.lower().map(mapping)
        approved = int((status == 1).sum())
        rejected = total - approved
        missing = df_full.isnull().any().any()

    csv_path = find_path("loan_approval_dataset.csv")
    if not csv_path:
        raise FileNotFoundError("raw CSV not found for column count")
    n_raw_cols = len(pd.read_csv(csv_path, nrows=0).columns)

    p_app = approved / total * 100
    p_rej = rejected / total * 100
    miss_msg = "Some missing values present." if missing else "No missing values."

    ff.write(
        f"Dataset: Archit Sharma\u2019s Kaggle Loan Approval Prediction\n"
        f"  \u2022 {total} rows  \u2022  {n_raw_cols} raw features\n"
        f"  \u2022 Approved: {approved} ({p_app:.1f}%)  Rejected: {rejected}"
        f" ({p_rej:.1f}%)\n"
        f"  \u2022 {miss_msg}"
    )


def dump_lr_params(ff) -> None:
    p = read_latest_glob("cv_results_lr_*.csv")
    if not p:
        raise FileNotFoundError("no LR CV results")
    df = pd.read_csv(p)
    best = df[df["rank_test_roc_auc"] == 1].iloc[0]
    params = {
        "penalty": best["param_clf__penalty"],
        "C": best["param_clf__C"],
        "class_weight": best["param_clf__class_weight"],
        "sampler": best["param_sampler"],
    }
    ff.write(", ".join(f"{k}={v}" for k, v in params.items()))


def dump_cart_params(ff) -> None:
    p = read_latest_glob("cv_results_tree_cart_*.csv")
    if not p:
        raise FileNotFoundError("no CART CV results")
    df = pd.read_csv(p)
    best = df[df["rank_test_roc_auc"] == 1].iloc[0]
    params = {
        "max_depth": best["param_clf__max_depth"],
        "min_samples_leaf": best["param_clf__min_samples_leaf"],
        "ccp_alpha": best["param_clf__ccp_alpha"],
        "class_weight": best["param_clf__class_weight"],
        "sampler": best["param_sampler"],
    }
    ff.write(", ".join(f"{k}={v}" for k, v in params.items()))


def dump_feature_counts(ff) -> None:
    p = find_path("feature_registry.json")
    if not p:
        raise FileNotFoundError("feature_registry.json missing")
    reg = json.load(p.open())
    num = len(reg.get("numeric", []))
    cat = len(reg.get("categorical", []))
    dum = len(reg.get("dummy_0_1", []))
    ff.write(
        f"Total engineered features: {num + cat + dum} "
        f"(numeric={num}, categorical={cat}, dummy={dum})"
    )


def dump_corr_top10(ff) -> None:
    p = find_path("numerical_correlations.csv")
    if not p:
        raise FileNotFoundError("numerical_correlations.csv missing")
    corr = pd.read_csv(p)
    top = corr.assign(abs_coef=corr["coef"].abs()).nlargest(10, "abs_coef")
    top = top.loc[:, ["var1", "var2", "coef", "q"]]
    top.columns = ["feature1", "feature2", "ρ", "q_value"]
    ff.write(top.to_string(index=False, float_format="%.3f"))


def dump_dropped_twins(ff) -> None:
    p = find_path("feature_registry.json")
    if not p:
        raise FileNotFoundError("feature_registry.json missing")
    dropped = json.load(p.open()).get("numeric_to_drop_lm", [])
    ff.write("\n".join(dropped) if dropped else "None dropped")


def dump_skew_profile(ff) -> None:
    p = find_path("skew_profile.csv")
    if not p:
        raise FileNotFoundError("skew_profile.csv missing")
    df = pd.read_csv(p)
    heavy = int(df["heavy_tail"].sum())
    total = len(df)
    ff.write(
        f"Heavy-tailed features: {heavy}/{total} ({heavy/total*100:.1f}%)\n"
        f"Median outlier rate (>3σ): {df['out_pct'].median():.1f}%"
    )


def dump_cart_overfit(ff) -> None:
    sum_p = read_latest_glob("summary_metrics_cart.csv")
    if not sum_p:
        raise FileNotFoundError("summary_metrics_cart.csv missing")
    cv_roc = pd.read_csv(sum_p, index_col=0).loc["mean", "roc_auc"]
    t_p = find_path("test_metrics_cart.json")
    if not t_p:
        raise FileNotFoundError("test_metrics_cart.json missing")
    md = json.load(t_p.open())
    test_roc = None
    for key in ("ROC_AUC", "ROC_AUC@Youden", "ROC_AUC@0.50"):
        if key in md:
            test_roc = md[key]
            break
    if test_roc is None:
        raise KeyError("no ROC_AUC key in test_metrics_cart.json")
    ff.write(f"CART CV vs Test ROC AUC gap: {cv_roc - test_roc:.4f}")


# main ----------------------------------------------------------------------


def main(args: list[str] | None = None) -> None:
    """Collect tables and figures into ``report_artifacts/``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT.mkdir(parents=True, exist_ok=True)

    with open(OUT / "pipeline_tables.txt", "w") as f:
        write_section(f, "0. Dataset & class overview", dump_dataset_overview)
        write_section(f, "0A. Best hyperparameters (LR)", dump_lr_params)
        write_section(f, "0B. Best hyperparameters (CART)", dump_cart_params)
        write_section(f, "0C. Engineered-feature counts", dump_feature_counts)
        write_section(f, "0D. Top-10 absolute Spearman correlations", dump_corr_top10)
        write_section(f, "0E. Features dropped for high |ρ|", dump_dropped_twins)
        write_section(f, "0F. Skewness profile", dump_skew_profile)
        write_section(f, "9. Overfitting check (CART ROC AUC gap)", dump_cart_overfit)

    diagrams = [
        "roc_lr.png",
        "roc_cart.png",
        "pr_lr.png",
        "pr_cart.png",
        "calibration_lr.png",
        "calibration_cart.png",
        "cm_lr.png",
        "cm_cart.png",
    ]
    for name in diagrams:
        src = find_path(name)
        if not src:
            logging.warning("Diagram not found: %s", name)
            continue
        dst = OUT / name
        if src.resolve() != dst.resolve():
            shutil.copy(src, dst)
            logging.info("Copied diagram: %s", name)

    print(f"Report artifacts written to {OUT}")


if __name__ == "__main__":
    main()
