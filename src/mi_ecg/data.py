from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


PHYSIONET_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"


@dataclass(frozen=True)
class SplitConfig:
    train_folds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    val_folds: tuple[int, ...] = (9,)
    test_folds: tuple[int, ...] = (10,)


def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def ptbxl_root(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    if not root.is_absolute():
        root = project_root_from_file() / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def download_file(url: str, out_path: Path, overwrite: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return
    with requests.get(url, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0") or 0)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with tmp.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=out_path.name,
            leave=False,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(out_path)


def ensure_ptbxl_metadata(data_dir: str | Path, overwrite: bool = False) -> tuple[Path, Path]:
    root = ptbxl_root(data_dir)
    metadata = root / "ptbxl_database.csv"
    scp = root / "scp_statements.csv"
    download_file(f"{PHYSIONET_BASE_URL}/ptbxl_database.csv", metadata, overwrite=overwrite)
    download_file(f"{PHYSIONET_BASE_URL}/scp_statements.csv", scp, overwrite=overwrite)
    return metadata, scp


def load_ptbxl_tables(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_path, scp_path = ensure_ptbxl_metadata(data_dir)
    df = pd.read_csv(metadata_path)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(scp_path, index_col=0)
    return df, scp


def diagnostic_codes_for_class(scp: pd.DataFrame, diagnostic_class: str) -> set[str]:
    diagnostic_class = diagnostic_class.upper()
    diagnostic_col = "diagnostic" if "diagnostic" in scp.columns else None
    mask = scp["diagnostic_class"].fillna("").str.upper().eq(diagnostic_class)
    if diagnostic_col:
        mask &= scp[diagnostic_col].fillna(0).astype(float).astype(bool)
    return set(scp.index[mask].astype(str))


def add_mi_label(df: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    mi_codes = diagnostic_codes_for_class(scp, "MI")
    labeled = df.copy()
    labeled["target_mi"] = labeled["scp_codes"].apply(
        lambda codes: int(any(code in mi_codes and float(score) > 0 for code, score in codes.items()))
    )
    labeled["target_codes"] = labeled["scp_codes"].apply(
        lambda codes: sorted([code for code, score in codes.items() if code in mi_codes and float(score) > 0])
    )
    return labeled


def select_records(
    df: pd.DataFrame,
    split: SplitConfig = SplitConfig(),
    max_records: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    allowed_folds = set(split.train_folds + split.val_folds + split.test_folds)
    selected = df[df["strat_fold"].isin(allowed_folds)].copy()
    if max_records is None or max_records <= 0 or max_records >= len(selected):
        return selected.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    group_cols = ["strat_fold", "target_mi"]
    counts = selected.groupby(group_cols).size().reset_index(name="n")
    counts["take"] = np.maximum(1, np.floor(counts["n"] / len(selected) * max_records).astype(int))
    diff = max_records - int(counts["take"].sum())
    if diff > 0:
        order = counts.sort_values("n", ascending=False).index[:diff]
        counts.loc[order, "take"] += 1
    for _, row in counts.iterrows():
        pool = selected[
            (selected["strat_fold"] == row["strat_fold"]) & (selected["target_mi"] == row["target_mi"])
        ]
        take = min(int(row["take"]), len(pool))
        if take:
            idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
            parts.append(selected.loc[idx])
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def waveform_column(sampling_rate: int) -> str:
    if sampling_rate == 100:
        return "filename_lr"
    if sampling_rate == 500:
        return "filename_hr"
    raise ValueError("sampling_rate must be 100 or 500 for PTB-XL raw WFDB files")


def download_ptbxl_waveforms(
    records: pd.DataFrame,
    data_dir: str | Path,
    sampling_rate: int = 100,
    overwrite: bool = False,
) -> None:
    root = ptbxl_root(data_dir)
    col = waveform_column(sampling_rate)
    for record_name in tqdm(records[col].astype(str).tolist(), desc="PTB-XL WFDB records"):
        for suffix in (".hea", ".dat"):
            rel = f"{record_name}{suffix}"
            download_file(f"{PHYSIONET_BASE_URL}/{rel}", root / rel, overwrite=overwrite)


def load_waveforms(
    records: pd.DataFrame,
    data_dir: str | Path,
    sampling_rate: int = 100,
) -> np.ndarray:
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("wfdb is required to read PTB-XL WFDB files. Install with `pip install wfdb`.") from exc

    root = ptbxl_root(data_dir)
    col = waveform_column(sampling_rate)
    xs = []
    for record_name in tqdm(records[col].astype(str).tolist(), desc="Reading waveforms"):
        signal, _ = wfdb.rdsamp(str(root / record_name))
        xs.append(signal.T.astype(np.float32))
    return np.stack(xs, axis=0)


def split_masks(records: pd.DataFrame, split: SplitConfig = SplitConfig()) -> dict[str, np.ndarray]:
    folds = records["strat_fold"].to_numpy()
    return {
        "train": np.isin(folds, np.array(split.train_folds)),
        "val": np.isin(folds, np.array(split.val_folds)),
        "test": np.isin(folds, np.array(split.test_folds)),
    }


def build_dataset(
    data_dir: str | Path,
    sampling_rate: int = 100,
    max_records: int | None = None,
    seed: int = 42,
    download: bool = True,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    df, scp = load_ptbxl_tables(data_dir)
    labeled = add_mi_label(df, scp)
    records = select_records(labeled, max_records=max_records, seed=seed)
    if download:
        download_ptbxl_waveforms(records, data_dir, sampling_rate=sampling_rate, overwrite=overwrite)
    x = load_waveforms(records, data_dir, sampling_rate=sampling_rate)
    y = records["target_mi"].to_numpy(dtype=np.int64)
    masks = split_masks(records)
    return x, y, records, masks


def write_dataset_card(records: pd.DataFrame, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_records": int(len(records)),
        "n_patients": int(records["patient_id"].nunique()) if "patient_id" in records else None,
        "mi_positive": int(records["target_mi"].sum()),
        "mi_negative": int((1 - records["target_mi"]).sum()),
        "fold_counts": [
            {"strat_fold": int(fold), "target_mi": int(label), "n": int(n)}
            for (fold, label), n in records.groupby(["strat_fold", "target_mi"]).size().items()
        ],
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def labels_summary(records: pd.DataFrame) -> pd.DataFrame:
    return (
        records.groupby(["strat_fold", "target_mi"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["strat_fold", "target_mi"])
    )
