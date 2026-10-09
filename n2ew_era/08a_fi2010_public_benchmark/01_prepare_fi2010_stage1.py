from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW_ZIP = DATA / "raw" / "BenchmarkDatasets.zip"
EXTRACTED = DATA / "extracted"
CACHE = DATA / "cache"
RESULTS = ROOT / "results"

NOAUCTION_ZSCORE_PREFIX = "BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/"
TRAIN_MEMBER = (
    NOAUCTION_ZSCORE_PREFIX
    + "NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt"
)
TEST_MEMBERS = [
    NOAUCTION_ZSCORE_PREFIX
    + "NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt",
    NOAUCTION_ZSCORE_PREFIX
    + "NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt",
    NOAUCTION_ZSCORE_PREFIX
    + "NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt",
]

WINDOW = 100
TRAIN_FRACTION = 0.8
HORIZONS = [10, 20, 30, 50, 100]
CLASS_NAMES = {
    1: "up",
    2: "stationary",
    3: "down",
}
OFFICIAL_ZIP_SIZE = 1_864_361_899
OFFICIAL_ZIP_SHA256 = "cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664"
PUBLISHED_TOTAL_REPRESENTATIONS = 394_337


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extracted_path(member: str) -> Path:
    return EXTRACTED / member


def load_fi2010_txt(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 149:
        raise RuntimeError(f"Expected FI-2010 txt shape (149, N), got {arr.shape} for {path}")
    features = np.ascontiguousarray(arr[:40, :].T.astype(np.float32))
    labels = np.ascontiguousarray(arr[-5:, :].T.astype(np.int8))
    del arr
    unique = set(int(v) for v in np.unique(labels))
    if not unique.issubset({1, 2, 3}):
        raise RuntimeError(f"Unexpected label values {sorted(unique)} in {path}")
    return features, labels, (features.shape[0], features.shape[1])


def cache_one(name: str, member: str) -> dict[str, object]:
    source = extracted_path(member)
    if not source.exists():
        raise FileNotFoundError(f"Missing extracted file: {source}")
    out = CACHE / f"{name}.npz"
    if out.exists():
        with np.load(out) as z:
            features_shape = tuple(int(x) for x in z["features"].shape)
            labels_shape = tuple(int(x) for x in z["labels"].shape)
    else:
        features, labels, _ = load_fi2010_txt(source)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(out, features=features, labels=labels)
        features_shape = tuple(int(x) for x in features.shape)
        labels_shape = tuple(int(x) for x in labels.shape)
    return {
        "cache_name": name,
        "source_member": member,
        "source_path": str(source.relative_to(ROOT)),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "cache_path": str(out.relative_to(ROOT)),
        "features_shape": "x".join(str(x) for x in features_shape),
        "labels_shape": "x".join(str(x) for x in labels_shape),
        "num_events": features_shape[0],
        "num_features_used": features_shape[1],
    }


def zip_member_manifest() -> list[dict[str, object]]:
    with ZipFile(RAW_ZIP) as zf:
        rows = []
        for info in zf.infolist():
            if not info.filename.startswith(NOAUCTION_ZSCORE_PREFIX):
                continue
            rows.append(
                {
                    "member": info.filename,
                    "uncompressed_size_bytes": info.file_size,
                    "compressed_size_bytes": info.compress_size,
                    "crc": f"{info.CRC:08x}",
                }
            )
    return rows


def label_distribution_rows(split: str, labels: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for col, horizon in enumerate(HORIZONS):
        values, counts = np.unique(labels[:, col], return_counts=True)
        by_value = {int(v): int(c) for v, c in zip(values, counts)}
        total = int(labels.shape[0])
        for label_value in [1, 2, 3]:
            count = by_value.get(label_value, 0)
            rows.append(
                {
                    "split": split,
                    "horizon": horizon,
                    "class_label": label_value,
                    "class_name": CLASS_NAMES[label_value],
                    "count": count,
                    "share": count / total if total else 0.0,
                }
            )
    return rows


def save_label_distributions(cache_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    train_npz = CACHE / "train_cf7.npz"
    test_npzs = [CACHE / "test_cf7.npz", CACHE / "test_cf8.npz", CACHE / "test_cf9.npz"]
    with np.load(train_npz) as z:
        train_labels_all = z["labels"]
        train_cut = int(np.floor(train_labels_all.shape[0] * TRAIN_FRACTION))
        train_fit_labels = train_labels_all[:train_cut]
        val_labels = train_labels_all[train_cut:]
        train_window_labels = train_fit_labels[WINDOW - 1 :]
        val_window_labels = val_labels[WINDOW - 1 :]
    test_labels_parts = []
    for path in test_npzs:
        with np.load(path) as z:
            test_labels_parts.append(z["labels"])
    test_labels_all = np.concatenate(test_labels_parts, axis=0)
    test_window_labels = test_labels_all[WINDOW - 1 :]

    snapshot_rows: list[dict[str, object]] = []
    snapshot_rows += label_distribution_rows("train_cf7_all_snapshots", train_labels_all)
    snapshot_rows += label_distribution_rows("train_fit_head_80pct_snapshots", train_fit_labels)
    snapshot_rows += label_distribution_rows("validation_tail_20pct_snapshots", val_labels)
    snapshot_rows += label_distribution_rows("test_cf7_cf8_cf9_concat_snapshots", test_labels_all)

    window_rows: list[dict[str, object]] = []
    window_rows += label_distribution_rows("train_fit_head_80pct_windows", train_window_labels)
    window_rows += label_distribution_rows("validation_tail_20pct_windows", val_window_labels)
    window_rows += label_distribution_rows("test_cf7_cf8_cf9_concat_windows", test_window_labels)

    fields = ["split", "horizon", "class_label", "class_name", "count", "share"]
    write_csv(RESULTS / "label_distribution_snapshots.csv", snapshot_rows, fields)
    write_csv(RESULTS / "label_distribution_windows.csv", window_rows, fields)

    observed_total = int(train_labels_all.shape[0] + test_labels_all.shape[0])
    count_rows = [
        {
            "split": "train_cf7_all_snapshots",
            "num_events": int(train_labels_all.shape[0]),
            "num_windows": int(train_labels_all.shape[0] - WINDOW + 1),
        },
        {
            "split": "train_fit_head_80pct",
            "num_events": int(train_fit_labels.shape[0]),
            "num_windows": int(train_window_labels.shape[0]),
        },
        {
            "split": "validation_tail_20pct",
            "num_events": int(val_labels.shape[0]),
            "num_windows": int(val_window_labels.shape[0]),
        },
        {
            "split": "test_cf7_cf8_cf9_concat",
            "num_events": int(test_labels_all.shape[0]),
            "num_windows": int(test_window_labels.shape[0]),
        },
    ]
    write_csv(RESULTS / "window_counts.csv", count_rows, ["split", "num_events", "num_windows"])

    check_rows = [
        {
            "check": "official_zip_size_bytes",
            "expected": OFFICIAL_ZIP_SIZE,
            "observed": RAW_ZIP.stat().st_size,
            "status": "pass" if RAW_ZIP.stat().st_size == OFFICIAL_ZIP_SIZE else "fail",
        },
        {
            "check": "official_zip_sha256",
            "expected": OFFICIAL_ZIP_SHA256,
            "observed": sha256_file(RAW_ZIP),
            "status": "pass" if sha256_file(RAW_ZIP) == OFFICIAL_ZIP_SHA256 else "fail",
        },
        {
            "check": "published_total_representations",
            "expected": PUBLISHED_TOTAL_REPRESENTATIONS,
            "observed": observed_total,
            "status": "pass" if observed_total == PUBLISHED_TOTAL_REPRESENTATIONS else "fail",
        },
    ]
    write_csv(RESULTS / "dataset_integrity_checks.csv", check_rows, ["check", "expected", "observed", "status"])
    return count_rows


def write_published_comparisons() -> None:
    rows = [
        {
            "model": "DeepLOB",
            "parameters": "60k",
            "input": "100x40 raw LOB",
            "protocol": "Setup 2, first 7 days train, last 3 days test",
            "f1_k10": 83.40,
            "f1_k20": 72.82,
            "f1_k50": 80.35,
            "source": "Zhang, Zohren, Roberts 2018/2020 arXiv v6, Table II; parameters Table III",
            "source_url": "https://arxiv.org/pdf/1808.03668",
        },
        {
            "model": "SVM",
            "parameters": "N/A",
            "input": "FI-2010 features, as quoted by DeepLOB",
            "protocol": "Setup 2 as quoted by DeepLOB",
            "f1_k10": 35.88,
            "f1_k20": 43.20,
            "f1_k50": 49.42,
            "source": "Zhang, Zohren, Roberts 2018/2020 arXiv v6, Table II, citing [28]",
            "source_url": "https://arxiv.org/pdf/1808.03668",
        },
        {
            "model": "MLP",
            "parameters": "N/A",
            "input": "FI-2010 features, as quoted by DeepLOB",
            "protocol": "Setup 2 as quoted by DeepLOB",
            "f1_k10": 48.27,
            "f1_k20": 51.12,
            "f1_k50": 55.95,
            "source": "Zhang, Zohren, Roberts 2018/2020 arXiv v6, Table II, citing [28]",
            "source_url": "https://arxiv.org/pdf/1808.03668",
        },
        {
            "model": "LSTM",
            "parameters": "N/A",
            "input": "FI-2010 100x40-style sequence, as quoted by DeepLOB",
            "protocol": "Setup 2 as quoted by DeepLOB",
            "f1_k10": 66.33,
            "f1_k20": 62.37,
            "f1_k50": 61.43,
            "source": "Zhang, Zohren, Roberts 2018/2020 arXiv v6, Table II and III, citing [28]",
            "source_url": "https://arxiv.org/pdf/1808.03668",
        },
        {
            "model": "RR",
            "parameters": "N/A",
            "input": "144 FI-2010 representation features",
            "protocol": "Original FI-2010 anchored 9-fold, not Setup 2",
            "f1_k10": 41.00,
            "f1_k20": 44.00,
            "f1_k50": 42.70,
            "source": "Ntakaris et al. 2018 Journal of Forecasting, Table 7; DeepLOB Table I repeats k10/k50",
            "source_url": "https://pure.au.dk/portal/files/143396530/Ntakaris_et_al_2018_Journal_of_Forecasting.pdf",
        },
        {
            "model": "SLFN",
            "parameters": "N/A",
            "input": "144 FI-2010 representation features",
            "protocol": "Original FI-2010 anchored 9-fold, not Setup 2",
            "f1_k10": 32.70,
            "f1_k20": 32.70,
            "f1_k50": 45.90,
            "source": "Ntakaris et al. 2018 Journal of Forecasting, Table 7; DeepLOB Table I repeats k10/k50",
            "source_url": "https://pure.au.dk/portal/files/143396530/Ntakaris_et_al_2018_Journal_of_Forecasting.pdf",
        },
    ]
    fields = [
        "model",
        "parameters",
        "input",
        "protocol",
        "f1_k10",
        "f1_k20",
        "f1_k50",
        "source",
        "source_url",
    ]
    write_csv(RESULTS / "published_comparisons.csv", rows, fields)


def write_manifest(cache_rows: list[dict[str, object]], zip_rows: list[dict[str, object]]) -> None:
    lines = [
        "# FI-2010 Data Manifest",
        "",
        "Status: official FI-2010 archive was already available locally; this phase started from extraction and verification, with no download step.",
        "",
        "## Official Dataset",
        "",
        "- Dataset: Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods.",
        "- Dataset ID: `73eb48d7-4dbc-4a10-a52a-da745b47a649`.",
        "- Repository: Fairdata/Etsin.",
        "- Dataset page: <https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649>.",
        "- Paper: Ntakaris, Magris, Kanniainen, Gabbouj, and Iosifidis (2018), Journal of Forecasting.",
        "- License reported by Fairdata metadata: CC BY 4.0.",
        "",
        "## Local Archive Used",
        "",
        f"- Local path: `{RAW_ZIP.relative_to(ROOT)}`.",
        "- Acquisition note: the zip was already in hand before this Stage 1 run; the script does not download data and raises if the archive is missing.",
        f"- Size bytes: `{RAW_ZIP.stat().st_size}`.",
        f"- SHA256: `{sha256_file(RAW_ZIP)}`.",
        f"- Expected size bytes from Fairdata metadata: `{OFFICIAL_ZIP_SIZE}`.",
        f"- Expected SHA256 from Fairdata metadata: `{OFFICIAL_ZIP_SHA256}`.",
        "",
        "## Extracted Files Used In Stage 1",
        "",
        "| cache | source | size bytes | sha256 | events | features |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in cache_rows:
        lines.append(
            f"| `{row['cache_name']}` | `{row['source_path']}` | {row['source_size_bytes']} | `{row['source_sha256']}` | {row['num_events']} | {row['num_features_used']} |"
        )
    lines.extend(
        [
            "",
            "## NoAuction/Zscore Members In Official Zip",
            "",
            "Only `NoAuction/1.NoAuction_Zscore` is used. Other FI-2010 variants remain untouched.",
            "",
            "| member | uncompressed bytes | compressed bytes | crc |",
            "|---|---:|---:|---|",
        ]
    )
    for row in zip_rows:
        lines.append(
            f"| `{row['member']}` | {row['uncompressed_size_bytes']} | {row['compressed_size_bytes']} | `{row['crc']}` |"
        )
    lines.extend(
        [
            "",
            "## Protocol Notes",
            "",
            "- Setup 2 uses `Train_Dst_NoAuction_ZScore_CF_7.txt` for the first 7 days.",
            "- The last 3 days are represented by the concatenated testing files `CF_7`, `CF_8`, and `CF_9`.",
            "- The first 40 rows are used as raw LOB features; the last five rows are official labels.",
            "- No project-internal data and no NEW1-NEW5 split were read or used.",
            "",
            "Generated by `01_prepare_fi2010_stage1.py`.",
        ]
    )
    (ROOT / "data_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"Missing archive: {RAW_ZIP}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    zip_rows = zip_member_manifest()
    write_csv(
        RESULTS / "zip_noauction_zscore_members.csv",
        zip_rows,
        ["member", "uncompressed_size_bytes", "compressed_size_bytes", "crc"],
    )

    cache_rows = [
        cache_one("train_cf7", TRAIN_MEMBER),
        cache_one("test_cf7", TEST_MEMBERS[0]),
        cache_one("test_cf8", TEST_MEMBERS[1]),
        cache_one("test_cf9", TEST_MEMBERS[2]),
    ]
    write_csv(
        RESULTS / "file_manifest.csv",
        cache_rows,
        [
            "cache_name",
            "source_member",
            "source_path",
            "source_size_bytes",
            "source_sha256",
            "cache_path",
            "features_shape",
            "labels_shape",
            "num_events",
            "num_features_used",
        ],
    )
    count_rows = save_label_distributions(cache_rows)
    write_published_comparisons()
    write_manifest(cache_rows, zip_rows)
    summary = {"cache_rows": cache_rows, "window_counts": count_rows}
    (RESULTS / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
