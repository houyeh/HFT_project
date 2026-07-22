from __future__ import annotations

import html
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "34_report_expansion_figures_and_methods"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

V18 = PROJECT_ROOT / "new_era" / "18_queue_aware"
V21 = PROJECT_ROOT / "new_era" / "21_stress_robust_policy"
V28 = PROJECT_ROOT / "new_era" / "28_residual_quote_ev_veto"
V30 = PROJECT_ROOT / "new_era" / "30_ml_training_adequacy_audit"
V31 = PROJECT_ROOT / "new_era" / "31_opportunity_cost_audit"
V32 = PROJECT_ROOT / "new_era" / "32_final_falsification_audits"
V33 = PROJECT_ROOT / "new_era" / "33_corrected_final_falsification_audits"

REQUIRED_PRESENTATION_COLUMNS = ["side", "spread_ticks", "quote_ev_moderate", "eval_group"]


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_placeholder_figure(path: Path, title: str, message: str) -> None:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.40, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#30343b",
            "axes.labelcolor": "#20242b",
            "xtick.color": "#20242b",
            "ytick.color": "#20242b",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#d9dee7",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def fmt_money(value: float) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    sign = "-" if value < 0 else ""
    return f"{sign}{abs(value):,.0f}"


def load_full_action_universe() -> pd.DataFrame:
    df = read_parquet(V31 / "data" / "full_action_universe.parquet")
    df["eval_group"] = "phase031_" + df["phase031_split"].astype(str)
    df["quote_ev_moderate"] = df["quote_ev_moderate"].astype(float)
    return df


def load_unused_oos_actions() -> pd.DataFrame:
    path = PROJECT_ROOT / "new_era" / "29_bid_only_all_unused_oos" / "results" / "bid_only_unused_oos_action_details.parquet"
    expected_parent = PROJECT_ROOT / "new_era" / "29_bid_only_all_unused_oos"
    if expected_parent not in path.parents:
        raise AssertionError(f"Unexpected unused OOS path: {path}")
    df = read_parquet(path)
    df = df.rename(columns={"quote_value_moderate": "quote_ev_moderate"})
    df["eval_group"] = "unused_" + df["data_dir"].astype(str) + "_" + df["symbol"].astype(str)
    df["phase031_split"] = "unused_oos"
    return df


def load_presentation_action_universe() -> pd.DataFrame:
    phase = load_full_action_universe()
    phase = phase[phase["phase031_split"].isin(["test", "new4"])].copy()
    phase_splits = set(phase["phase031_split"].astype(str).unique())
    if not phase_splits.issubset({"test", "new4"}):
        raise AssertionError(f"Phase031 presentation rows must only include test/new4, got {sorted(phase_splits)}")
    unused = load_unused_oos_actions()
    keep = sorted(set(phase.columns).intersection(unused.columns))
    out = pd.concat([phase[keep], unused[keep]], ignore_index=True)
    assert_required_columns(out, REQUIRED_PRESENTATION_COLUMNS, "presentation action universe")
    counts = (
        out.groupby(["phase031_split", "eval_group"], observed=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["phase031_split", "eval_group"])
    )
    print("Phase034 presentation action universe counts:")
    print(counts.to_string(index=False))
    out["side"] = out["side"].astype(str)
    out["spread_ticks"] = out["spread_ticks"].astype(float)
    out["selected_baseline"] = (out["side"].eq("bid") & (out["spread_ticks"] >= 1.0))
    return out


def assert_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise AssertionError(f"{name} missing required columns: {missing}")


def find_existing_digest() -> tuple[Path | None, Path | None]:
    candidates = []
    for name in ["Week14_HFT_FPGA_Advisor_Digest.md", "Week14_HFT_FPGA_Advisor_Digest.html"]:
        candidates.extend(PROJECT_ROOT.glob(f"**/{name}"))
    md = next((p for p in candidates if p.suffix.lower() == ".md"), None)
    html_path = next((p for p in candidates if p.suffix.lower() == ".html"), None)
    return md, html_path


def markdown_to_html(markdown_text: str, title: str) -> str:
    body, renderer_name = render_markdown(markdown_text)
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1120px;margin:36px auto;padding:0 24px;line-height:1.55;color:#17202a}"
        "h1,h2,h3{color:#111827} img{max-width:100%;border:1px solid #d6dae1;margin:12px 0 24px 0}"
        "code{background:#f3f4f6;padding:2px 4px;border-radius:3px} table{border-collapse:collapse} th,td{border:1px solid #d6dae1;padding:6px 8px}</style>"
        f"</head><body><p style=\"color:#667085;font-size:12px\">Rendered with {html.escape(renderer_name)}.</p>"
        + body
        + "</body></html>"
    )


def markdown_to_simple_html(markdown_text: str, title: str) -> str:
    return markdown_to_html(markdown_text, title)


def render_markdown(markdown_text: str) -> tuple[str, str]:
    try:
        import markdown as py_markdown

        return (
            py_markdown.markdown(
                markdown_text,
                extensions=["extra", "sane_lists", "tables", "toc"],
                output_format="html5",
            ),
            "python-markdown",
        )
    except ModuleNotFoundError:
        pass
    try:
        import mistune

        renderer = mistune.create_markdown(plugins=["table", "strikethrough", "url"])
        return renderer(markdown_text), "mistune"
    except ModuleNotFoundError:
        pass
    try:
        from markdown_it import MarkdownIt

        return MarkdownIt("commonmark", {"html": False}).enable("table").render(markdown_text), "markdown-it-py"
    except ModuleNotFoundError:
        raise RuntimeError("No real markdown renderer found. Install python-markdown, mistune, or markdown-it-py.")


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return True
    return False
