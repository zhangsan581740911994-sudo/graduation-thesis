#!/usr/bin/env python3
"""
汇总「高噪声消融」实验：扫描目录下各次 run 的 progress.csv，输出对比表。

默认根目录：项目根下的 outputs_portfolio_ablation_noise（与 scripts/run_ablation_noise.sh 一致）

用法（在 graduation-thesis 项目根目录）：
  python scripts/summarize_ablation_noise.py
  python scripts/summarize_ablation_noise.py --root /root/autodl-tmp/graduation-thesis/outputs_portfolio_ablation_noise
  python scripts/summarize_ablation_noise.py --root . --json-out ablation_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_run_name(name: str) -> dict:
    """从目录名推断算法 / seed / 噪声标签（启发式）。"""
    out = {"raw": name, "algo": "?", "seed": None, "noise_tag": None}
    lower = name.lower()
    if "pure_td3" in lower or "td3" in lower and "edp" not in lower:
        out["algo"] = "pure_td3"
    if "edp" in lower:
        out["algo"] = "edp"
    m = re.search(r"_s(\d+)_", name)
    if m:
        out["seed"] = int(m.group(1))
    m = re.search(r"noise(\d+)", lower)
    if m:
        out["noise_tag"] = f"{m.group(1)}%"
    if "noise80" in lower:
        out["noise_tag"] = "80%"
    return out


def _best_metrics(df: pd.DataFrame) -> dict:
    """从 progress 表提取 best / last eval 指标。"""
    cols = list(df.columns)
    best_cols = [c for c in cols if "best_normalized_return" in c]
    avg_cols = [c for c in cols if "average_normalizd_return" in c or "average_normalized_return" in c]

    def _col_max(c: str):
        s = pd.to_numeric(df[c], errors="coerce")
        return float(s.max()) if len(s) else float("nan")

    def _col_last(c: str):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        return float(s.iloc[-1]) if len(s) else float("nan")

    row = {
        "epochs_logged": len(df),
        "best_normalized_return_max": float("nan"),
        "last_average_normalized_return": float("nan"),
    }
    if best_cols:
        row["best_normalized_return_max"] = max(_col_max(c) for c in best_cols)
    # 主评估列：优先无后缀的 average_normalizd_return
    for prefer in ["average_normalizd_return", "average_normalized_return"]:
        matches = [c for c in avg_cols if c == prefer]
        if matches:
            row["last_average_normalized_return"] = _col_last(matches[0])
            break
    else:
        if avg_cols:
            row["last_average_normalized_return"] = _col_last(avg_cols[0])

    return row


def summarize(root: Path) -> list[dict]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    rows: list[dict] = []
    for progress in sorted(root.rglob("progress.csv")):
        run_dir = progress.parent
        try:
            name = str(run_dir.relative_to(root)).replace(os.sep, "/")
        except ValueError:
            name = str(run_dir).replace(os.sep, "/")
        meta = _parse_run_name(run_dir.name)
        try:
            df = pd.read_csv(progress)
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "run_dir": name,
                    "error": str(e),
                    **meta,
                }
            )
            continue
        m = _best_metrics(df)
        variant_path = run_dir / "variant.json"
        variant = {}
        if variant_path.is_file():
            try:
                variant = json.loads(variant_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        seed = meta["seed"]
        if seed is None and isinstance(variant, dict):
            seed = variant.get("seed")

        rows.append(
            {
                "run_dir": name,
                "algo_guess": meta["algo"],
                "seed": seed,
                "noise_tag": meta["noise_tag"],
                **m,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize ablation noise runs (progress.csv).")
    ap.add_argument(
        "--root",
        type=str,
        default=str(_project_root() / "outputs_portfolio_ablation_noise"),
        help="Root directory containing experiment subfolders (default: ./outputs_portfolio_ablation_noise)",
    )
    ap.add_argument("--json-out", type=str, default="", help="Optional path to write JSON summary.")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"目录不存在: {root.resolve()}")
        print("请先跑完 scripts/run_ablation_noise.sh，或用 --root 指向实际输出目录。")
        return 1

    try:
        rows = summarize(root)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if not rows:
        print(f"No progress.csv found under: {root}")
        print("确认已跑完消融且输出目录正确；或用 --root 指定路径。")
        return 0

    # 控制台表
    df = pd.DataFrame(rows)
    cols = [
        "run_dir",
        "algo_guess",
        "seed",
        "epochs_logged",
        "best_normalized_return_max",
        "last_average_normalized_return",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    show = df[cols].copy()
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)
    print(f"Root: {root.resolve()}\n")
    print(show.to_string(index=False))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
