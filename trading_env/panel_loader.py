"""读取组合收益率面板：兼容宽表与离线长表 split。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _load_wide_panel(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """宽表：首列 date，其余列均为各标的收益率。"""
    tickers = [c for c in df.columns if c != "date"]
    if not tickers:
        raise ValueError("No return columns besides 'date'.")
    df = df.dropna(subset=tickers, how="any")
    if df.empty:
        raise ValueError("No rows left after dropping NA returns.")
    returns = df[tickers].to_numpy(dtype=np.float32)
    return returns, tickers


def _load_long_split(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    长表 split：至少包含 date/symbol/return 三列。
    通过 pivot 转为 (T, N) 的收益率矩阵。
    """
    required_cols = {"date", "symbol", "return"}
    if not required_cols.issubset(df.columns):
        missing = sorted(required_cols - set(df.columns))
        raise ValueError(f"Long split CSV missing columns: {missing}")

    work = df[["date", "symbol", "return"]].copy()
    work["symbol"] = work["symbol"].astype(str).str.zfill(6)
    work = work.dropna(subset=["date", "symbol", "return"])

    panel = work.pivot_table(
        index="date",
        columns="symbol",
        values="return",
        aggfunc="last",
    ).sort_index()

    panel = panel.dropna(axis=0, how="any")
    if panel.empty:
        raise ValueError("No aligned rows left after pivot/dropna in long split CSV.")

    tickers = list(panel.columns.astype(str))
    returns = panel.to_numpy(dtype=np.float32)
    return returns, tickers


def load_returns_panel_csv(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    读取 CSV，自动兼容两种格式：
    1) 宽表：首列 `date`，其余列为各标的日简单收益率（已对齐交易日）；
    2) 长表 split：包含 `date/symbol/return` 三列（例如离线预处理输出）。

    Returns
    -------
    returns : ndarray, shape (T, N)
    tickers : list of column names (excluding date)
    """
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("Panel CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 长表 split：含 symbol + return
    if "symbol" in df.columns and "return" in df.columns:
        return _load_long_split(df)

    # 宽表
    return _load_wide_panel(df)
