"""
读取 csi1000_top50_weight_3years_data/ 下个股 CSV，共同交易日对齐后切分 train/val/test。

输出：
  csi1000_top50_offline_full.csv
  csi1000_top50_offline_train.csv
  csi1000_top50_offline_val.csv
  csi1000_top50_offline_test.csv

用法：
  python data_thesis/preprocess_csi1000_top50_offline_dataset.py
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "csi1000_top50_weight_3years_data"

FULL_OUT = BASE_DIR / "csi1000_top50_offline_full.csv"
TRAIN_OUT = BASE_DIR / "csi1000_top50_offline_train.csv"
VAL_OUT = BASE_DIR / "csi1000_top50_offline_val.csv"
TEST_OUT = BASE_DIR / "csi1000_top50_offline_test.csv"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 东财 stock_zh_a_hist 常为中文列名；新浪部分环境为英文。统一成下游使用的英文列。
_EASTMONEY_TO_EN = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将日线行情表规范为列 date,open,high,low,close,volume,amount（字符串日期可保留，下游再 to_datetime）。"""
    if df is None or df.empty:
        raise ValueError("empty dataframe")
    out = df.copy()
    for zh, en in _EASTMONEY_TO_EN.items():
        if zh in out.columns and en not in out.columns:
            out = out.rename(columns={zh: en})
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"缺少字段: {sorted(missing)}")
    return out[list(required)].copy()


def _read_one_stock_csv(path: Path) -> pd.DataFrame:
    symbol = path.stem.split("_", 1)[0]
    df = pd.read_csv(path)
    df = normalize_ohlcv_columns(df)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df["symbol"] = symbol
    keep_cols = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    return df[keep_cols]


def build_aligned_dataset() -> pd.DataFrame:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")
    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"目录中没有任何 .csv 股票文件: {INPUT_DIR}。"
            "请先成功运行 python data_thesis/get_csi1000_top50_by_weight.py（云主机建议 pip install baostock）。"
        )
    frames = [_read_one_stock_csv(f) for f in csv_files]
    df_all = pd.concat(frames, ignore_index=True)
    n_symbols = df_all["symbol"].nunique()
    day_counts = df_all.groupby("date")["symbol"].nunique()
    common_dates = day_counts[day_counts == n_symbols].index
    df_all = df_all[df_all["date"].isin(common_dates)].copy()
    df_all = df_all.sort_values(["symbol", "date"]).reset_index(drop=True)
    df_all["return"] = df_all.groupby("symbol")["close"].pct_change()
    df_all = df_all.dropna(subset=["return"]).reset_index(drop=True)
    return df_all


def split_by_date(df: pd.DataFrame):
    unique_dates = sorted(df["date"].unique())
    n = len(unique_dates)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])
    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    return train_df, val_df, test_df


def main():
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO 必须等于 1")
    df = build_aligned_dataset()
    df.to_csv(FULL_OUT, index=False, encoding="utf-8-sig")
    train_df, val_df, test_df = split_by_date(df)
    train_df.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_OUT, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_OUT, index=False, encoding="utf-8-sig")
    print("预处理完成。")
    print(f"股票数: {df['symbol'].nunique()}")
    print(
        f"日期区间: {df['date'].min().date()} ~ {df['date'].max().date()}, "
        f"交易日数: {df['date'].nunique()}"
    )
    print(f"train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)} 行")


if __name__ == "__main__":
    main()
