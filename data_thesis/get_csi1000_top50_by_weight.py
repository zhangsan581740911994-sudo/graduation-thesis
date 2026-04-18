"""
中证1000：按官方最新权重取 Top50 成分股，并下载近 N 年日线（前复权）。

指数代码：000852（中证 CSI 1000）
输出：
  - csi1000_top50_by_official_weight.csv  名单
  - csi1000_top50_weight_3years_data/*.csv  个股行情

用法：
  cd 项目根目录
  python data_thesis/get_csi1000_top50_by_weight.py

依赖：akshare；若东财接口失败，可改用 download_top20_3years.py 中的新浪源思路改成本脚本。
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

# 与 get_top20_hs300_by_weight.py 一致：代理环境下东财可用
DISABLE_PROXY = True
NO_PROXY_HOSTS = (
    "push2his.eastmoney.com,eastmoney.com,push2.eastmoney.com,"
    "push2cdn.eastmoney.com"
)
if DISABLE_PROXY:
    os.environ["NO_PROXY"] = NO_PROXY_HOSTS
    os.environ["no_proxy"] = NO_PROXY_HOSTS

# 中证1000
CSI1000_INDEX = "000852"
TOP_N = 50
YEARS = 3


def get_top50_by_weight() -> pd.DataFrame:
    print(f"正在从中证指数获取 中证1000({CSI1000_INDEX}) 官方权重 Top{TOP_N} ...")
    df = ak.index_stock_cons_weight_csindex(symbol=CSI1000_INDEX)
    df_sorted = df.sort_values(by="权重", ascending=False).reset_index(drop=True)
    top = df_sorted.head(TOP_N)[["成分券代码", "成分券名称", "权重"]].copy()
    top = top.rename(
        columns={"成分券代码": "symbol", "成分券名称": "name", "权重": "weight"}
    )
    top["symbol"] = top["symbol"].astype(str).str.zfill(6)
    print(top.head(10))
    print("...")
    print(f"共 {len(top)} 只")
    return top


def fetch_history(symbols_df: pd.DataFrame, years: int = YEARS) -> Path:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    base = Path(__file__).resolve().parent
    data_dir = base / "csi1000_top50_weight_3years_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n下载近 {years} 年日线: {start_str} ~ {end_str}")
    print(f"保存目录: {data_dir}")

    for _, row in symbols_df.iterrows():
        sym = row["symbol"]
        name = str(row["name"])
        w = row["weight"]
        fpath = data_dir / f"{sym}_{name}.csv"
        if fpath.exists() and fpath.stat().st_size > 0:
            print(f"[skip] {sym} {name}")
            continue
        print(f"[fetch] {sym} {name} 权重={w}%")
        ok = False
        for retry in range(3):
            try:
                d = ak.stock_zh_a_hist(
                    symbol=sym,
                    period="daily",
                    start_date=start_str,
                    end_date=end_str,
                    adjust="qfq",
                )
                if d is None or d.empty:
                    raise RuntimeError("empty")
                d.to_csv(fpath, index=False, encoding="utf-8-sig")
                print(f"  -> ok rows={len(d)}")
                ok = True
                break
            except Exception as e:
                print(f"  -> retry {retry + 1}: {e}")
                time.sleep(2)
        if not ok:
            print(f"  -> FAIL {sym} {name}")
        time.sleep(0.5)

    return data_dir


def main():
    top = get_top50_by_weight()
    if top.empty:
        return
    base = Path(__file__).resolve().parent
    list_path = base / "csi1000_top50_by_official_weight.csv"
    top.to_csv(list_path, index=False, encoding="utf-8-sig")
    print(f"\n名单已保存: {list_path}")
    fetch_history(top, years=YEARS)
    print("\n完成。下一步运行: python data_thesis/preprocess_csi1000_top50_offline_dataset.py")


if __name__ == "__main__":
    main()
