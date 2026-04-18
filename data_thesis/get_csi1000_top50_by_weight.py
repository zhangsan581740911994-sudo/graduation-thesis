"""
中证1000：按官方最新权重取 Top50 成分股，并下载近 N 年日线（前复权）。

指数代码：000852（中证 CSI 1000）
输出：
  - csi1000_top50_by_official_weight.csv  名单
  - csi1000_top50_weight_3years_data/*.csv  个股行情

日线拉取策略（云主机上东财/新浪 HTTP 常 RemoteDisconnected）：
  1) 优先 BaoStock（baostock，独立服务，AutoDL 上通常比 akshare 新浪接口稳）
  2) 失败再试 akshare.stock_zh_a_daily（新浪）

依赖：akshare；建议 pip install baostock

用法：
  cd 项目根目录
  python data_thesis/get_csi1000_top50_by_weight.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from preprocess_csi1000_top50_offline_dataset import normalize_ohlcv_columns

try:
    import baostock as bs

    _HAS_BAOSTOCK = True
except ImportError:
    bs = None  # type: ignore[assignment]
    _HAS_BAOSTOCK = False

# 中证1000
CSI1000_INDEX = "000852"
TOP_N = 50
YEARS = 3

AKSHARE_RETRIES = 10
SLEEP_BETWEEN_RETRIES = 2.5
SLEEP_BETWEEN_STOCKS = 1.5


def _to_sina_symbol(sym: str) -> str:
    sym = str(sym).zfill(6)
    return f"sh{sym}" if sym.startswith("6") else f"sz{sym}"


def _to_baostock_code(sym: str) -> str:
    """BaoStock 代码：6 开头上海 sh，其余深圳 sz。"""
    s = str(sym).zfill(6)
    return f"sh.{s}" if s.startswith("6") else f"sz.{s}"


def _ymd_compact_to_hyphen(ymd: str) -> str:
    """20230419 -> 2023-04-19"""
    ymd = ymd.strip()
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"


def _try_normalize(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    try:
        return normalize_ohlcv_columns(df)
    except ValueError:
        return None


def _fetch_baostock(sym: str, start_hyphen: str, end_hyphen: str) -> pd.DataFrame | None:
    if not _HAS_BAOSTOCK or bs is None:
        return None
    code = _to_baostock_code(sym)
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume,amount",
        start_date=start_hyphen,
        end_date=end_hyphen,
        frequency="d",
        adjustflag="2",  # 前复权
    )
    if rs.error_code != "0":
        return None
    rows: list[list[str]] = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    df = pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "volume", "amount"]
    )
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume", "amount"])
    return _try_normalize(df)


def _fetch_sina(sym: str, start_str: str, end_str: str) -> pd.DataFrame | None:
    sina_sym = _to_sina_symbol(sym)
    for retry in range(AKSHARE_RETRIES):
        try:
            df = ak.stock_zh_a_daily(
                symbol=sina_sym,
                start_date=start_str,
                end_date=end_str,
                adjust="qfq",
            )
            out = _try_normalize(df)
            if out is not None:
                return out
            raise RuntimeError("empty or unparseable columns")
        except Exception as e:
            print(f"  -> akshare(新浪) retry {retry + 1}/{AKSHARE_RETRIES}: {e}")
            time.sleep(SLEEP_BETWEEN_RETRIES)
    return None


def _csv_on_disk_ok(fpath: Path) -> bool:
    if not fpath.exists() or fpath.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(fpath)
        normalize_ohlcv_columns(df)
        return True
    except Exception:
        return False


def fetch_history(symbols_df: pd.DataFrame, years: int = YEARS) -> Path:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    start_hyphen = _ymd_compact_to_hyphen(start_str)
    end_hyphen = _ymd_compact_to_hyphen(end_str)

    base = Path(__file__).resolve().parent
    data_dir = base / "csi1000_top50_weight_3years_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n下载近 {years} 年日线: {start_str} ~ {end_str}")
    print(f"保存目录: {data_dir}")
    if _HAS_BAOSTOCK:
        print("数据源: BaoStock 优先，失败则 akshare 新浪")
    else:
        print("数据源: 仅 akshare 新浪（建议: pip install baostock）")

    bs_logged_in = False
    if _HAS_BAOSTOCK and bs is not None:
        lg = bs.login()
        if lg.error_code == "0":
            bs_logged_in = True
        else:
            print(f"[warn] baostock 登录失败: {lg.error_msg}，将仅用 akshare")

    try:
        for _, row in symbols_df.iterrows():
            sym = row["symbol"]
            name = str(row["name"])
            w = row["weight"]
            fpath = data_dir / f"{sym}_{name}.csv"

            if _csv_on_disk_ok(fpath):
                print(f"[skip] {sym} {name}（已有有效 CSV）")
                time.sleep(SLEEP_BETWEEN_STOCKS)
                continue
            if fpath.exists():
                print(f"[warn] 删除无效/旧文件: {fpath.name}")
                fpath.unlink(missing_ok=True)

            print(f"[fetch] {sym} {name} 权重={w}%")
            out = None
            source = ""
            if bs_logged_in:
                try:
                    out = _fetch_baostock(sym, start_hyphen, end_hyphen)
                    if out is not None:
                        source = "baostock"
                except Exception as e:
                    print(f"  -> baostock 异常: {e}")
            if out is None:
                out = _fetch_sina(sym, start_str, end_str)
                if out is not None:
                    source = "akshare_sina"

            if out is None:
                print(f"  -> FAIL {sym} {name}")
            else:
                out.to_csv(fpath, index=False, encoding="utf-8-sig")
                print(f"  -> ok rows={len(out)} via {source}")

            time.sleep(SLEEP_BETWEEN_STOCKS)
    finally:
        if bs_logged_in and bs is not None:
            bs.logout()

    return data_dir


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


def _print_startup_banner() -> None:
    """便于在 AutoDL 上确认：已拉取含 BaoStock 的脚本，且已 pip install baostock。"""
    line = "=" * 72
    print(line)
    print(" get_csi1000_top50_by_weight  |  日线优先 BaoStock，其次 akshare 新浪")
    print(f" BaoStock: {'已安装（推荐）' if _HAS_BAOSTOCK else '未安装 — 请先: pip install baostock'}")
    print(f" akshare 失败重试次数: {AKSHARE_RETRIES}（若你看到本行仍是 retry 1/2/3，说明代码未更新，请 git pull）")
    print(line)


if __name__ == "__main__":
    _print_startup_banner()
    main()
