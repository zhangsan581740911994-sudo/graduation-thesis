import akshare as ak
import pandas as pd
from datetime import datetime
import os
import time
from pathlib import Path


# ====================== 基本配置 ======================
CODE = "510300"  # 沪深300ETF（华泰柏瑞），可按需替换
SINA_SYMBOL = "sh510300"  # sina 接口常用格式
START_DATE = "20180101"  # 建议至少 3–5 年
END_DATE = datetime.today().strftime("%Y%m%d")

# 与脚本同目录（data_thesis/），不依赖本机绝对路径
BASE_DIR = Path(__file__).resolve().parent
BASE_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = str(BASE_DIR)

RAW_CSV = os.path.join(BASE_DIR, "510300_hs300_etf_daily_raw.csv")
CLEAN_CSV = os.path.join(BASE_DIR, "510300_hs300_etf_daily_clean.csv")
TRAIN_CSV = os.path.join(BASE_DIR, "510300_hs300_etf_train.csv")
VAL_CSV = os.path.join(BASE_DIR, "510300_hs300_etf_val.csv")
TEST_CSV = os.path.join(BASE_DIR, "510300_hs300_etf_test.csv")
DISABLE_PROXY = True
NO_PROXY_HOSTS = "push2his.eastmoney.com,eastmoney.com"


def download_from_akshare() -> pd.DataFrame:
    """从 AkShare 拉取 ETF 日线数据（优先东财，失败回退新浪）。"""
    old_no_proxy = os.environ.get("NO_PROXY")
    old_no_proxy_lower = os.environ.get("no_proxy")

    if DISABLE_PROXY:
        # 仅对东方财富域名直连，不影响其它请求继续使用系统代理
        os.environ["NO_PROXY"] = NO_PROXY_HOSTS
        os.environ["no_proxy"] = NO_PROXY_HOSTS

    last_err = None
    try:
        df = None

        # 方案 A：东方财富（日线，支持前复权）
        for i in range(3):
            try:
                df = ak.fund_etf_hist_em(
                    symbol=CODE,
                    period="daily",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    adjust="qfq",
                )
                break
            except Exception as e:
                last_err = e
                print(f"[EM] 第 {i + 1} 次下载失败: {e}")
                time.sleep(1.5)

        # 方案 B：新浪回退（不支持 qfq，但可保证先拿到数据）
        if df is None:
            print("\n正在尝试回退数据源: fund_etf_hist_sina ...")
            for i in range(3):
                try:
                    df_sina = ak.fund_etf_hist_sina(symbol=SINA_SYMBOL)
                    df_sina["日期"] = pd.to_datetime(df_sina["date"]).dt.strftime("%Y-%m-%d")
                    df_sina["开盘"] = df_sina["open"]
                    df_sina["收盘"] = df_sina["close"]
                    df_sina["最高"] = df_sina["high"]
                    df_sina["最低"] = df_sina["low"]
                    df_sina["成交量"] = df_sina.get("volume", pd.NA)
                    df_sina["成交额"] = df_sina.get("amount", pd.NA)
                    df = df_sina[["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]].copy()

                    start_dt = pd.to_datetime(START_DATE, format="%Y%m%d")
                    end_dt = pd.to_datetime(END_DATE, format="%Y%m%d")
                    df["日期"] = pd.to_datetime(df["日期"])
                    df = df[(df["日期"] >= start_dt) & (df["日期"] <= end_dt)].copy()
                    df["日期"] = df["日期"].dt.strftime("%Y-%m-%d")
                    print("已切换到新浪数据源。")
                    break
                except Exception as e:
                    last_err = e
                    print(f"[SINA] 第 {i + 1} 次下载失败: {e}")
                    time.sleep(1.5)

        if df is None:
            raise RuntimeError(
                "AkShare 下载失败：东方财富与新浪数据源均不可用。"
                f"原始错误: {last_err}"
            )
    finally:
        # 恢复原始环境变量，确保不影响后续进程行为
        if old_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = old_no_proxy

        if old_no_proxy_lower is None:
            os.environ.pop("no_proxy", None)
        else:
            os.environ["no_proxy"] = old_no_proxy_lower

    print("原始列名：", df.columns.tolist())
    df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    print(f"原始数据已保存：{RAW_CSV}（{len(df)} 行）")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名为英文，保留核心字段并按日期排序。"""
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns=rename_map)
    keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    df = df[keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_returns_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """计算简单收益率并去除缺失行。"""
    df["return"] = df["close"].pct_change()
    df = df.dropna().reset_index(drop=True)
    return df


def split_train_val_test(df: pd.DataFrame) -> None:
    """按时间切分 train/val/test，并分别保存为 csv。"""
    # 示例划分：train 到 2022，val 为 2023 上半年，test 为 2023 下半年及以后
    train_end = pd.Timestamp("2022-12-31")
    val_end = pd.Timestamp("2023-06-30")

    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]

    train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    val.to_csv(VAL_CSV, index=False, encoding="utf-8-sig")
    test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    print("\n📊 数据集切分（可直接写论文）")
    print(f"训练集: {len(train)} 行, 至 {train['date'].max().date() if not train.empty else 'N/A'}")
    print(f"验证集: {len(val)} 行, 至 {val['date'].max().date() if not val.empty else 'N/A'}")
    print(f"测试集: {len(test)} 行, 至 {test['date'].max().date() if not test.empty else 'N/A'}")
    print(f"TRAIN 保存于: {TRAIN_CSV}")
    print(f"VAL   保存于: {VAL_CSV}")
    print(f"TEST  保存于: {TEST_CSV}")


def main() -> None:
    df_raw = download_from_akshare()
    df = normalize_columns(df_raw)
    df = add_returns_and_clean(df)
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 清洗后数据已保存：{CLEAN_CSV}（{len(df)} 行）")

    split_train_val_test(df)


if __name__ == "__main__":
    main()