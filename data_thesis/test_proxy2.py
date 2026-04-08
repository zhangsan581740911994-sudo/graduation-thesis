import akshare as ak
import os

# 彻底禁用 requests 库的代理
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

try:
    df = ak.stock_zh_a_spot_em()
    print("spot_em ok! 行数:", len(df))
except Exception as e:
    print("spot_em 失败:", e)

try:
    df2 = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20240301", end_date="20240407", adjust="qfq")
    print("hist ok! 行数:", len(df2))
    print(df2.tail(5)[['日期', '成交额']])
except Exception as e:
    print("hist 失败:", e)
