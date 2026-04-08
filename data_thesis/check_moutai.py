import akshare as ak
import os

DISABLE_PROXY = True
NO_PROXY_HOSTS = "push2his.eastmoney.com,eastmoney.com,push2.eastmoney.com,push2cdn.eastmoney.com"

if DISABLE_PROXY:
    os.environ["NO_PROXY"] = NO_PROXY_HOSTS
    os.environ["no_proxy"] = NO_PROXY_HOSTS

df = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20240301", end_date="20240407", adjust="qfq")
print(df.tail(10)[['日期', '成交额']])
