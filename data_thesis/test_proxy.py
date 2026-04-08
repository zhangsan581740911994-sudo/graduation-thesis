import akshare as ak
import os
import urllib.request

# 清除所有代理环境变量
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(k, None)

# 设置空代理以防止 urllib 使用系统注册表代理
proxy_support = urllib.request.ProxyHandler({})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)

try:
    df = ak.stock_zh_a_spot_em()
    print("spot_em ok! 行数:", len(df))
    print(df.head())
except Exception as e:
    print("spot_em 失败:", e)

try:
    df2 = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20240301", end_date="20240407", adjust="qfq")
    print("hist ok! 行数:", len(df2))
except Exception as e:
    print("hist 失败:", e)
