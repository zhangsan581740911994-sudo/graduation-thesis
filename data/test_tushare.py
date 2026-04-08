import os
import tushare as ts

token = os.environ.get("TUSHARE_TOKEN")
if not token:
    raise ValueError("没有读到 TUSHARE_TOKEN 环境变量")

pro = ts.pro_api(token)

df = pro.index_daily(
    ts_code="000300.SH",
    start_date="20240101",
    end_date="20241231"
)

print(df.head())
print("rows:", len(df))