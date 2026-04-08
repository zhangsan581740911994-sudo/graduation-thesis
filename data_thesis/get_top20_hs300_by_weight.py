import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os
import time

# 代理设置
DISABLE_PROXY = True
NO_PROXY_HOSTS = "push2his.eastmoney.com,eastmoney.com,push2.eastmoney.com,push2cdn.eastmoney.com"
if DISABLE_PROXY:
    os.environ["NO_PROXY"] = NO_PROXY_HOSTS
    os.environ["no_proxy"] = NO_PROXY_HOSTS

def get_top20_by_weight():
    """获取沪深300官方最新权重排名前20的成分股"""
    print("正在从中证指数官方获取沪深300最新权重...")
    try:
        df = ak.index_stock_cons_weight_csindex(symbol="000300")
        df_sorted = df.sort_values(by="权重", ascending=False).reset_index(drop=True)
        top20 = df_sorted.head(20)[["成分券代码", "成分券名称", "权重"]]
        
        # 将列名重命名，方便后续使用
        top20 = top20.rename(columns={
            "成分券代码": "symbol",
            "成分券名称": "name",
            "权重": "weight"
        })
        
        print("\n==== 沪深300 官方最新权重前20名 ====")
        print(top20)
        return top20
    except Exception as e:
        print(f"获取权重失败: {e}")
        return pd.DataFrame()

def fetch_history_data(symbols_df, years=3):
    """提取前20支股票的近 N 年数据"""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "top20_weight_3years_data")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"\n开始下载近 {years} 年历史数据 (从 {start_str} 到 {end_str})...")
    
    for _, row in symbols_df.iterrows():
        sym = row["symbol"]
        name = row["name"]
        weight = row["weight"]
        
        file_path = os.path.join(DATA_DIR, f"{sym}_{name}.csv")
        if os.path.exists(file_path):
            print(f"[{sym} {name}] 数据已存在，跳过。")
            continue
            
        print(f"正在拉取 [{sym} {name}] (权重: {weight}%)...")
        for retry in range(3):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=sym,
                    period="daily",
                    start_date=start_str,
                    end_date=end_str,
                    adjust="qfq"
                )
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
                print(f"  -> 成功，共 {len(df)} 条数据")
                break
            except Exception as e:
                print(f"  -> 第 {retry+1} 次失败: {e}")
                time.sleep(1)
        time.sleep(0.5)
        
    print(f"\n全部下载完成！数据保存在: {DATA_DIR}")

def main():
    top20_df = get_top20_by_weight()
    if not top20_df.empty:
        # 保存前20名名单
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        top20_df.to_csv(os.path.join(BASE_DIR, "top20_hs300_by_official_weight.csv"), index=False, encoding="utf-8-sig")
        
        # 提取近3年数据
        fetch_history_data(top20_df, years=3)

if __name__ == "__main__":
    main()
