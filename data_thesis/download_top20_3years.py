import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os
import time

def fetch_history_data_sina(symbols_df, years=3):
    """提取前20支股票的近 N 年数据 (使用新浪数据源，东财易被代理拦截)"""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "top20_weight_3years_data")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"\n开始下载近 {years} 年历史数据 (从 {start_str} 到 {end_str})...")
    
    for _, row in symbols_df.iterrows():
        sym = str(row["symbol"]).zfill(6)
        name = row["name"]
        weight = row["weight"]
        
        file_path = os.path.join(DATA_DIR, f"{sym}_{name}.csv")
        if os.path.exists(file_path):
            print(f"[{sym} {name}] 数据已存在，跳过。")
            continue
            
        print(f"正在拉取 [{sym} {name}] (权重: {weight}%)...")
        for retry in range(3):
            try:
                # 转换新浪股票代码格式，如 sh600519, sz300750
                sina_symbol = f"sh{sym}" if sym.startswith("6") else f"sz{sym}"
                
                # 新浪接口：不依赖东财环境，代理下不易报错
                df = ak.stock_zh_a_daily(symbol=sina_symbol, start_date=start_str, end_date=end_str, adjust="qfq")
                if df.empty:
                    raise Exception("返回数据为空")
                    
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
                print(f"  -> 成功，共 {len(df)} 条数据")
                break
            except Exception as e:
                print(f"  -> 第 {retry+1} 次失败: {e}")
                time.sleep(2)
        time.sleep(1)
        
    print(f"\n全部下载完成！数据保存在: {DATA_DIR}")

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "top20_hs300_by_official_weight.csv")
    
    if not os.path.exists(csv_path):
        print(f"未找到 {csv_path}，请先运行上一步获取权重！")
        return
        
    top20_df = pd.read_csv(csv_path, dtype={"symbol": str})
    
    print("\n\n=============== 尝试使用新浪源拉取数据 ===============")
    print("由于东财接口在您的环境中被代理拦截，我们切换到新浪接口重试...")
    
    fetch_history_data_sina(top20_df, years=3)

if __name__ == "__main__":
    main()
