import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 解决系统代理导致的访问错误（针对东方财富的数据接口）
DISABLE_PROXY = True
NO_PROXY_HOSTS = "push2his.eastmoney.com,eastmoney.com,push2.eastmoney.com,push2cdn.eastmoney.com"

if DISABLE_PROXY:
    os.environ["NO_PROXY"] = NO_PROXY_HOSTS
    os.environ["no_proxy"] = NO_PROXY_HOSTS

def get_hs300_cons():
    """获取沪深300成分股代码"""
    # 000300 是沪深300指数的代码
    df = ak.index_stock_cons(symbol="000300")
    # 兼容可能的列名变动
    code_col = "品种代码" if "品种代码" in df.columns else df.columns[0]
    name_col = "品种名称" if "品种名称" in df.columns else df.columns[1]
    
    return df[code_col].tolist(), df.set_index(code_col)[name_col].to_dict()

def get_stock_amount(symbol):
    """获取单只股票过去 30 个交易日的平均成交额"""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=60) # 多取几天以确保有30个交易日
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    try:
        # 尝试获取A股历史数据 (日线前复权)
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_str,
            end_date=end_str,
            adjust="qfq"
        )
        if df.empty:
            return symbol, 0.0
            
        # 取最近30个交易日
        df = df.tail(30)
        
        # '成交额' 列的单位是元
        avg_amount = df['成交额'].mean()
        return symbol, avg_amount
    except Exception as e:
        return symbol, 0.0

def main():
    print("1. 获取沪深300成分股列表...")
    try:
        symbols, name_dict = get_hs300_cons()
        print(f"   成功获取 {len(symbols)} 只成分股")
    except Exception as e:
        print(f"获取成分股失败: {e}")
        return
    
    print("2. 正在拉取各只股票近期历史数据，计算平均成交额(使用5线程)...")
    results = []
    
    # 使用多线程加速下载
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(get_stock_amount, sym): sym for sym in symbols}
        
        for future in tqdm(as_completed(futures), total=len(symbols), desc="下载进度"):
            sym, amount = future.result()
            results.append((sym, amount))
            
    print("3. 按流动性（近30日平均成交额）排序并取前 20 只...")
    # 排序，按成交额从高到低
    results.sort(key=lambda x: x[1], reverse=True)
    top20 = results[:20]
    
    print("\n========== 沪深300 流动性前 20 只股票 ==========")
    print(f"{'排名':<4} | {'代码':<8} | {'名称':<10} | {'近30日日均成交额(亿元)':<15}")
    print("-" * 55)
    
    df_res = []
    for i, (sym, amount) in enumerate(top20, 1):
        name = name_dict.get(sym, "未知")
        amount_yi = amount / 1e8 # 转换为亿元
        print(f"{i:<6} | {sym:<10} | {name:<12} | {amount_yi:.2f}")
        df_res.append({"rank": i, "symbol": sym, "name": name, "avg_amount_yi": amount_yi})
        
    # 保存结果
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top20_hs300_liquid.csv")
    pd.DataFrame(df_res).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 结果已保存至 {out_csv}")

if __name__ == "__main__":
    main()
