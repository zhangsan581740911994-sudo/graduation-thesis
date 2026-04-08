import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from tqdm import tqdm

# ==================== 网络代理配置 ====================
# 如果您的网络直连总是报错 RemoteDisconnected，请保留默认设置，使用系统代理；
# 如果使用系统代理报错 ProxyError，可以取消下面两行的注释以直连：
# os.environ['NO_PROXY'] = '*'
# os.environ['no_proxy'] = '*'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONS_FILE = os.path.join(BASE_DIR, "hs300_cons.csv")
DATA_DIR = os.path.join(BASE_DIR, "hs300_daily_data")

def step1_get_components():
    """步骤一：获取并保存沪深300成分股"""
    print("=== 步骤一：获取沪深300成分股 ===")
    if os.path.exists(CONS_FILE):
        df = pd.read_csv(CONS_FILE, dtype=str)
        print(f"已存在成分股缓存文件，直接加载 {len(df)} 只股票。")
        return df
    
    print("正在从 akshare 拉取成分股...")
    # 000300 是沪深300指数的代码
    df = ak.index_stock_cons(symbol="000300")
    df.to_csv(CONS_FILE, index=False, encoding="utf-8-sig")
    print(f"成功拉取并保存到 {CONS_FILE}")
    return df

def step2_fetch_stock_data(df_cons):
    """步骤二：分批拉取历史数据，带断点续传"""
    print("\n=== 步骤二：逐个拉取股票历史数据 ===")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    code_col = "品种代码" if "品种代码" in df_cons.columns else df_cons.columns[0]
    symbols = df_cons[code_col].tolist()
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=60) # 获取近60天数据，保证有30个交易日
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    fail_list = []
    
    for sym in tqdm(symbols, desc="数据拉取进度"):
        file_path = os.path.join(DATA_DIR, f"{sym}.csv")
        if os.path.exists(file_path):
            continue # 已下载则跳过（支持断点续传，随时可以重新运行）
            
        # 加入重试机制
        success = False
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
                success = True
                break # 成功则跳出重试循环
            except Exception:
                time.sleep(1) # 遇到错误短暂等待后重试
        
        if not success:
            fail_list.append(sym)
            
        # 限制请求频率，避免因请求过快被服务器断开连接
        time.sleep(0.5) 
            
    if fail_list:
        print(f"\n注意：有 {len(fail_list)} 只股票下载失败。")
        print("因为已经实现了断点续传，您可以随时【再次运行本脚本】，它会自动跳过已成功的数据并重试失败的部分！")
    else:
        print("\n所有成分股数据已下载完成！")

def step3_calculate_top20(df_cons):
    """步骤三：计算平均成交额并输出前20"""
    print("\n=== 步骤三：计算并提取流动性前20名 ===")
    
    code_col = "品种代码" if "品种代码" in df_cons.columns else df_cons.columns[0]
    name_col = "品种名称" if "品种名称" in df_cons.columns else df_cons.columns[1]
    name_dict = df_cons.set_index(code_col)[name_col].to_dict()
    
    results = []
    for sym in df_cons[code_col].tolist():
        file_path = os.path.join(DATA_DIR, f"{sym}.csv")
        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_csv(file_path)
            if df.empty or "成交额" not in df.columns:
                continue
                
            # 取最近30个交易日
            df = df.tail(30)
            avg_amount = df['成交额'].mean()
            results.append({"代码": sym, "名称": name_dict.get(sym, "未知"), "平均成交额": avg_amount})
        except Exception:
            continue
        
    if not results:
        print("未找到任何有效的数据，请确认步骤二是否下载成功。")
        return
        
    df_res = pd.DataFrame(results)
    # 按成交额从高到低排序
    df_res = df_res.sort_values(by="平均成交额", ascending=False).reset_index(drop=True)
    
    # 取前20只
    top20 = df_res.head(20).copy()
    top20["排名"] = range(1, len(top20) + 1)
    top20["近30日日均成交额(亿元)"] = (top20["平均成交额"] / 1e8).round(2)
    
    print("\n========== 沪深300 流动性前 20 只股票 ==========")
    print(top20[["排名", "代码", "名称", "近30日日均成交额(亿元)"]].to_string(index=False))
    
    out_csv = os.path.join(BASE_DIR, "top20_hs300_liquid_final.csv")
    top20.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 最终排序结果已保存至: {out_csv}")

def main():
    df_cons = step1_get_components()
    step2_fetch_stock_data(df_cons)
    step3_calculate_top20(df_cons)

if __name__ == "__main__":
    main()
