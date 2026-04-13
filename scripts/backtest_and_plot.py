#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文图表生成脚本：累计净值曲线 (Cumulative NAV) 与动态回撤曲线 (Drawdown)
用法示例:
python scripts/backtest_and_plot.py \
    --model_dir outputs_portfolio_warmstart_top20_test_from_s43_seed42/default_xxx \
    --test_csv data_thesis/hs300_top20_offline_test.csv \
    --output_prefix bull_market
"""

import argparse
import json
import os
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from diffusion.diffusion import DiffusionQL
from trading_env.gym_env import PortfolioGymEnv
from trading_env.panel_loader import load_returns_panel_csv
from utilities.jax_utils import batch_to_jax

# 设置绘图风格，适用于学术论文
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def load_model(model_dir):
    """从指定目录加载训练好的 EDP 模型"""
    model_dir = Path(model_dir)
    variant_path = model_dir / "variant.json"
    
    if not variant_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {variant_path}")
        
    with open(variant_path, "r") as f:
        variant = json.load(f)
        
    algo_cfg = variant["algo_cfg"]
    # 兼容旧版本配置
    if "nstep" not in algo_cfg:
        algo_cfg["nstep"] = 1
        
    # 寻找最新的 checkpoint
    pkl_files = list(model_dir.glob("model_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"在 {model_dir} 中找不到 model_*.pkl 文件")
        
    # 按 epoch 数字排序，取最大的
    def get_epoch(p):
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1
            
    latest_pkl = max(pkl_files, key=get_epoch)
    print(f"正在加载模型: {latest_pkl}")
    
    # 重新构建模型架构
    # 注意：这里需要根据你的实际环境维度来初始化
    # 我们先创建一个临时环境来获取维度
    dummy_env = PortfolioGymEnv(variant["portfolio_eval_csv"])
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.shape[0]
    
    model = DiffusionQL(
        algo_cfg,
        obs_dim,
        act_dim,
        variant["max_traj_length"],
        variant["policy_arch"],
        variant["qf_arch"],
        variant["orthogonal_init"],
        variant["policy_log_std_multiplier"],
        variant["policy_log_std_offset"],
        variant["qf_layer_norm"],
        variant["policy_layer_norm"],
        variant["activation"],
    )
    
    # 加载权重
    model.load(str(latest_pkl))
    return model, variant


def calculate_drawdown(nav_array):
    """计算动态回撤"""
    running_max = np.maximum.accumulate(nav_array)
    drawdown = (nav_array - running_max) / running_max
    return drawdown


def run_backtest(model, test_csv, sample_method="ddpm"):
    """在测试集上运行回测，收集每天的收益"""
    print(f"正在测试集上运行回测: {test_csv}")
    env = PortfolioGymEnv(test_csv)
    obs = env.reset()
    done = False
    
    edp_returns = []
    bh_returns = []
    ew_returns = []
    dates = []
    
    # 获取日期列表用于画图 (跳过第一天，因为第一天是初始状态)
    returns_matrix, date_list = load_returns_panel_csv(test_csv)
    
    # 确保使用正确的随机数种子
    rng = jax.random.PRNGKey(42)
    
    step = 0
    while not done:
        # 获取 EDP 动作
        obs_jax = batch_to_jax(np.expand_params(obs, axis=0))
        rng, act_rng = jax.random.split(rng)
        
        # 使用扩散模型采样动作
        action, _ = model.act(obs_jax, act_rng, sample_method=sample_method)
        action = np.array(action[0])
        
        # 执行环境 step
        next_obs, reward, done, info = env.step(action)
        
        # 记录收益 (注意：环境返回的 reward 可能是 norm 过的，我们需要真实的 portfolio return)
        # 真实收益 = 动作权重 * 当天真实收益率
        actual_return = info.get('portfolio_return', reward)
        
        edp_returns.append(actual_return)
        bh_returns.append(info.get('bh_return', 0.0))
        ew_returns.append(info.get('ew_return', 0.0))
        
        if step < len(date_list):
            dates.append(date_list[step])
            
        obs = next_obs
        step += 1
        
    return {
        'dates': pd.to_datetime(dates[:len(edp_returns)]),
        'edp_returns': np.array(edp_returns),
        'bh_returns': np.array(bh_returns),
        'ew_returns': np.array(ew_returns)
    }


def plot_cumulative_nav(results, output_path):
    """绘制累计净值曲线"""
    dates = results['dates']
    # 计算累计净值 (初始净值为 1.0)
    edp_nav = np.cumprod(1.0 + results['edp_returns'])
    bh_nav = np.cumprod(1.0 + results['bh_returns'])
    ew_nav = np.cumprod(1.0 + results['ew_returns'])
    
    # 在开头插入 1.0 作为起点
    edp_nav = np.insert(edp_nav, 0, 1.0)
    bh_nav = np.insert(bh_nav, 0, 1.0)
    ew_nav = np.insert(ew_nav, 0, 1.0)
    
    # 日期也要在开头插入第一天的前一天 (为了画图对齐)
    plot_dates = [dates[0] - pd.Timedelta(days=1)] + list(dates)
    
    plt.figure(figsize=(10, 6))
    plt.plot(plot_dates, edp_nav, label='EDP Strategy (Ours)', color='#d62728', linewidth=2.5)
    plt.plot(plot_dates, bh_nav, label='Buy & Hold', color='#1f77b4', linewidth=1.5, linestyle='--')
    plt.plot(plot_dates, ew_nav, label='Equal Weight', color='#2ca02c', linewidth=1.5, linestyle='-.')
    
    plt.title('Cumulative Net Asset Value (NAV)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative NAV', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 格式化 x 轴日期显示
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"累计净值曲线已保存至: {output_path}")
    plt.close()
    
    # 返回最终净值用于打印
    return edp_nav[-1], bh_nav[-1], ew_nav[-1]


def plot_drawdown(results, output_path):
    """绘制动态回撤曲线"""
    dates = results['dates']
    
    # 计算累计净值
    edp_nav = np.cumprod(1.0 + results['edp_returns'])
    bh_nav = np.cumprod(1.0 + results['bh_returns'])
    
    # 计算回撤
    edp_dd = calculate_drawdown(edp_nav) * 100  # 转换为百分比
    bh_dd = calculate_drawdown(bh_nav) * 100
    
    plt.figure(figsize=(10, 4))
    
    # EDP 回撤 (红色阴影)
    plt.fill_between(dates, edp_dd, 0, color='#d62728', alpha=0.3, label='EDP Drawdown')
    plt.plot(dates, edp_dd, color='#d62728', linewidth=1)
    
    # Buy&Hold 回撤 (蓝色阴影)
    plt.fill_between(dates, bh_dd, 0, color='#1f77b4', alpha=0.2, label='Buy & Hold Drawdown')
    plt.plot(dates, bh_dd, color='#1f77b4', linewidth=1, linestyle='--')
    
    plt.title('Dynamic Drawdown (%)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Drawdown (%)', fontsize=14)
    plt.legend(fontsize=12, loc='lower left')
    plt.grid(True, alpha=0.3)
    
    # 格式化 x 轴日期显示
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"动态回撤曲线已保存至: {output_path}")
    plt.close()
    
    # 返回最大回撤用于打印
    return np.min(edp_dd), np.min(bh_dd)


def main():
    parser = argparse.ArgumentParser(description="生成论文图表：累计净值与回撤")
    parser.add_argument("--model_dir", type=str, required=True, help="包含 variant.json 和 model_*.pkl 的目录")
    parser.add_argument("--test_csv", type=str, required=True, help="用于回测的测试集 CSV 路径")
    parser.add_argument("--output_prefix", type=str, default="backtest", help="输出图片文件名的前缀")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs("docs/thesis_materials/figures", exist_ok=True)
    
    try:
        # 1. 加载模型
        model, variant = load_model(args.model_dir)
        
        # 2. 运行回测
        results = run_backtest(model, args.test_csv, sample_method=variant.get("sample_method", "ddpm"))
        
        # 3. 画图并保存
        nav_path = f"docs/thesis_materials/figures/{args.output_prefix}_cumulative_nav.png"
        dd_path = f"docs/thesis_materials/figures/{args.output_prefix}_drawdown.png"
        
        edp_final, bh_final, ew_final = plot_cumulative_nav(results, nav_path)
        edp_mdd, bh_mdd = plot_drawdown(results, dd_path)
        
        # 4. 打印统计摘要
        print("\n" + "="*50)
        print(f"回测摘要 ({args.output_prefix})")
        print("="*50)
        print(f"测试集: {args.test_csv}")
        print(f"测试天数: {len(results['dates'])} 天")
        print(f"EDP 最终累计收益: {(edp_final - 1.0) * 100:.2f}%")
        print(f"B&H 最终累计收益: {(bh_final - 1.0) * 100:.2f}%")
        print(f"E.W 最终累计收益: {(ew_final - 1.0) * 100:.2f}%")
        print("-" * 50)
        print(f"EDP 最大回撤 (MDD): {edp_mdd:.2f}%")
        print(f"B&H 最大回撤 (MDD): {bh_mdd:.2f}%")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n[错误] 运行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
