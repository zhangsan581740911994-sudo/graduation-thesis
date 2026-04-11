# HANDOFF：论文收尾与图表生成指南（后续专用）

> 用途：核心实验（牛市/熊市/消融）已全部跑通！本阶段目标是**“画图表 + 补深度消融”**，直接产出能插入毕业论文的高清图片和深度分析。  
> 更新时间：2026-04-11

---

## 0. 新环境恢复指南 (AutoDL 换机必读)

如果你租用了一台全新的 AutoDL 机器，请在开始后续实验前，严格按照以下步骤恢复环境：

### 第一步：拉取完整的代码和数据集
在新机子的终端里运行：
```bash
cd /root/autodl-tmp
git clone https://github.com/zhangsan581740911994-sudo/graduation-thesis.git
cd graduation-thesis
```

### 第二步：恢复模型权重
1. 将本地备份的 `model_weights_backup.zip` 上传到新机子的 `graduation-thesis` 目录下。
2. 在终端运行解压命令：
```bash
unzip model_weights_backup.zip
rm model_weights_backup.zip  # 解压确认无误后可删除压缩包省空间
```

### 第三步：重新配置 Python 虚拟环境
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-thesis.txt
```

### 第四步：召唤 AI 助手
环境准备好后，在 Cursor 里新建一个对话框（Chat），把本文档底部的 **“给新助手的启动指令”** 发给 AI。

---

## 1) 当前已达成成就（给新助手的上下文）

- **牛市进攻达成**：在 179 天强反弹窗口，EDP (O2O) 取得 30.3% 均值收益，超过 25% 目标。
- **熊市防守达成**：在 96 天单边下跌窗口，大盘亏损，EDP (O2O) 取得 5.03% 均值正收益，防守反击成功。
- **基础消融完成**：在 20 只股票的小规模连续控制任务中，证明了纯 TD3 (30.7%) 与 EDP (30.3%) 表现相当，确立了算法的边界条件。

---

## 1) 接下来要做什么？（核心任务清单）

本阶段分为两条线，可并行推进：

### 任务 A：生成论文级高清图表（优先做，直接出结果）
利用现有的 `progress.csv` 和测试集数据，使用 Python (matplotlib/seaborn) 自动生成以下 8 类图表：

1. **累计净值曲线图 (Cumulative NAV)**：牛市/熊市下 EDP vs Buy&Hold vs EqualWeight。
2. **动态回撤曲线图 (Drawdown)**：直观展示 EDP 在熊市的抗跌能力。
3. **验证集得分曲线 (Eval Return)**：证明离线训练的收敛性。
4. **损失函数下降图 (Loss Curve)**：`qf_loss` 和 `diff_loss` 的收敛过程。
5. **持仓权重热力图 (Weight Heatmap)**：展示模型在不同时间段的调仓逻辑（可解释性）。
6. **每日换手率柱状图 (Turnover Bar)**：证明模型没有被手续费拖垮。
7. **多维度雷达图 (Radar Chart)**：收益、夏普、回撤、波动、胜率的“六边形战士”对比。
8. **多 Seed 收益箱线图 (Boxplot)**：证明算法的稳定性（方差小）。

### 任务 B：深度消融实验（拔高论文档次，证明扩散模型 > TD3）
由于在 20 只股票的简单任务中，EDP 和纯 TD3 表现接近，为了在学术上证明 EDP（扩散模型）的绝对优势，我们需要设计一个**“更难、更嘈杂”**的环境。

**推荐方案：构建“混合噪声”数据集（最省算力，效果最好）**
- **做法**：在现有的 `train.csv` 中，故意掺杂 30% 的“随机权重交易数据”或“亏损交易数据”。
- **预期**：纯 TD3 在面对这种“多模态、嘈杂”的次优数据时，性能会大幅崩溃；而 EDP 作为生成模型，能完美拟合复杂分布并提取高回报动作，依然保持高收益。
- **结论**：证明 EDP 在复杂/嘈杂的真实金融数据下具有极强的鲁棒性。

---

## 2) 给新助手的启动指令（直接复制发送）

```text
你好，我的核心实验（牛市/熊市/基础消融）已经全部跑完，数据已记录在 `docs/thesis_materials/论文结果_填写版.md` 中。

请阅读 `docs/thesis_materials/HANDOFF_实验续跑指南.md`，然后：
1. 帮我写 Python 脚本，优先生成“任务 A”中的第 1 类和第 2 类图表（累计净值曲线图和动态回撤曲线图）。
2. 告诉我生成这些图表需要我提供哪些具体的 CSV 文件路径或数据格式。
```

---

## 3) 备忘：当前重要文件路径

- **牛市测试集**：`data_thesis/hs300_top20_offline_test.csv`
- **熊市测试集**：`data_thesis/hs300_top20_bear_test.csv`
- **牛市 EDP O2O 结果目录**：`./outputs_portfolio_warmstart_top20_test_from_s43_seed42/43/44`
- **熊市 EDP O2O 结果目录**：`./outputs_portfolio_warmstart_top20_bear_test_s42/43/44`
- **纯 TD3 离线结果目录**：`./outputs_portfolio_ablation_pure_td3_val_s42/43/44`

---

## 4) 论文撰写建议（随时查阅）

- **不要回避“纯 TD3 表现好”**：在论文中诚实写出“在 20 维简单连续控制任务中，TD3 已足够优秀；EDP 的价值在于向更大规模、更嘈杂的真实环境扩展”。
- **强调“防守反击”**：牛市赚 30.3%，熊市赚 5.03%（大盘亏损），这是你这篇论文最核心的业务亮点，一定要在摘要和结论里大写特写。