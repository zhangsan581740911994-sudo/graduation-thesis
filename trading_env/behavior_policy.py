"""
多规则混合行为策略：用于从行情构造离线数据集时的 action 列。

设计要点
--------
- 动作空间与 ETFTradingEnv 一致：目标仓位 w ∈ [-1, 1]。
- 动量/反转类规则只使用 **上一日收益** prev_return，避免用当日收益决策当日仓位（与常见「开盘用昨收信息」一致，无未来函数）。
- MA 类规则仅使用当日行内已有特征 ma5_gap / ma10_gap（由历史收盘价滚动得到）。
- 混合方式二选一：
  - sample：每步按概率抽一条规则（解释成「混合策略族」）；
  - blend：对各规则输出做加权平均后再 clip，轨迹更平滑。

论文中可写：行为数据由动量、均值回归、均线趋势与随机探索按固定比例混合生成，以兼顾覆盖度与可解释性。
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd

RULE_NAMES: Tuple[str, ...] = ("momentum", "mean_reversion", "ma_trend", "random")


class MixedBehaviorPolicy:
    """多规则混合行为策略。"""

    def __init__(
        self,
        seed: int = 0,
        rule_probs: Tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20),
        momentum_scale: float = 25.0,
        reversion_scale: float = 25.0,
        ma_spread_scale: float = 15.0,
        mixture_mode: Literal["sample", "blend"] = "sample",
    ) -> None:
        """
        Parameters
        ----------
        rule_probs
            顺序对应 RULE_NAMES：momentum, mean_reversion, ma_trend, random。
            须在 sample / blend 模式下均归一化为概率权重。
        momentum_scale
            动量：a = clip(scale * prev_return, -1, 1)。日收益很小，需放大到合理幅度。
        reversion_scale
            均值回归：a = clip(-scale * prev_return, -1, 1)。
        ma_spread_scale
            均线趋势：a = clip(scale * (ma5_gap - ma10_gap), -1, 1)。
        mixture_mode
            sample：每步按 rule_probs 抽一条规则；
            blend：对四条规则输出的动作按 rule_probs 加权平均再 clip。
        """
        if len(rule_probs) != 4:
            raise ValueError("rule_probs 长度必须为 4，对应四类规则。")
        s = float(sum(rule_probs))
        if s <= 0:
            raise ValueError("rule_probs 之和必须为正。")
        self._p = tuple(float(x) / s for x in rule_probs)
        self.momentum_scale = float(momentum_scale)
        self.reversion_scale = float(reversion_scale)
        self.ma_spread_scale = float(ma_spread_scale)
        self.mixture_mode = mixture_mode
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def prev_return_from_df(df: pd.DataFrame, t: int) -> float:
        """第 t 根 K 线决策时使用的上一日简单收益；t==0 时为 0。"""
        if t <= 0:
            return 0.0
        return float(df.iloc[t - 1]["return"])

    def _rule_momentum(self, prev_return: float) -> float:
        return float(np.clip(self.momentum_scale * prev_return, -1.0, 1.0))

    def _rule_mean_reversion(self, prev_return: float) -> float:
        return float(np.clip(-self.reversion_scale * prev_return, -1.0, 1.0))

    def _rule_ma_trend(self, row: pd.Series) -> float:
        spread = float(row["ma5_gap"]) - float(row["ma10_gap"])
        return float(np.clip(self.ma_spread_scale * spread, -1.0, 1.0))

    def _rule_random(self) -> float:
        return float(self.rng.uniform(-1.0, 1.0))

    def act(self, row: pd.Series, prev_return: float) -> float:
        """对单行行情 + 上一日收益给出目标仓位。"""
        a_m = self._rule_momentum(prev_return)
        a_r = self._rule_mean_reversion(prev_return)
        a_ma = self._rule_ma_trend(row)
        a_z = self._rule_random()
        actions = (a_m, a_r, a_ma, a_z)

        if self.mixture_mode == "blend":
            a = sum(p * a for p, a in zip(self._p, actions))
            return float(np.clip(a, -1.0, 1.0))

        idx = int(self.rng.choice(4, p=self._p))
        return float(actions[idx])

    def act_with_info(
        self, row: pd.Series, prev_return: float
    ) -> Tuple[float, str]:
        """返回 (action, 若 sample 则为被选规则名，blend 则为 'blend')。"""
        a_m = self._rule_momentum(prev_return)
        a_r = self._rule_mean_reversion(prev_return)
        a_ma = self._rule_ma_trend(row)
        a_z = self._rule_random()
        actions = (a_m, a_r, a_ma, a_z)

        if self.mixture_mode == "blend":
            a = sum(p * x for p, x in zip(self._p, actions))
            return float(np.clip(a, -1.0, 1.0)), "blend"

        idx = int(self.rng.choice(4, p=self._p))
        return float(actions[idx]), RULE_NAMES[idx]

    def rollout_actions(self, df: pd.DataFrame) -> np.ndarray:
        """对整表逐行生成动作序列，长度 len(df)，与 env 逐步交互时第 t 步用 actions[t]。"""
        n = len(df)
        out = np.empty(n, dtype=np.float32)
        for t in range(n):
            pr = self.prev_return_from_df(df, t)
            out[t] = self.act(df.iloc[t], pr)
        return out
