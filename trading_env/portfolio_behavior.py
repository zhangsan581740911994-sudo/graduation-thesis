"""
多标的行为策略：与单标 MixedBehaviorPolicy 同类规则，但在每个标的上生成分量，
再按 sample/blend 混合，输出 N 维 logits（与环境 softmax 一致）。
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


class MixedPortfolioBehaviorPolicy:
    """基于上一日截面收益向量与随机探索，输出 [-1,1]^N 的 logits。"""

    def __init__(
        self,
        returns: np.ndarray,
        seed: int = 0,
        rule_probs: Tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20),
        momentum_scale: float = 25.0,
        reversion_scale: float = 25.0,
        mixture_mode: Literal["sample", "blend"] = "sample",
    ) -> None:
        self.returns = np.asarray(returns, dtype=np.float32)
        if self.returns.ndim != 2:
            raise ValueError("returns must be (T, N).")
        self.n = self.returns.shape[1]
        if len(rule_probs) != 4:
            raise ValueError("rule_probs must have length 4.")
        s = float(sum(rule_probs))
        self._p = tuple(float(x) / s for x in rule_probs)
        self.momentum_scale = float(momentum_scale)
        self.reversion_scale = float(reversion_scale)
        self.mixture_mode = mixture_mode
        self.rng = np.random.default_rng(seed)

    def _prev_returns(self, t: int) -> np.ndarray:
        if t <= 0:
            return np.zeros(self.n, dtype=np.float32)
        return self.returns[t - 1].copy()

    def _rule_momentum(self, prev_r: np.ndarray) -> np.ndarray:
        return np.clip(self.momentum_scale * prev_r, -1.0, 1.0).astype(np.float32)

    def _rule_mean_reversion(self, prev_r: np.ndarray) -> np.ndarray:
        return np.clip(-self.reversion_scale * prev_r, -1.0, 1.0).astype(np.float32)

    def _rule_equal_preference(self) -> np.ndarray:
        """全零 logits → softmax 为等权，对应「无信息」基准。"""
        return np.zeros(self.n, dtype=np.float32)

    def _rule_random(self) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=self.n).astype(np.float32)

    def act(self, t: int) -> np.ndarray:
        prev_r = self._prev_returns(t)
        a_m = self._rule_momentum(prev_r)
        a_r = self._rule_mean_reversion(prev_r)
        a_eq = self._rule_equal_preference()
        a_z = self._rule_random()
        actions = (a_m, a_r, a_eq, a_z)

        if self.mixture_mode == "blend":
            out = sum(p * a for p, a in zip(self._p, actions))
            return np.clip(out, -1.0, 1.0).astype(np.float32)

        idx = int(self.rng.choice(4, p=self._p))
        return np.asarray(actions[idx], dtype=np.float32)
