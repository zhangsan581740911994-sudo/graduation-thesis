"""
多标的行为策略：与单标 MixedBehaviorPolicy 同类规则，但在每个标的上生成分量，
再按 sample/blend 混合，输出 N 维 logits（与环境 softmax 一致）。

环境变量（可选）：
- PORTFOLIO_BEHAVIOR_MODE=multimodal
  构造「低覆盖 + 多峰」离线数据：多数步为窄扰动（接近等权），少数步为尖峰集中仓位，
  用于消融中放大 TD3（单峰倾向）与 EDP（多模态生成）的差异。
  可选：PORTFOLIO_MULTIMODAL_CONSERVATIVE_FRAC（默认 0.2）、
        PORTFOLIO_MULTIMODAL_CONSERVATIVE_STD（默认 0.12）、
        PORTFOLIO_MULTIMODAL_EXTREME_K_MAX（默认 1）。
  额外可选：PORTFOLIO_MULTIMODAL_EXTREME_STYLE（默认 rank_bimodal）
        - rank_bimodal：在同一状态下随机选择「追涨极端峰」或「反转极端峰」
        - random_spike：随机重仓（旧逻辑）
  设置 multimodal 时不再使用 PORTFOLIO_NOISE_LEVEL 的四规则混合。
"""

from __future__ import annotations

import os
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
        self.n = int(self.returns.shape[1])
        self.rng = np.random.default_rng(seed)

        self._mode = os.environ.get("PORTFOLIO_BEHAVIOR_MODE", "").strip().lower()
        if self._mode == "multimodal":
            self._conservative_frac = float(
                os.environ.get("PORTFOLIO_MULTIMODAL_CONSERVATIVE_FRAC", "0.2")
            )
            self._conservative_std = float(
                os.environ.get("PORTFOLIO_MULTIMODAL_CONSERVATIVE_STD", "0.12")
            )
            self._extreme_k_max = int(
                os.environ.get("PORTFOLIO_MULTIMODAL_EXTREME_K_MAX", "1")
            )
            self._extreme_k_max = max(1, min(self._extreme_k_max, self.n))
            self._extreme_style = os.environ.get(
                "PORTFOLIO_MULTIMODAL_EXTREME_STYLE", "rank_bimodal"
            ).strip().lower()
            print(
                "[Behavior Policy] multimodal："
                f"conservative_frac={self._conservative_frac}, "
                f"conservative_std={self._conservative_std}, "
                f"extreme_k_max={self._extreme_k_max}, "
                f"extreme_style={self._extreme_style}"
            )
            self.momentum_scale = float(momentum_scale)
            self.reversion_scale = float(reversion_scale)
            self.mixture_mode = mixture_mode
            self._p = (1.0, 0.0, 0.0, 0.0)
            return

        noise_level = os.environ.get("PORTFOLIO_NOISE_LEVEL")
        if noise_level is not None:
            nl = float(noise_level)
            rem = (1.0 - nl) / 3.0
            rule_probs = (rem, rem, rem, nl)
            print(
                f"[Behavior Policy] 开启高噪声模式！Noise Level = {nl}, "
                f"行为策略概率调整为: {rule_probs}"
            )

        if len(rule_probs) != 4:
            raise ValueError("rule_probs must have length 4.")
        s = float(sum(rule_probs))
        self._p = tuple(float(x) / s for x in rule_probs)
        self.momentum_scale = float(momentum_scale)
        self.reversion_scale = float(reversion_scale)
        self.mixture_mode = mixture_mode

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

    def _act_multimodal(self, t: int) -> np.ndarray:
        """低覆盖多峰动作：
        - conservative：窄 logits（近等权）
        - extreme：两个相反极端峰（二选一），默认基于前一日收益排序构造
        """
        prev_r = self._prev_returns(t)
        if self.rng.random() < self._conservative_frac:
            x = self.rng.normal(0.0, self._conservative_std, size=self.n)
            return np.clip(x, -1.0, 1.0).astype(np.float32)
        k_hi = int(self.rng.integers(1, self._extreme_k_max + 1))
        logits = np.full(self.n, -1.0, dtype=np.float32)

        if self._extreme_style == "rank_bimodal":
            ranks = np.argsort(prev_r)  # 升序
            # mode=0: 追涨峰（top-k）；mode=1: 反转峰（bottom-k）
            mode = int(self.rng.integers(0, 2))
            if mode == 0:
                idx = ranks[-k_hi:]
            else:
                idx = ranks[:k_hi]
        else:
            # 兼容旧逻辑：随机尖峰
            idx = self.rng.choice(self.n, size=k_hi, replace=False)

        logits[idx] = 1.0
        return np.clip(logits, -1.0, 1.0).astype(np.float32)

    def act(self, t: int) -> np.ndarray:
        if self._mode == "multimodal":
            return self._act_multimodal(t)

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
