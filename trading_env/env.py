import numpy as np
import pandas as pd


class ETFTradingEnv:
    """
    Minimal single-asset trading environment for offline/online RL.

    - Action: target position weight w_t in [-1, 1] (short to long)
    - Reward: pnl - transaction_cost
    - Observation: [return_t, w_{t-1}, ma5_gap, ma10_gap]
    """

    def __init__(self, csv_path: str, fee_rate: float = 0.001, initial_cash: float = 1.0):
        self.csv_path = csv_path
        self.fee_rate = fee_rate
        self.initial_cash = initial_cash

        self.df = self._load_data(csv_path)
        self.n_steps = len(self.df)

        self.t = 0
        self.position = 0.0
        self.portfolio_value = initial_cash
        self.done = False

    @staticmethod
    def _load_data(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if "date" not in df.columns or "close" not in df.columns:
            raise ValueError("CSV must contain at least ['date', 'close'] columns.")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if "return" not in df.columns:
            df["return"] = df["close"].pct_change()

        # Basic technical features for a minimal baseline.
        ma5 = df["close"].rolling(5, min_periods=1).mean()
        ma10 = df["close"].rolling(10, min_periods=1).mean()
        df["ma5_gap"] = (df["close"] / ma5) - 1.0
        df["ma10_gap"] = (df["close"] / ma10) - 1.0

        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        return df

    def reset(self):
        self.t = 0
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.t]
        obs = np.array(
            [
                float(row["return"]),
                float(self.position),
                float(row["ma5_gap"]),
                float(row["ma10_gap"]),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before next step().")

        action = float(np.clip(action, -1.0, 1.0))
        row = self.df.iloc[self.t]
        r_t = float(row["return"])

        # Reward uses previous weight w_{t-1} to avoid look-ahead.
        pnl = self.position * r_t
        trade_cost = self.fee_rate * abs(action - self.position)
        reward = pnl - trade_cost

        self.portfolio_value *= (1.0 + reward)
        self.position = action
        self.t += 1

        if self.t >= self.n_steps - 1:
            self.done = True

        next_obs = self._get_obs() if not self.done else np.zeros(4, dtype=np.float32)
        info = {
            "date": str(row["date"].date()),
            "return": r_t,
            "position": self.position,
            "pnl": pnl,
            "trade_cost": trade_cost,
            "portfolio_value": self.portfolio_value,
        }
        return next_obs, reward, self.done, info

