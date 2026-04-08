import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def load_returns_panel(csv_path: Path):
    df = pd.read_csv(csv_path)
    if not {"date", "symbol", "return"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain date/symbol/return")
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    panel = (
        df[["date", "symbol", "return"]]
        .pivot_table(index="date", columns="symbol", values="return", aggfunc="last")
        .sort_index()
    )
    panel = panel.dropna(axis=0, how="any")
    returns = panel.to_numpy(dtype=np.float64)
    symbols = list(panel.columns)
    return returns, symbols


def run_episode(returns, theta, fee_rate=0.001):
    """
    组合环境奖励同 portfolio_env:
      reward_t = w_{t-1}^T r_t - fee * ||w_t - w_{t-1}||_1
    策略：
      logits_t = theta * r_{t-1}（逐资产动量系数）
      w_t = softmax(logits_t)
    """
    n_assets = returns.shape[1]
    weights = np.ones(n_assets, dtype=np.float64) / n_assets
    prev_r = np.zeros(n_assets, dtype=np.float64)

    rewards = []
    turnovers = []
    pv = 1.0

    for t in range(returns.shape[0]):
        logits = theta * prev_r
        w_new = softmax(logits)

        r_vec = returns[t]
        pnl = float(np.dot(weights, r_vec))
        turnover = float(np.sum(np.abs(w_new - weights)))
        reward = pnl - fee_rate * turnover

        pv *= 1.0 + reward
        rewards.append(reward)
        turnovers.append(turnover)

        prev_r = r_vec
        weights = w_new

    rewards = np.asarray(rewards, dtype=np.float64)
    turnovers = np.asarray(turnovers, dtype=np.float64)
    return {
        "cum_return": float(pv - 1.0),
        "mean_reward": float(np.mean(rewards)),
        "vol_reward": float(np.std(rewards) + 1e-12),
        "sharpe_daily": float(np.mean(rewards) / (np.std(rewards) + 1e-12)),
        "max_drawdown": max_drawdown(np.cumprod(1.0 + rewards)),
        "avg_turnover": float(np.mean(turnovers)),
    }


def max_drawdown(nav):
    peak = np.maximum.accumulate(nav)
    dd = (peak - nav) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def train_theta_es(
    train_returns,
    val_returns,
    seed=42,
    n_iter=200,
    pop_size=24,
    sigma=0.10,
    lr=0.20,
    fee_rate=0.001,
):
    n_assets = train_returns.shape[1]
    rng = np.random.default_rng(seed)
    theta = np.zeros(n_assets, dtype=np.float64)
    best_theta = theta.copy()
    best_val = -1e18

    for i in range(n_iter):
        eps = rng.normal(0.0, 1.0, size=(pop_size, n_assets))
        scores = []
        for k in range(pop_size):
            th = theta + sigma * eps[k]
            metric = run_episode(train_returns, th, fee_rate=fee_rate)["cum_return"]
            scores.append(metric)
        scores = np.asarray(scores, dtype=np.float64)

        # 标准化后做 ES 更新
        s = (scores - scores.mean()) / (scores.std() + 1e-12)
        grad = (s[:, None] * eps).mean(axis=0) / sigma
        theta = theta + lr * grad

        if (i + 1) % 10 == 0:
            val_metric = run_episode(val_returns, theta, fee_rate=fee_rate)["cum_return"]
            if val_metric > best_val:
                best_val = val_metric
                best_theta = theta.copy()

    return best_theta


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_groups(spec: str):
    """
    参数组格式：
      g1:n_iter=200,pop_size=24,sigma=0.10,lr=0.20,fee_rate=0.001;
      g2:n_iter=400,pop_size=32,sigma=0.08,lr=0.15,fee_rate=0.001
    """
    groups = []
    for chunk in [c.strip() for c in spec.split(";") if c.strip()]:
        if ":" not in chunk:
            raise ValueError(f"Bad group spec: {chunk}")
        name, body = chunk.split(":", 1)
        cfg = {"name": name.strip()}
        for kv in [x.strip() for x in body.split(",") if x.strip()]:
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k in {"n_iter", "pop_size"}:
                cfg[k] = int(v)
            elif k in {"sigma", "lr", "fee_rate"}:
                cfg[k] = float(v)
            else:
                raise ValueError(f"Unsupported key in group: {k}")
        for k, default in {
            "n_iter": 200,
            "pop_size": 24,
            "sigma": 0.10,
            "lr": 0.20,
            "fee_rate": 0.001,
        }.items():
            cfg.setdefault(k, default)
        groups.append(cfg)
    return groups


def run_one(
    train_r,
    val_r,
    test_r,
    symbols,
    out_dir: Path,
    seed: int,
    n_iter: int,
    pop_size: int,
    sigma: float,
    lr: float,
    fee_rate: float,
    run_test: bool,
    group_name: str = "single",
):
    best_theta = train_theta_es(
        train_r,
        val_r,
        seed=seed,
        n_iter=n_iter,
        pop_size=pop_size,
        sigma=sigma,
        lr=lr,
        fee_rate=fee_rate,
    )
    train_metrics = run_episode(train_r, best_theta, fee_rate=fee_rate)
    val_metrics = run_episode(val_r, best_theta, fee_rate=fee_rate)
    test_metrics = run_episode(test_r, best_theta, fee_rate=fee_rate) if run_test else None

    result = {
        "method": "Simple ES momentum policy",
        "group_name": group_name,
        "seed": seed,
        "n_iter": n_iter,
        "pop_size": pop_size,
        "sigma": sigma,
        "lr": lr,
        "fee_rate": fee_rate,
        "n_assets": len(symbols),
        "symbols": symbols,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_evaluated": bool(run_test),
    }
    with open(out_dir / "train_val_test_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.save(out_dir / "best_theta.npy", best_theta)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train/val(/test) for portfolio baseline with single or batch mode."
    )
    parser.add_argument(
        "--run_test",
        action="store_true",
        help="If set, evaluate test split after selecting params on val.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for single run.")
    parser.add_argument("--n_iter", type=int, default=200, help="Training iterations.")
    parser.add_argument("--pop_size", type=int, default=24, help="ES population size.")
    parser.add_argument("--sigma", type=float, default=0.10, help="ES perturbation std.")
    parser.add_argument("--lr", type=float, default=0.20, help="ES update learning rate.")
    parser.add_argument("--fee_rate", type=float, default=0.001, help="Transaction cost.")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch mode: parameter groups x seeds, and output summary mean/std.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds for batch mode, e.g. 42,43,44.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=(
            "g1:n_iter=200,pop_size=24,sigma=0.10,lr=0.20,fee_rate=0.001;"
            "g2:n_iter=400,pop_size=24,sigma=0.08,lr=0.15,fee_rate=0.001"
        ),
        help="Parameter group spec string for batch mode.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data_thesis"
    train_csv = data_dir / "hs300_top20_offline_train.csv"
    val_csv = data_dir / "hs300_top20_offline_val.csv"
    test_csv = data_dir / "hs300_top20_offline_test.csv"

    train_r, symbols = load_returns_panel(train_csv)
    val_r, _ = load_returns_panel(val_csv)
    test_r, _ = load_returns_panel(test_csv)

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_out = repo_root / "outputs_portfolio"
    base_out.mkdir(parents=True, exist_ok=True)

    if not args.batch:
        out_dir = base_out / f"simple_train_val_test_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_one(
            train_r, val_r, test_r, symbols, out_dir,
            seed=args.seed, n_iter=args.n_iter, pop_size=args.pop_size,
            sigma=args.sigma, lr=args.lr, fee_rate=args.fee_rate,
            run_test=args.run_test, group_name="single",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nSaved to: {out_dir}")
        return

    # Batch mode
    batch_dir = base_out / f"batch_train_val_test_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_list(args.seeds)
    groups = parse_groups(args.groups)
    rows = []
    total_runs = len(groups) * len(seeds)
    done_runs = 0
    for g in groups:
        for s in seeds:
            run_name = f"{g['name']}_seed{s}"
            run_dir = batch_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            result = run_one(
                train_r, val_r, test_r, symbols, run_dir,
                seed=s,
                n_iter=g["n_iter"],
                pop_size=g["pop_size"],
                sigma=g["sigma"],
                lr=g["lr"],
                fee_rate=g["fee_rate"],
                run_test=args.run_test,
                group_name=g["name"],
            )
            done_runs += 1
            print(
                f"[{done_runs}/{total_runs}] finished {run_name} | "
                f"val_cum_return={result['val_metrics']['cum_return']:.6f}",
                flush=True,
            )
            rows.append({
                "group_name": g["name"],
                "seed": s,
                "n_iter": g["n_iter"],
                "pop_size": g["pop_size"],
                "sigma": g["sigma"],
                "lr": g["lr"],
                "fee_rate": g["fee_rate"],
                "train_cum_return": result["train_metrics"]["cum_return"],
                "val_cum_return": result["val_metrics"]["cum_return"],
                "val_sharpe_daily": result["val_metrics"]["sharpe_daily"],
                "val_max_drawdown": result["val_metrics"]["max_drawdown"],
                "test_cum_return": None if result["test_metrics"] is None else result["test_metrics"]["cum_return"],
            })

    detail_df = pd.DataFrame(rows)
    detail_csv = batch_dir / "batch_detail.csv"
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    agg = detail_df.groupby("group_name").agg(
        n_runs=("seed", "count"),
        val_cum_return_mean=("val_cum_return", "mean"),
        val_cum_return_std=("val_cum_return", "std"),
        val_sharpe_daily_mean=("val_sharpe_daily", "mean"),
        val_sharpe_daily_std=("val_sharpe_daily", "std"),
        val_max_drawdown_mean=("val_max_drawdown", "mean"),
        val_max_drawdown_std=("val_max_drawdown", "std"),
    ).reset_index()
    summary_csv = batch_dir / "batch_summary_mean_std.csv"
    agg.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"Batch done. Detail: {detail_csv}")
    print(f"Summary: {summary_csv}\n")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
