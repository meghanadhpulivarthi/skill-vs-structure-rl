"""Train residual-PPO on the allocation env. Traceable, seeded, config-driven."""
import json
import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.synthetic_market import generate_market
from src.allocation_env import AllocationEnv

# Config — edit these directly for the default smoke run
DEFAULT_CONFIG = {
    "base_name": "vol_scaled",
    "window": 20,
    "cost_bps": 10.0,
    "total_timesteps": 20_000,
    "seed": 0,
    "n_assets": 4,
    "n_steps": 4000,
    "signal_strength": 0.95,
}


def build_env(market: dict, config: dict) -> AllocationEnv:
    return AllocationEnv(
        market,
        base_name=config["base_name"],
        window=config["window"],
        cost_bps=config["cost_bps"],
    )


def train_agent(market: dict, config: dict) -> PPO:
    env = Monitor(build_env(market, config))
    model = PPO("MlpPolicy", env, seed=config["seed"], verbose=0)
    model.learn(total_timesteps=config["total_timesteps"], progress_bar=True)
    return model


def _run_default():
    config = dict(DEFAULT_CONFIG)
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : {config}")
    print("=" * 60)

    market = generate_market(config["n_assets"], config["n_steps"],
                             seed=config["seed"], signal_strength=config["signal_strength"])
    model = train_agent(market, config)

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_train-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    model.save(out_dir / "ppo_model")
    with open(out_dir / "results.json", "w") as f:
        json.dump({"status": "trained", "model_path": str(out_dir / "ppo_model.zip")}, f, indent=2)
    print(f"Saved run to: {out_dir}")


if __name__ == "__main__":
    _run_default()
