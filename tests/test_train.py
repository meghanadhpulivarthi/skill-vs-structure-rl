import numpy as np
import pytest
from src.synthetic_market import generate_market
from src.train import build_env, train_agent


@pytest.mark.xfail(
    reason="OPEN RESEARCH GATE (2026-07-18): residual PPO does not yet beat the "
    "base even with a near-perfect signal — it over-tilts and pays turnover "
    "(mean|action|~0.7, skill<0 at 150k steps w/ VecNormalize). Blocks Task 8 "
    "(RQ2). Needs reward/action redesign — see docs checkpoint. Do NOT weaken "
    "this assertion; remove the xfail only once skill is genuinely positive.",
    strict=False,
)
def test_smoke_training_runs_and_beats_base_with_signal():
    # With a strong signal, a briefly-trained agent should earn positive mean
    # structure-baselined reward on a fresh episode (i.e. beat the base).
    market = generate_market(n_assets=4, n_steps=4000, seed=11, signal_strength=0.95)
    config = {"base_name": "vol_scaled", "window": 20, "cost_bps": 10.0,
              "total_timesteps": 20_000, "seed": 0}
    model = train_agent(market, config)

    eval_market = generate_market(n_assets=4, n_steps=4000, seed=12, signal_strength=0.95)
    env = build_env(eval_market, config)
    obs, _ = env.reset(seed=0)
    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)
        done = term or trunc
    assert np.isfinite(rewards).all()          # no NaNs (stability)
    assert np.mean(rewards) > 0.0               # adds skill when signal exists
