import numpy as np
from src.synthetic_market import generate_risky_safe_market
from src.train import build_env, train_agent


def test_gate_agent_beats_base_when_signal_exists():
    # With a strong leading signal, the de-risking-gate agent should earn
    # positive mean structure-baselined reward on a held-out market (i.e. its
    # timed de-risking beats the base after costs). This is the end-to-end
    # check that the method can detect skill when skill is possible.
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 1, "total_timesteps": 120_000, "seed": 0}
    market = generate_risky_safe_market(6000, seed=11, signal_strength=0.95)
    model = train_agent(market, config)

    eval_market = generate_risky_safe_market(6000, seed=12, signal_strength=0.95)
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
    assert np.mean(rewards) > 0.0               # detects skill when signal exists
