import numpy as np
from src.synthetic_market import generate_multi_regime_market
from src.train import build_env, train_agent


def test_build_env_defaults_to_gate():
    market = generate_multi_regime_market(n_risky=3, n_safe=2, n_steps=300, seed=1, signal_strength=0.9)
    env = build_env(market, {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0})
    assert env.action_mode == "gate"                      # default preserved


def test_tilt_agent_beats_base_when_signal_exists():
    # With a strong leading signal, the tilt agent should earn positive mean
    # structure-baselined reward OOS (times a tilt toward the safe block). End-to-end
    # check that the expressive action can detect skill when skill is possible.
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "action_mode": "tilt", "max_tilt": 0.15, "total_timesteps": 120_000, "seed": 0}
    market = generate_multi_regime_market(3, 2, 6000, seed=11, signal_strength=0.95)
    model = train_agent(market, config)

    eval_market = generate_multi_regime_market(3, 2, 6000, seed=12, signal_strength=0.95)
    env = build_env(eval_market, config)
    obs, _ = env.reset(seed=0)
    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)
        done = term or trunc
    assert np.isfinite(rewards).all()
    assert np.mean(rewards) > 0.0                          # detects skill when signal exists
