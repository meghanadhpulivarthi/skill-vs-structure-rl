import numpy as np
from src.interventions import feature_groups, rollout_observations, make_gate_fn


def test_feature_groups_partition_the_obs_vector():
    groups = feature_groups(window=20, n_assets=2)
    assert groups["returns"] == list(range(0, 40))
    assert groups["short_vol"] == [40, 41]
    assert groups["signal"] == [42]
    # groups must exactly partition the 43-dim gate obs (no overlap, full cover)
    all_idx = groups["returns"] + groups["short_vol"] + groups["signal"]
    assert sorted(all_idx) == list(range(43))


class _ConstGate:
    # minimal duck-typed stand-in for an SB3 model: gate = clip(mean of obs, 0, 1)
    def predict(self, obs, deterministic=True):
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        return obs.mean(axis=1, keepdims=True), None


def test_rollout_and_gate_fn_shapes_and_clipping():
    from src.synthetic_market import generate_risky_safe_market
    market = generate_risky_safe_market(300, seed=1, signal_strength=0.95)
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 1, "action_mode": "gate"}
    model = _ConstGate()
    obs_stack = rollout_observations(model, market, config)
    # one observation per decision: T = n_steps - window
    assert obs_stack.shape == (300 - 20, 43)
    assert np.isfinite(obs_stack).all()
    gate_fn = make_gate_fn(model)
    gates = gate_fn(obs_stack)
    assert gates.shape == (280,)
    assert (gates >= 0.0).all() and (gates <= 1.0).all()  # clipped to the env's [0,1]
