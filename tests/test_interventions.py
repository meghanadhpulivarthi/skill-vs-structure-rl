import numpy as np
from src.interventions import feature_groups, rollout_observations, make_gate_fn, freeze_group, permute_group, causal_effect


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


def test_freeze_removes_variance_permute_preserves_marginal():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(200, 43)).astype(np.float32)
    frozen = freeze_group(obs, [42])
    assert np.allclose(frozen[:, 42].var(), 0.0)          # variation removed
    assert np.allclose(frozen[:, :42], obs[:, :42])       # other columns untouched
    permuted = permute_group(obs, [42], np.random.default_rng(1))
    assert np.allclose(np.sort(permuted[:, 42]), np.sort(obs[:, 42]))  # marginal preserved
    assert np.allclose(permuted[:, :42], obs[:, :42])


def test_causal_effect_zero_for_inert_feature_positive_for_used_feature():
    # A gate that depends ONLY on the signal feature (index 42): ablating the
    # signal must move the gate; ablating an inert feature (index 0) must not.
    def signal_only_gate(observations):
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        return 1.0 / (1.0 + np.exp(-4.0 * obs[:, 42]))
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(300, 43)).astype(np.float32)
    signal_effect = causal_effect(signal_only_gate, obs, [42], "freeze")
    inert_effect = causal_effect(signal_only_gate, obs, [0], "freeze")
    assert signal_effect > 1e-3
    assert inert_effect < 1e-9
    # permute mode is also well-defined and non-negative
    assert causal_effect(signal_only_gate, obs, [42], "permute", seed=3) > 1e-3


def test_causal_effect_rejects_unknown_mode():
    import pytest
    with pytest.raises(ValueError):
        causal_effect(lambda o: np.zeros(len(o)), np.zeros((5, 43)), [0], "scramble")
