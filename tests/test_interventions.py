import numpy as np
from src.interventions import feature_groups, rollout_observations, make_gate_fn, freeze_group, permute_group, causal_effect, inject_vol_shock, flip_signal, feature_groups_tilt


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


def test_inject_vol_shock_is_local_and_nonmutating():
    from src.synthetic_market import generate_risky_safe_market
    market = generate_risky_safe_market(300, seed=2, signal_strength=0.95)
    original_returns = np.array(market["returns"])   # snapshot
    shocked = inject_vol_shock(market, t0=100, width=5, multiplier=5.0, risky_index=0)
    assert np.allclose(market["returns"], original_returns)          # input untouched
    assert np.allclose(shocked["returns"][100:105, 0],
                       original_returns[100:105, 0] * 5.0)            # intended slice scaled
    assert np.allclose(shocked["returns"][:100], original_returns[:100])  # elsewhere unchanged
    assert np.allclose(shocked["returns"][:, 1], original_returns[:, 1])  # safe asset unchanged
    assert np.allclose(shocked["signal"], market["signal"])          # signal preserved


def test_flip_signal_sets_one_step_without_mutating():
    from src.synthetic_market import generate_risky_safe_market
    market = generate_risky_safe_market(300, seed=3, signal_strength=0.95)
    original_signal = np.array(market["signal"])
    flipped = flip_signal(market, t0=150, value=1.0)
    assert np.allclose(market["signal"], original_signal)   # input untouched
    assert flipped["signal"][150] == 1.0
    assert np.allclose(np.delete(flipped["signal"], 150), np.delete(original_signal, 150))


def test_feature_groups_tilt_partitions_the_121_dim_obs():
    groups = feature_groups_tilt(window=20, n_assets=5)
    assert groups["returns"] == list(range(0, 100))
    assert groups["short_vol"] == [100, 101, 102, 103, 104]
    assert groups["long_vol"] == [105, 106, 107, 108, 109]
    assert groups["momentum"] == [110, 111, 112, 113, 114]
    assert groups["base_weights"] == [115, 116, 117, 118, 119]
    assert groups["signal"] == [120]
    all_idx = sum(groups.values(), [])
    assert sorted(all_idx) == list(range(121))


from src.interventions import make_safe_weight_fn
from src.simplex import project_to_simplex


class _ConstTiltModel:
    # returns a fixed 5-dim action regardless of obs
    def __init__(self, action):
        self._action = np.asarray(action, dtype=float)
    def predict(self, obs, deterministic=True):
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        return np.tile(self._action, (len(obs), 1)), None


def test_safe_weight_fn_matches_hand_computed_projection():
    n_assets = 5
    max_tilt = 0.15
    base_obs_idx = [115, 116, 117, 118, 119]
    safe_asset_idx = [3, 4]
    action = np.array([2.0, -1.0, 0.5, 1.5, -0.5])
    model = _ConstTiltModel(action)

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(6, 121)).astype(np.float32)
    # put a valid equal-weight base into the base_weights block
    obs[:, base_obs_idx] = 1.0 / n_assets

    fn = make_safe_weight_fn(model, base_obs_idx, safe_asset_idx, max_tilt)
    got = fn(obs)

    # hand-compute the expected safe-block weight
    base = obs[0, base_obs_idx].astype(float)
    w = project_to_simplex(base + max_tilt * np.tanh(action))
    expected = w[safe_asset_idx].sum()
    assert got.shape == (6,)
    assert np.allclose(got, expected, atol=1e-6)          # constant action -> constant safe weight
    assert (got >= -1e-9).all() and (got <= 1.0 + 1e-9).all()
