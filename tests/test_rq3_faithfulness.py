# tests/test_rq3_faithfulness.py
import pytest
import numpy as np
import torch
from src.rq3_faithfulness import run_probe
from src.interventions import feature_groups


def test_run_probe_identifies_signal_as_causal_driver_and_emits_verdict():
    # Ground-truth gate that uses ONLY the signal feature (index 42). Both the
    # numpy gate_fn and the differentiable gate_mean_fn express the same policy,
    # so causal ablation AND faithful attribution must both rank `signal` top.
    def gate_fn(observations):
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        return 1.0 / (1.0 + np.exp(-4.0 * obs[:, 42]))

    def gate_mean_fn(obs_tensor):
        return torch.sigmoid(4.0 * obs_tensor[:, 42])

    rng = np.random.default_rng(0)
    obs = rng.normal(scale=0.3, size=(200, 43)).astype(np.float32)
    groups = feature_groups(window=20, n_assets=2)
    verdict = run_probe(gate_fn, gate_mean_fn, obs, groups, seed=0)

    # premise: the signal is the dominant CAUSAL driver (this is ground truth)
    assert verdict["top_group"]["causal_freeze"] == "signal"
    assert verdict["causal"]["freeze"]["signal"] > verdict["causal"]["freeze"]["returns"]
    # a faithful attribution recovers it too (this stub policy IS faithful)
    assert verdict["top_group"]["saliency"] == "signal"
    # verdict object has the full required shape
    assert set(verdict) == {"causal", "attribution", "spearman", "top_group"}
    assert set(verdict["spearman"]) == {"saliency_freeze", "saliency_permute",
                                        "shap_freeze", "shap_permute"}
    for rho in verdict["spearman"].values():
        assert -1.0 <= rho <= 1.0


@pytest.mark.parametrize("driver_index,expected_group", [
    (0, "returns"),    # a return feature
    (40, "short_vol"), # a vol feature
    (42, "signal"),    # the signal
])
def test_all_methods_follow_the_true_single_feature_driver(driver_index, expected_group):
    # A gate that depends ONLY on feature `driver_index`. After the scale (grad x std)
    # and cardinality (normalized shares) fixes, every method must rank the group that
    # CONTAINS the true driver as top — regardless of that feature's raw scale or how
    # many features its group has. Inputs deliberately have UNEQUAL per-column scale so
    # this exercises the scale regime the old equal-variance tests missed.
    def gate_fn(observations):
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        return 1.0 / (1.0 + np.exp(-4.0 * obs[:, driver_index]))

    def gate_mean_fn(obs_tensor):
        return torch.sigmoid(4.0 * obs_tensor[:, driver_index])

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(200, 43)).astype(np.float32)
    # impose unequal column scales, incl. a large-scale signal like the real market
    scales = np.ones(43, dtype=np.float32) * 0.01
    scales[42] = 0.35
    obs = obs * scales
    groups = feature_groups(window=20, n_assets=2)
    verdict = run_probe(gate_fn, gate_mean_fn, obs, groups, seed=0)

    assert verdict["top_group"]["causal_freeze"] == expected_group
    assert verdict["top_group"]["causal_permute"] == expected_group
    assert verdict["top_group"]["saliency"] == expected_group
    assert verdict["top_group"]["shap"] == expected_group


from src.rq3_faithfulness import gate_response_to_vol_shock


def test_vol_shock_response_returns_aligned_gate_trajectories():
    from src.synthetic_market import generate_risky_safe_market
    from src.interventions import make_gate_fn

    class _RiskGate:
        # a gate that de-risks when recent risky-asset moves are large (uses vol feature 40)
        def predict(self, obs, deterministic=True):
            obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
            return np.clip(obs[:, 40:41] * 20.0, 0.0, 1.0), None

    market = generate_risky_safe_market(400, seed=4, signal_strength=0.95)
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 1, "action_mode": "gate"}
    curves = gate_response_to_vol_shock(_RiskGate(), market, config,
                                        t0=200, width=5, multiplier=6.0)
    assert set(curves) == {"baseline", "shocked"}
    assert len(curves["baseline"]) == len(curves["shocked"]) > 0
    assert np.isfinite(curves["baseline"]).all() and np.isfinite(curves["shocked"]).all()
