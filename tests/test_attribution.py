# tests/test_attribution.py
import numpy as np
import torch
from src.attribution import saliency_importance


def test_saliency_ranks_features_by_weight_on_linear_policy():
    # Known-answer calibration: a linear gate g = sigmoid(w . o). The saliency of
    # feature i is |w_i| * sigmoid'(.) > proportional ordering by |w_i|. Feature 5
    # has the largest weight; feature 10 has zero weight (inert).
    weights = np.zeros(43, dtype=np.float32)
    weights[5] = 3.0
    weights[10] = 0.0
    weights[20] = 1.0
    w = torch.tensor(weights)

    def linear_gate_mean(obs_tensor):
        return torch.sigmoid(obs_tensor @ w)

    rng = np.random.default_rng(0)
    obs = rng.normal(scale=0.1, size=(200, 43)).astype(np.float32)
    importance = saliency_importance(linear_gate_mean, obs)

    assert importance.shape == (43,)
    assert np.isfinite(importance).all()
    assert importance.argmax() == 5                 # largest-weight feature ranked top
    assert importance[10] < importance[20] < importance[5]   # inert < small < large
    assert importance[10] < 1e-6                    # inert feature ~ zero saliency


from src.attribution import shap_importance, aggregate_to_groups


def test_shap_ranks_features_by_weight_on_linear_policy():
    # Same known-answer calibration as saliency, for the model-agnostic path.
    weights = np.zeros(43, dtype=np.float32)
    weights[5] = 3.0
    weights[20] = 1.0  # feature 10 stays inert (weight 0)

    def linear_gate(observations):
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        return 1.0 / (1.0 + np.exp(-(obs @ weights)))

    rng = np.random.default_rng(0)
    obs = rng.normal(scale=0.1, size=(120, 43)).astype(np.float32)
    importance = shap_importance(linear_gate, obs, n_background=30, n_explain=40, seed=0)

    assert importance.shape == (43,)
    assert np.isfinite(importance).all()
    assert importance.argmax() == 5              # largest-weight feature ranked top
    assert importance[10] < importance[20]       # inert below the small-weight feature


def test_aggregate_to_groups_sums_absolute_importance():
    importance = np.zeros(43)
    importance[0:40] = 0.01     # returns block sums to 0.4
    importance[40:42] = 0.5     # short_vol sums to 1.0
    importance[42] = 2.0        # signal
    groups = {"returns": list(range(40)), "short_vol": [40, 41], "signal": [42]}
    agg = aggregate_to_groups(importance, groups)
    assert np.isclose(agg["returns"], 0.4)
    assert np.isclose(agg["short_vol"], 1.0)
    assert np.isclose(agg["signal"], 2.0)
