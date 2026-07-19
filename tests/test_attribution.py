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
