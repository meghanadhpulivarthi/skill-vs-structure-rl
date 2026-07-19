# src/attribution.py
"""Post-hoc attribution of the trained gate agent's de-risking decisions.

The ATTRIBUTION track for the RQ3 faithfulness experiment: gradient saliency and
KernelSHAP, both returning a per-feature importance vector over the 43-dim gate
observation. Compared against the CAUSAL track (src/interventions.py) to test
whether attribution identifies the true (signal) driver. Functions take injected
callables so they can be calibrated on a known linear policy before use on PPO.
"""
import numpy as np
import torch


def make_gate_mean_fn(model):
    """Adapter: the SB3 policy's deterministic gate (Gaussian mean, PRE-clip) as a
    differentiable function of the observation tensor. Pre-clip is intentional —
    the env's [0,1] clip has zero gradient in saturation and would zero out
    saliency exactly where the agent is decisive."""
    def gate_mean_fn(obs_tensor: torch.Tensor) -> torch.Tensor:
        distribution = model.policy.get_distribution(obs_tensor)
        return distribution.distribution.mean[:, 0]
    return gate_mean_fn


def saliency_importance(gate_mean_fn, observations: np.ndarray) -> np.ndarray:
    """Per-feature gradient saliency: mean_t |d gate / d obs_i| over the stack."""
    obs_tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    obs_tensor.requires_grad_(True)
    gate = gate_mean_fn(obs_tensor)
    # Rows are independent, so grad of the sum w.r.t. row i equals grad of gate_i.
    gate.sum().backward()
    grads = obs_tensor.grad.detach().numpy()
    return np.mean(np.abs(grads), axis=0)
