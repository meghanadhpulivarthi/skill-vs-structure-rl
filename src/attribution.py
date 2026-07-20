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
    """Per-feature saliency in gradient x feature-std units: mean_t |d gate / d obs_i|
    scaled by the feature's standard deviation over the stack. Scaling by std puts
    saliency in the SAME 'sensitivity per realistic (1-std) move' units as KernelSHAP
    and freeze-ablation, so the causal-vs-attribution comparison is apples-to-apples
    (raw |grad| is scale-invariant and would not be comparable to the scale-sensitive
    causal/SHAP tracks). See docs/design_2026-07-19_mc2-causal-probing.md and the final
    review."""
    observations = np.asarray(observations, dtype=np.float32)
    obs_tensor = torch.as_tensor(observations)
    obs_tensor.requires_grad_(True)
    gate = gate_mean_fn(obs_tensor)
    gate.sum().backward()
    grads = obs_tensor.grad.detach().numpy()
    mean_abs_grad = np.mean(np.abs(grads), axis=0)
    feature_std = observations.std(axis=0)
    return mean_abs_grad * feature_std


def shap_importance(gate_fn, observations, n_background: int = 40,
                    n_explain: int = 60, seed: int = 0) -> np.ndarray:
    """Per-feature KernelSHAP importance: mean |SHAP value| over an explained
    sample, using a seeded background sub-sample. `gate_fn` is the numpy gate
    adapter (src.interventions.make_gate_fn)."""
    import shap
    observations = np.asarray(observations, dtype=np.float32)
    rng = np.random.default_rng(seed)
    n = len(observations)
    background = observations[rng.choice(n, size=min(n_background, n), replace=False)]
    explain = observations[rng.choice(n, size=min(n_explain, n), replace=False)]
    # KernelExplainer uses the legacy global RNG internally; seed it for determinism.
    np.random.seed(seed)
    explainer = shap.KernelExplainer(gate_fn, background)
    values = np.asarray(explainer.shap_values(explain, silent=True))
    return np.mean(np.abs(values), axis=0)


def aggregate_to_groups(importance: np.ndarray, groups: dict) -> dict:
    """Sum |importance| within each semantic feature group."""
    importance = np.asarray(importance, dtype=float)
    return {name: float(np.abs(importance[indices]).sum()) for name, indices in groups.items()}


def project_to_simplex_torch(v: torch.Tensor) -> torch.Tensor:
    """Batched, differentiable Euclidean projection onto the probability simplex
    (Duchi et al. 2008), mirroring src/simplex.project_to_simplex so the tilt->weights
    map can be backpropagated for saliency. v: [B, n] -> [B, n]."""
    n = v.shape[1]
    u, _ = torch.sort(v, descending=True, dim=1)
    cssv = torch.cumsum(u, dim=1) - 1.0
    ind = torch.arange(1, n + 1, dtype=v.dtype, device=v.device)
    cond = (u - cssv / ind) > 0
    rho = cond.sum(dim=1, keepdim=True)                       # count of positive terms, >=1
    theta = torch.gather(cssv, 1, (rho - 1).clamp(min=0)) / rho.to(v.dtype)
    return torch.clamp(v - theta, min=0.0)
