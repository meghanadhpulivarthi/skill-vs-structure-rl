"""Phase-randomization placebo: the real-data overfitting/luck null.

Destroys any timeable temporal structure while preserving each asset's marginal
variance, spectrum, and contemporaneous cross-asset correlation. Running the full
walk-forward on these surrogates measures the skill the pipeline manufactures from
noise; the RQ1 headline is real OOS skill NET of this null (open-questions.md).
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.walk_forward import walk_forward_gate


def phase_randomize(returns: np.ndarray, seed: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    n_steps, n_assets = returns.shape
    rng = np.random.default_rng(seed)

    demeaned = returns - returns.mean(axis=0)
    spectrum = np.fft.rfft(demeaned, axis=0)
    magnitude = np.abs(spectrum)
    n_freq = spectrum.shape[0]

    # SHARED random phases across assets preserve cross-asset correlation.
    random_phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    random_phase[0] = 0.0                                      # keep the DC term real
    if n_steps % 2 == 0:
        random_phase[-1] = 0.0                                 # Nyquist term stays real
    phased = magnitude * np.exp(1j * random_phase)[:, None]
    surrogate = np.fft.irfft(phased, n=n_steps, axis=0)
    return surrogate + returns.mean(axis=0)                    # restore per-asset mean


def placebo_null(returns: np.ndarray, config: dict, n_placebo: int, seed: int, run_dir=None) -> dict:
    run_dir = Path(run_dir) if run_dir is not None else None
    placebo_skills = []
    for draw in tqdm(range(n_placebo), desc="placebo"):
        surrogate = phase_randomize(returns, seed=seed + draw)
        draw_dir = run_dir / f"placebo_{draw:02d}" if run_dir is not None else None
        result = walk_forward_gate(surrogate, {**config, "seed": config["seed"] + draw}, run_dir=draw_dir)
        placebo_skills.append(result["mean_skill"])
        print(f"placebo_null: draw {draw} skill = {result['mean_skill']:.6e}")
    return {
        "placebo_skills": [float(x) for x in placebo_skills],
        "mean": float(np.mean(placebo_skills)),
        "std": float(np.std(placebo_skills)),
    }
