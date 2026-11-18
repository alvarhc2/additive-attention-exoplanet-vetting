"""Figures drawn for the repository, not copied from the article."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_phase_fold_gif(path: str | Path, frames: int = 48) -> None:
    """Animate a light curve folding onto its transit.

    The motion follows the phase-folding sketch in the Astronet repository.
    The samples and the rendering are original.
    """

    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    orbits = 4
    samples = 720
    time = np.linspace(0.0, orbits, samples, endpoint=False)
    phase = np.mod(time, 1.0)
    # Transit centered at phase 0, drawn across the fold boundary.
    distance = np.minimum(phase, 1.0 - phase)
    transit = distance < 0.04
    flux = 1.0 + 0.004 * np.sin(2 * np.pi * time * 1.7)
    flux -= 0.018 * np.exp(-0.5 * (distance / 0.012) ** 2)
    flux += rng.normal(0.0, 0.0015, size=samples)

    figure, axes = plt.subplots(figsize=(7.4, 3.5), dpi=120)
    figure.patch.set_facecolor("#07080b")
    axes.set_facecolor("#07080b")
    for spine in axes.spines.values():
        spine.set_color("#2a3140")
    axes.tick_params(colors="#9aa3b2", labelsize=9)
    axes.set_ylabel("Flux", color="#c5ccd6")
    axes.set_ylim(0.965, 1.012)
    scatter = axes.scatter([], [], s=8, linewidths=0)
    caption = axes.text(0.0, 1.02, "", transform=axes.transAxes, color="#d5dbe3", fontsize=12)

    def draw(frame: int):
        progress = frame / (frames - 1)
        # Hold the time series, fold, then hold the folded curve.
        if progress < 0.2:
            mix = 0.0
            label = "Unfolded light curve"
            xlabel = "Time (orbits)"
            left, right = -0.05, orbits + 0.05
        elif progress < 0.8:
            mix = (progress - 0.2) / 0.6
            mix = mix * mix * (3 - 2 * mix)
            label = "Phase folding"
            xlabel = "Time  →  phase"
            left = -0.05 * (1 - mix) + (-0.02) * mix
            right = (orbits + 0.05) * (1 - mix) + 1.02 * mix
        else:
            mix = 1.0
            label = "Phase-folded transit"
            xlabel = "Phase"
            left, right = -0.02, 1.02
        position = (1 - mix) * time + mix * phase
        color = np.where(transit, "#e07a3d", "#e8edf2")
        order = np.argsort(position)
        scatter.set_offsets(np.column_stack((position[order], flux[order])))
        scatter.set_color(color[order])
        axes.set_xlim(left, right)
        axes.set_xlabel(xlabel, color="#c5ccd6")
        caption.set_text(label)
        return (scatter, caption)

    writer = PillowWriter(fps=12)
    with writer.saving(figure, str(path), dpi=120):
        for frame in range(frames):
            draw(frame)
            writer.grab_frame()
    plt.close(figure)
