import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import norm
from scipy.signal import argrelextrema

"""
Bayesian GMM fits + visualizations for your 6 CSVs.

Behavior:
- Fits BayesianGaussianMixture(n_components=4, dirichlet process prior)
- Plots histogram (density) + mixture density + components
- For Channel 2: highlights the FIRST gaussian (smallest mean)
- For Channel 9: highlights the LAST gaussian (largest mean)

Files (expected to contain a 'channel' column):
  histogram_testing/histogram_ch2_z12_x300_y800_box100.csv
  histogram_testing/histogram_ch9_z12_x300_y800_box100.csv
  histogram_testing/histogram_ch2_z12_x300_y400_box100.csv
  histogram_testing/histogram_ch9_z12_x300_y400_box100.csv
  histogram_testing/histogram_ch2_z12_x600_y400_box100.csv
  histogram_testing/histogram_ch9_z12_x600_y400_box100.csv
"""

DATA_FILES = [
    ("histogram_testing/histogram_ch2_z12_x300_y800_box100.csv", "Channel 2 | x=300, y=800"),
    ("histogram_testing/histogram_ch9_z12_x300_y800_box100.csv", "Channel 9 | x=300, y=800"),
    ("histogram_testing/histogram_ch2_z12_x300_y400_box100.csv", "Channel 2 | x=300, y=400"),
    ("histogram_testing/histogram_ch9_z12_x300_y400_box100.csv", "Channel 9 | x=300, y=400"),
    ("histogram_testing/histogram_ch2_z12_x600_y400_box100.csv", "Channel 2 | x=600, y=400"),
    ("histogram_testing/histogram_ch9_z12_x600_y400_box100.csv", "Channel 9 | x=600, y=400"),
]


def fit_bayesian_gmm_1d(x, max_components=4, random_state=42,
                        weight_concentration_prior=1e-2, mean_precision_prior=0.8):
    """
    Fit a 1-D Bayesian Gaussian Mixture using sklearn's BayesianGaussianMixture.

    - weight_concentration_prior (gamma): smaller -> encourages fewer active components.
    - mean_precision_prior: controls how tightly the prior concentrates means around mean_prior.
    """
    X = np.asarray(x).reshape(-1, 1)
    bgm = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=weight_concentration_prior,
        mean_precision_prior=mean_precision_prior,
        max_iter=1000,
        n_init=5,
        init_params="kmeans",
        random_state=random_state,
    )
    bgm.fit(X)
    return bgm


def compute_decision_boundary(x_grid, total_pdf, means, stds, policy,
                              max_idx, min_idx, sigma_factor):
    if min_idx.size == 0:
        return None

    maxima_points = np.sort(x_grid[max_idx]) if max_idx.size else np.array([])
    minima_points = np.sort(x_grid[min_idx])

    direction = 1 if policy == "first" else -1
    if direction == -1:
        maxima_points = maxima_points[::-1]
        minima_points = minima_points[::-1]

    first_min = minima_points[0] if minima_points.size else None
    if first_min is None:
        return None

    second_min = minima_points[1] if minima_points.size > 1 else None
    second_max = maxima_points[1] if maxima_points.size > 1 else None

    sorted_indices = np.argsort(means)
    if len(sorted_indices) < 2:
        sigma_second = None
    else:
        if policy == "first":
            second_idx = sorted_indices[1]
        else:
            second_idx = sorted_indices[-2]
        sigma_second = stds[second_idx]

    use_second_min = False
    if (sigma_second is not None) and (second_max is not None) and (sigma_second > 0):
        distance = abs(first_min - second_max)
        if distance <= sigma_factor * sigma_second:
            use_second_min = second_min is not None

    boundary = second_min if use_second_min else first_min

    midpoint = 0.5 * (np.min(means) + np.max(means))
    if policy == "first":
        boundary = min(boundary, midpoint)
    else:
        boundary = max(boundary, midpoint)

    return boundary


def plot_bgm_on_axis(ax, data, bgm, title, highlight_policy="auto", show_extrema=True,
                     boundary_factor=0.5):
    """
    ax: matplotlib axis
    data: 1D numpy array
    bgm: fitted BayesianGaussianMixture
    highlight_policy:
      - 'auto' : choose 'first' for Channel 2 titles and 'last' for Channel 9 titles
      - 'first' : highlight component with smallest mean
      - 'last'  : highlight component with largest mean
    """
    ax.hist(data, bins=100, density=True, alpha=0.35, edgecolor="k")

    x_grid = np.linspace(np.min(data), np.max(data), 1200)

    weights = bgm.weights_            # shape (K,)
    means = bgm.means_.ravel()       # shape (K,)
    # covariances_ is (K,1,1) for 'full' covariance on 1D -> extract scalars
    covs = np.array([bgm.covariances_[k].reshape(-1).item() if bgm.covariances_.ndim == 3 else bgm.covariances_[k]
                     for k in range(bgm.n_components)])
    stds = np.sqrt(covs)

    # component pdfs (weighted)
    comp_pdfs = []
    for k in range(bgm.n_components):
        pdf_k = weights[k] * norm.pdf(x_grid, loc=means[k], scale=stds[k])
        comp_pdfs.append(pdf_k)
    comp_pdfs = np.vstack(comp_pdfs)  # (K, len(x_grid))

    # total mixture density
    total_pdf = comp_pdfs.sum(axis=0)
    ax.plot(x_grid, total_pdf, lw=2, label="Mixture density")

    # plot components (dashed lines). Use alpha proportional to weight (for visibility)
    for k in range(bgm.n_components):
        alpha_k = float(min(max(weights[k] * 8, 0.12), 0.95))
        ax.plot(x_grid, comp_pdfs[k], lw=1.5, ls="--", alpha=alpha_k,
                label=f"comp {k}: w={weights[k]:.3f}, μ={means[k]:.2f}, σ={stds[k]:.2f}")

    # Determine which component to highlight
    if highlight_policy == "auto":
        if "Channel 2" in title:
            policy = "first"
        elif "Channel 9" in title:
            policy = "last"
        else:
            policy = "last"
    else:
        policy = highlight_policy

    if policy == "first":
        target_idx = int(np.argmin(means))
        label_text = "Target (first = smallest mean)"
    else:
        target_idx = int(np.argmax(means))
        label_text = "Target (last = largest mean)"

    # Only highlight if its weight is meaningful (otherwise indicate it's near-zero)
    if weights[target_idx] > 1e-3:
        ax.fill_between(x_grid, 0, comp_pdfs[target_idx], alpha=0.35, label=label_text)
    else:
        ax.fill_between(x_grid, 0, comp_pdfs[target_idx], alpha=0.15, label=label_text + " (tiny weight)")

    ax.axvline(means[target_idx], linestyle=":", linewidth=1.3, color="k")

    max_idx = argrelextrema(total_pdf, np.greater)[0]
    min_idx = argrelextrema(total_pdf, np.less)[0]

    if show_extrema:
        if max_idx.size:
            ax.scatter(x_grid[max_idx], total_pdf[max_idx], marker='x', s=35,
                       color='black', linewidths=1.2, label='Local maxima')
        if min_idx.size:
            ax.scatter(x_grid[min_idx], total_pdf[min_idx], marker='x', s=35,
                       color='black', linewidths=1.2, label='Local minima')

    boundary = compute_decision_boundary(x_grid, total_pdf, means, stds, policy,
                                         max_idx, min_idx, boundary_factor)
    if boundary is not None:
        ax.axvline(boundary, color='crimson', linestyle='-', linewidth=1.6,
                   label='Decision boundary')

    ax.set_title(title)
    ax.set_xlabel("Channel value")
    ax.set_ylabel("Density")
    ax.legend(fontsize="small")


def load_all_datasets():
    datasets = []
    for fname, title in DATA_FILES:
        try:
            df = pd.read_csv(fname)
            if "channel" not in df.columns:
                raise ValueError("CSV has no 'channel' column")
            arr = np.array(df["channel"].dropna()).astype(float)
            datasets.append((arr, title))
        except Exception as e:
            print(f"Warning: could not load {fname}: {e}")
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise Bayesian GMM fits for histogram CSVs.")
    parser.add_argument("--hide-extrema", action="store_true",
                        help="Do not mark local minima/maxima of the mixture density.")
    parser.add_argument("--boundary-factor", type=float, default=0.5,
                        help="Multiplier for σ when deciding if the boundary can move to the second local minima.")
    args = parser.parse_args()

    datasets = load_all_datasets()
    if len(datasets) == 0:
        raise SystemExit("No datasets loaded. Put your CSVs in histogram_testing/ with a 'channel' column.")

    n = len(datasets)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, (data, title) in zip(axes, datasets):
        print(f"Fitting Bayesian GMM for {title} (n_samples={len(data)})")
        # Tunable priors:
        # - weight_concentration_prior: smaller encourages sparsity (fewer active components).
        # - mean_precision_prior: how strongly means are pulled toward mean_prior (defaults to data mean)
        bgm = fit_bayesian_gmm_1d(data, max_components=4,
                                  random_state=42,
                                  weight_concentration_prior=1e-2,
                                  mean_precision_prior=0.8)
        plot_bgm_on_axis(ax, data, bgm, title, highlight_policy="auto",
                         show_extrema=not args.hide_extrema,
                         boundary_factor=args.boundary_factor)

    # hide unused axes if any
    for ax in axes[len(datasets):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
