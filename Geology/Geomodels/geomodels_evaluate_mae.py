"""
python evaluate_diversity.py --conditioning_idx 100
"""

import os
import argparse
import numpy as np
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt
import pandas as pd



parser = argparse.ArgumentParser(description="Evaluate training diversity experiment")
parser.add_argument('--conditioning_idx', type=int, default=100,
                    help='R2 patch index used for conditioning (default: 100)')
parser.add_argument('--max_lag_m',        type=int, default=800,
                    help='Max lag distance in metres for variogram MAE (default: 800)')
parser.add_argument('--mask_position',    type=int, default=120,
                    help='Mask position in pixels (default: 120)')
parser.add_argument('--jump_n_sample',    type=int, default=20,
                    help='jump_n_sample used during RePaint (default: 20)')
args = parser.parse_args()

CONDITIONING_IDX = args.conditioning_idx
MAX_LAG_M        = args.max_lag_m
MASK_POSITION    = args.mask_position
JUMP_N_SAMPLE    = args.jump_n_sample

BASE_ROBUST = (
    "/Home/siv36/hesal5042/Research/NORCE/"
    "inPainting_diffusionModel/Geology/"
    "sliced_data/Additional/robustness"
)

BASE_RUNS = (
    "/Home/siv36/hesal5042/Research/NORCE/"
    "inPainting_diffusionModel/Geology/"
    "training_runs"
)

SAVE_DIR = os.path.join(BASE_ROBUST, "evaluation_results")
os.makedirs(SAVE_DIR, exist_ok=True)


CONFIGS = [ 
    ('M1', 42),
    ('M1', 123),
    ('M1', 456), 
    ('M1', 789),
    ('M1', 321),
    ('M1', 654),
    ('M1', 987),
    ('M1', 213),

    
    ('M2', 42),
    ('M2', 123),
    ('M2', 456),
    ('M2', 789),
    ('M2', 321),
    ('M2', 654),
    ('M2', 987),
    ('M2', 213),
    
    ('M3', 42),
    ('M3', 123),
    ('M3', 456),
    ('M3', 789),
    ('M3', 321),
    ('M3', 654),
    ('M3', 987),
    ('M3', 213),
]

def compute_variograms(samples, mask_position):
    samples = np.squeeze(samples)
    if samples.ndim == 2:
        samples = samples[np.newaxis, :, :]

    N, H, W     = samples.shape
    boundary    = mask_position
    half_window = 4
    points_y    = [10, H // 2, H - 10]
    max_lag     = W - boundary - 5
    lags        = np.arange(1, max_lag)
    lags_m      = lags * 10

    variograms = {}
    for py in points_y:
        y0     = max(0, py - half_window)
        y1     = min(H, py + half_window + 1)
        window = samples[:, y0:y1, boundary:]
        gamma  = []
        for h in lags:
            d = window[:, :, h:] - window[:, :, :-h]
            gamma.append(0.5 * np.mean(d ** 2))
        variograms[py] = np.array(gamma)

    return variograms, lags_m, points_y


def variogram_mae(var_ref, var_pred, points_y, max_lag_m=800):
    max_lag_px = int(max_lag_m / 10)
    row_maes   = {}
    for py in points_y:
        ref  = var_ref[py][:max_lag_px]
        pred = var_pred[py][:max_lag_px]
        mae  = np.mean(np.abs(pred - ref))
        row_maes[py] = mae
        print(f"    Row {py} MAE: {mae:.6f}")
    overall_mae = np.mean(list(row_maes.values()))
    return overall_mae, row_maes

def percentile_coverage(ensemble, truth, mask_position):
    ens = ensemble[:, :, mask_position:]
    ref = truth[:,    mask_position:]

    intervals = {
        'P40-P60': (40, 60,  20.0),
        'P25-P75': (25, 75,  50.0),
        'P10-P90': (10, 90,  80.0),
    }

    results = {}
    for name, (lo, hi, nominal) in intervals.items():
        lower    = np.percentile(ens, lo, axis=0)
        upper    = np.percentile(ens, hi, axis=0)
        coverage = np.mean((ref >= lower) & (ref <= upper)) * 100.0
        results[name] = {'coverage': coverage, 'nominal': nominal}

    return results

def plot_variograms(var_ref, var_cond, var_uncond, lags_m, points_y,
                    model, seed, conditioning_idx, save_dir):

    ymax = max(
        max(np.max(var_ref[py])    for py in points_y),
        max(np.max(var_cond[py])   for py in points_y),
        max(np.max(var_uncond[py]) for py in points_y),
    ) * 1.05

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    for i, py in enumerate(points_y):
        axes[i].plot(lags_m, var_ref[py],    label="Test data",               linewidth=2)
        axes[i].plot(lags_m, var_cond[py],   label="Conditional Diffusion",   linewidth=2)
        axes[i].plot(lags_m, var_uncond[py], label="Unconditional Diffusion",  linewidth=2)
        axes[i].set_title(f"Row {py}")
        axes[i].set_xlabel("Lag distance (m)")
        axes[i].set_xlim(0, 1400)
        axes[i].set_ylim(0, ymax)
        axes[i].grid(True)

    axes[0].set_ylabel("Semivariance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    plt.suptitle(f"{model} seed{seed} | Conditioning {conditioning_idx}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = os.path.join(save_dir, f"{model}_seed{seed}_variogram.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Variogram plot saved: {out_path}")


def plot_variance_map(ensemble, model, seed, conditioning_idx,
                      mask_position, save_dir):

    ensemble = np.squeeze(ensemble)
    ensemble[ensemble < 1e-8] = 0

    variance_map = np.var(ensemble, axis=0)
    variance_map[:, :mask_position] = 0.0

    nz, nx = variance_map.shape
    x      = np.linspace(0, 2560, nx)
    z      = np.linspace(0,  640, nz)
    X, Z   = np.meshgrid(x, z)

    vmin        = 0.0
    vmax        = 0.01024

    levels      = np.linspace(vmin, vmax, 30)
    line_levels = np.linspace(vmin, vmax, 8)

    plt.figure(figsize=(12, 5))

    contour = plt.contourf(X, Z, variance_map,
                           levels=levels, cmap="terrain",
                           vmin=vmin, vmax=vmax)
    lines   = plt.contour(X, Z, variance_map,
                          levels=line_levels,
                          colors="black", linewidths=0.4)
    plt.clabel(lines, inline=True, fontsize=7, fmt="%.5f")

    cbar = plt.colorbar(contour)
    cbar.set_label("Porosity Variance")

    plt.axvline(x=mask_position * 10, color="red",
                linestyle="--", linewidth=2)
    plt.xticks(np.arange(0, 2560, 200))
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    plt.title(f"Variance Contour Map — {model} seed{seed}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{model}_seed{seed}_variance.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Variance map saved: {out_path}")




def plot_percentile_analysis(ensemble, truth, mask_position,
                              model, seed, conditioning_idx, save_dir):

    ens = ensemble[:, :, mask_position:]   # (N, H, W_inpaint)
    ref = truth[:,    mask_position:]      # (H, W_inpaint)

    eps = 1e-7


    percentiles       = 100.0 * np.mean(ens <= ref[np.newaxis, :, :],       axis=0)
    percentiles_plus  = 100.0 * np.mean(ens <= ref[np.newaxis, :, :] + eps, axis=0)
    percentiles_minus = 100.0 * np.mean(ens <= ref[np.newaxis, :, :] - eps, axis=0)

    over  = percentiles_minus > 95
    under = percentiles_plus  < 5

    outlier_map = np.zeros_like(percentiles)
    outlier_map[over]  =  1
    outlier_map[under] = -1

    n_rows, n_cols = percentiles.shape

    plt.figure(figsize=(10, 6))
    plt.imshow(outlier_map, aspect='auto', cmap="coolwarm", vmin=-1, vmax=1)
    cbar = plt.colorbar()
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["Under (<5th)", "Normal", "Over (>95th)"])
    plt.title(f"Outlier Map — {model} seed{seed}")
    plt.xlabel("Distance (pixels, inpainted region)")
    plt.ylabel("Depth (pixels)")
    plt.tight_layout()
    out1 = os.path.join(save_dir, f"{model}_seed{seed}_outlier_map.png")
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Outlier map saved: {out1}")

    COLORMAP   = "coolwarm"
    highlighted = list(range(6)) + [10] + list(range(25, n_cols, 25))
    n_colors    = len(highlighted)
    colors      = matplotlib.colormaps[COLORMAP](np.linspace(0, 1, n_colors))

    sorted_percentiles = np.sort(percentiles.ravel())
    n_all              = len(sorted_percentiles)

    plt.figure(figsize=(10, 6))
    for i, col_idx in enumerate(highlighted):
        if col_idx >= n_cols:
            continue
        col_sorted = np.sort(percentiles[:, col_idx])
        x          = np.linspace(0, 1, len(col_sorted))
        lw         = 0.8
        label      = f"Column {col_idx}" if i in [0, 5, 6] else None
        plt.plot(x, col_sorted, color=colors[i], linewidth=lw, label=label)

    plt.plot(np.linspace(0, 1, n_all), sorted_percentiles,
             color="black", linewidth=1.5, label="All pixels")
    plt.plot([0, 1], [0, 100],
             linestyle="--", color="red", linewidth=1.5, label="Theoretical (uniform)")
    plt.axhline(95, color="gray", linestyle=":", linewidth=1.2, label="Outlier threshold P5/P95")
    plt.axhline(5,  color="gray", linestyle=":", linewidth=1.2)

    plt.xlabel("Rank (normalized)")
    plt.ylabel("Percentile")
    plt.title(f"Sorted Percentiles — {model} seed{seed}")
    plt.legend(ncol=2)
    plt.grid(True)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_colors - 1)
    sm   = plt.cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    cbar = plt.colorbar(sm, ax=plt.gca(), label="Column index")
    cbar.set_ticks(np.linspace(0, n_colors - 1, n_colors))
    cbar.set_ticklabels([str(highlighted[j]) for j in range(n_colors)])

    out2 = os.path.join(save_dir, f"{model}_seed{seed}_sorted_percentiles.png")
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sorted percentiles saved: {out2}")

    uniform      = np.linspace(0, 100, n_rows)
    ks_distances = np.array([
        np.max(np.abs(np.sort(percentiles[:, c]) - uniform))
        for c in range(n_cols)
    ])

    ks_all_distances = np.zeros(n_cols)
    for c in range(n_cols):
        col_sorted = np.sort(percentiles[:, c])
        all_interp = np.interp(
            np.linspace(0, 1, n_rows),
            np.linspace(0, 1, n_all),
            sorted_percentiles
        )
        ks_all_distances[c] = np.max(np.abs(col_sorted - all_interp))

    plt.figure(figsize=(10, 6))
    plt.plot(ks_distances,     label="vs. uniform",        color="red",      linewidth=1.2)
    plt.plot(ks_all_distances, label="vs. all-pixel curve", color="steelblue", linewidth=1.2)
    plt.xlabel("Column index (inpainted region)")
    plt.ylabel("KS distance (percentile units)")
    plt.title(f"Per-column KS Distance — {model} seed{seed}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out3 = os.path.join(save_dir, f"{model}_seed{seed}_ks_distance.png")
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  KS distance saved: {out3}")

def plot_multiseed_summary(df, save_dir, conditioning_idx):

    metrics = [
        ("variogram_mae", "Variogram MAE ↓", None),
        ("P40_P60", "P40–P60 coverage (%)", 20),
        ("P25_P75", "P25–P75 coverage (%)", 50),
        ("P10_P90", "P10–P90 coverage (%)", 80),
    ]

    models = ["M1", "M2", "M3"]

    # Consistent colors for each model
    model_colors = {
        "M1": "#0072B2",   # blue
        "M2": "#D55E00",   # orange
        "M3": "#009E73",   # green
    }

    x = np.arange(len(models))

    # 2x2 layout for better readability
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()

    for ax, (metric, title, nominal) in zip(axes, metrics):

        means = []
        stds = []

        for i, model in enumerate(models):

            values = df.loc[
                df["model"] == model, metric
            ].values

            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1))

            # Individual training runs
            ax.scatter(
                np.full(len(values), i),
                values,
                s=45,
                alpha=0.75,
                color=model_colors[model]
            )

        means = np.array(means)
        stds = np.array(stds)

        # Mean +/- 1 standard deviation
        ax.errorbar(
            x,
            means,
            yerr=stds,
            fmt="_",
            markersize=28,
            color="black",
            ecolor="black",
            elinewidth=2,
            capsize=7,
            capthick=2
        )

        # Nominal coverage line
        if nominal is not None:
            ax.axhline(
                nominal,
                linestyle="--",
                linewidth=1.5,
                color="crimson"
            )

            ax.text(
                2.42,
                nominal + 1,
                "nominal",
                color="crimson",
                fontsize=10
            )

            ax.set_ylim(0, 100)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [r"$M_1$", r"$M_2$", r"$M_3$"],
            fontsize=12
        )

        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", alpha=0.25)

    # Shared legend with matching model colors
    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=model_colors["M1"],
            markeredgecolor=model_colors["M1"],
            markersize=8,
            label=r"$M_1$ (1 geomodel)"
        ),
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=model_colors["M2"],
            markeredgecolor=model_colors["M2"],
            markersize=8,
            label=r"$M_2$ (2 geomodels)"
        ),
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=model_colors["M3"],
            markeredgecolor=model_colors["M3"],
            markersize=8,
            label=r"$M_3$ (4 geomodels)"
        ),
        plt.Line2D(
            [0], [0],
            color="black",
            linewidth=2,
            label="mean ± 1 s.d."
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    out_path = os.path.join(
        save_dir,
        f"multiseed_summary_cond{conditioning_idx}.png"
    )

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Multi-seed summary figure saved: {out_path}")



def main():
    print("=" * 70)
    print("TRAINING DIVERSITY EVALUATION")
    print(f"Conditioning index : {CONDITIONING_IDX}")
    print(f"Max lag for MAE    : {MAX_LAG_M} m")
    print(f"jump_n_sample      : {JUMP_N_SAMPLE}")
    print("=" * 70)

    rows = []

    for model, seed in CONFIGS:

        run_name = f"{model}_seed{seed}"
        print(f"\n--- {run_name} ---")

        ensemble_path = os.path.join(
            BASE_ROBUST, run_name,
            f"jump{JUMP_N_SAMPLE}",
            f"conditioning_{CONDITIONING_IDX}",
            "ensemble.npy"
        )
        truth_path = os.path.join(
            BASE_ROBUST, run_name,
            f"jump{JUMP_N_SAMPLE}",
            f"conditioning_{CONDITIONING_IDX}",
            "truth.npy"
        )
        uncond_path = os.path.join(
            BASE_RUNS, run_name,
            "unconditional", "unconditional_1000.npy"
        )

        if not os.path.exists(ensemble_path):
            print(f"  MISSING ensemble: {ensemble_path} — skipping")
            continue
        if not os.path.exists(truth_path):
            print(f"  MISSING truth: {truth_path} — skipping")
            continue

        ensemble = np.load(ensemble_path).astype(np.float32)
        truth    = np.load(truth_path).astype(np.float32)

        print(f"  Ensemble shape: {ensemble.shape}")
        print(f"  Truth shape:    {truth.shape}")

        run_save_dir = os.path.join(SAVE_DIR, run_name)
        os.makedirs(run_save_dir, exist_ok=True)

        var_ref,  lags_m, points_y = compute_variograms(truth[np.newaxis], MASK_POSITION)
        var_cond, _,      _        = compute_variograms(ensemble,           MASK_POSITION)

        mae, row_maes = variogram_mae(var_ref, var_cond, points_y, MAX_LAG_M)
        print(f"  Variogram MAE (conditional): {mae:.6f}")

        if os.path.exists(uncond_path):
            uncond           = np.load(uncond_path).astype(np.float32)
            var_uncond, _, _ = compute_variograms(uncond, MASK_POSITION)
            uncond_mae, _    = variogram_mae(var_ref, var_uncond, points_y, MAX_LAG_M)
            print(f"  Variogram MAE (unconditional): {uncond_mae:.6f}")
        else:
            print(f"  Unconditional not found — skipping")
            var_uncond = {py: np.zeros_like(var_ref[py]) for py in points_y}
            uncond_mae = np.nan

        cov = percentile_coverage(ensemble, truth, MASK_POSITION)
        print(f"  P40-P60: {cov['P40-P60']['coverage']:.2f}%  (nominal 20%)")
        print(f"  P25-P75: {cov['P25-P75']['coverage']:.2f}%  (nominal 50%)")
        print(f"  P10-P90: {cov['P10-P90']['coverage']:.2f}%  (nominal 80%)")

        plot_variograms(var_ref, var_cond, var_uncond, lags_m, points_y,
                        model, seed, CONDITIONING_IDX, run_save_dir)
        plot_variance_map(ensemble, model, seed, CONDITIONING_IDX,
                          MASK_POSITION, run_save_dir)
        plot_percentile_analysis(ensemble, truth, MASK_POSITION,   # ← add this
                                 model, seed, CONDITIONING_IDX, run_save_dir)

        rows.append({
            'model':         model,
            'seed':          seed,
            'variogram_mae': mae,
            'uncond_mae':    uncond_mae,
            'P40_P60':       cov['P40-P60']['coverage'],
            'P25_P75':       cov['P25-P75']['coverage'],
            'P10_P90':       cov['P10-P90']['coverage'],
        })

    if not rows:
        print("\nNo results found — check ensembles have been generated.")
        return


    df = pd.DataFrame(rows)
    print("FULL RESULTS")
    print(df.to_string(index=False))

    plot_multiseed_summary(
    df,
    SAVE_DIR,
    CONDITIONING_IDX
)
    

    #1 Variogram MAE
    m1 = df[df["model"] == "M1"]["variogram_mae"].values
    m2 = df[df["model"] == "M2"]["variogram_mae"].values
    m3 = df[df["model"] == "M3"]["variogram_mae"].values

    stat, p = friedmanchisquare(m1, m2, m3)

    print("\nFriedman test — Variogram MAE")
    print(f"Statistic = {stat:.4f}")
    print(f"p-value   = {p:.4f}")


    #2 Percentile coverage errors
    coverage_tests = {
        "P40-P60": ("P40_P60", 20.0),
        "P25-P75": ("P25_P75", 50.0),
        "P10-P90": ("P10_P90", 80.0),
    }

    for name, (column, nominal) in coverage_tests.items():

        m1 = np.abs(
            df[df["model"] == "M1"][column].values - nominal
        )
        m2 = np.abs(
            df[df["model"] == "M2"][column].values - nominal
        )
        m3 = np.abs(
            df[df["model"] == "M3"][column].values - nominal
        )

        stat, p = friedmanchisquare(m1, m2, m3)

        print(f"\nFriedman test — {name} coverage error")
        print(f"Statistic = {stat:.4f}")
        print(f"p-value   = {p:.4f}")


    summary = df.groupby('model').agg(
        vario_mae_mean = ('variogram_mae', 'mean'),
        vario_mae_std  = ('variogram_mae', 'std'),
        P40_P60_mean   = ('P40_P60',       'mean'),
        P40_P60_std    = ('P40_P60',       'std'),
        P25_P75_mean   = ('P25_P75',       'mean'),
        P25_P75_std    = ('P25_P75',       'std'),
        P10_P90_mean   = ('P10_P90',       'mean'),
        P10_P90_std    = ('P10_P90',       'std'),
    ).reset_index()

    print("SUMMARY — Mean +/- Std across seeds")
    print(summary.to_string(index=False))

    csv_full    = os.path.join(SAVE_DIR, f"results_all_seeds_cond{CONDITIONING_IDX}.csv")
    csv_summary = os.path.join(SAVE_DIR, f"results_summary_cond{CONDITIONING_IDX}.csv")
    df.to_csv(csv_full,    index=False)
    summary.to_csv(csv_summary, index=False)

    print(f"\nFull results : {csv_full}")
    print(f"Summary      : {csv_summary}")


if __name__ == "__main__":
    main()

    #python geomodels_evaluate_mae.py --conditioning_idx 100