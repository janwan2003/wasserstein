import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("SimpleDecent"))
sys.path.append(os.path.abspath("magnetstein_data"))

from wasserstein import NMRSpectrum
from SimpleDecent.test_utils import (
    signif_features,
    multidiagonal_cost,
    reg_distribiution,
    warmstart_sparse,
    UtilsSparse,
    get_nus,
)

# ─── Output Configuration ────────────────────────────────────────────────────────
OUTPUT_FOLDER = "parameter_tuning_plots_final"

# ─── Centralized Hyperparameter Configuration ───────────────────────────────────
HYPERPARAMETERS = {
    "optimization": {
        "C": 20,
        "reg": 1.5,
        "eta_G": 5e-3,
        "eta_p": 5e-4,
        "max_iter": 5000,
        "gamma": 0.998,
        "tol": 6e-4,
        "patience": 50,
        "total_features": 5000,
    },
    "parameter_search": {
        "reg_m1_range": (10, 300),
        "reg_m2_range": (10, 300),
        "n_points": 10,
    },
}


def get_hyperparameter_text() -> str:
    """Generate formatted hyperparameter text for plots."""
    params = HYPERPARAMETERS["optimization"]
    search = HYPERPARAMETERS["parameter_search"]

    text = (
        f"Fixed: C={params['C']}, reg={params['reg']}, η_G={params['eta_G']:.0e}, "
        f"η_p={params['eta_p']:.0e}, max_iter={params['max_iter']}\n"
        f"Search: τ₁∈{search['reg_m1_range']}, τ₂∈{search['reg_m2_range']}, "
        f"Grid: {search['n_points']}×{search['n_points']}"
    )
    return text


# Use same experimental configurations as comparison script
experiments_folders = {
    "experiment_1": "magnetstein_data/experiment_1_intensity_difference",
    "experiment_2": "magnetstein_data/experiment_2_overlapping",
    "experiment_6": "magnetstein_data/experiment_6_miniperfumes",
    "experiment_7": "magnetstein_data/experiment_7_overlapping_and_intensity_difference",
    "experiment_8": "magnetstein_data/experiment_8_different_solvents",
    # "experiment_3": "magnetstein_data/experiment_3_perfumes_and_absent_components",
    # "experiment_5": "magnetstein_data/experiment_5_metabolites",
    # "experiment_4": "magnetstein_data/experiment_9_and_4_shim",
    # "experiment_9": "magnetstein_data/experiment_9_and_4_shim",
    # "experiment_10": "magnetstein_data/experiment_10_bcaa",
    # "experiment_11": "magnetstein_data/experiment_11_real_food_product",
}

components_dictionary = {
    "experiment_1": ["Pinene", "Benzyl benzoate"],
    "experiment_2": ["Pinene", "Limonene"],
    "experiment_6": ["Pinene", "Benzyl benzoate"],
    "experiment_7": ["Benzyl benzoate", "m Anisaldehyde"],
    "experiment_8": ["Benzyl benzoate", "m Anisaldehyde"],
    "experiment_3": [
        "Isopropyl myristate",
        "Benzyl benzoate",
        "Alpha pinene",
        "Limonene",
    ],
    "experiment_5": [
        "Lactate",
        "Alanine",
        "Creatine",
        "Creatinine",
        "Choline chloride",
    ],
    "experiment_4": [
        "Lactate",
        "Alanine",
        "Creatine",
        "Creatinine",
        "Choline chloride",
    ],
    "experiment_9": [
        "Lactate",
        "Alanine",
        "Creatine",
        "Creatinine",
        "Choline chloride",
    ],
    "experiment_10": ["Leucine", "Isoleucine", "Valine"],
    "experiment_11": ["Leucine", "Isoleucine", "Valine"],
}

protons_dictionary = {
    "experiment_1": [16, 12],
    "experiment_2": [16, 16],
    "experiment_6": [16, 12],
    "experiment_7": [12, 8],
    "experiment_8": [12, 8],
    "experiment_3": [34, 12, 16, 16],
    "experiment_5": [4, 4, 5, 5, 13],
    "experiment_4": [4, 4, 5, 5, 13],
    "experiment_9": [4, 4, 5, 5, 13],
    "experiment_10": [10, 10, 8],
    "experiment_11": [10, 10, 8],
}

ground_truth_molar_proportions = {
    "experiment_1": [0.09088457406472417, 0.9091154259352758],
    "experiment_2": [0.505, 0.495],
    "experiment_6": [0.3865, 0.6135],
    "experiment_7": [0.8403875207510383, 0.1596124792489616],
    "experiment_8": [0.3702, 0.6298],
    "experiment_3": [
        0.7264578344443725,
        0.10578603326645526,
        0.081968804608116,
        0.08578732768105625,
    ],
    "experiment_5": [0.3022, 0.2240, 0.1253, 0.2028, 0.1457],
    "experiment_4": [0.3022, 0.2240, 0.1253, 0.2028, 0.1457],
    "experiment_9": [0.3022, 0.2240, 0.1253, 0.2028, 0.1457],
    "experiment_10": [0.2538, 0.4949, 0.2513],  # variant 3
    "experiment_11": [0.4855, 0.2427, 0.2718],
}


def load_experiment_data(
    exp_num: str, variant: int = 3
) -> Tuple[List[NMRSpectrum], NMRSpectrum]:
    """Load spectra and mixture data for a given experiment."""
    folder = experiments_folders[exp_num]
    num = exp_num.split("_")[1]

    # Load mixture
    if exp_num == "experiment_10":
        mix_file = f"preprocessed_mix_variant_{variant + 1}.csv"
    elif num in ["9", "4"]:
        mix_file = f"preprocessed_exp{num}_mix.csv"
    else:
        mix_file = "preprocessed_mix.csv"

    mix_path = os.path.join(folder, mix_file)
    mix_arr = np.loadtxt(mix_path, delimiter=",")
    mix_spec = NMRSpectrum(confs=list(zip(mix_arr[:, 0], mix_arr[:, 1])))

    # Load component spectra
    comps = []
    for i in range(len(components_dictionary[exp_num])):
        if exp_num == "experiment_10":
            comp_file = f"preprocessed_variant_{variant + 1}_comp{i}.csv"
        elif num in ["9", "4"]:
            comp_file = f"preprocessed_exp{num}_comp{i}.csv"
        else:
            comp_file = f"preprocessed_comp{i}.csv"

        data = np.loadtxt(os.path.join(folder, comp_file), delimiter=",")
        comps.append(
            NMRSpectrum(
                confs=list(zip(data[:, 0], data[:, 1])),
                protons=protons_dictionary[exp_num][i],
            )
        )

    # Clean and normalize
    mix_spec.trim_negative_intensities()
    mix_spec.normalize()
    for comp in comps:
        comp.trim_negative_intensities()
        comp.normalize()

    return comps, mix_spec


def run_joint_md_single(
    spectra: List[NMRSpectrum],
    mix: NMRSpectrum,
    reg_m1: float,
    reg_m2: float,
    total_features: int = None,
) -> Dict:
    """Run joint mirror descent optimization with specific reg_m1 and reg_m2 values."""
    # Use centralized hyperparameters
    params = HYPERPARAMETERS["optimization"]
    if total_features is None:
        total_features = params["total_features"]

    n_components = len(spectra)
    features_per_comp = total_features // n_components

    # Extract features
    spectra_reduced = [
        signif_features(spectrum, features_per_comp) for spectrum in spectra
    ]

    # Get unique values across all spectra
    unique_v = sorted({v for spectrum in spectra_reduced for v, _ in spectrum.confs})
    total_unique_v = len(unique_v)

    mix_reduced = signif_features(mix, total_unique_v)

    # Initial uniform proportions
    ratio = np.ones(n_components) / n_components
    nus = get_nus([si.confs for si in spectra_reduced])

    a = np.array([p for _, p in mix_reduced.confs])
    b = sum(nu_i * p_i for nu_i, p_i in zip(nus, ratio))

    v1 = np.array([v for v, _ in mix_reduced.confs])
    v2 = unique_v

    # Use centralized parameters
    M = multidiagonal_cost(v1, v2, params["C"])
    warmstart = warmstart_sparse(a, b, params["C"])
    c = reg_distribiution(total_unique_v, params["C"])

    sparse = UtilsSparse(a, b, c, warmstart, M, params["reg"], reg_m1, reg_m2)

    try:
        spectrum_confs = [si.confs for si in spectra_reduced]
        result = sparse.joint_md(
            spectrum_confs,
            params["eta_G"],
            params["eta_p"],
            params["max_iter"],
            params["tol"],
            params["gamma"],
            params["patience"],
        )

        if isinstance(result, tuple) and len(result) == 2:
            G, p_final = result
            final_cost = sparse.sparse_dot(G.data, G.offsets)

            return {
                "proportions": p_final.tolist(),
                "cost": final_cost,
                "success": True,
                "features_used": total_unique_v,
                "hyperparameters": params,
                "reg_m1": reg_m1,
                "reg_m2": reg_m2,
            }
        else:
            return {"success": False, "error": "Invalid result format"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_l1_error(estimated: List[float], ground_truth: List[float]) -> float:
    """Calculate L1 error between estimated and ground truth proportions."""
    return sum(abs(e - g) for e, g in zip(estimated, ground_truth))


def run_parameter_combination(args):
    """Run optimization for a single parameter combination across all experiments."""
    reg_m1, reg_m2, experiments = args

    experiment_l1_errors = []
    experiment_results = []

    print(f"Testing reg_m1={reg_m1:.1f}, reg_m2={reg_m2:.1f}")

    for exp_num in experiments:
        try:
            # Load data
            spectra, mix = load_experiment_data(exp_num)
            ground_truth = ground_truth_molar_proportions[exp_num]

            # Run optimization
            result = run_joint_md_single(spectra, mix, reg_m1, reg_m2)

            if result["success"]:
                # Calculate L1 error
                l1_error = calculate_l1_error(result["proportions"], ground_truth)
                experiment_l1_errors.append(l1_error)

                experiment_results.append(
                    {
                        "experiment": exp_num,
                        "reg_m1": reg_m1,
                        "reg_m2": reg_m2,
                        "l1_error": l1_error,
                        "proportions": result["proportions"],
                        "ground_truth": ground_truth,
                        "cost": result["cost"],
                        "features_used": result["features_used"],
                    }
                )

                print(f"  {exp_num}: L1 error: {l1_error:.4f}")
            else:
                print(f"  {exp_num}: Failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"  {exp_num}: Error: {e}")

    # Calculate average L1 error across experiments
    if experiment_l1_errors:
        avg_l1_error = np.mean(experiment_l1_errors)
        std_l1_error = np.std(experiment_l1_errors)

        print(
            f"  reg_m1={reg_m1:.1f}, reg_m2={reg_m2:.1f}: Average L1 error: {avg_l1_error:.4f} ± {std_l1_error:.4f}"
        )

        return {
            "reg_m1": reg_m1,
            "reg_m2": reg_m2,
            "avg_l1_error": avg_l1_error,
            "std_l1_error": std_l1_error,
            "num_experiments": len(experiment_l1_errors),
            "individual_results": experiment_results,
        }
    else:
        print(f"  reg_m1={reg_m1:.1f}, reg_m2={reg_m2:.1f}: No successful experiments")
        return None


def parameter_tuning_experiment():
    """Perform grid search over reg_m1 and reg_m2 parameters using parallel processing."""
    print("Starting parameter tuning for reg_m1 and reg_m2")

    # Print hyperparameters
    print("\n" + "=" * 60)
    print("HYPERPARAMETER CONFIGURATION")
    print("=" * 60)
    for category, params in HYPERPARAMETERS.items():
        print(f"\n{category.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    print("=" * 60)

    # Use centralized hyperparameters
    search_params = HYPERPARAMETERS["parameter_search"]
    reg_m1_min, reg_m1_max = search_params["reg_m1_range"]
    reg_m2_min, reg_m2_max = search_params["reg_m2_range"]
    n_points = search_params["n_points"]

    reg_m1_values = np.linspace(reg_m1_min, reg_m1_max, n_points)
    reg_m2_values = np.linspace(reg_m2_min, reg_m2_max, n_points)

    # Experiments to test
    experiments = list(experiments_folders.keys())

    # Create parameter combinations
    param_combinations = [
        (reg_m1, reg_m2, experiments)
        for reg_m1 in reg_m1_values
        for reg_m2 in reg_m2_values
    ]

    total_combinations = len(param_combinations)

    print(
        f"Testing {len(reg_m1_values)} × {len(reg_m2_values)} = {total_combinations} parameter combinations"
    )
    print(f"Across {len(experiments)} experiments each")

    start_time = time.time()
    results = []

    # Use parallel processing
    max_workers = min(mp.cpu_count(), total_combinations)
    print(f"Running with {max_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all parameter combination jobs
        future_to_params = {
            executor.submit(run_parameter_combination, args): args[
                :2
            ]  # (reg_m1, reg_m2)
            for args in param_combinations
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_params):
            reg_m1, reg_m2 = future_to_params[future]
            completed += 1

            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                print(
                    f"Completed {completed}/{total_combinations}: reg_m1={reg_m1:.1f}, reg_m2={reg_m2:.1f}"
                )

            except Exception as exc:
                print(
                    f"Parameter combination reg_m1={reg_m1:.1f}, reg_m2={reg_m2:.1f} generated an exception: {exc}"
                )

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    return results


def analyze_and_visualize_results(results: List[Dict]):
    """Analyze results and create visualizations with hyperparameter info."""
    if not results:
        print("No results to analyze")
        return

    # Style setup similar to pretty_plots.ipynb
    import seaborn as sns
    import matplotlib as mpl

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "axes.labelsize": 14,
            "font.size": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 1.6,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 300,
        }
    )

    # Convert to DataFrame
    df_summary = pd.DataFrame(
        [
            {
                "reg_m1": r["reg_m1"],
                "reg_m2": r["reg_m2"],
                "avg_l1_error": r["avg_l1_error"],
                "std_l1_error": r["std_l1_error"],
                "num_experiments": r["num_experiments"],
            }
            for r in results
            if "avg_l1_error" in r
        ]
    )

    if df_summary.empty:
        print("No successful results to analyze")
        return

    # Find best parameters
    best_idx = df_summary["avg_l1_error"].idxmin()
    best_params = df_summary.iloc[best_idx]

    print(f"\n=== BEST PARAMETERS ===")
    print(f"reg_m1: {best_params['reg_m1']:.1f}")
    print(f"reg_m2: {best_params['reg_m2']:.1f}")
    print(
        f"Average L1 error: {best_params['avg_l1_error']:.4f} ± {best_params['std_l1_error']:.4f}"
    )

    # Save detailed results
    df_summary.to_csv("parameter_tuning_regm_summary.csv", index=False)

    # Save individual results
    individual_results = []
    for r in results:
        if "individual_results" in r:
            individual_results.extend(r["individual_results"])

    if individual_results:
        df_individual = pd.DataFrame(individual_results)
        df_individual.to_csv("parameter_tuning_regm_individual.csv", index=False)

    # Create visualizations directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Enhanced Heatmap with hyperparameters
    pivot_table = df_summary.pivot(
        index="reg_m2", columns="reg_m1", values="avg_l1_error"
    )

    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)

    # Use a better colormap and add contour lines
    im = ax.imshow(pivot_table.values, cmap="viridis", aspect="auto", origin="lower")

    # Add contour lines for better readability
    X, Y = np.meshgrid(range(len(pivot_table.columns)), range(len(pivot_table.index)))
    contours = ax.contour(
        X, Y, pivot_table.values, colors="white", alpha=0.4, linewidths=0.8
    )

    # Colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Average L₁ Error", fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=11)

    # Better axis labels and formatting
    ax.set_xlabel("Regularization Parameter λ₁", fontsize=14, labelpad=10)
    ax.set_ylabel("Regularization Parameter λ₂", fontsize=14, labelpad=10)
    ax.set_title(
        "Parameter Optimization: L₁ Error Landscape",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Set tick labels with better spacing
    n_ticks = 5
    x_indices = np.linspace(0, len(pivot_table.columns) - 1, n_ticks, dtype=int)
    y_indices = np.linspace(0, len(pivot_table.index) - 1, n_ticks, dtype=int)

    ax.set_xticks(x_indices)
    ax.set_yticks(y_indices)
    ax.set_xticklabels([f"{pivot_table.columns[i]:.0f}" for i in x_indices])
    ax.set_yticklabels([f"{pivot_table.index[i]:.0f}" for i in y_indices])

    # Mark best point with a star
    best_m1_idx = list(pivot_table.columns).index(best_params["reg_m1"])
    best_m2_idx = list(pivot_table.index).index(best_params["reg_m2"])
    ax.scatter(
        best_m1_idx,
        best_m2_idx,
        color="red",
        s=200,
        marker="*",
        linewidths=2,
        edgecolors="white",
        label=f"Optimum ({best_params['reg_m1']:.0f}, {best_params['reg_m2']:.0f})",
    )

    # Add legend for the optimum point
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add hyperparameter text
    fig.text(
        0.02,
        0.02,
        hyperparams_text,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.8),
        verticalalignment="bottom",
    )

    plt.savefig(f"{OUTPUT_FOLDER}/regm_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{OUTPUT_FOLDER}/regm_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Enhanced 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    reg_m1_mesh, reg_m2_mesh = np.meshgrid(pivot_table.columns, pivot_table.index)

    # Create surface with better styling
    surf = ax.plot_surface(
        reg_m1_mesh,
        reg_m2_mesh,
        pivot_table.values,
        cmap="viridis",
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )

    # Add wireframe for better structure visualization
    ax.plot_wireframe(
        reg_m1_mesh,
        reg_m2_mesh,
        pivot_table.values,
        color="white",
        alpha=0.1,
        linewidth=0.5,
    )

    # Mark best point
    ax.scatter(
        best_params["reg_m1"],
        best_params["reg_m2"],
        best_params["avg_l1_error"],
        color="red",
        s=100,
        marker="*",
        linewidths=2,
        edgecolors="white",
    )

    # Better axis labels
    ax.set_xlabel("λ₁", fontsize=14, labelpad=10)
    ax.set_ylabel("λ₂", fontsize=14, labelpad=10)
    ax.set_zlabel("Average L₁ Error", fontsize=14, labelpad=10)
    ax.set_title(
        "Parameter Optimization Surface", fontsize=16, fontweight="bold", pad=20
    )

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label("Average L₁ Error", fontsize=12, labelpad=15)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/regm_surface.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{OUTPUT_FOLDER}/regm_surface.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Enhanced line plots with error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Colors for consistency
    color1, color2 = "#2E86C1", "#E74C3C"

    # Fix reg_m2 at best value, vary reg_m1
    best_m2 = best_params["reg_m2"]
    subset_m1 = df_summary[df_summary["reg_m2"] == best_m2].sort_values("reg_m1")

    ax1.errorbar(
        subset_m1["reg_m1"],
        subset_m1["avg_l1_error"],
        yerr=subset_m1["std_l1_error"],
        fmt="o-",
        color=color1,
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=6,
    )
    ax1.axvline(
        best_params["reg_m1"],
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"Optimum λ₁ = {best_params['reg_m1']:.0f}",
    )
    ax1.set_xlabel("Regularization Parameter λ₁", fontsize=14)
    ax1.set_ylabel("Average L₁ Error", fontsize=14)
    ax1.set_title(f"Error vs λ₁ (λ₂ = {best_m2:.0f})", fontsize=14, fontweight="bold")
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Fix reg_m1 at best value, vary reg_m2
    best_m1 = best_params["reg_m1"]
    subset_m2 = df_summary[df_summary["reg_m1"] == best_m1].sort_values("reg_m2")

    ax2.errorbar(
        subset_m2["reg_m2"],
        subset_m2["avg_l1_error"],
        yerr=subset_m2["std_l1_error"],
        fmt="o-",
        color=color2,
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=6,
    )
    ax2.axvline(
        best_params["reg_m2"],
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"Optimum λ₂ = {best_params['reg_m2']:.0f}",
    )
    ax2.set_xlabel("Regularization Parameter λ₂", fontsize=14)
    ax2.set_ylabel("Average L₁ Error", fontsize=14)
    ax2.set_title(f"Error vs λ₂ (λ₁ = {best_m1:.0f})", fontsize=14, fontweight="bold")
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Add hyperparameter text
    fig.text(
        0.02,
        0.02,
        hyperparams_text,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.8),
        verticalalignment="bottom",
    )

    plt.savefig(f"{OUTPUT_FOLDER}/regm_line_plots.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{OUTPUT_FOLDER}/regm_line_plots.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 4. Summary statistics plot with dedicated hyperparameter space
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)

    # Create a grid layout with space for hyperparameters
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.15], hspace=0.3, wspace=0.2)

    # Main plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Hyperparameter info space
    ax_hyper = fig.add_subplot(gs[2, :])
    ax_hyper.axis("off")

    # Error distribution histogram
    ax1.hist(
        df_summary["avg_l1_error"],
        bins=15,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        linewidth=1,
    )
    ax1.axvline(
        best_params["avg_l1_error"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Minimum: {best_params['avg_l1_error']:.4f}",
    )
    ax1.set_xlabel("Average L₁ Error", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Distribution of L₁ Errors", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Parameter correlation
    scatter = ax2.scatter(
        df_summary["reg_m1"],
        df_summary["reg_m2"],
        c=df_summary["avg_l1_error"],
        cmap="viridis",
        s=50,
        alpha=0.7,
    )
    ax2.scatter(
        best_params["reg_m1"],
        best_params["reg_m2"],
        color="red",
        s=100,
        marker="*",
        edgecolors="white",
        linewidth=2,
    )
    ax2.set_xlabel("λ₁", fontsize=12)
    ax2.set_ylabel("λ₂", fontsize=12)
    ax2.set_title("Parameter Space Exploration", fontsize=14, fontweight="bold")
    cbar2 = plt.colorbar(scatter, ax=ax2)
    cbar2.set_label("L₁ Error", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Error vs reg_m1 (all reg_m2 values)
    for m2_val in sorted(df_summary["reg_m2"].unique())[
        ::2
    ]:  # Skip every other for clarity
        subset = df_summary[df_summary["reg_m2"] == m2_val].sort_values("reg_m1")
        ax3.plot(
            subset["reg_m1"],
            subset["avg_l1_error"],
            "o-",
            alpha=0.7,
            label=f"λ₂ = {m2_val:.0f}",
        )
    ax3.set_xlabel("λ₁", fontsize=12)
    ax3.set_ylabel("Average L₁ Error", fontsize=12)
    ax3.set_title("Error Trends Across λ₁", fontsize=14, fontweight="bold")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Error vs reg_m2 (all reg_m1 values)
    for m1_val in sorted(df_summary["reg_m1"].unique())[
        ::2
    ]:  # Skip every other for clarity
        subset = df_summary[df_summary["reg_m1"] == m1_val].sort_values("reg_m2")
        ax4.plot(
            subset["reg_m2"],
            subset["avg_l1_error"],
            "o-",
            alpha=0.7,
            label=f"λ₁ = {m1_val:.0f}",
        )
    ax4.set_xlabel("λ₂", fontsize=12)
    ax4.set_ylabel("Average L₁ Error", fontsize=12)
    ax4.set_title("Error Trends Across λ₂", fontsize=14, fontweight="bold")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax4.grid(True, alpha=0.3)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Add hyperparameter text in dedicated space
    hyperparams_text = get_hyperparameter_text()
    ax_hyper.text(
        0.5,
        0.5,
        hyperparams_text,
        ha="center",
        va="center",
        transform=ax_hyper.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.8),
        wrap=True,
    )

    plt.savefig(f"{OUTPUT_FOLDER}/regm_summary.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{OUTPUT_FOLDER}/regm_summary.png", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"\nResults saved to:")
    print(f"- parameter_tuning_regm_summary.csv")
    print(f"- parameter_tuning_regm_individual.csv")
    print(f"- {OUTPUT_FOLDER}/regm_heatmap.pdf/.png")
    print(f"- {OUTPUT_FOLDER}/regm_surface.pdf/.png")
    print(f"- {OUTPUT_FOLDER}/regm_line_plots.pdf/.png")
    print(f"- {OUTPUT_FOLDER}/regm_summary.pdf/.png")


def main():
    """Main execution function."""
    print("Parameter Tuning for reg_m1 and reg_m2")
    print("=" * 50)

    # Run parameter tuning experiment
    results = parameter_tuning_experiment()

    # Analyze and visualize results
    analyze_and_visualize_results(results)


if __name__ == "__main__":
    main()
