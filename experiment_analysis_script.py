#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

# ─── Import Custom Modules ───────────────────────────────────────────────────────
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
    get_nus,  # Add alignment function
)
from magnetstein_data.simple_wasserstein_descent import estimate_proportions_wasserstein

warnings.filterwarnings("ignore")

# ─── Output Configuration ────────────────────────────────────────────────────────
OUTPUT_FOLDER = "experiment_plots_final_4"

# ─── Centralized Hyperparameter Configuration ───────────────────────────────────
HYPERPARAMETERS = {
    # Joint MD Parameters
    "joint_md": {
        "C": 30,  # Bandwidth parameter for multidiagonal cost
        "reg": 1.5,  # KL regularization strength
        "eta_G": 0.5,  # Learning rate for transport plan
        "eta_p": 0.05,  # Learning rate for proportions
        "max_iter": 2000,  # Maximum iterations
        "gamma": 0.995,  # Learning rate decay
        "tol": 1e-4,  # Convergence tolerance
        "patience": 50,  # Early stopping patience
        "total_features": 3000,  # Total features to extract
        "mass_preservation_target": 90.0,  # Percentage of mass to preserve
    },
    # Simple Wasserstein Parameters
    "simple_ws": {
        "learning_rate": 0.05,  # Fixed learning rate
        "gamma": 0.995,  # Learning rate decay
        "T": 5000,  # Maximum iterations
        "patience": 80,  # Early stopping patience
        "tol": 1e-5,  # Convergence tolerance
        "total_features": 3000,  # Total features to extract
        "mass_preservation_target": 90.0,  # Percentage of mass to preserve
    },
    # Parameter grid search ranges
    "parameter_search": {
        "reg_m1_range": (2, 102),  # (min, max) for τ₁
        "reg_m2_range": (2, 102),  # (min, max) for τ₂
        "n_points": 3,  # Grid resolution
    },
}


def calculate_features_for_mass_target(
    spectrum: NMRSpectrum, target_mass_pct: float
) -> Dict:
    """
    Calculate how many features are needed to preserve exactly target_mass_pct% of total mass.

    Parameters:
        spectrum (NMRSpectrum): Input spectrum
        target_mass_pct (float): Target mass preservation percentage (0-100)

    Returns:
        Dict: Contains required_features, actual_mass_pct, total_features, target_mass_pct
    """
    # Sort features by intensity (descending)
    spectrum_confs_sorted = sorted(spectrum.confs, key=lambda x: x[1], reverse=True)
    original_mass = sum(intensity for _, intensity in spectrum.confs)
    target_mass = original_mass * (target_mass_pct / 100.0)

    cumulative_mass = 0
    required_features = 0

    for i, (_, intensity) in enumerate(spectrum_confs_sorted):
        cumulative_mass += intensity
        required_features = i + 1

        if cumulative_mass >= target_mass:
            break

    # Calculate actual mass preservation achieved
    actual_mass_pct = (
        (cumulative_mass / original_mass * 100) if original_mass > 0 else 0.0
    )

    return {
        "required_features": required_features,
        "actual_mass_pct": actual_mass_pct,
        "total_features": len(spectrum.confs),
        "target_mass_pct": target_mass_pct,
        "efficiency": required_features / len(spectrum.confs),
    }


def calculate_dynamic_features(
    spectra: List[NMRSpectrum], mix: NMRSpectrum, target_mass_pct: float = 90.0
) -> Dict:
    """
    Calculate the number of unique features needed to preserve target_mass_pct% for mix and all components.

    Returns:
        Dict: Contains feature analysis and the final unique feature count to use
    """
    print(f"Calculating features needed for {target_mass_pct}% mass preservation...")

    # Calculate features needed for mixture
    mix_analysis = calculate_features_for_mass_target(mix, target_mass_pct)
    print(
        f"  Mixture: {mix_analysis['required_features']} features for {mix_analysis['actual_mass_pct']:.1f}% mass"
    )

    # Calculate features needed for each component
    component_analyses = []
    for i, spectrum in enumerate(spectra):
        comp_analysis = calculate_features_for_mass_target(spectrum, target_mass_pct)
        comp_analysis["component_index"] = i
        component_analyses.append(comp_analysis)
        print(
            f"  Component {i}: {comp_analysis['required_features']} features for {comp_analysis['actual_mass_pct']:.1f}% mass"
        )

    # Extract features from all spectra based on calculated requirements
    mix_features_needed = mix_analysis["required_features"]
    mix_confs_sorted = sorted(mix.confs, key=lambda x: x[1], reverse=True)
    mix_top_features = set(v for v, _ in mix_confs_sorted[:mix_features_needed])

    component_top_features = set()
    for i, spectrum in enumerate(spectra):
        comp_features_needed = component_analyses[i]["required_features"]
        comp_confs_sorted = sorted(spectrum.confs, key=lambda x: x[1], reverse=True)
        comp_features = set(v for v, _ in comp_confs_sorted[:comp_features_needed])
        component_top_features.update(comp_features)

    # Calculate unique features across all spectra
    all_unique_features = mix_top_features.union(component_top_features)
    unique_features_count = len(all_unique_features)

    print(f"  Total unique features needed: {unique_features_count}")
    print(
        f"  Mix features: {len(mix_top_features)}, Component features: {len(component_top_features)}"
    )
    print(
        f"  Overlap: {len(mix_top_features.intersection(component_top_features))} features"
    )

    return {
        "mix_analysis": mix_analysis,
        "component_analyses": component_analyses,
        "mix_top_features": mix_top_features,
        "component_top_features": component_top_features,
        "all_unique_features": all_unique_features,
        "unique_features_count": unique_features_count,
        "target_mass_pct": target_mass_pct,
    }


def extract_features_from_unique_set(
    spectrum: NMRSpectrum, unique_features: set
) -> NMRSpectrum:
    """
    Extract features from spectrum that are in the unique_features set.
    """
    # Filter spectrum to only include features in the unique set
    filtered_confs = [
        (v, intensity) for v, intensity in spectrum.confs if v in unique_features
    ]

    # Sort by intensity (descending) to maintain signif_features behavior
    filtered_confs_sorted = sorted(filtered_confs, key=lambda x: x[1], reverse=True)

    # Create new spectrum and normalize
    reduced_spectrum = NMRSpectrum(
        confs=filtered_confs_sorted, protons=spectrum.protons
    )
    reduced_spectrum.normalize()

    return reduced_spectrum


def get_hyperparameter_text() -> str:
    """Generate formatted hyperparameter text for plots."""
    joint_params = HYPERPARAMETERS["joint_md"]
    simple_params = HYPERPARAMETERS["simple_ws"]

    text = (
        f"Joint MD: C={joint_params['C']}, reg={joint_params['reg']}, "
        f"η_G={joint_params['eta_G']:.0e}, η_p={joint_params['eta_p']:.0e}\n"
        f"Simple WS: lr={simple_params['learning_rate']}, "
        f"γ={simple_params['gamma']}, T={simple_params['T']}\n"
        f"Mass target: {joint_params['mass_preservation_target']}%"
    )
    return text


# ─── Style Setup (similar to pretty_plots) ──────────────────────────────────────
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


# ─── Experimental Configurations ────────────────────────────────────────────────
experiments_folders = {
    "experiment_1": "magnetstein_data/experiment_1_intensity_difference",
    # "experiment_2": "magnetstein_data/experiment_2_overlapping",
    "experiment_6": "magnetstein_data/experiment_6_miniperfumes",
    # "experiment_7": "magnetstein_data/experiment_7_overlapping_and_intensity_difference",
    # "experiment_8": "magnetstein_data/experiment_8_different_solvents",
    "experiment_3": "magnetstein_data/experiment_3_perfumes_and_absent_components",
    "experiment_5": "magnetstein_data/experiment_5_metabolites",
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


# ─── Data Loading Functions ──────────────────────────────────────────────────────
def load_experiment_data(
    exp_num: str, variant: int = 3
) -> Tuple[List[NMRSpectrum], NMRSpectrum, Dict]:
    """Load spectra and mixture data for a given experiment with dynamic feature calculation."""
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
        comp = NMRSpectrum(
            confs=list(zip(data[:, 0], data[:, 1])),
            protons=protons_dictionary[exp_num][i],
        )
        comp.label = components_dictionary[exp_num][i]
        comps.append(comp)

    # Clean and normalize
    mix_spec.trim_negative_intensities()
    mix_spec.normalize()
    for comp in comps:
        comp.trim_negative_intensities()
        comp.normalize()

    # Calculate dynamic features
    target_mass_pct = HYPERPARAMETERS["joint_md"]["mass_preservation_target"]
    feature_analysis = calculate_dynamic_features(comps, mix_spec, target_mass_pct)

    return comps, mix_spec, feature_analysis


def calculate_l1_error(estimated: List[float], ground_truth: List[float]) -> float:
    """Calculate L1 error between estimated and ground truth proportions."""
    return sum(abs(e - g) for e, g in zip(estimated, ground_truth))


# ─── Optimization Functions ──────────────────────────────────────────────────────
def run_joint_md_with_params(
    spectra: List[NMRSpectrum],
    mix: NMRSpectrum,
    reg_m1: float,
    reg_m2: float,
    total_features: int = None,
    feature_analysis: Dict = None,
) -> Dict:
    """Run joint mirror descent with specific parameters and dynamic feature selection."""
    # Use centralized hyperparameters
    params = HYPERPARAMETERS["joint_md"]
    if total_features is None:
        total_features = params["total_features"]

    n_components = len(spectra)

    if feature_analysis is not None:
        # Use dynamic feature calculation
        unique_features = feature_analysis["all_unique_features"]
        actual_features_used = len(unique_features)

        # Extract features using the unique feature set
        spectra_reduced = [
            extract_features_from_unique_set(spectrum, unique_features)
            for spectrum in spectra
        ]
        mix_reduced = extract_features_from_unique_set(mix, unique_features)

        print(f"Using {actual_features_used} dynamically calculated features")

    else:
        # Fallback to original method
        features_per_comp = total_features // n_components

        # Extract features
        spectra_reduced = [
            signif_features(spectrum, features_per_comp) for spectrum in spectra
        ]
        unique_v = sorted(
            {v for spectrum in spectra_reduced for v, _ in spectrum.confs}
        )
        total_unique_v = len(unique_v)
        mix_reduced = signif_features(mix, total_unique_v)
        actual_features_used = total_unique_v

    # Initial uniform proportions (no ground truth initialization)
    ratio = np.ones(n_components) / n_components

    nus = get_nus([si.confs for si in spectra_reduced])
    a = np.array([p for _, p in mix_reduced.confs])
    b = sum(nu_i * p_i for nu_i, p_i in zip(nus, ratio))
    v1 = np.array([v for v, _ in mix_reduced.confs])
    v2 = sorted({v for spectrum in spectra_reduced for v, _ in spectrum.confs})

    # Setup optimization using centralized parameters
    M = multidiagonal_cost(v1, v2, params["C"])
    warmstart = warmstart_sparse(a, b, params["C"])
    c = reg_distribiution(len(v2), params["C"])
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
                "features_used": actual_features_used,
                "hyperparameters": params.copy(),
                "reg_m1": reg_m1,
                "reg_m2": reg_m2,
                "feature_analysis": feature_analysis,
            }
        else:
            return {"success": False, "error": "Invalid result format"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_simple_wasserstein_with_params(
    spectra: List[NMRSpectrum],
    mix: NMRSpectrum,
    learning_rate: float = None,
    total_features: int = None,
    feature_analysis: Dict = None,
) -> Dict:
    """Run simple Wasserstein with specific parameters and dynamic feature selection."""
    # Use centralized hyperparameters
    params = HYPERPARAMETERS["simple_ws"]
    if learning_rate is None:
        learning_rate = params["learning_rate"]
    if total_features is None:
        total_features = params["total_features"]

    n_components = len(spectra)

    if feature_analysis is not None:
        # Use dynamic feature calculation
        unique_features = feature_analysis["all_unique_features"]
        actual_features_used = len(unique_features)

        # Extract features using the unique feature set
        spectra_to_use = [
            extract_features_from_unique_set(spectrum, unique_features)
            for spectrum in spectra
        ]
        mix_to_use = extract_features_from_unique_set(mix, unique_features)

        try:
            p_final, _, scores, ws_dist, ws_uniform = estimate_proportions_wasserstein(
                mix_to_use,
                spectra_to_use,
                learning_rate=learning_rate,
                gamma=params["gamma"],
                T=params["T"],
                tol=params["tol"],
                patience=params["patience"],
                n_features=None,  # Features already extracted
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    else:
        # Fallback to original method
        features_per_comp = total_features // n_components
        actual_features_used = total_features

        try:
            p_final, _, scores, ws_dist, ws_uniform = estimate_proportions_wasserstein(
                mix,
                spectra,
                learning_rate=learning_rate,
                gamma=params["gamma"],
                T=params["T"],
                tol=params["tol"],
                patience=params["patience"],
                n_features=features_per_comp,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    return {
        "proportions": p_final.tolist(),
        "cost": ws_dist,
        "ws_uniform": ws_uniform,
        "iterations": len(scores),
        "converged": len(scores) < params["T"],
        "success": True,
        "features_used": actual_features_used,
        "hyperparameters": params.copy(),
        "learning_rate": learning_rate,
        "feature_analysis": feature_analysis,
    }


# ─── Parameter Grid Generation (Parallel Worker Function) ────────────────────────
def process_parameter_combination(args):
    """Process a single parameter combination for an experiment."""
    exp_num, reg_m1, reg_m2 = args

    try:
        # Load data with dynamic feature calculation
        spectra, mix, feature_analysis = load_experiment_data(exp_num)
        ground_truth = ground_truth_molar_proportions[exp_num]

        # Run joint MD with dynamic features
        result = run_joint_md_with_params(
            spectra, mix, reg_m1, reg_m2, feature_analysis=feature_analysis
        )
        if result["success"]:
            l1_error = calculate_l1_error(result["proportions"], ground_truth)
            return {
                "experiment": exp_num,
                "method": "Joint MD",
                "param1": reg_m1,
                "param2": reg_m2,
                "param1_name": "reg_m1",
                "param2_name": "reg_m2",
                "l1_error": l1_error,
                "proportions": result["proportions"],
                "cost": result["cost"],
                "success": True,
                "features_used": result["features_used"],
                "target_mass_pct": feature_analysis["target_mass_pct"]
                if feature_analysis
                else None,
            }
        else:
            return {"success": False, "error": result.get("error", "Unknown error")}

    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_parameter_grid_results(exp_num: str, n_points: int = None) -> pd.DataFrame:
    """Generate results for parameter grid for a specific experiment using parallel processing."""
    print(f"Generating parameter grid for {exp_num}...")

    # Use centralized hyperparameters
    search_params = HYPERPARAMETERS["parameter_search"]
    if n_points is None:
        n_points = search_params["n_points"]

    print(
        f"  Using n_points = {n_points} (from {'parameter' if n_points != search_params['n_points'] else 'centralized config'})"
    )

    # Parameter ranges
    reg_m1_min, reg_m1_max = search_params["reg_m1_range"]
    reg_m2_min, reg_m2_max = search_params["reg_m2_range"]
    reg_m1_values = np.linspace(reg_m1_min, reg_m1_max, n_points)
    reg_m2_values = np.linspace(reg_m2_min, reg_m2_max, n_points)

    # Create parameter combinations
    param_combinations = [
        (exp_num, reg_m1, reg_m2)
        for reg_m1 in reg_m1_values
        for reg_m2 in reg_m2_values
    ]

    results = []

    # Use parallel processing for parameter sweep
    max_workers = min(mp.cpu_count(), len(param_combinations))
    print(
        f"  Running {len(param_combinations)} parameter combinations with {max_workers} workers..."
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(process_parameter_combination, args): args[1:]
            for args in param_combinations
        }

        for future in as_completed(future_to_params):
            reg_m1, reg_m2 = future_to_params[future]
            try:
                result = future.result()
                if result["success"]:
                    results.append(result)
            except Exception as exc:
                print(
                    f"  Parameter combination reg_m1={reg_m1:.1f}, reg_m2={reg_m2:.1f} failed: {exc}"
                )

    # Add single Simple Wasserstein comparison with dynamic features
    try:
        spectra, mix, feature_analysis = load_experiment_data(exp_num)
        ground_truth = ground_truth_molar_proportions[exp_num]
        simple_result = run_simple_wasserstein_with_params(
            spectra, mix, learning_rate=0.05, feature_analysis=feature_analysis
        )
        if simple_result["success"]:
            l1_error = calculate_l1_error(simple_result["proportions"], ground_truth)
            results.append(
                {
                    "experiment": exp_num,
                    "method": "Simple WS",
                    "param1": 0.05,
                    "param2": None,
                    "param1_name": "learning_rate",
                    "param2_name": None,
                    "l1_error": l1_error,
                    "proportions": simple_result["proportions"],
                    "cost": simple_result["cost"],
                    "success": True,
                    "features_used": simple_result["features_used"],
                    "target_mass_pct": feature_analysis["target_mass_pct"],
                }
            )
    except Exception as e:
        print(f"  Simple Wasserstein failed for {exp_num}: {e}")

    print(f"  Completed {exp_num}: {len(results)} successful results")
    return pd.DataFrame(results)


# ─── Visualization Functions ─────────────────────────────────────────────────────
def create_experiment_heatmap(df: pd.DataFrame, exp_num: str):
    """Create parameter heatmap for Joint MD method."""
    joint_data = df[df["method"] == "Joint MD"].copy()
    if joint_data.empty:
        print(f"No Joint MD data for {exp_num}")
        return

    # Create pivot table
    pivot_table = joint_data.pivot(index="param2", columns="param1", values="l1_error")

    # Find best parameters
    best_idx = joint_data["l1_error"].idxmin()
    best_params = joint_data.iloc[best_idx]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    # Create heatmap with enhanced styling
    im = ax.imshow(pivot_table.values, cmap="viridis", aspect="auto", origin="lower")

    # Add contour lines for better readability
    X, Y = np.meshgrid(range(len(pivot_table.columns)), range(len(pivot_table.index)))
    contours = ax.contour(
        X, Y, pivot_table.values, colors="white", alpha=0.4, linewidths=0.8
    )

    # Colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("L₁ Error", fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=11)

    # Better axis labels and formatting
    ax.set_xlabel("Regularization Parameter τ₁", fontsize=14, labelpad=10)
    ax.set_ylabel("Regularization Parameter τ₂", fontsize=14, labelpad=10)
    ax.set_title(
        f"{exp_num}: Joint MD Parameter Optimization",
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
    best_m1_idx = list(pivot_table.columns).index(best_params["param1"])
    best_m2_idx = list(pivot_table.index).index(best_params["param2"])
    ax.scatter(
        best_m1_idx,
        best_m2_idx,
        color="red",
        s=200,
        marker="*",
        linewidths=2,
        edgecolors="white",
        label=f"Optimum ({best_params['param1']:.0f}, {best_params['param2']:.0f})",
    )

    # Add legend for the optimum point
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save
    os.makedirs(f"{OUTPUT_FOLDER}/{exp_num}", exist_ok=True)
    plt.savefig(
        f"{OUTPUT_FOLDER}/{exp_num}/parameter_heatmap.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        f"{OUTPUT_FOLDER}/{exp_num}/parameter_heatmap.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def create_3d_surface_plot(df: pd.DataFrame, exp_num: str):
    """Create enhanced 3D surface plot."""
    joint_data = df[df["method"] == "Joint MD"].copy()
    if joint_data.empty:
        print(f"No Joint MD data for {exp_num}")
        return

    # Create pivot table
    pivot_table = joint_data.pivot(index="param2", columns="param1", values="l1_error")
    best_idx = joint_data["l1_error"].idxmin()
    best_params = joint_data.iloc[best_idx]

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
        best_params["param1"],
        best_params["param2"],
        best_params["l1_error"],
        color="red",
        s=100,
        marker="*",
        linewidths=2,
        edgecolors="white",
    )

    # Better axis labels
    ax.set_xlabel("τ₁", fontsize=14, labelpad=10)
    ax.set_ylabel("τ₂", fontsize=14, labelpad=10)
    ax.set_zlabel("L₁ Error", fontsize=14, labelpad=10)
    ax.set_title(
        f"{exp_num}: Parameter Optimization Surface",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label("L₁ Error", fontsize=12, labelpad=15)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_FOLDER}/{exp_num}/parameter_surface.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        f"{OUTPUT_FOLDER}/{exp_num}/parameter_surface.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def create_method_comparison_chart(df: pd.DataFrame, exp_num: str):
    """Create bar chart comparing methods at their best parameters."""
    joint_best = (
        df[df["method"] == "Joint MD"]["l1_error"].min()
        if not df[df["method"] == "Joint MD"].empty
        else float("inf")
    )
    simple_best = (
        df[df["method"] == "Simple WS"]["l1_error"].min()
        if not df[df["method"] == "Simple WS"].empty
        else float("inf")
    )
    ground_truth = ground_truth_molar_proportions[exp_num]

    # Get best results
    joint_best_row = (
        df[(df["method"] == "Joint MD") & (df["l1_error"] == joint_best)].iloc[0]
        if joint_best != float("inf")
        else None
    )
    simple_best_row = (
        df[(df["method"] == "Simple WS") & (df["l1_error"] == simple_best)].iloc[0]
        if simple_best != float("inf")
        else None
    )

    # Component names
    comp_names = components_dictionary[exp_num]

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Proportions comparison
    x = np.arange(len(comp_names))
    width = 0.25

    ax1.bar(
        x - width, ground_truth, width, label="Ground Truth", alpha=0.8, color="green"
    )

    if joint_best_row is not None:
        ax1.bar(
            x,
            joint_best_row["proportions"],
            width,
            label="Joint MD",
            alpha=0.8,
            color="blue",
        )
    if simple_best_row is not None:
        ax1.bar(
            x + width,
            simple_best_row["proportions"],
            width,
            label="Simple WS",
            alpha=0.8,
            color="red",
        )

    # Add value labels
    for i, gt in enumerate(ground_truth):
        ax1.text(
            i - width, gt + 0.01, f"{gt:.3f}", ha="center", va="bottom", fontsize=10
        )

    if joint_best_row is not None:
        for i, jt in enumerate(joint_best_row["proportions"]):
            ax1.text(i, jt + 0.01, f"{jt:.3f}", ha="center", va="bottom", fontsize=10)

    if simple_best_row is not None:
        for i, sw in enumerate(simple_best_row["proportions"]):
            ax1.text(
                i + width, sw + 0.01, f"{sw:.3f}", ha="center", va="bottom", fontsize=10
            )

    ax1.set_xlabel("Components", fontsize=14)
    ax1.set_ylabel("Proportion", fontsize=14)
    ax1.set_title(
        f"{exp_num}: Best Proportion Estimates", fontsize=14, fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Error comparison
    methods = []
    errors = []
    colors = []

    if joint_best != float("inf"):
        methods.append("Joint MD")
        errors.append(joint_best)
        colors.append("blue")

    if simple_best != float("inf"):
        methods.append("Simple WS")
        errors.append(simple_best)
        colors.append("red")

    if methods:
        bars = ax2.bar(methods, errors, color=colors, alpha=0.8)
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.001,
                f"{error:.4f}",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    ax2.set_ylabel("L₁ Error", fontsize=14)
    ax2.set_title(f"{exp_num}: Best L₁ Errors", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Add hyperparameter and feature information
    hyperparams_text = get_hyperparameter_text()

    # Add feature information if available
    features_info = ""
    if joint_best_row is not None and "features_used" in joint_best_row:
        features_info = f"\nFeatures used: {joint_best_row['features_used']}"
        if (
            "target_mass_pct" in joint_best_row
            and joint_best_row["target_mass_pct"] is not None
        ):
            features_info += f" (targeting {joint_best_row['target_mass_pct']}% mass)"

    full_text = hyperparams_text + features_info

    fig.text(
        0.02,
        0.02,
        full_text,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", alpha=0.8),
        verticalalignment="bottom",
    )

    # Save
    plt.savefig(
        f"{OUTPUT_FOLDER}/{exp_num}/method_comparison.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        f"{OUTPUT_FOLDER}/{exp_num}/method_comparison.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def create_parameter_trends(df: pd.DataFrame, exp_num: str):
    """Create enhanced parameter trend analysis plots focused on Joint MD."""
    joint_data = df[df["method"] == "Joint MD"].copy()
    simple_data = df[df["method"] == "Simple WS"].copy()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(16, 12), constrained_layout=True
    )

    # Colors for consistency
    color1, color2 = "#2E86C1", "#E74C3C"

    # Joint MD: Fix param2 at best value, vary param1
    if not joint_data.empty:
        best_idx = joint_data["l1_error"].idxmin()
        best_param2 = joint_data.iloc[best_idx]["param2"]
        best_param1 = joint_data.iloc[best_idx]["param1"]

        subset = joint_data[joint_data["param2"] == best_param2].sort_values("param1")
        if not subset.empty:
            ax1.errorbar(
                subset["param1"],
                subset["l1_error"],
                fmt="o-",
                color=color1,
                capsize=5,
                capthick=2,
                linewidth=2,
                markersize=6,
            )
            ax1.axvline(
                best_param1,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Optimum τ₁ = {best_param1:.0f}",
            )
        ax1.set_xlabel("τ₁", fontsize=14)
        ax1.set_ylabel("L₁ Error", fontsize=14)
        ax1.set_title(
            f"Error vs τ₁ (τ₂ = {best_param2:.0f})", fontsize=14, fontweight="bold"
        )
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Joint MD: Fix param1 at best value, vary param2
        subset = joint_data[joint_data["param1"] == best_param1].sort_values("param2")
        if not subset.empty:
            ax2.errorbar(
                subset["param2"],
                subset["l1_error"],
                fmt="o-",
                color=color2,
                capsize=5,
                capthick=2,
                linewidth=2,
                markersize=6,
            )
            ax2.axvline(
                best_param2,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Optimum τ₂ = {best_param2:.0f}",
            )
        ax2.set_xlabel("τ₂", fontsize=14)
        ax2.set_ylabel("L₁ Error", fontsize=14)
        ax2.set_title(
            f"Error vs τ₂ (τ₁ = {best_param1:.0f})", fontsize=14, fontweight="bold"
        )
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
    else:
        ax1.text(
            0.5,
            0.5,
            "No Joint MD data",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax2.text(
            0.5,
            0.5,
            "No Joint MD data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    # Parameter sensitivity analysis across different values
    if not joint_data.empty:
        # Show how error varies across all τ₁ values (average across τ₂)
        param1_avg = (
            joint_data.groupby("param1")["l1_error"].agg(["mean", "std"]).reset_index()
        )
        ax3.errorbar(
            param1_avg["param1"],
            param1_avg["mean"],
            yerr=param1_avg["std"],
            fmt="o-",
            color=color1,
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=6,
        )
        ax3.axvline(
            best_param1,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Optimum τ₁ = {best_param1:.0f}",
        )
        ax3.set_xlabel("τ₁", fontsize=14)
        ax3.set_ylabel("Average L₁ Error", fontsize=14)
        ax3.set_title("Parameter Sensitivity: τ₁", fontsize=14, fontweight="bold")
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
    else:
        ax3.text(
            0.5,
            0.5,
            "No Joint MD data",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )

    # Error distribution comparison between methods
    all_joint_errors = joint_data["l1_error"] if not joint_data.empty else []
    simple_error = simple_data["l1_error"].iloc[0] if not simple_data.empty else None

    if len(all_joint_errors) > 0:
        ax4.hist(
            all_joint_errors,
            bins=15,
            alpha=0.7,
            color="blue",
            label="Joint MD",
            density=True,
        )
        if simple_error is not None:
            ax4.axvline(
                simple_error,
                color="red",
                linestyle="--",
                linewidth=3,
                alpha=0.8,
                label=f"Simple WS: {simple_error:.4f}",
            )
        best_joint = min(all_joint_errors)
        ax4.axvline(
            best_joint,
            color="blue",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Best Joint MD: {best_joint:.4f}",
        )

    if len(all_joint_errors) == 0:
        ax4.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )

    ax4.set_xlabel("L₁ Error", fontsize=14)
    ax4.set_ylabel("Density", fontsize=14)
    ax4.set_title(
        "Error Distribution & Method Comparison", fontsize=14, fontweight="bold"
    )
    if len(all_joint_errors) > 0:
        ax4.legend(frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Save
    plt.savefig(
        f"{OUTPUT_FOLDER}/{exp_num}/parameter_trends.pdf", bbox_inches="tight", dpi=300
    )
    plt.savefig(
        f"{OUTPUT_FOLDER}/{exp_num}/parameter_trends.png", bbox_inches="tight", dpi=300
    )
    plt.close()


def create_summary_statistics(df: pd.DataFrame, exp_num: str):
    """Create comprehensive summary statistics plot with hyperparameters."""
    joint_data = df[df["method"] == "Joint MD"].copy()
    simple_data = df[df["method"] == "Simple WS"].copy()

    # Create figure with extra space for hyperparameters - increased figure size for additional plot
    fig = plt.figure(figsize=(20, 16), constrained_layout=True)

    # Create a grid layout with space for hyperparameters - now 3x2 for 6 plots
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.2], hspace=0.3, wspace=0.2)

    # Main plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Hyperparameter info space
    ax_hyper = fig.add_subplot(gs[2, :])
    ax_hyper.axis("off")

    # 1. Error distribution histogram
    ax1.hist(
        df["l1_error"],
        bins=15,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        linewidth=1,
    )
    ax1.axvline(
        df["l1_error"].min(),
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Best Joint MD",
    )
    ax1.axvline(
        simple_data["l1_error"].min() if not simple_data.empty else np.nan,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Best Simple WS",
    )
    ax1.set_xlabel("L₁ Error", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Distribution of L₁ Errors", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 2. Parameter correlation scatter (Joint MD only)
    if not joint_data.empty:
        scatter = ax2.scatter(
            joint_data["param1"],
            joint_data["param2"],
            c=joint_data["l1_error"],
            cmap="viridis",
            s=50,
            alpha=0.7,
        )
        best_idx = joint_data["l1_error"].idxmin()
        best_params = joint_data.iloc[best_idx]
        ax2.scatter(
            best_params["param1"],
            best_params["param2"],
            color="red",
            s=100,
            marker="*",
            edgecolors="white",
            linewidth=2,
        )
        ax2.set_xlabel("τ₁", fontsize=12)
        ax2.set_ylabel("τ₂", fontsize=12)
        ax2.set_title("Parameter Space Exploration", fontsize=14, fontweight="bold")
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("L₁ Error", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
    else:
        ax2.text(
            0.5,
            0.5,
            "No Joint MD data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    # 3. Average L1 and L2 errors by reg_m1 and reg_m2 parameters
    if not joint_data.empty:
        # Calculate L2 error for joint data (if not already present)
        if "l2_error" not in joint_data.columns:
            # We need ground truth to calculate L2 error
            ground_truth = ground_truth_molar_proportions[exp_num]
            joint_data["l2_error"] = joint_data["proportions"].apply(
                lambda props: np.sqrt(
                    sum((p - g) ** 2 for p, g in zip(props, ground_truth))
                )
            )

        # Group by parameters and calculate mean errors
        param_avg = (
            joint_data.groupby(["param1", "param2"])
            .agg({"l1_error": "mean", "l2_error": "mean"})
            .reset_index()
        )

        # Create bar plot comparing L1 and L2 errors
        x_pos = np.arange(len(param_avg))
        width = 0.35

        bars1 = ax3.bar(
            x_pos - width / 2,
            param_avg["l1_error"],
            width,
            label="L₁ Error",
            alpha=0.8,
            color="blue",
        )
        bars2 = ax3.bar(
            x_pos + width / 2,
            param_avg["l2_error"],
            width,
            label="L₂ Error",
            alpha=0.8,
            color="red",
        )

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.001,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        for bar in bars2:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.001,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Create x-axis labels showing parameter combinations
        param_labels = [
            f"({row['param1']:.0f},{row['param2']:.0f})"
            for _, row in param_avg.iterrows()
        ]
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=10)
        ax3.set_xlabel("(τ₁, τ₂) Parameter Pairs", fontsize=12)
        ax3.set_ylabel("Average Error", fontsize=12)
        ax3.set_title(
            "L₁ vs L₂ Error by Parameter Combination", fontsize=14, fontweight="bold"
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
    else:
        ax3.text(
            0.5,
            0.5,
            "No Joint MD data",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )

    # 4. Method comparison if both exist
    if not joint_data.empty and not simple_data.empty:
        methods = ["Joint MD", "Simple WS"]
        best_errors = [joint_data["l1_error"].min(), simple_data["l1_error"].min()]
        colors = ["blue", "red"]

        bars = ax4.bar(methods, best_errors, color=colors, alpha=0.8)
        for bar, error in zip(bars, best_errors):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.001,
                f"{error:.4f}",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    ax4.set_ylabel("Best L₁ Error", fontsize=12)
    ax4.set_title("Method Comparison", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Add hyperparameter information in dedicated space
    hyperparams_text = get_hyperparameter_text()
    search_params = HYPERPARAMETERS["parameter_search"]
    full_text = (
        f"Hyperparameters: {hyperparams_text}\n"
        f"Parameter Search: τ₁∈[{search_params['reg_m1_range'][0]}, {search_params['reg_m1_range'][1]}], "
        f"τ₂∈[{search_params['reg_m2_range'][0]}, {search_params['reg_m2_range'][1]}], "
        f"Grid: {search_params['n_points']}×{search_params['n_points']}"
    )

    ax_hyper.text(
        0.5,
        0.5,
        full_text,
        ha="center",
        va="center",
        transform=ax_hyper.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", alpha=0.8),
        wrap=True,
    )

    # Save
    plt.savefig(
        f"{OUTPUT_FOLDER}/{exp_num}/summary_statistics.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        f"{OUTPUT_FOLDER}/{exp_num}/summary_statistics.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def create_cross_experiment_summary():
    """Create summary visualizations across all experiments."""
    all_results = []

    # Collect best results from each experiment
    for exp_num in experiments_folders.keys():
        csv_path = f"{OUTPUT_FOLDER}/{exp_num}/parameter_results.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Get best result for Joint MD and single Simple WS result
            joint_data = df[df["method"] == "Joint MD"]
            simple_data = df[df["method"] == "Simple WS"]

            if not joint_data.empty:
                best_joint = joint_data.iloc[joint_data["l1_error"].idxmin()]
                all_results.append(
                    {
                        "experiment": exp_num,
                        "method": "Joint MD",
                        "l1_error": best_joint["l1_error"],
                        "n_components": len(components_dictionary[exp_num]),
                        "best_param1": best_joint["param1"],
                        "best_param2": best_joint["param2"],
                    }
                )

            if not simple_data.empty:
                simple_result = simple_data.iloc[0]  # Only one Simple WS result
                all_results.append(
                    {
                        "experiment": exp_num,
                        "method": "Simple WS",
                        "l1_error": simple_result["l1_error"],
                        "n_components": len(components_dictionary[exp_num]),
                        "best_param1": simple_result["param1"],
                        "best_param2": None,
                    }
                )

    if not all_results:
        print("No results found for summary")
        return

    summary_df = pd.DataFrame(all_results)

    # Create summary plots with dedicated space for hyperparameters
    fig = plt.figure(figsize=(18, 16), constrained_layout=True)

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

    # 1. L1 Error comparison across all experiments
    pivot_errors = summary_df.pivot(
        index="experiment", columns="method", values="l1_error"
    )
    pivot_errors.plot(kind="bar", ax=ax1, color=["blue", "red"], alpha=0.8)
    ax1.set_title(
        "Best L₁ Errors Across All Experiments", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("L₁ Error", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(title="Method")
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 2. Error vs number of components
    joint_summary = summary_df[summary_df["method"] == "Joint MD"]
    simple_summary = summary_df[summary_df["method"] == "Simple WS"]

    if not joint_summary.empty:
        ax2.scatter(
            joint_summary["n_components"],
            joint_summary["l1_error"],
            color="blue",
            s=100,
            alpha=0.7,
            label="Joint MD",
        )
    if not simple_summary.empty:
        ax2.scatter(
            simple_summary["n_components"],
            simple_summary["l1_error"],
            color="red",
            s=100,
            alpha=0.7,
            label="Simple WS",
        )

    ax2.set_xlabel("Number of Components", fontsize=12)
    ax2.set_ylabel("Best L₁ Error", fontsize=12)
    ax2.set_title("Error vs Problem Complexity", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # 3. Win/Loss ratio
    exp_comparison = []
    for exp in pivot_errors.index:
        if pd.notna(pivot_errors.loc[exp, "Joint MD"]) and pd.notna(
            pivot_errors.loc[exp, "Simple WS"]
        ):
            joint_err = pivot_errors.loc[exp, "Joint MD"]
            simple_err = pivot_errors.loc[exp, "Simple WS"]
            if joint_err < simple_err:
                exp_comparison.append("Joint MD")
            elif simple_err < joint_err:
                exp_comparison.append("Simple WS")
            else:
                exp_comparison.append("Tie")

    if exp_comparison:
        win_counts = pd.Series(exp_comparison).value_counts()
        colors = [
            "blue" if x == "Joint MD" else "red" if x == "Simple WS" else "gray"
            for x in win_counts.index
        ]
        ax3.pie(
            win_counts.values, labels=win_counts.index, autopct="%1.1f%%", colors=colors
        )
        ax3.set_title("Overall Win/Loss Ratio", fontsize=14, fontweight="bold")
    else:
        ax3.text(
            0.5,
            0.5,
            "No comparison data",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("Overall Win/Loss Ratio", fontsize=14, fontweight="bold")

    # 4. Parameter distribution for Joint MD
    if not joint_summary.empty:
        scatter = ax4.scatter(
            joint_summary["best_param1"],
            joint_summary["best_param2"],
            c=joint_summary["l1_error"],
            cmap="viridis",
            s=100,
            alpha=0.7,
        )
        ax4.set_xlabel("Best τ₁", fontsize=12)
        ax4.set_ylabel("Best τ₂", fontsize=12)
        ax4.set_title(
            "Optimal Parameters Distribution (Joint MD)", fontsize=14, fontweight="bold"
        )
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label("L₁ Error", fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
    else:
        ax4.text(
            0.5,
            0.5,
            "No Joint MD data",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Optimal Parameters Distribution", fontsize=14, fontweight="bold")

    # Add hyperparameter information in dedicated space
    hyperparams_text = get_hyperparameter_text()
    search_params = HYPERPARAMETERS["parameter_search"]
    full_text = (
        f"Configuration: {hyperparams_text} | "
        f"Search Space: τ₁∈{search_params['reg_m1_range']}, τ₂∈{search_params['reg_m2_range']}"
    )

    ax_hyper.text(
        0.5,
        0.5,
        full_text,
        ha="center",
        va="center",
        transform=ax_hyper.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.8),
        wrap=True,
    )

    # Save summary
    os.makedirs(f"{OUTPUT_FOLDER}/summary", exist_ok=True)
    plt.savefig(
        f"{OUTPUT_FOLDER}/summary/cross_experiment_summary.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        f"{OUTPUT_FOLDER}/summary/cross_experiment_summary.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Save summary data
    summary_df.to_csv(
        f"{OUTPUT_FOLDER}/summary/cross_experiment_results.csv", index=False
    )

    print(f"Cross-experiment summary saved to {OUTPUT_FOLDER}/summary/")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("CROSS-EXPERIMENT SUMMARY")
    print("=" * 60)

    if not joint_summary.empty:
        print(f"Joint MD - Best L1 Error: {joint_summary['l1_error'].min():.4f}")
        print(
            f"Joint MD - Average L1 Error: {joint_summary['l1_error'].mean():.4f} ± {joint_summary['l1_error'].std():.4f}"
        )

    if not simple_summary.empty:
        print(f"Simple WS - Best L1 Error: {simple_summary['l1_error'].min():.4f}")
        print(
            f"Simple WS - Average L1 Error: {simple_summary['l1_error'].mean():.4f} ± {simple_summary['l1_error'].std():.4f}"
        )

    if exp_comparison:
        joint_wins = exp_comparison.count("Joint MD")
        simple_wins = exp_comparison.count("Simple WS")
        ties = exp_comparison.count("Tie")
        print(f"Win/Loss: Joint MD {joint_wins}, Simple WS {simple_wins}, Ties {ties}")


# ─── Run Analysis for All Experiments ───────────────────────────────────────────
if __name__ == "__main__":
    # Start total timing
    total_start_time = time.time()

    # Print hyperparameters at start
    print("=" * 60)
    print("HYPERPARAMETER CONFIGURATION")
    print("=" * 60)
    for category, params in HYPERPARAMETERS.items():
        print(f"\n{category.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    print("=" * 60)

    # Create main output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Analyze each experiment
    for exp_num in experiments_folders.keys():
        exp_start_time = time.time()
        try:
            # Generate parameter grid results with dynamic features
            df = generate_parameter_grid_results(exp_num)

            if df.empty:
                print(f"No successful results for {exp_num}")
                continue

            # Create all visualizations
            create_experiment_heatmap(df, exp_num)
            create_3d_surface_plot(df, exp_num)
            create_method_comparison_chart(df, exp_num)
            create_parameter_trends(df, exp_num)
            create_summary_statistics(df, exp_num)

            # Save results
            df.to_csv(f"{OUTPUT_FOLDER}/{exp_num}/parameter_results.csv", index=False)

            exp_duration = time.time() - exp_start_time
            print(
                f"Completed analysis for {exp_num} in {exp_duration:.2f} seconds ({exp_duration / 60:.1f} minutes)"
            )
        except Exception as e:
            exp_duration = time.time() - exp_start_time
            print(f"Error analyzing {exp_num} after {exp_duration:.2f} seconds: {e}")
            import traceback

            traceback.print_exc()
            continue

    total_duration = time.time() - total_start_time
    print(
        f"Analysis complete! Check the {OUTPUT_FOLDER}/ directory for all visualizations."
    )
    print(
        f"Total analysis time: {total_duration:.2f} seconds ({total_duration / 60:.1f} minutes)"
    )

    # Create summary comparison across all experiments
    print("\nCreating cross-experiment summary...")
    summary_start_time = time.time()
    create_cross_experiment_summary()
    summary_duration = time.time() - summary_start_time
    print(f"Cross-experiment summary completed in {summary_duration:.2f} seconds")

    # Print final hyperparameter summary
    print("\n" + "=" * 60)
    print("FINAL HYPERPARAMETER SUMMARY")
    print("=" * 60)
    print("HYPERPARAMETERS used in this run:")
    for category, params in HYPERPARAMETERS.items():
        print(f"\n{category.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    print("=" * 60)

    final_total_duration = time.time() - total_start_time
    print(
        f"\nTOTAL RUNTIME: {final_total_duration:.2f} seconds ({final_total_duration / 60:.1f} minutes)"
    )
