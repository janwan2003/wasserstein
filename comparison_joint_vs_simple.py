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
    align_spectrum_shift,  # Add alignment function
)
from magnetstein_data.simple_wasserstein_descent import (
    estimate_proportions_wasserstein,
)

# ─── Output Configuration ────────────────────────────────────────────────────────
OUTPUT_FOLDER = "comparison_plots_final"

# Experimental configurations
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

experiments_folders = {
    "experiment_1": "magnetstein_data/experiment_1_intensity_difference",
    "experiment_2": "magnetstein_data/experiment_2_overlapping",
    "experiment_6": "magnetstein_data/experiment_6_miniperfumes",
    "experiment_7": "magnetstein_data/experiment_7_overlapping_and_intensity_difference",
    "experiment_8": "magnetstein_data/experiment_8_different_solvents",
    "experiment_3": "magnetstein_data/experiment_3_perfumes_and_absent_components",
    "experiment_5": "magnetstein_data/experiment_5_metabolites",
    "experiment_4": "magnetstein_data/experiment_9_and_4_shim",
    "experiment_9": "magnetstein_data/experiment_9_and_4_shim",
    "experiment_10": "magnetstein_data/experiment_10_bcaa",
    "experiment_11": "magnetstein_data/experiment_11_real_food_product",
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

    # Calculate dynamic features for 90% mass preservation
    target_mass_pct = HYPERPARAMETERS["joint_md"]["mass_preservation_target"]
    feature_analysis = calculate_dynamic_features(comps, mix_spec, target_mass_pct)

    return comps, mix_spec, feature_analysis


# ─── Centralized Hyperparameter Configuration ───────────────────────────────────
HYPERPARAMETERS = {
    # Joint MD Parameters
    "joint_md": {
        "C": 30,
        "reg": 1.5,
        "reg_m1": 40,
        "reg_m2": 50,
        "eta_G": 0.5,
        "eta_p": 0.05,
        "max_iter": 1000,
        "gamma": 0.995,
        "tol": 1e-4,
        "patience": 80,
        "mass_preservation_target": 90.0,  # Percentage of mass to preserve
    },
    # Simple Wasserstein Parameters
    "simple_ws": {
        "learning_rate": 0.05,
        "gamma": 0.995,
        "T": 5000,
        "patience": 80,
        "tol": 5e-5,
        "mass_preservation_target": 95.0,  # Percentage of mass to preserve
    },
}


def run_joint_md(
    spectra: List[NMRSpectrum],
    mix: NMRSpectrum,
    feature_analysis: Dict = None,
    total_features: int = None,
    **params,
) -> Dict:
    """Run joint mirror descent optimization with dynamic feature selection."""
    # Use centralized hyperparameters, allow override
    default_params = HYPERPARAMETERS["joint_md"].copy()
    default_params.update(params)

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
        if total_features is None:
            total_features = 2500  # Default fallback

        n_components = len(spectra)
        features_per_comp = total_features // n_components

        # Extract features from components
        spectra_reduced = [
            signif_features(spectrum, features_per_comp) for spectrum in spectra
        ]

        # Get unique values across all spectra
        unique_v = sorted(
            {v for spectrum in spectra_reduced for v, _ in spectrum.confs}
        )
        actual_features_used = len(unique_v)

        # Extract mixture features using the same unique values
        mix_reduced = signif_features(mix, actual_features_used)

    # Initial uniform proportions
    n_components = len(spectra)
    ratio = np.ones(n_components) / n_components

    # Prepare data for joint optimization
    nus = get_nus([si.confs for si in spectra_reduced])

    a = np.array([p for _, p in mix_reduced.confs])
    b = sum(nu_i * p_i for nu_i, p_i in zip(nus, ratio))

    v1 = np.array([v for v, _ in mix_reduced.confs])
    v2 = sorted({v for spectrum in spectra_reduced for v, _ in spectrum.confs})

    # Setup sparse optimization using centralized parameters
    C = default_params["C"]
    reg = default_params["reg"]
    reg_m1 = default_params["reg_m1"]
    reg_m2 = default_params["reg_m2"]
    eta_G = default_params["eta_G"]
    eta_p = default_params["eta_p"]
    max_iter = default_params["max_iter"]
    gamma = default_params["gamma"]
    tol = default_params["tol"]
    patience = default_params["patience"]

    M = multidiagonal_cost(v1, v2, C)
    warmstart = warmstart_sparse(a, b, C)
    c = reg_distribiution(len(v2), C)

    sparse = UtilsSparse(a, b, c, warmstart, M, reg, reg_m1, reg_m2)

    start_time = time.time()
    try:
        spectrum_confs = [si.confs for si in spectra_reduced]
        result = sparse.joint_md(
            spectrum_confs, eta_G, eta_p, max_iter, tol, gamma, patience
        )
        if isinstance(result, tuple) and len(result) == 2:
            G, p_final = result
        else:
            raise ValueError(f"joint_md returned unexpected result: {type(result)}")
    except Exception as e:
        print(f"Error in joint_md: {e}")
        raise
    end_time = time.time()

    # Calculate final cost
    final_cost = sparse.sparse_dot(G.data, G.offsets)

    return {
        "proportions": p_final.tolist(),
        "cost": final_cost,
        "time": end_time - start_time,
        "features_used": actual_features_used,
        "converged": True,
        "hyperparameters": default_params,
        "feature_analysis": feature_analysis,
        # Return the feature-reduced spectra for use in Simple WS
        "spectra_reduced": spectra_reduced,
        "mix_reduced": mix_reduced,
    }


def run_simple_wasserstein(
    spectra: List[NMRSpectrum],
    mix: NMRSpectrum,
    feature_analysis: Dict = None,
    total_features: int = None,
    spectra_reduced=None,
    mix_reduced=None,
    **params,
) -> Dict:
    """Run simple Wasserstein optimization using the same features as Joint MD."""
    # Use centralized hyperparameters, allow override
    default_params = HYPERPARAMETERS["simple_ws"].copy()
    default_params.update(params)

    # If pre-reduced spectra are provided (from Joint MD), use them
    # to ensure exactly the same features are used
    if spectra_reduced is not None and mix_reduced is not None:
        spectra_to_use = spectra_reduced
        mix_to_use = mix_reduced
        actual_features_used = len({v for v, _ in mix_reduced.confs})

    elif feature_analysis is not None:
        # Use dynamic feature calculation
        unique_features = feature_analysis["all_unique_features"]
        actual_features_used = len(unique_features)

        # Extract features using the unique feature set
        spectra_to_use = [
            extract_features_from_unique_set(spectrum, unique_features)
            for spectrum in spectra
        ]
        mix_to_use = extract_features_from_unique_set(mix, unique_features)

    else:
        # Fallback: extract features independently
        if total_features is None:
            total_features = 2500

        n_components = len(spectra)
        n_features_per_comp = total_features // n_components
        spectra_to_use = spectra
        mix_to_use = mix
        actual_features_used = total_features

    # Parameters from centralized config
    learning_rate = default_params["learning_rate"]
    gamma = default_params["gamma"]
    T = default_params["T"]
    patience = default_params["patience"]
    tol = default_params["tol"]

    start_time = time.time()

    if spectra_reduced is not None or feature_analysis is not None:
        # Use pre-reduced spectra or dynamic features
        p_final, traj, scores, ws_dist, ws_uniform = estimate_proportions_wasserstein(
            mix_to_use,
            spectra_to_use,
            learning_rate=learning_rate,
            gamma=gamma,
            T=T,
            tol=tol,
            patience=patience,
            n_features=None,  # Features already extracted
        )
    else:
        # Extract features within the function
        p_final, traj, scores, ws_dist, ws_uniform = estimate_proportions_wasserstein(
            mix_to_use,
            spectra_to_use,
            learning_rate=learning_rate,
            gamma=gamma,
            T=T,
            tol=tol,
            patience=patience,
            n_features=n_features_per_comp,
        )

    end_time = time.time()

    iterations = len(scores)
    converged = iterations < T

    return {
        "proportions": p_final.tolist(),
        "cost": ws_dist,
        "ws_uniform": ws_uniform,
        "time": end_time - start_time,
        "iterations": iterations,
        "converged": converged,
        "scores": scores,
        "features_used": actual_features_used,
        "hyperparameters": default_params,
    }


def calculate_metrics(estimated: List[float], ground_truth: List[float]) -> Dict:
    """Calculate comparison metrics."""
    l1_error = sum(abs(e - g) for e, g in zip(estimated, ground_truth))
    l2_error = np.sqrt(sum((e - g) ** 2 for e, g in zip(estimated, ground_truth)))
    max_error = max(abs(e - g) for e, g in zip(estimated, ground_truth))

    return {"l1_error": l1_error, "l2_error": l2_error, "max_error": max_error}


def run_single_experiment(args):
    """Run both joint_md and simple Wasserstein for a single experiment."""
    exp_num, joint_md_params, simple_ws_params = args

    try:
        # Load data with dynamic feature calculation
        spectra, mix, feature_analysis = load_experiment_data(exp_num)
        ground_truth = ground_truth_molar_proportions[exp_num]

        print(f"Processing {exp_num}: Components: {len(spectra)}")
        print(f"  Dynamic features: {feature_analysis['unique_features_count']}")
        print(f"  Mass preservation target: {feature_analysis['target_mass_pct']}%")

        # Run joint mirror descent with dynamic features
        joint_result = run_joint_md(
            spectra, mix, feature_analysis=feature_analysis, **joint_md_params
        )
        joint_metrics = calculate_metrics(joint_result["proportions"], ground_truth)

        # Run simple Wasserstein using the SAME features as Joint MD
        simple_result = run_simple_wasserstein(
            spectra,
            mix,
            spectra_reduced=joint_result["spectra_reduced"],
            mix_reduced=joint_result["mix_reduced"],
            **simple_ws_params,
        )
        simple_metrics = calculate_metrics(simple_result["proportions"], ground_truth)

        # Store results
        result = {
            "experiment": exp_num,
            "n_components": len(spectra),
            "ground_truth": ground_truth,
            # Feature analysis results
            "target_mass_pct": feature_analysis["target_mass_pct"],
            "mix_features_for_target": feature_analysis["mix_analysis"][
                "required_features"
            ],
            "mix_actual_mass_pct": feature_analysis["mix_analysis"]["actual_mass_pct"],
            # Joint MD results
            "joint_proportions": joint_result["proportions"],
            "joint_cost": joint_result["cost"],
            "joint_time": joint_result["time"],
            "joint_features": joint_result["features_used"],
            "joint_l1_error": joint_metrics["l1_error"],
            "joint_l2_error": joint_metrics["l2_error"],
            "joint_max_error": joint_metrics["max_error"],
            # Simple WS results
            "simple_proportions": simple_result["proportions"],
            "simple_cost": simple_result["cost"],
            "simple_ws_uniform": simple_result["ws_uniform"],
            "simple_time": simple_result["time"],
            "simple_iterations": simple_result["iterations"],
            "simple_converged": simple_result["converged"],
            "simple_features": simple_result["features_used"],
            "simple_l1_error": simple_metrics["l1_error"],
            "simple_l2_error": simple_metrics["l2_error"],
            "simple_max_error": simple_metrics["max_error"],
        }

        # Add individual component results
        comp_names = components_dictionary[exp_num]
        for i, name in enumerate(comp_names):
            result[f"gt_{name}"] = ground_truth[i]
            result[f"joint_{name}"] = joint_result["proportions"][i]
            result[f"simple_{name}"] = simple_result["proportions"][i]
            # Add component mass analysis
            comp_analysis = feature_analysis["component_analyses"][i]
            result[f"comp_{i}_features_for_target"] = comp_analysis["required_features"]
            result[f"comp_{i}_actual_mass_pct"] = comp_analysis["actual_mass_pct"]

        print(
            f"Completed {exp_num}: Joint L1={joint_metrics['l1_error']:.4f}, Simple L1={simple_metrics['l1_error']:.4f}"
        )
        print(
            f"  Features used: {joint_result['features_used']}, Mass preserved: {feature_analysis['mix_analysis']['actual_mass_pct']:.1f}%"
        )
        return result

    except Exception as e:
        print(f"Error processing {exp_num}: {e}")
        return None


def get_hyperparameter_text() -> str:
    """Generate formatted hyperparameter text for plots."""
    joint_params = HYPERPARAMETERS["joint_md"]
    simple_params = HYPERPARAMETERS["simple_ws"]

    text = (
        f"Joint MD: C={joint_params['C']}, reg={joint_params['reg']}, "
        f"τ₁={joint_params['reg_m1']}, τ₂={joint_params['reg_m2']}\n"
        f"Simple WS: lr={simple_params['learning_rate']}, "
        f"γ={simple_params['gamma']}, T={simple_params['T']}\n"
        f"Mass target: {joint_params['mass_preservation_target']}%"
    )
    return text


def create_individual_experiment_chart(result: Dict):
    """Create a comparison chart for a single experiment."""
    exp_num = result["experiment"]
    comp_names = components_dictionary[exp_num]

    # Extract proportions
    ground_truth = [result[f"gt_{name}"] for name in comp_names]
    joint_props = [result[f"joint_{name}"] for name in comp_names]
    simple_props = [result[f"simple_{name}"] for name in comp_names]

    # Create bar chart
    x = np.arange(len(comp_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 8))

    bars1 = ax.bar(
        x - width, ground_truth, width, label="Ground Truth", alpha=0.8, color="green"
    )
    bars2 = ax.bar(
        x, joint_props, width, label="Joint Mirror Descent", alpha=0.8, color="blue"
    )
    bars3 = ax.bar(
        x + width,
        simple_props,
        width,
        label="Simple Wasserstein",
        alpha=0.8,
        color="red",
    )

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    # Customize chart
    ax.set_xlabel("Components", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title(
        f"{exp_num} - Proportion Estimation Comparison\n"
        f"Joint MD L1 Error: {result['joint_l1_error']:.4f}, "
        f"Simple WS L1 Error: {result['simple_l1_error']:.4f}",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(comp_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add hyperparameter and feature information
    hyperparams_text = get_hyperparameter_text()
    ax.text(
        0.02,
        0.02,
        hyperparams_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", alpha=0.8),
    )

    # Add performance metrics and dynamic feature info
    metrics_text = (
        f"Joint MD: Time={result['joint_time']:.2f}s, Features={result['joint_features']}\n"
        f"Simple WS: Time={result['simple_time']:.2f}s, Iterations={result['simple_iterations']}\n"
        f"Mass target: {result['target_mass_pct']}%, Mix actual: {result['mix_actual_mass_pct']:.1f}%"
    )
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save chart
    os.makedirs(f"{OUTPUT_FOLDER}/individual", exist_ok=True)
    plt.savefig(
        f"{OUTPUT_FOLDER}/individual/{exp_num}_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def run_comparison_experiment():
    """Run comparison between joint_md and simple Wasserstein on all experiments."""
    # Use centralized hyperparameters
    joint_md_params = HYPERPARAMETERS["joint_md"]
    simple_ws_params = HYPERPARAMETERS["simple_ws"]

    # Prepare arguments for parallel processing
    experiment_args = [
        (exp_num, joint_md_params, simple_ws_params)
        for exp_num in experiments_folders.keys()
    ]

    results = []

    # Use parallel processing
    max_workers = min(mp.cpu_count(), len(experiment_args))
    print(
        f"Running {len(experiment_args)} experiments in parallel with {max_workers} workers..."
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_exp = {
            executor.submit(run_single_experiment, args): args[0]
            for args in experiment_args
        }

        # Collect results as they complete
        for future in as_completed(future_to_exp):
            exp_num = future_to_exp[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    # Create individual chart immediately
                    create_individual_experiment_chart(result)
            except Exception as exc:
                print(f"{exp_num} generated an exception: {exc}")

    return results


def create_visualizations(results: List[Dict]):
    """Create comparison visualizations."""
    if not results:
        print("No results to visualize")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)

    # L1 Error comparison
    plt.figure(figsize=(14, 8))
    x = range(len(df))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        df["joint_l1_error"],
        width,
        label="Joint Mirror Descent",
        alpha=0.8,
        color="blue",
    )
    plt.bar(
        [i + width / 2 for i in x],
        df["simple_l1_error"],
        width,
        label="Simple Wasserstein",
        alpha=0.8,
        color="red",
    )

    # Add value labels on bars
    for i, (joint_err, simple_err) in enumerate(
        zip(df["joint_l1_error"], df["simple_l1_error"])
    ):
        plt.text(
            i - width / 2,
            joint_err + 0.005,
            f"{joint_err:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        plt.text(
            i + width / 2,
            simple_err + 0.005,
            f"{simple_err:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Experiment", fontsize=12)
    plt.ylabel("L1 Error", fontsize=12)
    plt.title("L1 Error Comparison: Joint MD vs Simple Wasserstein", fontsize=14)
    plt.xticks(x, df["experiment"], rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_FOLDER}/l1_error_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Time comparison
    plt.figure(figsize=(14, 8))
    plt.bar(
        [i - width / 2 for i in x],
        df["joint_time"],
        width,
        label="Joint Mirror Descent",
        alpha=0.8,
        color="blue",
    )
    plt.bar(
        [i + width / 2 for i in x],
        df["simple_time"],
        width,
        label="Simple Wasserstein",
        alpha=0.8,
        color="red",
    )

    # Add value labels on bars
    for i, (joint_time, simple_time) in enumerate(
        zip(df["joint_time"], df["simple_time"])
    ):
        plt.text(
            i - width / 2,
            joint_time + 0.5,
            f"{joint_time:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        plt.text(
            i + width / 2,
            simple_time + 0.5,
            f"{simple_time:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Experiment", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Computation Time Comparison", fontsize=14)
    plt.xticks(x, df["experiment"], rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/time_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Scatter plot: Joint vs Simple L1 errors
    plt.figure(figsize=(10, 10))
    plt.scatter(
        df["joint_l1_error"], df["simple_l1_error"], alpha=0.7, s=150, c="purple"
    )

    # Add diagonal line
    min_val = min(df["joint_l1_error"].min(), df["simple_l1_error"].min())
    max_val = max(df["joint_l1_error"].max(), df["simple_l1_error"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, linewidth=2)

    plt.xlabel("Joint Mirror Descent L1 Error", fontsize=12)
    plt.ylabel("Simple Wasserstein L1 Error", fontsize=12)
    plt.title("L1 Error: Joint MD vs Simple Wasserstein", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add experiment labels
    for i, row in df.iterrows():
        plt.annotate(
            row["experiment"].replace("experiment_", "exp"),
            (row["joint_l1_error"], row["simple_l1_error"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            alpha=0.8,
        )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/l1_error_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Summary statistics chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Method comparison by number of components
    n_comp_groups = df.groupby("n_components").agg(
        {
            "joint_l1_error": "mean",
            "simple_l1_error": "mean",
        }
    )

    comp_counts = n_comp_groups.index
    ax1.bar(
        [x - 0.2 for x in comp_counts],
        n_comp_groups["joint_l1_error"],
        0.4,
        label="Joint MD",
        alpha=0.8,
        color="blue",
    )
    ax1.bar(
        [x + 0.2 for x in comp_counts],
        n_comp_groups["simple_l1_error"],
        0.4,
        label="Simple WS",
        alpha=0.8,
        color="red",
    )
    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("Average L1 Error")
    ax1.set_title("Average L1 Error by Component Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Win/Loss ratio
    joint_wins = sum(df["joint_l1_error"] < df["simple_l1_error"])
    simple_wins = sum(df["simple_l1_error"] < df["joint_l1_error"])
    ties = len(df) - joint_wins - simple_wins

    ax2.pie(
        [joint_wins, simple_wins, ties],
        labels=[
            f"Joint MD ({joint_wins})",
            f"Simple WS ({simple_wins})",
            f"Ties ({ties})",
        ],
        autopct="%1.1f%%",
        colors=["blue", "red", "gray"],
    )
    ax2.set_title("Win/Loss Ratio (Lower L1 Error Wins)")

    # Time efficiency
    ax3.scatter(
        df["joint_time"],
        df["joint_l1_error"],
        alpha=0.7,
        s=100,
        color="blue",
        label="Joint MD",
    )
    ax3.scatter(
        df["simple_time"],
        df["simple_l1_error"],
        alpha=0.7,
        s=100,
        color="red",
        label="Simple WS",
    )
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("L1 Error")
    ax3.set_title("Time vs Accuracy Trade-off")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Features used
    ax4.bar(df["experiment"], df["joint_features"], alpha=0.8, color="green")
    ax4.set_xlabel("Experiment")
    ax4.set_ylabel("Features Used (Joint MD)")
    ax4.set_title("Features Used by Joint MD")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/summary_statistics.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main execution function."""
    print("Starting comparison experiment: Joint Mirror Descent vs Simple Wasserstein")
    print("Using dynamic feature calculation based on mass preservation")

    # Print hyperparameters
    print("\n" + "=" * 60)
    print("HYPERPARAMETER CONFIGURATION")
    print("=" * 60)
    for method, params in HYPERPARAMETERS.items():
        print(f"\n{method.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    print("=" * 60)

    start_time = time.time()
    results = run_comparison_experiment()
    end_time = time.time()

    if not results:
        print("No successful experiments to analyze")
        return

    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("joint_vs_simple_comparison_results_dynamic.csv", index=False)

    # Create visualizations
    create_visualizations(results)

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total experiments: {len(results)}")
    print(
        f"Average features used: {df['joint_features'].mean():.0f} ± {df['joint_features'].std():.0f}"
    )
    print(f"Average mass target: {df['target_mass_pct'].mean():.1f}%")
    print(
        f"Average actual mix mass preserved: {df['mix_actual_mass_pct'].mean():.1f}% ± {df['mix_actual_mass_pct'].std():.1f}%"
    )
    print(
        f"Joint MD average L1 error: {df['joint_l1_error'].mean():.4f} ± {df['joint_l1_error'].std():.4f}"
    )
    print(
        f"Simple WS average L1 error: {df['simple_l1_error'].mean():.4f} ± {df['simple_l1_error'].std():.4f}"
    )
    print(
        f"Joint MD average time: {df['joint_time'].mean():.2f} ± {df['joint_time'].std():.2f} seconds"
    )
    print(
        f"Simple WS average time: {df['simple_time'].mean():.2f} ± {df['simple_time'].std():.2f} seconds"
    )

    # Count wins
    joint_wins = sum(df["joint_l1_error"] < df["simple_l1_error"])
    simple_wins = sum(df["simple_l1_error"] < df["joint_l1_error"])
    ties = len(df) - joint_wins - simple_wins
    print(f"Joint MD wins: {joint_wins}/{len(results)}")
    print(f"Simple WS wins: {simple_wins}/{len(results)}")
    print(f"Ties: {ties}/{len(results)}")

    print(f"\nResults saved to: joint_vs_simple_comparison_results_dynamic.csv")
    print(f"Plots saved to: {OUTPUT_FOLDER}/")
    print(f"Individual experiment charts: {OUTPUT_FOLDER}/individual/")


if __name__ == "__main__":
    main()
