import sys
import os

sys.path.append(os.path.abspath(".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from wasserstein import NMRSpectrum
from simple_wasserstein_descent import (
    estimate_proportions_wasserstein,
    ws_mix,
    numpy_to_torch_tensor,
)

plot_dir = "plots"

components_dictionary = {
    "experiment_1": ["Pinene", "Benzyl benzoate"],
    "experiment_2": ["Pinene", "Limonene"],
    "experiment_7": ["Benzyl benzoate", "m Anisaldehyde"],
    "experiment_3": [
        "Isopropyl myristate",
        "Benzyl benzoate",
        "Alpha pinene",
        "Limonene",
    ],
    "experiment_6": ["Pinene", "Benzyl benzoate"],
    "experiment_5": [
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
    "experiment_4": [
        "Lactate",
        "Alanine",
        "Creatine",
        "Creatinine",
        "Choline chloride",
    ],
    "experiment_8": ["Benzyl benzoate", "m Anisaldehyde"],
    "experiment_10": ["Leucine", "Isoleucine", "Valine"],
    "experiment_11": ["Leucine", "Isoleucine", "Valine"],
}

protons_dictionary = {
    "experiment_1": [16, 12],
    "experiment_2": [16, 16],
    "experiment_7": [12, 8],
    "experiment_3": [34, 12, 16, 16],
    "experiment_6": [16, 12],
    "experiment_5": [4, 4, 5, 5, 13],
    "experiment_9": [4, 4, 5, 5, 13],
    "experiment_4": [4, 4, 5, 5, 13],
    "experiment_8": [12, 8],
    "experiment_10": [10, 10, 8],
    "experiment_11": [10, 10, 8],
}

# comment out if want to run only a subset of experiments
experiments_folders = {
    "experiment_1": "experiment_1_intensity_difference",
    "experiment_6": "experiment_6_miniperfumes",
    # "experiment_5": "experiment_5_metabolites",
    "experiment_7": "experiment_7_overlapping_and_intensity_difference",
    # "experiment_9": "experiment_9_and_4_shim",
    # "experiment_4": "experiment_9_and_4_shim",
    "experiment_8": "experiment_8_different_solvents",
    # "experiment_3": "experiment_3_perfumes_and_absent_components",
    "experiment_2": "experiment_2_overlapping",
    # "experiment_10": "experiment_10_bcaa",
    # "experiment_11": "experiment_11_real_food_product",
}

# parameters
learning_rate = 0.05
T = 5000
gamma = 0.995
patience = 100
tol = 1e-6
n_features_list = np.linspace(200, 70400, num=20, dtype=int)
# n_features_list = np.logspace(np.log2(200), np.log2(70400), num=20, dtype=int, base=2)

print(f"Using n_features_list: {n_features_list}")
# add variant index and ground truth proportions
variant = 3
ground_truth_molar_proportions = {
    "experiment_1": [0.09088457406472417, 0.9091154259352758],
    "experiment_2": [0.505, 0.495],
    "experiment_7": [0.8403875207510383, 0.1596124792489616],
    "experiment_3": [
        0.7264578344443725,
        0.10578603326645526,
        0.081968804608116,
        0.08578732768105625,
    ],
    "experiment_6": [0.3865, 0.6135],
    "experiment_5": [0.3022, 0.2240, 0.1253, 0.2028, 0.1457],
    "experiment_9": [0.3022, 0.2240, 0.1253, 0.2028, 0.1457],
    "experiment_4": [0.3022, 0.2240, 0.1253, 0.2028, 0.1457],
    "experiment_8": [0.3702, 0.6298],
    "experiment_10": [
        0.3401,
        0.3299,
        0.3299,  # variant 1
    ]
    if variant == 0
    else [
        0.2525,
        0.2475,
        0.5,  # variant 2
    ]
    if variant == 1
    else [
        0.2538,
        0.4949,
        0.2513,  # variant 3
    ]
    if variant == 2
    else [
        0.2075,
        0.3942,
        0.3983,  # variant 4
    ],
    "experiment_11": [0.4855, 0.2427, 0.2718],
}

all_results = []
for n_features in n_features_list:
    print(f"Running experiments with n_features={n_features}")
    results = []
    for exp_num, folder in experiments_folders.items():
        num = exp_num.split("_")[1]

        # load mix (special cases 4,9,10)
        if exp_num == "experiment_10":
            mix_file = f"preprocessed_mix_variant_{variant + 1}.csv"
        elif num in ["9", "4"]:
            mix_file = f"preprocessed_exp{num}_mix.csv"
        else:
            mix_file = "preprocessed_mix.csv"
        mix_path = os.path.join(folder, mix_file)
        mix_arr = np.loadtxt(mix_path, delimiter=",")
        mix_spec = NMRSpectrum(confs=list(zip(mix_arr[:, 0], mix_arr[:, 1])))

        # load component spectra (special cases 4,9,10)
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

        # estimate
        p_final, traj, scores, ws_dist, ws_uniform = estimate_proportions_wasserstein(
            mix_spec,
            comps,
            learning_rate=learning_rate,
            gamma=gamma,
            T=T,
            tol=tol,
            patience=patience,
            n_features=n_features,
        )

        # convergence info
        iterations = len(scores)
        converged = iterations < T
        print(
            f"{exp_num}: ws_fit={ws_dist:.4f}, ws_uniform={ws_uniform:.4f}, "
            f"iterations={iterations}, converged={converged}"
        )

        # convergence plot
        plt.figure()
        plt.plot(range(1, iterations + 1), scores, marker="o")
        plt.title(f"{exp_num} convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Wasserstein distance")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{exp_num}_convergence.png"))
        plt.close()

        # WS error against ground truth mix
        gt = ground_truth_molar_proportions[exp_num]
        spectra_t = [numpy_to_torch_tensor(s.confs) for s in comps]
        mix_t = numpy_to_torch_tensor(mix_spec.confs)
        ws_est_actual = ws_mix(
            torch.tensor(gt, dtype=torch.float64), spectra_t, mix_t
        ).item()
        print(f"{exp_num}: ws_error_vs_truth={ws_est_actual:.4f}")

        # collect results
        res = dict(
            experiment=exp_num,
            ws_distance=ws_dist,
            ws_uniform=ws_uniform,
            ws_error_vs_truth=ws_est_actual,
            iterations=iterations,
            converged=converged,
            error_L1=sum(abs(p - g) for p, g in zip(p_final.tolist(), gt)),
        )
        for name, val in zip(components_dictionary[exp_num], p_final.tolist()):
            res[name] = val
        for name, g in zip(components_dictionary[exp_num], gt):
            res[f"gt_{name}"] = g

        results.append(res)
    # tag and accumulate
    for r in results:
        r["n_features"] = n_features
    all_results.extend(results)

# # save combined results
df = pd.DataFrame(all_results)
# df.to_csv("simple_ws_estimation_results.csv", index=False)

# create output directory for plots
os.makedirs(plot_dir, exist_ok=True)

# plot L1 error vs n_features for each experiment
plt.figure(figsize=(12, 8))
df_pivot = df.pivot(index="n_features", columns="experiment", values="error_L1")
for exp in df_pivot.columns:
    plt.plot(df_pivot.index, df_pivot[exp], marker="o", label=exp)
plt.xlabel("n_features")
plt.ylabel("L1 error")
plt.title("L1 error vs n_features per experiment")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig(os.path.join(plot_dir, "l1_error_vs_n_features.png"))
plt.close()

# per‐experiment comparison: estimated vs ground truth
comparison_plot_dir = os.path.join(plot_dir, "comparisons")
os.makedirs(comparison_plot_dir, exist_ok=True)
for _, row in df.iterrows():
    exp = row["experiment"]
    comps = components_dictionary[exp]
    est = [row[c] for c in comps]
    gt = [row[f"gt_{c}"] for c in comps]
    x = np.arange(len(comps))
    width = 0.35
    n_features = row["n_features"]
    l1_error = row["error_L1"]

    plt.figure()
    rects1 = plt.bar(x - width / 2, est, width, label="Estimated")
    rects2 = plt.bar(x + width / 2, gt, width, label="Ground truth")

    # Add labels on top of bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            plt.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    plt.xticks(x, comps, rotation=45, ha="right")
    plt.ylabel("Proportion")
    plt.title(
        f"{exp}: estimated vs ground truth\n"
        f"n_features: {n_features}, L1 error: {l1_error:.4f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(comparison_plot_dir, f"{exp}_n{n_features}_comparison.png")
    )
    plt.close()

# plot Wasserstein distances across experiments
plt.figure()
plt.plot(df["experiment"], df["ws_distance"], marker="o", label="fitted")
plt.plot(df["experiment"], df["ws_uniform"], marker="x", label="uniform")
plt.xticks(rotation=45)
plt.title("Wasserstein distances")
plt.xlabel("Experiment")
plt.ylabel("Distance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "ws_distances.png"))
plt.close()
