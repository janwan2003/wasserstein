import sys
import os

sys.path.append(os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

from SimpleDecent.test_utils import (
    multidiagonal_cost,
    reg_distribiution,
    warmstart_sparse,
    UtilsSparse,
    signif_features,
    dia_matrix
)
from wasserstein import NMRSpectrum, Spectrum
from binsearch_estimation import binsearch_p

# experiment settings
protons_dict = {
    "experiment_1": [16, 12],
    "experiment_2": [16, 16],
    "experiment_6": [16, 12],
    "experiment_7": [12, 8],
    "experiment_8": [12, 8],
}
ground_truth = {
    "experiment_1": 0.09088457406472417,
    "experiment_2": 0.505,
    "experiment_6": 0.3865,
    "experiment_7": 0.8403875207510383,
    "experiment_8": 0.3702,
}
folders = {
    exp: Path(f"{exp}_intensity_difference")
    if exp == "experiment_1"
    else Path(f"{exp}_miniperfumes")
    if exp == "experiment_6"
    else Path(f"{exp}_overlapping_and_intensity_difference")
    if exp == "experiment_7"
    else Path(f"{exp}_different_solvents")
    if exp == "experiment_8"
    else Path(f"{exp}_overlapping")
    for exp in ground_truth
}
METHOD = "mirror_descent"  # or "lbfgsb"


# construct data for one experiment and p
def construct_data(exp: str, N: int):
    folder = folders[exp]
    # load mix spectrum and downsample
    mix_arr = np.loadtxt(folder / "preprocessed_mix.csv", delimiter=",")
    mix_spec = NMRSpectrum(confs=list(zip(mix_arr[:, 0], mix_arr[:, 1])))
    mix_spec = signif_features(mix_spec, 2 * N)

    # load and downsample component spectra
    comps = []
    for i in range(2):
        arr = np.loadtxt(folder / f"preprocessed_comp{i}.csv", delimiter=",")
        comp_spec = NMRSpectrum(
            confs=list(zip(arr[:, 0], arr[:, 1])), protons=protons_dict[exp][i]
        )
        comps.append(signif_features(comp_spec, N))

    return mix_spec, comps


# average L1 error across experiments for given hyperparameters
def avg_l1_error(
    mix,
    spectra,
    exp: str,
    N: int,
    C: int,
    reg: float,
    regm1: float,
    regm2: float,
    max_iter: int,
    gamma: float,
):
    true_p = ground_truth[exp]
    p_hat = binsearch_p(mix, spectra, exp, N, C, reg, regm1, regm2, max_iter, gamma)[0]
    # print(f"Experiment: {exp}, regm1: {regm1}, regm2: {regm2}, p_hat: {p_hat:.4f}, true_p: {true_p:.4f}")
    # calculate L1 error
    return abs(p_hat - true_p)


# wrapper for hyperparam pairs
def process_pair(args):
    mix, spectra, exp, N, C, reg, r1, r2, max_iter, gamma = args
    return (r1, r2), avg_l1_error(
        mix, spectra, exp, N, C, reg, r1, r2, max_iter, gamma
    )


# plot heatmap
def plot_heatmap(exp, data, xvals, yvals, N, C, reg):
    fig, ax = plt.subplots(figsize=(11, 11))
    im = ax.imshow(data, origin="lower", cmap="viridis_r")
    ax.set_title(f"N={N}, C={C}, reg={reg} — L1 Error")
    ax.set_xlabel("regm2")
    ax.set_ylabel("regm1")
    ax.set_xticks(range(len(xvals)))
    ax.set_yticks(range(len(yvals)))
    ax.set_xticklabels([f"{v:.3f}" for v in xvals], rotation=45, ha="right")
    ax.set_yticklabels([f"{v:.3f}" for v in yvals])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", fontsize=8)
    ax.figure.colorbar(im, ax=ax, label="L1 Error")
    ax.set_aspect("equal")
    plt.tight_layout()
    out = Path(f"plots/l1_error_heatmaps/{exp}")
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "avg_l1_error_heatmap.png")
    plt.close()

# main
def main():
    N, C = 500, 20
    reg, max_iter = 1.5, 400
    gamma = 1.0 - (10.0 / max_iter)
    regm1_vals = np.linspace(20, 240, 6)
    regm2_vals = np.linspace(20, 240, 6)

    
    for exp in ground_truth.keys():
        if exp == "experiment_6" or exp == "experiment_1":
            continue
        print(f"Processing {exp}...")
        mix, spectra = construct_data(exp, N)

        grid = np.zeros((len(regm1_vals), len(regm2_vals)))
        tasks = [
            (mix, spectra, exp, N, C, reg, r1, r2, max_iter, gamma)
            for r1 in regm1_vals
            for r2 in regm2_vals
        ]
        results = []
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [
                executor.submit(process_pair, task) for task in tasks
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing"
            ):
                results.append(future.result())
        for (r1, r2), val in results:
            i = np.where(regm1_vals == r1)[0][0]
            j = np.where(regm2_vals == r2)[0][0]
            grid[i, j] = val
        plot_heatmap(exp, grid, regm2_vals, regm1_vals, N, C, reg)


if __name__ == "__main__":
    main()
