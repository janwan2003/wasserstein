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
)
from wasserstein import NMRSpectrum

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
def construct_data(exp: str, N: int, C: int, p: float):
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

    # mixture vector a
    a = np.array([inten for _, inten in mix_spec.confs])
    # build b on mix support
    comp_maps = [{freq: inten for freq, inten in comp.confs} for comp in comps]
    b = np.array(
        [
            p * comp_maps[0].get(freq, 0) + (1 - p) * comp_maps[1].get(freq, 0)
            for freq, _ in mix_spec.confs
        ]
    )

    v1 = np.array([freq for freq, _ in mix_spec.confs])
    v2 = v1.copy()
    M = multidiagonal_cost(v1, v2, C)
    c = reg_distribiution(len(a), C)
    return a, b, c, M


# compute cost for a single p (top-level for pickling)
def compute_cost(args):
    exp, p, N, C, reg, regm1, regm2, max_iter, gamma, step = args

    a, b, c, M = construct_data(exp, N, C, p)
    G0 = warmstart_sparse(a, b, C)
    sparse = UtilsSparse(a, b, c, G0, M, reg, regm1, regm2)

    if METHOD == "lbfgsb":
        _, G = sparse.lbfgsb_unbalanced(numItermax=max_iter)
    else:
        _, G = sparse.mirror_descent_unbalanced(
            numItermax=max_iter,
            gamma=gamma,
            step_size=step,
            stopThr=1e-6,
            patience=100,
        )

    # --- Filter only integer offsets from G dict ---
    G_offsets = []
    G_data = []
    for key, diag in G.items():
        try:
            off = int(key)
        except (ValueError, TypeError):
            continue
        G_offsets.append(off)
        G_data.append(diag)

    transport_cost = sparse.sparse_dot(G_data, G_offsets)
    rm1 = sparse.marg_tv_sparse_rm1(G_data, G_offsets)
    rm2 = sparse.marg_tv_sparse_rm2(G_data, G_offsets)

    cost = transport_cost + rm1 / regm1 + rm2 / regm2
    return cost, p


# find best p for one experiment
def find_best_p(
    exp: str,
    N: int,
    C: int,
    reg: float,
    regm1: float,
    regm2: float,
    p_values: np.ndarray,
    max_iter: int,
    gamma: float,
    step: float,
):
    tasks = [(exp, p, N, C, reg, regm1, regm2, max_iter, gamma, step) for p in p_values]
    best_cost, best_p = np.inf, p_values[0]
    with ProcessPoolExecutor(
        max_workers=min(multiprocessing.cpu_count(), 12)
    ) as executor:
        futures = [executor.submit(compute_cost, t) for t in tasks]
        for future in as_completed(futures):
            cost, p = future.result()
            if cost < best_cost:
                best_cost, best_p = cost, p
    return best_p


# average L1 error across experiments for given hyperparameters
def avg_l1_error(
    regm1: float,
    regm2: float,
    N: int,
    C: int,
    reg: float,
    max_iter: int,
    gamma: float,
    step: float,
    num_p: int,
    p_span: float,
):
    errors = []
    for exp, true_p in ground_truth.items():
        low = max(0, true_p - p_span)
        high = min(1, true_p + p_span)
        p_vals = np.linspace(low, high, num_p)
        p_hat = find_best_p(exp, N, C, reg, regm1, regm2, p_vals, max_iter, gamma, step)
        errors.append(abs(p_hat - true_p) + abs((1 - p_hat) - (1 - true_p)))
    return sum(errors) / len(errors)


# wrapper for hyperparam pairs
def process_pair(args):
    regm1, regm2, N, C, reg, max_iter, gamma, step, num_p, p_span = args
    return (regm1, regm2), avg_l1_error(
        regm1, regm2, N, C, reg, max_iter, gamma, step, num_p, p_span
    )


# plot heatmap
def plot_heatmap(data, xvals, yvals, N, C, reg):
    fig, ax = plt.subplots(figsize=(11, 11))
    im = ax.imshow(data, origin="lower", cmap="viridis_r")
    ax.set_title(f"N={N}, C={C}, reg={reg} — Avg L1/2 Error")
    ax.set_xlabel("regm2")
    ax.set_ylabel("regm1")
    ax.set_xticks(range(len(xvals)))
    ax.set_yticks(range(len(yvals)))
    ax.set_xticklabels([f"{v:.3f}" for v in xvals], rotation=45, ha="right")
    ax.set_yticklabels([f"{v:.3f}" for v in yvals])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j] / 2:.4f}", ha="center", va="center", fontsize=8)
    ax.figure.colorbar(im, ax=ax, label="L1 Error")
    ax.set_aspect("equal")
    plt.tight_layout()
    out = Path("plots/l1_error_heatmaps")
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "avg_l1_error_heatmap.png")
    plt.close()


# main
def main():
    N, C = 1000, 20
    reg, max_iter = 1.5, 1000
    step = 0.001
    gamma = 1.0 - (20.0 / max_iter)
    num_p, p_span = 24, 0.1
    regm1_vals = np.linspace(0.1, 10, 3)
    regm2_vals = np.linspace(0.1, 10, 3)
    grid = np.zeros((len(regm1_vals), len(regm2_vals)))
    tasks = [
        (r1, r2, N, C, reg, max_iter, gamma, step, num_p, p_span)
        for r1 in regm1_vals
        for r2 in regm2_vals
    ]
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for (r1, r2), val in tqdm(
            executor.map(process_pair, tasks),
            total=len(tasks),
            desc="Hyperparam search",
        ):
            i = np.where(regm1_vals == r1)[0][0]
            j = np.where(regm2_vals == r2)[0][0]
            grid[i, j] = val
    plot_heatmap(grid, regm2_vals, regm1_vals, N, C, reg)


if __name__ == "__main__":
    main()
