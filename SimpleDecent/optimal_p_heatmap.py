import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from test_utils import (
    load_data,
    signif_features,
    Spectrum,
    multidiagonal_cost,
    reg_distribiution,
    warmstart_sparse,
    UtilsSparse,
)
from tqdm import tqdm

METHOD = "mirror_descent"


def construct_data(N, C, p):
    spectra, mix = load_data()
    mix_og = signif_features(mix, 2 * N)
    ratio = np.array([p, 1 - p])
    mix_aprox = Spectrum.ScalarProduct(
        [signif_features(spectra[0], N), signif_features(spectra[1], N)], ratio
    )
    mix_aprox.normalize()
    a = np.array([f for _, f in mix_og.confs])
    b = np.array([f for _, f in mix_aprox.confs])
    v1 = np.array([x for x, _ in mix_og.confs])
    v2 = np.array([x for x, _ in mix_aprox.confs])
    M = multidiagonal_cost(v1, v2, C)
    c = reg_distribiution(2 * N, C)
    return a, b, c, M


def process_wrapper(arg_tuple):
    p, N, C, reg, regm1, regm2, max_iter, gamma, step_size = arg_tuple
    a, b, c, M = construct_data(N, C, p)
    G0 = warmstart_sparse(a, b, C)
    sparse = UtilsSparse(a, b, c, G0, M, reg, regm1, regm2)
    if METHOD == "lbfgsb":
        G_data, log = sparse.lbfgsb_unbalanced(numItermax=max_iter)
    else:
        G_data, log = sparse.mirror_descent_unbalanced(
            numItermax=max_iter, gamma=gamma, step_size=step_size
        )
    return p, log["final_distance"]


def get_optimal_p(N, C, reg, regm1, regm2, p_values, max_iter, gamma, step_size):
    args = [(p, N, C, reg, regm1, regm2, max_iter, gamma, step_size) for p in p_values]
    results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_wrapper, a) for a in args]
        for f in as_completed(futures):
            results.append(f.result())
    if not results:
        return None, float("inf")
    ps, costs = zip(*results)
    idx = np.argmin(costs)
    return ps[idx], costs[idx]


def plot_heatmap_grid(name, data, xvals, yvals, N, C, reg):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, origin="lower", cmap="viridis_r")
    ax.set_title(f"2N={2 * N}  C={C}  reg={reg} — {name}")
    ax.set_xlabel("regm2")
    ax.set_ylabel("regm1")
    ax.set_xticks(np.arange(len(xvals)))
    ax.set_yticks(np.arange(len(yvals)))
    ax.set_xticklabels([f"{x:.2f}" for x in xvals], rotation=45)
    ax.set_yticklabels([f"{y:.2f}" for y in yvals])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label=name)
    plt.tight_layout()

    save_path = "plots/optimal_p_heatmap_lowregs"
    os.makedirs(save_path, exist_ok=True)
    filename = f"{save_path}/{name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Heatmap saved to {filename}")


def main():
    N = 1000
    C = 20
    reg = 1.5
    max_iter = 1000
    step_size = 0.001
    goal = 0.3865
    gamma = 1.0 - (20.0 / max_iter)
    p_values = np.linspace(0.35, 0.50, 12)
    regm1_vals = np.logspace(np.log2(1), np.log2(100), num=5, base=2)
    regm2_vals = np.logspace(np.log2(1), np.log2(100), num=5, base=2)
    heatmap = np.zeros((len(regm1_vals), len(regm2_vals)))

    for i, r1 in enumerate(tqdm(regm1_vals, desc="regm1")):
        for j, r2 in enumerate(regm2_vals):
            optimal_p, _ = get_optimal_p(
                N, C, reg, r1, r2, p_values, max_iter, gamma, step_size
            )
            heatmap[i, j] = abs(optimal_p - goal)

    plot_heatmap_grid(
        "Optimal p difference", heatmap, regm2_vals, regm1_vals, N, C, reg
    )


if __name__ == "__main__":
    main()
