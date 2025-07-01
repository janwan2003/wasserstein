import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from test_utils import (
    load_data,
    signif_features,
    Spectrum,
    multidiagonal_cost,
    reg_distribiution,
    warmstart_sparse,
    UtilsSparse,
)

# The method to use for the optimization. Can be "mirror_descent" or "lbfgsb".
METHOD = "mirror_descent"


def construct_data(N, C, p):
    """
    Constructs the data needed for the optimization problem.
    This function is based on the implementation in p_curves.py.
    """
    spectra, mix = load_data()

    mix_og = signif_features(mix, 2 * N)

    ratio = np.array([p, 1 - p])
    mix_aprox = Spectrum.ScalarProduct(
        [signif_features(spectra[0], N), signif_features(spectra[1], N)], ratio
    )
    mix_aprox.normalize()

    a = np.array([p for _, p in mix_og.confs])
    b = np.array([p for _, p in mix_aprox.confs])

    v1 = np.array([v for v, _ in mix_og.confs])
    v2 = np.array([v for v, _ in mix_aprox.confs])

    M = multidiagonal_cost(v1, v2, C)
    c = reg_distribiution(2 * N, C)
    return a, b, c, M


def get_optimal_p(N, C, reg, regm1, regm2, p_values, max_iter, gamma, step_size):
    """
    For a given set of hyperparameters, this function finds the optimal p value
    from a list of candidates by running optimizations in parallel.
    """
    args_list = [
        (p, N, C, reg, regm1, regm2, max_iter, gamma, step_size) for p in p_values
    ]
    results = []

    # Disable the progress bar for the inner loop to avoid clutter
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_wrapper, arg) for arg in args_list]
        # Using as_completed to get results as they are ready
        for future in as_completed(futures):
            results.append(future.result())

    if not results:
        return None, float("inf")

    # Find the value of p that minimizes the final distance
    costs = [res[1]["final distance"] for res in results]
    min_cost_index = np.argmin(costs)
    optimal_p = results[min_cost_index][0]
    min_cost = costs[min_cost_index]

    return optimal_p, min_cost


def process_wrapper(arg_tuple):
    """
    A wrapper function for parallel execution. It computes the cost for a given p.
    This version creates its own warmstart to prevent shape mismatches.
    """
    p, N, C, reg, regm1, regm2, max_iter, gamma, step_size = arg_tuple
    a, b, c, M = construct_data(N, C, p)

    # Create a simple warmstart for each process to avoid shape mismatch errors
    G0 = warmstart_sparse(a, b, C)

    sparse = UtilsSparse(a, b, c, G0, M, reg, regm1, regm2)

    if METHOD == "lbfgsb":
        G, _ = sparse.lbfgsb_unbalanced(numItermax=max_iter)
    else:
        G, _ = sparse.mirror_descent_unbalanced(
            numItermax=max_iter,
            gamma=gamma,
            step_size=step_size,
            stopThr=1e-6,
            patience=100,
        )

    transport_cost = sparse.sparse_dot(G, sparse.offsets)
    marginal_penalty = sparse.marg_tv_sparse(G, sparse.offsets)
    final_distance = transport_cost + marginal_penalty / (regm1 + regm2)
    metrics = {
        "final distance": final_distance,
    }
    return p, metrics


def plot_heatmap_grid(metric_name, data, xvals, yvals, N, C, reg):
    """
    Plots and saves a heatmap of the results.
    """
    fig, ax = plt.subplots(figsize=(11, 11))

    im = ax.imshow(data, origin="lower", cmap="viridis_r")  # Reversed viridis

    ax.set_title(f"2*N: {2 * N}, C: {C}, reg: {reg} — {metric_name}")
    ax.set_xlabel("regm2")
    ax.set_ylabel("regm1")

    ax.set_xticks(np.arange(len(xvals)))
    ax.set_yticks(np.arange(len(yvals)))
    ax.set_xticklabels([f"{x:.1f}" for x in xvals])
    ax.set_yticklabels([f"{y:.1f}" for y in yvals])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                f"{data[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                weight="bold",
            )

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(metric_name)

    ax.set_aspect("equal")
    plt.tight_layout()

    save_path = "plots/optimal_p_heatmap_lowregs"
    os.makedirs(save_path, exist_ok=True)
    filepath = f"{save_path}/{metric_name.lower().replace(' ', '_')}.png"
    plt.savefig(filepath)
    plt.close()
    print(f"Heatmap saved to {filepath}")


def main():
    """
    Main function to run the hyperparameter search and generate a heatmap.
    """
    # Fixed parameters for the experiment
    N = 1000
    C = 20
    reg = 1.5
    max_iter = 1000
    step_size = 0.001
    gamma = 1.0 - (20.0 / max_iter)
    p_values = np.linspace(0.3, 0.5, 24)

    # Define the hyperparameter search space for the heatmap
    regm1_values = np.logspace(np.log2(0.1), np.log2(10), num=10, base=2)
    regm2_values = np.logspace(np.log2(0.1), np.log2(10), num=10, base=2)

    heatmap_data = np.zeros((len(regm1_values), len(regm2_values)))

    print("Starting hyperparameter search for optimal p...")
    for i, regm1 in enumerate(tqdm(regm1_values, desc="regm1 progress")):
        for j, regm2 in enumerate(regm2_values):
            optimal_p, _ = get_optimal_p(
                N, C, reg, regm1, regm2, p_values, max_iter, gamma, step_size
            )

            if optimal_p is not None:
                heatmap_data[i, j] = abs(optimal_p - 3865)
            else:
                heatmap_data[i, j] = np.nan

    # Plot the results
    plot_heatmap_grid(
        "Distance of Optimal p from 0.3865",
        heatmap_data,
        regm2_values,
        regm1_values,
        N,
        C,
        reg,
    )


if __name__ == "__main__":
    main()
