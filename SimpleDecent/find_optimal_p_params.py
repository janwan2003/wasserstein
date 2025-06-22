import numpy as np
from tqdm import tqdm
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
    dia_matrix,
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


def get_optimal_p(N, C, reg, regm1, regm2, p_values, max_iter, G0, gamma):
    """
    For a given set of hyperparameters, this function finds the optimal p value
    from a list of candidates by running optimizations in parallel.
    """
    args_list = [(p, N, C, reg, regm1, regm2, max_iter, G0, gamma) for p in p_values]
    results = []

    desc = f"C={C}, reg={reg:.1f}, regm1={regm1:.0f}, regm2={regm2:.0f}"
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_wrapper, arg) for arg in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            results.append(future.result())

    if not results:
        return None, float("inf")

    # Find the value of p that minimizes the total cost
    costs = [res[1]["Total Cost"] for res in results]
    min_cost_index = np.argmin(costs)
    optimal_p = results[min_cost_index][0]
    min_cost = costs[min_cost_index]

    return optimal_p, min_cost


def process_wrapper(arg_tuple):
    """
    A wrapper function for parallel execution. It computes the cost for a given p.
    """
    p, N, C, reg, regm1, regm2, max_iter, G0, gamma = arg_tuple
    a, b, c, M = construct_data(N, C, p)
    sparse = UtilsSparse(a, b, c, G0, M, reg, regm1, regm2)

    if METHOD == "lbfgsb":
        G, _ = sparse.lbfgsb_unbalanced(numItermax=max_iter)
    else:
        G, _ = sparse.mirror_descent_unbalanced(numItermax=max_iter, gamma=gamma)

    tc = sparse.sparse_dot(G, sparse.offsets)
    reg_val = sparse.reg_kl_sparse(G, sparse.offsets)
    marg_rm1 = sparse.marg_tv_sparse_rm1(G, sparse.offsets)
    marg_rm2 = sparse.marg_tv_sparse_rm2(G, sparse.offsets)

    metrics = {
        "Total Cost": tc + reg_val + marg_rm1 + marg_rm2,
        "Transport Cost": tc,
        "Regularization Term": reg_val,
        "Marginal Penalty": marg_rm1 + marg_rm2,
        "Marginal Penalty Normalized": marg_rm1 / regm1 + marg_rm2 / regm2,
    }
    return p, metrics


def main():
    """
    Main function to run the hyperparameter search experiment.
    """
    # Fixed parameters for the experiment
    N = 2000
    max_iter = 200
    step_size = 0.0002
    gamma = 1.0 - 1.0 / max_iter

    p_values = np.linspace(0.3, 0.7, 32)

    # --- Define the hyperparameter search space ---
    C_values = [20]
    reg_values = [0.01]
    regm1_values = np.linspace(200, 300, num=5)
    regm2_values = np.linspace(100, 200, num=5)

    found_params = []

    # --- Start the search ---
    print("\nStarting hyperparameter search...")
    for C in C_values:
        # --- Construct a good warmstart transport plan for the current C ---
        print(f"Constructing warmstart transport plan for C={C}...")
        a_warm, b_warm, c_warm, M_warm = construct_data(N, C, 0.5)
        _G0_warm = warmstart_sparse(a_warm, b_warm, C)

        # Use mirror descent to get a robust warmstart plan
        sparse_warm = UtilsSparse(
            a_warm, b_warm, c_warm, _G0_warm, M_warm, reg=1.5, reg_m1=230, reg_m2=115
        )
        _G0_data, _ = sparse_warm.mirror_descent_unbalanced(
            numItermax=max_iter, gamma=gamma, step_size=step_size
        )
        G0 = dia_matrix(
            (_G0_data, sparse_warm.offsets),
            shape=(sparse_warm.n, sparse_warm.m),
            dtype=np.float64,
        )
        print("Warmstart constructed.")

        for reg in reg_values:
            for regm1 in regm1_values:
                for regm2 in regm2_values:
                    optimal_p, min_cost = get_optimal_p(
                        N, C, reg, regm1, regm2, p_values, max_iter, G0, gamma
                    )

                    print(f"  -> Optimal p = {optimal_p:.4f} (cost: {min_cost:.4f})")

                    if optimal_p is not None and 0.39 <= optimal_p <= 0.41:
                        print("  Found a good set of parameters!")
                        params = {
                            "N": N,
                            "C": C,
                            "reg": reg,
                            "regm1": regm1,
                            "regm2": regm2,
                            "optimal_p": optimal_p,
                            "min_cost": min_cost,
                        }
                        found_params.append(params)

    # --- Print the final results ---
    print("\n--- Experiment Finished ---")
    if found_params:
        print("Found the following parameter sets where optimal p is in [0.39, 0.41]:")
        for params in found_params:
            print(params)
    else:
        print("Could not find any suitable parameters in the defined search space.")


if __name__ == "__main__":
    main()
