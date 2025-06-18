from test_utils import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

METHOD = "mirror_descent"
# METHOD = "lbfgsb"

def construct_data(N, C, p):
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

# Parameters
regm1 = 230
regm2 = 115
reg = 1.5
N = 1000
C = 20
max_iter = 1000
p_values = np.linspace(0.3, 0.7, 32)

# Construct warmstart
print("Constructing warmstart...")

a, b, c, M = construct_data(N, C, 0.5)
_G0 = warmstart_sparse(a, b, C)
sparse = UtilsSparse(a, b, c, _G0, M, reg, regm1, regm2)
_G0, _ = sparse.mirror_descent_unbalanced(numItermax=max_iter)
G0 = dia_matrix((_G0, sparse.offsets), shape=(sparse.n, sparse.m), dtype=np.float64)

if METHOD == "lbfgsb":
    save_path = "p_curves_lbfgsb"
    print("Using LBFGSB method.")
    # okazuje sie ze nasz warmstart slabo dziala dla lbfgsb wiec trzeba sie poluzyc tym od md
    sparse = UtilsSparse(a, b, c, G0, M, reg, regm1, regm2)
    _G0, _ = sparse.lbfgsb_unbalanced(numItermax=max_iter)
    G0 = dia_matrix((_G0, sparse.offsets), shape=(sparse.n, sparse.m), dtype=np.float64)
else:
    save_path = "p_curves_md"
    print("Using Mirror Descent method.")

print("Warmstart constructed.")

shape = (len(p_values),)

def new_grid(): return np.zeros(shape)

metrics_s1 = {
    "Total Cost": new_grid(),
    "Transport Cost": new_grid(),
    "Regularization Term": new_grid(),
    "Marginal Penalty": new_grid(),
    "Marginal Penalty Normalized": new_grid()
}

os.makedirs(f"plots/jit/{save_path}", exist_ok=True)

args_list = [(i, p) for i, p in enumerate(p_values)]

results = []

def process_wrapper(arg_tuple):
    i, p = arg_tuple
    a, b, c, M = construct_data(N, C, p)
    sparse = UtilsSparse(a, b, c, G0, M, reg, regm1, regm2)

    if METHOD == "lbfgsb":
        G, _ = sparse.lbfgsb_unbalanced(numItermax=max_iter)
    else:
        G, _ = sparse.mirror_descent_unbalanced(numItermax=max_iter)

    # print(f"Sum of G for p={p}: {np.sum(G)}")

    tc = sparse.sparse_dot(G, sparse.offsets)
    reg_val = sparse.reg_kl_sparse(G, sparse.offsets)
    marg_rm1 = sparse.marg_tv_sparse_rm1(G, sparse.offsets)
    marg_rm2 = sparse.marg_tv_sparse_rm2(G, sparse.offsets)    

    return (i, {
        "Total Cost": tc + reg_val + marg_rm1 + marg_rm2,
        "Transport Cost": tc,
        "Regularization Term": reg_val,
        "Marginal Penalty": marg_rm1 + marg_rm2,
        "Marginal Penalty Normalized": marg_rm1 / regm1 + marg_rm2 / regm2
    })

# if METHOD == "lbfgsb":
for i, p in tqdm(args_list, desc="Processing"):
    results.append(process_wrapper((i, p)))
# else:
#     with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
#         futures = [executor.submit(process_wrapper, arg) for arg in args_list]
#         for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
#             results.append(future.result())

# Fill metrics
for i, metrics in results:
    for key in metrics_s1.keys():
        metrics_s1[key][i] = metrics[key]

# Plot curves
def plot_metric_curve(metric_name, y_vals, x_vals):
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o')
    plt.title(f"Metric: {metric_name} vs p (2N={2*N}, C={C}, reg={reg}, regm1={regm1}, regm2={regm2})")
    plt.xlabel("p")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.savefig(f"plots/jit/{save_path}/{metric_name.lower().replace(' ', '_')}.png")
    plt.close()

for metric in metrics_s1:
    plot_metric_curve(metric, metrics_s1[metric], p_values)