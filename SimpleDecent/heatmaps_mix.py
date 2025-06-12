from test_utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

# METHOD = "mirror_descent"
METHOD = "lbfgsb"

def construct_data(N, C):
    spectra, mix = load_data()

    mix_og = signif_features(mix, 2 * N)

    ratio = np.array([0.4, 1 - 0.6])
    mix_aprox = Spectrum.ScalarProduct(
        [signif_features(spectra[0], N), signif_features(spectra[1], N)], ratio
    )
    mix_aprox.normalize()

    a = np.array([p for _, p in mix_og.confs])
    b = np.array([p for _, p in mix_aprox.confs])

    v1 = np.array([v for v, _ in mix_og.confs])
    v2 = np.array([v for v, _ in mix_aprox.confs])

    M = multidiagonal_cost(v1, v2, C)
    # G0 = warmstart_sparse(a, b, C)

    c = reg_distribiution(2 * N, C)

    return a, b, c, M

# Parameters
regm1_values = np.linspace(200, 400, num=20)
regm2_values = np.linspace(100, 300, num=20)
reg = 1.5
N = 1000
C = 20
max_iter = 1000

# Warmstart
a, b, c, M = construct_data(N, C)
_G0 = warmstart_sparse(a, b, C)
_regm1 = (regm1_values[0] + regm1_values[-1]) / 2
_regm2 = (regm2_values[0] + regm2_values[-1]) / 2
sparse = UtilsSparse(a, b, c, _G0, M, reg, _regm1, _regm2)

print("Constructing warmstart...")
if METHOD == "lbfgsb":
    save_path = "heatmaps_lbfgsb"
    print("Using LBFGSB method.")
    _G0, _ = sparse.lbfgsb_unbalanced(numItermax=max_iter)
else:
    save_path = "heatmaps_md"
    print("Using Mirror Descent method.")
    _G0, _ = sparse.mirror_descent_unbalanced(numItermax=max_iter)
G0 = dia_matrix((_G0, sparse.offsets), shape=(sparse.n, sparse.m), dtype=np.float64)
print("Warmstart constructed.")

shape = (len(regm1_values), len(regm2_values))

# Initialize metric containers
def new_grid(): return np.zeros(shape)
metrics_s1 = {
    "Total Cost": new_grid(),
    "Transport Cost": new_grid(),
    "Regularization Term": new_grid(),
    "Marginal Penalty": new_grid(),
    "Marginal Penalty Normalized": new_grid()
}

os.makedirs(f"plots/{save_path}/heatmaps_mix", exist_ok=True)

args_list = [(i, j, regm1, regm2) for i, regm1 in enumerate(regm1_values)
                                    for j, regm2 in enumerate(regm2_values)]

results = []

def process_wrapper(arg_tuple):
    i, j, regm1, regm2 = arg_tuple
    sparse = UtilsSparse(a, b, c, G0, M, reg, regm1, regm2)

    if METHOD == "lbfgsb":
        G, _ = sparse.lbfgsb_unbalanced(numItermax=max_iter)
    else:
        G, _ = sparse.mirror_descent_unbalanced(numItermax=max_iter)

    tc = sparse.sparse_dot(G, sparse.offsets)
    reg_val = sparse.reg_kl_sparse(G, sparse.offsets)
    # marg1 = sparse.marg_tv_sparse(G, sparse.offsets)
    marg_rm1 = sparse.marg_tv_sparse_rm1(G, sparse.offsets)
    marg_rm2 = sparse.marg_tv_sparse_rm2(G, sparse.offsets)    

    return (i, j, {
        "Total Cost": tc + reg_val + marg_rm1 + marg_rm2,
        "Transport Cost": tc,
        "Regularization Term": reg_val,
        "Marginal Penalty": marg_rm1 + marg_rm2,
        "Marginal Penalty Normalized": marg_rm1 / regm1 + marg_rm2 / regm2
    })

if METHOD == "lbfgsb":
    for i, p in tqdm(args_list, desc="Processing"):
        results.append(process_wrapper((i, p)))
else:
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_wrapper, arg) for arg in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())

for i, j, metrics in results:
    for key in metrics_s1.keys():
        val1 = metrics[key]
        metrics_s1[key][i, j] = val1

def plot_heatmap_grid(metric_name, data, xvals, yvals):
    fig, ax = plt.subplots(figsize=(11, 11))

    im = ax.imshow(data, origin='lower', cmap='viridis')

    ax.set_title(f"2*N: {2*N}, C: {C}, reg: {reg} — 40/60 mix: {metric_name}")
    ax.set_xlabel("regm2")
    ax.set_ylabel("regm1")

    ax.set_xticks(np.arange(len(xvals)))
    ax.set_yticks(np.arange(len(yvals)))
    ax.set_xticklabels([f"{x:.1f}" for x in xvals])
    ax.set_yticklabels([f"{y:.1f}" for y in yvals])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}",
                    ha="center", va="center", color="white", fontsize=6, weight='bold')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(metric_name)

    ax.set_aspect('equal')
    plt.tight_layout()

    filepath = f"plots/{save_path}/heatmaps_mix/{metric_name.lower().replace(' ', '_')}.png"
    plt.savefig(filepath)
    plt.close()

for metric in metrics_s1.keys():
    plot_heatmap_grid(metric, metrics_s1[metric], regm2_values, regm1_values)
