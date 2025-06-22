from test_utils import (
    load_data,
    signif_features,
    Spectrum,
    multidiagonal_cost,
    warmstart_sparse,
    reg_distribiution,
    UtilsSparse,
    dia_matrix,
)
import matplotlib.pyplot as plt
import numpy as np
import os

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
    G0 = warmstart_sparse(a, b, C)

    c = reg_distribiution(2 * N, C)

    return v1, v2, a, b, c, M, G0


# Parameters
regm1 = 230
regm2 = 115
reg = 1.5
N = 1000
C = 20
max_iter = 3000

# Data
print("Constructing transport plan...")

v1, v2, a, b, c, M, _G0 = construct_data(N, C)
sparse = UtilsSparse(a, b, c, _G0, M, reg, regm1, regm2)
_G0, _ = sparse.mirror_descent_unbalanced(numItermax=max_iter)
G = dia_matrix((_G0, sparse.offsets), shape=(sparse.n, sparse.m), dtype=np.float64)

if METHOD == "lbfgsb":
    save_path = "marginals_lbfgsb"
    print("Using LBFGSB method.")
    # okazuje sie ze nasz warmstart slabo dziala dla lbfgsb wiec trzeba sie poluzyc tym od md
    sparse = UtilsSparse(a, b, c, G, M, reg, regm1, regm2)
    _G0, _ = sparse.lbfgsb_unbalanced(numItermax=max_iter)
    G = dia_matrix((_G0, sparse.offsets), shape=(sparse.n, sparse.m), dtype=np.float64)
else:
    save_path = "marginals_md"
    print("Using Mirror Descent method.")

print("Transport plan constructed.")

G1 = np.array(G.sum(axis=1)).flatten()  # G * 1
G2 = np.array(G.sum(axis=0)).flatten()  # 1^T * G

assert len(a) == len(G1) == len(v1), "Length mismatch in a, G1, and v1"
assert len(b) == len(G2) == len(v2), "Length mismatch in b, G2, and v2"

# print("Sum of G:", np.sum(G))

os.makedirs(f"plots/{save_path}", exist_ok=True)

# Combined plot with 6 subplots: 3 rows × 2 columns
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Top row
axs[0, 0].scatter(v1, a, color="blue", s=10)
axs[0, 0].set_title("Original source (a)")
axs[0, 0].set_xlabel("v1")
axs[0, 0].set_ylabel("Mass")

axs[0, 1].scatter(v2, b, color="green", s=10)
axs[0, 1].set_title("Original target (b)")
axs[0, 1].set_xlabel("v2")
axs[0, 1].set_ylabel("Mass")

# Middle row
axs[1, 0].scatter(v1, G1, color="orange", s=10)
axs[1, 0].set_title("Transported source (G * 1)")
axs[1, 0].set_xlabel("v1")
axs[1, 0].set_ylabel("Mass")

axs[1, 1].scatter(v2, G2, color="red", s=10)
axs[1, 1].set_title("Transported target (1ᵀ * G)")
axs[1, 1].set_xlabel("v2")
axs[1, 1].set_ylabel("Mass")

# Bottom row: Deltas
delta_source = a - G1
delta_target = b - G2

axs[2, 0].scatter(v1, delta_source, color="purple", s=10)
axs[2, 0].axhline(0, color="black", linestyle="--", linewidth=1)
axs[2, 0].set_title("Delta source (a - G * 1)")
axs[2, 0].set_xlabel("v1")
axs[2, 0].set_ylabel("Delta Mass")

axs[2, 1].scatter(v2, delta_target, color="brown", s=10)
axs[2, 1].axhline(0, color="black", linestyle="--", linewidth=1)
axs[2, 1].set_title("Delta target (b - 1ᵀ * G)")
axs[2, 1].set_xlabel("v2")
axs[2, 1].set_ylabel("Delta Mass")

# title
plt.suptitle(
    f"(2N={2 * N}, C={C}, reg={reg}, regm1={regm1}, regm2={regm2}, maxiter={max_iter})",
    fontsize=16,
)

plt.tight_layout()

output_path = f"plots/{save_path}/marginals_comparison_with_deltas.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot with deltas saved to {output_path}")
