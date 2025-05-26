"""
Utilities for sparse Wasserstein distance computation and optimization.
This module provides implementations of sparse matrix operations, optimization routines,
and testing functions for efficient Wasserstein distance computation.
"""

import numpy as np
import os
import sys
import time
from scipy.sparse import dia_matrix
from scipy.optimize import minimize, Bounds
from typing import List
from ot.backend import get_backend
import ot

sys.path.append(os.path.abspath(".."))
from wasserstein import Spectrum, NMRSpectrum

# ------------------------------------------------------------------------------
# Sparse Matrix Operations
# ------------------------------------------------------------------------------


def multidiagonal_cost(v1, v2, C):
    """
    Construct a multidiagonal sparse matrix where M[i,j] = abs(v1[i] - v2[j]) if abs(i-j) < C.

    Parameters:
        v1, v2 (array-like): Input vectors (must be same length)
        C (int): Bandwidth parameter controlling how many diagonals are non-zero

    Returns:
        dia_matrix: The constructed sparse matrix in DIAgonal format
    """
    n = len(v1)
    assert len(v2) == n, "v1 and v2 must have the same length"

    offsets = np.arange(-C + 1, C)  # Diagonals from -C+1 to C-1

    # Preallocate the data array
    data = np.zeros((len(offsets), n))

    # For each diagonal offset
    for i, k in enumerate(offsets):
        # Calculate valid indices for this diagonal
        i_min = max(0, -k)
        i_max = min(n, n - k)
        diag_length = i_max - i_min

        # Calculate values for this diagonal
        diag_values = np.abs(v1[i_min:i_max] - v2[i_min + k : i_max + k])

        # Place values in the correct position with proper padding
        if k < 0:
            # Lower diagonal: place values at the beginning, pad at the end
            data[i, :diag_length] = diag_values
        else:
            # Upper diagonal: place values at the end, pad at the beginning
            data[i, n - diag_length :] = diag_values

    return dia_matrix((data, offsets), shape=(n, n))


def reg_distribiution(N, C):
    """
    Construct a multidiagonal sparse matrix of regularization coefficients.

    Parameters:
        N (int): Size of the matrix
        C (int): Bandwidth parameter controlling how many diagonals are non-zero

    Returns:
        dia_matrix: The constructed sparse matrix in DIAgonal format
    """
    offsets = np.arange(-C + 1, C)  # Diagonals from -C+1 to C-1

    # Preallocate the data array
    data = np.zeros((len(offsets), N))

    # Calculate the value once outside the loop
    value = 1 / (N * (2 * C - 1) - C * (C - 1))

    # For each diagonal offset
    for i, k in enumerate(offsets):
        diag_length = N - np.abs(k)

        # Place values in the correct position with proper padding
        if k < 0:
            # Lower diagonal: place values at the beginning, pad at the end
            data[i, :diag_length] = value
        else:
            # Upper diagonal: place values at the end, pad at the beginning
            data[i, N - diag_length :] = value

    return dia_matrix((data, offsets), shape=(N, N))


def warmstart_sparse(p1, p2, C):
    """
    Create a warm-start sparse multidiagonal transport plan where:
    - Diagonal (i,i) gets min(p1[i], p2[i])
    - Remaining mass is distributed to adjacent cells (i±k) within bandwidth C

    Args:
        p1 (np.array): Source distribution (size n)
        p2 (np.array): Target distribution (size n)
        C (int): Bandwidth (max allowed offset from diagonal)

    Returns:
        dia_matrix: Sparse multidiagonal matrix of shape (n,n)
    """
    n = len(p1)
    assert len(p2) == n, "p1 and p2 must have same length"

    # Initialize diagonals with proper padding
    offsets = list(range(-C + 1, C))

    # Create storage for diagonals with proper padding
    diagonals = {}
    for offset in offsets:
        diagonals[offset] = np.zeros(n)

    for i in range(n):
        remaining_mass = p1[i]

        # Step 1: Assign to diagonal (i,i)
        assign = min(remaining_mass, p2[i])
        diagonals[0][i] = assign
        remaining_mass -= assign

        # Step 2: Distribute remaining mass to adjacent cells
        radius = 1
        while remaining_mass > 1e-10 and radius < C:
            # Right neighbor (i, i+radius)
            if i + radius < n:
                idx = i + radius
                assign = min(remaining_mass, p2[idx])
                diagonals[radius][idx] += assign
                remaining_mass -= assign

            # Left neighbor (i, i-radius)
            if i - radius >= 0 and remaining_mass > 1e-10:
                idx = i - radius
                assign = min(remaining_mass, p2[idx])
                diagonals[-radius][idx] += assign
                remaining_mass -= assign

            radius += 1

    diagonals = [diagonals[offset] for offset in offsets]
    sum = np.sum(diagonals)
    diagonals /= sum

    return dia_matrix((np.array(diagonals), offsets), shape=(n, n))


def flatten_multidiagonal(matrix_data, offsets):
    """
    Flatten a multidiagonal matrix data into a 1D array.

    Parameters:
        matrix_data: Array of diagonal data
        offsets: Array of diagonal offsets

    Returns:
        np.array: Flattened 1D array
    """
    flat_vector = []

    for diag, offset in zip(matrix_data, offsets):
        if offset < 0:
            flat_vector.extend(diag[:offset])
        else:
            flat_vector.extend(diag[offset:])

    return np.array(flat_vector)


def reconstruct_multidiagonal(flat_vector, offsets, N):
    """
    Reconstruct the diagonal data from the flat vector.

    Parameters:
        flat_vector: Flat array of meaningful diagonal values
        offsets: The offsets array
        N: Size of the matrix (N x N)

    Returns:
        List of arrays containing the diagonal data
    """
    M_data = []
    ptr = 0

    for offset in offsets:
        diag_length = N - abs(offset)
        diag_values = flat_vector[ptr : ptr + diag_length]

        # Reconstruct the diagonal with proper padding
        if offset < 0:
            padded_diag = np.pad(diag_values, (0, abs(offset)), mode="constant")
        else:
            padded_diag = np.pad(diag_values, (abs(offset), 0), mode="constant")

        M_data.append(padded_diag)
        ptr += diag_length

    return np.array(M_data)


# ------------------------------------------------------------------------------
# Data Loading and Processing
# ------------------------------------------------------------------------------


def load_data():
    """
    Load NMR spectral data from files.

    Returns:
        Tuple[List[NMRSpectrum], NMRSpectrum]: Component spectra and mixture spectrum
    """
    components_names = ["Pinene", "Benzyl benzoate"]

    protons_list = [16, 12]

    filename = "preprocessed_mix.csv"
    mix = np.loadtxt(filename, delimiter=",")

    how_many_components = len(components_names)
    names = ["comp" + str(i) for i in range(how_many_components)]

    files_with_components = ["preprocessed_comp0.csv", "preprocessed_comp1.csv"]
    spectra = []
    for i in range(how_many_components):
        filename = files_with_components[i]
        spectra.append(np.loadtxt(filename, delimiter=","))

    spectra2: List[NMRSpectrum] = []
    names = []
    for i in range(len(spectra)):
        spectra2.append(
            NMRSpectrum(
                confs=list(zip(spectra[i][:, 0], spectra[i][:, 1])),
                protons=protons_list[i],
            )
        )
        names.append("comp" + str(i))

    spectra = spectra2
    del spectra2
    mix = NMRSpectrum(confs=list(zip(mix[:, 0], mix[:, 1])))
    mix.trim_negative_intensities()
    mix.normalize()
    for sp in spectra:
        sp.trim_negative_intensities()
        sp.normalize()

    return spectra, mix


def signif_features(spectrum, n_features):
    """
    Extract the most significant features from a spectrum.

    Parameters:
        spectrum (NMRSpectrum): Input spectrum
        n_features (int): Number of features to extract

    Returns:
        NMRSpectrum: Spectrum with only the significant features
    """
    spectrum_confs = sorted(spectrum.confs, key=lambda x: x[1], reverse=True)[
        :n_features
    ]
    spectrum_signif = NMRSpectrum(confs=spectrum_confs, protons=spectrum.protons)
    spectrum_signif.normalize()
    return spectrum_signif


# ------------------------------------------------------------------------------
# Optimization Classes
# ------------------------------------------------------------------------------


class UtilsSparse:
    """
    Utilities for sparse matrix optimization of optimal transport.

    This class implements optimization routines for unbalanced optimal transport
    using sparse multidiagonal matrices.
    """

    def __init__(self, a, b, c, G0_sparse, M_sparse, reg, reg_m1, reg_m2, damp):
        """
        Initialize the sparse optimization utilities.

        Parameters:
            a (np.array): Source distribution
            b (np.array): Target distribution
            c (dia_matrix): Reference distribution for KL regularization
            G0_sparse (dia_matrix): Initial transport plan
            M_sparse (dia_matrix): Cost matrix
            reg (float): KL regularization strength
            reg_m1 (float): Regularization for source marginal
            reg_m2 (float): Regularization for target marginal
            damp (float): Damping parameter
        """
        self.a = a
        self.b = b
        self.c_data = c.data
        self.m, self.n = M_sparse.shape

        assert self.m == self.n, "Only square multidiagonal matrices supported"
        assert len(a) == len(b) == self.m, "Marginals must match matrix dimensions"

        self.G0_sparse = G0_sparse
        self.offsets = M_sparse.offsets
        self.data = M_sparse.data

        self.reg = reg
        self.reg_m1 = reg_m1
        self.reg_m2 = reg_m2
        self.damp = damp

        self.data /= damp
        self.reg /= damp
        self.reg_m1 /= damp
        self.reg_m2 /= damp

    def sparse_dot(self, G_data, G_offsets):
        """
        Efficient dot product between two multidiagonal matrices.

        Parameters:
            G_data: Diagonal data of the second matrix
            G_offsets: Offsets of the second matrix

        Returns:
            float: Dot product value
        """
        total = 0.0
        for i, offset1 in enumerate(self.offsets):
            for j, offset2 in enumerate(G_offsets):
                if offset1 == offset2:
                    min_len = min(len(self.data[i]), len(G_data[j]))
                    total += np.sum(self.data[i][:min_len] * G_data[j][:min_len])
        return total

    def sparse_row_sum(self, G_data, G_offsets):
        """Row sums for multidiagonal matrix (square matrix version)"""
        row_sums = np.zeros(self.m)
        for offset, diag in zip(G_offsets, G_data):
            if offset == 0:  # Main diagonal
                row_sums += diag[: self.m]
            elif offset < 0:  # Lower diagonal
                k = -offset
                row_sums[k:] += diag[: self.m - k]
            else:  # Upper diagonal
                row_sums[:-offset] += diag[offset : offset + self.m - offset]
        return row_sums

    def sparse_col_sum(self, G_data, _G_offsets):
        """Column sums for multidiagonal matrix (square matrix version)"""
        col_sums = np.zeros(self.m)
        for diag in G_data:
            col_sums += diag
        return col_sums

    def reg_kl_sparse(self, G_data, G_offsets):
        G_flat = flatten_multidiagonal(G_data, G_offsets)
        C_flat = flatten_multidiagonal(self.c_data, G_offsets)

        return np.sum(G_flat * np.log(G_flat / C_flat + 1e-16)) + np.sum(
            C_flat - G_flat
        )

    def grad_kl_sparse(self, G_data, G_offsets):
        """Gradient of KL divergence for sparse matrix"""
        G_flat = flatten_multidiagonal(G_data, G_offsets)
        C_flat = flatten_multidiagonal(self.c_data, G_offsets)

        grad_flat = np.log(G_flat / C_flat + 1e-16)
        return reconstruct_multidiagonal(grad_flat, G_offsets, self.m) * self.reg

    def marg_tv_sparse(self, G_data, G_offsets):
        """TV marginal penalty for sparse matrix"""
        row_sums = self.sparse_row_sum(G_data, G_offsets)
        col_sums = self.sparse_col_sum(G_data, G_offsets)
        return self.reg_m1 * np.sum(np.abs(row_sums - self.a)) + self.reg_m2 * np.sum(
            np.abs(col_sums - self.b)
        )

    def grad_marg_tv_sparse(self, G_data, G_offsets):
        """Gradient of TV marginal penalty"""
        row_sums = self.sparse_row_sum(G_data, G_offsets)
        col_sums = self.sparse_col_sum(G_data, G_offsets)

        # Compute sign terms for rows and columns
        row_signs = self.reg_m1 * np.sign(row_sums - self.a)
        col_signs = self.reg_m2 * np.sign(col_sums - self.b)

        grad_data = []
        for offset in G_offsets:
            if offset == 0:  # Main diagonal
                grad_diag = row_signs + col_signs
            elif offset < 0:  # Lower diagonal
                k = -offset
                grad_diag = row_signs[k:] + col_signs[: self.m - k]
                grad_diag = np.pad(grad_diag, (0, k), "constant")
            else:  # Upper diagonal
                grad_diag = row_signs[: self.m - offset] + col_signs[offset:]
                grad_diag = np.pad(grad_diag, (offset, 0), "constant")
            grad_data.append(grad_diag)

        return np.array(grad_data)

    def func_sparse(self, G_flat):
        """Combined loss function and gradient for sparse optimization"""
        # Reconstruct sparse matrix from flattened representation
        G_data = reconstruct_multidiagonal(G_flat, self.offsets, self.m)

        # Compute loss
        transport_cost = self.sparse_dot(G_data, self.offsets)
        marginal_penalty = self.marg_tv_sparse(G_data, self.offsets)
        val = transport_cost + marginal_penalty
        if self.reg > 0:
            val += self.reg_kl_sparse(G_data, self.offsets)

        # Compute gradient
        grad = self.data + self.grad_marg_tv_sparse(G_data, self.offsets)

        # Combine gradients
        grad_flat = np.zeros_like(G_flat)

        if self.reg > 0:
            grad += self.grad_kl_sparse(G_data, self.offsets)

        grad_flat = flatten_multidiagonal(grad, self.offsets)

        return val, grad_flat

    def lbfgsb_unbalanced(self, numItermax=1000, stopThr=1e-15):
        _func = self.func_sparse

        # panic for now
        # if self.c is None:
        #     raise ValueError("Reference distribution 'c' must be provided for unbalanced OT")
        if self.G0_sparse is None:
            raise ValueError("Initial transport plan 'G0' must be provided")

        res = minimize(
            _func,
            flatten_multidiagonal(self.G0_sparse.data, self.G0_sparse.offsets),
            method="L-BFGS-B",
            jac=True,
            bounds=Bounds(0, np.inf),
            tol=stopThr,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": numItermax},
        )

        G = reconstruct_multidiagonal(res.x, self.G0_sparse.offsets, self.m) * self.damp

        log = {
            "total_cost": res.fun * self.damp,
            "cost": self.sparse_dot(G, self.offsets),
            "res": res,
        }
        return G, log


class UtilsDense:
    """
    Utilities for dense matrix optimization of optimal transport.

    This class implements optimization routines for unbalanced optimal transport
    using dense matrices (for comparison and validation).
    """

    def __init__(self, a, b, c, G0, M, reg, reg_m1, reg_m2):
        """
        Initialize the dense optimization utilities.

        Parameters:
            a (np.array): Source distribution
            b (np.array): Target distribution
            c (dia_matrix): Reference distribution for KL regularization
            G0 (np.array): Initial transport plan
            M (np.array): Cost matrix
            reg (float): KL regularization strength
            reg_m1 (float): Regularization for source marginal
            reg_m2 (float): Regularization for target marginal
        """
        self.a = a
        self.b = b
        self.c = c.toarray()
        self.m, self.n = M.shape
        assert self.m == self.n, "Only square multidiagonal matrices supported"
        assert len(a) == len(b) == self.m, "Marginals must match matrix dimensions"

        self.G0 = G0
        self.M = M

        self.reg = reg
        self.reg_m1 = reg_m1
        self.reg_m2 = reg_m2

    def reg_kl(self, G):
        nx_numpy = get_backend(self.M, self.a, self.b)
        return nx_numpy.kl_div(G, self.c, mass=True)

    def grad_kl(self, G):
        return np.log(G / self.c + 1e-16)

    def marg_tv(self, G):
        return self.reg_m1 * np.sum(np.abs(G.sum(1) - self.a)) + self.reg_m2 * np.sum(
            np.abs(G.sum(0) - self.b)
        )

    def grad_marg_tv(self, G):
        return self.reg_m1 * np.outer(
            np.sign(G.sum(1) - self.a), np.ones(self.n)
        ) + self.reg_m2 * np.outer(np.ones(self.m), np.sign(G.sum(0) - self.b))

    def func(self, G):
        G = G.reshape((self.m, self.n))

        # compute loss
        val = np.sum(G * self.M) + self.marg_tv(G)
        if self.reg > 0:
            val = val + self.reg * self.reg_kl(G)
        # compute gradient
        grad = self.M + self.grad_marg_tv(G)
        if self.reg > 0:
            grad = grad + self.reg * self.grad_kl(G)

        return val, grad

    def lbfgsb_unbalanced(self, numItermax=1000, stopThr=1e-15):
        _func = self.func

        G0 = self.a[:, None] * self.b[None, :] if self.G0 is None else self.G0

        res = minimize(
            _func,
            G0.ravel(),
            method="L-BFGS-B",
            jac=True,
            bounds=Bounds(0, np.inf),
            tol=stopThr,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": numItermax},
        )

        G = res.x.reshape(self.M.shape)

        log = {"total_cost": res.fun, "cost": np.sum(G * self.M), "res": res}
        return G, log


# ------------------------------------------------------------------------------
# Testing and Experiment Functions
# ------------------------------------------------------------------------------


def test_sparse(N, C, p, reg, reg_m1, reg_m2, damp=1, max_iter=5000, debug=False):
    """
    Test the sparse optimization algorithm on NMR spectra.

    Parameters:
        N (int): Number of features to use for each component
        C (int): Bandwidth parameter
        p (float): Mixing ratio for the first component
        reg (float): KL regularization strength
        reg_m1 (float): Source marginal regularization
        reg_m2 (float): Target marginal regularization
        damp (float): Damping parameter
        max_iter (int): Maximum number of iterations
        debug (bool): Whether to print debug information

    Returns:
        None
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
    G0 = warmstart_sparse(a, b, C)

    # print(G0.data)
    # print(G0.offsets)
    # print(G0.toarray())
    # print(a)
    # print(b)

    c = reg_distribiution(2 * N, C)

    sparse = UtilsSparse(a, b, c, G0, M, reg, reg_m1, reg_m2, damp)
    # print(sparse.sparse_row_sum(M.data, M.offsets))
    G, log_s = sparse.lbfgsb_unbalanced(numItermax=max_iter)
    G /= damp
    transport_cost = sparse.sparse_dot(G, sparse.offsets)
    # marginal_penalty = sparse.marg_tv_sparse_huber(G, sparse.offsets)
    regularization_term = sparse.reg_kl_sparse(G, sparse.offsets)
    marginal_penalty = sparse.marg_tv_sparse(G, sparse.offsets)

    if debug:
        emd = ot.emd2_1d(x_a=v1, x_b=v2, a=a, b=b, metric="euclidean")
        print(f"EMD: {emd}")
        print(
            f"N: {N}, C: {C}, p: {p}, reg: {reg}, reg_m1: {reg_m1}, reg_m2: {reg_m2}, damp: {damp}"
        )
        print("Transport cost: ", transport_cost * damp)
        print("Regularization term:", regularization_term * damp)
        print("Marginal penalty: ", marginal_penalty * damp)
        print(
            "Marginal penalty normalized: ", marginal_penalty * damp / (reg_m1 + reg_m2)
        )
        print(
            "Final distance: ",
            (transport_cost + marginal_penalty / (reg_m1 + reg_m2) * damp),
        )
        val, _ = sparse.func_sparse(flatten_multidiagonal(G, sparse.offsets))
        print("Value: ", val * damp)
        print("G sum: ", np.sum(G))
        print(log_s)

    # validation
    if not reg > 0:
        dense = UtilsDense(a, b, c, G0.toarray(), M.toarray(), reg, reg_m1, reg_m2)
        G_dense = dia_matrix((G * damp, sparse.offsets), shape=(2 * N, 2 * N)).toarray()

        G_d, log_d = dense.lbfgsb_unbalanced(numItermax=max_iter)

        transport_cost2 = np.sum(G_dense * dense.M)
        marginal_penalty2 = dense.marg_tv(G_dense)
        print("Dense trasport cost: ", transport_cost2 * damp)
        print("Dense marginal penalty: ", marginal_penalty2 * damp)
        val2, _ = dense.func(G_dense)
        print("Value: ", val2 * damp)
        print("Dense sum G: ", np.sum(G_d))

        print(log_d)


def run_timing_experiment():
    """
    Run timing experiments for the sparse optimization algorithm with different matrix sizes.
    Analyzes the computational complexity based on timing results.
    """

    # Fixed parameters
    C = 15
    p = 0.46
    reg = 0.01
    reg_a, reg_b = 22, 25
    max_iter = 5000

    # Values of N to test
    n_values = [40, 100, 200, 400, 800, 1200, 1600, 2000]
    times = []

    print("\nTiming Experiment Results:")
    print("N\tTime (seconds)")
    print("-" * 30)

    for N in n_values:
        start_time = time.time()
        test_sparse(N, C, p, reg, reg_a, reg_b, max_iter=max_iter)
        end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

        print(f"{N}\t{elapsed_time:.4f}")

    # Analyze complexity
    log_n = np.log(n_values)
    log_t = np.log(times)

    # Linear regression to fit power function
    coeffs = np.polyfit(log_n, log_t, 1)
    b = coeffs[0]  # Exponent
    a = np.exp(coeffs[1])  # Coefficient

    print("\nComplexity Analysis:")
    print(f"Estimated complexity: O(N^{b:.2f})")
    print(f"Fitted function: time ≈ {a:.6f} * N^{b:.2f}")

    # Interpret complexity
    if 0.8 <= b <= 1.2:
        print("Approximately linear complexity O(N)")
    elif 1.8 <= b <= 2.2:
        print("Approximately quadratic complexity O(N²)")
    elif 2.8 <= b <= 3.2:
        print("Approximately cubic complexity O(N³)")
    else:
        print(f"Complexity of O(N^{b:.2f})")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    reg_a, reg_b = 22, 25
    reg = 0.01
    N = 400
    C = 15
    max_iter = 5000

    # Uncomment the desired function to run
    # test_sparse(N, C, 0.46, reg, reg_a, reg_b, max_iter=max_iter, debug=True)
    run_timing_experiment()
