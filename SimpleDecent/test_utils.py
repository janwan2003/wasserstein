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
from numba import njit
sys.path.append(os.path.abspath(".."))
from wasserstein import Spectrum, NMRSpectrum


# --- Numba JIT-compiled Helper Functions (outside the class) ---
# These functions now take all necessary data explicitly as arguments,
# allowing Numba to compile them effectively in nopython mode.

@njit
def _compute_flat_length(matrix_data, offsets):
    total_len = 0
    for i in range(len(offsets)):
        offset = offsets[i]
        if offset < 0:
            total_len += matrix_data[i].shape[0] + offset  # slice is [:offset]
        else:
            total_len += matrix_data[i].shape[0] - offset  # slice is [offset:]
    return total_len

@njit
def flatten_multidiagonal(matrix_data, offsets):
    """
    Flatten a multidiagonal matrix data into a 1D array using prange.

    Parameters:
        matrix_data: 2D NumPy array of shape (num_diagonals, m)
        offsets: 1D array of diagonal offsets

    Returns:
        np.array: Flattened 1D array
    """
    num_diags = len(offsets)
    # First compute total length
    total_len = _compute_flat_length(matrix_data, offsets)

    flat_vector = np.empty(total_len, dtype=matrix_data[0].dtype)
    positions = np.empty(num_diags + 1, dtype=np.int32)
    positions[0] = 0

    # Prefix sum to get start positions for each diag
    for i in range(num_diags):
        offset = offsets[i]
        if offset < 0:
            diag_len = matrix_data[i].shape[0] + offset
        else:
            diag_len = matrix_data[i].shape[0] - offset
        positions[i + 1] = positions[i] + diag_len

    # Parallel fill
    for i in range(num_diags):
        offset = offsets[i]
        diag = matrix_data[i]
        start = positions[i]

        if offset < 0:
            flat_vector[start:start + diag.shape[0] + offset] = diag[:offset]
        else:
            flat_vector[start:start + diag.shape[0] - offset] = diag[offset:]

    return flat_vector

@njit
def reconstruct_multidiagonal(flat_vector, offsets, N):
    num_diags = len(offsets)
    max_len = N
    M_data = np.zeros((num_diags, max_len))

    # Precompute the start index for each diagonal in flat_vector
    ptrs = np.zeros(num_diags, dtype=np.int64)
    ptr = 0
    for i in range(num_diags):
        ptrs[i] = ptr
        ptr += N - abs(offsets[i])

    # Fill M_data in parallel
    for i in range(num_diags):
        offset = offsets[i]
        diag_len = N - abs(offset)
        start_ptr = ptrs[i]

        for j in range(diag_len):
            if offset < 0:
                M_data[i, j] = flat_vector[start_ptr + j]
            else:
                M_data[i, j + offset] = flat_vector[start_ptr + j]

    return M_data

@njit
def _sparse_dot_numba(data1, offsets1, data2, offsets2):
    tmp = np.zeros(len(offsets1))
    for i in range(len(offsets1)):
        offset1 = offsets1[i]
        for j in range(len(offsets2)):
            offset2 = offsets2[j]
            if offset1 == offset2:
                diag1 = data1[i]
                diag2 = data2[j]
                min_len = min(len(diag1), len(diag2))
                acc = 0.0
                for k in range(min_len):
                    acc += diag1[k] * diag2[k]
                tmp[i] += acc
    return np.sum(tmp)

@njit
def _sparse_row_sum_numba(m, G_data, G_offsets):
    """Jitted helper for sparse_row_sum."""
    row_sums = np.zeros(m, dtype=G_data[0].dtype)
    for k in range(len(G_offsets)):
        offset = G_offsets[k]
        diag = G_data[k]
        
        # Determine the effective length of the diagonal within the matrix bounds
        if offset == 0:  # Main diagonal
            diag_start_row = 0
            diag_len = m
        elif offset < 0:  # Lower diagonal
            diag_start_row = -offset
            diag_len = m + offset
        else:  # Upper diagonal
            diag_start_row = 0
            diag_len = m - offset

        for i in range(diag_len):
            row_sums[diag_start_row + i] += diag[i]
    return row_sums

@njit
def _sparse_col_sum_numba(m, G_data, G_offsets):
    """Jitted helper for sparse_col_sum."""
    col_sums = np.zeros(m, dtype=G_data[0].dtype)
    for k in range(len(G_offsets)):
        offset = G_offsets[k]
        diag = G_data[k]
        
        # Determine the effective length of the diagonal within the matrix bounds
        if offset == 0: # Main diagonal
            diag_start_col = 0
            diag_len = m
        elif offset < 0: # Lower diagonal
            diag_start_col = 0
            diag_len = m + offset
        else: # Upper diagonal
            diag_start_col = offset
            diag_len = m - offset

        for i in range(diag_len):
            col_sums[diag_start_col + i] += diag[i]
    return col_sums

@njit
def _reg_kl_sparse_numba(c_data, m, G_data, G_offsets):
    """Jitted helper for reg_kl_sparse."""
    G_flat = flatten_multidiagonal(G_data, G_offsets)
    C_flat = flatten_multidiagonal(c_data, G_offsets)

    term1 = 0.0
    for i in range(len(G_flat)):
        # Added 1e-16 to denominator and log input for numerical stability
        term1 += G_flat[i] * np.log(G_flat[i] / (C_flat[i] + 1e-16) + 1e-16)

    term2 = np.sum(C_flat - G_flat)
    return term1 + term2

@njit
def _grad_kl_sparse_numba(c_data, m, reg, G_data, G_offsets):
    """Jitted helper for grad_kl_sparse."""
    G_flat = flatten_multidiagonal(G_data, G_offsets)
    C_flat = flatten_multidiagonal(c_data, G_offsets)

    grad_flat = np.empty_like(G_flat)
    for i in range(len(G_flat)):
        grad_flat[i] = np.log(G_flat[i] / (C_flat[i] + 1e-16) + 1e-16)

    reconstructed_grad = reconstruct_multidiagonal(grad_flat, G_offsets, m)
    
    # Element-wise multiplication with reg on each array in the list
    for i in range(len(reconstructed_grad)):
        reconstructed_grad[i] *= reg
            
    return reconstructed_grad

@njit
def _marg_tv_sparse_numba(m, reg_m1, reg_m2, a, b, G_data, G_offsets):
    """Jitted helper for marg_tv_sparse."""
    row_sums = _sparse_row_sum_numba(m, G_data, G_offsets)
    col_sums = _sparse_col_sum_numba(m, G_data, G_offsets)
    return reg_m1 * np.sum(np.abs(row_sums - a)) + \
           reg_m2 * np.sum(np.abs(col_sums - b))

@njit
def _marg_tv_sparse_rm1_numba(m, reg_m1, a, G_data, G_offsets):
    """Jitted helper for marg_tv_sparse_rm1."""
    row_sums = _sparse_row_sum_numba(m, G_data, G_offsets)
    return reg_m1 * np.sum(np.abs(row_sums - a))

@njit
def _marg_tv_sparse_rm2_numba(m, reg_m2, b, G_data, G_offsets):
    """Jitted helper for marg_tv_sparse_rm2."""
    col_sums = _sparse_col_sum_numba(m, G_data, G_offsets)
    return reg_m2 * np.sum(np.abs(col_sums - b))

@njit
def _grad_marg_tv_sparse_numba(m, reg_m1, reg_m2, a, b, G_data, G_offsets):
    """Jitted version of grad_marg_tv_sparse that returns padded 2D array like original."""
    
    # Compute row and column sums
    row_sums = _sparse_row_sum_numba(m, G_data, G_offsets)
    col_sums = _sparse_col_sum_numba(m, G_data, G_offsets)

    # Compute sign vectors
    row_signs = reg_m1 * np.sign(row_sums - a)
    col_signs = reg_m2 * np.sign(col_sums - b)

    num_diagonals = len(G_offsets)
    grad_data_array = np.zeros((num_diagonals, m))  # full padded output

    for i in range(num_diagonals):
        offset = G_offsets[i]

        if offset == 0:
            grad_data_array[i, :] = row_signs + col_signs

        elif offset < 0:
            k = -offset
            diag_len = m - k
            grad_diag = row_signs[k:] + col_signs[:diag_len]
            grad_data_array[i, 0:diag_len] = grad_diag

        else:  # offset > 0
            diag_len = m - offset
            grad_diag = row_signs[:diag_len] + col_signs[offset:]
            grad_data_array[i, offset:offset + diag_len] = grad_diag

    return grad_data_array


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

    return dia_matrix((np.array(diagonals), offsets), shape=(n, n), dtype=np.float64)


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

    def __init__(self, a, b, c, G0_sparse, M_sparse, reg, reg_m1, reg_m2):
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

        if self.reg > 0:
            grad += self.grad_kl_sparse(G_data, self.offsets)

        grad_flat = flatten_multidiagonal(grad, self.offsets)

        return val, grad_flat

    def sparse_dot(self, G_data, G_offsets):
        """
        Efficient dot product between two multidiagonal matrices.
        Calls a jitted helper function.
        """
        return _sparse_dot_numba(self.data, self.offsets, G_data, G_offsets)

    def sparse_row_sum(self, G_data, G_offsets):
        """Row sums for multidiagonal matrix. Calls a jitted helper."""
        return _sparse_row_sum_numba(self.m, G_data, G_offsets)

    def sparse_col_sum(self, G_data, _G_offsets):
        """Column sums for multidiagonal matrix. Calls a jitted helper."""
        return _sparse_col_sum_numba(self.m, G_data, _G_offsets)

    def reg_kl_sparse(self, G_data, G_offsets):
        """Calls a jitted helper for KL regularization."""
        return _reg_kl_sparse_numba(self.c_data, self.m, G_data, G_offsets)

    def grad_kl_sparse(self, G_data, G_offsets):
        """Calls a jitted helper for KL gradient."""
        return _grad_kl_sparse_numba(self.c_data, self.m, self.reg, G_data, G_offsets)

    def marg_tv_sparse(self, G_data, G_offsets):
        """Calls a jitted helper for TV marginal penalty."""
        return _marg_tv_sparse_numba(
            self.m, self.reg_m1, self.reg_m2, self.a, self.b, G_data, G_offsets
        )

    def marg_tv_sparse_rm1(self, G_data, G_offsets):
        """Calls a jitted helper for row TV marginal penalty."""
        return _marg_tv_sparse_rm1_numba(self.m, self.reg_m1, self.a, G_data, G_offsets)
    
    def marg_tv_sparse_rm2(self, G_data, G_offsets):
        """Calls a jitted helper for column TV marginal penalty."""
        return _marg_tv_sparse_rm2_numba(self.m, self.reg_m2, self.b, G_data, G_offsets)

    def grad_marg_tv_sparse(self, G_data, G_offsets):
        """Calls a jitted helper for TV marginal gradient."""
        return _grad_marg_tv_sparse_numba(
            self.m, self.reg_m1, self.reg_m2, self.a, self.b, G_data, G_offsets
        )

    # def func_sparse(self, G_flat):
    #     """
    #     Combined loss function and gradient for sparse optimization.
    #     This function remains a standard Python method as it orchestrates calls
    #     to jitted functions and interacts with non-jittable Python objects/structures
    #     like lists of arrays and `scipy.optimize.minimize`.
    #     """
    #     # Reconstruct sparse matrix from flattened representation
    #     G_data = reconstruct_multidiagonal(G_flat, self.offsets, self.m)

    #     # Compute loss (these calls use the jitted helpers)
    #     transport_cost = self.sparse_dot(G_data, self.offsets)
    #     marginal_penalty = self.marg_tv_sparse(G_data, self.offsets)
    #     val = transport_cost + marginal_penalty
    #     if self.reg > 0:
    #         val += self.reg_kl_sparse(G_data, self.offsets)

    #     # Compute gradient (these calls use the jitted helpers)
    #     # Create a new list for grad_data to avoid modifying self.data in place
    #     grad_data = [arr.copy() for arr in self.data] 

    #     grad_marg_data = self.grad_marg_tv_sparse(G_data, self.offsets)
        
    #     # Perform element-wise addition of arrays in the lists (Python loop)
    #     for i in range(len(grad_data)):
    #         # Assuming grad_data[i] and grad_marg_data[i] have compatible shapes
    #         grad_data[i] += grad_marg_data[i]

    #     if self.reg > 0:
    #         grad_kl_data = self.grad_kl_sparse(G_data, self.offsets)
    #         for i in range(len(grad_data)):
    #             grad_data[i] += grad_kl_data[i]

    #     # Flatten the combined gradient for the optimizer
    #     grad_flat = flatten_multidiagonal(grad_data, self.offsets, self.m)

    #     return val, grad_flat

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
            options={
                # "ftol": 1e-12, 
                # "gtol": 1e-8, 
                "maxiter": numItermax
            },
        )

        G = reconstruct_multidiagonal(res.x, self.G0_sparse.offsets, self.m)

        log = {
            "total_cost": res.fun,
            "cost": self.sparse_dot(G, self.offsets),
            "res": res,
        }
        # print(res)
        return G, log

    def mirror_descent_unbalanced(self, numItermax=1000, step_size=0.1, stopThr=1e-9):
        """
        Solves the unbalanced OT problem using mirror descent with exponential updates.

        Parameters:
            numItermax (int): Maximum number of iterations
            step_size (float): Fixed step size for gradient updates
            stopThr (float): Stopping threshold for relative change in objective

        Returns:
            tuple: (G, log) where G is the optimal transport plan and log contains information about the optimization
        """
        if self.G0_sparse is None:
            raise ValueError("Initial transport plan 'G0' must be provided")

        G_offsets = self.G0_sparse.offsets.copy()
        G_flat = flatten_multidiagonal(self.G0_sparse.data, G_offsets)

        val_prev, grad_flat = self.func_sparse(G_flat)

        log = {
            "loss": [val_prev],
            "step_sizes": [step_size], # ?
        }

        for i in range(numItermax):
            grad_flat_clipped = np.clip(grad_flat, -100, 100)
            G_w = G_flat * np.exp(-step_size * grad_flat_clipped)
            G_w = np.maximum(G_w, 1e-15)
            G_flat_new = G_w / np.sum(G_w)

            val_new, grad_flat_new = self.func_sparse(G_flat_new)

            rel_change = abs(val_new - val_prev) / max(abs(val_prev), 1e-10)
            log["loss"].append(val_new)
            log["step_sizes"].append(step_size) # ?

            if rel_change < stopThr:
                log["convergence"] = True
                log["iterations"] = i + 1
                G_flat = G_flat_new
                break

            G_flat = G_flat_new
            grad_flat = grad_flat_new
            val_prev = val_new

        else:
            log["convergence"] = False
            log["iterations"] = numItermax

        G = reconstruct_multidiagonal(G_flat, G_offsets, self.m)
        log["total_cost"] = val_prev
        log["cost"] = self.sparse_dot(G, G_offsets)

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


def test_sparse(N, C, p, reg, reg_m1, reg_m2, max_iter=5000, debug=False):
    """
    Test the sparse optimization algorithm on NMR spectra.

    Parameters:
        N (int): Number of features to use for each component
        C (int): Bandwidth parameter
        p (float): Mixing ratio for the first component
        reg (float): KL regularization strength
        reg_m1 (float): Source marginal regularization
        reg_m2 (float): Target marginal regularization
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

    sparse = UtilsSparse(a, b, c, G0, M, reg, reg_m1, reg_m2)
    # print(sparse.sparse_row_sum(M.data, M.offsets))
    G, log_s = sparse.lbfgsb_unbalanced(numItermax=max_iter)
    transport_cost = sparse.sparse_dot(G, sparse.offsets)
    # marginal_penalty = sparse.marg_tv_sparse_huber(G, sparse.offsets)
    regularization_term = sparse.reg_kl_sparse(G, sparse.offsets)
    marginal_penalty = sparse.marg_tv_sparse(G, sparse.offsets)

    if debug:
        emd = ot.emd2_1d(x_a=v1, x_b=v2, a=a, b=b, metric="euclidean")
        print(f"EMD: {emd}")
        print(
            f"N: {N}, C: {C}, p: {p}, reg: {reg}, reg_m1: {reg_m1}, reg_m2: {reg_m2}"
        )
        print("Transport cost: ", transport_cost)
        print("Regularization term:", regularization_term)
        print("Marginal penalty: ", marginal_penalty)
        print(
            "Marginal penalty normalized: ", marginal_penalty / (reg_m1 + reg_m2)
        )
        print(
            "Final distance: ",
            (transport_cost + marginal_penalty / (reg_m1 + reg_m2)),
        )
        val, _ = sparse.func_sparse(flatten_multidiagonal(G, sparse.offsets))
        print("Value: ", val)
        print("G sum: ", np.sum(G))
        print(log_s)

    # validation
    if not reg > 0:
        dense = UtilsDense(a, b, c, G0.toarray(), M.toarray(), reg, reg_m1, reg_m2)
        G_dense = dia_matrix((G, sparse.offsets), shape=(2 * N, 2 * N)).toarray()

        G_d, log_d = dense.lbfgsb_unbalanced(numItermax=max_iter)

        transport_cost2 = np.sum(G_dense * dense.M)
        marginal_penalty2 = dense.marg_tv(G_dense)
        print("Dense trasport cost: ", transport_cost2)
        print("Dense marginal penalty: ", marginal_penalty2)
        val2, _ = dense.func(G_dense)
        print("Value: ", val2)
        print("Dense sum G: ", np.sum(G_d))

        print(log_d)


def test_sparse_mirror_descent(
    N,
    C,
    p,
    reg,
    reg_m1,
    reg_m2,
    max_iter=5000,
    step_size=0.1,
    debug=False,
    warmstart=None,
):
    """
    Test the sparse optimization algorithm using mirror descent on NMR spectra.

    Parameters:
        N (int): Number of features to use for each component
        C (int): Bandwidth parameter
        p (float): Mixing ratio for the first component
        reg (float): KL regularization strength
        reg_m1 (float): Source marginal regularization
        reg_m2 (float): Target marginal regularization
        max_iter (int): Maximum number of iterations
        step_size (float): Initial step size for mirror descent
        adaptive_step (bool): Whether to use adaptive step size
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
    if warmstart is None:
        warmstart = warmstart_sparse(a, b, C)
    # print("Warmstart shape: ", warmstart.shape)
    c = reg_distribiution(2 * N, C)

    sparse = UtilsSparse(a, b, c, warmstart, M, reg, reg_m1, reg_m2)

    # Use mirror descent instead of L-BFGS-B
    start_time = time.time()
    G, log_s = sparse.mirror_descent_unbalanced(
        numItermax=max_iter, step_size=step_size
    )
    end_time = time.time()

    transport_cost = sparse.sparse_dot(G, sparse.offsets)
    regularization_term = sparse.reg_kl_sparse(G, sparse.offsets)
    marginal_penalty = sparse.marg_tv_sparse(G, sparse.offsets)
    final_distance = transport_cost + marginal_penalty / (reg_m1 + reg_m2)
    log_s["final_distance"] = final_distance
    if debug:
        emd = ot.emd2_1d(x_a=v1, x_b=v2, a=a, b=b, metric="euclidean")
        print(f"EMD: {emd}")
        print(
            f"N: {N}, C: {C}, p: {p}, reg: {reg}, reg_m1: {reg_m1}, reg_m2: {reg_m2}"
        )
        print("Optimization method: Mirror Descent")
        print(f"Iterations: {log_s.get('iterations', max_iter)}")
        print(f"Converged: {log_s.get('convergence', False)}")
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        print("Transport cost: ", transport_cost)
        print("Regularization term:", regularization_term)
        print("Marginal penalty: ", marginal_penalty)
        print(
            "Marginal penalty normalized: ", marginal_penalty / (reg_m1 + reg_m2)
        )
        print(
            "Final distance: ",
            (transport_cost + marginal_penalty / (reg_m1 + reg_m2)),
        )
        print("G sum: ", np.sum(G))

        # If we have loss history, we can print it or plot it
        if "loss" in log_s and len(log_s["loss"]) > 0:
            print(f"Initial loss: {log_s['loss'][0]}")
            print(f"Final loss: {log_s['loss'][-1]}")
            print(f"Loss reduction: {log_s['loss'][0] - log_s['loss'][-1]}")

    return G, log_s


def test_ws_distance(
    N,
    C,
    p,
    reg,
    reg_m1,
    reg_m2,
    max_iter=5000,
    step_size=0.1,
    debug=False,
):
    """
    Test the sparse optimization algorithm using mirror descent on NMR spectra.

    Parameters:
        N (int): Number of features to use for each component
        C (int): Bandwidth parameter
        p (float): Mixing ratio for the first component
        reg (float): KL regularization strength
        reg_m1 (float): Source marginal regularization
        reg_m2 (float): Target marginal regularization
        max_iter (int): Maximum number of iterations
        step_size (float): Initial step size for mirror descent
        adaptive_step (bool): Whether to use adaptive step size
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

    return mix_og.WSDistance(mix_aprox)


# ------------------------------------------------------------------------------
# Parameter Tuning Functions
# ------------------------------------------------------------------------------


def find_optimal_reg():
    # 1.5 co ciekawe
    N, C, p = 400, 15, 0.46
    reg_a = reg_b = 22
    max_iter = 2000

    low, high = 0.001, 10
    best_reg, best_cost = None, float("inf")

    for _ in range(10):
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        _, log1 = test_sparse_mirror_descent(
            N, C, p, mid1, reg_a, reg_b, max_iter=max_iter, step_size=0.002
        )
        _, log2 = test_sparse_mirror_descent(
            N, C, p, mid2, reg_a, reg_b, max_iter=max_iter, step_size=0.002
        )

        cost1 = log1["final_distance"]
        cost2 = log2["final_distance"]

        if cost1 < cost2:
            high = mid2
            if cost1 < best_cost:
                best_reg, best_cost = mid1, cost1
        else:
            low = mid1
            if cost2 < best_cost:
                best_reg, best_cost = mid2, cost2

    print(f"Optimal reg: {best_reg}, Cost: {best_cost}")
    return best_reg


def find_optimal_reg_marginals(
    N=400, C=15, p=0.46, reg=1.5, max_iter=2000, low=1, high=500
):
    # 333 ale w sumie to może nie być dobra metoda, może lepiej porównać różnicę między tym a 0.46 np. zamiast maksymalizować
    best_reg_m, best_cost = None, float("inf")

    for _ in range(10):
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        _, log1 = test_sparse_mirror_descent(
            N, C, p, reg, mid1, mid1, max_iter=max_iter, step_size=0.002
        )
        _, log2 = test_sparse_mirror_descent(
            N, C, p, reg, mid2, mid2, max_iter=max_iter, step_size=0.002
        )

        cost1 = log1["final_distance"]
        cost2 = log2["final_distance"]

        if cost1 < cost2:
            high = mid2
            if cost1 < best_cost:
                best_reg_m, best_cost = mid1, cost1
        else:
            low = mid1
            if cost2 < best_cost:
                best_reg_m, best_cost = mid2, cost2

    print(f"Optimal reg_a/reg_b: {best_reg_m}, Cost: {best_cost}")
    return best_reg_m


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    reg_a, reg_b = 50, 50
    reg = 1.5
    N = 400
    C = 15
    max_iter = 1000
    step_size = 0.0005

    # Uncomment the desired function to run
    # test_sparse(N, C, 0.46, reg, reg_a, reg_b, max_iter=max_iter, debug=True)
    # test_sparse(N, C, 0.4, reg, reg_a, reg_b, max_iter=max_iter, debug=True)

    # for p in [0.3, 0.35, 0.4, 0.45, 0.5]:
    #     test_sparse_mirror_descent(
    #         N,
    #         C,
    #         p,
    #         reg,
    #         reg_a,
    #         reg_b,
    #         max_iter=max_iter,
    #         step_size=step_size,
    #         debug=True,
    #     )
    test_sparse_mirror_descent(
        N, C, 0.4, reg, reg_a, reg_b, max_iter=max_iter, step_size=step_size, debug=True
    )
    # run_comparison_experiment()

    # Uncomment to find optimal parameters
    find_optimal_reg()
    # find_optimal_reg_marginals(reg=1.5)
