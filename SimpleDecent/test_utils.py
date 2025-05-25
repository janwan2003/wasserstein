import numpy as np
from scipy.sparse import dia_matrix, issparse
from scipy.optimize import minimize, Bounds
from ot.backend import get_backend
import sys
import os
sys.path.append(os.path.abspath(".."))
from wasserstein import Spectrum, NMRSpectrum
from typing import List
import ot
import math


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
    
    data = []
    for k in offsets:
        # For diagonal k, compute valid indices
        i_min = max(0, -k)
        i_max = min(n, n - k)
        diag_length = i_max - i_min
        
        # Compute the values for this diagonal
        diag_values = np.abs(v1[i_min:i_max] - v2[i_min + k:i_max + k])
        
        # Pad to ensure the diagonal has length `n`
        if k < 0:
            # Lower diagonal: pad at the end
            diag_values_padded = np.pad(diag_values, (0, n - diag_length), mode='constant')
        else:
            # Upper diagonal: pad at the beginning
            diag_values_padded = np.pad(diag_values, (n - diag_length, 0), mode='constant')
        
        data.append(diag_values_padded)
    
    data = np.array(data)
    return dia_matrix((data, offsets), shape=(n, n))


def reg_distribiution(N, C):
    """
    Construct a multidiagonal sparse matrix where M[i,j] = abs(v1[i] - v2[j]) if abs(i-j) < C.
    
    Parameters:
    v1, v2 (array-like): Input vectors (must be same length)
    C (int): Bandwidth parameter controlling how many diagonals are non-zero
    
    Returns:
    dia_matrix: The constructed sparse matrix in DIAgonal format
    """
    
    offsets = np.arange(-C + 1, C)  # Diagonals from -C+1 to C-1
    
    data = []
    for k in offsets:
        diag_values = np.array([1 / (N * (2 * C - 1) - C * (C - 1))] * (N - np.abs(k)))
        
        # Pad to ensure the diagonal has length `n`
        if k < 0:
            # Lower diagonal: pad at the end
            diag_values_padded = np.pad(diag_values, (0, np.abs(k)), mode='constant')
        else:
            # Upper diagonal: pad at the beginning
            diag_values_padded = np.pad(diag_values, (np.abs(k), 0), mode='constant')
        
        data.append(diag_values_padded)
    
    data = np.array(data)
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


def load_data():
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
                confs=list(zip(spectra[i][:, 0], spectra[i][:, 1])), protons=protons_list[i]
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
    spectrum_confs = sorted(spectrum.confs, key=lambda x: x[1], reverse=True)[:n_features]
    spectrum_signif = NMRSpectrum(confs=spectrum_confs, protons=spectrum.protons)
    spectrum_signif.normalize()
    return spectrum_signif


def flatten_multidiagonal(matrix_data, offsets):
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
        diag_values = flat_vector[ptr:ptr + diag_length]
        
        # Reconstruct the diagonal with proper padding
        if offset < 0:
            padded_diag = np.pad(diag_values, (0, abs(offset)), mode='constant')
        else:
            padded_diag = np.pad(diag_values, (abs(offset), 0), mode='constant')
        
        M_data.append(padded_diag)
        ptr += diag_length
    
    return np.array(M_data)


class UtilsSparse:
    def __init__(self, a, b, c, G0_sparse, M_sparse, reg, reg_m1, reg_m2, damp):
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
        """Efficient dot product between two multidiagonal matrices"""
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
                row_sums += diag[:self.m]
            elif offset < 0:  # Lower diagonal
                k = -offset
                row_sums[k:] += diag[:self.m - k]
            else:  # Upper diagonal
                row_sums[:-offset] += diag[offset:offset + self.m - offset]
        return row_sums
    
    def sparse_col_sum(self, G_data, _G_offsets):
        """Column sums for multidiagonal matrix (square matrix version)"""
        col_sums = np.zeros(self.m)
        for diag in G_data:
            col_sums += diag
        return col_sums

    # def reg_kl_sparse2(self, G_data, G_offsets):
    #     """KL divergence regularization for sparse matrix"""
    #     kl_sum = 0.0
    #     total_mass = 0.0
        
    #     for j, offset in enumerate(G_offsets):
    #         diag = G_data[j]
    #         if offset == 0:  # Main diagonal
    #             idx = np.arange(self.m)
    #         elif offset < 0:  # Lower diagonal
    #             idx = np.arange(-offset, self.m)
    #         else:  # Upper diagonal
    #             idx = np.arange(0, self.m-offset)
            
    #         c_values = self.c_diag[idx]
    #         with np.errstate(divide='ignore', invalid='ignore'):
    #             kl_terms = diag[:len(idx)] * np.log(diag[:len(idx)] / c_values + 1e-16)
    #             kl_terms = np.nan_to_num(kl_terms, nan=0.0, posinf=0.0, neginf=0.0)
    #             kl_sum += np.sum(kl_terms)
    #             total_mass += np.sum(diag[:len(idx)])
        
    #     return self.reg * (kl_sum - total_mass + np.sum(self.c_diag))
    
    def reg_kl_sparse(self, G_data, G_offsets):
        G_flat = flatten_multidiagonal(G_data, G_offsets)
        C_flat = flatten_multidiagonal(self.c_data, G_offsets)

        return np.sum(G_flat * np.log(G_flat / C_flat + 1e-16)) + np.sum(C_flat - G_flat)

    # def grad_kl_sparse(self, G_data, G_offsets):
    #     """Gradient of KL divergence for sparse matrix"""
    #     grad_data = []
    #     for j, offset in enumerate(G_offsets):
    #         diag = G_data[j]
    #         if offset == 0:  # Main diagonal
    #             idx = np.arange(self.m)
    #         elif offset < 0:  # Lower diagonal
    #             idx = np.arange(-offset, self.m)
    #         else:  # Upper diagonal
    #             idx = np.arange(0, self.m-offset)
            
    #         c_values = self.c_diag[idx]
    #         grad_diag = np.log(diag[:len(idx)] / c_values + 1e-16)
    #         grad_diag = np.nan_to_num(grad_diag, nan=0.0, posinf=0.0, neginf=0.0)
            
    #         # Pad to original diagonal length
    #         if offset < 0:
    #             grad_diag = np.pad(grad_diag, (0, -offset), 'constant')
    #         elif offset > 0:
    #             grad_diag = np.pad(grad_diag, (offset, 0), 'constant')
            
    #         grad_data.append(self.reg * grad_diag)
        
    #     return grad_data
    
    def grad_kl_sparse(self, G_data, G_offsets):
        """Gradient of KL divergence for sparse matrix"""
        G_flat = flatten_multidiagonal(G_data, G_offsets)
        C_flat = flatten_multidiagonal(self.c_data, G_offsets)

        grad_flat =  np.log(G_flat / C_flat + 1e-16)
        return reconstruct_multidiagonal(grad_flat, G_offsets, self.m) * self.reg
    
    @staticmethod
    def huber_loss(x, delta=1e-8):
        """Huber loss function (smooth approximation of L1)"""
        abs_x = np.abs(x)
        return np.where(abs_x < delta, 0.5 * x**2 / delta, abs_x - 0.5 * delta)
    
    @staticmethod
    def huber_gradient(x, delta=1e-8):
        """Gradient of Huber loss"""
        return np.where(np.abs(x) < delta, x / delta, np.sign(x))

    # def marg_tv_sparse_huber(self, G_data, G_offsets, delta=1e-6):
    #     """Huber-loss marginal penalty for sparse matrix"""
    #     row_sums = self.sparse_row_sum(G_data, G_offsets)
    #     col_sums = self.sparse_col_sum(G_data, G_offsets)
    #     return (self.reg_m1 * np.sum(self.huber_loss(row_sums - self.a, delta))) + \
    #             (self.reg_m2 * np.sum(self.huber_loss(col_sums - self.b, delta)))

    # def grad_marg_tv_sparse_huber(self, G_data, G_offsets, delta=1e-6):
    #     """Gradient of Huber marginal penalty"""
    #     row_sums = self.sparse_row_sum(G_data, G_offsets)
    #     col_sums = self.sparse_col_sum(G_data, G_offsets)
        
    #     # Compute Huber gradient terms
    #     row_grad = self.reg_m1 * self.huber_gradient(row_sums - self.a, delta)
    #     col_grad = self.reg_m2 * self.huber_gradient(col_sums - self.b, delta)
        
    #     grad_data = []
    #     for offset in G_offsets:
    #         if offset == 0:  # Main diagonal
    #             grad_diag = row_grad + col_grad
    #         elif offset < 0:  # Lower diagonal
    #             k = -offset
    #             grad_diag = row_grad[k:] + col_grad[:self.m - k]
    #             grad_diag = np.pad(grad_diag, (0, k), 'constant')
    #         else:  # Upper diagonal
    #             grad_diag = row_grad[:self.m - offset] + col_grad[offset:]
    #             grad_diag = np.pad(grad_diag, (offset, 0), 'constant')
    #         grad_data.append(grad_diag)
        
    #     return np.array(grad_data)

    def marg_tv_sparse(self, G_data, G_offsets):
        """TV marginal penalty for sparse matrix"""
        row_sums = self.sparse_row_sum(G_data, G_offsets)
        col_sums = self.sparse_col_sum(G_data, G_offsets)
        return self.reg_m1 * np.sum(np.abs(row_sums - self.a)) + self.reg_m2 * np.sum(np.abs(col_sums - self.b))

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
                grad_diag = row_signs[k:] + col_signs[:self.m - k]
                grad_diag = np.pad(grad_diag, (0, k), 'constant')
            else:  # Upper diagonal
                grad_diag = row_signs[:self.m - offset] + col_signs[offset:]
                grad_diag = np.pad(grad_diag, (offset, 0), 'constant')
            grad_data.append(grad_diag)
        
        return np.array(grad_data)

    def func_sparse(self, G_flat):
        """Combined loss function and gradient for sparse optimization"""
        # Reconstruct sparse matrix from flattened representation
        G_data = reconstruct_multidiagonal(G_flat, self.offsets, self.m)

        # Compute loss
        transport_cost = self.sparse_dot(G_data, self.offsets)
        marginal_penalty = self.marg_tv_sparse(G_data, self.offsets)
        # marginal_penalty = self.marg_tv_sparse_huber(G_data, self.offsets)
        val = transport_cost + marginal_penalty
        if self.reg > 0:
            val += self.reg_kl_sparse(G_data, self.offsets)
        
        # Compute gradient
        grad = self.data + self.grad_marg_tv_sparse(G_data, self.offsets)
        # grad = self.data + self.grad_marg_tv_sparse_huber(G_data, self.offsets)
        
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
            options={
                'ftol': 1e-12, 
                'gtol': 1e-8,
                'maxiter': numItermax
            }
        )

        G = reconstruct_multidiagonal(res.x, self.G0_sparse.offsets, self.m) * self.damp

        log = {"total_cost": res.fun * self.damp, "cost": self.sparse_dot(G, self.offsets), "res": res}
        return G, log
    

class UtilsDense:
    def __init__(self, a, b, c, G0, M, reg, reg_m1, reg_m2):
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
        return self.reg_m1 * np.outer(np.sign(G.sum(1) - self.a), np.ones(self.n)) + self.reg_m2 * np.outer(
            np.ones(self.m), np.sign(G.sum(0) - self.b)
        )
    
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
        # c = self.a[:, None] * self.b[None, :] if self.c is None else self.c

        res = minimize(
            _func,
            G0.ravel(),
            method="L-BFGS-B",
            jac=True,
            bounds=Bounds(0, np.inf),
            tol=stopThr,
            options={
                'ftol': 1e-12, 
                'gtol': 1e-8,
                'maxiter': numItermax
            }
        )

        G = res.x.reshape(self.M.shape)

        log = {"total_cost": res.fun, "cost": np.sum(G * self.M), "res": res}
        return G, log
    

def test_sparse(N, C, p, reg, reg_m1, reg_m2, damp, max_iter):
    print(f"N: {N}, C: {C}, p: {p}, reg: {reg}, reg_m1: {reg_m1}, reg_m2: {reg_m2}, damp: {damp}")

    spectra, mix = load_data()

    mix_og = signif_features(mix, 2*N)

    ratio = np.array([p, 1 - p])
    mix_aprox = Spectrum.ScalarProduct([signif_features(spectra[0], N), signif_features(spectra[1], N)], ratio)
    mix_aprox.normalize()

    a = np.array([p for _, p in mix_og.confs])
    b = np.array([p for _, p in mix_aprox.confs])

    v1 = np.array([v for v, _ in mix_og.confs])
    v2 = np.array([v for v, _ in mix_aprox.confs])

    emd = ot.emd2_1d(x_a=v1, x_b=v2, a=a, b=b, metric="euclidean")
    print(f"EMD: {emd}")

    M = multidiagonal_cost(v1, v2, C)
    G0 = warmstart_sparse(a, b, C)

    # print(G0.data)
    # print(G0.offsets)
    # print(G0.toarray())
    # print(a)
    # print(b)

    c = reg_distribiution(2*N, C)

    sparse = UtilsSparse(a, b, c, G0, M, reg, reg_m1, reg_m2, damp)
    # print(sparse.sparse_row_sum(M.data, M.offsets))
    G, log_s = sparse.lbfgsb_unbalanced(numItermax=max_iter)
    G /= damp
    transport_cost = sparse.sparse_dot(G, sparse.offsets)
    # marginal_penalty = sparse.marg_tv_sparse_huber(G, sparse.offsets)
    regularization_term = sparse.reg_kl_sparse(G, sparse.offsets)
    marginal_penalty = sparse.marg_tv_sparse(G, sparse.offsets)
    print("Transport cost: ", transport_cost * damp)
    print("Regularization term:", regularization_term * damp)
    print("Marginal penalty: ", marginal_penalty * damp)
    print("Marginal penalty normalized: ", marginal_penalty * damp / reg_m1)
    print("Final distance: ", (transport_cost + marginal_penalty / reg_m1) * damp)
    val, _ = sparse.func_sparse(flatten_multidiagonal(G, sparse.offsets))
    print("Value: ", val * damp)
    print("G sum: ", np.sum(G))
    print(log_s)

    # validation
    if not reg > 0:
        dense = UtilsDense(a, b, c, G0.toarray(), M.toarray(), reg, reg_m1, reg_m2)
        G_dense = dia_matrix((G * damp, sparse.offsets), shape=(2*N, 2*N)).toarray()

        G_d, log_d = dense.lbfgsb_unbalanced(numItermax=max_iter)

        transport_cost2 = np.sum(G_dense * dense.M)
        marginal_penalty2 = dense.marg_tv(G_dense)
        print("Dense trasport cost: ", transport_cost2 * damp)
        print("Dense marginal penalty: ", marginal_penalty2 * damp)
        val2, _ = dense.func(G_dense)
        print("Value: ", val2 * damp)
        print("Dense sum G: ", np.sum(G_d)) 

        print(log_d)


def main():
    np.random.seed(420)

    N = 10

    v1 = np.random.rand(N)
    v2 = np.random.rand(N)
    C = 3  # Bandwidth
    M_sparse = multidiagonal_cost(v1, v2, C)
    M = M_sparse.toarray()

    # Uniform marginals
    a = np.random.rand(N)
    b = np.random.rand(N)
    a /= np.sum(a)
    b /= np.sum(b)

    # Parameters
    reg = 0.01
    reg_m1 = 0.1
    reg_m2 = 0.1

    g1 = np.random.rand(N)
    g2 = np.random.rand(N)
    Gtest_sparse = multidiagonal_cost(g1, g2, C)
    Gtest_data = Gtest_sparse.data
    Gtest_offsets = Gtest_sparse.offsets
    Gtest = Gtest_sparse.toarray()

    G0_sparse = warmstart_sparse(a, b, C)
    G0_flat = flatten_multidiagonal(G0_sparse.data, G0_sparse.offsets)
    G0 = G0_sparse.toarray()
    c = reg_distribiution(N, C)
    print("c: ", c.toarray())
    print(np.sum(c.data))
    # print("G0 sparse data: ", G0_sparse.data)
    # print("G0 sparse offsets: ", G0_sparse.offsets)
    # print(G0_sparse.data.ravel())
    # print("M_sparse data: ", M_sparse.data.ravel())
    # print(flatten_multidiagonal(M_sparse.data, M_sparse.offsets))

    # M_flat = flatten_multidiagonal(M_sparse.data, M_sparse.offsets)

    # M_data = reconstruct_multidiagonal(M_flat, M_sparse.offsets, N)
    # print("M_data: ", M_data)
    # print("M_sparse data: ", M_sparse.data)

    sparse = UtilsSparse(a, b, c, G0_sparse, M_sparse, reg, reg_m1, reg_m2, damp=1)
    dense = UtilsDense(a, b, c, G0, M, reg, reg_m1, reg_m2)

    # idenpotency test
    # def float_lists_equal(list1, list2, rel_tol=1e-9, abs_tol=1e-9):
    #     if len(list1) != len(list2):
    #         return False
    #     return all(math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) 
    #         for a, b in zip(list1, list2))
    
    # flat = flatten_multidiagonal(Gtest_data, sparse.offsets)
    # flat2 = flatten_multidiagonal(reconstruct_multidiagonal(flat, sparse.offsets, N), sparse.offsets)
    
    # print(float_lists_equal(flat, flat2, rel_tol=1e-9, abs_tol=1e-9))
    # dense.reg_kl(Gtest)
    # print(sparse.reg_kl_sparse(G0_sparse.data, G0_sparse.offsets))

    # _, log_s = sparse.lbfgsb_unbalanced()
    # _, log_d = dense.lbfgsb_unbalanced()

    # print(log_s)
    # print(log_d)

    # print(Gtest_sparse.data)
    # print(Gtest)

    # print("Dense row sum:  ", Gtest.sum(axis=1))
    # print("Dense col sum:  ", Gtest.sum(axis=0))
    # print("Sparse row sum: ", sparse.sparse_row_sum(G0_sparse.data, G0_sparse.offsets))
    # print("Sparse col sum: ", sparse.sparse_col_sum(Gtest_data, Gtest_offsets))

    # print(sparse.marg_tv_sparse(Gtest_data, Gtest_offsets))
    # print(dense.marg_tv(Gtest))
    # assert np.isclose(sparse.marg_tv_sparse(Gtest_data, Gtest_offsets), dense.marg_tv(Gtest))
    # print(sparse.reg_kl_sparse(Gtest_data, Gtest_offsets), dense.reg_kl(Gtest))
    # assert np.isclose(sparse.reg_kl_sparse(Gtest_data, Gtest_offsets), dense.reg_kl(Gtest))

    # print(dense.grad_marg_tv(Gtest))
    # print(sparse.grad_marg_tv_sparse(Gtest_data, Gtest_offsets))
    # cost_d, grad_d = dense.func(G0)
    # cost_s, grad_s = sparse.func_sparse(G0_flat)
    # print(cost_d, cost_s)
    # print(grad_d)
    # print(dia_matrix((reconstruct_multidiagonal(grad_s, G0_sparse.offsets, N), G0_sparse.offsets), shape=(N, N)).toarray())


    # print(np.sum(G0test * M))
    # print(sparse.sparse_dot(G0test_data, G0test_offsets))
    # assert np.sum(G0test * M) == sparse.sparse_dot(G0test_data, G0test_offsets)


if __name__ == "__main__":
    # for reg_m in [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1, 1.5, 2, 3.5, 5, 7, 10]:
    #     d = 1
    #     # damp = d * reg_m
    #     damp = 1
    #     print(f"Testing reg_m: {reg_m}, damp: {damp}")
    #     test_sparse(10000, 20, 0.5, reg_m, reg_m, damp, 1000)
    test_sparse(20, 5, 0.2, 0.01, 1000.5, 1000.5, 1, 5000)
    test_sparse(20, 5, 0.4, 0.01, 1000.5, 1000.5, 1, 5000)
    test_sparse(20, 5, 0.6, 0.01, 1000.5, 1000.5, 1, 5000)
    test_sparse(20, 5, 0.8, 0.01, 1000.5, 1000.5, 1, 5000)
    # main()