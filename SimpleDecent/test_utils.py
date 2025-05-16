import numpy as np
from scipy.sparse import dia_matrix, issparse
from scipy.optimize import minimize, Bounds
from ot.backend import get_backend
import sys
import os
sys.path.append(os.path.abspath(".."))
from wasserstein import Spectrum, NMRSpectrum
from typing import List


def multidiagonal_matrix(v1, v2, C):
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
        diag_len = n - abs(offset)
        diagonals[offset] = np.zeros(diag_len)
    
    for i in range(n):
        remaining_mass = p1[i]
        
        # Step 1: Assign to diagonal (i,i)
        assign = min(remaining_mass, p2[i])
        diagonals[0][i] = assign
        remaining_mass -= assign
        
        # Step 2: Distribute remaining mass to adjacent cells
        radius = 1
        while remaining_mass > 1e-10 and radius < C:
            # Left neighbor (i, i-radius)
            if i - radius >= 0:
                idx = i - radius
                assign = min(remaining_mass, p2[idx])
                diagonals[-radius][idx] += assign
                remaining_mass -= assign
            
            # Right neighbor (i, i+radius)
            if i + radius < n and remaining_mass > 1e-10:
                assign = min(remaining_mass, p2[i + radius])
                diagonals[radius][i] += assign
                remaining_mass -= assign
            
            radius += 1
    
    # Pad diagonals to equal length for dia_matrix
    padded_data = []
    for offset in offsets:
        diag = diagonals[offset]
        if offset < 0:
            # Lower diagonal: pad at end
            padded = np.pad(diag, (0, -offset), 'constant')
        elif offset > 0:
            # Upper diagonal: pad at beginning
            padded = np.pad(diag, (offset, 0), 'constant')
        else:
            # Main diagonal: no padding needed
            padded = diag
        padded_data.append(padded)
    
    return dia_matrix((np.array(padded_data), offsets), shape=(n, n))


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


class UtilsSparse:
    def __init__(self, a, b, c, G0_sparse, M_sparse, reg, reg_m1, reg_m2):
        self.a = a
        self.b = b
        self.c = c
        self.m, self.n = M_sparse.shape

        assert self.m == self.n, "Only square multidiagonal matrices supported"
        assert len(a) == len(b) == self.m, "Marginals must match matrix dimensions"
        
        self.G0_sparse = G0_sparse
        self.offsets = M_sparse.offsets
        self.data = M_sparse.data

        if issparse(c):
            self.c_diag = c.diagonal()
        else:
            self.c_diag = np.asarray(c).reshape(-1)

        self.reg = reg
        self.reg_m1 = reg_m1
        self.reg_m2 = reg_m2

    def sparse_dot(self, G_data, G_offsets):
        """Efficient dot product between two multidiagonal matrices"""
        total = 0.0
        for i, offset1 in enumerate(self.offsets):
            for j, offset2 in enumerate(G_offsets):
                if offset1 == offset2:
                    min_len = min(len(self.data[i]), len(G_data[j]))
                    total += np.sum(self.data[i][:min_len] * G_data[j][:min_len])
        return total

    # def dense_sparse_dot(self, dense_mat):
    #     """
    #     Compute dot product between a dense matrix and this sparse multidiagonal matrix.
        
    #     Args:
    #         dense_mat (np.ndarray): Dense matrix of shape (m, n)
            
    #     Returns:
    #         np.ndarray: Resulting dense matrix of shape (m, n)
    #     """
    #     m, n = dense_mat.shape
    #     result = np.zeros((m, n))
        
    #     for diag, offset in zip(self.data, self.offsets):
    #         diag_len = len(diag)
            
    #         if offset < 0:
    #             # Lower diagonal
    #             start_row = -offset
    #             for i in range(min(diag_len, m - start_row)):
    #                 j = i + start_row + offset
    #                 result[start_row + i, j] = dense_mat[start_row + i, j] * diag[i]
    #         elif offset > 0:
    #             # Upper diagonal
    #             start_col = offset
    #             for j in range(min(diag_len, n - start_col)):
    #                 i = j
    #                 result[i, start_col + j] = dense_mat[i, start_col + j] * diag[j]
    #         else:
    #             # Main diagonal
    #             for k in range(min(diag_len, m, n)):
    #                 result[k, k] = dense_mat[k, k] * diag[k]
        
    #     return result
    
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

    def reg_kl_sparse(self, G_data, G_offsets):
        """KL divergence regularization for sparse matrix"""
        kl_sum = 0.0
        total_mass = 0.0
        
        for j, offset in enumerate(G_offsets):
            diag = G_data[j]
            if offset == 0:  # Main diagonal
                idx = np.arange(self.m)
            elif offset < 0:  # Lower diagonal
                idx = np.arange(-offset, self.m)
            else:  # Upper diagonal
                idx = np.arange(0, self.m-offset)
            
            c_values = self.c_diag[idx]
            with np.errstate(divide='ignore', invalid='ignore'):
                kl_terms = diag[:len(idx)] * np.log(diag[:len(idx)] / c_values + 1e-16)
                kl_terms = np.nan_to_num(kl_terms, nan=0.0, posinf=0.0, neginf=0.0)
                kl_sum += np.sum(kl_terms)
                total_mass += np.sum(diag[:len(idx)])
        
        return kl_sum - total_mass + np.sum(self.c_diag)

    def grad_kl_sparse(self, G_data, G_offsets):
        """Gradient of KL divergence for sparse matrix"""
        grad_data = []
        for j, offset in enumerate(G_offsets):
            diag = G_data[j]
            if offset == 0:  # Main diagonal
                idx = np.arange(self.m)
            elif offset < 0:  # Lower diagonal
                idx = np.arange(-offset, self.m)
            else:  # Upper diagonal
                idx = np.arange(0, self.m-offset)
            
            c_values = self.c_diag[idx]
            grad_diag = np.log(diag[:len(idx)] / c_values + 1e-16)
            grad_diag = np.nan_to_num(grad_diag, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Pad to original diagonal length
            if offset < 0:
                grad_diag = np.pad(grad_diag, (0, -offset), 'constant')
            elif offset > 0:
                grad_diag = np.pad(grad_diag, (offset, 0), 'constant')
            
            grad_data.append(grad_diag)
        
        return grad_data

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
        G_data = []
        ptr = 0
        for offset in self.offsets:
            G_data.append(G_flat[ptr:ptr+self.m])
            ptr += self.m

        # Compute loss
        transport_cost = self.sparse_dot(G_data, self.offsets)
        marginal_penalty = self.marg_tv_sparse(G_data, self.offsets)
        val = transport_cost + marginal_penalty
        if self.reg > 0:
            if self.reg_div == "kl":
                kl_penalty = self.reg * self.reg_kl_sparse(G_data, self.offsets)
                val += kl_penalty
        
        # Compute gradient
        grad_marg = self.grad_marg_tv_sparse(G_data, self.offsets)
        
        # Combine gradients
        grad_flat = np.zeros_like(G_flat)
        ptr = 0
        for i, offset in enumerate(self.offsets):
            # Transport cost gradient (M)
            grad_flat[ptr:ptr+self.m] += self.data[i][:self.m]
            
            # Marginal penalty gradient
            grad_flat[ptr:ptr+self.m] += grad_marg[i][:self.m]
            
            # KL divergence gradient if needed
            if self.reg > 0 and self.reg_div == "kl":
                grad_kl_diag = self.grad_kl_sparse(G_data, self.offsets)[i][:self.m]
                grad_flat[ptr:ptr+self.m] += self.reg * grad_kl_diag
            
            ptr += self.m
        
        return val, grad_flat
    
    def lbfgsb_unbalanced(self, numItermax=1000, stopThr=1e-15, verbose=False):
        _func = self.func_sparse

        # panic for now
        # if self.c is None:
        #     raise ValueError("Reference distribution 'c' must be provided for unbalanced OT")
        if self.G0_sparse is None:
            raise ValueError("Initial transport plan 'G0' must be provided")
            

        res = minimize(
            _func,
            self.G0_sparse.data.ravel(),
            method="L-BFGS-B",
            jac=True,
            bounds=Bounds(0, np.inf),
            tol=stopThr,
            options=dict(maxiter=numItermax, disp=verbose),
        )

        G = res.x.reshape(self.data.shape)

        log = {"cost": self.sparse_dot(G, self.offsets), "res": res}
        log["total_cost"] = res.fun
        return G, log
    

class UtilsDense:
    def __init__(self, a, b, c, G0, M, reg, reg_m1, reg_m2):
        self.a = a
        self.b = b
        self.c = c
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

        return val, grad.ravel()

    def lbfgsb_unbalanced(self, numItermax=1000, stopThr=1e-15, verbose=False):
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
            options=dict(maxiter=numItermax, disp=verbose),
        )

        G = res.x.reshape(self.M.shape)

        log = {"cost": np.sum(G * self.M), "res": res}
        log["total_cost"] = res.fun
        return G, log
    
def main():
    np.random.seed(420)

    N = 200

    v1 = np.random.rand(N)
    v2 = np.random.rand(N)
    C = 4  # Bandwidth
    M_sparse = multidiagonal_matrix(v1, v2, C)
    M = M_sparse.toarray()

    # Uniform marginals
    a = np.random.rand(N)
    b = np.random.rand(N)
    a /= np.sum(a)
    b /= np.sum(b)

    # Parameters
    reg = 0
    reg_m1 = 0.1
    reg_m2 = 0.1

    g1 = np.random.rand(N)
    g2 = np.random.rand(N)
    Gtest_sparse = multidiagonal_matrix(g1, g2, C)
    Gtest_data = Gtest_sparse.data
    Gtest_offsets = Gtest_sparse.offsets
    Gtest = Gtest_sparse.toarray()

    c = None
    G0_sparse = warmstart_sparse(a, b, C)
    G0 = G0_sparse.toarray()

    sparse = UtilsSparse(a, b, c, G0_sparse, M_sparse, reg, reg_m1, reg_m2)
    dense = UtilsDense(a, b, c, G0, M, reg, reg_m1, reg_m2)

    _, log_s = sparse.lbfgsb_unbalanced()
    _, log_d = dense.lbfgsb_unbalanced()

    print(log_s)
    print(log_d)

    # print(G0test_sparse.data)
    # print(G0test)

    # print("Dense row sum:  ", G0test.sum(axis=1))
    # print("Dense col sum:  ", G0test.sum(axis=0))
    # print("Sparse row sum: ", sparse.sparse_row_sum(G0test_data, G0test_offsets))
    # print("Sparse col sum: ", sparse.sparse_col_sum(G0test_data, G0test_offsets))

    # print(sparse.marg_tv_sparse(G0test_data, G0test_offsets))
    # print(dense.marg_tv(G0test))
    # assert sparse.marg_tv_sparse(G0test_data, G0test_offsets) == dense.marg_tv(G0test)
    # assert sparse.grad_marg_tv_sparse(G0test_data, G0test_offsets) == dense.grad_marg_tv(G0test)

    # print(dense.grad_marg_tv(G0test))
    # print(sparse.grad_marg_tv_sparse(G0test_data, G0test_offsets))
    # cost_d, grad_d = dense.func(G0test)
    # cost_s, grad_s = sparse.func_sparse(G0test_data.ravel())
    # print(G0test_data.ravel())
    # print(cost_d, cost_s)
    # print(grad_d)
    # print(grad_s)


    # print(np.sum(G0test * M))
    # print(sparse.sparse_dot(G0test_data, G0test_offsets))
    # assert np.sum(G0test * M) == sparse.sparse_dot(G0test_data, G0test_offsets)

    # print(warmstart_sparse(a, b, C).toarray())


if __name__ == "__main__":
    main()