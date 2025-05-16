import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.sparse import dia_matrix, issparse

from ot.backend import get_backend
from ot.utils import list_to_array, get_parameter_pair


def get_loss_unbalanced_sparse(a, b, c, M_sparse, reg, reg_m1, reg_m2, reg_div="kl", regm_div="kl"):
    m, n = M_sparse.shape
    assert m == n, "Only square multidiagonal matrices supported"
    assert len(a) == len(b) == m, "Marginals must match matrix dimensions"
    
    offsets = M_sparse.offsets
    data = M_sparse.data
    
    if issparse(c):
        c_diag = c.diagonal()
    else:
        c_diag = np.asarray(c).reshape(-1)
    
    def sparse_dot(G_data, G_offsets):
        """Efficient dot product between two multidiagonal matrices"""
        total = 0.0
        for i, offset1 in enumerate(offsets):
            for j, offset2 in enumerate(G_offsets):
                if offset1 == offset2:
                    min_len = min(len(data[i]), len(G_data[j]))
                    total += np.sum(data[i][:min_len] * G_data[j][:min_len])
        return total
    
    def sparse_row_sum(G_data, G_offsets):
        """Row sums for multidiagonal matrix"""
        row_sums = np.zeros(m)
        for j, offset in enumerate(G_offsets):
            diag = G_data[j]
            if offset == 0:  # Main diagonal
                row_sums += diag
            elif offset < 0:  # Lower diagonal
                row_sums[-offset:] += diag[:m+offset]
            else:  # Upper diagonal
                row_sums[:-offset] += diag[:m-offset]
        return row_sums
    
    def sparse_col_sum(G_data, G_offsets):
        """Column sums for multidiagonal matrix"""
        return sparse_row_sum(G_data, -np.array(G_offsets))
    
    def reg_kl(G_data, G_offsets):
        """KL divergence regularization for sparse matrix"""
        kl_sum = 0.0
        total_mass = 0.0
        
        for j, offset in enumerate(G_offsets):
            diag = G_data[j]
            if offset == 0:  # Main diagonal
                idx = np.arange(m)
            elif offset < 0:  # Lower diagonal
                idx = np.arange(-offset, m)
            else:  # Upper diagonal
                idx = np.arange(0, m-offset)
            
            c_values = c_diag[idx]
            with np.errstate(divide='ignore', invalid='ignore'):
                kl_terms = diag[:len(idx)] * np.log(diag[:len(idx)] / c_values + 1e-16)
                kl_terms = np.nan_to_num(kl_terms, nan=0.0, posinf=0.0, neginf=0.0)
                kl_sum += np.sum(kl_terms)
                total_mass += np.sum(diag[:len(idx)])
        
        return kl_sum - total_mass + np.sum(c_diag)
    
    def grad_kl(G_data, G_offsets):
        """Gradient of KL divergence for sparse matrix"""
        grad_data = []
        for j, offset in enumerate(G_offsets):
            diag = G_data[j]
            if offset == 0:  # Main diagonal
                idx = np.arange(m)
            elif offset < 0:  # Lower diagonal
                idx = np.arange(-offset, m)
            else:  # Upper diagonal
                idx = np.arange(0, m-offset)
            
            c_values = c_diag[idx]
            grad_diag = np.log(diag[:len(idx)] / c_values + 1e-16)
            grad_diag = np.nan_to_num(grad_diag, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Pad to original diagonal length
            if offset < 0:
                grad_diag = np.pad(grad_diag, (0, -offset), 'constant')
            elif offset > 0:
                grad_diag = np.pad(grad_diag, (offset, 0), 'constant')
            
            grad_data.append(grad_diag)
        
        return grad_data
    
    def marg_tv(G_data, G_offsets):
        """TV marginal penalty for sparse matrix"""
        row_sums = sparse_row_sum(G_data, G_offsets)
        col_sums = sparse_col_sum(G_data, G_offsets)
        return reg_m1 * np.sum(np.abs(row_sums - a)) + reg_m2 * np.sum(np.abs(col_sums - b))
    
    def grad_marg_tv(G_data, G_offsets):
        """Gradient of TV marginal penalty"""
        row_sums = sparse_row_sum(G_data, G_offsets)
        col_sums = sparse_col_sum(G_data, G_offsets)
        
        grad_data = []
        for offset in G_offsets:
            if offset == 0:  # Main diagonal
                grad_diag = reg_m1 * np.sign(row_sums - a) + reg_m2 * np.sign(col_sums - b)
            elif offset < 0:  # Lower diagonal
                grad_diag = reg_m1 * np.sign(row_sums[-offset:] - a[-offset:])
                grad_diag = np.pad(grad_diag, (0, -offset), 'constant')
            else:  # Upper diagonal
                grad_diag = reg_m2 * np.sign(col_sums[offset:] - b[offset:])
                grad_diag = np.pad(grad_diag, (offset, 0), 'constant')
            grad_data.append(grad_diag)
        
        return grad_data
    
    def _func(G_flat):
        """Combined loss function and gradient for sparse optimization"""
        # Reconstruct sparse matrix from flattened representation
        G_data = []
        ptr = 0
        for offset in offsets:
            diag_len = m - abs(offset)
            G_data.append(G_flat[ptr:ptr+diag_len])
            ptr += diag_len
        
        # Compute loss
        transport_cost = sparse_dot(G_data, offsets)
        marginal_penalty = marg_tv(G_data, offsets)
        val = transport_cost + marginal_penalty
        
        if reg > 0:
            if reg_div == "kl":
                kl_penalty = reg * reg_kl(G_data, offsets)
                val += kl_penalty
        
        # Compute gradient
        grad_marg = grad_marg_tv(G_data, offsets)
        
        # Combine gradients
        grad_flat = np.zeros_like(G_flat)
        ptr = 0
        for i, offset in enumerate(offsets):
            diag_len = m - abs(offset)
            
            # Transport cost gradient (M)
            grad_flat[ptr:ptr+diag_len] += data[i][:diag_len]
            
            # Marginal penalty gradient
            grad_flat[ptr:ptr+diag_len] += grad_marg[i][:diag_len]
            
            # KL divergence gradient if needed
            if reg > 0 and reg_div == "kl":
                grad_kl_diag = grad_kl(G_data, offsets)[i][:diag_len]
                grad_flat[ptr:ptr+diag_len] += reg * grad_kl_diag
            
            ptr += diag_len
        
        return val, grad_flat
    
    return _func

    # def _func(G):
    #     G = G.reshape((m, n))

    #     # compute loss
    #     val = np.sum(G * M) + regm_fun(G)
    #     if reg > 0:
    #         val = val + reg * reg_fun(G)
    #     # compute gradient
    #     grad = M + grad_regm_fun(G)
    #     if reg > 0:
    #         grad = grad + reg * grad_reg_fun(G)

    #     return val, grad.ravel()

    # return _func


# def lbfgsb_unbalanced(
#     a,
#     b,
#     M,
#     reg,
#     reg_m,
#     c=None,
#     reg_div="kl",
#     regm_div="kl",
#     G0=None,
#     numItermax=1000,
#     stopThr=1e-15,
#     method="L-BFGS-B",
#     verbose=False,
#     log=False,
# ):
#     r"""
#     Solve the unbalanced optimal transport problem and return the OT plan using L-BFGS-B algorithm.
#     The function solves the following optimization problem:

#     .. math::
#         W = \arg \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
#         \mathrm{reg} \mathrm{div}(\gamma, \mathbf{c}) +
#         \mathrm{reg_{m1}} \cdot \mathrm{div_m}(\gamma \mathbf{1}, \mathbf{a}) +
#         \mathrm{reg_{m2}} \cdot \mathrm{div_m}(\gamma^T \mathbf{1}, \mathbf{b})

#         s.t.
#              \gamma \geq 0

#     where:

#     - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
#     - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
#     - :math:`\mathbf{c}` is a reference distribution for the regularization
#     - :math:`\mathrm{div_m}` is a divergence, either Kullback-Leibler divergence,
#     or half-squared :math:`\ell_2` divergence, or Total variation
#     - :math:`\mathrm{div}` is a divergence, either Kullback-Leibler divergence,
#     or half-squared :math:`\ell_2` divergence

#     .. note:: This function is backend-compatible and will work on arrays
#         from all compatible backends. First, it converts all arrays into Numpy arrays,
#         then uses the L-BFGS-B algorithm from scipy.optimize to solve the optimization problem.

#     Parameters
#     ----------
#     a : array-like (dim_a,)
#         Unnormalized histogram of dimension `dim_a`
#         If `a` is an empty list or array ([]),
#         then `a` is set to uniform distribution.
#     b : array-like (dim_b,)
#         Unnormalized histogram of dimension `dim_b`
#         If `b` is an empty list or array ([]),
#         then `b` is set to uniform distribution.
#     M : array-like (dim_a, dim_b)
#         loss matrix
#     reg: float
#         regularization term >=0
#     c : array-like (dim_a, dim_b), optional (default = None)
#         Reference measure for the regularization.
#         If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
#     reg_m: float or indexable object of length 1 or 2
#         Marginal relaxation term: nonnegative (including 0) but cannot be infinity.
#         If :math:`\mathrm{reg_{m}}` is a scalar or an indexable object of length 1,
#         then the same :math:`\mathrm{reg_{m}}` is applied to both marginal relaxations.
#         If :math:`\mathrm{reg_{m}}` is an array, it must be a Numpy array.
#     reg_div: string, optional
#         Divergence used for regularization.
#         Can take three values: 'entropy' (negative entropy), or
#         'kl' (Kullback-Leibler) or 'l2' (half-squared) or a tuple
#         of two calable functions returning the reg term and its derivative.
#         Note that the callable functions should be able to handle Numpy arrays
#         and not tesors from the backend
#     regm_div: string, optional
#         Divergence to quantify the difference between the marginals.
#         Can take three values: 'kl' (Kullback-Leibler) or 'l2' (half-squared) or 'tv' (Total Variation)
#     G0: array-like (dim_a, dim_b)
#         Initialization of the transport matrix
#     numItermax : int, optional
#         Max number of iterations
#     stopThr : float, optional
#         Stop threshold on error (> 0)
#     verbose : bool, optional
#         Print information along iterations
#     log : bool, optional
#         record log if True

#     Returns
#     -------
#     gamma : (dim_a, dim_b) array-like
#             Optimal transportation matrix for the given parameters
#     log : dict
#         log dictionary returned only if `log` is `True`

#     Examples
#     --------
#     >>> import ot
#     >>> import numpy as np
#     >>> a=[.5, .5]
#     >>> b=[.5, .5]
#     >>> M=[[1., 36.],[9., 4.]]
#     >>> np.round(ot.unbalanced.lbfgsb_unbalanced(a, b, M, reg=0, reg_m=5, reg_div='kl', regm_div='kl'), 2)
#     array([[0.45, 0.  ],
#            [0.  , 0.34]])
#     >>> np.round(ot.unbalanced.lbfgsb_unbalanced(a, b, M, reg=0, reg_m=5, reg_div='l2', regm_div='l2'), 2)
#     array([[0.4, 0. ],
#            [0. , 0.1]])

#     References
#     ----------
#     .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
#         Unbalanced optimal transport through non-negative penalized
#         linear regression. NeurIPS.
#     See Also
#     --------
#     ot.lp.emd2 : Unregularized OT loss
#     ot.unbalanced.sinkhorn_unbalanced2 : Entropic regularized OT loss
#     """

#     # wrap the callable function to handle numpy arrays
#     # if isinstance(reg_div, tuple):
#     #     f0, df0 = reg_div
#     #     try:
#     #         f0(G0)
#     #         df0(G0)
#     #     except BaseException:
#     #         warnings.warn(
#     #             "The callable functions should be able to handle numpy arrays, wrapper ar added to handle this which comes with overhead"
#     #         )

#     #         def f(x):
#     #             return nx.to_numpy(f0(nx.from_numpy(x, type_as=M0)))

#     #         def df(x):
#     #             return nx.to_numpy(df0(nx.from_numpy(x, type_as=M0)))

#     #         reg_div = (f, df)

#     # else:
#     #     reg_div = reg_div.lower()
#     #     if reg_div not in ["entropy", "kl", "l2"]:
#     #         raise ValueError(
#     #             "Unknown reg_div = {}. Must be either 'entropy', 'kl' or 'l2', or a tuple".format(
#     #                 reg_div
#     #             )
#     #         )

#     # regm_div = regm_div.lower()
#     # if regm_div not in ["kl", "l2", "tv"]:
#     #     raise ValueError(
#     #         "Unknown regm_div = {}. Must be either 'kl', 'l2' or 'tv'".format(regm_div)
#     #     )

#     reg_m1, reg_m2 = get_parameter_pair(reg_m)

#     M, a, b = list_to_array(M, a, b)
#     nx = get_backend(M, a, b)
#     M0 = M

#     dim_a, dim_b = M.shape

#     if len(a) == 0:
#         a = nx.ones(dim_a, type_as=M) / dim_a
#     if len(b) == 0:
#         b = nx.ones(dim_b, type_as=M) / dim_b

#     # convert to numpy
#     a, b, M, reg_m1, reg_m2, reg = nx.to_numpy(a, b, M, reg_m1, reg_m2, reg)
#     G0 = a[:, None] * b[None, :] if G0 is None else nx.to_numpy(G0)
#     c = a[:, None] * b[None, :] if c is None else nx.to_numpy(c)

#     _func = _get_loss_unbalanced(a, b, c, M, reg, reg_m1, reg_m2, reg_div, regm_div)

#     res = minimize(
#         _func,
#         G0.ravel(),
#         method=method,
#         jac=True,
#         bounds=Bounds(0, np.inf),
#         tol=stopThr,
#         options=dict(maxiter=numItermax, disp=verbose),
#     )

#     G = nx.from_numpy(res.x.reshape(M.shape), type_as=M0)

#     if log:
#         log = {"cost": nx.sum(G * M), "res": res}
#         log["total_cost"] = nx.from_numpy(res.fun, type_as=M0)
#         return G, log
#     else:
#         return G