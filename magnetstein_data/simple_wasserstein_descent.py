import torch
from wasserstein import NMRSpectrum


def numpy_to_torch_tensor(pairs):
    vals = torch.tensor([float(v) for v, _ in pairs], dtype=torch.float64)
    probs = torch.tensor([float(p) for _, p in pairs], dtype=torch.float64)
    return {
        "values": vals.clone().detach().requires_grad_(True),
        "probs": probs.clone().detach().requires_grad_(True),
    }


def weighted_quantile_torch(spectre, quantiles):
    values, probs = spectre["values"], spectre["probs"]
    idx = torch.argsort(values)
    values, probs = values[idx], probs[idx]
    cum = torch.cumsum(probs, dim=0)
    indices = torch.clamp(
        torch.searchsorted(cum, quantiles, right=True), 0, len(values) - 1
    )
    return values[indices]


def wasserstein_distance(mu, nu, p=1):
    cum_mu = torch.cumsum(mu["probs"], dim=0)
    cum_nu = torch.cumsum(nu["probs"], dim=0)
    t = torch.sort(torch.cat([cum_mu, cum_nu]))[0]
    Fmu = weighted_quantile_torch(mu, t)
    Fnu = weighted_quantile_torch(nu, t)
    integral = torch.trapz(torch.abs(Fmu - Fnu) ** p, t)
    return integral ** (1 / p)


def ws_mix(p, spectra_torch, mix_torch):
    # stack values/probs: shape (n_spectra, L)
    values_stack = torch.stack([s["values"] for s in spectra_torch], dim=0)
    probs_stack = torch.stack([s["probs"] for s in spectra_torch], dim=0)
    # compute weighted sum across spectra
    est_vals = (p.unsqueeze(1) * values_stack).sum(dim=0)
    est_pr = (p.unsqueeze(1) * probs_stack).sum(dim=0)
    est_pr = est_pr / est_pr.sum()
    return wasserstein_distance(mix_torch, {"values": est_vals, "probs": est_pr})


def mirror_descent_torch(
    spectra_torch,
    mix_torch,
    learning_rate=1.0,
    gamma=0.99,
    T=1000,
    tol=1e-6,
    patience=5,
):
    n = len(spectra_torch)
    # initialize uniform weights of length n
    p = torch.full((n,), 1.0 / n, dtype=torch.float64, requires_grad=True)
    history, scores = [p.clone()], []
    prev_ws = None
    no_change = 0
    for t in range(T):
        ws = ws_mix(p, spectra_torch, mix_torch)
        curr = ws.item()
        # count consecutive small‐change iterations
        if prev_ws is not None and abs(prev_ws - curr) < tol:
            no_change += 1
        else:
            no_change = 0
        if no_change >= patience:
            break
        prev_ws = curr
        ws.backward()
        lr_t = learning_rate * (gamma**t)
        with torch.no_grad():
            w = p * torch.exp(-lr_t * p.grad)
            p.copy_(w / w.sum())
            p.grad.zero_()
        history.append(p.clone())
        scores.append(curr)
    return p, torch.stack(history), scores


def estimate_proportions_wasserstein(
    mix,
    spectra,
    learning_rate=0.01,
    gamma=0.99,
    T=200,
    tol=1e-6,
    patience=5,
    given=None,
    n_features=None,
):
    # if requested, reduce to top‐N features before torch conversion
    if n_features is not None:
        mix = signif_features(mix, len(spectra) * n_features)
        spectra = [signif_features(s, n_features) for s in spectra]

    # convert spectra + mix
    spectra_torch = [numpy_to_torch_tensor(s.confs) for s in spectra]
    mix_torch = numpy_to_torch_tensor(mix.confs)
    final_p, traj, scores = mirror_descent_torch(
        spectra_torch,
        mix_torch,
        learning_rate=learning_rate,
        gamma=gamma,
        T=T,
        tol=tol,
        patience=patience,
    )
    ws_dist = ws_mix(final_p, spectra_torch, mix_torch).item()
    if given is None:
        # default to uniform if not provided
        given = tuple([1.0 / len(spectra)] * len(spectra))
    ws_dist_given = ws_mix(
        torch.tensor(given, dtype=torch.float64), spectra_torch, mix_torch
    ).item()
    return final_p, traj, scores, ws_dist, ws_dist_given


# new helper: extract top‐N by probability
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
