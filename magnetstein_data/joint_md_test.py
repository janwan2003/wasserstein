import numpy as np
from typing import List
import sys
import os

sys.path.append(os.path.abspath(".."))
from wasserstein import NMRSpectrum
from SimpleDecent.test_utils import (
    signif_features,
    Spectrum,
    multidiagonal_cost,
    reg_distribiution,
    warmstart_sparse,
    UtilsSparse,
    get_nus,
)


def load_data():
    components_names = ["Pinene", "Benzyl benzoate"]

    protons_list = [16, 12]

    experiment = "experiment_6_miniperfumes/"

    filename = experiment + "preprocessed_mix.csv"
    mix = np.loadtxt(filename, delimiter=",")

    how_many_components = len(components_names)
    names = ["comp" + str(i) for i in range(how_many_components)]

    files_with_components = [
        experiment + "preprocessed_comp0.csv",
        experiment + "preprocessed_comp1.csv",
    ]
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


def test_joint_md(
    spectra,
    mix,
    N,
    C,
    reg,
    reg_m1,
    reg_m2,
    eta_G=1e-3,  # tym
    eta_p=5 * 1e-4,  # i tym trzeba sie pobawic nie recze za defaulty
    gamma=1.0,
    max_iter=1000,
    tol=1e-6,
    patience=50,
):
    spectra = [signif_features(spectrum, N) for spectrum in spectra]

    ratio = np.ones(len(spectra)) / len(spectra)

    unique_v = sorted({v for spectrum in spectra for v, _ in spectrum.confs})
    total_unique_v = len(unique_v)

    mix_og = signif_features(mix, total_unique_v)

    nus = get_nus([si.confs for si in spectra])

    a = np.array([p for _, p in mix_og.confs])
    b = sum(nu_i * p_i for nu_i, p_i in zip(nus, ratio))

    v1 = np.array([v for v, _ in mix_og.confs])
    v2 = unique_v

    print(len(v1), len(v2))

    M = multidiagonal_cost(v1, v2, C)
    warmstart = warmstart_sparse(a, b, C)
    c = reg_distribiution(total_unique_v, C)

    sparse = UtilsSparse(a, b, c, warmstart, M, reg, reg_m1, reg_m2)
    confs = [spectrum.confs for spectrum in spectra]
    G, p = sparse.joint_md(confs, eta_G, eta_p, max_iter, tol, gamma, patience)

    print("final p: ", p)


spectra, mix = load_data()
test_joint_md(spectra, mix, 1000, 20, 1.5, 230, 115, 1e-3, 5 * 1e-4, 1.0, 1000)
