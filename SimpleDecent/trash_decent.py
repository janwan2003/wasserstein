from typing import List
import sys
import os
import numpy as np
from tqdm import trange
import math
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))
from nmr_spectrum import NMRSpectrum
from spectrum import Spectrum


def load_data():
    components_names = ["Pinene", "Benzyl benzoate"]

    protons_list = [16, 12]

    filename = "preprocessed_mix.csv"
    mix = np.loadtxt(filename, delimiter=",")
    # If you are using file exported from Mnova, comment line above and uncomment line below.
    # mix = np.loadtxt(filename, delimiter='\t', usecols=[0,1])

    how_many_components = len(components_names)
    names = ["comp" + str(i) for i in range(how_many_components)]

    files_with_components = ["preprocessed_comp0.csv", "preprocessed_comp1.csv"]
    spectra = []
    for i in range(how_many_components):
        filename = files_with_components[i]
        spectra.append(np.loadtxt(filename, delimiter=","))
        # If you are using file exported from Mnova, comment line above and uncomment line below.
        # spectra.append(np.loadtxt(filename, delimiter='\t', usecols=[0,1]))

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
    plt.title("Mixture")
    mix.plot(profile=True)
    for i, sp in enumerate(spectra):
        plt.title("Component " + str(i))
        sp.plot(profile=True)
    print(spectra[0].confs[:10])


def WSDistanceMoves(first, second):
    """Return the optimal transport plan between self and other."""
    try:
        ii = 0
        leftoverprob = second.confs[0][1]
        for mass, prob in first.confs:
            while leftoverprob <= prob:
                yield (second.confs[ii][0], mass, leftoverprob)
                prob -= leftoverprob
                ii += 1
                leftoverprob = second.confs[ii][1]
            yield (second.confs[ii][0], mass, prob)
            leftoverprob -= prob
    except IndexError:
        return


def WSDistance(first, second, n_features=None):
    """
    Compute Wasserstein distance using only n most significant features.
    If n_features is None, use all features.
    """
    if n_features is not None and n_features < len(first.confs) + len(second.confs):
        first, second = filter_significant_features(first, second, n_features)

    if not np.isclose(sum(x[1] for x in first.confs), 1.0):
        raise ValueError("Self is not normalized.")
    if not np.isclose(sum(x[1] for x in second.confs), 1.0):
        raise ValueError("Other is not normalized.")

    return math.fsum(abs(x[0] - x[1]) * x[2] for x in WSDistanceMoves(first, second))


def WSGradient(source, target, n_features=None, epsilon=1e-5):
    """
    Compute numerical gradient of WSDistance with respect to mz values of `source`
    using only n most significant features.
    """
    if n_features is not None and n_features < len(source.confs) + len(target.confs):
        source, target = filter_significant_features(source, target, n_features)

    source.normalize()
    target.normalize()

    mzs = np.array([mz for mz, inten in source.confs])
    intensities = np.array([inten for mz, inten in source.confs])
    grad = np.zeros(len(mzs))

    for i in trange(len(mzs), desc="Computing WSDistance gradient"):
        mz_plus = mzs.copy()
        mz_minus = mzs.copy()
        mz_plus[i] += epsilon
        mz_minus[i] -= epsilon

        source_plus = Spectrum(confs=list(zip(mz_plus, intensities)))
        source_minus = Spectrum(confs=list(zip(mz_minus, intensities)))

        source_plus.normalize()
        source_minus.normalize()

        dist_plus = WSDistance(target, source_plus)
        dist_minus = WSDistance(target, source_minus)

        grad[i] = (dist_plus - dist_minus) / (2 * epsilon)

    return grad


def calculate_gradient(mix, comp0, comp1, p, epsilon=1e-5):
    delta = epsilon

    p_plus = np.array([p[0] + delta, p[1] - delta])
    p_minus = np.array([p[0] - delta, p[1] + delta])

    p_plus = np.clip(p_plus, 1e-6, 1)
    p_plus /= p_plus.sum()

    p_minus = np.clip(p_minus, 1e-6, 1)
    p_minus /= p_minus.sum()

    sp_plus = Spectrum.ScalarProduct([comp0, comp1], p_plus)
    sp_minus = Spectrum.ScalarProduct([comp0, comp1], p_minus)

    sp_plus.normalize()
    sp_minus.normalize()

    grad = (WSDistance(mix, sp_plus) - WSDistance(mix, sp_minus)) / (2 * epsilon)

    # Gradient of p[1] is -grad because p[1] = 1 - p[0]
    grad_vec = np.array([grad, -grad])

    return grad_vec


def mirror_descent_two_weights(
    mix, comp0, comp1, learning_rate=1.0, T=100, epsilon=1e-5, lmd=0.95
):
    p = np.array([0.5, 0.5])  # start from uniform mixture
    history = [p.copy()]
    scores = []

    for _ in trange(T, desc="Mirror Descent (2 weights)"):
        # Track score (distance from current estimate to true mixture)
        estimated_mix = Spectrum.ScalarProduct([comp0, comp1], p)
        estimated_mix.normalize()
        ws = WSDistance(mix, estimated_mix)
        scores.append(ws)

        # Compute gradient
        grad_vec = calculate_gradient(mix, comp0, comp1, p, epsilon)

        # Mirror descent update
        w = p * np.exp(-learning_rate * grad_vec)
        p = w / w.sum()
        learning_rate *= lmd

        history.append(p.copy())

    return p, np.array(history), np.array(scores)
