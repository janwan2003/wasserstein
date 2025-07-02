import sys
import os

sys.path.append(os.path.abspath(".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wasserstein import NMRSpectrum, Spectrum
from SimpleDecent.test_utils import (
    signif_features,
    Spectrum,
    multidiagonal_cost,
    reg_distribiution,
    warmstart_sparse,
    UtilsSparse,
    dia_matrix,
)

def signif_features(spectrum, n_features):
    spectrum_confs = sorted(spectrum.confs, key=lambda x: x[1], reverse=True)[
        :n_features
    ]
    spectrum_signif = NMRSpectrum(confs=spectrum_confs, protons=spectrum.protons)
    spectrum_signif.normalize()
    return spectrum_signif

def construct_data(mix, spectra, N, C, ratio):
    mix_aprox = Spectrum.ScalarProduct(spectra, ratio)
    mix_aprox.normalize()

    a = np.array([p for _, p in mix.confs])
    b = np.array([p for _, p in mix_aprox.confs])

    v1 = np.array([v for v, _ in mix.confs])
    v2 = np.array([v for v, _ in mix_aprox.confs])

    M = multidiagonal_cost(v1, v2, C)

    c = reg_distribiution(2 * N, C)

    return a, b, c, M

def derivative_estimate(mix, spectra, N, C, G0, reg, regm1, regm2, ratio, eps, max_iter):
    p_signs = np.zeros(len(spectra) - 1)
    for i in range(len(spectra) - 1):
        ratio[i] += eps
        ap, bp, c, M = construct_data(mix, spectra, N, C, ratio)
        ratio[i] -= 2*eps
        am, bm, c, M = construct_data(mix, spectra, N, C, ratio)
        ratio[i] += eps

        sparse_p = UtilsSparse(ap, bp, c, G0, M, reg, regm1, regm2)
        sparse_m = UtilsSparse(am, bm, c, G0, M, reg, regm1, regm2)
        _, logp = sparse_p.mirror_descent_unbalanced(numItermax=max_iter)
        _, logm = sparse_m.mirror_descent_unbalanced(numItermax=max_iter)

        p_signs[i] = np.sign(logp["total_cost"] - logm["total_cost"])
    return p_signs

def binsearch_p(mix, spectra, exp_num, N, C, reg=1.5, regm1=230, regm2=115, max_iter=1000, iter=10, eps=1*1e-2):
    if n_features is not None:
        mix = signif_features(mix, 2 * N)
        spectra = [signif_features(s, N) for s in spectra]

    ratio = np.array(
        ground_truth_molar_proportions[exp_num]
    )
    a, b, c, M = construct_data(mix, spectra, N, C, ratio)
    _G0 = warmstart_sparse(a, b, C)
    sparse = UtilsSparse(a, b, c, _G0, M, reg, regm1, regm2)
    _G0, _ = sparse.mirror_descent_unbalanced(numItermax=max_iter)
    G0 = dia_matrix((_G0, sparse.offsets), shape=(sparse.n, sparse.m), dtype=np.float64)

    ratio = np.array(1 / len(spectra) * np.ones(len(spectra)))

    step = 1 / (len(spectra) * len(spectra))

    for _ in range(iter):
        p_signs = derivative_estimate(mix, spectra, N, C, G0, reg, regm1, regm2, ratio, eps, max_iter)
        plus_count = np.sum(p_signs > 0)
        minus_count = np.sum(p_signs < 0)
        for sign in range(len(p_signs)):
            if p_signs[sign] < 0:
                ratio[sign] += step * minus_count / len(p_signs)
            elif p_signs[sign] > 0:
                ratio[sign] -= step * plus_count / len(p_signs)
        
        ratio[-1] = 1 - np.sum(ratio[:-1])
        step /= len(spectra)

    return ratio


components_dictionary = {
    "experiment_1": ["Pinene", "Benzyl benzoate"],
    "experiment_2": ["Pinene", "Limonene"],
    "experiment_7": ["Benzyl benzoate", "m Anisaldehyde"],
    "experiment_3": [
        "Isopropyl myristate",
        "Benzyl benzoate",
        "Alpha pinene",
        "Limonene",
    ],
    "experiment_6": ["Pinene", "Benzyl benzoate"],
    "experiment_5": [
        "Lactate",
        "Alanine",
        "Creatine",
        "Creatinine",
        "Choline chloride",
    ],
    "experiment_9": [
        "Lactate",
        "Alanine",
        "Creatine",
        "Creatinine",
        "Choline chloride",
    ],
    "experiment_4": [
        "Lactate",
        "Alanine",
        "Creatine",
        "Creatinine",
        "Choline chloride",
    ],
    "experiment_8": ["Benzyl benzoate", "m Anisaldehyde"],
    "experiment_10": ["Leucine", "Isoleucine", "Valine"],
    "experiment_11": ["Leucine", "Isoleucine", "Valine"],
}

protons_dictionary = {
    "experiment_1": [16, 12],
    "experiment_2": [16, 16],
    "experiment_7": [12, 8],
    "experiment_3": [34, 12, 16, 16],
    "experiment_6": [16, 12],
    "experiment_5": [4, 4, 5, 5, 13],
    "experiment_9": [4, 4, 5, 5, 13],
    "experiment_4": [4, 4, 5, 5, 13],
    "experiment_8": [12, 8],
    "experiment_10": [10, 10, 8],
    "experiment_11": [10, 10, 8],
}

# comment out if want to run only a subset of experiments
experiments_folders = {
    "experiment_1": "experiment_1_intensity_difference",
    "experiment_6": "experiment_6_miniperfumes",
    # "experiment_5": "experiment_5_metabolites",
    "experiment_7": "experiment_7_overlapping_and_intensity_difference",
    # "experiment_9": "experiment_9_and_4_shim",
    # "experiment_4": "experiment_9_and_4_shim",
    "experiment_8": "experiment_8_different_solvents",
    # "experiment_3": "experiment_3_perfumes_and_absent_components",
    "experiment_2": "experiment_2_overlapping",
    # "experiment_10": "experiment_10_bcaa",
    # "experiment_11": "experiment_11_real_food_product",
}

# parameters
n_features_list = np.linspace(200, 70400, num=20, dtype=int)
# n_features_list = np.logspace(np.log2(200), np.log2(70400), num=20, dtype=int, base=2)

print(f"Using n_features_list: {n_features_list}")
# add variant index and ground truth proportions
variant = 3
ground_truth_molar_proportions = {
    "experiment_1": [0.09088457406472417, 0.9091154259352758],
    "experiment_2": [0.505, 0.495],
    "experiment_7": [0.8403875207510383, 0.1596124792489616],
    "experiment_3": [
        0.7264578344443725,
        0.10578603326645526,
        0.081968804608116,
        0.08578732768105625,
    ],
    "experiment_6": [0.3865, 0.6135],
    "experiment_5": [0.3022, 0.2240, 0.1253, 0.2028, 0.1457],
    "experiment_9": [0.3022, 0.2240, 0.1253, 0.2028, 0.1457],
    "experiment_4": [0.3022, 0.2240, 0.1253, 0.2028, 0.1457],
    "experiment_8": [0.3702, 0.6298],
    "experiment_10": [
        0.3401,
        0.3299,
        0.3299,  # variant 1
    ]
    if variant == 0
    else [
        0.2525,
        0.2475,
        0.5,  # variant 2
    ]
    if variant == 1
    else [
        0.2538,
        0.4949,
        0.2513,  # variant 3
    ]
    if variant == 2
    else [
        0.2075,
        0.3942,
        0.3983,  # variant 4
    ],
    "experiment_11": [0.4855, 0.2427, 0.2718],
}

all_results = []
for n_features in n_features_list:
    print(f"Running experiments with n_features={n_features}")
    results = []
    for exp_num, folder in experiments_folders.items():
        num = exp_num.split("_")[1]

        # load mix (special cases 4,9,10)
        if exp_num == "experiment_10":
            mix_file = f"preprocessed_mix_variant_{variant + 1}.csv"
        elif num in ["9", "4"]:
            mix_file = f"preprocessed_exp{num}_mix.csv"
        else:
            mix_file = "preprocessed_mix.csv"
        mix_path = os.path.join(folder, mix_file)
        mix_arr = np.loadtxt(mix_path, delimiter=",")
        mix_spec = NMRSpectrum(confs=list(zip(mix_arr[:, 0], mix_arr[:, 1])))

        # load component spectra (special cases 4,9,10)
        comps = []
        for i in range(len(components_dictionary[exp_num])):
            if exp_num == "experiment_10":
                comp_file = f"preprocessed_variant_{variant + 1}_comp{i}.csv"
            elif num in ["9", "4"]:
                comp_file = f"preprocessed_exp{num}_comp{i}.csv"
            else:
                comp_file = f"preprocessed_comp{i}.csv"
            data = np.loadtxt(os.path.join(folder, comp_file), delimiter=",")
            comps.append(
                NMRSpectrum(
                    confs=list(zip(data[:, 0], data[:, 1])),
                    protons=protons_dictionary[exp_num][i],
                )
            )

        # estimate
        p = binsearch_p(
            mix=mix_spec,
            spectra=comps,
            exp_num=exp_num,
            N=n_features,
            C=20,
            reg=1.5,
            regm1=230,
            regm2=115,
            max_iter=1000,
            iter=10,
        )
        print(f"Estimated proportions for {exp_num}: {p}")
        break
    break
