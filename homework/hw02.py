import matplotlib.pyplot as plt
import scipy.stats as sps
from scipy.special import erfinv
import numpy as np
import math
import sys
import os
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '../practice'))
from pr01 import generate_n_random_numbers


P = 0.95
PERCENTILE = ((1 - P) / 2) * 100
ANALYTICAL_EXPECTATION = 0
ITERATIONS = 10 ** 3
BOOTSTRAP_SIZE = 1000
NS = [10, 100, 1000]


def calculate_c_star(x: list):
    return sum(x) / len(x)


def traditional_confidence_interval(sample: list,
                                    quantil: float,
                                    sigma_squared: float) -> tuple:
    term = quantil * math.sqrt(sigma_squared) / math.sqrt(len(sample))
    c_star = calculate_c_star(sample)
    return (c_star - term, c_star + term)


def bootstrap_confidence_interval(sample: list) -> tuple:
    bootstrap_samples = bootstrap(sample)
    bootstrap_means = [np.mean(sample) for sample in bootstrap_samples]
    return (np.percentile(bootstrap_means, PERCENTILE),
            np.percentile(bootstrap_means, 100 - PERCENTILE))


def bootstrap(data: list):
    return [np.random.choice(data, size=len(data), replace=True)
            for _ in range(BOOTSTRAP_SIZE)]


z_score = np.sqrt(2) * erfinv(P)


def jackknife_confidence_interval(sample: list) -> tuple:
    sample_mean = np.mean(sample)
    jack_means = np.apply_along_axis(np.mean, 1, jackknife(sample))
    mean_jack_means = np.mean(jack_means)
    # jackknife bias
    n = len(sample)
    bias = (n - 1) * (mean_jack_means - sample_mean)
    # jackknife standard error
    terms = np.apply_along_axis(lambda x: pow(x - mean_jack_means, 2), 0, jack_means)
    std_err = np.sqrt((n - 1) * np.mean(terms))
    # bias-corrected "jackknifed estimate"
    estimate = sample_mean - bias
    return (estimate - std_err * z_score,
            estimate + std_err * z_score)


def jackknife(data: list):
    return [np.delete(data, i) for i in range(len(data))]


def task():
    distros = ["Uniform[-1, 1]", "Norm(0, 1)", "2xUniform[-1, 1]"]
    sigma_squared = {"Uniform[-1, 1]": 1 / 3,
                     "Norm(0, 1)": 1,
                     "2xUniform[-1, 1]": 2 / 3}
    quantil = sps.norm(loc=0, scale=1).ppf((1 + P) / 2)

    print("Start generating samples...")
    samples = {d: [[generate_n_random_numbers(n, d) for _ in range(ITERATIONS)]
                   for n in NS] for d in tqdm(distros)}
    print("Samples generated.")

    _, axis = plt.subplots(3, 3)
    for i, d in enumerate(distros):
        results = []
        results_bs = []
        results_jk = []
        for n_id, n in enumerate(NS):
            hits_counter = 0
            hits_counter_bs = 0
            hits_counter_jk = 0
            print(f"Distribution {d}, samples size = {n}")
            for sample in tqdm(samples[d][n_id]):
                d_left, d_right = traditional_confidence_interval(sample, quantil, sigma_squared[d])
                hits_counter += d_left <= ANALYTICAL_EXPECTATION <= d_right

                d_left_bs, d_right_bs = bootstrap_confidence_interval(sample)
                hits_counter_bs += d_left_bs <= ANALYTICAL_EXPECTATION <= d_right_bs

                d_left_jk, d_right_jk = jackknife_confidence_interval(sample)
                hits_counter_jk += d_left_jk <= ANALYTICAL_EXPECTATION <= d_right_jk

            results.append(hits_counter / ITERATIONS)
            results_bs.append(hits_counter_bs / ITERATIONS)
            results_jk.append(hits_counter_jk / ITERATIONS)

        print(f"Results for distribution {d}\n"
              f"{results} (tranditional)\n"
              f"{results_bs} (bootstrap)\n"
              f"{results_jk} (jackknife)\n", '-' * 40, sep='')
        axis[i][0].plot(range(len(results)), results)
        axis[i][1].plot(range(len(results_bs)), results_bs)
        axis[i][2].plot(range(len(results_jk)), results_jk)
    plt.show()


if __name__ == "__main__":
    task()
