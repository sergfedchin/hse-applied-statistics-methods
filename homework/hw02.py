import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np
import math
import sys
import os
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '../practice'))
from pr01 import generate_n_random_numbers


P = 0.95
ANALYTICAL_EXPECTATION = 0
ITERATIONS = 10 ** 4


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
    bootstrap_means.sort()
    shift = (1 - P) / 2
    return (np.percentile(bootstrap_means, shift),
            np.percentile(bootstrap_means, 1 - shift))


def bootstrap(data, n_bootstrap_samples=1000):
    return [np.random.choice(data, size=len(data), replace=True)
            for _ in range(n_bootstrap_samples)]


def task_2():
    distros = ["uniform_-1_1", "norm_0_1", "2xuniform_-1_1"]
    sigma_squared = {"uniform_-1_1": 1 / 3,
                     "norm_0_1": 1,
                     "2xuniform_-1_1": 2 / 3}
    quantil = sps.norm(loc=0, scale=1).ppf((1 + P) / 2)

    ns = [10, 100, 1000, 5000]
    print("Start generating samples...")
    samples = {d: [[generate_n_random_numbers(n, d) for _ in range(ITERATIONS)]
                   for n in ns] for d in distros}
    print("Samples generated.")
    _, axis = plt.subplots(3, 1)
    for i, d in enumerate(distros):
        results = []
        for n_id, n in enumerate(ns):
            hits_counter = 0
            print(f"Distribution {d}, samples size = {n}")
            for sample in tqdm(samples[d][n_id]):
                # d_left, d_right = traditional_confidence_interval(sample, quantil, sigma_squared[d])
                d_left, d_right = bootstrap_confidence_interval(sample)
                hits_counter += d_left <= ANALYTICAL_EXPECTATION <= d_right
            results.append(hits_counter / ITERATIONS)
        print(d, results)
        axis[i].plot(range(len(results)), results)
    plt.show()


if __name__ == "__main__":
    task_2()
