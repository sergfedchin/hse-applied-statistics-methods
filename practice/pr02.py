from pr01 import generate_n_random_numbers
import matplotlib.pyplot as plt
import math
import scipy.stats as sps


def calculate_c_star(x: list):
    return sum(x) / len(x)


def task_1():
    distros_1 = ["uniform_-1_1", "norm_0_1", "cauchy"]
    powers = list(range(1, 8))
    ns_1 = [10 ** d for d in powers]
    c_stars = {d: [calculate_c_star(generate_n_random_numbers(n, d))
                   for n in ns_1] for d in distros_1}

    for d in distros_1:
        plt.plot(powers, c_stars[d])

    plt.xlabel('log(n)')
    plt.ylabel('c *')
    plt.show()


def task_2():
    P = 0.95
    distros = ["uniform_-1_1", "norm_0_1", "2xuniform_-1_1"]
    sigma_squared = {"uniform_-1_1": 1 / 3,
                     "norm_0_1": 1,
                     "2xuniform_-1_1": 2 / 3}
    quantil = sps.norm(loc=0, scale=1).ppf((1 + P) / 2)

    ns = [10, 100, 1000, 5000]
    samples = {d: [[generate_n_random_numbers(n, d) for _ in range(10**4)]
                   for n in ns] for d in distros}

    _, axis = plt.subplots(3, 1)

    for i, d in enumerate(distros):
        results = []
        for n_id, n in enumerate(ns):
            term = quantil * math.sqrt(sigma_squared[d]) / math.sqrt(n)
            results.append([(calculate_c_star(sample) - term <= 0 <=
                             calculate_c_star(sample) + term) for sample in
                            samples[d][n_id]].count(True) / 10 ** 4)
        print(d, results)
        axis[i].plot(range(len(results)), results)
    plt.show()


if __name__ == "__main__":
    task_1()
    task_2()
