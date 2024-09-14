import random
import math
from matplotlib import pyplot as plt
# from scipy.integrate import quad


def generate_n_random_numbers(N: int, distribution: str):
    """
    N - size of sample
    distribution - ["uniform_0_1", "uniform_-1_1", "2xuniform_-1_1", "3xuniform_-1_1", "cauchy", norm_0_1]
    """
    match distribution:
        case "uniform_0_1":
            return [random.uniform(0, 1) for _ in range(N)]
        case "uniform_-1_1":
            return [random.uniform(0, 1) * 2 - 1 for _ in range(N)]
        case "2xuniform_-1_1":
            t = generate_n_random_numbers(2 * N, "uniform_-1_1")
            return [t[2 * i] + t[2 * i + 1] for i in range(N)]
        case "3xuniform_-1_1":
            t = generate_n_random_numbers(3 * N, "uniform_-1_1")
            return [t[3 * i] + t[3 * i + 1] + t[3 * i + 2] for i in range(N)]
        case "cauchy":
            t = generate_n_random_numbers(N, "uniform_0_1")
            return list(map(lambda x: math.tan(math.pi * (x - 0.5)), t))
        case "norm_0_1":
            return [random.normalvariate(0, 1) for _ in range(N)]
        case _:
            return []


# def charachteristic_function(mu, d, b):
#     return math.exp(-d * abs(mu) - pow(b, 2) * pow(mu, 2) * 0.5)


# def integrand_for_density(t):
#     math.exp(-math.i)

if __name__ == "__main__":
    figure, axis = plt.subplots(2, 2)

    N = int(input("Enter N: "))
    for i, distr in enumerate(["uniform_0_1", "uniform_-1_1", "3xuniform_-1_1", "cauchy"]):
        result = generate_n_random_numbers(N, distr)
        # print(result)
        bins = 500
        if distr == "cauchy":
            axis[i // 2, i % 2].hist(result, bins=bins, density=True, range=(-6, 6))
        else:
            axis[i // 2, i % 2].hist(result, bins=bins, density=True)

    plt.show()
