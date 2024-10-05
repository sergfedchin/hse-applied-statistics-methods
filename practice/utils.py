import random
import math


def generate_n_random_numbers(N: int, distribution: str):
    """
    N - size of sample\n
    distribution - [Uniform[0, 1], Uniform[-1, 1], 2xUniform[-1, 1],
    3xUniform[-1, 1], Cauchy(0, 1), Norm(0, 1)]
    """
    if distribution not in ["Uniform[0, 1]", "Uniform[-1, 1]",
                            "2xUniform[-1, 1]", "3xUniform[-1, 1]",
                            "Cauchy(0, 1)", "Norm(0, 1)"]:
        raise ValueError(f"Distribution '{distribution}' is not supported")
    match distribution:
        case "Uniform[0, 1]":
            return [random.uniform(0, 1) for _ in range(N)]
        case "Uniform[-1, 1]":
            return [random.uniform(0, 1) * 2 - 1 for _ in range(N)]
        case "2xUniform[-1, 1]":
            t = generate_n_random_numbers(2 * N, "Uniform[-1, 1]")
            return [t[2 * i] + t[2 * i + 1] for i in range(N)]
        case "3xUniform[-1, 1]":
            t = generate_n_random_numbers(3 * N, "Uniform[-1, 1]")
            return [t[3 * i] + t[3 * i + 1] + t[3 * i + 2] for i in range(N)]
        case "Cauchy(0, 1)":
            t = generate_n_random_numbers(N, "Uniform[0, 1]")
            return list(map(lambda x: math.tan(math.pi * (x - 0.5)), t))
        case "Norm(0, 1)":
            return [random.normalvariate(0, 1) for _ in range(N)]
        case _:
            return []
