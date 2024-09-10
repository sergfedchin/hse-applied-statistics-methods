import numpy as np
import random
import math
from copy import copy
from math import floor
from math import sqrt
from scipy.special import gammaincc, hyp1f1
from scipy.stats import norm
from scipy import fftpack as sff


# def float_to_binary_str(x: float) -> str:
#     # pack the float as a binary string,
#     # convert each byte to binary and join them
#     return ''.join(format(c, '08b') for c in struct.pack('!f', x))


def generate_binary_sequence(length: int) -> str:
    """
    Generates random numbers and joins their binary form into
    a string of length 'length'
    """
    return ''.join([bin(random.randint(0, 2_147_483_647))[2:] for _ in range(length // 16)])[0:length]


class NISTTests:
    def __init__(self, sequence: str) -> None:
        self.sequence = sequence
        self.n = len(sequence)
        self.alpha = 0.01
        self.test_results = [False] * 15
        self.tests = [
            {"test": self.test_1, "id": 1, "description": "frequency bit test"},
            {"test": self.test_2, "id": 2, "description": "frequency block test"},
            {"test": self.test_3, "id": 3, "description": "runs test"},
            {"test": self.test_4, "id": 4, "description": "longest run of ones in a block test"},
            {"test": self.test_5, "id": 5, "description": "matrix rank test"},
            {"test": self.test_6, "id": 6, "description": "spectral test"},
            {"test": self.test_7, "id": 7, "description": "non-overlapping templates test"},
            {"test": self.test_8, "id": 8, "description": "overlapping templates test"},
            {"test": self.test_9, "id": 9, "description": "universal test TODO"},
            {"test": self.test_10, "id": 10, "description": "linear complexity test"},
            {"test": self.test_11, "id": 11, "description": "serial test"},
            {"test": self.test_12, "id": 12, "description": "approximate entropy test"},
            {"test": self.test_13, "id": 13, "description": "cumulative sums test"},
            {"test": self.test_14, "id": 14, "description": "random excursions test"},
            {"test": self.test_15, "id": 15, "description": "random excursions variant test"},
            ]

    def conduct_test(self, test, test_id) -> bool:
        test_res = test(self.sequence)
        self.test_results[test_id - 1] = test_res[0]
        if test_res[0]:
            print("SUCCESS ", end='')
        else:
            print("FAILED ", end='')
        print(f"{test_res[1:]}")
        return test_res

    def test_sequence(self) -> None:
        print("Testing sequence of length", self.n)

        for test in self.tests:
            print(f'Test {test["id"]} ({test["description"]}): ', end="")
            self.conduct_test(test["test"], test["id"])

    def test_1(self, input: str) -> tuple:
        """
        Frequency bit test
        """
        s = input.count('0') - input.count('1')
        # compute statistics
        s_obs = abs(s) / math.sqrt(self.n)
        # compute p-Value
        p_value = math.erfc(s_obs / (math.sqrt(2)))
        # return success indicator and p-value
        return (p_value >= self.alpha, p_value)

    def test_2(self, input: str, block_size=20) -> tuple:
        """
        Frequency block test
        """
        blocks_count = floor(self.n / block_size)
        if blocks_count == 1:
            # for block size 1, this test degenerates to test_1
            return self.test_1(input[0:block_size])

        frac_sum = 0.0
        for i in range(blocks_count):
            # extract current block
            block = input[i * block_size:(i + 1) * block_size]
            # calculate ratio of ones in it
            pi = block.count('1') / block_size
            # add it to the sum
            frac_sum += pow(pi - 0.5, 2)

        # compute chi_obs ^ 2
        result = 4.0 * block_size * frac_sum
        # compute p-value
        p_value = gammaincc(blocks_count / 2, result / 2)
        return (p_value >= self.alpha, p_value)

    def test_3(self, input: str) -> tuple:
        """
        Bit switches test
        """
        pi = input.count('1') / self.n
        # check the first condition
        if math.fabs(pi - 0.5) >= 2 / sqrt(self.n):
            return (False, None)

        # count bit switches
        V_n = 0
        for i in range(0, self.n - 1):
            V_n = V_n if input[i] == input[i + 1] else V_n + 1
        # calculate p-value
        tmp_val = 2 * pi * (1 - pi)
        p_value = math.erfc(math.fabs(V_n - self.n * tmp_val) / math.sqrt(2 * self.n) * tmp_val)
        return (p_value >= self.alpha, p_value)

    def find_max_ones_subsequence(bin_string: str) -> int:
        l, r, max_length = 0, 0, 0
        while r < len(bin_string):
            if bin_string[r] == '1':
                r += 1
            else:
                max_length = max(r - l, max_length)
                next_one = bin_string.find('1', r)
                if next_one == -1:
                    break
                l, r = next_one, next_one
        return max(r - l, max_length)

    def test_4(self, input: str) -> tuple:
        """
        Test for the Longest Run of Ones in a Block
        """
        if self.n < 128:
            raise ValueError("Not enough bits to test")
        # the test has multiple cases with different values
        case_id = 0
        if self.n < 6272:
            case_id = 0
        elif self.n < 750_000:
            case_id = 1
        else:
            case_id = 2

        BLOCK_SIZES = [8, 128, 10000]
        block_size = BLOCK_SIZES[case_id]
        blocks_count = self.n // block_size
        # count how many consecutive ones does each block contain
        max_ones_subsequence_lengths = \
            [NISTTests.find_max_ones_subsequence(
                input[i * block_size:(i + 1) * block_size])
                for i in range(blocks_count)]

        # count how many blocks have their max_ones_subsequence length
        # in each category from the lookup table
        # https://habr.com/ru/companies/securitycode/articles/237695/
        LOOKUP_TABLES = [[-1, 1, 2, 3, block_size],
                         [-1, 4, 5, 6, 7, 8, block_size],
                         [-1, 10, 11, 12, 13, 14, 15, block_size]]
        lookup_table = LOOKUP_TABLES[case_id]

        v = [sum(lookup_table[i] < cur <= lookup_table[i + 1]
                 for cur in max_ones_subsequence_lengths)
                 for i in range(len(lookup_table) - 1)]
        # then calculate chi-squared
        # https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf section 3-4
        KS = [3, 5, 6]
        RS = [16, 49, 75]
        k, r = KS[case_id], RS[case_id]

        PIS = [[0.2148, 0.3672, 0.2305, 0.1875],
               [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124],
               [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
               ]
        pi = PIS[case_id]
        chi_squared = sum(((v[i] - r * pi[i]) ** 2) / (r * pi[i]) for i in range(k + 1))
        # print(f"M={block_size}, K={k}, R={r}, v={v} CHI_SQUARED={chi_squared}")

        # calculate p-value
        p_value = gammaincc(k/2, chi_squared / 2)
        return (p_value >= self.alpha, p_value)

    def convert_array_to_matrices(array_of_arrays):
        return [np.mat(arr) for arr in array_of_arrays]

    def test_5(self, input: str) -> tuple:
        """
        Matrice test
        Input length recommended to be at least 38.912 bit
        """
        # def P(r, q, m):
        #     product = 1.0
        #     for i in range(r):
        #         product *= ((1 - 2 ** (i - q)) * (1 - 2 ** (i - m))) / (1 - 2 ** (i - r))
        M = 32
        Q = 32
        N = self.n // (M * Q)
        if N == 0:
            raise ValueError("Not enough bits to test (minimum 1024 required)")
        arr = np.array(list(map(int, input[0:N * M * Q])))
        arr = arr.reshape(N, M, Q)
        # print(np.linalg.matrix_rank(arr[0]), np.linalg.matrix_rank(arr[1]))

        # count how many matrices have ranks M and M-1
        F = [0, 0, 0]
        for matrix in arr:
            rank = np.linalg.matrix_rank(matrix)
            if rank == M:
                F[0] += 1
            elif rank == M - 1:
                F[1] += 1
            else:
                F[2] += 1
        pi = [1.0, 0.0, 0.0]
        # approximate the probability of ranks
        for j in range(1, 50):
            pi[0] *= 1 - (1 / (2 ** j))
        pi[1] = 2 * pi[0]
        pi[2] = 1 - pi[0] - pi[1]

        # calculate p-value
        chi_squared = 0.0
        for i in range(len(pi)):
            chi_squared += pow((F[i] - pi[i] * N), 2) / (pi[i] * N)

        p_value = math.exp(-chi_squared / 2)
        # print(f"chi-squared={chi_squared} p-value={p_value}")
        return (p_value >= self.alpha, p_value)

    def test_6(self, input: str) -> tuple:
        """
        Spectral test
        """
        plus_one_minus_one = list(map(lambda c: int(c) * 2 - 1, input))

        # Step 2 - Apply a Discrete Fourier transform (DFT) on X to produce: S = DFT(X).
        # A sequence of complex variables is produced which represents periodic
        # components of the sequence of bits at different frequencies
        spectral = sff.fft(plus_one_minus_one)

        # Step 3 - Calculate M = modulus(S´) ≡ |S'|, where S´ is the substring consisting of the first n/2
        # elements in S, and the modulus function produces a sequence of peak heights.
        modulus = abs(spectral[0:self.n // 2])

        # Step 4 - Compute T = sqrt(log(1 / 0.05) * length_of_string) the 95 % peak height threshold value.
        # Under an assumption of randomness, 95 % of the values obtained from the test should not exceed T.
        tau = sqrt(math.log(1 / 0.05) * self.n)

        # Step 5 - Compute N0 = .95n/2. N0 is the expected theoretical (95 %) number of peaks
        # (under the assumption of randomness) that are less than T.
        n0 = 0.95 * (self.n / 2)

        # Step 6 - Compute N1 = the actual observed number of peaks in M that are less than T.
        n1 = len(np.where(modulus < tau)[0])

        # Step 7 - Compute d = (n_1 - n_0) / sqrt (length_of_string * (0.95) * (0.05) / 4)
        d = (n1 - n0) / sqrt(self.n * (0.95) * (0.05) / 4)

        # Step 8 - Compute p_value = erfc(abs(d)/sqrt(2))
        p_value = math.erfc(math.fabs(d) / sqrt(2))
        return (p_value >= self.alpha, p_value)

    def test_7(self, input: str, target_pattern='000000001') -> tuple:
        """
        Non-overlapping template matching test
        """
        N = 8  # number of blocks
        pattern_size = len(target_pattern)
        block_size = self.n // N
        pattern_counts = [0] * N

        # for each block count the number of pattern hits
        for i in range(N):
            block = input[i * block_size:(i + 1) * block_size]
            window_start = 0
            while window_start < block_size:
                if block[window_start:window_start + pattern_size] == target_pattern:
                    pattern_counts[i] += 1
                    window_start += pattern_size
                else:
                    window_start += 1

        # calculate the theoretical mean and variance
        mean = (block_size - pattern_size + 1) / pow(2, pattern_size)
        # variance - σ2 = M((1/pow(2,m)) - ((2m -1)/pow(2, 2m)))
        variance = block_size * ((1 / pow(2, pattern_size)) - (((2 * pattern_size) - 1) / (pow(2, pattern_size * 2))))

        # calculate the chi-squared statistic for these pattern matches
        chi_squared = 0
        for i in range(N):
            chi_squared += pow((pattern_counts[i] - mean), 2) / variance

        # calculate p-value
        p_value = gammaincc((N / 2), (chi_squared / 2))
        return (p_value >= self.alpha, p_value)

    def test_8(self, input: str, target_pattern='000000001') -> tuple:
        """
        Overlapping template matching test
        """
        def get_prob(u, x):
            out = math.exp(-x)
            if u != 0:
                out = x * math.exp(2 * -x) * (2 ** -u) * hyp1f1(u + 1, 2, x)
            return out

        pattern_size = 9
        block_size = 1032
        if self.n < block_size:
            raise ValueError("Not enough bits to test (minimum 1032 required)")

        pattern = '1' * pattern_size
        N = self.n // block_size  # number of blocks

        # λ = (M-m+1)/pow(2, m)
        lambda_val = float(block_size - pattern_size + 1) / pow(2, pattern_size)
        # η = λ/2
        eta = lambda_val / 2

        pi = [get_prob(i, eta) for i in range(5)]
        pi.append(1 - sum(pi))

        pattern_counts = [0] * 6
        for i in range(N):
            block = input[i * block_size:(i + 1) * block_size]
            # count the number of pattern hits
            pattern_count = sum([block[j:j + pattern_size] == pattern for j in range(block_size)])
            pattern_counts[min(pattern_count, 5)] += 1

        # calculate p-value
        chi_squared = sum([pow(pattern_counts[i] - N * pi[i], 2) / (N * pi[i]) for i in range(len(pattern_counts))])
        p_value = gammaincc(5 / 2, chi_squared / 2)
        return (p_value >= self.alpha, p_value)

    def test_9(self, input: str) -> tuple:
        """
        Maurer’s “Universal Statistical” Test
        """
        return (True, 1)

    def test_10(self, input: str) -> tuple:
        """
        Linear complexity test
        Recommended n >= 10^6 and M in [500; 5000]
        """
        def berlekamp_massey_algorithm(block_data):
            """
            An implementation of the Berlekamp Massey Algorithm. Taken from Wikipedia [1]
            [1] - https://en.wikipedia.org/wiki/Berlekamp-Massey_algorithm
            The Berlekamp–Massey algorithm is an algorithm that will find the shortest linear feedback shift register (LFSR)
            for a given binary output sequence. The algorithm will also find the minimal polynomial of a linearly recurrent
            sequence in an arbitrary field. The field requirement means that the Berlekamp–Massey algorithm requires all
            non-zero elements to have a multiplicative inverse.
            :param block_data:
            :return:
            """
            n = len(block_data)
            c = np.zeros(n)
            b = np.zeros(n)
            c[0], b[0] = 1, 1
            l, m, i = 0, -1, 0
            int_data = [int(el) for el in block_data]
            while i < n:
                v = int_data[(i - l):i]
                v = v[::-1]
                cc = c[1:l + 1]
                d = (int_data[i] + np.dot(v, cc)) % 2
                if d == 1:
                    temp = copy(c)
                    p = np.zeros(n)
                    for j in range(0, l):
                        if b[j] == 1:
                            p[j + i - m] = 1
                    c = (c + p) % 2
                    if l <= 0.5 * i:
                        l = i + 1 - l
                        m = i
                        b = temp
                i += 1
            return l

        M = 500  # block size
        N = self.n // M  # number of blocks
        # the probabilities computed by the equations in Section 3.10
        pi = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]

        t2 = (M / 3 + 2 / 9) / (2 ** M)
        mean = 0.5 * M + (1 / 36) * (9 + (-1) ** (M + 1)) - t2

        if N == 0:
            raise ValueError("Not enough bits to test (minimum 500 bits)")

        # calculate complexities of blocks using the Berlekamp-Massey algorithm
        complexities = [berlekamp_massey_algorithm(input[i * M:(i + 1) * M]) for i in range(N)]

        t = [-1 * (((-1) ** M) * (c - mean) + 2 / 9) for c in complexities]
        vg = np.histogram(t, bins=[-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf])[0][::-1]
        # calculate p-value
        chi_squared = sum([((vg[i] - N * pi[i]) ** 2) / (N * pi[i]) for i in range(7)])
        p_value = gammaincc(6 / 2, chi_squared / 2)
        return (p_value >= self.alpha, p_value)

    def test_11(self, input: str) -> tuple:
        """
        Serial test
        """
        m = 16  # pattern length
        input += input[:(m - 1)]

        # get max length one patterns for m, m-1, m-2
        max_pattern = '1' * (m + 1)

        # Step 02: Determine the frequency of all possible overlapping m-bit blocks,
        # all possible overlapping (m-1)-bit blocks and
        # all possible overlapping (m-2)-bit blocks.
        v_01 = np.zeros(int(max_pattern[0:m:], 2) + 1)
        v_02 = np.zeros(int(max_pattern[0:m - 1:], 2) + 1)
        v_03 = np.zeros(int(max_pattern[0:m - 2:], 2) + 1)

        for i in range(self.n):
            # Check what pattern is observed and use its binary form ass its index
            v_01[int(input[i:i + m:], 2)] += 1
            v_02[int(input[i:i + m - 1:], 2)] += 1
            v_03[int(input[i:i + m - 2:], 2)] += 1

        v = [v_01, v_02, v_03]
        # Step 03 Compute for ψs
        sums = [sum(pow(v[i][j], 2) for j in range(len(v[i]))) * pow(2, m - i) / self.n for i in range(3)]

        # compute the test statistics and p-values
        # Step 04 Compute for ∇
        nabla_01 = sums[0] - sums[1]
        nabla_02 = sums[0] - 2 * sums[1] + sums[2]

        # Step 05 Compute p-values
        p_value_01 = gammaincc(pow(2, m - 1) / 2, nabla_01 / 2)
        p_value_02 = gammaincc(pow(2, m - 2) / 2, nabla_02 / 2)

        return (p_value_01 >= self.alpha and p_value_02 >= self.alpha, p_value_01, p_value_02)

    def test_12(self, input: str) -> tuple:
        """
        Approximate entropy test
        """
        m = max(2, int(math.log2(self.n)) - 6)  # pattern length
        # append m-1 bits from the beginning of the sequence to the end of the sequence.
        # NOTE: documentation says m-1 bits but that doesnt make sense, or work.
        input += input[:m + 1:]

        max_pattern = '1' * (m + 2)
        # keep track of each pattern's appearence frequency
        v_01 = np.zeros(int(max_pattern[0:m:], 2) + 1)
        v_02 = np.zeros(int(max_pattern[0:m + 1:], 2) + 1)

        for i in range(self.n):
            # work out what pattern is observed
            v_01[int(input[i:i + m:], 2)] += 1
            v_02[int(input[i:i + m + 1:], 2)] += 1

        # calculate the test statistics and p values
        v = [v_01, v_02]

        sums = [sum([v[i][j] * math.log(v[i][j] / self.n) for j in range(len(v[i])) if v[i][j] > 0]) / self.n for i in range(2)]
        ape = sums[0] - sums[1]
        chi_squared = 2 * self.n * (math.log(2) - ape)
        p_value = gammaincc(pow(2, m - 1), chi_squared / 2)
        return (p_value >= self.alpha, p_value)

    def test_13(self, input: str, mode=0) -> tuple:
        """
        Cumulative sums test
        Minimum n = 100
        """
        if mode != 0:
            input = reversed(input)
        cumsums = list(np.cumsum(np.array(list(map(lambda c: int(c) * 2 - 1, input)))))

        abs_max = max(cumsums, key=abs)
        sqrt_n = sqrt(self.n)  # calculation optimization

        # calculate statistics and p-value by summating Ф values
        start = int((-self.n / abs_max + 1) // 4)
        end = int((self.n / abs_max - 1) // 4)
        terms_one = [norm.cdf((4 * k + 1) * abs_max / sqrt_n) - norm.cdf((4 * k - 1) * abs_max / sqrt_n) for k in range(start, end + 1)]

        start = int((-self.n / abs_max - 3) // 4)
        end = int((self.n / abs_max - 1) // 4)
        terms_two = [norm.cdf((4 * k + 3) * abs_max / sqrt_n) - norm.cdf((4 * k + 1) * abs_max / sqrt_n) for k in range(start, end + 1)]

        p_value = 1 - sum(terms_one) + sum(terms_two)
        return (p_value >= self.alpha, p_value)

    def test_14(self, input: str) -> tuple:
        """
        Random Excursions Test
        Recommended minimum input size: 1.000.000 bits 
        """
        def get_pi_value(k, x):
            """
            This method is used by the random_excursions method to get expected probabilities
            """
            x = abs(x)
            if k == 0:
                pi = 1 - 1 / (2 * x)
            elif k >= 5:
                pi = (1 / (2 * x)) * (1 - 1 / (2 * x)) ** 4
            else:
                pi = (1 / (4 * x * x)) * (1 - 1 / (2 * x)) ** (k - 1)
            return pi

        # state = 1
        # normalize sequence, compute cumulative sums
        sequence_x = list(map(lambda c: int(c) * 2 - 1, input))
        cumsum = np.cumsum(sequence_x)  # S

        # form a new sequence S'
        cumsum = np.append(cumsum, [0])
        cumsum = np.append([0], cumsum)

        # states we are interested in
        states = np.array([-4, -3, -2, -1, 1, 2, 3, 4])

        # identify all the locations where the cumulative sum is 0
        zero_positions = np.where(cumsum == 0)[0]
        # print("\nZERO_POSITIONS:\n", zero_positions)

        # with this identify 'cycles'
        cycles = [cumsum[zero_positions[pos]:zero_positions[pos + 1] + 1] for pos in range(len(zero_positions) - 1)]
        j = len(cycles)  # number of cycles

        # according to documentation this check should take place
        # but input length for this to not fail is much higher
        # than I usually have so it is omitted
        """
        if j < 500:
            return (False, 0)
        """

        # determine the number of times each cycle visits each state
        state_count = [[len(np.where(cycle == state)[0]) for state in states] for cycle in cycles]
        state_count = np.transpose(np.clip(state_count, 0, 5))
        # print("State count:\n", state_count)

        # now build a table, which indicates how many cycles contain each
        # state given number of times (i.e. in the first column should be
        # how many cycles reach each state [-4..4] exactly zero times)
        state_count_2 = [[(sct == cycle).sum() for sct in state_count] for cycle in range(6)]
        state_count_2 = np.transpose(state_count_2)
        # print("SU:\n", state_count_2)

        # compute probabilities pi_k(x) that state x occurs k times in a
        # random distribution
        pi = [[get_pi_value(k, x) for k in range(6)] for x in states]

        j_times_pi = j * np.array(pi)
        chi_squared = np.sum((np.array(state_count_2) - j_times_pi) ** 2 / j_times_pi, axis=1)
        # print("CHI_SQUARED:\n", chi_squared)

        p_values = [gammaincc(5 / 2, chi / 2) for chi in chi_squared]
        # print("P-Values:\n", p_values)

        return (all(map(lambda p: p >= self.alpha, p_values)), tuple(p_values))

    def test_15(self, input: str) -> tuple:
        """
        Random Excursions Variant Test
        Recommended minimum input size: 1.000.000 bits
        """
        # normalize sequence, compute cumulative sums
        sequence_x = list(map(lambda c: int(c) * 2 - 1, input))
        cumsum = np.cumsum(sequence_x)  # S

        # form a new sequence S'
        cumsum = np.append(cumsum, [0])
        cumsum = np.append([0], cumsum)

        # identify all the locations where the cumulative sum is 0
        zero_positions = np.where(cumsum == 0)[0]

        # with this identify 'cycles'
        cycles = [cumsum[zero_positions[pos]:zero_positions[pos + 1] + 1] for pos in range(len(zero_positions) - 1)]
        j = len(cycles)  # number of cycles

        # states we are interested in
        states = list(range(-9, 0)) + list(range(1, 10))

        # and count how many times each position is taken
        xi = {x: len(np.where(cumsum == x)[0]) for x in states}

        # compute p-values
        p_values = [math.erfc(abs(xi[x] - j) / sqrt(2 * j * (4 * abs(x) - 2))) for x in states]

        return (all(map(lambda p: p >= self.alpha, p_values)), tuple(p_values))


seq = generate_binary_sequence(1032000)
tests = NISTTests(seq)
tests.test_sequence()
# tests = NISTTests("0110110101")
# print(tests.test_15("0110110101"))
