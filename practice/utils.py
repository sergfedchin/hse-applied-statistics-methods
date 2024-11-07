import random
import math
import numpy as np
from collections.abc import Iterable


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


def all_keys(*args: dict,):
    '''Get all unique keys of the input dicts'''
    return np.unique(np.concat([list(a.keys()) for a in args
                                if isinstance(a, dict)]))


def scale_dict(d: dict, x: float | int):
    return {k: v * x for k, v in d.items()}


def isnumber(x):
    return np.isreal(x)
    try:
        _ = x + 1
        return True
    except TypeError:
        return False


def isiterable(x):
    try:
        _ = iter(x)
        return True
    except TypeError:
        return False


def type_name(x):
    return type(x).__name__


def raise_operand_exception(x, y, op: str):
        raise ValueError(f"unsupported operand type(s) for {op}: '{type_name(x)}' and '{type_name(y)}'")


class IntervalArithmetics:
    def __init__(self, a: int | float, b: int | float = None):
        self.a = a
        if b is not None:
            if b < a:
                raise ValueError(
                    "left boundary should be less than right boundary")
            self.b = b
        else:
            self.b = a

    def radius(self):
        return (self.b - self.a) / 2

    def __repr__(self):
        return f"[{self.a}; {self.b}]"

    def __pos__(self):
        return self

    def __add__(self, o):
        if not isinstance(o, IntervalArithmetics):
            o = IntervalArithmetics(o)
        return IntervalArithmetics(self.a + o.a, self.b + o.b)

    def __radd__(self, o):
        return self + o

    def __iadd__(self, o):
        return self + o

    def __neg__(self):
        return IntervalArithmetics(-self.b, -self.a)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) - (-o)

    def __isub__(self, o):
        return self - o

    def __mul__(self, o):
        if not isinstance(o, IntervalArithmetics):
            o = IntervalArithmetics(o)
        r = [self.a * o.a, self.a * o.b, self.b * o.a, self.b * o.b]
        return IntervalArithmetics(min(r), max(r))

    def __rmul__(self, o):
        return self * o

    def __imul__(self, o):
        return self * o

    def _inverse(interval):
        if interval.a <= 0 <= interval.b:
            raise ZeroDivisionError("divider interval cannot contain zero.")
        r = [1.0 / interval.a, 1.0 / interval.b]
        return IntervalArithmetics(min(r), max(r))

    def __truediv__(self, o):
        if not isinstance(o, IntervalArithmetics):
            if o == 0:
                raise ZeroDivisionError("divider interval cannot be zero.")
            o = IntervalArithmetics(o)
        return self * IntervalArithmetics._inverse(o)

    def __rtruediv__(self, o):
        return o * IntervalArithmetics._inverse(self)

    def __itruediv__(self, o):
        return self / o

    def __pow__(self, o):
        if not isinstance(o, IntervalArithmetics):
            o = IntervalArithmetics(o)
        r = [self.a ** o.a, self.a ** o.b, self.b ** o.a, self.b ** o.b]
        return IntervalArithmetics(min(r), max(r))

    def __rpow__(self, o):
        return IntervalArithmetics(o) ** self

    def abs(self):
        if self.a < 0 and self.b > 0:
            return IntervalArithmetics(0, max(-self.a, self.b))
        r = [abs(self.a), abs(self.b)]
        return IntervalArithmetics(min(r), max(r))


class AffineArithmetics:
    cnt = 0

    def __init__(self, a: IntervalArithmetics | int | float,
                 b: int | float = None):
        self.cv: float = 0
        self.deviations: dict[int, float] = {}
        self.cnt: int = 0
        if isinstance(a, IntervalArithmetics):
            self.cv = (a.a + a.b) / 2
            self.deviations[AffineArithmetics._next_id()] = (a.b - a.a) / 2
        else:
            if AffineArithmetics._isnumber(a):
                if b is None:
                    self.cv = a
                elif AffineArithmetics._isnumber(b):
                    if b < a:
                        raise ValueError("left boundary "
                                         "should be less than right boundary")
                    self.cv = (a + b) / 2
                    self.deviations[AffineArithmetics._next_id()] = (b - a) / 2

    def _next_id():
        res = AffineArithmetics.cnt
        AffineArithmetics.cnt += 1
        return res

    def _isnumber(x):
        return isinstance(x, int) or isinstance(x, float)

    def _all_keys(*args: dict):
        return list(map(int, np.concat([list(a.keys())
                                        for a in args if
                                        isinstance(a, dict)]).astype(int)))

    def radius(self):
        return sum([abs(x_i) for x_i in self.deviations.values()])

    def to_IA(self):
        rad = self.radius()
        return IntervalArithmetics(self.cv - rad, self.cv + rad)

    def __repr__(self):
        return self.to_IA().__repr__()

    def __eq__(self, other):
        return self.cv == other.cv and self.deviations == other.deviations

    def __pos__(self):
        return self

    def __add__(self, other):
        if not isinstance(other, AffineArithmetics):
            if AffineArithmetics._isnumber(other):
                other = AffineArithmetics(other)
            else:
                raise TypeError(f"Cannot add {AffineArithmetics} "
                                f"and {type(other)}")
        res = AffineArithmetics(self.cv + other.cv)
        for i in AffineArithmetics._all_keys(self.deviations,
                                             other.deviations):
            res.deviations[i] = (self.deviations.get(i, 0) +
                                 other.deviations.get(i, 0))
        return res

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __neg__(self):
        res = AffineArithmetics(-self.cv)
        for i, x_i in self.deviations.items():
            res.deviations[i] = -x_i
        return res

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) - (-other)

    def __isub__(self, other):
        return self - other

    # https://asp-eurasipjournals.springeropen.com/articles/10.1186/1687-6180-2014-36
    def __mul__(self, other):
        if AffineArithmetics._isnumber(other):
            res = AffineArithmetics(self.cv * other)
            res.deviations = {i: x_i * other for i, x_i in
                              self.deviations.items()}
            return res
        if isinstance(other, IntervalArithmetics):
            other = AffineArithmetics(other)
        if not isinstance(other, AffineArithmetics):
            raise TypeError(f"unsupported operand type(s) for *: "
                            f"'{AffineArithmetics.__name__}' "
                            f"and '{type(other).__name__}'")
        x_0, y_0 = self.cv, other.cv
        res = AffineArithmetics(x_0 * y_0)
        for i in AffineArithmetics._all_keys(self.deviations,
                                             other.deviations):
            x_i, y_i = self.deviations.get(i, 0), other.deviations.get(i, 0)
            res.deviations[i] = x_0 * y_i + y_0 * x_i
        res.deviations[AffineArithmetics._next_id()] = (self.radius() *
                                                        other.radius())
        return res

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def _inverse(self):
        if (len(self.deviations) == 0):
            return AffineArithmetics(1 / self.cv)
        r = self.radius()
        a, b = self.cv - r, self.cv + r
        if a <= 0 <= b:
            raise ZeroDivisionError("divider interval cannot contain zero")
        fa, fb = 1 / a, 1 / b
        if a > 0:
            p = -fb / b
            ya = fa - p * a
            yb = 2 * fb
        else:
            p = -fa / a
            ya = 2 * fa
            yb = fb - p * b
        res = AffineArithmetics(p * self.cv + (ya + yb) / 2)
        res.deviations = {i: x_i * p for i, x_i in self.deviations.items()}
        res.deviations[AffineArithmetics._next_id()] = (ya - yb) / 2
        return res

    # https://www.jstage.jst.go.jp/article/elex/1/7/1_7_176/_pdf/-char/en
    # def _inverse(self):
    #     if (len(self.deviations) == 0):
    #         return AA(1 / self.cv)
    #     r = self._radius()
    #     y_lower, y_upper = self.cv - r, self.cv + r
    #     if y_lower <= 0 <= y_upper:
    #         raise ZeroDivisionError("divider interval cannot contain zero")
    #     y_0 = self.cv
    #     prod = y_lower * y_upper
    #     inv_prod = 1 / prod
    #     sqrt_prod = math.sqrt(prod)

    #     if 0 < y_lower:
    #         res = AA(-inv_prod * y_0 + (y_lower + y_upper + 2 * sqrt_prod)
    #                  * inv_prod * 0.5)
    #         res.deviations = {i: -y_i * inv_prod for i, y_i in
    #                           self.deviations.items()}
    #         res.deviations[AA._next_id()] = (y_lower + y_upper - 2 *
    #                                          sqrt_prod) * inv_prod * 0.5
    #         return res
    #     if y_upper < 0:
    #         res = AA(-inv_prod * y_0 + (y_lower + y_upper - 2 * sqrt_prod) *
    #                  inv_prod * 0.5)
    #         res.deviations = {i: -y_i * inv_prod for i, y_i in
    #                           self.deviations.items()}
    #         res.deviations[AA._next_id()] = (-y_lower - y_upper - 2 *
    #                                          sqrt_prod) * inv_prod * 0.5
    #         return res
    #     raise Exception()

    def __truediv__(self, other):
        if AffineArithmetics._isnumber(other):
            res = AffineArithmetics(self.cv / other)
            res.deviations = {i: x_i / other for i, x_i in
                              self.deviations.items()}
            return res
        if isinstance(other, IntervalArithmetics):
            other = AffineArithmetics(other)
        if not isinstance(other, AffineArithmetics):
            raise TypeError(f"unsupported operand type(s) for /: "
                            f"'{AffineArithmetics.__name__}' "
                            f"and '{type(other).__name__}'")
        return self * AffineArithmetics._inverse(other)

    def __rtruediv__(self, other):
        return other * AffineArithmetics._inverse(self)

    def __itruediv__(self, other):
        return self / other


class FwdAAD:
    all_names = set()

    def __init__(self, real: float | int, dual: dict):
        self.real = real
        self.dual = dual
        self.original_name = None

    def value(self) -> float | int:
        return self.real

    def get_gradient(self) -> list[float]:
        return list(self.dual.values())

    def print_gradient(self, precision: int = 6) -> None:
        print(', '.join(['d/d{} = {:.{}f}'.format(key, d, precision) for key, d in sorted(list(self.dual.items()), key=lambda x: x[0] if len(x[0].split('_')) == 1 else int(x[0].split('_')[-1]))]))

    def get_variable(symbol: str, value: float | int = None, ignore_existent: bool = False):
        """
        Create variable symbol.

        :param symbol: unique name to indicate the variable
        :param value: value of the variable
        :param ignore_existent: do not raise an exception if variable with
        name `symbol` already exists
        :returns: FwdAAD  variable
        :raises ValueError: if variable with name `symbol` already exists and
        `ignore_existent` is False
        """
        if symbol not in FwdAAD.all_names or ignore_existent:
            FwdAAD.all_names.add(symbol)
            res = FwdAAD(value, {symbol: 1})
            res.original_name = symbol
            return res
        else:
            raise ValueError(f"Variable with name '{symbol}' already exists."
                             "Please provide a unique name.")

    def get_vector(symbol: str, values: Iterable = None, length: int = None, ignore_existent: bool = False) -> list:
        """
        Create a list of variable symbols.

        :param symbol: unique name to indicate the variables. Each variable will have a name in format 'symbol'_i
        :param values: values of the variables
        :param length: length of the vector (is required if `values = None`, otherwise ignored)
        :param ignore_existent: do not raise an exception if a variable with name `symbol` with index in range
        [1; len(values)] already exists
        :returns: FwdAAD  variables list
        :raises ValueError: if a variable with name `symbol` with index in range [1; len(values)] already exists and
        `ignore_existent` is False
        """
        if values is not None:
            return [FwdAAD.get_variable(symbol + '_' + str(idx + 1), x, ignore_existent) for idx, x in enumerate(values)]
        elif length is not None:
            return [FwdAAD.get_variable(symbol + '_' + str(idx + 1), None, ignore_existent) for idx in range(length)]
        else:
            raise ValueError("Please provide values or specify length of the vector.")

    def set_value(self, value: int | float) -> None:
        if not isnumber(value):
            raise ValueError(f"Expected a number, got '{type_name(value)}'")
        self.real = value
        self.dual = {self.original_name: 1}

    def set_name(self, name: str) -> None:
        derivative = self.dual[self.original_name]
        del self.dual[self.original_name]
        FwdAAD.all_names.remove(self.original_name)
        self.original_name = name
        FwdAAD.all_names.add(name)
        self.dual[name] = derivative

    def set_vector_values(vector, values: Iterable[int | float]) -> None:
        if not isiterable(vector):
            raise ValueError(f"object '{type_name(vector)}' is not iterable")
        if not isiterable(values):
            raise ValueError(f"object '{type_name(values)}' is not iterable")
        if len(vector) != len(values):
            raise ValueError("Variables vector and values sizes mismatch")
        for var, val in zip(vector, values):
            if not isinstance(var, FwdAAD):
                raise ValueError(f"Expected '{FwdAAD.__name__}', got '{type_name(var)}'")
            var.set_value(val)

    def _raise_operand_exception(x, y, op: str):
        raise ValueError(f"unsupported operand type(s) for {op}: '{type_name(x)}' and '{type_name(y)}'")

    def __repr__(self):
        return str(self.real)

    def __add__(self, other):
        if isinstance(other, FwdAAD):
            real = self.real + other.real
            dual = {key: self.dual.get(key, 0) + other.dual.get(key, 0) for key in all_keys(self.dual, other.dual)}
            return FwdAAD(real, dual)
        elif isnumber(other):
            return FwdAAD(self.real + other, self.dual)
        else:
            FwdAAD._raise_operand_exception(self, other, '+')

    def __pos__(self):
        return self

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        if not isinstance(other, FwdAAD) or isnumber(other):
            FwdAAD._raise_operand_exception(self, other, '-')
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, FwdAAD):
            real = self.real * other.real
            dual = {key: self.dual.get(key, 0) * other.real + self.real * other.dual.get(key, 0) for key in all_keys(self.dual, other.dual)}
            return FwdAAD(real, dual)
        elif isnumber(other):
            return FwdAAD(self.real * other, scale_dict(self.dual, other))
        else:
            FwdAAD._raise_operand_exception(self, other, '*')

    def _inverse(self):
        real2 = self.real * self.real
        dual = {key: -d / real2 for key, d in self.dual.items()}
        return FwdAAD(1 / self.real, dual)

    def __truediv__(self, other):
        if isinstance(other, FwdAAD):
            return self * other._inverse()
        elif isnumber(other):
            return self * (1 / other)
        else:
            FwdAAD._raise_operand_exception(self, other, '/')

    def __pow__(self, other):
        if isinstance(other, FwdAAD):
            real = pow(self.real, other.real)
            # (f^g)' = f^{g-1} * (g*f' + f*ln(f)*g')
            real_x_log_real = self.real * math.log(self.real)
            pow_self_real_other_real_minus_one = pow(self.real, other.real - 1)
            dual = {key: pow_self_real_other_real_minus_one * (other.real * self.dual.get(key, 0) +  real_x_log_real * other.dual.get(key, 0)) for key in all_keys(self.dual, other.dual)}
        elif isnumber(other):
            real = pow(self.real, other)
            term = other * pow(self.real, other - 1)  # optimization
            dual = scale_dict(self.dual, term)
        else:
            FwdAAD._raise_operand_exception(self, other, '** or pow()')
        return FwdAAD(real, dual)

    def __rpow__(self, other):
        if isnumber(other):
            real = pow(other, self.real)
            term = real * math.log(other)
            dual = {key: term * d for key, d in self.dual.items()}
            return FwdAAD(real, dual)
        else:
            FwdAAD._raise_operand_exception(self, other, '** or pow()')

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self._inverse()
