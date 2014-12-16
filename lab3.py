#!/usr/bin/env python
from decimal import Decimal, ROUND_DOWN
from functools import wraps
import math


def simple_equation_solver(matrix, vector):
    new_vector = [None] * 3
    new_vector[2] = vector[2][0] / matrix[2][2]
    new_vector[1] = (vector[1][0] - matrix[1][2] * new_vector[2]) / matrix[1][1]
    new_vector[0] = (vector[0][0] - matrix[0][1] * new_vector[1] - matrix[0][2] * new_vector[2]) / matrix[0][0]
    return new_vector


def multiply(X, Y):
    return [
        [
            sum(a*b for a, b in zip(X_row, Y_col))
            for Y_col in zip(*Y)
        ]
        for X_row in X
    ]


def decimal_auto_quantize(precision):
    precision = Decimal('.' + '0'*precision)

    def truncate_wrapper(method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            result = method(*args, **kwargs)
            if isinstance(result, Decimal):
                return result.quantize(precision, rounding=ROUND_DOWN)
            return result
        return wrapper

    methods = ['__add__', '__sub__', '__mul__', '__floordiv__', '__mod__',
               '__divmod__', '__pow__']
    for method in methods:
        setattr(Decimal, method, truncate_wrapper(getattr(Decimal, method)))


if __name__ == '__main__':
    patch_decimal(2)

    N = int(raw_input())
    M = 21 - N
    A = 0.1 * M + 0.01 * N
    B = 0.2 * M + 0.02 * N + 0.01 * M * N + 0.001 * N * N
    matrix_A = [[1.2345, 3.1415, 1], [2.3456, 5.9690, 0], [3.4567, 2.1828, (2 + 0.1 * N)]]
    vector_b = [[7.5175 + A], [14.2836], [7.8223 + B]]
