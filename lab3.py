#!/usr/bin/env python

from copy import deepcopy
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


def patch_decimal(precision):
    decimal_precision = Decimal('.' + '0'*precision)

    def truncate_wrapper(method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            result = method(*args, **kwargs)
            if isinstance(result, Decimal):
                return result.quantize(decimal_precision, rounding=ROUND_DOWN)
            return result
        return wrapper

    methods = ['__add__', '__sub__', '__mul__', '__floordiv__', '__mod__',
               '__divmod__', '__pow__']
    for method in methods:
        setattr(Decimal, method, truncate_wrapper(getattr(Decimal, method)))

    def new_wrapper(wrapped_func):
        def wrapper(cls, value, *args, **kwargs):
            if isinstance(value, float):
                value = ('{:.' + str(precision) + 'f}').format(value)
            return wrapped_func(cls, value, *args, **kwargs)
        return staticmethod(wrapper)
    Decimal.__new__ = new_wrapper(Decimal.__new__)


def to_decimal_matrix(matrix):
    return [
        [Decimal(x) for x in row]
        for row in matrix
    ]


def generate_matrix_and_vector(n):
    m = 21 - n
    a = 0.1 * m + 0.01 * n
    b = 0.2 * m + 0.02 * n + 0.01 * m * n + 0.001 * n * n
    matrix = [
        [1.2345, 3.1415, 1],
        [2.3456, 5.9690, 0],
        [3.4567, 2.1828, (2 + 0.1 * n)]
    ]
    vector = [[7.5175 + a], [14.2836], [7.8223 + b]]
    return map(to_decimal_matrix, (matrix, vector))


def simple_equation_solver(matrix, vector):
    new_vector = [None] * 3
    new_vector[2] = vector[2][0] / matrix[2][2]
    new_vector[1] = (vector[1][0] - matrix[1][2] * new_vector[2]) / matrix[1][1]
    new_vector[0] = (vector[0][0] - matrix[0][1] * new_vector[1] - matrix[0][2] * new_vector[2]) / matrix[0][0]
    return new_vector


def gauss_transformation(matrix, vector):
    matrix = deepcopy(matrix)
    vector = deepcopy(vector)
    for column in range(0, 2):
        max = column;
        for line in range(column + 1, 3):
            if (abs(matrix[max][column]) < abs(matrix[x][column])):
                max = x
        if (max != column):
            matrix[column], matrix[max] = matrix[max], matrix[column]
            vector[column], vector[max] = vector[max], vector[column]
        for line in range(column + 1, 3):
            if (matrix[line][column] != Decimal(0.0)):
                koef = matrix[column][column] / matrix[line][column]
                vector[line][0] *= koef
                matrix[line][column] = 0;
                for rest_index in range(column + 1, 3):
                    matrix[line][rest_index] *= koef
                    matrix[line][rest_index] -= matrix[column][rest_index]
    return matrix, vector


def vector_distance(vector1, vector2):
    accum = 0.0
    for x in range(0, 3):
        diff = abs(vector1[x] - vector2[x])
        accum += diff * diff
    return math.sqrt(accum)


if __name__ == '__main__':
    patch_decimal(2)
    n = int(raw_input())
    m, v = generate_matrix_and_vector(n)
    print repr(m)
    print repr(v)
