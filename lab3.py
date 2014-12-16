#!/usr/bin/env python
from decimal import getcontext, Decimal, ROUND_DOWN
import math
import sys

import numpy


def solve():
    pass


def parse_matrix(lines):
    def parse_row(row):
        return map(Decimal, row.strip().split(' '))
    return numpy.matrix(map(parse_row, lines))


if __name__ == '__main__':

    N = int(raw_input())
    M = 21 - N
    A = 0.1 * M + 0.01 * N
    B = 0.2 * M + 0.02 * N + 0.01 * M * N + 0.001 * N * N
    matrix_A = [[1.2345, 3.1415, 1], [2.3456, 5.9690, 0], [3.4567, 2.1828, (2 + 0.1 * N)]]
    vector_b = [[7.5175 + A], [14.2836], [7.8223 + B]]


    getcontext().rounding = ROUND_DOWN
    print 'Matrix:'
    A = parse_matrix(iter(raw_input, ''))
    print 'Vector:'
    b = parse_matrix(iter(raw_input, ''))
    print 'Solve Ax = b.'
    print 'A:'
    print A
    print 'b:'
    print b
    # input("qwe")

    for k in (2, 4, 6):
        getcontext().prec = k
        solve()

