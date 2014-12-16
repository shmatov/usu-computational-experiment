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

