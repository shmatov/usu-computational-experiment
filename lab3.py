#!/usr/bin/env python
import decimal
from decimal import Decimal
import math

import numpy


def solve():
    pass


if __name__ == '__main__':
    decimal.getcontext().rounding = decimal.ROUND_DOWN
    for k in (2, 4, 6):
        decimal.getcontext().prec = k
        solve()

