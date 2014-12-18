#!/usr/bin/env python
import math


eps = 0.5 * 10**-5
interval = (0, 2)


def sign(x):
    return 1 if x >= 0 else -1


def f(x):
    """f(x) = 2*cos(x) - e^x"""
    return 2 * math.cos(x) - math.e ** x


def df(x):
    """f'(x) = -2*sin(x) - e^x"""
    return -2 * math.sin(x) - math.e ** x


def dichotomy():
    a, b = interval
    x = a
    assert(sign(f(a)) != sign(f(b)))
    i = []
    while abs(b - a) > eps:
        x = (a + b) / 2.
        i.append(x)
        if sign(f(x)) == sign(f(a)):
            a = x
        if sign(f(x)) == sign(f(b)):
            b = x
    return x, i


def newton():
    x0, _ = interval
    i = []
    while True:
        x1 = x0 - (f(x0) / df(x0))
        i.append(x1)
        if abs(x1 - x0) < eps:
            return x1, i
        x0 = x1


def static_hords():
    x0, x_curr = interval
    i = []
    while True:
        x_next = x_curr - (f(x_curr)*(x0 - x_curr)) / (f(x0) - f(x_curr))
        i.append(x_next)
        if abs(x_curr - x_next) < eps:
            return x_next, i
        x_curr = x_next


def moving_hords():
    x_prev, x_curr = interval
    i = []
    while True:
        x_next = x_curr - (f(x_curr)*(x_curr - x_prev)) / (f(x_curr) - f(x_prev))
        i.append(x_next)
        if abs(x_curr - x_next) < eps:
            return x_next, i
        x_prev, x_curr = x_curr, x_next


if __name__ == '__main__':
    def fmt(*cells):
        return '\n'.join(map(str, cells)) + '\n' + '-' * 80

    def compute(method):
        result, iterations = method()
        return method.func_name, result, iterations

    print f.func_doc
    print df.func_doc
    print 'interval:', list(interval)
    print 'eps:', eps
    print '-'*80

    print fmt(*compute(dichotomy))
    print fmt(*compute(static_hords))
    print fmt(*compute(moving_hords))
    print fmt(*compute(newton))
