#!/usr/bin/env python
import math
from table import ASCIITable


def sign(num):
    return math.copysign(1, num)


def f1(x):
    B = 1  # random const from my head
    return math.sqrt(1 + x + x*x + B)


def f2(x):
    return 1/(1 + x*x)


def f3(x):
    return math.atan(x) / (1 + x*x*x)


def integral_core(function, accumulate, left, right, step):
    assert(left < right and step > 0)
    integral_acc = 0
    while left < right:
        left_actual = min(left + step, right)
        integral_acc += accumulate(left, left_actual)
        left = left_actual
    return integral_acc


def rectangle_method(function, left, right, step):
    def accumulator(a, b):
        return (b - a) * function((a + b) / 2)
    return integral_core(function, accumulator, left, right, step)


def trapezoidal_rule(function, left, right, step):
    def accumulator(a, b):
        return (b - a) * (function(a) + function(b)) / 2
    return integral_core(function, accumulator, left, right, step)


def simpson(function, left, right, step):
    def accumulator(a, b):
        return (b - a) / 6 * (function(a) + function(b) + 4 * function((a + b) / 2))
    return integral_core(function, accumulator, left, right, step)


def compute_intergral_with_eps(function, function_c2, left, right, eps):
    pass


def dichotomy(f, a, b, eps):
    x = None
    assert(sign(f(a)) != sign(f(b)))
    while abs(b - a) > eps:
        x = (a + b) / 2.
        if sign(f(x)) == sign(f(a)):
            a = x
        if sign(f(x)) == sign(f(b)):
            b = x
    return x


def solve_task01():
    left = 0
    right = 1
    step = 0.1
    table = ASCIITable(['method\computation', 'Sn[f]', 'S2n[f]', 'Runge'])
    for integral_method in [(rectangle_method, 2), (trapezoidal_rule, 2), (simpson, 4)]:
        integral, algebraic_accuracy = integral_method
        sn = integral(f1, left, right, step)
        s2n = integral(f1, left, right, step / 2)
        runge = abs(sn - s2n) / (2 ** algebraic_accuracy - 1)
        table.add_row([integral.func_name, sn, s2n, runge])
    print 'TASK01'
    print table


def solve_task02():
    pass


def solve_task03():
    eps = 0.005
    left = 0
    right = round(1 + math.sqrt(math.pi / (4 * eps)))
    table = ASCIITable(['compute integral with epsilon'])
    table.add_row([compute_intergral_with_eps(trapezoidal_rule, f3, left, right, eps)])
    print 'TASK03'
    print table


# document = LaTeX() uncomment at global instance
if __name__ == '__main__':
    solve_task01()
    solve_task02()
    solve_task03()
