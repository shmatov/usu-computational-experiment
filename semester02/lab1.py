#!/usr/bin/env python
import math
from table import ASCIITable
import pylab


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
    left = 0.0
    right = 1.0
    n = 30
    methods = [rectangle_method, trapezoidal_rule, simpson]
    method_names = map(_parse_method_name, methods)
    headers = [y for x in method_names for y in [x, 'error']]
    errors = {method: [] for method in method_names}

    table = ASCIITable(['n'] + headers)
    for i in range(1, n + 1):
        step = (right - left) / float(i)
        table_row = [i]
        for method in methods:
            integral = method(f2, left, right, step)
            error = math.fabs(math.pi / 4 - integral)
            errors[_parse_method_name(method)].append(error)
            table_row.extend([integral, error])
        table.add_row(table_row)
    print 'TASK02'
    print table
    plot_errors(errors)


def _parse_method_name(method):
    return method.func_name.split('_')[0]


def plot_errors(errors):
    for name, values in errors.items():
        pylab.plot(range(len(values)), values, label=name)
    pylab.legend()
    pylab.show()


def solve_task03():
    eps = 0.005
    left = 0
    right = round(1 + math.sqrt(math.pi / (4 * eps))) # A(eps)
    max_f2c = 0.3256314123833 # M2, we know this a hard way ;[
    # maximize ((18 x^4)/(x^3+1)^3-(6 x)/(x^3+1)^2) tan^(-1)(x)-(6 x^2)/((x^2+1) (x^3+1)^2)-(2 x)/((x^2+1)^2 (x^3+1)) over [0, 14]
    step = math.pow(12 * eps / max_f2c, 1/3)

    table = ASCIITable(['compute integral with epsilon'])
    table.add_row([trapezoidal_rule(f3, left, right, step)])
    print 'TASK03'
    print table



if __name__ == '__main__':
    solve_task01()
    solve_task02()
    solve_task03()
