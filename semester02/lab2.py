#!/usr/bin/env python
import math
from table import ASCIITable
import pylab


def sign(num):
    return math.copysign(1, num)


# INITIAL

initial_y_0 = 0.1
initial_a = 0.0
initial_b = 1.0

def initial_y_(x, y):
    return 30 * y * (x - 0.2) * (x - 0.7)

def initial_y__(x, y):
    return 30 * (y_(x, y) * (x - 0.2) * (x - 0.7) + y * (2 * x - 0.9))

def initial_y___(x, y):
    return 30 * (y__(x, y) * (x - 0.2) * (x - 0.7) + 2 * y_(x, y) * (2 * x - 0.9) + 2 * y)


# ONE STEP CALCULATIONS

def one_step_explicit_euler(x, y, step, y_, **kwargs):
    return y + step * y_(x, y)

def one_step_backward_euler(x, y, step, y_, **kwargs): # also known as 'with recalculation'
    y_new = one_step_explicit_euler(x, y, step, y_)
    return y + step * (y_(x, y) + y_(x + step, y_new)) / 2

def one_step_implicit_euler(x, y, step, y_, **kwargs):
    return None

def one_step_cauchy(x, y, step, y_, **kwargs):
    y_plus_halfstep = y + step * y_(x, y) / 2
    return y + step * f(x + step / 2, y_plus_halfstep)

def one_step_taylor_degree_2(x, y, step, y_, **kwargs):
    return y + step * y_(x, y)

def one_step_taylor_degree_3(x, y, step, y_, **kwargs):
    y__ = kwargs['y__']
    return y + step * y_(x, y) + step * step * y__(x, y)

def one_step_taylor_degree_4(x, y, step, y_, **kwargs):
    y__ = kwargs['y__']
    y___ = kwargs['y___']
    return y + step * y_(x, y) + step * step * y__(x, y) + step * step * step * y___(x, y)

def one_step_adams_two_steps_usage(x, y, step, y_, **kwargs):
    x_prev1, y_prev1 = kwargs['prev1']
    return y + step * (3 * f(x, y) - f(x_prev1, y_prev1)) / 2

def one_step_adams_three_steps_usage(x, y, step, y_, **kwargs):
    x_prev1, y_prev1 = kwargs['prev1']
    x_prev2, y_prev2 = kwargs['prev2']
    return y + step * (23 * f(x, y) - 16 * f(x_prev1, y_prev1) + 5 * f(x_prev2, y_prev2)) / 12

def one_step_runge_kutta(x, y, step, y_, **kwargs):
    return None


# TIED UP ONE STEP METHODS AND ACCELERATORS

EXPLICIT_EULER = (one_step_explicit_euler, 0, None)
BACKWARD_EULER = (one_step_backward_euler, 0, None)
IMPLICIT_EULER = (one_step_implicit_euler, 0, None)
CAUCHY = (one_step_cauchy, 0, None)
TAYLOR_DEGREE_2 = (one_step_taylor_degree_2, 0, None)
TAYLOR_DEGREE_3 = (one_step_taylor_degree_3, 0, None)
TAYLOR_DEGREE_4 = (one_step_taylor_degree_4, 0, None)
ADAMS_2_STEPS = (one_step_adams_two_steps_usage, 1, one_step_backward_euler)
ADAMS_3_STEPS = (one_step_adams_three_steps_usage, 2, one_step_backward_euler)
RUNGE_KUTTA = (one_step_runge_kutta, 0, None)


# SOLVER

def common_solver(method_description, a, b, steps_count, y0, y_, y__, y___):
    assert(isinstance(a, float))
    assert(isinstance(b, float))
    assert(a < b)
    assert(isinstance(steps_count, int))
    assert(isinstance(y0, float))

    x = a
    y = y0
    pair_list = [(x, y)]
    step = (b - a) / steps_count

    major_method, precalc_steps_count, precalc_method = method_description

    for _ in range(precalc_steps_count):
        x, y = x + step, precalc_method(x, y, step, y_)
        pair_list.append((x, y))

    for _ in range(steps_count - precalc_steps_count)
        additional_method_args = dict(
            y__ = y__,
            y___ = y___,
            prev1 = pair_list[-2] if len(pair_list) > 1 else None,
            prev2 = pair_list[-3] if len(pair_list) > 2 else None
        )
        x, y = x + step, major_method(x, y, step, y_, **additional_method_args)
        pair_list.append((x, y))

    return pair_list

if __name__ == '__main__':



def f1(x):
    B = 1  # random const from my head
    return math.sqrt(1 + x + x*x + B)


def f2(x):
    return 1/(1 + x*x)


def f3(x):
    return math.atan(x) / (1 + x*x*x)


def integral_core(fxy_tion, accumulate, left, right, step):
    assert(isinstance(left, float))
    assert(isinstance(right, float))
    assert(left < right and step > 0)
    integral_acc = 0
    while left < right:
        left_actual = min(left + step, right)
        integral_acc += accumulate(left, left_actual)
        left = left_actual
    return integral_acc


def rectangle_method(fxy_tion, left, right, step):
    def accumulator(a, b):
        return (b - a) * fxy_tion((a + b) / 2)
    return integral_core(fxy_tion, accumulator, left, right, step)


def trapezoidal_rule(fxy_tion, left, right, step):
    def accumulator(a, b):
        return (b - a) * (fxy_tion(a) + fxy_tion(b)) / 2
    return integral_core(fxy_tion, accumulator, left, right, step)


def simpson(fxy_tion, left, right, step):
    def accumulator(a, b):
        return (b - a) / 6 * (fxy_tion(a) + fxy_tion(b) + 4 * fxy_tion((a + b) / 2))
    return integral_core(fxy_tion, accumulator, left, right, step)



def solve_task01():
    left = 0.0
    right = 1.0
    step = 0.1
    table = ASCIITable(['method\computation', 'Sn[f]', 'S2n[f]', 'Runge'])
    for integral_method in [(rectangle_method, 2), (trapezoidal_rule, 2), (simpson, 4)]:
        integral, algebraic_accuracy = integral_method
        sn = integral(f1, left, right, step)
        s2n = integral(f1, left, right, step / 2)
        runge = abs(sn - s2n) / (2 ** algebraic_accuracy - 1)
        table.add_row([integral.fxy__name, sn, s2n, runge])
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
    return method.fxy__name.split('_')[0]


def plot_errors(errors):
    for name, values in errors.items():
        pylab.plot(range(len(values)), values, label=name)
    pylab.legend()
    pylab.show()


def solve_task03():
    eps = 0.005
    left = 0.0
    right = round(1 + math.sqrt(math.pi / (4 * eps))) # A(eps)
    max_f2c = 0.3256314123833 # M2, we know this a hard way ;[
    # maximize ((18 x^4)/(x^3+1)^3-(6 x)/(x^3+1)^2) tan^(-1)(x)-(6 x^2)/((x^2+1) (x^3+1)^2)-(2 x)/((x^2+1)^2 (x^3+1)) over [0, 14]
    step = math.pow(12 * eps / max_f2c, 1/3)

    table = ASCIITable(['compute integral with epsilon'])
    table.add_row([trapezoidal_rule(f3, left, right, step)])
    print 'TASK03'
    print table



if __name__ == '__main__':
    print 1
