#!/usr/bin/env python
import math
from table import ASCIITable
import pylab


def sign(num):
    return math.copysign(1, num)


# TASK INITIAL

initial_y0 = 0.1
initial_a = 0.0
initial_b = 1.0

def initial_dy(x, y):
    return 30 * y * (x - 0.2) * (x - 0.7)

def initial_ddy(x, y):
    return 30 * (initial_dy(x, y) * (x - 0.2) * (x - 0.7) + y * (2 * x - 0.9))

def initial_dddy(x, y):
    return 30 * (initial_ddy(x, y) * (x - 0.2) * (x - 0.7) + 2 * initial_dy(x, y) * (2 * x - 0.9) + 2 * y)

def initial_euler_backwards(x, y, step):
    return y / (1 - step * (x + step - 0.2) * (x + step - 0.7))

def initial_solution_of_y(x):
    return math.pow(math.exp, x * (x ** 2 / 3 - 0.45 * x + 0.14))


# ONE STEP CALCULATIONS

def one_step_explicit_euler(x, y, step, dy, **_):
    return y + step * dy(x, y)

def one_step_recalculation_euler(x, y, step, dy, **_):
    y_new = one_step_explicit_euler(x, y, step, dy)
    return y + step * (dy(x, y) + dy(x + step, y_new)) / 2

def one_step_implicit_euler(x, y, step, dy, dy_euler_backwards, **_):
    return dy_euler_backwards(x, y, step)

def one_step_cauchy(x, y, step, dy, **_):
    y_plus_halfstep = y + step * dy(x, y) / 2
    return y + step * dy(x + step / 2, y_plus_halfstep)

def one_step_taylor_degree_2(x, y, step, dy, **_):
    return y + step * dy(x, y)

def one_step_taylor_degree_3(x, y, step, dy, ddy, **_):
    return y + step * dy(x, y) + step ** 2 * ddy(x, y)

def one_step_taylor_degree_4(x, y, step, dy, ddy, dddy, **_):
    return y + step * dy(x, y) + step ** 2 * ddy(x, y) + step ** 3 * dddy(x, y)

def one_step_adams_two_steps_usage(x, y, step, dy, prev1, **_):
    x_prev1, y_prev1 = prev1
    return y + step * (3 * dy(x, y) - dy(x_prev1, y_prev1)) / 2

def one_step_adams_three_steps_usage(x, y, step, dy, prev1, prev2, **_):
    x_prev1, y_prev1 = prev1
    x_prev2, y_prev2 = prev2
    return y + step * (23 * dy(x, y) - 16 * dy(x_prev1, y_prev1) + 5 * dy(x_prev2, y_prev2)) / 12

def one_step_runge_kutta(x, y, step, dy, **kwargs):
    k_1 = dy(x, y)
    k_2 = dy(x + step / 2, y + step / 2 * k_1)
    k_3 = dy(x + step / 2, y + step / 2 * k_2)
    k_4 = dy(x + step, y + step * k_3)
    return y + step / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


# TIED UP ONE STEP METHODS AND ACCELERATORS

def EXPLICIT_EULER():
    return (one_step_explicit_euler, 0, None)
def RECALCULATION_EULER():
    return (one_step_recalculation_euler, 0, None)
def IMPLICIT_EULER():
    return (one_step_implicit_euler, 0, None)
def CAUCHY():
    return (one_step_cauchy, 0, None)
def TAYLOR_DEGREE_2():
    return (one_step_taylor_degree_2, 0, None)
def TAYLOR_DEGREE_3():
    return (one_step_taylor_degree_3, 0, None)
def TAYLOR_DEGREE_4():
    return (one_step_taylor_degree_4, 0, None)
def ADAMS_2_STEPS():
    return (one_step_adams_two_steps_usage, 1, one_step_recalculation_euler)
def ADAMS_3_STEPS():
    return (one_step_adams_three_steps_usage, 2, one_step_recalculation_euler)
def RUNGE_KUTTA():
    return (one_step_runge_kutta, 0, None)


# SOLVER

def common_solver(method_description, a, b, steps_count, y0, dy, ddy, dddy, dy_euler_backwards):
    assert(isinstance(a, float))
    assert(isinstance(b, float))
    assert(a < b)
    assert(isinstance(steps_count, int) and steps_count > 2)
    assert(isinstance(y0, float))

    x = a
    y = y0
    pair_list = [(x, y)]
    step = (b - a) / steps_count

    major_method, precalc_steps_count, precalc_method = method_description

    for _ in range(precalc_steps_count):
        x, y = x + step, precalc_method(x, y, step, dy)
        pair_list.append((x, y))

    for _ in range(steps_count - precalc_steps_count):
        additional_method_kwargs = {
            'ddy': ddy,
            'dddy': dddy,
            'prev1': pair_list[-2] if len(pair_list) > 1 else None,
            'prev2': pair_list[-3] if len(pair_list) > 2 else None,
            'dy_euler_backwards': dy_euler_backwards
        }
        x, y = x + step, major_method(x, y, step, dy, **additional_method_kwargs)
        pair_list.append((x, y))

    return pair_list


# MAIN

def get_plot_list_for_method(method, n):
    return common_solver(method(),
            initial_a,
            initial_b,
            n,
            initial_y0,
            initial_dy,
            initial_ddy,
            initial_dddy,
            initial_euler_backwards)


if __name__ == '__main__':
    methods = [
        EXPLICIT_EULER,
        RECALCULATION_EULER,
        IMPLICIT_EULER,
        CAUCHY,
        TAYLOR_DEGREE_2,
        TAYLOR_DEGREE_3,
        TAYLOR_DEGREE_4,
        ADAMS_2_STEPS,
        ADAMS_3_STEPS,
        RUNGE_KUTTA
    ]

    n = 100
    benchmark_run_count = 10000
    for _ in range(benchmark_run_count):
        for method in methods:
            plot_list = get_plot_list_for_method(method, n)

    # # TO SEE IF IT WORK
    # for method in methods:
    #     print method.func_name
    #     table = ASCIITable(['x', 'y'])
    #     pair_list = get_plot_list_for_method(method, n)
    #     for pair in pair_list:
    #         table.add_row([pair[0], pair[1]])
    #     print table
