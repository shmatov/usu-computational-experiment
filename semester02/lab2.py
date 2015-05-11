#!/usr/bin/env python
import math  # noqa
import matplotlib.pyplot as plt


# INITIAL DATA

# COMMON

initial_a = 0.0
initial_b = 1.0
variant = 4


# TASK A

task_a_initial_y0 = 0.1


def task_a_initial_dy(x, y):
    return 30 * y * (x - 0.2) * (x - 0.7)


def task_a_initial_ddy(x, y):
    return 30 * (task_a_initial_dy(x, y) * (x - 0.2) * (x - 0.7) + y * (2 * x - 0.9))


def task_a_initial_dddy(x, y):
    return 30 * (task_a_initial_ddy(x, y) * (x - 0.2) * (x - 0.7) + 2 * task_a_initial_dy(x, y) * (2 * x - 0.9) + 2 * y)


def task_a_initial_euler_backwards(x, y, step):
    return y / (1 - 30 * step * (x + step - 0.2) * (x + step - 0.7))


def task_a_initial_solution_of_y(x):
    return math.exp(x * (10 * x ** 2 - 13.5 * x + 4.2)) / 10


TASK_A = {
    'a': initial_a,
    'b': initial_b,
    'y0': task_a_initial_y0,
    'dy': task_a_initial_dy,
    'ddy': task_a_initial_ddy,
    'dddy': task_a_initial_dddy,
    'euler_backwards': task_a_initial_euler_backwards,
    'solution_of_y': task_a_initial_solution_of_y
}

# TASK B

task_b_initial_y0 = 0.1


def task_b_initial_dy(x, y):
    return 50 * y * (x - 0.6) * (x - 0.85)


def task_b_initial_ddy(x, y):
    return 50 * (
        task_b_initial_dy(x, y) * (x - 0.6) * (x - 0.85) +
        y * (2 * x - 1.45)
    )


def task_b_initial_dddy(x, y):
    return 50 * (
        task_b_initial_ddy(x, y) * (x - 0.6) * (x - 0.85) +
        2 * task_b_initial_dy(x, y) * (2 * x - 1.45) +
        2 * y
    )


def task_b_initial_euler_backwards(x, y, step):
    return y / (1 - 50 * step * (x + step - 0.6) * (x + step - 0.85))


def task_b_initial_solution_of_y(x):
    return math.exp(x * (50 / 3. * x ** 2 - 36.25 * x + 25.5)) / 10


TASK_B = {
    'a': initial_a,
    'b': initial_b,
    'y0': task_b_initial_y0,
    'dy': task_b_initial_dy,
    'ddy': task_b_initial_ddy,
    'dddy': task_b_initial_dddy,
    'euler_backwards': task_b_initial_euler_backwards,
    'solution_of_y': task_b_initial_solution_of_y
}


# TASK C

task_c_initial_y0 = 0.5


def task_c_initial_dy(x, y):
    return -20 * y ** 2 * (x - (variant * 0.1))


def task_c_initial_ddy(x, y):
    return -20 * (2 * task_c_initial_dy(x, y) * (x - (variant * 0.1)) + y ** 2)


def task_c_initial_dddy(x, y):
    return -20 * (
        2 * task_c_initial_ddy(x, y) * (x - (variant * 0.1)) + 2 * task_c_initial_dy(x, y) + 2 * y
    )


def task_c_initial_euler_backwards(x, y, step):
    return (
        (-1 + math.sqrt(1 + 80 * step * (x + step - (variant * 0.1)) * y)) /
        (40 * step * (x + step - (variant * 0.1)))
    )


def task_c_initial_solution_of_y(x):
    return - 0.05 / ((variant * 0.1) * x - 0.5 * x ** 2 - 0.1)


TASK_C = {
    'a': initial_a,
    'b': initial_b,
    'y0': task_c_initial_y0,
    'dy': task_c_initial_dy,
    'ddy': task_c_initial_ddy,
    'dddy': task_c_initial_dddy,
    'euler_backwards': task_c_initial_euler_backwards,
    'solution_of_y': task_c_initial_solution_of_y
}


# ONE STEP CALCULATIONS

def one_step_explicit_euler(x, y, step, dy, **_):
    return y + step * dy(x, y)


def one_step_recalculation_euler(x, y, step, dy, **_):
    y_new = one_step_explicit_euler(x, y, step, dy)
    return y + step * (dy(x, y) + dy(x + step, y_new)) / 2


def one_step_implicit_euler(x, y, step, dy, euler_backwards, **_):
    return euler_backwards(x, y, step)


def one_step_cauchy(x, y, step, dy, **_):
    y_plus_half_step = y + step * dy(x, y) / 2
    return y + step * dy(x + step / 2, y_plus_half_step)


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


def one_step_runge_kutta(x, y, step, dy, **_):
    k_1 = dy(x, y)
    k_2 = dy(x + step / 2, y + step / 2 * k_1)
    k_3 = dy(x + step / 2, y + step / 2 * k_2)
    k_4 = dy(x + step, y + step * k_3)
    return y + step / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


def one_step_analytic_solution(x, y, step, dy, solution_of_y, **_):
    return solution_of_y(x + step)


# TIED UP ONE STEP METHODS AND ACCELERATORS

METHODS = {
    'EXPLICIT_EULER': (one_step_explicit_euler, 0, None),
    'RECALCULATION_EULER': (one_step_recalculation_euler, 0, None),
    'IMPLICIT_EULER': (one_step_implicit_euler, 0, None),
    'CAUCHY': (one_step_cauchy, 0, None),
    'TAYLOR_DEGREE_2': (one_step_taylor_degree_2, 0, None),
    'TAYLOR_DEGREE_3': (one_step_taylor_degree_3, 0, None),
    'TAYLOR_DEGREE_4': (one_step_taylor_degree_4, 0, None),
    'ADAMS_2_STEPS': (one_step_adams_two_steps_usage, 1, one_step_recalculation_euler),
    'ADAMS_3_STEPS': (one_step_adams_two_steps_usage, 2, one_step_recalculation_euler),
    'RUNGE_KUTTA': (one_step_runge_kutta, 0, None),
    'ANALYTIC_SOLUTION': (one_step_analytic_solution, 0, None)
}


# SOLVER

def common_solver(method, steps_count, a, b, y0, dy, ddy, dddy, euler_backwards, solution_of_y):
    assert (isinstance(a, float))
    assert (isinstance(b, float))
    assert (a < b)
    assert (isinstance(steps_count, int) and steps_count > 2)
    assert (isinstance(y0, float))

    x = a
    y = y0
    pair_list = [(x, y)]
    step = (b - a) / steps_count

    major_method, precalc_steps_count, precalc_method = method

    for _ in range(precalc_steps_count):
        x, y = x + step, precalc_method(x, y, step, dy)
        pair_list.append((x, y))

    for _ in range(steps_count - precalc_steps_count):
        additional_method_kwargs = {
            'ddy': ddy,
            'dddy': dddy,
            'prev1': pair_list[-2] if len(pair_list) > 1 else None,
            'prev2': pair_list[-3] if len(pair_list) > 2 else None,
            'euler_backwards': euler_backwards,
            'solution_of_y': solution_of_y
        }
        x, y = x + step, major_method(x, y, step, dy, **additional_method_kwargs)
        pair_list.append((x, y))

    return pair_list


# MAIN

def plot(methods, steps):
    plt.figure()
    ax = plt.subplot(111)

    for method_name, points in methods.items():
        plt.plot(map(lambda x: x[0], points),
                 map(lambda x: x[1], points),
                 label=method_name)

    plt.title('steps = {}'.format(steps))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show(block=False)


def compute(steps):
    data = {}
    for name, method in METHODS.iteritems():
        points = common_solver(
            method,
            steps if name != 'ANALYTIC_SOLUTION' else 10000,
            **TASK_C
        )
        data[name] = points
    plot(data, steps)


if __name__ == '__main__':
    while True:
        try:
            steps = int(raw_input('steps = '))
        except:
            break
        compute(steps)
