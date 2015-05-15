from collections import defaultdict
import math
import sys

from numpy.linalg import solve, LinAlgError
import matplotlib.pyplot as plt
from scipy import interpolate

# THE OPTIONS

variant = 1
n = 10

# CONSTANT CONDITIONS

x0 = 0.0
xN = 1.0

alpha = 2 + 0.1 * variant

y0 = 0.0
dyN = math.e - (1.0 / math.e) + alpha


def h():
    return (xN - x0) / n


# TASK

def initial_ddy(x, y):
    return y + 2 * alpha + 2 + alpha * x * (1 - x)


def initial_solution_of_y(x):
    return -2 + alpha * x * (x - 1) + math.exp(-x) + math.exp(x)


def pair_list_for_solution(n2):
    h2 = (xN - x0) / n2
    x = x0
    pair_list = []
    for _ in range(n2 + 1):
        y = initial_solution_of_y(x)
        pair_list.append((x, y))
        x += h2
    return pair_list


# Tridiagonal matrix solution way

def lagrange_internal(last_i, constants, divider):
    assert isinstance(last_i, int)
    assert (len(constants) - 1) <= last_i <= n
    assert isinstance(divider, float)
    for const in constants:
        assert isinstance(const, float)
    result = [0.0] * (n + 1)
    for const in reversed(constants):
        result[last_i] = const
        last_i -= 1
    result = map(lambda x: x / divider, result)
    return result


def lagrange_dy_0h(last_i):
    return lagrange_internal(last_i, (-1.0, 1.0), h())


def lagrange_dy_0h2_by_first(last_i):
    return lagrange_internal(last_i, (-3.0, 4.0, -1.0), 2.0 * h())


def lagrange_dy_0h2_by_second(last_i):
    return lagrange_internal(last_i, (-2.0, 0.0, 2.0), 2.0 * h())


def lagrange_dy_0h2_by_third(last_i):
    return lagrange_internal(last_i, (1.0, -4.0, 3.0), 2.0 * h())


def lagrange_ddy_0h2(last_i):
    return lagrange_internal(last_i, (1.0, -2.0, 1.0), h() ** 2)


lagrange_usage = {
    "LAGRANGE: y'[O(h)], y''[O(h^2)]": {
        'dy': lagrange_dy_0h,
        'ddy': lagrange_ddy_0h2
    },
    "LAGRANGE: y'[O(h^2), L'(x[i-2]) by nodes i-2, i-1, i], y''[O(h^2)]": {
        'dy': lagrange_dy_0h2_by_first,
        'ddy': lagrange_ddy_0h2
    },
    "LAGRANGE: y'[O(h^2), L'(x[i-1]) by nodes i-2, i-1, i], y''[O(h^2)]": {
        'dy': lagrange_dy_0h2_by_second,
        'ddy': lagrange_ddy_0h2
    },
    "LAGRANGE: y'[O(h^2), L'(x[i]) by nodes i-2, i-1, i], y''[O(h^2)]": {
        'dy': lagrange_dy_0h2_by_third,
        'ddy': lagrange_ddy_0h2
    }
}
lagrange_usage_variant1 = {
    'dy': lagrange_dy_0h,
    'ddy': lagrange_ddy_0h2
}
lagrange_usage_variant2 = {
    'dy': lagrange_dy_0h2_by_third,
    'ddy': lagrange_ddy_0h2
}


def linear_dependence_y0():
    dependence = [0.0] * (n + 2)
    dependence[0] = 1.0
    dependence[n + 1] = 0.0
    return dependence


# noinspection PyPep8Naming
def linear_dependence_dyN(lagrange_options):
    dy_lagrange_to_use = lagrange_options['dy']
    dependence = dy_lagrange_to_use(n)
    dependence.append(dyN)
    return dependence


def linear_dependence_of_ddy(lagrange_options, i):
    ddy_lagrange_to_use = lagrange_options['ddy']

    dependence = ddy_lagrange_to_use(i + 1)
    dependence[i] -= 1.0
    x_i = x0 + i * h()
    dependence.append(2 * alpha + 2 + alpha * x_i * (1 - x_i))

    return dependence


def get_all_dependencies_matrix(lagrange_options):
    matrix = [linear_dependence_y0()]
    for i in range(1, n):
        matrix.append(linear_dependence_of_ddy(lagrange_options, i))
    matrix.append(linear_dependence_dyN(lagrange_options))
    return matrix


def print_mx_line_specific(mx_line):
    first = None
    for elem in mx_line:
        first = elem
        if first != 0.0:
            break
    mx_line = map(lambda x: x / first, mx_line)
    # print(mx_line)


def solve_dependencies_matrix(mx):
    vec = []
    for mx_line in mx:
        print_mx_line_specific(mx_line)
        vec.append(mx_line[-1])
        del mx_line[-1]
    solution = solve(mx, vec)

    pair_list = []
    x = x0
    for y in solution:
        pair_list.append((x, y))
        x += h()
    return pair_list


# THE SHOOTING METHOD


def cauchy_solution_of_fxyy(fxyy, y_start, dy_start):
    x = x0
    y, r = y_start, dy_start
    assert isinstance(y, float)
    assert isinstance(r, float)
    y_pairs_list = [(x, y)]
    r_pairs_list = [(x, r)]
    for _ in range(n):
        x += h()
        y += h() * r
        r = h() * fxyy(x, y)
        y_pairs_list.append((x, y))
        r_pairs_list.append((x, r))
    return y_pairs_list, r_pairs_list


eps = 0.00001


def interpolate_x_in_pair_list(x_fx_pairs, x_to_calculate):
    # lagrange by nodes
    xs = map(lambda pair: pair[0], x_fx_pairs)
    ys = map(lambda pair: pair[1], x_fx_pairs)
    xs[-1] = 1.0  # hack because of fucking 0.9999999999999
    return interpolate.interp1d(xs, ys)(x_to_calculate)


def calculate_next_phi(mu_n, fxyy):
    cauchy_solution = cauchy_solution_of_fxyy(fxyy, y0, mu_n)[1]
    return interpolate_x_in_pair_list(cauchy_solution, xN) - dyN


def calculate_next_dphi(mu_n, fxyy):
    y_mu_n = cauchy_solution_of_fxyy(fxyy, y0, mu_n)[0]
    y_mu_n_prev_step = cauchy_solution_of_fxyy(fxyy, y0, mu_n - h())[0]
    r = 1.0  # WHY? because the analytic of our fxyy says so: y = mu, y'_mu = 1
    y = 0.0  # WHY? because the analytic of our fxyy says so: other condition
    x = x0
    for _ in range(n):
        x += h()
        y += h() * r
        r += h() * calculate_dy_by_ymu_diff(y_mu_n, y_mu_n_prev_step, x)
    return r


def calculate_dy_by_ymu_diff(y_mu, y_mu_prev_step, x):
    return (interpolate_x_in_pair_list(y_mu, x) - interpolate_x_in_pair_list(y_mu_prev_step, x)) / h()


def shoot_method_solution(fxyy):
    mu_n = 0.0  # Kandoba said that

    while True:
        phi = calculate_next_phi(mu_n, fxyy)
        dphi = calculate_next_dphi(mu_n, fxyy)
        mu_n -= phi / dphi
        if abs(phi) < eps:
            break
    return cauchy_solution_of_fxyy(fxyy, y0, mu_n)[0]


# MAIN


def plot(title, data, block=True):
    plt.figure()
    ax = plt.subplot(1, 1, 1)

    for method_name, points in data.items():
        plt.plot(map(lambda x: x[0], points),
                 map(lambda x: x[1], points),
                 label=method_name)

    plt.title(title)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    ax.legend(bbox_to_anchor=(0., 1.1, 1., 0), loc=3, mode="expand",
              borderaxespad=0.)

    plt.show(block=block)


def compute(steps):
    global n
    old_n, n = n, steps

    data = {}
    for name, lagrange_options in lagrange_usage.items():
        try:
            data[name] = solve_dependencies_matrix(
                get_all_dependencies_matrix(lagrange_options)
            )
        except LinAlgError:
            pass
    data['SHOOTING METHOD'] = shoot_method_solution(initial_ddy)

    n = old_n
    return data


def compute_errors(steps_list):
    methods_errors = defaultdict(list)
    for steps in steps_list:
        methods = compute(steps)
        analytic = pair_list_for_solution(steps)
        for method_name, points in methods.iteritems():
            assert len(points) == len(analytic)
            assert all(
                map(
                    lambda (x1, x2): x1 == x2,
                    zip(
                        map(lambda (x, y): x, points),
                        map(lambda (x, y): x, analytic)
                    )
                )
            )

            error = sum(
                map(
                    lambda (y1, y2): (y1 - y2) ** 2,
                    zip(
                        map(lambda (x, y): y, points),
                        map(lambda (x, y): y, analytic)
                    )
                )
            )
            methods_errors[method_name].append(
                (steps, math.sqrt(error) / steps)
            )
    return methods_errors



def get_mode():
    try:
        if sys.argv[1] == 'errors':
            return 'errors'
    except IndexError:
        pass
    return 'interactive'


def main():
    mode = get_mode()
    if mode == 'errors':
        errors = compute_errors(range(10, 100, 5))
        plot('errors', errors)

    elif mode == 'interactive':
        steps = n
        data = compute(steps)
        plot('steps = {}'.format(steps), dict(
            {
                'ANALYTIC': pair_list_for_solution(10000)
            }.items() + data.items()
        ))


if __name__ == '__main__':
    main()
