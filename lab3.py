#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
from decimal import Decimal, ROUND_DOWN
import math

from latex import LatexDocument


WIDTH = 80


class Matrix(object):
    def __init__(self, matrix):
        self._matrix = matrix

    def __iter__(self):
        return iter(self._matrix)

    def __mul__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError(
                '{} * {}'.format(type(self).__name__, type(other).__name__)
            )

        return Matrix([
            [
                sum(a * b for a, b in zip(self_row, other_col))
                for other_col in zip(*other)
            ]
            for self_row in self
        ])

    def __getitem__(self, index):
        return self._matrix[index]

    def __setitem__(self, index, value):
        self._matrix[index] = value
        return value

    def __repr__(self):
        max_lengths = [max(len(str(x)) for x in col) for col in zip(*self)]
        return '\n'.join([
            '  '.join([
                str(x).rjust(l) for x, l in zip(row, max_lengths)
            ])
            for row in self
        ])

    def __len__(self):
        return len(self._matrix)

    @property
    def rows(self):
        return len(self._matrix)

    @property
    def cols(self):
        return len(self[0])

    def map(self, func):
        return Matrix([[func(x) for x in row] for row in self])

    def lu_decomposition(self):
        n = self.rows

        l = self.map(lambda x: x - x)
        u = self.map(lambda x: x - x)

        for j in xrange(n):
            l[j][j] = abs(self[j][j] / self[j][j])

            for i in xrange(j+1):
                s1 = sum(u[k][j] * l[i][k] for k in xrange(i))
                u[i][j] = self[i][j] - s1

            for i in xrange(j, n):
                s2 = sum(u[k][j] * l[i][k] for k in xrange(j))
                l[i][j] = (self[i][j] - s2) / u[j][j]

        return l, u

    def to_tex(self):
        data = r' \\ '.join(' & '.join(map(str, row)) for row in self)
        return r'\begin{pmatrix}' + data + '\end{pmatrix}'


class Num(Decimal):
    precision = 2

    def __new__(cls, value):
        if isinstance(value, float):
            value = ('{:.' + str(cls.precision) + 'f}').format(value)

        return super(Num, cls).__new__(cls, value)

    def _quantize(self, value):
        if isinstance(value, Decimal):
            decimal_precision = Decimal('.' + '0' * self.precision)
            return Num(value.quantize(decimal_precision, rounding=ROUND_DOWN))
        return value

    def __mul__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__mul__(*args, **kwargs))

    def __add__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__add__(*args, **kwargs))

    def __sub__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__sub__(*args, **kwargs))

    def __div__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__div__(*args, **kwargs))

    def __mod__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__mod__(*args, **kwargs))

    def __pow__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__pow__(*args, **kwargs))

    def __repr__(self):
        return 'Num({})'.format(self)


def generate_matrix_and_vector_and_answer(n):
    m = 21 - n
    a = 0.1 * m + 0.01 * n
    b = 0.2 * m + 0.02 * n + 0.01 * m * n + 0.001 * n * n
    matrix = Matrix([
        [1.2345, 3.1415, 1],
        [2.3456, 5.9690, 0],
        [3.4567, 2.1828, (2 + 0.1 * n)]
    ])
    vector = Matrix([
        [7.5175 + a],
        [14.2836],
        [7.8223 + b]
    ])
    answer = Matrix([
        [1],
        [2],
        [0.1 * m + 0.01 * n]
    ])
    return matrix.map(Num), vector.map(Num), answer


def lower_zeros_simple_equation_solver(mx, vec):
    nvec = [None] * 3
    nvec[2] = vec[2][0] / mx[2][2]
    nvec[1] = (vec[1][0] - mx[1][2] * nvec[2]) / mx[1][1]
    nvec[0] = (vec[0][0] - mx[0][1] * nvec[1] - mx[0][2] * nvec[2]) / mx[0][0]
    return Matrix([[x] for x in nvec])

def upper_zeros_simple_equation_solver(mx, vec):
    nvec = [None] * 3
    nvec[0] = vec[0][0] / mx[0][0]
    nvec[1] = (vec[1][0] - mx[1][0] * nvec[0]) / mx[1][1]
    nvec[2] = (vec[2][0] - mx[2][1] * nvec[1] - mx[2][0] * nvec[0]) / mx[2][2]
    return Matrix([[x] for x in nvec])


def gauss_transformation(matrix, vector):
    matrix = deepcopy(matrix)

    vector = deepcopy(vector)
    for column in range(0, 2):
        maximum = column
        for line in range(column + 1, 3):
            if abs(matrix[maximum][column]) < abs(matrix[line][column]):
                maximum = line
        if maximum != column:
            matrix[column], matrix[maximum] = matrix[maximum], matrix[column]
            vector[column], vector[maximum] = vector[maximum], vector[column]
        for line in range(column + 1, 3):
            if matrix[line][column] != 0:
                k = matrix[column][column] / matrix[line][column]
                vector[line][0] *= k
                vector[line][0] -= vector[column][0]
                matrix[line][column] = 0
                for rest_index in range(column + 1, 3):
                    matrix[line][rest_index] *= k
                    matrix[line][rest_index] -= matrix[column][rest_index]
    return matrix, vector


def vector_distance(v1, v2):
    squares = map(lambda (x, y): float(x[0] - y[0]) ** 2, zip(v1, v2))
    return math.sqrt(sum(squares))


def show(name, matrix):
    print ''
    print name
    print repr(matrix)


def solver_1(mx, vec):
    print ''
    print 'Method 1'.center(WIDTH)
    print ''
    print '-' * WIDTH

    print 'LU decomposition.'
    l, u = mx.lu_decomposition()
    show('L', l)
    y = upper_zeros_simple_equation_solver(l, vec)
    show('y', y)

    show('U', u)
    x = lower_zeros_simple_equation_solver(u, y)
    show('x', x)
    # show('LU', l * u)

    print 'Diff between real and calculated: {}'.format(
        vector_distance(ans, x.map(float))
    )
    print '=' * WIDTH


def solver_2(mx, vec):
    print ''
    print 'Method 2'.center(WIDTH)
    print ''
    print '-' * WIDTH

    m_gauss, v_gauss = gauss_transformation(mx, vec)
    print 'After gauss transformation.'
    show('Matrix:', m_gauss)
    show('Vector:', v_gauss)
    print '-' * WIDTH

    calc_ans = lower_zeros_simple_equation_solver(m_gauss, v_gauss)
    print 'Answer after calc:\n'
    print repr(calc_ans)
    print '-' * WIDTH

    print 'Diff between real and calculated: {}'.format(
        vector_distance(ans, calc_ans.map(float))
    )
    print '=' * WIDTH


if __name__ == '__main__':
    n = int(raw_input('Solve task for N = '))
    mx, vec, ans = generate_matrix_and_vector_and_answer(n)

    show('Matrix:', mx)
    show('Vector:', vec)
    show('Answer:', ans)
    print '=' * WIDTH

    solver_1(mx, vec)
    solver_2(mx, vec)
