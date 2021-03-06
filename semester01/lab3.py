#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
import math
import sys

from latex import LatexDocument, Math, Section, Text
from fixed_precision import Num


doc = LatexDocument()


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

            for i in xrange(j + 1):
                s1 = sum(u[k][j] * l[i][k] for k in xrange(i))
                u[i][j] = self[i][j] - s1

            for i in xrange(j, n):
                s2 = sum(u[k][j] * l[i][k] for k in xrange(j))
                l[i][j] = (self[i][j] - s2) / u[j][j]

        return l, u

    def to_tex(self):
        data = r' \\ '.join(' & '.join(map(str, row)) for row in self)
        return r'\begin{pmatrix}' + data + '\end{pmatrix}'


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
    return matrix, vector, answer


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


def solver_1(mx, vec):
    l, u = mx.lu_decomposition()
    y = upper_zeros_simple_equation_solver(l, vec)
    x = lower_zeros_simple_equation_solver(u, y)

    doc.add(Text('Ax = b'))
    doc.add(Text('Выполним LU-разложение для матрицы A. A = LU'))
    doc.add(Math('L = ', l))
    doc.add(Math('U = ', u))
    doc.add(Text('Получаем LUx = b'))
    doc.add(Text('Считая, что Ux = y, решим систему Ly = b подстановкой:'))
    doc.add(Math('y = ', y))
    doc.add(Text('Теперь решим систему Ux = y.'))
    return x


def solver_2(mx, vec):
    m_gauss, v_gauss = gauss_transformation(mx, vec)
    x = lower_zeros_simple_equation_solver(m_gauss, v_gauss)

    doc.add(Text('После приведения к нижнетреугольной матрице:'))
    doc.add(Math(r'\acute{A} = ', m_gauss))
    doc.add(Math(r'\acute{b} = ', v_gauss))
    return x


if __name__ == '__main__':
    n = int(sys.argv[1])
    original_mx, original_vec, ans = generate_matrix_and_vector_and_answer(n)


    doc.add(Section('Решение Ax = b  различными методами для N={}(вариант)'.format(n)))
    doc.add(Math('A = ', original_mx.map('{:.8g}'.format)))
    doc.add(Math('b = ', original_vec.map('{:.8g}'.format)))
    doc.add(Text('Точный ответ:'))
    doc.add(Math(r'\bar{x} = ', ans))

    for method in [1, 2]:
        for k in [2, 4, 6]:
            Num.precision = k
            mx = original_mx.map(Num)
            vec = original_vec.map(Num)
            slv_name = 'компактной схемы Гаусса(LU-разложение)' if method == 1 else 'Гаусса с выбором главного элемента'
            doc.add(Text(''))
            doc.add(Text(''))
            doc.add(Section('Решение системы методом {} при k={}(точность)'.format(slv_name, k)))
            calc_ans = solver_1(mx, vec) if method == 1 else solver_2(mx, vec)
            doc.add(Text('Вычисленное значение:'))
            doc.add(Math(r'\tilde{x} = ', calc_ans))
            doc.add(Math(r'||\bar{x} - \tilde{x}|| = ', vector_distance(ans, calc_ans.map(float))))
    print doc
