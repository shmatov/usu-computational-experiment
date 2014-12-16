#!/usr/bin/env python

from copy import deepcopy

def simple_equation_solver(matrix, vector):
    new_vector = [None] * 3
    new_vector[2] = vector[2][0] / matrix[2][2]
    new_vector[1] = (vector[1][0] - matrix[1][2] * new_vector[2]) / matrix[1][1]
    new_vector[0] = (vector[0][0] - matrix[0][1] * new_vector[1] - matrix[0][2] * new_vector[2]) / matrix[0][0]
    return new_vector

def gauss_transformation(matrix, vector):
    matrix = deepcopy(matrix)
    vector = deepcopy(vector)
    for column in range(0, 2):
        max = column;
        for line in range(column + 1, 3):
            if (abs(matrix[max][column]) < abs(matrix[x][column])):
            max = x
        if (max != column):
            matrix[column], matrix[max] = matrix[max], matrix[column]
            vector[column], vector[max] = vector[max], vector[column]
        for line in range(column + 1, 3):
            if (matrix[line][column] != Decimal(0.0)):
                koef = matrix[column][column] / matrix[line][column]
                vector[line][0] *= koef
                matrix[line][column] = 0;
                for rest_index in range(column + 1, 3):
                    matrix[line][rest_index] *= koef
                    matrix[line][rest_index] -= matrix[column][rest_index]
    return matrix, vector

def vector_distance(vector1, vector2):
    accum = 0.0
    for x in range(0, 3):
        diff = abs(vector1[x] - vector2[x])
        accum += diff * diff
    return math.sqrt(accum)
