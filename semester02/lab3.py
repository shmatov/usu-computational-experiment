

variant = 4


def initial_ddy(x, y, dy):
    alpha = 2 + 0.1 * variant
    return y + 2 * alpha + 2 + alpha * x * (1 - x)




