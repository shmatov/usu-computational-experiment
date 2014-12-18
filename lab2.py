#!/usr/bin/env python
import math


eps = 0.5 * 10**-5
interval = (0, 2)
func_plots = []


def sign(x):
    return 1 if x >= 0 else -1


def f(x):
    """f(x) = 2*cos(x) - e^x"""
    return 2 * math.cos(x) - math.e ** x


def df(x):
    """f'(x) = -2*sin(x) - e^x"""
    return -2 * math.sin(x) - math.e ** x


def f2(x):
    """1 + sin(x) - 1.2*e^-x"""
    return 1 + math.sin(x) - 1.2 / math.e ** x


def df2(x):
    """cos(x) + 1.2*e^-x"""
    return math.cos(x) + 1.2 / math.e ** x


def dichotomy():
    a, b = interval
    x = a
    assert(sign(f(a)) != sign(f(b)))
    i = []
    while abs(b - a) > eps:
        x = (a + b) / 2.
        i.append(x)
        if sign(f(x)) == sign(f(a)):
            a = x
        if sign(f(x)) == sign(f(b)):
            b = x
    return x, i


def newton():
    x0, _ = interval
    i = []
    while True:
        x1 = x0 - (f(x0) / df(x0))
        i.append(x1)
        if abs(x1 - x0) < eps:
            return x1, i
        x0 = x1


def static_hords():
    x0, x_curr = interval
    i = []
    while True:
        x_next = x_curr - (f(x_curr)*(x0 - x_curr)) / (f(x0) - f(x_curr))
        i.append(x_next)
        if abs(x_curr - x_next) < eps:
            return x_next, i
        x_curr = x_next


def moving_hords():
    x_prev, x_curr = interval
    i = []
    while True:
        x_next = x_curr - (f(x_curr)*(x_curr - x_prev)) / (f(x_curr) - f(x_prev))
        i.append(x_next)
        if abs(x_curr - x_next) < eps:
            return x_next, i
        x_prev, x_curr = x_curr, x_next

def muller():
	_, i = dichotomy()
	i = i[:3]
	while abs(i[-1] - i[-2]) > eps:
		x1, x2, x3 = i[-3:] #Xk-2, Xk-1, Xk 
		q = (x3 - x2) / (x2 - x1)
		A = q * f(x3) - q * (1 + q) * f(x2) + q * q * f(x1)
		B = (2 * q + 1) * f(x3) - (1 + q) * (1 + q) * f(x2) + q * q * f(x1)
		C = (1 + q) * f(x3)
		expr1 = B + math.sqrt(B * B - 4 * A * C)
		expr2 = B - math.sqrt(B * B - 4 * A * C)
		expr = expr1 if (abs(expr1) > abs(expr2)) else expr2
		x_next = x3 - (x3 - x2) * 2 * C / expr
		i.append(x_next)
	return i[-1], i


if __name__ == '__main__':
    def fmt(*cells):
        return '|' + '|'.join(map("{:^20}".format, cells)) + '|'

    def compute(method):
        result, iterations = method()
        iterations = zip(iterations, map(f, iterations))
        func_plots.append((method.func_name, iterations))
        # iterations == [(x0, f(x0), .., (xn, f(xn))]
        return method.func_name, result, len(iterations)


    print f.func_doc
    print df.func_doc
    print 'interval:', list(interval)
    print 'eps:', eps

    print '-'*64
    print fmt('method', 'result', 'iterations')
    print '-'*64
    print fmt(*compute(dichotomy))
    print fmt(*compute(static_hords))
    print fmt(*compute(moving_hords))
    print fmt(*compute(newton))
    print fmt(*compute(muller))
    print '-'*64

    for func in func_plots:
    	print func[0]
    	print 'plot{' + ', '.join(map(lambda (_, y): "{}".format(abs(y)), func[1])) + '}'
