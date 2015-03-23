#!/usr/bin/env python
import math

"""

decomposition:
for lab 1.1:
1)Interface: method(Function function, double from, double to, double step) -> double
Srednie_pryamougolniki
Trapecii
Simpson

2)Rn[f] - ? I forgot how to count that
Rerror(Function function, double step) -> double

3)LaTeX: tags for tables


for lab 1.2:
<pick some N from imagination> [I recommend 15]
from i = 1 to N
    step = 1/i;
    using 1) from 1.1 use method(func, step)
using 3) from 1.1 make table
4)Make N decart graphs with point to point lines, where point is (x, f(x)), x from each step;


for lab 1.3:
use analytics to spent error and make integral a finite one.
N = 2;
while(abs(prev_integral - act_intergral) < epsilon)
    N = N * 2;
    prev_integral = act_intergral
    act_intergral = using any method 1) from 1.1 with step 1/N;
print act_intergral


"""

B = 1 #random const from my head

def f1(x):
    return math.sqrt(1 + x + x*x + B)

def f2(x):
    return 1/(1 + x*x)

def f3(x):
    return math.arctg(x) / (1 + x*x*x)


def integral_core(function, accumulate, left, right, step):
	assert(left < right and step > 0)
	integral_acc = 0
	while left < right:
		step_actual = min(left + step, right) - left
		integral_acc = integral_acc + accumulate(left, left + step_actual)
		left = left + step
	return integral_acc

def rectangle_method(function, left, right, step):
	def accumulator(a, b):
		return (b - a) * function((a + b) / 2)
	return integral_core(function, *accumulator, left, right, step)

def trapezoidal_rule(function, left, right, step):
	def accumulator(a, b):
		return (b - a) * (function(a) + function(b)) / 2
	return integral_core(function, *accumulator, left, right, step)

def simpson(function, left, right, step):
	def accumulator(a, b):
		return (b - a) / 6 * (function(a) + function(b) + 4 * function((a + b) / 2))
	return integral_core(function, *accumulator, left, right, step)



def step_for_interval(left, right, n):
    return (right - left) / n

def compute_intergral_with_eps(method, function, left, right, eps):
    N = 3 #bullshit
    prev_integral = method(function, left, right, step_for_interval(left, right, N - 1))
    current_integral = method(function, left, right, step_for_interval(left, right, N))
    while abs(prev_integral - current_integral) < eps:
        N = N + 1 #bullshit
        prev_integral = current_integral
        current_integral = method(function, left, right, step_for_interval(left, right, N))
    return current_integral




def solve_task01():
    left = 0
    right = 1

    print rectangle_method(*f1, left, right, 0.1)
    print rectangle_method(*f1, left, right, 0.05)
    print trapezoidal_rule(*f1, left, right, 0.1)
    print trapezoidal_rule(*f1, left, right, 0.05)
    print simpson(*f1, left, right, 0.1)
    print simpson(*f1, left, right, 0.05)
    # Rn - ?
    # LATEX TABLES

def solve_task02():
    pass


def solve_task03():
	eps = 0.005
    left = 0
    right = 13 # roundup(sqrt(pi / (4 * eps)))
    print compute_intergral_with_eps(*trapezoidal_rule, *f3, left, right, eps)


# document = LaTeX(); uncomment at global instance
if __name__ == '__main__':
    solve_task01();
    solve_task02();
    solve_task03();



