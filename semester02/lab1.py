#!/usr/bin/env python
import math

"""

decomposition:
for lab 1.1:
1)Interface: method(Function function, double step) -> double
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

if __name__ == '__main__':
    print 1