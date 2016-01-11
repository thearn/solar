from sympy import *
import numpy as np 

"""
derivative derivations
"""

cap = symbols("cap")
gen1,gen2,gen3,gen4,gen5 = symbols("gen1 gen2 gen3 gen4 gen5")
con1,con2,con3,con4,con5 = symbols("con1 con2 con3 con4 con5")

SOC = 1.0

old_SOC = SOC
available = SOC * cap
generated = gen1
consumed = con1

delta = available + generated - consumed
SOC = (delta) / cap

soc1 = (SOC + SOC)/2.0 

# -----------------------
old_SOC = SOC
available = SOC * cap
generated = gen2
consumed = con2

delta = available + generated - consumed
SOC = (delta) / cap

soc2 = (SOC + soc1)/2.0 


# -----------------------

old_SOC = SOC
available = SOC * cap
generated = gen3
consumed = con3

delta = available + generated - consumed
SOC = (delta) / cap

soc3 = (SOC + soc2)/2.0 

# ------------------------
old_SOC = SOC
available = SOC * cap
generated = gen4
consumed = con4

delta = available + generated - consumed
SOC = (delta) / cap

soc4 = (SOC + soc3)/2.0 
#-----------------------
old_SOC = SOC
available = SOC * cap
generated = gen5
consumed = con5

delta = available + generated - consumed
SOC = (delta) / cap

soc5 = (SOC + soc4)/2.0 

idx_1 = -np.array([1.0] + [1. - 1./2.0**i for i in xrange(1, 1)][::-1])/cap
idx_2 = -np.array([1.0] + [1. - 1./2.0**i for i in xrange(1, 2)][::-1])/cap

idx_5 = -np.array([1.0] + [1. - 1./2.0**i for i in xrange(1, 5)][::-1])/cap
idx_4 = -np.array([1.0] + [1. - 1./2.0**i for i in xrange(1, 4)][::-1])/cap

print 10*"----"
print diff(soc1, gen1).factor()
print
print diff(soc1, gen2).factor()
print
print diff(soc1, gen3).factor()
print
print diff(soc1, gen4).factor()
print
print diff(soc1, gen5).factor()
print


print 10*"----"
print diff(soc2, gen1).factor()
print
print diff(soc2, gen2).factor()
print
print diff(soc2, gen3).factor()
print
print diff(soc2, gen4).factor()
print
print diff(soc2, gen5).factor()
print


print 10*"----"
print diff(soc3, gen1).factor()
print
print diff(soc3, gen2).factor()
print
print diff(soc3, gen3).factor()
print
print diff(soc3, gen4).factor()
print
print diff(soc3, gen5).factor()
print


print 10*"----"
print diff(soc4, gen1).factor()
print
print diff(soc4, gen2).factor()
print
print diff(soc4, gen3).factor()
print
print diff(soc4, gen4).factor()
print
print diff(soc4, gen5).factor()
print



print 10*"----"
print diff(soc5, gen1).factor()
print
print diff(soc5, gen2).factor()
print
print diff(soc5, gen3).factor()
print
print diff(soc5, gen4).factor()
print
print diff(soc5, gen5).factor()
print

n = 100
x = np.linspace(0, n - 1, n)
y = np.linspace(0, n - 1, n)

idx = np.where((x == (n - 1) | (x >= 1023)))
x[idx] = 1023
y[idx] = 1023


xx, yy = np.meshgrid(x, y)
A =  (xx >= yy ) * (1. - 1./2.0**(-yy + xx + 1))
A = A[::-1, ::-1]
A[0,0] = 1.0

print A

"""
print ((1- soc1)/cap).factor()
print diff(soc1, cap).factor()
print
print ((1- soc2)/cap).factor()
print diff(soc2, cap).factor()
print
print ((1- soc3)/cap).factor()
print diff(soc3, cap).factor()
print
print ((1- soc4)/cap).factor()
print diff(soc4, cap).factor()
print
print ((1- soc5)/cap).factor()
print diff(soc5, cap).factor()
"""