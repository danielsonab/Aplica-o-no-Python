import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def f(y,x):
    Y, Yw = y
    A = q/(2*EI)
    dif = [Yw, A*(L*x-x**2)]
    return dif

EI = 1.4*10**7
q = 10*10**3
L = 4
Y0 = 0
Yw0 = 0
yi = [Y0, Yw0]
x = np.arange(0,4+0.25,0.25)

from scipy.optimize import fsolve
def g(Yw0):
    yi = [Y0, Yw0]
    sol = odeint(f,yi, x)
    sol = sol.T
    a = sol[0]
   
    return a[-1] 
Yw1, = fsolve(g,0)

yi = [Y0, Yw1]
sol = odeint(f, yi, x)
plt.figure()
plt.plot(x, sol[:,0], 'r-', label='Curva de deflexão')
plt.xlabel('Comprimento(m)')
plt.ylabel('Deflexão(m)')
plt.legend(bbox_to_anchor = (1.04,1))

vmax = min(sol[:,0])
xl = np.argmin(sol[:,0])
print('Deflexão máxima',vmax, 'em x =', x[xl])