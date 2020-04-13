import math
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4,4,100)

def func(x):
    c1=1/(math.sqrt(2*math.pi)*1)
    c2=-(x**2)/2
    return c1*math.exp(c2)

y = list(map(func,x))
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y)

plt.show()
