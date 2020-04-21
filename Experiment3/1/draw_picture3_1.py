import math
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,15,5000)

def func(x):
    k = 0.2 #阻尼比系数
    b = math.acos(k) #阻尼角
    wn = 1 #无阻尼振荡角频率
    c1 = math.sqrt(1-k**2)  
    wd = wn*c1 #阻尼振荡角频率
    ans = 1-math.exp(-k*wn*x)*math.sin(wd*x+b)/c1
    return ans
y = list(map(func,t))
print(math.cos(math.pi))
plt.xlabel('t')
plt.ylabel('y')
plt.plot(t,y)
plt.plot([0,15],[1,1],color='red', linewidth=1, linestyle='--')

plt.show()
