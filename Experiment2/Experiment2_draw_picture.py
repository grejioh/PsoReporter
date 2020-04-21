import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
#构建(x,y,z)点集
L = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(L, L)
Z = np.exp((X**2+Y**2)*(-0.5))/(2*np.pi)
#绘制
ax.plot_surface(X, Y, Z, rstride= 1, cstride=1, cmap=plt.get_cmap('rainbow'),alpha=0.8)
#投影，显示出等高线
ax.contourf(X, Y, Z, zdir='z', offset=-0.1, cmap=plt.get_cmap('rainbow'),alpha=0.8)

ax.set_xlabel(r'$x1$')
ax.set_ylabel(r'$x2$')
ax.set_zlabel(r'$P(x)$')
ax.set_zlim(-0.1, 0.20)
plt.show()
