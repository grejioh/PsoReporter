import math
from random import random
import numpy as np
import matplotlib.pyplot as plt
class Pso():
    def __init__(self,pN,dim,max_iter,func): 
        self.w = 0.5 #惯性因子
        self.c1 = 1.5 #自身认知因子
        self.c2 = 1.5 #社会任质因子
        self.pN = pN  #粒子数量
        self.maxv = 0.5 #最大速度
        self.dim = dim #维数
        self.max_iter = max_iter #迭代维数
        self.X = np.zeros((self.pN,self.dim)) #各维坐标，初值为全为0
        self.V = np.zeros((self.pN,self.dim)) #各维速度，初值为全为0
        self.pbest = np.zeros((self.pN,self.dim)) #粒子最优位置
        self.gbest = np.zeros((1,self.dim)) #全局最优位置
        self.p_bestfit = np.zeros(self.pN) #粒子最佳位置对应值
        self.fit = -1e15 #为求最大值，初始设为一足够小的值
        self.func = func #待求解函数
        
    #待求解函数
    def function(self,x):
        return self.func(x)

    #初始化粒子群
    def init_pop(self,):
        for i in range(self.pN):
            self.X[i] = np.random.uniform(0.1,0.99,[1,self.dim])
            self.V[i] = np.random.uniform(0,self.maxv,[1,self.dim])

            self.pbest[i] = self.X[i] #更新粒子最佳位置
            self.p_bestfit[i] = self.function(self.X[i]) #更新对应值
        for i in range(self.pN):
            if(self.p_bestfit[i] > self.fit):
                self.gbest = self.X[i]
                self.fit = self.p_bestfit[i]

    #开始迭代
    def update(self):
        fitness = []

        for QAQ in range(self.max_iter): #迭代次数
            for i in range(self.pN):
                temp = self.function(self.X[i]) #获得该粒子位置的函数值
                if(temp > self.p_bestfit[i]):
                    self.p_bestfit[i] = temp
                    self.pbest[i] = self.X[i]
                    if(self.p_bestfit[i]>self.fit): #更新全局最优
                        self.fit = self.p_bestfit[i]
                        self.gbest = self.X[i]
            for i in range(self.pN):
                self.V[i] = min(self.w*self.V[i]+\
                            self.c1*random()*(self.pbest[i]-self.X[i])+\
                            self.c2*random()*(self.gbest-self.X[i]),self.maxv)
                
                self.X[i] = self.X[i] + self.V[i]
                if self.X[i]<0.1: self.X[i] = 0.1
                if self.X[i]>=1: self.X[i] = 0.99 

            fitness.append(-self.fit)

        return self.gbest,self.fit,fitness

                

def func(x,k):
    #k = 0.707 阻尼比系数
    b = math.acos(k) #阻尼角
    wn = 1 #无阻尼振荡角频率
    c1 = math.sqrt(1-k**2)  
    wd = wn*c1 #阻尼振荡角频率
    ans = 1-math.exp(-k*wn*x)*math.sin(wd*x+b)/c1
    return ans

def realTime(k):
    t = np.linspace(0,30,num=10000) # delta t = 0.001
    seqk = [k]*10000
    y = list(map(func,t,seqk))
    realtime = 0
    
    for i in range(1,10000+1):
        if  y[-i]<= 1-0.05 or y[-i] >= 1+0.05:
            realtime=t[-i]
            break
    return realtime
                        
def count_func(x):
    return -realTime(x[0])

def solve():
    iternum = 100
    pso = Pso(pN = 50,dim = 1,max_iter = iternum, func = count_func)
    pso.init_pop()
    x_best,fit_best,fitness= pso.update()
    print("bestZeta={:.6f}  bestTime={:.6}".format(x_best[0],-fit_best))

    x = range(1,iternum+1) 
    plt.plot(x,fitness,label ='Convergence curve')
    plt.xlabel(r'$Number\ of\ iterations$')
    plt.ylabel(r'$BestTime$')
    plt.legend()
    plt.show()

solve()
    








        
