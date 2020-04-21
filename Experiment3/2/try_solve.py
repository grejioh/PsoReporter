import math
import numpy as np
import matplotlib.pyplot as plt


def func(x,k):
    #k = 0.707 阻尼比系数
    b = math.acos(k) #阻尼角
    wn = 1 #无阻尼振荡角频率
    c1 = math.sqrt(1-k**2)  
    wd = wn*c1 #阻尼振荡角频率
    ans = 1-math.exp(-k*wn*x)*math.sin(wd*x+b)/c1
    return ans

def realTime(k):
    t = np.linspace(0,30,num=30000) # delta t = 0.001
    seqk = [k]*30000
    y = list(map(func,t,seqk))
    realtime = 0
    ttt = []
    for i in range(1,30000+1):
        if  y[-i]<= 1-0.05 or y[-i] >= 1+0.05:
            realtime=t[-i]
            break
    return realtime

def envelopeTime(k):
    return -(math.log(0.05)+math.log(math.sqrt(1-k**2)))/k
def estimateTime(k):
    return 3/k

def solve():
    k = np.linspace(0.1,0.999,num=1000)
    realtime = list(map(realTime,k))
    envelopetime = list(map(envelopeTime,k))
    estimatetime = list(map(estimateTime,k))
    plt.figure()
    plt.plot(k,realtime,label='Real Time')
    plt.plot(k,envelopetime,label ='Envelope Time')
    plt.plot(k,estimatetime,label = 'Estimate Time')
    plt.legend()
    plt.xlabel(r'$\zeta$')
    plt.ylabel(r'$\omega_nt$')
    plt.show()
    

solve()
