# 本例通过两个点拟合出y=ax^2+bx+c曲线，a、b、c为参数，需要我们解出来，当然，会有无数个a、b、c的值满足要求，这就好比深度学习中参数远远多于样本数。
# 正则化可以使得参数的值尽可能接近0。
# p1=（x1,y1),p2= (x2,y2)为两个样本点，本例取 p1 = [5, 3],p2= [-3, -9]
# ————————————————
#版权声明：本文为CSDN博主「yangyangyangr」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
#原文链接：https://blog.csdn.net/weixin_38698995/article/details/105428190

import numpy as np
import matplotlib.pyplot as plt

def No_rugularization(e,f):
    '''
    没有用正则化 abc系数都很大
    :param e: e f为传入的两个点，列表形式如[2,4]
    :param f: 同上
    :return: a,b,c。即一元二次方程的三个系数
    '''
    x1=e[0]
    y1=e[1]
    x2=f[0]
    y2=f[1]
    a=-1000000
    c,b=1000 ,1000
    L=(a*x1**2+b*x1+c-y1)**2+(a*x2**2+b*x2+c-y2)**2
    print(L)
    for i in range(100000):
        a = a - (2*(a*x1**2+b*x1+c-y1)*(x1**2)+2*(a*x2**2+b*x2+c-y2)*(x2**2))*0.001
        b = b - (2*(a*x1**2+b*x1+c-y1)*(x1)+2*(a*x2**2+b*x2+c-y2)*(x2))*0.001
        c = c - (2*(a*x1**2+b*x1+c-y1)+2*(a*x2**2+b*x2+c-y2))*0.001
    L1 = (a * x1 ** 2 + b * x1 + c - y1) ** 2 + (a * x2 ** 2 + b * x2 + c - y2) ** 2
    print(L1,'L1')
    print(a,b,c)
    return a,b,c

def ZhengzehuaL2(e,f):
    x1=e[0]
    y1=e[1]
    x2=f[0]
    y2=f[1]
    a=-1000000
    c,b=1000 ,1000#给一个较大初始值
    L=(a*x1**2+b*x1+c-y1)**2+(a*x2**2+b*x2+c-y2)**2+a**2+b**2+c**2
    print(L)
    for i in range(100000):#学习率设为0.001
        a = a - (2*(a*x1**2+b*x1+c-y1)*(x1**2)+2*(a*x2**2+b*x2+c-y2)*(x2**2)+a*2)*0.001
        b = b - (2*(a*x1**2+b*x1+c-y1)*(x1)+2*(a*x2**2+b*x2+c-y2)*(x2)+b*2)*0.001
        c = c - (2*(a*x1**2+b*x1+c-y1)+2*(a*x2**2+b*x2+c-y2)+c*2)*0.001
    L1 = (a * x1 ** 2 + b * x1 + c - y1) ** 2 + (a * x2 ** 2 + b * x2 + c - y2) ** 2
    print(L1,'L1')
    print(a,b,c)
    return a,b,c


def testfunction(f):
    p1 = [5, 3]
    p2 = [-3, -9]
    a, b, c = f(p1, p2)
    x = np.arange(-20, 20)
    y = a * x ** 2 + b * x + c
    plt.title("Matplotlib demo")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    plt.plot(x, y)
    [a1, b1] = p1 
    [a2, b2] = p2
    plt.plot(a1, b1, 'ob')
    plt.plot(a2, b2, 'ob')
    plt.show()

testfunction(No_rugularization)#实验结果表明所得的系数绝对值很大
testfunction(ZhengzehuaL2)#实验结果表明  正则化 让所得的系数绝对值非常小
