# coding=utf-8
import numpy as np
import copy
import matplotlib.pyplot as plt
dataSet = [
        [0.665, 0.09, '是'],
        [0.242, 0.266, '是'],
        [0.244, 0.056, '是'],
        [0.342, 0.098, '是'],
        [0.638, 0.16, '是'],
        [0.656, 0.197, '是'],
        [0.359, 0.369, '是'],
        [0.592, 0.041, '是'],
        [0.718, 0.102, '是'],
        [0.696, 0.459, '否'],
        [0.773, 0.375, '否'],
        [0.633, 0.263, '否'],
        [0.607, 0.317, '否'],
        [0.555, 0.214, '否'],
        [0.402, 0.236, '否'],
        [0.48, 0.148, '否'],
        [0.436, 0.21, '否'],
        [0.557, 0.216, '否']
    ]
for i in range(len(dataSet)):   # '是'换为1，'否'换为-1。
    if dataSet[i][-1] == '是':
        dataSet[i][-1] = 1
    else:
        dataSet[i][-1] = -1
dataSet = np.array(dataSet)


def Err(splite,num,D): # 计算分类器的总误差
    err = [0,0]
    for i in range(2):
        if i == 0: # 大于分类值为正例
            for j in range(np.shape(dataSet)[0]):
                if (dataSet[j,num] >= splite and dataSet[j,-1] == -1) or (dataSet[j,num] < splite and dataSet[j,-1] == 1):
                    err[0] += D[j]  #该样本的权重作为错误率
        
        else: # 小于分类值为正例
            for j in range(np.shape(dataSet)[0]):
                if (dataSet[j,num] <= splite and dataSet[j,-1] == -1) or (dataSet[j,num] > splite and dataSet[j,-1] == 1):
                    err[1] += D[j]  
    
    return min(err),err.index(min(err))

def CreateTree(D): # 生成决策树
    base = {} # 决策树桩
    m, n = np.shape(dataSet)
    minErr = 10000
    for i in range(n-1):
        list1 = [x[i] for x in dataSet]
        list1.sort(reverse=False)
        for j in range(m-1): 
            splite = (list1[j]+list1[j+1])/2
            err,symbol = Err(splite,i,D)
            if err < minErr:  # 选择误差最小的分类器
                minErr = err
                base['feature'] = i # 哪个属性
                base['splite'] = splite
                base['symbol'] = symbol # 大于还是小于
                base['err'] = err

    return base

def predict(h,data): # 根据分类器得到预测结果
    if h['symbol'] == 0:
        if h['splite'] <= data[h['feature']]:
            return 1
        else:
            return -1
    else:
        if h['splite'] >= data[h['feature']]:
            return 1
        else:
            return -1

def AdaBoost(T): # 主函数
    Dt = np.ones([np.shape(dataSet)[0],1])/np.shape(dataSet)[0]
    h = [] # 基学习器
    for i in range(T):
        h.append(CreateTree(Dt)) # 生成决策树

        alpha = np.log((1 - h[i]['err']) / h[i]['err']) / 2
        Dtt = np.ones(np.shape(Dt)) # 权值迭代
        for j in range(np.shape(dataSet)[0]):
            Dtt[j,0] = Dt[j,0]*np.exp(-1*alpha*dataSet[j,-1]*predict(h[i],dataSet[j]))

        Dtt = Dtt/np.sum(Dtt) # 除以规范因子使分布规范化
        Dt = copy.deepcopy(Dtt)

        h[i]["alpha"] = alpha
    return h

def AdaBoost_Predict(data,h):
    final = []
    num = 0
    all = 0
    for i in data:
        hx = 0
        for j in h:
            hx += j['alpha']*predict(j,i)
        if hx >0: final.append(1)
        elif hx <0: final.append(-1)
        else: final.append(0)

        if final[-1] != i[-1]:
            num += 1
        all += 1

    return final,(1-num/all)


def Plot(h):
    # 画点
    data1 = []
    data2 = []
    for i in dataSet:
        if i[-1] == 1:
            data1.append(i)
        else:
            data2.append(i)
    X1 = [i[0] for i in data1]
    Y1 = [i[1] for i in data1]
    plt.plot(X1,Y1,'r+',label = '正例')

    X2 = [i[0] for i in data2]
    Y2 = [i[1] for i in data2]
    plt.plot(X2,Y2,'b_',label = '反例')

    plt.xlabel('属性1')
    plt.ylabel('属性2')
    plt.legend(loc="best")
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签

    # 画基学习器的边界
    x = np.linspace(0, 0.8, 100)
    y = np.linspace(0, 0.6, 100)
    for i in h:
        z = [i['splite']]*100
        if i['feature'] == 0:
            if i['symbol'] == 0:
                plt.plot(z,y,'r-')
            else:
                plt.plot(z,y,'k--')
        else:
            if i['symbol'] == 0:
                plt.plot(x,z,'r-')
            else:
                plt.plot(x,z,'k--')
    plt.show()

h = AdaBoost(9)
final,acc = AdaBoost_Predict(dataSet,h)
for i in h:
    print(i)
print('\n')
print(final,acc)
Plot(h)


