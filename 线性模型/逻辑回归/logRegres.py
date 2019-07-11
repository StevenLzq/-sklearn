import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')  #打开文本
    for line in fr.readlines():  #逐行读取，，第三个值是数据对应的标签，将X0的值设置为1
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat


#定义sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


#梯度上升优化
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()#转换轴，行向量变成列向量
    m,n=np.shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=np.ones((n,1))
    for  k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=labelMat-h
        weights=weights+alpha*dataMatrix.transpose()*error #这部分关注shape
    return weights


#######分析数据：画出决策边界
#画出数据集和Logistic回归最佳拟合曲线



def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()#数据
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    #绘制数据样本散点图
    for i in range(n):
        if int(labelMat[i]) == 1:  ##xcord1标签值为1
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])  ##xcord2标签值为0

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s',alpha=.5)#大小，颜色，形状，透明度
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]  ##0为两类的分界处，故假定0=w0x0+w1x1+w2x2,x0=1
    y = y.transpose()
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



#####随机梯度上升
"""梯度上升在处理小的特征的的时候表现尚可，但是在很大的数据集的时候
，一一求梯度就显得不那么聪明了，随机梯度是一个不错且有效的办法，能帮助加速收敛
"""
#粗糙简陋的随机方式
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    sumWeights = []
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#随机梯度的改进
def  stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01    #可变步长，随着次数增肌不断减小
            # 随机选取样本，以他的梯度作为方向，减少周期性波动
            randIndex=int(np.random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            #参数更新
            weights=weights+alpha*error*dataMatrix[randIndex]
            #删除该值进行下一轮迭代
            del(list(dataIndex)[randIndex])
    return weights

#####用逻辑回归预测马的死亡率
#我们使用的而数据都经过了预处理这一点要明白

#输入：特征向量与回归系数，用来验证模型效果
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    #大于0.5 返回 1；否则返回0
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('HorseColicTraining.txt')
    frTest = open('HorseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    # trainingSet 中存储训练数据集的特征，trainingLabels 存储训练数据集的样本对应的分类标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t') #分割
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        #存入训练样本特征
        trainingSet.append(lineArr)
        #存入训练样本标签
        trainingLabels.append(float(currLine[21]))
    #使用改进后的随机梯度下降算法得到回归系数
    trainingWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)

    #统计
    errorCount = 0#预测错误个数
    numTestVec = 0.0#总个数
    for line in frTest.readlines():
        #循环一次，样本数加1
        numTestVec += 1.0
        currLine = line.strip().split('\t') #分割
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 利用分类预测函数对该样本进行预测，并与样本标签进行比较
        if int(classifyVector(np.array(lineArr), trainingWeights)) != int(currLine[21]):
            #如果预测错误，错误数加1
            errorCount += 1
    #计算错误率
    errorRate = (float(errorCount) / numTestVec)
    print('the error rate of this test is : %f' % errorRate)
    return errorRate



def multiTest():
    numTests=10
    errorSum=0
    for k in range(numTests):
        errorSum+=colicTest()
    print('after %d iterations the average errorrate is %f:'%(numTests,errorSum/float(numTests)))


if __name__=='__main__':
    #dataArr, labelMat = loadDataSet()
    # weights=gradAscent(dataArr, labelMat)
    # print(weights)
    # plotBestFit(weights)

    # weights=stocGradAscent0(np.array(dataArr),labelMat )
    # plotBestFit(weights)


    # weights=stocGradAscent1(np.array(dataArr),labelMat)
    # plotBestFit(weights)
####马的死亡率
    multiTest()

