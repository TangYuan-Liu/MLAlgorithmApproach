import numpy as np
import matplotlib.pyplot as plt

def loadDataset():
    f = open("./Dataset.txt")
    dataMat = []
    labelMat = []
    for line in f.readlines():
        lineData = (line.strip().split())
        dataMat.append([1.0, float(lineData[0]), float(lineData[1])])
        labelMat.append(int(lineData[2]))
    return dataMat, labelMat

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def grandAscent(dataMat, labels):
    dataMatrix = np.mat(dataMat)
    labelMat = np.mat(labels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    iterations = 500
    weights = np.ones([n,1])
    for k in range(iterations):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotFig(weights, dataset, labels):
    type1x = []
    type1y = []
    type2x = []
    type2y = []
    for i in range(len(dataset)):
        if(labels[i] == 1):
            type1x.append(dataset[i][1])
            type1y.append(dataset[i][2])
        else:
            type2x.append(dataset[i][1])
            type2y.append(dataset[i][2])
    weights = np.asarray(weights)    
    x = np.arange(-4.0, 4.0, 0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.scatter(type1x, type1y, color = 'b', marker = 's')
    ax.scatter(type2x, type2y, color = 'r')
    plt.show()


if __name__ == "__main__":
    dataMat,labelMat = loadDataset()
    weights = grandAscent(dataMat, labelMat)
    plotFig(weights, dataMat, labelMat)
