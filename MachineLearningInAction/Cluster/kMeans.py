#聚类辅助函数

import numpy as np

def randCent(dataset,k):
    n = np.shape(dataset)[1]
    centroids = np.zeros([k,n])
    for j in range(n):
        minJ = min(dataset[:,j])
        rangeJ = np.float(max(dataset[:,j] - minJ)
        center[:,j] = minJ + rangeJ * np.random.rand(k,1)
    return center

def loadData(FileName):
    datamat = []
    f = open(FileName)
    for line in f.readlines():
        curline = line.strip().split('\t')
        fline = map(float,curline)
        datamat.append(fline)
    return datamat

def distance(vecA,vecB):
    return np.sqrt(np.sum((vecA - vecB)**2))
