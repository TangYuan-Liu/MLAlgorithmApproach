import numpy as np
import matplotlib.pyplot as plt
import sys
def LoadDataset(filename):
    path = "./" + filename
    f = open(path)
    dataset = []
    for line in f.readlines():
        temp = line.strip().split()
        dataset.append(map(float,temp))
    
    dataset = np.asarray(dataset)
    n = np.shape(dataset)[0]
    hashlist = np.zeros([n,])
    shuffle = np.zeros([n,2])
    for i in range(n):
        temp = dataset[i]
        flag = 1
        while(flag):
            newid = int(np.floor(n*np.random.rand(1,)[0]))
            if(hashlist[newid] == 0):
                hashlist[newid] = 1
                shuffle[newid] = temp
                flag = 0
            else:
                flag = 1
    return shuffle


def FindRandomCenter(dataset,k):
    center = np.zeros([k,2])
    for i in range(k):
        RangeX = np.max(dataset[:,0]) - np.min(dataset[:,0])
        RangeY = np.max(dataset[:,1]) - np.min(dataset[:,1])
        x = np.min(dataset[:,0]) + np.random.rand(1,)[0] * RangeX
        y = np.min(dataset[:,1]) + np.random.rand(1,)[0] * RangeY
        center[i][0] = x
        center[i][1] = y
    print("Original Center:")
    return center

def Distance(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

def kMeans(dataset,k):
    inf = float("inf")
    pointnum = np.shape(dataset)[0]
    center = FindRandomCenter(dataset,k)
    ClusterChanged = True
    ClusterAssment = np.mat(np.zeros([pointnum,2]))
    while ClusterChanged:
        ClusterChanged = False
        for i in range(pointnum):
            minDist = inf
            minIndex = -1
            for j in range(k):
                dis = Distance(center[j],dataset[i])
                if(minDist > dis):
                    minDist = dis
                    minIndex = j
            if(ClusterAssment[i,0] != minIndex):
                ClusterChanged = True
                ClusterAssment[i,:] = minIndex,minDist**2
             
        for cent in range(k):
            temp = dataset[np.nonzero(ClusterAssment[:,0].A == cent)[0]]
            center[cent] = np.mean(temp,axis=0)
    return center,ClusterAssment


if __name__ == "__main__":
    args = sys.argv[1:]
    name = args[0]
    dataset = LoadDataset(name)
    cent,assm = kMeans(dataset,4)
    print("Final is:")
    print cent
    print("Assment:")
    print assm
    plt.figure()
    plt.scatter(dataset[:,0],dataset[:,1],color="b")
    plt.scatter(cent[:,0],cent[:,1],color="r")
    plt.show()
