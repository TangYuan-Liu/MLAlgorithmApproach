import numpy as np

def LoadDataset(filename):
    path = "./" + filename
    f = open(path)
    dataset = []
    for line in f.readlines():
        temp = line.strip().split()
        dataset.append(map(float,temp))
    
    dataset = np.asarray(dataset)
    return dataset


def FindRandomCenter(dataset,k):
    center = np.zeros([k,2])
    for i in range(k):
        RangeX = np.max(dataset[:,0]) - np.min(dataset[:,0])
        RangeY = np.max(dataset[:,1]) - np.min(dataset[:,1])
        x = np.min(dataset[:,0]) + np.random.rand(1,)[0] * RangeX
        y = np.min(dataset[:,1]) + np.random.rand(1,)[0] * RangeY
        center[i][0] = x
        center[i][1] = y

    return center

def Distance(vecA,vecB):
    return np.sqrt(np.sum((vecA-vecB)**2))

def kMeans(dataset,k):
    pointnum = np.shape(dataset)[0]
    center = FindRandomCenter(dataset,k)
    ClusterChanged = True
    ClusterAssment = np.mat(np.zeros([pointnum,2]))
    while ClusterChanged:
        ClusterChanged = False
        for i in range(pointnum):
            minDist = None
            minIndex = None
            for j in range(k):
                dis = Distance(center[j],dataset[i])
                if(minDist == minIndex == None):
                    minDist = dis
                    minIndex = j
                else:
                    if(minDist > dis):
                        minDist = dis
                        minIndex = j
            if(ClusterAssment[i,0] != minIndex):
                ClusterChanged = True
                ClusterAssment[i,:] = minIndex,minDist**2
               
        print center
        for cent in range(k):
            temp = dataset[np.nonzero(ClusterAssment[:,0].A == cent)]
            center[cent] = np.mean(temp,axis=0)
    
    return center,ClusterAssment


if __name__ == "__main__":
    name = "dataset"
    dataset = LoadDataset(name)
    kMeans(dataset,4)

 

