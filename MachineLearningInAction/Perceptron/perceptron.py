#This program is a implementation of a simple perceptron
#我们使用最简单的基本训练算法，即每次随机挑选一个误差点对w，b进行调整。

import numpy as np

class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None

    def prediction(self, x, option=False):

        x = np.asarray(x,np.float32)
        y_predict = np.dot(x,self.w) + self.b

        if(option == True):
            return np.sigh(y_prediction).astype(np.float32)
        else:
            return y_prediction

    def simple_train(self, x, y_label, lr=0.01, epoch=1000):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        self.w = np.zeros(x.shape[1])
        self.b = 0.

        for _ in range(epoch):
            err = -y * self.prediction(x)
            idx = np.argmax(err)

            if (err[idx] <= 0):
                break
            else:
                delta = lr * y[idx]
                self.w += delta * x[idx]
                self.b += delta
