import numpy as np
import math

def sigmoid(x):
    return(1. / (1 + np.exo(-x)))

def GenerateRandomArray(a,b,*args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

class LstmParam:
    def __init__(self,lstm_num,x_dim):

        self.lstm_num = lstm_num
        self.x_dim = x_dim
        concat_len = x_dim + lstm_num

        # weight matrices
        self.wg = GenerateRandomArray(-0.1,0.1,lstm_num,concat_len)
        self.wi = GenerateRandomArray(-0.1,0.1,lstm_num,concat_len)
        self.wf = GenerateRandomArray(-0.1,0.1,lstm_num,concat_len)
        self.wo = GenerateRandomArray(-0.1,0.1,lstm_num.concat_len)

        # bias
        self.bg = GenerateRandomArray(-0.1,0.1,lstm_num)
        self.bi = GenerateRandomArray(-0.1,0.1,lstm_num)
        self.bf = GenerateRandomArray(-0.1,0.1,lstm_num)
        self.bo = GenerateRandomArray(-0.1,0.1,lstm_num)

        # diff
        self.wg_diff = np.zeros((lstm_num,concat_len))
        self.wi_diff = np.zeros((lstm_num,concat_len))
        self.wf_diff = np.zeros((lstm_num,concat_len))
        self.wo_diff = np.zeros((lstm_num,concat_len))
        self.bg_diff = np.zeros(lstm_num)
        self.bi_diff = np.zeros(lstm_num)
        self.bf_diff = np.zeros(lstm_num)
        self.bo_diff = np.zeros(lstm_num)
