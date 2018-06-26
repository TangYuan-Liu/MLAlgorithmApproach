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


    def apply_diff(self, lr = 1):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff

        #Reset diff to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)

    class LstmState:
        def __init__(self, mem_cell_ct, x_dim):
            self.g = np.zeros(mem_cell_ct)
            self.i = np.zeros(mem_cell_ct)
            self.f = np.zeros(mem_cell_ct)
            self.o = np.zeros(mem_cell_ct)
            self.s = np.zeros(mem_cell_ct)
            self.h = np.zeros(mem_cell_ct)
            self.bottom_diff_h = np.zeros_like(self.h)
            self.bottom_diff_s = np.zeros_like(self.s)
            self.bottom_diff_x = np.zeros_like(x_dim)

    class LstmNode:
        def __init__(self, lstm_param, lstm_state):
            self.state = lstm_state
            self.param = lstm_param
            self.x = None
            self.xc = None

        def bottom_data_is(self, x, s_prev = None, h_prev = None):
            
       
