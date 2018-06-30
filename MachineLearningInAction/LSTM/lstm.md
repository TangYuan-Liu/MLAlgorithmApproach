# Long-Short Term Memory Cell-----Part 1
这一部分将对LSTM进行介绍，同时对算法进行推导。
## INTRODUCTION
LSTM网络实际上是一种模仿生物神经元记忆功能的深度神经网络，其原理如下图所示：
<div align="center">
<img style="flex-grow:1; flex-shrink:1; border: 1px solid black;" src="./lstmcell.png" width="900" alt="cluster" />
</div>
<p align="center">图1 LSTM Cell结构</p>
其中，每个lstm单元都有两个输出，上面的是记忆，下面的是单元在t时刻的输出。  

#### Forget Gate
左边第一个即为遗忘门，它将t-1时刻的单元输出和t时刻的输入合并后，经过一个sigmoid变换，直接对t-1时刻的记忆进行作用。
#### Input Gate
左边第二个即为输入门，它将t时刻输入与t-1时刻输出合并后，经过sigmoid变换，再配合左边第三个网络层变换tanh，以一定比例对  
记忆产生作用。
#### Output Gate
第四个即为输出门，每一个时刻的输出，既依赖于当前时刻的输入，也依赖于历史记忆。所以输出门将t-1时刻lstm单元的输出、t时刻的输入以及
记忆相叠加后，得到t时刻的lstm单元的输出。

## ALGORITHM
算法的具体推导将会在当前文件夹下**AlgorithmDeduction.pdf**中进行展示。
## KEY POINTS
初学者在使用深度学习框架进行实验时，很容易对LSTM类中参数**rnn_units**产生疑问，不清楚这个参数的含义。从字面上理解，rnn_units表示隐藏层节点数量，但实际应用时仍不理解其中的含义。实际上，我们很容易被CNN的网络结构所误导，认为RNN同样作为深度神经网络，也是类似的结构。实际上，如图1所示，LSTM在内部计算时，并不是类似CNN一条网络主线、多层接力传递这种模式，而是另外一种状态。举例来说，如果我们给LSTM网络的输入数据是[T,N]，T实际上是时间步的长度，也就是time_step，而个时间步输入的大小就是N。我们用f表示LSTM对于c(t)的函数变换、用m表示LSTM对h(t)的函数变换。  
具体每个时刻的数据流结果计算如下：  
1.t = 0   **h(t-1)** = [0,0,0,...0](size = [10,])   **x(t)** = dataset[0,:]       **c(t)** = f(x(t), h(t-1))  **h(t)** = m(x(t), h(t-1))  
2.t = 1   **h(t-1) = h(t=0)**  **x(t=1)** = dataset[1,:]  **c(t)** = f(x(t), h(t-1))  **ht(t)** = m(x(t), h(t-1))  
.  
.  
.
