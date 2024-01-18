# 实现深度神经网络

我们现在将利用我们对GPU编程的积累知识，使用PyCUDA实现我们自己的深度神经网络（DNN）。在过去的十年里，DNN吸引了很多关注，因为它们为机器学习（ML）提供了一个强大而优雅的模型。DNN也是第一个能够展示GPU真正强大之处的应用之一（除了渲染图形），通过利用它们的大规模并行吞吐量，最终帮助NVIDIA成为人工智能领域的主要参与者。

在本书的过程中，我们大多是按章节覆盖单个主题的——在这里，我们将在我们学到的许多主题的基础上构建我们自己的DNN实现。虽然目前有几个面向普通公众的基于GPU的DNN的开源框架，例如Google的TensorFlow和Keras，微软的CNTK，Facebook的Caffe2和PyTorch，但从头开始实现一个DNN非常有教育意义，这将使我们更深入地了解和欣赏DNN所需的基础技术。我们有很多材料要涵盖，所以在简要介绍一些基本概念后，我们将直奔主题。

在本章中，我们将研究以下内容：

+   理解**人工神经元**（AN）是什么

+   理解如何将多个AN组合在一起形成**深度神经网络**（DNN）

+   在CUDA和Python中从头开始实现DNN

+   理解如何使用交叉熵损失来评估神经网络的输出

+   实现梯度下降来训练NN

+   学习如何在小数据集上训练和测试NN

# 技术要求

本章需要一台装有现代NVIDIA GPU（2016年以后）的Linux或Windows 10 PC，并安装了所有必要的GPU驱动程序和CUDA Toolkit（9.0及以上）。还需要一个合适的Python 2.7安装（如Anaconda Python 2.7），并安装了PyCUDA模块。

本章的代码也可以在GitHub上找到：[https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)。

有关本章的先决条件的更多信息，请查看本书的前言。有关软件和硬件要求，请查看[https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)中的README文件。

# 人工神经元和神经网络

让我们简要回顾一些**机器学习（ML）**和**神经网络（NNs）**的基础知识。在机器学习中，我们的目标是使用具有特定标记类别或特征的数据集，并利用这些示例来训练我们的系统以预测未来数据的值。我们称根据先前训练数据预测未来数据的类别或标签的程序或函数为**分类器**。

有许多类型的分类器，但在这里我们将专注于NNs。NNs的理念是它们（据说）以类似于人脑的方式工作，通过使用一组**人工神经元（ANs）**来学习和分类数据，所有这些神经元连接在一起形成特定的结构。不过，让我们暂停一下，看看一个单独的AN是什么。在数学上，这只是从线性空间**R^n**到**R**的*仿射*函数，如下所示：

![](assets/25478608-c882-49f1-85a9-9e3f6b0d472d.png)

我们可以看到这可以被描述为一个常量权重向量***w***和输入向量***x***之间的点积，最后加上一个额外的偏置常量*b*。（再次强调，这个函数的唯一*输入*是*x*；其他值都是常数！）

现在，单个AN本身是相当无用（而且愚蠢）的，只有当它们与大量其他AN合作时，它们的*智能*才会显现出来。我们的第一步是将一系列相似的AN堆叠在一起，以形成我们将称之为**密集层（DL）**的东西。这是密集的，因为每个神经元将处理来自*x*的每个输入值 - 每个AN将接收来自**R^n**的数组或向量值，并在**R**中输出一个值。由于有*m*个神经元，这意味着它们的输出集体位于**R^m**空间中。我们将注意到，如果我们堆叠我们层中每个神经元的权重，以形成一个*m x n*的权重矩阵，然后我们可以通过矩阵乘法计算每个神经元的输出，然后加上适当的偏差：

![](assets/03bf9c48-e6dc-4f86-9b18-9a0eac05c7cb.png)

现在，假设我们想要构建一个能够对*k*个不同类别进行分类的NN分类器；我们可以创建一个新的附加密集层，该层接收来自先前密集层的*m*个值，并输出*k*个值。假设我们对每一层都有适当的权重和偏差值（这显然不容易找到），并且在每一层之后也有适当的**激活函数**设置（我们稍后会定义），这将作为我们*k*个不同类别之间的分类器，根据最终层的输出给出*x*落入每个类别的概率。当然，我们在这里走得太远了，但这就是NN的工作原理。

现在，似乎我们可以将密集层连接到长链中以实现分类。这就是所谓的DNN。当我们有一层不直接连接到输入或输出时，这就是一个隐藏层。DNN的优势在于额外的层允许NN捕捉浅层NN无法捕捉到的数据的抽象和细微差别。

# 实现人工神经元的密集层

现在，让我们实现NN最重要的构建模块，**密集层**。让我们从声明CUDA内核开始，就像这样：

```py
__global__ void dense_eval(int num_outputs, int num_inputs, int relu, int sigmoid, float * w, float * b, float * x, float *y, int batch_size, int w_t, int b_t, float delta)
```

让我们逐个检查输入。`num_outputs`当然表示这一层的总输出数量；这正好是该层的神经元数量。`num_inputs`告诉我们输入数据的大小。为`relu`和`sigmoid`设置正值将表示我们应该在该层的输出上使用相应的激活函数，我们稍后会定义。`w`和`b`是包含该层权重和偏差的数组，而`x`和`y`将作为我们的输入和输出。通常，我们希望一次对多个数据进行分类。我们可以通过将`batch_size`设置为我们希望预测的点数来指示这一点。最后，`w_t`、`b_t`和`delta`将在训练过程中使用，通过**梯度下降**来确定该层的适当权重和偏差。（我们将在后面的部分中更多地了解梯度下降。）

现在，让我们开始编写我们的内核。我们将在每个输出上并行计算，因此我们将设置一个整数`i`作为全局线程ID，然后使用适当的`if`语句让任何不必要的额外线程不执行任何操作：

```py
{
 int i = blockDim.x*blockIdx.x + threadIdx.x;

 if (i < num_outputs)
 {
```

现在，让我们用适当的`for`循环迭代批处理中的每个数据点：

```py
for(int k=0; k < batch_size; k++)
 { 
```

我们将从权重和输入中的32位浮点数中进行乘法和累加，得到64位双精度`temp`，然后加上适当的偏差点。然后我们将这个值强制转换回32位浮点数并将值放入输出数组中，然后关闭对`k`的循环：

```py
double temp = 0.0f;
 for (int j = 0; j < num_inputs; j++)
 {
   temp += ((double) w[(num_inputs)*i + j ] ) * ( (double) x[k*num_inputs + j]);
 }
 temp += (double) b[i];
 y[k * num_outputs + i] = (float) temp;  
}
```

*乘法和累加*类型的操作通常会导致严重的数值精度损失。这可以通过使用更高精度的临时变量来存储操作过程中的值，然后在操作完成后将此变量强制转换回原始精度来减轻。 

要训练一个NN，我们最终将不得不计算我们的NN对每个权重和偏差的导数（来自微积分），这是针对特定输入批次的。请记住，数学函数*f*在值*x*处的导数可以估计为*f**(x + δ) - f(x) / δ*，其中delta（δ）是某个足够小的正值。我们将使用输入值`w_t`和`b_t`来指示内核是否要计算相对于特定权重或偏差的导数；否则，我们将将这些输入值设置为负值，仅对此层进行评估。我们还将设置delta为适当小的值，用于计算导数，并使用它来增加适当偏差或权重的值：

```py
if( w_t >= 0 && i == (w_t / num_inputs))
 {
 int j = w_t % num_inputs;
 for(int k=0; k < batch_size; k++)
  y[k*num_outputs + i] += delta*x[k*num_inputs+j];
}
if( b_t >= 0 && i == b_t )
 {
  for(int k=0; k < batch_size; k++)
  y[k*num_outputs + i] += delta;
 }
```

现在，我们将添加一些代码，用于所谓的**修正线性单元**（或**ReLU**）和**Sigmoid激活函数**。这些用于处理密集神经层的直接输出。ReLU只将所有负值设为0，同时作为正输入的恒等式，而sigmoid只计算每个值上的`sigmoid`函数的值（*1 / (1 + e^(-x))*）。ReLU（或任何其他激活函数）用于NN中隐藏层之间，作为使整个NN成为非线性函数的手段；否则，整个NN将构成一个微不足道（且计算效率低下）的矩阵操作。（虽然可以在层之间使用许多其他非线性激活函数，但发现ReLU对训练来说是一个特别有效的函数。）Sigmoid用作NN中用于**标签**的最终层，即可能为给定输入分配多个标签的层，而不是将输入分配给单个类别。

让我们在文件中稍微上移一点，甚至在我们开始定义这个CUDA内核之前，将这些操作定义为C宏。我们还要记得在我们写完CUDA-C代码后将其放入其中：

```py
DenseEvalCode = '''
#define _RELU(x) ( ((x) > 0.0f) ? (x) : 0.0f )
#define _SIGMOID(x) ( 1.0f / (1.0f + expf(-(x)) ))
```

现在，我们将使用内核输入`relu`和`sigmoid`来指示我们是否应该使用这些额外的层；我们将从这些中获取积极的输入，以指示它们应该分别使用。我们可以添加这个，关闭我们的内核，并将其编译成可用的Python函数：

```py
if(relu > 0 || sigmoid > 0)
for(int k=0; k < batch_size; k++)
 { 
   float temp = y[k * num_outputs + i];
   if (relu > 0)
    temp = _RELU(temp);
   if (sigmoid > 0)
    temp = _SIGMOID(temp);
   y[k * num_outputs + i] = temp; 
  }
 }
 return;
}
'''
eval_mod = SourceModule(DenseEvalCode)
eval_ker = eval_mod.get_function('dense_eval')
```

现在，让我们回到文件的开头，并设置适当的导入语句。请注意，我们将包括`csv`模块，该模块将用于处理测试和训练的数据输入：

```py
from __future__ import division
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import numpy as np
from Queue import Queue
import csv
import time
```

现在，让我们继续设置我们的密集层；我们将希望将其包装在一个Python类中以便使用，这将使我们在开始将这些密集层连接成一个完整的NN时更加轻松。我们将称之为`class DenseLayer`并开始编写构造函数。这里的大部分输入和设置应该是不言自明的：我们绝对应该添加一个选项，从预训练的网络中加载权重和偏差，并且我们还将包括指定默认*delta*值以及默认流的选项。（如果没有给出权重或偏差，权重将被初始化为随机值，而所有偏差将设置为0。）我们还将在这里指定是否使用ReLU或sigmoid层。最后，请注意我们如何设置块和网格大小：

```py
class DenseLayer:
    def __init__(self, num_inputs=None, num_outputs=None, weights=None, b=None, stream=None, relu=False, sigmoid=False, delta=None):
        self.stream = stream

        if delta is None:
            self.delta = np.float32(0.001)
        else:
            self.delta = np.float32(delta)

        if weights is None:
            weights = np.random.rand(num_outputs, num_inputs) - .5
            self.num_inputs = np.int32(num_inputs)
        self.num_outputs = np.int32(num_outputs) 

        if type(weights) != pycuda.gpuarray.GPUArray:
            self.weights = gpuarray.to_gpu_async(np.array(weights, 
            dtype=np.float32) , stream = self.stream)
        else:
            self.weights = weights

        if num_inputs is None or num_outputs is None:
            self.num_inputs = np.int32(self.weights.shape[1])
            self.num_outputs = np.int32(self.weights.shape[0])

        else:
            self.num_inputs = np.int32(num_inputs)
            self.num_outputs = np.int32(num_outputs)

        if b is None:
            b = gpuarray.zeros((self.num_outputs,),dtype=np.float32)

        if type(b) != pycuda.gpuarray.GPUArray:
            self.b = gpuarray.to_gpu_async(np.array(b, 
            dtype=np.float32) , stream = self.stream)
        else:
            self.b = b 

        self.relu = np.int32(relu)
        self.sigmoid = np.int32(sigmoid)

        self.block = (32,1,1)
        self.grid = (int(np.ceil(self.num_outputs / 32)), 1,1)
```

现在，我们将在这个类中设置一个函数来评估来自这一层的输入；我们将仔细检查输入（x）以确定它是否已在GPU上（如果没有，则将其转移到`gpuarray`），并让用户指定预分配的`gpuarray`用于输出（y），如果没有指定，则手动分配一个输出数组。我们还将检查训练时的delta和`w_t`/`b_t`值，以及`batch_size`。然后我们将在`x`输入上运行核函数，输出到`y`，最后将`y`作为输出值返回：

```py
def eval_(self, x, y=None, batch_size=None, stream=None, delta=None, w_t = None, b_t = None):

if stream is None:
    stream = self.stream

if type(x) != pycuda.gpuarray.GPUArray:
    x = gpuarray.to_gpu_async(np.array(x,dtype=np.float32), stream=self.stream)

if batch_size is None:
    if len(x.shape) == 2:
        batch_size = np.int32(x.shape[0])
    else:
        batch_size = np.int32(1)

if delta is None:
    delta = self.delta

delta = np.float32(delta)

if w_t is None:
    w_t = np.int32(-1)

if b_t is None:
    b_t = np.int32(-1)

if y is None:
    if batch_size == 1:
        y = gpuarray.empty((self.num_outputs,), dtype=np.float32)
    else:
        y = gpuarray.empty((batch_size, self.num_outputs), dtype=np.float32)

    eval_ker(self.num_outputs, self.num_inputs, self.relu, self.sigmoid, self.weights, self.b, x, y, np.int32(batch_size), w_t, b_t, delta , block=self.block, grid=self.grid , stream=stream)

 return y
```

现在，我们已经完全实现了一个密集层！

# 实现softmax层

现在，让我们看看如何实现**softmax层**。正如我们已经讨论过的，sigmoid层用于为类分配标签——也就是说，如果您想要从输入中推断出多个非排他性特征，您应该使用sigmoid层。**softmax层**用于仅通过推断为样本分配单个类别——这是通过计算每个可能类别的概率来实现的（当然，所有类别的概率总和为100%）。然后我们可以选择具有最高概率的类别来给出最终分类。

现在，让我们看看softmax层到底做了什么——给定一组*N*个实数(*c[0], ..., c[N-1]*)，我们首先计算每个数字的指数函数的总和，然后计算每个数字除以这个总和的指数，得到softmax：

![](assets/ce20fc54-36ee-46a9-86a9-40387fb73545.png)

让我们从我们的实现开始。我们将首先编写两个非常简短的CUDA内核：一个用于计算每个输入的指数，另一个用于计算所有点的平均值：

```py
SoftmaxExpCode='''
__global__ void softmax_exp( int num, float *x, float *y, int batch_size)
{
 int i = blockIdx.x * blockDim.x + threadIdx.x;

 if (i < num)
 {
  for (int k=0; k < batch_size; k++)
  {
   y[num*k + i] = expf(x[num*k+i]);
  }
 }
}
'''
exp_mod = SourceModule(SoftmaxExpCode)
exp_ker = exp_mod.get_function('softmax_exp')

SoftmaxMeanCode='''
__global__ void softmax_mean( int num, float *x, float *y, int batch_size)
{
 int i = blockDim.x*blockIdx.x + threadIdx.x;

 if (i < batch_size)
 {
  float temp = 0.0f;

  for(int k=0; k < num; k++)
   temp += x[i*num + k];

  for(int k=0; k < num; k++)
   y[i*num+k] = x[i*num+k] / temp;
 }

 return;
}'''

mean_mod = SourceModule(SoftmaxMeanCode)
mean_ker = mean_mod.get_function('softmax_mean')
```

现在，让我们编写一个Python包装类，就像我们之前做的那样。首先，我们将从构造函数开始，并使用`num`指示输入和输出的数量。如果需要，我们还可以指定默认流：

```py
class SoftmaxLayer:
    def __init__(self, num=None, stream=None):
     self.num = np.int32(num)
     self.stream = stream
```

现在，让我们编写一个类似于密集层的`eval_`函数：

```py
def eval_(self, x, y=None, batch_size=None, stream=None):
 if stream is None:
 stream = self.stream

 if type(x) != pycuda.gpuarray.GPUArray:
  temp = np.array(x,dtype=np.float32)
  x = gpuarray.to_gpu_async( temp , stream=stream)

 if batch_size==None:
  if len(x.shape) == 2:
   batch_size = np.int32(x.shape[0])
  else:
   batch_size = np.int32(1)
 else:
  batch_size = np.int32(batch_size)

 if y is None:
  if batch_size == 1:
   y = gpuarray.empty((self.num,), dtype=np.float32)
 else:
  y = gpuarray.empty((batch_size, self.num), dtype=np.float32)

 exp_ker(self.num, x, y, batch_size, block=(32,1,1), grid=(int( np.ceil( self.num / 32) ), 1, 1), stream=stream)

 mean_ker(self.num, y, y, batch_size, block=(32,1,1), grid=(int( np.ceil( batch_size / 32)), 1,1), stream=stream)

 return y

```

# 实现交叉熵损失

现在，让我们实现所谓的**交叉熵损失**函数。这用于在训练过程中测量神经网络在数据子集上的准确性；损失函数输出的值越大，我们的神经网络在正确分类给定数据方面就越不准确。我们通过计算期望输出和实际输出之间的标准平均对数熵差来实现这一点。为了数值稳定性，我们将限制输出值为`1`：

```py
MAX_ENTROPY = 1

def cross_entropy(predictions=None, ground_truth=None):

 if predictions is None or ground_truth is None:
  raise Exception("Error! Both predictions and ground truth must be float32 arrays")

 p = np.array(predictions).copy()
 y = np.array(ground_truth).copy()

 if p.shape != y.shape:
  raise Exception("Error! Both predictions and ground_truth must have same shape.")

 if len(p.shape) != 2:
  raise Exception("Error! Both predictions and ground_truth must be 2D arrays.")

 total_entropy = 0

 for i in range(p.shape[0]):
  for j in range(p.shape[1]):
   if y[i,j] == 1: 
    total_entropy += min( np.abs( np.nan_to_num( np.log( p[i,j] ) ) ) , MAX_ENTROPY) 
   else: 
    total_entropy += min( np.abs( np.nan_to_num( np.log( 1 - p[i,j] ) ) ), MAX_ENTROPY)

 return total_entropy / p.size
```

# 实现一个顺序网络

现在，让我们实现最后一个类，将多个密集层和softmax层对象组合成一个连贯的前馈顺序神经网络。这将作为另一个类来实现，它将包含其他类。让我们首先从编写构造函数开始——我们将能够在这里设置最大批处理大小，这将影响为此网络分配多少内存——我们将在列表变量`network_mem`中存储一些用于权重和每个层的输入/输出的分配内存。我们还将在列表network中存储DenseLayer和SoftmaxLayer对象，并在network_summary中存储有关NN中每个层的信息。请注意，我们还可以在这里设置一些训练参数，包括delta，用于梯度下降的流的数量（稍后我们将看到），以及训练时期的数量。

我们还可以看到开始时的另一个输入称为layers。在这里，我们可以通过描述每个层来指示NN的构造，构造函数将通过迭代layers的每个元素并调用`add_layer`方法来创建它，接下来我们将实现这个方法：

```py
class SequentialNetwork:
 def __init__(self, layers=None, delta=None, stream = None, max_batch_size=32, max_streams=10, epochs = 10):

 self.network = []
 self.network_summary = []
 self.network_mem = []

 if stream is not None:
  self.stream = stream
 else:
  self.stream = drv.Stream()

 if delta is None:
  delta = 0.0001

 self.delta = delta
 self.max_batch_size=max_batch_size
 self.max_streams = max_streams
 self.epochs = epochs

 if layers is not None:
  for layer in layers:
   add_layer(self, layer)
```

现在，让我们实现`add_layer`方法。我们将使用字典数据类型将有关该层的所有相关信息传递给顺序网络，包括层的类型（密集、softmax等）、输入/输出的数量、权重和偏差。这将向对象的网络和`network_summary`列表变量追加适当的对象和信息，并适当地为`network_mem`列表分配`gpuarray`对象：

```py
def add_layer(self, layer):
 if layer['type'] == 'dense':
  if len(self.network) == 0:
   num_inputs = layer['num_inputs']
  else:
   num_inputs = self.network_summary[-1][2]

  num_outputs = layer['num_outputs']
  sigmoid = layer['sigmoid']
  relu = layer['relu']
  weights = layer['weights']
  b = layer['bias']

  self.network.append(DenseLayer(num_inputs=num_inputs, num_outputs=num_outputs, sigmoid=sigmoid, relu=relu, weights=weights, b=b))
  self.network_summary.append( ('dense', num_inputs, num_outputs))

  if self.max_batch_size > 1:
   if len(self.network_mem) == 0:
self.network_mem.append(gpuarray.empty((self.max_batch_size, self.network_summary[-1][1]), dtype=np.float32))
 self.network_mem.append(gpuarray.empty((self.max_batch_size, self.network_summary[-1][2] ), dtype=np.float32 ) ) 
 else:
 if len(self.network_mem) == 0:
 self.network_mem.append( gpuarray.empty( (self.network_summary[-1][1], ), dtype=np.float32 ) )
 self.network_mem.append( gpuarray.empty((self.network_summary[-1][2], ), dtype=np.float32 ) ) 

 elif layer['type'] == 'softmax':

  if len(self.network) == 0:
   raise Exception("Error! Softmax layer can't be first!")

  if self.network_summary[-1][0] != 'dense':
   raise Exception("Error! Need a dense layer before a softmax layer!")

  num = self.network_summary[-1][2]
  self.network.append(SoftmaxLayer(num=num))
  self.network_summary.append(('softmax', num, num))

  if self.max_batch_size > 1:
   self.network_mem.append(gpuarray.empty((self.max_batch_size, self.network_summary[-1][2] ), dtype=np.float32)) 
  else:
   self.network_mem.append( gpuarray.empty((self.network_summary[-1][2], ), dtype=np.float32))
```

# 推断方法的实现

我们现在将在我们的`SequentialNetwork`类中添加两种推断方法——即，根据特定输入预测输出的方法。我们将首先称之为`predict`，这将由最终用户使用。在训练过程中，我们将不得不基于仅部分层的部分结果进行预测，我们将为此制作另一种方法，称为`partial_predict`。

让我们从实现*predict*开始。这将接受两个输入——以一维或二维NumPy数组形式的样本集，可能还有用户定义的CUDA流。我们将从样本（这里称为`x`）进行一些类型检查和格式化，记住样本将以行方式存储：

```py
def predict(self, x, stream=None):

 if stream is None:
  stream = self.stream

 if type(x) != np.ndarray:
  temp = np.array(x, dtype = np.float32)
  x = temp

 if(x.size == self.network_mem[0].size):
  self.network_mem[0].set_async(x, stream=stream)
 else:

  if x.size > self.network_mem[0].size:
   raise Exception("Error: batch size too large for input.")

  x0 = np.zeros((self.network_mem[0].size,), dtype=np.float32)
  x0[0:x.size] = x.ravel()
  self.network_mem[0].set_async(x0.reshape( self.network_mem[0].shape), stream=stream)

 if(len(x.shape) == 2):
  batch_size = x.shape[0]
 else:
  batch_size = 1
```

现在，让我们执行实际的推断步骤。我们只需遍历整个神经网络，在每一层上执行`eval_`：

```py
for i in xrange(len(self.network)):
 self.network[i].eval_(x=self.network_mem[i], y= self.network_mem[i+1], batch_size=batch_size, stream=stream)
```

现在，我们将提取NN的最终输出，GPU，并将其返回给用户。如果`x`中的样本数量实际上小于最大批处理大小，则在返回之前我们将适当地切片输出数组：

```py
y = self.network_mem[-1].get_async(stream=stream)

if len(y.shape) == 2:
 y = y[0:batch_size, :]

return y
```

现在，完成了这一步，让我们实现`partial_predict`。让我们简要讨论一下这个想法。当我们处于训练过程中时，我们将评估一组样本，然后看看如何通过逐个添加*delta*到每个权重和偏差上会影响输出。为了节省时间，我们可以计算每一层的输出并将它们存储给定的样本集，然后只重新计算我们改变权重的层的输出，以及所有后续层的输出。我们很快就会更深入地了解这个想法，但现在，我们可以这样实现：

```py
def partial_predict(self, layer_index=None, w_t=None, b_t=None, partial_mem=None, stream=None, batch_size=None, delta=None):

 self.network[layer_index].eval_(x=self.network_mem[layer_index], y = partial_mem[layer_index+1], batch_size=batch_size, stream = stream, w_t=w_t, b_t=b_t, delta=delta)

 for i in xrange(layer_index+1, len(self.network)):
  self.network[i].eval_(x=partial_mem[i], y =partial_mem[i+1], batch_size=batch_size, stream = stream)
```

# 梯度下降

我们现在将以**批量随机梯度下降（BSGD）**的形式对我们的NN进行训练方法的完整实现。让我们逐字逐句地思考这意味着什么。**批量**意味着这个训练算法将一次操作一组训练样本，而不是同时处理所有样本，而**随机**表示每个批次是随机选择的。**梯度**意味着我们将使用微积分中的梯度，这里是每个权重和偏差对损失函数的导数集合。最后，**下降**意味着我们试图减少损失函数——我们通过迭代地对权重和偏差进行微小的更改来实现这一点，通过*减去*梯度。

从微积分中我们知道，一个点的梯度总是指向最大*增加*的方向，其相反方向是最大*减少*的方向。因为我们想要*减少*，所以我们减去梯度。

我们现在将在我们的`SequentialNetwork`类中实现BSGD作为`bsgd`方法。让我们逐一讨论`bsgd`的输入参数：

+   `training`将是一个二维NumPy数组的训练样本

+   `labels`将是NN最终层的期望输出，对应于每个训练样本

+   `delta`将指示我们在计算导数时应该增加权重多少

+   `max_streams`将指示BSGD将在其上执行计算的最大并发CUDA流的数量

+   `batch_size`将指示我们希望对每次更新权重计算损失函数的批次有多大

+   `epochs`将指示我们对当前样本集的顺序进行多少次洗牌，分成一系列批次，然后进行BSGD

+   `training_rate`将指示我们使用梯度计算更新权重和偏差的速率

我们将像往常一样开始这个方法，并进行一些检查和类型转换，将CUDA流对象的集合设置为Python列表，并在另一个列表中分配一些额外需要的GPU内存：

```py
def bsgd(self, training=None, labels=None, delta=None, max_streams = None, batch_size = None, epochs = 1, training_rate=0.01):

 training_rate = np.float32(training_rate)

 training = np.float32(training)
 labels = np.float32(labels)

 if( training.shape[0] != labels.shape[0] ):
  raise Exception("Number of training data points should be same as labels!")

 if max_streams is None:
  max_streams = self.max_streams

 if epochs is None:
 epochs = self.epochs

 if delta is None:
 delta = self.delta

 streams = []
 bgd_mem = []

 # create the streams needed for training
 for _ in xrange(max_streams):
  streams.append(drv.Stream())
  bgd_mem.append([])

 # allocate memory for each stream
 for i in xrange(len(bgd_mem)):
  for mem_bank in self.network_mem:
   bgd_mem[i].append( gpuarray.empty_like(mem_bank) )
```

现在，我们可以开始训练。我们将从每个`epoch`开始执行整个BSGD的迭代，对每个epoch对整个数据集进行随机洗牌。我们还将在终端打印一些信息，以便用户在训练过程中获得一些状态更新：

```py
num_points = training.shape[0]

if batch_size is None:
 batch_size = self.max_batch_size

index = range(training.shape[0])

for k in xrange(epochs): 

 print '-----------------------------------------------------------'
 print 'Starting training epoch: %s' % k
 print 'Batch size: %s , Total number of training samples: %s' % (batch_size, num_points)
 print '-----------------------------------------------------------'

 all_grad = []

 np.random.shuffle(index)
```

现在，我们将循环遍历洗牌数据集中的每个批次。我们首先计算当前批次的熵，然后将其打印出来。如果用户看到熵的减少，那么他们将知道梯度下降在这里起作用：

```py
for r in xrange(int(np.floor(training.shape[0]/batch_size))):

 batch_index = index[r*batch_size:(r+1)*batch_size] 

 batch_training = training[batch_index, :]
 batch_labels = labels[batch_index, :]

 batch_predictions = self.predict(batch_training)

 cur_entropy = cross_entropy(predictions=batch_predictions, ground_truth=batch_labels)

 print 'entropy: %s' % cur_entropy
```

我们现在将迭代我们的NN的每个密集层，计算整套权重和偏差的梯度。我们将把这些导数存储在*扁平化*（一维）数组中，这将对应于我们的CUDA内核中的`w_t`和`b_t`索引，它们也是扁平化的。由于我们将有多个流处理不同权重的不同输出，我们将使用Python队列容器来存储尚未处理的这一批权重和偏差：然后我们可以从该容器的顶部弹出值到下一个可用流（我们将这些存储为元组，第一个元素指示这是权重还是偏差）：

```py
for i in xrange(len(self.network)):

 if self.network_summary[i][0] != 'dense':
  continue

 all_weights = Queue()

 grad_w = np.zeros((self.network[i].weights.size,), dtype=np.float32)
 grad_b = np.zeros((self.network[i].b.size,), dtype=np.float32)

 for w in xrange( self.network[i].weights.size ):
  all_weights.put( ('w', np.int32(w) ) )

 for b in xrange( self.network[i].b.size ):
  all_weights.put(('b', np.int32(b) ) )
```

现在，我们需要迭代每一个权重和偏差，我们可以使用`while`循环来检查我们刚刚设置的`queue`对象是否为空。我们将设置另一个队列`stream_weights`，这将帮助我们组织每个流处理的权重和偏差。适当设置权重和偏差输入后，我们现在可以使用`partial_predict`，使用当前流和相应的GPU内存数组：

请注意，我们已经对这批样本执行了`predict`来计算熵，所以我们现在可以对这批样本执行`partial_predict`，只要我们小心使用内存和层。

```py
while not all_weights.empty():

 stream_weights = Queue()

 for j in xrange(max_streams):

  if all_weights.empty():
    break

  wb = all_weights.get()

  if wb[0] == 'w':
   w_t = wb[1]
   b_t = None
  elif wb[0] == 'b':
   b_t = wb[1]
   w_t = None

  stream_weights.put( wb )

  self.partial_predict(layer_index=i, w_t=w_t, b_t=b_t, partial_mem=bgd_mem[j], stream=streams[j], batch_size=batch_size, delta=delta)
```

我们只计算了一小部分权重和偏差的输出预测。我们将不得不为每个计算熵，然后将导数值存储在扁平化的数组中：

```py
for j in xrange(max_streams):

 if stream_weights.empty():
  break

 wb = stream_weights.get()

 w_predictions = bgd_mem[j][-1].get_async(stream=streams[j])

 w_entropy = cross_entropy(predictions=w_predictions[ :batch_size,:], ground_truth=batch_labels)

 if wb[0] == 'w':
  w_t = wb[1]
  grad_w[w_t] = -(w_entropy - cur_entropy) / delta

 elif wb[0] == 'b':
  b_t = wb[1]
  grad_b[b_t] = -(w_entropy - cur_entropy) / delta
```

我们现在已经完成了`while`循环。一旦我们到达外部，我们将知道我们已经计算了这个特定层的所有权重和偏差的导数。在迭代到下一层之前，我们将把当前权重和偏差的梯度计算值附加到`all_grad`列表中。我们还将把扁平化的权重列表重新整形成原始形状：

```py
all_grad.append([np.reshape(grad_w,self.network[i].weights.shape) , grad_b])
```

在迭代每一层之后，我们可以对这一批的NN的权重和偏差进行优化。请注意，如果`training_rate`变量远小于`1`，这将减少权重更新的速度：

```py
for i in xrange(len(self.network)):
 if self.network_summary[i][0] == 'dense':
  new_weights = self.network[i].weights.get()
  new_weights += training_rate*all_grad[i][0]
  new_bias = self.network[i].b.get()
  new_bias += training_rate*all_grad[i][1]
  self.network[i].weights.set(new_weights)
  self.network[i].b.set(new_bias)
```

我们已经完全实现了（非常简单的）基于GPU的DNN！

# 数据的调整和归一化

在我们继续训练和测试全新的NN之前，我们需要退后一步，谈谈**数据调整**和**数据归一化**。NN对数值误差非常敏感，特别是当输入的规模差异很大时。这可以通过正确**调整**我们的训练数据来减轻，这意味着对于输入样本中的每个点，我们将计算所有样本中每个点的平均值和方差，然后在输入到NN进行训练或推断（预测）之前，对每个样本中的每个点减去平均值并除以标准差。这种方法称为**归一化**。让我们组合一个小的Python函数来为我们做这个：

```py
def condition_data(data, means=None, stds=None):

 if means is None:
  means = np.mean(data, axis=0)

 if stds is None:
  stds = np.std(data, axis = 0)

 conditioned_data = data.copy()
 conditioned_data -= means
 conditioned_data /= stds

 return (conditioned_data, means, stds)
```

# 鸢尾花数据集

我们现在将为一个真实问题构建我们自己的DNN：根据花瓣的测量来对花的类型进行分类。我们将使用众所周知的*鸢尾花数据集*。该数据集存储为逗号分隔值（CSV）文本文件，每行包含四个不同的数值（花瓣测量），然后是花的类型（这里有三个类别——*山鸢尾*、*变色鸢尾*和*维吉尼亚鸢尾*）。我们现在将设计一个小型DNN，根据这个数据集对鸢尾花的类型进行分类。

在我们继续之前，请下载鸢尾花数据集并将其放入您的工作目录。这可以从UC Irvine机器学习存储库中获取，网址为：[https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)。

我们将首先将此文件处理成适当的数据数组，以便用于训练和验证我们的DNN。让我们从打开我们的主函数开始；我们需要将花的名称转换为DNN可以输出的实际类别，因此让我们创建一个小字典，为每个类别提供相应的标签。我们还将设置一些空列表来存储我们的训练数据和标签：

```py
if __name__ == '__main__':
 to_class = { 'Iris-setosa' : [1,0,0] , 'Iris-versicolor' : [0,1,0], 'Iris-virginica' : [0,0,1]}

 iris_data = []
 iris_labels = []
```

现在，让我们从CSV文件中读取。我们将使用Python的`csv`模块中的`reader`函数，这是我们之前导入的：

```py
with open('C:/Users/btuom/examples/9/iris.data', 'rb') as csvfile:
 csvreader = csv.reader(csvfile, delimiter=',')
 for row in csvreader:
  newrow = []
  if len(row) != 5:
   break
  for i in range(4):
   newrow.append(row[i])
  iris_data.append(newrow)
  iris_labels.append(to_class[row[4]])
```

我们现在将随机洗牌数据，并使用三分之二的样本作为训练数据。剩下的三分之一将用于测试（验证）数据：

```py
iris_len = len(iris_data)
shuffled_index = list(range(iris_len))
np.random.shuffle(shuffled_index)

iris_data = np.float32(iris_data)
iris_labels = np.float32(iris_labels)
iris_data = iris_data[shuffled_index, :]
iris_labels = iris_labels[shuffled_index,:]

t_len = (2*iris_len) // 3

iris_train = iris_data[:t_len, :]
label_train = iris_labels[:t_len, :]

iris_test = iris_data[t_len:,:]
label_test = iris_labels[t_len:, :]
```

现在，最后，我们可以开始构建我们的DNN！首先，让我们创建一个`SequentialNetwork`对象。我们将`max_batch_size`设置为`32`：

```py
sn = SequentialNetwork( max_batch_size=32 )
```

现在，让我们创建我们的NN。这将包括四个密集层（两个隐藏层）和一个softmax层。我们将逐层增加神经元的数量，直到最后一层，最后一层只有三个输出（每个类别一个）。每层神经元数量的增加允许我们捕捉数据的一些细微差别：

```py

sn.add_layer({'type' : 'dense', 'num_inputs' : 4, 'num_outputs' : 10, 'relu': True, 'sigmoid': False, 'weights' : None, 'bias' : None} ) 
sn.add_layer({'type' : 'dense', 'num_inputs' : 10, 'num_outputs' : 15, 'relu': True, 'sigmoid': False, 'weights': None, 'bias' : None} ) 
sn.add_layer({'type' : 'dense', 'num_inputs' : 15, 'num_outputs' : 20, 'relu': True, 'sigmoid': False, 'weights': None, 'bias' : None} ) 
sn.add_layer({'type' : 'dense', 'num_inputs' : 20, 'num_outputs' : 3, 'relu': True, 'sigmoid': False, 'weights': None , 'bias': None } ) 
sn.add_layer({'type' : 'softmax'})
```

我们现在将调整我们的训练数据，并使用我们刚刚实现的BSGD方法进行训练。我们将使用`batch_size`设置为`16`，`max_streams`设置为`10`，`epochs`的数量设置为100，`delta`设置为0.0001，`training_rate`设置为1——这些将是几乎任何现代GPU可接受的参数。我们还将计时训练过程，这可能会非常耗时。

```py
ctrain, means, stds = condition_data(iris_train)

t1 = time()
sn.bsgd(training=ctrain, labels=label_train, batch_size=16, max_streams=20, epochs=100 , delta=0.0001, training_rate=1)
training_time = time() - t1
```

现在，我们的DNN已经完全训练好了。我们准备开始验证过程！让我们设置一个名为`hits`的Python变量来计算正确分类的总数。我们还需要对验证/测试数据进行条件设置。还有一件事——我们通过DNN的softmax层的最大值对应的索引来确定类别。我们可以使用NumPy的`argmax`函数来检查这是否给出了正确的分类，就像这样：

```py
hits = 0
ctest, _, _ = condition_data(iris_test, means=means, stds=stds)
for i in range(ctest.shape[0]):
 if np.argmax(sn.predict(ctest[i,:])) == np.argmax( label_test[i,:]):
  hits += 1
```

现在，我们准备检查我们的DNN实际工作得有多好。让我们输出准确率以及总训练时间：

```py
print 'Percentage Correct Classifications: %s' % (float(hits ) / ctest.shape[0])
print 'Total Training Time: %s' % training_time
```

现在，我们完成了。我们现在可以完全用Python和CUDA实现一个DNN！一般来说，您可以期望在这个特定问题上的准确率在80%-97%之间，使用任何Pascal级别的GPU的训练时间为10-20分钟。

本章的代码可在本书的GitHub存储库的适当目录下的`deep_neural_network.py`文件中找到。

# 总结

在本章中，我们首先给出了人工神经网络的定义，并向您展示了如何将单个AN组合成密集层，然后再将其组合成完整的深度神经网络。然后，我们在CUDA-C中实现了一个密集层，并制作了一个相应的Python包装类。我们还包括了在密集层的输出上添加ReLU和sigmoid层的功能。我们看到了使用softmax层的定义和动机，这用于分类问题，然后在CUDA-C和Python中实现了这一功能。最后，我们实现了一个Python类，以便我们可以从先前的类构建一个顺序前馈DNN；我们实现了一个交叉熵损失函数，然后在我们的梯度下降实现中使用这个损失函数来训练我们DNN中的权重和偏差。最后，我们使用我们的实现在真实数据集上构建、训练和测试了一个DNN。

现在我们对我们的CUDA编程能力有了很大的自信，因为我们可以编写自己基于GPU的DNN！我们现在将在接下来的两章中学习一些非常高级的内容，我们将看看如何编写我们自己的接口到编译后的CUDA代码，以及一些关于NVIDIA GPU非常技术性的细节。

# 问题

1.  假设您构建了一个DNN，并在训练后，它只产生垃圾。经过检查，您发现所有的权重和偏差要么是巨大的数字，要么是NaN。问题可能是什么？

1.  小`training_rate`值可能存在的一个问题是什么？

1.  大`training_rate`值可能存在的一个问题是什么？

1.  假设我们想要训练一个DNN，将多个标签分配给动物的图像（“黏滑的”，“毛茸茸的”，“红色的”，“棕色的”等等）。在DNN的末尾应该使用sigmoid还是softmax层？

1.  假设我们想要将单个动物的图像分类为猫或狗。我们应该使用sigmoid还是softmax？

1.  如果我们减小批量大小，梯度下降训练过程中权重和偏差的更新会更多还是更少？
