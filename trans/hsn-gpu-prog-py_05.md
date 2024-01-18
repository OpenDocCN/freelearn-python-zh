# 流、事件、上下文和并发性

在之前的章节中，我们看到在与GPU交互时，主机执行的两个主要操作是：

+   将内存数据从GPU复制到GPU，以及从GPU复制到内存

+   启动内核函数

我们知道，在单个内核中，其许多线程之间存在一定程度的并发性；然而，在多个内核和GPU内存操作之间，还存在另一种并发性。这意味着我们可以同时启动多个内存和内核操作，而无需等待每个操作完成。然而，另一方面，我们必须有一定的组织能力，以确保所有相互依赖的操作都得到同步；这意味着我们不应该在其输入数据完全复制到设备内存之前启动特定的内核，或者在内核执行完成之前，不应该将已启动内核的输出数据复制到主机。

为此，我们有所谓的**CUDA** **流**—**流**是在GPU上按顺序运行的一系列操作。单独一个流是没有用的—重点是通过使用多个流在主机上发出的GPU操作来获得并发性。这意味着我们应该交错启动与不同流对应的GPU操作，以利用这个概念。

我们将在本章中广泛涵盖流的概念。此外，我们还将研究**事件**，这是流的一个特性，用于精确计时内核，并指示主机在给定流中已完成哪些操作。

最后，我们将简要介绍CUDA **上下文**。**上下文**可以被视为类似于操作系统中的进程，因为GPU将每个上下文的数据和内核代码*隔离*并封装起来，使其与当前存在于GPU上的其他上下文相分离。我们将在本章末尾看到这方面的基础知识。

本章的学习成果如下：

+   理解设备和流同步的概念

+   学习如何有效地使用流来组织并发的GPU操作

+   学习如何有效地使用CUDA事件

+   理解CUDA上下文

+   学习如何在给定上下文中显式同步

+   学习如何显式创建和销毁CUDA上下文

+   学习如何使用上下文允许主机上的多个进程和线程共享GPU使用

# 技术要求

本章需要一台带有现代NVIDIA GPU（2016年以后）的Linux或Windows 10 PC，并安装了所有必要的GPU驱动程序和CUDA Toolkit（9.0及以上）。还需要一个合适的Python 2.7安装（如Anaconda Python 2.7），并安装了PyCUDA模块。

本章的代码也可以在GitHub上找到：

[https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)

有关先决条件的更多信息，请查看本书的*前言*，有关软件和硬件要求，请查看[https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)中的README。

# CUDA设备同步

在我们可以使用CUDA流之前，我们需要了解**设备同步**的概念。这是一种操作，其中主机阻塞任何进一步的执行，直到发给GPU的所有操作（内存传输和内核执行）都已完成。这是为了确保依赖于先前操作的操作不会被无序执行—例如，确保CUDA内核启动在主机尝试读取其输出之前已完成。

在CUDA C中，设备同步是通过`cudaDeviceSynchronize`函数执行的。这个函数有效地阻止了主机上的进一步执行，直到所有GPU操作完成。`cudaDeviceSynchronize`是如此基本，以至于它通常是CUDA C大多数书籍中最早涉及的主题之一-我们还没有看到这一点，因为PyCUDA已经在需要时自动为我们调用了这个函数。让我们看一个CUDA C代码的例子，看看如何手动完成这个操作：

```py
// Copy an array of floats from the host to the device.
cudaMemcpy(device_array, host_array, size_of_array*sizeof(float), cudaMemcpyHostToDevice);
// Block execution until memory transfer to device is complete.
cudaDeviceSynchronize();
// Launch CUDA kernel.
Some_CUDA_Kernel <<< block_size, grid_size >>> (device_array, size_of_array);
// Block execution until GPU kernel function returns.
cudaDeviceSynchronize();
// Copy output of kernel to host.
cudaMemcpy(host_array,  device_array, size_of_array*sizeof(float), cudaMemcpyDeviceToHost); // Block execution until memory transfer to host is complete.
cudaDeviceSynchronize();
```

在这段代码块中，我们看到我们必须在每个单独的GPU操作之后直接与设备进行同步。如果我们只需要一次调用单个CUDA内核，就像这里看到的那样，这是可以的。但是，如果我们想要同时启动多个独立的内核和操作不同数据数组的内存操作，跨整个设备进行同步将是低效的。在这种情况下，我们应该跨多个流进行同步。我们现在就来看看如何做到这一点。

# 使用PyCUDA流类

我们将从一个简单的PyCUDA程序开始；它只是生成一系列随机的GPU数组，对每个数组进行简单的内核处理，然后将数组复制回主机。然后我们将修改它以使用流。请记住，这个程序将完全没有任何意义，只是为了说明如何使用流以及一些基本的性能提升。（这个程序可以在GitHub存储库的`5`目录下的`multi-kernel.py`文件中看到。）

当然，我们将首先导入适当的Python模块，以及`time`函数：

```py
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time
```

我们现在将指定要处理多少个数组-在这里，每个数组将由不同的内核启动进行处理。我们还指定要生成的随机数组的长度，如下所示：

```py
num_arrays = 200
array_len = 1024**2
```

我们现在有一个在每个数组上操作的内核；它只是迭代数组中的每个点，并将其乘以2再除以2，重复50次，最终保持数组不变。我们希望限制每个内核启动将使用的线程数，这将帮助我们在GPU上的许多内核启动之间获得并发性，以便我们可以让每个线程通过`for`循环迭代数组的不同部分。（再次提醒，这个内核函数除了用来学习流和同步之外，完全没有任何用处！）如果每个内核启动使用的线程太多，以后获得并发性将更加困难：

```py
ker = SourceModule(""" 
__global__ void mult_ker(float * array, int array_len)
{
     int thd = blockIdx.x*blockDim.x + threadIdx.x;
     int num_iters = array_len / blockDim.x;

     for(int j=0; j < num_iters; j++)
     {
         int i = j * blockDim.x + thd;

         for(int k = 0; k < 50; k++)
         {
              array[i] *= 2.0;
              array[i] /= 2.0;
         }
     }
}
""")

mult_ker = ker.get_function('mult_ker')
```

现在，我们将生成一些随机数据数组，将这些数组复制到GPU，迭代地在每个数组上启动我们的内核，然后将输出数据复制回主机，并使用NumPy的`allclose`函数来断言它们是否相同。我们将使用Python的`time`函数来计算从开始到结束的所有操作的持续时间：

```py
data = []
data_gpu = []
gpu_out = []

# generate random arrays.
for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))

t_start = time()

# copy arrays to GPU.
for k in range(num_arrays):
    data_gpu.append(gpuarray.to_gpu(data[k]))

# process arrays.
for k in range(num_arrays):
    mult_ker(data_gpu[k], np.int32(array_len), block=(64,1,1), grid=(1,1,1))

# copy arrays from GPU.
for k in range(num_arrays):
    gpu_out.append(data_gpu[k].get())

t_end = time()

for k in range(num_arrays):
    assert (np.allclose(gpu_out[k], data[k]))

print 'Total time: %f' % (t_end - t_start)
```

我们现在准备运行这个程序。我现在就要运行它：

![](assets/e590fec8-0e98-4fce-bd4e-091871758825.png)

所以，这个程序完成需要了将近三秒的时间。我们将进行一些简单的修改，使我们的程序可以使用流，然后看看我们是否可以获得任何性能提升（这可以在存储库中的`multi-kernel_streams.py`文件中看到）。

首先，我们注意到对于每个内核启动，我们有一个单独的数据数组进行处理，这些数组存储在Python列表中。我们将不得不为每个单独的数组/内核启动对创建一个单独的流对象，所以让我们首先添加一个空列表，名为`streams`，用来保存我们的流对象：

```py
data = []
data_gpu = []
gpu_out = []
streams = []
```

我们现在可以生成一系列流，我们将使用它们来组织内核启动。我们可以使用`pycuda.driver`子模块和`Stream`类来获取流对象。由于我们已经导入了这个子模块并将其别名为`drv`，我们可以用以下方式用新的流对象填充我们的列表：

```py
for _ in range(num_arrays):
    streams.append(drv.Stream())
```

现在，我们将首先修改将数据传输到GPU的内存操作。考虑以下步骤：

1.  查找第一个循环，使用`gpuarray.to_gpu`函数将数组复制到GPU。我们将希望切换到此函数的异步和适用于流的版本`gpu_array.to_gpu_async`。（现在我们还必须使用`stream`参数指定每个内存操作应该使用哪个流）：

```py
for k in range(num_arrays):
    data_gpu.append(gpuarray.to_gpu_async(data[k], stream=streams[k]))
```

1.  我们现在可以启动我们的内核。这与以前完全相同，只是我们必须通过使用`stream`参数来指定要使用哪个流：

```py
for k in range(num_arrays):
    mult_ker(data_gpu[k], np.int32(array_len), block=(64,1,1), grid=(1,1,1), stream=streams[k])
```

1.  最后，我们需要从GPU中取出数据。我们可以通过将`gpuarray get`函数切换为`get_async`来实现这一点，并再次使用`stream`参数，如下所示：

```py
for k in range(num_arrays):
    gpu_out.append(data_gpu[k].get_async(stream=streams[k]))
```

我们现在准备运行我们适用于流的修改后的程序：

![](assets/1e11ea75-4947-459b-a915-363bdad7b241.png)

在这种情况下，我们获得了三倍的性能提升，考虑到我们需要进行的修改非常少，这并不算太糟糕。但在我们继续之前，让我们尝试更深入地理解为什么这样做有效。

让我们考虑两个CUDA内核启动的情况。在我们启动内核之前和之后，我们还将执行与每个内核相对应的GPU内存操作，总共有六个操作。我们可以通过图表来可视化在GPU上随时间发生的操作，如此—在*x*轴上向右移动对应于时间持续时间，而*y*轴对应于在特定时间执行的GPU上的操作。这可以用以下图表来描述：

![](assets/9c272bdc-c6e1-4438-96ad-392af521175d.png)

很容易想象为什么流在性能提高方面效果如此好——因为单个流中的操作直到所有*必要*的先前操作完成之后才会被阻塞，我们将获得不同GPU操作之间的并发性，并充分利用我们的设备。这可以通过并发操作的大量重叠来看出。我们可以将基于流的并发性随时间的变化可视化如下：

![](assets/d01793e3-d5a6-4ff8-b3fc-f22c198c7962.png)

# 使用CUDA流进行并发康威生命游戏

我们现在将看到一个更有趣的应用——我们将修改上一章的LIFE（康威的*生命游戏*）模拟，以便我们将同时显示四个独立的动画窗口。（建议您查看上一章的这个示例，如果您还没有的话。）

让我们从上一章的存储库中获取旧的LIFE模拟，应该在`4`目录中的`conway_gpu.py`下。现在我们将把这个修改为基于新的CUDA流的并发LIFE模拟。（我们将在稍后看到的这种基于流的新模拟也可以在本章目录`5`中的`conway_gpu_streams.py`文件中找到。）

转到文件末尾的主函数。我们将设置一个新变量，用`num_concurrent`表示我们将同时显示多少个并发动画（其中`N`表示模拟栅格的高度/宽度，与以前一样）。我们在这里将其设置为`4`，但您可以随意尝试其他值：

```py
if __name__ == '__main__':

    N = 128
    num_concurrent = 4
```

我们现在需要一组`num_concurrent`流对象，并且还需要在GPU上分配一组输入和输出栅格。当然，我们只需将这些存储在列表中，并像以前一样初始化栅格。我们将设置一些空列表，并通过循环填充每个适当的对象，如下所示（请注意我们如何在每次迭代中设置一个新的初始状态栅格，将其发送到GPU，并将其连接到`lattices_gpu`）：

```py
streams = []
lattices_gpu = []
newLattices_gpu = []

for k in range(num_concurrent):
    streams.append(drv.Stream())
    lattice = np.int32( np.random.choice([1,0], N*N, p=[0.25, 0.75]).reshape(N, N) )
    lattices_gpu.append(gpuarray.to_gpu(lattice)) 
    newLattices_gpu.append(gpuarray.empty_like(lattices_gpu[k])) 
```

由于我们只在程序启动期间执行此循环一次，并且几乎所有的计算工作都将在动画循环中进行，因此我们实际上不必担心实际使用我们刚刚生成的流。

我们现在将使用Matplotlib使用子图函数设置环境；请注意，我们可以通过设置`ncols`参数来设置多个动画图。我们将有另一个列表结构，它将对应于动画更新中所需的图像`imgs`。请注意，我们现在可以使用`get_async`和相应的流来设置这个：

```py
fig, ax = plt.subplots(nrows=1, ncols=num_concurrent)
imgs = []

for k in range(num_concurrent):
    imgs.append( ax[k].imshow(lattices_gpu[k].get_async(stream=streams[k]), interpolation='nearest') )

```

在主函数中要更改的最后一件事是以`ani = animation.FuncAnimation`开头的倒数第二行。让我们修改`update_gpu`函数的参数，以反映我们正在使用的新列表，并添加两个参数，一个用于传递我们的`streams`列表，另一个用于指示应该有多少并发动画的参数：

```py
ani = animation.FuncAnimation(fig, update_gpu, fargs=(imgs, newLattices_gpu, lattices_gpu, N, streams, num_concurrent) , interval=0, frames=1000, save_count=1000)    
```

现在我们需要对`update_gpu`函数进行必要的修改以接受这些额外的参数。在文件中向上滚动一点，并按照以下方式修改参数：

`def update_gpu(frameNum, imgs, newLattices_gpu, lattices_gpu, N, streams, num_concurrent)`:

现在，我们需要修改这个函数，使其迭代`num_concurrent`次，并像以前一样设置`imgs`的每个元素，最后返回整个`imgs`列表：

```py
for k in range(num_concurrent):
    conway_ker( newLattices_gpu[k], lattices_gpu[k], grid=(N/32,N/32,1), block=(32,32,1), stream=streams[k] )
     imgs[k].set_data(newLattices_gpu[k].get_async(stream=streams[k]) )
     lattices_gpu[k].set_async(newLattices_gpu[k], stream=streams[k])

 return imgs
```

注意我们所做的更改——每个内核都在适当的流中启动，而`get`已经切换为与相同流同步的`get_async`。

最后，在循环中的最后一行将GPU数据从一个设备数组复制到另一个设备数组，而不进行任何重新分配。在此之前，我们可以使用简写切片操作符`[:]`直接在数组之间复制元素，而不在GPU上重新分配任何内存；在这种情况下，切片操作符符号充当PyCUDA `set`函数的别名，用于GPU数组。 （当然，`set`是将一个GPU数组复制到另一个相同大小的数组，而不进行任何重新分配的函数。）幸运的是，确实有一个流同步版本的这个函数，`set_async`，但我们需要明确地使用这个函数来调用这个函数，明确指定要复制的数组和要使用的流。

我们现在已经完成并准备运行。转到终端并在命令行输入`python conway_gpu_streams.py`来观看展示：

![](assets/93d56393-5968-409d-bcab-d56330f6bc91.png)

# 事件

**事件**是存在于GPU上的对象，其目的是作为操作流的里程碑或进度标记。事件通常用于在设备端精确计时操作的时间持续时间；到目前为止，我们所做的测量都是使用基于主机的Python分析器和标准Python库函数，如`time`。此外，事件还可以用于向主机提供有关流状态和已完成的操作的状态更新，以及用于显式基于流的同步。

让我们从一个不使用显式流并使用事件来测量仅一个单个内核启动的示例开始。（如果我们的代码中没有显式使用流，CUDA实际上会隐式定义一个默认流，所有操作都将放入其中）。

在这里，我们将使用与本章开头相同的无用的乘法/除法循环内核和标题，并修改大部分以下内容。我们希望为此示例运行一个长时间的单个内核实例，因此我们将生成一个巨大的随机数数组供内核处理，如下所示：

```py
array_len = 100*1024**2
data = np.random.randn(array_len).astype('float32')
data_gpu = gpuarray.to_gpu(data)
```

现在，我们使用`pycuda.driver.Event`构造函数构造我们的事件（当然，`pycuda.driver`已经被我们之前的导入语句别名为`drv`）。

我们将在这里创建两个事件对象，一个用于内核启动的开始，另一个用于内核启动的结束（我们将始终需要*两个*事件对象来测量任何单个GPU操作，很快我们将看到）：

```py
start_event = drv.Event()
end_event = drv.Event()
```

现在，我们准备启动我们的内核，但首先，我们必须使用事件记录函数标记`start_event`实例在执行流中的位置。我们启动内核，然后标记`end_event`在执行流中的位置，并且使用`record`：

```py
start_event.record()
mult_ker(data_gpu, np.int32(array_len), block=(64,1,1), grid=(1,1,1))
end_event.record()
```

事件具有一个二进制值，指示它们是否已经到达，这是由函数query给出的。让我们在内核启动后立即为两个事件打印状态更新：

```py
print 'Has the kernel started yet? {}'.format(start_event.query())
 print 'Has the kernel ended yet? {}'.format(end_event.query())
```

现在让我们运行这个，看看会发生什么：

![](assets/b7cd2aa1-cb0c-485a-939b-cf2d7fe35d1e.png)

我们的目标是最终测量内核执行的时间持续，但内核似乎还没有启动。在PyCUDA中，内核是异步启动的（无论它们是否存在于特定流中），因此我们必须确保我们的主机代码与GPU正确同步。

由于`end_event`是最后一个，我们可以通过此事件对象的`synchronize`函数阻止进一步的主机代码执行，直到内核完成；这将确保在执行任何进一步的主机代码之前内核已经完成。让我们在适当的位置添加一行代码来做到这一点：

```py
end_event.synchronize()

print 'Has the kernel started yet?  {}'.format(start_event.query())

print 'Has the kernel ended yet? {}'.format(end_event.query())
```

最后，我们准备测量内核的执行时间；我们可以使用事件对象的`time_till`或`time_since`操作来与另一个事件对象进行比较，以获取这两个事件之间的时间（以毫秒为单位）。让我们使用`start_event`的`time_till`操作在`end_event`上：

```py
print 'Kernel execution time in milliseconds: %f ' % start_event.time_till(end_event)
```

时间持续可以通过GPU上已经发生的两个事件之间的`time_till`和`time_since`函数来测量。请注意，这些函数始终以毫秒为单位返回值！

现在让我们再次运行我们的程序：

![](assets/29aadcbf-d395-487a-ac70-f3c422ae6f12.png)

（此示例也可在存储库中的`simple_event_example.py`文件中找到。）

# 事件和流

我们现在将看到如何在流方面使用事件对象；这将使我们对各种GPU操作的流程具有高度复杂的控制，使我们能够准确了解每个单独流的进展情况，甚至允许我们在忽略其他流的情况下与主机同步特定流。

首先，我们必须意识到这一点——每个流必须有自己专用的事件对象集合；多个流不能共享一个事件对象。让我们通过修改之前的示例`multi_kernel_streams.py`来确切了解这意味着什么。在内核定义之后，让我们添加两个额外的空列表——`start_events`和`end_events`。我们将用事件对象填充这些列表，这些事件对象将对应于我们拥有的每个流。这将允许我们在每个流中计时一个GPU操作，因为每个GPU操作都需要两个事件：

```py
data = []
data_gpu = []
gpu_out = []
streams = []
start_events = []
end_events = []

for _ in range(num_arrays):
    streams.append(drv.Stream())
    start_events.append(drv.Event())
    end_events.append(drv.Event())
```

现在，我们可以通过修改第二个循环来逐个计时每个内核启动，使用事件的开始和结束记录。请注意，由于这里有多个流，我们必须将适当的流作为参数输入到每个事件对象的`record`函数中。还要注意，我们可以在第二个循环中捕获结束事件；这仍然可以完美地捕获内核执行持续时间，而不会延迟启动后续内核。现在考虑以下代码：

```py
for k in range(num_arrays):
    start_events[k].record(streams[k])
    mult_ker(data_gpu[k], np.int32(array_len), block=(64,1,1), grid=(1,1,1), stream=streams[k])

for k in range(num_arrays):
    end_events[k].record(streams[k])
```

现在我们将提取每个单独内核启动的持续时间。让我们在迭代断言检查之后添加一个新的空列表，并通过`time_till`函数将其填充为持续时间：

```py
kernel_times = []
for k in range(num_arrays):
   kernel_times.append(start_events[k].time_till(end_events[k]))
```

现在让我们在最后添加两个`print`语句，告诉我们内核执行时间的平均值和标准偏差：

```py
print 'Mean kernel duration (milliseconds): %f' % np.mean(kernel_times)
print 'Mean kernel standard deviation (milliseconds): %f' % np.std(kernel_times)
```

我们现在可以运行这个：

![](assets/40d38973-beee-4d08-8e0b-3847ae757c8e.png)

（此示例也可在存储库中的`multi-kernel_events.py`中找到。）

我们看到内核持续时间的标准偏差相对较低，这是好事，考虑到每个内核在相同的块和网格大小上处理相同数量的数据——如果存在较高的偏差，那将意味着我们在内核执行中使用GPU的不均匀程度很高，我们将不得不重新调整参数以获得更高的并发水平。

# 上下文

CUDA上下文通常被描述为类似于操作系统中的进程。让我们来看看这意味着什么——进程是计算机上运行的单个程序的实例；操作系统内核之外的所有程序都在一个进程中运行。每个进程都有自己的一组指令、变量和分配的内存，一般来说，它对其他进程的操作和内存是盲目的。当一个进程结束时，操作系统内核会进行清理，确保进程分配的所有内存都已被释放，并关闭进程使用的任何文件、网络连接或其他资源。（好奇的Linux用户可以使用命令行`top`命令查看计算机上运行的进程，而Windows用户可以使用Windows任务管理器查看）。

类似于进程，上下文与正在使用GPU的单个主机程序相关联。上下文在内存中保存了所有正在使用的CUDA内核和分配的内存，并且对其他当前存在的上下文的内核和内存是盲目的。当上下文被销毁（例如，在基于GPU的程序结束时），GPU会清理上下文中的所有代码和分配的内存，为其他当前和未来的上下文释放资源。到目前为止，我们编写的程序都存在于一个单一的上下文中，因此这些操作和概念对我们来说是不可见的。

让我们也记住，一个单独的程序开始时是一个单独的进程，但它可以分叉自身以在多个进程或线程中运行。类似地，一个单独的CUDA主机程序可以在GPU上生成和使用多个CUDA上下文。通常，当我们分叉主机进程的新进程或线程时，我们将创建一个新的上下文以获得主机端的并发性。（然而，应该强调的是，主机进程和CUDA上下文之间没有确切的一对一关系）。

与生活中的许多其他领域一样，我们将从一个简单的例子开始。我们将首先看看如何访问程序的默认上下文并在其间进行同步。

# 同步当前上下文

我们将看到如何在Python中显式同步我们的设备上下文，就像在CUDA C中一样；这实际上是CUDA C中最基本的技能之一，在其他大多数关于这个主题的书籍的第一章或第二章中都有涵盖。到目前为止，我们已经能够避免这个话题，因为PyCUDA自动使用`pycuda.gpuarray`函数（如`to_gpu`或`get`）执行大多数同步；否则，在本章开头我们看到的`to_gpu_async`或`get_async`函数的情况下，同步是由流处理的。

我们将谦卑地开始修改我们在[第3章](6ab0cd69-e439-4cfb-bf1a-4247ec58c94e.xhtml)中编写的程序，*使用PyCUDA入门*，它使用显式上下文同步生成Mandelbrot集的图像。（这在存储库的`3`目录下的文件`gpu_mandelbrot0.py`中可用。）

我们在这里不会获得任何性能提升，与我们最初的Mandelbrot程序相比；这个练习的唯一目的是帮助我们理解CUDA上下文和GPU同步。

看一下头部，我们当然看到了`import pycuda.autoinit`一行。我们可以使用`pycuda.autoinit.context`访问当前上下文对象，并且可以通过调用`pycuda.autoinit.context.synchronize()`函数在当前上下文中进行同步。

现在让我们修改`gpu_mandelbrot`函数以处理显式同步。我们看到的第一行与GPU相关的代码是这样的：

`mandelbrot_lattice_gpu = gpuarray.to_gpu(mandelbrot_lattice)`

我们现在可以将其更改为显式同步。我们可以使用`to_gpu_async`异步将数据复制到GPU，然后进行同步，如下所示：

```py
mandelbrot_lattice_gpu = gpuarray.to_gpu_async(mandelbrot_lattice) pycuda.autoinit.context.synchronize()
```

然后我们看到下一行使用`gpuarray.empty`函数在GPU上分配内存。由于GPU架构的性质，CUDA中的内存分配始终是自动同步的；这里没有*异步*内存分配的等价物。因此，我们保持这行与之前一样。

CUDA中的内存分配始终是同步的！

现在我们看到接下来的两行 - 我们的Mandelbrot内核通过调用`mandel_ker`启动，并且我们通过调用`get`复制了Mandelbrot `gpuarray`对象的内容。在内核启动后我们进行同步，将`get`切换为`get_async`，最后进行最后一行同步：

```py
mandel_ker( mandelbrot_lattice_gpu, mandelbrot_graph_gpu, np.int32(max_iters), np.float32(upper_bound))
pycuda.autoinit.context.synchronize()
mandelbrot_graph = mandelbrot_graph_gpu.get_async()
pycuda.autoinit.context.synchronize()
```

现在我们可以运行这个程序，它将像[第3章](6ab0cd69-e439-4cfb-bf1a-4247ec58c94e.xhtml)中的*Getting Started with PyCUDA*一样在磁盘上生成一个Mandelbrot图像。

（此示例也可在存储库中的`gpu_mandelbrot_context_sync.py`中找到。）

# 手动上下文创建

到目前为止，我们一直在所有PyCUDA程序的开头导入`pycuda.autoinit`；这实际上在程序开始时创建一个上下文，并在结束时销毁它。

让我们尝试手动操作。我们将制作一个小程序，只是将一个小数组复制到GPU，然后将其复制回主机，打印数组，然后退出。

我们从导入开始：

```py
import numpy as np
from pycuda import gpuarray
import pycuda.driver as drv
```

首先，我们使用`pycuda.driver.init`函数初始化CUDA，这里被别名为`drv`：

```py
drv.init()
```

现在我们选择要使用的GPU；在拥有多个GPU的情况下是必要的。我们可以使用`pycuda.driver.Device`选择特定的GPU；如果您只有一个GPU，就像我一样，可以使用`pycuda.driver.Device(0)`访问它，如下所示：

```py
dev = drv.Device(0)
```

现在我们可以使用`make_context`在此设备上创建一个新上下文，如下所示：

```py
ctx = dev.make_context()
```

现在我们有了一个新的上下文，这将自动成为默认上下文。让我们将一个数组复制到GPU，然后将其复制回主机并打印出来：

```py
x = gpuarray.to_gpu(np.float32([1,2,3]))
print x.get()
```

现在我们完成了。我们可以通过调用`pop`函数销毁上下文：

```py
ctx.pop()
```

就是这样！我们应该始终记得在程序退出之前销毁我们显式创建的上下文。

（此示例可以在存储库中的本章目录下的`simple_context_create.py`文件中找到。）

# 主机端多进程和多线程

当然，我们可以通过在主机的CPU上使用多个进程或线程来实现并发。让我们现在明确区分主机端操作系统进程和线程。

每个存在于操作系统内核之外的主机端程序都作为一个进程执行，并且也可以存在于多个进程中。进程有自己的地址空间，因为它与所有其他进程同时运行并独立运行。进程通常对其他进程的操作视而不见，尽管多个进程可以通过套接字或管道进行通信。在Linux和Unix中，使用fork系统调用生成新进程。

相比之下，主机端线程存在于单个进程中，多个线程也可以存在于单个进程中。单个进程中的多个线程并发运行。同一进程中的所有线程共享进程内的相同地址空间，并且可以访问相同的共享变量和数据。通常，资源锁用于在多个线程之间访问数据，以避免竞争条件。在编译语言（如C、C++或Fortran）中，多个进程线程通常使用Pthreads或OpenMP API进行管理。

线程比进程更轻量级，操作系统内核在单个进程中的多个线程之间切换任务要比在多个进程之间切换任务快得多。通常，操作系统内核会自动在不同的CPU核心上执行不同的线程和进程，以建立真正的并发性。

Python的一个特点是，虽然它通过`threading`模块支持多线程，但所有线程都将在同一个CPU核心上执行。这是由于Python是一种解释脚本语言的技术细节所致，与Python的全局标识符锁（GIL）有关。要通过Python在主机上实现真正的多核并发，不幸的是，我们必须使用`multiprocessing`模块生成多个进程。（不幸的是，由于Windows处理进程的方式，`multiprocessing`模块目前在Windows下并不完全可用。Windows用户将不幸地不得不坚持单核多线程，如果他们想要在主机端实现任何形式的并发。）

我们现在将看到如何在Python中使用两个线程来使用基于GPU的操作；Linux用户应该注意，通过将`threading`的引用切换到`multiprocessing`，将`Thread`的引用切换到`Process`，可以很容易地将其扩展到进程，因为这两个模块看起来和行为都很相似。然而，由于PyCUDA的性质，我们将不得不为每个将使用GPU的线程或进程创建一个新的CUDA上下文。让我们立即看看如何做到这一点。

# 主机并发的多个上下文

让我们首先简要回顾一下如何在Python中创建一个可以通过简单示例返回值给主机的单个主机线程。（这个例子也可以在存储库中的`single_thread_example.py`文件的`5`下看到。）我们将使用`threading`模块中的`Thread`类来创建`Thread`的子类，如下所示：

```py
import threading
class PointlessExampleThread(threading.Thread):
```

现在我们设置我们的构造函数。我们调用父类的构造函数，并在对象中设置一个空变量，这将是线程返回的值：

```py
def __init__(self):
    threading.Thread.__init__(self)
    self.return_value = None
```

现在我们在我们的线程类中设置运行函数，这是在启动线程时将被执行的内容。我们只需要让它打印一行并设置返回值：

```py
def run(self):
    print 'Hello from the thread you just spawned!'
    self.return_value = 123
```

最后，我们需要设置join函数。这将允许我们从线程中接收一个返回值：

```py
def join(self):
    threading.Thread.join(self)
    return self.return_value
```

现在我们已经设置好了我们的线程类。让我们将这个类的一个实例作为`NewThread`对象启动，通过调用`start`方法来生成新的线程，然后通过调用`join`来阻塞执行并从主机线程获取输出：

```py
NewThread = PointlessExampleThread()
NewThread.start()
thread_output = NewThread.join()
print 'The thread completed and returned this value: %s' % thread_output
```

现在让我们运行这个：

![](assets/9dc8f524-03ac-4f2c-a21c-8736c1feb1cf.png)

现在，我们可以在主机上的多个并发线程之间扩展这个想法，通过多个上下文和线程来启动并发的CUDA操作。现在我们来看最后一个例子。让我们重新使用本章开头的无意义的乘法/除法内核，并在我们生成的每个线程中启动它。

首先，让我们看一下导入部分。由于我们正在创建显式上下文，请记住删除`pycuda.autoinit`，并在最后添加一个`threading`导入：

```py
import pycuda
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time
import threading 
```

我们将使用与之前相同的数组大小，但这次线程的数量和数组的数量将直接对应。通常情况下，我们不希望在主机上生成超过20个左右的线程，所以我们只会使用`10`个数组。因此，现在考虑以下代码：

```py
num_arrays = 10
array_len = 1024**2
```

现在，我们将把旧的内核存储为一个字符串对象；由于这只能在一个上下文中编译，我们将不得不在每个线程中单独编译这个内核：

```py
kernel_code = """ 
__global__ void mult_ker(float * array, int array_len)
{
     int thd = blockIdx.x*blockDim.x + threadIdx.x;
     int num_iters = array_len / blockDim.x;
    for(int j=0; j < num_iters; j++)
     {
     int i = j * blockDim.x + thd;
     for(int k = 0; k < 50; k++)
     {
         array[i] *= 2.0;
         array[i] /= 2.0;
     }
 }
}
"""
```

现在我们可以开始设置我们的类。我们将像以前一样创建`threading.Thread`的另一个子类，并设置构造函数以接受一个输入数组作为参数。我们将用`None`初始化一个输出变量，就像以前一样：

```py
class KernelLauncherThread(threading.Thread):
    def __init__(self, input_array):
        threading.Thread.__init__(self)
        self.input_array = input_array
        self.output_array = None
```

现在我们可以编写`run`函数。我们选择我们的设备，在该设备上创建一个上下文，编译我们的内核，并提取内核函数引用。注意`self`对象的使用：

```py
def run(self):
    self.dev = drv.Device(0)
    self.context = self.dev.make_context()
    self.ker = SourceModule(kernel_code)
    self.mult_ker = self.ker.get_function('mult_ker')
```

现在我们将数组复制到GPU，启动内核，然后将输出复制回主机。然后我们销毁上下文：

```py
self.array_gpu = gpuarray.to_gpu(self.input_array)
self.mult_ker(self.array_gpu, np.int32(array_len), block=(64,1,1), grid=(1,1,1))
self.output_array = self.array_gpu.get()
self.context.pop()
```

最后，我们设置了join函数。这将把`output_array`返回到主机：

```py
 def join(self):
     threading.Thread.join(self)
     return self.output_array
```

我们现在已经完成了我们的子类。我们将设置一些空列表来保存我们的随机测试数据、线程对象和线程输出值，与之前类似。然后我们将生成一些随机数组进行处理，并设置一个核发射器线程列表，这些线程将分别操作每个数组：

```py
data = []
gpu_out = []
threads = []
for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))
for k in range(num_arrays):
 threads.append(KernelLauncherThread(data[k]))
```

现在我们将启动每个线程对象，并通过使用`join`将其输出提取到`gpu_out`列表中：

```py
for k in range(num_arrays):
    threads[k].start()

for k in range(num_arrays):
    gpu_out.append(threads[k].join())
```

最后，我们只需对输出数组进行简单的断言，以确保它们与输入相同：

```py
for k in range(num_arrays):
    assert (np.allclose(gpu_out[k], data[k]))

```

这个示例可以在存储库中的`multi-kernel_multi-thread.py`文件中看到。

# 总结

我们通过学习设备同步和从主机上同步GPU操作的重要性开始了本章；这允许依赖操作在进行之前完成。到目前为止，这个概念一直被PyCUDA自动处理，然后我们学习了CUDA流，它允许在GPU上同时执行独立的操作序列而无需在整个GPU上进行同步，这可以大大提高性能；然后我们学习了CUDA事件，它允许我们在给定流中计时单个CUDA内核，并确定流中的特定操作是否已发生。接下来，我们学习了上下文，它类似于主机操作系统中的进程。我们学习了如何在整个CUDA上下文中显式同步，然后看到了如何创建和销毁上下文。最后，我们看到了如何在GPU上生成多个上下文，以允许主机上的多个线程或进程使用GPU。

# 问题

1.  在第一个示例中内核的启动参数中，我们的内核每个都是在64个线程上启动的。如果我们将线程数增加到并超过GPU中的核心数，这会如何影响原始版本和流版本的性能？

1.  考虑本章开头给出的CUDA C示例，它演示了`cudaDeviceSynchronize`的使用。您认为在不使用流，只使用`cudaDeviceSynchronize`的情况下，是否可能在多个内核之间获得一定程度的并发？

1.  如果您是Linux用户，请修改上一个示例，使其在进程上运行而不是线程。

1.  考虑`multi-kernel_events.py`程序；我们说内核执行持续时间的标准偏差很低是件好事。如果标准偏差很高会有什么坏处？

1.  在上一个示例中，我们只使用了10个主机端线程。列出两个原因，说明为什么我们必须使用相对较少数量的线程或进程来在主机上启动并发的GPU操作。
