# 构建光学字符识别的神经网络模块

本章介绍以下主题：

+   使用**光学字符识别**（**OCR**）系统

+   使用软件可视化光学字符

+   使用神经网络构建光学字符识别器

+   应用 OCR 系统

# 介绍

OCR 系统用于将文本图像转换为字母、单词和句子。它被广泛应用于各个领域，用于从图像中提取信息。它还用于签名识别、自动数据评估和安全系统。它在商业上用于验证数据记录、护照文件、发票、银行对账单、电脑收据、名片、静态数据的打印输出等。OCR 是模式识别、人工智能和计算机视觉的研究领域。

# 可视化光学字符

光学字符可视化是一种常见的数字化印刷文本的方法，使得这些文本可以进行电子编辑、搜索、紧凑存储和在线显示。目前，它们广泛应用于认知计算、机器翻译、文本转语音转换、文本挖掘等领域。

# 如何做…

1.  导入以下软件包：

```py
import os 
import sys 
import cv2 
import numpy as np 
```

1.  加载输入数据：

```py
in_file = 'words.data'  
```

1.  定义可视化参数：

```py
scale_factor = 10 
s_index = 6 
e_index = -1 
h, w = 16, 8 
```

1.  循环直到遇到*Esc*键：

```py
with open(in_file, 'r') as f: 
  for line in f.readlines(): 
    information = np.array([255*float(x) for x in line.split('t')[s_index:e_index]]) 
    image = np.reshape(information, (h,w)) 
    image_scaled = cv2.resize(image, None, fx=scale_factor, fy=scale_factor) 
    cv2.imshow('Image', image_scaled) 
    a = cv2.waitKey() 
    if a == 10: 
      break 
```

1.  键入`python visualize_character.py`来执行代码：

![](img/ca2ad22d-0452-4ddb-923a-04c264e6bf16.png)

1.  执行`visualize_character.py`时得到的结果如下：

![](img/55eb558d-22dc-4190-9199-38d47ff974c6.png)

# 使用神经网络构建光学字符识别器

本节描述基于神经网络的光学字符识别方案。

# 如何做…

1.  导入以下软件包：

```py
import numpy as np 
import neurolab as nl 
```

1.  读取输入文件：

```py
in_file = 'words.data'
```

1.  考虑 20 个数据点来构建基于神经网络的系统：

```py
# Number of datapoints to load from the input file 
num_of_datapoints = 20
```

1.  表示不同的字符：

```py
original_labels = 'omandig' 
# Number of distinct characters 
num_of_charect = len(original_labels) 
```

1.  使用 90%的数据来训练神经网络，剩下的 10%用于测试：

```py
train_param = int(0.9 * num_of_datapoints) 
test_param = num_of_datapoints - train_param 
```

1.  定义数据集提取参数：

```py
s_index = 6 
e_index = -1 
```

1.  构建数据集：

```py
information = [] 
labels = [] 
with open(in_file, 'r') as f: 
  for line in f.readlines(): 
    # Split the line tabwise 
    list_of_values = line.split('t') 
```

1.  实施错误检查以确认字符：

```py
    if list_of_values[1] not in original_labels: 
      continue 
```

1.  提取标签并将其附加到主列表：

```py
    label = np.zeros((num_of_charect , 1)) 
    label[original_labels.index(list_of_values[1])] = 1 
    labels.append(label)
```

1.  提取字符并将其添加到主列表：

```py
    extract_char = np.array([float(x) for x in     list_of_values[s_index:e_index]]) 
    information.append(extract_char)
```

1.  一旦加载所需数据集，退出循环：

```py
    if len(information) >= num_of_datapoints: 
      break 
```

1.  将信息和标签转换为 NumPy 数组：

```py
information = np.array(information) 
labels = np.array(labels).reshape(num_of_datapoints, num_of_charect) 
```

1.  提取维度的数量：

```py
num_dimension = len(information[0]) 
```

1.  创建和训练神经网络：

```py
neural_net = nl.net.newff([[0, 1] for _ in range(len(information[0]))], [128, 16, num_of_charect]) 
neural_net.trainf = nl.train.train_gd 
error = neural_net.train(information[:train_param,:], labels[:train_param,:], epochs=10000, show=100, goal=0.01) 
```

1.  预测测试输入的输出：

```py
p_output = neural_net.sim(information[train_param:, :]) 
print "nTesting on unknown data:" 
  for i in range(test_param): 
    print "nOriginal:", original_labels[np.argmax(labels[i])] 
    print "Predicted:", original_labels[np.argmax(p_output[i])]
```

1.  执行`optical_character_recognition.py`时得到的结果如下截图所示：

![](img/9c777c60-2961-4856-b0b3-6966565f610f.png)

# 工作原理…

构建了一个神经网络支持的光学字符识别系统，用于从图像中提取文本。该过程涉及训练神经网络系统，测试和验证使用字符数据集。

读者可以参考文章*基于神经网络的光学字符识别系统*，了解 OCR 背后的基本原理：[`ieeexplore.ieee.org/document/6419976/`](http://ieeexplore.ieee.org/document/6419976/)

# 另请参阅

请参考以下内容：

+   [`searchcontentmanagement.techtarget.com/definition/OCR-optical-character-recognition`](https://searchcontentmanagement.techtarget.com/definition/OCR-optical-character-recognition)

+   [`thecodpast.org/2015/09/top-5-ocr-apps/`](https://thecodpast.org/2015/09/top-5-ocr-apps/)

+   [`convertio.co/ocr/`](https://convertio.co/ocr/)

# OCR 系统的应用

OCR 系统广泛用于从图像中提取/转换文本（字母和数字）。OCR 系统被广泛用于验证商业文件、自动车牌识别以及从文件中提取关键字符。它还用于使打印文件的电子图像可搜索，并为盲人和视障用户构建辅助技术。
