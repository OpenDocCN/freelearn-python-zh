# 实现、应用和工具

学习算法而没有任何现实生活的应用仍然是一种纯粹的学术追求。在本章中，我们将探讨正在塑造我们世界的数据结构和算法。

这个时代的一个黄金机会是数据的丰富。电子邮件、电话号码、文本文档和图像包含大量的数据。在这些数据中，有着使数据更加重要的有价值信息。但是要从原始数据中提取这些信息，我们必须使用专门从事这项任务的数据结构、过程和算法。

机器学习使用大量算法来分析和预测某些变量的发生。仅基于纯数字的数据分析仍然使得许多潜在信息埋藏在原始数据中。因此，通过可视化呈现数据，使人们能够理解并获得有价值的见解。

在本章结束时，您应该能够做到以下几点：

+   精确修剪和呈现数据

+   为了预测，需要同时使用监督学习和无监督学习算法。

+   通过可视化呈现数据以获得更多见解

# 技术要求

为了继续本章，您需要安装以下包。这些包将用于对正在处理的数据进行预处理和可视化呈现。其中一些包还包含对我们的数据进行操作的算法的良好实现。

最好使用`pip`安装这些模块。因此，首先，我们需要使用以下命令为 Python 3 安装 pip：

+   `sudo apt-get update`

+   `sudo apt-get install python3-pip`

此外，需要运行以下命令来安装`numpy`、`scikit-learn`、`matplotlib`、`pandas`和`textblob`包：

```py
# pip3 install numpy
# pip3 install scikit-learn
# pip3 install matplotlib
# pip3 install pandas
# pip3 install textblob  
```

如果您使用的是旧版本的 Python（即 Python 2），则可以使用相同的命令来安装这些包，只需将`pip3`替换为`pip`。

您还需要安装`nltk`和`punkt`包，这些包提供了内置的文本处理功能。要安装它们，请打开 Python 终端并运行以下命令：

```py
>>import nltk
>>nltk.download('punkt')
```

这些包可能需要先安装其他特定于平台的模块。请注意并安装所有依赖项：

+   **NumPy**：一个具有操作 n 维数组和矩阵功能的库。

+   **Scikit-learn**：用于机器学习的高级模块。它包含许多用于分类、回归和聚类等算法的实现。

+   **Matplotlib**：这是一个绘图库，利用 NumPy 绘制各种图表，包括折线图、直方图、散点图，甚至 3D 图表。

+   **Pandas**：这个库处理数据操作和分析。

GitHub 链接如下：[`github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-3.x-Second-Edition/tree/master/Chapter14`](https://github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-3.x-Second-Edition/tree/master/Chapter14)。

# 数据预处理

首先，要分析数据，我们必须对数据进行预处理，以去除噪音并将其转换为适当的格式，以便进一步分析。来自现实世界的数据集大多充满噪音，这使得直接应用任何算法变得困难。收集到的原始数据存在许多问题，因此我们需要采取方法来清理数据，使其适用于进一步的研究。

# 处理原始数据

收集到的数据可能与随时间收集的其他记录不一致。重复条目的存在和不完整的记录要求我们以这样的方式处理数据，以揭示隐藏的有用信息。

为了清理数据，我们完全丢弃了不相关和嘈杂的数据。缺失部分或属性的数据可以用合理的估计值替换。此外，当原始数据存在不一致性时，检测和纠正就变得必要了。

让我们探讨如何使用`NumPy`和`pandas`进行数据预处理技术。

# 缺失数据

如果数据存在缺失值，机器学习算法的性能会下降。仅仅因为数据集存在缺失字段或属性并不意味着它没有用处。可以使用几种方法来填补缺失值。其中一些方法如下：

+   使用全局常数填补缺失值。

+   使用数据集中的均值或中位数值。

+   手动提供数据。

+   使用属性的均值或中位数来填补缺失值。选择基于数据将要使用的上下文和敏感性。

例如，以下数据：

```py
    import numpy as np 
    data = pandas.DataFrame([ 
        [4., 45., 984.], 
        [np.NAN, np.NAN, 5.], 
        [94., 23., 55.], 
    ]) 
```

可以看到，数据元素`data[1][0]`和`data[1][1]`的值为`np.NAN`，表示它们没有值。如果不希望在给定数据集中存在`np.NAN`值，可以将其设置为一个常数。

让我们将值为`np.NAN`的数据元素设置为`0.1`：

```py
    print(data.fillna(0.1)) 
```

数据的新状态如下：

```py
0     1      2
0   4.0  45.0  984.0
1   0.1   0.1    5.0
2  94.0  23.0   55.0
```

要应用均值，我们需要做如下操作：

```py
    print(data.fillna(data.mean()))
```

为每列计算均值，并将其插入到具有`np.NAN`值的数据区域中：

```py
0     1      2
0   4.0  45.0  984.0
1  49.0  34.0    5.0
2  94.0  23.0   55.0
```

对于第一列，列`0`，均值通过`(4 + 94)/2`得到。然后将结果`49.0`存储在`data[1][0]`中。对列`1`和`2`也进行类似的操作。

# 特征缩放

数据框中的列称为其特征。行称为记录或观察。如果一个属性的值比其他属性的值具有更高的范围，机器学习算法的性能会下降。因此，通常需要将属性值缩放或归一化到一个公共范围内。

考虑一个例子，以下数据矩阵。这些数据将在后续部分中被引用，请注意：

```py
data1= ([[  58.,    1.,   43.],
 [  10.,  200.,   65.],
 [  20.,   75.,    7.]]
```

特征一的数据为`58`、`10`和`20`，其值位于`10`和`58`之间。对于特征二，数据位于`1`和`200`之间。如果将这些数据提供给任何机器学习算法，将产生不一致的结果。理想情况下，我们需要将数据缩放到一定范围内以获得一致的结果。

再次仔细检查发现，每个特征（或列）的均值都在不同的范围内。因此，我们要做的是使特征围绕相似的均值对齐。

特征缩放的一个好处是它提升了机器学习的学习部分。`scikit`模块有大量的缩放算法，我们将应用到我们的数据中。

# 最小-最大标量形式的归一化

最小-最大标量形式的归一化使用均值和标准差将所有数据装箱到位于某些最小和最大值之间的范围内。通常，范围设置在`0`和`1`之间；尽管可以应用其他范围，但`0`到`1`范围仍然是默认值：

```py
from sklearn.preprocessing import MinMaxScaler

scaled_values = MinMaxScaler(feature_range=(0,1)) 
results = scaled_values.fit(data1).transform(data1) 
print(results) 
```

使用`MinMaxScaler`类的一个实例，范围为`(0,1)`，并传递给`scaled_values`变量。调用`fit`函数进行必要的计算，用于内部使用以改变数据集。`transform`函数对数据集进行实际操作，并将值返回给`results`：

```py
[[ 1\.          0\.          0.62068966]
 [ 0\.          1\.          1\.        ]
 [ 0.20833333  0.3718593   0\.        ]]
```

从前面的输出中可以看出，所有数据都经过了归一化，并位于`0`和`1`之间。这种输出现在可以提供给机器学习算法。

# 标准缩放

我们初始数据集或表中各特征的均值分别为 29.3、92 和 38。为了使所有数据具有相似的均值，即数据的均值为零，方差为单位，我们可以应用标准缩放算法，如下所示：

```py
    stand_scalar =  preprocessing.StandardScaler().fit(data) 
    results = stand_scalar.transform(data) 
    print(results)
```

`data`被传递给从实例化`StandardScaler`类返回的对象的`fit`方法。`transform`方法作用于数据元素，并将输出返回给结果：

```py
[[ 1.38637564 -1.10805456  0.19519899]
 [-0.93499753  1.31505377  1.11542277]
 [-0.45137812 -0.2069992  -1.31062176]]
```

检查结果，我们观察到所有特征现在都是均匀分布的。

# 二值化数据

要对给定的特征集进行二值化，我们可以使用一个阈值。如果给定数据集中的任何值大于阈值，则该值将被替换为`1`，如果该值小于阈值，则替换为`0`。考虑以下代码片段，我们以 50 作为阈值来对原始数据进行二值化：

```py
 results = preprocessing.Binarizer(50.0).fit(data).transform(data) 
 print(results) 
```

创建一个`Binarizer`的实例，并使用参数`50.0`。`50.0`是将在二值化算法中使用的阈值：

```py
[[ 1\. 0\. 0.]
 [ 0\. 1\. 1.]
 [ 0\. 1\. 0.]] 
```

数据中所有小于 50 的值将为`0`，否则为`1`。

# 学习机器学习

机器学习是人工智能的一个子领域。机器学习基本上是一个可以从示例数据中学习并可以基于此提供预测的算法。机器学习模型从数据示例中学习模式，并使用这些学习的模式来预测未见数据。例如，我们将许多垃圾邮件和正常邮件的示例输入来开发一个机器学习模型，该模型可以学习邮件中的模式，并可以将新邮件分类为垃圾邮件或正常邮件。

# 机器学习类型

机器学习有三个广泛的类别，如下：

+   **监督学习**：在这里，算法会接收一组输入和相应的输出。然后算法必须找出对于未见过的输入，输出将会是什么。监督学习算法试图学习输入特征和目标输出中的模式，以便学习的模型可以预测新的未见数据的输出。分类和回归是使用监督学习方法解决的两种问题，其中机器学习算法从给定的数据和标签中学习。分类是一个将给定的未见数据分类到预定义类别集合中的过程，给定一组输入特征和与其相关的标签。回归与分类非常相似，唯一的区别在于，在回归中，我们有连续的目标值，而不是固定的预定义类别集合（名义或分类属性），我们预测连续响应中的值。这样的算法包括朴素贝叶斯、支持向量机、k-最近邻、线性回归、神经网络和决策树算法。

+   **无监督学习**：无监督学习算法仅使用输入来学习数据中的模式和聚类，而不使用存在于一组输入和输出变量之间的关系。无监督算法用于学习给定输入数据中的模式，而不带有与其相关的标签。聚类问题是使用无监督学习方法解决的最流行的问题之一。在这种情况下，数据点根据特征之间的相似性被分组成组或簇。这样的算法包括 k 均值聚类、凝聚聚类和层次聚类。

+   **强化学习**：在这种学习方法中，计算机动态地与环境交互，以改善其性能。

# 你好分类器

让我们举一个简单的例子来理解机器学习的工作原理；我们从一个文本分类器的`hello world`例子开始。这是对机器学习的一个温和的介绍。

这个例子将预测给定文本是否带有负面或正面的含义。在这之前，我们需要用一些数据来训练我们的算法（模型）。

朴素贝叶斯模型适用于文本分类目的。基于朴素贝叶斯模型的算法通常速度快，产生准确的结果。它基于特征相互独立的假设。要准确预测降雨的发生，需要考虑三个条件。这些条件是风速、温度和空气中的湿度量。实际上，这些因素确实会相互影响，以确定降雨的可能性。但朴素贝叶斯的抽象是假设这些特征在任何方面都是无关的，因此独立地影响降雨的可能性。朴素贝叶斯在预测未知数据集的类别时非常有用，我们很快就会看到。

现在，回到我们的 hello 分类器。在我们训练模型之后，它的预测将属于正类别或负类别之一：

```py
    from textblob.classifiers import NaiveBayesClassifier 
    train = [ 
        ('I love this sandwich.', 'pos'), 
        ('This is an amazing shop!', 'pos'), 
        ('We feel very good about these beers.', 'pos'), 
        ('That is my best sword.', 'pos'), 
        ('This is an awesome post', 'pos'), 
        ('I do not like this cafe', 'neg'), 
        ('I am tired of this bed.', 'neg'), 
        ("I can't deal with this", 'neg'), 
        ('She is my sworn enemy!', 'neg'), 
        ('I never had a caring mom.', 'neg') 
    ] 
```

首先，我们将从`textblob`包中导入`NaiveBayesClassifier`类。这个分类器非常容易使用，基于贝叶斯定理。

`train`变量由每个包含实际训练数据的元组组成。每个元组包含句子和它所关联的组。

现在，为了训练我们的模型，我们将通过传递`train`来实例化一个`NaiveBayesClassifier`对象：

```py
    cl = NaiveBayesClassifier(train) 
```

更新后的朴素贝叶斯模型`cl`将预测未知句子所属的类别。到目前为止，我们的模型只知道短语可以属于`neg`和`pos`两个类别中的一个。

以下代码使用我们的模型运行测试：

```py
    print(cl.classify("I just love breakfast")) 
    print(cl.classify("Yesterday was Sunday")) 
    print(cl.classify("Why can't he pay my bills")) 
    print(cl.classify("They want to kill the president of Bantu")) 
```

我们测试的输出如下：

```py
pos 
pos 
neg 
neg 
```

我们可以看到算法在正确将输入短语分类到它们的类别方面取得了一定程度的成功。

这个刻意构造的例子过于简单，但它确实显示了如果提供了正确数量的数据和合适的算法或模型，机器是可以在没有任何人类帮助的情况下执行任务的。

在我们的下一个例子中，我们将使用`scikit`模块来预测一个短语可能属于的类别。

# 一个监督学习的例子

让我们考虑一个文本分类问题的例子，可以使用监督学习方法来解决。文本分类问题是在我们有一组与固定数量的类别相关的文档时，将新文档分类到预定义的文档类别集合之一。与监督学习一样，我们需要首先训练模型，以便准确预测未知文档的类别。

# 收集数据

`scikit`模块带有我们可以用于训练机器学习模型的示例数据。在这个例子中，我们将使用包含 20 个文档类别的新闻组文档。为了加载这些文档，我们将使用以下代码行：

```py
 from sklearn.datasets import fetch_20newsgroups 
 training_data = fetch_20newsgroups(subset='train', categories=categories,   
                                           shuffle=True, random_state=42)
```

让我们只取四个文档类别来训练模型。在我们训练模型之后，预测的结果将属于以下类别之一：

```py
    categories = ['alt.atheism', 
                  'soc.religion.christian','comp.graphics', 'sci.med'] 
```

我们将用作训练数据的记录总数是通过以下方式获得的：

```py
 print(len(training_data)) 
```

机器学习算法不能直接处理文本属性，因此每个文档所属类别的名称被表示为数字（例如，`alt.atheism`表示为`0`），使用以下代码行：

```py
    print(set(training_data.target)) 
```

类别具有整数值，我们可以使用`print(training_data.target_names[0])`将其映射回类别本身。

在这里，`0`是从`set(training_data.target)`中随机选择的数字索引。

现在训练数据已经获得，我们必须将数据提供给机器学习算法。词袋模型是一种将文本文档转换为特征向量的方法，以便将文本转换为学习算法或模型可以应用的形式。此外，这些特征向量将用于训练机器学习模型。

# 词袋模型

词袋是一种模型，用于表示文本数据，它不考虑单词的顺序，而是使用单词计数。让我们看一个例子来理解词袋方法如何用于表示文本。看看以下两个句子：

```py
    sentence_1 = "as fit as a fiddle"
    sentence_2 = "as you like it"
```

词袋使我们能够将文本拆分为由矩阵表示的数值特征向量。

为了使用词袋模型减少我们的两个句子，我们需要获得所有单词的唯一列表：

```py
    set((sentence_1 + sentence_2).split(" "))
```

这个集合将成为我们矩阵中的列，被称为机器学习术语中的特征。矩阵中的行将代表用于训练的文档。行和列的交集将存储单词在文档中出现的次数。使用我们的两个句子作为例子，我们得到以下矩阵：

|  | **as** | **fit** | **a** | **fiddle** | **you** | **like** | **it** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **句子 1** | 2 | 1 | 1 | 1 | 0 | 0 | 0 |
| **句子 2** | 1 | 0 | 0 | 0 | 1 | 1 | 1 |

前面的数据有很多特征，通常对文本分类不重要。停用词可以被移除，以确保只分析相关的数据。停用词包括 is，am，are，was 等等。由于词袋模型在分析中不包括语法，停用词可以安全地被删除。

为了生成进入矩阵列的值，我们必须对我们的训练数据进行标记化：

```py
    from sklearn.feature_extraction.text import CountVectorizer 
    from sklearn.feature_extraction.text import TfidfTransformer 
    from sklearn.naive_bayes import MultinomialNB 
    count_vect = CountVectorizer() 
    training_matrix = count_vect.fit_transform(training_data.data) 
```

在这个例子中，`training_matrix`的维度为（2,257 x 35,788），对应于我们在这个例子中使用的四个数据类别。这意味着 2,257 对应于文档的总数，而 35,788 对应于列的数量，即构成所有文档中唯一单词集的特征的总数。

我们实例化`CountVectorizer`类，并将`training_data.data`传递给`count_vect`对象的`fit_transform`方法。结果存储在`training_matrix`中。`training_matrix`包含所有唯一的单词及其相应的频率。

有时，频率计数对于文本分类问题表现不佳；我们可以使用**词项频率-逆文档频率**（**TF-IDF**）加权方法来表示特征，而不是使用频率计数。

在这里，我们将导入`TfidfTransformer`，它有助于为我们的数据中的每个特征分配权重：

```py
    matrix_transformer = TfidfTransformer() 
    tfidf_data = matrix_transformer.fit_transform(training_matrix) 

    print(tfidf_data[1:4].todense()) 
```

`tfidf_data[1:4].todense()`只显示了一个三行 35,788 列矩阵的截断列表。所见的值是 TF-IDF；与使用频率计数相比，它是一种更好的表示方法。

一旦我们提取了特征并以表格格式表示它们，我们就可以应用机器学习算法进行训练。有许多监督学习算法；让我们看一个朴素贝叶斯算法的例子来训练文本分类器模型。

朴素贝叶斯算法是一种简单的分类算法，它基于贝叶斯定理。它是一种基于概率的学习算法，通过使用特征/单词/术语的词频来计算属于的概率来构建模型。朴素贝叶斯算法将给定的文档分类为预定义类别中的一个，其中新文档中观察到的单词的最大概率所在的类别。朴素贝叶斯算法的工作方式如下——首先，处理所有训练文档以提取出现在文本中的所有单词的词汇，然后计算它们在不同目标类别中的频率以获得它们的概率。接下来，将新文档分类到具有属于特定类别的最大概率的类别中。朴素贝叶斯分类器基于这样的假设，即单词出现的概率与文本中的位置无关。多项式朴素贝叶斯可以使用`scikit`库的`MultinomialNB`函数来实现，如下所示：

```py
 model = MultinomialNB().fit(tfidf_data, training_data.target) 
```

`MultinomialNB`是朴素贝叶斯模型的一个变体。我们将经过合理化的数据矩阵`tfidf_data`和类别`training_data.target`传递给其`fit`方法。

# 预测

为了测试训练模型如何预测未知文档的类别，让我们考虑一些示例测试数据来评估模型：

```py
    test_data = ["My God is good", "Arm chip set will rival intel"] 
    test_counts = count_vect.transform(test_data) 
    new_tfidf = matrix_transformer.transform(test_counts)
```

将`test_data`列表传递给`count_vect.transform`函数，以获得测试数据的向量化形式。为了获得测试数据集的 TF-IDF 表示，我们调用`matrix_transformer`对象的`transform`方法。当我们将新的测试数据传递给机器学习模型时，我们必须以与准备训练数据相同的方式处理数据。

为了预测文档可能属于哪个类别，我们使用`predict`函数如下：

```py
    prediction = model.predict(new_tfidf)  
```

循环可以用于迭代预测，显示它们被预测属于的类别：

```py
    for doc, category in zip(test_data, prediction): 
        print('%r => %s' % (doc, training_data.target_names[category])) 
```

当循环运行完成时，将显示短语及其可能属于的类别。示例输出如下：

```py
'My God is good' => soc.religion.christian
'Arm chip set will rival intel' => comp.graphics
```

到目前为止，我们所看到的都是监督学习的一个典型例子。我们首先加载已知类别的文档。然后将这些文档输入到最适合文本处理的机器学习算法中，基于朴素贝叶斯定理。一组测试文档被提供给模型，并预测类别。

探索一个无监督学习算法的例子，我们将讨论 k 均值算法对一些数据进行聚类。

# 无监督学习示例

无监督学习算法能够发现数据中可能存在的固有模式，并以这样的方式将它们聚类成组，使得一个组中的数据点非常相似，而来自两个不同组的数据点在性质上非常不相似。这些算法的一个例子就是 k 均值算法。

# k 均值算法

k 均值算法使用给定数据集中的均值点来对数据进行聚类并发现数据集中的组。`K`是我们希望发现的聚类的数量。k 均值算法生成了分组/聚类之后，我们可以将未知数据传递给该模型，以预测新数据应该属于哪个聚类。

请注意，在这种算法中，只有原始的未分类数据被输入到算法中，没有任何与数据相关联的标签。算法需要找出数据是否具有固有的组。

k 均值算法通过迭代地根据提供的特征之间的相似性将数据点分配到聚类中。k 均值聚类使用均值点将数据点分组成 k 个聚类/组。它的工作方式如下。首先，我们创建 k 个非空集合，并计算数据点与聚类中心之间的距离。接下来，我们将数据点分配给具有最小距离且最接近的聚类。然后，我们重新计算聚类点，并迭代地遵循相同的过程，直到所有数据都被聚类。

为了理解这个算法的工作原理，让我们检查包含 x 和 y 值的 100 个数据点（假设有两个属性）。我们将把这些值传递给学习算法，并期望算法将数据聚类成两组。我们将对这两组进行着色，以便看到聚类。

让我们创建一个包含 100 条*x*和*y*对的样本数据：

```py
    import numpy as np 
    import matplotlib.pyplot as plt 
    original_set = -2 * np.random.rand(100, 2) 
    second_set = 1 + 2 * np.random.rand(50, 2) 
    original_set[50: 100, :] = second_set 
```

首先，我们创建 100 条记录，其中包含`-2 * np.random.rand(100, 2)`。在每条记录中，我们将使用其中的数据来表示最终将绘制的*x*和*y*值。

`original_set`中的最后 50 个数字将被`1+2*np.random.rand(50, 2)`替换。实际上，我们已经创建了两个数据子集，其中一个集合中的数字为负数，而另一个集合中的数字为正数。现在算法的责任是适当地发现这些段。

我们实例化`KMeans`算法类，并传递`n_clusters=2`。这使得算法将其所有数据聚类成两组。在 k 均值算法中，簇的数量必须事先知道。使用`scikit`库实现 k 均值算法如下所示：

```py
    from sklearn.cluster import KMeans 
    kmean = KMeans(n_clusters=2) 

    kmean.fit(original_set) 

    print(kmean.cluster_centers_) 

    print(kmean.labels_) 
```

数据集被传递给`kmean`的`fit`函数，`kmean.fit(original_set)`。算法生成的聚类将围绕某个平均点旋转。定义这两个平均点的点是通过`kmean.cluster_centers_`获得的。

打印出的平均点如下所示：

```py
[[ 2.03838197 2.06567568]
 [-0.89358725 -0.84121101]]
```

`original_set`中的每个数据点在我们的 k 均值算法完成训练后将属于一个簇。k 均值算法将它发现的两个簇表示为 1 和 0。如果我们要求算法将数据分成四个簇，这些簇的内部表示将是 0、1、2 和 3。要打印出每个数据集所属的不同簇，我们执行以下操作：

```py
    print(kmean.labels_) 
```

这将产生以下输出：

```py
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
```

有`100`个 1 和 0。每个显示每个数据点所属的簇。通过使用`matplotlib.pyplot`，我们可以绘制每个组的点并适当着色以显示簇：

```py
    import matplotlib.pyplot as plt 
    for i in set(kmean.labels_): 
        index = kmean.labels_ == i 
        plt.plot(original_set[index,0], original_set[index,1], 'o')
```

`index = kmean.labels_ == i`是一种巧妙的方法，通过它我们选择与组`i`对应的所有点。当`i=0`时，所有属于零组的点都返回到变量`index`。对于`index =1, 2`，依此类推。

`plt.plot(original_set[index,0], original_set[index,1], 'o')`然后使用`o`作为绘制每个点的字符绘制这些数据点。

接下来，我们将绘制形成簇的质心或平均值：

```py
    plt.plot(kmean.cluster_centers_[0][0],kmean.cluster_centers_[0][1], 
             '*', c='r', ms=10) 
    plt.plot(kmean.cluster_centers_[1][0],kmean.cluster_centers_[1][1], 
             '*', c='r', ms=10) 
```

最后，我们使用代码片段`plt.show()`显示整个图形，其中两个平均值用红色星号表示，如下所示：

![](img/9130b340-90ee-4cee-b5ff-ca93c2b01e27.png)

该算法在我们的样本数据中发现了两个不同的簇。

# 预测

有了我们得到的两个簇，我们可以预测新一组数据可能属于哪个组。

让我们预测点`[[-1.4, -1.4]]`和`[[2.5, 2.5]]`将属于哪个组：

```py
    sample = np.array([[-1.4, -1.4]]) 
    print(kmean.predict(sample)) 

    another_sample = np.array([[2.5, 2.5]]) 
    print(kmean.predict(another_sample)) 
```

输出如下：

```py
[1]
[0] 
```

在这里，两个测试样本分配到了两个不同的簇。

# 数据可视化

数值分析有时不那么容易理解。在本节中，我们将向您展示一些可视化数据和结果的方法。图像是分析数据的一种快速方式。图像中大小和长度的差异是快速标记，可以得出结论。在本节中，我们将介绍表示数据的不同方法。除了这里列出的图表外，在处理数据时还可以实现更多。

# 条形图

要将值 25、5、150 和 100 绘制成条形图，我们将把这些值存储在一个数组中，并将其传递给`bar`函数。图中的条代表*y*轴上的大小：

```py
    import matplotlib.pyplot as plt 

    data = [25., 5., 150., 100.] 
    x_values = range(len(data)) 
    plt.bar(x_values, data) 

    plt.show()
```

`x_values`存储由`range(len(data))`生成的值数组。此外，`x_values`将确定在*x*轴上绘制条形的点。第一根条将在*x*轴上绘制，其中*x*为零。第二根带有数据 5 的条将在*x*轴上绘制，其中*x*为 1：

![](img/5c2618e0-70aa-41eb-aa32-ff94e8b3f8c6.png)

通过修改以下行可以改变每个条的宽度：

```py
    plt.bar(x_values, data, width=1.)  
```

这应该产生以下图形：

![](img/9cd8e72f-79e7-4c7e-9cfc-e80a375153f0.png)

然而，这样做并不直观，因为条之间不再有空间，这使得看起来很笨拙。每个条现在在*x*轴上占据一个单位。

# 多条形图

在尝试可视化数据时，堆叠多个条使人能够进一步了解一条数据或变量相对于另一条数据或变量的变化：

```py
    data = [ 
            [8., 57., 22., 10.], 
            [16., 7., 32., 40.],
           ] 

    import numpy as np 
    x_values = np.arange(4) 
    plt.bar(x_values + 0.00, data[0], color='r', width=0.30) 
    plt.bar(x_values + 0.30, data[1], color='y', width=0.30) 

    plt.show() 
```

第一批数据的`y`值为`[8., 57., 22., 10.]`。第二批数据为`[16., 7., 32., 40.]`。当条形图绘制时，8 和 16 将占据相同的`x`位置，侧边相邻。

`x_values = np.arange(4)`生成值为`[0, 1, 2, 3]`的数组。第一组条形图首先绘制在位置`x_values + 0.30`。因此，第一个 x 值将被绘制在`0.00, 1.00, 2.00 和 3.00`。

第二组`x_values`将被绘制在`0.30, 1.30, 2.30`和`3.30`：

![](img/d0624b77-b67d-470b-96fb-22a604eafad6.png)

# 箱线图

箱线图用于可视化分布的中位数值和低高范围。它也被称为箱线图。

让我们绘制一个简单的箱线图。

我们首先生成`50`个来自正态分布的数字。然后将它们传递给`plt.boxplot(data)`进行绘图：

```py
    import numpy as np 
    import matplotlib.pyplot as plt 

    data = np.random.randn(50) 

    plt.boxplot(data) 
    plt.show() 
```

以下图表是产生的：

![](img/05876ad6-cc46-4141-bfa8-bb3b119b6c0e.png)

对于前面的图表，一些注释——箱线图的特点包括跨越四分位距的箱子，用于测量离散度；数据的外围由连接到中心箱子的须表示；红线代表中位数。

箱线图可用于轻松识别数据集中的异常值，以及确定数据集可能偏向的方向。

# 饼图

饼图解释和直观地表示数据，就像适合放在圆圈里一样。个别数据点被表示为圆圈的扇形，总和为 360 度。这种图表适合显示分类数据和总结：

```py
    import matplotlib.pyplot as plt 
    data = [500, 200, 250] 

    labels = ["Agriculture", "Aide", "News"] 

    plt.pie(data, labels=labels,autopct='%1.1f%%') 
    plt.show() 
```

图表中的扇形用标签数组中的字符串标记：

![](img/17ea1d44-5589-427c-8c5c-0d37c801d374.png)

# 气泡图

散点图的另一种变体是气泡图。在散点图中，我们只绘制数据的`x`和`y`点。气泡图通过展示点的大小添加了另一个维度。这第三个维度可以表示市场的规模甚至利润：

```py
    import numpy as np 
    import matplotlib.pyplot as plt 

    n = 10 
    x = np.random.rand(n) 
    y = np.random.rand(n) 
    colors = np.random.rand(n) 
    area = np.pi * (60 * np.random.rand(n))**2 

    plt.scatter(x, y, s=area, c=colors, alpha=0.5) 
    plt.show() 
```

通过`n`变量，我们指定了随机生成的`x`和`y`值的数量。这个数字也用于确定我们的`x`和`y`坐标的随机颜色。随机气泡大小由`area = np.pi * (60 * np.random.rand(n))**2`确定。

以下图表显示了这个气泡图：

![](img/9b77b68c-6fed-44e9-93f5-c9f23357954a.png)

# 总结

在本章中，我们探讨了数据和算法如何结合起来帮助机器学习。通过数据清洗技术和缩放和归一化过程，我们首先对大量数据进行了整理。将这些数据输入到专门的学习算法中，我们能够根据算法从数据中学到的模式来预测未知数据的类别。我们还讨论了机器学习算法的基础知识。

我们详细解释了监督和无监督的机器学习算法，使用朴素贝叶斯和 k 均值聚类算法。我们还使用基于 Python 的`scikit-learn`机器学习库提供了这些算法的实现。最后，我们讨论了一些重要的可视化技术，因为对压缩数据进行图表化和绘图有助于更好地理解和做出有见地的发现。

希望您在阅读本书时有一个愉快的体验，并且它能够帮助您在未来的数据结构和 Python 3.7 的学习中取得成功！
