# 第四章：预测词语中的情感

本章介绍以下主题：

+   构建朴素贝叶斯分类器

+   逻辑回归分类器

+   将数据集分割为训练集和测试集

+   使用交叉验证评估准确性

+   分析一个句子的情感

+   使用主题建模识别文本中的模式

+   情感分析的应用

# 构建朴素贝叶斯分类器

朴素贝叶斯分类器使用贝叶斯定理构建监督模型。

# 如何做...

1.  导入以下软件包：

```py
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
```

1.  使用以下包含逗号分隔的算术数据的数据文件：

```py
in_file = 'data_multivar.txt'
a = []
b = []
with open(in_file, 'r') as f:
  for line in f.readlines():
    data = [float(x) for x in line.split(',')]
    a.append(data[:-1])
    b.append(data[-1])
a = np.array(a)
b = np.array(b)
```

1.  构建朴素贝叶斯分类器：

```py
classification_gaussiannb = GaussianNB()
classification_gaussiannb.fit(a, b)
b_pred = classification_gaussiannb.predict(a)
```

1.  计算朴素贝叶斯的准确性：

```py
correctness = 100.0 * (b == b_pred).sum() / a.shape[0]
print "correctness of the classification =", round(correctness, 2), "%"
```

1.  绘制分类器结果：

```py
def plot_classification(classification_gaussiannb, a , b):
  a_min, a_max = min(a[:, 0]) - 1.0, max(a[:, 0]) + 1.0
  b_min, b_max = min(a[:, 1]) - 1.0, max(a[:, 1]) + 1.0
  step_size = 0.01
  a_values, b_values = np.meshgrid(np.arange(a_min, a_max,   step_size), np.arange(b_min, b_max, step_size))
  mesh_output1 = classification_gaussiannb.predict(np.c_[a_values.ravel(), b_values.ravel()])
  mesh_output2 = mesh_output1.reshape(a_values.shape)
  plt.figure()
  plt.pcolormesh(a_values, b_values, mesh_output2, cmap=plt.cm.gray)
  plt.scatter(a[:, 0], a[:, 1], c=b , s=80, edgecolors='black', linewidth=1,cmap=plt.cm.Paired)
```

1.  指定图的边界：

```py
plt.xlim(a_values.min(), a_values.max())
plt.ylim(b_values.min(), b_values.max())
*# specify the ticks on the X and Y axes* plt.xticks((np.arange(int(min(a[:, 0])-1), int(max(a[:, 0])+1), 1.0)))
plt.yticks((np.arange(int(min(a[:, 1])-1), int(max(a[:, 1])+1), 1.0)))
plt.show()
plot_classification(classification_gaussiannb, a, b)
```

执行朴素贝叶斯分类器后获得的准确性如下截图所示：

![](img/e4b9d171-b071-433f-90a1-e585f8ffe86a.png)

# 另请参阅

请参考以下文章：

+   要了解分类器如何工作的示例，请参考以下链接：

[`en.wikipedia.org/wiki/Naive_Bayes_classifier`](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

+   要了解更多关于使用提议的分类器进行文本分类的信息，请参考以下链接：

[`sebastianraschka.com/Articles/2014_naive_bayes_1.html`](http://sebastianraschka.com/Articles/2014_naive_bayes_1.html)

+   要了解更多关于朴素贝叶斯分类算法的信息，请参考以下链接：

[`software.ucv.ro/~cmihaescu/ro/teaching/AIR/docs/Lab4-NaiveBayes.pdf`](http://software.ucv.ro/~cmihaescu/ro/teaching/AIR/docs/Lab4-NaiveBayes.pdf)

# 逻辑回归分类器

可以选择这种方法，其中输出只能取两个值，0 或 1，通过/失败，赢/输，活着/死亡，健康/生病等。在因变量有两个以上的结果类别的情况下，可以使用多项逻辑回归进行分析。

# 如何做...

1.  安装必要的软件包后，让我们构建一些训练标签：

```py
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
a = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
b = np.array([1, 1, 1, 2, 2, 2])
```

1.  初始化分类器：

```py
classification = linear_model.LogisticRegression(solver='liblinear', C=100)
classification.fit(a, b)
```

1.  绘制数据点和边界：

```py
def plot_classification(classification, a , b):
  a_min, a_max = min(a[:, 0]) - 1.0, max(a[:, 0]) + 1.0
  b_min, b_max = min(a[:, 1]) - 1.0, max(a[:, 1]) + 1.0 step_size = 0.01
  a_values, b_values = np.meshgrid(np.arange(a_min, a_max, step_size), np.arange(b_min, b_max, step_size))
  mesh_output1 = classification.predict(np.c_[a_values.ravel(), b_values.ravel()])
  mesh_output2 = mesh_output1.reshape(a_values.shape)
  plt.figure()
  plt.pcolormesh(a_values, b_values, mesh_output2, cmap=plt.cm.gray)
  plt.scatter(a[:, 0], a[:, 1], c=b , s=80, edgecolors='black',linewidth=1,cmap=plt.cm.Paired)
 # specify the boundaries of the figure  plt.xlim(a_values.min(), a_values.max())
  plt.ylim(b_values.min(), b_values.max())
 # specify the ticks on the X and Y axes  plt.xticks((np.arange(int(min(a[:, 0])-1), int(max(a[:, 0])+1), 1.0)))
  plt.yticks((np.arange(int(min(a[:, 1])-1), int(max(a[:, 1])+1), 1.0)))
  plt.show()
  plot_classification(classification, a, b)
```

执行逻辑回归的命令如下截图所示：

![](img/ac242a00-2916-40dc-9710-78d6b2719307.png)

# 将数据集分割为训练集和测试集

分割有助于将数据集分割为训练和测试序列。

# 如何做...

1.  将以下代码片段添加到同一个 Python 文件中：

```py
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
in_file = 'data_multivar.txt'
a = []
b = []
with open(in_file, 'r') as f:
  for line in f.readlines():
    data = [float(x) for x in line.split(',')]
    a.append(data[:-1])
    b.append(data[-1])
a = np.array(a)
b = np.array(b)
```

1.  将 75%的数据用于训练，25%的数据用于测试：

```py
a_training, a_testing, b_training, b_testing = cross_validation.train_test_split(a, b, test_size=0.25, random_state=5)
classification_gaussiannb_new = GaussianNB()
classification_gaussiannb_new.fit(a_training, b_training)
```

1.  在测试数据上评估分类器的性能：

```py
b_test_pred = classification_gaussiannb_new.predict(a_testing)
```

1.  计算分类器系统的准确性：

```py
correctness = 100.0 * (b_testing == b_test_pred).sum() / a_testing.shape[0]
print "correctness of the classification =", round(correctness, 2), "%"
```

1.  绘制测试数据的数据点和边界：

```py
def plot_classification(classification_gaussiannb_new, a_testing , b_testing):
  a_min, a_max = min(a_testing[:, 0]) - 1.0, max(a_testing[:, 0]) + 1.0
  b_min, b_max = min(a_testing[:, 1]) - 1.0, max(a_testing[:, 1]) + 1.0
  step_size = 0.01
  a_values, b_values = np.meshgrid(np.arange(a_min, a_max, step_size), np.arange(b_min, b_max, step_size))
  mesh_output = classification_gaussiannb_new.predict(np.c_[a_values.ravel(), b_values.ravel()])
  mesh_output = mesh_output.reshape(a_values.shape)
  plt.figure()
  plt.pcolormesh(a_values, b_values, mesh_output, cmap=plt.cm.gray)
  plt.scatter(a_testing[:, 0], a_testing[:, 1], c=b_testing , s=80, edgecolors='black', linewidth=1,cmap=plt.cm.Paired)
 # specify the boundaries of the figure  plt.xlim(a_values.min(), a_values.max())
  plt.ylim(b_values.min(), b_values.max())
  # specify the ticks on the X and Y axes
  plt.xticks((np.arange(int(min(a_testing[:, 0])-1), int(max(a_testing[:, 0])+1), 1.0)))
  plt.yticks((np.arange(int(min(a_testing[:, 1])-1), int(max(a_testing[:, 1])+1), 1.0)))
  plt.show()
plot_classification(classification_gaussiannb_new, a_testing, b_testing)
```

在以下截图中显示了数据集分割时获得的准确性：

![](img/9e8bf137-ddd2-4813-bebc-d619b5943868.png)

# 使用交叉验证评估准确性

交叉验证在机器学习中是必不可少的。最初，我们将数据集分割为训练集和测试集。接下来，为了构建一个健壮的分类器，我们重复这个过程，但需要避免过度拟合模型。过度拟合表示我们对训练集获得了很好的预测结果，但对测试集获得了非常糟糕的结果。过度拟合导致模型的泛化能力差。

# 如何做...

1.  导入软件包：

```py
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
import numpy as np
in_file = 'cross_validation_multivar.txt'
a = []
b = []
with open(in_file, 'r') as f:
  for line in f.readlines():
    data = [float(x) for x in line.split(',')]
    a.append(data[:-1])
    b.append(data[-1])
a = np.array(a)
b = np.array(b)
classification_gaussiannb = GaussianNB()
```

1.  计算分类器的准确性：

```py
num_of_validations = 5
accuracy = cross_validation.cross_val_score(classification_gaussiannb, a, b, scoring='accuracy', cv=num_of_validations)
print "Accuracy: " + str(round(100* accuracy.mean(), 2)) + "%"
f1 = cross_validation.cross_val_score(classification_gaussiannb, a, b, scoring='f1_weighted', cv=num_of_validations)
print "f1: " + str(round(100*f1.mean(), 2)) + "%"
precision = cross_validation.cross_val_score(classification_gaussiannb,a, b, scoring='precision_weighted', cv=num_of_validations)
print "Precision: " + str(round(100*precision.mean(), 2)) + "%"
recall = cross_validation.cross_val_score(classification_gaussiannb, a, b, scoring='recall_weighted', cv=num_of_validations)
print "Recall: " + str(round(100*recall.mean(), 2)) + "%"
```

1.  执行交叉验证后获得的结果如下所示：

![](img/c897f11e-7072-4708-bfd2-1c07e9f7d59a.png)

为了了解它在给定的句子数据集上的工作情况，请参考以下链接：

+   逻辑回归简介：

[`machinelearningmastery.com/logistic-regression-for-machine-learning/`](https://machinelearningmastery.com/logistic-regression-for-machine-learning/)

# 分析一个句子的情感

情感分析是指找出特定文本部分是积极的、消极的还是中性的过程。这种技术经常被用来了解人们对特定情况的看法。它评估了消费者在不同形式中的情感，比如广告活动、社交媒体和电子商务客户。

# 如何做...

1.  创建一个新文件并导入所选的包：

```py
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
```

1.  描述一个提取特征的函数：

```py
def collect_features(word_list):
  word = []
  return dict ([(word, True) for word in word_list])
```

1.  采用 NLTK 中的电影评论作为训练数据：

```py
if __name__=='__main__':
  plus_filenum = movie_reviews.fileids('pos')
  minus_filenum = movie_reviews.fileids('neg')
```

1.  将数据分成积极和消极的评论：

```py
  feature_pluspts = [(collect_features(movie_reviews.words(fileids=[f])),
'Positive') for f in plus_filenum]
  feature_minuspts = [(collect_features(movie_reviews.words(fileids=[f])),
'Negative') for f in minus_filenum]
```

1.  将数据分成训练和测试数据集：

```py
  threshold_fact = 0.8
  threshold_pluspts = int(threshold_fact * len(feature_pluspts))
  threshold_minuspts = int(threshold_fact * len(feature_minuspts))
```

1.  提取特征：

```py
  feature_training = feature_pluspts[:threshold_pluspts] + feature_minuspts[:threshold_minuspts]
  feature_testing = feature_pluspts[threshold_pluspts:] + feature_minuspts[threshold_minuspts:]
  print "nNumber of training datapoints:", len(feature_training)
  print "Number of test datapoints:", len(feature_testing)
```

1.  考虑朴素贝叶斯分类器，并用指定的目标进行训练：

```py
  # Train a Naive Bayes classifiers
  classifiers = NaiveBayesClassifier.train(feature_training)
  print "nAccuracy of the classifiers:",nltk.classify.util.accuracy(classifiers,feature_testing)
  print "nTop 10 most informative words:"
  for item in classifiers.most_informative_features()[:10]:print item[0]
 # Sample input reviews  in_reviews = [
  "The Movie was amazing",
  "the movie was dull. I would never recommend it to anyone.",
  "The cinematography is pretty great in the movie",
  "The direction was horrible and the story was all over the place"
  ]
  print "nPredictions:"
  for review in in_reviews:
    print "nReview:", review
  probdist = classifiers.prob_classify(collect_features(review.split()))
  predict_sentiment = probdist.max()
  print "Predicted sentiment:", predict_sentiment
  print "Probability:", round(probdist.prob(predict_sentiment), 2)
```

1.  情感分析的结果如下所示：

![](img/74bc9668-f70b-4188-b868-a0a1cb5aa4a6.png)

# 使用主题建模在文本中识别模式

主题建模是指识别手稿信息中隐藏模式的过程。其目标是在一系列文件中揭示一些隐藏的主题结构。

# 如何做...

1.  导入以下包：

```py
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora
from nltk.corpus import stopwords
```

1.  加载输入数据：

```py
def load_words(in_file):
  element = []
  with open(in_file, 'r') as f:
    for line in f.readlines():
      element.append(line[:-1])
  return element
```

1.  预处理文本的类：

```py
classPreprocedure(object):
  def __init__(self):
 # Create a regular expression tokenizer    self.tokenizer = RegexpTokenizer(r'w+')
```

1.  获取停用词列表以终止程序执行：

```py
    self.english_stop_words= stopwords.words('english')
```

1.  创建一个 Snowball 词干提取器：

```py
    self.snowball_stemmer = SnowballStemmer('english')  
```

1.  定义一个执行标记化、停用词去除和词干处理的函数：

```py
  def procedure(self, in_data):
# Tokenize the string
    token = self.tokenizer.tokenize(in_data.lower())
```

1.  从文本中消除停用词：

```py
    tokenized_stopwords = [x for x in token if not x in self.english_stop_words]
```

1.  对标记进行词干处理：

```py
    token_stemming = [self.snowball_stemmer.stem(x) for x in tokenized_stopwords]
```

1.  返回处理过的标记：

```py
    return token_stemming
```

1.  从`main`函数加载输入数据：

```py
if __name__=='__main__':
 # File containing input data  in_file = 'data_topic_modeling.txt'
 # Load words  element = load_words(in_file)
```

1.  创建一个对象：

```py
  preprocedure = Preprocedure()
```

1.  处理文件并提取标记：

```py
  processed_tokens = [preprocedure.procedure(x) for x in element]
```

1.  根据标记化的文档创建一个字典：

```py
  dict_tokens = corpora.Dictionary(processed_tokens)
  corpus = [dict_tokens.doc2bow(text) for text in processed_tokens]
```

1.  开发一个 LDA 模型，定义所需的参数，并初始化 LDA 目标：

```py
  num_of_topics = 2
  num_of_words = 4
  ldamodel = models.ldamodel.LdaModel(corpus,num_topics=num_of_topics, id2word=dict_tokens, passes=25)
  print "Most contributing words to the topics:"
  for item in ldamodel.print_topics(num_topics=num_of_topics, num_words=num_of_words):
    print "nTopic", item[0], "==>", item[1]
```

1.  执行`topic_modelling.py`时获得的结果如下截图所示：

![](img/5c09b91d-5a91-4538-b54a-46ff45e443ee.png)

# 情感分析的应用

情感分析在社交媒体如 Facebook 和 Twitter 中使用，以找出公众对某个问题的情感（积极/消极）。它们还用于确定人们对广告的情感以及人们对您的产品、品牌或服务的感受。
