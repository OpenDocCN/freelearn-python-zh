# 执行词形还原

如何做

+   一些过程，比如我们将使用的过程，需要额外下载它们用于执行各种分析的各种数据集。可以通过执行以下操作来下载它们：安装 NLTK

+   安装 NLTK

+   ![](img/75aae6dd-a071-42ea-a6be-8ccd5f961b95.png)NTLK GUI

+   执行词干提取

+   首先我们从 NLTK 导入句子分词器：

+   识别和删除短词

+   句子分割的第一个例子在`07/01_sentence_splitting1.py`文件中。这使用 NLTK 中内置的句子分割器，该分割器使用内部边界检测算法：

+   识别和删除罕见单词

+   然后使用`sent_tokenize`分割句子，并报告句子：

+   我们按照以下步骤进行：

+   介绍

+   您可以使用语言参数选择所需的语言。例如，以下内容将基于德语进行分割：

+   阅读和清理工作列表中的描述

# 挖掘数据通常是工作中最有趣的部分，文本是最常见的数据来源之一。我们将使用 NLTK 工具包介绍常见的自然语言处理概念和统计模型。我们不仅希望找到定量数据，比如我们已经抓取的数据中的数字，还希望能够分析文本信息的各种特征。这种文本信息的分析通常被归类为自然语言处理（NLP）的一部分。Python 有一个名为 NLTK 的库，提供了丰富的功能。我们将调查它的几种功能。

在 Mac 上，这实际上会弹出以下窗口：

# 如何做

NLTK 的核心可以使用 pip 安装：

# 从 StackOverflow 抓取工作列表

文本整理和分析

1.  选择安装所有并按下下载按钮。工具将开始下载许多数据集。这可能需要一段时间，所以喝杯咖啡或啤酒，然后不时地检查一下。完成后，您就可以继续进行下一个步骤了。

```py
pip install nltk
```

1.  执行句子分割

```py
import nltk nltk.download() showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
```

1.  删除标点符号

在本章中，我们将涵盖：

执行标记化

# 在这个配方中，我们学习安装 Python 的自然语言工具包 NTLK。

许多 NLP 过程需要将大量文本分割成句子。这可能看起来是一个简单的任务，但对于计算机来说可能会有问题。一个简单的句子分割器可以只查找句号（。），或者使用其他算法，比如预测分类器。我们将使用 NLTK 来检查两种句子分割的方法。

# 这将产生以下输出：

我们将使用存储在`07/sentence1.txt`文件中的句子。它包含以下内容，这些内容是从 StackOverflow 上的随机工作列表中提取的：

我们正在寻找具有以下经验的开发人员：ASP.NET，C＃，SQL Server 和 AngularJS。我们是一个快节奏，高度迭代的团队，必须随着我们的工厂的增长迅速适应。我们需要那些习惯于解决新问题，创新解决方案，并且每天与公司的各个方面进行互动的人。有创意，有动力，能够承担责任并支持您创建的应用程序。帮助我们更快地将火箭送出去！

识别和删除停用词

1.  从 StackOverflow 工作列表创建词云

```py
from nltk.tokenize import sent_tokenize
```

1.  然后加载文件：

```py
with open('sentence1.txt', 'r') as myfile:
  data=myfile.read().replace('\n', '')
```

1.  拼接 n-gram

```py
sentences = sent_tokenize(data)   for s in sentences:
  print(s)
```

如果您想创建自己的分词器并自己训练它，那么可以使用`PunktSentenceTokenizer`类。`sent_tokenize`实际上是这个类的派生类，默认情况下实现了英语的句子分割。但是您可以从 17 种不同的语言模型中选择：

```py
We are seeking developers with demonstrable experience in: ASP.NET, C#, SQL Server, and AngularJS.
We are a fast-paced, highly iterative team that has to adapt quickly as our factory grows.
We need people who are comfortable tackling new problems, innovating solutions, and interacting with every facet of the company on a daily basis.
Creative, motivated, able to take responsibility and support the applications you create.
Help us get rockets out the door faster!
```

1.  执行句子分割

```py
Michaels-iMac-2:~ michaelheydt$ ls ~/nltk_data/tokenizers/punkt PY3   finnish.pickle  portuguese.pickle README   french.pickle  slovene.pickle czech.pickle  german.pickle  spanish.pickle danish.pickle  greek.pickle  swedish.pickle dutch.pickle  italian.pickle  turkish.pickle english.pickle  norwegian.pickle estonian.pickle  polish.pickle
```

1.  计算单词的频率分布

```py
sentences = sent_tokenize(data, language="german") 
```

# 还有更多...

要了解更多关于这个算法的信息，可以阅读[`citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.85.5017&rep=rep1&type=pdf`](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.85.5017&rep=rep1&type=pdf)上提供的源论文。

# 执行标记化

标记化是将文本转换为标记的过程。这些标记可以是段落、句子和常见的单词，通常是基于单词级别的。NLTK 提供了许多标记器，将在本教程中进行演示。

# 如何做

这个示例的代码在`07/02_tokenize.py`文件中。它扩展了句子分割器，演示了五种不同的标记化技术。文件中的第一句将是唯一被标记化的句子，以便我们保持输出的数量在合理范围内：

1.  第一步是简单地使用内置的 Python 字符串`.split()`方法。结果如下：

```py
print(first_sentence.split())
['We', 'are', 'seeking', 'developers', 'with', 'demonstrable', 'experience', 'in:', 'ASP.NET,', 'C#,', 'SQL', 'Server,', 'and', 'AngularJS.'] 
```

句子是在空格边界上分割的。注意，诸如“:”和“,”之类的标点符号包括在生成的标记中。

1.  以下演示了如何使用 NLTK 中内置的标记器。首先，我们需要导入它们：

```py
from nltk.tokenize import word_tokenize, regexp_tokenize, wordpunct_tokenize, blankline_tokenize
```

以下演示了如何使用`word_tokenizer`：

```py
print(word_tokenize(first_sentence))
['We', 'are', 'seeking', 'developers', 'with', 'demonstrable', 'experience', 'in', ':', 'ASP.NET', ',', 'C', '#', ',', 'SQL', 'Server', ',', 'and', 'AngularJS', '.'] 
```

结果现在还将标点符号分割为它们自己的标记。

以下使用了正则表达式标记器，它允许您将任何正则表达式表达为标记器。它使用了一个`'\w+'`正则表达式，结果如下：

```py
print(regexp_tokenize(first_sentence, pattern='\w+')) ['We', 'are', 'seeking', 'developers', 'with', 'demonstrable', 'experience', 'in', 'ASP', 'NET', 'C', 'SQL', 'Server', 'and', 'AngularJS']
```

`wordpunct_tokenizer`的结果如下：

```py
print(wordpunct_tokenize(first_sentence))
['We', 'are', 'seeking', 'developers', 'with', 'demonstrable', 'experience', 'in', ':', 'ASP', '.', 'NET', ',', 'C', '#,', 'SQL', 'Server', ',', 'and', 'AngularJS', '.']
```

`blankline_tokenize`产生了以下结果：

```py
print(blankline_tokenize(first_sentence))
['We are seeking developers with demonstrable experience in: ASP.NET, C#, SQL Server, and AngularJS.']
```

可以看到，这并不是一个简单的问题。根据被标记化的文本类型的不同，你可能会得到完全不同的结果。

# 执行词干提取

词干提取是将标记减少到其*词干*的过程。从技术上讲，它是将屈折（有时是派生）的单词减少到它们的词干形式的过程-单词的基本根形式。例如，单词*fishing*、*fished*和*fisher*都来自根词*fish*。这有助于将被处理的单词集合减少到更容易处理的较小基本集合。

最常见的词干提取算法是由 Martin Porter 创建的，NLTK 提供了 PorterStemmer 中这个算法的实现。NLTK 还提供了 Snowball 词干提取器的实现，这也是由 Porter 创建的，旨在处理英语以外的其他语言。NLTK 还提供了一个名为 Lancaster 词干提取器的实现。Lancaster 词干提取器被认为是这三种中最激进的词干提取器。

# 如何做

NLTK 在其 PorterStemmer 类中提供了 Porter 词干提取算法的实现。可以通过以下代码轻松创建一个实例：

```py
>>> from nltk.stem import PorterStemmer
>>> pst = PorterStemmer() >>> pst.stem('fishing') 'fish'
```

`07/03_stemming.py`文件中的脚本将 Porter 和 Lancaster 词干提取器应用于我们输入文件的第一句。执行词干提取的主要部分是以下内容：

```py
pst = PorterStemmer() lst = LancasterStemmer() print("Stemming results:")   for token in regexp_tokenize(sentences[0], pattern='\w+'):
  print(token, pst.stem(token), lst.stem(token))
```

结果如下：

```py
Stemming results:
We We we
are are ar
seeking seek seek
developers develop develop
with with with
demonstrable demonstr demonst
experience experi expery
in in in
ASP asp asp
NET net net
C C c
SQL sql sql
Server server serv
and and and
AngularJS angularj angulars
```

从结果可以看出，Lancaster 词干提取器确实比 Porter 词干提取器更激进，因为后者将几个单词进一步缩短了。

# 执行词形还原

词形还原是一个更系统的过程，将单词转换为它们的基本形式。词干提取通常只是截断单词的末尾，而词形还原考虑了单词的形态分析，评估上下文和词性以确定屈折形式，并在不同规则之间做出决策以确定词根。

# 如何做

在 NTLK 中可以使用`WordNetLemmatizer`进行词形还原。这个类使用 WordNet 服务，一个在线语义数据库来做出决策。`07/04_lemmatization.py`文件中的代码扩展了之前的词干提取示例，还计算了每个单词的词形还原。重要的代码如下：

```py
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

pst = PorterStemmer() lst = LancasterStemmer() wnl = WordNetLemmatizer()   print("Stemming / lemmatization results") for token in regexp_tokenize(sentences[0], pattern='\w+'):
  print(token, pst.stem(token), lst.stem(token), wnl.lemmatize(token))
```

结果如下：

```py
Stemming / lemmatization results
We We we We
are are ar are
seeking seek seek seeking
developers develop develop developer
with with with with
demonstrable demonstr demonst demonstrable
experience experi expery experience
in in in in
ASP asp asp ASP
NET net net NET
C C c C
SQL sql sql SQL
Server server serv Server
and and and and
AngularJS angularj angulars AngularJS
```

使用词形还原过程的结果有一些差异。这表明，根据您的数据，其中一个可能比另一个更适合您的需求，因此如果需要，可以尝试所有这些方法。

# 确定和去除停用词

停用词是在自然语言处理情境中不提供太多上下文含义的常见词。这些词通常是语言中最常见的词。这些词在英语中至少包括冠词和代词，如*I*，*me*，*the*，*is*，*which*，*who*，*at*等。在处理文档中的含义时，通常可以通过在处理之前去除这些词来方便处理，因此许多工具都支持这种能力。NLTK 就是其中之一，并且支持大约 22 种语言的停用词去除。

# 如何做

按照以下步骤进行（代码在`07/06_freq_dist.py`中可用）：

1.  以下演示了使用 NLTK 去除停用词。首先，从导入停用词开始：

```py
>>> from nltk.corpus import stopwords
```

1.  然后选择所需语言的停用词。以下选择英语：

```py
>>> stoplist = stopwords.words('english')
```

1.  英语停用词列表有 153 个单词：

```py
>>> len(stoplist) 153
```

1.  这不是太多，我们可以在这里展示它们所有：

```py
>>> stoplist
 ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
```

1.  从单词列表中去除停用词可以通过简单的 Python 语句轻松完成。这在`07/05_stopwords.py`文件中有演示。脚本从所需的导入开始，并准备好我们要处理的句子：

```py
from nltk.tokenize import sent_tokenize
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords

with open('sentence1.txt', 'r') as myfile:
  data = myfile.read().replace('\n', '')   sentences = sent_tokenize(data) first_sentence = sentences[0]   print("Original sentence:") print(first_sentence)
```

1.  这产生了我们熟悉的以下输出：

```py
Original sentence:
We are seeking developers with demonstrable experience in: ASP.NET, C#, SQL Server, and AngularJS.
```

1.  然后我们对该句子进行标记化：

```py
tokenized = regexp_tokenize(first_sentence, '\w+') print("Tokenized:", tokenized)
```

1.  使用以下输出：

```py
Tokenized: ['We', 'are', 'seeking', 'developers', 'with', 'demonstrable', 'experience', 'in', 'ASP', 'NET', 'C', 'SQL', 'Server', 'and', 'AngularJS']
```

1.  然后我们可以使用以下语句去除停用词列表中的标记：

```py
stoplist = stopwords.words('english') cleaned = [word for word in tokenized if word not in stoplist] print("Cleaned:", cleaned)
```

使用以下输出：

```py
Cleaned: ['We', 'seeking', 'developers', 'demonstrable', 'experience', 'ASP', 'NET', 'C', 'SQL', 'Server', 'AngularJS']
```

# 还有更多...

去除停用词有其目的。这是有帮助的，正如我们将在后面的一篇文章中看到的，我们将在那里创建一个词云（停用词在词云中不提供太多信息），但也可能是有害的。许多其他基于句子结构推断含义的自然语言处理过程可能会因为去除停用词而受到严重阻碍。

# 计算单词的频率分布

频率分布计算不同数据值的出现次数。这些对我们很有价值，因为我们可以用它们来确定文档中最常见的单词或短语，从而推断出哪些具有更大或更小的价值。

可以使用几种不同的技术来计算频率分布。我们将使用内置在 NLTK 中的工具来进行检查。

# 如何做

NLTK 提供了一个类，`ntlk.probabilities.FreqDist`，可以让我们非常容易地计算列表中值的频率分布。让我们使用这个类来进行检查（代码在`07/freq_dist.py`中）：

1.  要使用 NLTK 创建频率分布，首先从 NTLK 中导入该功能（还有标记器和停用词）：

```py
from nltk.probabilities import FreqDist
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
```

1.  然后我们可以使用`FreqDist`函数根据单词列表创建频率分布。我们将通过读取`wotw.txt`（《世界大战》- 古腾堡出版社提供）的内容，对其进行标记化并去除停用词来进行检查：

```py
with open('wotw.txt', 'r') as file:
  data = file.read() tokens = [word.lower() for word in regexp_tokenize(data, '\w+')] stoplist = stopwords.words('english') without_stops = [word for word in tokens if word not in stoplist]
```

1.  然后我们可以计算剩余单词的频率分布：

```py
freq_dist = FreqDist(without_stops)
```

1.  `freq_dist`是一个单词到单词计数的字典。以下打印了所有这些单词（只显示了几行输出，因为有成千上万个唯一单词）：

```py
print('Number of words: %s' % len(freq_dist)) for key in freq_dist.keys():
  print(key, freq_dist[key])
**Number of words: 6613
shall 8
dwell 1
worlds 2
inhabited 1
lords 1
world 26
things 64**
```

1.  我们可以使用频率分布来识别最常见的单词。以下报告了最常见的 10 个单词：

```py
print(freq_dist.most_common(10))
[('one', 201), ('upon', 172), ('said', 166), ('martians', 164), ('people', 159), ('came', 151), ('towards', 129), ('saw', 129), ('man', 126), ('time', 122)] 
```

我希望火星人在前 5 名中。它是第 4 名。

# 还有更多...

我们还可以使用这个来识别最不常见的单词，通过使用`.most_common()`的负值进行切片。例如，以下内容找到了最不常见的 10 个单词：

```py
print(freq_dist.most_common()[-10:])
[('bitten', 1), ('gibber', 1), ('fiercer', 1), ('paler', 1), ('uglier', 1), ('distortions', 1), ('haunting', 1), ('mockery', 1), ('beds', 1), ('seers', 1)]
```

有相当多的单词只出现一次，因此这只是这些值的一个子集。只出现一次的单词数量可以通过以下方式确定（由于有 3,224 个单词，已截断）：

```py
dist_1 = [item[0] for item in freq_dist.items() if item[1] == 1] print(len(dist_1), dist_1)

3224 ['dwell', 'inhabited', 'lords', 'kepler', 'quoted', 'eve', 'mortal', 'scrutinised', 'studied', 'scrutinise', 'multiply', 'complacency', 'globe', 'infusoria', ...
```

# 识别和去除罕见单词

我们可以通过利用查找低频词的能力来删除低频词，这些词在某个领域中属于正常范围之外，或者只是从给定领域中被认为是罕见的单词列表中删除。但我们将使用的技术对两者都适用。

# 如何做

罕见单词可以通过构建一个罕见单词列表然后从正在处理的标记集中删除它们来移除。罕见单词列表可以通过使用 NTLK 提供的频率分布来确定。然后您决定应该使用什么阈值作为罕见单词的阈值：

1.  `07/07_rare_words.py` 文件中的脚本扩展了频率分布配方，以识别出现两次或更少的单词，然后从标记中删除这些单词：

```py
with open('wotw.txt', 'r') as file:
  data = file.read()   tokens = [word.lower() for word in regexp_tokenize(data, '\w+')] stoplist = stopwords.words('english') without_stops = [word for word in tokens if word not in stoplist]   freq_dist = FreqDist(without_stops)   print('Number of words: %s' % len(freq_dist))   # all words with one occurrence dist = [item[0] for item in freq_dist.items() if item[1] <= 2] print(len(dist)) not_rare = [word for word in without_stops if word not in dist]   freq_dist2 = FreqDist(not_rare) print(len(freq_dist2))
```

输出结果为：

```py
Number of words: 6613
4361
2252
```

通过这两个步骤，删除停用词，然后删除出现 2 次或更少的单词，我们将单词的总数从 6,613 个减少到 2,252 个，大约是原来的三分之一。

# 识别和删除罕见单词

删除短单词也可以用于去除内容中的噪声单词。以下内容检查了删除特定长度或更短单词。它还演示了通过选择不被视为短的单词（长度超过指定的短单词长度）来进行相反操作。

# 如何做

我们可以利用 NLTK 的频率分布有效地计算短单词。我们可以扫描源中的所有单词，但扫描结果分布中所有键的长度会更有效，因为它将是一个显著较小的数据集：

1.  `07/08_short_words.py` 文件中的脚本举例说明了这个过程。它首先加载了 `wotw.txt` 的内容，然后计算了单词频率分布（删除短单词后）。然后它识别了三个字符或更少的单词：

```py
short_word_len = 3 short_words = [word for word in freq_dist.keys() if len(word) <= short_word_len] print('Distinct # of words of len <= %s: %s' % (short_word_len, len(short_words))) 
```

这将导致：

```py
Distinct # of words of len <= 3: 184
```

1.  通过更改列表推导中的逻辑运算符可以找到不被视为短的单词：

```py
unshort_words = [word for word in freq_dist.keys() if len(word) > short_word_len] print('Distinct # of word > len %s: %s' % (short_word_len, len(unshort_words)))
```

结果为：

```py
Distinct # of word > len 3: 6429
```

# 删除标点符号

根据使用的分词器和这些分词器的输入，可能希望从生成的标记列表中删除标点符号。`regexp_tokenize` 函数使用 `'\w+'` 作为表达式可以很好地去除标点符号，但 `word_tokenize` 做得不太好，会将许多标点符号作为它们自己的标记返回。

# 如何做

通过列表推导和仅选择不是标点符号的项目，类似于从标记中删除其他单词的标点符号的删除。`07/09_remove_punctuation.py` 文件演示了这一点。让我们一起走过这个过程：

1.  我们将从以下开始，它将从工作列表中`word_tokenize`一个字符串：

```py
>>> content = "Strong programming experience in C#, ASP.NET/MVC, JavaScript/jQuery and SQL Server" >>> tokenized = word_tokenize(content) >>> stop_list = stopwords.words('english') >>> cleaned = [word for word in tokenized if word not in stop_list] >>> print(cleaned)
['Strong', 'programming', 'experience', 'C', '#', ',', 'ASP.NET/MVC', ',', 'JavaScript/jQuery', 'SQL', 'Server'] 
```

1.  现在我们可以用以下方法去除标点符号：

```py
>>> punctuation_marks = [':', ',', '.', "``", "''", '(', ')', '-', '!', '#'] >>> tokens_cleaned = [word for word in cleaned if word not in punctuation_marks] >>> print(tokens_cleaned)
['Strong', 'programming', 'experience', 'C', 'ASP.NET/MVC', 'JavaScript/jQuery', 'SQL', 'Server']
```

1.  这个过程可以封装在一个函数中。以下是在 `07/punctuation.py` 文件中，将删除标点符号：

```py
def remove_punctuation(tokens):
  punctuation = [':', ',', '.', "``", "''", '(', ')', '-', '!', '#']
  return [token for token in tokens if token not in punctuation]
```

# 还有更多...

删除标点符号和符号可能是一个困难的问题。虽然它们对许多搜索没有价值，但标点符号也可能需要保留作为标记的一部分。以搜索工作网站并尝试找到 C# 编程职位为例，就像在这个配方中的示例一样。C# 的标记化被分成了两个标记：

```py
>>> word_tokenize("C#") ['C', '#']
```

实际上我们有两个问题。将 C 和 # 分开后，我们失去了 C# 在源内容中的信息。然后，如果我们从标记中删除 #，那么我们也会失去这些信息，因为我们也无法从相邻的标记中重建 C#。

# 拼接 n-gram

关于 NLTK 被用于识别文本中的 n-gram 已经写了很多。n-gram 是文档/语料库中常见的一组单词，长度为*n*个单词（出现 2 次或更多）。2-gram 是任何常见的两个单词，3-gram 是一个三个单词的短语，依此类推。我们不会研究如何确定文档中的 n-gram。我们将专注于从我们的标记流中重建已知的 n-gram，因为我们认为这些 n-gram 对于搜索结果比任何顺序中找到的 2 个或 3 个独立单词更重要。

在解析工作列表的领域中，重要的 2-gram 可能是诸如**计算机科学**、**SQL Server**、**数据科学**和**大数据**之类的东西。此外，我们可以将 C#视为`'C'`和`'#'`的 2-gram，因此在处理工作列表时，我们可能不希望使用正则表达式解析器或`'#'`作为标点符号。

我们需要有一个策略来识别我们的标记流中的这些已知组合。让我们看看如何做到这一点。

# 如何做到这一点

首先，这个例子并不打算进行详尽的检查或者最佳性能的检查。只是一个简单易懂的例子，可以轻松应用和扩展到我们解析工作列表的例子中：

1.  我们将使用来自`StackOverflow` SpaceX 的工作列表的以下句子来检查这个过程：

*我们正在寻找具有以下方面经验的开发人员：ASP.NET、C#、SQL Server 和 AngularJS。我们是一个快节奏、高度迭代的团队，随着我们的工厂的增长，我们必须快速适应。*

1.  这两个句子中有许多高价值的 2-gram（我认为工作列表是寻找 2-gram 的好地方）。仅仅看一下，我就可以挑出以下内容是重要的：

+   +   ASP.NET

+   C#

+   SQL Server

+   快节奏

+   高度迭代

+   快速适应

+   可证明的经验

1.  现在，虽然这些在技术上的定义可能不是 2-gram，但当我们解析它们时，它们都将被分开成独立的标记。这可以在`07/10-ngrams.py`文件中显示，并在以下示例中显示：

```py
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

with open('job-snippet.txt', 'r') as file:
  data = file.read()   tokens = [word.lower() for word in word_tokenize(data)] stoplist = stopwords.words('english') without_stops = [word for word in tokens if word not in stoplist] print(without_stops)
```

这产生了以下输出：

```py
['seeking', 'developers', 'demonstrable', 'experience', ':', 'asp.net', ',', 'c', '#', ',', 'sql', 'server', ',', 'angularjs', '.', 'fast-paced', ',', 'highly', 'iterative', 'team', 'adapt', 'quickly', 'factory', 'grows', '.']
```

我们希望从这个集合中去掉标点，但我们希望在构建一些 2-gram 之后再去做，特别是这样我们可以将"C#"拼接成一个单个标记。

1.  `07/10-reconstruct-2grams.py`文件中的脚本演示了一个函数来实现这一点。首先，我们需要描述我们想要重建的 2-gram。在这个文件中，它们被定义为以下内容：

```py
grams = {
  "c": [{"#": ""}],
  "sql": [{"server": " "}],
  "fast": [{"paced": "-"}],
  "highly": [{"iterative": " "}],
  "adapt": [{"quickly": " "}],
  "demonstrable": [{"experience", " "}]
}
```

`grams`是一个字典，其中键指定了 2-gram 的“左”侧。每个键都有一个字典列表，其中每个字典键可以是 2-gram 的右侧，值是将放在左侧和右侧之间的字符串。

1.  有了这个定义，我们能够看到我们的标记中的`"C"`和`"#"`被重构为"C#"。`"SQL"`和`"Server"`将成为`"SQL Server"`。`"fast"`和`"paced"`将导致`"faced-paced"`。

所以我们只需要一个函数来使这一切工作。这个函数在`07/buildgrams.py`文件中定义：

```py
def build_2grams(tokens, patterns):
  results = []
  left_token = None
 for i, t in enumerate(tokens):
  if left_token is None:
  left_token = t
            continue    right_token = t

        if left_token.lower() in patterns:
  right = patterns[left_token.lower()]
  if right_token.lower() in right:
  results.append(left_token + right[right_token.lower()] + right_token)
  left_token = None
 else:
  results.append(left_token)
  else:
  results.append(left_token)
  left_token = right_token

    if left_token is not None:
  results.append(left_token)
  return results
```

1.  这个函数，给定一组标记和一个以前描述的格式的字典，将返回一组修订后的标记，其中任何匹配的 2-gram 都被放入一个单个标记中。以下演示了它的一些简单用法：

```py
grams = {
  'c': {'#': ''} } print(build_2grams(['C'], grams)) print(build_2grams(['#'], grams)) print(build_2grams(['C', '#'], grams)) print(build_2grams(['c', '#'], grams))
```

这导致以下输出：

```py
['C']
['#']
['C#']
['c#']
```

1.  现在让我们将其应用到我们的输入中。这个完整的脚本在`07/10-reconstruct-2grams.py`文件中（并添加了一些 2-gram）：

```py
grams = {
  "c": {"#": ""},
  "sql": {"server": " "},
  "fast": {"paced": "-"},
  "highly": {"iterative": " "},
  "adapt": {"quickly": " "},
  "demonstrable": {"experience": " "},
  "full": {"stack": " "},
  "enterprise": {"software": " "},
  "bachelor": {"s": "'"},
  "computer": {"science": " "},
  "data": {"science": " "},
  "current": {"trends": " "},
  "real": {"world": " "},
  "paid": {"relocation": " "},
  "web": {"server": " "},
  "relational": {"database": " "},
  "no": {"sql": " "} }   with open('job-snippet.txt', 'r') as file:
  data = file.read()   tokens = word_tokenize(data) stoplist = stopwords.words('english') without_stops = [word for word in tokens if word not in stoplist] result = remove_punctuation(build_2grams(without_stops, grams)) print(result)
```

结果如下：

```py
['We', 'seeking', 'developers', 'demonstrable experience', 'ASP.NET', 'C#', 'SQL Server', 'AngularJS', 'We', 'fast-paced', 'highly iterative', 'team', 'adapt quickly', 'factory', 'grows']
```

完美！

# 还有更多...

我们向`build_2grams()`函数提供一个字典，该字典定义了识别 2-gram 的规则。在这个例子中，我们预定义了这些 2-gram。可以使用 NLTK 来查找 2-gram（以及一般的 n-gram），但是在这个小样本的一个工作职位中，可能找不到任何 2-gram。

# 从 StackOverflow 抓取工作列表

现在让我们将一些内容整合起来，从 StackOverflow 的工作列表中获取信息。这次我们只看一个列表，这样我们就可以了解这些页面的结构并从中获取信息。在后面的章节中，我们将研究如何从多个列表中聚合结果。现在让我们学习如何做到这一点。

# 准备就绪

实际上，StackOverflow 使得从他们的页面中抓取数据变得非常容易。我们将使用来自[`stackoverflow.com/jobs/122517/spacex-enterprise-software-engineer-full-stack-spacex?so=p&sec=True&pg=1&offset=22&cl=Amazon%3b+`](https://stackoverflow.com/jobs/122517/spacex-enterprise-software-engineer-full-stack-spacex?so=p&sec=True&pg=1&offset=22&cl=Amazon%3b+)的内容。在您阅读时，这可能不再可用，因此我已经在`07/spacex-job-listing.html`文件中包含了此页面的 HTML，我们将在本章的示例中使用。

StackOverflow 的工作列表页面非常有结构。这可能是因为它们是由程序员创建的，也是为程序员创建的。页面（在撰写本文时）看起来像下面这样：

![](img/dfb46fdc-eb94-4b80-93a8-885f9ce8a756.png)StackOverflow 工作列表

所有这些信息都被编码在页面的 HTML 中。您可以通过分析页面内容自行查看。但 StackOverflow 之所以如此出色的原因在于它将其大部分页面数据放在一个嵌入的 JSON 对象中。这是放置在`<script type="application/ld+json>`HTML 标签中的，所以很容易找到。下面显示了此标签的截断部分（描述被截断，但所有标记都显示出来）：

![](img/5fecad0b-3acf-4cb6-90d9-4da452fd469b.png)工作列表中嵌入的 JSON

这使得获取内容非常容易，因为我们可以简单地检索页面，找到这个标签，然后使用`json`库将此 JSON 转换为 Python 对象。除了实际的工作描述，还包括了工作发布的大部分“元数据”，如技能、行业、福利和位置信息。我们不需要在 HTML 中搜索信息-只需找到这个标签并加载 JSON。请注意，如果我们想要查找项目，比如工作职责**，我们仍然需要解析描述。还要注意，描述包含完整的 HTML，因此在解析时，我们仍需要处理 HTML 标记。

# 如何做到这一点

让我们去获取这个页面的工作描述。我们将在下一个示例中对其进行清理。

这个示例的完整代码在`07/12_scrape_job_stackoverflow.py`文件中。让我们来看一下：

1.  首先我们读取文件：

```py
with open("spacex-job-listing.txt", "r") as file:
  content = file.read()
```

1.  然后，我们将内容加载到`BeautifulSoup`对象中，并检索`<script type="application/ld+json">`标签：

```py
bs = BeautifulSoup(content, "lxml") script_tag = bs.find("script", {"type": "application/ld+json"})
```

1.  现在我们有了这个标签，我们可以使用`json`库将其内容加载到 Python 字典中：

```py
job_listing_contents = json.loads(script_tag.contents[0]) print(job_listing_contents)
```

这个输出看起来像下面这样（为了简洁起见，这是截断的）：

```py
{'@context': 'http://schema.org', '@type': 'JobPosting', 'title': 'SpaceX Enterprise Software Engineer, Full Stack', 'skills': ['c#', 'sql', 'javascript', 'asp.net', 'angularjs'], 'description': '<h2>About this job</h2>\r\n<p><span>Location options: <strong>Paid relocation</strong></span><br/><span>Job type: <strong>Permanent</strong></span><br/><span>Experience level: <strong>Mid-Level, Senior</strong></span><br/><span>Role: <strong>Full Stack Developer</strong></span><br/><span>Industry: <strong>Aerospace, Information Technology, Web Development</strong></span><br/><span>Company size: <strong>1k-5k people</strong></span><br/><span>Company type: <strong>Private</strong></span><br/></p><br/><br/><h2>Technologies</h2> <p>c#, sql, javascript, asp.net, angularjs</p> <br/><br/><h2>Job description</h2> <p><strong>Full Stack Enterprise&nbsp;Software Engineer</strong></p>\r\n<p>The EIS (Enterprise Information Systems) team writes the software that builds rockets and powers SpaceX. We are responsible for 
```

1.  这很棒，因为现在我们可以做一些简单的任务，而不涉及 HTML 解析。例如，我们可以仅使用以下代码检索工作所需的技能：

```py
# print the skills for skill in job_listing_contents["skills"]:
  print(skill)
```

它产生以下输出：

```py
c#
sql
javascript
asp.net
angularjs
```

# 还有更多...

描述仍然存储在此 JSON 对象的描述属性中的 HTML 中。我们将在下一个示例中检查该数据的解析。

# 阅读和清理工作列表中的描述

工作列表的描述仍然是 HTML。我们将要从这些数据中提取有价值的内容，因此我们需要解析这个 HTML 并执行标记化、停用词去除、常用词去除、进行一些技术 2-gram 处理，以及一般的所有这些不同的过程。让我们来做这些。

# 准备就绪

我已经将确定基于技术的 2-gram 的代码折叠到`07/tech2grams.py`文件中。我们将在文件中使用`tech_2grams`函数。

# 如何做...

这个示例的代码在`07/13_clean_jd.py`文件中。它延续了`07/12_scrape_job_stackoverflow.py`文件的内容：

1.  我们首先从我们加载的描述的描述键创建一个`BeautifulSoup`对象。我们也会打印出来看看它是什么样子的：

```py
desc_bs = BeautifulSoup(job_listing_contents["description"], "lxml") print(desc_bs) <p><span>Location options: <strong>Paid relocation</strong></span><br/><span>Job type: <strong>Permanent</strong></span><br/><span>Experience level: <strong>Mid-Level, Senior</strong></span><br/><span>Role: <strong>Full Stack Developer</strong></span><br/><span>Industry: <strong>Aerospace, Information Technology, Web Development</strong></span><br/><span>Company size: <strong>1k-5k people</strong></span><br/><span>Company type: <strong>Private</strong></span><br/></p><br/><br/><h2>Technologies</h2> <p>c#, sql, javascript, asp.net, angularjs</p> <br/><br/><h2>Job description</h2> <p><strong>Full Stack Enterprise Software Engineer</strong></p>
<p>The EIS (Enterprise Information Systems) team writes the software that builds rockets and powers SpaceX. We are responsible for all of the software on the factory floor, the warehouses, the financial systems, the restaurant, and even the public home page. Elon has called us the "nervous system" of SpaceX because we connect all of the other teams at SpaceX to ensure that the entire rocket building process runs smoothly.</p>
<p><strong>Responsibilities:</strong></p>
<ul>
<li>We are seeking developers with demonstrable experience in: ASP.NET, C#, SQL Server, and AngularJS. We are a fast-paced, highly iterative team that has to adapt quickly as our factory grows. We need people who are comfortable tackling new problems, innovating solutions, and interacting with every facet of the company on a daily basis. Creative, motivated, able to take responsibility and support the applications you create. Help us get rockets out the door faster!</li>
</ul>
<p><strong>Basic Qualifications:</strong></p>
<ul>
<li>Bachelor's degree in computer science, engineering, physics, mathematics, or similar technical discipline.</li>
<li>3+ years of experience developing across a full-stack:  Web server, relational database, and client-side (HTML/Javascript/CSS).</li>
</ul>
<p><strong>Preferred Skills and Experience:</strong></p>
<ul>
<li>Database - Understanding of SQL. Ability to write performant SQL. Ability to diagnose queries, and work with DBAs.</li>
<li>Server - Knowledge of how web servers operate on a low-level. Web protocols. Designing APIs. How to scale web sites. Increase performance and diagnose problems.</li>
<li>UI - Demonstrated ability creating rich web interfaces using a modern client side framework. Good judgment in UX/UI design.  Understands the finer points of HTML, CSS, and Javascript - know which tools to use when and why.</li>
<li>System architecture - Knowledge of how to structure a database, web site, and rich client side application from scratch.</li>
<li>Quality - Demonstrated usage of different testing patterns, continuous integration processes, build deployment systems. Continuous monitoring.</li>
<li>Current - Up to date with current trends, patterns, goings on in the world of web development as it changes rapidly. Strong knowledge of computer science fundamentals and applying them in the real-world.</li>
</ul> <br/><br/></body></html>
```

1.  我们想要浏览一遍，去掉所有的 HTML，只留下描述的文本。然后我们将对其进行标记。幸运的是，使用`BeautifulSoup`很容易就能去掉所有的 HTML 标签：

```py
just_text = desc_bs.find_all(text=True) print(just_text)

['About this job', '\n', 'Location options: ', 'Paid relocation', 'Job type: ', 'Permanent', 'Experience level: ', 'Mid-Level, Senior', 'Role: ', 'Full Stack Developer', 'Industry: ', 'Aerospace, Information Technology, Web Development', 'Company size: ', '1k-5k people', 'Company type: ', 'Private', 'Technologies', ' ', 'c#, sql, javascript, asp.net, angularjs', ' ', 'Job description', ' ', 'Full Stack Enterprise\xa0Software Engineer', '\n', 'The EIS (Enterprise Information Systems) team writes the software that builds rockets and powers SpaceX. We are responsible for all of the software on the factory floor, the warehouses, the financial systems, the restaurant, and even the public home page. Elon has called us the "nervous system" of SpaceX because we connect all of the other teams at SpaceX to ensure that the entire rocket building process runs smoothly.', '\n', 'Responsibilities:', '\n', '\n', 'We are seeking developers with demonstrable experience in: ASP.NET, C#, SQL Server, and AngularJS. We are a fast-paced, highly iterative team that has to adapt quickly as our factory grows. We need people who are comfortable tackling new problems, innovating solutions, and interacting with every facet of the company on a daily basis. Creative, motivated, able to take responsibility and support the applications you create. Help us get rockets out the door faster!', '\n', '\n', 'Basic Qualifications:', '\n', '\n', "Bachelor's degree in computer science, engineering, physics, mathematics, or similar technical discipline.", '\n', '3+ years of experience developing across a full-stack:\xa0 Web server, relational database, and client-side (HTML/Javascript/CSS).', '\n', '\n', 'Preferred Skills and Experience:', '\n', '\n', 'Database - Understanding of SQL. Ability to write performant SQL. Ability to diagnose queries, and work with DBAs.', '\n', 'Server - Knowledge of how web servers operate on a low-level. Web protocols. Designing APIs. How to scale web sites. Increase performance and diagnose problems.', '\n', 'UI - Demonstrated ability creating rich web interfaces using a modern client side framework. Good judgment in UX/UI design.\xa0 Understands the finer points of HTML, CSS, and Javascript - know which tools to use when and why.', '\n', 'System architecture - Knowledge of how to structure a database, web site, and rich client side application from scratch.', '\n', 'Quality - Demonstrated usage of different testing patterns, continuous integration processes, build deployment systems. Continuous monitoring.', '\n', 'Current - Up to date with current trends, patterns, goings on in the world of web development as it changes rapidly. Strong knowledge of computer science fundamentals and applying them in the real-world.', '\n', ' ']
```

太棒了！我们现在已经有了这个，它已经被分解成可以被视为句子的部分！

1.  让我们把它们全部连接在一起，对它们进行词标记，去掉停用词，并应用常见的技术工作 2-gram：

```py
joined = ' '.join(just_text) tokens = word_tokenize(joined)   stop_list = stopwords.words('english') with_no_stops = [word for word in tokens if word not in stop_list] cleaned = remove_punctuation(two_grammed) print(cleaned)
```

这样就会得到以下输出：

```py
['job', 'Location', 'options', 'Paid relocation', 'Job', 'type', 'Permanent', 'Experience', 'level', 'Mid-Level', 'Senior', 'Role', 'Full-Stack', 'Developer', 'Industry', 'Aerospace', 'Information Technology', 'Web Development', 'Company', 'size', '1k-5k', 'people', 'Company', 'type', 'Private', 'Technologies', 'c#', 'sql', 'javascript', 'asp.net', 'angularjs', 'Job', 'description', 'Full-Stack', 'Enterprise Software', 'Engineer', 'EIS', 'Enterprise', 'Information', 'Systems', 'team', 'writes', 'software', 'builds', 'rockets', 'powers', 'SpaceX', 'responsible', 'software', 'factory', 'floor', 'warehouses', 'financial', 'systems', 'restaurant', 'even', 'public', 'home', 'page', 'Elon', 'called', 'us', 'nervous', 'system', 'SpaceX', 'connect', 'teams', 'SpaceX', 'ensure', 'entire', 'rocket', 'building', 'process', 'runs', 'smoothly', 'Responsibilities', 'seeking', 'developers', 'demonstrable experience', 'ASP.NET', 'C#', 'SQL Server', 'AngularJS', 'fast-paced', 'highly iterative', 'team', 'adapt quickly', 'factory', 'grows', 'need', 'people', 'comfortable', 'tackling', 'new', 'problems', 'innovating', 'solutions', 'interacting', 'every', 'facet', 'company', 'daily', 'basis', 'Creative', 'motivated', 'able', 'take', 'responsibility', 'support', 'applications', 'create', 'Help', 'us', 'get', 'rockets', 'door', 'faster', 'Basic', 'Qualifications', 'Bachelor', "'s", 'degree', 'computer science', 'engineering', 'physics', 'mathematics', 'similar', 'technical', 'discipline', '3+', 'years', 'experience', 'developing', 'across', 'full-stack', 'Web server', 'relational database', 'client-side', 'HTML/Javascript/CSS', 'Preferred', 'Skills', 'Experience', 'Database', 'Understanding', 'SQL', 'Ability', 'write', 'performant', 'SQL', 'Ability', 'diagnose', 'queries', 'work', 'DBAs', 'Server', 'Knowledge', 'web', 'servers', 'operate', 'low-level', 'Web', 'protocols', 'Designing', 'APIs', 'scale', 'web', 'sites', 'Increase', 'performance', 'diagnose', 'problems', 'UI', 'Demonstrated', 'ability', 'creating', 'rich', 'web', 'interfaces', 'using', 'modern', 'client-side', 'framework', 'Good', 'judgment', 'UX/UI', 'design', 'Understands', 'finer', 'points', 'HTML', 'CSS', 'Javascript', 'know', 'tools', 'use', 'System', 'architecture', 'Knowledge', 'structure', 'database', 'web', 'site', 'rich', 'client-side', 'application', 'scratch', 'Quality', 'Demonstrated', 'usage', 'different', 'testing', 'patterns', 'continuous integration', 'processes', 'build', 'deployment', 'systems', 'Continuous monitoring', 'Current', 'date', 'current trends', 'patterns', 'goings', 'world', 'web development', 'changes', 'rapidly', 'Strong', 'knowledge', 'computer science', 'fundamentals', 'applying', 'real-world']
```

我认为这是从工作清单中提取出来的一组非常好的和精细的关键词。
