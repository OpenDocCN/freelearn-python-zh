# 划分文本数据和构建文本分类器

本章介绍了以下主题：

+   构建文本分类器

+   使用标记化预处理数据

+   词干化文本数据

+   使用分块划分文本

+   构建词袋模型

+   文本分类器的应用

# 介绍

本章介绍了构建文本分类器的方法。这包括从数据库中提取重要特征、训练、测试和验证文本分类器。最初，使用常用词训练文本分类器。后来，训练好的文本分类器用于预测。构建文本分类器包括使用标记化预处理数据、词干化文本数据、使用分块划分文本和构建词袋模型。

# 构建文本分类器

分类器单元通常被认为是将数据库分成各种类别的。朴素贝叶斯分类器方案被广泛认为是根据训练模型将文本分隔的文献。本章的这一部分最初考虑了一个带有关键词的文本数据库；特征提取从文本中提取关键短语并训练分类器系统。然后，实施**词频-逆文档频率**（**tf-idf**）转换以指定单词的重要性。最后，使用分类器系统预测并打印输出。

# 如何做...

1.  在新的Python文件中包含以下行以添加数据集：

```py
from sklearn.datasets import fetch_20newsgroups 
category_mapping = {'misc.forsale': 'Sellings', 'rec.motorcycles': 'Motorbikes', 
        'rec.sport.baseball': 'Baseball', 'sci.crypt': 'Cryptography', 
        'sci.space': 'OuterSpace'} 

training_content = fetch_20newsgroups(subset='train', 
categories=category_mapping.keys(), shuffle=True, random_state=7) 
```

1.  执行特征提取以从文本中提取主要单词：

```py
from sklearn.feature_extraction.text import CountVectorizer 

vectorizing = CountVectorizer() 
train_counts = vectorizing.fit_transform(training_content.data) 
print "nDimensions of training data:", train_counts.shape 
```

1.  训练分类器：

```py
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import TfidfTransformer 

input_content = [ 
    "The curveballs of right handed pitchers tend to curve to the left", 
    "Caesar cipher is an ancient form of encryption", 
    "This two-wheeler is really good on slippery roads" 
] 

tfidf_transformer = TfidfTransformer() 
train_tfidf = tfidf_transformer.fit_transform(train_counts) 
```

1.  实现多项式朴素贝叶斯分类器：

```py
classifier = MultinomialNB().fit(train_tfidf, training_content.target) 
input_counts = vectorizing.transform(input_content) 
input_tfidf = tfidf_transformer.transform(input_counts) 
```

1.  预测输出类别：

```py
categories_prediction = classifier.predict(input_tfidf) 
```

1.  打印输出：

```py
for sentence, category in zip(input_content, categories_prediction): 
    print 'nInput:', sentence, 'nPredicted category:',  
            category_mapping[training_content.target_names[category]] 
```

以下屏幕截图提供了根据数据库输入预测对象的示例：

![](Images/14d70abf-d300-4e88-9ece-6318c2e1abb0.png)

# 它是如何工作的...

本章的前一部分提供了有关实施分类器部分和一些样本结果的见解。分类器部分是基于训练好的朴素贝叶斯中的先前文本与测试序列中的关键测试之间的比较工作的。

# 另请参阅

请参阅以下文章：

+   *情感分析算法和应用：调查*在[https://www.sciencedirect.com/science/article/pii/S2090447914000550](https://www.sciencedirect.com/science/article/pii/S2090447914000550)。

+   *在线评论的情感分类：使用基于句子的语言模型*来学习情感预测的工作方式[https://www.tandfonline.com/doi/abs/10.1080/0952813X.2013.782352?src=recsys&journalCode=teta20](https://www.tandfonline.com/doi/abs/10.1080/0952813X.2013.782352?src=recsys&journalCode=teta20)。

+   *使用产品评论数据进行情感分析*和*在存在情感的情况下进行句子级情感分析*，以了解推荐系统中使用的各种指标[https://journalofbigdata.springeropen.com/articles/10.1186/s40537-015-0015-2](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-015-0015-2)和[https://link.springer.com/chapter/10.1007/978-3-642-54903-8_1](https://link.springer.com/chapter/10.1007/978-3-642-54903-8_1)。

# 使用标记化预处理数据

数据的预处理涉及将现有文本转换为学习算法的可接受信息。

标记化是将文本分成一组有意义的片段的过程。这些片段被称为标记。

# 如何做...

1.  介绍句子标记化：

```py
from nltk.tokenize import sent_tokenize
```

1.  形成一个新的文本标记器：

```py
tokenize_list_sent = sent_tokenize(text)
print "nSentence tokenizer:" 
print tokenize_list_sent 
```

1.  形成一个新的单词标记器：

```py
from nltk.tokenize import word_tokenize 
print "nWord tokenizer:" 
print word_tokenize(text) 
```

1.  引入一个新的WordPunct标记器：

```py
from nltk.tokenize import WordPunctTokenizer 
word_punct_tokenizer = WordPunctTokenizer() 
print "nWord punct tokenizer:" 
print word_punct_tokenizer.tokenize(text) 
```

标记器得到的结果如下所示。它将一个句子分成单词组：

![](Images/fb4eb393-47a2-496c-a3cb-a2f99a36dfa6.png)

# 词干化文本数据

词干处理过程涉及为标记器的单词创建适当的缩写单词。

# 如何做...

1.  使用新的Python文件初始化词干处理过程：

```py
from nltk.stem.porter import PorterStemmer 
from nltk.stem.lancaster import LancasterStemmer 
from nltk.stem.snowball import SnowballStemmer 
```

1.  让我们描述一些要考虑的词，如下所示：

```py
words = ['ability', 'baby', 'college', 'playing', 'is', 'dream', 'election', 'beaches', 'image', 'group', 'happy'] 
```

1.  确定要使用的一组`词干处理器`：

```py
stemmers = ['PORTER', 'LANCASTER', 'SNOWBALL'] 
```

1.  为所选的“词干处理器”初始化必要的任务：

```py
stem_porter = PorterStemmer() 
stem_lancaster = LancasterStemmer() 
stem_snowball = SnowballStemmer('english') 
```

1.  格式化表格以打印结果：

```py
formatted_row = '{:>16}' * (len(stemmers) + 1) 
print 'n', formatted_row.format('WORD', *stemmers), 'n' 
```

1.  反复检查单词列表，并使用选择的“词干处理器”对它们进行排列：

```py
for word in words:
  stem_words = [stem_porter.stem(word), 
  stem_lancaster.stem(word), 
  stem_snowball.stem(word)] 
  print formatted_row.format(word, *stem_words) 
```

词干处理过程得到的结果如下截图所示：

![](Images/c6b683e2-a8d2-4273-89f8-7b841628e28b.png)

# 使用分块分割文本

分块过程可用于将大文本分成小的、有意义的单词。

# 如何做…

1.  使用Python开发并导入以下包：

```py
import numpy as np 
from nltk.corpus import brown 
```

1.  描述将文本分成块的函数：

```py
# Split a text into chunks 
def splitter(content, num_of_words): 
   words = content.split(' ') 
   result = [] 
```

1.  初始化以下编程行以获取分配的变量：

```py
   current_count = 0 
   current_words = []
```

1.  开始使用单词进行迭代：

```py
   for word in words: 
     current_words.append(word) 
     current_count += 1 
```

1.  获取必要数量的单词后，重新组织变量：

```py
     if current_count == num_of_words: 
       result.append(' '.join(current_words)) 
       current_words = [] 
       current_count = 0 
```

1.  将块附加到输出变量：

```py
       result.append(' '.join(current_words)) 
       return result 
```

1.  导入“布朗语料库”的数据，并考虑前10000个单词：

```py
if __name__=='__main__': 
  # Read the data from the Brown corpus 
  content = ' '.join(brown.words()[:10000]) 
```

1.  描述每个块中的字大小：

```py
  # Number of words in each chunk 
  num_of_words = 1600 
```

1.  初始化一对重要的变量：

```py
  chunks = [] 
  counter = 0 
```

1.  通过调用“分割器”函数打印结果：

```py
  num_text_chunks = splitter(content, num_of_words) 
  print "Number of text chunks =", len(num_text_chunks) 
```

1.  分块后得到的结果如下截图所示：

![](Images/e33566a4-1380-4420-91db-00b582161836.png)

# 构建词袋模型

在处理包含大单词的文本文档时，我们需要将它们转换为几种类型的算术表示。我们需要将它们制定为适合机器学习算法的形式。这些算法需要算术信息，以便它们可以检查数据并提供重要的细节。词袋程序帮助我们实现这一点。词袋创建一个文本模型，使用文档中的所有单词来发现词汇。随后，它通过构建文本中所有单词的直方图来为每个文本创建模型。

# 如何做…

1.  通过导入以下文件初始化一个新的Python文件：

```py
import numpy as np 
from nltk.corpus import brown 
from chunking import splitter 
```

1.  定义`main`函数并从“布朗语料库”中读取输入数据：

```py
if __name__=='__main__': 
        content = ' '.join(brown.words()[:10000]) 
```

1.  将文本内容分成块：

```py
    num_of_words = 2000 
    num_chunks = [] 
    count = 0 
    texts_chunk = splitter(content, num_of_words) 
```

1.  基于这些“文本”块构建词汇表：

```py
    for text in texts_chunk: 
      num_chunk = {'index': count, 'text': text} 
      num_chunks.append(num_chunk) 
      count += 1
```

1.  提取文档词矩阵，有效地计算文档中每个单词的出现次数：

```py
  from sklearn.feature_extraction.text      
  import CountVectorizer
```

1.  提取文档术语“矩阵”：

```py
from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(min_df=5, max_df=.95) 
matrix = vectorizer.fit_transform([num_chunk['text'] for num_chunk in num_chunks]) 
```

1.  提取词汇并打印它：

```py
vocabulary = np.array(vectorizer.get_feature_names()) 
print "nVocabulary:" 
print vocabulary 
```

1.  打印文档术语“矩阵”：

```py
print "nDocument term matrix:" 
chunks_name = ['Chunk-0', 'Chunk-1', 'Chunk-2', 'Chunk-3', 'Chunk-4'] 
formatted_row = '{:>12}' * (len(chunks_name) + 1) 
print 'n', formatted_row.format('Word', *chunks_name), 'n' 
```

1.  迭代单词，并打印每个单词在不同块中的重现：

```py
for word, item in zip(vocabulary, matrix.T): 
# 'item' is a 'csr_matrix' data structure 
 result = [str(x) for x in item.data] 
 print formatted_row.format(word, *result)
```

1.  执行词袋模型后获得的结果如下所示：

![](Images/be04e751-2b4b-4148-a279-59415d35db0a.png)![](Images/c0088dd1-48ad-4485-b29b-14b920f4998e.png)

为了理解它在给定句子上的工作原理，请参考以下内容：

+   *情感分析简介*，在这里解释：[https://blog.algorithmia.com/introduction-sentiment-analysis/](https://blog.algorithmia.com/introduction-sentiment-analysis/)

# 文本分类器的应用

文本分类器用于分析客户情绪，在产品评论中，当在互联网上搜索查询时，在社交标签中，预测研究文章的新颖性等等。
