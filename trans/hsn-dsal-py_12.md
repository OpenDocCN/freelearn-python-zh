# 字符串算法和技术

根据所解决的问题，有许多流行的字符串处理算法。然而，最重要、最流行和最有用的字符串处理问题之一是从给定文本中找到给定的子字符串或模式。它有各种应用，例如从文本文档中搜索元素，检测抄袭等。

在本章中，我们将学习标准的字符串处理或模式匹配算法，以找出给定模式或子字符串在给定文本中的位置。我们还将讨论暴力算法，以及Rabin-Karp、Knuth-Morris-Pratt（KMP）和Boyer-Moore模式匹配算法。我们还将讨论与字符串相关的一些基本概念。我们将用简单的解释、示例和实现来讨论所有算法。

本章旨在讨论与字符串相关的算法。本章将涵盖以下主题：

+   学习Python中字符串的基本概念

+   学习模式匹配算法及其实现

+   理解和实现Rabin-Karp模式匹配算法

+   理解和实现Knuth-Morris-Pratt（KMP）算法

+   理解和实现Boyer-Moore模式匹配算法

# 技术要求

本章讨论的基于本章讨论的概念和算法的所有程序都在书中以及GitHub存储库中提供，链接如下：[https://github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter12](https://github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter12)。

# 字符串符号和概念

字符串基本上是一系列对象，主要是一系列字符。与其他任何数据类型（如int或float）一样，我们需要存储数据和要应用的操作。字符串数据类型允许我们存储数据，Python提供了一组丰富的操作和函数，可以应用于字符串数据类型的数据。Python 3.7提供的大多数操作和函数，可以应用于字符串的数据，都在[第1章](2818f56c-fbcf-422f-83dc-16cbdbd8b5bf.xhtml)中详细描述了*Python对象、类型和表达式*。

字符串主要是文本数据，通常处理得非常高效。以下是一个字符串（S）的示例——`"packt publishing"`。

子字符串也是给定字符串的一部分字符序列。例如，`"packt"`是字符串`"packt publishing"`的子字符串。

子序列是从给定字符串中删除一些字符但保持字符出现顺序的字符序列。例如，`"pct pblishing"`是字符串`"packt publishing"`的有效子序列，通过删除字符`a`、`k`和`u`获得。但是，这不是一个子字符串。子序列不同于子字符串，因为它可以被认为是子字符串的泛化。

字符串`s`的前缀是字符串`s`的子字符串，它出现在字符串的开头。还有另一个字符串`u`，它存在于前缀之后的字符串s中。例如，子字符串`"pack"`是字符串`(s) = "packt publishing"`的前缀，因为它是起始子字符串，之后还有另一个子字符串。

后缀`(d)`是一个子字符串，它出现在字符串（s）的末尾，以便在子字符串d之前存在另一个非空子字符串。例如，子字符串`"shing"`是字符串`"packt publishing"`的后缀。Python具有内置函数，用于检查字符串是否具有给定的前缀或后缀，如下面的代码片段所示：

```py
string =  "this is data structures book by packt publisher"; suffix =  "publisher"; prefix = "this"; print(string.endswith(suffix))  #Check if string contains given suffix.
print(string.startswith(prefix)) #Check if string starts with given prefix.

#Outputs
>>True
>>True
```

模式匹配算法是最重要的字符串处理算法，我们将在后续章节中讨论它们。

# 模式匹配算法

模式匹配算法用于确定给定模式字符串（P）在文本字符串（T）中匹配的索引位置。如果模式在文本字符串中不匹配，则返回`"pattern not found"`。例如，对于给定字符串（s）=`"packt publisher"`，模式（p）=`"publisher"`，模式匹配算法返回模式在文本字符串中匹配的索引位置。

在本节中，我们将讨论四种模式匹配算法，即暴力方法，以及Rabin-Karp算法，Knuth-Morris-Pratt（KMP）和Boyer Moore模式匹配算法。

# 暴力算法

暴力算法，或者模式匹配算法的朴素方法，非常基础。使用这种方法，我们简单地测试给定字符串中输入模式的所有可能组合，以找到模式的出现位置。这个算法非常朴素，如果文本很长就不适用。

在这里，我们首先逐个比较模式和文本字符串的字符，如果模式的所有字符与文本匹配，我们返回模式的第一个字符放置的文本的索引位置。如果模式的任何字符与文本字符串不匹配，我们将模式向右移动一个位置。我们继续比较模式和文本字符串，通过将模式向右移动一个索引位置。

为了更好地理解暴力算法的工作原理，让我们看一个例子。假设我们有一个文本字符串(T)=**acbcabccababcaacbcac**，模式字符串(P)是**acbcac**。现在，模式匹配算法的目标是确定给定文本T中模式字符串的索引位置，如下图所示：

![](Images/248df9b8-9f4d-4cb4-a404-b08d3cc5cef8.png)

我们首先比较文本的第一个字符，即**a**，和模式的字符。在这里，模式的初始五个字符匹配，最后一个字符不匹配。由于不匹配，我们进一步将模式向右移动一个位置。我们再次开始逐个比较模式的第一个字符和文本字符串的第二个字符。在这里，文本字符串的字符**c**与模式的字符**a**不匹配。由于不匹配，我们将模式向右移动一个位置，如前面的图所示。我们继续比较模式和文本字符串的字符，直到遍历整个文本字符串。在上面的例子中，我们在索引位置**14**找到了匹配，用箭头指向**aa**。

在这里，让我们考虑模式匹配的暴力算法的Python实现：

```py
def brute_force(text, pattern):
    l1 = len(text)      # The length of the text string
    l2 = len(pattern)   # The length of the pattern 
    i = 0
    j = 0               # looping variables are set to 0
    flag = False        # If the pattern doesn't appear at all, then set this to false and execute the last if statement

    while i < l1:         # iterating from the 0th index of text
        j = 0
        count = 0    
        # Count stores the length upto which the pattern and the text have matched

        while j < l2:
            if i+j < l1 and text[i+j] == pattern[j]:  
        # statement to check if a match has occoured or not
        count += 1     # Count is incremented if a character is matched 
            j += 1
        if count == l2:   # it shows a matching of pattern in the text 
                print("\nPattern occours at index", i) 
                  # print the starting index of the successful match
                flag = True 
     # flag is True as we wish to continue looking for more matching of  
      pattern in the text. 
            i += 1
    if not flag: 
        # If the pattern doesn't occours at all, means no match of  
         pattern in the text string
        print('\nPattern is not at all present in the array')

brute_force('acbcabccababcaacbcac','acbcac')         # function call

#outputs
#Pattern occours at index 14
```

在暴力方法的上述代码中，我们首先计算给定文本字符串和模式的长度。我们还用`0`初始化循环变量，并将标志设置为`False`。这个变量用于在字符串中继续搜索模式的匹配。如果标志在文本字符串结束时为`False`，这意味着在文本字符串中根本没有模式的匹配。

接下来，我们从文本字符串的`0th`索引开始搜索循环，直到末尾。在这个循环中，我们有一个计数变量，用于跟踪匹配的模式和文本的长度。接下来，我们有另一个嵌套循环，从`0th`索引运行到模式的长度。在这里，变量`i`跟踪文本字符串中的索引位置，变量`j`跟踪模式中的字符。接下来，我们使用以下代码片段比较模式和文本字符串的字符：

```py
if i+j<l1 and text[i+j] == pattern[j]:
```

此外，我们在文本字符串中每次匹配模式的字符后递增计数变量。然后，我们继续匹配模式和文本字符串的字符。如果模式的长度等于计数变量，那么就意味着有匹配。

如果在文本字符串中找到了模式的匹配，我们会打印文本字符串的索引位置，并将标志变量保持为“True”，因为我们希望继续在文本字符串中搜索更多模式的匹配。最后，如果标志变量的值为“False”，这意味着在文本字符串中根本没有找到模式的匹配。

朴素字符串匹配算法的最佳情况和最坏情况的时间复杂度分别为`O(n)`和`O(m*(n-m+1))`。最佳情况是模式在文本中找不到，并且模式的第一个字符根本不在文本中，例如，如果文本字符串是`ABAACEBCCDAAEE`，模式是`FAA`。在这种情况下，由于模式的第一个字符在文本中不匹配，比较次数将等于文本的长度(`n`)。

最坏情况发生在文本字符串和模式的所有字符都相同的情况下，例如，如果文本字符串是`AAAAAAAAAAAAAAAA`，模式是`AAAA`。另一个最坏情况是只有最后一个字符不同，例如，如果文本字符串是`AAAAAAAAAAAAAAAF`，模式是`AAAAF`。因此，最坏情况的时间复杂度将是`O(m*(n-m+1))`。

# 拉宾-卡普算法

拉宾-卡普模式匹配算法是改进后的蛮力方法，用于在文本字符串中找到给定模式的位置。拉宾-卡普算法的性能通过减少比较次数来改进，借助哈希。我们在[第7章](7caf334d-44bb-4c49-bb74-4f6e1ac8a8e4.xhtml)中详细描述了哈希，*哈希和符号表*。哈希函数为给定的字符串返回一个唯一的数值。

这种算法比蛮力方法更快，因为它避免了不必要的逐个字符比较。相反，模式的哈希值一次性与文本字符串的子字符串的哈希值进行比较。如果哈希值不匹配，模式就向前移动一位，因此无需逐个比较模式的所有字符。

这种算法基于这样的概念：如果两个字符串的哈希值相等，那么假定这两个字符串也相等。这种算法的主要问题是可能存在两个不同的字符串，它们的哈希值相等。在这种情况下，算法可能无法工作；这种情况被称为虚假命中。为了避免这个问题，在匹配模式和子字符串的哈希值之后，我们通过逐个比较它们的字符来确保模式实际上是匹配的。

拉宾-卡普模式匹配算法的工作原理如下：

1.  首先，在开始搜索之前，我们对模式进行预处理，即计算长度为`m`的模式的哈希值以及长度为`m`的文本的所有可能子字符串的哈希值。因此，可能的子字符串的总数将是`(n-m+1)`。这里，`n`是文本的长度。

1.  我们比较模式的哈希值，并逐一与文本的子字符串的哈希值进行比较。

1.  如果哈希值不匹配，我们就将模式向前移动一位。

1.  如果模式的哈希值和文本的子字符串的哈希值匹配，那么我们逐个比较模式和子字符串的字符，以确保模式实际上在文本中找到。

1.  我们继续进行步骤2-4的过程，直到达到给定文本字符串的末尾。

在这个算法中，我们可以使用Horner法则或任何返回给定字符串唯一值的哈希函数来计算数值哈希值。我们也可以使用字符串所有字符的序数值之和来计算哈希值。

让我们举个例子来理解Rabin-Karp算法。假设我们有一个文本字符串（T）=“publisher paakt packt”，模式（P）=“packt”。首先，我们计算模式（长度为`m`）的哈希值和文本字符串的所有子字符串（长度为`m`）的哈希值。

我们开始比较模式“packt”的哈希值与第一个子字符串“publi”的哈希值。由于哈希值不匹配，我们将模式移动一个位置，然后再次比较模式的哈希值与文本的下一个子字符串“ublis”的哈希值。由于这些哈希值也不匹配，我们再次将模式移动一个位置。如果哈希值不匹配，我们总是将模式移动一个位置。

此外，如果模式的哈希值和子字符串的哈希值匹配，我们逐个比较模式和子字符串的字符，并返回文本字符串的位置。在这个例子中，这些值在位置`17`匹配。重要的是要注意，可能有一个不同的字符串，其哈希值可以与模式的哈希值匹配。这种情况称为虚假命中，是由于哈希冲突而引起的。Rabin-Karp算法的功能如下所示：

![](Images/06dc36ee-ebaf-43b7-b19e-564e36fafb04.png)

# 实现Rabin-Karp算法

实现Rabin-Karp算法的第一步是选择哈希函数。我们使用字符串所有字符的序数值之和作为哈希函数。

我们首先存储文本和模式的所有字符的序数值。接下来，我们将文本和模式的长度存储在`len_text`和`len_pattern`变量中。然后，我们通过对模式中所有字符的序数值求和来计算模式的哈希值。

接下来，我们创建一个名为`len_hash_array`的变量，它存储了使用`len_text - len_pattern + 1`的长度（等于模式的长度）的所有可能子字符串的总数，并创建了一个名为`hash_text`的数组，它存储了所有可能子字符串的哈希值。

接下来，我们开始一个循环，它将运行所有可能的文本子字符串。最初，我们通过使用`sum(ord_text[:len_pattern])`对其所有字符的序数值求和来计算第一个子字符串的哈希值。此外，所有子字符串的哈希值都是使用其前一个子字符串的哈希值计算的，如`((hash_text[i-1] - ord_text[i-1]) + ord_text[i+len_pattern-1])`。

计算哈希值的完整Python实现如下所示：

```py
def generate_hash(text, pattern):
      ord_text = [ord(i) for i in text]   
                       # stores unicode value of each character in text 
      ord_pattern = [ord(j) for j in pattern] 
                   # stores unicode value of each character in pattern
      len_text = len(text)           # stores length of the text 
      len_pattern = len(pattern)     # stores length of the pattern
      hash_pattern = sum(ord_pattern)
      len_hash_array = len_text - len_pattern + 1    
       #stores the length of new array that will contain the hash 
       values of text
      hash_text = [0]*(len_hash_array) 
                         # Initialize all the values in the array to 0.
      for i in range(0, len_hash_array): 
           if i == 0:  
                hash_text[i] = sum(ord_text[:len_pattern]) 
                                      # initial value of hash function
           else:
                hash_text[i] = ((hash_text[i-1] - ord_text[i-1]) + 
                ord_text[i+len_pattern-1]) 
                    # calculating next hash value using previous value

      return [hash_text, hash_pattern]         # return the hash values
```

在预处理模式和文本之后，我们有预先计算的哈希值，我们将用它们来比较模式和文本。

主要的Rabin-Karp算法实现如下。首先，我们将给定的文本和模式转换为字符串格式，因为只能为字符串计算序数值。

接下来，我们调用`generate_hash`函数来计算哈希值。我们还将文本和模式的长度存储在`len_text`和`len_pattern`变量中。我们还将`flag`变量初始化为`False`，以便跟踪模式是否至少出现一次在文本中。

接下来，我们开始一个循环，实现算法的主要概念。这个循环将运行`hash_text`的长度，这是可能子字符串的总数。最初，我们通过使用`if hash_text[i] == hash_pattern`比较子字符串的第一个哈希值和模式的哈希值。它们不匹配；我们什么也不做，寻找另一个子字符串。如果它们匹配，我们通过循环使用`if pattern[j] == text[i+j]`逐个字符比较子字符串和模式。

然后，我们创建一个`count`变量来跟踪模式和子字符串中匹配的字符数。如果计数的长度和模式的长度变得相等，这意味着所有字符都匹配，并且返回模式被找到的索引位置。最后，如果`flag`变量保持为`False`，这意味着模式在文本中根本不匹配。

Rabin-Karp算法的完整Python实现如下所示：

```py
def Rabin_Karp_Matcher(text, pattern):
    text = str(text)                 # convert text into string format
    pattern = str(pattern)           # convert pattern into string format
    hash_text, hash_pattern = generate_hash(text, pattern) 
                    # generate hash values using generate_hash function
    len_text = len(text)              # length of text
    len_pattern = len(pattern)        # length of pattern
    flag = False # checks if pattern is present atleast once or not at all
    for i in range(len(hash_text)): 
        if hash_text[i] == hash_pattern:     # if the hash value matches
            count = 0 
            for j in range(len_pattern): 
                if pattern[j] == text[i+j]: 
                        # comparing patten and substring character by character
                    count += 1  
                else:
                    break
                if count == len_pattern:       # Pattern is found in the text
                    flag = True                # update flag accordingly
                    print("Pattern occours at index", i)
                if not flag:                # Pattern doesn't match even once.
                    print("Pattern is not at all present in the text")
```

Rabin-Karp模式匹配算法在搜索之前预处理模式，即计算模式的哈希值，其复杂度为`O(m)`。此外，Rabin-Karp算法的最坏情况运行时间复杂度为`O(m *(n-m+1))`。

最坏情况是模式根本不在文本中出现。

平均情况将发生在模式至少出现一次的情况下。

# Knuth-Morris-Pratt算法

**Knuth-Morris-Pratt**（**KMP**）算法是一种基于预先计算的前缀函数的模式匹配算法，该函数存储了模式中重叠文本部分的信息。KMP算法预处理这个模式，以避免在使用前缀函数时进行不必要的比较。该算法利用前缀函数来估计模式应该移动多少来搜索文本字符串中的模式，每当我们得到一个不匹配时。KMP算法是高效的，因为它最小化了给定模式与文本字符串的比较。

KMP算法背后的动机可以在以下解释性图表中看到：

![](Images/8f518fde-532f-4f1a-855d-94e9c0b80d22.png)

# 前缀函数

`prefix`函数（也称为失败函数）在模式中查找模式本身。当出现不匹配时，它试图找出由于模式本身的重复而可以重复使用多少之前的比较。它的值主要是最长的前缀，也是后缀。

例如，如果我们有一个模式的`prefix`函数，其中所有字符都不同，那么`prefix`函数的值将为`0`，这意味着如果我们找到任何不匹配，模式将被移动到模式中的字符数。这也意味着模式中没有重叠，并且不会重复使用任何先前的比较。如果文本字符串只包含不同的字符，我们将从模式的第一个字符开始比较。考虑以下示例：模式**abcde**包含所有不同的字符，因此它将被移动到模式中的字符数，并且我们将开始比较模式的第一个字符与文本字符串的下一个字符，如下图所示：

![](Images/ef15fd04-c114-4737-9dbe-344b65cf61a8.png)

让我们考虑另一个示例，以更好地理解`prefix`函数如何为模式（P）**abcabbcab**工作，如下图所示：

![](Images/0a762ce6-7571-451d-a0cb-378a379fc3f1.png)

在上图中，我们从索引**1**开始计算`prefix`函数的值。如果字符没有重复，我们将值赋为**0**。在上面的例子中，我们为索引位置**1**到**3**的`prefix`函数分配了**0**。接下来，在索引位置**4**，我们可以看到有一个字符**a**，它是模式中第一个字符的重复，所以我们在这里分配值**1**，如下所示：

![](Images/58b14e31-38ce-4f51-b496-bdf83615502d.png)

接下来，我们看索引位置**5**处的下一个字符。它有最长的后缀模式**ab**，因此它的值为**2**，如下图所示：

![](Images/cd0fc819-6318-4b89-926b-d283d072cd15.png)

同样，我们看下一个索引位置**6**。这里，字符是**b**。这个字符在模式中没有最长的后缀，所以它的值是**0**。接下来，我们在索引位置**7**处赋值**0**。然后，我们看索引位置**8**，并将值**1**分配给它，因为它有长度为**1**的最长后缀。最后，在索引位置**9**，我们有长度为**2**的最长后缀：

![](Images/04aa8d29-9b55-4e3f-9d39-f519846bc1c5.png)

`prefix`函数的值显示了如果不匹配，字符串的开头有多少可以重复使用。例如，如果在索引位置**5**处比较失败，`prefix`函数的值为**2**，这意味着不需要比较前两个字符。

# 理解KMP算法

KMP模式匹配算法使用具有模式本身重叠的模式，以避免不必要的比较。KMP算法的主要思想是根据模式中的重叠来检测模式应该移动多少。算法的工作原理如下：

1.  首先，我们为给定的模式预先计算`prefix`函数，并初始化一个表示匹配字符数的计数器q。

1.  我们从比较模式的第一个字符与文本字符串的第一个字符开始，如果匹配，则递增模式的计数器**q**和文本字符串的计数器，并比较下一个字符。

1.  如果不匹配，我们将预先计算的`prefix`函数的值赋给**q**的索引值。

1.  我们继续在文本字符串中搜索模式，直到达到文本的末尾，即如果我们找不到任何匹配。如果模式中的所有字符都在文本字符串中匹配，我们返回模式在文本中匹配的位置，并继续搜索另一个匹配。

让我们考虑以下示例来理解这一点：

给定模式的`prefix`函数如下：

![](Images/628b4eb1-1001-487f-9197-1dea713c5750.png)

现在，我们开始比较模式的第一个字符与文本字符串的第一个字符，并继续比较，直到找到匹配。例如，在下图中，我们从比较文本字符串的字符**a**和模式的字符**a**开始。由于匹配，我们继续比较，直到找到不匹配或者我们已经比较了整个模式。在这里，我们在索引位置**6**找到了不匹配，所以现在我们必须移动模式。

我们使用`prefix`函数的帮助来找到模式应该移动的次数。这是因为在不匹配的位置（即`prefix_function(6)`为**2**）上，`prefix`函数的值为**2**，所以我们从模式的索引位置`2`开始比较模式。由于KMP算法的效率，我们不需要比较索引位置**1**的字符，我们比较模式的字符**c**和文本的字符**b**。由于它们不匹配，我们将模式向右移动**1**个位置，如下所示：

![](Images/b4b25215-9e10-4cd4-b815-12f07cb088ad.png)

接下来，我们比较的字符是**b**和**a**——它们不匹配，所以我们将模式向右移动**1**个位置。接下来，我们比较模式和文本字符串，并在文本的索引位置10处找到字符**b**和**c**之间的不匹配。在这里，我们使用预先计算的“前缀”函数来移动模式，因为`prefix_function(4)`是**2**，所以我们将其移动到索引位置**2**，如下图所示：

![](Images/44d1fb23-e596-45c4-a474-374590b295a9.png)

之后，由于字符**b**和**c**不匹配，我们将模式向右移动1个位置。接下来，我们比较文本中索引为**11**的字符，直到找到不匹配为止。我们发现字符**b**和**c**不匹配，如下图所示。由于`prefix_function(2)`是`0`，我们将模式移动到模式的索引`0`。我们重复相同的过程，直到达到字符串的末尾。我们在文本字符串的索引位置**13**找到了模式的匹配，如下所示：

![](Images/bebe5024-3aff-4b76-8623-1c2681510c3d.png)

KMP算法有两个阶段，预处理阶段，这是我们计算“前缀”函数的地方，它的空间和时间复杂度为`O(m)`，然后，在第二阶段，即搜索阶段，KMP算法的时间复杂度为`O(n)`。

现在，我们将讨论如何使用Python实现KMP算法。

# 实现KMP算法

这里解释了KMP算法的Python实现。我们首先为给定的模式实现“前缀”函数。为此，首先我们使用`len()`函数计算模式的长度，然后初始化一个列表来存储“前缀”函数计算出的值。

接下来，我们开始执行循环，从2到模式的长度。然后，我们有一个嵌套循环，直到我们处理完整个模式为止。变量`k`初始化为`0`，这是模式的第一个元素的“前缀”函数。如果模式的第`k`个元素等于第`q`个元素，那么我们将`k`的值增加`1`。

k的值是由“前缀”函数计算得出的值，因此我们将其分配给模式的`q`的索引位置。最后，我们返回具有模式每个字符的计算值的“前缀”函数列表。以下是“前缀”函数的代码：

```py
def pfun(pattern): # function to generate prefix function for the given pattern
    n = len(pattern) # length of the pattern
    prefix_fun = [0]*(n) # initialize all elements of the list to 0
    k = 0
    for q in range(2,n):
         while k>0 and pattern[k+1] != pattern[q]:
            k = prefix_fun[k]
         if pattern[k+1] == pattern[q]: # If the kth element of the pattern is equal to the qth element
            k += 1            # update k accordingly
         prefix_fun[q] = k
    return prefix_fun         # return the prefix function 
```

一旦我们创建了“前缀”函数，我们就实现了主要的KMP匹配算法。我们首先计算文本字符串和模式的长度，它们分别存储在变量`m`和`n`中。以下代码详细显示了这一点：

```py

def KMP_Matcher(text,pattern): 
    m = len(text)
    n = len(pattern)
    flag = False
    text = '-' + text       # append dummy character to make it 1-based indexing
    pattern = '-' + pattern       # append dummy character to the pattern also
    prefix_fun = pfun(pattern) # generate prefix function for the pattern
    q = 0
    for i in range(1,m+1):
        while q>0 and pattern[q+1] != text[i]: 
        # while pattern and text are not equal, decrement the value of q if it is > 0
            q = prefix_fun[q]
        if pattern[q+1] == text[i]: # if pattern and text are equal, update value of q
            q += 1
        if q == n: # if q is equal to the length of the pattern, it means that the pattern has been found.
            print("Pattern occours with shift",i-n) # print the index,
```

```py
where first match occours.
            flag = True
            q = prefix_fun[q]
    if not flag:
            print('\nNo match found')

KMP_Matcher('aabaacaadaabaaba','abaac')         #function call
```

# Boyer-Moore算法

正如我们已经讨论过的，字符串模式匹配算法的主要目标是通过避免不必要的比较来尽可能地跳过比较。

Boyer-Moore模式匹配算法是另一种这样的算法（除了KMP算法），它通过使用一些方法跳过一些比较来进一步提高模式匹配的性能。您需要理解以下概念才能使用Boyer-Moore算法：

1.  在这个算法中，我们将模式从左向右移动，类似于KMP算法

1.  我们从右向左比较模式和文本字符串的字符，这与KMP算法相反

1.  该算法通过使用好后缀和坏字符移位的概念来跳过不必要的比较

# 理解Boyer-Moore算法

Boyer-Moore算法从右到左比较文本上的模式。它通过预处理模式来使用模式中各种可能的对齐信息。这个算法的主要思想是我们将模式的末尾字符与文本进行比较。如果它们不匹配，那么模式可以继续移动。如果末尾的字符不匹配，就没有必要进行进一步的比较。此外，在这个算法中，我们还可以看到模式的哪一部分已经匹配（与匹配的后缀），因此我们利用这个信息，通过跳过任何不必要的比较来对齐文本和模式。

当我们发现不匹配时，Boyer-Moore算法有两个启发式来确定模式的最大可能移位：

+   坏字符启发式

+   好后缀启发式

在不匹配时，每个启发式都建议可能的移位，而Boyer-Moore算法通过考虑由于坏字符和好后缀启发式可能的最大移位来移动模式。坏字符和好后缀启发式的详细信息将在以下子节中通过示例详细解释。

# 坏字符启发式

Boyer-Moore算法将模式和文本字符串从右到左进行比较。它使用坏字符启发式来移动模式。根据坏字符移位的概念，如果模式的字符与文本不匹配，那么我们检查文本的不匹配字符是否出现在模式中。如果这个不匹配的字符（也称为坏字符）不出现在模式中，那么模式将被移动到这个字符的旁边，如果该字符在模式中的某处出现，我们将模式移动到与文本字符串的坏字符对齐的位置。

让我们通过一个例子来理解这个概念。考虑一个文本字符串（T）和模式={**acacac**}。我们从右到左比较字符，即文本字符串的字符**b**和模式的字符**c**。它们不匹配，所以我们在模式中寻找文本字符串的不匹配字符**b**。由于它不在模式中出现，我们将模式移动到不匹配的字符旁边，如下图所示：

![](Images/cc298adf-fb27-4d70-9f29-5148bc40d532.png)

让我们看另一个例子。我们从右到左比较文本字符串和模式的字符，对于文本的字符**d**，我们得到了不匹配。在这里，后缀**ac**是匹配的，但是字符**d**和**c**不匹配，不匹配的字符**d**不在模式中出现。因此，我们将模式移动到不匹配的字符旁边，如下图所示：

![](Images/9fc8421b-9ed9-4282-bf63-522846e13eb9.png)

让我们考虑坏字符启发式的另一个例子。在这里，后缀**ac**是匹配的，但是接下来的字符**a**和**c**不匹配，因此我们在模式中搜索不匹配的字符**a**的出现。由于它在模式中出现了两次，我们有两个选项来对齐不匹配的字符，如下图所示。在这种情况下，我们有多个选项来移动模式，我们移动模式的最小次数以避免任何可能的匹配。（换句话说，它将是模式中该字符的最右出现位置。）如果模式中只有一个不匹配的字符的出现，我们可以轻松地移动模式，使不匹配的字符对齐。

在以下示例中，我们更喜欢选项**1**来移动模式：

![](Images/284beb26-731a-441f-a1c3-2eb800c011c1.png)

# 好后缀启发式

坏字符启发式并不总是提供良好的建议。Boyer-Moore算法还使用好后缀启发式来将模式移位到文本字符串上，以找出匹配模式的位置。

好后缀启发式是基于匹配的后缀。在这里，我们将模式向右移动，以使匹配的后缀子模式与模式中另一个相同后缀的出现对齐。它的工作原理是：我们从右到左开始比较模式和文本字符串。如果我们找到任何不匹配，那么我们检查到目前为止已经匹配的后缀的出现。这被称为好后缀。我们以这样的方式移动模式，以便将好后缀的另一个出现对齐到文本上。好后缀启发式主要有两种情况：

1.  匹配的后缀在模式中有一个或多个出现。

1.  匹配后缀的某部分存在于模式的开头（这意味着匹配后缀的后缀存在于模式的前缀中）。

让我们通过以下示例了解这些情况。假设我们有一个模式**acabac**。我们对字符**a**和**b**进行不匹配，但此时，我们已经匹配了后缀，即**ac**。现在，我们在模式中搜索好后缀**ac**的另一个出现，并通过对齐后缀来移动模式，如下所示：

![](Images/19965040-07c0-4fc5-84d1-a785c3b25d1c.png)

让我们考虑另一个例子，我们有两个选项来对齐模式的移位，以便获得两个好后缀字符串。在这里，我们将选择**1**来通过考虑具有最小移位的选项来对齐好后缀，如下所示：

![](Images/467f95d8-927f-44e1-9ff8-dc722ba839ae.png)

让我们再看一个例子。在这里，我们得到了**aac**的后缀匹配，但对于字符**b**和**a**，我们得到了不匹配。我们搜索好后缀**aac**，但在模式中找不到另一个出现。但是，我们发现模式开头的前缀**ac**与整个后缀不匹配，但与匹配后缀**aac**的后缀**ac**匹配。在这种情况下，我们通过将模式与后缀对齐来移动模式，如下所示：

![](Images/323ab0d1-b503-42f7-ab66-aa97a9fc0299.png)

另一个好后缀启发式的案例如下。在这种情况下，我们匹配后缀**aac**，但在字符**b**和**a**处不匹配。我们尝试在模式中搜索匹配的后缀，但在模式中没有后缀的出现，所以在这种情况下，我们将模式移位到匹配的后缀之后，如下所示：

![](Images/e84ff37e-2ddd-4ece-9883-2011cd8531ed.png)

我们通过坏字符启发式和好后缀启发式给出的更长距离来移动模式。

Boyer-Moore算法在模式的预处理中需要`O(m)`的时间，进一步搜索需要`O(mn)`的时间复杂度。

# 实现Boyer-Moore算法

让我们了解Boyer-Moore算法的实现。最初，我们有文本字符串和模式。在初始化变量之后，我们开始使用while循环，该循环从模式的最后一个字符开始与文本的相应字符进行比较。

然后，通过使用嵌套循环从模式的最后一个索引到模式的第一个字符，从右到左比较字符。这使用`range(len(pattern)-1, -1, -1)`。

外部while循环跟踪文本字符串中的索引，而内部for循环跟踪模式中的索引位置。

接下来，我们开始使用`pattern[j] != text[i+j]`来比较字符。如果它们不匹配，我们将使标志变量`False`，表示存在不匹配。

现在，我们通过使用条件`j == len(pattern)-1`来检查好后缀是否存在。如果这个条件为真，意味着没有可能的好后缀，所以我们检查坏字符启发式，即如果模式中存在或不存在不匹配的字符，使用条件`text[i+j] in pattern[0:j]`，如果条件为真，则意味着坏字符存在于模式中。在这种情况下，我们使用`i=i+j-pattern[0:j].rfind(text[i+j])`将模式移动到与模式中此字符的其他出现对齐。这里，`(i+j)`是坏字符的索引。

如果坏字符不在模式中（它不在`else`部分），我们使用索引`i=i+j+1`将整个模式移动到不匹配的字符旁边。

接下来，我们进入条件的`else`部分，检查好后缀。当我们发现不匹配时，我们进一步测试，看看我们的模式前缀中是否有任何好后缀的子部分。我们通过使用以下条件来做到这一点：

```py
 text[i+j+k:i+len(pattern)] not in pattern[0:len(pattern)-1]
```

此外，我们检查好后缀的长度是否为`1`。如果好后缀的长度为`1`，我们不考虑这个移动。如果好后缀大于`1`，我们通过好后缀启发式找出移动次数，并将其存储在`gsshift`变量中。这是将模式移动到与文本的好后缀匹配的位置的指令。此外，我们计算由于坏字符启发式可能的移动次数，并将其存储在`bcshift`变量中。当坏字符存在于模式中时，可能的移动次数是`i+j-pattern[0:j].rfind(text[i+j])`，当坏字符不在模式中时，可能的移动次数将是`i+j+1`。

接下来，我们通过使用坏字符和好后缀启发式的最大移动次数将模式移动到文本字符串上。最后，我们检查标志变量是否为`True`。如果为`True`，这意味着找到了模式，并且匹配的索引已存储在`matched_indexes`变量中。

Boyer-Moore算法的完整实现如下所示：

```py
text= "acbaacacababacacac"
pattern = "acacac"

matched_indexes = []
i=0
flag = True
while i<=len(text)-len(pattern):
    for j in range(len(pattern)-1, -1, -1): #reverse searching
        if pattern[j] != text[i+j]:
            flag = False #indicates there is a mismatch
            if j == len(pattern)-1: #if good-suffix is not present, we test bad character 
                if text[i+j] in pattern[0:j]:
                    i=i+j-pattern[0:j].rfind(text[i+j]) #i+j is index of bad character, this line is used for jumping pattern to match bad character of text with same character in pattern
                else:
                    i=i+j+1 #if bad character is not present, jump pattern next to it
            else:
                k=1
                while text[i+j+k:i+len(pattern)] not in pattern[0:len(pattern)-1]: #used for finding sub part of a good-suffix
                    k=k+1
                if len(text[i+j+k:i+len(pattern)]) != 1: #good-suffix should not be of one character
                    gsshift=i+j+k-pattern[0:len(pattern)-1].rfind(text[i+j+k:i+len(pattern)]) #jumps pattern to a position where good-suffix of pattern matches with good-suffix of text
                else:
                    #gsshift=i+len(pattern)
                    gsshift=0 #when good-suffix heuristic is not applicable, we prefer bad character heuristic
                if text[i+j] in pattern[0:j]:
                    bcshift=i+j-pattern[0:j].rfind(text[i+j]) #i+j is index of bad character, this line is used for jumping pattern to match bad character of text with same character in pattern
                else:
                    bcshift=i+j+1
                i=max((bcshift, gsshift))
            break
    if flag: #if pattern is found then normal iteration
        matched_indexes.append(i)
        i = i+1
    else: #again set flag to True so new string in text can be examined
        flag = True

print ("Pattern found at", matched_indexes)

```

# 总结

在本章中，我们已经讨论了在实时场景中具有广泛应用的最流行和重要的字符串处理算法。我们从查看与字符串相关的基本概念和定义开始了本章。接下来，我们详细描述了用于模式匹配问题的暴力、Rabin-Karp、KMP和Boyer-Moore模式匹配算法。我们已经看到，暴力模式匹配算法非常慢，因为它逐个比较模式和文本字符串的字符。

在模式匹配算法中，我们试图找到跳过不必要比较的方法，并尽快将模式移动到文本上，以快速找到匹配模式的位置。KMP算法通过查看模式本身中的重叠子字符串来找出不必要的比较，以避免不重要的比较。此外，我们讨论了Boyer-Moore算法，在文本和模式很长时非常高效。这是实践中使用的最流行的模式匹配算法。

在下一章中，我们将更详细地讨论数据结构设计策略和技术。
