# 6

# TC1 汇编器和模拟器设计

在本章中，我们将结合之前章节学到的知识，构建一个计算机模拟器。本章我们将涵盖的关键主题如下：

+   分析指令

+   处理汇编指令

+   构建二进制指令

+   pre-TC1（实际模拟器的前传）

+   TC1 模拟器程序

+   一个 TC1 汇编语言程序

+   TC1 后记

到本章结束时，您应该了解模拟器是如何设计的，并且能够创建一个模拟器。接下来的两个章节将专注于扩展模拟器并提供更多功能，例如输入错误检测。

# 技术要求

您可以在 GitHub 上找到本章使用的程序：[`github.com/PacktPublishing/Practical-Computer-Architecture-with-Python-and-ARM/tree/main/Chapter06`](https://github.com/PacktPublishing/Practical-Computer-Architecture-with-Python-and-ARM/tree/main/Chapter06)。

为了构建一个基于 Python 的模拟器，您需要之前章节中使用的相同工具；也就是说，您需要一个编辑器来创建 Python 程序，以及一个 Python 解释器。这些工具包含在我们之前在*第一章*中介绍的免费 Python 软件包中。

# 分析指令

在本节中，我们将探讨如何将表示汇编语言指令的文本字符串转换为可以被模拟器执行的二进制代码。

有趣的是，汇编器可能比实际的模拟器更复杂。实际上，在本章中，我们对模拟器本身的篇幅相对较少。我们实际上不需要汇编器，因为将汇编级操作手动翻译成二进制代码很容易；这只是填写 32 位指令格式字段的问题。例如，*将常量值 42 加载到寄存器 R7 中*可以写成`LDRL` `R7`,`42`。这个指令有 7 位操作码，`01` `01010`，目标寄存器是`r7`（代码`111`），两个源寄存器未使用，它们的字段都可以设置为`000`。常量是`42`，或者作为 16 位二进制值`0000000000101010`。二进制编码的指令如下：

0001010111`000000`0000000000101010

手动翻译代码很容易，但并不有趣。我们将创建一个汇编器来自动化翻译过程，并允许您使用符号名称而不是实际的常量（常数）。考虑以下汇编语言代码的示例。这是使用*数值*（阴影标注）而不是符号名称编写的。这并不是一个特定的汇编语言；它旨在说明基本概念：

```py

        LDRL R0,60              @ Load R0 with the time factor, 60
        .
        CMP  R3,R5              @ Compare R3 and R5
        BEQ  1                  @ If equal, jump to next-but-one instruction
        ADD  R2,R2,4
        SUB  R7,R1,R2
```

在以下示例中，常量值已被替换为符号名称。这些名称被阴影标注：

```py

Minutes EQU  60                 @ Set up a constant
Test    EQU  4
.       LDRL R0,Minutes         @ Load R0 with the time factor
.       CMP  R3,R5
        BEQ  Next
        ADD  R2,R2,Test
Next    SUB  R7,R1,R2
```

Python 的字典结构确实使得处理符号名称变得非常容易。前面的示例展示了处理包含汇编语言的文本文件的过程。这个文件被称为`sFile`，它只是一个包含汇编语言指令的`.txt`文件。

## 处理输入

我们现在将探讨如何处理原始输入——即包含汇编语言源代码的文本文件。原则上，有一个源文件，其中汇编语言指令都完美格式化和排列，那将是很好的。

在现实中，一个程序可能不是理想地格式化的；例如，可能有空白行或需要忽略的程序注释。

我们设计这种汇编语言，以便在编写 TC1 程序时具有相当大的灵活性。实际上，它允许一种大多数真实汇编器都没有实现的自由格式。我们采取这种方法的几个原因包括。首先，它展示了如何执行文本处理，这是汇编器设计的基本部分。其次，自由格式意味着你不必记住是否使用大写或小写名称和标签。

一些语言是大小写敏感的，而另一些则不是。我们设计的汇编语言是*不区分大小写*的；也就是说，你可以写 `ADD r0,r1,r2` 或者 `ADD R0,R1,R2`。因此，我们可以以下列所有形式编写加载寄存器立即指令来执行*加载* *寄存器索引*操作：

`LDRI` `R2,[R1],10` 或者

`LDRI` `R2,r1,10` 或者

`LDRI` `R2,[R1,10]` 或者 `,`

`LDRI r2,r1,10`

这种记法自由度之所以可能，是因为`[]`括号实际上并不必要来识别指令；它们在程序中使用，因为程序员将`[r0]`与间接寻址联系起来。换句话说，括号是为了程序员而不是计算机而存在的，是多余的。

然而，这种自由度并不一定是可取的，因为它可能导致错误，并使一个人阅读另一个人的程序变得更加困难。所有设计决策都伴随着利弊。

以下 Python 示例包含一个简短的嵌入式汇编语言程序。Python 代码被设计成你可以使用汇编器的一部分汇编语言程序（这只是为了测试和调试目的，因为它避免了每次想要测试一个特性时都需要进入文本编辑器的情况）或者磁盘上的文本形式程序。在这个例子中，我们在我的电脑上找到了测试程序，位于`E:\testCode.txt`。当演示文本处理代码运行时，它会询问你是从磁盘读取代码还是读取嵌入式代码。输入`d`读取磁盘，输入任何其他输入读取嵌入式代码。

汇编语言程序的文件名为 `testCode = 'E://testCode.txt'`。在 Python 程序中，我们使用双反斜杠代替传统的文件命名约定。

文本处理程序移除空白行，将文本转换为大写（允许您写`r0`或`R0`），并允许您使用逗号或空格作为分隔符（您可以写`ADD R0,R1,R2`或`ADD r0 r1 r2`）。我们还移除了指令前后多余的空格。最终结果是标记化列表；也就是说，`ADD r0,r1,r2`被转换为`['ADD','R0','R1','R2']`列表。现在，汇编器可以查找指令，然后提取它所需的信息（寄存器编号和字面量）。

在以下程序中，我们每次处理一行时都会使用一个新的变量，以便帮助您跟踪变量。以下是一个示例：

```py

sFile2 = [i.upper() for i in sFile1]          # Convert in to uppercase
sFile3 = [i.split('@')[0] for i in sFile2]    # Remove comments
```

我们为了清晰起见使用了不同的变量名。通常，你会这样写：

```py

sFile = [i.upper() for i in sFile]            #
sFile = [i.split('@')[0] for i in sFile]      #
```

我们使用文件理解来移除代码中的注释：

```py

sFile3 = [i.split('@')[0] for i in sFile2]    # Remove comments
```

这是一个相当巧妙的技巧，需要解释。它将`sFile2`中的每一行复制到`sFile3`。然而，对于每一行的复制值是`i.split('@')[0]`，其中`i`是当前行。`split('@')`方法使用`'@'`作为分隔符将列表分割成字符串。如果原始字符串中没有`'@'`，则字符串将被复制。如果有`'@'`，它将被复制为两个字符串；例如，`ADD R1,R2,R3 @ Sum the totals`将被复制到`sFile3`作为`'ADD R1,R2,R3','@ Sum the totals'`。然而，由于`[0]`索引，只复制列表的第一个元素；也就是说，只复制`'ADD R1,R2,R3'`，并移除了注释。

文本输入处理块如下所示：

```py

testCode = 'E://testCode.txt'
altCode  = ['nop', 'NOP 5', 'add R1,R2','', 'LDR r1,[r2]', \
            'ldr r1,[R2]','\n', 'BEQ test @www','\n']
x = input('For disk enter d, else any character ')
if x == 'd':
    with open(testCode, 'r') as source0:
         source = source0.readlines()
    source = [i.replace('\n','') for i in source]
else:    source = altCode
print('Source code to test is',source)
sFile0 = []
for i in range(0,len(source)):                # Process the source file in list sFile
    t1 =  source[i].replace(',',' ')          # Replace comma with space
    t2 =  t1.replace('[',' ')                 # Remove [ brackets
    t3 =  t2.replace(']',' ')                 # Remove ] brackets
    t4 =  t3.replace('  ',' ')                # Remove any double spaces
    sFile0.append(t4)                         # Add result to source file
sFile1= [i for i in sFile0 if i[-1:]!='\n']   # Remove end-of-lines
sFile2= [i.upper() for i in sFile1]           # All uppercase
sFile3= [i.split('@')[0] for i in sFile2]     # Remove comments with @
sFile4= [i.rstrip(' ') for i in sFile3 ]      # Remove trailing spaces
sFile5= [i.lstrip(' ') for i in sFile4 ]      # Remove leading spaces
sFile6=[i for i in sFile5 if i != '']         # Remove blank lines
print ('Post-processed output',  sFile6)
```

以下是用此代码的两个示例。在第一种情况下，用户输入是`d`，表示磁盘程序，在第二种情况下，输入是`x`，表示使用嵌入式源程序。在每种情况下，都会打印课程和输出值以演示字符串处理操作。

### 情况 1 – 磁盘输入

```py

For disk enter d, else any character d
Source code to test is ['diskCode', 'b', 'add r1,r2,[r3]', '', 'ADD r3 @test', ' ', 'r2,,r3', ' ', 'gg']
Post-processed output ['DISKCODE', 'B', 'ADD R1 R2 R3', 'ADD R3', 'R2 R3', 'GG']
```

### 情况 2 – 使用嵌入式测试程序

```py

For disk enter d, else any character x
Source code to test is ['nop', 'NOP 5', 'add R1,R2', '', 'LDR r1,[r2]', 'ldr r1,[R2]', '\n', 'BEQ test @www', '\n']
Post-processed output ['NOP', 'NOP 5', 'ADD R1 R2', 'LDR R1 R2', 'LDR R1 R2', 'BEQ TEST']
```

上述代码并不代表一个最优的文本处理系统。它被设计用来演示在处理文本之前涉及的基本过程。然而，这些概念将在 TC1 中再次出现。

### 处理助记符

名字里有什么？我们如何知道一个`NOP`指令是独立的，而一个`ADD`指令需要三个寄存器？在本节中，我们将开始讨论汇编语言指令是如何被处理的，以便提取它们的含义（即，将它们转换为二进制形式）。

考虑以下 TC1 汇编语言的片段：

```py

ADD  R1,R3,R7        @ Three operands (three registers)
NOP                  @ No operands
LDRL R4,27           @ Two operands (register and a literal value)
```

当汇编器读取一行时，它需要知道如何处理操作码及其操作数。那么，它是如何知道如何进行操作的？我们可以使用 Python 的字典功能以非常简单的方式解决这个问题，只需查看表格以查看操作码需要哪些信息。

回想一下，字典是一组或集合的项目，其中每个项目有两个组成部分；例如，一个英德词典包含由一个英语单词及其德语对应词组成的项目。你查找的单词称为 *键*，它提供 *值*。例如，在英德词典中，项目 `'town':'Stadt'` 由键 `town` 和值 `Stadt` 组成。字典是 *查找表* 的一个高级名称。

Python 中的字典由其标点符号定义（即，它不需要任何特殊的保留 Python 词汇）；它是一种由花括号 `{}` 包围的列表类型。每个列表项由一个键及其值组成，键和值之间用冒号分隔。连续的项用逗号分隔，就像列表一样。

*键* 用于访问字典中适当的价值。在 TC1 中，键是用于查找指令详细信息的 *助记符*。让我们创建一个名为 `codes` 的字典，其中包含三个键，这些键是表示有效 TC1 指令的字符串：`STOP`、`ADD` 和 `LDRL`。这个字典可以写成以下形式：

```py

codes = {'STOP':P, 'ADD':Q, 'LDRL':R}               # P, Q, R are variables
```

每个键都是一个以冒号结尾的字符串，后面跟着它的值。键不一定是字符串。在这种情况下，它是一个字符串，因为我们正在使用它来查找助记符，这些助记符是文本字符串。第一个 `key:value` 对是 `'STOP':P`，其中 `'STOP'` 是键，`P` 是它的值。假设我们想知道 `ADD` 是否是一个合法的指令（即，在字典中）。我们可以通过以下方式测试这个指令（即，键）是否在字典中：

```py

    if 'ADD' in codes:  # Test whether 'ADD' is a valid mnemonic in the dictionary
```

如果键在字典中，则返回 `True`，否则返回 `False`。你可以使用 `not in` 来测试某个元素是否不在字典中。

Python 允许将任何有效的对象与键关联，例如，一个列表。例如，我们可以写出以下 `key:value` 对：

```py

 'ADD': [3, 0b1101001, 'Addition', '07/05/2021', timesUsed]
```

在这里，与键关联的值是一个包含五个元素的列表，它将 `ADD` 助记符与操作数的数量、其二进制编码、其名称、设计日期以及它在当前程序中被使用的次数（以及能够从字典中读取值，你还可以写入并更新它）相关联。

以下代码设置了一个字典，将助记符与变量（预设为整数 `1,2,3,4`）绑定：

```py

P,Q,R,N = 1,2,3,4                                   # Set up dummy opcodes
validCd = {'STOP':P, 'ADD':Q, 'LDRL':R, 'NOP':N}    # Dictionary of codes
x = input('Please enter a code  ')                  # Request an opcode
if x not in validCd:                                # Check dictionary for errors
    print('Error! This is not valid')
if x in validCd:                                    # Check for valid opcode
    print('Valid op ', validCd.get(x))              # If found, read its value
```

在这个例子中，我们使用了 `get()` 方法来读取与键关联的值。如果键是 `x`，其值由 `validCd.get(x)` 给出；即，语法是 `dictionaryName.get(key)`。

汇编语言包含要执行的指令。然而，它还包含称为 *汇编指令* 的信息，这些信息告诉程序有关环境的一些信息；例如，数据在内存中的位置或如何将符号名称绑定到值。我们现在将探讨汇编指令。

# 处理汇编指令

在本节中，我们将学习以下内容：

+   汇编指令的作用

+   如何创建一个将符号名称与值链接的符号表

+   如何访问符号表

+   如何更新符号表

+   处理标签

我们将演示程序员选择的名称是如何被操作和转换为适当的数值的。

TC1 的第一个版本要求你为所有名称和标签提供实际值。如果你想跳转到一条指令，你必须提供要跳转的行数。允许程序员编写以下内容会更好：

`JMP next`

在这里，`next` 是目标行的标签。这比编写以下内容更受欢迎：

`JMP 21`

类似地，如果文字 `60` 代表一小时中的分钟，请编写以下内容：

`MULL ,R1,MINUTES`

这比以下内容更受欢迎：

`MULL ,R1,60`

我们需要一种方法来 *链接* `next` 与 `21` 和 `MINUTES` 与 `60`。

Python 的 *字典* 结构解决了这个问题。我们只需创建 `key:value` 对，其中 `key` 是我们想要定义的标签，`value` 是它的值。在这个例子中，前面例子中的字典将是 `{'NEXT':21, 'MINUTES':60}`。注意这个例子使用 *整数* 作为值。在这本书中，我们也将使用 *字符串* 作为值，因为我们以文本形式输入数据；例如，`'MINUTES':'60'`。

`EQU` 汇编指令将一个值与一个符号名称关联。例如，TC1 允许你编写以下内容：

`MINUTES EQU 60`

## 使用字典

`MINUTES EQU 60` 汇编指令有三个令牌：一个标签、一个函数（等价）和一个值。我们从源代码中提取 `'MINUTES':60` 字典对，并将其插入到名为 `symbolTab` 的字典中。以下代码演示了该过程。第一行设置符号表。我们用虚拟条目 `'START':0` 初始化它。我们创建了这个初始条目用于测试目的：

```py

symbolTab = {'START':0}                              # Symbol table for labels
for i in range (0,len(sFile)):                       # Deal with equates
    if len(sFile[i]) > 2 and sFile[i][1] == 'EQU':   # Is token 'EQU'?
        symbolTab[sFile[i][0]] = sFile[i][2]         # If so, update table
sFile = [i for i in sFile if i.count('EQU') == 0]    # Delete EQU from source
```

`for` 循环（阴影部分）读取源代码的每一行，`sFile`，并检查 `'EQU'` 是否是该行的第二个令牌。`len(sFile[i]) > 2` 的比较确保该行至少有三个令牌，以确保它是一个有效的等价指令。文本以粗体字体显示。

我们可以通过使用 `and` 布尔运算符同时执行两个测试，这样测试只有在两个条件都为真时才为真。

我们检查第二个令牌是否为 `'EQU'`，使用 `sFile[i][1] == 'EQU'`。`sFile[i][1]` 表示法有两个列表索引。第一个，以粗体显示，表示源代码的第 `i` 行，第二个索引表示该行的令牌 1；也就是说，它是第二个元素。

如果找到 `'EQU'`，我们将第一个令牌 `[`sFile[i][0]`]` 作为键添加到符号表中，第三个令牌 `sFile[i][2]` 作为值。

考虑以下 `MINUTES EQU 60` 源代码行。

关键是 `sFile[i][0]` 和它的值是 `sFile[i][2]`，因为 `MINUTES` 是第 `i` 行的第一个标记，而 `60` 是第 `i` 行的第三个标记。存储的键是 `'MINUTES'`，其值是 `60`。但请注意，值 `60` 是以 *字符串* 形式存在，而不是 *整数* 形式。为什么？因为汇编指令是一个字符串，而不是一个整数。如果我们想要数值，我们必须使用 `int()`。

这段代码的最后一行如下：

```py

sFile = [i for i in sFile if i.count('EQU') == 0]
```

这一行使用列表推导来扫描源文件，并删除任何包含 `EQU` 的行，因为只有指令被加载到程序内存中。包含 `EQU` 的行是一个指令，而不是指令。这个操作使用计数方法，`i.count('EQU')` 来计算 `EQU` 在一行中出现的次数，然后如果计数不是 `0`，则删除该行。我们在移动（即保留）一行之前测试的条件如下：

`if i.count('EQU') ==` `0:`

在这里，`i` 是当前正在处理的行。将 `count` 方法应用于当前行，并计算 `'EQU'` 字符串在该行中出现的次数。只有当计数是 `0`（即，它不是一个带有 `EQU` 指令的行）时，该行才会被复制到 `sFile`。

由于检测 `EQU` 指令、将其放入符号表和从代码中删除非常重要，我们将通过一小段测试代码来演示其操作。以下代码片段在 `sFile` 中设置了一个包含三个指令的列表以进行测试。请记住，`sFile` 是一个列表的列表，每个列表是一个由标记组成的指令，每个标记都是一个字符串：

```py

sFile=[['test','EQU','5'],['not','a','thing'],['xxx','EQU','88'], \
       ['ADD','r1','r2','r3']]
print('Source: ', sFile)
symbolTab = {}                                    # Creates empty symbol table
for i in range (0,len(sFile)):                    # Deal with equates e.g., PQR EQU 25
    print('sFile[i]', sFile[i])
    if len(sFile[i]) > 2 and sFile[i][1] == 'EQU':  # Is the second token 'EQU'?
        print('key/val', sFile[i][0], sFile[i][2])  # Display key-value pair
        symbolTab[sFile[i][0]] = sFile[i][2]        # Now update symbol table
sFile = [i for i in sFile if i.count('EQU') == 0]   # Delete equates from source file
print('Symbol table: ', symbolTab)
print('Processed input: ',sFile)
```

粗体的代码是我们讨论过的代码。其余的代码是由 `print` 语句组成的，用于观察代码的行为。代码中的关键行如下：

`symbolTab[sFile[i][0]] =` `sFile[i][2]`

这通过以下格式的 `key:value` 对更新符号表：

`symbolTab[key] =` `value`

当运行此代码时，它生成以下输出：

```py

Source [['test','EQU','5'],['not','a','thing'],['xxx','EQU','88'], ['ADD','r1','r2','r3']]
sFile[i] ['test', 'EQU', '5']
key/val test 5
sFile[i] ['not', 'a', 'thing']
sFile[i] ['xxx', 'EQU', '88']
key/val xxx 88
sFile[i] ['ADD', 'r1', 'r2', 'r3']
Symbol table {'test': '5', 'xxx': '88'}
Processed input [['not', 'a', 'thing'], ['ADD', 'r1', 'r2', 'r3']]
```

最后两行给出了符号表和 `sFile` 的后处理版本。两个等式已经被加载到字典（符号表）中，并且处理后的输出已经去除了这两个等式。

向字典中添加新的 `key:value` 对的方法有很多。我们本可以应用 `update` 方法到 `symbolTab` 并编写以下内容：

```py

symbolTab.update({[sFile[i][0]]:sFile[i][2]})
```

在汇编器的后续示例中，我们将采用不同的汇编指令约定，并使用 `.`equ `name value` 的格式，因为这种约定被 ARM 处理器采用，正如我们将在后面的章节中看到的。表示汇编指令的方法通常不止一种，每种方法都有其自身的优缺点（例如，编码的简便性或与特定标准和约定相匹配）。

## 标签

处理源文件的下一步是处理标签。以下是一个示例：

```py

      DEC  r1                                   @ Decrement r1
      BEQ  NEXT1                                @ If result in r1 is 0, then jump to line NEXT1
      INC  r2                                   @ If result not 0, increment r2
      .
NEXT1 .
```

在这个例子中，递减操作从寄存器 `r1` 的内容中减去 `1`。如果结果是 `0`，则设置 `Z flag`。下一条指令是 *零分支到 NEXT1*。如果 *Z = 1*，则跳转到标签为 `NEXT1` 的行；否则，执行紧随 `BEQ` 后面的 `INC r2` 指令。

由 TC1 生成的二进制程序（机器代码）不存储或使用标签。它需要下一个指令的实际地址或其相对地址（即，它需要从当前位置跳转多远）。换句话说，我们需要将 `NEXT1` 标签转换为程序中的实际地址。

这是字典的工作。我们只需将标签作为键放入字典，然后将相应的地址作为与键关联的值插入。以下三行 Python 代码演示了如何收集标签地址并将它们放入符号表中：

```py

1. for i in range(0,len(sFile)):                  # Add branch addresses to symbol tab
2.     if sFile[i][0] not in codes:               # If first token not an opcode, it's a label
3.        symbolTab.update({sFile[i][0]:str(i)})  # Add pc value, i to sym tab as string
4. print('\nEquate and branch table\n')           # Display symbol table
5. for x,y in symbolTab.items():                  # Step through symbol table
6.     print('{:<8}'.format(x),y)
```

这三条线，1 到 3，定义了一个 `for` 循环，遍历 `sFile` 中的每一行源代码。因为我们已经处理了代码，将每条指令转换成令牌列表，所以每一行以有效的助记符或标签开始。我们只需检查一行上的第一个令牌是否在助记符列表（或字典）中。如果第一个令牌在列表中，则它是一条指令。如果不在列表中，则它是一个标签（我们忽略它是错误的情况）。

我们使用以下方式来检查有效的助记符：

```py

2\. if sFile[i][0] not in codes:
```

在这里，`sFile[i][0]` 代表字典中第 `i` 行的第一个项目（即令牌）。Python 中的 `not in` 代码返回 `True` 如果助记符不在名为 `codes` 的字典中。如果测试返回 `True`，则我们有一个标签，必须使用以下操作将其放入符号表中：

```py

3\. symbolTab.update({sFile[i][0]:str(i)})                  # i is the pc value
```

这个表达式表示，“*将指定的* `key:value` *对添加到名为* `symbolTable`* 的字典中*。”为什么与标签关联的值给定为 `i`？与标签关联的值是该行的地址（即，程序计数器 `pc` 在执行该行时的值）。由于我们是逐行遍历源代码，计数器 `i` 是程序计数器的对应值。

`update` 方法应用于符号表，其中 sFile[i][0] 作为键，`str(i)` 作为值。键是 sFile[i][0]，即标签（即字符串）。然而，`i` 的 *值* 不是一个 *字符串*。这个值是一个 *整数*，`i`，它是当前行地址。我们通过 str(i) 将整数地址转换为字符串，因为等式以字符串的形式存储在表中（即，这是由我做出的设计决策）。

下两行打印符号表：

```py

4\. print('\nEquate and branch table\n')                    # Display symbol table
5\. for x,y in symbolTab.items(): print('{:<8}'.format(x),y) # Step through symbol table
```

使用 `for` 循环打印符号表中的值。我们使用以下方式提取 `key:value` 对：

```py

5\. for x,y in symbolTab.items():
```

`items()` 方法遍历 `symbolTab` 字典的所有元素，并允许我们打印每个 `key:pair` 值（即，所有名称/标签及其值）。`print` 语句使用 `{:<8}.format(x)` 格式化 `x` 的值，以显示八个字符，右对齐。

解码指令后，我们接下来必须将其转换为适当的二进制代码。

# 构建二进制指令

汇编过程的下一步是为每个指令生成适当的二进制模式。在本节中，我们展示了如何将指令的各个组件组合起来，以创建可以由计算机稍后执行的二进制值。

注意，本节中的代码描述了分析指令时涉及的一些指令处理。实际的模拟器在细节上有所不同，尽管原理是相同的。

我们首先必须提取助记符，将其转换为二进制，然后提取寄存器编号（如果适用），最后插入 16 位字面量。此外，因为汇编器是文本形式，我们必须能够处理符号字面量（即，它们是名称而不是数字）、十进制、负数、二进制或十六进制；也就是说，我们必须处理以下形式的指令：

```py

LDRL r0,24                   @ Decimal numeric value
LDRL r0,0xF2C3               @ Hexadecimal numeric value
LDRL r0,$F2C3                @ Hexadecimal numeric value (alternative representation)
LDRL r0,%00110101            @ Binary numeric value
LDRL r0,0b00110101           @ Binary numeric value (alternative representation)
LDRL r0,-234                 @ Negative decimal numeric value
LDRL r0,ALAN2                @ Symbolic value requiring symbol table look-up
```

汇编器查看源代码的每一行，并提取助记符。指令是一系列标记（例如，`'NEXT'`、`'ADD'`、`'r1'`、`'r2'`、`'0x12FA'`，这是五个标记，或者 `'STOP'`，这是一个标记）。情况变得更加复杂，因为助记符可能是第一个标记，或者如果指令有标签，则是第二个标记。在以下示例中，`sFile` 包含程序作为指令列表，我们正在处理第 `i` 行，`sFile[i]`。我们的解决方案如下：

1.  读取第一个标记，`sFile[i][0]`。如果此标记在代码列表中，则它是一个指令。如果不在此代码列表中，则它是一个标签，第二个标记 `sFile[i][1]` 是指令。

1.  获取指令详细信息。这些信息存储在一个名为 `codes` 的字典中。如果助记符在字典中，则键返回一个包含两个组件的元组。第一个组件是指令的格式，它定义了所需的操作数序列 `rD`, `rS1`, `rS2`, `literal`；例如，代码 `1001` 表示一个具有目标寄存器和字面量的指令。元组的第二个组件是操作码的值。我们使用十进制值（理想情况下，为了可读性，它应该是二进制，但二进制值太长，使得文本难以阅读）。

1.  从指令中的标记读取寄存器编号；例如，`ADD` `r3`,`r2`,`r7` 将返回 `3`,`2`,`7`，而 `NOP` 将返回 `0`,`0`,`0`（如果寄存器字段未使用，则将其设置为 `0`）。

1.  读取任何字面量并将其转换为 16 位整数。这是最复杂的操作，因为字面量可能有之前描述的七种不同格式之一。

1.  本节中的讨论指的是在章节末尾完整呈现的 TC1 程序。在这里，我们展示该程序的部分内容，并解释它们是如何工作的以及汇编过程中的步骤。

## 提取指令及其参数

以下代码片段展示了扫描源代码并创建二进制值的循环的开始。此代码初始化变量，提取操作码作为助记符，提取任何标签，提取助记符所需的参数，并查找操作码及其格式：

```py

for i in range(0,len(sFile)):                     # Assembly loop reads instruction
    opCode,label,literal,predicate = [],[],[],[]  # Initialize opcode, label, literal, predicate
    rD, rS1, rS2  = 0, 0, 0                       # Clear register-select fields to zeros
    if sFile[i][0] in codes: opCode = sFile[i][0]   # If first token is a valid opcode, get it
    else:                    opCode = sFile[i][1]   # If not, then opcode is second token
    if (sFile[i][0] in codes) and (len(sFile[i]) > 1): # If opcode valid and length > 1
        predicate = sFile[i][1:]
    else:
        if len(sFile[i]) > 2: predicate = sFile[i][2:] \
                                              # Lines with a label longer than 2 tokens
    form = codes.get(opCode)                  # Use mnemonic to read instruction format
    if form[0] & 0b1000 == 0b1000:            # Bit 4 of format selects destination register rD
    if predicate[0] in symbolTab:                 # If first token in symbol tab, it's a label
            rD = int(symbolTab[predicate[0]][1:]) # If it is a label, then get its value
```

循环中的第 2 和 3 行声明并初始化变量，并提供了默认值。

第 4 行的第一个`if…else`语句检查源代码第`i`行的第一个令牌`sFile[i][0]`。如果该令牌在`codes`字典中，则`sFile[i][0]`是操作码。如果不在此字典中，则该令牌必须是标签，第二个令牌是操作码（第 4 和 5 行）：

```py

4.    if sFile[i][0] in codes: opcode = sFile[i][0] # If first token is a valid opcode, get it
5.    else:                    opCode = sFile[i][1] # If not, then it's the second token
```

如果我们遇到标签，我们可以将其转换为其实际地址，该地址在`symbolTab`中，使用以下方法：

```py

    if sFile[i][0] in symbolTab: label = sFile[i][0] # Get label
```

第 6、7、8 和 9 行从汇编语言中提取谓词。记住，谓词由助记符之后的令牌组成，包括指令所需的任何寄存器和字面量：

```py

6\. if (sFile[i][0] in codes) and (len(sFile[i])>1): # Get everything after opcode
7.                        predicate = sFile[i][1:]  # Line with opcode
8\. else:
9.    if len(sFile[i])>2: predicate = sFile[i][2:]  # If label and len > 2 tokens
```

我们必须处理两种情况：第一个令牌是助记符，第二个令牌也是助记符。我们还检查该行是否足够长以包含谓词。如果有谓词，它将通过第 7 和第 9 行被提取：

```py

7.          predicate = sFile[i][1:]     # The predicate is the second and following tokens
9.          predicate = sFile[i][2:]     # The predicate is the third and following tokens
```

符号`[2:]`表示从令牌 2 到行尾的所有内容。这是 Python 的一个非常好的特性，因为它不需要你明确地声明行的长度。一旦我们提取了包含寄存器和字面量信息的谓词，我们就可以开始汇编指令。

接下来，我们提取当前行的代码格式以获取从谓词中所需的信息。第 10 行，形式`= codes.get(opCode)`，访问`codes`字典以查找`opCode`变量中的助记符。`get`方法应用于`codes`，`form`变量接收键值，即（格式，代码）元组，例如（8，10）。`form[0]`变量是指令格式，`form[1]`是操作码：

```py

10\. form = codes.get(opCode)                    # Use opcode to read instruction format
11\. if form[0] & 0b1000 == 0b1000:              # Bit 3 of format selects destination reg rD
12.     if predicate[0] in symbolTab:           # Check whether first token is symbol table
13.        rD =int(symbolTab[predicate[0]][1:]) # If it's a label, then get its value
```

元组的第二个元素 `form[1]` 提供了 7 位指令码；即 `0100010` 对于 `LDRL`。行 `10` 到 `13` 展示了如何提取目标寄存器。我们首先使用 `AND` `form[0]` 与 `0b1000` 来测试最高有效位，该位指示是否需要此指令的目标寄存器 `rD`。如果需要，我们首先测试寄存器是否以 `R0` 的形式表示，或者是否以名称给出，例如 `TIME`。我们必须这样做，因为 TC1 允许你使用 `EQU` 指令重命名寄存器。

你可以使用 `dictionary` 来检查一个项目是否在字典中。以下是一个例子：

if `'INC'` in opCodes:

要获取有关特定助记符的信息，我们可以使用 `get` 方法读取与键 `–` 关联的值，例如，`format =` `opCodes.get('INC')`。

前面的例子返回 `format = (8,82)`。`8` 指的是格式代码 `0b1000`（指定目标寄存器）。`82` 是该指令的指令码。我们可以使用以下方式访问与 `'INC'` 关联的值的两个字段：

```py

binaryCode  = format[0]
formatStyle = format[1]
```

我们首先在行 `12` 中测试寄存器是否有符号名称。`if predicate[0]` in `symbolTab:`，如果它在符号表中，我们在行 `13` 中读取它的值。

`rD =` `int(symbolTab[predicate[0]][1:])`

我们使用键来查询符号表，这个键是谓词的第一个元素，因为在 TC1 汇编语言指令中（例如在 `ADD` r4,`r7`,`r2` 中），目标寄存器总是第一个。寄存器由 `predicate[0]` 提供。`symbolTab[predicate[0]]` 表达式查找符号名称并提供其值；例如，考虑 `TIME EQU R3`。`INC TIME` 汇编语言指令将查找 `TIME` 并返回 `R3`。我们现在有了目标操作数，但它是一个字符串，`'R3'`，而不是一个数字。我们只需要 `3`，并必须使用 `int` 函数将字符串格式的数字转换为整数值。

让我们简化 Python 表达式，以便更容易解释。假设我们写下以下内容：

`destReg =` `symbolTab`[predicate[0]]

`destReg` 的值是表示目标寄存器的字符串。假设这是 `'R3'`。我们需要做的是从 `'R3'` 中隔离 `'3'`，然后将字符 `'3'` 转换为整数 `3`。我们可以写 `destRegNum = destReg[1:]` 来返回字符串中除了初始 `'R'` 之外的所有字符。最后一步是将它转换为整数，我们可以用 `rD =` `int(destRegNum)` 来完成。

记住，`[1:]` 表示第一个字符 `'R'` 之后的所有字符。因此，如果寄存器是 `'R3'`，则返回 `'3'`。我们本来可以写 `[1:2]` 而不是 `[1:]`，因为数字在 1 到 7 的范围内。然而，通过使用 `[1:]` 符号，我们可以在不改变程序的情况下，以后增加超过 9 个寄存器的数量。

将这三个步骤结合起来，我们得到`rD = int(symbolTab[predicate[0]][1:])`。

以下 Python 代码显示了整个解码过程：

```py

form = codes.get(opCode)                      # Use opcode to read type of instruction
if form[0] & 0b1000 == 0b1000:                # Bit 4 of format selects destination register rD
    if predicate[0] in symbolTab:             # Check whether first token is sym tab
          rD = int(symbolTab[predicate[0]][1:])  # If it is, then get its value
    else: rD = int(predicate[0][1:])          # If it's not a label, get from the predicate
if form[0] & 0b0100 == 0b0100:                # Bit 3 selects register source register 1, rS1
    if predicate[1] in symbolTab:
          rS1 = int(symbolTab[predicate[1]][1:])
    else: rS1 = int(predicate[1][1:])
if form[0] & 0b0010 == 0b0010:                # Bit 2 of format selects register rS1
    if predicate[2] in symbolTab:
          rS2 = int(symbolTab[predicate[2]][1:])
    else: rS2 = int(predicate[2][1:])
if form[0] & 0b0001 == 0b0001:                # Bit 1 of format indicates a literal
    if predicate[-1] in symbolTab:            # If literal in symbol table, get it
        predicate[-1] = symbolTab[predicate[-1]]
    elif type(predicate[-1]) == 'int':                                # Integer
        literal = str(literal)
    elif predicate[-1][0]    == '%':                                  # Binary
        literal=int(predicate[-1][1:],2) 
    elif predicate[-1][0:2]  == '0B':                                 # Binary 
        literal=int(predicate[-1][2:],2)
    elif predicate[-1][0:1]  == '$':                                  # Hex
        literal=int(predicate[-1][1:],16)
    elif predicate[-1][0:2]  == '0X':                                 # Hex
        literal=int(predicate[-1][2:],16)
    elif predicate[-1].isnumeric():                                   # Decimal
        literal=int(predicate[-1])
    elif predicate[-1][0]    == '-':                                  # Negative
        literal=(-int(predicate[-1][1:]))&0xFFFF
    else:  literal = 0                                                # Default
```

这段代码块执行相同的操作序列三次，以相同的方式处理`rD`，然后是`rS1`（第一个源寄存器），然后是`rS2`（第二个源寄存器）。这个代码块的最后部分（阴影部分）更复杂，因为我们允许字面量的多种表示。我们使用`if…elif`结构来测试符号字面量、二进制字面量、十六进制字面量、无符号十进制数值字面量，最后是负十进制数值字面量。

字面量是指令使用的数值常数。然而，在汇编语言中，字面量由一个文本字符串表示；也就是说，如果字面量是`12`，它就是字符串`'12'`而不是数值`12`。它必须通过`int()`函数转换成数值形式。

我们最初决定允许十进制、二进制或十六进制整数。后来，我们包括了符号名称，因为它们使用 Python 的字典处理起来非常容易。假设我们有一个指令，它已经被分解成一个助记符和一个包含寄存器和字面量或符号名称的谓词，例如，`['R1', 'R2', 'myData']`。考虑以下代码：

```py

if predicate[-1] in symbolTab:                 # If literal is in symbol table, look up value
   predicate[-1] = symbolTab[predicate[-1]]    # Get its value from the symbol table
```

这取谓词的最后一个元素（由`[-1]`索引指示）并查看它是否在符号表中。如果没有，代码将测试其他类型的字面量。如果在符号表中，它将被提取出来，并且`myData`符号名称将被其实际值替换。

表中的字面量可能是一个整数或字符串。以下代码将其转换为字符串（如果它是字面量的话）：

```py

if type(predicate[-1])=='int': literal=str(literal) # Integer to string
```

`if`结构使用`type()`函数，该函数返回对象的类型。在这种情况下，如果对象是整数，它将是`'int'`。`str()`函数将整数对象转换为字符串对象。

这个操作看起来可能很奇怪，因为我们正在将一个整数（我们想要的）转换成一个字符串（我们不想要的）。这种异常的原因是我们稍后将要测试十六进制、二进制和有符号值，这些值将是字符串，并且将所有字面量都作为字符串来处理可以简化编码。

以下代码演示了如何将三种数字格式转换为整数形式，以便将其打包到最终的 32 位 TC1 机器指令中：

```py

if   predicate[-1][0]   == '%':  literal = int(predicate[-1][1:],2)
elif predicate[-1][0:2  == '0B': literal = int(predicate[-1][2:],2)
elif predicate[-1][0:1] == '$':  literal = int(predicate[-1][1:],16)
elif predicate[-1][0:2] == '0X': literal = int(predicate[-1][2:],16)
elif predicate[-1].isnumeric():  literal = int(predicate[-1])
```

在 TC1 汇编语言中，二进制数以`%`或`0b`为前缀，十六进制值以`$`或`0x`为前缀。常数被测试以确定它是十进制、二进制还是十六进制，然后执行适当的转换。将二进制字符串`x`转换为整数`y`使用`y = int(x,2)`完成。粗体参数是数字基数。在这种情况下，它是二进制格式的`2`。在十六进制格式中，它是`16`。

让我们看看十六进制转换。我们必须进行两个选择：首先是标记，然后是标记的具体字符。考虑`ADDL R1,R2,0XF2A4`。谓词是`'R1 R2 0XF2A4'`，它被标记为`predicate = ['R1', '``R2', '0XF2A4']`。

`predicate[-1]`的值是`'0XF2A4'`。为了测试十六进制值，我们必须查看前两个字符，看它们是否是`'0X'`。注意`0X`不是`0x`，因为 TC1 将输入转换为大写。我们可以编写以下代码：

```py

lastToken = predicate[-1]              # Get the last token from the predicate
prefix = lastToken[0:2]                # Get the first two characters of this token to test for '0X'
```

我们可以通过合并两个列表索引后缀`[-1]`和`[0:2]`来节省一行，即`predicate[-1][0:2]`。

代码的第三行`elif predicate[-1].isnumeric(): literal=int(predicate[-1])`检测十进制字符串并将它们转换为数值形式。由于十进制值没有前缀，我们使用`isnumeric`方法来测试具有数值的字符串。这一行读作，“*如果谓词中的最后一个标记是数值，则将其转换为* *整数值*。”

最后，我们必须处理负数（例如，-5）。如果一个字面量以-开头，则读取剩余的字符串并将其转换为 16 位二进制补码形式。这是必要的，因为 TC1 计算机以 16 位二进制补码形式表示有符号整数。

生成指令的最终 32 位二进制代码很容易。我们有一个操作码和零到四个字段可以插入。字段最初设置为全零（默认值）。然后，每个字段向左移动到指令中所需的位置，并通过位运算`OR`将其插入到指令中。相应的代码如下：

```py

s2      = s2      << 16                # Shift source 2 16 places left
s1      = s1      << 19                # Shift source 1 19 places left
destReg = destReg << 22                # Shift destination register 22 places left
op      = op      << 25                # Shift opcode 25 places left
binCode = lit | s2 | s1 | destReg | op # Logical OR the fields
```

我们可以一行完成所有这些，如下所示：

```py

binCode = lit | (s2 << 16) | (s1 << 19) | (destReg << 22)| (op << 25)
```

在下一章中，我们将回到 TC1 模拟器并对其进行扩展。我们还将展示如何通过向指令集添加新操作和打印模拟器结果的一些方法来扩展 TC1 模拟器。

在展示完整的 TC1 之前，我们将演示一个简化的版本，它可以执行汇编语言程序，本质上与 TC1 相同。然而，这个版本已被设计为通过省略诸如符号名称或指定常量时使用不同数基的能力等特性来降低总复杂性。在这种情况下，所有字面量都是简单的十进制整数。

# 休息：TC1 之前

为了提供一个更完整的 CPU 模拟器操作概述，我们将引入一个高度简化的但完整的版本，以便在我们创建更复杂的系统之前，给你一个了解事物如何组合的概念。

在本节中，你将学习如何设计一个模拟器，它不包含与完整设计相关的某些复杂性。

这个版本的 TC1，称为 TC1mini，可以执行汇编语言。然而，我们使用固定的汇编级指令格式（输入区分大小写）和固定的字面量格式（不支持十六进制或二进制数字），并且我们不支持标签和符号名称。这种方法有助于防止细节干扰大局。

## 模拟器

模拟器支持寄存器到寄存器的操作，例如 `ADD r1,r2,r3`。它的唯一内存访问是基于指针的，即 `LDRI r1,[r2]` 和 `STRI r1,[r2]`。它提供递增和递减指令，`INC r1` 和 `INC r2`。有两种比较操作：`CMPI r1,5` 和 `CMP r1,r2`（前者比较寄存器与字面量，后者比较两个寄存器）。为了保持简单，唯一的状态标志是 `z`（零），并且这个标志仅用于比较和减法操作。

提供了三种分支指令（无条件分支、零分支和非零分支）。由于这个模拟器不支持符号名称，分支需要一个字面量来指示目标。分支相对于分支指令的当前位置；例如，`BRA 3` 表示跳转到三个位置之后的指令，而 `BRA -2` 表示跳转回两个指令之前。

我没有提供基于文件的程序输入机制（即，将源程序作为文本文件读取）。要执行的汇编语言程序作为名为 `sFile` 的 Python 字符串列表嵌入。你可以轻松修改它或替换代码以输入文件。

指令集被设置为以下形式的字典：

```py

codes = {'STOP':[0], 'LDRL':[3], 'STRL':[7]}
```

`key:value` 对使用助记符作为键，并使用一个包含一个项目的列表作为值，该列表表示指令的类。类从 `0`（没有操作数的助记符）到 `7`（具有寄存器和寄存器间接操作数的助记符）。我们没有实现 TC1 的 4 位格式代码，该代码用于确定指令所需的参数，因为该信息在类中是隐含的。此外，我们不会将指令汇编成二进制代码。我们以文本形式读取助记符，并直接执行它。

当读取指令时，它首先被标记化以创建一个包含一到四个标记的列表，例如，`['CMPL', 'r3', '5']`。当从源读取指令时，确定类并用于从标记中提取所需信息。

一旦知道了助记符、寄存器编号/值和字面量，就使用简单的 `if .. elif` 结构来选择适当的指令，然后执行它。大多数指令都在 Python 的单行中解释。

在指令读取和执行循环结束时，你会被邀请按下一个键来按顺序执行下一个指令。每条指令之后显示的数据是程序计数器、z 位、指令、寄存器和内存位置。我们只使用四个寄存器和八个内存位置。

我们将这个程序分成几个部分，每个部分之间都有简短的描述。第一部分提供源代码作为内置列表。它定义了指令类并提供了一组操作码及其类。我们为此不使用字典。然而，我们为寄存器及其间接版本提供了字典，以简化指令分析。例如，我们可以在`LDRI r1,[r2]`指令中查找`r1`和`r2`：

```py

sFile = ['LDRL r2,1','LDRL r0,4','NOP','STRI r0,[r2]','LDRI r3,[r2]',   \
         'INC r3','ADDL r3,r3,2','NOP','DEC r3', 'BNE -2','DEC r3','STOP']
                                            # Source program for testing
# Simple CPU instruction interpreter. Direct instruction interpretation. 30 September 2022\. V1.0
# Class 0: no operand                   NOP
# Class 1: literal                      BEQ  3
# Class 2: register                     INC  r1
# Class 3: register,literal             LDRL r1,5
# Class 4: register,register,           MOV  r1,r2
# Class 5: register,register,literal    ADDL r1,r2,5
# Class 6: register,register,register   ADD  r1,r2,r3
# Class 7: register,[register]          LDRI r1,[r2]
codes = {'NOP':[0],'STOP':[0],'BEQ':[1],'BNE':[1],'BRA':[1],  \
         'INC':[2],'DEC':[2],'CMPL':[3],'LDRL':[3],'MOV':[4],  \
         'CMP':[4],'SUBL':[5],'ADDL':[5],'ANDL':[5],'ADD':[6], \
         'SUB':[6], 'AND':[6],'LDRI':[7],'STRI':[7]}
reg1  = {'r0':0,'r1':1,'r2':2,'r3':3}       # Legal registers
reg2  = {'[r0]':0,'[r1]':1,'[r2]':2,'[r3]':3} # Legal pointer registers
r = [0] * 4                                 # Four registers
r[0],r[1],r[2],r[3] = 1,2,3,4               # Preset registers for testing
m  = [0] * 8                                # Eight memory locations
pc = 0                                      # Program counter initialize to 0
go = 1                                      # go is the run control (1 to run)
z  = 0                                      # z is the zero flag. Set/cleared by SUB, DEC, CMP
while go == 1:                              # Repeat execute fetch and execute loop
    thisLine = sFile[pc]                    # Get current instruction
    pc = pc + 1                             # Increment pc
    pcOld = pc                              # Remember pc value for this cycle
    temp = thisLine.replace(',',' ')        # Remove commas: ADD r1,r2,r3 to ADD r1 r2 r3
```

```py
    tokens = temp.split(' ')                # Tokenize:  ADD r1 r2 r3 to ['ADD','r1','r2','r3']
```

在下一节中，我们分析一条指令以提取指令所需的操作数值。这是通过查看指令的 op-class 然后提取适当的信息（例如，寄存器号）来实现的：

```py

    mnemonic = tokens[0]                  # Extract first token, the mnemonic
    opClass = codes[mnemonic][0]          # Extract instruction class
                                          # Process the current instruction and analyze it
    rD,rDval,rS1,rS1val,rS2,rS2val,lit, rPnt,rPntV = 0,0,0,0,0,0,0,0,0 
                                          # Clear all parameters
    if opClass in [0]: pass               # If class 0, nothing to be done (simple opcode only)
    if opClass in [2,3,4,5,6,7,8]:        # Look for ops with destination register rD
        rD     = reg1[tokens[1]]          # Get token 1 and use it to get register number as rD
        rDval  = r[rD]                    # Get contents of register rD
    if opClass in [4,5,6]:                # Look at instructions with first source register rS1
        rS1    = reg1[tokens[2]]          # Get rS1 register number and then contents
        rS1val = r[rS1]
    if opClass in [6]:                    # If class 6, it's got three registers. Extract rS2
        rS2    = reg1[tokens[3]]          # Get rS2 and rS2val
        rS2val = r[rS2]
    if opClass in [1,3,5,8]:              # The literal is the last element in instructions
        lit    = int(tokens[-1])          # Get the literal
    if opClass in [7]:                    # Class 7 involves register indirect addressing
        rPnt   = reg2[tokens[2]]          # Get the pointer (register) and value of the pointer
        rPntV  = r[rPnt]                  # Get the register number
    if mnemonic == 'STOP':                # Now execute instructions. If STOP, clear go and exit
        go = 0
        print('Program terminated')
```

这是程序中的指令执行部分。我们使用一系列比较助记符与操作码，然后直接执行指令。与 TC1 不同，我们不将助记符转换为二进制代码，然后再通过将二进制代码转换为适当的操作来执行它：

```py

    elif mnemonic == 'NOP':  pass         # NOP does nothing. Just drop to end of loop
    elif mnemonic == 'INC': r[rD] = rDval + 1  # Increment: add 1 to destination register
    elif mnemonic == 'DEC':               # Decrement: subtract 1 from register and update z bit
        z = 0
        r[rD] = rDval - 1
        if r[rD] == 0: z = 1
    elif mnemonic == 'BRA':               # Unconditional branch
        pc = pc + lit - 1
    elif mnemonic == 'BEQ':               # Conditional branch on zero
        if z == 1: pc = pc + lit - 1
    elif mnemonic == 'BNE':               # Conditional branch on not zero
        if z == 0: pc = pc + lit - 1
    elif mnemonic == 'ADD': r[rD]=rS1val+rS2val # Add
    elif mnemonic == 'ADDL': r[rD] = rS1val+lit # Add literal
    elif mnemonic == 'SUB':                     # Subtract and set/clear z
        r[rD] = rS1val - rS2val
        z = 0
        if r[rD] == 0: z = 1
    elif mnemonic == 'SUBL':                    # Subtract literal
        r[rD] = rS1val - lit
        z = 0
        if r[rD] == 0: z = 1
    elif mnemonic == 'CMPL':                    # Compare literal
        diff = rDval - lit
        z = 0
        if diff == 0 : z = 1
    elif mnemonic == 'CMP':                     # Compare
        diff = rDval - rS1val
        z = 0
        if diff == 0: z = 1
    elif mnemonic == 'MOV':  r[rD] = rS1val     # Move, load, and store operations
    elif mnemonic == 'LDRL': r[rD] = lit
    elif mnemonic == 'LDRI': r[rD] = m[rPntV]
    elif mnemonic == 'STRI': m[rPntV] = rDval
    regs = ' '.join('%02x' % b for b in r)      # Format memory locations hex
    mem  = ' '.join('%02x' % b for b in m)      # Format registers hex
    print('pc =','{:<3}'.format(pcOld), '{:<14}'.format(thisLine), \
          'Regs =',regs, 'Mem =',mem, 'z =', z)
    x = input('>>> ')               # Request keyboard input before dealing with next instruction
```

注意，执行循环以从键盘请求输入结束。这样，在按下*Enter*/*Return*键之前，下一个循环不会执行。

以下展示了模拟器在执行嵌入式程序时的输出。更改的寄存器、内存位置和标志值以粗体显示：

```py

pc = 1   LDRL r2,1     Regs = 01 02 01 04 Mem = 00 00 00 00 00 00 00 00 z = 0
pc = 2   LDRL r0,4     Regs = 04 02 01 04 Mem = 00 00 00 00 00 00 00 00 z = 0
pc = 3   NOP           Regs = 04 02 01 04 Mem = 00 00 00 00 00 00 00 00 z = 0
pc = 4   STRI r0,[r2]  Regs = 04 02 01 04 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 5   LDRI r3,[r2]  Regs = 04 02 01 04 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 6   INC r3        Regs = 04 02 01 05 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 7   ADDL r3,r3,2  Regs = 04 02 01 07 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 8   NOP           Regs = 04 02 01 07 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 9   DEC r3        Regs = 04 02 01 06 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 10  BNE -2        Regs = 04 02 01 06 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 8   NOP           Regs = 04 02 01 06 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 9   DEC r3        Regs = 04 02 01 05 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 10  BNE -2        Regs = 04 02 01 05 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 8   NOP           Regs = 04 02 01 05 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 9   DEC r3        Regs = 04 02 01 04 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 10  BNE -2        Regs = 04 02 01 04 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 8   NOP           Regs = 04 02 01 04 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 9   DEC r3        Regs = 04 02 01 03 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 10  BNE -2        Regs = 04 02 01 03 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 8   NOP           Regs = 04 02 01 03 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 9   DEC r3        Regs = 04 02 01 02 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 10  BNE -2        Regs = 04 02 01 02 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 8   NOP           Regs = 04 02 01 02 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 9   DEC r3        Regs = 04 02 01 01 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 10  BNE -2        Regs = 04 02 01 01 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 8   NOP           Regs = 04 02 01 01 Mem = 00 04 00 00 00 00 00 00 z = 0
pc = 9   DEC r3        Regs = 04 02 01 00 Mem = 00 04 00 00 00 00 00 00 z = 1
pc = 10  BNE -2        Regs = 04 02 01 00 Mem = 00 04 00 00 00 00 00 00 z = 1
pc = 11  DEC r3        Regs = 04 02 01 -1 Mem = 00 04 00 00 00 00 00 00 z = 0
Program terminated
pc = 12  STOP          Regs = 04 02 01 -1 Mem = 00 04 00 00 00 00 00 00 z = 0
```

我们现在将查看 TC1 模拟器的程序。在提供代码之前，我们将简要介绍其一些功能。

# TC1 模拟器程序

在本节中，我们提供了 TC1 汇编器和模拟器的完整代码。这将使您能够构建和修改一个可以执行 TC1 支持的代码或您自己的指令集（如果您修改了 TC1）的计算机汇编器和模拟器。

汇编器是更复杂的一部分，因为它涉及到读取文本、分析它并将其格式化为二进制代码。模拟器本身只是读取每个二进制代码然后执行相应的操作。

模拟器包括我们在前面的章节中尚未介绍的功能（例如，调试和跟踪功能）。在本书的第一稿中，TC1 的功能相对更基础，具有最小子集的功能。随着书籍的编辑和程序的修改，功能集得到了增强，使其成为一个更实用的工具。我们首先简要介绍一些这些功能，以帮助理解程序。

## 单步执行

计算机按顺序执行指令，除非遇到分支或子程序调用。在测试模拟器时，您经常希望一起执行一批指令（即，不打印寄存器值），或者您可能希望在每条指令执行后按*Enter*/*Return*键逐条执行指令，或者执行指令直到您按下特定的指令。

在 TC1 的这个版本中，你可以执行并显示一条指令，跳过显示接下来的 n 条指令，或者直到遇到改变流程的指令才显示指令。程序加载后，会显示输入提示。如果你输入回车，模拟器将执行下一条指令并等待。如果你输入一个整数（然后回车），模拟器将执行指定数量的指令而不显示结果。如果你输入 b 然后回车，模拟器将执行指令而不显示它们，直到遇到下一个分支指令。

考虑以下示例。代码只是一组用于演示的随机指令。我使用了无操作(`nop`)作为填充。我还测试了字面地址格式（十六进制和二进制），并展示了不区分大小写：

```py

@ test trace modes
    nop
    nop
    inc r1
    NOP
    dec r2
    ldrl r6,0b10101010
    bra abc
    nop
    inc R7
    nop
abc ldrl r3,$ABCD
    nop
    inc r3
    INC r4
    nop
    nop
    inc r5
    END!
```

我已经编辑了它，移除了未访问的内存位置。在提示符`>>>`之后，你选择要执行的操作：跟踪一条指令，执行`n`条指令而不停止或显示寄存器，或者执行代码到下一个分支指令而不显示它。在每种情况下，以下输出中都会突出显示程序计数器的值。粗体的文本是我留在当前行操作上的注释（跟踪表示按下了*Return*/*Enter*键，这将执行下一条指令）：

```py

  >>>  trace
0      NOP           PC= 0 z=0 n=0 c=0 R 0000 0000 0000 0000 0000 0000 0000 0000
>>>3 jump 3 instructions (silent trace)
4      DEC R2        PC= 4 z=0 n=1 c=1 R 0000 0001 ffff 0000 0000 0000 0000 0000
>>>b jump to branch (silent mode up to next branch/rts/jsr)
6      BRA ABC       PC= 6 z=0 n=1 c=1 R 0000 0001 ffff 0000 0000 0000 00aa 0000
>>>  trace Here's the sample run
10 ABC LDRL R3 $ABCD PC=10 z=0 n=1 c=1 R 0000 0001 ffff abcd 0000 0000 00aa 0000
>>>  trace
11     NOP           PC=11 z=0 n=1 c=1 R 0000 0001 ffff abcd 0000 0000 00aa 0000
>>>4 jump 4
16     INC R5        PC=16 z=0 n=0 c=1 R 0000 0001 ffff abce 0001 0001 00aa 0000
>>>  trace
17     END!          PC=17 z=0 n=0 c=1 R 0000 0001 ffff abce 0001 0001 00aa 0000
```

## 文件输入

当我们最初开始编写模拟器时，我们通过逐个输入指令的简单方式输入测试程序。这对于最简单的测试是有效的，但很快变得繁琐。后来，程序以文本文件的形式输入。当文件名较短时，例如`t.txt`，效果很好，但随着文件名的变长（例如，当我在特定目录中存储源代码时），它变得更加繁琐。

然后，我们将文件名包含在实际的 TC1 程序中。当你需要反复运行相同的程序来测试模拟器的各种功能时，这很方便。我们需要的是一种方法，在大多数时候使用我的工作程序（嵌入在模拟器中），但在需要时切换到替代程序。

一个合理的解决方案是生成一个输入提示，提示你按*Enter*键选择默认文件，或者为替代源程序提供文件名，例如，*按 Enter 键选择默认文件*或*输入文件名以选择替代源程序*。我们决定使用 Python 的异常机制来实现这一点。在计算机科学中，*异常*（也称为*软件中断*）是一种设计用来处理意外事件的机制。在 Python 中，异常处理器使用两个保留字：`try`和`exception`。

如其名所示，`try` 会让 Python 执行其后的代码块，而 `exception` 是在 `try` 块失败时执行的代码块。本质上，它意味着“如果你不能这样做，就做那件事。”`if` 和 `try` 的区别在于，`if` 返回 `True` 或 `False`，并在 `True` 时执行指定的操作，而 `try` 则尝试运行一个代码块，如果失败（即崩溃），则会调用异常。

`try` 允许你尝试打开一个文件，如果文件不存在（即避免致命错误），则提供一种退出方式。考虑以下情况：

```py

myProg = 'testException1.txt'                      # Name of default program
try:                                               # Check whether this file exists
    with open(myProg,'r') as prgN:                 # If it's there, open it and read it
        myFile = prgN.readlines()
except:                                            # Call exception if file not there
    altProg = input('Enter source file name: ')    # Request a filename
    with open(altProg,'r') as prgN:                # Open the user file
        myFile = prgN.readlines()
print('File loaded: ', myFile)
```

这段代码寻找一个名为 `testException1.txt` 的文件。如果存在（如本例所示），模拟器会运行它，我们得到以下输出：

```py

>>> %Run testTry.py
File loaded:  ['   @ Test exception file\n', ' nop\n', ' nop\n', ' inc\n', ' end!']
```

在下一个案例中，我们已删除 `testException1.txt`。现在在提示符后得到以下输出：

```py

>>> %Run testTry.py
Enter source file name: testException2.txt
File loaded:  ['   @ Test exception file TWO\n', ' dec r1\n', ' nop\n', ' inc r2\n', ' end!']
```

粗体显示的行是备选的文件名。

在 TC1 程序中，我进一步简化了事情，通过在异常中包含文件目录（因为我总是使用相同的目录）并包含文件扩展名 `.txt`。它看起来如下：

```py

prgN = 'E://ArchitectureWithPython//' + prgN + '.txt'
```

这个表达式自动提供文件名和文件类型的路径。

记住，Python 允许你使用 `+` 操作符来连接字符串。

### TC1 程序

程序的第一部分提供了指令列表及其编码。这段文本放置在两个 `'''` 标记之间，表示它不是程序的一部分。这样可以避免每行都从 `#` 开始。这种三引号称为文档字符串注释。

TC1 的第一部分是指令列表。这些是为了使程序更容易理解而提供的：

```py

### TC1 computer simulator and assembler. Version of 11 September 2022
''' This is the table of instructions for reference and is not part of the program code
00 00000  stop operation            STOP             00 00000 000 000 000 0  0000
00 00001  no operation              NOP              00 00001 000 000 000 0  0000
00 00010  get character from keyboard  GET  r0         00 00010 rrr 000 000 0  1000
00 00011  get character from keyboard  RND  r0         00 00011 rrr 000 000 L  1001
00 00100  swap bytes in register        SWAP r0         00 00100 rrr 000 000 0  1000
00 01000  print hex value in register     PRT r0          00 01000 rrr 000 000 0  1000
00 11111  terminate program         END!            00 11111 000 000 000 0  0000
01 00000  load register from register     MOVE r0,r1      01 00000 rrr aaa 000 0  1100
01 00001  load register from memory   LDRM r0,L       01 00001 rrr 000 000 L  1001
01 00010  load register with literal       LDRL r0,L       01 00010 rrr 000 000 L  1001
01 00011  load register indirect        LDRI r0,[r1,L]  01 00011 rrr aaa 000 L  1101
01 00100  store register in memory      STRM r0,L       01 00100 rrr 000 000 L  1001
01 00101  store register indirect       STRI r0,[r1,L]  01 00101 rrr aaa 000 L  1101
10 00000  add register to register      ADD  r0,r1,r2   10 00000 rrr aaa bbb 0  1110
10 00001  add literal to register        ADDL r0,r1,L    10 00001 rrr aaa 000 L  1101
10 00010  subtract register from register SUB  r0,r1,r2   10 00010 rrr aaa bbb 0  1110
10 00011  subtract literal from register    SUBL r0,r1,L    10 00011 rrr aaa 000 L  1101
10 00100  multiply register by register   MUL  r0,r1,r2   10 00100 rrr aaa bbb 0  1110
10 00101  multiply literal by register     MULL r0,r1,L    10 00101 rrr aaa 000 L  1101
10 00110  divide register by register     DIV  r0,r1,r2   10 00110 rrr aaa bbb 0  1110
10 00111  divide register by literal       DIVL r0,r1,L    10 00111 rrr aaa 000 L  1101
10 01000  mod register by register      MOD  r0,r1,r2   10 01000 rrr aaa bbb 0  1110
10 01001  mod register by literal        MODL r0,r1,L    10 01001 rrr aaa 000 L  1101
10 01010  AND register to register      AND  r0,r1,r2   10 01000 rrr aaa bbb 0  1110
10 01011  AND register to literal         ANDL r0,r1,L    10 01001 rrr aaa 000 L  1101
10 01100  OR register to register        OR   r0,r1,r2   10 01010 rrr aaa bbb 0  1110
10 01101  NOR register to literal        ORL  r0,r1,L    10 01011 rrr aaa 000 L  1101
10 01110  EOR register to register       OR   r0,r1,r2   10 01010 rrr aaa bbb 0  1110
10 01111  EOR register to literal       ORL  r0,r1,L    10 01011 rrr aaa 000 L  1101
10 10000  NOT register              NOT  r0         10 10000 rrr 000 000 0  1000
10 10010  increment register          INC  r0          10 10010 rrr 000 000 0  1000
10 10011  decrement register         DEC  r0         10 10011 rrr 000 000 0  1000
10 10100  compare register with register CMP  r0,r1      10 10100 rrr aaa 000 0  1100
10 10101  compare register with literal   CMPL r0,L       10 10101 rrr 000 000 L  1001
10 10110  add with carry            ADC              10 10110 rrr aaa bbb 0  1110
10 10111  subtract with borrow        SBC             10 10111 rrr aaa bbb 0  1110
10 11000  logical shift left           LSL  r0,L       10 10000 rrr 000 000 0  1001
10 11001  logical shift left literal       LSLL r0,L       10 10000 rrr 000 000 L  1001
10 11010  logical shift right          LSR  r0,L       10 10001 rrr 000 000 0  1001
10 11011  logical shift right literal      LSRL r0,L       10 10001 rrr 000 000 L  1001
10 11100  rotate left               ROL  r0,L        10 10010 rrr 000 000 0  1001
10 11101  rotate left literal           ROLL r0,L        10 10010 rrr 000 000 L  1001
10 11110  rotate right              ROR  r0,L        10 10010 rrr 000 000 0  1001
10 11111  rotate right literal          RORL r0,L        10 10010 rrr 000 000 L  1001
11 00000  branch unconditionally       BRA  L           11 00000 000 000 000 L  0001
11 00001  branch on zero           BEQ  L           11 00001 000 000 000 L  0001
11 00010  branch on not zero         BNE  L           11 00010 000 000 000 L  0001
11 00011  branch on minus           BMI  L           11 00011 000 000 000 L  0001
11 00100  branch to subroutine       BSR  L           11 00100 000 000 000 L  0001
11 00101  return from subroutine       RTS              11 00101 000 000 000 0  0000
11 00110  decrement & branch on not zero DBNE r0,L       11 00110 rrr 000 000 L  1001
11 00111  decrement & branch on zero  DBEQ r0,L        11 00111 rrr 000 000 L  1001
11 01000  push register on stack       PUSH r0           11 01000 rrr 000 000 0  1000
11 01001  pull register off stack       PULL r0           11 01001 rrr 000 000 0  1000
'''
import random                       # Get library for random number generator
def alu(fun,a,b):                   # Alu defines operation and a and b are inputs
   global c,n,z                     # Status flags are global and are set up here
    if   fun == 'ADD': s = a + b
    elif fun == 'SUB': s = a - b
    elif fun == 'MUL': s = a * b
    elif fun == 'DIV': s = a // b   # Floor division returns an integer result
    elif fun == 'MOD': s = a % b    # Modulus operation gives remainder: 12 % 5 = 2
    elif fun == 'AND': s = a & b    # Logic functions
    elif fun == 'OR':  s = a | b
    elif fun == 'EOR': s = a & b
    elif fun == 'NOT': s = ~a
    elif fun == 'ADC': s = a + b + c # Add with carry
    elif fun == 'SBC': s = a - b – c # Subtract with borrow
    c,n,z = 0,0,0                    # Clear flags before recalculating them
    if s & 0xFFFF == 0: z = 1        # Calculate the c, n, and z flags
    if s & 0x8000 != 0: n = 1        # Negative if most sig bit 15 is 1
    if s & 0xFFFF != 0: c = 1        # Carry set if bit 16 is 1
    return (s & 0xFFFF)              # Return the result constrained to 16 bits
```

由于左移和右移、变长移位、加移位和旋转操作相当复杂，我们提供了一个实现移位的函数。这个函数接受移位类型、方向和移动位数作为输入参数，以及要移动的单词：

```py

def shift(dir,mode,p,q):   # Shifter: performs shifts and rotates. dir = left/right, mode = logical/rotate
    global z,n,c                    # Make flag bits global. Note v-bit not implemented
    if dir == 0:                    # dir = 0 for left shift, 1 for right shift
        for i in range (0,q):       # Perform q left shifts on p
            sign = (0x8000 & p) >> 15            # Sign bit
            p = (p << 1) & 0xFFFF                # Shift p left one place
            if mode == 1:p = (p & 0xFFFE) | sign # For rotate left, add in bit shifted out
    else:                                        # dir = 1 for right shift
        for i in range (0,q):                    # Perform q right shifts
            bitOut = 0x0001 & p                  # Save lsb shifted out
            sign = (0x8000 & p) >> 15            # Get sign-bit for ASR
            p = p >> 1                           # Shift p one place right
            if mode == 1:p = (p&0x7FFF)|(bitOut<<15) # If mode = 1, insert bit rotated out
            if mode == 2:p = (p&0x7FFF)|(sign << 15) # If mode = 2, propagate sign bit
    z,c,n = 0,0,0                                # Clear all flags
    if p == 0:          z = 1                    # Set z if p is zero
    if p & 0x8000 != 0: n = 1                    # Set n-bit if p = 1
    if (dir == 0) and (sign == 1):   c = 1       # Set carry if left shift and sign 1
    if (dir == 1) and (bitOut == 1): c = 1  # Set carry bit if right shift and bit moved out = 1
    return(0xFFFF & p)               # Ensure output is 16 bits wide
def listingP():                      # Function to perform listing and formatting of source code
    global listing                   # Listing contains the formatted source code
    listing = [0]*128                # Create formatted listing file for display
    if debugLevel > 1: print('Source assembly code listing ')
    for i in range (0,len(sFile)):        # Step through the program
        if sFile[i][0] in codes:          # Is first token in opcodes (no label)?
            i2 =  (' ').join(sFile[i])    # Convert tokens into string for printing
            i1 = ''                       # Dummy string i1 represents missing label
        else:
            i2 = (' ').join(sFile[i][1:]) # If first token not opcode, it's a label
            i1 = sFile[i][0]              # i1 is the label (first token)
        listing[i] = '{:<3}'.format(i) + '{:<7}'.format(i1) + \
                     '{:<10}'.format(i2)  # Create listing table entry
        if debugLevel  > 1:               # If debug  = 1, don't print source program
            print('{:<3}'.format(i),'{:<7}'.format(i1),'{:<10}'.format(i2)) \
                                          # print: pc, label, opcode
    return()
```

这是处理字面量的函数 `getLit`。它可以处理多种可能的格式中的字面量，包括十进制、二进制、十六进制和符号形式：

```py

def getLit(litV):                                  # Extract a literal
    if  litV[0]    == '#': litV = litV[1:]         # Some systems prefix literal with '#
    if  litV in symbolTab:                         # Look in sym tab and get value if there
        literal = symbolTab[litV]                  # Read the symbol value as a string
        literal = int(literal)                     # Convert string into integer
    elif  litV[0]   == '%': literal = int(litV[1:],2)
                                                   # If first char is %, convert to integer
    elif  litV[0:2] == '0B':literal = int(litV[2:],2)
                                                   # If prefix 0B, convert binary to integer
    elif  litV[0:2] == '0X':literal = int(litV[2:],16)
                                                   # If 0x, convert hex string to integer
    elif  litV[0:1] == '$': literal = int(litV[1:],16)
                                                   # If $, convert hex string to integer
    elif  litV[0]   == '-': literal = (-int(litV[1:]))&0xFFFF 
                                                   # Convert 2's complement to int
    elif  litV.isnumeric():  literal = int(litV)
                                                   # If decimal string, convert to integer
    else:                    literal = 0           # Default value 0 (default value)
```

```py
    return(literal)
```

`Print` 语句可能有点复杂。因此，我们创建了一个 `print` 函数，用于显示寄存器和内存内容。我们将在本书的其他地方讨论打印格式。我们生成要打印的字符串数据 `m`、`m1` 和 `m2`，然后使用适当的格式打印它们：

```py

def printStatus():                             # Display machine status (registers, memory)
    text = '{:<27}'.format(listing[pcOld])     # Format instruction for listing
    m = mem[0:8]                               # Get the first 8 memory locations
    m1 = ' '.join('%04x' % b for b in m)       # Format memory location's hex
    m2 = ' '.join('%04x' % b for b in r)       # Format register's hex
    print(text, 'PC =', '{:>2}'.format(pcOld) , 'z =',z,'n =',n,'c =',c, m1,\
    'Registers ', m2)
    if debugLevel == 5:
        print('Stack =', ' '.join('%04x' % b for b in stack), \
        'Stack pointer =', sp)
    return()
print('TC1 CPU simulator 11 September 2022 ')  # Print the opening banner
debugLevel = input('Input debug level 1 - 5: ') # Ask for debugging level
if debugLevel.isnumeric():                     # If debug level is an integer, get it
    debugLevel = int(debugLevel)               # Convert text to integer
else: debugLevel = 1                           # Else, set default value to level 1
if debugLevel not in range (1,6): debugLevel = 1 # Ensure range 1 to 5
print()                                        # New line
```

上述代码块提供了调试功能，旨在演示调试的概念，并提供在汇编阶段显示中间信息以检查汇编过程的功能。在程序开始时，从键盘读取一个变量 `debugLevel`。这决定了调试功能级别，从 1（无）到 5（最大）。调试信息可以包括源代码、解码操作和其他参数：

```py

global c,n,z                                   # Processor flags (global variables)
symbolTab = {'START':0}             # Create symbol table for labels + equates with dummy entry
c,n,z = 0,0,0                                  # Initialize flags: carry, negative, zero
sFile = ['']* 128                              # sFile holds the source text
memP  = [0] * 128                              # Create program memory of 128 locations
mem   = [0] * 128                              # Create data memory of 128 locations
stack = [0] * 16                               # Create a stack for return addresses
# codes is a dictionary of instructions {'mnemonic':(x.y)} where x is the instruction operand format, and y the opcode
codes = {                                                            \
        'STOP':(0,0),  'NOP' :(0,1),  'GET' :(8,2),  'RND' : (9,3),  \
        'SWAP':(8,4),  'SEC' :(0,5),  'PRT' :(8,8),  'END!':(0,31),  \
        'MOVE':(12,32),'LDRM':(9,33), 'LDRL':(9,34), 'LDRI':(13,35), \
        'STRM':(9,36), 'STRI':(13,37),'ADD' :(14,64),'ADDL':(13,65), \
        'SUB' :(14,66),'SUBL':(13,67),'MUL' :(14,68),'MULL':(13,69), \
        'DIV' :(14,70),'DIVL':(13,71),'MOD' :(14,72),'MODL':(13,73), \
        'AND' :(14,74),'ANDL':(13,75),'OR'  :(14,76),'ORL' :(13,77), \
        'EOR' :(14,78),'EORL':(13,79),'NOT' :(8,80), 'INC' :(8,82),  \
        'DEC' :(8,83), 'CMP' :(12,84),'CMPL':(9,85), 'LSL' :(12,88), \
        'LSLL':(9,89), 'LSR' :(12,90),'LSRL':(9,91), 'ROL' :(12,92), \
        'ROLL':(9,93), 'ROR' :(12,94),'RORL':(9,95), 'ADC':(14,102), \
        'SBC':(14,103),'BRA' :(1,96), 'BEQ' :(1,97), 'BNE' :(1,98),  \
        'BMI' :(1,99), 'BSR' :(1,100),'RTS' :(0,101),'DBNE':(9,102), \
        'DBEQ':(9,103),'PUSH':(8,104),'PULL':(8,105) }
branchGroup = ['BRA', 'BEQ', 'BNE', 'BSR', 'RTS'] # Operations responsible for flow control
```

以下一小节负责读取要汇编和执行的源文件。此源代码应采用`.txt`文件的形式。请注意，此代码使用 Python 的`try`和`except`机制，该机制能够执行一个动作（在这种情况下，尝试从磁盘加载文件）并在动作失败时执行另一个动作。在这里，我们用它来测试默认文件名，如果该文件不存在，则从终端获取一个文件名：

```py

# Read the input source code text file and format it. This uses a default file and a user file if default is absent
prgN = 'E://ArchitectureWithPython//C_2_test.txt' # prgN = program name: default test file
try:                                              # Check whether this file exists
    with open(prgN,'r') as prgN:                  # If it's there, open it and read it
        prgN = prgN.readlines()
except:                                           # Call exception program if not there
    prgN = input('Enter source file name: ')    # Request a filename (no extension needed)
    prgN = 'E://ArchitectureWithPython//' + prgN + '.txt' # Build filename
    with open(prgN,'r') as prgN:                  # Open user file
        prgN = prgN.readlines()                   # Read it
for i in range (0,len(prgN)):                     # Scan source prgN and copy it to sFile
    sFile[i] = prgN[i]                            # Copy prgN line to sFile line
    if 'END!' in prgN[i]: break                   # If END! found, then stop copying
             # Format source code
sFile = [i.split('@')[0] for i in sFile]          # But first, remove comments     ###
for i in range(0,len(sFile)):                     # Repeat: scan input file line by line
    sFile[i] = sFile[i].strip()                   # Remove leading/trailing spaces and eol
    sFile[i] = sFile[i].replace(',',' ')          # Allow use of commas or spaces
    sFile[i] = sFile[i].replace('[','')           # Remove left bracket
    sFile[i] = sFile[i].replace(']','')       # Remove right bracket and convert [R4] to R4
    while '  ' in sFile[i]:                       # Remove multiple spaces
        sFile[i] = sFile[i].replace('  ', ' ')
sFile = [i.upper() for i in sFile]                # Convert to uppercase
```

```py
sFile = [i.split(' ') for i in sFile if i != '']  # Split the tokens into list items
```

这一小节处理等价汇编指令，并使用`EQU`指令将值绑定到符号名称。这些绑定放置在`符号表`字典中，等价项从源代码中移除：

```py

                                    # Remove assembler directives from source code
for i in range (0,len(sFile)):      # Deal with equates of the form PQR EQU 25
    if len(sFile[i]) > 2 and sFile[i][1] == 'EQU': # If line is > 2 tokens and second is EQU
        symbolTab[sFile[i][0]] = sFile[i][2]       # Put third token EQU in symbol table
sFile = [i for i in sFile if i.count('EQU') == 0]  # Remove all lines with 'EQU'
                                    # Debug: 1 none, 2 source, 3 symbol tab, 4 Decode i, 5 stack
listingP()                          # List the source code if debug level is 1
```

在这里，我们执行指令解码；也就是说，我们分析每条指令的文本以提取操作码和参数：

```py

                                    # Look for labels and add to symbol table
for i in range(0,len(sFile)):       # Add branch addresses to symbol table
    if sFile[i][0] not in codes:    # If first token not opcode, then it is a label
        symbolTab.update({sFile[i][0]:str(i)})     # Add it to the symbol table
if debugLevel > 2:                  # Display symbol table if debug level 2
    print('\nEquate and branch table\n')           # Display the symbol table
    for x,y in symbolTab.items(): print('{:<8}'.format(x),y) \
                                             # Step through the symbol table dictionary
    print('\n')
            # Assemble source code in sFile
if debugLevel > 3: print('Decoded instructions')   # If debug level 4/5, print decoded ops
for pcA in range(0,len(sFile)):              # ASSEMBLY: pcA = prog counter in assembly
    opCode, label, literal, predicate = [], [], 0, []   # Initialize variables
                                             # Instruction = label + opcode + predicate
    rD, rS1, rS2  = 0, 0, 0                  # Clear all register-select fields
    thisOp = sFile[pcA]                      # Get current instruction, thisOPp, in text form
                                             # Instruction: label + opcode or opcode
    if thisOp[0] in codes: opCode = thisOp[0]      # If token opcode, then get token
    else:                                    # Otherwise, opcode is second token
        opCode = thisOp[1]                   # Read the second token to get opcode
        label = sFile[i][0]                  # Read the first token to get the label
    if (thisOp[0] in codes) and (len(thisOp) > 1): # If first token opcode, rest is predicate
        predicate = thisOp[1:]               # Now get the predicate
    else:                                    # Get predicate if the line has a label
        if len(thisOp) > 2: predicate = thisOp[2:]
    form = codes.get(opCode)                 # Use opcode to read type (format)
                                             # Now check the bits of the format code
    if form[0] & 0b1000 == 0b1000:           # Bit 4 selects destination register rD
        if predicate[0] in symbolTab:        # Check if first token in symbol table
            rD = int(symbolTab[predicate[0]][1:]) # If it is, then get its value
        else: rD = int(predicate[0][1:])     # If not label, get register from the predicate
    if form[0] & 0b0100 == 0b0100:           # Bit 3 selects source register 1, rS1
        if predicate[1] in symbolTab:
            rS1 = int(symbolTab[predicate[1]][1:])
        else: rS1 = int(predicate[1][1:])
    if form[0] & 0b0010 == 0b0010:           # Bit 2 of format selects register rS1
        if predicate[2] in symbolTab:
            rS2 = int(symbolTab[predicate[2]][1:])
        else: rS2 = int(predicate[2][1:])
    if form[0] & 0b0001 == 0b0001:           # Bit 1 of format selects the literal field
        litV = predicate[-1]
        literal = getLit(litV)
```

这一节是在 TC1 开发之后添加的。我们引入了调试级别的概念。也就是说，在模拟运行开始时，你可以设置一个范围在`1`到`3`之间的参数，以确定在汇编处理过程中显示多少信息。这允许你在测试程序时获取更多关于指令编码的信息：

```py

    if debugLevel > 3:                       # If debug level > 3, print decoded fields
        t0 = '%02d' % pcA                    # Format instruction counter
        t1 = '{:<23}'.format(' '.join(thisOp)) # Format operation to 23 spaces
        t3 = '%04x' % literal                # Format literal to 4-character hex
        t4 = '{:04b}'.format(form[0])        # Format the 4-bit opcode format field
        print('pc =',t0,'Op =',t1,'literal',t3,'Dest reg =',rD,'rS1 =', \
              'rS1,'rS2 =',rS2,'format =',t4)  # Concatenate fields to create 32-bit opcode
    binCode = form[1]<<25|(rD)<<22|(rS1)<<19|(rS2)<<16|literal # Binary pattern
    memP[pcA] = binCode                      # Store instruction in program memory
                                             # End of the assembly portion of the program
```

我们即将执行指令。在我们这样做之前，有必要初始化一些与当前操作相关的变量（例如，跟踪）：

```py

                                           # The code is executed here
r = [0] * 8                                # Define registers r[0] to r[7]
pc = 0                                     # Set program counter to 0
run = 1                                    # run = 1 during execution
sp = 16                                    # Initialize the stack pointer (BSR/RTS)
goCount = 0                                # goCount executes n operations with no display
traceMode    = 0                           # Set to 1 to execute n instructions without display
skipToBranch = 0                           # Used when turning off tracing until a branch
```

```py
silent = 0                                 # silent = 1 to turn off single stepping
```

这是主循环，我们在这里解码指令以提取参数（寄存器编号和字面量）：

```py

                                           # Executes instructions when run is 1
while run == 1:                            # Step through instructions: first, decode them!
    binCode = memP[pc]                     # Read binary code of instruction
    pcOld = pc                             # pc in pcOld (for display purposes)
    pc = pc + 1                            # Increment the pc
    binOp = binCode >> 25                  # Extract the 7-bit opcode as binOp
    rD    = (binCode >> 22) & 7            # Extract the destination register, rD
    rS1   = (binCode >> 19) & 7            # Extract source register 1, rS1
    rS2   = (binCode >> 16) & 7            # Extract source register 2, rS2
    lit   = binCode & 0xFFFF               # Extract the 16-bit literal
    op0 = r[rD]                            # Get contents of destination register
    op1 = r[rS1]                           # Get contents of source register 1
    op2 = r[rS2]                           # Get contents of source register 2
```

在下一节中，我们将从 TC1 的原始版本开始。TC1 的第一个版本将操作码解码为二进制字符串，然后进行查找。然而，由于我们有源文件，直接从助记符文本执行会更简单。这使得代码更容易阅读：

```py

# Instead of using the binary opcode to determine the instruction, I use the text opcode
# It makes the code more readable if I use 'ADD' rather than its opcode
    mnemonic=next(key for key,value in codes.items() if value[1]==binOp
                                           # Get mnemonic from dictionary
### INTERPRET INSTRUCTIONS                    # Examine the opcode and execute it
    if   mnemonic == 'STOP': run = 0       # STOP ends the simulation
    elif mnemonic == 'END!': run = 0       # END! terminates reading source code and stops
    elif mnemonic == 'NOP':  pass          # NOP is a dummy instruction that does nothing
    elif mnemonic == 'GET':                # Reads integer from the keyboard
        printStatus()
        kbd = (input('Type integer '))     # Get input
        kbd = getLit(kbd)                  # Convert string to integer
        r[rD] = kbd                        # Store in register
        continue
    elif mnemonic == 'RND':  r[rD] = random.randint(0,lit)
                                           # Generate random number
    elif mnemonic == 'SWAP': r[rD] = shift(0,1,r[rD],8)
                                           # Swap bytes in a 16-bit word
    elif mnemonic == 'SEC':  c = 1         # Set carry flag
    elif mnemonic == 'LDRL': r[rD] = lit   # LDRL R0,20 loads R0 with literal 20
    elif mnemonic == 'LDRM': r[rD] = mem[lit]
                                           # Load register with memory location (LDRM)
    elif mnemonic == 'LDRI': r[rD] = mem[op1 + lit]
                                                   # LDRI r1,[r2,4] memory location [r2]+4
    elif mnemonic == 'STRM': mem[lit] = r[rD]      # STRM stores register in memory
    elif mnemonic == 'STRI': mem[op1 + lit] = r[rD] # STRI stores rD at location [rS1]+L
    elif mnemonic == 'MOVE': r[rD] = op1           # MOVE copies register rS1 to rD
    elif mnemonic == 'ADD':  r[rD] = alu('ADD',op1, op2)
                                                   # Adds [r2] to [r3] and puts result in r1
    elif mnemonic == 'ADDL': r[rD] = alu('ADD',op1,lit) # Adds 12 to [r2] and puts result in r1
    elif mnemonic == 'SUB':  r[rD] = alu('SUB',op1,op2) #
    elif mnemonic == 'SUBL': r[rD] = alu('SUB',op1,lit)
    elif mnemonic == 'MUL':  r[rD] = alu('MUL',op1,op2)
    elif mnemonic == 'MULL': r[rD] = alu('MUL',op1,lit)
    elif mnemonic == 'DIV':  r[rD] = alu('DIV',op1,op2) # Logical OR
    elif mnemonic == 'DIVL': r[rD] = alu('DIV',op1,lit)
    elif mnemonic == 'MOD':  r[rD] = alu('MOD',op1,op2) # Modulus
    elif mnemonic == 'MODL': r[rD] = alu('MOD',op1,lit)
    elif mnemonic == 'AND':  r[rD] = alu('AND',op1,op2) # Logical AND
    elif mnemonic == 'ANDL': r[rD] = alu('AND',op1,lit)
    elif mnemonic == 'OR':   r[rD] = alu('OR', op1,op2) # Logical OR
    elif mnemonic == 'ORL':  r[rD] = alu('OR', op1,lit)
    elif mnemonic == 'EOR':  r[rD] = alu('EOR',op1,op2) # Exclusive OR
    elif mnemonic == 'EORL': r[rD] = alu('EOR',op1,lit)
    elif mnemonic == 'NOT':  r[rD] = alu('NOT',op0,1)   # NOT r1 uses only one operand
    elif mnemonic == 'INC':  r[rD] = alu('ADD',op0,1)
    elif mnemonic == 'DEC':  r[rD] = alu('SUB',op0,1)
    elif mnemonic == 'CMP':  rr    = alu('SUB',op0,op1) # rr is a dummy variable
    elif mnemonic == 'CMPL': rr    = alu('SUB',op0,lit)
    elif mnemonic == 'ADC':  r[rD] = alu('ADC',op1,op2)
    elif mnemonic == 'SBC':  r[rD] = alu('SBC',op1,op2)
    elif mnemonic == 'LSL':  r[rD] = shift(0,0,op0,op1)
    elif mnemonic == 'LSLL': r[rD] = shift(0,0,op0,lit)
    elif mnemonic == 'LSR':  r[rD] = shift(1,0,op0,op1)
    elif mnemonic == 'LSRL': r[rD] = shift(1,0,op0,lit)
    elif mnemonic == 'ROL':  r[rD] = shift(1,1,op0,op2)
    elif mnemonic == 'ROLL': r[rD] = shift(1,1,op0,lit)
    elif mnemonic == 'ROR':  r[rD] = shift(0,1,op0,op2)
    elif mnemonic == 'RORL': r[rD] = shift(0,1,op0,lit)
    elif mnemonic == 'PRT':  print('Reg',rD,'=', '%04x' % r[rD])
    elif mnemonic == 'BRA':             pc = lit
    elif mnemonic == 'BEQ' and  z == 1: pc = lit
    elif mnemonic == 'BNE' and  z == 0: pc = lit
    elif mnemonic == 'BMI' and  n == 1: pc = lit
    elif mnemonic == 'DBEQ':                     # Decrement register and branch on zero
        r[rD] = r[rD] - 1
        if r[rD] != 0: pc = lit
    elif mnemonic == 'DBNE':            # Decrement register and branch on not zero
        r[rD] = alu('SUB',op0,1)        # Note the use of the alu function
        if z == 0: pc = lit
    elif mnemonic == 'BSR':             # Stack-based operations. Branch to subroutine
        sp = sp - 1                     # Pre-decrement stack pointer
        stack[sp] = pc                  # Push the pc (return address)
        pc = lit                        # Jump to target address
    elif mnemonic == 'RTS':             # Return from subroutine
        pc = stack[sp]                  # Pull pc address of the stack
        sp = sp + 1                     # Increment stack pointer
    elif mnemonic == 'PUSH':            # Push register to stack
        sp = sp - 1                     # Move stack pointer up to make space
        stack[sp] = op0                 # Push register in op on the stack
    elif mnemonic == 'PULL':            # Pull register off the stack
        r[rD] = stack[sp]               # Transfer stack value to register
        sp = sp + 1                     # Move stack down
```

这一小节执行一个名为跟踪的功能，允许我们在执行代码时列出寄存器的内容或关闭列表：

```py

                                        # Instruction interpretation complete. Deal with display
    if silent == 0:                     # Read keyboard ONLY if not in silent mode
       x = input('>>>')                 # Get keyboard input to continue
       if x == 'b': skipToBranch = 1    # Set flag to execute to branch with no display
       if x.isnumeric():                # Is this a trace mode with a number of steps to skip?
           traceMode = 1                # If so, set traceMode
           goCount   = getLit(x) + 1    # Record the number of lines to skip printing
    if skipToBranch == 1:               # Are we in skip-to-branch mode?
        silent = 1                      # If so, turn off printing status
        if mnemonic in branchGroup:     # Have we reached a branch?
            silent = 0                  # If branch, turn off silent mode and allow tracing
            skipToBranch = 0            # Turn off skip-to-branch mode
    if traceMode == 1:                  # If in silent mode (no display of data)
        silent = 1                      # Set silent flag
        goCount = goCount – 1           # Decrement silent mode count
        if goCount == 0:                # If we've reached zero, turn display on
            traceMode = 0               # Leave trace mode
            silent = 0                  # Set silent flag back to zero (off)
```

```py
    if silent == 0: printStatus()
```

现在我们已经解释了 TC1 模拟器，我们将演示其使用方法。

# TC1 汇编语言程序的示例

在这里，我们演示了一个 TC1 汇编语言程序。这提供了一种测试模拟器和展示其工作方式的方法。我们希望测试一系列功能，因此我们应该包括循环、条件测试和基于指针的内存访问。我们将编写一个程序来完成以下任务：

1.  将内存从位置 0 到 4 的区域填充为随机数。

1.  反转数字的顺序。

由于这个问题使用内存和顺序地址，它涉及到寄存器间接寻址，即`LDRI`和`STRI`指令。通过以下方式创建随机数并将它们按顺序存储在内存中：

```py

Set a pointer to the first memory location (i.e.,0)
Set a counter to 5 (we are going to access five locations 0 to 4)
Repeat
  Generate a random number
  Store this number at the pointer address
  Point to next number (i.e., add 1 to the pointer)
  Decrement the counter (i.e., counter 5,4,3,2,1,0)
  Until counter = 0
```

在 TC1 代码中，我们可以将其翻译如下：

```py

        LDRL r0,0             @ Use r0 as a memory pointer and set it to 0
        LDRL r1,5             @ Use r1 as the loop counter
  Loop1 RND  r2               @ Loop: Generate a random number in r2
        STRI r2,[r0],0        @ Store the random number in memory using pointer r0
        INC  r0               @ Point to the next location (add 1 to the pointer)
        DEC  r1               @ Decrement the loop counter (subtract 1 from the counter)
        BNE  Loop1            @ Repeat until 0 (branch back to Loop1 if the last result was not 0)
```

我们已经用随机值填充了一个内存区域。现在我们需要反转它们的顺序。有许多方法可以反转数字的顺序。一种是将数字从源位置移动到内存中的一个临时位置，然后以相反的顺序写回。当然，这需要额外的内存来存储临时副本。考虑另一种不需要缓冲区的解决方案。我们将在目标地址上方写下源地址：

`原始（源）        0   1   2   3   4`

`交换（目标）    4   3   2   1   0`

正如你所见，位置`0`与位置`4`交换，然后位置`1`与位置`3`交换；接着，在位置`2`，我们达到了中间点，反转完成。为了执行这个动作，我们需要两个指针，每个字符串的一端一个。我们选择字符串两端的两个字符并交换它们。然后，我们将指针向内移动并执行第二次交换。当指针在中点相遇时，任务完成。请注意，这假设要反转的项目数量是奇数：

```py

Set upper pointer to top
Set lower pointer to bottom
Repeat
   Get value at upper pointer
   Get value at lower pointer
   Swap values and store
Until upper pointer and lower pointer are equal
```

在 TC1 汇编语言中，这看起来如下：

```py

      LDRL r0,0                @ Lower pointer points at first entry in table
      LDRL r1,4                @ Upper pointer points at last entry in table
Loop2 LDRI r2,[r0],0           @ REPEAT: Get lower value pointed at by r0
      LDRI r3,[r1],0           @ Get upper value pointed at by r1
      MOVE r2,r4               @ Save lower value in r4 temporarily
      STRI r3,[r0],0           @ Store upper value in lower entry position
      STRI r4,[r1],0           @ Store saved lower value in upper entry position
      INC  r0                  @ Increase lower pointer
      DEC  r1                  @ Decrease upper pointer
      CMP  r0,r1               @ Compare pointers
      BNE  Loop2               @ UNTIL all characters moved
```

以下显示了程序逐条指令执行时的输出。为了简化数据的阅读，我们将寄存器和内存值的变化用粗体表示。分支操作被阴影覆盖。比较指令用斜体表示。

第一个块是在指令执行开始前 TC1 打印的源代码：

```py

TC1 CPU simulator 11 September 2022
Input debug level 1 - 5: 4
Source assembly code listing
0           LDRL R0 0
1           LDRL R1 5
2   LOOP1   RND  R2
3           STRI R2 R0 0
4           INC  R0
5           DEC  R1
6           BNE  LOOP1
7           NOP
8           LDRL R0 0
9           LDRL R1 4
10  LOOP2   LDRI R2 R0 0
11          LDRI R3 R1 0
12          MOVE R4 R2
13          STRI R3 R0 0
14          STRI R4 R1 0
15          INC  R0
16          DEC  R1
17          CMP  R0 R1
18          BNE  LOOP2
19          NOP
20          STOP
21          END!
Equate and branch table
START    0
LOOP1    2
LOOP2    10
```

第二个代码块显示了汇编器在指令解码时的输出。你可以看到各种寄存器、字面量和格式字段：

```py

Decoded instructions
pc=00 Op =       LDRL R0 0        literal 0000 RD=0 rS1=0 rS2=0 format=1001
pc=01 Op =       LDRL R1 5        literal 0005 RD=1 rS1=0 rS2=0 format=1001
pc=02 Op=  LOOP1 RND R2 0XFFFF    literal ffff RD=2 rS1=0 rS2=0 format=1001
pc=03 Op =       STRI R2 R0 0     literal 0000 RD=2 rS1=0 rS2=0 format=1101
pc=04 Op =       INC  R0          literal 0000 RD=0 rS1=0 rS2=0 format=1000
pc=05 Op =       DEC  R1          literal 0000 RD=1 rS1=0 rS2=0 format=1000
pc=06 Op =       BNE  LOOP1       literal 0002 RD=0 rS1=0 rS2=0 format=0001
pc=07 Op =       NOP              literal 0000 RD=0 rS1=0 rS2=0 format=0000
pc=08 Op =       LDRL R0 0        literal 0000 RD=0 rS1=0 rS2=0 format=1001
pc=09 Op =       LDRL R1 4        literal 0004 RD=1 rS1=0 rS2=0 format=1001
pc=10 Op=  LOOP2 LDRI R2 R0 0     literal 0000 RD=2 rS1=0 rS2=0 format=1101
pc=11 Op =       LDRI R3 R1 0     literal 0000 RD=3 rS1=1 rS2=0 format=1101
pc=12 Op =       MOVE R4 R2       literal 0000 RD=4 rS1=2 rS2=0 format=1100
pc=13 Op =       STRI R3 R0 0     literal 0000 RD=3 rS1=0 rS2=0 format=1101
pc=14 Op =       STRI R4 R1 0     literal 0000 RD=4 rS1=1 rS2=0 format=1101
pc=15 Op =       INC  R0          literal 0000 RD=0 rS1=0 rS2=0 format=1000
pc=16 Op =       DEC  R1          literal 0000 RD=1 rS1=0 rS2=0 format=1000
pc=17 Op =       CMP  R0 R1       literal 0000 RD=0 rS1=1 rS2=0 format=1100
pc=18 Op =       BNE  LOOP2       literal 000a RD=0 rS1=0 rS2=0 format=0001
pc=19 Op =       NOP              literal 0000 RD=0 rS1=0 rS2=0 format=0000
pc=20 Op =       STOP             literal 0000 RD=0 rS1=0 rS2=0 format=0000
pc=21 Op =       END!             literal 0000 RD=0 rS1=0 rS2=0 format=0000
```

以下提供了使用此程序运行时的输出。我们将跟踪级别设置为`4`以显示源代码（在文本处理之后）、符号表和解码指令。

然后，我们逐行执行了代码。为了使输出更易于阅读并适应页面，我们移除了不改变的寄存器和内存位置，并突出显示了由于指令的结果而改变的值（内存、寄存器和 z 标志）。你可以跟随这些内容，看看内存/寄存器是如何随着每个指令而改变的。

正如你所见，我们在内存位置`0`到`4`中创建了五个随机数，然后反转它们的顺序。这并不匹配打印状态的输出，因为它已经被修改以适应打印：

```py

0         LDRL R0 0         PC =  0 z = 0 0000 0000 0000 0000 0000
                            R  0000 0000 0000 0000 0000
1         LDRL R1 5         PC =  1 z = 0 0000 0000 0000 0000 0000
                            R  0000 0005 0000 0000 0000
2  LOOP1  RND  R2           PC =  2 z = 0 0000 0000 0000 0000 0000
                            R  0000 0005 9eff 0000 0000
3         STRI R2 R0 0      PC =  3 z = 0 9eff 0000 0000 0000 0000
                            R  0000 0005 9eff 0000 0000
4         INC  R0           PC =  4 z = 0 9eff 0000 0000 0000 0000
                            R  0001 0005 9eff 0000 0000
5         DEC  R1           PC =  5 z = 0 9eff 0000 0000 0000 0000
                            R  0001 0004 9eff 0000 0000
6         BNE  LOOP1        PC =  6 z = 0 9eff 0000 0000 0000 0000
                            R  0001 0004 9eff 0000 0000
2  LOOP1  RND  R2           PC =  2 z = 0 9eff 0000 0000 0000 0000
                            R  0001 0004 6d4a 0000 0000
3         STRI R2 R0 0      PC =  3 z = 0 9eff 6d4a 0000 0000 0000
                            R  0001 0004 6d4a 0000 0000
4         INC  R0           PC =  4 z = 0 9eff 6d4a 0000 0000 0000
                            R  0002 0004 6d4a 0000 0000
5         DEC  R1           PC =  5 z = 0 9eff 6d4a 0000 0000 0000
                            R  0002 0003 6d4a 0000 0000
6         BNE  LOOP1        PC =  6 z = 0 9eff 6d4a 0000 0000 0000
                            R  0002 0003 6d4a 0000 0000
2  LOOP1  RND  R2           PC =  2 z = 0 9eff 6d4a 0000 0000 0000
                            R  0002 0003 a387 0000 0000
3         STRI R2 R0 0      PC =  3 z = 0 9eff 6d4a a387 0000 0000
                            R  0002 0003 a387 0000 0000
4         INC  R0           PC =  4 z = 0 9eff 6d4a a387 0000 0000
                            R  0003 0003 a387 0000 0000
5         DEC  R1           PC =  5 z = 0 9eff 6d4a a387 0000 0000
                            R  0003 0002 a387 0000 0000
6         BNE  LOOP1        PC =  6 z = 0 9eff 6d4a a387 0000 0000
                            R  0003 0002 a387 0000 0000
2  LOOP1  RND  R2           PC =  2 z = 0 9eff 6d4a a387 0000 0000
                            R  0003 0002 2937 0000 0000
3         STRI R2 R0 0      PC =  3 z = 0 9eff 6d4a a387 2937 0000
                            R  0003 0002 2937 0000 0000
4         INC  R0           PC =  4 z = 0 9eff 6d4a a387 2937 0000
                            R  0004 0002 2937 0000 0000
5         DEC  R1           PC =  5 z = 0 9eff 6d4a a387 2937 0000
                            R  0004 0001 2937 0000 0000
6         BNE  LOOP1        PC =  6 z = 0 9eff 6d4a a387 2937 0000
                            R  0004 0001 2937 0000 0000
2  LOOP1  RND  R2           PC =  2 z = 0 9eff 6d4a a387 2937 0000
                            R  0004 0001 db95 0000 0000
3         STRI R2 R0 0      PC =  3 z = 0 9eff 6d4a a387 2937 db95
                            R  0004 0001 db95 0000 0000
4         INC  R0           PC =  4 z = 0 9eff 6d4a a387 2937 db95
                            R  0005 0001 db95 0000 0000
5         DEC  R1           PC =  5 z = 1 9eff 6d4a a387 2937 db95
                            R  0005 0000 db95 0000 0000
6         BNE LOOP1         PC =  6 z = 1 9eff 6d4a a387 2937 db95
                            R  0005 0000 db95 0000 0000
7         NOP               PC =  7 z = 1 9eff 6d4a a387 2937 db95
                            R  0005 0000 db95 0000 0000
8         LDRL R0 0         PC =  8 z = 1 9eff 6d4a a387 2937 db95
                            R  0000 0000 db95 0000 0000
9         LDRL R1 4         PC =  9 z = 1 9eff 6d4a a387 2937 db95
                            R  0000 0004 db95 0000 0000
10 LOOP2  LDRI R2 R0 0      PC = 10 z = 1 9eff 6d4a a387 2937 db95
                            R  0000 0004 9eff 0000 0000
11        LDRI R3 R1 0      PC = 11 z = 1 9eff 6d4a a387 2937 db95
                            R  0000 0004 9eff db95 0000
12        MOVE R4 R2        PC = 12 z = 1 9eff 6d4a a387 2937 db95
                            R  0000 0004 9eff db95 9eff
13        STRI R3 R0 0      PC = 13 z = 1 db95 6d4a a387 2937 db95
                            R  0000 0004 9eff db95 9eff
14        STRI R4 R1 0      PC = 14 z = 1 db95 6d4a a387 2937 9eff
                            R  0000 0004 9eff db95 9eff
15        INC  R0           PC = 15 z = 0 db95 6d4a a387 2937 9eff
                            R  0001 0004 9eff db95 9eff
16        DEC  R1           PC = 16 z = 0 db95 6d4a a387 2937 9eff
                            R  0001 0003 9eff db95 9eff
17        CMP  R0 R1        PC = 17 z = 0 db95 6d4a a387 2937 9eff
                            R  0001 0003 9eff db95 9eff
18        BNE  LOOP2        PC = 18 z = 0 db95 6d4a a387 2937 9eff
                            R  0001 0003 9eff db95 9eff
10 LOOP2  LDRI R2 R0 0      PC = 10 z = 0 db95 6d4a a387 2937 9eff
                            R  0001 0003 6d4a db95 9eff
11        LDRI R3 R1 0      PC = 11 z = 0 db95 6d4a a387 2937 9eff
                            R  0001 0003 6d4a 2937 9eff
12        MOVE R4 R2        PC = 12 z = 0 db95 6d4a a387 2937 9eff
                            R  0001 0003 6d4a 2937 6d4a
13        STRI R3 R0 0      PC = 13 z = 0 db95 2937 a387 2937 9eff
                            R  0001 0003 6d4a 2937 6d4a
14        STRI R4 R1 0      PC = 14 z = 0 db95 2937 a387 6d4a 9eff
                            R  0001 0003 6d4a 2937 6d4a
15        INC  R0           PC = 15 z = 0 db95 2937 a387 6d4a 9eff
                            R  0002 0003 6d4a 2937 6d4a
16        DEC  R1           PC = 16 z = 0 db95 2937 a387 6d4a 9eff
                            R  0002 0002 6d4a 2937 6d4a
17        CMP  R0 R1        PC = 17 z = 1 db95 2937 a387 6d4a 9eff
                            R  0002 0002 6d4a 2937 6d4a
18        BNE  LOOP2        PC = 18 z = 1 db95 2937 a387 6d4a 9eff
                            R  0002 0002 6d4a 2937 6d4a
19        NOP               PC = 19 z = 1 db95 2937 a387 6d4a 9eff
                            R  0002 0002 6d4a 2937 6d4a
20        STOP              PC = 20 z = 1 db95 2937 a387 6d4a 9eff
                            R  0002 0002 6d4a 2937 6d4a
```

在下一节中，我们将展示如何测试 TC1 的操作。我们将涵盖以下内容：

+   测试汇编器（例如，使用代码自由格式的能力）

+   测试流程控制指令（分支）

+   测试移位操作

# 测试汇编器

由于 TC1 汇编器可以处理多种排版特性（例如，大写或小写和多个空格），测试汇编器的一个简单方法是为它提供一个包含各种条件的文件来汇编，例如多个空格、等价和大小写转换。我的初始测试源代码如下：

```py

     NOP
 BRA eee
      INC r4
alan inc r5
eee    STOP
aa NOP @comment2
bb NOP     1
      LDRL      r0,   12
      LDRL r3,0x123 @ comment1
      LDRL r7,     0xFF
      INC R2
  BRA last
test1     EQU    999
  @comment3
@comment4
  @ qqq EQU 7
www STRI r1,r2,1
abc   equ 25
qwerty  equ   888
last LDRL r5,0xFAAF
  beQ Aa
      STOP 2
```

这段代码并不十分优雅；它只是随机测试代码。在下面的代码中，我们提供了汇编器在`调试`模式下的输出。这包括代码的格式化（删除空白行并将小写转换为大写）。第一个列表提供了指令作为令牌列表的数组：

```py

TC1 CPU simulator 11 September 2022
Input debug level 1 - 5: 4
Source assembly code listing
0           NOP
1           BRA EEE
2           INC R4
3   ALAN    INC R5
4   EEE     STOP
5   AA      NOP
6   BB      NOP 1
7           LDRL R0 12
8           LDRL R3 0X123
9           LDRL R7 0XFF
10          INC R2
11          BRA LAST
12  WWW     STRI R1 R2 1
13  LAST    LDRL R5 0XFAAF
14          BEQ AA
15          STOP 2
```

第二个列表是将符号名称和标签与整数值关联的符号表：

```py

Equate and branch table
START    0
TEST1    999
ABC      25
QWERTY   888
ALAN     3
EEE      4
AA       5
BB       6
WWW      12
LAST     13
LOOP1    18
LOOP2    26
```

下一个列表主要用于调试，当指令未按预期行为时。它让您确定指令是否被正确解码：

```py

Decoded instructions
pc=0  op=NOP                  literal 000 Dest reg=0 rS1-0 rS2=0 format=0000
pc=00 Op=NOP                  literal 0000 Dest reg=0 rS1=0 rS2=0 format=0000
pc=01 Op=BRA EEE              literal 0004 Dest reg=0 rS1=0 rS2=0 format=0001
pc=02 Op=INC R4               literal 0000 Dest reg=4 rS1=0 rS2=0 format=1000
pc=03 Op=ALAN INC R5          literal 0000 Dest reg=5 rS1=0 rS2=0 format=1000
pc=04 Op=EEE STOP             literal 0000 Dest reg=0 rS1=0 rS2=0 format=0000
pc=05 Op=AA NOP               literal 0000 Dest reg=0 rS1=0 rS2=0 format=0000
pc=06 Op=BB NOP 1             literal 0000 Dest reg=0 rS1=0 rS2=0 format=0000
pc=07 Op=LDRL R0 12           literal 000c Dest reg=0 rS1=0 rS2=0 format=1001
pc=08 Op=LDRL R3 0X123        literal 0123 Dest reg=3 rS1=0 rS2=0 format=1001
pc=09 Op=LDRL R7 0XFF         literal 00ff Dest reg=7 rS1=0 rS2=0 format=1001
pc=10 Op=INC R2               literal 0000 Dest reg=2 rS1=0 rS2=0 format=1000
pc=11 Op=BRA LAST             literal 000d Dest reg=0 rS1=0 rS2=0 format=0001
pc=12 Op=WWW STRI R1 R2 1     literal 0001 Dest reg=1 rS1=2 rS2=0 format=1101
pc=13 Op=LAST LDRL R5 0XFAAF   literal faaf Dest reg=5 rS1=0 rS2=0 format=1001
pc=14 Op=BEQ AA               literal 0005 Dest reg=0 rS1=0 rS2=0 format=0001
pc=15 Op=STOP 2               literal 0000 Dest reg=0 rS1=0 rS2=0 format=0000
>>>
```

## 测试流程控制操作

在这里，我们演示如何测试计算机最重要的操作类别之一，即流程控制指令，即条件分支。

需要测试的最重要的一类指令是改变控制流程的指令：分支和子程序调用指令。下面的代码片段也是无意义的（它仅用于测试指令执行）且仅设计用于测试循环。一个循环使用非零操作的分支构建，另一个使用通过递减寄存器并分支直到寄存器递减到零的自动循环机制。`DBNE` `r0`,`loop`，其中`r0`是被递减的计数器，`loop`是分支目标地址。

我们首先提供源列表和符号表：

```py

>>> %Run TC1_FinalForBook_V1.2_20220911.py
TC1 CPU simulator 11 September 2022
Input debug level 1 - 5: 4
Source assembly code listing
0           NOP
1           BRA LAB1
2           INC R0
3   LAB1    INC R2
4           NOP
5           BRA LAB6
6           NOP
7   LAB2    LDRL R2 3
8   LAB4    DEC R2
9           NOP
10          BNE LAB4
11          NOP
12          BSR LAB7
13          NOP
14          LDRL R3 4
15  LAB5    NOP
16          INC R7
17          DBNE R3 LAB5
18          NOP
19          STOP
20  LAB6    BRA LAB2
21          NOP
22  LAB7    DEC R7
23          DEC R7
24          RTS
25          END!
Equate and branch table
START    0
LAB1     3
LAB2     7
LAB4     8
LAB5     15
LAB6     20
LAB7     22
```

以下是在调试会话后的输出。如您所见，分支序列被忠实实现。请注意，我们已经突出显示了分支动作和后果（即，下一个指令）：

```py

0         NOP               PC= 0 z=0 n=0 c=0 
                            R  0000 0000 0000 0000 0000 0000 0000 0000
1         BRA LAB1          PC= 1 z=0 n=0 c=0
                            R  0000 0000 0000 0000 0000 0000 0000 0000
3  LAB1   INC R2            PC= 3 z=0 n=0 c=1 
                            R  0000 0000 0001 0000 0000 0000 0000 0000
4         NOP               PC= 4 z=0 n=0 c=1 
                            R  0000 0000 0001 0000 0000 0000 0000 0000
5         BRA LAB6          PC= 5 z=0 n=0 c=1 
                            R  0000 0000 0001 0000 0000 0000 0000 0000
20 LAB6   BRA LAB2          PC=20 z=0 n=0 c=1 
                            R  0000 0000 0001 0000 0000 0000 0000 0000
7  LAB2   LDRL R2 3         PC= 7 z=0 n=0 c=1 
                            R  0000 0000 0003 0000 0000 0000 0000 0000
8  LAB4   DEC R2            PC= 8 z=0 n=0 c=1 
                            R  0000 0000 0002 0000 0000 0000 0000 0000
9         NOP               PC= 9 z=0 n=0 c=1 
                            R  0000 0000 0002 0000 0000 0000 0000 0000
10        BNE LAB4          PC=10 z=0 n=0 c=1 
                            R  0000 0000 0002 0000 0000 0000 0000 0000
8  LAB4   DEC R2            PC= 8 z=0 n=0 c=1 
                            R  0000 0000 0001 0000 0000 0000 0000 0000
9         NOP               PC= 9 z=0 n=0 c=1 
                            R  0000 0000 0001 0000 0000 0000 0000 0000
10        BNE LAB4          PC=10 z=0 n=0 c=1 
                            R  0000 0000 0001 0000 0000 0000 0000 0000
8  LAB4   DEC R2            PC= 8 z=1 n=0 c=0 
                            R  0000 0000 0000 0000 0000 0000 0000 0000
9         NOP               PC= 9 z=1 n=0 c=0 
                            R  0000 0000 0000 0000 0000 0000 0000 0000
10        BNE LAB4          PC=10 z=1 n=0 c=0 
                            R  0000 0000 0000 0000 0000 0000 0000 0000
11        NOP               PC=11 z=1 n=0 c=0 
                            R  0000 0000 0000 0000 0000 0000 0000 0000
12        BSR LAB7          PC=12 z=1 n=0 c=0 
                            R  0000 0000 0000 0000 0000 0000 0000 0000
22 LAB7   DEC R7            PC=22 z=0 n=1 c=1 
                            R  0000 0000 0000 0000 0000 0000 0000 ffff
23        DEC R7            PC=23 z=0 n=1 c=1 
                            R  0000 0000 0000 0000 0000 0000 0000 fffe
24        RTS               PC=24 z=0 n=1 c=1 
                            R  0000 0000 0000 0000 0000 0000 0000 fffe
13        NOP               PC=13 z=0 n=1 c=1 
                            R  0000 0000 0000 0000 0000 0000 0000 fffe
14        LDRL R3 4         PC=14 z=0 n=1 c=1 
                            R  0000 0000 0000 0004 0000 0000 0000 fffe
15 LAB5   NOP               PC=15 z=0 n=1 c=1 
                            R  0000 0000 0000 0004 0000 0000 0000 fffe
16        INC R7            PC=16 z=0 n=1 c=1 
                            R  0000 0000 0000 0004 0000 0000 0000 ffff
17        DBNE R3 LAB5      PC=17 z=0 n=1 c=1 
                            R  0000 0000 0000 0003 0000 0000 0000 ffff
15 LAB5   NOP               PC=15 z=0 n=1 c=1 
                            R  0000 0000 0000 0003 0000 0000 0000 ffff
16        INC R7            PC=16 z=1 n=0 c=0 
                            R  0000 0000 0000 0003 0000 0000 0000 0000
17        DBNE R3 LAB5      PC=17 z=1 n=0 c=0 
                            R  0000 0000 0000 0002 0000 0000 0000 0000
15 LAB5   NOP               PC=15 z=1 n=0 c=0 
                            R  0000 0000 0000 0002 0000 0000 0000 0000
16        INC R7            PC=16 z=0 n=0 c=1 
                            R  0000 0000 0000 0002 0000 0000 0000 0001
17        DBNE R3 LAB5      PC=17 z=0 n=0 c=1 
                            R  0000 0000 0000 0001 0000 0000 0000 0001
15 LAB5   NOP               PC=15 z=0 n=0 c=1 
                            R  0000 0000 0000 0001 0000 0000 0000 0001
16        INC R7            PC=16 z=0 n=0 c=1 
                            R  0000 0000 0000 0001 0000 0000 0000 0002
17        DBNE R3 LAB5      PC=17 z=0 n=0 c=1 
                            R  0000 0000 0000 0000 0000 0000 0000 0002
18        NOP               PC=18 z=0 n=0 c=1 
                            R  0000 0000 0000 0000 0000 0000 0000 0002
19        STOP              PC=19 z=0 n=0 c=1 
                            R  0000 0000 0000 0000 0000 0000 0000 0002
```

在下一章中，我们将探讨如何增强 TC1 程序以添加错误检查、包含新指令和特殊功能（如可变长度操作数字段）等功能。

## 测试移位操作

TC1 支持两种移位类型：*逻辑*和*旋转*。逻辑移位将位向左或向右移动。在一边，空出的位被零替换，在另一边，移出的位被复制到进位标志。在旋转中，从一边移出的位被复制到另一边；也就是说，位串被当作一个环。无论进行多少次移位，都不会丢失任何位。在每次移位时，移到另一边的位也被复制到进位位。

大多数真实计算机还有两种其他移位变体：算术移位，在右移时保留二进制补码数的符号（除以 2 操作），以及带进位的旋转移位，其中从一边移入的位是旧的进位位，移出的位成为新的进位位。本质上，如果寄存器有`m`位，进位位被包含以创建一个`m+1`位的字。这个特性用于多精度算术。我们没有在 TC1 中包含这些模式。

除了指定移位类型外，我们还需要指定移位方向（左或右）。大多数计算机允许你指定移位的次数。我们提供了这两种设施，并且移位的次数可以使用寄存器或字面量来指定。在多长度移位中，进位位的当前状态是最后一位移入进位的位。移位操作（附示例）如下：

| **移位类型** | **寄存器/字面量** | **示例** |
| --- | --- | --- |
| 逻辑左移 | 字面量 | `LSLL r0,r1,2` |
| 逻辑左移 | 寄存器 | `LSL r3,r1,r4` |
| 逻辑右移 | 字面量 | `LSRL r0,r1,2` |
| 逻辑右移 | 寄存器 | `LSR r3,r1,r2` |
| 左旋转 | 字面量 | `ROLL r0,r1,2` |
| 左旋转 | 寄存器 | `ROL` `r3,r1,r0` |
| 右旋转 | 字面量 | `RORL r0,r3,2` |
| 右旋转 | 寄存器 | `ROR r3,r1,r0` |

表 6.1 – TC1 移位模式

当我们测试这些指令时，我们必须确保移位方向正确，移位的次数正确，末位（那些移出或移入的位）的行为正确，并且标志位被适当地设置。

考虑以下使用 16 位值`1000000110000001`的一系列移位代码片段：

```py

LDRL r1,%1000000110000001
LSLL r0,r1,1
LSLL r0,r1,2
LSRL r0,r1,1
LSRL r0,r1,1
LDRL r1,%1000000110000001
LDRL r2,1
LDRL r3,2
LSLL r0,r1,r2
LSLL r0,r1,r3
LSRL r0,r1,r2
LSRL r0,r1,r2
```

以下是从模拟器输出的信息（编辑后仅显示相关信息），显示了在执行前面的代码时寄存器和条件码。寄存器`r0`的二进制值显示在右侧。这使我们能够通过人工检查来验证操作是否正确执行：

```py

1  LDRL R1 %1000000110000001 z = 0 n = 0 c = 0 
   Regs 0 - 3  0000 8181 0000 0000  R0 =  0000000000000000
2  LSLL R0 R1 1              z = 0 n = 0 c = 1 
   Regs 0 – 3  0302 8181 0000 0000  R0 =  0000001100000010
3  LSLL R0 R1 2              z = 0 n = 0 c = 0 
   Regs 0 - 3  0604 8181 0000 0000  R0 =  0000011000000100
4  LSRL R0 R1 1              z = 0 n = 0 c = 1 
   Regs 0 - 3  40c0 8181 0000 0000  R0 =  0100000011000000
5  LSRL R0 R1 1              z = 0 n = 0 c = 1 
   Regs 0 - 3  40c0 8181 0000 0000  R0 =  0100000011000000
6  LDRL R1 %1000000110000001 z = 0 n = 0 c = 1 
   Regs 0 - 3  40c0 8181 0000 0000  R0 =  0100000011000000
7  LDRL R2 1                 z = 0 n = 0 c = 1 
   Regs 0 - 3  40c0 8181 0001 0000  R0 =  0100000011000000
8  LDRL R3 2                 z = 0 n = 0 c = 1 
   Regs 0 - 3  40c0 8181 0001 0002  R0 =  0100000011000000
9  LSL  R0 R1 R2             z = 0 n = 0 c = 1 
   Regs 0 - 3  0302 8181 0001 0002  R0 =  0000001100000010
10 LSL  R0 R1 R3             z = 0 n = 0 c = 0 
   Regs 0 - 3  0604 8181 0001 0002  R0 =  0000011000000100
11 LSR  R0 R1 R2             z = 0 n = 0 c = 1 
   Regs 0 - 3  40c0 8181 0001 0002  R0 =  0100000011000000
12 LSR  R0 R1 R2             z = 0 n = 0 c = 1 
```

```py
   Regs 0 – 3  40c0 8181 0001 0002  R0 =  0100000011000000
```

注意，加载操作不会影响 z 位。一些计算机几乎在每次操作后都会更新 z 位。一些计算机在需要时更新 z 位（例如，我们稍后将要介绍的 ARM），而一些计算机仅在特定操作后更新它。

本章的最后一部分涵盖了向 TC1 添加后缀的内容，我们提供了一个更简单的示例，它执行相同的基本功能，但以不同的方式执行某些操作，例如指令解码。这样做的目的是为了说明构建模拟器有许多方法。

# TC1 后缀

这里展示的 TC1 版本是在本书的开发过程中逐渐形成的。当前版本比原型具有更多功能；例如，最初它不包括符号分支地址，并要求用户输入实际的行号。

在这里，我们展示的是 TC1 的一个简化版本，称为 TC1mini，我们在其中做一些不同的事情；例如，不允许自由格式（助记符必须为大写，寄存器为小写，并且不能使用空格和逗号作为可互换的分隔符）。在这个版本中，一个简单的函数检查助记符是否有效，如果无效则终止程序。同样，我们增加了一个功能，检查由指针生成的地址是否在内存空间范围内。下一节提供了一些关于这个版本的注释。

## classDecode 函数

TC1 将每个指令与一个 4 位二进制值关联起来，以指示当前指令需要参数；例如，`1101`表示寄存器`rD`、`rS1`和一个字面量。TC1mini 版本将一个*类号*与范围`0`到`7`中的每个指令关联起来，以描述其类型。类别从`0`（无参数的助记符）到`7`（具有间接地址的助记符，如`LDRI r2,[r4]`）。与 TC1 不同，TC1mini 汇编语言中的`[]`括号不是可选的。

两个模拟器 TC1 和 TC1mini 之间的区别在于 4 位二进制代码提供了*预解码*；也就是说，模拟器不需要计算指令所需的参数，因为代码直接告诉你。如果你使用类号，你必须解码类号以确定实际所需的参数。然而，类号可以非常创新。TC1mini 使用七种不同的指令格式，并需要定义至少七个类别。如果你有，比如说，14 个类别，每个寻址模式类别可以分成两个子类别，以给你更大的控制权来执行指令过程。

`classDecode`函数接收一个指令的谓词并返回四个谓词值、目标寄存器、源寄存器`1`、源寄存器`2`和字面量。当然，指令可能包含从零到四个这样的值。因此，这些参数最初被设置为哑值，要么是`null`字符串，要么是零。

在继续之前，请记住 Python 的`in`操作符对于测试一个元素是否是集合的成员非常有用。例如，如果一个操作在类别 2、4、5 和 9 中，我们可以写出以下代码：

```py

if thisThing in [2, 4, 5, 9]:                         # Test for membership of the set
def classDecode(predicate):
    lit,rD,rS1,rS2 = '',0,0,0                         # The literal is a null string initially
    if opClass in [1]:      lit = predicate           # Class 1 is mnemonic plus a literal
    if opClass in [2]:      rD  = reg1[predicate]     # Class 2 is mnemonic plus a literal
    if opClass in [3,4,5,6,7]:                   # Classes 3 to 7 have multiple parameters
        predicate = predicate.split(',')              # So, split predicate into tokens
        rD = reg1[predicate[0]]                       # Get first token (register number)
```

当前指令的`opClass`用于提取参数。我们不是使用`if`结构，而是使用了 Python 的`if in [list]`结构；例如，`if opClass in [3,4,5,6,7]`如果指令在类别`3`到`7`中，则返回`True`。如果是这样，谓词（一个字符串）使用`split()`函数分割成列表，并读取第一个元素以提取目标寄存器`rD`。请注意，我们只需要分割谓词一次，因为所有后续的情况也属于这个组。

## 测试函数`testLine`

TC1 的另一个限制是缺乏测试和验证；例如，我有时会输入`MOVE`而不是`MOV`，程序就会崩溃。通常这不是问题；你只需重新编辑源程序。然而，在调试 TC1 时，我经常假设错误是由于我新代码中的错误造成的，结果发现只是汇编语言程序中的一个误打印。因此，我添加了一些测试。以下提供了测试函数：

```py

def testLine(tokens):                    # Check whether there's a valid instruction in this line
    error = 1                            # error flag = 1 for no error and 0 for an error state
    if len(tokens) == 1:                 # If the line is a single token, it must be a mnemonic
        if tokens[0] in codes: error = 0 # If the token is in codes, there's no error
    else:                                # Otherwise, we have a multi-token line
        if (tokens[0] in codes) or (tokens[1] in codes): error = 0:
    return(error)                        # Return the error code
```

唯一值得关注的一行是以下内容：

```py

if (tokens[0] in codes) or (tokens[1] in codes): error = 0:
```

需要考虑两种情况：带有标签的指令和无标签的指令。在前一种情况下，助记符是指令中的第二个令牌，在后一种情况下，助记符是第一个令牌。我们可以通过使用 Python 的`if ... in`结构来测试令牌是否是助记符。比如说我们有以下结构：

`if token[0]` `in codes`

如果第一个令牌是有效的助记符，则返回`True`。我们可以使用`or`布尔运算符将两个测试组合起来得到前面的表达式。在程序中，我们使用`tokens`参数调用`testLine`，它返回一个错误。我们使用错误来打印消息，并通过`sys.exit()`函数将其返回给操作系统。

## testIndex()函数

此模拟器提供了一种形式为`LDRI r1,[r2]`的指令，以提供内存间接寻址（即基于指针或索引寻址）。

在这种情况下，寄存器`r1`被加载了由寄存器`r2`指向的内存内容。如果指针寄存器包含一个无效值，该值超出了合法地址范围，程序将会崩溃。通过测试索引，我们可以确保检测到越界索引。请注意，只有第一个源寄存器`rS1`被用作内存指针：

```py

def testIndex():                        # Test for register or memory index out of range
    if (rD > 7) or (rS1 > 7) or (rS2 > 7): # Ensure register numbers are in the range 0 to 7
        print('Register number error')
        sys.exit()                      # Call operating system to leave the Python program
    if mnemonic in ['LDRI', 'STRI']:    # Memory index testing only for memory load and store
        if r[rS1] > len(m) - 1:         # Test rS1 contents are less than memory size
            print(' Memory index error')
            sys.exit()
    return()
```

## 一般注释

以下行演示了如何从助记符中提取操作类。由于`()`和`[]`括号的存在，表达式看起来很奇怪。`codes.get(key)`操作使用`key`从`codes`字典中获取相关值：

```py

opClass = codes.get(mnemonic)[0]     # Use mnemonic to read opClass from the codes dictionary
```

在这种情况下，键是助记符，返回的值是操作类；例如，如果助记符是`'LDRL'`，相应的值是`[3]`。请注意，返回的值不是`3`！它是一个包含单个值`3`的*列表*。因此，我们必须通过指定第一个项目来从列表中提取值，即`mnemonic[0]`。

构建指令有许多方法。在 TC1 中，我们创建一个二进制值，就像真正的汇编器一样。在 TC1mini 中，我们直接从汇编语言形式执行指令。因此，当我们编译指令时，我们创建一个文本形式的程序。为此，我们需要将标签、助记符、寄存器和立即数组合成一个列表。

实现该功能的代码如下：

```py

thisLine = list((i,label,mnemonic,predicate,opClass))
                                        # Combine the component parts in a list
prog.append(thisLine)                   # Add the new line to the existing program
```

在本例中，我们使用`list()`函数将项目组合成一个列表，然后使用`append()`将此项目添加到现有列表中。注意`list()`的语法。你可能期望它是`list(a,b,c)`。不，它是`list((a,b,c))`。`list()`函数使用括号作为正常操作，但列表本身必须放在括号内。这是因为列表项构成了列表的*单个*参数。

## TC1tiny 代码列表

这是 TC1 简化的版本列表。指令根据操作数的数量和排列分为八个类别。每个指令都在字典`codes`中，它提供了用于解码操作数的类别编号。指令本身直接从其助记符执行。与 TC1 不同，没有中间的二进制代码。同样，寄存器名称和间接寄存器名称也都在字典中，以简化指令解码：

```py

# Simple CPU instruction interpreter. Direct instruction interpretation. 30 September 2022\. V1.0
# Class 0: no operand                   NOP
# Class 1: literal                      BEQ  3
# Class 2: register                     INC  r1
# Class 3: register,literal             LDRL r1,5
# Class 4: register,register,           MOV  r1,r2
# Class 5: register,register,literal    ADDL r1,r2,5
# Class 6: register,register,register   ADD  r1,r2,r3
# Class 7: register,[register]          LDRI r1,[r2]
import sys                              #NEW
codes = {'NOP':[0],'STOP':[0],'END':[0],'ERR':[0], 'BEQ':[1],'BNE':[1], \
         'BRA':[1],'INC':[2],'DEC':[2],'NOT':[2],'CMPL':[3],'LDRL':[3], \
         'DBNE':[3],'MOV':[4],'CMP':[4],'SUBL':[5],'ADDL':[5],'ANDL':[5], \
         'ADD':[6],'SUB':[6],'AND':[6],'OR':[6],'LDRI':[7],'STRI':[7]}
reg1  = {'r0':0,   'r1':1,   'r2':2,  'r3':3,   'r4':4,   'r5':5, \
         'r6':6,  'r7':7}               # Registers
reg2  = {'[r0]':0, '[r1]':1, '[r2]':2,'[r3]':3, '[r4]':4, \
         '[r5]':5, '[r6]':6,'[r7]':7}   # Pointer registers
symTab = {}                             # Symbol table
r = [0] * 8                             # Register set
m = [0] * 8
prog = [] * 32                          # Program memory
def equates():                          # Process directives and delete from source
    global symTab, sFile
    for i in range (0,len(sFile)):      # Deal with equates
        tempLine = sFile[i].split()
        if len(tempLine) > 2 and tempLine[1] == 'EQU':
                                        # If line > 2 tokens and second EQU
            print('SYMB' , sFile[i])
            symTab[tempLine[0]] = tempLine[2] # Put third token EQU in symbol table
    sFile = [ i for i in sFile if i.count('EQU') == 0] # Remove all lines with 'EQU'
    print('Symbol table ', symTab, '\n')
    return()
```

本节处理将指令解码为适当的任务，以便正确地使用适当的参数执行它们：

```py

def classDecode(predicate):
    lit,rD,rS1,rS2 = '',0,0,0                         # Initialize variables
    if opClass in [1]:      lit =  predicate
    if opClass in [2]:      rD  = reg1[predicate]
    if opClass in [3,4,5,6,7]:
        predicate = predicate.split(',')
        rD = reg1[predicate[0]]
    if opClass in [4,5,6]:  rS1 = reg1[predicate[1]] \
                                                # Get source reg 1 for classes 4, 5, and 6
    if opClass in [3,5]:    lit = (predicate[-1])     # Get literal for classes 3 and 5
    if opClass in [6]:      rS2 = reg1[predicate[2]]  # Get source reg 2 for class 6
    if opClass in [7]:      rS1 = reg2[predicate[1]]  # Get source pointer reg for class 7
    return(lit,rD,rS1,rS2)
```

与 TC1 不同，我们在输入上进行了少量测试，例如，内存或寄存器索引是否超出范围。这只是一个数据验证的示例：

```py

def testLine(tokens):   # Check there's a valid instruction in this line
    error = 1
    if len(tokens) == 1:
        if tokens[0] in codes: error = 0
    else:
        if (tokens[0] in codes) or (tokens[1] in codes): error = 0
    return(error)
    def testIndex():                    # Test for reg or memory index out of range
    print('rD,rS1 =', rD,rS1, 'r[rS1] =', r[rS1], 'len(m)', len(m),\
    'mnemonic =', mnemonic)
    if rD > 7 or rS1 > 7 or rS2 > 7:
        print('Register number error')
        sys.exit()                                  # Exit program on register error
    if mnemonic in ['LDRI', 'STRI']:
        if r[rS1] > len(m) - 1:
            print(' Memory index error')
            sys.exit()                              # Exit program on pointer error
    return()
def getLit(litV):                                   # Extract a literal (convert formats)
    if litV == '': return(0)                        # Return 0 if literal field empty
    if  litV in symTab:                    # Look in symbol table and get value if there
        litV = symTab[litV]                         # Read the symbol value as a string
        lit = int(litV)                             # Convert string to integer
    elif  litV[0]    == '%': lit = int(litV[1:],2)  # If % convert binary to int
    elif  litV[0:1]  == '$': lit = int(litV[1:],16) # If first symbol $, convert hex to int
    elif  litV[0]    == '-':
        lit = (-int(litV[1:]))&0xFFFF               # Deal with negative values
    elif  litV.isnumeric():  lit = int(litV)        # Convert decimal string to integer
    else:                    lit = 0                # Default value 0 (if all else fails)
    return(lit)
prgN = 'E://ArchitectureWithPython//NewIdeas_1.txt' # prgN = program name:  test file
sFile = [ ]                                         # sFile source data
with open(prgN,'r') as prgN:                        # Open it and read it
    prgN = prgN.readlines()
for i in range(0,len(prgN)):                        # First level of text-processing
    prgN[i] = prgN[i].replace('\n','')              # Remove newline code in source
    prgN[i] = ' '.join(prgN[i].split())             # Remove multiple spaces
    prgN[i] = prgN[i].strip()                       # First strip spaces
prgN = [i.split('@')[0] for i in prgN]              # Remove comment fields
while '' in prgN: prgN.remove('')                   # Remove blank lines
for i in range(0,len(prgN)):                        # Copy source to sFile: stop on END
    sFile.append(prgN[i])                           # Build new source text file sFile
    if 'END' in sFile[i]: break            # Leave on 'END' and ignore any more source text
for i in range(0,len(sFile)): print(sFile[i])
print()
equates()                                           # Deal with equates
for i in range(0,len(sFile)): print(sFile[i])
print()
for i in range(0,len(sFile)):         # We need to compile a list of labels
    label = ''                        # Give each line a default empty label
    predicate = ''                    # Create default predicate (label + mnemonic + predicate)
    tokens = sFile[i].split(' ')      # Split into separate groups
    error = testLine(tokens)          # Test for an invalid instruction
    if error == 1:                    # If error found
        print('Illegal instruction', tokens, 'at',i)
        sys.exit()                    # Exit program
    numTokens = len(tokens)           # Process this line
    if numTokens == 1: mnemonic = tokens[0]
    if numTokens > 1:
        if tokens[0][-1] == ':':
            symTab.update({tokens[0][0:-1]:i})    # Insert new value and line number
            label = tokens[0][0:-1]
            mnemonic = tokens[1]
        else: mnemonic = tokens[0]
        predicate = tokens[-1]
    opClass = codes.get(mnemonic)[0] # Use the mnemonic to read opClass from codes dictionary
    thisLine = list((i,label,mnemonic,predicate,opClass))
    prog.append(thisLine)            # Program line + label + mnemonic + predicate + opClass
print('Symbol table ', symTab, '\n') # Display symbol table for equates and line labels
```

下面是实际的指令执行循环。正如你所见，它非常紧凑：

```py

                                             # Instruction execution
run = 1
z = 0
pc = 0
while run == 1:
    thisOp = prog[pc]
    if thisOp[2] in ['STOP', 'END']: run = 0 # Terminate on STOP or END (comment on this)
    pcOld = pc
    pc = pc + 1
    mnemonic  = thisOp[2]
    predicate = thisOp[3]
    opClass   = thisOp[4]
    lit,rD,rS1,rS2 = classDecode(predicate)
    lit = getLit(lit)
    if   mnemonic == 'NOP': pass
    elif mnemonic == 'BRA': pc = lit
    elif mnemonic == 'BEQ':
        if z == 1: pc = lit
    elif mnemonic == 'BNE':
        if z == 0: pc = lit
    elif mnemonic == 'INC': r[rD] = r[rD] + 1
    elif mnemonic == 'DEC':
        z = 0
        r[rD] = r[rD] - 1
        if r[rD] == 0: z = 1
    elif mnemonic == 'NOT': r[rD] = (~r[rD])&0xFFFF  # Logical NOT
    elif mnemonic == 'CMPL':
        z = 0
        diff = r[rD] - lit
        if diff == 0: z = 1
    elif mnemonic == 'LDRL': r[rD] = lit
    elif mnemonic == 'DBNE':
        r[rD] = r[rD] - 1
        if r[rD] != 0: pc = lit
    elif mnemonic == 'MOV':  r[rD] = r[rS1]
    elif mnemonic == 'CMP':
        z = 0
        diff = r[rD] - r[rS1]
        if diff == 0: z = 1
    elif mnemonic == 'ADDL': r[rD] = r[rS1] + lit
    elif mnemonic == 'SUBL': r[rD] = r[rS1] - lit
    elif mnemonic == 'ADD':  r[rD] = r[rS1] + r[rS2]
    elif mnemonic == 'SUB':  r[rD] = r[rS1] - r[rS2]
    elif mnemonic == 'AND':  r[rD] = r[rS1] & r[rS2]
    elif mnemonic == 'OR':   r[rD] = r[rS1] | r[rS2]
    elif mnemonic == 'LDRI':
        testIndex()
        r[rD] = m[r[rS1]]
    elif mnemonic == 'STRI':
        testIndex()
        m[r[rS1]] = r[rD]
    regs = ' '.join('%04x' % b for b in r)           # Format memory location's hex
    mem  = ' '.join('%04x' % b for b in m)           # Format register's hex
    print('pc =','{:<3}'.format(pcOld),'{:<18}'.format(sFile[pcOld]),\
          'regs =',regs,'Mem =',mem,'z =',z)
```

代码执行循环，就像我们讨论的大多数模拟器一样，非常直接。当前指令被检索并解码为助记符、类别和寄存器号。程序计数器递增，助记符被呈现给一系列的`then...elif`语句。

许多指令仅用一行代码执行；例如，`ADD`通过将两个寄存器相加来实现：`r[rD] = r[rS1] + r[rS2]`。一些指令，如`compare`，需要从两个寄存器中减去，然后相应地设置状态位。

我们包括了一条相对复杂的指令，即非零减量和分支，该指令会减去一个寄存器的值，然后如果寄存器没有减到`0`，则跳转到目标地址。

在最后一节，我们将探讨 TC1 的另一种变体。

# TC1 后记 II

如果一个后记很好，两个就更好了。我们添加了这个主题的第二种变体来展示不同的做事方式。程序的大部分结构与之前相同。功能如下：

+   直接执行（回顾）

+   避免为相同的基本操作使用不同的助记符（例如，`ADD`和`ADDL`）的能力

主要的增强是处理指令和解码指令的方式。在 TC1 中，我们使用 4 位代码来定义每个指令的结构，从其参数的角度来看。当在字典中查找助记符时，它会返回一个代码，该代码给出了所需的参数。

TC1 的一个特性（或问题）是，对于指令的不同变体，我们使用不同的助记符，例如`ADD`和`ADDL`。后缀`L`告诉汇编器需要一个立即数操作数（而不是寄存器号）。在这个例子中，我们通过将指令分类来避免不同的指令格式，并使用单个助记符。每个类别定义了一种指令格式，从类别`0`（无参数的指令）到类别`9`（包含*四个*寄存器的指令）。

这个例子使用了指令的直接执行。也就是说，我们不是将指令编译成二进制然后执行二进制，而是直接从其助记符执行指令。

这种安排的后果是，一条指令可能属于多个类别；例如，`LDR`属于`三个`类别，而不是有`LDR`、`LDRL`和`LDRI`变体。当遇到一条指令时，它会与每个类别进行比对。如果助记符属于某个类别，则在决定是否找到了正确的类别之前，会检查指令的属性。

考虑`ADD`指令。我们可以写成`ADD r1,r2,5`或者`ADD r1,r2,r3`；也就是说，加到寄存器的第二个数字可以是立即数或寄存器。因此，`ADD`属于`5`类和`6`类。为了解决歧义，我们查看最后一个操作数；如果是立即数，则属于`5`类，如果是寄存器，则属于`6`类。

检查寄存器很容易，因为我们已经将寄存器放入了字典中，所以只需要检查最终操作数是否在字典中即可。考虑类别`3`：

```py

if (mnemonic in class3) and (predLen == 2) and (predicate[1] not in regList)
```

在这里，我们进行三次测试。首先，我们检查助记符是否在类别`3`中。然后，我们测试谓词长度（对于两个操作数，如`CMP r1,5`，它是`2`）。最后，我们通过确保操作数不在寄存器列表中来测试第二个操作数是否为数值。

这个实验的 Python 程序如下。

```py

# Instruction formats
# NOP             # class 0
# BRA 4           # Class 1
# INC r1          # class 2
# LDR r1,#4       # class 3
# MOV r1,r2       # class 4
# ADD r1,r2,5     # class 5
# ADD r1,r2,r3    # class 6
# LDR r1,[r2]     # class 7
# LDR r1,[r2],4   # class 8
# MLA r1,r2,r3,r4 # class 9 [r1] = [r2] + [r3] * [r3]
def getLit(lit):                        # Extract a literal
    if    lit in symTab:    literal = symTab[lit] \
                                        # Look in symbol table and get if there
    elif  lit       == '%': literal = iint(lit[1:],2) \
                                        # If first symbol is %, convert binary to integer
    elif  lit[0:1]  == '$': literal = int(lit[1:],16) \
                                        # If first symbol is $, convert hex to integer
    elif  lit[0]    == '-': literal = i(-int(lit[1:]))&0xFFFF \
                                        # Deal with negative values
    elif  lit.isnumeric():  literal = iint(lit) \
                                        # If number is a decimal string, then convert to integer
    else:                   literal = 0 # Default value 0 if all else fails
    return(literal)
regList = {'r0':0,'r1':1,'r2':2,'r3':3,'r4':4,'r5':5,'r6':6,'r7':7}
iRegList = {'[r0]':0,'[r1]':1,'[r2]':2,'[r3]':3,'[r4]':4,'[r5]':5, \
            '[r6]':6,'[r7]':7}
class0 = ['NOP','STOP','RTS']           # none
class1 = ['BRA','BEQ', 'BSR']           # register
class2 = ['INC', 'DEC']                 # register
class3 = ['LDR', 'STR','CMP','DBNE','LSL','LSR','ROR']  # register, literal
class4 = ['MOV','CMP','ADD']            # register, register Note ADD r1,r2
class5 = ['ADD','SUB']                  # register, register, literal
class6 = ['ADD','SUB']                  # register, register, register
class7 = ['LDR','STR']                  # register, pointer
class8 = ['LDR','STR']                  # register, pointer, literal
class9 = ['MLA']                        # register, register, register, register
inputSource = 0                         # Manual (keyboard) input if 0; file input if 1
singleStep  = 0                         # Select single-step mode or execute all-to-end mode
x = input('file input? type y or n ')   # Ask for file input (y) or keyboard input (any key)
if x == 'y':
    inputSource = 1
    x = input('Single step type y ')    # Type 'y' for single-step mode
    if x == 'y': singleStep = 1
    with open('C:/Users/AlanClements/Desktop/c.txt','r') as fileData:
        fileData = fileData.readlines()
    for i in range (0,len(fileData)):   # Remove leading and trailing spaces
        fileData[i] = fileData[i].strip()
r =     [0] * 8                         # Eight registers
m =     [0] * 16                        # 16 memory locations
stack = [0] * 8                         # Stack for return addresses (BSR/RTS)
prog =  []  * 64                        # Program memory
progDisp = [] * 64                      # Program for display
symTab = {}                             # Symbol table for symbolic name to value binding
run = True
pc = 0                                  # Clear program counter
sp = 7                                  # Set stack pointer to bottom of stack
while run == True:                      # Program processing loop
    predicate = []                      # Dummy
    if inputSource == 1:                # Get instruction from file
        line = fileData[pc]
    else: line = input('>> > ')         # Or input instruction from keyboard
    if line == '':
        run = False
        break
    line = ' '.join(line.split())       # Remove multiple spaces. Uses join and split
    progDisp.append(line)               # Make a copy of this line for later display
    line = line.replace(',',' ')
    line = line.split(' ')              # Split instruction into tokens
    if (len(line) > 1) and (line[0][-1] == ':'): # Look for a label (token 0 ending in :)
        label = line[0]
        symTab[line[0]] = pc            # Put a label in symTab alongside the pc
    else:
        line.insert(0,'    :')          # If no label+, insert a dummy one (for pretty printing)
    mnemonic  = line[1]                 # Get the mnemonic, second token
    predicate = line[2:]                # What's left is the predicate (registers and literal)
    prog.append(line)                   # Append the line to the program
    pc = pc + 1                         # And bump up the program counter
    progLength = pc – 1                 # Record the total number of instructions
for i in range (0,pc-1):
    print('pc =', f'{i:3}', (' ').join(prog[i])) # Print the program
print('Symbol table =', symTab, '\n')   # Display the symbol table
pc = 0
run = True
z = 0
c = 0
classNim = 10
while run == True:                      # Program execution loop
    instruction = prog[pc]
    pcOld = pc
    pc = pc + 1
    if instruction[1] == 'STOP':        # Halt on STOP instruction
        print('End of program exit')
        break
    mnemonic  = instruction[1]
    predicate = instruction[2:]
    predLen   = len(predicate)
    if (predLen > 0) and (mnemonic not in class1): rD = regList[predicate[0]]
                                        # Get rD for classes 2 to 8
```

在这个模拟器中，我们按类别而不是按助记符处理指令。这个特性意味着相同的助记符可以有不同的寻址方式，例如立即数、寄存器，甚至内存。第一个类别`0`是为没有操作数的助记符保留的，例如`NOP`。当然，这种机制使得发明一种新的操作成为可能，比如，比如`NOP 4`，它以不同的方式执行：

```py

    if mnemonic in class0:              # Deal with instructions by their group (class)
        classNum = 0
        if mnemonic == 'NOP': pass
        if mnemonic == 'RTS':           # Return from subroutine pull address off the stack
            pc = stack[sp]
            sp = sp + 1
    if mnemonic in class1:              # Class deals with branch operations so get literal
        classNum = 1
        literal = getLit(predicate[0])
        if   mnemonic == 'BRA': pc = literal
        elif mnemonic == 'BEQ':
            if z == 1: pc = literal
        elif mnemonic == 'BSR':         # Deal with subroutine call
            sp = sp - 1                 # Push return address on the stack
            stack[sp] = pc
            pc = literal
    if mnemonic in class2:                 # Class 2 increment and decrement so get register
        classNum = 2
        if mnemonic == 'INC': r[rD] = r[rD] + 1
        if mnemonic == 'DEC':
            r[rD] = r[rD] - 1
            if r[rD] == 0: z = 1           # Decrement sets z flag
            else: z = 0
    if (mnemonic in class3) and (predLen == 2) and \
    (predicate[1] not in regList):         
        classNum = 3
        literal = getLit(predicate[-1])
        if mnemonic == 'CMP':
            diff = r[rD] - literal
            if diff == 0: z = 1
            else:         z = 0
        elif mnemonic == 'LDR': r[rD] = literal
        elif mnemonic == 'STR': m[literal] = r[rD]
        elif mnemonic == 'DBNE':
            r[rD] = r[rD] - 1
            if r[rD] != 0: pc = literal        # Note we don't use z flag
        elif mnemonic == 'LSL':
            for i in range(0,literal):
                c = ((0x8000) & r[rD]) >> 16
                r[rD] = (r[rD] << 1) & 0xFFFF  # Shift left and constrain to 16 bits
        elif mnemonic == 'LSR':
            for i in range(0,literal):
                c = ((0x0001) & r[rD])
                r[rD] = r[rD] >> 1
        elif mnemonic == 'ROR':
            for i in range(0,literal):
                c = ((0x0001) & r[rD])
                r[rD] = r[rD] >> 1
                r[rD] = r[rD] | (c << 15)
    if (mnemonic in class4) and (predLen == 2) and (predicate[1]\
    in regList):                           #
        classNum = 4
        rS1 = regList[predicate[1]]        # Get second register
        if mnemonic == 'MOV':              # Move source register to destination register
           r[rD] = r[rS1]
        elif mnemonic == 'CMP':
            diff = r[rD] -  r[rS1]
            if diff == 0: z = 1
            else:         z = 0
        elif mnemonic == 'ADD':            # Add source to destination register
            r[rD] = r[rD] + r[rS1]
    if (mnemonic in class5) and (predLen == 3) and (predicate[2] not\
    in regList):
        classNum = 5                       # Class 5 is register with literal operand
        literal = getLit(predicate[2])
        rS1 = regList[predicate[1]]
        if   mnemonic == 'ADD': r[rD] = r[rS1] + literal
        elif mnemonic == 'SUB': r[rD] = r[rS1] - literal
    if (mnemonic in class6) and (predLen == 3) and (predicate[-1]\
    in regList):
        classNum = 6                       # Class 6 uses three registers
        rS1 = regList[predicate[1]]
        rS2 = regList[predicate[2]]
        if   mnemonic == 'ADD': r[rD] = r[rS1] + r[rS2]
        elif mnemonic == 'SUB': r[rD] = r[rS1] - r[rS2]
    if (mnemonic in class7) and (predLen == 2) and (predicate[1]\
    in iRegList):
        classNum = 7                       # Class 7 uses a pointer register with load and store
        pReg  = predicate[1]
        pReg1 = iRegList[pReg]
        pReg2 = r[pReg1]
        if   mnemonic == 'LDR': r[rD] = m[pReg2]
        elif mnemonic == 'STR': m[pReg2] = r[rD]
    if (mnemonic in class8) and (predLen == 3):
        classNum = 8                       # Class 8 uses a pointer register and a literal offset
        pReg  = predicate[1]
        pReg1 = iRegList[pReg]
        pReg2 = r[pReg1]
        literal = getLit(predicate[2])
        if   mnemonic == 'LDR': r[rD] = m[pReg2 + literal]
        elif mnemonic == 'STR': m[pReg2 + literal] = r[rD]
    if mnemonic in class9:                 # Class 9 demonstrates a 4-operand instruction
        classNum = 9
        if mnemonic == 'MLA':
            rS1 = regList[predicate[1]]
            rS2 = regList[predicate[2]]
            rS3 = regList[predicate[3]]
            r[rD] = r[rS1] * r[rS2] + r[rS3]
    pInst = ' '.join(instruction)          ##############
    Regs = ' '.join('%04x' % i for i in r)
    print('pc {:<2}'.format(pcOld),'Class =', classNum,      \
          '{:<20}'.format(pInst),'Regs: ', regs, 'Mem', m,   \
          'r[0] =', '{:016b}'.format(r[0]),                  \
          'c =', c, 'z =', z, '\n')
    print(progDisp[pcOld])
    if singleStep == 1: input(' >>> ')
```

之前程序的目的在于演示另一种对指令进行分类和使用操作数数量来区分指令类型的方法，例如`ADD r1,r2`和`ADD r1,r2,r3`。

# 概述

在本章中，我们介绍了 TC1 模拟器，它可以接受 TC1 汇编语言的文本文件，将其转换为机器代码，然后执行。TC1 的指令集架构接近经典的 RISC 架构，具有寄存器到寄存器的架构（即数据操作发生在寄存器的内容上）。唯一允许的内存操作是从内存（或立即数）中加载寄存器，或将寄存器存储在内存中。

模拟器有两个基本组件：一个汇编器，它将类似于`ADD r1,r2,r3`的助记符转换为 32 位二进制指令，以及一个解释器，它读取指令，提取必要的信息，然后执行指令。

TC1 的一些元素相当不寻常。提供了源代码的免费格式结构；例如，你可以编写`ADD r1,r2,r3`或`adD R1 r2 r3`，这两条指令都将被愉快地接受。为什么？首先，这是为了演示 Python 中字符串处理的使用。其次，它使得用户更容易以他们选择的格式输入。所有输入都会自动转换为大写，以使语言不区分大小写。同样，逗号或空格可以作为参数之间的分隔符。最后，去掉了表示间接寻址所需的`[]`括号。用户可以输入`LDRI r0,[r1]`或`LDRI r0,r1`。

同样，数字可以以不同的形式输入（十进制、二进制或十六进制）；例如，基数可以用*摩托罗拉格式*或*Python 格式*表示。大多数实际的汇编器不允许这种奢侈。

TC1 的早期版本要求所有地址都是数字的；如果你想跳转到行`30`，你必须写`BRA 30`。正是 Python 字典结构的非凡功能和易用性使得包含标签变得如此简单。你所要做的就是识别一个标签，将其值与其一起放入字典中，然后，每当遇到该标签时，只需在字典中查找其值即可。

我们还提供了一个示例汇编语言程序来测试 TC1，并对如何测试各种指令进行了简要讨论。

设计了 TC1 之后，我们创建了一个相当简化的版本，并将其称为 TC1mini。这个模拟器在编写指令方面没有提供相同的灵活性，它也没有一个大的指令集。它也没有将指令编码成二进制形式然后再解码并执行。它直接执行汇编指令（再次感谢 Python 的字典机制）。

在本章的结尾，我们提供了另一个简化的计算机模拟器，旨在强调计算机模拟器的结构，并提供一个修改基本设计的方法示例。

在本章的关键部分，我们介绍了 TC1 计算机模拟器并展示了其设计。我们还研究了 TC1 的变体，以帮助创建一个更完整的模拟器和汇编器的图景。在下一章中，我们将更进一步，探讨模拟器的更多方面。我们将描述几个具有不同架构的模拟器。
