# 第9章。输入/输出、物理格式和逻辑布局

在本章中，我们将看以下配方：

+   使用pathlib处理文件名

+   使用上下文管理器读写文件

+   替换文件并保留先前版本

+   使用CSV模块读取分隔文件

+   使用正则表达式读取复杂格式

+   读取JSON文档

+   读取XML文档

+   读取HTML文档

+   从DictReader升级CSV到命名元组读取器

+   从DictReader升级CSV到命名空间读取器

+   使用多个上下文读写文件

# 介绍

术语**文件**有许多含义：

+   **操作系统**（**OS**）使用文件来组织数据的字节。字节可以表示图像、一些声音样本、单词，甚至可执行程序。所有这些截然不同的内容都被简化为一组字节。应用软件理解这些字节。

有两种常见的操作系统文件：

+   块文件存在于诸如磁盘或**固态驱动器**（**SSD**）等设备上。这些文件可以按字节块读取。操作系统可以随时在文件中寻找任何特定的字节。

+   字符文件是管理设备的一种方式，比如连接到计算机的网络连接或键盘。文件被视为一系列单独的字节流，这些字节在看似随机的时间点到达。在字节流中没有办法向前或向后寻找。

+   *文件*一词还定义了Python运行时使用的数据结构。Python文件抽象包装了各种操作系统文件实现。当我们打开一个文件时，Python抽象、操作系统实现和磁盘或其他设备上的字节集之间存在绑定。

+   文件也可以被解释为Python对象的集合。从这个角度来看，文件的字节表示Python对象，如字符串或数字。文本字符串文件非常常见且易于处理。Unicode字符通常使用UTF-8编码方案编码为字节，但还有许多其他选择。Python提供了诸如`shelve`和`pickle`等模块，以将更复杂的Python对象编码为字节。

通常，我们会谈论对象是如何序列化的。当对象被写入文件时，Python对象状态信息被转换为一系列字节。反序列化是从字节中恢复Python对象的反向过程。我们也可以称之为状态的表示，因为我们通常将每个单独对象的状态与类定义分开序列化。

当我们处理文件中的数据时，我们经常需要做两个区分：

+   **数据的物理格式**：这回答了文件中的字节编码的Python数据结构是什么。字节可以是Unicode文本。文本可以表示**逗号分隔值**（**CSV**）或JSON文档。物理格式通常由Python库处理。

+   **数据的逻辑布局**：布局查看数据中的各种CSV列或JSON字段的细节。在某些情况下，列可能带有标签，或者可能有必须按位置解释的数据。这通常是我们应用程序的责任。

物理格式和逻辑布局对解释文件中的数据至关重要。我们将看一些处理不同物理格式的方法。我们还将研究如何使我们的程序与逻辑布局的某些方面分离。

# 使用pathlib处理文件名

大多数操作系统使用分层路径来标识文件。以下是一个示例文件名：

```py
 **/Users/slott/Documents/Writing/Python Cookbook/code** 

```

这个完整的路径名有以下元素：

+   前导`/`表示名称是绝对的。它从文件系统的根目录开始。在Windows中，名称前面可以有一个额外的字母，比如`C:`，以区分每个存储设备上的文件系统。Linux和Mac OS X将所有设备视为单个大文件系统。

+   `Users`，`slott`，`Documents`，`Writing`，`Python Cookbook`和`code`等名称代表文件系统的目录（或文件夹）。必须有一个顶层的`Users`目录。它必须包含`slott`子目录。对于路径中的每个名称都是如此。

+   在Windows中，操作系统使用`\`来分隔路径上的项目。Python使用`/`。Python的标准`/`会被优雅地转换为Windows路径分隔符字符；我们通常可以忽略Windows的`\`。

无法确定名称`code`代表什么类型的对象。有许多种文件系统对象。名称`code`可能是一个命名其他文件的目录。它可能是一个普通的数据文件，或者是一个指向面向流的设备的链接。还有额外的目录信息显示这是什么类型的文件系统对象。

没有前导`/`的路径是相对于当前工作目录的。在Mac OS X和Linux中，`cd`命令设置当前工作目录。在Windows中，`chdir`命令执行此操作。当前工作目录是与操作系统的登录会话相关的特性。它由shell可见。

我们如何以与特定操作系统无关的方式处理路径名？我们如何简化常见操作，使它们尽可能统一？

## 准备工作

重要的是要区分两个概念：

+   标识文件的路径

+   文件的内容

路径提供了一个可选的目录名称序列和最终的文件名。它可能通过文件扩展名提供有关文件内容的一些信息。目录包括文件名，有关文件创建时间、所有者、权限、大小以及其他详细信息。文件的内容与目录信息和名称是分开的。

通常，文件名具有后缀，可以提供有关物理格式的提示。以`.csv`结尾的文件可能是可以解释为数据行和列的文本文件。名称和物理格式之间的绑定并不是绝对的。文件后缀只是一个提示，可能是错误的。

文件的内容可能有多个名称。多个路径可以链接到单个文件。提供文件内容的目录条目是使用链接（`ln`）命令创建的。Windows使用`mklink`。这被称为**硬链接**，因为它是名称和内容之间的低级连接。

除了硬链接，我们还可以有**软链接**或**符号链接**（或连接点）。软链接是一种不同类型的文件，链接很容易被看作是对另一个文件的引用。操作系统的GUI呈现可能会将这些显示为不同的图标，并称其为别名或快捷方式以使其清晰可见。

在Python中，`pathlib`模块处理所有与路径相关的处理。该模块在路径之间进行了几个区分：

+   可能或可能不引用实际文件的纯路径

+   解析并引用实际文件的具体路径

这种区别使我们能够为我们的应用程序可能创建或引用的文件创建纯路径。我们还可以为实际存在于操作系统上的文件创建具体路径。应用程序可以解析纯路径以创建具体路径。

`pathlib`模块还区分Linux路径对象和Windows路径对象。这种区分很少需要；大多数情况下，我们不想关心路径的操作系统级细节。使用`pathlib`的一个重要原因是，我们希望处理的方式与底层操作系统无关。我们可能想要使用`PureLinuxPath`对象的情况很少。

本节中的所有迷你配方都将利用以下内容：

```py
 **>>> from pathlib import Path** 

```

我们很少需要`pathlib`中的其他类定义。

我们假设使用`argparse`来收集文件或目录名称。有关`argparse`的更多信息，请参见[第5章](text00063.html#page "第5章. 用户输入和输出")中的*使用argparse获取命令行输入*配方，*用户输入和输出*。我们将使用`options`变量，该变量具有配方处理的`input`文件名或目录名。

为了演示目的，通过提供以下`Namespace`对象显示了模拟参数解析：

```py
 **>>> from argparse import Namespace 
>>> options = Namespace( 
...     input='/path/to/some/file.csv', 
...     file1='/Users/slott/Documents/Writing/Python Cookbook/code/ch08_r09.py', 
...     file2='/Users/slott/Documents/Writing/Python Cookbook/code/ch08_r10.py', 
... )** 

```

这个`options`对象有三个模拟参数值。`input`值是一个纯路径：它不一定反映实际文件。`file1`和`file2`值反映了作者计算机上存在的具体路径。这个对象的行为与`argparse`模块创建的选项相同。

## 如何做...

我们将展示一些常见的路径名操作作为单独的迷你配方。这将包括以下操作：

+   从输入文件名制作输出文件名

+   制作多个兄弟输出文件

+   创建一个目录和一些文件

+   比较文件日期以查看哪个更新

+   删除一个文件

+   查找所有与给定模式匹配的文件

### 通过更改输入后缀来制作输出文件名

执行以下步骤，通过更改输入后缀来生成输出文件名：

1.  从输入文件名字符串创建`Path`对象。`Path`类将正确解析字符串以确定路径的元素：

```py
     **>>> input_path = Path(options.input) 
          >>> input_path 
          PosixPath('/path/to/some/file.csv')** 

    ```

在这个例子中，显示了`PosixPath`类，因为作者使用Mac OS X。在Windows机器上，该类将是`WindowsPath`。

1.  使用`with_suffix()`方法创建输出`Path`对象：

```py
     **>>> output_path = input_path.with_suffix('.out') 
          >>> output_path 
          PosixPath('/path/to/some/file.out')** 

    ```

所有的文件名解析都由`Path`类无缝处理。`with_suffix()`方法使我们不必手动解析文件名的文本。

### 制作具有不同名称的多个兄弟输出文件

执行以下步骤，制作具有不同名称的多个兄弟输出文件：

1.  从输入文件名字符串创建`Path`对象。`Path`类将正确解析字符串以确定路径的元素：

```py
     **>>> input_path = Path(options.input) 
          >>> input_path 
          PosixPath('/path/to/some/file.csv')** 

    ```

在这个例子中，显示了`PosixPath`类，因为作者使用Linux。在Windows机器上，该类将是`WindowsPath`。

1.  从文件名中提取父目录和干部。干部是没有后缀的名称：

```py
     **>>> input_directory = input_path.parent 
          >>> input_stem = input_path.stem** 

    ```

1.  构建所需的输出名称。在这个例子中，我们将在文件名后附加`_pass`。输入文件`file.csv`将产生输出`file_pass.csv`：

```py
     **>>> output_stem_pass = input_stem+"_pass" 
          >>> output_stem_pass 
          'file_pass'** 

    ```

1.  构建完整的`Path`对象：

```py
     **>>> output_path = (input_directory / output_stem_pass).with_suffix('.csv') 
          >>> output_path 
          PosixPath('/path/to/some/file_pass.csv')** 

    ```

`/`运算符从`path`组件组装一个新路径。我们需要将其放在括号中，以确保它首先执行并创建一个新的`Path`对象。`input_directory`变量具有父`Path`对象，`output_stem_pass`是一个简单的字符串。使用`/`运算符组装新路径后，使用`with_suffix()`方法来确保使用特定的后缀。

### 创建一个目录和一些文件

以下步骤是为了创建一个目录和一些文件：

1.  从输入文件名字符串创建`Path`对象。`Path`类将正确解析字符串以确定路径的元素：

```py
     **>>> input_path = Path(options.input) 
          >>> input_path 
          PosixPath('/path/to/some/file.csv')** 

    ```

在这个例子中，显示了`PosixPath`类，因为作者使用Linux。在Windows机器上，该类将是`WindowsPath`。

1.  为输出目录创建`Path`对象。在这种情况下，我们将创建一个`output`目录作为与源文件相同父目录的子目录：

```py
     **>>> output_parent = input_path.parent / "output" 
          >>> output_parent 
          PosixPath('/path/to/some/output')** 

    ```

1.  使用输出`Path`对象创建输出文件名。在这个例子中，输出目录将包含一个与输入文件同名但具有不同后缀的文件：

```py
     **>>> input_stem = input_path.stem 
          >>> output_path = (output_parent / input_stem).with_suffix('.src')** 

    ```

我们使用`/`运算符从父`Path`和基于文件名的干部的字符串组装一个新的`Path`对象。创建了`Path`对象后，我们可以使用`with_suffix()`方法为文件设置所需的后缀。

### 比较文件日期以查看哪个更新

以下是通过比较来查看更新文件日期的步骤：

1.  从输入文件名字符串创建`Path`对象。`Path`类将正确解析字符串以确定路径的元素：

```py
     **>>> file1_path = Path(options.file1) 
          >>> file1_path 
          PosixPath('/Users/slott/Documents/Writing/Python Cookbook/code/ch08_r09.py') 
          >>> file2_path = Path(options.file2) 
          >>> file2_path 
          PosixPath('/Users/slott/Documents/Writing/Python Cookbook/code/ch08_r10.py')** 

    ```

1.  使用每个`Path`对象的`stat()`方法获取文件的时间戳。这个方法返回一个`stat`对象，在`stat`对象中，该对象的`st_mtime`属性提供了文件的最近修改时间：

```py
     **>>> file1_path.stat().st_mtime 
          1464460057.0 
          >>> file2_path.stat().st_mtime 
          1464527877.0** 

    ```

这些值是以秒为单位测量的时间戳。我们可以轻松比较这两个值，看哪个更新。

如果我们想要一个对人们有意义的时间戳，我们可以使用`datetime`模块从中创建一个合适的`datetime`对象：

```py
 **>>> import datetime 
>>> mtime_1 = file1_path.stat().st_mtime 
>>> datetime.datetime.fromtimestamp(mtime_1) 
datetime.datetime(2016, 5, 28, 14, 27, 37)** 

```

我们可以使用`strftime()`方法格式化`datetime`对象，或者我们可以使用`isoformat()`方法提供一个标准化的显示。请注意，时间将隐含地应用于操作系统时间戳的本地时区偏移；根据操作系统的配置，笔记本电脑可能不会显示与创建它的服务器相同的时间，因为它们处于不同的时区。

### 删除文件

删除文件的Linux术语是**unlinking**。由于文件可能有许多链接，直到所有链接都被删除，实际数据才会被删除：

1.  从输入文件名字符串创建`Path`对象。`Path`类将正确解析字符串以确定路径的元素：

```py
          **>>> input_path = Path(options.input) 
          >>> input_path 
          PosixPath('/path/to/some/file.csv')** 

    ```

1.  使用这个`Path`对象的`unlink()`方法来删除目录条目。如果这是数据的最后一个目录条目，那么空间可以被操作系统回收：

```py
     **>>> try: 
          ...     input_path.unlink() 
          ... except FileNotFoundError as ex: 
          ...     print("File already deleted") 
          File already deleted** 

    ```

如果文件不存在，将引发`FileNotFoundError`。在某些情况下，这个异常需要用`pass`语句来消除。在其他情况下，警告消息可能很重要。也有可能缺少文件代表严重错误。

此外，我们可以使用`Path`对象的`rename()`方法重命名文件。我们可以使用`symlink_to()`方法创建新的软链接。要创建操作系统级别的硬链接，我们需要使用`os.link()`函数。

### 查找所有与给定模式匹配的文件

以下是查找所有与给定模式匹配的文件的步骤：

1.  从输入目录名称创建`Path`对象。`Path`类将正确解析字符串以确定路径的元素：

```py
     **>>> directory_path = Path(options.file1).parent 
          >>> directory_path 
          PosixPath('/Users/slott/Documents/Writing/Python Cookbook/code')** 

    ```

1.  使用`Path`对象的`glob()`方法来定位所有与给定模式匹配的文件。默认情况下，这不会递归遍历整个目录树：

```py
     **>>> list(directory_path.glob("ch08_r*.py")) 
          [PosixPath('/Users/slott/Documents/Writing/Python Cookbook/code/ch08_r01.py'),
           PosixPath('/Users/slott/Documents/Writing/Python Cookbook/code/ch08_r02.py'), 
           PosixPath('/Users/slott/Documents/Writing/Python Cookbook/code/ch08_r06.py'),
           PosixPath('/Users/slott/Documents/Writing/Python Cookbook/code/ch08_r07.py'),
           PosixPath('/Users/slott/Documents/Writing/Python Cookbook/code/ch08_r08.py'),
           PosixPath('/Users/slott/Documents/Writing/Python Cookbook/code/ch08_r09.py'),
           PosixPath('/Users/slott/Documents/Writing/Python Cookbook/code/ch08_r10.py')]** 

    ```

## 工作原理...

在操作系统内部，路径是一系列目录（文件夹是目录的一种表示）。在诸如`/Users/slott/Documents/writing`的名称中，根目录`/`包含一个名为`Users`的目录。这个目录包含一个子目录`slott`，其中包含`Documents`，其中包含`writing`。

在某些情况下，简单的字符串表示用于总结从根目录到目标目录的导航。然而，字符串表示使许多种路径操作变成复杂的字符串解析问题。

`Path`类定义简化了许多纯路径上的操作。纯`Path`可能反映实际的文件系统资源，也可能不反映。`Path`上的操作包括以下示例：

+   提取父目录，以及所有封闭目录名称的序列。

+   提取最终名称、最终名称的干部和最终名称的后缀。

+   用新后缀替换后缀或用新名称替换整个名称。

+   将字符串转换为`Path`。还可以将`Path`转换为字符串。许多操作系统函数和Python的部分偏好使用文件名字符串。

+   使用`/`运算符从现有`Path`连接的字符串构建一个新的`Path`对象。

具体的`Path`表示实际的文件系统资源。对于具体的`Path`，我们可以对目录信息进行许多额外的操作：

+   确定这是什么类型的目录项：普通文件、目录、链接、套接字、命名管道（或fifo）、块设备或字符设备。

+   获取目录详细信息，包括时间戳、权限、所有权、大小等。我们也可以修改这些内容。

+   我们可以取消链接（或删除）目录项。

几乎可以使用`pathlib`模块对文件的目录项执行任何想要的操作。少数例外情况属于`os`或`os.path`模块的一部分。

## 还有更多...

当我们在本章的其余部分查看其他与文件相关的示例时，我们将使用`Path`对象来命名文件。目标是避免尝试使用字符串来表示路径。

`pathlib`模块在Linux纯`Path`对象和Windows纯`Path`对象之间做了一个小区别。大多数情况下，我们不关心路径的操作系统级细节。

有两种情况可以帮助为特定操作系统生成纯路径：

+   如果我们在Windows笔记本电脑上进行开发，但在Linux服务器上部署Web服务，可能需要使用`PureLinuxPath`。这使我们能够在Windows开发机器上编写测试用例，反映出在Linux服务器上的实际使用意图。

+   如果我们在Mac OS X（或Linux）笔记本电脑上进行开发，但专门部署到Windows服务器，可能需要使用`PureWindowsPath`。

我们可能会有类似这样的东西：

```py
 **>>> from pathlib import PureWindowsPath 
>>> home_path = PureWindowsPath(r'C:\Users\slott') 
>>> name_path = home_path / 'filename.ini' 
>>> name_path 
PureWindowsPath('C:/Users/slott/filename.ini') 
>>> str(name_path) 
'C:\\Users\\slott\\filename.ini'** 

```

请注意，当显示`WindowsPath`对象时，`/`字符会从Windows标准化为Python表示法。使用`str()`函数检索适合Windows操作系统的路径字符串。

如果我们尝试使用通用的`Path`类，我们将得到一个适合用户环境的实现，这可能不是Windows。通过使用`PureWindowsPath`，我们已经绕过了映射到用户实际操作系统的过程。

## 另请参阅

+   在*替换文件并保留上一个版本*示例中，我们将看到如何利用`Path`的特性创建临时文件，然后将临时文件重命名以替换原始文件

+   在[第5章](text00063.html#page "第5章。用户输入和输出")的*使用argparse获取命令行输入*示例中，我们将看到获取用于创建`Path`对象的初始字符串的一种非常常见的方法

# 使用上下文管理器读写文件

许多程序将访问外部资源，如数据库连接、网络连接和操作系统文件。对于一个可靠、行为良好的程序来说，可靠而干净地释放所有外部纠缠是很重要的。

引发异常并最终崩溃的程序仍然可以正确释放资源。这包括关闭文件并确保任何缓冲数据被正确写入文件。

这对于长时间运行的服务器尤为重要。Web服务器可能会打开和关闭许多文件。如果服务器没有正确关闭每个文件，那么数据对象可能会留在内存中，减少可用于进行网络服务的空间。工作内存的丢失看起来像是一个缓慢的泄漏。最终服务器需要重新启动，降低可用性。

我们如何确保资源被正确获取和释放？我们如何避免资源泄漏？

## 准备就绪

昂贵和重要资源的一个常见例子是外部文件。已经打开进行写入的文件也是宝贵的资源；毕竟，我们运行程序来创建文件形式的有用输出。Python应用程序必须清楚地释放与文件相关的操作系统级资源。我们希望确保无论应用程序内部发生什么，缓冲区都会被刷新，文件都会被正确关闭。

当我们使用上下文管理器时，我们可以确保我们的应用程序使用的文件得到正确处理。特别是，即使在处理过程中引发异常，文件也始终会被关闭。

例如，我们将使用一个脚本来收集关于目录中文件的一些基本信息。这可以用于检测文件更改，这种技术通常用于在文件被替换时触发处理。

我们将编写一个摘要文件，其中包含文件名、修改日期、大小以及从文件中的字节计算出的校验和。然后我们可以检查目录并将其与摘要文件中的先前状态进行比较。这个函数可以准备单个文件的详细描述：

```py
    from types import SimpleNamespace 
    import datetime 
    from hashlib import md5 

    def file_facts(path): 
        return SimpleNamespace( 
            name = str(path), 
            modified = datetime.datetime.fromtimestamp( 
                path.stat().st_mtime).isoformat(), 
            size = path.stat().st_size, 
            checksum = md5(path.read_bytes()).hexdigest() 
        ) 

```

这个函数从`path`参数中的给定`Path`对象获取相对文件名。我们还可以使用`resolve()`方法获取绝对路径名。`Path`对象的`stat()`方法返回一些操作系统状态值。状态的`st_mtime`值是最后修改时间。表达式`path.stat().st_mtime`获取文件的修改时间。这用于创建完整的`datetime`对象。然后，`isoformat()`方法提供了修改时间的标准化显示。

`path.stat().st_size`的值是文件的当前大小。`path.read_bytes()`的值是文件中的所有字节，这些字节被传递给`md5`类，使用MD5算法创建校验和。结果`md5`对象的`hexdigest()`函数给出了一个足够敏感的值，可以检测到文件中的任何单字节更改。

我们想将这个应用到目录中的多个文件。如果目录正在被使用，例如，文件经常被写入，那么我们的分析程序在尝试读取被另一个进程写入的文件时可能会崩溃并出现I/O异常。

我们将使用上下文管理器来确保程序即使在罕见的崩溃情况下也能提供良好的输出。

## 如何做...

1.  我们将使用文件路径，因此重要的是导入`Path`类：

```py
            from pathlib import Path 

    ```

1.  创建一个标识输出文件的`Path`：

```py
            summary_path = Path('summary.dat') 

    ```

1.  `with`语句创建`file`对象，并将其分配给变量`summary_file`。它还将这个`file`对象用作上下文管理器：

```py
            with summary_path.open('w') as summary_file: 

    ```

现在我们可以使用`summary_file`变量作为输出文件。无论`with`语句内部引发什么异常，文件都将被正确关闭，所有操作系统资源都将被释放。

以下语句将把当前工作目录中文件的信息写入打开的摘要文件。这些语句缩进在`with`语句内部：

```py
    base = Path(".") 
    for member in base.glob("*.py"): 
        print(file_facts(member), file=summary_file) 

```

这将为当前工作目录创建一个`Path`，并将对象保存在`base`变量中。`Path`对象的`glob()`方法将生成与给定模式匹配的所有文件名。之前显示的`file_facts()`函数将生成一个具有有用信息的命名空间对象。我们可以将每个摘要打印到`summary_file`。

我们省略了将事实转换为更有用的表示。如果数据以JSON表示法序列化，可以稍微简化后续处理。

当`with`语句结束时，文件将被关闭。这将发生无论是否引发了任何异常。

## 工作原理...

上下文管理器对象和`with`语句一起工作，以管理宝贵的资源。在这种情况下，文件连接是一个相对昂贵的资源，因为它将操作系统资源与我们的应用程序绑定在一起。它也很珍贵，因为它是脚本的有用输出。

当我们写`with x:`时，对象`x`是上下文管理器。上下文管理器对象响应两种方法。这两种方法是由提供的对象上的`with`语句调用的。重要事件如下：

+   在上下文的开始时评估`x.__enter__()`。

+   在上下文结束时评估`x.__exit__(*details)`。`__exit__()`是无论上下文中是否引发了任何异常都会被保证执行的。异常细节会提供给`__exit__()`方法。如果有异常，上下文管理器可能会有不同的行为。

文件对象和其他几种对象都设计为与此对象管理器协议一起使用。

以下是描述上下文管理器如何使用的事件序列：

1.  评估`summary_path.open('w')`以创建一个文件对象。这保存在`summary_file`中。

1.  在上下文开始时评估`summary_file.__enter__()`。

1.  在`with`语句上下文中进行处理。这将向给定文件写入几行。

1.  在`with`语句结束时，评估`summary_file.__exit__()`。这将关闭输出文件，并释放所有操作系统资源。

1.  如果在`with`语句内引发了异常并且未处理，则现在重新引发该异常，因为文件已正确关闭。

文件关闭操作由`with`语句自动处理。它们总是执行，即使有异常被引发。这个保证对于防止资源泄漏至关重要。

有些人喜欢争论关于“总是”这个词：他们喜欢寻找上下文管理器无法正常工作的极少数情况。例如，有可能整个Python运行时环境崩溃；这将使所有语言保证失效。如果Python上下文管理器没有正确关闭文件，操作系统将关闭文件，但最终的数据缓冲区可能会丢失。甚至有可能整个操作系统崩溃，或者硬件停止，或者在僵尸启示录期间计算机被摧毁；上下文管理器在这些情况下也不会关闭文件。

## 还有更多...

许多数据库连接和网络连接也可以作为上下文管理器。上下文管理器保证连接被正确关闭并释放资源。

我们也可以为输入文件使用上下文管理器。最佳实践是对所有文件操作使用上下文管理器。本章中的大多数配方都将使用文件和上下文管理器。

在罕见的情况下，我们需要为一个对象添加上下文管理能力。`contextlib`包括一个名为`closing()`的函数，它将调用对象的`close()`方法。

我们可以使用这个来包装一个缺乏适当上下文管理器功能的数据库连接：

```py
    from contextlib import closing 
    with closing(some_database()) as database: 
        process(database) 

```

这假设`some_database()`函数创建了与数据库的连接。这种连接不能直接用作上下文管理器。通过将连接包装在`closing()`函数中，我们添加了必要的功能，使其成为一个适当的连接管理器对象，以确保数据库被正确关闭。

## 另请参阅

+   有关多个上下文的更多信息，请参阅*使用多个上下文读写文件*配方

# 替换文件同时保留先前的版本

我们可以利用`pathlib`的强大功能来支持各种文件名操作。在*使用pathlib处理文件名*配方中，我们看了一些管理目录、文件名和文件后缀的最常见技术。

一个常见的文件处理要求是以安全失败的方式创建输出文件。也就是说，应用程序应该保留任何先前的输出文件，无论应用程序如何失败或者在何处失败。

考虑以下情景：

1.  在时间*t*[0]，有一个有效的`output.csv`文件，是昨天使用`long_complex.py`应用程序的结果。

1.  在时间*t*[1]，我们开始运行`long_complex.py`应用程序。它开始覆盖`output.csv`文件。预计在时间*t*[3]正常完成。

1.  在时间*t*[2]，应用程序崩溃。部分`output.csv`文件是无用的。更糟糕的是，从时间*t*[0]开始的有效文件也不可用，因为它已经被覆盖。

显然，我们可以备份文件。这引入了一个额外的处理步骤。我们可以做得更好。创建一个安全失败的文件的好方法是什么？

## 准备工作

安全失败的文件输出通常意味着我们不覆盖先前的文件。相反，应用程序将使用临时名称创建一个新文件。如果文件成功创建，那么可以使用重命名操作替换旧文件。

目标是以这样的方式创建文件，以便在重命名之前的任何时间点，崩溃都会保留原始文件。在重命名之后的任何时间点，新文件都已经就位并且有效。

有几种方法可以解决这个问题。我们将展示一种使用三个单独文件的变体：

+   输出文件最终将被覆盖：`output.csv`。

+   文件的临时版本：`output.csv.tmp`。有各种命名这个文件的约定。有时会在文件名上加上`~`或`#`等额外字符，以表示它是一个临时工作文件。有时它会在`/tmp`文件系统中。

+   文件的先前版本：`name.out.old`。任何先前的`.old`文件都将在最终输出时被删除。

## 如何做到...

1.  导入`Path`类：

```py
     **>>> from pathlib import Path** 

    ```

1.  为了演示目的，我们将通过提供以下`Namespace`对象来模拟参数解析：

```py
     **>>> from argparse import Namespace 
          >>> options = Namespace( 
          ...     target='/Users/slott/Documents/Writing/Python Cookbook/code/output.csv' 
          ... )** 

    ```

我们为`target`命令行参数提供了一个模拟值。这个`options`对象的行为类似于`argparse`模块创建的选项。

1.  为所需的输出文件创建纯`Path`。这个文件还不存在，这就是为什么这是一个纯路径：

```py
     **>>> output_path = Path(options.target) 
          >>> output_path 
          PosixPath('/Users/slott/Documents/Writing/Python Cookbook/code/output.csv')** 

    ```

1.  创建一个临时输出文件的纯`Path`。这将用于创建输出：

```py
          >>> output_temp_path = output_path.with_suffix('.csv.tmp') 

    ```

1.  将内容写入临时文件。当然，这是应用程序的核心。通常相当复杂。对于这个例子，我们将它缩短为只写一个字面字符串：

```py
     **>>> output_temp_path.write_text("Heading1,Heading2\r\n355,113\r\n")** 

    ```

### 注意

这里的任何失败都不会影响原始输出文件；原始文件没有被触及。

1.  删除任何先前的`.old文件`：

```py
     **>>> output_old_path = output_path.with_suffix('.csv.old') 
          >>> try: 
          ...     output_old_path.unlink() 
          ... except FileNotFoundError as ex: 
          ...     pass # No previous file** 

    ```

### 注意

此时的任何失败都不会影响原始输出文件。

1.  如果存在文件，将其重命名为`.old文件`：

```py
     **>>> output_path.rename(output_old_path)** 

    ```

在此之后的任何失败都会保留`.old`文件。这个额外的文件可以作为恢复过程的一部分重命名。

1.  将临时文件重命名为新的输出文件：

```py
     **>>> output_temp_path.rename(output_path)** 

    ```

1.  此时，文件已经被重命名临时文件覆盖。一个`.old`文件将保留下来，以防需要将处理回滚到先前的状态。

## 它是如何工作的...

这个过程涉及三个单独的操作系统操作，一个unlink和两个重命名。这导致了一个情况，即`.old`文件需要用来恢复先前的良好状态。

这是一个显示各种文件状态的时间表。我们已经将内容标记为版本 1（先前的内容）和版本 2（修订后的内容）：

| **时间** | **操作** | **.csv.old** | **.csv** | **.csv.tmp** |
| *t* [0] |  | 版本 0 | 版本 1 |  |
| *t*[1] | 写入 | 版本 0 | 版本 1 | 进行中 |
| *t* [2] | 关闭 | 版本 0 | 版本 1 | 版本 2 |
| *t* [3] | unlink `.csv.old` |  | 版本 1 | 版本 2 |
| *t*[4] | 将`.csv`重命名为`.csv.old` | 版本 1 |  | 版本 2 |
| *t* [5] | 将`.csv.tmp`重命名为`.csv` | 版本 1 | 版本 2 |  |

虽然存在几种失败的机会，但是关于哪个文件有效没有任何歧义：

+   如果有`.csv`文件，则它是当前的有效文件

+   如果没有`.csv`文件，则`.csv.old`文件是备份副本，可用于恢复

由于这些操作都不涉及实际复制文件，因此它们都非常快速且非常可靠。

## 还有更多...

在许多情况下，输出文件涉及根据时间戳可选地创建目录。 这也可以通过`pathlib`模块优雅地处理。 例如，我们可能有一个存档目录，我们将在其中放入旧文件：

```py
    archive_path = Path("/path/to/archive") 

```

我们可能希望创建日期戳子目录以保存临时或工作文件：

```py
    import datetime 
    today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 

```

然后我们可以执行以下操作来定义工作目录：

```py
    working_path = archive_path / today 
    working_path.mkdir(parents=True, exists_ok=True) 

```

`mkdir()`方法将创建预期的目录。 包括`parents=True`参数，以确保还将创建所有父目录。 这在首次执行应用程序时非常方便。 `exists_ok=True`很方便，因此可以在不引发异常的情况下重用现有目录。

`parents=True`不是默认值。 使用`parents=False`的默认值时，当父目录不存在时，应用程序将崩溃，因为所需的文件不存在。

同样，`exists_ok=True`不是默认值。 默认情况下，如果目录存在，则会引发`FileExistsError`异常。 包括使操作在目录存在时保持安静的选项。

此外，有时适合使用`tempfile`模块创建临时文件。 该模块可以创建保证唯一的文件名。 这允许复杂的服务器进程创建临时文件，而不考虑文件名冲突。

## 另请参阅

+   在*使用pathlib处理文件名*配方中，我们研究了`Path`类的基本原理。

+   在[第11章](text00120.html#page "第11章。测试")中，*测试*，我们将研究一些编写单元测试的技术，以确保其中的部分行为正常

# 使用CSV模块读取分隔文件

常用的数据格式之一是CSV。 我们可以很容易地将逗号视为许多候选分隔符字符之一。 我们可能有一个使用`|`字符作为数据列之间分隔符的CSV文件。 这种泛化使CSV文件特别强大。

我们如何处理各种各样的CSV格式之一的数据？

## 准备就绪

文件内容的摘要称为模式。 必须区分模式的两个方面：

+   **文件的物理格式**：对于CSV，这意味着文件包含文本。 文本被组织成行和列。 将有一个行分隔符字符（或字符）； 也将有一个列分隔符字符。 许多电子表格产品将使用`,`作为列分隔符和`\r\n`字符序列作为行分隔符。 其他格式也是可能的，而且很容易更改分隔列和行的标点符号。 特定的标点符号组合称为CSV方言。

+   **文件中数据的逻辑布局**：这是存在的数据列的顺序。 处理CSV文件中的逻辑布局有几种常见情况：

+   该文件有一行标题。 这是理想的，并且与CSV模块的工作方式非常匹配。 最好的标题是适当的Python变量名。

+   文件没有标题，但列位置是固定的。 在这种情况下，我们可以在打开文件时对文件施加标题。

+   如果文件没有标题并且列位置不固定，则通常会出现严重问题。 这很难解决。 需要额外的模式信息； 例如，列定义的单独列表可以使文件可用。

+   文件有多行标题。在这种情况下，我们必须编写特殊处理来跳过这些行。我们还必须用Python替换复杂的标题为更有用的内容。

+   更困难的情况是文件不符合**第一范式**（**1NF**）。在1NF中，每行都独立于所有其他行。当文件不符合这个正常形式时，我们需要添加一个生成器函数来将数据重新排列为1NF。参见[第4章](text00048.html#page "第4章.内置数据结构-列表、集合、字典")中的*切片和切块列表*配方，*内置数据结构-列表、集合、字典*，以及[第8章](text00088.html#page "第8章.功能和响应式编程特性")中的*使用堆叠的生成器表达式*配方，*功能和响应式编程特性*，了解其他规范化数据结构的配方。

我们将查看一个相对简单的CSV文件，其中包含从帆船日志记录的实时数据。这是`waypoints.csv`文件。数据如下所示：

```py
    lat,lon,date,time 
    32.8321666666667,-79.9338333333333,2012-11-27,09:15:00 
    31.6714833333333,-80.93325,2012-11-28,00:00:00 
    30.7171666666667,-81.5525,2012-11-28,11:35:00 

```

这些数据有四列，需要重新格式化以创建更有用的信息。

## 如何做...

1.  导入`csv`模块和`Path`类：

```py
            import csv 

    ```

1.  从`pathlib`导入`Path`检查数据以确认以下特性：

+   列分隔符字符：`','`是默认值。

+   行分隔符字符：`'\r\n'`在Windows和Linux中广泛使用。这可能是Excel的一个特性，但非常普遍。Python的通用换行符功能意味着Linux标准的`'\n'`将与行分隔符一样有效。

+   单行标题的存在。如果不存在，可以单独提供此信息。

1.  创建标识文件的`Path`对象：

```py
            data_path = Path('waypoints.csv') 

    ```

1.  使用`Path`对象在`with`语句中打开文件：

```py
            with data_path.open() as data_file: 

    ```

有关with语句的更多信息，请参阅*使用上下文管理器读写文件*配方。

1.  从打开文件对象创建CSV读取器。这在`with`语句内缩进：

```py
            data_reader = csv.DictReader(data_file) 

    ```

1.  读取（和处理）各行数据。这在`with`语句内正确缩进。对于此示例，我们将只打印它们：

```py
            for row in data_reader: 
                print(row) 

    ```

输出是一系列如下的字典：

```py
    {'date': '2012-11-27', 
     'lat': '32.8321666666667', 
     'lon': '-79.9338333333333', 
     'time': '09:15:00'} 

```

由于行已转换为字典，列键不是按原始顺序排列的。如果我们使用`pprint()`来自`pprint`模块，键往往会按字母顺序排序。现在我们可以通过引用`row['date']`来处理数据。使用列名称比按位置引用列更具描述性：`row[0]`难以理解。

## 工作原理...

`csv`模块处理物理格式工作，将行与行分开，并将每行内的列分开。默认规则确保每个输入行都被视为单独的行，并且列由`","`分隔。

当我们需要使用列分隔符字符作为数据的一部分时会发生什么？我们可能会有这样的数据：

```py
    lan,lon,date,time,notes 
    32.832,-79.934,2012-11-27,09:15:00,"breezy, rainy" 
    31.671,-80.933,2012-11-28,00:00:00,"blowing ""like stink""" 

```

`notes`列在第一行中包含了`","`列分隔符字符的数据。CSV的规则允许列的值被引号括起来。默认情况下，引号字符是`"`。在这些引号字符内，列和行分隔符字符被忽略。

为了在带引号的字符串中嵌入引号字符，需要加倍。第二个示例行显示了当在带引号的列内使用引号字符时，值`"blowing "like stink""`是如何通过加倍引号字符来编码的。这些引用规则意味着CSV文件可以表示任何组合的字符，包括行和列分隔符字符。

CSV文件中的值始终为字符串。像`7331`这样的字符串值对我们来说可能看起来像一个数字，但在`csv`模块处理时，它只是文本。这使处理简单而统一，但对于人类用户来说可能有些尴尬。

一些CSV数据是从数据库或Web服务器等软件导出的。这些数据往往是最容易处理的，因为各行往往是一致地组织的。

当数据从手动准备的电子表格保存时，数据可能会显示桌面软件内部数据显示规则的怪癖。例如，通常会出现一个在桌面软件上显示为日期的数据列，在CSV文件中却显示为简单的浮点数。

日期作为数字的问题有两种解决方案。一种是在源电子表格中添加一列，以正确格式化日期为字符串。理想情况下，这是使用ISO规则完成的，以便日期以YYYY-MM-DD格式表示。另一种解决方案是将电子表格日期识别为某个纪元日期之后的秒数。纪元日期略有不同，但通常是1900年1月1日或1904年1月1日。

## 还有更多...

正如我们在*组合映射和减少转换*配方中所看到的，通常有一个包括源数据清洗和转换的处理流水线。在这个特定的例子中，没有额外需要消除的行。然而，每一列都需要转换成更有用的东西。

为了将数据转换为更有用的形式，我们将使用两部分设计。首先，我们将定义一个行级清洗函数。在这种情况下，我们将通过添加额外的类似列的值来更新行级字典对象：

```py
    import datetime 
    def clean_row(source_row): 
        source_row['lat_n']= float(source_row['lat']) 
        source_row['lon_n']= float(source_row['lon']) 
        source_row['ts_date']= datetime.datetime.strptime( 
            source_row['date'],'%Y-%m-%d').date() 
        source_row['ts_time']= datetime.datetime.strptime( 
            source_row['time'],'%H:%M:%S').time() 
        source_row['timestamp']= datetime.datetime.combine( 
            source_row['ts_date'], 
            source_row['ts_time'] 
            ) 
        return source_row 

```

我们创建了新的列值`lat_n`和`lon_n`，它们具有适当的浮点值而不是字符串。我们还解析了日期和时间值，创建了`datetime.date`和`datetime.time`对象。我们还将日期和时间合并成一个单一的有用值，即`timestamp`列的值。

一旦我们有了一个用于清理和丰富数据的行级函数，我们就可以将这个函数映射到数据源中的每一行。我们可以使用`map(clean_row, reader)`，或者我们可以编写一个体现这个处理循环的函数：

```py
    def cleanse(reader): 
        for row in reader: 
             yield clean_row(row) 

```

这可以用来从每一行提供更有用的数据：

```py
    with data_path.open() as data_file: 
        data_reader = csv.DictReader(data_file) 
        clean_data_reader = cleanse(data_reader) 
        for row in clean_data_reader: 
            pprint(row) 

```

我们注入了`cleanse()`函数来创建一个非常小的转换规则堆栈。堆栈以`data_reader`开始，只有另一个项目。这是一个很好的开始。随着应用软件扩展到更多的计算，堆栈将扩展。

这些清洁和丰富的行如下：

```py
    {'date': '2012-11-27', 
     'lat': '32.8321666666667', 
     'lat_n': 32.8321666666667, 
     'lon': '-79.9338333333333', 
     'lon_n': -79.9338333333333, 
     'time': '09:15:00', 
     'timestamp': datetime.datetime(2012, 11, 27, 9, 15), 
     'ts_date': datetime.date(2012, 11, 27), 
     'ts_time': datetime.time(9, 15)} 

```

我们添加了诸如`lat_n`和`lon_n`这样的列，它们具有适当的数值而不是字符串。我们还添加了`timestamp`，它具有完整的日期时间值，可以用于简单计算航点之间的经过时间。

## 另请参阅

+   有关处理管道或堆栈概念的更多信息，请参阅*组合映射和减少转换*配方

+   有关处理不符合1NF的CSV文件的更多信息，请参阅[第4章](text00048.html#page "第4章.内置数据结构-列表、集合、字典")的*切片和切块列表*配方，以及[第8章](text00088.html#page "第8章.功能和反应式编程特性")的*使用堆叠的生成器表达式*配方。

# 使用正则表达式阅读复杂格式

许多文件格式缺乏CSV文件的优雅规律。一个常见的文件格式，而且相当难以解析的是Web服务器日志文件。这些文件往往具有复杂的数据，没有单一的分隔符字符或一致的引用规则。

当我们在[第8章](text00088.html#page "第8章.功能和反应式编程特性")的*使用yield语句编写生成器函数*配方中查看简化的日志文件时，我们看到行如下：

```py
 **[2016-05-08 11:08:18,651] INFO in ch09_r09: Sample Message One 
[2016-05-08 11:08:18,651] DEBUG in ch09_r09: Debugging 
[2016-05-08 11:08:18,652] WARNING in ch09_r09: Something might have gone wrong** 

```

这个文件中使用了各种标点符号。`csv`模块无法处理这种复杂性。

我们如何以CSV文件的简洁简单方式处理这种类型的数据？我们能把这些不规则的行转换成更规则的数据结构吗？

## 准备好

解析具有复杂结构的文件通常涉及编写一个行为有点像`csv`模块中的`reader()`函数的函数。在某些情况下，创建一个行为像`DictReader`类的小类可能会稍微容易一些。

读取器的核心特性是一个函数，它将把一行文本转换成一个字典或一组单独的字段值。这项工作通常可以通过`re`包来完成。

在我们开始之前，我们需要开发（和调试）适当解析输入文件的每一行的正则表达式。有关更多信息，请参阅[第1章](text00014.html#page "第1章。数字、字符串和元组")中的*使用正则表达式解析字符串*配方，*数字、字符串和元组*。

对于这个例子，我们将使用以下代码。我们将定义一个模式字符串，其中包含一系列用于行的各个元素的正则表达式：

```py
 **>>> import re 
>>> pattern_text = (r'\[(\d+-\d+-\d+ \d+:\d+:\d+,\d+)\]' 
...     '\s+(\w+)' 
...     '\s+in' 
...     '\s+([\w_\.]+):' 
...     '\s+(.*)') 
>>> pattern = re.compile(pattern_text)** 

```

日期时间戳是各种数字、连字符、冒号和逗号；它被`[`和`]`包围。我们不得不使用`\[`和`\]`来转义正则表达式中`[`和`]`的正常含义。日期戳后面是一个严重级别，它是一系列字符的单次运行。字符`in`可以被忽略；没有`()`来捕获匹配的数据。模块名称是一系列字母字符，由字符类`\w`总结，还包括`_`和`.`。模块名称后面还有一个额外的`:`字符，也可以被忽略。最后，有一条消息延伸到行的末尾。我们用`()`包装了有趣的数据字符串，以便在正则表达式处理中捕获每个字符串。

请注意，我们还包括了`\s+`序列，以静默地跳过任意数量的类似空格的字符。看起来样本数据都使用单个空格作为分隔符。然而，当吸收空白时，使用`\s+`似乎是一个稍微更一般化的方法，因为它允许额外的空格。

这是这种模式的工作方式：

```py
 **>>> sample_data = '[2016-05-08 11:08:18,651] INFO in ch09_r09: Sample Message One' 
>>> match = pattern.match(sample_data) 
>>> match.groups() 
('2016-05-08 11:08:18,651', 'INFO', 'ch09_r09', 'Sample Message One')** 

```

我们提供了一行样本数据。匹配对象`match`有一个`groups()`方法，返回每个有趣的字段。我们可以使用`(?P<name>...)`来为每个捕获命名字段，而不仅仅是`(...)`，将其转换为字典。

## 如何做到这一点...

这个配方有两个部分-为单行定义一个解析函数，并使用解析函数处理每行输入。

### 定义解析函数

为定义解析函数执行以下步骤：

1.  定义编译的正则表达式对象：

```py
            import re 
            pattern_text = (r'\[(?P<date>\d+-\d+-\d+ \d+:\d+:\d+,\d+)\]' 
                '\s+(?P<level>\w+)' 
                '\s+in\s+(?P<module>[\w_\.]+):' 
                '\s+(?P<message>.*)') 
            pattern = re.compile(pattern_text) 

    ```

我们使用了`(?P<name>...)`正则表达式构造来为每个捕获的组提供名称。生成的字典将与`csv.DictReader`的结果相同。

1.  定义一个接受文本行作为参数的函数：

```py
            def log_parser(source_line): 

    ```

1.  应用正则表达式创建匹配对象。我们将其分配给`match`变量：

```py
            match = pattern.match(source_line) 

    ```

1.  如果匹配对象是`None`，则该行与模式不匹配。这行可能会被静默地跳过。在某些应用中，应该以某种方式记录它，以提供有用于调试或增强应用的信息。对于无法解析的输入行，提出异常也可能是有意义的：

```py
            if match is None: 
                raise ValueError( 
                    "Unexpected input {0!r}".format(source_line)) 

    ```

1.  返回一个有用的数据结构，其中包含来自此输入行的各个数据片段：

```py
            return match.groupdict() 

    ```

这个函数可以用来解析每一行输入。文本被转换成一个带有字段名和值的字典。

### 使用解析函数

1.  导入`csv`模块和`Path`类：

```py
            import csv 

    ```

1.  从`pathlib`导入`PathCreate`，标识文件的`Path`对象：

```py
            data_path = Path('sample.log') 

    ```

1.  使用`Path`对象在`with`语句中打开文件：

```py
            with data_path.open() as data_file: 

    ```

### 注意

有关`with`语句的更多信息，请参阅*使用上下文管理器读写文件*配方。

1.  从打开的文件对象`data_file`创建日志文件解析器。在这种情况下，我们将使用`map()`将解析器应用于源文件的每一行：

```py
            data_reader = map(log_parser, data_file) 

    ```

1.  读取（和处理）各行数据。在这个例子中，我们将只是打印它们：

```py
            for row in data_reader: 
                pprint(row) 

    ```

输出是一系列如下所示的字典：

```py
    {'date': '2016-05-08 11:08:18,651', 
     'level': 'INFO', 
     'message': 'Sample Message One', 
     'module': 'ch09_r09'} 
    {'date': '2016-05-08 11:08:18,651', 
     'level': 'DEBUG', 
     'message': 'Debugging', 
     'module': 'ch09_r09'} 
    {'date': '2016-05-08 11:08:18,652', 
     'level': 'WARNING', 
     'message': 'Something might have gone wrong', 
     'module': 'ch09_r09'} 

```

我们可以对这些字典进行比对原始文本行更有意义的处理。这使我们能够按严重程度级别过滤数据，或者基于提供消息的模块创建`Counter`。

## 工作原理...

这个日志文件是典型的第一正规形式文件。数据组织成代表独立实体或事件的行。每行具有一致数量的属性或列，每列的数据是原子的或不能进一步有意义地分解。与CSV文件不同，该格式需要复杂的正则表达式来解析。

在我们的日志文件示例中，时间戳具有许多单独的元素——年、月、日、小时、分钟、秒和毫秒，但进一步分解时间戳没有太大价值。更有帮助的是将其用作单个`datetime`对象，并从该对象中派生详细信息（如一天中的小时），而不是将各个字段组装成新的复合数据。

在复杂的日志处理应用程序中，可能会有几种消息字段的变体。可能需要使用单独的模式解析这些消息类型。当我们需要这样做时，它揭示了日志中的各行在格式和属性数量上不一致，打破了第一正规形式的假设之一。

在数据不一致的情况下，我们将不得不创建更复杂的解析器。这可能包括复杂的过滤规则，以分离出可能出现在Web服务器日志文件中的各种信息。这可能涉及解析行的一部分，以确定必须使用哪个正则表达式来解析行的其余部分。

我们一直依赖使用`map()`高阶函数。这将`log_parse()`函数应用于源文件的每一行。这种直接的简单性提供了一些保证，即创建的数据对象数量将精确匹配日志文件中的行数。

我们通常遵循*使用cvs模块读取分隔文件*配方中的设计模式，因此读取复杂日志几乎与读取简单CSV文件相同。事实上，我们可以看到主要区别在于一行代码：

```py
    data_reader = csv.DictReader(data_file) 

```

与之相比：

```py
    data_reader = map(log_parser, data_file) 

```

这种并行结构允许我们在许多输入文件格式上重用分析函数。这使我们能够创建一个可以用于许多数据源的工具库。

## 还有更多...

在读取非常复杂的文件时，最常见的操作之一是将其重写为更易处理的格式。我们经常希望以CSV格式保存数据以供以后处理。

其中一些与*使用cvs模块读取和写入多个上下文*配方类似，该配方还显示了多个打开上下文。我们将从一个文件中读取并写入另一个文件。

文件写入过程如下所示：

```py
    import csv 
    data_path = Path('sample.log') 
    target_path = data_path.with_suffix('.csv') 
    with target_path.open('w', newline='') as target_file: 
        writer = csv.DictWriter( 
            target_file, 
            ['date', 'level', 'module', 'message'] 
            ) 
        writer.writeheader() 

        with data_path.open() as data_file: 
            reader = map(log_parser, data_file) 
            writer.writerows(reader) 

```

脚本的第一部分定义了给定文件的CSV写入器。输出文件的路径`target_path`基于输入名称`data_path`。后缀从原始文件名的后缀更改为`.csv`。

该文件使用`newline=''`选项关闭换行符打开。这允许`csv.DictWriter`类插入适合所需CSV方言的换行符。

创建了一个`DictWriter`对象来写入给定文件。提供了一系列列标题。这些标题必须与用于将每行写入文件的键匹配。我们可以看到这些标题与产生数据的正则表达式的`(?P<name>...)`部分匹配。

`writeheader()`方法将列名写为输出的第一行。这使得读取文件稍微容易，因为提供了列名。CSV文件的第一行可以是一种显式模式定义，显示了存在哪些数据。

源文件如前面的配方所示打开。由于`csv`模块的写入器的工作方式，我们可以将`reader()`生成器函数提供给写入器的`writerows()`方法。`writerows()`方法将消耗`reader()`函数生成的所有数据。这将反过来消耗打开文件生成的所有行。

我们不需要编写任何显式的`for`语句来确保处理所有输入行。`writerows()`函数保证了这一点。

输出文件如下：

```py
    date,level,module,message 
    "2016-05-08 11:08:18,651",INFO,ch09_r09,Sample Message One 
    "2016-05-08 11:08:18,651",DEBUG,ch09_r09,Debugging 
    "2016-05-08 11:08:18,652",WARNING,ch09_r09,Something might have gone wrong 

```

该文件已从相当复杂的输入格式转换为更简单的CSV格式。

## 另请参阅

+   在[第8章](text00088.html#page "第8章。功能和响应式编程特性")的*使用yield语句编写生成器函数*配方中，*功能和响应式编程特性*显示了此日志格式的其他处理

+   在*使用CSV模块读取分隔文件*配方中，我们将研究此通用设计模式的其他应用

+   在*从Dictreader升级CSV到命名元组读取器*和*从Dictreader升级CSV到命名空间读取器*的配方中，我们将研究更复杂的处理技术

# 阅读JSON文档

用于序列化数据的JSON表示法非常受欢迎。有关详细信息，请参阅[http://json.org](http://json.org)。Python包括`json`模块，用于在此表示法中序列化和反序列化数据。

JSON文档被JavaScript应用广泛使用。使用JSON表示法在基于Python的服务器和基于JavaScript的客户端之间交换数据是很常见的。应用程序堆栈的这两个层通过HTTP协议发送的JSON文档进行通信。有趣的是，数据持久化层也可以使用HTTP协议和JSON表示法。

我们如何在Python中使用`json`模块解析JSON数据？

## 准备工作

我们已经收集了一些帆船比赛结果，保存在`race_result.json`中。该文件包含有关团队、航段以及各个团队完成比赛航段的顺序的信息。

在许多情况下，当船只没有启动，没有完成，或者被取消比赛资格时，会出现空值。在这些情况下，完成位置被分配一个比最后位置多一个的分数。如果有七艘船，那么团队将得到八分。这是一个相当大的惩罚。

数据具有以下模式。整个文档内有两个字段：

+   `legs`：显示起始港口和目的港口的字符串数组。

+   `teams`：包含有关每个团队的详细信息的对象数组。在每个团队对象内部，有几个数据字段：

+   `name`：团队名称字符串。

+   `position`：包含位置的整数和空值的数组。此数组中项目的顺序与legs数组中项目的顺序相匹配。

数据如下：

```py
    { 
      "teams": [ 
        { 
          "name": "Abu Dhabi Ocean Racing", 
          "position": [ 
            1, 
            3, 
            2, 
            2, 
            1, 
            2, 
            5, 
            3, 
            5 
          ] 
        }, 
        ... 
      ], 
      "legs": [ 
        "ALICANTE - CAPE TOWN", 
        "CAPE TOWN - ABU DHABI", 
        "ABU DHABI - SANYA", 
        "SANYA - AUCKLAND", 
        "AUCKLAND - ITAJA\u00cd", 
        "ITAJA\u00cd - NEWPORT", 
        "NEWPORT - LISBON", 
        "LISBON - LORIENT", 
        "LORIENT - GOTHENBURG" 
      ] 
    } 

```

我们只显示了第一个团队。在这场比赛中总共有七个团队。

JSON格式的数据看起来像一个包含列表的Python字典。Python语法和JSON语法之间的重叠可以被认为是一个幸运的巧合：它使得更容易可视化从JSON源文档构建的Python数据结构。

并非所有的JSON结构都只是Python对象。有趣的是，JSON文档中有一个空项，它映射到Python的`None`对象。含义是相似的，但语法不同。

此外，其中一个字符串包含一个Unicode转义序列`\u00cd`，而不是实际的Unicode字符Í。这是一种常用的技术，用于编码超出128个ASCII字符的字符。

## 如何做...

1.  导入`json`模块：

```py
     **>>> import json** 

    ```

1.  定义一个标识要处理的文件的`Path`对象：

```py
     **>>> from pathlib import Path 
          >>> source_path = Path("code/race_result.json")** 

    ```

`json`模块目前不能直接处理`Path`对象。因此，我们将把内容读取为一个大文本块，并处理该文本对象。

1.  通过解析JSON文档创建Python对象：

```py
     **>>> document = json.loads(source_path.read_text())** 

    ```

我们使用了`source_path.read_text()`来读取由`Path`命名的文件。我们将这个字符串提供给`json.loads()`函数进行解析。

一旦我们解析文档创建了一个Python字典，我们就可以看到各种部分。例如，字段`teams`包含了每个团队的所有结果。它是一个数组，该数组中的第0项是第一个团队。

每个团队的数据将是一个带有两个键`name`和`position`的字典。我们可以组合各种键来获得第一个团队的名称：

```py
 **>>> document['teams'][0]['name'] 
'Abu Dhabi Ocean Racing'** 

```

我们可以查看`legs`字段内的每条赛道的名称：

```py
 **>>> document['legs'][5] 
'ITAJAÍ - NEWPORT'** 

```

请注意，JSON源文件包含了`'\u00cd'`的Unicode转义序列。这被正确解析，Unicode输出显示了正确的Í字符。

## 工作原理...

JSON文档是JavaScript对象表示法中的数据结构。JavaScript程序可以轻松解析文档。其他语言必须多做一些工作来将JSON转换为本地数据结构。

一个JSON文档包含三种结构：

+   **映射到Python字典的对象**：JSON的语法类似于Python：`{"key": "value"}`。与Python不同，JSON只使用`"`作为字符串引号。JSON表示对字典值末尾的额外`,`不容忍。除此之外，这两种表示法是相似的。

+   **映射到Python列表的数组**：JSON语法使用`[item, ...]`，看起来像Python。JSON不容忍数组值末尾的额外`,`。

+   **基本值**：有五种值：字符串，数字，`true`，`false`和`null`。字符串用`"`括起来，并使用各种`\转义`序列，这与Python的类似。数字遵循浮点值的规则。其他三个值是简单的文字；这些与Python的`True`，`False`和`None`相对应。

没有其他类型的数据规定。这意味着Python程序必须将复杂的Python对象转换为更简单的表示，以便它们可以以JSON表示法进行序列化。

相反，我们经常应用额外的转换来从简化的JSON表示中重建复杂的Python对象。`json`模块有一些地方可以应用额外的处理来创建更复杂的Python对象。

## 还有更多...

一般来说，一个文件包含一个单独的JSON文档。标准没有提供一种简单的方法在单个文件中编码多个文档。例如，如果我们想要分析网站日志，JSON可能不是保留大量信息的最佳表示法。

我们经常需要解决的另外两个问题：

+   序列化复杂对象以便将它们写入文件

+   从从文件读取的文本中反序列化复杂对象

当我们将Python对象的状态表示为一串文本字符时，我们已经对对象进行了序列化。许多Python对象需要保存在文件中或传输到另一个进程。这些传输需要对象状态的表示。我们将分别查看序列化和反序列化。

### 序列化复杂数据结构

我们还可以从Python数据结构创建JSON文档。因为Python非常复杂和灵活，我们可以轻松地创建无法在JSON中表示的Python数据结构。

如果我们创建的Python对象仅限于简单的`dict`，`list`，`str`，`int`，`float`，`bool`和`None`值，那么将其序列化为JSON会得到最佳结果。如果我们小心谨慎，我们可以构建快速序列化并可以被不同语言编写的多个程序广泛使用的对象。

这些类型的值都不涉及Python`sets`或其他类定义。这意味着我们经常被迫将复杂的Python对象转换为字典以在JSON文档中表示它们。

例如，假设我们已经分析了一些数据并创建了一个结果为`Counter`对象：

```py
 **>>> import random 
>>> random.seed(1) 
>>> from collections import Counter 
>>> colors = (["red"]*18)+(["black"]*18)+(["green"]*2) 
>>> data = Counter(random.choice(colors) for _ in range(100)) 
Because this data is - effectively - a dict, we can serialie this very easily into JSON: 
>>> print(json.dumps(data, sort_keys=True, indent=2)) 
{ 
  "black": 53, 
  "green": 7, 
  "red": 40 
}** 

```

我们已经以JSON表示法转储了数据，并将键排序为顺序。这确保了一致的输出。缩进为两个将显示每个`{}`对象和每个`[]`数组在视觉上缩进，以便更容易看到文档的结构。

我们可以通过一个相对简单的操作将其写入文件：

```py
 **output_path = Path("some_path.json") 
    output_path.write_text( 
        json.dumps(data, sort_keys=True, indent=2))** 

```

当我们重新阅读这个文档时，我们将不会从JSON加载操作中得到一个`Counter`对象。我们只会得到一个字典实例。这是JSON简化为非常简单值的结果。

一个常用的数据结构，不容易序列化的是`datetime.datetime`对象。当我们尝试时会发生什么：

```py
 **>>> import datetime 
>>> example_date = datetime.datetime(2014, 6, 7, 8, 9, 10) 
>>> document = {'date': example_date}** 

```

我们创建了一个简单的文档，其中只有一个字段。字段的值是一个`datetime`实例。当我们尝试将其序列化为JSON时会发生什么？

```py
 **>>> json.dumps(document)  
Traceback (most recent call last): 
  ... 
TypeError: datetime.datetime(2014, 6, 7, 8, 9, 10) is not JSON serializable** 

```

这表明无法序列化的对象将引发`TypeError`异常。避免此异常可以通过两种方式之一来完成。我们可以在构建文档之前转换数据，或者我们可以向JSON序列化过程添加一个钩子。

一种技术是在将其序列化为JSON之前将`datetime`对象转换为字符串：

```py
 **>>> document_converted = {'date': example_date.isoformat()} 
>>> json.dumps(document_converted) 
'{"date": "2014-06-07T08:09:10"}'** 

```

这使用ISO日期格式创建一个可以序列化的字符串。读取此数据的应用程序然后可以将字符串转换回`datetime`对象。

序列化复杂数据的另一种技术是提供一个在序列化期间自动使用的默认函数。这个函数必须将一个复杂对象转换为可以安全序列化的东西。通常它会创建一个具有字符串和数值的简单字典。它还可能创建一个简单的字符串值。

```py
 **>>> def default_date(object): 
...     if isinstance(object, datetime.datetime): 
...         return example_date.isoformat() 
...     return object** 

```

我们定义了一个函数`default_date()`，它将对`datetime`对象应用特殊的转换规则。这些将被转换为可以由`json.dumps()`函数序列化的字符串对象。

我们使用`default`参数将此函数提供给`dumps()`函数，如下所示：

```py
 **>>> document = {'date': example_date} 
>>> print( 
...     json.dumps(document, default=default_date, indent=2)) 
{ 
  "date": "2014-06-07T08:09:10" 
}** 

```

在任何给定的应用程序中，我们需要扩展这个函数，以处理我们可能想要以JSON表示的更复杂的Python对象。如果有大量非常复杂的数据结构，我们通常希望有一个比精心将每个对象转换为可序列化对象更一般的解决方案。有许多设计模式可以在对象状态的序列化细节中包含类型信息。

### 反序列化复杂数据结构

在将JSON反序列化为Python对象时，还有另一个钩子可以用于将数据从JSON字典转换为更复杂的Python对象。这称为`object_hook`，它在`json.loads()`处理期间用于检查每个复杂对象，以查看是否应该从该字典创建其他内容。

我们提供的函数要么创建一个更复杂的Python对象，要么只是保持字典不变：

```py
 **>>> def as_date(object): 
...     if 'date' in object: 
...         return datetime.datetime.strptime( 
...            object['date'], '%Y-%m-%dT%H:%M:%S') 
...     return object** 

```

这个函数将检查解码的每个对象，看看对象是否有一个名为`date`的字段。如果有，整个对象的值将被替换为`datetime`对象。

我们向`json.loads()`函数提供一个函数，如下所示：

```py
 **>>> source= '''{"date": "2014-06-07T08:09:10"}''' 
>>> json.loads(source, object_hook=as_date) 
datetime.datetime(2014, 6, 7, 8, 9, 10)** 

```

这解析了一个非常小的JSON文档，符合包含日期的标准。从JSON序列化中找到的字符串值构建了生成的Python对象。

在更大的上下文中，处理日期的这个特定示例并不理想。使用单个`'date'`字段表示日期对象可能会导致使用`as_date()`函数反序列化更复杂对象时出现问题。

一个更一般的方法要么寻找一些独特的、非Python的东西，比如`'$date'`。另一个特性是确认特殊指示符是对象的唯一键。当满足这两个标准时，对象可以被特殊处理。

我们还可能希望设计我们的应用程序类，以提供额外的方法来帮助序列化。一个类可能包括一个`to_json()`方法，以统一的方式序列化对象。这种方法可能提供类信息。它可以避免序列化任何派生属性或计算属性。同样，我们可能需要提供一个静态的`from_json()`方法，用于确定给定的字典对象实际上是给定类的实例。

## 另请参阅

+   *阅读HTML文档*的示例将展示我们如何从HTML源准备这些数据

# 阅读XML文档

XML标记语言被广泛用于组织数据。有关详细信息，请参阅[http://www.w3.org/TR/REC-xml/](http://www.w3.org/TR/REC-xml/)。Python包括许多用于解析XML文档的库。

XML被称为标记语言，因为感兴趣的内容是用`<tag>`和`</tag>`构造标记的，这些标记定义了数据的结构。整个文件包括内容和XML标记文本。

因为标记与我们的文本交织在一起，所以必须使用一些额外的语法规则。为了在我们的数据中包含`<`字符，我们将使用XML字符实体引用以避免混淆。我们使用`&lt;`来在文本中包含`<`。类似地，`&gt;`代替`>`，`&amp;`代替`&`，`&quot;`也用于嵌入属性值中的`"`。

因此，文档将包含以下项目：

```py
    <team><name>Team SCA</name><position>...</position></team> 

```

大多数XML处理允许在XML中添加额外的`\n`和空格字符，以使结构更加明显：

```py
    <team> 
        <name>Team SCA</name> 
        <position>...</position> 
    </team> 

```

一般来说，内容被标签包围。整个文档形成了一个大的、嵌套的容器集合。从另一个角度来看，文档形成了一个树，根标签包含了所有其他标签及其嵌入的内容。在标签之间，有额外的内容完全是空白的，在这个例子中将被忽略。

使用正则表达式非常困难。我们需要更复杂的解析器来处理嵌套的语法。

有两个可用于解析XML-SAX和Expat的二进制库。Python包括`xml.sax`和`xml.parsers.expat`来利用这两个模块。

除此之外，在`xml.etree`包中还有一套非常复杂的工具。我们将专注于使用`ElementTree`模块来解析和分析XML文档。

我们如何使用`xml.etree`模块在Python中解析XML数据？

## 准备工作

我们已经收集了`race_result.xml`中的一些帆船比赛结果。该文件包含了关于团队、赛段以及各个团队完成每个赛段的顺序的信息。

在许多情况下，当船只没有起航，没有完成比赛或被取消资格时，会出现空值。在这些情况下，得分将比船只数量多一个。如果有七艘船，那么团队将得到八分。这是一个很大的惩罚。

根标签是`<results>`文档。这是以下模式：

+   `<legs>`标签包含命名每个赛段的单独的`<leg>`标签。赛段名称在文本中包含起始港口和终点港口。

+   `<teams>`标签包含一些`<team>`标签，其中包含每个团队的详细信息。每个团队都有用内部标签结构化的数据：

+   `<name>`标签包含团队名称。

+   `<position>`标签包含一些`<leg>`标签，其中包含给定赛段的完成位置。每个赛段都有编号，编号与`<legs>`标签中的赛段定义相匹配。

数据如下所示：

```py
    <?xml version="1.0"?> 
    <results> 
        <teams> 
                <team> 
                        <name> 
                                Abu Dhabi Ocean Racing 
                        </name> 
                        <position> 
                                <leg n="1"> 
                                        1 
                                </leg> 
                                <leg n="2"> 
                                        3 
                                </leg> 
                                <leg n="3"> 
                                        2 
                                </leg> 
                                <leg n="4"> 
                                        2 
                                </leg> 
                                <leg n="5"> 
                                        1 
                                </leg> 
                                <leg n="6"> 
                                        2 
                                </leg> 
                                <leg n="7"> 
                                        5 
                                </leg> 
                                <leg n="8"> 
                                        3 
                                </leg> 
                                <leg n="9"> 
                                        5 
                                </leg> 
                        </position> 
                </team> 
                ... 
        </teams> 
        <legs> 
        ... 
        </legs> 
    </results> 

```

我们只展示了第一个团队。在这场比赛中总共有七个团队。

在XML标记中，应用程序数据显示在两种地方。在标签之间；例如，`<name>阿布扎比海洋赛艇</name>`。标签是`<name>`，在`<name>`和`</name>`之间的文本是该标签的值。

此外，数据显示为标签的属性。例如，在`<leg n="1">`中。标签是`<leg>`；标签具有一个名为`n`的属性，其值为`1`。标签可以具有无限数量的属性。

`<leg>`标签包括作为属性`n`给出的腿编号，以及作为标签内文本给出的腿的位置。一般的方法是将重要数据放在标签内，将补充或澄清数据放在属性中。两者之间的界限非常模糊。

XML允许**混合内容模型**。这反映了XML与文本混合的情况，XML标记内外都会有文本。以下是混合内容的示例：

```py
    <p>This has <strong>mixed</strong> content.</p> 

```

一些文本位于`<p>`标签内，一些文本位于`<strong>`标签内。`<p>`标签的内容是文本和带有更多文本的标签的混合。

我们将使用`xml.etree`模块来解析数据。这涉及从文件中读取数据并将其提供给解析器。生成的文档将会相当复杂。

我们没有为我们的示例数据提供正式的模式定义，也没有提供**文档类型定义**（**DTD**）。这意味着XML默认为混合内容模式。此外，XML结构无法根据模式或DTD进行验证。

## 如何做...

1.  我们需要两个模块—`xml.etree`和`pathlib`：

```py
     **>>> import xml.etree.ElementTree as XML 
          >>> from pathlib import Path** 

    ```

我们已将`ElementTree`模块名称更改为`XML`，以使其更容易输入。通常也会将其重命名为类似`ET`的名称。

1.  定义一个定位源文档的`Path`对象：

```py
     **>>> source_path = Path("code/race_result.xml")** 

    ```

1.  通过解析源文件创建文档的内部`ElementTree`版本：

```py
     **>>> source_text = source_path.read_text(encoding='UTF-8') 
          >>> document = XML.fromstring(source_text)** 

    ```

XML解析器不太容易使用`Path`对象。我们选择从`Path`对象中读取文本，然后解析该文本。

一旦我们有了文档，就可以搜索其中的相关数据。在这个例子中，我们将使用`find()`方法来定位给定标签的第一个实例：

```py
 **>>> teams = document.find('teams') 
>>> name = teams.find('team').find('name') 
>>> name.text.strip() 
'Abu Dhabi Ocean Racing'** 

```

在这种情况下，我们定位了`<teams>`标签，然后找到该列表中第一个`<team>`标签的实例。在`<team>`标签内，我们定位了第一个`<name>`标签，以获取团队名称的值。

因为XML是混合内容模型，内容中的所有`\n`、`\t`和空格字符都会被完全保留。我们很少需要这些空白字符，因此在处理有意义的内容之前和之后使用`strip()`方法去除所有多余的字符是有意义的。

## 工作原理...

XML解析器模块将XML文档转换为基于文档对象模型的相当复杂的对象。在`etree`模块的情况下，文档将由通常表示标签和文本的`Element`对象构建。

XML还包括处理指令和注释。这些通常被许多XML处理应用程序忽略。

XML的解析器通常具有两个操作级别。在底层，它们识别事件。解析器找到的事件包括元素开始、元素结束、注释开始、注释结束、文本运行和类似的词法对象。在更高的级别上，这些事件用于构建文档的各种`元素`。

每个`Element`实例都有一个标签、文本、属性和尾部。标签是`<tag>`内的名称。属性是跟在标签名称后面的字段。例如，`<leg n="1">`标签的标签名称是`leg`，属性名为`n`。在XML中，值始终是字符串。

文本包含在标签的开始和结束之间。因此，例如`<name>SCA团队</name>`这样的标签，对于代表`<name>`标签的`Element`的`text`属性来说是`"SCA团队"`。

注意，标签还有一个尾部属性：

```py
    <name>Team SCA</name> 
    <position>...</position> 

```

在`</name>`标签关闭后和`<position>`标签打开前有一个`\n`字符。这是`<name>`标签的尾部。当使用混合内容模型时，尾部值可能很重要。在非混合内容模型中，尾部值通常是空白。

## 还有更多...

因为我们不能简单地将XML文档转换为Python字典，所以我们需要一种方便的方法来搜索文档内容。`ElementTree`模块提供了一种搜索技术，这是**XML路径语言**（**XPath**）的部分实现，用于指定XML文档中的位置。XPath表示法给了我们相当大的灵活性。

XPath查询与`find()`和`findall()`方法一起使用。以下是我们如何找到所有的名称：

```py
 **>>> for tag in document.findall('teams/team/name'): 
...      print(tag.text.strip()) 
Abu Dhabi Ocean Racing 
Team Brunel 
Dongfeng Race Team 
MAPFRE 
Team Alvimedica 
Team SCA 
Team Vestas Wind** 

```

我们已经查找了顶级的`<teams>`标签。在该标签内，我们想要`<team>`标签。在这些标签内，我们想要`<name>`标签。这将搜索所有这种嵌套标签结构的实例。

我们也可以搜索属性值。这可以方便地找到每个队伍在比赛的特定赛段上的表现。数据位于每个队伍的`<position>`标签内的`<leg>`标签中。

此外，每个`<leg>`都有一个属性值n，显示它代表比赛的哪个赛段。以下是我们如何使用这个属性从XML文档中提取特定数据的方法：

```py
 **>>> for tag in document.findall("teams/team/position/leg[@n='8']"): 
...     print(tag.text.strip()) 
3 
5 
7 
4 
6 
1 
2** 

```

这显示了每个队伍在比赛的第8赛段上的完赛位置。我们正在寻找所有带有`<leg n="8">`的标签，并显示该标签内的文本。我们必须将这些值与队名匹配，以查看Team SCA在这个赛段上第一名，而东风队在这个赛段上最后一名。

## 另请参阅

+   *阅读HTML文档*的示例展示了我们如何从HTML源准备这些数据

# 阅读HTML文档

网络上有大量使用HTML标记的内容。浏览器可以很好地呈现数据。我们如何解析这些数据，以从显示的网页中提取有意义的内容？

我们可以使用标准库`html.parser`模块，但这并不是有帮助的。它只提供低级别的词法扫描信息，但并不提供描述原始网页的高级数据结构。

我们将使用Beautiful Soup模块来解析HTML页面。这可以从**Python包索引**（**PyPI**）中获得。请参阅[https://pypi.python.org/pypi/beautifulsoup4](https://pypi.python.org/pypi/beautifulsoup4)。

这必须下载并安装才能使用。通常情况下，`pip`命令可以很好地完成这项工作。

通常情况下，这很简单，就像下面这样：

```py
 **pip install beautifulsoup4** 

```

对于Mac OS X和Linux用户，需要使用`sudo`命令来提升用户的权限：

```py
 **sudo pip install beautifulsoup4** 

```

这将提示用户输入密码。用户必须能够提升自己以获得根权限。

在极少数情况下，如果您有多个版本的Python，请确保使用匹配的pip版本。在某些情况下，我们可能需要使用以下内容：

```py
 **sudo pip3.5 install beautifulsoup4** 

```

使用与Python 3.5配套的`pip`。

## 准备工作

我们已经收集了一些帆船赛的结果，保存在`Volvo Ocean Race.html`中。这个文件包含了关于队伍、赛段以及各个队伍在每个赛段中的完成顺序的信息。它是从Volvo Ocean Race网站上抓取的，并且在浏览器中打开时看起来很棒。

HTML标记非常类似于XML。内容被`<tag>`标记包围，显示数据的结构和呈现方式。HTML早于XML，XHTML标准调和了两者。浏览器必须能够容忍旧的HTML甚至结构不正确的HTML。损坏的HTML的存在可能会使分析来自万维网的数据变得困难。

HTML页面包含大量的开销。通常有大量的代码和样式表部分，以及不可见的元数据。内容可能被广告和其他信息包围。一般来说，HTML页面具有以下整体结构：

```py
    <html> 
        <head>...</head> 
        <body>...</body> 
    </html> 

```

在`<head>`标签中将会有指向JavaScript库的链接，以及指向**层叠样式表**（**CSS**）文档的链接。这些通常用于提供交互功能和定义内容的呈现。

大部分内容在`<body>`标签中。许多网页非常繁忙，提供了一个非常复杂的内容混合。网页设计是一门复杂的艺术，内容被设计成在大多数浏览器上看起来很好。在网页上跟踪相关数据可能很困难，因为重点是人们如何看待它，而不是自动化工具如何处理它。

在这种情况下，比赛结果在HTML的`<table>`标签中，很容易找到。我们看到页面中相关内容的整体结构如下：

```py
    <table> 
        <thead> 
            <tr> 
                <th>...</th> 
                ... 
            </tr> 
        </thead> 
        <tbody> 
            <tr> 
                <td>...</td> 
                ... 
            </tr> 
            ... 
        </tbody> 
    </table> 

```

`<thead>`标签包括表格的列标题。有一个单一的表格行标签`<tr>`，包含表头`<th>`标签，其中包含内容。内容有两部分；基本显示是比赛每条腿的编号。这是标签的内容。除了显示的内容，还有一个属性值，被一个JavaScript函数使用。当光标悬停在列标题上时，这个属性值会显示。JavaScript函数会弹出腿部名称。

`<tbody>`标签包括团队名称和每场比赛的结果。表格行（`<tr>`）包含每个团队的详细信息。团队名称（以及图形和总体完成排名）显示在表格数据`<td>`的前三列中。表格数据的其余列包含比赛每条腿的完成位置。

由于帆船比赛的相对复杂性，一些表格数据单元格中包含了额外的注释。这些被包含为属性，用于提供关于单元格值原因的补充数据。在某些情况下，团队没有开始一条腿，或者没有完成一条腿，或者退出了一条腿。

这是HTML中典型的`<tr>`行：

```py
    <tr class="ranking-item"> 
        <td class="ranking-position">3</td> 
        <td class="ranking-avatar"> 
            <img src="..."> </td> 
        <td class="ranking-team">Dongfeng Race Team</td> 
        <td class="ranking-number">2</td> 
        <td class="ranking-number">2</td> 
        <td class="ranking-number">1</td> 
        <td class="ranking-number">3</td> 
        <td class="ranking-number" tooltipster data-></td> 
        <td class="ranking-number">1</td> 
        <td class="ranking-number">4</td> 
        <td class="ranking-number">7</td> 
        <td class="ranking-number">4</td> 
        <td class="ranking-number total">33<span class="asterix">*</span></td> 
    </tr> 

```

`<tr>`标签具有一个类属性，用于定义此行的样式。CSS为这个数据类提供了样式规则。此标签上的`class`属性帮助我们的数据收集应用程序定位相关内容。

`<td>`标签也有类属性，用于定义数据单元格的样式。在这种情况下，类信息澄清了单元格内容的含义。

其中一个单元格没有内容。该单元格具有`data-title`属性。这被一个JavaScript函数用来在单元格中显示额外信息。

## 如何做...

1.  我们需要两个模块：bs4和pathlib：

```py
     **>>> from bs4 import BeautifulSoup 
          >>> from pathlib import Path** 

    ```

我们只从`bs4`模块中导入了`BeautifulSoup`类。这个类将提供解析和分析HTML文档所需的所有功能。

1.  定义一个命名源文档的`Path`对象：

```py
     **>>> source_path = Path("code/Volvo Ocean Race.html")** 

    ```

1.  从HTML内容创建soup结构。我们将把它分配给一个变量`soup`：

```py
     **>>> with source_path.open(encoding='utf8') as source_file: 
          ...     soup = BeautifulSoup(source_file, 'html.parser')** 

    ```

我们使用上下文管理器来访问文件。作为替代，我们可以简单地使用`source_path.read_text(encodig='utf8')`来读取内容。这与为`BeautifulSoup`类提供一个打开的文件一样有效。

变量`soup`中的soup结构可以被处理，以定位各种内容。例如，我们可以提取腿部细节如下：

```py
    def get_legs(soup) 
        legs = [] 
        thead = soup.table.thead.tr 
        for tag in thead.find_all('th'): 
            if 'data-title' in tag.attrs: 
                leg_description_text = clean_leg(tag.attrs['data-title']) 
                legs.append(leg_description_text) 
        return legs 

```

表达式`soup.table.thead.tr`将找到第一个`<table>`标签。在其中，第一个`<thead>`标签；在其中，第一个`<tr>`标签。我们将这个`<tr>`标签分配给一个名为`thead`的变量，可能会误导。然后我们可以使用`findall()`来定位容器内的所有`<th>`标签。

我们将检查每个标签的属性，以定位`data-title`属性的值。这将包含腿部名称信息。腿部名称内容如下：

```py
    <th tooltipster data->LEG 1</th> 

```

`data-title`属性值包括值内的一些额外的HTML标记。这不是HTML的标准部分，`BeautifulSoup`解析器不会在属性值内查找这个HTML。

我们有一小段HTML需要解析，所以我们可以创建一个小的`soup`对象来解析这段文本：

```py
    def clean_leg(text): 
        leg_soup = BeautifulSoup(text, 'html.parser') 
        return leg_soup.text 

```

我们从`data-title`属性的值创建一个小的`BeautifulSoup`对象。这个soup将包含关于标签`<strong>`和文本的信息。我们使用文本属性来获取所有文本，而不包含任何标签信息。

## 它是如何工作的...

`BeautifulSoup`类将HTML文档转换为基于**文档对象模型**（**DOM**）的相当复杂的对象。结果结构将由`Tag`、`NavigableString`和`Comment`类的实例构建。

通常，我们对包含网页内容的标签感兴趣。这些是`Tag`和`NavigableString`类的对象。

每个`Tag`实例都有一个名称、字符串和属性。名称是`<`和`>`之间的单词。属性是跟在标签名称后面的字段。例如，`<td class="ranking-number">1</td>`的标签名称是`td`，有一个名为`class`的属性。值通常是字符串，但在一些情况下，值可以是字符串列表。`Tag`对象的字符串属性是标签包围的内容；在这种情况下，它是一个非常短的字符串`1`。

HTML是一个混合内容模型。这意味着标签可以包含除可导航文本之外的子标签。文本是混合的，它可以在任何子标签内部或外部。当查看给定标签的子级时，将会有一系列标签和文本自由混合。

HTML的最常见特性之一是包含换行字符的可导航文本小块。当我们有这样的一段代码时：

```py
    <tr> 
        <td>Data</td> 
    </tr> 

```

`<tr>`标签内有三个子元素。以下是该标签的子元素的显示：

```py
 **>>> example = BeautifulSoup(''' 
...     <tr> 
...         <td>data</td> 
...     </tr> 
... ''', 'html.parser') 
>>> list(example.tr.children) 
['\n', <td>data</td>, '\n']** 

```

两个换行字符是`<td>`标签的同级，并且被解析器保留。这是包围子标签的可导航文本。

`BeautifulSoup`解析器依赖于另一个更低级的过程。较低级的过程可以是内置的`html.parser`模块。也有其他可安装的替代方案。`html.parser`是最容易使用的，覆盖了最常见的用例。还有其他可用的替代方案，Beautiful Soup文档列出了可以用来解决特定网页解析问题的其他低级解析器。

较低级的解析器识别事件；这些事件包括元素开始、元素结束、注释开始、注释结束、文本运行和类似的词法对象。在更高的层次上，这些事件用于构建Beautiful Soup文档的各种对象。

## 还有更多...

Beautiful Soup的`Tag`对象表示文档结构的层次结构。标签之间有几种导航方式：

+   除了特殊的根`[document]`容器，所有标签都会有一个父级。顶级`<html>`标签通常是根文档容器的唯一子级。

+   `parents`属性是一个给定标签的所有父级的生成器。这是通过层次结构到达给定标签的路径。

+   所有`Tag`对象都可以有子级。一些标签，如`<img/>`和`<hr/>`没有子级。`children`属性是一个生成器，产生标签的子级。

+   具有子级的标签可能有多个级别的标签。例如，整个`<html>`标签具有整个文档作为后代。`children`属性具有直接子级；`descendants`属性生成所有子级的子级。

+   标签也可以有兄弟标签，这些标签位于同一个容器内。由于标签有一个定义好的顺序，所以有一个`next_sibling`和`previous_sibling`属性来帮助遍历标签的同级。

在某些情况下，文档将具有一般直观的组织结构，通过`id`属性或`class`属性的简单搜索将找到相关数据。以下是对给定结构的典型搜索：

```py
 **>>> ranking_table = soup.find('table', class_="ranking-list")** 

```

请注意，我们必须在Python查询中使用`class_`来搜索名为`class`的属性。鉴于整个文档，我们正在搜索任何`<table class="ranking-list">`标签。这将在网页中找到第一个这样的表。由于我们知道只会有一个这样的表，这种基于属性的搜索有助于区分网页上的任何其他表格数据。

这是这个`<table>`标签的父级：

```py
 **>>> list(tag.name for tag in ranking_table.parents) 
['section', 'div', 'div', 'div', 'div', 'body', 'html', '[document]']** 

```

我们只显示了上面给定的`<table>`的每个父级标签的标签名。请注意，有四个嵌套的`<div>`标签包裹着包含`<table>`的`<section>`。这些`<div>`标签中的每一个可能都有一个不同的class属性，以正确定义内容和内容样式。

`[document]`是包含各种标签的`BeautifulSoup`容器。这是以独特的方式显示出来，以强调它不是一个真正的标签，而是顶级`<html>`标签的容器。

## 另请参阅

+   *读取JSON文档*和*读取XML文档*配方都使用类似的数据。示例数据是通过使用这些技术从HTML页面抓取而为它们创建的。

# 从DictReader升级CSV到命名元组读取器

当我们从CSV格式文件中读取数据时，对于结果数据结构有两种一般选择：

+   当我们使用`csv.reader()`时，每一行都变成了一个简单的列值列表。

+   当我们使用`csv.DictReader`时，每一行都变成了一个字典。默认情况下，第一行的内容成为行字典的键。另一种方法是提供一个值列表，将用作键。

在这两种情况下，引用行内的数据都很笨拙，因为它涉及相当复杂的语法。当我们使用`csv`读取器时，我们必须使用`row[2]`：这个语义完全晦涩。当我们使用`DictReader`时，我们可以使用`row['date']`，这不那么晦涩，但仍然需要大量输入。

在一些现实世界的电子表格中，列名是不可能的长字符串。很难处理`row['Total of all locations excluding franchisees']`。

我们可以做些什么来用更简单的东西替换复杂的语法？

## 准备工作

改善处理电子表格的程序的可读性的一种方法是用`namedtuple`对象替换列的列表。这提供了由`namedtuple`定义的易于使用的名称，而不是`.csv`文件中可能杂乱无章的列名。

更重要的是，它允许更好的语法来引用各个列。除了`row[0]`，我们还可以使用`row.date`来引用名为`date`的列。

列名（以及每列的数据类型）是给定数据文件的模式的一部分。在一些CSV文件中，列标题的第一行是文件的模式。这个模式是有限的，它只提供属性名称；数据类型是未知的，必须被视为字符串处理。

这指出了在电子表格的行上强加外部模式的两个原因：

+   我们可以提供有意义的名称

+   我们可以在必要时执行数据转换

我们将查看一个相对简单的CSV文件，其中记录了一艘帆船的日志中的一些实时数据。这是`waypoints.csv`文件，数据如下：

```py
    lat,lon,date,time 
    32.8321666666667,-79.9338333333333,2012-11-27,09:15:00 
    31.6714833333333,-80.93325,2012-11-28,00:00:00 
    30.7171666666667,-81.5525,2012-11-28,11:35:00 

```

数据有四列。其中两列是航点的纬度和经度。它有一个包含日期和时间的列。这并不理想，我们将分别查看各种数据清洗步骤。

在这种情况下，列标题恰好是有效的Python变量名。这很少见，但可能会导致略微简化。我们将在下一节中看看其他选择。

最重要的一步是将数据收集为`namedtuples`。

## 如何做...

1.  导入所需的模块和定义。在这种情况下，它们将来自`collections`，`csv`和`pathlib`：

```py
            from collections import namedtuple 
            from pathlib import Path 
            import csv 

    ```

1.  定义与实际数据匹配的`namedtuple`。在这种情况下，我们称之为`Waypoint`并为四列数据提供名称。在这个例子中，属性恰好与列名匹配；这不是必须的：

```py
            Waypoint = namedtuple('Waypoint', ['lat', 'lon', 'date', 'time'])
    ```

1.  定义引用数据的`Path`对象：

```py
            waypoints_path = Path('waypoints.csv') 

    ```

1.  为打开的文件创建处理上下文：

```py
            with waypoints_path.open() as waypoints_file: 

    ```

1.  为数据定义一个CSV读取器。我们将其称为原始读取器。从长远来看，我们将遵循[第8章](text00088.html#page "第8章。功能和响应式编程特性")中的*使用堆叠的生成器表达式*配方，*功能和响应式编程特性*和[第8章](text00088.html#page "第8章。功能和响应式编程特性")中的*使用一堆生成器表达式*配方，*功能和响应式编程特性*来清理和过滤数据：

```py
            raw_reader = csv.reader(waypoints_file) 

    ```

1.  定义一个生成器，从输入数据的元组构建`Waypoint`对象：

```py
            waypoints_reader = (Waypoint(*row) for row in raw_reader) 

    ```

现在我们可以使用`waypoints_reader`生成器表达式来处理行：

```py
    for row in waypoints_reader: 
        print(row.lat, row.lon, row.date, row.time) 

```

`waypoints_reader`对象还将提供标题行，我们希望忽略它。我们将在下一节讨论过滤和转换。

表达式`(Waypoint(*row) for row in raw_reader)`会将`row`元组的每个值扩展为`Waypoint`函数的位置参数值。这是因为CSV文件中的列顺序与`namedtuple`定义中的列顺序匹配。

这种构造也可以使用`itertools`模块来执行。`starmap()`函数可以用作`starmap(Waypoint, raw_reader)`。这也将使`raw_reader`中的每个元组扩展为`Waypoint`函数的位置参数。请注意，我们不能使用内置的`map()`函数。`map()`函数假定函数接受单个参数值。我们不希望每个四项`row`元组都被用作`Waypoint`函数的唯一参数。我们需要将四个项目拆分为四个位置参数值。

## 它是如何工作的...

这个配方有几个部分。首先，我们使用`csv`模块对数据的行和列进行基本解析。我们利用了*使用cvs模块读取分隔文件*配方来处理数据的物理格式。

其次，我们定义了一个`namedtuple()`，为我们的数据提供了一个最小的模式。这并不是非常丰富或详细。它提供了一系列列名。它还简化了访问特定列的语法。

最后，我们将`csv`读取器包装在一个生成器函数中，为每一行构建`namedtuple`对象。这对默认处理来说是一个微小的改变，但它会导致后续编程的更好风格。

现在我们可以使用`row.date`而不是`row[2]`或`row['date']`来引用特定的列。这是一个可以简化复杂算法呈现的小改变。

## 还有更多...

处理输入的初始示例存在两个额外的问题。首先，标题行与有用的数据行混在一起；这个标题行需要通过某种过滤器被拒绝。其次，数据都是字符串，需要进行一些转换。我们将通过扩展配方来解决这两个问题。

有两种常见的技术可以丢弃不需要的标题行：

+   我们可以使用显式迭代器并丢弃第一项。总体思路如下：

```py
            with waypoints_path.open() as waypoints_file: 
                raw_reader = csv.reader(waypoints_file) 
                waypoints_iter = iter(waypoints_reader) 
                next(waypoints_iter)  # The header 
                for row in waypoints_iter: 
                    print(row) 

    ```

这个片段展示了如何从原始CSV读取器创建一个迭代器对象`waypoints_iter`。我们可以使用`next()`函数从这个读取器中跳过一个项目。剩下的项目可以用来构建有用的数据行。我们也可以使用`itertools.islice()`函数来实现这一点。

+   我们可以编写一个生成器或使用`filter()`函数来排除选定的行：

```py
            with waypoints_path.open() as waypoints_file: 
                raw_reader = csv.reader(waypoints_file) 
                skip_header = filter(lambda row: row[0] != 'lat', raw_reader) 
                waypoints_reader = (Waypoint(*row) for row in skip_header) 
                for row in waypoints_reader: 
                    print(row) 

    ```

这个例子展示了如何从原始CSV读取器创建过滤生成器`skip_header`。过滤器使用一个简单的表达式`row[0] != 'lat'`来确定一行是否是标题或者有用的数据。只有有用的行通过了这个过滤器。标题行被拒绝了。

我们还需要做的另一件事是将各种数据项转换为更有用的值。我们将遵循[第8章](text00088.html#page "第8章。功能和反应式编程特性")中的*Simplifying complex algorithms with immutable data structures*配方的例子，从原始输入数据构建一个新的`namedtuple`：

```py
    Waypoint_Data = namedtuple('Waypoint_Data', ['lat', 'lon', 'timestamp']) 

```

在大多数项目的这个阶段，很明显`Waypoint namedtuple`的原始名称选择不当。代码需要重构以更改名称以澄清原始`Waypoint`元组的角色。随着设计的演变，这种重命名和重构将多次发生。根据需要重命名是很重要的。我们不会在这里进行重命名：我们将把它留给读者重新设计名称。

为了进行转换，我们需要一个处理单个`Waypoint`字段的函数。这将创建更有用的值。它涉及对纬度和经度值使用`float()`。它还需要对日期值进行一些仔细的解析。

这是处理单独的日期和时间的第一部分。这是两个lambda对象-只有一个单一表达式的小函数，将日期或时间字符串转换为日期或时间值：

```py
    import datetime 
    parse_date = lambda txt: datetime.datetime.strptime(txt, '%Y-%m-%d').date() 
    parse_time = lambda txt: datetime.datetime.strptime(txt, '%H:%M:%S').time() 

```

我们可以使用这些来从原始`Waypoint`对象构建一个新的`Waypoint_data`对象：

```py
    def convert_waypoint(waypoint): 
        return Waypoint_Data( 
            lat = float(waypoint.lat), 
            lon = float(waypoint.lon), 
            timestamp = datetime.datetime.combine( 
                parse_date(waypoint.date), 
                parse_time(waypoint.time) 
            )     
        ) 

```

我们应用了一系列函数，从现有的数据结构构建了一个新的数据结构。纬度和经度值使用`float()`函数进行转换。日期和时间值使用`parse_date`和`parse_time` lambda与`datetime`类的`combine()`方法转换为`datetime`对象。

这个函数允许我们为源数据构建一个更完整的处理步骤堆栈：

```py
    with waypoints_path.open() as waypoints_file: 
        raw_reader = csv.reader(waypoints_file) 
        skip_header = filter(lambda row: row[0] != 'lat', raw_reader) 
        waypoints_reader = (Waypoint(*row) for row in skip_header) 
        waypoints_data_reader = (convert_waypoint(wp) for wp in waypoints_reader) 
        for row in waypoints_data_reader: 
            print(row.lat, row.lon, row.timestamp) 

```

原始读取器已经补充了一个跳过标题的过滤函数，一个用于创建`Waypoint`对象的生成器，以及另一个用于创建`Waypoint_Data`对象的生成器。在`for`语句的主体中，我们有一个简单易用的数据结构，具有愉快的名称。我们可以引用`row.lat`而不是`row[0]`或`row['lat']`。

请注意，每个生成器函数都是惰性的，它不会获取比产生一些输出所需的更多输入。这个生成器函数堆栈使用的内存很少，可以处理无限大小的文件。

## 参见

+   *从dict reader升级CSV到namespace reader*配方使用了可变的`SimpleNamespace`数据结构

# 从DictReader升级CSV到命名空间读取器

当我们从CSV格式文件中读取数据时，我们有两种一般的选择结果数据结构：

+   当我们使用`csv.reader()`时，每一行都变成了一个简单的列值列表。

+   当我们使用`csv.DictReader`时，每一行都变成了一个字典。默认情况下，第一行的内容成为行字典的键。我们还可以提供一个值列表，将用作键。

在这两种情况下，引用行内的数据都很笨拙，因为它涉及相当复杂的语法。当我们使用读取器时，我们必须使用`row[0]`，这个语义完全晦涩。当我们使用`DictReader`时，我们可以使用`row['date']`，这不那么晦涩，但是要输入很多。

在一些现实世界的电子表格中，列名是不可能很长的字符串。很难使用`row['Total of all locations excluding franchisees']`。

我们可以用什么简单的方法来替换复杂的语法？

## 准备工作

列名（以及每列的数据类型）是我们数据的模式。列标题是嵌入在CSV数据的第一行中的模式。这个模式只提供了属性名称；数据类型是未知的，必须被视为字符串。

这指出了在电子表格的行上强加外部模式的两个原因：

+   我们可以提供有意义的名称。

+   我们可以在必要时进行数据转换。

我们还可以使用模式来定义数据质量和清洗处理。这可能变得非常复杂。我们将限制使用模式来提供列名和数据转换。

我们将查看一个相对简单的CSV文件，其中记录了一艘帆船日志的实时数据。这是`waypoints.csv`文件。数据看起来像下面这样：

```py
    lat,lon,date,time 
    32.8321666666667,-79.9338333333333,2012-11-27,09:15:00 
    31.6714833333333,-80.93325,2012-11-28,00:00:00 
    30.7171666666667,-81.5525,2012-11-28,11:35:00 

```

这个电子表格有四列。其中两列是航点的纬度和经度。它有一个包含日期和时间的列。这并不理想，我们将分别查看各种数据清洗步骤。

在这种情况下，列标题是有效的Python变量名。这导致了处理中的一个重要简化。在没有列名或列名不是Python变量的情况下，我们将不得不应用从列名到首选属性名的映射。

## 如何做...

1.  导入所需的模块和定义。在这种情况下，它将是来自`types`，`csv`和`pathlib`：

```py
            from types import SimpleNamespace 
            from pathlib import Path 

    ```

1.  导入`csv`并定义一个指向数据的`Path`对象：

```py
            waypoints_path = Path('waypoints.csv') 

    ```

1.  为打开的文件创建处理上下文：

```py
            with waypoints_path.open() as waypoints_file: 

    ```

1.  为数据定义一个CSV读取器。我们将其称为原始读取器。从长远来看，我们将遵循[第8章](text00088.html#page "第8章。功能和响应式编程特性")中的*使用堆叠的生成器表达式*，*功能和响应式编程特性*并使用多个生成器表达式来清理和过滤数据：

```py
            raw_reader = csv.DictReader(waypoints_file) 

    ```

1.  定义一个生成器，将这些字典转换为`SimpleNamespace`对象：

```py
            ns_reader = (SimpleNamespace(**row) for row in raw_reader) 

    ```

这使用了通用的`SimpleNamespace`类。当我们需要使用更具体的类时，我们可以用应用程序特定的类名替换`SimpleNamespace`。该类的`__init__`必须使用与电子表格列名匹配的关键字参数。

现在我们可以从这个生成器表达式中处理行：

```py
    for row in ns_reader: 
        print(row.lat, row.lon, row.date, row.time) 

```

## 它是如何工作的...

这个食谱有几个部分。首先，我们使用了`csv`模块来对数据的行和列进行基本解析。我们利用了*使用cvs模块读取分隔文件*的方法来处理数据的物理格式。CSV格式的想法是在每一行中有逗号分隔的文本列。有规则可以使用引号来允许列内的数据包含逗号。所有这些规则都在`csv`模块中实现，省去了我们编写解析器的麻烦。

其次，我们将`csv`读取器包装在一个生成器函数中，为每一行构建一个`SimpleNamespace`对象。这是对默认处理的微小扩展，但可以使后续编程风格更加优雅。现在我们可以使用`row.date`来引用特定列，而不是`row[2]`或`row['date']`。这是一个小改变，可以简化复杂算法的呈现。

## 还有更多...

我们可能有两个额外的问题要解决。是否需要这些取决于数据和数据的用途：

+   我们如何处理不是合适的Python变量的电子表格名称？

+   我们如何将数据从文本转换为Python对象？

事实证明，这两个需求都可以通过一个逐行转换数据的函数来优雅处理，并且还可以处理任何必要的列重命名：

```py
    def make_row(source): 
        return SimpleNamespace( 
            lat = float(source['lat']), 
            lon = float(source['lon']), 
            timestamp = make_timestamp(source['date'], source['time']), 
        )     

```

这个函数实际上是原始电子表格的模式定义。这个函数中的每一行提供了几个重要的信息：

+   `SimpleNamespace`中的属性名称

+   从源数据转换

+   映射到最终结果的源列名称

目标是定义任何必要的辅助或支持函数，以确保转换函数的每一行与所示的行类似。该函数的每一行都是结果列的完整规范。作为额外的好处，每一行都是用Python符号表示的。

这个函数可以替换`ns_reader`语句中的`SimpleNamespace`。现在所有的转换工作都集中在一个地方：

```py
    ns_reader = (make_row(row) for row in raw_reader) 

```

这一行变换函数依赖于`make_timestamp()`函数。该函数将两个源列转换为一个结果为`datetime`对象的函数。该函数如下所示：

```py
    import datetime 
    make_date = lambda txt: datetime.datetime.strptime( 
        txt, '%Y-%m-%d').date() 
    make_time = lambda txt: datetime.datetime.strptime( 
        txt, '%H:%M:%S').time() 

    def make_timestamp(date, time): 
        return datetime.datetime.combine( 
                make_date(date), 
                make_time(time) 
             ) 

```

`make_timestamp()`函数将时间戳的创建分为三个部分。前两部分非常简单，只需要一个lambda对象。这些是从文本转换为`datetime.date`或`datetime.time`对象。每个转换使用`strptime()`方法来解析日期或时间字符串，并返回适当的对象类。

第三部分也可以是lambda，因为它也是一个单一表达式。但是，它是一个很长的表达式，将其包装为`def`语句似乎更清晰一些。这个表达式使用`datetime`的`combine()`方法将日期和时间组合成一个对象。

## 另请参阅

+   *从字典读取器升级CSV到命名元组读取器*的方法是使用不可变的`namedtuple`数据结构，而不是`SimpleNamespace`

# 使用多个上下文来读写文件

通常需要将数据从一种格式转换为另一种格式。例如，我们可能有一个复杂的网络日志，我们希望将其转换为更简单的格式。

请参阅*使用正则表达式读取复杂格式*食谱以了解复杂的网络日志格式。我们希望只进行一次解析。

之后，我们希望使用更简单的文件格式，更像*从字典读取器升级CSV到命名元组读取器*或*从字典读取器升级CSV到命名空间读取器*的格式。CSV格式的文件可以使用`csv`模块进行读取和解析，简化物理格式的考虑。

我们如何从一种格式转换为另一种格式？

## 准备工作

将数据文件从一种格式转换为另一种格式意味着程序需要有两个打开的上下文：一个用于读取，一个用于写入。Python使这变得容易。使用`with`语句上下文确保文件被正确关闭，并且所有相关的操作系统资源都被完全释放。

我们将研究总结许多网络日志文件的常见问题。源代码格式与[第8章](text00088.html#page "第8章. 函数式和响应式编程特性")中*使用yield语句编写生成器函数*食谱中看到的格式相同，也与本章中*使用正则表达式读取复杂格式*食谱中看到的格式相同。行如下所示：

```py
    [2016-05-08 11:08:18,651] INFO in ch09_r09: Sample Message One
    [2016-05-08 11:08:18,651] DEBUG in ch09_r09: Debugging
    [2016-05-08 11:08:18,652] WARNING in ch09_r09: Something might have gone wrong

```

这些很难处理。需要复杂的正则表达式来解析它们。对于大量的数据，它也相当慢。

以下是行中各个元素的正则表达式模式：

```py
    import re 
    pattern_text = (r'\[(?P<date>\d+-\d+-\d+ \d+:\d+:\d+,\d+)\]' 
        '\s+(?P<level>\w+)' 
        '\s+in\s+(?P<module>[\w_\.]+):' 
        '\s+(?P<message>.*)') 
    pattern = re.compile(pattern_text) 

```

这个复杂的正则表达式有四个部分：

+   日期时间戳用`[ ]`括起来，包含各种数字、连字符、冒号和逗号。它将被捕获并通过`?P<date>`前缀分配名称`date`给`()`组。

+   严重级别，这是一系列字符。这是通过下一个`()`组的`?P<level>`前缀捕获并命名为level。

+   该模块是一个包括`_`和`.`的字符序列。它被夹在`in`和`:`之间。被分配名称`module`。

+   最后，有一条消息延伸到行尾。这是通过最后一个`()`内的`?P<message>`分配给消息的。

模式还包括空白符的运行，`\s+`，它们不在任何`()`组中捕获。它们被静默忽略。

当我们使用这个正则表达式创建一个`match`对象时，该`match`对象的`groupdict()`方法将生成一个包含每行名称和值的字典。这与`csv`读取器的工作方式相匹配。它提供了处理复杂数据的通用框架。

我们将在迭代日志数据行的函数中使用这个。该函数将应用正则表达式，并生成组字典：

```py
    def extract_row_iter(source_log_file): 
        for line in source_log_file: 
            match = log_pattern.match(line) 
            if match is None: 
                # Might want to write a warning 
                continue 
            yield match.groupdict() 

```

这个函数查看给定输入文件中的每一行。它将正则表达式应用于该行。如果该行匹配，它将捕获相关的数据字段。如果没有匹配，该行没有遵循预期的格式；这可能值得一个错误消息。没有有用的数据可以产生，所以`continue`语句跳过了`for`语句的其余部分。

`yield`语句产生匹配的字典。每个字典将有四个命名字段和从日志中捕获的数据。数据将仅为文本，因此额外的转换将需要分别应用。

我们可以使用`csv`模块中的`DictWriter`类来发出一个CSV文件，其中这些各种数据元素被整齐地分隔。一旦我们创建了一个CSV文件，我们就可以简单地处理数据，比原始日志行快得多。

## 如何做...

1.  这个食谱将需要三个组件：

```py
            import re 
            from pathlib import Path 
            import csv 

    ```

1.  这是匹配简单Flask日志的模式。对于其他类型的日志，或者配置到Flask中的其他格式，将需要不同的模式：

```py
            log_pattern = re.compile( 
                r"\[(?P<timestamp>.*?)\]" 
                r"\s(?P<levelname>\w+)" 
                r"\sin\s(?P<module>[\w\._]+):" 
                r"\s(?P<message>.*)") 

    ```

1.  这是产生匹配行的字典的函数。这应用了正则表达式模式。不匹配的行将被静默跳过。匹配将产生一个项目名称及其值的字典：

```py
            def extract_row_iter(source_log_file): 
                for line in source_log_file: 
                    match = log_pattern.match(line) 
                    if match is None: continue 
                    yield match.groupdict() 

    ```

1.  我们将为生成的日志摘要文件定义`Path`对象：

```py
            summary_path = Path('summary_log.csv') 

    ```

1.  然后我们可以打开结果上下文。因为我们使用了`with`语句，所以可以确保无论在脚本中发生什么，文件都会被正确关闭：

```py
            with summary_path.open('w') as summary_file: 

    ```

1.  由于我们正在基于字典编写CSV文件，我们将定义一个`csv.DictWriter`。这是在`with`语句内缩进了四个空格。我们必须提供输入字典中的预期键。这将定义结果文件中列的顺序：

```py
            writer = csv.DictWriter(summary_file, 
                ['timestamp', 'levelname', 'module', 'message']) 
            writer.writeheader() 

    ```

1.  我们将为包含日志文件的源目录定义`Path`对象。在这种情况下，日志文件碰巧在脚本所在的目录中。这是罕见的，使用环境变量可能会更有用：

```py
            source_log_dir = Path('.') 

    ```

我们可以想象使用`os.environ.get('LOG_PATH', '/var/log')`作为一个比硬编码路径更一般的解决方案。

1.  我们将使用`Path`对象的`glob()`方法来查找所有与所需名称匹配的文件：

```py
            for source_log_path in source_log_dir.glob('*.log'): 

    ```

这也可以从环境变量或命令行参数中获取模式字符串。

1.  我们将为每个源文件定义一个读取上下文。这个上下文管理器将确保输入文件被正确关闭并释放资源。请注意，这是在前面的`with`和`for`语句内缩进，总共有八个空格。在处理大量文件时，这一点尤为重要：

```py
            with source_log_path.open() as source_log_file: 

    ```

1.  我们将使用写入器的`writerows()`方法来从`extract_row_iter()`函数中写入所有有效行。这是在两个`with`语句以及`for`语句内缩进的。这是整个过程的核心：

```py
            writer.writerows(extract_row_iter(source_log_file) ) 

    ```

1.  我们还可以编写一个摘要。这是在外部`with`和`for`语句内缩进的。它总结了前面的`with`语句的处理：

```py
            print('Converted', source_log_path, 'to', summary_path) 

    ```

## 工作原理...

Python与多个上下文管理器很好地配合。我们可以轻松地有深度嵌套的`with`语句。每个`with`语句可以管理不同的上下文对象。

由于打开的文件是上下文对象，将每个打开的文件包装在`with`语句中是最合理的，以确保文件被正确关闭并且所有操作系统资源都从文件中释放。

我们使用`Path`对象来表示文件系统位置。这使我们能够根据输入名称轻松创建输出名称，或在处理后重命名文件。有关更多信息，请参阅*使用pathlib处理文件名*配方。

我们使用生成器函数来组合两个操作。首先，有一个从源文本到单独字段的映射。其次，有一个排除不匹配预期模式的源文本的过滤器。在许多情况下，我们可以使用`map()`和`filter()`函数来使这一点更清晰。

然而，在使用正则表达式匹配时，要分离操作的映射和过滤部分就不那么容易了。正则表达式可能不匹配一些输入行，这就成了一种捆绑到映射中的过滤。因此，生成器函数非常有效。

`csv`写入器有一个`writerows()`方法。这个方法接受一个迭代器作为参数值。这样很容易向写入器提供一个生成器函数。写入器将消耗生成器产生的对象。这种方式可以处理非常大的文件，因为不会将整个文件读入内存，只需读取足够的文件来创建完整的数据行。

## 还有更多...

通常需要对从每个源文件读取的日志文件行数、因为它们不匹配而被丢弃的行数以及最终写入摘要文件的行数进行摘要计数。

在使用生成器时，这是具有挑战性的。生成器产生大量数据行。它如何产生一个摘要呢？

答案是我们可以向生成器提供一个可变对象作为参数。理想的可变对象是`collections.Counter`的一个实例。我们可以用它来计算包括有效记录、无效记录，甚至特定数据值的出现次数。可变对象可以被生成器和整个主程序共享，以便主程序可以将计数信息打印到日志中。

以下是将文本转换为有用的字典对象的映射-过滤函数。我们编写了一个名为`counting_extract_row_iter()`的第二个版本，以强调额外的特性：

```py
    def counting_extract_row_iter(counts, source_log_file): 
        for line in source_log_file: 
            match = log_pattern.match(line) 
            if match is None: 
                counts['non-match'] += 1 
                continue 
            counts['valid'] += 1 
            yield match.groupdict() 

```

我们提供了一个额外的参数`counts`。当我们发现不匹配正则表达式的行时，我们可以增加`Counter`中的`non-match`键。当我们发现正确匹配的行时，我们可以增加`Counter`中的`valid`键。这提供了一个摘要，显示了从给定文件中处理了多少行。

整体处理脚本如下所示：

```py
    summary_path = Path('summary_log.csv') 
    with summary_path.open('w') as summary_file: 

        writer = csv.DictWriter(summary_file, 
            ['timestamp', 'levelname', 'module', 'message']) 
        writer.writeheader() 

        source_log_dir = Path('.') 
        for source_log_path in source_log_dir.glob('*.log'): 
            counts = Counter() 
            with source_log_path.open() as source_log_file: 
                writer.writerows( 
                    counting_extract_row_iter(counts, source_log_file) 
                    ) 

            print('Converted', source_log_path, 'to', summary_path) 
            print(counts) 

```

我们做了三个小改动：

+   在处理源日志文件之前，创建一个空的`Counter`对象。

+   将`Counter`对象提供给`counting_extract_row_iter()`函数。该函数在处理行时会更新计数器。

+   在处理文件后打印`counter`的值。这种未加修饰的输出并不太美观，但它讲述了一个重要的故事。

我们可能会看到以下输出：

```py
 **Converted 20160612.log to summary_log.csv 
Counter({'valid': 86400}) 
Converted 20160613.log to summary_log.csv 
Counter({'valid': 86399, 'non-match': 1)** 

```

这种输出方式向我们展示了`summary_log.csv`的大小，也显示了`20160613.log`文件中出现了问题。

我们可以很容易地扩展这一点，将所有单独的源文件计数器组合起来，在处理结束时产生一个单一的大输出。我们可以使用`+`运算符来组合多个`Counter`对象，以创建所有数据的总和。具体细节留给读者作为练习。

## 另请参阅

+   有关上下文的基础知识，请参阅*使用上下文管理器读写文件*配方
