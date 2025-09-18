# 前言

每个特工都需要一套好的工具和设备。当一个特工的任务涉及收集数据时，就需要进行高强度的数据处理。这本书将为你提供各种信息处理工具，帮助你收集、分析和传达总部所需的数据。

Python 允许特工编写简单的脚本来收集数据、执行复杂的计算并产生有用的结果。特工还可以使用 Python 从本地文件、HTTP 网页服务器和 FTP 文件服务器中提取数据。

Python 有许多附加包。这本书将探讨其中的两个：Pillow 允许进行复杂的图像转换和处理，而 BeautifulSoup 允许特工从 HTML 网页中提取数据。有特定需求的特工可能需要探索 Python 自然语言工具包（NLTK）、数值 Python（NumPy）甚至科学 Python（SciPy）。

# 本书涵盖的内容

第一章，*我们的间谍工具包*，揭示了安装和使用 Python 的基础知识。我们将编写脚本来帮助特工处理外币转换，并学习特工如何从 ZIP 存档中恢复丢失的密码。

第二章，*获取情报数据*，展示了我们如何使用 Python 从各种类型的文件服务器中提取信息。特工将学习如何与不同的互联网协议一起工作，并使用表示状态转移（REST）与网络服务进行交互。这包括处理加密货币，如比特币的技术。

第三章，*使用隐写术编码秘密信息*，展示了我们如何将 Pillow 工具集添加到图像处理中。拥有 Pillow 的特工可以创建缩略图，转换、裁剪和增强图像。我们还将探索一些隐写术算法，将我们的信息隐藏在图像文件中。

第四章，*投递点、藏身之处、会面和巢穴*，更深入地探讨了地理编码和地理定位。这包括使用网络服务将地址转换为经纬度。我们还将学习如何将经纬度转换回地址。我们将研究哈弗辛公式来获取地点之间的正确距离。我们还将探讨一些表示地理位置的方法，以便于整洁的存储和通信。

第五章，*间谍大师的更敏感分析*，展示了我们如何使用 Python 进行基本的数据分析。一个好的特工不仅仅是说出事实和数字；一个好的特工会进行足够的分析来确认数据是真实的。能够检查数据集之间的相关性是创造有价值的情报资产的关键。

# 你需要这本书的内容

一个神秘代理需要一个他们有管理权限的计算机。我们将安装额外的软件。如果没有管理密码，他们可能难以安装 Python 3、Pillow 或 BeautifulSoup。

对于使用 Windows 的代理，我们想要添加的包是预构建的。

对于使用 Linux 的代理，需要开发者工具。Linux 拥有一套完整的开发者工具，这些工具很常见。GNU C 编译器（GCC）是这些工具的支柱。

对于使用 Mac OS X 的代理，所需的开发者工具是 Xcode ([`developer.apple.com/xcode/`](https://developer.apple.com/xcode/))。我们还需要安装一个名为 homebrew ([`brew.sh`](http://brew.sh)) 的工具，以帮助我们向 Mac OS X 添加 Linux 包。

Python 3 可从 Python 下载页面 [`www.python.org/download`](https://www.python.org/download) 获取。

我们将下载并安装除了 Python 3.3 以外的几个东西：

+   setuptools 包，包括 `easy_install-3.3`，将帮助我们添加包。它可以从 [`pypi.python.org/pypi/setuptools`](https://pypi.python.org/pypi/setuptools) 下载。

+   PIP 包也将帮助我们安装额外的包。一些经验丰富的现场代理更喜欢 PIP 而不是 setuptools。它可以从 [`pypi.python.org/pypi/pip/1.5.6`](https://pypi.python.org/pypi/pip/1.5.6) 下载。

+   Pillow 包将使我们能够处理图像文件。它可以从 [`pypi.python.org/pypi/Pillow/2.4.0`](https://pypi.python.org/pypi/Pillow/2.4.0) 下载。

+   BeautifulSoup 版本 4 包将使我们能够处理 HTML 网页。它可以从 [`pypi.python.org/pypi/beautifulsoup4/4.3.2`](https://pypi.python.org/pypi/beautifulsoup4/4.3.2) 下载。

从这里，我们将看到 Python 的可扩展性。几乎任何代理可能需要的功能可能已经编写好并通过 Python 包索引（PyPi）提供，可以从 [`pypi.python.org/pypi`](https://pypi.python.org/pypi) 下载。

# 这本书面向的对象

这本书是为那些不太了解 Python 但舒适安装新软件并准备好在 Python 中进行一些巧妙编程的神秘代理而写的。一个以前从未编程过的代理可能会发现其中的一些内容有点高级；一本涵盖 Python 基础的入门教程可能会有所帮助。

我们期望使用这本书的代理对简单的数学感到舒适。这包括货币转换的乘法和除法。它还包括多项式、简单的三角学和几个统计公式。

我们还期望使用这本书的神秘代理会进行自己的调查。这本书的示例旨在帮助代理开始开发有趣、有用的应用程序。每个代理都必须自己进一步探索。

# 习惯用法

在这本书中，你将找到多种文本样式，用以区分不同类型的信息。以下是一些这些样式的示例，以及它们的意义解释。

文本中的代码词汇、包名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名将如下所示：“`size_list`变量是由编码大小的字节组成的八个元组的序列。”

代码块将以如下方式设置：

```py
message_bytes= message.encode("UTF-8")
bits_list = list(to_bits(c) for c in message_bytes )
len_h, len_l = divmod( len(message_bytes), 256 )
size_list = [to_bits(len_h), to_bits(len_l)]
bit_sequence( size_list+bits_list )
```

当我们希望引起你对代码块中特定部分的注意时，相关的行或项目将以粗体显示：

```py
w, h = ship.size
for p,m in enumerate( bit_sequence(size_list+bits_list) ):
    y, x = divmod( p, w )
    r, g, b = ship.getpixel( (x,y) )
    r_new = (r & 0xfe) | m
    print( (r, g, b), m, (r_new, g, b) )
    ship.putpixel( (x,y), (r_new, g, b)  )
```

任何命令行输入或输出都应如下编写：

```py
$ python3.3 -m doctest ourfile.py

```

**新术语**和**重要词汇**将以粗体显示。屏幕上看到的词汇，例如在菜单或对话框中，将以文本中的这种方式显示：“有一个定义这些文件关联的**高级设置**面板。”

### 注意

警告或重要注意事项将以这样的框显示。

### 提示

技巧和窍门将以这样的形式出现。

# 读者反馈

我们欢迎读者的反馈。告诉我们你对这本书的看法——你喜欢什么或可能不喜欢什么。读者反馈对我们开发你真正能从中获得最大收益的标题非常重要。

要发送给我们一般性的反馈，只需发送一封电子邮件到`<feedback@packtpub.com>`，并在邮件的主题中提及书名。

如果你在一个主题上具有专业知识，并且你对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

既然你已经是 Packt Publishing 书籍的骄傲拥有者，我们有许多事情可以帮助你从购买中获得最大收益。

## 下载示例代码

你可以从你购买的所有 Packt Publishing 书籍的账户中下载示例代码文件。[`www.packtpub.com`](http://www.packtpub.com)。如果你在其他地方购买了这本书，你可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给你。

## 错误清单

尽管我们已经尽一切努力确保我们内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站，或添加到该标题的勘误部分下的现有勘误列表中。任何现有勘误都可以通过从 [`www.packtpub.com/support`](http://www.packtpub.com/support) 选择您的标题来查看。

## 盗版

在互联网上，版权材料的盗版是一个跨所有媒体的持续问题。在 Packt Publishing，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，无论形式如何，请立即向我们提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 `<copyright@packtpub.com>` 与我们联系，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们作者和提供有价值内容方面的帮助。

## 问题

如果你在本书的任何方面遇到问题，可以通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决。
