# 前言

在本书中，您将介绍比特币和区块链，以及如何参与比特币生态系统。您将了解比特币及其特性、区块链以及两者如何共同工作。您还将学习如何使用 Pi 比特币工具来用 Python 编程比特币。您将学习如何使用 Python 以编程方式与区块链 API 进行交互，以及比特币挖矿以及如何开始进行。我们还将探讨比特币交易机器人。本书还涉及探索和分析比特币生态系统中产生的大量数据；如何获取、清理、操作和可视化比特币价格数据；以及如何使用 Python 的数据分析工具分析比特币骰子游戏的数据。

# 为了充分利用本书

任何有一些 Python 经验的人都可以从本书中受益，他们希望探索 Python 比特币编程并开始构建基于比特币的 Python 应用程序。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便文件直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压或提取文件夹：

+   WinRAR/7-Zip 适用于 Windows

+   Zipeg/iZip/UnRarX 适用于 Mac

+   7-Zip/PeaZip 适用于 Linux

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Bitcoin-Programming-with-Python`](https://github.com/PacktPublishing/Hands-On-Bitcoin-Programming-with-Python)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还提供了来自我们丰富书籍和视频目录的其他代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。来看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/HandsOnBitcoinProgrammingwithPython_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/HandsOnBitcoinProgrammingwithPython_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这是一个例子：“使用`privtopub`函数从这些私钥创建三个公钥。”

代码块设置如下：

```py
# Generate Public Key
my_public_key = privtopub(my_private_key)
print("Public Key: %s\n" % my_public_key)
```

任何命令行输入或输出都以以下方式编写：

```py
pip install bitcoin
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这种方式出现在文本中。这是一个例子：“屏幕截图显示了统计数据（DATA | Stats）”

警告或重要提示会出现在这样的样式中。提示和技巧会出现在这样的样式中。
