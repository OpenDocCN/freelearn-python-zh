# 前言

密码学在保护关键系统和敏感信息方面有着悠久而重要的历史。本书将向您展示如何使用 Python 加密、评估、比较和攻击数据。总的来说，本书将帮助您处理加密中的常见错误，并向您展示如何利用这些错误。

# 这本书适合谁

这本书适用于希望学习如何加密数据、评估和比较加密方法以及如何攻击它们的安全专业人员。

# 本书内容

第一章，*混淆*，介绍了凯撒密码和 ROT13，简单的字符替换密码，以及 base64 编码。然后我们转向 XOR。最后，有一些挑战来测试您的学习，包括破解凯撒密码、反向 base64 编码和解密 XOR 加密而不使用密钥。

第二章，*哈希*，介绍了较旧的 MD5 和较新的 SHA 哈希技术，以及 Windows 密码哈希。最弱的哈希类型是常见的使用，其次是 Linux 密码哈希，这是常见使用中最强大的哈希类型。之后，有一些挑战需要完成。首先是破解一些 Windows 哈希并恢复密码，然后您将被要求破解哈希，甚至不知道使用了多少轮哈希算法，最后您将被要求破解那些强大的 Linux 哈希。

第三章，*强加密*，介绍了当今用于隐藏数据的主要模式。它足够强大，可以满足美国军方的需求。然后，介绍了它的两种模式，ECB 和 CBC；CBC 是更强大和更常见的模式。我们还将讨论填充预言攻击，这使得可能克服 AES CBC 的一些部分，如果设计者犯了错误，并且过于详细的错误消息向攻击者提供了信息。最后，我们介绍了 RSA，这是当今主要的公钥算法，它使得可以在不交换给定私钥的情况下通过不安全的通道发送秘密信息。在此之后，我们将进行一个挑战，我们将破解 RSA，即当它错误地使用两个相似的质数而不是两个随机质数时。

# 充分利用本书

您不需要有编程经验或任何特殊的计算机。任何能运行 Python 的计算机都可以完成这些项目，您也不需要太多的数学，因为我们不会发明新的加密技术，只是学习如何使用现有的标准加密技术，这些技术不需要比基本代数更多的东西。

# 下载本书的示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)注册，直接将文件发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩软件解压缩文件夹：

+   WinRAR/7-Zip for Windows

+   Mac 的 Zipeg/iZip/UnRarX

+   Linux 的 7-Zip/PeaZip

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Cryptography-with-Python`](https://github.com/PacktPublishing/Hands-On-Cryptography-with-Python)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有其他代码包，来自我们丰富的图书和视频目录，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/HandsOnCryptographywithPython_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/HandsOnCryptographywithPython_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这是一个例子："如果我们输入`HELLO`，它会打印出`KHOOR`的正确答案。"

代码块设置如下：

```py
alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
str_in = raw_input("Enter message, like HELLO: ")

n = len(str_in)
str_out = ""

for i in range(n):
   c = str_in[i]
   loc = alpha.find(c)
   print i, c, loc, 
   newloc = loc + 3
   str_out += alpha[newloc]
   print newloc, str_out

print "Obfuscated version:", str_out
```

任何命令行输入或输出都以以下形式编写：

```py
$ python
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会在文本中显示为这样。这是一个例子："从管理面板中选择系统信息"。

警告或重要说明会以这种方式出现。提示和技巧会以这种方式出现。
