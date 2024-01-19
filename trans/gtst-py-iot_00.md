# 前言

这个学习路径将带您进入机器人世界，并教会您如何利用树莓派和 Python 实现一切。

它教会您如何利用树莓派 3 和树莓派零的力量构建卓越的自动化系统，可以改变您的业务。您将学会创建文本分类器，预测单词的情感，并使用 Tkinter 库开发应用程序。当您使用 Python 构建人脸检测和识别系统以及家庭自动化系统时，事情将变得更有趣，不同的设备将使用树莓派进行控制。通过这些多样化的机器人项目，您将掌握机器人学的基础知识和功能，并了解机器人与物联网环境的整合。

通过学习路径的最后，您将涵盖从配置机器人控制器到使用 Python 创建自驾动机器车的一切。

这个学习路径包括以下 Packt 产品的内容：

+   Raspberry Pi 3 Cookbook for Python Programmers - Third Edition by Tim Cox, Dr. Steven Lawrence Fernandes

+   Python Programming with Raspberry Pi by Sai Yamanoor, Srihari Yamanoor

+   Python Robotics Projects by Prof. Diwakar Vaish

# 本书适合对象

这本书专为想要通过创建可以改善人们生活的机器人来提升技能的 Python 开发人员设计。熟悉 Python 和电子学将有助于理解本学习路径中的概念。

# 充分利用本书

要开始使用本书，读者应该了解 Python 编程的基础知识。读者对机器学习、计算机视觉和神经网络有基本的了解将是有益的。还建议使用以下硬件：

+   带有任何操作系统的笔记本电脑

+   树莓派

+   一个 8GB 或 16GB 的 MicroSD 卡

+   USB 键盘、鼠标和 WiFi 卡

+   带有 HDMI 输入的显示器

+   电源供应，最低 500 毫安

+   显示器电缆和其他配件

读者需要下载并安装 RASPBIAN STRETCH WITH DESKTOP；这将为我们提供树莓派的 GUI 界面

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

文件下载完成后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip 适用于 Windows

+   Zipeg/iZip/UnRarX 适用于 Mac

+   7-Zip/PeaZip 适用于 Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/GettingStartedwithPythonfortheInternetofThings`](https://github.com/PacktPublishing/GettingStartedwithPythonfortheInternetofThings)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有其他代码包，来自我们丰富的图书和视频目录，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。去看看吧！

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。以下是一个例子：“`input()`方法用于从用户那里获取输入。”

代码块设置如下：

```py
try:         
    input_value = int(value)      
except ValueError as error:         
    print("The value is invalid %s" % error)
```

任何命令行输入或输出都以以下形式编写：

```py
sudo pip3 install schedule
```

**粗体**：表示一个新术语，一个重要词或屏幕上看到的词。例如，菜单或对话框中的单词会在文本中显示为这样。这是一个例子：“如果你需要不同的东西，点击页眉中的**下载**链接以获取所有可能的下载：”

警告或重要说明会显示为这样。提示和技巧会显示为这样。
