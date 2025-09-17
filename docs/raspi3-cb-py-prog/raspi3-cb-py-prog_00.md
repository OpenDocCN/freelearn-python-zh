# 前言

这本书旨在为任何想要使用树莓派构建软件应用或硬件项目的人提供帮助。本书逐步介绍了文本分类、创建游戏、3D 图形和情感分析等内容。我们还逐步深入到更高级的主题，例如构建计算机视觉应用、机器人以及神经网络应用。具备基本的 Python 知识将是非常理想的；然而，所有编程概念都进行了详细的解释。所有示例均使用 Python 3 编写，并提供了清晰且详细的解释，以便您能够适应并在自己的项目中使用所有这些信息。到本书结束时，您将具备使用树莓派构建创新软件应用和硬件项目所需的所有技能。

# 这本书面向的对象

这本书适合任何想要使用 Raspberry Pi 3 掌握 Python 编程技能的人。具备 Python 基础知识将是一个额外的优势。

# 为了最大限度地利用这本书

阅读者应了解 Python 编程的基础知识。

读者对机器学习、计算机视觉和神经网络有一个基本了解将会有益。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便直接将文件通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packtpub.com](http://www.packtpub.com/support) 登录或注册。

1.  选择“支持”标签。

1.  点击代码下载与勘误表。

1.  在搜索框中输入书籍名称，并遵循屏幕上的指示。

一旦文件下载完成，请确保您使用最新版本解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，地址为**[`github.com/PacktPublishing/Raspberry-Pi-3-Cookbook-for-Python-Programmers-Third-Edition`](https://github.com/PacktPublishing/Raspberry-Pi-3-Cookbook-for-Python-Programmers-Third-Edition)**。如果代码有更新，它将在现有的 GitHub 仓库中进行更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到。去看看吧！

# 下载彩色图片

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：

[Raspberry Pi 3 Cookbook for Python Programmers Third Edition with Color Images](http://www.packtpub.com/sites/default/files/downloads/RaspberryPi3CookbookforPythonProgrammersThirdEdition_ColorImages.pdf).

# 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：

"在这里我们使用`bind`函数，它将此小部件（`the_canvas`）上发生的特定事件绑定到特定的动作或按键。"

代码块设置如下：

```py
#!/usr/bin/python3 
# bouncingball.py 
import tkinter as TK 
import time 

VERT,HOREZ=0,1 
xTOP,yTOP = 0,1 
xBTM,yBTM = 2,3 
MAX_WIDTH,MAX_HEIGHT = 640,480 
xSTART,ySTART = 100,200 
BALL_SIZE=20 
RUNNING=True 
```

任何命令行输入或输出都应按照以下格式编写：

```py
sudo nano /boot/config.txt
```

**粗体**: 表示新术语、重要单词或屏幕上出现的单词。例如，菜单或对话框中的单词在文本中会这样显示。以下是一个示例：“点击“配对”按钮开始配对过程并输入设备的 PIN 码。”

警告或重要提示会像这样显示。

小贴士和技巧看起来是这样的。

# 部分

在这本书中，您会发现几个经常出现的标题（*准备就绪*，*如何操作...*，*它是如何工作的...*，*还有更多...*，以及*另请参阅*）。

为了清楚地说明如何完成食谱，请按照以下部分使用：

# 准备就绪

本节向您介绍在食谱中可以期待的内容，并描述如何设置任何软件或为食谱所需的任何初步设置。

# 如何做到这一点...

本节包含遵循食谱所需的步骤。

# 它是如何工作的...

本节通常包含对上一节发生事件的详细解释。

# 还有更多...

本节包含有关食谱的附加信息，以便您对食谱有更深入的了解。

# 参见

本节提供了对其他有用信息的链接，以帮助食谱的制作。

# 联系我们

我们始终欢迎读者的反馈。

**总体反馈**：请发送电子邮件至 `feedback@packtpub.com` 并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过 `questions@packtpub.com` 发送邮件给我们。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告这一点，我们将不胜感激。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过发送链接至`copyright@packtpub.com`与我们联系。

**如果您想成为一名作者**：如果您在某个领域有专业知识，并且对撰写或参与一本书籍感兴趣，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评价。一旦您阅读并使用了这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 公司可以了解您对我们产品的看法，而我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问[packtpub.com](https://www.packtpub.com/).
