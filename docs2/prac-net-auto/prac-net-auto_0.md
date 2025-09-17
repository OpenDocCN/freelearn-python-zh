# 前言

网络自动化是使用 IT 控制来监督和执行日常网络管理功能的应用。它在网络虚拟化技术和网络功能中发挥着关键作用。

本书首先提供网络自动化、SDN 以及网络自动化各种应用的介绍，包括将 DevOps 工具集成到网络中以实现高效自动化。然后，它指导你完成不同的网络自动化任务，并涵盖各种数据挖掘和报告方法，如 IPv6 迁移、数据中心搬迁和接口解析，同时保持安全性并提高数据中心鲁棒性。接着，本书转向 Python 的使用以及 SSH 密钥的管理，用于机器到机器（M2M）通信，所有这些都伴随着实际用例。它还涵盖了 Ansible 在网络自动化中的重要性，包括自动化最佳实践、使用不同工具测试自动化网络的方法以及其他重要技术。

到本书结束时，你将熟悉网络自动化的各个方面。

# 本书涵盖的内容

第一章，*基本概念*，介绍了如何开始自动化。

第二章*，网络工程师的 Python*，介绍了 Python 作为脚本语言，并提供示例来解释 Python 在访问网络设备和从设备输出中解析数据的使用。

第三章，*从网络访问和挖掘数据*，介绍了在保持安全性和提高数据中心鲁棒性的同时，提供按需自助容量和资源。

第四章*，自动化触发器的 Web 框架*，讨论了如何对自动化框架进行可扩展调用并生成自定义和动态的 HTML 页面。

第五章*，网络自动化中的 Ansible*，解释了如何虚拟化 Oracle 数据库并动态扩展以确保达到服务水平。

第六章*，网络工程师的持续集成*，概述了网络工程师的集成原则，以管理快速增长并实现高可用性和快速灾难恢复。

第七章*，网络自动化中的 SDN 概念*，讨论了将企业 Java 应用程序迁移到虚拟化的 x86 平台，以更好地利用资源，并简化生命周期和可伸缩性管理。

# 你需要为本书准备的内容

本书的硬件和软件要求包括 Python（3.5 及以上版本）、IIS、Windows、Linux、Ansible 安装和 GNS3（用于测试）或真实路由器。

您需要互联网连接来下载 Python 库。此外，还需要具备 Python 的基本知识、网络知识以及像 IIS 这样的 Web 服务器的基本熟悉度。

# 本书面向对象

如果您是一位寻找广泛指南以帮助您高效自动化和管理网络的网络工程师，那么这本书就是为您准备的。

# 术语约定

在本书中，您将找到许多不同的文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名显示如下：“从安装目录中，我们只需要调用`python.exe`，这将调用 Python 解释器。”

代码块设置为以下格式：

```py
#PowerShell sample code
$myvalue=$args[0]
write-host ("Argument passed to Powershell is "+$myvalue)
```

任何命令行输入或输出都按照以下方式编写：

```py
python checkargs.py 5 6
```

**新术语**和**重要词汇**以粗体显示。

警告或重要注意事项看起来像这样。

小贴士和技巧看起来像这样。

# 读者反馈

我们始终欢迎读者的反馈。请告诉我们您对这本书的看法——您喜欢什么或不喜欢什么。读者反馈对我们来说非常重要，因为它帮助我们开发出您真正能从中获得最大收益的标题。要发送给我们一般性的反馈，请简单地发送电子邮件至 `feedback@packtpub.com`，并在邮件的主题中提及书的标题。如果您在某个领域有专业知识，并且对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南，网址为 [www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已经成为 Packt 图书的骄傲拥有者，我们有一些东西可以帮助您从您的购买中获得最大收益。

# 下载示例代码

您可以从您的账户中下载本书的示例代码文件，网址为 [`www.packtpub.com`](http://www.packtpub.com)。如果您在其他地方购买了这本书，您可以访问 [`www.packtpub.com/support`](http://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给您。您可以通过以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的“支持”标签上。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书的名称。

1.  选择您想要下载代码文件的书籍。

1.  从下拉菜单中选择您购买这本书的地方。

1.  点击“代码下载”。

文件下载完成后，请确保您使用最新版本解压或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

本书的相关代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Practical-Network-Automation`](https://github.com/PacktPublishing/Practical-Network-Automation)。我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。请查看它们！

# 下载本书的颜色图像

我们还为您提供了一个包含本书中使用的截图/图表的颜色图像的 PDF 文件。这些颜色图像将帮助您更好地理解输出的变化。您可以从[`www.packtpub.com/sites/default/files/downloads/PracticalNetworkAutomation_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/PracticalNetworkAutomation_ColorImages.pdf)下载此文件。

# 勘误

尽管我们已经尽最大努力确保内容的准确性，错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以避免其他读者感到沮丧，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。要查看之前提交的勘误，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在勘误部分下。

# 侵权

互联网上版权材料的侵权是一个持续存在的问题，涉及所有媒体。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。请通过提供涉嫌侵权材料的链接，通过`copyright@packtpub.com`联系我们。我们感谢您在保护我们作者和我们为您提供有价值内容的能力方面的帮助。

# 询问

如果您对本书的任何方面有问题，您可以通过`questions@packtpub.com`联系我们，我们将尽力解决问题。
