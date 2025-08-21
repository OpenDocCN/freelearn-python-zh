# 前言

测试一直是软件开发的一部分。几十年来，全面的测试是由复杂的手动测试程序支持的，而这些程序又由庞大的预算支持；但是在 1998 年发生了一些革命性的事情。在他的《更好的 Smalltalk 指南》中，Smalltalk 大师 Kent Beck 引入了一个名为 SUnit 的自动化测试框架。这引发了一系列测试框架，包括 JUnit、PyUnit 和许多其他针对不同语言和各种平台的框架，被称为 xUnit 运动。当 17 位顶级软件专家在 2001 年签署了《敏捷宣言》时，自动化测试成为了敏捷运动的基石。

测试包括许多不同的风格，包括单元测试、集成测试、验收测试、烟测试、负载测试等等。本书深入探讨了并探索了在使用 Python 的灵活力量的同时在所有重要级别进行测试。它还演示了许多工具。

这本书旨在扩展您对测试的知识，使其不再是您听说过的东西，而是您可以在任何级别应用以满足您在改进软件质量方面的需求。

关于，或者稍微练习了一下，变成了您可以在任何级别应用以满足您在改进软件质量方面的需求。我们希望为您提供工具，以获得更好的软件开发和客户满意度的巨大回报。

# 这本书适合谁

如果您是一名希望将测试提升到更高水平并希望扩展您的测试技能的 Python 开发人员，那么这本书适合您。假设您具有一些 Python 编程知识。

# 本书涵盖了什么

第一章《使用 Unittest 开发基本测试》为您快速介绍了 Python 社区中最常用的测试框架。

第二章《使用 Nose 运行自动化测试套件》介绍了最普遍的 Python 测试工具，并向您展示如何编写专门的插件。

第三章《使用 doctest 创建可测试文档》展示了使用 Python 的文档字符串构建可运行的 doctests 以及编写自定义测试运行器的许多不同方法。

第四章《使用行为驱动开发测试客户故事》深入探讨了使用 doctest、mocking 和 Lettuce/Should DSL 编写易于阅读的可测试的客户故事。

第五章《使用验收测试进行高级客户场景》，帮助您进入客户的思维模式，并使用 Pyccuracy 和 Robot Framework 从他们的角度编写测试。

第六章《将自动化测试与持续集成集成》向您展示了如何使用 Jenkins 和 TeamCity 将持续集成添加到您的开发流程中。

第七章《通过测试覆盖率衡量您的成功》探讨了如何创建覆盖率报告并正确解释它们。它还深入探讨了如何将它们与您的持续集成系统结合起来。

第八章《烟/负载测试-测试主要部分》着重介绍了如何创建烟测试套件以从系统中获取脉搏。它还演示了如何将系统置于负载之下，以确保它能够处理当前的负载，并找到未来负载的下一个破坏点。

第九章《新旧系统的良好测试习惯》带您经历了作者从软件测试方面学到的许多不同的经验教训。

*第十章*，*使用 Selenium 进行 Web UI 测试*，教你如何为他们的软件编写合适的测试集。它将解释要使用的各种测试集和框架。本章不包含在书中，可以在以下链接在线获取：[`www.packtpub.com/sites/default/files/downloads/Web_UI_Testing_Using_Selenium.pdf`](https://www.packtpub.com/sites/default/files/downloads/Web_UI_Testing_Using_Selenium.pdf)

# 要充分利用本书

您需要在您的计算机上安装 Python。本书使用许多其他 Python 测试工具，但包括详细的步骤，显示如何安装和使用它们。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择 SUPPORT 选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Python-Testing-Cookbook-Second-Edition`](https://github.com/PacktPublishing/Python-Testing-Cookbook-Second-Edition)。我们还有来自丰富书籍和视频目录的其他代码包可用于**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。去看看吧！

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这是一个例子：“您还可以使用`pip install virtualenv`。”

代码块设置如下：

```py
if __name__== "__main__": 
    suite = unittest.TestLoader().loadTestsFromTestCase(\
              RomanNumeralConverterTest) 
    unittest.TextTestRunner(verbosity=2).run(suite) 
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体设置：

```py
def test_bad_inputs(self): 
    r = self.cvt.convert_to_roman 
    d = self.cvt.convert_to_decimal 
    edges = [("equals", r, "", None),\ 
```

任何命令行输入或输出都以以下方式编写：

```py
$ python recipe3.py
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这种方式出现在文本中。这是一个例子：“选择一个要测试的类。这被称为**被测试的类**。”

警告或重要说明会出现在这样的地方。提示和技巧会出现在这样的地方。

# 章节

在本书中，您会经常看到几个标题（*准备就绪*，*如何做...*，*它是如何工作的...*，*还有更多...*和*另请参阅*）。

为了清晰地说明如何完成配方，使用以下各节： 

# 准备就绪

本节告诉您配方中可以期望发生什么，并描述如何设置配方所需的任何软件或任何预备设置。

# 如何做...

本节包含遵循该配方所需的步骤。

# 它是如何工作的...

本节通常包括对前一节中发生的事情的详细解释。

# 还有更多...

本节包括有关配方的其他信息，以使您对配方更加了解。

# 另请参阅

本节提供了有关配方的其他有用信息的链接。
