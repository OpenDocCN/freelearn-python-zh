# 前言

自动化测试是开发人员工具中非常重要的工具。拥有一组自动化测试不仅可以提高生产力和软件质量；它还可以作为开发人员的安全网，并在代码重构方面带来信心。Python自带了一个标准的`unittest`模块，用于编写自动化测试，但也有一个替代方案：pytest。pytest框架简单易用，从简单的单元测试一直扩展到复杂的集成测试。许多人认为它在方法上是真正Pythonic的，具有简单的功能、简单的断言、固定装置、插件和一整套功能。越来越多的开发人员正在采用全面的测试方法，那么为什么不使用一个既简单又强大的框架，许多人认为它是一种真正的乐趣呢？

# 这本书适合谁

这本书适合任何希望开始使用pytest来提高其日常工作中测试技能的人。它涵盖了从安装pytest及其更重要的功能，一直到将现有的基于`unittest`的测试套件转换为pytest的技巧和技巧。还有一些基于作者多年的测试和pytest经验的技巧和讨论。在本书中，我们将通过几个代码示例，并且只需要中级水平的Python，尽管如果您有一些`unittest`经验，您会更喜欢这本书。

# 本书涵盖内容

[第1章](d72f0a2a-17dc-43ec-84eb-7dfb259d69f9.xhtml)，*介绍pytest*，讨论了为什么测试很重要，快速概述了标准的`unittest`模块，最后介绍了pytest的主要特性。

[第2章](6a63dd32-2dad-46e8-8267-46bca6349500.xhtml)，*编写和运行测试*，涵盖了pytest的安装，pytest如何仅使用`assert`语句来检查值，测试模块组织，以及一些非常有用的命令行选项，以提高生产力。

[第3章](6a37176b-78d0-4e6c-8185-460d48869d23.xhtml)，*标记和参数化*，解释了pytest的*标记*如何工作，如何根据特定条件跳过测试，并讨论了预期失败和不稳定测试之间的区别（以及如何处理）。最后，我们将学习如何使用`parametrize`标记将不同的输入集应用于相同的测试代码片段，避免重复，并邀请我们覆盖更多的输入情况。

[第4章](bf8b3438-83e6-4ce5-9df4-4da086636ef7.xhtml)，*固定装置*，探讨了pytest更受欢迎的功能之一，*固定装置*。我们还将介绍一些内置的固定装置，最后介绍一些技巧和诀窍，以更充分地利用测试套件中的固定装置。

[第5章](fe24ed0e-18cb-4676-84ac-460c76a710d6.xhtml)，*插件*，介绍了如何在丰富的插件生态系统中安装和搜索有用的插件，还介绍了一系列作者在日常工作中发现有趣和/或必须的各种插件。

[第6章](71fd05cd-2267-4c11-8b49-d9c44718320a.xhtml)，*将unittest套件转换为pytest*，介绍了一系列技术，将帮助您开始使用pytest，即使您所有的测试都是使用`unittest`框架编写的。它涵盖了从无需更改即可运行测试套件，一直到使用经过时间考验的技术将其转换为利用pytest功能。

[第7章](ef3bb2d7-4f68-4bfe-91b9-19d9d0fd6666.xhtml)，*总结*，介绍了如果您想巩固pytest技能的可能下一步。我们还将看看友好的pytest社区以及如何更多地参与其中。

# 要充分利用本书

以下是您开始所需的简短清单：

+   台式电脑或笔记本电脑：pytest适用于Linux、Windows和macOS-X，因此选择您喜欢的任何系统。

+   Python 3：所有示例都是用Python 3.6编写的，但它们应该可以在Python 3.4或更高版本上使用，如果有的话，可以进行轻微修改。大多数示例也可以移植到Python 2，但需要更多的努力，但强烈建议使用Python 3。

+   您喜欢的文本编辑器或IDE来处理代码。

+   熟悉Python：不需要太高级的知识，但是Python概念，如`with`语句和装饰器是很重要的。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便直接将文件发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在GitHub上，网址为**[https://github.com/PacktPublishing/pytest-Quick-Start-Guide](https://github.com/PacktPublishing/pytest-Quick-Start-Guide)**。如果代码有更新，将在现有的GitHub存储库上进行更新。

我们还有来自丰富图书和视频目录的其他代码包，可在**[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**上找到。去看看吧！

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：指示文本中的代码单词，数据库表名，文件夹名，文件名，文件扩展名，路径名，虚拟URL，用户输入和Twitter句柄。这是一个例子：“在命令提示符中键入此命令以创建`virtualenv`。”

代码块设置如下：

```py
 # contents of test_player_mechanics.py
    def test_player_hit():
        player = create_player()
        assert player.health == 100
        undead = create_undead()
        undead.hit(player)
        assert player.health == 80
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```py
def test_empty_name():
    with pytest.raises(InvalidCharacterNameError):
        create_character(name='', class_name='warrior')

def test_invalid_class_name():
    with pytest.raises(InvalidClassNameError):
        create_character(name='Solaire', class_name='mage')
```

任何命令行输入或输出都以以下方式编写：

```py
λ pip install pytest
```
