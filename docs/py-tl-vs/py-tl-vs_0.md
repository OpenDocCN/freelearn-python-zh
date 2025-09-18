# 前言

和许多其他开发者一样，Python 开发者总是需要找到方法来管理不同工具之间的开发工作流程。大多数时候，这都是在没有使用专为 Python 开发设计的完整 IDE 中的综合指南的情况下发生的。

提供完整指南的稀有、卓越的 IDE 通常价格昂贵，并且不提供实际步骤来帮助加快开发过程。

在过去几十年中，作为成熟且功能丰富的工具，Visual Studio 在编译语言和严格针对 Windows 和.NET 的语言市场中占据主导地位。它集成了许多实用工具和功能，以加快和简化开发者的工作流程，帮助用户处理重复性任务、管理项目，并深入了解项目结构。然而，最重要的是，它帮助用户清晰地了解代码的内部结构。

在过去几年中，微软开始探索如何将新语言集成到 Visual Studio 中；因此，Python Tools for Visual Studio (PTVS)被开发出来。这是一个经过良好开发的工具，已经发布了第二个版本，并且通常被专业开发者作为他们选择的新 IDE 用于 Python 项目。

PTVS 拥有 Python 开发者所能梦想的一切：一致的项目文件管理、与微软 IntelliSense 技术相结合的交互式调试和代码补全功能、项目模板、一流的 Django 集成包、IDE 中的虚拟环境管理、REPL 以及基于本地的快速加载和响应的 IDE。

本书将更多地关注 Python 在 Visual Studio 中的集成，而不是语言本身。它将尝试深入探讨工具提供的功能，并探讨其日常使用对开发者的可行性。我们将展示如何使用 PTVS 与 Django 结合的实例，以及如何在将知名库集成到 Microsoft Windows 上的 Python 项目中遇到困难时的处理方法。

# 本书涵盖内容

第一章，*PTVS 简介*，提供了 PTVS 和 Visual Studio 与 Python 解释器之间交互的高级概述。

第二章，*Visual Studio 中的 Python 工具*，对 PTVS 的工具、类型检查、内部功能以及自动化（IntelliSense 和 REPL）进行了深入分析。

第三章，*日常编码工具*，讨论了浏览代码和灵活设置 Python 环境。它还讨论了重构和调试过程。

第四章，*PTVS 中的 Django*，展示了如何利用强大的 Visual Studio IDE 和工具来加速 Django 开发。

第五章，*PTVS 中的高级 Django*，深入探讨了使用第三方 Python 库 Fabric 和 South 进行远程任务管理和模式迁移。

第六章，*PTVS 中的 IPython 和 IronPython*，概述了 IPython 库及其在 Visual Studio 中的集成。它还介绍了 IronPython 及其与.NET 框架的集成。

# 您需要为本书准备的内容

您需要具备 Python 的基本知识、安装了 Windows 的计算机以及互联网连接。为了完成练习和示例，我们建议您拥有 Visual Studio。

# 本书面向的对象

这本书旨在为那些希望通过 Visual Studio 为.NET 社区提供的自动化工具来提高 Python 项目生产力的开发者编写。对 Python 编程的基本知识是必不可少的。

# 规范

在本书中，您将找到多种文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词如下所示：“我们可以通过使用`include`指令来包含其他上下文。”

代码块设置如下：

```py
class foo:
    """
    Documentation of the class.
    It can be multiline and contain any amount of text
    """
    @classmethod
    def bar(self, first=0, second=0):
        """This is the documentation for the method"""
        return first + second

print(foo.bar())
```

任何命令行输入或输出都应如下编写：

```py
python manage.py schemamigration south2ptvs –-initial

```

**新术语**和**重要词汇**以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，在文本中如下显示：“点击**下一步**按钮将您带到下一屏幕。”

### 注意

警告或重要提示将以如下框中的形式出现。

### 小贴士

技巧和窍门如下所示。

# 读者反馈

我们欢迎读者的反馈。请告诉我们您对本书的看法——您喜欢什么或可能不喜欢什么。读者反馈对我们开发您真正从中受益的标题非常重要。

要向我们发送一般反馈，只需发送电子邮件至`<feedback@packtpub.com>`，并在邮件主题中提及书名。

如果您在某个主题上具有专业知识，并且您有兴趣撰写或为本书做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已成为 Packt 图书的骄傲拥有者，我们有一些东西可以帮助您充分利用您的购买。

# 下载本书中的彩色图像

我们还为您提供了一个包含本书中使用的截图/图表彩色图像的 PDF 文件。彩色图像将帮助您更好地理解输出的变化。您可以从以下链接下载此文件：[`www.packtpub.com/sites/default/files/downloads/8687OS_ColoredImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/8687OS_ColoredImages.pdf)

## 错误清单

尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站，或添加到该标题的勘误部分下的现有勘误列表中。任何现有勘误都可以通过从 [`www.packtpub.com/support`](http://www.packtpub.com/support) 选择您的标题来查看。

## 侵权

互联网上版权材料的侵权是一个跨所有媒体持续存在的问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，无论形式如何，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过以下链接联系我们 `<copyright@packtpub.com>`，并提供涉嫌侵权材料的链接。

我们感谢您在保护我们作者以及为我们提供有价值内容的能力方面的帮助。

## 询问

如果您在本书的任何方面遇到问题，可以通过以下链接 `<questions@packtpub.com>` 联系我们，我们将尽力解决。

# 作者特别感谢

感谢 Packt Publishing 给我们机会为开发者社区出版这本书，以及他们在整个过程中提供的帮助：从注入想法到整个孕育过程。这是一段充满惊喜和发现的旅程。

我们还想要感谢我们的审稿人，Fabio Lonegro 和 Chris Marinic，他们在整个过程中提供了清晰且无偏见的反馈，为我们深入理解书籍的细节提供了宝贵的见解。

最后但同样重要的是，我们想感谢微软 PTVS 团队，特别是史蒂夫·道尔，他不仅亲自为本书做出了贡献，还通过提供技术支持在每一个细节上给予了帮助。感谢沙赫罗克·莫塔扎维通过推特联系我们([`twitter.com/cathycracks/status/421336498748006400`](https://twitter.com/cathycracks/status/421336498748006400))。史蒂夫和团队的其他成员给了我们很多帮助、见解和建议，帮助我们克服书中一些复杂但非常重要的部分。他们甚至邀请我们亲自拜访，以更深入地了解他们的工作。我们真心觉得 PTVS 是由一群热爱社区并渴望将 PTVS 发展成为更好、更有用工具的人开发的。在我们看来，微软 PTVS 团队到目前为止已经用这个工具做了很多出色的工作，我们期待着未来还有更多精彩。

我们到目前为止已经享受了这次旅程，我们非常高兴能一起努力使这本书变得生动。这是一个充满爱与深夜深入讨论的亲密而艰难的过程。我们希望您能像我们从这本书中学到的一样，享受并从中获得知识。

我们希望您会发现这本书很有趣，并且它能帮助您发现 PTVS 的内在力量，正如斯科特·汉斯勒在他的博客文章中所描述的，微软的最好保密之一 - Python Tools for Visual Studio (PTVS)，创建于 2013 年 7 月 2 日，可在[`www.hanselman.com/blog/OneOfMicrosoftsBestKeptSecretsPythonToolsForVisualStudioPTVS.aspx`](http://www.hanselman.com/blog/OneOfMicrosoftsBestKeptSecretsPythonToolsForVisualStudioPTVS.aspx)找到。
