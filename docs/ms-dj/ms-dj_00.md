# 前言

# 您需要为本书做好准备

**所需的编程知识**

本书的读者应该了解过程化和面向对象编程的基础知识：控制结构（如 if、while 或 for）、数据结构（列表、哈希/字典）、变量、类和对象。网页开发经验，正如您可能期望的那样，非常有帮助，但不是阅读本书的必要条件。在整本书中，我试图为缺乏这方面经验的读者推广网页开发的最佳实践。

**所需的 Python 知识**

在其核心，Django 只是用 Python 编写的一组库。要使用 Django 开发网站，您需要编写使用这些库的 Python 代码。因此，学习 Django 实际上就是学习如何在 Python 中编程以及理解 Django 库的工作原理。如果您有 Python 编程经验，那么您应该可以轻松上手。总的来说，Django 代码并不执行很多*魔术*（即，编程技巧，其实现很难解释或理解）。对您来说，学习 Django 将是学习 Django 的惯例和 API 的问题。

如果您没有 Python 编程经验，您将会有所收获。它很容易学习，也很愉快使用！尽管本书不包括完整的 Python 教程，但它会在适当的时候突出 Python 的特性和功能，特别是当代码不立即让人明白时。不过，我建议您阅读官方的 Python 教程（有关更多信息，请访问[`docs.python.org/tut/`](http://docs.python.org/tut/)）。我还推荐 Mark Pilgrim 的免费书籍*Dive Into Python*，可在线获取[`www.diveintopython.net/`](http://www.diveintopython.net/)，并由 Apress 出版。

**所需的 Django 版本**

本书涵盖 Django 1.8 LTS。这是 Django 的长期支持版本，将至少在 2018 年 4 月之前得到全面支持。

如果您使用的是 Django 的早期版本，建议您升级到最新版本的 Django 1.8 LTS。在印刷时（2016 年 7 月），Django 1.8 LTS 的最新生产版本是 1.8.13。

如果您安装了 Django 的较新版本，请注意，尽管 Django 的开发人员尽可能保持向后兼容性，但偶尔会引入一些向后不兼容的更改。每个版本的更改都总是在发布说明中进行了解，您可以在[`docs.djangoproject.com/en/dev/releases/`](https://docs.djangoproject.com/en/dev/releases)找到。

有任何疑问，请访问：[`masteringdjango.com`](http://masteringdjango.com)。

# 这本书是为谁准备的

本书假设您对互联网和编程有基本的了解。有 Python 或 Django 的经验会是一个优势，但不是必需的。这本书非常适合初学者和中级程序员，他们正在寻找一个快速、安全、可扩展和可维护的替代网页开发平台，而不是基于 PHP、Java 和 dotNET 的平台。

# 惯例

在本书中，您会发现一些区分不同信息类型的文本样式。以下是一些这些样式的示例及其含义的解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：“在命令提示符（或在`Applications/Utilities/Terminal`中，OS X 中）键入`python`。”

代码块设置如下：

```py
from django.http import HttpResponse
def hello(request):
return HttpResponse("Hello world")
```

任何命令行输入或输出都以以下方式编写：

```py
Python 2.7.5 (default, June 27 2015, 13:20:20)
[GCC x.x.x] on xxx
Type "help", "copyright", "credits" or "license" for more information.
>>>

```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如菜单或对话框中的单词，会以这样的方式显示在文本中：“您应该看到文本**Hello world**-这是您的 Django 视图的输出（图 2-1）。”

### 注意

警告或重要说明会以这样的方式显示在框中。

### 提示

提示和技巧是这样显示的。
