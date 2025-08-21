# 前言

Django 是用 Python 编写的，是一个旨在快速构建复杂 Web 应用程序的 Web 应用程序框架，无需任何麻烦。它松散地遵循 MVC 模式，并遵循不重复原则，使数据库驱动的应用程序高效且高度可扩展，并且是迄今为止最受欢迎和成熟的 Python Web 框架。

这本书是一本手册，将帮助您构建一个简单而有效的 Django Web 应用程序。它首先向您介绍 Django，并教您如何设置它并编写简单的程序。然后，您将学习构建您的第一个类似 Twitter 的应用程序。随后，您将介绍标签、Ajax（以增强用户界面）和推文。然后，您将继续创建管理界面，学习数据库连接，并使用第三方库。然后，您将学习调试和部署 Django 项目，并且还将一窥 Django 与 AngularJS 和 Elasticsearch。通过本书的最后，您将能够利用 Django 框架轻松开发出一个功能齐全的 Web 应用程序。

# 本书内容

第一章《Django 简介》向您介绍了 MVC Web 开发框架的历史，并解释了为什么 Python 和 Django 是实现本书目标的最佳工具。

第二章《入门》向您展示如何在 Unix/Linux、Windows 和 Mac OS X 上设置开发环境。我们还将看到如何创建我们的第一个项目并将其连接到数据库。

第三章《Django 中的代码风格》涵盖了构建网站所需的所有基本主题，例如更好的 Django Web 开发的编码实践，应该使用哪种 IDE 和版本控制。

第四章《构建类似 Twitter 的应用程序》带您了解主要的 Django 组件，并为您的 Twitter 应用程序开发一个工作原型。

第五章《引入标签》教您设计算法来构建标签模型以及在帖子中使用标签的机制。

第六章《使用 AJAX 增强用户界面》将帮助您使用 Django 的 Ajax 增强 UI 体验。

第七章《关注和评论》向您展示如何创建登录、注销和注册页面模板。它还将向您展示如何允许另一个用户关注您以及如何显示最受关注的用户。

第八章《创建管理界面》向您展示了使用 Django 的内置功能的管理员界面的功能，以及如何以自定义方式显示带有侧边栏或启用分页的推文。

第九章《扩展和部署》通过利用 Django 框架的各种功能，为您的应用程序准备部署到生产环境。它还向您展示如何添加对多种语言的支持，通过缓存提高性能，自动化测试，并配置项目以适用于生产环境。

第十章《扩展 Django》讨论了如何改进应用程序的各个方面，主要是性能和本地化。它还教您如何在生产服务器上部署项目。

第十一章《数据库连接》涵盖了各种数据库连接形式，如 MySQL，NoSQL，PostgreSQL 等，这是任何基于数据库的应用程序所需的。

第十二章《使用第三方包》讨论了开源以及如何在项目中使用和实现开源第三方包。

第十三章《调试的艺术》向您展示如何记录和调试代码，以实现更好和更高效的编码实践。

第十四章《部署 Django 项目》向您展示如何将 Django 项目从开发环境移动到生产环境，以及在上线之前需要注意的事项。

第十五章《接下来做什么？》将带您进入下一个级别，介绍 Django 项目中使用的两个最重要和首选组件 AngularJS 和 Elasticsearch。

# 您需要为本书做好准备

对于本书，您需要在 PC/笔记本电脑上运行最新（最好是）Ubuntu/Windows/Mac 操作系统，并安装 Python 2.7.X 版本。

除此之外，您需要 Django 1.7.x 和您喜欢的任何一个文本编辑器，如 Sublime Text 编辑器，Notepad++，Vim，Eclipse 等。

# 这本书适合谁

这本书适合想要开始使用 Django 进行 Web 开发的 Web 开发人员。需要基本的 Python 编程知识，但不需要了解 Django。

# 约定

在本书中，您会发现许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：“`username`变量是我们想要查看的推文的所有者。”

代码块设置如下：

```py
#!/usr/bin/env python
import os
import sys
if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_mytweets.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
```

任何命令行输入或输出都写成如下形式：

```py
Python 2.7.6 (default, Mar 22 2014, 22:59:56) 
[GCC 4.8.2] on linux2 
Type "help", "copyright", "credits" or "license" for more information.

```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如菜单或对话框中的单词，会在文本中以这种方式出现：“在那个链接中，我们会找到下载按钮，点击下载后，点击**下载 Bootstrap**。”

### 注意

警告或重要说明出现在这样的框中。

### 提示

提示和技巧看起来像这样。
