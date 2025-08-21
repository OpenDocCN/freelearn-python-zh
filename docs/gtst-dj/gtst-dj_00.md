# 前言

多年来，Web 开发已经通过框架得到了发展。Web 开发变得更加高效，质量也得到了提高。Django 是一个非常复杂和流行的框架。框架是一组旨在简化和标准化开发的工具。它允许开发人员从非常实用的工具中受益，以最小化开发时间。然而，使用框架进行开发需要了解框架及其正确的使用方法。本书使用逐步教学法帮助初学者开发人员学习如何轻松应对 Django 框架。本书中的示例解释了一个简单 Web 工具的开发：基于文本的任务管理器。

# 本书涵盖内容

第一章，“Django 在 Web 上的位置”，简要介绍了 Web 的历史和发展。它解释了框架和 MVC 模式是什么。最后介绍了 Django。

第二章，“创建 Django 项目”，涉及安装使用 Django 所需的软件。在本章结束时，您将拥有一个准备好编码的开发环境。

第三章，“使用 Django 的 Hello World!”，描述了在提醒正则表达式后的 Django 路由。最后以一个简单的控制器示例结束，该控制器在用户的浏览器上显示“Hello world!”。

第四章，“使用模板”，解释了 Django 模板的工作原理。它涵盖了模板语言的基础知识以及架构模板和 URL 创建的最佳实践。

第五章，“使用模型”，描述了在 Django 中构建模型。它还解释了如何生成数据库以及如何使用 South 工具进行维护。本章还向您展示了如何通过管理模块设置管理界面。

第六章，“使用 Querysets 获取模型数据”，解释了如何通过模型对数据库执行查询。使用示例来测试不同类型的查询。

第七章，“使用 Django 表单”，讨论了 Django 表单。它解释了如何使用 Django 创建表单以及如何处理它们。

第八章，“使用 CBV 提高生产力”，专注于 Django 的一个独特方面：基于类的视图。本章解释了如何在几秒钟内创建 CRUD 界面。

第九章，“使用会话”，解释了如何使用 Django 会话。不同的实际示例展示了会话变量的使用以及如何充分利用它们。

第十章，“认证模块”，解释了如何使用 Django 认证模块。它涵盖了注册、登录以及对某些页面的访问限制。

第十一章，“使用 Django 进行 AJAX”，描述了 jQuery 库的基础知识。然后，它展示了使用 Django 进行 AJAX 的实际示例，并解释了这些页面的特点。

第十二章，“使用 Django 进行生产”，解释了如何使用 Django Web 服务器（如 Nginx）和 PostgreSQL Web 系统数据库部署网站。

附录，“速查表”，是对 Django 开发人员有用的常见方法或属性的快速参考。

# 本书所需内容

Django 开发所需的软件如下：

+   Python 3

+   PIP 1.5

+   Django 1.6

# 本书适合对象

本书适用于希望学习如何使用高质量框架创建网站的 Python 开发人员。本书也适用于使用其他语言（如 PHP）的 Web 开发人员，他们希望提高网站的质量和可维护性。本书适用于具有 Python 基础和 Web 基础知识的任何人，他们希望在当今最先进的框架之一上工作。

# 惯例

在本书中，您会发现一些文本样式，用于区分不同类型的信息。以下是一些这些样式的例子，以及它们的含义解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名会显示如下：“我们可以通过使用`settings.py`指令来包含其他上下文。”

代码块设置如下：

```py
from django.conf.urls import patterns, include, url
from django.contrib import admin
admin.autodiscover()
urlpatterns = patterns('',
# Examples:
# url(r'^$', 'Work_msanager.views.home', name='home'),
# url(r'^blog/', include('blog.urls')),
url(r'^admin/', include(admin.site.urls)),
)
```

任何命令行输入或输出都以以下方式书写：

```py
root@debian: wget https://raw.github.com/pypa/pip/master/contrib
/get-pip.py
root@debian:python3 get-pip.py

```

**新术语**和**重要单词**以粗体显示。例如，屏幕上看到的单词，菜单或对话框中的单词会以这种方式出现在文本中：“点击**高级系统设置**。”

### 注意

警告或重要提示会以这样的框出现。

### 提示

提示和技巧会以这种方式出现。
