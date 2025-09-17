# 前言

we2py 是一个用于快速开发安全数据库驱动互联网应用程序的框架。它用 Python 编写，并可以用 Python 编程。它包括库、应用程序和可重用示例。

创建于 2007 年，web2py 在许多使用该框架的开发者的共同努力下，已经取得了巨大的成长、演变和改进。我们感谢他们所有人。

在过去两年中，web2py 发展迅速，以至于很难保持官方文档的时效性。尽管 web2py 始终向后兼容，但已创建了新的 API，提供了解决旧问题的新的方法。

在第三方网站上，如维基、博客和邮件列表中积累了大量的知识。特别是两个资源对 web2py 用户非常有价值：web2py Google Group 和[`www.web2pyslices.com/website`](http://www.web2pyslices.com/website)。然而，那里提供的信息质量参差不齐，因为一些食谱已经过时。

这本书始于收集这些信息、清理它、更新它，并将用户试图解决的重要和常见问题与其他问题分开，这些问题并不代表普遍利益。

用户遇到的最常见问题包括在生产环境中部署 web2py、使用可重用组件构建复杂应用程序、生成 PDF 报告、自定义表单和身份验证、使用第三方库（特别是 jQuery 插件），以及与第三方 Web 服务接口。

收集这些信息并将它们组织成这本书花费了我们超过一年的时间。比列出的作者更多的人有意或无意地做出了贡献。实际上，这里使用的某些代码是基于已经在线发布的代码，尽管这些代码在这里已经被重构、测试和更好地记录。

本书中的代码在 BSD 许可下发布，除非另有说明，并且可在以下列出的专用 GitHub 存储库中在线获取。Python 代码应遵循称为 PEP 8 的风格约定。我们遵循了这一约定来发布在线代码，但在印刷版书籍中压缩了列表，以遵循 Packt 风格指南，并减少对长行的换行需求。

我们相信这本书将是对新 web2py 开发者和经验丰富的开发者都非常有价值的一份资源。我们的目标仍然是使网络变得更加开放和易于访问。我们通过提供 web2py 及其文档来做出贡献，使任何人都能以敏捷和高效的方式构建新的基础设施和服务。

# 本书涵盖的内容

第一章，*部署 web2py*。在本章中，我们讨论了如何配置各种 Web 服务器与 web2py 协同工作。这是生产环境的一个必要设置。我们考虑了最流行的服务器，如 Apache、Cherokee、Lighttpd、Nginx、CGI 和 IIS。相应的配方提供了不同适配器的使用示例，例如`mod_wsgi`、`FastCGI`、`uWSGI`和`ISAPI`。因此，它们可以轻松扩展到许多其他 Web 服务器。使用生产 Web 服务器可以保证静态文件更快地提供服务、更好的并发性和增强的日志记录功能。

第二章，*构建您的第一个应用程序*。我们指导读者通过创建几个非平凡应用程序的过程，包括`Contacts`应用程序、`Reddit`克隆和`Facebook`克隆。这些应用程序都提供了用户认证、通过关系连接的多个表以及 Ajax 功能。在第二部分的章节中，我们讨论了通用 web2py 应用程序的进一步定制，例如构建用于服务静态页面的插件、在页眉中添加标志、自定义菜单以及允许用户选择他们首选的语言。本章的主要重点是模块化和可重用性。

第三章，*数据库抽象层*。DAL 可以说是 web2py 最重要的组件之一。在本章中，我们讨论了从现有来源（csv 文件、`mysql`和`postgresql`数据库）导入模型和数据以及创建新模型的各种方法。我们处理了诸如标记数据和高效使用标签搜索数据库等常见情况。我们使用前序遍历方法实现树表示。我们展示了如何绕过 Google App Engine 平台的一些限制。

第四章，*高级表单*。web2py 的一个优势是它能够自动从数据表示生成表单。然而，不可避免的是，最苛刻的用户会感到需要自定义这些表单。在本章中，我们提供了典型定制的示例，例如添加按钮、添加上传进度条、添加工具提示以及为上传的图像添加缩略图。我们还展示了如何创建向导表单并在一个页面上添加多个表单。

第五章，*添加 Ajax 效果*。本章是上一章的扩展。在这里，我们进一步增强了表单和表格，使用各种 jQuery 插件通过 Ajax 使它们更具交互性。

第六章，*使用第三方库*。web2py 可以使用任何 Python 第三方库。在本章中，我们通过使用随 web2py 一起提供的库（feedparser、`rss`）以及不提供的库（matplotlib）来给出一些示例。我们还提供了一个允许在应用程序级别进行自定义日志记录的配方，以及一个可以检索和显示 Twitter 流的应用程序。

第七章，*Web 服务。计算机可以通过协议进行通信，例如 JSON、JSONRPC、XMLRPC 和 SOAP*。在本章中，我们提供了允许 web2py 基于这些协议创建服务并消费其他服务提供的服务的方法。特别是，我们提供了与 Flex、Paypal、Flickr 和 GIS 集成的示例。

第八章，*认证和授权*。web2py 内置了一个 Auth 模块，用于处理认证和授权。在本章中，我们展示了各种自定义方法，包括向注册和登录表单添加 CAPTCHA，为表示用户添加全球认可的头像（gravatars），以及与使用 OAuth 2.0（例如 Facebook）的服务集成。我们还展示了如何利用`teacher/students`模式。

第九章，*路由配方*。本章包括使用缩短、更简洁和旧 URL 公开 web2py 操作的配方。例如，向 URL 添加前缀或从 URL 中省略应用程序名称。我们还展示了如何使用 web2py 路由机制的高级用法来处理 URL 中的特殊字符，使用 URL 指定首选语言，以及映射特殊文件，如`favicons.ico`和`robots.txt`。

第十章，*报告配方*。在 web2py 中使用标准 Python 库（如`reportlab`或`latex`）创建报告有许多方法。然而，为了方便用户，web2py 附带了一个名为`pyfpdf`的库，由*Mariano Reingart*创建，可以将 HTML 直接转换为 PDF。本章介绍了使用 web2py 模板系统和`pyfpdf`库创建 PDF 报告、列表、标签、徽章和发票的配方。

第十一章，*其他技巧和窍门*。在这里，我们查看那些不适合其他任何章节，但典型 web2py 用户认为很重要的配方。一个例子是使用 Eclipse 与 web2py 结合，Eclipse 是一个非常流行的 Java IDE，可以与 Python 一起使用。其他示例包括如何开发适合移动设备的应用程序，以及如何开发使用 wxPython GUI 的独立应用程序。

# 您需要这本书的内容

所需的唯一软件是 web2py，这是所有配方共有的。web2py 提供源代码版本和适用于 Mac 和 Windows 的二进制版本。可以从[`web2py.com`](http://web2py.com)下载。

我们确实推荐从源代码运行 web2py，在这种情况下，用户还应该安装最新的 Python 2.7 解释器，可以从[`python.org`](http://python.org)下载。

当一个菜谱有额外要求时，会在菜谱中明确说明（例如，有些需要 Windows，有些需要 IIS，有些需要额外的 Python 模块或 jQuery 插件）。

# 本书面向对象

本书针对对 web2py 有基本知识的 Python 开发者，他们希望掌握这个框架。

# 惯例

在这本书中，您将找到许多不同风格的文本，以区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码词如下所示：“使用`Lighttpd`运行 web2py。”

代码块以如下格式设置：

```py
from gluon.storage import Storage

settings = Storage()

settings.production = False
if settings.production:
	settings.db_uri = 'sqlite://production.sqlite'
	settings.migrate = False
else:
	settings.db_uri = 'sqlite://development.sqlite'
	settings.migrate = True

```

当我们希望引起您对代码块中特定部分的注意时，相关的行或项目将以粗体显示：

```py
{{extend 'layout.html'}}
<h2>Companies</h2>
<table>
	{{for company in companies:}}
 <tr>
	<td>
		{{=A(company.name, _href=URL('contacts', args=company.id))}} </td>
	<td>
		{{=A('edit', _href=URL('company_edit', args=company.id))}} </td>
	</tr>
		{{pass}}
<tr>
<td>{{=A('add company', _href=URL('company_create'))}}</td>
</tr>
</table>

```

任何命令行输入或输出都写成如下格式：

```py
python web2py.py -i 127.0.0.1 -p 8000 -a mypassword --nogui

```

**新术语**和**重要词汇**以粗体显示。屏幕上显示的词，例如在菜单或对话框中，在文本中显示如下：“一旦创建了网站，双击以下截图所示的**URLRewrite**：”。

### 注意

警告或重要注意事项以如下框的形式出现。

### 注意

小贴士和技巧看起来像这样。

# 读者反馈

我们欢迎读者的反馈。请告诉我们您对这本书的看法——您喜欢什么或可能不喜欢什么。读者反馈对我们开发您真正从中受益的标题非常重要。

要向我们发送一般反馈，只需发送一封电子邮件到 feedback@packtpub.com，并在邮件的主题中提及书名。

如果有您需要的书籍并希望我们出版，请通过[www.packtpub.com](http://www.packtpub.com)上的**建议标题**表单或发送电子邮件至 suggest@packtpub.com 给我们留言。

如果您在某个主题上有专业知识，并且您有兴趣撰写或为书籍做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您已经是 Packt 书籍的骄傲拥有者，我们有一些事情可以帮助您从您的购买中获得最大收益。

## 下载示例代码

您可以从[`www.PacktPub.com`](http://www.PacktPub.com)上的您的账户下载您购买的所有 Packt 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.PacktPub.com/support`](http://www.PacktPub.com/support)，并注册以将文件直接通过电子邮件发送给您。代码文件也上传到了以下仓库：[`github.com/mdipierro/web2py-recipes-source`](http://https://github.com/mdipierro/web2py-recipes-source)。

所有代码均在 BSD 许可证下发布([`www.opensource.org/licenses/bsd-license.php`](http://www.opensource.org/licenses/bsd-license.php))，除非源文件中另有说明。

## 勘误

尽管我们已经尽一切努力确保我们内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以避免其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/support`](http://www.packtpub.com/support)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站，或添加到该标题的勘误部分下的现有勘误列表中。您可以通过从[`www.packtpub.com/support`](http://www.packtpub.com/support)选择您的标题来查看任何现有的勘误。

## 盗版

在互联网上，版权材料的盗版是所有媒体持续存在的问题。在 Packt，我们非常重视保护我们的版权和许可证。如果您在网上遇到我们作品的任何非法副本，无论形式如何，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 copyright@packtpub.com 与我们联系，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们的作者和为您提供有价值内容的能力方面的帮助。

## 问题

您可以通过 questions@packtpub.com 与我们联系，如果您在本书的任何方面遇到问题，我们将尽力解决。
