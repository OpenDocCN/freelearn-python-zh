# 前言

情景很熟悉：你是一名网页开发者，已经使用过几种编程语言、框架和环境，决定学习足够的 Python 来制作一些玩具网页应用程序。也许你已经使用过一些 Python 网页框架来构建一个或两个应用程序，并且想探索一些你一直听说过的替代选项。

这通常是人们了解 Flask 的方式。

作为一个微框架，Flask 旨在帮助你，然后不再干涉你。与大多数其他通用网页框架采取非常不同的方法，Flask 由一个非常小的核心组成，处理和规范化 HTTP 和 WSGI 规范（通过 Werkzeug），并提供一个非常好的模板语言（通过 Jinja2）。Flask 的美妙之处在于其固有的可扩展性：由于它从一开始就被设计为做得很少，因此也很容易扩展。这样做的一个愉快的结果是，你不必受制于特定的数据库抽象层、身份验证协议或缓存机制。

学习一个新的框架不仅仅是学习提供给你的基本功能和对象：同样重要的是学习如何调整框架以帮助你构建应用程序的特定要求。

本书将演示如何使用 Python 网页微框架开发一系列网页应用程序项目，并利用扩展和外部 Python 库/API 来扩展各种更大更复杂的网页应用程序的开发。

# 本书内容

第一章，“从正确的角度开始-使用 Virtualenv”，开始了我们对 Python 网页应用程序开发的深入探讨，介绍了使用和管理虚拟环境来隔离应用程序依赖关系的基础知识。我们将研究安装和分发可重用的 Python 代码包的设置工具、pip、库和实用程序，以及 virtualenv，这是一个用于创建项目的基于 Python 软件要求的隔离环境的工具。我们还将讨论这些工具无法做到的事情，并研究 virtualenvwrapper 抽象，以增强 virtualenv 提供的功能。

第二章，“从小到大-扩展 Flask 应用程序结构”，探讨了你可能考虑为 Flask 应用程序考虑的各种基线布局和配置。随着我们从最简单的单文件应用程序结构逐渐进展到更复杂的多包蓝图架构，我们概述了每种方法的利弊。

第三章，“Snap-代码片段共享应用程序”，构建了我们的第一个简单的 Flask 应用程序，重点是学习最流行的关系数据库抽象之一，SQLAlchemy，以及一些最流行的 Flask 扩展：Flask-Login 用于处理经过身份验证的用户登录会话，Flask-Bcrypt 确保帐户密码以安全方式存储，Flask-WTF 用于创建和处理基于表单的输入数据。

第四章，“Socializer-可测试的时间轴”，为社交网页应用程序构建了一个非常简单的数据模型，主要关注使用 pytest，Python 测试框架和工具进行单元和功能测试。我们还将探讨应用程序工厂模式的使用，该模式允许我们为简化测试目的实例化我们应用程序的不同版本。此外，详细描述了 Blinker 库提供的常常被省略（和遗忘）的信号的使用和创建。

第五章，*Shutterbug，Photo Stream API*，围绕基于 JSON 的 API 构建了一个应用程序的框架，这是当今任何现代 Web 应用程序的要求。我们使用了许多基于 API 的 Flask 扩展之一，Flask-RESTful，用于原型设计 API，我们还深入研究了无状态系统的简单身份验证机制，并在此过程中编写了一些测试。我们还短暂地进入了 Werkzeug 的世界，这是 Flask 构建的 WSGI 工具包，用于构建自定义 WSGI 中间件，允许无缝处理基于 URI 的版本号，以适应我们新生 API 的需求。

第六章，*Hublot – Flask CLI Tools*，涵盖了大多数 Web 应用程序框架讨论中经常省略的一个主题：命令行工具。解释了 Flask-Script 的使用，并创建了几个基于 CLI 的工具，以与我们应用程序的数据模型进行交互。此外，我们将构建我们自己的自定义 Flask 扩展，用于包装现有的 Python 库，以从 GitHub API 获取存储库和问题信息。

第七章，*Dinnerly – Recipe Sharing*，介绍了 OAuth 授权流程的概念，这是许多大型 Web 应用程序（如 Twitter、Facebook 和 GitHub）实施的，以允许第三方应用程序代表帐户所有者行事，而不会损害基本帐户安全凭据。为食谱共享应用程序构建了一个简单的数据模型，允许所谓的社交登录以及将数据从我们的应用程序跨发布到用户连接的服务的 feeds 或 streams。最后，我们将介绍使用 Alembic 的数据库迁移的概念，它允许您以可靠的方式将 SQLAlchemy 模型元数据与基础关系数据库表的模式同步。

# 本书需要什么

要完成本书中大多数示例的操作，您只需要您喜欢的文本编辑器或 IDE，访问互联网（以安装各种 Flask 扩展，更不用说 Flask 本身了），一个关系数据库（SQLite、MySQL 或 PostgreSQL 之一），一个浏览器，以及对命令行的一些熟悉。我们已经注意到了在每一章中完成示例所需的额外软件包或库。

# 本书适合谁

本书是为希望深入了解 Web 应用程序开发世界的新 Python 开发人员，或者对学习 Flask 及其背后的基于扩展的生态系统感兴趣的经验丰富的 Python Web 应用程序专业人士而创建的。要充分利用每一章，您应该对 Python 编程语言有扎实的了解，对关系数据库系统有基本的了解，并且熟练掌握命令行。

# 约定

本书中，您将找到许多不同类型信息的文本样式。以下是一些样式的示例，以及它们的含义解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下："这将创建一个空的`app1`环境并激活它。您应该在 shell 提示符中看到一个（app1）标签。"

代码块设置如下：

```py
[default]
  <div>{{ form.password.label }}: {{ form.password }}</div>
  {% if form.password.errors %}
  <ul class="errors">{% for error in form.password.errors %}<li>{{ error }}</li>{% endfor %}</ul>
  {% endif %}

  <div><input type="submit" value="Sign up!"></div>
</form>

{% endblock %}
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目以粗体显示：

```py
    from application.users.views import users
    app.register_blueprint(users, url_prefix='/users')

 from application.posts.views import posts
 app.register_blueprint(posts, url_prefix='/posts')

        # …
```

任何命令行输入或输出都以以下方式编写：

```py
$ source ~/envs/testing/bin/activate
(testing)$ pip uninstall numpy

```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如菜单或对话框中的单词，会以这样的方式出现在文本中："然后它断言返回的 HTML 中出现了**注册！**按钮文本"。

### 注意

警告或重要说明显示在这样的框中。

### 提示

提示和技巧显示如下。
