# 第二章：第一个应用程序，有多难？

在一个完整的章节中没有一行代码，你需要这个，对吧？在这一章中，我们将逐行解释我们的第一个应用程序；我们还将介绍如何设置我们的环境，开发时使用什么工具，以及如何在我们的应用程序中使用 HTML。

# Hello World

当学习新技术时，人们通常会写一个 Hello World 应用程序，这个应用程序包含启动一个简单应用程序并显示文本"Hello World!"所需的最小可能代码。让我们使用 Flask 来做到这一点。

本书针对**Python 2.x**进行了优化，所以我建议你从现在开始使用这个版本。所有的示例和代码都针对这个 Python 版本，这也是大多数 Linux 发行版的默认版本。

# 先决条件和工具

首先，让我们确保我们的环境已经正确配置。在本课程中，我假设你使用的是类似 Debian 的 Linux 发行版，比如 Mint（[`www.linuxmint.com/`](http://www.linuxmint.com/)）或 Ubuntu（[`ubuntu.com/`](http://ubuntu.com/)）。所有的说明都将针对这些系统。

让我们从以下方式开始安装所需的 Debian 软件包：

```py
sudo apt-get install python-dev python-pip

```

这将安装 Python 开发工具和编译 Python 包所需的库，以及 pip：一个方便的工具，你可以用它来从命令行安装 Python 包。继续吧！让我们安装我们的虚拟环境管理工具：

```py
sudo pip install virtualenvwrapper
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc

```

解释一下我们刚刚做的事情：`sudo`告诉我们的操作系统，我们想要以管理员权限运行下一个命令，`pip`是默认的 Python 包管理工具，帮助我们安装`virtualenvwrapper`包。第二个命令语句添加了一个命令，将`virtualenvwrapper.sh`脚本与控制台一起加载，以便命令在你的 shell 内工作（顺便说一下，我们将使用它）。

# 设置虚拟环境

虚拟环境是 Python 将完整的包环境与其他环境隔离开来的方式。这意味着你可以轻松地管理依赖关系。想象一下，你想为一个项目定义最小必需的包；虚拟环境将非常适合让你测试和导出所需包的列表。我们稍后会讨论这个问题。现在，按下键盘上的*Ctrl* + *Shift* + *T*创建一个新的终端，并像这样创建我们的*hello world*环境：

```py
mkvirtualenv hello
pip install flask

```

第一行创建了一个名为"hello"的环境。你也可以通过输入`deactivate`来停用你的虚拟环境，然后可以使用以下命令再次加载它：

```py
workon hello  # substitute hello with the desired environment name if needed

```

第二行告诉 pip 在当前虚拟环境`hello`中安装 Flask 包。

# 理解"Hello World"应用程序

在设置好环境之后，我们应该使用什么来编写我们美丽的代码呢？编辑器还是集成开发环境？如果你的预算有限，可以尝试使用 Light Table 编辑器（[`lighttable.com/`](http://lighttable.com/)）。免费、快速、易于使用（*Ctrl* + *Spacebar* 可以访问所有可用选项），它还支持工作区！对于这个价钱来说，已经很难找到更好的了。如果你有 200 美元可以花（或者有免费许可证[`www.jetbrains.com/pycharm/buy/`](https://www.jetbrains.com/pycharm/buy/)），那就花钱购买 PyCharm 集成开发环境吧，这几乎是最适合 Python Web 开发的最佳 IDE。现在让我们继续。

创建一个文件夹来保存你的项目文件（你不需要，但如果你这样做，人们会更喜欢你），如下所示：

```py
mkdir hello_world

```

进入新的项目文件夹并创建`main.py`文件：

```py
cd hello_world
touch main.py

```

`main.py`文件将包含整个"Hello World"应用程序。我们的`main.py`内容应该像这样：

```py
# coding:utf-8
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run()
```

### 提示

**下载示例代码**

您可以从您在[`www.packtpub.com`](http://www.packtpub.com)的帐户中下载您购买的所有 Packt 图书的示例代码文件。如果您在其他地方购买了本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接发送到您的电子邮件。

哇！那需要一些打字，对吧？不是吗？是的，我知道。那么，我们刚刚做了什么？

第一行说明我们的`main.py`文件应该使用`utf-8`编码。所有酷孩子都这样做，所以不要对您的非英语朋友不友好，并在所有 Python 文件中使用它（这样做可能有助于您避免在大型项目中出现一些讨厌的错误）。

在第二行和第三行，我们导入我们的 Flask 类并对其进行实例化。我们应用程序的名称是“app”。几乎所有与它相关的东西都与它有关：视图、蓝图、配置等等。参数`__name__`是必需的，并且用于告诉应用程序在哪里查找静态内容或模板等资源。

为了创建我们的“Hello World”，我们需要告诉我们的 Flask 实例在用户尝试访问我们的 Web 应用程序（使用浏览器或其他方式）时如何响应。为此，Flask 有路由。

路由是 Flask 读取请求头并决定哪个视图应该响应该请求的方式。它通过分析请求的 URL 的路径部分，并找到注册了该路径的路由来实现这一点。

在*hello world*示例中，在第 5 行，我们使用路由装饰器将`hello`函数注册到`"/"`路径。每当应用程序接收到路径为`"/"`的请求时，`hello`都会响应该请求。以下代码片段显示了如何检查 URL 的路径部分：

```py
from urlparse import urlparse
parsed = urlparse("https://www.google.com/")
assert parsed.path == "/"
```

您还可以将多个路由映射到同一个函数，如下所示：

```py
@app.route("/")
@app.route("/index")
def hello():
    return "Hello World!"
```

在这种情况下，`"/"`和`"/index"`路径都将映射到`hello`。

在第 6 和第 7 行，我们有一个将响应请求的函数。请注意，它不接收任何参数并且以熟悉的字符串作出响应。它不接收任何参数，因为请求数据（如提交的表单）是通过一个名为**request**的线程安全变量访问的，我们将在接下来的章节中更多地了解它。

关于响应，Flask 可以以多种格式响应请求。在我们的示例中，我们以纯字符串作出响应，但我们也可以以 JSON 或 HTML 字符串作出响应。

第 9 和第 10 行很简单。它们检查`main.py`是作为脚本还是作为模块被调用。如果是作为脚本，它将运行与 Flask 捆绑在一起的内置开发服务器。让我们试试看：

```py
python main.py

```

您的终端控制台将输出类似以下内容：

```py
Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

```

只需在浏览器中打开`http://127.0.0.1:5000/`，即可查看您的应用程序运行情况。

将`main.py`作为脚本运行通常是一个非常简单和方便的设置。通常，您可以使用 Flask-Script 来处理为您调用开发服务器和其他设置。

如果您将`main.py`作为模块使用，只需按以下方式导入它：

```py
from main import what_I_want
```

通常，您会在测试代码中执行类似以下操作来导入应用工厂函数。

这基本上就是关于我们的“Hello World”应用程序的所有内容。我们的世界应用程序缺少的一件事是乐趣因素。所以让我们添加一些；让我们让您的应用程序有趣！也许一些 HTML、CSS 和 JavaScript 可以在这里起作用。让我们试试看！

# 提供 HTML 页面

首先，要使我们的`hello`函数以 HTML 响应，我们只需将其更改为以下内容：

```py
def hello():
    return "<html><head><title>Hi there!</title></head><body>Hello World!</body></html>", 200
```

在上面的示例中，`hello`返回一个 HTML 格式的字符串和一个数字。字符串将默认解析为 HTML，而`200`是一个可选的 HTTP 代码，表示成功的响应。默认情况下返回`200`。

如果您使用*F5*刷新浏览器，您会注意到没有任何变化。这就是为什么当源代码更改时，Flask 开发服务器不会重新加载。只有在调试模式下运行应用程序时才会发生这种情况。所以让我们这样做：

```py
app = Flask(__name__)
app.debug=True
```

现在去你的应用程序正在运行的终端，输入`Ctrl + C`然后重新启动服务器。你会注意到除了你的服务器正在运行的 URL 之外有一个新的输出——关于“stat”的内容。这表示你的服务器将在源代码修改时重新加载代码。这很好，但你注意到我们刚刚犯下的罪行了吗：在处理响应的函数内部定义我们的模板？小心，MVC 之神可能在看着你。让我们把我们定义视图的地方和定义控制器的地方分开。创建一个名为 templates 的文件夹，并在其中创建一个名为`index.html`的文件。`index.html`文件的内容应该像这样：

```py
<html>
<head><title>Hi there!</title></head>
<body>Hello World!</body>
</html>
```

现在改变你的代码像这样：

```py
from flask import Flask, render_response
@app.route("/")
def hello():
    return render_template("index.html")
```

你看到我们做了什么了吗？`render_response`能够从`templates/`文件夹（Flask 的默认文件夹）中加载模板，并且你可以通过返回输出来渲染它。

现在让我们添加一些 JavaScript 和 CSS 样式。默认情况下，Flask 内置的开发服务器会提供`project`文件夹中名为`static`的子文件夹中的所有文件。让我们创建我们自己的文件夹并向其中添加一些文件。你的项目树应该是这样的：

```py
project/
-main.py
-templates/
--index.html
-static/
--js
---jquery.min.js
---foundation.min.js
---modernizr.js
--css
---styles.css
---foundation.min.css
```

注意我从`foundation.zurb`框架中添加了文件，这是一个在[`foundation.zurb.com/`](http://foundation.zurb.com/)上可用的不错的 CSS 框架。我建议你也这样做，以便拥有一个现代、漂亮的网站。你模板中的静态文件路径应该是这样的：

```py
<script src='/static/js/modernizr.js'></script>
```

在真实文件路径之前的`/static`文件夹是 Flask 默认提供的路由，只在调试模式下起作用。在生产环境中，你将需要 HTTP 服务器为你提供静态文件。查看本章附带的代码以获取完整示例。

尝试用一些漂亮的 CSS 样式来改进“hello world”示例！

# 总结

建立开发环境是一项非常重要的任务，我们刚刚完成了这个任务！创建一个“Hello World”应用程序是向某人介绍新技术的好方法。我们也做到了。最后，我们学会了如何提供 HTML 页面和静态文件，这基本上是大多数 Web 应用程序所做的。你在本章中掌握了所有这些技能，我希望这个过程既简单又充实！

在下一章中，我们将通过更加冒险的模板来为我们的挑战增添一些调味。我们将学习如何使用 Jinja2 组件来创建强大的模板，从而让我们在输入更少的情况下做更多的事情。到时见！
