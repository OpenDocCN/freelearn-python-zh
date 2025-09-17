# 第十五章。下一步

在学习 Python 基础之后，接下来是什么？每个开发者的旅程将因他们将要构建的应用程序的一般架构而异。在本章中，我们将探讨四种类型的 Python 应用程序。我们将深入探讨 **命令行界面**（**CLI**）应用程序。我们也会简要地看看 **图形用户界面**（**GUI**）应用程序。我们可以使用许多图形库和框架来完成这项工作；很难涵盖所有替代方案。

网络服务器应用程序通常涉及一个复杂的网络框架，该框架处理标准化的开销。我们的 Python 代码将连接到这个框架。与 GUI 应用程序一样，有几个常用的框架。我们将快速查看网络框架的一些常见功能。我们还将探讨以 Hadoop 服务器流式接口为代表的大数据环境。

这并不是要全面或具有代表性。Python 被以许多不同的方式使用。

# 利用标准库

在实现 Python 解决方案时，扫描标准库以查找相关模块是有帮助的。这个库很大，一开始可能会让人感到有些畏惧。然而，我们可以集中我们的搜索。

我们可以将 *Python 标准库* 文档分为三个部分。前五章是所有 Python 程序员都需要了解的一般参考材料。接下来的 20 章以及第二十八章和第三十二章描述了我们可能将其纳入各种应用程序的模块。剩下的章节不太有用；它们更多地关注 Python 的内部结构和扩展语言本身的方法。

库目录表中的模块名称和摘要可能不足以展示模块可能被使用的所有方式。例如，`bisect` 模块可以被扩展来创建一个快速字典，该字典保留其键的既定顺序。如果不仔细阅读模块的描述，这一点并不明显。

一些库模块具有相对较小、易于理解的实现。对于较大的模块和包，通常有一些可以从上下文中提取出来并广泛重用的部分。例如，考虑一个使用 `http.client` 来进行 REST 网络服务请求的应用程序。我们经常需要 `urllib.parse` 模块中的函数来编码查询字符串或正确引用 URL 的部分。在 Python 应用程序的前端通常可以看到一个长长的导入列表。

# 利用 PyPI – Python 包索引

在扫描库之后，寻找更多 Python 包的下一个地方是位于 [`pypi.python.org/pypi`](https://pypi.python.org/pypi) 的 **Python 包索引**（**PyPI**）。这里列出了数千个包，它们的支持和质量各不相同。

如我们在第一章中所述，*入门*，Python 3.4 还安装了两个脚本来帮助我们添加包，`pip`和`easy_install`。这些工具在 PyPI 上搜索请求的包。大多数包可以通过使用它们的名称找到；工具定位适合平台和 Python 版本的适当版本。

我们在其他章节中提到了一些外部库：

+   使用`nose`编写测试，请参阅[`pypi.python.org/pypi/nose/1.3.6`](https://pypi.python.org/pypi/nose/1.3.6)

+   使用`docutils`编写文档，请参阅[`pypi.python.org/pypi/docutils/0.12`](https://pypi.python.org/pypi/docutils/0.12)

+   使用`Sphinx`编写复杂文档，请参阅[`pypi.python.org/pypi/Sphinx/1.3.1`](https://pypi.python.org/pypi/Sphinx/1.3.1)

此外，还有许多包集合可用：我们可能会安装 Anaconda、NumPy 或 SciPy，每个都包含一个整洁的分发中的多个其他包。请参阅[`continuum.io/downloads`](http://continuum.io/downloads)、[`www.numpy.org`](http://www.numpy.org)或[`www.scipy.org`](http://www.scipy.org)。

在某些情况下，我们可能有一些相互不兼容的 Python 配置。例如，我们可能需要在两个环境中工作，一个使用较旧的 Beautiful Soup 3，另一个使用较新的版本 4。请参阅[`pypi.python.org/pypi/beautifulsoup4/4.3.2`](https://pypi.python.org/pypi/beautifulsoup4/4.3.2)。为了简化这个切换，我们可以使用`virtualenv`工具创建具有自己复杂依赖模块树的隔离 Python 环境。请参阅[`virtualenv.pypa.io/en/latest/`](https://virtualenv.pypa.io/en/latest/)。

Python 生态系统庞大而复杂。没有好的理由在真空中发明解决方案。通常最好找到适当的组件或部分解决方案，然后下载并扩展它们。

# 应用程序类型

我们将探讨四种类型的 Python 应用程序。这些既不是最常见的也不是最受欢迎的 Python 应用程序类型；它们是根据作者的有限经验随机选择的。Python 被广泛使用，任何试图总结 Python 被使用的各种地方的努力都有误导而不是提供信息的风险。

我们将探讨 CLI 应用程序的两个原因。首先，它们可能相对简单，比其他类型的应用程序依赖更少的额外包或框架。其次，更复杂的应用程序通常从 CLI 主脚本启动。出于这些原因，CLI 功能似乎对 Python 的大多数使用都是基本的。

我们将探讨 GUI 应用程序，因为它们在桌面电脑上很受欢迎。这里的困难在于，Python 软件开发中有许多可用的 GUI 框架。以下是一个列表：[`wiki.python.org/moin/GuiProgramming`](https://wiki.python.org/moin/GuiProgramming)。我们将重点关注`turtle`包，因为它简单且内置。

我们将探讨 Web 应用程序，因为 Python 与 Django 或 Flask（以及其他许多框架）一起用于构建高流量网站。以下是一个 Python Web 框架列表：[`wiki.python.org/moin/WebFrameworks`](https://wiki.python.org/moin/WebFrameworks)。我们将重点关注 Flask，因为它相对简单。

我们还将探讨如何使用 Python 与 Hadoop 流进行数据分析。我们不会下载和安装 Apache Hadoop，而是简要介绍如何在我们的桌面上构建和测试管道映射-归约处理。

# 构建命令行应用程序

从第一章的初始脚本示例“入门”中，我们的重点是使用 CLI 脚本学习 Python 基础知识。CLI 应用程序具有许多共同特性：

+   它们通常从标准输入文件读取，写入标准输出文件，并在标准错误文件中产生日志或错误。操作系统保证这些文件始终可用。Python 通过`sys.stdin`、`sys.stdout`和`sys.stderr`提供它们。此外，`input()`和`print()`等函数默认使用这些文件。

+   它们通常使用环境变量进行配置。这些值通过`os.environ`可用。

+   它们也可能依赖于 shell 功能，如将`~`扩展为用户的家目录，这是由`os.path.expanduser()`完成的。

+   它们通常解析命令行参数。虽然变量`sys.argv`包含参数字符串，但直接使用它们很麻烦。我们将使用`argparse`模块来定义参数模式，解析字符串，并创建一个包含相关参数值的对象。

这些基本功能涵盖了多种编程选择。例如，一个网络服务器可以被视为一个永远运行的 CLI 程序，它从特定的端口号接收请求。一个 GUI 应用程序可能从命令行开始，但随后打开窗口以允许用户交互。

## 使用 argparse 获取命令行参数

我们将使用`argparse`模块创建一个解析器来使用命令行参数。一旦配置完成，我们可以使用这个解析器创建一个小的命名空间对象，该对象包含在命令行上提供的所有参数值，或者包含默认值。我们的应用程序可以使用这个对象来控制其行为。

通常，我们希望将命令行处理与我们的应用程序的其他部分隔离开。以下是一个处理解析的函数，然后使用解析的选项调用另一个函数来完成实际工作：

```py
logger= logging.getLogger(__name__)
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",
        action="store_const", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument("c", type=float)
    options= parser.parse_args()

    logging.getLogger().setLevel(options.verbose)
    logger.debug("Converting '{0!r}'".format(options.c))
    convert(options.c)
```

我们使用所有默认参数构建了一个`ArgumentParser`方法。我们本可以识别程序名称，提供使用说明，或者当有人使用`-h`选项获取帮助时显示任何其他内容。我们省略了这些额外的文档，以保持示例简洁。

我们为这个应用程序定义了两个参数：一个可选参数和一个位置参数。可选参数`-v`或`--verbose`在结果选项集合中存储一个常量值。这个属性的名称是参数的长名称，即`verbose`。提供的常量是`logging.DEBUG`；如果选项不存在，则默认值为`logging.INFO`。

位置参数`c`在解析完所有选项之后接受一个命令行参数。`nargs`的值可以省略；它可以设置为`'*'`以收集所有参数。我们提供了一个要求，即输入值通过`float()`函数转换，这意味着在参数解析期间将拒绝非数值值并显示错误。这将设置为结果对象的`c`属性。

当我们评估`parse_args()`方法时，定义的参数用于解析`sys.argv`中的命令行值。`options`对象将具有结果值或默认值。

在`main()`函数的第二部分，我们使用`options`对象通过`verbose`参数值设置根日志记录器的日志级别。然后我们使用全局`logger`对象将单个位置参数值输出，该值将被分配给`options`对象的`c`属性。

最后，我们使用输入参数值评估我们的应用程序函数；解析器将此分配给`options.c`变量。执行实际工作的函数设计为完全独立于用于调用它的命令行界面。该函数接受一个浮点值并将结果打印到标准输出。它可以利用模块全局`logger`对象。

我们设计命令行应用程序的目标是将有用的工作与所有界面考虑完全分离。这允许我们导入执行实际工作的函数，并从单个组件构建更大或更复杂的应用程序。这通常意味着命令行参数被转换为普通函数参数或类构造函数参数。

## 使用 cmd 模块进行交互式应用程序

一些命令行应用程序需要用户交互。例如，`sftp`命令可以从命令行使用，以与服务器交换文件。我们可以使用 Python 的`cmd`模块创建类似交互式应用程序。

要构建更复杂的交互式应用程序，我们可以创建一个扩展`cmd.Cmd`类的类。这个类中任何以`do_`开头命名的函数定义了一个交互式命令。例如，如果我们定义了一个方法`do_get()`，这意味着我们的应用程序现在有一个交互式的`get`命令。

用户输入`get`命令后的任何后续文本都将作为参数提供给`do_get()`方法。然后`do_get()`函数负责对命令之后的文本进行进一步解析和处理。

我们可以创建此类的一个实例，并调用继承的 `cmdloop()` 方法来拥有一个工作着的交互式应用程序。这使我们能够非常快速和简单地部署一个工作着的交互式应用程序。虽然我们受限于字符模式、命令行界面，但我们可以轻松地添加功能而无需做太多额外的工作。

# 构建 GUI 应用程序

我们可以区分仅与图形工作以及深度交互的应用程序。在前一种情况下，我们可能有一个命令行应用程序，它创建或修改图像文件。在第二种情况下，我们将定义一个响应输入事件的程序。这些交互式应用程序创建一个事件循环，它接受鼠标点击、屏幕手势、键盘字符和其他事件，并对这些事件做出响应。在某种程度上，GUI 程序的唯一独特特性是它响应事件的广泛多样性。

`tkinter` 模块是 Python 和 **Tk** 用户界面小部件工具包之间的接口。此模块帮助我们构建丰富的交互式应用程序。当我们使用 Python 内置的 IDLE 编辑器时，我们正在使用一个用 `tkinter` 构建的应用程序。`tkinter` 模块文档包括关于 **Tk** 小部件的背景信息。

`turtle` 模块还依赖于底层的 **Tk** 图形。此模块还允许我们构建简单的交互式应用程序。turtle 的想法来自 Logo 编程语言，其中图形命令用于使一个`turtle`在绘图空间中移动。`turtle` 模型为某些类型的图形提供了一种非常方便的规范。例如，绘制一个旋转的矩形可能涉及到一个相当复杂的计算，包括正弦和余弦来确定四个角最终的位置。或者，我们可以指导 turtle 使用诸如 `forward(w)`、`forward(l)` 和 `right(90)` 等命令，从任何起始位置和任何初始旋转绘制大小为 *w* × *l* 的矩形。

为了让学习 Python 更容易，`turtle` 模块提供了一些基本的类，这些类实现了 `Screen` 和 `Turtle`。该模块还包括一个丰富的函数集合，这些函数隐式地与单例 `Turtle` 和 `Screen` 对象一起工作，消除了设置图形环境的需求。对于初学者来说，这个仅提供函数的环境是一种简单的动词语言，可以用来学习编程的基础。

简单的程序看起来像这样：

```py
from turtle import *

def on_screen():
    x, y = pos()
    w, h = screensize()
    return -w <= x < w and -h <= y < h

def spiral(angle, incr, size=10):
    while on_screen():
        right(angle)
        forward(size)
        size *= incr
```

我们使用 `from turtle import *` 来引入所有单个函数。这是初学者的常见设置。

我们定义了一个函数，`on_screen()`，它将 `pos()` 函数给出的 turtle 位置与 `screensize()` 函数给出的屏幕整体大小进行比较。我们的函数使用一个简单的逻辑表达式来确定当前 turtle 位置是否仍然在显示边界内。

对于学习编程的人来说，`pos()`和`screensize()`函数的实现细节可能并不那么有帮助。更高级的程序员可能想知道`pos()`函数使用单例全局`Turtle`实例的`Turtle.pos()`方法。同样，`screensize()`函数使用单例全局`Screen`实例的`Screen.screensize()`方法。

`spiral()`函数将使用定义螺旋线段的三参数来绘制螺旋形状。这个函数依赖于`turtle`包中的`right()`和`forward()`函数来设置海龟的方向并绘制线段。虽然`forward()`函数绘制的线段端点的计算可能涉及一点三角学，但新程序员能够学习迭代的基本知识，而无需与正弦或余弦函数纠缠。

这是我们如何使用这个函数的方法：

```py
if __name__ == "__main__":
    speed(10)
    spiral(size=10, incr=1.05, angle = 67)
    done()
```

作为初始化的一部分，我们将海龟的速度设置为 10，这相当快。对于在循环或条件语句上遇到困难的人来说，较慢的速度可以帮助他们在观察海龟时更好地跟随代码。我们已经使用一组参数值评估了`spiral()`函数。

`done()`函数将启动一个 GUI 事件处理循环，等待用户交互。我们在绘制有趣的部分之后启动了循环，因为唯一预期的事件是图形窗口的关闭。当用户关闭窗口时，`done()`函数也会结束。然后我们的脚本可以正常结束。

如果我们要构建更复杂的交互式应用程序，有一个合适的`mainloop()`函数可以使用。这个函数可以捕获事件，使我们的程序能够对这些事件做出响应。

Logo 语言及其相关的`turtle`包允许初学者在不必须一次掌握太多细节的情况下学习编程的基本知识。`turtle`包并不是为了产生与**matplotlib**或**Pillow**等包相同类型的复杂技术图形。

## 使用更复杂的包

我们可以使用 Pillow 库创建复杂的图像处理应用程序。这个包允许我们创建大图像的缩略图，转换图像格式，并验证文件实际上是否包含编码的图像数据。我们还可以使用这个包创建简单的科学图形，显示数据点的二维图。这个包并不是为了构建完整的 GUI，因为它不会为我们处理输入事件。更多信息，请参阅[`pypi.python.org/pypi/Pillow/2.8.1`](https://pypi.python.org/pypi/Pillow/2.8.1)。

对于数学、科学和统计工作，matplotlib 包被广泛使用。这个包包括创建二维和三维基本数据图的非常复杂工具。这个包与 SciPy 和 Anaconda 捆绑在一起。更多信息，请参阅[`matplotlib.org`](http://matplotlib.org)。

有几个更通用的图形框架。其中一个常用于学习 Python 的是**Pygame**框架。它包含大量组件，包括图形、声音和图像处理工具。Pygame 包包括多个图形驱动程序，能够以平滑的方式处理大量移动对象。请参阅[`www.pygame.org/news.html`](http://www.pygame.org/news.html)。

# 构建 Web 应用程序

Web 应用程序涉及大量的处理，这最好描述为样板代码。例如，HTTP 协议的基本处理通常是标准化的，有库可以优雅地处理它。解析请求头和将 URL 路径映射到特定资源的细节不需要重新发明。

然而，简单地处理 HTTP 协议和将 URL 映射到特定应用程序资源之间存在深刻的区别。这两个层次推动了**Web 服务网关接口**（**WSGI**）设计和`wsgi`模块在标准库中的定义。有关更多信息，请参阅**Python 增强提案**（**PEP**）3333，[`www.python.org/dev/peps/pep-3333/`](https://www.python.org/dev/peps/pep-3333/)。

WSGI 背后的想法是所有 Web 服务都应该遵守处理 HTTP 请求和响应细节的单个、最小标准。这个标准允许复杂的 Web 服务器包含各种 Python 工具和框架，这些工具和框架通过 WSGI 组合在一起，以确保组件正确互联。URL 到资源的映射必须在标准上下文中处理。

可以将`mod_wsgi`模块插入到 Apache HTTPD 服务器中。此模块将在 Apache 前端和后端 Python 实例之间传递请求和响应。通过一点规划，我们可以确保静态内容（如图形、样式表、JavaScript 库等）由前端 Web 服务器处理。动态内容（如 HTML 页面、XML 或 JSON 文档）则由我们的 Python 应用程序处理。

有关`mod_wsgi`的更多信息，请参阅[`www.modwsgi.org/`](http://www.modwsgi.org/)。

## 使用 Web 框架

在这个背景下，Web 应用程序通常使用一个解析 URL 并调用 Python 函数以返回由 URL 定位的资源框架来构建。虽然这显然是创建 Web 服务器所需的最小要求，但通常还有大量我们希望拥有的附加功能。

例如，身份验证和授权是我们经常需要且希望不需要实现的功能。与一个允许我们添加 OAuth 客户端代码的框架一起工作会更好。使用 cookie 的网站也将从具有无缝集成的会话管理功能中受益。

许多网站提供 RESTful 网络服务。有时这些服务是数据库访问的薄包装。当数据库是关系型时，我们通常需要一个 **对象关系映射器**（**ORM**）层，它允许我们通过 RESTful 服务暴露更完整的对象。这也是一个良好的网络服务器框架选项。

在 Python 中提供网络服务有两种主要方法：套件和组件。套件方法以 Django 等包为代表，这些包提供了一个统一集合中的几乎所有可能需要的模块和包。请参阅 [`www.djangoproject.com`](https://www.djangoproject.com)。

组件方法可以在 Flask 等项目中看到。这被称为 **微框架**，因为它相对较少。Flask 服务器专注于 URL 路由，使其非常适合构建 RESTful 服务。它可能包括会话管理，使其可用于 HTML 网站。它与 Jinja2、WTForms、SQLAlchemy、OAuth 认证模块和其他许多模块很好地协作。有关更多信息，请参阅 [`flask.pocoo.org/docs/0.10/`](http://flask.pocoo.org/docs/0.10/)。

## 使用 Flask 构建 RESTful 网络服务

我们将演示一个非常简单的网络服务。我们将使用之前在 turtle 示例中展示的算法，进行一些小的修改，以创建动态图形下载。为了更容易创建可下载的文件，我们将放弃简单的 turtle 图形包，并使用 Pillow 包来创建图像文件。许多网站使用 Pillow 来验证上传的图像并创建缩略图。它是任何使用图像的网站的必要组成部分。

关于 Pillow 的更多信息，请参阅 [`pypi.python.org/pypi/Pillow/2.8.1`](https://pypi.python.org/pypi/Pillow/2.8.1)。

网络服务必须对 HTTP 请求提供资源。一个简单的 Flask 网站将有一个整体的应用程序对象和多个路由，这些路由将 URL（以及可能的方法名称）映射到函数。

这里有一个简单的例子：

```py
from flask import Flask, request
from PIL import Image, ImageDraw, ImageColor
import tempfile

spiral_app = Flask(__name__)

@spiral_app.route('/image/<spec>', methods=('GET',))
def image(spec):
    spec_uq= urllib.parse.unquote_plus(spec)
    spec_dict = urllib.parse.parse_qs(spec_uq)
    spiral_app.logger.info( 'image spec {0!r}'.format(spec_dict) )
    try:
        angle= float(spec_dict['angle'][0])
        incr= float(spec_dict['incr'][0])
        size= int(spec_dict['size'][0])
    except Exception as e:
        return make_response('URL {0} is invalid'.format(spec), 403)

    # Working dir should be under Apache Home.
    _, temp_name = tempfile.mkstemp('.png')

    im = Image.new('RGB', (400, 300), color=ImageColor.getrgb('white'))
    pen= Pen(im)
    spiral(pen, angle=angle, incr=incr, size=size)
    im.save(temp_name, format='png')

    # Should redirect so that Apache serves the image.
    spiral_app.logger.debug( 'image file {0!r}'.format(temp_name) )
    with open(temp_name, 'rb' ) as image_file:
        data = image_file.read()
    return (data, 200, {'Content-Type':'image/png'})
```

此示例展示了 Flask 应用程序的核心三个特性。此脚本定义了一个 `Flask` 实例。我们基于文件名定义了该实例，对于主脚本，它将是 `"__main__"`，而对于导入的脚本，它将是模块名。我们将该 `Flask` 容器分配给一个变量 `spiral_app`，以便在整个模块文件中使用。

一个更复杂的 Flask 应用程序可能在一个子模块包中包含多个单独的视图函数。这些中的每一个都将依赖于全局 Flask 应用程序。

我们通过`image()`函数创建图像资源。为此函数提供了一个`route`装饰器，它显示了 URL 路径以及与此资源一起工作的方法。为 HTTP 协议定义了大量的方法。许多 RESTful 网络服务专注于 POST、GET、PUT 和 DELETE，因为这些与常用的**创建、检索、更新和删除**（**CRUD**）规则相匹配，这些规则通常用于总结数据库操作。

我们将`image()`函数分解为四个独立的部分。首先，我们需要解析 URL。`route`包括一个占位符`<spec>`，Flask 会解析并提供给函数作为参数。这将是一个用于描述螺旋的 URL 编码参数。它可能看起来像这样：

```py
http://127.0.0.1:5000/image/size=10&angle=65.0&incr=1.05
```

一旦我们解码了规范，我们将有一个特殊的多元值字典。这看起来像是来自 HTML 表单的输入。结构将是表单字段名称到每个字段值的列表的映射。对象看起来像这样：

```py
{'size': ['10'], 'angle': ['65.0'], 'incr': ['1.05']}
```

`image()`函数只使用每个项目中的一个值；每个输入都必须转换为数值。我们将所有潜在的异常收集到一个单独的`except`子句中，从而掩盖了任何错误输入的细节。我们使用 Flask 的`make_response()`函数构建一个包含错误消息和状态码 403（“禁止”）的响应。一个更复杂的函数会使用**Accept**头根据客户端声明的偏好将响应格式化为 JSON 或 XML。我们将其保留为默认的 MIME 类型 text/plain。

图像被保存到一个临时文件中，该文件是用`tempfile.mkstemp()`函数创建的。在这种情况下，我们将从 Flask 应用程序中保存那个临时文件。对于低流量网站，这是可以接受的。对于高流量网站，Python 应用程序永远不应该处理下载。文件应该创建在 Apache HTTPD 服务器可以下载图像的目录中，而不是 Python 应用程序。

图像构建使用了一些 Pillow 定义的对象来定义绘图空间。一个定制的类定义了一个`Pen`实例，它与`turtle.Turtle`类平行。一旦图像构建完成，它就会使用给定的文件名保存。请注意，Pillow 包可以以多种格式保存文件；在这个例子中我们使用了`.png`格式。

最后一个部分下载文件。注释指出，高流量网站会重定向到一个 URL，Apache 会从该 URL 下载图像文件。这使 Flask 服务器能够处理另一个请求。

注意，在这个函数的本地命名空间中会有两个图像副本。`im`变量将保存整个详细的图像。`data`变量将保存图像文档的压缩文件系统版本。我们可以使用`del im`来删除图像对象；然而，通常将此分解为两个函数更好，这样命名空间会为我们处理对象删除。

我们可以使用以下脚本运行此服务的演示版本：

```py
if __name__ == '__main__':
    spiral_app.run(debug=True)
```

这允许我们在桌面上使用一个运行中的 Web 服务器。然后我们可以尝试不同的实现方案。

这个例子的重要之处在于我们可以——非常快速地——在我们的桌面环境中运行一个服务。然后我们可以轻松地探索和实验用户体验。例如，由于图像将嵌入到 HTML 页面中，我们希望为该页面设计和调试 HTML、CSS 和 JavaScript。当我们有一个简单、易于调整的 Web 服务器时，整个开发过程会变得更加容易。

# 连接到 MapReduce 框架

关于 Apache Hadoop 服务器的背景信息，请参阅[`hadoop.apache.org`](https://hadoop.apache.org)。以下是摘要：

> *Apache Hadoop 软件库是一个框架，它允许使用简单的编程模型在计算机集群上分布式处理大型数据集。它旨在从单个服务器扩展到数千台机器，每台机器都提供本地计算和存储。*

Hadoop 分布式处理的一部分是 MapReduce 模块。此模块允许我们将数据分析分解为两个互补的操作：映射和缩减。这些操作在 Hadoop 集群中分布，以并发运行。映射操作处理集群中分散的数据集的所有行。然后，映射操作的输出被输送到缩减操作以进行汇总。

Python 程序员可以使用 Hadoop 流接口。这涉及到一个 Hadoop“包装器”，它将数据作为标准输入文件呈现给 Python 映射程序。映射程序的标准输出必须是制表符分隔的键值对。这些被发送到缩减程序，再次作为标准输入。有关帮助 Python 程序员使用 Hadoop 的包的更多信息，请参阅[`blog.cloudera.com/blog/2013/01/a-guide-to-python-frameworks-for-hadoop/`](http://blog.cloudera.com/blog/2013/01/a-guide-to-python-frameworks-for-hadoop/)。

MapReduce 操作的一个常见示例是创建书中找到的单词的索引。映射操作将巨型文本文件转换为文本文件中找到的单词序列。缩减操作将计算每个单词的出现次数，从而得出单词及其流行度的最终汇总。（有关其重要性的更多信息，请访问 NLTK 网站：[`www.nltk.org`](http://www.nltk.org)。）

实际问题可能涉及多个映射和多个缩减。在许多情况下，映射似乎很简单：它们会从源数据中的每一行提取一个键和一个值。我们不会过多地研究 Hadoop，而是展示如何在我们的桌面上编写和测试映射器和缩减器。

我们的目标是拥有两个程序，`map.py`和`reduce.py`，它们可以组合成如下流：

```py
cat some_file.dat | python3 map.py | sort | python3 reduce.py
```

这种方法将通过向我们的`map.py`程序和`reduce.py`程序提供数据来模拟 Hadoop 流。这将成为我们映射和减少处理的一个简单集成测试。对于 Windows，我们将使用`type`命令而不是 Linux 的`cat`程序。

让我们看看美国国家海洋和大气管理局国家气候数据中心的一些原始气候数据。有关在线气候数据，请参阅[`www.ncdc.noaa.gov/cdo-web/`](http://www.ncdc.noaa.gov/cdo-web/)。我们可以请求包含特定时间段降雪等详细信息的文件。

我们的问题是“哪几个月在弗吉尼亚州里士满机场有降雪？”降雪数据属性名为`TSNW`。它的单位是 1/10 英寸，因此我们的映射器需要将其转换为`Decimal`英寸，以便更实用。

我们可以编写一个看起来像这样的映射脚本：

```py
import csv
import sys
import datetime
from decimal import Decimal
if __name__ == "__main__":
    rdr = csv.DictReader(sys.stdin)
    wtr = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    for row in rdr:
        date = datetime.datetime.strptime(row['DATE'], "%Y%m%d").date()
        if row['TSNW'] in ('0', '-9999', '9999'):
            continue # Zero or equipment error: reject
        wtr.writerow( [date.month, Decimal(row['TSNW'])/10] )
```

由于我们的输入在大约标准 CSV 表示法中——带有标题——我们可以使用`csv.DictReader`对象来解析输入。每一行数据都是一个`dict`对象，其键由 CSV 文件的第一行定义。输出更加专业化：在 Hadoop 中，它必须是一个制表符分隔的关键字和值，以换行符结束。

对于每个输入字典对象，我们将日期从文本转换为适当的 Python 日期，以便我们可以可靠地提取月份。我们可以通过使用`row['DATE'][4:6]`来完成此操作，但这似乎很模糊。映射器包括一个过滤器，以拒绝没有降雪的月份，或者有特定领域的空值（9999 或-9999）而不是测量值。

输出是一个键和一个值。我们的键是报告的月份；值是将十分之一英寸的降雪转换为英寸测量值。我们使用了`Decimal`类来避免引入浮点近似。

减少操作使用`Counter`对象来总结映射器产生的结果。对于这个例子，减少操作看起来像这样：

```py
import csv
import sys
from collections import Counter
from decimal import Decimal
if __name__ == "__main__":
    rdr= csv.DictReader(
        sys.stdin, fieldnames=("month","snowfall"),
        delimiter='\t', lineterminator='\n')
    counts = Counter()
    for line in rdr:
        counts[line['month']] += Decimal(line['snowfall'])
    print( counts )
```

减少读取器与映射器的写入器相匹配：它们都使用制表符作为分隔符，使用换行符作为行终止符。这遵循了 Hadoop 对从映射器到减少器的数据的要求。我们还创建了一个`Counter`对象来存储我们的降雪数据。

对于每一行输入，我们提取降雪英寸数，并将这些数累加到以月份数字为键的`Counter`对象中。最终结果将显示大里士满都会区每个月的降雪英寸数。

我们可以很容易地在我们的桌面上测试和实验这个。我们可以使用 shell 脚本或可能是一个像这样的小包装程序来执行映射器、排序和减少器的管道：

```py
import subprocess
dataset = "526212.csv"
command = """cat {dataset} | python3 -m Chapter_15.map | sort |
    python3 -m Chapter_15.reduce"""
command = command.format_map(locals())
result= subprocess.check_output(command, shell=True)
for line in result.splitlines():
      print( line.decode("ASCII") )
```

我们创建了一个在 Mac OS X 或 Linux 上工作的命令，并将文件名替换到该命令中。对于 Windows，我们可以使用`type`而不是`cat`；Python 程序可能被命名为`python`而不是`python3`。否则，shell 管道应该在 Windows 上正常工作。

我们使用了 `subprocess.check_output()` 函数来运行这个 shell 命令并收集输出。这是一种快速实验我们的 Hadoop 程序的方法，同时避免了使用繁忙的 Hadoop 集群相关的延迟。

只要我们坚持使用在 Hadoop 环境中正确安装的库元素，这种方法就能很好地工作。在某些情况下，我们的集群可能已经安装了 Anaconda，这使我们能够访问各种包。当我们想使用自己的包——一个在整个集群中未安装的包时，我们需要将额外的模块提供给 Hadoop 流命令，以确保我们的额外模块被下载到集群中的每个节点，包括我们的映射器和归约器。

# 摘要

在本章中，我们探讨了多种 Python 应用程序。虽然 Python 被广泛使用，但我们选择了一些重点关注的领域。我们研究了能够处理大量数据的 CLI 应用程序。命令行界面也存在于其他类型的应用程序中，这使得它成为任何程序的基本部分。

我们还研究了使用内置的 `turtle` 模块进行 GUI 程序。广泛使用的 GUI 框架涉及下载、安装和更复杂的编程，我们无法在一个章节中展示。有几个流行的选择；对于 GUI 应用程序来说，没有关于“最佳”包的共识。做出选择是困难的。

我们还研究了使用 Flask 模块的网络应用程序。这也是一个单独的下载。在许多情况下，有许多相关的下载将成为网络应用程序的一部分。我们可能包括 Jinja2、WTForms、OAuth、SQLAlchemy 和 Pillow，以扩展网络服务器的库。

我们还研究了如何利用桌面 Python 来开发 Hadoop 应用程序。我们不必下载和安装 Hadoop，可以创建一个遵循 Hadoop 方法的处理管道。我们可以仅使用桌面工具编写映射器和归约器，这样我们就可以创建可靠的单元测试。这使我们有了信心，当我们使用完整的数据集在 Hadoop 集群上运行应用程序时，我们会得到预期的结果。

当然，这还不是全部。Python 可以作为自动化其他应用程序的语言在另一个应用程序中使用。一个程序可以嵌入一个 Python 解释器，该解释器与整体应用程序交互。有关更多信息，请参阅 [`docs.python.org/2/extending/embedding.html`](https://docs.python.org/2/extending/embedding.html)。

我们可以将 Python 应用程序的世界想象成一个充满岛屿、群岛、湾口和河口的大水体。美国东海岸的切萨皮克湾就是一个例子。我们试图展示这个湾的主要特征：海角、尖端、浅滩和海岸线。我们避开了洋流、天气和潮汐的影响，以便我们可以专注于湾的必要特征。沿着特定路线进行实用导航需要更深入地研究感兴趣的区域：详细的航海图、飞行员指南以及来自其他船员的当地知识。

考虑 Python 世界的广度是很重要的。到达目的地的距离可能会显得令人畏惧。我们的目标一直是展示一些主要航标，这些航标可以帮助将漫长的航行分解成更短的航程。如果我们孤立出漫长旅程的各个部分，我们就可以分别解决它们，并从这些部分构建出一个更大的解决方案。
