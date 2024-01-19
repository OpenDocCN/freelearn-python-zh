# 让我们开始我们的自动化之旅

在本章中，我们将介绍以下内容：

+   创建虚拟环境

+   安装第三方包

+   创建带有格式化值的字符串

+   操作字符串

+   从结构化字符串中提取数据

+   使用第三方工具—parse

+   介绍正则表达式

+   深入了解正则表达式

+   添加命令行参数

# 介绍

本章的目标是介绍一些基本技术，这些技术将在整本书中都很有用。主要思想是能够创建一个良好的 Python 环境来运行接下来的自动化任务，并能够将文本输入解析为结构化数据。

Python 默认安装了大量工具，但也很容易安装第三方工具，这些工具可以简化处理文本时的常见操作。在本章中，我们将看到如何从外部来源导入模块并使用它们来充分发挥 Python 的潜力。

在任何自动化任务中，结构化输入数据的能力至关重要。本书中大部分将处理的数据来自未格式化的来源，如网页或文本文件。正如古老的计算机格言所说，*垃圾进，垃圾出*，因此对输入进行消毒非常重要。

# 创建虚拟环境

在使用 Python 时的第一步是明确定义工作环境。这有助于脱离操作系统解释器和环境，并正确定义将要使用的依赖关系。不这样做往往会产生混乱的情况。记住，*显式优于隐式！*

这在两种情况下尤为重要：

+   在同一台计算机上处理多个项目时，它们可能具有在某些时候会发生冲突的不同依赖关系。例如，不能在同一环境中安装同一模块的两个版本。

+   在开发将最终在远程服务器上运行的个人笔记本电脑上开发一些代码等情况下，需要在不同计算机上使用的项目上工作。

开发人员之间的一个常见笑话是对错误的回应是*它在我的机器上运行*，意思是它似乎在他们的笔记本电脑上工作，但在生产服务器上却不工作。尽管有大量因素可能导致此错误，但一个好的做法是创建一个可以自动复制的环境，减少对实际使用的依赖关系的不确定性。

使用`virtualenv`模块很容易实现这一点，它可以设置一个虚拟环境，因此不会与计算机上安装的 Python 版本共享任何已安装的依赖项。

在 Python3 中，`virtualenv`工具会自动安装，而在以前的版本中并非如此。

# 准备就绪

要创建新的虚拟环境，请执行以下操作：

1.  转到包含项目的主目录。

1.  输入以下命令：

```py
$ python3 -m venv .venv
```

这将创建一个名为`.venv`的子目录，其中包含虚拟环境。包含虚拟环境的目录可以位于任何位置。将其保留在相同的根目录下会很方便，并在其前面加上一个点可以避免在运行`ls`或其他命令时显示它。

1.  在激活虚拟环境之前，检查`pip`中安装的版本。这取决于您的操作系统，例如，MacOS High Sierra 10.13.4 的版本为 9.0.3。稍后将对其进行升级。还要检查引用的 Python 解释器，这将是主要操作系统的解释器：

```py
$ pip --version
pip 9.0.3 from /usr/local/lib/python3.6/site-packages/pip (python 3.6)
$ which python3
/usr/local/bin/python3
```

现在，您的虚拟环境已准备就绪。

# 如何做...

1.  通过运行以下命令激活虚拟环境：

```py
$ source .venv/bin/activate
```

您会注意到提示会显示`(.venv)`，表示虚拟环境已激活。

1.  请注意，所使用的 Python 解释器是虚拟环境中的解释器，而不是*准备就绪*中第 3 步中的一般操作系统解释器。检查虚拟环境中的位置：

```py
(.venv) $ which python
/root_dir/.venv/bin/python
(.venv) $ which pip
/root_dir/.venv/bin/pip
```

1.  升级`pip`的版本并检查版本：

```py
(.venv) $ pip install --upgrade pip
...
Successfully installed pip-10.0.1
(.venv) $ pip --version
pip 10.0.1 from /root_dir/.venv/lib/python3.6/site-packages/pip (python 3.6)
```

1.  退出环境并运行`pip`来检查版本，这将返回之前的环境。检查`pip`版本和 Python 解释器以显示激活虚拟环境之前的版本，如*准备就绪*部分的第 3 步所示。请注意，它们是不同的 pip 版本！

```py
(.venv) $ deactivate 
$ which python3
/usr/local/bin/python3
$ pip --version
pip 9.0.3 from /usr/local/lib/python3.6/site-packages/pip (python 3.6)
```

# 它是如何工作的...

请注意，在虚拟环境中，您可以使用`python`而不是`python3`，尽管`python3`也可用。这将使用环境中定义的 Python 解释器。

在一些像 Linux 这样的系统中，可能需要使用`python3.7`而不是`python3`。验证您正在使用的 Python 解释器是否为 3.7 或更高版本。

在虚拟环境中，*如何做...*部分的第 3 步安装了最新版本的`pip`，而不会影响外部安装。

虚拟环境包含`.venv`目录中的所有 Python 数据，而`activate`脚本指向所有环境变量。最好的是，它可以很容易地被删除和重新创建，消除了在一个封闭的沙盒中进行实验的恐惧。

请记住，目录名称显示在提示符中。如果需要区分环境，请使用描述性目录名称，例如`.my_automate_recipe`，或使用`--prompt`选项。

# 还有更多...

要删除虚拟环境，请停用它并删除目录：

```py
(.venv) $ deactivate
$ rm -rf .venv
```

`venv`模块有更多选项，可以使用`-h`标志显示：

```py
$ python3 -m venv -h
usage: venv [-h] [--system-site-packages] [--symlinks | --copies] [--clear]
 [--upgrade] [--without-pip] [--prompt PROMPT]
 ENV_DIR [ENV_DIR ...]
Creates virtual Python environments in one or more target directories.
positional arguments:
 ENV_DIR A directory to create the environment in.

optional arguments:
 -h, --help show this help message and exit
 --system-site-packages
 Give the virtual environment access to the system
 site-packages dir.
 --symlinks Try to use symlinks rather than copies, when symlinks
 are not the default for the platform.
 --copies Try to use copies rather than symlinks, even when
 symlinks are the default for the platform.
 --clear Delete the contents of the environment directory if it
 already exists, before environment creation.
 --upgrade Upgrade the environment directory to use this version
 of Python, assuming Python has been upgraded in-place.
 --without-pip Skips installing or upgrading pip in the virtual
 environment (pip is bootstrapped by default)
 --prompt PROMPT Provides an alternative prompt prefix for this
 environment.
Once an environment has been created, you may wish to activate it, for example, by
sourcing an activate script in its bin directory.
```

处理虚拟环境的一种便捷方式，特别是如果您经常需要在它们之间切换，就是使用`virtualenvwrapper`模块：

1.  要安装它，请运行以下命令：

```py
$ pip install virtualenvwrapper
```

1.  然后，将以下变量添加到您的启动脚本中，通常是`.bashrc`或`.bash_profile`。虚拟环境将安装在`WORKON_HOME`目录下，而不是与项目相同的目录下，如前面所示：

```py
export WORKON_HOME=~/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
```

运行启动脚本或打开新的终端将允许您创建新的虚拟环境：

```py
$ mkvirtualenv automation_cookbook
...
Installing setuptools, pip, wheel...done.
(automation_cookbook) $ deactivate
$ workon automation_cookbook
(automation_cookbook) $
```

有关更多信息，请查看`virtualenvwrapper`的文档：[`virtualenvwrapper.readthedocs.io/en/latest/index.html`](https://virtualenvwrapper.readthedocs.io/en/latest/index.html)。

在`workon`后按下*Tab*键，将自动完成可用的环境。

# 另请参阅

+   *安装第三方软件包*的步骤

+   *使用第三方工具—parse*的步骤

# 安装第三方软件包

Python 最强大的功能之一是能够使用一个令人印象深刻的第三方软件包目录，涵盖了不同领域的大量内容，从专门执行数值操作、机器学习和网络通信的模块，到命令行便利工具、数据库访问、图像处理等等！

其中大多数都可以在官方 Python 软件包索引（[`pypi.org/`](https://pypi.org/)）上找到，该索引拥有超过 130,000 个准备好使用的软件包。在本书中，我们将安装其中一些软件包，并且通常花一点时间研究外部工具来解决问题是值得的。很可能有人已经创建了一个解决问题的工具。

与找到并安装软件包一样重要的是跟踪使用了哪些软件包。这对于**可复制性**非常有帮助，意味着能够在任何情况下从头开始启动整个环境。

# 准备就绪

起点是找到一个在我们的项目中有用的软件包。

一个很棒的模块是`requests`，它处理 HTTP 请求并以其简单直观的界面以及出色的文档而闻名。查看文档，网址为：[`docs.python-requests.org/en/master/`](http://docs.python-requests.org/en/master)。

在本书中处理 HTTP 连接时，我们将使用`requests`。

下一步将是选择要使用的版本。在这种情况下，最新版本（在撰写时为 2.18.4）将是完美的。如果未指定模块的版本，默认情况下将安装最新版本，这可能会导致不同环境中的不一致性。

我们还将使用很棒的`delorean`模块来处理时间（版本 1.0.0 [`delorean.readthedocs.io/en/latest/`](http://delorean.readthedocs.io/en/latest/)）。

# 如何做...

1.  在我们的主目录中创建一个`requirements.txt`文件，其中将指定项目的所有要求。让我们从`delorean`和`requests`开始：

```py
delorean==1.0.0
requests==2.18.4
```

1.  使用`pip`命令安装所有要求：

```py
$ pip install -r requirements.txt
...
Successfully installed babel-2.5.3 certifi-2018.4.16 chardet-3.0.4 delorean-1.0.0 humanize-0.5.1 idna-2.6 python-dateutil-2.7.2 pytz-2018.4 requests-2.18.4 six-1.11.0 tzlocal-1.5.1 urllib3-1.22
```

1.  现在在使用虚拟环境时可以同时使用这两个模块：

```py
$ python
Python 3.6.5 (default, Mar 30 2018, 06:41:53)
[GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.39.2)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import delorean
>>> import requests
```

# 它是如何工作的...

`requirements.txt`文件指定了模块和版本，`pip`在[pypi.org](http://pypi.org)上进行搜索。

请注意，从头开始创建一个新的虚拟环境并运行以下操作将完全重新创建您的环境，这使得可复制性非常简单：

```py
$ pip install -r requirements.txt
```

请注意，*如何做...*部分的第 2 步会自动安装其他依赖模块，例如`urllib3`。

# 还有更多...

如果需要将任何模块更改为不同的版本，因为有新版本可用，可以使用要求进行更改，然后再次运行`install`命令：

```py
$ pip install -r requirements.txt
```

当需要包含新模块时，这也适用。

在任何时候，都可以使用`freeze`命令来显示所有已安装的模块。`freeze`以与`requirements.txt`兼容的格式返回模块，从而可以生成一个包含当前环境的文件：

```py
$ pip freeze > requirements.txt
```

这将包括依赖项，因此文件中会有更多的模块。

有时找到很棒的第三方模块并不容易。搜索特定功能可能效果很好，但有时会有一些出乎意料的很棒的模块，因为它们做了你从未想过的事情。一个很棒的策划列表是**Awesome Python**（[`awesome-python.com/`](https://awesome-python.com/)），它涵盖了很多常见 Python 用例的很棒工具，如加密、数据库访问、日期和时间处理等。

在某些情况下，安装软件包可能需要额外的工具，例如编译器或支持某些功能的特定库（例如特定的数据库驱动程序）。如果是这种情况，文档通常会解释依赖关系。

# 另请参阅

+   *创建虚拟环境*的步骤

+   *使用第三方工具—parse*的步骤

# 创建带有格式化值的字符串

处理文本和文档时的基本能力之一是能够将值正确格式化为结构化字符串。Python 在提供良好的默认值方面非常聪明，比如正确呈现数字，但是有很多选项和可能性。

我们将通过一个表格的示例来讨论创建格式化文本时的一些常见选项。

# 准备工作

在 Python 中格式化字符串的主要工具是`format`方法。它使用一个定义的迷你语言以这种方式呈现变量：

```py
result = template.format(*parameters)
```

`template`是一个基于迷你语言解释的字符串。在最简单的情况下，它会用参数替换大括号之间的值。以下是一些示例：

```py
>>> 'Put the value of the string here: {}'.format('STRING')
"Put the value of the string here: STRING"
>>> 'It can be any type ({}) and more than one ({})'.format(1.23, str)
"It can be any type (1.23) and more than one (<class 'str'>)"
>> 'Specify the order: {1}, {0}'.format('first', 'second')
'Specify the order: second, first'
>>> 'Or name parameters: {first}, {second}'.format(second='SECOND', first='FIRST')
'Or name parameters: FIRST, SECOND'
```

在 95%的情况下，这种格式化就足够了；保持简单是很好的！但是对于复杂的情况，比如自动对齐字符串和创建漂亮的文本表格时，迷你语言`format`有更多的选项。

# 如何做...

1.  编写以下脚本`recipe_format_strings_step1.py`，以打印一个对齐的表格：

```py
# INPUT DATA
data = [
    (1000, 10),
    (2000, 17),
    (2500, 170),
    (2500, -170),
]
# Print the header for reference
print('REVENUE | PROFIT | PERCENT')

# This template aligns and displays the data in the proper format
TEMPLATE = '{revenue:>7,} | {profit:>+7} | {percent:>7.2%}'

# Print the data rows
for revenue, profit in data:
    row = TEMPLATE.format(revenue=revenue, profit=profit, percent=profit / revenue)
    print(row)
```

1.  运行它以显示以下对齐的表格。请注意，`PERCENT`正确显示为百分比：

```py
REVENUE | PROFIT | PERCENT
 1,000 |    +10 |   1.00%
 2,000 |    +17 |   0.85%
 2,500 |   +170 |   6.80%
 2,500 |   -170 |  -6.80%
```

# 它是如何工作的...

`TEMPLATE`常量包含三列，每一列都有适当的名称（`REVENUE`，`PROFIT`，`PERCENT`）。这使得在格式调用上更加明确和简单。

在参数名称之后，有一个冒号，用于分隔格式定义。请注意，所有内容都在花括号内。在所有列中，格式规范将宽度设置为七个字符，并使用`>`符号将值对齐到右侧：

+   收入使用`,`符号添加千位分隔符-`[{revenue:>7,}]`。

+   利润为正值添加`+`符号。负值会自动添加`-`-`[{profit:>+7}]`。

+   百分比显示百分比值，精确到两位小数-`[{percent:>7.2%}]`。这是通过 0.2（精度）和添加`%`符号来完成的。

# 还有更多...

您可能也已经看到了使用`%`运算符的 Python 格式。虽然它适用于简单的格式，但它不如格式化的迷你语言灵活，不建议使用。

自 Python 3.6 以来的一个很棒的新功能是使用 f-strings，它使用定义的变量执行格式操作：

```py
>>> param1 = 'first'
>>> param2 = 'second'
>>> f'Parameters {param1}:{param2}'
'Parameters first:second'
```

这简化了很多代码，使我们能够创建非常描述性和可读性的代码。

在使用 f-strings 时要小心，确保字符串在适当的时间被替换。一个常见问题是，定义为呈现的变量尚未定义。例如，先前定义的`TEMPLATE`不会作为 f-string 定义，因为`revenue`和其他参数在那时不可用。

如果需要写大括号，需要重复两次。请注意，每个复制将显示为单个大括号，再加上一个大括号用于值替换，总共三个大括号：

```py
>> value = 'VALUE'
>>> f'This is the value, in curly brackets {{{value}}}'
'This is the value, in curly brackets {VALUE}'
```

这使我们能够创建元模板-生成模板的模板。在某些情况下，这将很有用，但请尽量限制它们的使用，因为它们会很快变得复杂，产生难以阅读的代码。

Python 格式规范迷你语言比这里显示的选项更多。

由于语言试图非常简洁，有时很难确定符号的位置。有时您可能会问自己问题，比如-`+`符号是在宽度参数之前还是之后。-请仔细阅读文档，并记住在格式规范之前始终包括一个冒号。

请在 Python 网站上查看完整的文档和示例（[`docs.python.org/3/library/string.html#formatspec`](https://docs.python.org/3/library/string.html#formatspec)）。

# 另请参阅

+   在第五章的*生成精彩报告*中的*模板报告*配方

+   *操作字符串*配方

# 操作字符串

处理文本时的基本能力是能够正确地操作该文本。这意味着能够将其连接，分割成常规块，或将其更改为大写或小写。我们将在以后讨论更高级的解析文本和分隔文本的方法，但在许多情况下，将段落分成行、句子甚至单词是有用的。有时，单词将必须删除一些字符或用规范版本替换以便与确定的值进行比较。

# 准备就绪

我们将定义一个基本文本，将其转换为其主要组件，然后重新构造它。例如，需要将报告转换为新格式以通过电子邮件发送。

我们将在此示例中使用的输入格式如下：

```py
    AFTER THE CLOSE OF THE SECOND QUARTER, OUR COMPANY, CASTAÑACORP
    HAS ACHIEVED A GROWTH IN THE REVENUE OF 7.47%. THIS IS IN LINE
    WITH THE OBJECTIVES FOR THE YEAR. THE MAIN DRIVER OF THE SALES HAS BEEN
    THE NEW PACKAGE DESIGNED UNDER THE SUPERVISION OF OUR MARKETING DEPARTMENT.
    OUR EXPENSES HAS BEEN CONTAINED, INCREASING ONLY BY 0.7%, THOUGH THE BOARD
    CONSIDERS IT NEEDS TO BE FURTHER REDUCED. THE EVALUATION IS SATISFACTORY
    AND THE FORECAST FOR THE NEXT QUARTER IS OPTIMISTIC. THE BOARD EXPECTS
    AN INCREASE IN PROFIT OF AT LEAST 2 MILLION DOLLARS.
```

我们需要编辑文本以消除对数字的任何引用。需要通过在每个句号后添加一个新行来正确格式化它，使其对齐为 80 个字符，并将其转换为 ASCII 以确保兼容性。

文本将存储在解释器中的`INPUT_TEXT`变量中。

# 如何做...

1.  输入文本后，将其拆分为单独的单词：

```py
>>> INPUT_TEXT = '''
...     AFTER THE CLOSE OF THE SECOND QUARTER, OUR COMPANY, CASTAÑACORP
...     HAS ACHIEVED A GROWTH IN THE REVENUE OF 7.47%. THIS IS IN LINE
...
'''
>>> words = INPUT_TEXT.split()
```

1.  用`'X'`字符替换任何数字：

```py
>>> redacted = [''.join('X' if w.isdigit() else w for w in word) for word in words]
```

1.  将文本转换为纯 ASCII（请注意，公司名称包含一个不是 ASCII 的字母`ñ`）：

```py
>>> ascii_text = [word.encode('ascii', errors='replace').decode('ascii')
...               for word in redacted]
```

1.  将单词分组为 80 个字符的行：

```py
>>> newlines = [word + '\n' if word.endswith('.') else word for word in ascii_text]
>>> LINE_SIZE = 80
>>> lines = []
>>> line = ''
>>> for word in newlines:
...     if line.endswith('\n') or len(line) + len(word) + 1 > LINE_SIZE:
...         lines.append(line)
...         line = ''
...     line = line + ' ' + word
```

1.  将所有行格式化为标题并将它们连接为单个文本片段：

```py
>>> lines = [line.title() for line in lines]
>>> result = '\n'.join(lines)
```

1.  打印结果：

```py
>>> print(result)
 After The Close Of The Second Quarter, Our Company, Casta?Acorp Has Achieved A
 Growth In The Revenue Of X.Xx%.

 This Is In Line With The Objectives For The Year.

 The Main Driver Of The Sales Has Been The New Package Designed Under The
 Supervision Of Our Marketing Department.

 Our Expenses Has Been Contained, Increasing Only By X.X%, Though The Board
 Considers It Needs To Be Further Reduced.

 The Evaluation Is Satisfactory And The Forecast For The Next Quarter Is
 Optimistic.
```

# 它是如何工作的...

每个步骤都对文本执行特定的转换：

+   第一个步骤在默认分隔符、空格和换行符上分割文本。这将它分割成没有行或多个空格用于分隔的单词。

+   为了替换数字，我们遍历每个单词的每个字符。对于每个字符，如果它是一个数字，就返回一个`'X'`。这是通过两个列表推导式完成的，一个用于遍历列表，另一个用于每个单词，只有在有数字时才进行替换——`['X' if w.isdigit() else w for w in word]`。请注意，这些单词再次连接在一起。

+   每个单词都被编码为 ASCII 字节序列，然后再次解码为 Python 字符串类型。注意使用`errors`参数来强制替换未知字符，如`ñ`。

字符串和字节之间的区别一开始并不直观，特别是如果你从来不用担心多种语言或编码转换。在 Python 3 中，字符串（内部 Python 表示）和字节之间有很强的分离，因此大多数适用于字符串的工具在字节对象中不可用。除非你很清楚为什么需要一个字节对象，总是使用 Python 字符串。如果你需要执行像这个任务中的转换，编码和解码在同一行中进行，这样你就可以保持对象在舒适的 Python 字符串领域。如果你有兴趣了解更多关于编码的信息，你可以查看这篇简短的文章（[`eli.thegreenplace.net/2012/01/30/the-bytesstr-dichotomy-in-python-3`](https://eli.thegreenplace.net/2012/01/30/the-bytesstr-dichotomy-in-python-3)）和这篇更长更详细的文章（[`www.diveintopython3.net/strings.html`](http://www.diveintopython3.net/strings.html)）。

+   这一步首先为所有以句号结尾的单词添加一个额外的换行符（`\n`字符）。这标记了不同的段落。之后，它创建一行并逐个添加单词。如果多一个单词会使它超过 80 个字符，它就结束该行并开始新的一行。如果该行已经以换行符结尾，它也结束并开始另一行。请注意，添加了额外的空格来分隔单词。

+   最后，每一行都被大写为标题（每个单词的第一个字母都是大写的），并且所有行都通过换行符连接在一起。

# 还有...

可以对字符串执行的一些其他有用操作如下：

+   字符串可以像任何其他列表一样切片。这意味着`'word'[0:2]`将返回`'wo'`。

+   使用`.splitlines()`通过换行符分隔行。

+   有`.upper()`和`.lower()`方法，它们返回一个所有字符都设置为大写或小写的副本。它们的使用非常类似于`.title()`：

```py
>>> 'UPPERCASE'.lower()
'uppercase'
```

+   对于简单的替换（例如，将所有`A`替换为`B`或将`mine`替换为`ours`），使用`.replace()`。这种方法对于非常简单的情况很有用，但替换很容易变得棘手。注意替换的顺序，以避免冲突和大小写敏感问题。请注意以下示例中错误的替换：

```py
>>> 'One ring to rule them all, one ring to find them, One ring to bring them all and in the darkness bind them.'.replace('ring', 'necklace')
'One necklace to rule them all, one necklace to find them, One necklace to bnecklace them all and in the darkness bind them.'
```

这类似于我们将在正则表达式中看到的问题，匹配代码的意外部分。

还有更多示例将在后面介绍。有关更多信息，请参阅正则表达式示例。

如果您使用多种语言，或者任何非英语输入，学习 Unicode 和编码的基础知识非常有用。简而言之，鉴于世界上所有不同语言中的大量字符，包括与拉丁语无关的字母表，如中文或阿拉伯语，有一个标准来尝试覆盖所有这些字符，以便计算机可以正确理解它们。Python 3 极大地改善了这种情况，使字符串成为内部对象，以处理所有这些字符。Python 使用的编码，也是最常见和兼容的编码，目前是 UTF-8。

了解有关 UTF-8 基础知识的好文章是这篇博文：([`www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/`](https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/))。

处理编码在从可以使用不同编码的外部文件中读取时仍然很重要（例如 CP-1252 或 windows-1252，这是由传统 Microsoft 系统生成的常见编码，或 ISO 8859-15，这是行业标准）。

# 另请参阅

+   *使用格式化值创建字符串*食谱

+   *介绍正则表达式*食谱

+   *深入研究正则表达式*食谱

+   第四章中的*处理编码*食谱，*搜索和读取本地文件*

# 从结构化字符串中提取数据

在许多自动化任务中，我们需要处理特定格式的输入文本并提取相关信息。例如，电子表格可能以文本形式定义百分比（例如 37.4%），我们希望以后以数值格式检索它（0.374，作为浮点数）。

在这个食谱中，我们将看到如何处理包含有关产品的内联信息的销售日志，例如已售出、价格、利润和其他一些信息。

# 准备工作

想象一下，我们需要解析存储在销售日志中的信息。我们将使用以下结构的销售日志：

```py
[<Timestamp in iso format>] - SALE - PRODUCT: <product id> - PRICE: $<price of the sale>
```

例如，特定的日志可能如下所示：

```py
[2018-05-05T10:58:41.504054] - SALE - PRODUCT: 1345 - PRICE: $09.99
```

请注意，价格有一个前导零。所有价格都将有两位数字的美元，两位数字的美分。

我们需要在开始之前激活我们的虚拟环境：

```py
$ source .venv/bin/activate
```

# 如何做...

1.  在 Python 解释器中，进行以下导入。记得激活你的`virtualenv`，就像*创建虚拟环境*食谱中描述的那样：

```py
>>> import delorean
>>> from decimal import Decimal
```

1.  输入要解析的日志：

```py
>>> log = '[2018-05-05T11:07:12.267897] - SALE - PRODUCT: 1345 - PRICE: $09.99'
```

1.  将日志分割为其部分，这些部分由` -`（注意破折号前后的空格）分隔。我们忽略`SALE`部分，因为它没有添加任何相关信息：

```py
>>> divide_it = log.split(' - ')
>>> timestamp_string, _, product_string, price_string = divide_it
```

1.  将`timestamp`解析为 datetime 对象：

```py
>>> timestamp = delorean.parse(tmp_string.strip('[]'))
```

1.  将`product_id`解析为整数：

```py
>>> product_id = int(product_string.split(':')[-1])
```

1.  将价格解析为`Decimal`类型：

```py
>>> price = Decimal(price_string.split('$')[-1])
```

1.  现在，您已经拥有了所有本机 Python 格式的值：

```py
>> timestamp, product_id, price
(Delorean(datetime=datetime.datetime(2018, 5, 5, 11, 7, 12, 267897), timezone='UTC'), 1345, Decimal('9.99'))
```

# 它是如何工作的...

这个基本的工作是隔离每个元素，然后将它们解析为适当的类型。第一步是将完整的日志分割成较小的部分。`-`字符串是一个很好的分隔符，因为它将其分成四个部分——一个时间戳部分，一个只有`SALE`一词的部分，产品和价格。

在时间戳的情况下，我们需要隔离日志中的 ISO 格式。这就是为什么它被剥离括号。我们使用`delorean`模块（之前介绍过）将其解析为`datetime`对象。

单词`SALE`被忽略。那里没有相关信息。

为了隔离产品 ID，我们将产品部分分割为冒号。然后，我们将最后一个元素解析为整数：

```py
>>> product_string.split(':')
['PRODUCT', ' 1345']
>>> int(' 1345')
1345
```

为了分割价格，我们使用美元符号作为分隔符，并将其解析为`Decimal`字符：

```py
>>> price_string.split('$')
['PRICE: ', '09.99']
>>> Decimal('09.99')
Decimal('9.99')
```

如下一节所述，不要将此值解析为浮点类型。

# 还有更多...

这些日志元素可以组合成一个单一对象，有助于解析和聚合它们。例如，我们可以在 Python 代码中以以下方式定义一个类：

```py
class PriceLog(object):
  def __init__(self, timestamp, product_id, price):
    self.timestamp = timestamp
    self.product_id = product_id
    self.price = price
  def __repr__(self):
    return '<PriceLog ({}, {}, {})>'.format(self.timestamp,
                                            self.product_id,
                                            self.price)
  @classmethod
  def parse(cls, text_log):
    '''
    Parse from a text log with the format
    [<Timestamp>] - SALE - PRODUCT: <product id> - PRICE: $<price>
    to a PriceLog object
    '''
    divide_it = text_log.split(' - ')
    tmp_string, _, product_string, price_string = divide_it
    timestamp = delorean.parse(tmp_string.strip('[]'))
    product_id = int(product_string.split(':')[-1])
    price = Decimal(price_string.split('$')[-1])
    return cls(timestamp=timestamp, product_id=product_id, price=price)
```

因此，解析可以按以下方式进行：

```py
>>> log = '[2018-05-05T12:58:59.998903] - SALE - PRODUCT: 897 - PRICE: $17.99'
>>> PriceLog.parse(log)
<PriceLog (Delorean(datetime=datetime.datetime(2018, 5, 5, 12, 58, 59, 998903), timezone='UTC'), 897, 17.99)>
```

避免使用浮点数类型来表示价格。浮点数存在精度问题，可能在聚合多个价格时产生奇怪的错误，例如：

```py
>>> 0.1 + 0.1 + 0.1 0.30000000000000004
```

尝试这两个选项以避免问题：

+   **使用整数分为基本单位**：这意味着将货币输入乘以 100，并将其转换为整数（或者正确的分数单位，根据所使用的货币而定）。在显示它们时，您可能仍然希望更改基数。

+   **解析为十进制类型**：`Decimal`类型保持固定精度，并且按预期工作。您可以在 Python 文档中找到有关`Decimal`类型的更多信息，网址为[`docs.python.org/3.6/library/decimal.html`](https://docs.python.org/3.6/library/decimal.html)。

如果使用`Decimal`类型，请直接从字符串解析结果为`Decimal`。如果首先将其转换为浮点数，则可能会将精度错误传递给新类型。

# 另请参阅

+   *创建虚拟环境*食谱

+   *使用第三方工具—解析*食谱

+   *介绍正则表达式*食谱

+   *深入了解正则表达式*食谱

# 使用第三方工具—解析

手动解析数据，如前一篇文章中所示，对于小字符串非常有效，但是要调整确切的公式以适应各种输入可能非常费力。如果输入有时有额外的破折号呢？或者根据某个字段的大小而变化的变长标题呢？

更高级的选项是使用正则表达式，我们将在下一篇文章中看到。但是 Python 中有一个名为`parse`的出色模块([`github.com/r1chardj0n3s/parse`](https://github.com/r1chardj0n3s/parse))，它允许我们反转格式字符串。这是一个强大、易于使用的工具，极大地提高了代码的可读性。

# 准备工作

将`parse`模块添加到虚拟环境中的`requirements.txt`文件中，并重新安装依赖项，如*创建虚拟环境*食谱中所示。

`requirements.txt`文件应如下所示：

```py
delorean==1.0.0
requests==2.18.3
parse==1.8.2
```

然后，在虚拟环境中重新安装模块：

```py
$ pip install -r requirements.txt
...
Collecting parse==1.8.2 (from -r requirements.txt (line 3))
 Using cached https://files.pythonhosted.org/packages/13/71/e0b5c968c552f75a938db18e88a4e64d97dc212907b4aca0ff71293b4c80/parse-1.8.2.tar.gz
...
Installing collected packages: parse
 Running setup.py install for parse ... done
Successfully installed parse-1.8.2
```

# 如何做...

1.  导入`parse`函数：

```py
>>> from parse import parse
```

1.  定义要解析的日志，格式与*从结构化字符串中提取数据*食谱中的格式相同：

```py
>>> LOG = '[2018-05-06T12:58:00.714611] - SALE - PRODUCT: 1345 - PRICE: $09.99'
```

1.  分析它并描述它，就像打印时所做的那样，如下所示：

```py
>>> FORMAT = '[{date}] - SALE - PRODUCT: {product} - PRICE: ${price}'
```

1.  运行`parse`并检查结果：

```py
>>> result = parse(FORMAT, LOG)
>>> result
<Result () {'date': '2018-05-06T12:58:00.714611', 'product': '1345', 'price': '09.99'}>
>>> result['date']
'2018-05-06T12:58:00.714611'
>>> result['product']
'1345'
>>> result['price']
'09.99'
```

1.  请注意，结果都是字符串。定义要解析的类型：

```py
>>> FORMAT = '[{date:ti}] - SALE - PRODUCT: {product:d} - PRICE: ${price:05.2f}'
```

1.  再次解析：

```py
>>> result = parse(FORMAT, LOG)
>>> result
<Result () {'date': datetime.datetime(2018, 5, 6, 12, 58, 0, 714611), 'product': 1345, 'price': 9.99}>
>>> result['date']
datetime.datetime(2018, 5, 6, 12, 58, 0, 714611)
>>> result['product']
1345
>>> result['price']
9.99
```

1.  定义自定义类型以避免浮点类型的问题：

```py
>>> from decimal import Decimal
>>> def price(string):
...   return Decimal(string)
...
>>> FORMAT = '[{date:ti}] - SALE - PRODUCT: {product:d} - PRICE: ${price:price}'
>>> parse(FORMAT, LOG, {'price': price})
<Result () {'date': datetime.datetime(2018, 5, 6, 12, 58, 0, 714611), 'product': 1345, 'price': Decimal('9.99')}>
```

# 工作原理...

`parse`模块允许我们定义一个格式，例如字符串，以便在解析值时反转格式方法。我们在创建字符串时讨论的许多概念也适用于此处—将值放在括号中，在冒号后定义类型等。

默认情况下，如第 4 步所示，值被解析为字符串。这是分析文本的一个很好的起点。值可以被解析为更有用的本机类型，如*如何做...*部分的第 5 和第 6 步所示。请注意，虽然大多数解析类型与 Python 格式规范迷你语言中的类型相同，但还有其他一些可用，例如用于 ISO 格式时间戳的`ti`。

如果本机类型不够用，我们可以定义自己的解析，如*如何做...*部分的第 7 步所示。请注意，价格函数的定义接收一个字符串并返回正确的格式，本例中为`Decimal`类型。

*从结构化字符串中提取数据*食谱的*还有更多*部分中描述的有关浮点数和价格信息的所有问题在这里同样适用。

# 还有更多...

时间戳也可以转换为`delorean`对象以保持一致性。此外，`delorean`对象携带时区信息。添加与上一个示例相同的结构，得到以下对象，可以解析日志：

```py
class PriceLog(object):
  def __init__(self, timestamp, product_id, price):
    self.timestamp = timestamp
    self.product_id = product_id
    self.price = price
  def __repr__(self):
    return '<PriceLog ({}, {}, {})>'.format(self.timestamp,
                                            self.product_id,
                                            self.price)
  @classmethod
  def parse(cls, text_log):
    '''
    Parse from a text log with the format
    [<Timestamp>] - SALE - PRODUCT: <product id> - PRICE: $<price>
    to a PriceLog object
    '''
    def price(string):
      return Decimal(string)
    def isodate(string):
      return delorean.parse(string)
    FORMAT = ('[{timestamp:isodate}] - SALE - PRODUCT: {product:d} - '
              'PRICE: ${price:price}')
    formats = {'price': price, 'isodate': isodate}
    result = parse.parse(FORMAT, text_log, formats)
    return cls(timestamp=result['timestamp'],
               product_id=result['product'],
               price=result['price'])
```

因此，解析它会返回类似的结果：

```py
>>> log = '[2018-05-06T14:58:59.051545] - SALE - PRODUCT: 827 - PRICE: $22.25'
>>> PriceLog.parse(log)
<PriceLog (Delorean(datetime=datetime.datetime(2018, 6, 5, 14, 58, 59, 51545), timezone='UTC'), 827, 22.25)>
```

此代码包含在 GitHub 文件`Chapter01/price_log.py`中。

所有`parse`支持的类型都可以在[`github.com/r1chardj0n3s/parse#format-specification`](https://github.com/r1chardj0n3s/parse#format-specification)的文档中找到。

# 另请参阅

+   *从结构化字符串中提取数据*示例

+   *介绍正则表达式*示例

+   *深入了解正则表达式*示例

# 介绍正则表达式

**正则表达式**，或**regex**，是一种用于*匹配*文本的模式。换句话说，它允许我们定义一个**抽象字符串**（通常是结构化文本的定义）来检查其他字符串是否匹配。

最好用示例来描述它们。想象一下，定义一个文本模式为“以大写 A 开头，之后只包含小写 N 和 A 的单词”。单词*Anna*符合此模式，但*Bob*、*Alice*和*James*不符合。单词*Aaan*、*Ana*、*Annnn*和*Aaaan*也符合，但*ANNA*不符合。

如果这听起来很复杂，那是因为它确实很复杂。正则表达式可能非常复杂，因为它们可能非常复杂且难以理解。但它们非常有用，因为它们允许我们执行非常强大的模式匹配。

正则表达式的一些常见用途如下：

+   **验证输入数据**：例如，电话号码只包含数字、破折号和括号。

+   **字符串解析**：从结构化字符串（如日志或 URL）中检索数据。这与前一个示例中描述的内容类似。

+   **抓取**：在长文本中查找某些内容的出现。例如，在网页中查找所有电子邮件。

+   **替换**：查找并用其他单词替换一个单词或多个单词。例如，将*the owner*替换为*John Smith*。

“有些人遇到问题时会想到“我知道了，我会使用正则表达式。”现在他们有了两个问题。”

- Jamie Zawinski

正则表达式在保持非常简单时效果最好。一般来说，如果有特定的工具可以做到，最好使用它而不是正则表达式。HTML 解析就是一个非常明显的例子；查看第三章，*构建您的第一个网络抓取应用程序*，以了解更好的工具来实现这一点。

一些文本编辑器也允许我们使用正则表达式进行搜索。虽然大多数是针对编写代码的编辑器，如 Vim、BBEdit 或 Notepad++，但它们也存在于更通用的工具中，如 MS Office、Open Office 或 Google 文档。但要小心，因为特定的语法可能略有不同。

# 准备工作

处理正则表达式的`python`模块称为`re`。我们将介绍的主要函数是`re.search()`，它返回一个关于匹配模式的*match*对象的信息。

由于正则表达式模式也是字符串，我们将它们区分开来，通过在前面加上*r*来区分它们，例如`r'pattern'`。这是 Python 标记文本为原始字符串文字的方式，这意味着其中的字符串会被直接接受，不会进行任何转义。这意味着`\`被用作反斜杠，而不是一个序列。例如，没有 r 前缀，`\n`表示换行符。

有些字符是特殊的，表示诸如*字符串结尾*、*任何数字*、*任何字符*、*任何空白字符*等概念。

最简单的形式只是一个字面字符串。例如，正则表达式模式`r'LOG'`匹配字符串`'LOGS'`，但不匹配字符串`'NOT A MATCH'`。如果没有匹配，搜索返回`None`：

```py
>>> import re
>>> re.search(r'LOG', 'LOGS')
<_sre.SRE_Match object; span=(0, 3), match='LOG'>
>>> re.search(r'LOG', 'NOT A MATCH')
>>>
```

# 如何做...

1.  导入`re`模块：

```py
>>> import re
```

1.  然后，匹配不位于字符串开头的模式：

```py
>>> re.search(r'LOG', 'SOME LOGS')
<_sre.SRE_Match object; span=(5, 8), match='LOG'>
```

1.  匹配仅位于字符串开头的模式。注意`^`字符：

```py
>>> re.search(r'^LOG', 'LOGS')
<_sre.SRE_Match object; span=(0, 3), match='LOG'>
>>> re.search(r'^LOG', 'SOME LOGS')
>>>
```

1.  仅在字符串末尾匹配模式。请注意`$`字符：

```py
>>> re.search(r'LOG$', 'SOME LOG')
<_sre.SRE_Match object; span=(5, 8), match='LOG'>
>>> re.search(r'LOG$', 'SOME LOGS')
>>>
```

1.  匹配单词`'thing'`（不包括`things`），但不匹配`something`或`anything`。请注意第二个模式的开头处的`\b`：

```py
>>> STRING = 'something in the things she shows me'
>>> match = re.search(r'thing', STRING)
>>> STRING[:match.start()], STRING[match.start():match.end()], STRING[match.end():]
('some', 'thing', ' in the things she shows me')
>>> match = re.search(r'\bthing', STRING)
>>> STRING[:match.start()], STRING[match.start():match.end()], STRING[match.end():]
('something in the ', 'thing', 's she shows me')

```

1.  匹配仅为数字和破折号（例如电话号码）的模式。检索匹配的字符串：

```py
>>> re.search(r'[0123456789-]+', 'the phone number is 1234-567-890')
<_sre.SRE_Match object; span=(20, 32), match='1234-567-890'>
>>> re.search(r'[0123456789-]+', 'the phone number is 1234-567-890').group()
'1234-567-890'
```

1.  天真地匹配电子邮件地址：

```py
>>> re.search(r'\S+@\S+', 'my email is email.123@test.com').group()
'email.123@test.com'
```

# 它是如何工作的...

`re.search`函数匹配模式，无论其在字符串中的位置如何。如前所述，如果未找到模式，将返回`None`，或者匹配对象。

使用以下特殊字符：

+   `^`：标记字符串的开头

+   `$`：标记字符串的结尾

+   `\b`：标记单词的开头或结尾

+   `\S`：标记任何非空白字符，包括特殊字符

更多特殊字符将在下一个配方中显示。

在*如何做...*部分的第 6 步中，`r'[0123456789-]+'`模式由两部分组成。第一部分在方括号之间，匹配`0`到`9`之间的任何单个字符（任何数字）和破折号（`-`）字符。之后的`+`表示该字符可以出现一次或多次。这在正则表达式中称为**量词**。这使得可以匹配任何数字和破折号的组合，无论长度如何。

步骤 7 再次使用`+`号匹配尽可能多的字符，然后再次使用`@`。在这种情况下，字符匹配是`\S`，它匹配任何非空白字符。

请注意，此处描述的电子邮件的天真模式非常天真，因为它将匹配无效的电子邮件，例如`john@smith@test.com`。对于大多数用途，更好的正则表达式是`r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"`。您可以访问[`emailregex.com/`](http://emailregex.com/)查找它并链接到更多信息。

请注意，解析包括边缘情况在内的有效电子邮件实际上是一个困难且具有挑战性的问题。前面的正则表达式对于本书涵盖的大多数用途应该都可以，但在诸如 Django 之类的通用框架项目中，电子邮件验证是一个非常冗长且非常难以阅读的正则表达式。

生成的匹配对象返回匹配模式开始和结束的位置（使用`start`和`end`方法），如步骤 5 所示，该步骤将字符串拆分为匹配部分，显示两个匹配模式之间的区别。

步骤 5 中显示的差异非常常见。尝试捕获 GP 可能最终捕获 eg**gp**lant 和 ba**gp**ipe！同样，`things\b`不会捕获 things。请务必测试并进行适当的调整，例如捕获`\bGP\b`以获取单词 GP。

可以通过调用`group()`来检索特定匹配的模式，如步骤 6 所示。请注意，结果始终是一个字符串。可以进一步使用我们之前看到的任何方法进行处理，例如通过破折号将电话号码拆分成组：

```py
>>> match = re.search(r'[0123456789-]+', 'the phone number is 1234-567-890')
>>> [int(n) for n in match.group().split('-')]
[1234, 567, 890]
```

# 还有更多...

处理正则表达式可能会很困难和复杂。请花时间测试您的匹配，并确保它们按照您的期望工作，以避免不愉快的惊喜。

您可以使用一些工具进行交互式地检查您的正则表达式。一个很好的免费在线工具是[`regex101.com/`](https://regex101.com/)，它显示每个元素并解释正则表达式。请仔细检查您是否使用了 Python 风格：

![](img/9c4e482e-ec3f-4e68-b39b-b58feebb6c34.png)

请注意，解释描述了`\b`匹配单词边界（单词的开头或结尾），以及*thing*字面上匹配这些字符。

在某些情况下，正则表达式可能非常缓慢，甚至会产生所谓的**正则表达式拒绝服务**，即创建一个字符串以混淆特定的正则表达式，使其花费大量时间，甚至在最坏的情况下阻塞计算机。虽然自动化任务可能不会让您陷入这些问题，但请注意，如果正则表达式花费的时间太长，请留意。

# 另请参阅

+   *从结构化字符串中提取数据*配方

+   *使用第三方工具—解析*配方

+   *深入了解正则表达式*配方

# 深入了解正则表达式

在这个配方中，我们将更多地了解如何处理正则表达式。在介绍基础知识之后，我们将深入了解模式元素，引入组作为检索和解析字符串的更好方法，看看如何搜索相同字符串的多个出现，并处理更长的文本。

# 如何做...

1.  导入`re`：

```py
>>> import re
```

1.  将电话模式作为组的一部分进行匹配（在括号中）。注意使用`\d`作为*任何数字*的特殊字符：

```py
>>> match = re.search(r'the phone number is ([\d-]+)', '37: the phone number is 1234-567-890')
>>> match.group()
'the phone number is 1234-567-890'
>>> match.group(1)
'1234-567-890'
```

1.  编译一个模式并捕获一个不区分大小写的模式，使用`yes|no`选项：

```py
>>> pattern = re.compile(r'The answer to question (\w+) is (yes|no)', re.IGNORECASE)
>>> pattern.search('Naturaly, the answer to question 3b is YES')
<_sre.SRE_Match object; span=(10, 42), match='the answer to question 3b is YES'>
>>> _.groups()
('3b', 'YES')
```

1.  在文本中匹配所有城市和州的缩写的出现。请注意，它们由一个单个字符分隔，城市的名称始终以大写字母开头。为简单起见，只匹配了四个州：

```py
>>> PATTERN = re.compile(r'([A-Z][\w\s]+).(TX|OR|OH|MI)')
>>> TEXT ='the jackalopes are the team of Odessa,TX while the knights are native of Corvallis OR and the mud hens come from Toledo.OH; the whitecaps have their base in Grand Rapids,MI'
>>> list(PATTERN.finditer(TEXT))
[<_sre.SRE_Match object; span=(31, 40), match='Odessa,TX'>, <_sre.SRE_Match object; span=(73, 85), match='Corvallis OR'>, <_sre.SRE_Match object; span=(113, 122), match='Toledo.OH'>, <_sre.SRE_Match object; span=(157, 172), match='Grand Rapids,MI'>]
>>> _[0].groups()
('Odessa', 'TX')
```

# 它是如何工作的...

引入的新特殊字符如下。请注意，大写或小写的相同字母表示相反的匹配，例如`\d`匹配数字，而`\D`匹配非数字。：

+   `\d`：标记任何数字（0 到 9）。

+   `\s`：标记任何空白字符，包括制表符和其他空白特殊字符。请注意，这与上一个配方中引入的`\S`相反**。**

+   `\w`：标记任何字母（包括数字，但不包括句号等字符）。

+   `.`：标记任何字符。

要定义组，请将定义的组放在括号中。可以单独检索组，使它们非常适合匹配包含稍后将处理的可变部分的更大模式，如步骤 2 中所示。请注意与上一个配方中步骤 6 模式的区别。在这种情况下，模式不仅是数字，而且包括前缀，即使我们随后提取数字。请查看这种差异，其中有一个不是我们想要捕获的数字：

```py
>>> re.search(r'the phone number is ([\d-]+)', '37: the phone number is 1234-567-890')
<_sre.SRE_Match object; span=(4, 36), match='the phone number is 1234-567-890'>
>>> _.group(1)
'1234-567-890'
>>> re.search(r'[0123456789-]+', '37: the phone number is 1234-567-890')
<_sre.SRE_Match object; span=(0, 2), match='37'>
>>> _.group()
'37'
```

记住，第 0 组（`.group()`或`.group(0)`）始终是整个匹配。其余的组按它们出现的顺序排列。

模式也可以编译。如果模式需要一遍又一遍地匹配，这样可以节省一些时间。要以这种方式使用它，编译模式，然后使用该对象执行搜索，如步骤 3 和 4 所示。可以添加一些额外的标志，例如使模式不区分大小写。

第 4 步的模式需要一点信息。它由两个组成，由一个单个字符分隔。特殊字符`.`表示它匹配一切，例如一个句号、一个空格和一个逗号。第二组是一组明确定义的选项，例如美国州的缩写。

第一组以大写字母（`[A-Z]`）开头，并接受任何字母或空格的组合（`[\w\s]+`），但不接受句号或逗号等标点符号。这匹配城市，包括由多个单词组成的城市。

请注意，这个模式从任何大写字母开始匹配，直到找到一个州，除非被标点符号分隔，这可能不是预期的结果，例如：

```py
>>> re.search(r'([A-Z][\w\s]+).(TX|OR|OH|MI)', 'This is a test, Escanaba MI')
<_sre.SRE_Match object; span=(16, 27), match='Escanaba MI'>
>>> re.search(r'([A-Z][\w\s]+).(TX|OR|OH|MI)', 'This is a test with Escanaba MI')
<_sre.SRE_Match object; span=(0, 31), match='This is a test with Escanaba MI'>
```

第 4 步还展示了如何在长文本中查找多个出现。虽然`.findall()`方法存在，但它不返回完整的匹配对象，而`.findalliter()`则返回。现在在 Python 3 中很常见，`.findalliter()`返回一个迭代器，可以在 for 循环或列表推导中使用。请注意，`.search()`仅返回模式的第一个匹配，即使出现更多匹配：

```py
>>> PATTERN.search(TEXT)
<_sre.SRE_Match object; span=(31, 40), match='Odessa,TX'>
>>> PATTERN.findall(TEXT)
[('Odessa', 'TX'), ('Corvallis', 'OR'), ('Toledo', 'OH')]
```

# 还有更多...

特殊字符可以反转，如果它们被大小写交换。例如，我们使用的特殊字符的反向如下：

+   `\D`：标记任何非数字

+   `\W`：标记任何非字母

+   `\B`：标记任何不在单词开头或结尾的字符

最常用的特殊字符通常是`\d`（数字）和`\w`（字母和数字），因为它们标记了常见的搜索模式，加号表示一个或多个。

组也可以分配名称。这样可以使它们更加明确，但会使组变得更冗长，形式如下—`(?P<groupname>PATTERN)`。可以通过名称引用组，使用`.group(groupname)`或通过调用`.groupdict()`来保持其数字位置。

例如，步骤 4 的模式可以描述如下：

```py
>>> PATTERN = re.compile(r'(?P<city>[A-Z][\w\s]+?).(?P<state>TX|OR|OH|MN)')
>>> match = PATTERN.search(TEXT)
>>> match.groupdict()
{'city': 'Odessa', 'state': 'TX'}
>>> match.group('city')
'Odessa'
>>> match.group('state')
'TX'
>>> match.group(1), match.group(2)
('Odessa', 'TX')
```

正则表达式是一个非常广泛的主题。有整本专门讨论它们的技术书籍，它们可能非常深奥。Python 文档是一个很好的参考（[`docs.python.org/3/library/re.html`](https://docs.python.org/3/library/re.html)）并且可以学到更多。

如果一开始感到有点害怕，这是完全正常的感觉。仔细分析每个模式，将其分成不同的部分，它们将开始变得有意义。不要害怕运行正则表达式交互式分析器！

正则表达式可能非常强大和通用，但它们可能不是您尝试实现的目标的合适工具。我们已经看到了一些细微差别和模式。作为一个经验法则，如果一个模式开始感觉复杂，那么是时候寻找另一个工具了。还记得之前的配方以及它们提供的选项，比如`parse`。

# 另请参阅

+   *介绍正则表达式*配方

+   *使用第三方工具—parse*配方

# 添加命令行参数

许多任务最好被构造为接受不同参数以改变工作方式的命令行接口，例如，抓取一个网页或另一个网页。Python 在标准库中包含了一个强大的`argparse`模块，可以轻松创建丰富的命令行参数解析。

# 准备工作

脚本中`argparse`的基本用法可以分为三个步骤：

1.  定义脚本将接受的参数，生成一个新的解析器。

1.  调用定义的解析器，返回一个包含所有结果参数的对象。

1.  使用参数调用脚本的入口点，这将应用定义的行为。

尝试使用以下通用结构编写脚本：

```py
IMPORTS

def main(main parameters):
  DO THINGS

if __name__ == '__main__':
    DEFINE ARGUMENT PARSER
    PARSE ARGS
    VALIDATE OR MANIPULATE ARGS, IF NEEDED
    main(arguments)
```

`main`函数使得很容易知道代码的入口点。`if`语句下的部分只有在文件直接调用时才会执行，而不是在导入时执行。我们将对所有步骤都遵循这一点。

# 如何做...

1.  创建一个脚本，它将接受一个单个整数作为位置参数，并打印出相应次数的哈希符号。`recipe_cli_step1.py`脚本如下，但请注意我们正在遵循之前介绍的结构，并且`main`函数只是打印参数：

```py
import argparse

def main(number):
    print('#' * number)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int, help='A number')
    args = parser.parse_args()

    main(args.number)
```

1.  调用脚本并查看参数的呈现方式。使用无参数调用脚本会显示自动帮助信息。使用自动参数`-h`显示扩展帮助信息：

```py
$ python3 recipe_cli_step1.py
usage: recipe_cli_step1.py [-h] number
recipe_cli_step1.py: error: the following arguments are required: number
$ python3 recipe_cli_step1.py -h
usage: recipe_cli_step1.py [-h] number
positional arguments:
 number A number
optional arguments:
 -h, --help show this help message and exit
```

1.  使用额外参数调用脚本会按预期工作：

```py
$ python3 recipe_cli_step1.py 4
####
$ python3 recipe_cli_step1.py not_a_number
usage: recipe_cli_step1.py [-h] number
recipe_cli_step1.py: error: argument number: invalid int value: 'not_a_number'
```

1.  更改脚本以接受一个可选参数用于打印的字符。默认值将是`'#'`。`recipe_cli_step2.py`脚本将如下所示：

```py
import argparse

def main(character, number):
    print(character * number)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int, help='A number')
    parser.add_argument('-c', type=str, help='Character to print',
                        default='#')

args = parser.parse_args()
main(args.c, args.number)
```

1.  帮助信息已更新，使用`-c`标志允许我们打印不同的字符：

```py
$ python3 recipe_cli_step2.py -h
usage: recipe_cli_step2.py [-h] [-c C] number

positional arguments:
 number A number

optional arguments:
 -h, --help show this help message and exit
 -c C Character to print
$ python3 recipe_cli_step2.py 4
####
$ python3 recipe_cli_step2.py 5 -c m
mmmmm
```

1.  添加一个标志，当存在时改变行为。`recipe_cli_step3.py`脚本如下：

```py
import argparse

def main(character, number):
    print(character * number)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int, help='A number')
    parser.add_argument('-c', type=str, help='Character to print',
                        default='#')
    parser.add_argument('-U', action='store_true', default=False,
                        dest='uppercase',
                        help='Uppercase the character')
    args = parser.parse_args()

    if args.uppercase:
        args.c = args.c.upper()

    main(args.c, args.number)
```

1.  如果添加了`-U`标志，则调用它会将字符转换为大写：

```py
$ python3 recipe_cli_step3.py 4 -c f
ffff
$ python3 recipe_cli_step3.py 4 -c f -U
FFFF
```

# 工作原理...

如*如何做...*部分中的步骤 1 所述，通过`.add_arguments`将参数添加到解析器中。一旦定义了所有参数，调用`parse_args()`将返回一个包含结果的对象（或者如果有错误则退出）。

每个参数都应该添加一个帮助描述，但它们的行为可能会有很大变化：

+   如果参数以`-`开头，则被视为可选参数，就像步骤 4 中的`-c`参数一样。如果不是，则是位置参数，就像步骤 1 中的`number`参数一样。

为了清晰起见，始终为可选参数定义默认值。如果不这样做，它将是`None`，但这可能会令人困惑。

+   记得始终添加一个带有参数描述的帮助参数；帮助将自动生成，如步骤 2 所示。

+   如果存在类型，将进行验证，例如，在步骤 3 中的`number`。默认情况下，类型将为字符串。

+   `store_true`和`store_false`操作可用于生成标志，不需要任何额外参数的参数。将相应的默认值设置为相反的布尔值。这在步骤 6 和 7 中的`U`参数中有所示。

+   `args`对象中属性的名称默认情况下将是参数的名称（如果存在破折号，则不包括）。您可以使用`dest`更改它。例如，在步骤 6 中，命令行参数`-U`被描述为`uppercase`。

在使用短参数（如单个字母）时，更改参数的名称以供内部使用非常有用。一个良好的命令行界面将使用`-c`，但在内部使用更详细的标签，如`configuration_file`可能是一个好主意。显式胜于隐式！

+   一些参数可以与其他参数协同工作，如步骤 3 所示。执行所有必需的操作，以清晰简洁的参数传递主要函数。例如，在步骤 3 中，只传递了两个参数，但可能已经修改了一个参数。

# 还有更多...

您也可以使用双破折号创建长参数，例如：

```py
 parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                     help='Enable verbose output')
```

这将接受`-v`和`--verbose`，并将存储名称`verbose`。

添加长名称是使界面更直观和易于记忆的好方法。几次之后很容易记住有一个冗长的选项，并且以`v`开头。

处理命令行参数时的主要不便之处可能是最终拥有太多参数。这会造成混乱。尽量使参数尽可能独立，不要在它们之间建立太多依赖关系，否则处理组合可能会很棘手。

特别是，尽量不要创建超过一对位置参数，因为它们没有助记符。位置参数也接受默认值，但大多数情况下这不是预期的行为。

有关详细信息，请查看 Python 的`argparse`文档（[`docs.python.org/3/library/argparse.html`](https://docs.python.org/3/library/argparse.html)）。

# 另请参阅

+   *创建虚拟环境*食谱

+   *安装第三方软件包*食谱
