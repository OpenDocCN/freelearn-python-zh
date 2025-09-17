# 第五章：渲染数据

能够编写自己的执行和状态模块对于开发者来说是一种强大的能力，但你不能忽视为那些没有能力提供自己模块的用户提供这种能力。

渲染器允许用户使用不同类型的数据输入格式向 Salt 的各个部分提供数据。Salt 附带的一小部分渲染器涵盖了大多数用例，但如果你用户需要以专用格式应用数据怎么办？或者甚至是一个尚未支持但更常见的格式，如 XML？在本章中，我们将讨论：

+   编写渲染器

+   解决渲染器问题

# 理解文件格式

默认情况下，Salt 使用 YAML 处理其各种文件。有两个主要原因：

+   YAML 可以轻松转换为 Python 数据结构

+   YAML 易于人类阅读和修改

Salt 配置文件也必须使用 YAML（或 JSON，可以被 YAML 解析器读取），但其他文件，如状态、支柱、反应器等，可以使用其他格式。数据序列化格式是最常见的，但任何可以转换为 Python 字典的格式都可以。

例如，Salt 附带三个不同的 Python 渲染器：`py`、`pyobjects`和`pydsl`。每个都有其优点和缺点，但最终结果相同：它们执行 Python 代码，生成字典，然后传递给 Salt。

一般而言，你会在 Salt 中找到两种类型的渲染器。第一种返回 Python 数据结构中的数据。序列化和基于代码的模块都属于这一类别。第二种用于管理文本格式化和模板。让我们依次讨论每个部分，然后在章节的后面构建我们自己的渲染器。

## 序列化数据

数据可以存储在任何数量的格式中，但最终，这些数据必须是能够转换为指令的东西。YAML 和 JSON 等格式是明显的选择，因为它们易于修改，并且反映了使用它们的程序中的结果数据结构。二进制格式，如 Message Pack，虽然不易于人类修改，但它们仍然产生相同的数据结构。

其他格式，如 XML，更难处理，因为它们并不直接类似于 Salt 等程序的内部数据结构。它们非常适合建模大量使用类的代码，但 Salt 并不大量使用这种代码。然而，当你知道如何将这种格式转换为 Salt 可以使用的数据结构时，为其构建渲染器并不困难。

## 与模板一起工作

模板很重要，因为它们允许最终用户使用某些程序元素，而无需编写实际的模块。变量无疑是模板引擎中最关键元素之一，但其他结构，如循环和分支，也可以给用户带来很大的权力。

模板渲染器与数据序列化渲染器不同，因为它们不是以字典格式返回数据，然后由 Salt 摄取，而是返回至少需要使用数据序列化渲染器转换一次的数据。

在某些层面上，这可能会显得有些反直觉，但使用渲染管道将这两个元素结合起来。

## 使用渲染管道

渲染管道基于 Unix 管道；数据可以通过一系列管道从模块传递到模块，以便到达最终的数据结构。你可能没有意识到，但如果你曾经编写过 SLS 文件，你就已经使用了渲染管道。

要设置渲染管道，你需要在要渲染的文件顶部添加一行，其中包含经典的 Unix hashbang，后面跟着要使用的渲染器，按照使用的顺序，由管道字符分隔。默认的渲染顺序实际上是：

```py
#!jinja|yaml

```

这意味着相关的文件将首先由 Jinja2 解析，然后编译成 YAML 库可以读取的格式。

通常来说，将两个以上的不同渲染器组合在一起并不合理或必要；使用的越多，结果文件对人类来说就越复杂，出错的可能性也越大。一般来说，一个添加了程序性快捷方式的模板引擎和一个数据序列化器就足够了。一个值得注意的例外是`gpg`渲染器，它可以用于静态加密场景。这个 hashbang 看起来会是这样：

```py
#!jinja|yaml|gpg

```

# 构建序列化渲染器

渲染器相对容易构建，因为它们通常做的只是导入一个库，将数据通过它，然后返回结果。我们的示例渲染器将使用 Python 自己的 Pickle 格式。

## 基本结构

在任何必要的导入之外，渲染器只需要一个`render()`函数。最重要的参数是第一个。与其他模块一样，这个参数的名称对 Salt 来说并不重要，只要它被定义即可。因为我们的例子使用了`pickle`库，所以我们将使用`pickle_data`作为我们的参数名称。

其他参数也会传递给渲染器，但在这个例子中，我们只会用它们来解决问题。特别是，我们需要接受`saltenv`和`sls`，稍后会显示它们的默认值。我们将在“*渲染器故障排除*”部分介绍它们，但现在我们只需使用`kwargs`来涵盖它们。

我们还需要从一种特殊的`import`开始，称为`absolute_import`，它允许我们从也称为`pickle`的文件中导入`pickle`库。

让我们先列出模块，然后讨论`render()`函数中的组件：

```py
'''
Render Pickle files.

This file should be saved as salt/renderers/pickle.py
'''
from __future__ import absolute_import
import pickle
from salt.ext.six import string_types

def render(pickle_data, saltenv='base', sls='', **kwargs):
    '''
    Accepts a pickle, and renders said data back to a python dict.
    '''
    if not isinstance(pickle_data, string_types):
        pickle_data = pickle_data.read()

    if pickle_data.startswith('#!'):
        pickle_data = pickle_data[(pickle_data.find('\n') + 1):]
    if not pickle_data.strip():
        return {}
    return pickle.loads(pickle_data)
```

这个函数除了以下内容之外不做太多：

+   首先，检查传入的数据是否为字符串，如果不是，则将其视为文件对象。

+   检查是否存在`#!`，表示使用了显式的渲染管道。因为那个管道在其他地方处理，并且会导致与`pickle`库的错误，所以将其丢弃。

+   检查结果内容是否为空。如果是，则返回一个空字典。

+   通过`pickle`库运行数据，并返回结果。

如果您开始将此代码与 Salt 附带的自定义渲染器进行比较，您会发现它们几乎完全相同。这在很大程度上是因为 Python 中许多数据序列化库使用完全相同的方法。

让我们创建一个可以使用的文件。我们将使用的示例数据如下：

```py
apache:
  pkg:
    - installed
    - refresh: True
```

创建此文件的最佳方式是使用 Python 本身。请打开 Python shell 并输入以下命令：

```py
>>> import pickle
>>> data = {'apache': {'pkg': ['installed', {'refresh': True}]}}
>>> out = open('/srv/salt/pickle.sls', 'w')
>>> pickle.dump(data, out)
>>> out.close()

```

当您退出 Python shell 时，您应该能够用您最喜欢的文本编辑器打开此文件。当您在顶部添加一个指定`pickle`渲染器的 hashbang 行时，您的文件可能看起来像这样：

```py
#!pickle
(dp0
S'apache'
p1
(dp2
S'pkg'
p3
(lp4
S'installed'
p5
a(dp6
S'refresh'
p7
I01
sass.
```

保存文件，并使用`salt-call`测试您的渲染器。这次，我们将告诉 Salt 将结果 SLS 以 Salt 看到的形式输出：

```py
# salt-call --local state.show_sls pickle --out=yaml
local:
 apache:
 __env__: base
 __sls__: !!python/unicode pickle
 pkg:
 - installed
 - refresh: true
 - order: 10000

```

Salt 的状态编译器添加了一些它内部使用的额外信息，但我们可以看到我们请求的基本内容都在那里。

# 构建模板渲染器

构建处理模板文件的渲染器与处理序列化的渲染器没有太大区别。实际上，渲染器本身除了库特定的代码外，几乎相同。这次，我们将使用一个名为`tenjin`的 Python 库。您可能需要使用 pip 安装它：

```py
# pip install tenjin

```

## 使用 Tenjin 进行模板化

此模块使用第三方库，因此将有一个`__virtual__()`函数来确保它已安装：

```py
'''
Conver a file using the Tenjin templating engine

This file should be saved as salt/renderers/tenjin.py
'''
from __future__ import absolute_import
try:
    import tenjin
    from tenjin.helpers import *
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
from salt.ext.six import string_types

def __virtual__():
    '''
    Only load if Tenjin is installed
    '''
    return HAS_LIBS

def render(tenjin_data, saltenv='base', sls='', **kwargs):
    '''
    Accepts a tenjin, and renders said data back to a python dict.
    '''
    if not isinstance(tenjin_data, string_types):
        tenjin_data = tenjin_data.read()

    if tenjin_data.startswith('#!'):
        tenjin_data = tenjin_data[(tenjin_data.find('\n') + 1):]
    if not tenjin_data.strip():
        return {}

    template = tenjin.Template(input=tenjin_data)
    return template.render(kwargs)
```

`render()`函数本身与用于`pickle`的函数基本相同，除了最后两行，它们对模板引擎的处理略有不同。

注意传递给此函数的`kwargs`。模板引擎通常具有合并外部数据结构的能力，这可以与模板引擎本身的各种数据结构一起使用。Salt 会在`kwargs`中提供一些数据，因此我们将传递这些数据以供 Tenjin 使用。

## 使用模板渲染器

当然，您需要在 SLS 文件中添加一个 hashbang 行，就像之前一样，但由于我们的 Tenjin 渲染器没有设置为直接返回数据，您需要将所需的数据序列化渲染器的名称添加到您的渲染管道中。我们将使用之前相同的实际 SLS 数据，但添加了一些 Tenjin 特定的元素：

```py
#!tenjin|yaml
<?py pkg = 'apache'?>
<?py refresh = True?>
#{pkg}:
  pkg:
    - installed
    - refresh: #{refresh}
```

我们在这里没有做任何特别的事情，只是设置了一些变量，然后使用了它们。结果内容将是 YAML 格式，因此我们向我们的渲染管道添加了`yaml`。

许多模板引擎，包括 Tenjin，都有能力处理输出字符串（如我们示例中所做）或实际数据结构（如数据序列化器返回的数据）的模板。当使用此类库时，请花点时间考虑您计划使用多少，以及是否需要为它创建两个不同的渲染器：一个用于数据，一个用于字符串。

测试与之前相同：

```py
# salt-call --local state.show_sls tenjin --out yaml
local:
 apache:
 pkg:
 - installed
 - refresh: true
 - order: 10000
 __sls__: !!python/unicode tenjin
 __env__: base

```

我们可以看到我们的第一个示例和第二个示例之间有一些细微的差异，但这些差异只是显示了用于渲染数据的模块。

# 渲染器故障排除

由于渲染器经常被用来管理 SLS 文件，因此使用状态编译器进行故障排除通常是最简单的，正如我们在本章中已经做的那样。

首先，生成一个包含你需要测试的特定元素的 SLS 小文件。这可能是使用序列化引擎格式的数据文件，或者是一个生成数据序列化文件格式的基于文本的文件。如果你正在编写模板渲染器，通常最简单的方法就是使用 YAML。

`state`执行模块包含许多主要用于故障排除的函数。我们在示例中使用了`state.show_sls`，并带有`--out yaml`选项，因为它以我们在 SLS 文件中已经习惯的格式显示输出。然而，还有一些其他有用的函数：

+   `state.show_low_sls`：显示单个 SLS 文件在状态编译器将其转换为低数据后的数据。在编写状态模块时，低数据通常更容易可视化。

+   `state.show_highstate`：显示所有状态，根据`top.sls`文件，它们将被应用到 Minion 上。此输出的外观就像所有 SLS 文件都被堆在一起一样。这在故障排除你认为跨越多个 SLS 文件的渲染问题时可能很有用。

+   `state.show_lowstate`：此函数返回的数据与`state.show_highstate`返回的数据相同，但经过状态编译器处理。这又像是`state.show_low_sls`的长版本。

# 摘要

渲染器用于将各种文件格式转换为 Salt 内部可用的数据结构。数据序列化渲染器以字典格式返回数据，而模板渲染器返回可以由数据序列化器处理的数据。这两种类型的渲染器看起来相同，都需要一个`render()`函数。

现在我们已经知道如何处理进入 Salt 的数据，是时候看看从 Salt 返回的数据了。接下来：处理返回数据。
