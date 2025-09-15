## 13

应用程序集成：配置

Python 的可扩展库概念为我们提供了对众多计算资源的丰富访问。该语言提供了使更多资源可用的途径。这使得 Python 程序特别擅长集成组件以创建复杂的复合处理。在本章中，我们将讨论创建复杂应用程序的基本原则：管理配置文件、日志记录以及允许自动化测试的脚本设计模式。

这些新食谱借鉴了其他章节中食谱中的想法。具体来说，在第六章的使用 argparse 获取命令行输入、使用 cmd 创建命令行应用程序和使用 OS 环境设置的食谱中，展示了创建顶级（主）应用程序脚本的一些特定技术。回顾这些食谱可能有助于查看 Python 应用程序脚本的示例。在第十一章中，我们探讨了文件系统的输入和输出。

在本章中，我们将探讨处理配置文件的各种方法。有许多文件格式可以用来存储长期配置信息：

+   configparser 模块处理的 INI 文件格式。

+   TOML 文件格式非常易于处理，但需要一个不是 Python 发行版中当前部分的附加模块。我们将在使用 TOML 作为配置文件的食谱中探讨这一点。

+   属性文件格式是 Java 编程的典型格式，在 Python 中无需编写太多代码即可处理。一些语法与 Python 脚本和 TOML 文件重叠。从属性文件格式切换到 TOML 只需要将任何 name:value 更改为 name = "value"，允许使用 TOML 解析器。

+   对于 Python 脚本，具有赋值语句的文件看起来很像属性文件，并且可以使用 compile()和 exec()函数非常容易地处理。我们将在使用 Python 作为配置文件的食谱中探讨这一点。

+   具有类定义的 Python 模块是一种使用 Python 语法但将设置隔离到单独类中的变体。这可以通过导入语句进行处理。我们将在使用类作为配置的命名空间的食谱中探讨这一点。

本章中的一些食谱将扩展第七章和第八章中的一些概念。本章将应用这些概念来定义使用类定义配置文件。

考虑配置文件中所需的信息类型非常重要。在配置文件中不小心包含密码或安全令牌可能会对数据的安全使用造成致命影响。在配置文件中包含个人信息也是一种常见的安全弱点。请参阅[常见弱点枚举](https://cwe.mitre.org/index.html)了解设计不良配置文件的其他更具体问题。

在本章中，我们将探讨以下菜谱：

+   查找配置文件

+   使用 TOML 配置文件

+   使用 Python 配置文件

+   将类用作配置的命名空间

+   设计用于组合的脚本

+   使用日志进行控制和审计输出

我们将从一个处理必须组合的多个配置文件的菜谱开始。这为用户提供了一些有用的灵活性。从那里，我们可以深入了解一些常见配置文件格式的具体细节。

# 13.1 查找配置文件

许多应用程序将具有配置选项的层次结构。这个层次结构的基础通常是应用程序中内置的默认值。这些可能由来自集中式配置文件的整个服务器（或集群）范围内的值补充。也可能有特定于用户的文件，或者甚至可能是启动程序时提供的配置文件。

在许多情况下，配置参数是以文本文件的形式编写的，因此它们是持久的并且易于更改。在 Linux 中，常见的传统是将系统范围的配置放在/etc 目录中。用户的个人更改将放在他们的家目录中，通常命名为~username 或$HOME。

在这个示例中，我们将看到应用程序如何支持配置文件的丰富层次结构。

## 13.1.1 准备工作

我们将使用的示例是一个模拟掷骰子的应用程序。该应用程序在第六章的几个菜谱中有所展示。6。具体来说，请查看使用 argparse 获取命令行输入和使用 cmd 创建命令行应用程序。

我们将遵循 Bash shell 的设计模式，它会在以下位置查找配置文件：

1.  它从/etc/profile 文件开始，适用于使用系统的每个人。

1.  在读取该文件后，它会按以下顺序查找这些文件之一：

    1.  ~/.bash_profile

    1.  ~/.bash_login

    1.  ~/.profile

其他 shell，如 zsh，使用一些额外的文件，但遵循通过一系列文件顺序工作的模式。

在符合 POSIX 标准的操作系统上，shell 将~展开为登录用户的家目录。通常，Python 的 pathlib 模块通过 Path.home()方法自动处理 Windows、Linux 和 macOS。

在后面的菜谱中，我们将探讨解析和处理配置文件特定格式的各种方法。对于本菜谱的目的，我们不会选择特定的格式。相反，我们将假设已经定义了一个名为 load_config_file()的现有函数，该函数将从配置文件的 内容中加载特定的配置映射。

函数看起来是这样的：

```py
def load_config_file(config_path: Path) -> dict[str, Any]: 

    """Loads a configuration mapping object with the contents 

    of a given file. 

    :param config_path: Path to be read. 

    :returns: mapping with configuration parameter value 

    """ 

    # Details omitted.
```

我们将探讨实现此函数的多种不同方法。

### 为什么有这么多选择？

在讨论这类设计时，有时会出现一个相关话题——为什么有这么多选择？为什么不指定一个确切的位置？

提供一个分布特有的变体，但另一个分布不典型的情况很常见。此外，用户的期望取决于他们已经熟悉的软件；这很难预测。当然，当处理 Windows 时，还可能出现更多仅适用于该平台的变体文件路径。出于这些原因，提供多个位置并允许用户或管理员选择他们偏好的位置更容易。

## 13.1.2 如何做到这一点...

我们将利用 pathlib 模块提供一种方便的方式来处理各种位置上的文件。我们还将使用 collections 模块提供非常有用的 ChainMap 类：

1.  导入 Path 类和 ChainMap 类。还需要几个类型提示：

    ```py
    from pathlib import Path 

    from collections import ChainMap 

    from typing import TextIO, Any
    ```

1.  定义一个获取配置文件的整体函数：

    ```py
    def get_config() -> ChainMap[str, Any]:
    ```

1.  为配置文件的各个位置创建路径。这些被称为纯路径，并以潜在文件的名字开头。我们可以将这些位置分解为系统路径和一系列本地路径。以下是两个赋值语句：

    ```py
     system_path = Path("/etc") / "some_app" / "config" 

        local_paths = [ 

        ".some_app_settings", 

        ".some_app_config", 

        ]
    ```

1.  将应用程序的内置默认值定义为一个字典列表：

    ```py
     configuration_items = [ 

            dict( 

                some_setting="Default Value", 

                another_setting="Another Default", 

                some_option="Built-In Choice", 

            ) 

        ]
    ```

    每个单独的配置文件是从键到值的映射。这些映射对象中的每一个都被组合成一个列表；这成为最终的 ChainMap 配置映射。

1.  如果存在系统范围的配置文件，则加载此文件：

    ```py
     if system_path.exists(): 

            configuration_items.append( 

                load_config_file(system_path))
    ```

1.  遍历其他位置以查找要加载的文件。这将加载它找到的第一个文件，并使用 break 语句在找到第一个文件后停止：

    ```py
     for config_name in local_paths: 

            config_path = Path.home() / config_name 

            if config_path.exists(): 

                configuration_items.append( 

                    load_config_file(config_path)) 

                break
    ```

1.  反转列表并创建最终的 ChainMap 映射：

    ```py
     configuration = ChainMap( 

            *reversed(configuration_items) 

        )
    ```

    列表需要反转，以便首先搜索附加在最后的本地文件，然后是系统设置，最后是应用程序默认设置。当然，可以按相反的顺序组装列表以避免 reversed()函数；我们把这个可能的变化留给你作为练习。

一旦我们构建了配置对象，我们就可以像简单的映射一样使用最终的配置。此对象支持所有预期的字典操作。

## 13.1.3 它是如何工作的...

在第五章的 创建字典——插入和更新 配方中，我们探讨了使用字典的基本知识。在这里，我们将几个字典组合成一个链。当一个键不在链的第一个字典中时，则检查链中的后续字典。这是一种为映射中的每个键提供默认值的好方法。由于 ChainMap 几乎与内置的 dict 类无法区分，它允许在实现细节上具有很大的灵活性：任何可以读取以创建字典的配置文件都是完全可接受的。应用程序的其余部分可以基于字典，而不必暴露配置构建的细节。

## 13.1.4 更多...

单一的全局配置文件和本地配置文件替代名称集合之间的微妙区别并不理想。这种单例和选择列表之间的区别似乎没有特定的用途。通常，我们希望扩展这种设计，而微小的非对称性会导致复杂性。

我们将考虑将配置修改为以下四个层级：

1.  内置的默认值。

1.  在像 /etc 或 /opt 这样的中心目录中的主机级配置。这通常用于此容器的 OS 或网络上下文的详细信息。

1.  为运行应用程序的用户配置的 home 目录。这可能用于区分测试和生产实例。

1.  当前工作目录中的本地文件。这可能由开发者或测试人员使用。

这表明对配方进行修改以使用路径的嵌套列表。外层列表包含所有配置层级。在每个层级内，一个列表将包含配置文件的替代位置。

```py
local_names = (’.some_app_settings’, ’.some_app_config’) 

config_paths = [ 

    [ 

        base / ’some_app’ / ’config’ 

        for base in (Path(’/etc’), Path(’/opt’)) 

    ], 

    [ 

        Path.home() / name 

        for name in local_names 

    ], 

    [ 

        Path.cwd() / name 

        for name in local_names 

    ], 

]
```

这种 list[list[Path]] 结构提供了三个配置文件层级。每个层级都有多个替代名称。层级的顺序以及每个层级内的名称都很重要。较低层级提供对较高层级的覆盖。然后我们可以使用嵌套的 for 语句来检查所有替代位置。

```py
def get_config_2() -> ChainMap[str, Any]: 

    configuration_items = [ 

        DEFAULT_CONFIGURATION 

    ] 

    for tier_paths in config_paths: 

        for alternative in tier_paths: 

            if alternative.exists(): 

                configuration_items.append( 

                    load_config_file(alternative)) 

                break 

    configuration = ChainMap( 

        *reversed(configuration_items) 

    ) 

    return configuration
```

我们已经将默认配置提取到一个名为 DEFAULT_CONFIGURATION 的全局变量中。我们明显地留下了名为 config_paths 的配置路径集合。不清楚这是否应该是全局的（并且使用全大写的全局变量名）或者是否应该是 get_config() 函数的一部分。我们通过使用小写名称并将其放在函数外部，采取了一部分两者的做法。

config_paths 的值可能不会在其他地方需要，因此将其作为全局变量是一个糟糕的选择。然而，这却是一样可能会改变的东西——也许在下一个主要版本中——并且值得暴露出来以便进行更改。

## 13.1.5 参见

+   在本章的 使用 TOML 作为配置文件 和 使用 Python 作为配置文件 菜谱中，我们将探讨实现 load_config_file() 函数的方法。

+   在第十五章的 模拟外部资源 菜谱中，我们将探讨测试此类函数的方法，这些函数与外部资源交互。

+   pathlib 模块是此处理的核心。此模块提供了 Path 类定义，它提供了关于操作系统文件的许多复杂信息。有关更多信息，请参阅第十一章的 使用 pathlib 处理文件名 菜谱。

# 13.2 使用 TOML 作为配置文件

Python 提供了多种打包应用程序输入和配置文件的方式。我们将探讨使用 TOML 语法编写文件，因为这种格式优雅且简单。有关此格式的更多信息，请参阅 [`toml.io/en/`](https://toml.io/en/)。

大多数 TOML 文件看起来与 INI 格式文件非常相似。这种重叠是有意为之的。在 Python 中解析时，TOML 文件将是一个嵌套字典结构。

我们可能有一个这样的文件：

```py
[some_app] 

    option_1 = "useful value" 

    option_2 = 42 

[some_app.feature] 

    option_1 = 7331
```

这将变成如下所示的字典：

```py
{’some_app’: {’feature’: {’option_1’: 7331}, 

              ’option_1’: ’useful value’, 

              ’option_2’: 42}}
```

[some_app.feature] 被称为“表格”。在键中使用 . 会创建一个嵌套表格。

## 13.2.1 准备工作

我们经常会使用本章前面展示的 寻找配置文件 菜谱，以检查给定配置文件的各种位置。这种灵活性对于创建易于在各种平台上使用的应用程序通常是必不可少的。

在本菜谱中，我们将构建 寻找配置文件 菜谱中缺失的部分，即 load_config_file() 函数。以下是需要填写的模板：

```py
def load_config_file_draft(config_path: Path) -> dict[str, Any]: 

    """Loads a configuration mapping object with contents 

    of a given file. 

    :param config_path: Path to be read. 

    :returns: mapping with configuration parameter values 

    """ 

    # Details omitted.
```

在本菜谱中，我们将填写 Details 省略行保留的空间，以加载 TOML 格式的配置文件。

## 13.2.2 如何做...

本菜谱将使用 tomllib 模块解析 YAML-TOML 文件：

1.  导入 tomllib 模块以及 Path 定义和 load_config_file() 函数定义所需的类型提示：

    ```py
    from pathlib import Path 

    from typing import Any 

    import tomllib
    ```

1.  使用 tomllib.load() 函数加载 TOML 语法文档：

    ```py
    import tomllib 

    def load_config_file(config_path: Path) -> dict[str, Any]: 

        """Loads a configuration mapping object with contents 

        of a given file. 

        :param config_path: Path to be read. 

        :returns: mapping with configuration parameter values 

        """ 

        with config_path.open(’b’) as config_file: 

            document = tomllib.load(config_file)
    ```

    Python 中 TOML 解析的一个不寻常的要求是我们需要在使用 load() 函数时以“二进制”模式打开文件。我们可以使用 'rb' 作为模式，以明确表示文件是为读取而打开的。

    另一种选择是使用 loads() 函数对一个文本块进行操作。它看起来像这样：

    ```py
        document = tomllib.loads(config_path.read_text())
    ```

此 load_config_file() 函数生成了所需的字典结构。它可以适应 寻找配置文件 菜谱中的设计，以使用 TOML 语法加载配置文件。

## 13.2.3 它是如何工作的...

如上所述，TOML 语法的理念是易于阅读，并直接映射到 Python 字典。TOML 语法与 INI 文件语法之间有一些有意重叠。它也与属性文件语法的某些方面有重叠。

TOML 语法的核心是键值对，通常写作 key = value。键包括有效的 Python 符号。这意味着任何具有字典映射的数据类或 Pydantic 结构都可以映射到 TOML 语法中。

TOML 的键可以包含连字符，这不是 Python 允许的名称的一部分。键也可以是引号字符串。这允许有相当广泛的替代键。这些特性在使用配置字典对象时可能需要一些谨慎。

键也可以是点分隔的；这将创建子字典。以下是一个点分隔键的示例：

```py
some_app.option_1 = "useful value" 

some_app.option_2 = 42 

some_app.feature.option_1 = 7331
```

这看起来与常用于 Java 应用程序的属性文件非常相似。这通过在点字符处分解键来创建嵌套字典。

可用的值种类繁多，包括字符串值、整数值、浮点值和布尔值（使用 true 和 false 作为字面值）。此外，TOML 还识别 ISO 日期时间字符串；有关支持的格式，请参阅 [RFC 3339](https://tools.ietf.org/html/rfc3339)。

TOML 允许两种数据结构：

+   数组用 `[` 和 `]` 括起来。我们可以使用 `sizes = [1, 2, 3]` 来创建一个 Python 列表值。

+   可以使用 `{` 和 `}` 包围一个或多个键值对来创建内联表格。例如，`sample = {x = 10, y = 8.4}` 创建了一个嵌套字典值。

TOML 语法的一个重要特性是使用 [table] 作为嵌套字典的键。我们经常会看到这种情况：

```py
[some_app] 

    option_1 = "useful value" 

    option_2 = 42 

[some_app.feature] 

    option_1 = 7331
```

`[some_app]` 是一个字典的键，该字典包含缩进的键值对。TOML 语法中的 `[some_app.feature]` 定义了一个更深层次的字典。使用点分隔的键意味着字符串 "some_app" 将成为包含键 "feature" 的字典的键。与该键关联的值将是一个包含键 "option_1" 的字典。在 TOML 中，[table] 前缀用于嵌套值，创建了一个视觉组织，使得查找和更改配置设置更加容易。

## 13.2.4 更多内容...

TOML 语法用于描述 Python 项目的整体 `pyproject.toml` 文件。此文件通常有两个顶级表格：[project] 和 [build-system]。项目表将包含有关 [project] 的元数据。以下是一个示例：

```py
[project] 

name = "python_cookbook_3e" 

version = "2024.1.0" 

description = "All of the code examples for Modern Python Cookbook, 3rd Ed." 

readme = "README.rst" 

requires-python = ">=3.12" 

license = {file = "LICENSE.txt"}
```

`[build-system]` 表格提供了有关安装模块、软件包或应用程序所需工具的信息。以下是一个示例：

```py
[build-system] 

build-backend = ’setuptools.build_meta’ 

requires = [ 

    ’setuptools’, 

]
```

此文件提供了关于项目的一些基本信息。使用 TOML 语法使得阅读和更改相对容易。

## 13.2.5 参见

+   参考本章前面的查找配置文件配方，了解如何搜索多个文件系统位置以查找配置文件。我们可以轻松地将应用程序默认设置、系统级设置和个人设置分别存入不同的文件，并由应用程序组合。

+   更多关于 TOML 语法的详细信息，请参阅[`toml.io/en/`](https://toml.io/en/)。

+   更多关于 pyproject.toml 文件的信息，请参阅 Python 打包权威机构文档[`pip.pypa.io/en/stable/reference/build-system/pyproject-toml/`](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/)。

# 13.3 使用 Python 配置文件

除了提供配置数据的 TOML 语法之外，我们还可以用 Python 符号编写文件；它既优雅又简单。由于配置文件是一个 Python 模块，因此它提供了极大的灵活性。

## 13.3.1 准备工作

Python 赋值语句对于创建配置文件来说特别优雅。语法可以简单、易于阅读且非常灵活。如果我们使用赋值语句，我们可以从单独的模块导入应用程序的配置细节。这个模块可以命名为 settings.py，以表明该模块专注于配置参数。

由于 Python 将每个导入的模块视为一个全局 Singleton 对象，因此我们可以让应用程序的多个部分都使用 import settings 语句来获取当前全局应用程序配置参数的一致视图。我们不需要担心使用 Singleton 设计模式管理对象，因为 Python 已经包含了这部分。

我们希望能够在文本文件中提供如下定义：

```py
"""Weather forecast for Offshore including the Bahamas 

""" 

query = {’mz’: 

    [’ANZ532’, 

     ’AMZ117’, 

     ’AMZ080’] 

} 

base_url = "https://forecast.weather.gov/shmrn.php"
```

此配置是一个 Python 脚本。参数包括两个变量，query 和 base_url。query 变量的值是一个包含单个键'mz'和一系列值的字典。

这可以被视为一系列相关 URL 的规范，这些 URL 都与[`forecast.weather.gov/shmrn.php?mz=ANZ532`](http://forecast.weather.gov/shmrn.php?mz=ANZ532)类似。

我们经常使用查找配置文件的配方来检查给定配置文件的各种位置。这种灵活性对于创建易于在各种平台上使用的应用程序通常是必不可少的。

在这个配方中，我们将构建查找配置文件配方中缺失的部分，即 load_config_file()函数。以下是需要填写的模板：

```py
def load_config_file_draft(config_path: Path) -> dict[str, Any]: 

    """Loads a configuration mapping object with contents 

    of a given file. 

    :param config_path: Path to be read. 

    :returns: mapping with configuration parameter values 

    """ 

    # Details omitted.
```

在这个配方中，我们将填充由# Details omitted 行保留的空间，以在 Python 格式中加载配置文件。

## 13.3.2 如何做...

我们可以利用 pathlib 模块来定位文件。我们还将利用内置的 compile()和 exec()函数来处理配置文件中的代码：

1.  导入 Path 定义和 load_config_file()函数定义所需的类型提示：

    ```py
    from pathlib import Path 

    from typing import Any
    ```

1.  使用内置的 compile() 函数将 Python 模块编译成可执行形式。此函数需要源文本以及读取文本的文件名。文件名对于创建有用且正确的回溯消息至关重要：

    ```py
    def load_config_file(config_path: Path) -> dict[str, Any]: 

        code = compile( 

            config_path.read_text(), 

            config_path.name, 

            "exec")
    ```

    在代码不来自文件的情况下，通常的做法是为文件名提供一个名称，如 <string>。

1.  执行由 compile() 函数创建的代码对象。这需要两个上下文。全局上下文提供了任何先前导入的模块，以及 __builtins__ 模块。局部上下文是 locals 字典；这是新变量将被创建的地方：

    ```py
     locals: dict[str, Any] = {} 

        exec( 

            code, 

            {"__builtins__": __builtins__}, 

            locals 

        ) 

        return locals
    ```

此 load_config_file() 函数生成所需的字典结构。它可以适应从 查找配置文件 配方中加载配置文件的设计，使用 Python 语法。

## 13.3.3 它是如何工作的...

Python 语言细节——语法和语义——体现在内置的 compile() 和 exec() 函数中。三个基本步骤如下：

1.  读取文本。

1.  使用 compile() 函数编译文本以创建代码对象。

1.  使用 exec() 函数执行代码对象。

exec() 函数反映了 Python 处理全局和局部变量的方式。为此函数提供了两个命名空间（映射）。这些可以通过 globals() 和 locals() 函数访问。

我们可以向 exec() 函数提供两个不同的字典：

+   全局对象的字典。最常见的用途是提供对导入的模块的访问，这些模块始终是全局的。可以在该字典中提供 __builtins__ 模块。在某些情况下，还需要添加其他模块，如 pathlib。

+   一个用于 locals 的字典，它将由每个赋值语句创建（或更新）。此局部字典允许我们在执行设置模块时捕获创建的变量。

exec() 函数将更新 locals 字典。我们预计全局字典不会被更新，并将忽略此集合发生的任何更改。

## 13.3.4 更多内容...

此配方建议配置文件完全是一系列 name = value 赋值语句。赋值语句使用 Python 语法，变量名和字面量语法也是如此。这允许配置利用 Python 的内置类型的大量集合。此外，Python 语句的全系列都可用。这导致了一些工程权衡。

因为配置文件中可以使用任何语句，这可能导致复杂性。如果配置文件中的处理变得过于复杂，文件就不再是配置，而是应用程序的第一级部分。非常复杂的功能应该通过修改应用程序编程来实现，而不是通过配置设置进行黑客攻击。Python 应用程序包含完整的源代码，因为通常修复源代码比创建超复杂的配置文件更容易。目标是让配置文件提供值以定制操作，而不是提供插件功能。

我们可能希望将操作系统环境变量作为配置的全局变量的一部分。这样做有助于确保配置值与当前环境设置匹配。这可以通过 os.environ 映射来完成。

对于相关设置进行一些处理也是合理的。例如，编写一个包含多个相邻路径的配置文件可能会有所帮助：

```py
"""Config with related paths""" 

base = Path(os.environ.get("APP_HOME", "/opt/app")) 

log = base / ’log’ 

out = base / ’out’
```

在许多情况下，设置文件是由可以信赖的人编辑的。尽管如此，错误仍然会发生，因此对提供给 exec()函数的全局字典中可用的函数保持谨慎是明智的。提供最窄的函数集以支持配置是推荐的做法。

## 13.3.5 参见

+   请参阅本章前面的查找配置文件配方，了解如何搜索多个文件系统位置以查找配置文件。

# 13.4 使用类作为配置的命名空间

Python 提供了多种打包应用程序输入和配置文件的方法。我们将继续探讨使用 Python 符号编写文件，因为它优雅且熟悉的语法可以导致易于阅读的配置文件。许多项目允许我们使用类定义来提供配置参数。这当然使用 Python 语法。它还使用类定义作为命名空间，以允许在单个模块中提供多个配置。使用类层次结构意味着可以使用继承技术来简化参数的组织。

这避免了使用 ChainMap 来允许用户对通用设置进行特定覆盖。相反，它使用普通的继承。

我们永远不会创建这些类的实例。我们将使用类定义的属性并依赖类继承方法来追踪属性的适当值。这与本章中的其他配方不同，因为它将生成一个 ConfigClass 对象，而不是一个 dict[str, Any]对象。

在本配方中，我们将探讨如何使用 Python 类符号表示配置细节。

## 13.4.1 准备工作

Python 定义类属性的方式可以简单、易于阅读，并且相当灵活。我们可以通过一些工作，定义一个复杂的配置语言，允许某人快速且可靠地更改 Python 应用程序的配置参数。

我们可以将这种语言基于类定义。这允许我们将多个配置选项打包在一个模块中。应用程序可以加载该模块并从模块中选择相关的类定义。

我们希望能够提供如下所示的定义：

```py
class Configuration: 

    """ 

    Generic Configuration with a sample query. 

    """ 

    base = "https://forecast.weather.gov/shmrn.php" 

    query = {"mz": ["GMZ856"]}
```

我们可以在 settings.py 文件中创建这个类定义来创建一个设置模块。要使用配置，主应用程序可以这样做：

```py
>>> from settings import Configuration 

>>> Configuration.base 

’https://forecast.weather.gov/shmrn.php’
```

应用程序将使用名为 settings 的模块名和名为 Configuration 的类名来收集设置。

配置文件的存储位置遵循 Python 查找模块的规则。我们不需要自己实现配置的搜索，而是可以利用 Python 内置的 sys.path 搜索以及 PYTHONPATH 环境变量的使用。

在这个菜谱中，我们将构建一个类似于查找配置文件菜谱的缺失部分，即 load_config_file()函数。然而，会有一个重要的区别：我们将返回一个对象而不是一个字典。然后我们可以通过属性名来引用配置值，而不是使用更繁琐的字典表示法。以下是需要填写的修订模板：

```py
ConfigClass = type[object] 

def load_config_file_draft( 

    config_path: Path, classname: str = "Configuration" 

) -> ConfigClass: 

    """Loads a configuration mapping object with contents 

    of a given file. 

    :param config_path: Path to be read. 

    :returns: mapping with configuration parameter values 

    """ 

    # Details omitted.
```

我们在本章的多个菜谱中使用了类似的模板。对于这个菜谱，我们添加了一个参数并更改了返回类型。在之前的菜谱中，没有 classname 参数，但在这里它被用来从由 config_path 参数命名的文件系统位置选择模块中的一个类。

## 13.4.2 如何实现...

我们可以利用 pathlib 模块来定位文件。我们将利用内置的 compile()和 exec()函数来处理配置文件中的代码：

1.  导入 Path 定义以及 load_config_file()函数定义所需的类型提示：

    ```py
    from pathlib import Path 

    import platform 
    ```

1.  使用内置的 compile()函数将 Python 模块编译成可执行形式。此函数需要源文本以及从其中读取文本的文件名。文件名对于创建有用且正确的回溯消息是必不可少的：

    ```py
    def load_config_file( 

        config_path: Path, classname: str = "Configuration" 

    ) -> ConfigClass: 

        code = compile( 

            config_path.read_text(), 

            config_path.name, 

            "exec")
    ```

1.  执行由 compile()方法创建的代码对象。我们需要提供两个上下文。全局上下文可以提供 __builtins__ 模块、Path 类和 platform 模块。局部上下文是新变量将被创建的地方：

    ```py
     globals = { 

            "__builtins__": __builtins__, 

            "Path": Path, 

            "platform": platform} 

        locals: dict[str, ConfigClass] = {} 

        exec(code, globals, locals) 

        return locals[classname]
    ```

    这将在 locals()映射中定位命名的类并返回该类作为配置对象。这不会返回一个字典。

在 load_config_file() 函数的这种变体中，产生了一个有用的结构，可以通过属性名称来访问。它并不提供 查找配置文件 菜单中预期的设计。由于它使用属性名称，因此产生的配置对象比简单的字典更有用。

## 13.4.3 它是如何工作的...

我们可以通过使用 compile() 和 exec() 来加载 Python 模块。从模块中，我们可以提取包含各种应用程序设置的单独类名。总体来说，它看起来像以下示例：

```py
>>> configuration = load_config_file( 

... Path(’src/ch13/settings.py’), ’Chesapeake’) 

>>> configuration.__doc__.strip() 

’Weather for Chesapeake Bay’ 

>>> configuration.query 

{’mz’: [’ANZ532’]} 

>>> configuration.base 

’https://forecast.weather.gov/shmrn.php’
```

我们可以将任何类型的对象放入配置类的属性中。我们的示例显示了字符串列表和字符串。使用类定义时，任何类的任何对象都成为可能。

我们可以在类声明中包含复杂的计算。我们可以使用这个功能来创建由其他属性派生出来的属性。我们可以执行任何类型的语句，包括 if 语句和 for 语句，来创建属性值。

然而，我们不会创建给定类的实例。像 Pydantic 这样的工具将验证类的实例，但对于验证类定义并不有帮助。任何类型的验证规则都必须定义在用于构建结果配置类的元类中。此外，类的一般方法将不会被使用。如果需要一个类似函数的定义，它必须用 @classmethod 装饰器来使其有用。

## 13.4.4 更多...

使用类定义意味着我们将利用继承来组织配置值。我们可以轻松地创建 Configuration 的多个子类，其中一个将被选中用于应用程序。

配置可能看起来像这样：

```py
class Configuration: 

    """ 

    Generic Configuration with a sample query. 

    """ 

    base = "https://forecast.weather.gov/shmrn.php" 

    query = {"mz": ["GMZ856"]} 

class Bahamas(Configuration): 

    """ 

    Weather forecast for Offshore including the Bahamas 

    """ 

    query = {"mz": ["AMZ117", "AMZ080"]} 

class Chesapeake(Configuration): 

    """ 

    Weather for Chesapeake Bay 

    """ 

    query = {"mz": ["ANZ532"]}
```

我们的应用程序必须从 settings 模块中可用的类中选择一个合适的类。我们可能使用操作系统环境变量或命令行选项来指定要使用的类名。我们的程序可以这样执行：

```py
(cookbook3) % python3 some_app.py -c settings.Chesapeake
```

这将定位 settings 模块中的 Chesapeake 类。然后，处理将基于该特定配置类中的详细信息。这个想法导致了 load_config_class() 函数的扩展。

为了选择一个可用的类，我们可以通过在命令行参数值中查找 "." 分隔符来分隔模块名称和类名称：

```py
import importlib 

def load_config_class(name: str) -> ConfigClass: 

    module_name, _, class_name = name.rpartition(".") 

    settings_module = importlib.import_module(module_name) 

    result: ConfigClass = vars(settings_module)[class_name] 

    return result
```

我们不是手动编译和执行模块，而是使用了更高层次的 importlib 模块。此模块包含实现导入语句语法的函数。请求的模块被导入，然后编译和执行，结果模块对象被分配给名为 result 的变量。

现在，我们可以如下使用这个函数：

```py
>>> configuration = load_config_class( 

... ’settings.Chesapeake’) 

>>> configuration.__doc__.strip() 

’Weather for Chesapeake Bay’ 

>>> configuration.query 

{’mz’: [’ANZ532’]} 

>>> configuration.base 

’https://forecast.weather.gov/shmrn.php’
```

我们已经在 settings 模块中找到了 Chesapeake 配置类，并从中提取了应用程序需要的各种设置。

## 13.4.5 参见

+   我们将在第七章和第八章中详细探讨类定义。

+   在本章中查看查找配置文件的配方，了解一种不使用类定义的替代方法。

# 13.5 设计用于组合的脚本

整体应用程序设计的一个重要部分是创建一个可以处理命令行参数和配置文件的脚本。此外，设计脚本以便它可以被测试以及与其他脚本组合成一个复合应用程序也非常重要。

想法是这样的：许多好主意通过一系列阶段演变而来。这样的演变可能是一条以下路径：

1.  这个想法最初是一系列单独的笔记本，用于处理更大任务的不同部分。

1.  在探索和实验的初期阶段之后，这变成了一项简单的重复性任务。与其手动打开和点击来运行笔记本，不如将其保存到脚本文件中，然后可以从命令行运行这些脚本文件。

1.  在将这一过程作为组织运营的常规部分的一段时间后，三部分脚本需要合并成一个单一的脚本。此时，需要进行重构。

在将多个脚本组合成一个应用程序并发现意外问题时，重构是最痛苦的时刻。这通常是因为当多个脚本集成时，全局变量将被共享。

在项目生命周期的早期，这是一个不那么痛苦的时刻。一旦创建了脚本，就应该努力设计脚本以便进行测试和组合到更大的应用程序中。

## 13.5.1 准备工作

在这个配方中，我们将探讨构成脚本良好设计的要素。特别是，我们想要确保在设计时考虑参数和配置文件。

目标是拥有以下结构：

+   对整个模块或脚本的一个文档字符串。

+   导入语句。这些有一个内部顺序。像 isort 和 ruff 这样的工具可以处理这个问题。

+   应用于脚本的类和函数定义。

+   一个函数，用于将配置文件选项和运行时参数收集到一个单一的对象中，该对象可以被其他类和函数使用。

+   一个执行有用工作的单个函数。这通常被称为 main()，但这个名字并没有什么神圣的。

+   一个只有当模块作为脚本运行时才会执行的小块代码，而模块被导入时则不会执行：

    ```py
     if __name__ == "__main__": 

        main()
    ```

## 13.5.2 如何实现...

以目标设计为目标，这里有一个方法：

1.  首先在文件顶部编写一个总结文档字符串。开始时先有一个大致的想法，然后再添加细节。以下是一个示例：

    ```py
     """ 

        Some Script. 

        What it does. How it works. 

        Who uses it. When do they use it. 

    """
    ```

1.  导入语句跟在文档字符串后面。事先预见所有导入并不总是可能的。随着模块的编写和修改，导入将被添加和删除。

1.  接下来是类和函数的定义。顺序对于在 def 或 class 语句中解析类型名称很重要。这意味着最基本类型定义必须首先进行。

    再次强调，在设计的第一波中，并不总是能够按照正确的顺序编写所有定义。重要的是保持它们在逻辑组织中的统一，并重新排列它们，以便阅读代码的人能够理解顺序。

1.  编写一个函数（例如 get_config()），用于获取所有配置参数。通常，这包括两个部分；有时需要将它们分解为两个单独的函数，因为每个部分可能相当复杂。

1.  然后是 main() 函数。它执行脚本的必要工作。当从笔记本演变而来时，这可以由单元格序列构建。

1.  在最后添加 Main-Import 切换代码块：

    ```py
    if __name__ == "__main__": 

        main()
    ```

生成的模块将作为脚本正常工作。它还可以更容易地进行测试，因为测试工具如 pytest 可以导入模块，而无需在尝试处理数据时对文件系统进行更改。它可以与其他脚本集成，以创建有用的复合应用程序。

## 13.5.3 它是如何工作的...

设计脚本时的核心考虑是区分模块的两个用途：

+   当从命令行运行时。在这种情况下，内置的全局变量 __name__ 将具有 "__main__" 的值。

+   当作为测试的一部分或作为更大、复合应用程序的一部分导入时。在这种情况下，__name__ 将具有模块的名称值。

当导入模块时，我们不希望它开始执行工作。在导入过程中，我们不希望模块打开文件、读取数据、进行计算或产生输出。所有这些工作都是在模块作为主程序运行时才能发生的事情。

笔记本或脚本语句的原始单元格现在成为 main() 函数的主体，因此脚本将正常工作。然而，它也将以可以测试的形式存在。它还可以集成到更大、更复杂的应用程序中。

## 13.5.4 更多内容...

在开始将代码转换为应用程序时，main() 函数通常相当长。有两种方法可以使处理过程更清晰：

+   显著的公告牌注释

+   通过重构创建多个较小的函数

我们可能从一个具有如下注释的脚本开始：

```py
# # Some complicated process 

# 

# Some additional markdown details. 

# In[12]: 

print("Some useful code here") 

# In[21]: 

print("More code here")
```

In[n]: 注释由 JupyterLab 提供，用于识别笔记本中的单元格。我们可以创建类似这样的公告牌注释：

```py
#################################### 

# Some complicated process         # 

#                                  # 

# Some additional markdown details.# 

#################################### 

print("Some useful code here") 

#################################### 

# Another step in the process      # 

#################################### 

print("More code here")
```

这并不理想。这是一个可接受的临时措施，但这些步骤应该是适当的函数，每个函数都有一个文档字符串和测试用例。在那些没有适当文档字符串且缺乏利用文档字符串的文档生成器的语言中，公告牌注释是传统的。

Python 有文档字符串和几个工具——如 Sphinx——可以从文档字符串创建文档。

## 13.5.5 相关内容

+   有关使用 argparse 从用户获取输入的背景信息，请参阅第六章中的使用 argparse 获取命令行输入配方。

+   在本章中，请参阅使用 TOML 配置文件、使用 Python 配置文件和使用类作为配置命名空间的相关配方。

+   本章后面的使用日志记录进行控制和审计输出配方探讨了日志记录。

+   在第十四章的将两个应用程序合并为一个配方中，我们将探讨遵循此设计模式的应用程序组合方法。

+   测试和集成的详细信息在其他章节中介绍。有关创建测试的详细信息，请参阅第十五章。有关组合应用程序的详细信息，请参阅第十四章。

# 13.6 使用日志记录进行控制和审计输出

当我们考虑一个应用程序时，我们可以将整体计算分解为三个不同的方面：

+   收集输入

+   将输入转换为输出的基本处理过程

+   生成输出

应用程序会产生几种不同的输出：

+   主要输出有助于用户做出决策或采取行动。在某些情况下，这可能是通过 Web 服务器下载的 JSON 格式文档。它可能是一组更复杂的文档，这些文档一起创建一个 PDF 文件。

+   确认程序完全且正确运行的控件信息。

+   可以用来跟踪持久数据库中状态变化历史的审计摘要。

+   任何指示应用程序为何无法工作的错误消息。

将所有这些不同的方面都合并到写入标准输出的 print()请求中并不是最佳选择。实际上，这可能会导致混淆，因为太多的不同输出可能会在单个流中交错。

操作系统为每个运行进程提供两个输出文件，标准输出和标准错误。这些在 Python 中通过 sys 模块的 sys.stdout 和 sys.stderr 名称可见。默认情况下，print()函数写入 sys.stdout 文件。我们可以更改目标文件，并将控制、审计和错误消息写入 sys.stderr。这是正确方向上的一个重要步骤。

Python 还提供了日志记录包，可以用来将辅助输出导向一个单独的文件（以及/或其他输出通道，如数据库）。它还可以用来格式化和过滤额外的输出。

在这个配方中，我们将探讨使用日志模块的好方法。

## 13.6.1 准备工作

满足各种输出需求的一种方法是创建多个记录器，每个记录器具有不同的意图。通常，记录器的命名与记录器关联的模块或类相关。我们也可以围绕审计或控制等整体目的命名记录器。

记录器的名称构成一个层次结构，由..分隔。根记录器是所有记录器的父级，名称为""。这表明我们可以有专注于特定类、模块或功能的记录器家族。

一组顶级记录器可以包括多个不同的关注领域，包括：

+   错误将为警告和错误的所有记录器添加前缀。

+   调试将为调试消息的所有记录器添加前缀。

+   审计将命名带有计数和总计的记录器，用于确认数据已被完全处理。

+   控制将命名提供有关应用程序运行时间、环境、配置文件和命令行参数值的记录器。

在大多数情况下，将错误和调试放在单个记录器中是有帮助的。在其他情况下——例如，一个网络服务器——请求错误响应日志应与任何内部错误或调试日志分开。

一个复杂的应用程序可能包含几个名为 audit.input 和 audit.output 的记录器，以显示消耗的数据计数和生成数据的计数。将这些记录器分开可以帮助关注数据提供者的问题。

严重程度级别为每个记录器提供了一种过滤机制。在日志包中定义的严重程度级别包括以下内容：

调试：这些消息通常不会显示，因为它们的目的是支持调试。上面，我们建议这是一种独特的调试类型。我们建议应用程序创建一个日志调试器，并使用普通的 INFO 消息进行调试条目。

信息：这些消息提供了有关正常、愉快的处理过程的信息。

警告：这些消息表明处理可能以某种方式受损。警告的最合理用例是当函数或类已被弃用时：它们仍然工作，但应该被替换。

错误：处理无效，输出不正确或不完整。在长时间运行的服务器的情况下，单个请求可能存在问题，但服务器整体可以继续运行。

严重：更严重的错误级别。通常，这用于长时间运行的服务器，其中服务器本身无法继续运行，即将崩溃。

每个记录器都有与严重程度级别相似的方法名称。我们使用 info()方法以 INFO 严重程度级别写入消息。

对于错误处理，严重级别大多是合适的。然而，调试日志记录器通常会生成大量需要单独保留的数据。此外，任何审计和控制输出似乎没有严重级别。严重级别似乎仅关注错误日志记录。因此，似乎更好的是具有如 debug.some_function 之类的名称的独立日志。然后我们可以通过启用或禁用这些日志记录器的输出以及配置严重级别为 INFO 来配置调试。

## 13.6.2 如何实现...

我们将在两个迷你食谱中查看类和函数中的日志记录。

### 在类中记录日志

1.  确保已导入日志记录模块。

1.  在 __init__()方法中，包括以下内容以创建错误和调试日志记录器：

    ```py
     self.err_logger = logging.getLogger( 

                f"error.{self.__class__.__name__}") 

            self.dbg_logger = logging.getLogger( 

                f"debug.{self.__class__.__name__}")
    ```

1.  在任何可能需要未来调试的方法中，使用调试日志记录器的函数将详细信息写入日志。虽然可以使用 f-string 来编写日志消息，但它们涉及将值插入文本的一些开销。当配置静默日志记录器的输出时，使用日志记录器的格式化选项和单独的参数值涉及的计算量略少：

    ```py
     self.dbg_logger.info( 

                "Some computation with %r", some_variable) 

            # Some complicated computation with some_variable 

            self.dbg_logger.info( 

                "Result details = %r", result)
    ```

1.  在几个关键位置，包含整体状态消息。这些通常在整体应用程序控制类中：

    ```py
     # Some complicated input processing and parsing 

            self.err_logger.info("Input processing completed.") 
    ```

### 在函数中记录日志

1.  确保已导入日志记录模块。

1.  对于更大和更复杂的函数，将日志记录器包含在函数内部是有意义的：

    ```py
    def large_and_complicated(some_parameter: Any) -> Any: 

        dbg_logger = logging.getLogger("debug.large_and_complicated") 

        dbg_logger.info("some_parameter= %r", some_parameter)
    ```

    由于日志记录器被缓存，第一次调用 get_logger()时才会涉及任何显著的开销。所有后续请求都是字典查找。

1.  对于较小的函数，全局定义日志记录器是有意义的。这有助于减少函数体内的视觉混乱：

    ```py
     very_small_dbg_logger = logging.getLogger("debug.very_small") 

    def very_small(some_parameter: Any) -> Any: 

        very_small_dbg_logger.info("some_parameter= %r", some_parameter)
    ```

注意，如果没有进一步的配置，将不会产生任何输出。这是因为每个日志记录器的默认严重级别将是 WARNING，这意味着处理程序将不会显示 INFO-或 DEBUG 级别的消息。

## 13.6.3 它是如何工作的...

引入日志记录到应用程序中有三个部分：

+   使用 getLogger()函数创建 Logger 对象。

+   使用类似于 info()或 error()的每个日志记录器的方法之一，将日志消息放置在重要的状态变化附近。

+   当应用程序运行时，整体配置日志系统。这对于查看日志记录器的输出至关重要。我们将在本食谱的 There’s more...部分中探讨这一点。

创建日志记录器可以通过多种方式完成。一种常见的方法是创建一个与模块同名的日志记录器：

```py
logger = logging.getLogger(__name__)
```

对于顶级主脚本，这将具有 __main__ 的名称。对于导入的模块，名称将与模块名称匹配。

在更复杂的应用程序中，可能有各种日志记录器服务于各种目的。在这些情况下，仅仅将日志记录器命名为模块名称可能不足以提供所需级别的灵活性。

还可以使用日志模块本身作为根记录器。这意味着一个模块可以使用 logging.info() 函数，例如。这不推荐，因为根记录器是匿名的，我们牺牲了使用记录器名称作为重要信息来源的可能性。

这个配方建议根据受众或用例命名记录器。最顶层的名称——例如，debug.——将区分日志的受众或目的。这可以使将给定父记录器下的所有记录器路由到特定处理器变得容易。

将日志消息与代码执行的重要状态变化相关联是有帮助的。

记录的第三个方面是配置记录器，以便它们将请求路由到适当的目的地。默认情况下，如果没有进行任何配置，记录器实例将默默地忽略正在创建的各种消息。

使用最小配置，我们可以在控制台上看到所有日志事件。这可以通过以下方式完成：

```py
if __name__ == "__main__": 

    logging.basicConfig(level=logging.INFO)
```

## 13.6.4 更多内容...

为了将不同的记录器路由到不同的目的地，我们需要更复杂的配置。通常，这超出了我们使用 basicConfig() 函数所能构建的内容。我们需要使用 logging.config 模块和 dictConfig() 函数。这可以提供完整的配置选项。使用此函数的最简单方法是使用 TOML 编写配置：

```py
version = 1 

[formatters.default] 

    style = "{" 

    format = "{levelname}:{name}:{message}" 

[formatters.timestamp] 

    style = "{" 

    format = "{asctime}//{levelname}//{name}//{message}" 

[handlers.console] 

    class = "logging.StreamHandler" 

    stream = "ext://sys.stderr" 

    formatter = "default" 

[handlers.file] 

    class = "logging.FileHandler" 

    filename = "data/write.log" 

    formatter = "timestamp" 

[loggers] 

    overview_stats.detail = {handlers = ["console"]} 

    overview_stats.write = {handlers = ["file", "console"] } 

    root = {level = "INFO"}
```

在这个 TOML 配置中，以下是一些关键点：

+   版本键的值必须是 1。这是必需的。

+   格式化程序表中的值定义了可用的日志格式。如果没有指定格式化程序，内置的格式化程序将只显示消息正文：

    +   示例中定义的默认格式化程序与 basicConfig() 函数创建的格式相匹配。这包括消息严重级别和记录器名称。

    +   示例中定义的新日期时间戳格式化程序是一个更复杂的格式，它包括记录的日期时间戳。为了使文件更容易解析，使用了 // 作为列分隔符。

+   处理器表定义了记录器可用的处理器：

    +   控制台处理器写入 sys.stderr 流，并使用默认格式化程序。以 "ext://..." 开头的文本是配置文件如何引用在 Python 环境中定义的对象的方式——在这种情况下，来自 sys 模块的 sys.stderr 值。

    +   文件处理器使用 FileHandler 类将内容写入文件。打开文件的默认模式是 a，这将追加到任何现有的日志文件。配置指定了用于文件的日期时间戳格式化程序。

+   记录器表为应用将使用的两个特定命名的记录器提供了配置。任何以 overview_stats.detail 开头的记录器名称将由控制台处理器处理。任何以 overview_stats.write 开头的记录器名称将同时发送到文件处理器和控制台处理器。

+   特殊的根键定义了顶级记录器。在代码中引用时，它有一个名为 ""（空字符串）的名称。在配置文件中，它有根键。

    在根记录器上设置严重性级别将设置用于显示或隐藏此记录器所有子记录器消息的级别。这将显示严重性为 INFO 或更高的消息，包括警告、错误和严重错误。

假设这个文件的内容存储在一个名为 config_toml 的变量中，包裹 main() 函数的配置将看起来像这样：

```py
if __name__ == "__main__": 

    logging.config.dictConfig( 

        tomllib.loads(config_toml)) 

    main() 

    logging.shutdown()
```

这将启动日志记录到一个已知的状态。它将处理应用程序。它将最终化所有的日志缓冲区，并正确关闭任何文件。

## 13.6.5 参考信息

+   在本章前面的设计用于组合的脚本配方中查看，以了解此应用的补充部分。

+   在本章中查看使用 TOML 作为配置文件的配方，了解更多关于解析 TOML 文档的信息。

# 加入我们的社区 Discord 空间

加入我们的 Python Discord 工作空间，讨论并了解更多关于这本书的信息：[`packt.link/dHrHU`](https://packt.link/dHrHU)

![PIC](img/file1.png)
