# 第十三章。应用程序集成

在本章中，我们将探讨以下示例：

+   查找配置文件

+   使用 YAML 进行配置文件

+   使用 Python 进行配置文件

+   使用类作为命名空间进行配置值

+   为构图设计脚本

+   使用日志进行控制和审计输出

+   将两个应用程序合并为一个

+   使用命令设计模式组合多个应用程序

+   在复合应用程序中管理参数和配置

+   包装和组合 CLI 应用程序

+   包装程序并检查输出

+   控制复杂的步骤序列

# 介绍

Python 的可扩展库为我们提供了丰富的访问多种计算资源的途径。这使得 Python 程序特别擅长于集成组件以创建复杂的复合处理。

在第五章中的*使用 argparse 获取命令行输入*，*使用 cmd 创建命令行应用程序*和*使用 OS 环境设置*的示例中，展示了创建顶层（主要）应用程序脚本的特定技术。在第九章中，我们研究了文件系统的输入和输出。在第十二章中，我们研究了创建服务器，这些服务器是从客户端接收请求的主要应用程序。

所有这些示例展示了 Python 中的应用程序编程的一些方面。还有一些其他有用的技术：

+   从文件中处理配置。在第五章的*使用 argparse 获取命令行输入*中，我们展示了解析命令行参数的技术。在*使用 OS 环境设置*的示例中，我们涉及了其他类型的配置细节。在本章中，我们将探讨处理配置文件的多种方法。有许多文件格式可用于存储长期配置信息：

+   INI 文件格式由`configparser`模块处理。

+   YAML 文件格式非常易于使用，但需要一个不是 Python 发行版的附加模块。我们将在*使用 YAML 进行配置文件*中进行讨论。

+   属性文件格式是 Java 编程的典型格式，可以在 Python 中处理而不需要编写太多代码。语法与 Python 脚本重叠。

+   对于 Python 脚本，具有赋值语句的文件看起来很像属性文件，并且非常容易使用`compile()`和`exec()`方法进行处理。我们将在*使用 Python 进行配置文件*中进行讨论。

+   Python 模块与类定义是一种使用 Python 语法的变体，但将设置隔离到单独的类中。这可以通过`import`语句进行处理。我们将在*使用类作为命名空间进行配置*中进行讨论。

+   在本章中，我们将探讨设计应用程序的方法，这些应用程序可以组合在一起创建更大、更复杂的复合应用程序。

+   我们将探讨由复合应用程序引起的复杂性以及需要集中一些功能（如命令行解析）的需求。

+   我们将扩展第六章和第七章中的一些概念，应用命令设计模式的想法到 Python 程序中。

# 查找配置文件

许多应用程序将具有配置选项的层次结构。可能会有内置于特定版本的默认值。可能会有服务器范围（或集群范围）的值。可能会有特定用户的值，或者甚至是特定程序调用的本地配置文件。

在许多情况下，这些配置参数将被写入文件中，以便更改。Linux 中的常见传统是将系统范围的配置放在 `/etc` 目录中。用户的个人更改将在其主目录中，通常命名为 `~username` 。

我们如何支持丰富的配置文件位置层次结构？

## 准备工作

示例将是一个为用户提供卡牌的网络服务。该服务在 第十二章 中的多个配方中都有展示，*网络服务* 。我们将忽略服务的一些细节，以便专注于从各种文件系统位置获取配置参数。

我们将遵循 **bash** shell 的设计模式，该模式在几个地方寻找配置文件：

1.  它始于 `/etc/profile` 文件。

1.  读取该文件后，它会按照以下顺序寻找其中一个文件：

1.  `~/.bash_profile` 。

1.  `~/.bash_login` 。

1.  `~/.profile` 。

在符合 POSIX 的操作系统中，shell 会将 `~` 扩展为已登录用户的主目录。这被定义为 `HOME` 环境变量的值。一般来说，Python 的 `pathlib` 模块可以自动处理这个问题。

有几种方法可以保存程序的配置参数：

+   使用类定义的优势在于极大的灵活性和相对简单的 Python 语法。它可以使用普通的继承来包含默认值。当参数有多个来源时，它的工作效果就不那么好，因为没有简单的方法来改变类定义。

+   对于映射参数，我们可以使用 `collections` 模块中的 `ChainMap` 集合来搜索多个不同来源的字典。

+   对于 `SimpleNamespace` 实例，`types` 模块提供了这个可变的类，可以从多个来源进行更新。

+   `argparse` 模块中的 `Namespace` 实例可能很方便，因为它反映了来自命令行的选项。

bash shell 的设计模式使用了两个单独的文件。当我们包含应用程序范围的默认值时，实际上有三个配置级别。这可以通过映射和 `collections` 模块中的 `ChainMap` 类来优雅地实现。

在后续的配方中，我们将探讨解析和处理配置文件的方法。在本配方中，我们将假设已定义了一个名为 `load_config_file()` 的函数，该函数将从文件内容中加载配置映射：

```py
    def load_config_file(config_file): 
        '''Loads a configuration mapping object with contents 
        of a given file. 

        :param config_file: File-like object that can be read. 
        :returns: mapping with configuration parameter values 
        ''' 
        # Details omitted. 

```

我们将分别研究实现此功能的方法。本章还涵盖了实现的变体，包括 *使用 YAML 进行配置文件* 和 *使用 Python 进行配置文件* 配方。

`pathlib` 模块可以帮助处理这个问题。该模块提供了 `Path` 类的定义，可以提供有关操作系统文件的复杂信息。有关更多信息，请参阅 第九章 中的 *使用 pathlib 处理文件名* 配方，*输入/输出、物理格式、逻辑布局* 。

### 为什么有这么多选择？

在讨论这种设计时，有时会出现一个侧边栏话题——为什么有这么多选择？为什么不明确指定两个地方？

答案取决于设计的上下文。当创建一个全新的应用程序时，选择可能被限制在两个选项之间。然而，当替换遗留应用程序时，通常会有一个新的位置，在某些方面比遗留位置更好，但仍然需要支持遗留位置。经过几次这样的演变变化后，通常会看到一些文件的替代位置。

此外，由于 Linux 发行版之间的差异，通常会看到对于一个发行版来说是典型的变化，但对于另一个发行版来说是非典型的变化。当处理 Windows 时，也会有独特于该平台的变体文件路径。

## 如何做...

1.  导入`Path`类和`ChainMap`类：

```py
        from pathlib import Path 
        from collections import ChainMap

```

1.  定义一个获取配置文件的整体函数：

```py
        def get_config():

```

1.  为各种位置创建路径。这些被称为纯路径，因为它们与文件系统没有关系。它们起初是*潜在*文件的名称：

```py
        system_path = Path('/etc/profile') 
        home_path = Path('~').expanduser() 
        local_paths = [home_path/'.bash_profile', 
            home_path/'.bash_login', 
            home_path/'.profile'] 

```

1.  定义应用程序的内置默认值：

```py
        configuration_items = [ 
            dict( 
                some_setting = 'Default Value', 
                another_setting = 'Another Default', 
                some_option = 'Built-In Choice', 
            ) 
        ] 

```

1.  每个单独的配置文件都是从键到值的映射。各种映射对象将形成一个列表；这将成为最终的`ChainMap`配置映射。我们将通过追加项目来组装映射列表，然后在加载文件后反转顺序。

1.  如果存在系统范围的配置文件，则加载该文件：

```py
        if system_path.exists(): 
            with system_path.open() as config_file: 
                configuration_items.append(config_file) 

```

1.  遍历其他位置，寻找要加载的文件。这会加载它找到的第一个文件：

```py
        for config_path in local_paths:
            if config_path.exists(): 
                with config_path.open() as config_file: 
                    configuration_items.append(config_file) 
                break

```

我们已经包含了**if-break**模式，以在找到第一个文件后停止。这修改了循环的默认语义，从 For All 变为 There Exists。有关更多信息，请参阅避免使用 break 语句配方中的潜在问题。

1.  反转列表并创建最终的`ChainMap`。需要反转列表，以便首先搜索本地文件，然后是系统设置，最后是应用程序默认设置：

```py
        configuration = ChainMap(*reversed(configuration_items)) 

```

1.  返回最终的配置映射：

```py
        return configuration

```

一旦我们构建了`configuration`对象，我们就可以像使用简单映射一样使用最终的配置。这个对象支持所有预期的字典操作。

## 工作原理...

任何面向对象语言的最优雅的特性之一是能够创建简单的对象集合。在这种情况下，对象是文件系统`Path`对象。

如在第九章的*使用 pathlib 处理文件名*配方中所述，`Path`对象有一个`resolve()`方法，可以返回从纯`Path`构建的具体`Path`。在这个配方中，我们使用了`exists()`方法来确定是否可以构建一个具体路径。当用于读取文件时，`open()`方法将解析纯`Path`并打开相关文件。

在第四章的*创建字典-插入和更新*配方中，我们看了一下使用字典的基础知识。在这里，我们将几个字典合并成一个链。当一个键在链中的第一个字典中找不到时，会检查链中后面的字典。这是一种为映射中的每个键提供默认值的方便方法。

这里有一个手动创建`ChainMap`的示例：

```py
 **>>> from collections import ChainMap 
>>> config = ChainMap( 
...     {'another_setting': 2}, 
...     {'some_setting': 1}, 
...     {'some_setting': 'Default Value', 
...      'another_setting': 'Another Default', 
...      'some_option': 'Built-In Choice'})** 

```

`config`对象是从三个单独的映射构建而成的。第一个可能是来自本地文件的细节，比如`~/.bash_login`。第二个可能是来自`/etc/profile`文件的系统范围设置。第三个包含应用程序范围的默认值。

当我们查询这个对象的值时，我们会看到以下内容：

```py
 **>>> config['another_setting'] 
2 
>>> config['some_setting'] 
1 
>>> config['some_option'] 
'Built-In Choice'** 

```

对于任何给定键的值都取自映射链中的第一个实例。这允许一种非常简单的方式来拥有覆盖系统范围值的本地值，覆盖内置默认值。

## 还有更多...

在第十一章的*Mocking External Resources*配方中，我们讨论了模拟外部资源的方法，以便我们可以编写一个单元测试，而不会意外删除文件。这个配方中的代码的测试需要通过模拟`Path`类来模拟文件系统资源。下面是单元测试的高级概述：

```py
    import unittest 
    from unittest.mock import * 

    class GIVEN_get_config_WHEN_load_THEN_overrides(unittest.TestCase): 
        def setUp(self): 

        def runTest(self): 

```

这为单元测试提供了一个样板结构。由于涉及的不同对象数量，模拟`Path`变得相当复杂。以下是发生的各种对象创建的总结：

1.  对`Path`类的调用创建一个`Path`对象。测试过程将创建两个`Path`对象，因此我们可以使用`side_effect`特性返回每个对象。我们需要确保基于要测试的代码的正确顺序返回这些值：

```py
        self.mock_path = Mock( 
            side_effect = [self.mock_system_path, self.mock_home_path] 
        ) 

```

1.  对于`system_path`的值，将调用`Path`对象的`exists()`方法；这将确定具体文件是否存在。然后将调用打开文件并读取内容：

```py
        self.mock_path = Mock( 
            side_effect = [self.mock_system_path, self.mock_home_path] 
        ) 

```

1.  对于`home_path`的值，将调用`expanduser()`方法将`~`更改为正确的主目录：

```py
        self.mock_home_path = Mock( 
            expanduser = Mock( 
                return_value = self.mock_expanded_home_path 
            ) 
        ) 

```

1.  然后，使用`/`运算符将扩展的`home_path`与三个备用目录一起创建：

```py
        self.mock_expanded_home_path = MagicMock( 
            __truediv__ = Mock( 
                side_effect = [self.not_exist, self.exist, self.exist] 
            ) 
        ) 

```

1.  为了进行单元测试，我们决定第一个要搜索的路径不存在。其他两个存在，但我们期望只有一个会被读取。第二个将被忽略：

+   对于不存在的模拟路径，我们可以使用这个：

```py
            self.not_exist = Mock( 
                exists = Mock(return_value=False) )
```

+   对于存在的模拟路径，我们将有更复杂的东西：

```py
            self.exist = Mock( exists = Mock(return_value=True), open = mock_open() ) 

```

我们还必须通过模拟模块中的`mock_open()`函数来处理文件的处理。这可以处理文件作为上下文管理器使用的各种细节，这变得相当复杂。`with`语句需要`__enter__()`和`__exit__()`方法，这由`mock_open()`处理。

我们必须按照相反的顺序组装这些模拟对象。这样可以确保每个变量在使用之前都已经创建好了。下面是整个`setUp()`方法，显示了对象的正确顺序：

```py
    def setUp(self): 
        self.mock_system_path = Mock( 
            exists = Mock(return_value=True), 
            open = mock_open() 
        ) 
        self.exist = Mock( 
            exists = Mock(return_value=True), 
            open = mock_open() 
        ) 
        self.not_exist = Mock( 
            exists = Mock(return_value=False) 
        ) 
        self.mock_expanded_home_path = MagicMock( 
            __truediv__ = Mock( 
                side_effect = [self.not_exist, self.exist, self.exist] 
            ) 
        ) 
        self.mock_home_path = Mock( 
            expanduser = Mock( 
                return_value = self.mock_expanded_home_path 
            ) 
        ) 
        self.mock_path = Mock( 
            side_effect = [self.mock_system_path, self.mock_home_path] 
        ) 

        self.mock_load = Mock( 
            side_effect = [{'some_setting': 1}, {'another_setting': 2}] 
        ) 

```

除了对`Path`操作的模拟之外，我们还添加了一个模拟模块。`mock_load`对象是未定义的`load_config_file()`的替身。我们希望将这个测试与路径处理分开，因此模拟对象使用`side_effect`属性返回两个单独的值，期望它将被调用两次。

以下是一些测试，将确认路径搜索是否按照广告进行。每个测试都从应用两个修补程序开始，以创建一个修改后的上下文，用于测试`get_config()`函数：

```py
    def runTest(self): 
        with patch('__main__.Path', self.mock_path), \ 
        patch('__main__.load_config_file', self.mock_load): 
            config = get_config() 
        # print(config) 
        self.assertEqual(2, config['another_setting']) 
        self.assertEqual(1, config['some_setting']) 
        self.assertEqual('Built-In Choice', config['some_option']) 

```

`patch()`的第一个用法是用`self.mock_path`替换`Path`类。`patch()`的第二个用法是用`self.mock_load`函数替换`load_config_file()`函数；这个函数将返回两个小的配置文档。在这两种情况下，被修补的上下文是当前模块，`__name__`的值为`"__main__"`。在单元测试位于一个单独的模块的情况下，将导入被测试的模块，并使用该模块的名称。

我们可以通过检查对`self.mock_load`的调用来验证`load_config_file()`是否被正确调用。在这种情况下，每个配置文件应该有一个调用：

```py
    self.mock_load.assert_has_calls( 
        [
            call(self.mock_system_path.open.return_value.__enter__.return_value), 
            call(self.exist.open.return_value.__enter__.return_value) 
        ] 
    ) 

```

我们确保首先检查`self.mock_system_path`文件。注意调用链——`Path()`返回一个`Path`对象。该对象的`open()`必须返回一个将被`load_config_file()`函数使用的值。上下文的`__enter__()`方法是`load_config_file()`函数将使用的对象。

我们确保另一个路径是`exists()`方法返回`True`的路径。这是构建文件名的检查：

```py
    self.mock_expanded_home_path.assert_has_calls( 
        [call.__truediv__('.bash_profile'), 
        call.__truediv__('.bash_login'), 
        call.__truediv__('.profile')] 
    ) 

```

`/`运算符由`__truediv__()`方法实现。每次调用都会构建一个单独的`Path`实例。我们可以确认，总体上，`Path`对象只使用了两次。一次是用于字面量`'/etc/profile'`，一次是用于字面量`'~'`：

```py
    self.mock_path.assert_has_calls( 
        [call('/etc/profile'), call('~')] 
    ) 

```

请注意，两个文件都对`exists()`方法返回`True`。然而，我们期望只有这两个文件中的一个会被检查。找到一个后，第二个文件将被忽略。以下是一个确认只有一个存在检查的测试：

```py
    self.exist.assert_has_calls( [call.exists()] ) 

```

为了完整起见，我们还检查了存在的文件是否会通过整个上下文管理序列：

```py
    self.exist.open.assert_has_calls( 
        [call(), call().__enter__(), call().__exit__(None, None, None)] 
    )

```

第一次调用是为了`self.exist`对象的`open()`方法。从这里返回的是一个上下文，将执行`__enter__()`方法以及`__exit__()`方法。在前面的代码中，我们检查了从`__enter__()`返回的值是否被读取以获取配置文件内容。

## 另请参阅

+   在*使用 YAML 进行配置文件*和*使用 Python 进行配置文件*的方法中，我们将研究实现`load_config_file()`函数的方法。

+   在第十一章的*模拟外部资源*方法中，我们研究了如何测试与外部资源交互的函数，比如这个函数。

# 使用 YAML 进行配置文件

Python 提供了多种打包应用程序输入和配置文件的方式。我们将研究使用 YAML 符号写文件，因为它简洁而简单。

我们如何用 YAML 符号表示配置细节？

## 准备工作

Python 没有内置的 YAML 解析器。我们需要使用`pip`软件包管理系统将`pyyaml`项目添加到我们的库中。安装的步骤如下：

```py
 **MacBookPro-SLott:pyweb slott$ pip3.5 install pyyaml** 

 **Collecting pyyaml** 

 **Downloading PyYAML-3.11.zip (371kB)** 

 **100% |████████████████████████████████| 378kB 2.5MB/s** 

 **Installing collected packages: pyyaml 
  Running setup.py install for pyyaml ... done 
Successfully installed pyyaml-3.11** 

```

YAML 语法的优雅之处在于简单的缩进用于显示文档的结构。以下是我们可能在 YAML 中编码的一些设置的示例：

```py
 **query: 
  mz: 
    - ANZ532 
    - AMZ117 
    - AMZ080 
url: 
  scheme: http 
  netloc: forecast.weather.gov 
  path: /shmrn.php 
description: > 
  Weather forecast for Offshore including the Bahamas** 

```

这个文档可以被看作是一系列相关的 URL 的规范，它们都类似于[`forecast.weather.gov/shmrn.php?mz=ANZ532`](http://forecast.weather.gov/shmrn.php?mz=ANZ532)。文档包含了有关从方案、网络位置、基本路径和几个查询字符串构建 URL 的信息。`yaml.load()`函数可以加载这个 YAML 文档；它将创建以下 Python 结构：

```py
 **{'description': 'Weather forecast for Offshore including the Bahamas\n', 
 'query': {'mz': ['ANZ532', 'AMZ117', 'AMZ080']}, 
 'url': {'netloc': 'forecast.weather.gov', 
         'path': 'shmrn.php', 
         'scheme': 'http'}}** 

```

这种*字典-字典*结构可以被应用程序用来定制其操作。在这种情况下，它指定了一个要查询的 URL 序列，以组装更大的天气简报。

我们经常使用*查找配置文件*的方法来检查各种位置以找到给定的配置文件。这种灵活性通常对于创建一个可以轻松在各种平台上使用的应用程序至关重要。

在这个方法中，我们将构建前一个示例中缺失的部分，即`load_config_file()`函数。以下是需要填写的模板：

```py
    def load_config_file(config_file) -> dict: 
        '''Loads a configuration mapping object with contents 
        of a given file. 

        :param config_file: File-like object that can be read. 
        :returns: mapping with configuration parameter values 
        ''' 
        # Details omitted. 

```

## 如何做…

1.  导入`yaml`模块：

```py
        import yaml 

```

1.  使用`yaml.load()`函数加载 YAML 语法文档：

```py
        def load_config_file(config_file) -> dict: 
            '''Loads a configuration mapping object with contents 
            of a given file. 

            :param config_file: File-like object that can be read. 
            :returns: mapping with configuration parameter values 
            ''' 
            document = yaml.load(config_file) 
        return document 

```

## 工作原理…

YAML 语法规则在[`yaml.org`](http://yaml.org)中定义。YAML 的理念是提供具有更灵活、人性化语法的类似 JSON 的数据结构。JSON 是更一般的 YAML 语法的特例。

这里的权衡是，JSON 中的一些空格和换行不重要——有可见的标点来显示文档的结构。在一些 YAML 变体中，换行和缩进决定了文档的结构；使用空格意味着 YAML 文档中的换行很重要。

JSON 语法中可用的主要数据结构如下：

+   **序列**：`[item, item, ...]`

+   **映射**：`{key: value, key: value, ...}`

+   **标量**：

+   字符串：`"value"`

+   数字：`3.1415926`

+   字面值：`true`，`false`和`null`

JSON 语法是 YAML 的一种风格；它被称为流风格。在这种风格中，文档结构由显式指示符标记。语法要求使用`{...}`和`[...]`来显示结构。

YAML 提供的另一种选择是块样式。文档结构由换行和缩进定义。此外，长字符串标量值可以使用普通、带引号和折叠样式的语法。以下是替代 YAML 语法的工作方式：

+   **块序列**：我们用-在序列的每一行前面加上。这看起来像一个项目列表，很容易阅读。这是一个例子：

```py
 **zoneid: 
        - ANZ532 
        - AMZ117 
        - AMZ080** 

```

加载后，这将在 Python 中创建一个带有字符串列表的字典：`{zoneid: ['ANZ532', 'AMZ117', 'AMZ080']}`。

+   **块映射**：我们可以使用简单的`key: value`语法将键与简单的标量关联起来。我们可以单独在一行上使用`key:`；值缩进在下面的行上。这是一个例子：

```py
 **url: 
        scheme: http 
        netloc: marine.weather.gov** 

```

这将创建一个嵌套字典，在 Python 中看起来像这样：`{'url': {'scheme': 'http', 'netloc': 'marine.weather.gov'}}`。

我们还可以使用显式的键和值标记`?`和`:`。当键特别长或对象更复杂时，这可能有所帮助：

```py
 **? scheme 
: http 
? netloc 
: marine.weather.gov** 

```

YAML 的一些更高级功能将利用键和值之间的显式分隔：

+   对于短字符串标量值，我们可以保持它们的普通状态，YAML 规则将简单地使用所有字符，并去除前导和尾随空格。这些示例都使用了这种假设的字符串值。

+   引号可以用于字符串，就像在 JSON 中一样。

+   对于较长的字符串，YAML 引入了`|`前缀；在此之后的行将保留所有的间距和换行符。

它还引入了`>`前缀，将单词保留为一长串文本——任何换行符都被视为单个空格字符。这在连续文本中很常见。

在这两种情况下，缩进决定了文档的哪部分是文本的一部分。

+   在某些情况下，值可能是模棱两可的。例如，美国的邮政编码都是数字——`22102`。尽管 YAML 规则会将其解释为数字，但应该理解为字符串。当然，引号可能会有所帮助。为了更明确，可以在值的前面使用`!!str`本地标签来强制指定数据类型。例如，`!!str 22102`确保数字将被视为字符串对象。

## 还有更多...

YAML 中有一些 JSON 中没有的其他功能：

+   注释以`#`开头，一直延续到行尾。它们几乎可以放在任何地方。JSON 不允许注释。

+   文档开始，由新文档开头的`---`行指示。这允许一个 YAML 文件包含许多单独的对象。JSON 限制每个文件只能有一个文档。一个文档-每个文件的替代方案是一个更复杂的解析算法。YAML 提供了显式的文档分隔符和一个非常简单的解析接口。

+   具有两个单独文档的 YAML 文件：

```py
 **>>> import yaml 
      >>> yaml_text = ''' 
      ... --- 
      ... id: 1 
      ... text: "Some Words." 
      ... --- 
      ... id: 2 
      ... text: "Different Words." 
      ... ''' 
      >>> document_iterator = iter(yaml.load_all(yaml_text)) 
      >>> document_1 = next(document_iterator) 
      >>> document_1['id'] 
      1 
      >>> document_2 = next(document_iterator) 
      >>> document_2['text'] 
      'Different Words.'** 

```

+   `yaml_text`值包含两个 YAML 文档，每个文档都以`---`开头。`load_all()`函数是一个迭代器，一次加载一个文档。应用程序必须迭代处理流中的每个文档的结果。

+   文档结束。`...`行是文档的结束。

+   映射的复杂键；JSON 映射键仅限于可用的标量类型——字符串、数字、`true`、`false`和`null`。YAML 允许将映射键设置得更复杂。

+   重要的是，Python 要求映射键的哈希表是不可变的对象。这意味着复杂的键必须转换为不可变的 Python 对象，通常是`tuple`。为了创建一个特定于 Python 的对象，我们需要使用更复杂的本地标签。以下是一个例子：

```py
 **>>> yaml.load(''' 
      ... ? !!python/tuple ["a", "b"] 
      ... : "value" 
      ... ''') 
      {('a', 'b'): 'value'}** 

```

+   这个例子使用`?`和`:`来标记映射的键和值。我们这样做是因为键是一个复杂对象。键`value`使用了一个本地标签`!!python/tuple`，来创建一个元组，而不是默认的`list`。键的文本使用了一个流类型的 YAML 值`["a", "b"]`。

+   JSON 没有集合的规定。YAML 允许我们使用`!!set`标签来创建一个集合，而不是一个简单的序列。集合中的项目以`?`前缀标识，因为它们被认为是一个映射的键，没有值。

+   请注意，`!!set`标签与集合中的值处于相同的缩进级别。它在`data_values`的字典键内缩进：

```py
 **>>> import yaml 
      >>> yaml_text = ''' 
      ... document: 
      ...     id: 3 
      ...     data_values: 
      ...       !!set 
      ...       ? some 
      ...       ? more 
      ...       ? words 
      ... ''' 
      >>> some_document = yaml.load(yaml_text) 
      >>> some_document['document']['id'] 
      3 
      >>> some_document['document']['data_values'] == {'some', 'more', 'words'} 
      True** 

```

+   `!!set`本地标签修改以下序列，使其成为一个`set`对象，而不是默认的列表对象。结果集等于预期的 Python 集合对象`{'some', 'more', 'words'}`。

+   Python 可变对象规则将被应用于集合的内容。无法构建一个包含`list`对象的集合，因为列表实例没有哈希值。必须使用`!!python/tuple`本地标签来构建一个元组集合。

+   我们还可以创建一个 Python 的两元组列表序列，它实现了有序映射。`yaml`模块不会为我们直接创建`OrderedDict`：

```py
 **>>> import yaml 
      >>> yaml_text = ''' 
      ... !!omap 
      ... - key1: string value 
      ... - numerator: 355 
      ... - denominator: 113 
      ... ''' 
      >>> yaml.load(yaml_text) 
      [('key1', 'string value'), ('numerator', 355), ('denominator', 113)]** 

```

+   请注意，很难在不指定大量细节的情况下，进一步创建`OrderedDict`。以下是创建`OrderedDict`实例的 YAML。

```py
        !!python/object/apply:collections.OrderedDict 
        args: 
            -   !!omap 
                -   key1: string value 
                -   numerator: 355 
                -   denominator: 113 

```

+   `args`关键字是必需的，以支持`!!python/object/apply`标签。只有一个位置参数，它是一个从键和值序列构建的 YAML`!!omap`。

+   几乎任何类的 Python 对象都可以使用 YAML 本地标签来构建。任何具有简单`__init__()`方法的类都可以从 YAML 序列化中构建。

这是一个简单的类定义：

```py
        class Card: 
            def __init__(self, rank, suit): 
                self.rank = rank 
                self.suit = suit 
            def __repr__(self): 
                return "{rank} {suit}".format_map(vars(self)) 

```

我们定义了一个具有两个位置属性的类。以下是该对象的 YAML 描述：

```py
        !!python/object/apply:__main__.Card 
        kwds: 
            rank: 7 
            suit: ♣

```

我们使用`kwds`键为`Card`构造函数提供了两个基于关键字的参数值。Unicode`♣`字符很好用，因为 YAML 文件是使用 UTF-8 编码的文本。

+   除了以`!!`开头的本地标签之外，YAML 还支持使用`tag:`方案的 URI 标签。这允许使用基于 URI 的类型规范，这些规范在全局范围内是唯一的。这可以使 YAML 文档在各种上下文中更容易处理。

标签包括权限名称、日期和具体细节，以`/`分隔的路径形式。例如，一个标签可能看起来像这样——`!<tag:www.someapp.com,2016:rules/rule1>`。

## 另请参阅

+   查看*查找配置文件*配方，了解如何在多个文件系统位置搜索配置文件。我们可以很容易地将应用程序默认值、系统范围的设置和个人设置构建到单独的文件中，并由应用程序组合。

![](img/614271.jpg)

# 使用 Python 进行配置文件

Python 提供了多种打包应用程序输入和配置文件的方式。我们将看一下使用 Python 符号写文件，因为它既优雅又简单。

许多包使用单独的模块中的赋值语句来提供配置参数。特别是 Flask 项目可以这样做。我们在*使用 Flask 框架进行 RESTful API*配方中看到了 Flask，以及第十二章中的一些相关配方，*网络服务*。

我们如何用 Python 模块符号表示配置细节？

## 准备工作

Python 赋值语句符号特别优雅。它非常简单，易于阅读，而且非常灵活。如果我们使用赋值语句，可以从一个单独的模块中导入配置细节。这个模块可以有一个名字，比如`settings.py`，以显示它专注于配置参数。

因为 Python 将每个导入的模块视为全局**单例**对象，所以我们可以让应用程序的几个部分都使用`import settings`语句来获得当前全局应用程序配置参数的一致视图。

然而，在某些情况下，我们可能希望选择几个备选的设置文件之一。在这种情况下，我们希望使用比`import`语句更灵活的技术来加载文件。

我们希望能够在文本文件中提供以下形式的定义：

```py
    '''Weather forecast for Offshore including the Bahamas 
    ''' 
    query = {'mz': ['ANZ532', 'AMZ117', 'AMZ080']} 
    url = { 
      'scheme': 'http', 
      'netloc': 'forecast.weather.gov', 
      'path': '/shmrn.php' 
    } 

```

这是 Python 语法。参数包括两个变量，`query`和`url`。`query`变量的值是一个带有单个键`mz`和一系列值的字典。

这可以看作是一系列相关的 URL 的规范，它们都类似于[`forecast.weather.gov/shmrn.php?mz=ANZ532`](http://forecast.weather.gov/shmrn.php?mz=ANZ532)。

我们经常使用*查找配置文件*配方来检查各种位置以找到给定的配置文件。这种灵活性通常对于创建一个在各种平台上易于使用的应用程序是至关重要的。

在这个配方中，我们将构建前面示例中缺失的部分，即`load_config_file()`函数。这是需要填写的模板：

```py
    def load_config_file(config_file) -> dict: 
        '''Loads a configuration mapping object with contents 
        of a given file. 

        :param config_file: File-like object that can be read. 
        :returns: mapping with configuration parameter values 
        ''' 
        # Details omitted. 

```

## 如何做...

这段代码替换了`load_config_file()`函数中的`# Details omitted.`行：

1.  使用内置的`compile()`函数编译配置文件中的代码。这个函数需要源文本以及从中读取文本的文件名。文件名对于创建有用和正确的回溯消息是必不可少的：

```py
        code = compile(config_file.read(), config_file.name, 'exec') 

```

1.  在罕见的情况下，代码不是来自文件时，一般的做法是提供一个名字，比如`<string>`，而不是文件名。

1.  执行`compile()`函数创建的代码对象。这需要两个上下文。全局上下文提供了任何先前导入的模块，以及`__builtins__`模块。本地上下文是新变量将被创建的地方：

```py
        locals = {} 
        exec(code, {'__builtins__':__builtins__}, locals) 
        return locals 

```

1.  当代码在脚本文件的顶层执行时——通常在`if __name__ == "__main__"`条件内部执行——它在全局和本地是相同的上下文中执行。当代码在函数、方法或类定义内部执行时，该上下文的本地变量与全局变量是分开的。

1.  通过创建一个单独的`locals`对象，我们确保导入的语句不会对任何其他全局变量进行意外更改。

## 它是如何工作的...

Python 语言的细节，语法和语义都体现在`compile()`和`exec()`函数中。当我们启动一个脚本时，过程基本上是这样的：

1.  阅读文本。使用`compile()`函数编译它以创建一个代码对象。

1.  使用`exec()`函数来执行该代码对象。

`__pycache__`目录保存代码对象，并保存未更改的文本文件的重新编译。这对处理没有实质影响。

`exec()`函数反映了 Python 处理全局和本地变量的方式。这个函数提供了两个命名空间。这些对于通过`globals()`和`locals()`函数运行的脚本是可见的。

我们提供了两个不同的字典：

+   全局对象的字典。这些变量可以通过`global`语句访问。最常见的用法是提供对导入模块的访问，这些模块始终是全局的。例如，通常会提供`__builtins__`模块。在某些情况下，可能需要添加其他模块。

+   本地提供的字典会被每个赋值语句更新。这个本地字典允许我们捕获在`settings`模块内创建的变量。

## 还有更多...

这个配方构建了一个配置文件，可以完全是一系列`name = value`赋值。这个语句直接由 Python 赋值语句语法支持。

此外，Python 编程的全部范围都是可用的。必须做出许多工程上的权衡。

配置文件中可以使用任何语句。但这可能会导致复杂性。如果处理变得太复杂，文件就不再是配置文件，而成为应用程序的一部分。非常复杂的功能应该通过修改应用程序编程来完成，而不是通过配置设置进行操作。由于 Python 应用程序包括源代码，这相对容易做到。

除了简单的赋值语句之外，使用`if`语句处理替代方案也是有道理的。文件可能提供一个特定运行时环境的独特特性部分。例如，`platform`包可以用于隔离特性。

可能包括类似于这样的内容：

```py
    import platform 
    if platform.system() == 'Windows': 
        tmp = Path(r"C:\TEMP") 
    else: 
        tmp = Path("/tmp") 

```

为了使这个工作，全局变量应该包括`platform`和`Path`。这是对`__builtins__`的合理扩展。

简单地进行一些处理也是有道理的，以便更容易地组织一些相关的设置。例如，一个应用程序可能有一些相关的文件。编写这样的配置文件可能会有所帮助：

```py
    base = Path('/var/app/') 
    log = base/'log' 
    out = base/'out' 

```

`log`和`out`的值被应用程序使用。`base`的值仅用于确保其他两个位置放置在同一个目录中。

这导致了对之前显示的`load_config_file()`函数的以下变化。这个版本包括一些额外的模块和全局类：

```py
    from pathlib import Path  
    import platform 
    def load_config_file_path(config_file) -> dict: 
        code = compile(config_file.read(), config_file.name, 'exec') 
        globals = {'__builtins__': __builtins__, 
            'Path': Path, 'platform': platform} 
        locals = {} 
        exec(code, globals, locals) 
        return locals 

```

包括`Path`和`platform`意味着可以编写配置文件而无需`import`语句的开销。这可以使设置更容易准备和维护。

## 参见

+   参见*查找配置文件*配方，了解如何搜索多个文件系统位置以查找配置文件。

# 使用类作为命名空间进行配置

Python 提供了各种打包应用程序输入和配置文件的方法。我们将研究使用 Python 符号写文件，因为它既优雅又简单。

许多项目允许使用类定义来提供配置参数。类层次结构的使用意味着可以使用继承技术来简化参数的组织。特别是 Flask 包可以做到这一点。我们在配方*使用 Flask 框架进行 RESTful API*以及一些相关的配方中看到了 Flask。

我们如何在 Python 类符号中表示配置细节？

## 准备工作

Python 用于定义类属性的符号特别优雅。它非常简单，易于阅读，并且相当灵活。我们可以很容易地定义一个复杂的配置语言，允许某人快速可靠地更改 Python 应用程序的配置参数。

我们可以基于类定义构建这种语言。这允许我们在单个模块中打包多个配置选项。应用程序可以加载模块并从模块中选择相关的类定义。

我们希望能够提供以下类似的定义：

```py
    class Configuration: 
        """ 
        Weather forecast for Offshore including the Bahamas 
        """ 
        query = {'mz': ['ANZ532', 'AMZ117', 'AMZ080']} 
        url = { 
          'scheme': 'http', 
          'netloc': 'forecast.weather.gov', 
          'path': '/shmrn.php' 
        } 

```

我们可以在单个`settings`模块中创建这个`Configuration`类。要使用配置，主应用程序将执行以下操作：

```py
    from settings import Configuration 

```

这使用一个固定的文件和一个固定的类名。尽管看起来缺乏灵活性，但这通常比其他选择更有用。我们有两种额外的方法来支持复杂的配置文件：

+   我们可以使用`PYTHONPATH`环境变量列出配置模块的多个位置。

+   使用多重继承和混合来将默认值、系统范围的设置和本地设置合并到配置类定义中

这些技术可能有所帮助，因为配置文件位置只需遵循 Python 查找模块的规则。我们不需要实现自己的搜索配置文件的方法。

在这个示例中，我们将构建前一个示例中缺失的部分，即`load_config_file()`函数。以下是需要填写的模板：

```py
    def load_config_file(config_file) -> dict: 
        '''Loads a configuration mapping object with contents 
        of a given file. 

        :param config_file: File-like object that can be read. 
        :returns: mapping with configuration parameter values 
        ''' 
        # Details omitted. 

```

## 操作步骤...

这段代码替换了`load_config_file()`函数中的`# Details omitted.`行：

1.  使用内置的`compile()`函数编译给定文件中的代码。这个函数需要源文本以及从中读取文本的文件名。文件名对于创建有用和正确的回溯消息是必不可少的：

```py
        code = compile(config_file.read(), config_file.name, 'exec') 

```

1.  执行`compile()`方法创建的代码对象。我们需要提供两个上下文。全局上下文可以提供`__builtins__`模块，以及`Path`类和`platform`模块。本地上下文是新变量将被创建的地方：

```py
        globals = {'__builtins__':__builtins__, 
                   'Path': Path, 
                   'platform': platform} 
        locals = {} 
        exec(code, globals, locals) 
        return locals['Configuration'] 

```

1.  这只会从执行模块设置的本地变量中返回定义的`Configuration`类。任何其他变量都将被忽略。

## 工作原理...

Python 语言的细节——语法和语义——体现在`compile()`和`exec()`函数中。`exec()`函数反映了 Python 处理全局和本地变量的方式。这个函数提供了两个命名空间。全局`namespace`实例包括`__builtins__`以及可能在文件中使用的类和模块。

本地变量命名空间将有新创建的类。这个命名空间有一个`__dict__`属性，使其可以通过字典方法访问。因此，我们可以通过名称提取类。该函数返回类对象，供整个应用程序使用。

我们可以将任何类型的对象放入类的属性中。我们的示例展示了映射对象。在创建类级别的属性时，没有任何限制。

我们可以在`class`语句内进行复杂的计算。我们可以利用这一点创建从其他属性派生的属性。我们可以执行任何类型的语句，包括`if`语句和`for`语句来创建属性值。

## 还有更多...

使用类定义意味着我们可以利用继承来组织配置值。我们可以轻松创建`Configuration`的多个子类，其中一个将被选中用于应用程序。配置可能如下所示：

```py
    class Configuration: 
        """ 
        Generic Configuration 
        """ 
        url = { 
          'scheme': 'http', 
          'netloc': 'forecast.weather.gov', 
          'path': '/shmrn.php' 
        } 

    class Bahamas(Configuration): 
        """ 
        Weather forecast for Offshore including the Bahamas 
        """ 
        query = {'mz': ['AMZ117', 'AMZ080']} 

    class Cheaspeake(Configuration): 
        """ 
        Weather for Cheaspeake Bay 
        """ 
        query = {'mz': ['ANZ532']} 

```

这意味着我们的应用程序必须从`settings`模块中的可用类中选择一个合适的类。我们可以使用操作系统环境变量或命令行选项来指定要使用的类名。这个想法是我们的程序是这样执行的：

```py
 **python3 some_app.py -c settings.Chesapeake** 

```

这将在`settings`模块中找到`Chesapeake`类。然后处理将基于该特定配置类中的细节。这个想法导致了`load_config_file()`函数的扩展。

为了选择可用类中的一个，我们将提供一个额外的参数，其中包含类名：

```py
    import importlib 
    def load_config_module(name): 
        module_name, _, class_name = name.rpartition('.') 
        settings_module = importlib.import_module(module_name) 
        return vars(settings_module)[class_name] 

```

我们没有手动编译和执行模块，而是使用了更高级别的`importlib`模块。该模块实现了`import`语句的语义。请求的模块被导入；编译和执行——并将生成的模块对象分配给变量名`settings_module`。

然后我们可以查看模块的变量并挑选出所请求的类。`vars()`内置函数将从模块、类甚至本地变量中提取内部字典。

现在我们可以按照以下方式使用这个函数：

```py
 **>>> configuration = load_config_module('settings.Chesapeake') 
>>> configuration.__doc__.strip() 
'Weather for Cheaspeake Bay' 
>>> configuration.query 
{'mz': ['ANZ532']} 
>>> configuration.url['netloc'] 
'forecast.weather.gov'** 

```

我们在`settings`模块中找到了`Chesapeake`配置类。

### 配置表示

使用类似这样的类的一个后果是，类的默认显示并不是太有信息。当我们尝试打印配置时，它看起来像这样：

```py
 **>>> print(configuration) 
<class 'settings.Chesapeake'>** 

```

这几乎没有用。它提供了一小部分信息，但这远远不够用于调试。

我们可以使用`vars()`函数查看更多细节。但是，这显示的是本地变量，而不是继承的变量：

```py
 **>>> pprint(vars(configuration)) 
mappingproxy({'__doc__': '\\n    Weather for Cheaspeake Bay\\n    ', 
              '__module__': 'settings', 
              'query': {'mz': ['ANZ532']}})** 

```

这样做更好，但仍然不完整。

为了查看所有设置，我们需要更复杂的东西。有趣的是，我们不能简单地为这个类定义`__repr__()`。在类中定义的方法将适用于该类的实例，而不是类本身。

我们创建的每个类对象都是内置`type`类的实例。我们可以使用元类调整`type`类的行为方式，并实现一个稍微更好的`__repr__()`方法，该方法查找所有父类的属性。

我们将使用一个`__repr__`扩展内置类型，该类型在显示工作配置时做得更好一些：

```py
    class ConfigMetaclass(type): 
        def __repr__(self): 
            name = (super().__name__ + '('  
                + ', '.join(b.__name__ for b in super().__bases__) + ')') 
            base_values = {n:v 
                for base in reversed(super().__mro__) 
                    for n, v in vars(base).items() 
                        if not n.startswith('_')} 
            values_text = ['    {0} = {1!r}'.format(name, value)  
                for name, value in base_values.items()] 
            return '\n'.join(["class {}:".format(name)] + values_text) 

```

类名可以从超类`type`中的`__name__`属性中获得。基类的名称也包括在内，以显示此配置类的继承层次结构。

`base_values`是从所有基类的属性构建的。每个类按照**方法解析顺序**（**MRO**）的相反顺序进行检查。按照反向 MRO 加载所有属性值意味着首先加载所有默认值，然后用子类值覆盖。

不包含`_`前缀的名称被包括在内。具有`_`前缀的名称被悄悄地忽略。

生成的值用于创建类似类定义的文本表示。这不是原始类源代码；这是原始类定义的净效果。

这是使用这个元类的`Configuration`类层次结构。基类`Configuration`包含元类，并提供默认定义。子类使用唯一于特定环境或上下文的值扩展这些定义：

```py
    class Configuration(metaclass=ConfigMetaclass): 
        unchanged = 'default' 
        override = 'default' 
        feature_override = 'default' 
        feature = 'default' 

    class Customized(Configuration): 
        override = 'customized' 
        feature_override = 'customized' 

```

我们可以利用 Python 的多重继承的所有功能来构建`Configuration`类定义。这可以提供将单独特性的细节合并到单个配置对象中的能力。

## 另见

+   我们将在第六章，*类和对象的基础*和第七章，*更高级的类设计*中查看类定义

# 为组合设计脚本

许多大型应用实际上是由多个较小的应用程序组合而成的。在企业术语中，它们通常被称为包含单独命令行应用程序程序的应用系统。

一些大型复杂的应用程序包括许多命令。例如，Git 应用程序有许多单独的命令，如`git pull`，`git commit`和`git push`。这些也可以看作是整个 Git 应用程序系统的一部分的单独应用程序。

一个应用程序可能起初是一组单独的 Python 脚本文件。在其演变过程中的某个时刻，有必要重构脚本，将特性组合起来，并从较旧的不连贯脚本创建新的组合脚本。另一条路径也是可能的，一个大型应用程序可能被分解和重构为一个新的组织。

我们如何设计一个脚本，以便将来的组合和重构尽可能简单？

## 准备就绪

我们需要区分 Python 脚本的几个设计特性：

+   我们已经看到了收集输入的几个方面：

+   从命令行界面和环境变量获取高度动态的输入。请参阅第五章中的*使用 argparse 获取命令行输入*。

+   从文件中获取更改配置选项变得很慢。请参阅*查找配置文件*，*使用 YAML 进行配置文件*和*使用 Python 进行配置文件*。

+   阅读任何输入文件，请参阅第九章中的*使用 CSV 模块读取分隔文件*、*使用正则表达式读取复杂格式*、*读取 JSON 文档*、*读取 XML 文档*和*读取 HTML 文档*的示例，*输入/输出、物理格式和逻辑布局*。

+   产生输出有几个方面：

+   创建日志并提供其他支持审计、控制和监控的功能。我们将在*使用日志进行控制和审计输出*的示例中看到其中一些。

+   创建应用程序的主要输出。这可能会被打印或写入输出文件，使用与解析输入相同的库模块。

+   应用程序的真正工作。这些是基本功能，与各种输入解析和输出格式考虑分离。该算法专门使用 Python 数据结构。

这种*关注点的分离*表明，无论多么简单的应用程序都应设计为几个单独的函数。这些函数应组合成一个完整的脚本。这样我们就可以将输入和输出与核心处理分开。处理是我们经常想要重用的部分。输入和输出格式应该是可以轻松更改的东西。

作为一个具体的例子，我们将看一个创建骰子掷出序列的简单应用程序。每个序列都将遵循 craps 游戏的规则。以下是规则：

1.  两个骰子的第一次掷出是*come out*掷：

1.  两点、三点或十二点的掷出是立即输。该序列有一个单一值，例如，`[(1, 1)]`。

1.  七点或十一点的掷出是立即赢。这个序列也有一个单一值，例如，`[(3, 4)]`。

1.  任何其他数字都会确定一个点。序列从点值开始，直到掷出七点或点值：

1.  最终的七点是输，例如，`[(3, 1), (3, 2), (1, 1), (5, 6), (4, 3)]`。

1.  原始点值的最终匹配是赢。至少会有两次掷骰子。游戏的长度没有上限，例如，`[(3, 1), (3, 2), (1, 1), (5, 6), (1, 3)]`。

输出是具有不同结构的项目序列。有些会是短列表。有些会是长列表。这是使用 YAML 格式文件的理想场所。

这个输出可以由两个输入控制——要创建多少样本，以及是否要给随机数生成器设定种子。出于测试目的，固定种子可能有所帮助。

## 如何做...

1.  将所有输出显示设计为两个广泛的领域：

1.  不进行处理但显示结果对象的函数（或类）。

1.  日志可能用于调试、监控、审计或其他控制。这是一个横切关注点，将嵌入到应用程序的其余部分中。

在这个例子中，有两个输出——序列的序列和一些额外信息，以确认处理是否正常工作。每个掷出的点数计数是确定模拟骰子是否公平的方便方法。

掷出的序列需要写入文件。这表明`write_rolls()`函数被给定一个迭代器作为参数。这是一个迭代并以 YAML 格式将值转储到文件的函数：

```py
        def write_rolls(output_path, roll_iterator): 
            face_count = Counter() 
            with output_path.open('w') as output_file: 
                for roll in roll_iterator: 
                    output_file.write( 
                        yaml.dump( 
                            roll, 
                            default_flow_style=True, 
                            explicit_start=True)) 
                    for dice in roll: 
                        face_count[sum(dice)] += 1 
            return face_count 

```

监控和控制输出应显示用于控制处理的输入参数。它还应提供显示骰子公平的计数：

```py
        def summarize(configuration, counts): 
            print(configuration) 
            print(counts) 

```

1.  设计（或重构）应用程序的基本处理，使其看起来像一个单一函数：

1.  所有输入都是参数。

1.  所有输出都由`return`或`yield`产生。使用`return`创建单一结果。使用`yield`生成多个结果的序列迭代。

在这个例子中，我们可以很容易地将核心功能设为一个发出值序列迭代的函数。输出函数可以使用这个迭代器：

```py
        def roll_iter(total_games, seed=None): 
            random.seed(seed) 
            for i in range(total_games): 
                sequence = craps_game() 
                yield sequence 

```

此函数依赖于`craps_game()`函数生成请求的样本数量。每个样本都是一个完整的游戏，显示所有的骰子掷出。此函数提供`face_count`计数器给这个低级函数以累积一些总数以确认一切是否正常工作。

`craps_game()`函数实现了 crap 游戏规则以发出一个或多个掷骰子的单个序列。这包括了单个游戏中的所有掷骰子。我们稍后会看一下这个`craps_game()`函数。

1.  将所有的输入收集重构为一个函数（或类），它收集各种输入源。这可以包括环境变量、命令行参数和配置文件。它还可以包括多个输入文件的名称：

```py
        def get_options(argv): 
            parser = argparse.ArgumentParser() 
            parser.add_argument('-s', '--samples', type=int) 
            parser.add_argument('-o', '--output') 
            options = parser.parse_args(argv) 

            options.output_path = Path(options.output) 

            if "RANDOMSEED" in os.environ: 
                seed_text = os.environ["RANDOMSEED"] 
                try: 
                    options.seed = int(seed_text) 
                except ValueError: 
                    sys.exit("RANDOMSEED={0!r} invalid".format(seed_text)) 
            else: 
                options.seed = None 
            return options 

```

这个函数收集命令行参数。它还检查`os.environ`环境变量的集合。

参数解析器将处理解析`--samples`和`--output`选项的细节。我们可以利用`argparse`的其他功能来更好地验证参数值。

`output_path`的值是从`--output`选项的值创建的。类似地，`RANDOMSEED`环境变量的值经过验证并放入`options`命名空间。`options`对象的使用将所有不同的参数放在一个地方。

1.  编写最终的`main()`函数，它包含了前面的三个元素，以创建最终的整体脚本：

```py
        def main(): 
            options = get_options(sys.argv[1:]) 
            face_count = write_rolls(options.output_path, roll_iter(options.samples, options.seed)) 
            summarize(options, face_count) 

```

这将应用程序的各个方面汇集在一起。它解析命令行和环境选项。它创建一个控制总计计数器。

`roll_iter()`函数是核心处理。它接受各种选项，并发出一系列掷骰子。

`roll_iter()`方法的主要输出由`write_rolls()`收集，并写入给定的输出路径。控制输出由一个单独的函数写入，这样我们可以在不影响主要输出的情况下更改摘要。

## 它的工作原理...

输出如下：

```py
 **slott$ python3 ch13_r05.py --samples 10 --output=x.yaml** 

 **Namespace(output='x.yaml', output_path=PosixPath('x.yaml'), samples=10, seed=None)** 

 **Counter({5: 7, 6: 7, 7: 7, 8: 5, 4: 4, 9: 4, 11: 3, 10: 1, 12: 1})** 

 **slott$ more x.yaml** 

 **--- [[5, 4], [3, 4]]** 

 **--- [[3, 5], [1, 3], [1, 4], [5, 3]]** 

 **--- [[3, 2], [2, 4], [6, 5], [1, 6]]** 

 **--- [[2, 4], [3, 6], [5, 2]]** 

 **--- [[1, 6]]** 

 **--- [[1, 3], [4, 1], [1, 4], [5, 6], [6, 5], [1, 5], [2, 6], [3, 4]]** 

 **--- [[3, 3], [3, 4]]** 

 **--- [[3, 5], [4, 1], [4, 2], [3, 1], [1, 4], [2, 3], [2, 6]]** 

 **--- [[2, 2], [1, 5], [5, 5], [1, 5], [6, 6], [4, 3]]** 

 **--- [[4, 5], [6, 3]]** 

```

命令行请求了十个样本，并指定了一个名为`x.yaml`的输出文件。控制输出是选项的简单转储。它显示了参数的值以及在`options`对象中设置的附加值。

控制输出包括来自十个样本的计数。这提供了一些信心，例如六、七和八的值更常出现。它显示了像三和十二这样的值出现得更少。

这里的核心前提是关注点的分离。处理有三个明显的方面：

+   **输入**：来自命令行和环境变量的参数由一个名为`get_options()`的单个函数收集。这个函数可以从各种来源获取输入，包括配置文件。

+   **输出**：主要输出由`write_rolls()`函数处理。其他控制输出由在`Counter`对象中累积总数，然后在最后转储此输出来处理。

+   **处理**：应用程序的基本处理被分解到`roll_iter()`函数中。这个函数可以在各种上下文中重复使用。

这种设计的目标是将`roll_iter()`函数与周围应用程序的细节分离开来。另一个应用程序可能有不同的命令行参数，或不同的输出格式，但可以重用基本算法。

例如，可能有第二个应用程序对掷骰子的序列进行一些统计分析。这可能包括掷骰子的次数，以及最终的输赢结果。我们可以假设这两个应用程序是`generator.py`（如前所示）和`overview_stats.py`。

在使用这两个应用程序创建骰子并总结它们之后，用户可能会确定将骰子创建和统计概述合并到一个单一应用程序中会更有利。因为每个应用程序的各个方面都被分开，所以重新排列功能并创建一个新应用程序变得相对容易。现在我们可以构建一个新应用程序，它将从以下两个导入开始：

```py
    from generator import roll_iter, craps_rules 
    from stats_overview import summarize 

```

这个新应用程序可以在不对其他两个应用程序进行任何更改的情况下构建。这样一来，引入新功能不会影响原始应用程序。

更重要的是，新应用程序没有涉及任何代码的复制或粘贴。新应用程序导入工作软件——对一个应用程序进行修复的更改也将修复其他应用程序中的潜在错误。

### 提示

通过复制和粘贴进行重用会产生技术债务。避免复制和粘贴代码。

当我们尝试从一个应用程序复制代码来创建一个新应用程序时，我们会创建一个混乱的情况。对一个副本进行的任何更改不会奇迹般地修复另一个副本中的潜在错误。当对一个副本进行更改，而另一个副本没有保持更新时，这是一种*代码腐烂*。

## 还有更多...

在前一节中，我们跳过了`craps_rules()`函数的细节。这个函数创建了一个包含单个*Craps*游戏的骰子投掷序列。它可以从单次投掷到无限长度的序列。大约 98%的游戏将是十三次或更少的骰子投掷。

规则取决于两个骰子的总和。捕获的数据包括两个骰子的分开面。为了支持这些细节，有一个`namedtuple`实例，具有这两个相关属性：

```py
    Roll = namedtuple('Roll', ('faces', 'total')) 
    def roll(n=2): 
        faces = list(random.randint(1, 6) for _ in range(n)) 
        total = sum(faces) 
        return Roll(faces, total) 

```

这个`roll()`函数创建一个`namedtuple`实例，其中包含显示骰子面和骰子总和的序列。`craps_game()`函数将生成足够的规则来返回一个完整的游戏：

```py
    def craps_game(): 
        come_out = roll() 
        if come_out.total in [2, 3, 12]: 
            return [come_out.faces] 
        elif come_out.total in [7, 11]: 
            return [come_out.faces] 
        elif come_out.total in [4, 5, 6, 8, 9, 10]: 
            sequence = [come_out.faces] 
            next = roll() 
            while next.total not in [7, come_out.total]: 
                sequence.append(next.faces) 
                next = roll() 
            sequence.append(next.faces) 
            return sequence 
        else: 
            raise Exception("Horrifying Logic Bug") 

```

`craps_game()`函数实现了 Craps 的规则。如果第一次投掷是两、三或十二，序列只有一个值，游戏就输了。如果第一次投掷是七或十一，序列也只有一个值，游戏就赢了。其余的值建立了一个点。投掷序列从点值开始。序列一直持续，直到被七或点值结束。

### 设计为类层次结构

`roll_iter()`，`roll()`和`craps_game()`方法之间的密切关系表明，将这些函数封装到一个单一的类定义中可能更好。下面是一个将所有这些功能捆绑在一起的类：

```py
    class CrapsSimulator: 
        def __init__(self, seed=None): 
            self.rng = random.Random(seed) 
            self.faces = None 
            self.total = None 

        def roll(self, n=2): 
            self.faces = list(self.rng.randint(1, 6) for _ in range(n)) 
            self.total = sum(self._faces) 

        def craps_game(sel): 
            self.roll() 
            if self.total in [2, 3, 12]: 
                return [self.faces] 
            elif self.total in [7, 11]: 
                return [self.faces] 
            elif self.total in [4, 5, 6, 8, 9, 10]: 
                point, sequence = self.total, [self.faces] 
                self.roll() 
                while self.total not in [7, point]: 
                    sequence.append(self.faces) 
                    self.roll() 
                sequence.append(self.faces) 
                return sequence 
            else: 
                raise Exception("Horrifying Logic Bug") 

        def roll_iter(total_games): 
            for i in range(total_games): 
                sequence = self.craps_game() 
                yield sequence 

```

这个类包括模拟器的初始化，包括自己的随机数生成器。它将使用给定的种子值，或者内部算法将选择种子值。`roll()`方法将设置`self.total`和`self.faces`实例变量。

`craps_game()`生成一个游戏的骰子序列。它使用`roll()`方法和两个实例变量`self.total`和`self.faces`来跟踪骰子的状态。

`roll_iter()`方法生成游戏序列。请注意，此方法的签名与前面的`roll_iter()`函数并不完全相同。这个类将随机数种子的生成与游戏创建算法分开。

重写`main()`以使用`CrapsSimulator`类留给读者作为练习。由于方法名称与原始函数名称相似，重构不应该太复杂。

## 另请参阅

+   在第五章的*用户输入和输出*中查看*使用 argparse 获取命令行输入*的方法，了解使用`argparse`从用户那里获取输入的背景知识

+   查看*查找配置文件*的方法，以便追踪配置文件

+   *使用日志记录控制和审计输出*的方法查看日志记录

+   在*将两个应用程序合并为一个*的配方中，我们将看看如何合并遵循这种设计模式的应用程序

# 使用日志记录控制和审计输出

在*为组合设计脚本*的配方中，我们考察了应用程序的三个方面：

+   收集输入

+   产生输出

+   连接输入和输出的基本处理

应用程序产生几种不同类型的输出：

+   帮助用户做出决策或采取行动的主要输出

+   确认程序完全正确工作的控制信息

+   用于跟踪持久数据库中状态变化历史的审计摘要

+   指示应用程序为什么不工作的任何错误消息

将所有这些不同方面都归并到写入标准输出的`print()`请求中并不是最佳选择。实际上，这可能会导致混乱，因为太多不同的输出被合并到一个流中。

操作系统提供了两个输出文件，标准输出和标准错误。在 Python 中，可以通过`sys`模块的`sys.stdout`和`sys.stderr`来看到这些文件。默认情况下，`print()`方法会写入`sys.stdout`文件。我们可以改变这一点，将控制、审计和错误消息写入`sys.stderr`。这是朝着正确方向迈出的重要一步。

Python 提供了`logging`包，可以用来将辅助输出定向到单独的文件。它还可以用来格式化和过滤附加输出。

我们如何正确使用日志记录？

## 准备工作

在*为组合设计脚本*的配方中，我们看了一个生成带有模拟原始输出的 YAML 文件的应用程序。在这个配方中，我们将看一个应用程序，它消耗这些原始数据并生成一些统计摘要。我们将称这个应用程序为`overview_stats.py`。

遵循分离输入、输出和处理的设计模式，我们将有一个类似这样的`main()`应用程序：

```py
    def main(): 
        options = get_options(sys.argv[1:]) 
        if options.output is not None: 
            report_path = Path(options.output) 
            with report_path.open('w') as result_file: 
                process_all_files(result_file, options.file) 
        else: 
            process_all_files(sys.stdout, options.file) 

```

这个函数将从各种来源获取选项。如果命名了输出文件，它将使用`with`语句上下文创建输出文件。然后这个函数将处理所有命令行参数文件作为输入，从中收集统计信息。

如果没有提供输出文件名，这个函数将写入`sys.stdout`文件。这将显示可以使用操作系统 shell 的`>`运算符重定向的输出，以创建一个文件。

`main()`函数依赖于`process_all_files()`函数。`process_all_files()`函数将遍历每个参数文件，并从该文件中收集统计信息。这个函数看起来是这样的：

```py
    def process_all_files(result_file, file_names): 
        for source_path in (Path(n) for n in file_names): 
            with source_path.open() as source_file: 
                game_iter = yaml.load_all(source_file) 
                statistics = gather_stats(game_iter) 
                result_file.write( 
                    yaml.dump(dict(statistics), explicit_start=True) 
                ) 

```

`process_all_files()`函数将`gather_stats()`应用于`file_names`可迭代中的每个文件。然后将生成的集合写入给定的`result_file`。

### 注意

这里显示的函数将处理和输出混合在一起，这种设计并不理想。我们将在*将两个应用程序合并为一个*的配方中解决这个设计缺陷。

基本处理在`gather_stats()`函数中。给定一个文件路径，它将读取并总结该文件中的游戏。然后产生的摘要对象可以作为整体显示的一部分，或者在这种情况下，附加到一系列 YAML 格式的摘要中：

```py
    def gather_stats(game_iter): 
        counts = Counter() 
        for game in game_iter: 
            if len(game) == 1 and sum(game[0]) in (2, 3, 12): 
                outcome = "loss" 
            elif len(game) == 1 and sum(game[0]) in (7, 11): 
                outcome = "win" 
            elif len(game) > 1 and sum(game[-1]) == 7: 
                outcome = "loss" 
            elif len(game) > 1 and sum(game[0]) == sum(game[-1]): 
                outcome = "win" 
            else: 
                raise Exception("Wait, What?") 
            event = (outcome, len(game)) 
            counts[event] += 1 
        return counts 

```

这个函数确定了四种游戏终止规则中的哪一种适用于骰子掷出的顺序。它首先打开给定的源文件，然后使用`load_all()`函数遍历所有的 YAML 文档。每个文档都是一个单独的游戏，表示为一系列骰子对。

这个函数使用第一个（和最后一个）掷骰子来确定游戏的整体结果。有四条规则，应该列举出所有可能的逻辑事件组合。如果在我们的推理中出现错误，异常将被引发以警示我们某种特殊情况没有符合设计的方式。

游戏被简化为一个具有结果和长度的单个事件。这些被累积到一个`Counter`对象中。游戏的结果和长度是我们正在计算的两个值。这些是更复杂或复杂的统计分析的替代品。

我们已经仔细地将几乎所有与文件相关的考虑从这个函数中分离出来。`gather_stats()`函数将使用任何可迭代的游戏数据源。

这是应用程序的输出。它不是很漂亮；这是一个 YAML 文档，可用于进一步处理：

```py
 **slott$ python3 ch13_r06.py x.yaml** 

 **---** 

 **? !!python/tuple [loss, 2]** 

 **: 2** 

 **? !!python/tuple [loss, 3]** 

 **: 1** 

 **? !!python/tuple [loss, 4]** 

 **: 1** 

 **? !!python/tuple [loss, 6]** 

 **: 1** 

 **? !!python/tuple [loss, 8]** 

 **: 1** 

 **? !!python/tuple [win, 1]** 

 **: 1** 

 **? !!python/tuple [win, 2]** 

 **: 1** 

 **? !!python/tuple [win, 4]** 

 **: 1** 

 **? !!python/tuple [win, 7]** 

 **: 1** 

```

我们需要将日志记录功能插入所有这些函数中，以显示正在读取的文件以及处理文件时的任何错误或问题。

此外，我们将创建两个日志。一个将有详细信息，另一个将有已创建文件的最小摘要。第一个日志可以进入`sys.stderr`，当程序运行时将在控制台显示。另一个日志将附加到长期的`log`文件中，以覆盖应用程序的所有用途。

满足不同需求的一种方法是创建两个记录器，每个记录器具有不同的意图。这两个记录器还将具有截然不同的配置。另一种方法是创建一个单一的记录器，并使用`Filter`对象来区分每个记录器的内容。我们将专注于创建单独的记录器，因为这样更容易开发和更容易进行单元测试。

每个记录器都有各种方法，反映了消息的严重性。`logging`包中定义的严重级别包括以下内容：

+   **DEBUG**：通常不显示这些消息，因为它们的目的是支持调试。

+   **INFO**：这些消息提供有关正常、顺利处理的信息。

+   **WARNING**：这些消息表明处理可能以某种方式受到影响。警告的最明智用例是当函数或类已被弃用时：它们可以工作，但应该被替换。这些消息通常会显示。

+   **ERROR**：处理无效，输出不正确或不完整。在长时间运行的服务器的情况下，单个请求可能会出现问题，但作为一个整体，服务器可以继续运行。

+   **CRITICAL**：更严重的错误级别。通常，这是由长时间运行的服务器使用的，其中服务器本身无法继续运行，并且即将崩溃。

方法名称与严重级别类似。我们使用`logging.info()`来写入 INFO 级别的消息。

## 如何做到...

1.  我们将首先将基本的日志记录功能实现到现有的函数中。这意味着我们需要`logging`模块：

```py
        import logging

```

应用程序的其余部分将使用许多其他软件包：

```py
        import argparse 
        import sys 
        from pathlib import Path 
        from collections import Counter 
        import yaml 

```

1.  我们将创建两个作为模块全局变量的记录器对象。创建函数可以放在创建全局变量的脚本的任何位置。一个位置是在`import`语句之后尽早放置这些内容。另一个常见的选择是在最后附近，但在任何`__name__ == "__main__"`脚本处理之外。这些变量必须始终被创建，即使模块作为库导入。

记录器具有分层名称。我们将使用应用程序名称和内容后缀来命名记录器。`overview_stats.detail`记录器将具有处理详细信息。`overview_stats.write`记录器将标识已读取和已写入的文件；这与审计日志的概念相对应，因为文件写入跟踪输出文件集合中的状态更改：

```py
        detail_log = logging.getLogger("overview_stats.detail") 
        write_log = logging.getLogger("overview_stats.write") 

```

我们现在不需要配置这些记录器。如果我们什么都不做，这两个记录器对象将默默地接受单独的日志条目，但不会进一步处理数据。

1.  我们将重写`main()`函数以总结处理的两个方面。这将使用`write_log`记录器对象来显示何时创建新文件：

```py
        def main(): 
            options = get_options(sys.argv[1:]) 
            if options.output is not None: 
                report_path = Path(options.output) 
                with report_path.open('w') as result_file: 
                    process_all_files(result_file, options.file) 
                write_log.info("wrote {}".format(report_path)) 
            else: 
                process_all_files(sys.stdout, options.file) 

```

我们添加了`write_log.info("wrote {}".format(result_path))`一行，将信息消息放入日志中已写入的文件。

1.  我们将重写`process_all_files()`函数，以在读取文件时提供注释：

```py
        def process_all_files(result_file, file_names): 
            for source_path in (Path(n) for n in file_names): 
                detail_log.info("read {}".format(source_path)) 
                with source_path.open() as source_file: 
                    game_iter = yaml.load_all(source_file) 
                    statistics = gather_stats(game_iter) 
                result_file.write( 
                    yaml.dump(dict(statistics), explicit_start=True) 
                ) 

```

我们添加了`detail_log.info("read {}".format(source_path))`行，以在每次读取文件时将信息消息放入详细日志中。

1.  `gather_stats()`函数可以添加一行日志来跟踪正常操作。此外，我们还为逻辑错误添加了一个日志条目：

```py
        def gather_stats(game_iter): 
            counts = Counter() 
            for game in game_iter: 
                if len(game) == 1 and sum(game[0]) in (2, 3, 12): 
                    outcome = "loss" 
                elif len(game) == 1 and sum(game[0]) in (7, 11): 
                    outcome = "win" 
                elif len(game) > 1 and sum(game[-1]) == 7: 
                    outcome = "loss" 
                elif len(game) > 1 and sum(game[0]) == sum(game[-1]): 
                    outcome = "win" 
                else: 
                    detail_log.error("problem with {}".format(game)) 
                    raise Exception("Wait, What?") 
                event = (outcome, len(game)) 
                detail_log.debug("game {} -> event {}".format(game, event)) 
                counts[event] += 1 
            return counts 

```

`detail_log`记录器用于收集调试信息。如果将整体日志级别设置为包括调试消息，我们将看到此额外输出。

1.  `get_options()`函数还将写入一个调试行。这可以通过将选项显示在日志中来帮助诊断问题：

```py
        def get_options(argv): 
            parser = argparse.ArgumentParser() 
            parser.add_argument('file', nargs='*') 
            parser.add_argument('-o', '--output') 
            options = parser.parse_args(argv) 
            detail_log.debug("options: {}".format(options)) 
            return options 

```

1.  我们可以添加一个简单的配置来查看日志条目。这是作为第一步来简单确认有两个记录器，并且它们被正确使用：

```py
        if __name__ == "__main__": 
            logging.basicConfig(stream=sys.stderr, level=logging.INFO) 
            main() 

```

此日志配置构建了默认处理程序对象。此对象仅在给定流上打印所有日志消息。此处理程序分配给根记录器；它将应用于此记录器的所有子记录器。因此，前面代码中创建的两个记录器将发送到同一个流。

以下是运行此脚本的示例：

```py
 **slott$ python3 ch13_r06a.py -o sum.yaml x.yaml 
      INFO:overview_stats.detail:read x.yaml 
      INFO:overview_stats.write:wrote sum.yaml** 

```

日志中有两行。两者的严重性都是 INFO。第一行来自`overview_stats.detail`记录器。第二行来自`overview_stats.write`记录器。默认配置将所有记录器发送到`sys.stdout`。

1.  为了将不同的记录器路由到不同的目的地，我们需要比`basicConfig()`函数更复杂的配置。我们将使用`logging.config`模块。`dictConfig()`方法可以提供完整的配置选项。这样做的最简单方法是将配置写入 YAML，然后使用`yaml.load()`函数将其转换为内部的`dict`对象：

```py
            import logging.config 
            config_yaml = ''' 
        version: 1 
        formatters: 
            default: 
                style: "{" 
                format: "{levelname}:{name}:{message}" 
                #   Example: INFO:overview_stats.detail:read x.yaml 
            timestamp: 
                style: "{" 
                format: "{asctime}//{levelname}//{name}//{message}" 

        handlers: 
            console: 
                class: logging.StreamHandler 
                stream: ext://sys.stderr 
                formatter: default 
            file: 
                class: logging.FileHandler 
                filename: write.log 
                formatter: timestamp 

        loggers: 
            overview_stats.detail: 
                handlers: 
                -   console 
            overview_stats.write: 
                handlers: 
                -   file 
                -   console 
        root: 
            level: INFO 
        ''' 

```

YAML 文档被包含在一个三重撇号字符串中。这使我们能够写入尽可能多的文本。我们使用 YAML 表示法在大块文本中定义了五件事：

+   `version`键的值必须为 1。

+   `formatters`键的值定义了日志格式。如果未指定此项，那么默认格式只显示消息正文，不包括级别或记录器信息：

+   此处定义的`default`格式化程序与`basicConfig()`函数创建的格式相同。

+   此处定义的`timestamp`格式化程序是一个更复杂的格式，包括记录的日期时间戳。为了使文件更容易解析，使用了`//`作为列分隔符。

+   `handlers`键定义了两个记录器的两个处理程序。`console`处理程序写入流`sys.stderr`。我们指定了此处理程序将使用的格式化程序。此定义与`basicConfig()`函数创建的配置相对应。

`file`处理程序写入文件。打开文件的默认模式是`a`，这将追加到文件，文件大小没有上限。还有其他处理程序可以在多个文件之间轮换，每个文件都有限制的大小。我们提供了一个显式的文件名，以及一个将在文件中放入比在控制台上显示的更多细节的格式化程序：

+   `loggers`键为应用程序将创建的两个记录器提供了配置。任何以`overview_stats.detail`开头的记录器名称将仅由控制台处理程序处理。任何以`overview_stats.write`开头的记录器名称将同时发送到文件处理程序和控制台处理程序。

+   `root`键定义了顶级记录器。它的名称是`''`（空字符串），以防我们需要在代码中引用它。设置根记录器的级别将为该记录器的所有子记录器设置级别。

1.  使用配置来包装`main()`函数，如下所示：

```py
        logging.config.dictConfig(yaml.load(config_yaml)) 
        main()
        logging.shutdown()
```

1.  这将以已知状态开始记录。它将处理应用程序。它将完成所有日志缓冲区的处理，并正确关闭任何文件。

## 工作原理...

将日志引入应用程序有三个部分：

+   创建记录器对象

+   在重要状态更改附近放置日志请求

+   作为一个整体配置日志系统

创建记录器可以通过多种方式完成。此外，也可以忽略。作为默认值，我们可以使用`logging`模块本身作为记录器。例如，如果我们使用`logging.info()`方法，这将隐式地使用根记录器。

更常见的方法是创建一个与模块名称相同的记录器：

```py
    logger = logging.getLogger(__name__) 

```

对于顶级主脚本，这将具有名称`"__main__"`。对于导入的模块，名称将与模块名称匹配。

在更复杂的应用程序中，将有各种记录器用于各种目的。在这些情况下，仅仅将记录器命名为模块可能无法提供所需的灵活性。

有两个概念可以用来为记录器分配名称。通常最好选择其中一个，并在整个大型应用程序中坚持使用它：

+   遵循包和模块的层次结构。这意味着特定于类的记录器可能具有类似`package.module.class`的名称。同一模块中的其他类将共享一个共同的父记录器名称。然后可以设置整个包的日志级别，特定模块之一的日志级别，或者只是其中一个类的日志级别。

+   根据受众或用例遵循层次结构。顶级名称将区分日志的受众或目的。我们可能会有名称为`event`，`audit`和可能`debug`的顶级记录器。这样，所有审计记录器的名称都将以`"audit."`开头。这样可以很容易地将给定父级下的所有记录器路由到特定处理程序。

在这个示例中，我们使用了第一种命名风格。记录器名称与软件架构相对应。将日志请求放置在所有重要状态更改附近应该相对简单。日志中应包含各种有趣的状态更改：

+   对持久资源的任何更改都可能是包含`INFO`级别消息的好地方。任何 OS 更改（通常是文件系统）都有可能进行日志记录。同样地，数据库更新和应该更改 Web 服务状态的请求也应该被记录。

+   每当出现无法进行持久状态更改的问题时，应该有一个`ERROR`消息。任何 OS 级别的异常在被捕获和处理时都可以被记录。

+   在长时间的复杂计算中，可能有助于在特别重要的赋值语句之后记录`DEBUG`消息。在某些情况下，这是一个提示，表明长时间的计算可能需要分解成函数，以便可以单独测试它们。

+   对内部应用程序资源的任何更改都应该产生一个`DEBUG`消息，以便可以通过日志跟踪对象状态更改。

+   当应用程序进入错误状态时。这通常是由于异常。在某些情况下，将使用`assert`语句来检测程序的状态，并在出现问题时引发异常。一些异常以`EXCEPTION`级别记录。然而，一些异常只需要`DEBUG`级别的消息，因为异常被屏蔽或转换。一些异常可能以`ERROR`或`CRITICAL`级别记录。

日志的第三个方面是配置记录器，以便将请求路由到适当的目的地。默认情况下，如果没有任何配置，记录器将悄悄地创建日志事件，但不会显示它们。

通过最小配置，我们可以在控制台上看到所有日志事件。这可以通过`basicConfig()`方法完成，并且涵盖了大量简单的用例，而无需任何真正的麻烦。我们可以使用文件名而不是流来提供命名文件。也许最重要的功能是通过`basicConfig()`方法在根记录器上设置日志级别，从而提供一种简单的启用调试的方法。

配方中的示例配置使用了两个常见的处理程序——`StreamHandler`和`FileHandler`类。还有十几个以上的处理程序，每个都具有独特的功能，用于收集和发布日志消息。

## 还有更多...

+   请参阅*为组合设计脚本*配方，了解这个应用程序的补充部分。

# 将两个应用程序合并为一个

在*为组合设计脚本*配方中，我们研究了一个简单的应用程序，通过模拟过程创建了一组统计数据。在*使用日志记录进行控制和审计输出*配方中，我们研究了一个总结统计数据的应用程序。在这个配方中，我们将结合这两个单独的应用程序，创建一个单一的复合应用程序，既创建又总结统计数据。

有几种常见的方法可以将这两个应用程序组合起来：

+   一个 shell 脚本可以运行模拟器，然后运行分析器

+   一个 Python 程序可以代替 shell 脚本，并使用`runpy`模块来运行每个程序

+   我们可以从每个应用程序的基本特性构建一个复合应用程序

在*为组合设计脚本*配方中，我们研究了应用程序的三个方面：

+   输入收集

+   产生输出

+   连接输入和输出的基本处理

在这个配方中，我们研究了一种设计模式，可以将几个 Python 语言组件组合成一个更大的应用程序。

我们如何将应用程序组合成一个复合应用程序？

## 准备工作

在*为组合设计脚本*和*使用日志记录进行控制和审计输出*的配方中，我们遵循了一个设计模式，将输入收集、基本处理和输出产生分开。这个设计模式的目标是将有趣的部分聚集在一起，以便将它们组合和重新组合成更高级的结构。

请注意，这两个应用程序之间存在微小的不匹配。我们可以借用数据库工程（也是电气工程）的一个短语，称之为阻抗不匹配。在电气工程中，这是一个电路设计问题，通常通过使用一个叫做变压器的设备来解决。这可以用来匹配电路组件之间的阻抗。

在数据库工程中，当数据库具有规范化的扁平数据，但编程语言使用丰富结构的复杂对象时，这种问题会出现。对于 SQL 数据库，这是一个常见问题，使用**SQLAlchemy**等包作为**对象关系管理**（**ORM**）层。这一层是扁平数据库行（通常来自多个表）和复杂 Python 结构之间的变压器。

在构建复合应用程序时，这个例子中出现的阻抗不匹配是一个主要问题。模拟器的设计是比统计摘要更频繁地运行。对于解决这类问题，我们有几种选择：

+   **总体重新设计**：这可能不是一个明智的选择，因为这两个组件应用程序有一定数量的用户基础。在其他情况下，新的用例是一个机会，可以进行全面的修复并清理一些技术债务。

+   **包括迭代器**：这意味着当我们构建复合应用程序时，我们将添加一个`for`语句来执行多次模拟运行，然后将其处理成一个单一的摘要。这与原始设计意图相符。

+   **一个列表**：这意味着复合应用程序将运行一个模拟，并将这个单一模拟输出提供给摘要。这修改了结构以进行更多的摘要;摘要可能需要组合成预期的单一结果。

在这两者之间的选择取决于首先导致创建复合应用程序的用户故事。这也可能取决于已建立的用户基础。对于我们的目的，我们将假设用户已经意识到 1,000 次模拟运行 1,000 个样本是标准的，并且他们希望遵循*包括迭代器*设计来创建一个复合过程。

作为练习，读者应该追求替代设计。假设用户更愿意在单个模拟中运行 1,000,000 个样本。对于这一点，用户更希望摘要工作采用*一个列表*设计。

我们还将看看另一个选项。在这种情况下，我们将执行 100 次模拟运行，分布在多个并发工作进程中。这将减少创建一百万个样本的时间。这是*包括迭代器*复合设计的变体。

## 如何做...

1.  遵循将复杂过程分解为与输入或输出细节无关的函数的设计模式。有关此内容的详细信息，请参阅*设计用于组合的脚本*食谱。

1.  从工作模块中导入基本函数。在这种情况下，这两个模块的名称相对不那么有趣，`ch13_r05`和`ch13_r06`：

```py
        from ch13_r05 import roll_iter 
        from ch13_r06 import gather_stats 

```

1.  导入所需的任何其他模块。在本示例中，我们将使用`Counter`函数来准备摘要：

```py
        from collections import Counter 

```

1.  创建一个新函数，该函数将来自其他应用程序的现有函数组合在一起。一个函数的输出是另一个函数的输入：

```py
        def summarize_games(total_games, *, seed=None): 
            game_statistics = gather_stats(roll_iter(total_games, seed=seed)) 
            return game_statistics 

```

在许多情况下，明确地堆叠函数，创建中间结果更有意义。当有多个函数创建一种映射-减少管道时，这一点尤为重要：

```py
        def summarize_games_2(total_games, *, seed=None): 
            game_roll_history = roll_iter(total_games, counts, seed=seed) 
            game_statistics = gather_stats(game_roll_history) 
            return game_statistics 

```

我们已将处理分解为具有中间变量的步骤。`game_roll_history`变量是`roll_iter()`函数的输出。这个生成器的输出是`gather_states()`函数的可迭代输入，保存在`game_statistics`变量中。

1.  编写使用此复合过程的输出格式化函数。例如，这是一个练习`summarize_games()`函数的复合过程。这也编写了输出报告：

```py
        def simple_composite(games=100000): 
            start = time.perf_counter() 
            stats = summarize_games(games) 
            end = time.perf_counter() 
            games = sum(stats.values()) 
            print('games', games) 
            print(win_loss(stats)) 
            print("{:.2f} seconds".format(end-start)) 

```

1.  可以使用`argparse`模块来收集命令行选项。有关此内容的示例包括*设计用于组合的脚本*食谱。

## 工作原理...

这种设计的核心特点是将应用程序的各种关注点分离为独立的函数或类。这两个组件应用程序从输入、处理和输出关注点开始进行了设计。从这个基础开始，很容易导入和重用处理。这也使得两个原始应用程序保持不变。

目标是从工作模块中导入函数，并避免复制和粘贴编程。从一个文件复制一个函数并粘贴到另一个文件意味着对一个文件所做的任何更改不太可能被应用到另一个文件。这两个副本将慢慢分歧，导致有时被称为*代码腐烂*的现象。

当一个类或函数做了几件事时，重用潜力会减少。这导致了**重用的反向幂定律**的观察——类或函数的可重用性*R(c)*与该类或函数中特性数量的倒数*F(c)*有关：

*R*（*c*）∝ 1 / *F*（*c*）

单一特性有助于重用。多个特性会减少组件重用的机会。

当我们查看*设计用于组合的脚本*和*使用日志记录进行控制和审计输出*食谱中的两个原始应用程序时，我们可以看到基本函数的特性很少。`roll_iter()`函数模拟了一个游戏，并产生了结果。`gather_stats()`函数从任何数据源中收集统计信息。

计数特性的想法当然取决于抽象级别。从小规模的视角来看，函数会做很多小事情。从非常大的尺度来看，函数需要几个辅助程序来形成一个完整的应用程序；从这个角度来看，单个函数只是一个特性的一部分。

我们的重点是软件的技术特性。这与敏捷概念中的特性作为多个用户故事背后的统一概念无关。在这种情况下，我们谈论的是软件架构技术特性——输入、输出、处理、使用的操作系统资源、依赖关系等等。

从实用的角度来看，相关的技术特性与用户故事相关。这将把规模问题置于用户所感知的软件属性领域。如果用户看到多个特性，这意味着重用可能会有困难。

在这种情况下，一个应用程序创建文件。第二个应用程序总结文件。用户的反馈可能表明区分并不重要，或者可能令人困惑。这导致重新设计，从两个原始步骤创建一个一步操作。

## 还有更多...

我们将看看另外三个可以成为组合应用程序一部分的架构特性：

+   **重构**：*将两个应用程序合并为一个*的方法没有正确区分处理和输出。在尝试创建一个组合应用程序时，我们可能需要重构组件模块。

+   **并发**：并行运行多个`roll_iter()`实例以使用多个核心。

+   **日志记录**：当多个应用程序组合在一起时，组合日志记录可能变得复杂。

### 重构

在某些情况下，有必要重新安排软件以提取有用的特性。在一个组件中，`ch13_r06`模块中有以下函数：

```py
    def process_all_files(result_file, file_names): 
        for source_path in (Path(n) for n in file_names): 
            detail_log.info("read {}".format(source_path)) 
            with source_path.open() as source_file: 
                game_iter = yaml.load_all(source_file) 
                statistics = gather_stats(game_iter) 
                result_file.write( 
                    yaml.dump(dict(statistics), explicit_start=True) 
                ) 

```

这将源文件迭代、详细处理和输出创建结合在一起。`result_file.write()`输出处理是一个单一的复杂语句，很难从这个函数中提取出来。

为了在两个应用程序之间正确地重用此特性，我们需要重构`ch13_r06`应用程序，以便文件输出不被埋在`process_all_files()`函数中。在这种情况下，重构并不太困难。在某些情况下，选择了错误的抽象，重构会变得非常困难。

一行代码，`result_file.write(...)`，需要用一个单独的函数替换。这是一个小改变。具体细节留给读者作为练习。当定义为一个单独的函数时，更容易替换。

这种重构使得新函数可以用于其他组合应用程序。当多个应用程序共享一个公共函数时，这样输出之间的兼容性更高。

### 并发

运行许多模拟，然后进行单个摘要的根本原因是一种 map-reduce 设计。详细的模拟可以并行运行，使用多个核心和多个处理器。然而，最终摘要需要通过统计减少从所有模拟中创建。

我们经常使用操作系统特性来运行多个并发进程。POSIX shell 包括`&`运算符，可以用于分叉并发子进程。Windows 有一个`**start**`命令，类似。我们可以直接利用 Python 来生成多个并发模拟进程。

用于执行此操作的一个模块是`concurrent`包中的`futures`模块。我们可以通过创建`ProcessPoolExecutor`的实例来构建一个并行模拟处理器。我们可以向这个执行程序提交请求，然后收集这些并发请求的结果：

```py
    import concurrent.futures 

    def parallel(): 
        start = time.perf_counter() 
        total_stats = Counter() 
        worker_list = [] 
        with concurrent.futures.ProcessPoolExecutor() as executor: 
            for i in range(100): 
                worker_list.append(executor.submit(summarize_games, 1000)) 
            for worker in worker_list: 
                stats = worker.result() 
                total_stats.update(stats) 
        end = time.perf_counter() 

        games = sum(total_stats.values()) 
        print('games', games) 
        print(win_loss(total_stats)) 
        print("{:.2f} seconds".format(end-start)) 

```

我们初始化了三个对象：`start`，`total_stats`和`worker_list`。`start`对象记录了处理开始的时间；`time.perf_counter()`通常是最准确的可用计时器。`total_stats`是一个`Counter`对象，将收集最终的统计摘要。`worker_list`将是一个单独的`Future`对象列表，每个请求都有一个。

`ProcessPoolExecutor`方法定义了一个处理上下文，其中有一个工作池可用于处理请求。默认情况下，池中的工作进程数量与处理器数量相同。每个工作进程都在导入给定模块的执行器中运行。模块中定义的所有函数和类都可供工作进程使用。

执行器的`submit()`方法会执行一个函数以及该函数的参数。在这个例子中，将进行 100 个请求，每个请求将模拟 1,000 场比赛，并返回这些比赛的骰子点数序列。`submit()`返回一个`Future`对象，它是工作请求的模型。

在提交所有 100 个请求后，收集结果。`Future`对象的`result()`方法等待处理完成并收集结果对象。在这个例子中，结果是 1,000 场比赛的统计摘要。然后将它们组合成整体的`total_stats`摘要。

以下是串行和并行执行之间的比较：

```py
 **games 100000** 

 **Counter({'loss': 50997, 'win': 49003})** 

 **2.83 seconds** 

 **games 100000** 

 **Counter({'loss': 50523, 'win': 49477})** 

 **1.49 seconds** 

```

处理时间减少了一半。由于有 100 个并发请求，为什么时间没有减少原始时间的 1/100？观察到在生成子进程、通信请求数据和通信结果数据方面存在相当大的开销。

### 记录

在*使用日志进行控制和审计输出*的示例中，我们看到了如何使用`logging`模块进行控制、审计和错误输出。当构建复合应用程序时，我们将不得不结合原始应用程序的每个日志功能。

记录涉及三个部分的步骤：

1.  创建记录器对象。通常是一行代码，如`logger = logging.get_logger('some_name')`。通常在类或模块级别执行一次。

1.  使用记录器对象收集事件。这涉及到诸如`logger.info('some message')`这样的行。这些行分散在整个应用程序中。

1.  整体配置日志系统。应用程序中有两种日志配置可能性：

+   尽可能外部化。在这种情况下，日志配置仅在应用程序的最外层全局范围内完成：

```py
            if __name__ == "__main__": 
                logging configuration goes only here. 
                main() 
                logging.shutdown() 

```

这保证了日志系统只有一个配置。

+   在类、函数或模块的某个地方。在这种情况下，我们可能有几个模块都试图进行日志配置。这是由日志系统容忍的。但是，调试可能会令人困惑。

这些示例都遵循第一种方法。如果所有应用程序都在最全局范围内配置日志记录，那么很容易理解如何配置复合应用程序。

在存在多个日志配置的情况下，复合应用程序可以采用两种方法：

+   复合应用程序包含最终配置，它有意覆盖了先前定义的所有记录器。这是默认设置，并可以通过在 YAML 配置文档中明确说明`incremental: false`来表示。

+   复合应用程序保留其他应用程序记录器，仅修改记录器配置，可能是通过设置整体级别。这是通过在 YAML 配置文档中包含`incremental: true`来完成的。

当组合 Python 应用程序时，增量配置对于不隔离日志配置的应用程序非常有用。为了正确为复合应用程序配置日志，可能需要一些时间来阅读和理解每个应用程序的代码。

## 另请参阅

+   在*为组合设计脚本*配方中，我们看了一个可组合应用程序的核心设计模式

# 使用 Command 设计模式组合多个应用程序

许多复杂的应用程序套件遵循与 Git 程序类似的设计模式。有一个基本命令`git`，有许多子命令。例如，`git pull`，`git commit`和`git push`。

这个设计的核心是一系列单独的命令。git 的各种功能可以被看作是执行给定功能的单独类定义。

当我们输入诸如`git pull`这样的命令时，就好像程序`git`正在定位一个实现该命令的类。

我们如何创建一系列密切相关的命令？

## 准备工作

我们将想象一个由三个命令构建的应用程序。这是基于*为组合设计脚本*，*使用日志进行控制和审计输出*和*将两个应用程序合并为一个*配方中显示的应用程序。我们将有三个应用程序——*模拟*，*总结*和一个名为*simsum*的组合应用程序。

这些功能基于诸如`ch13_r05`，`ch13_r06`和`ch13_r07`之类的模块。这个想法是我们可以将这些单独的模块重组成一个遵循 Command 设计模式的单一类层次结构。

这种设计有两个关键要素：

1.  客户端只依赖于抽象超类`Command`。

1.  `Command`超类的每个单独子类都有一个相同的接口。我们可以用其中任何一个替换其他任何一个。

当我们完成这个之后，一个整体的应用程序脚本可以创建和执行任何一个`Command`子类。

## 如何做...

1.  整体应用程序将具有一种结构，试图将功能分为两类——参数解析和命令执行。每个子命令都将包括处理和输出捆绑在一起。

这是`Command`超类：

```py
        from argparse import Namespace 

        class Command: 
            def execute(self, options: Namespace): 
                pass 

```

我们将依赖于`argparse.Namespace`为每个子类提供一个非常灵活的选项集合。这不是必需的，但在*管理复合应用程序中的参数和配置*配方中会很有帮助。由于该配方将包括选项解析，因此似乎最好专注于每个类使用`argparse.Namespace`。

1.  为`Simulate`命令创建`Command`超类的子类：

```py
        import ch13_r05 

        class Simulate(Command): 
            def __init__(self, seed=None): 
                self.seed = seed 
            def execute(self, options): 
                self.game_path = Path(options.game_file) 
                data = ch13_r05.roll_iter(options.games, self.seed) 
                ch13_r05.write_rolls(self.game_path, data) 

```

我们已经将`ch13_r05`模块的处理和输出包装到这个类的`execute()`方法中。

1.  为`Summarize`命令创建`Command`超类的子类：

```py
        import ch13_r06 

        class Summarize(Command): 
            def execute(self, options): 
                self.summary_path = Path(options.summary_file) 
                with self.summary_path.open('w') as result_file: 
                    ch13_r06.process_all_files(result_file, options.game_files) 

```

对于这个类，我们已经将文件创建和文件处理包装到类的`execute()`方法中。

1.  所有的整体过程都可以由以下`main()`函数执行：

```py
        from argparse import Namespace 

        def main(): 
            options_1 = Namespace(games=100, game_file='x.yaml') 
            command1 = Simulate() 
            command1.execute(options_1) 

            options_2 = Namespace(summary_file='y.yaml', game_files=['x.yaml']) 
            command2 = Summarize() 
            command2.execute(options_2) 

```

我们创建了两个命令，一个是`Simulate`类的实例，另一个是`Summarize`类的实例。这些可以被执行以提供一个同时模拟和总结数据的组合功能。

## 工作原理...

为各种子命令创建可互换的多态类是提供可扩展设计的一种方便方式。`Command`设计模式强烈鼓励每个单独的子类具有相同的签名，以便可以创建和执行任何命令。此外，可以添加适合框架的新命令。

SOLID 设计原则之一是**Liskov 替换原则**（**LSP**）。`Command`抽象类的任何子类都可以替代父类。

每个`Command`实例都有一个简单的接口。有两个功能：

+   `__init__()`方法期望由参数解析器创建的命名空间对象。每个类将只从这个命名空间中选择所需的值，忽略其他任何值。这允许子命令忽略不需要的全局参数。

+   `execute()`方法执行处理并写入任何输出。这完全基于初始化期间提供的值。

使用命令设计模式可以确保它们可以互换。整个`main()`脚本可以创建`Simulate`或`Summarize`类的实例。替换原则意味着任一实例都可以执行，因为接口是相同的。这种灵活性使得解析命令行选项并创建任一可用类的实例变得容易。我们可以扩展这个想法并创建单个命令实例的序列。

## 还有更多...

这种设计模式的更常见扩展之一是提供组合命令。在*将两个应用程序合并为一个*的示例中，我们展示了创建组合的一种方法。这是另一种方法，基于定义一个实现现有命令组合的新命令：

```py
    class CommandSequence(Command): 
        def __init__(self, *commands): 
            self.commands = [command() for command in commands] 
        def execute(self, options): 
            for command in self.commands: 
                command.execute(options) 

```

这个类将通过`*commands`参数接受其他`Command`类。这个序列将组合所有的位置参数值。它将从这些类中构建单独的类实例。

我们可以像这样使用`CommandSequence`类：

```py
    options = Namespace(games=100, game_file='x.yaml', 
        summary_file='y.yaml', game_files=['x.yaml'] 
    ) 
    sim_sum_command = CommandSequence(Simulate, Summarize) 
    sim_sum_command.execute(options) 

```

我们使用了两个其他类`Simulate`和`Summarize`创建了一个`CommandSequence`的实例。`__init__()`方法将构建这两个对象的内部序列。然后`sim_sum_command`对象的`execute()`方法将按顺序执行这两个处理步骤。

这种设计虽然简单，但暴露了许多实现细节。特别是两个类名和中间的`x.yaml`文件是可以封装到更好的类设计中的细节。

如果我们专门关注被组合的两个命令，我们可以创建一个稍微更好的`CommandSequence`子类参数。这将有一个`__init__()`方法，遵循其他`Command`子类的模式：

```py
    class SimSum(CommandSequence): 
        def __init__(self): 
            super().__init__(Simulate, Summarize) 

```

这个类定义将两个其他类合并到已定义的`CommandSequence`结构中。我们可以通过稍微修改选项来继续这个想法，以消除`Simulate`步骤中`game_file`的显式值，这也必须是`Summarize`步骤的`game_files`输入的一部分。

我们想要构建和使用一个更简单的`Namespace`，其选项如下：

```py
    options = Namespace(games=100, summary_file='y.yaml') 
    sim_sum_command = SimSum() 
    sim_sum_command.execute(options) 

```

这意味着一些缺失的选项必须由`execute()`方法注入。我们将把这个方法添加到`SimSum`类中：

```py
    def execute(self, options): 
        new_namespace = Namespace( 
            game_file='x.yaml', 
            game_files=['x.yaml'], 
            **vars(options) 
        ) 
        super().execute(new_namespace) 

```

这个`execute()`方法克隆了选项。它添加了两个额外的值，这些值是命令集成的一部分，但不是用户应该提供的。

这种设计避免了更新有状态的选项集。为了保持原始选项对象不变，我们进行了复制。`vars()`函数将`Namespace`公开为一个简单的字典。然后我们可以使用`**`关键字参数技术将字典转换为新的`Namespace`对象的关键字参数。这将创建一个浅拷贝。如果命名空间内的有状态对象被更新，原始的`options`和`new_namespace`参数都可以访问相同的基础值对象。

由于`new_namespace`是一个独立的集合，我们可以向这个`Namespace`实例添加新的键和值。这些只会出现在`new_namespace`中，不会影响原始选项对象。

## 另请参阅

+   在*为组合设计脚本*、*使用日志进行控制和审计输出*和*将两个应用程序合并为一个*的示例中，我们看了这个组合应用程序的组成部分。在大多数情况下，我们需要结合所有这些示例的元素来创建一个有用的应用程序。

+   我们经常需要遵循*在组合应用程序中管理参数和配置*的示例。

# 在组合应用程序中管理参数和配置

当我们有一套复杂的单独应用程序（或系统）时，几个应用程序共享共同特征是很常见的。当然，我们可以使用普通的继承来定义一个库模块，为复杂套件中的每个单独应用程序提供共同的类和函数。

创建许多单独应用程序的缺点是外部 CLI 直接与软件架构相关联。重新排列软件组件变得笨拙，因为更改也会改变可见的 CLI。

许多应用文件之间共同特征的协调可能变得笨拙。例如，定义命令行参数的各种一字母缩写选项是困难的。这需要在所有单个应用文件之外保持某种选项的主列表。看起来这应该在代码的某个地方集中保存。

是否有继承的替代方案？如何确保一套应用程序可以重构而不会对 CLI 造成意外更改或需要复杂的额外设计说明？

## 准备工作

许多复杂的应用套件遵循与 Git 使用的相似的设计模式。有一个基本命令`git`，带有许多子命令。例如，`git pull`，`git commit`和`git push`。命令行界面的核心可以由`git`命令集中。然后可以根据需要组织和重新组织子命令，而对可见 CLI 的更改较少。

我们将想象一个由三个命令构建的应用程序。这是基于*为组合设计脚本*，*使用日志记录进行控制和审计输出*和*将两个应用程序合并为一个*配方中显示的应用程序。我们将有三个应用程序，每个应用程序有三个命令：`craps simulate`，`craps summarize`和组合应用程序`craps simsum`。

我们将依赖于*使用命令设计模式合并多个应用程序*配方中的子命令设计。这将提供`Command`子类的方便层次结构：

+   `Command`类是一个抽象超类。

+   `Simulate`子类执行*为组合设计脚本*配方中的模拟功能。

+   `Summarize`子类执行*使用日志记录进行控制和审计输出*配方中的总结功能。

+   `SimSum`子类可以执行组合模拟和总结，遵循*将两个应用程序合并为一个*的想法。

为了创建一个简单的命令行应用程序，我们需要适当的参数解析。

这个参数解析将依赖于`argparse`模块的子命令解析能力。我们可以创建适用于所有子命令的一组公共命令选项。我们还可以为每个子命令创建唯一的选项。

## 如何做...

1.  定义命令界面。这是一种**用户体验**（**UX**）设计练习。虽然大多数 UX 都集中在 Web 和移动设备应用程序上，但核心原则也适用于 CLI 应用程序和服务器。

早些时候，我们注意到根应用程序将是`craps`。它将有以下三个子命令：

```py
 **craps simulate -o game_file -g games 
      craps summarize -o summary_file game_file ... 
      craps simsum -g games** 

```

1.  定义根 Python 应用程序。与本书中的其他文件一致，我们将称其为`ch13_r08.py`。在操作系统级别，我们可以提供一个别名或链接，使可见界面与用户对`craps`的期望相匹配。

1.  我们将从*使用命令设计模式合并多个应用程序*配方中导入类定义。这将包括`Command`超类和`Simulate`，`Summarize`和`SimSum`子类。

1.  创建整体参数解析器，然后创建一个子解析器构建器。`subparsers`对象将用于创建每个子命令的参数定义：

```py
        import argparse 
        def get_options(argv): 
            parser = argparse.ArgumentParser(prog='craps') 
            subparsers = parser.add_subparsers() 

```

对于每个命令，创建一个解析器，并添加该命令特有的参数。

1.  使用两个唯一于模拟的选项定义`simulate`命令。我们还将提供一个特殊的默认值，用于初始化生成的`Namespace`对象：

```py
            simulate_parser = subparsers.add_parser('simulate') 
            simulate_parser.add_argument('-g', '--games', type=int, default=100000) 
            simulate_parser.add_argument('-o', '--output', dest='game_file') 
            simulate_parser.set_defaults(command=Simulate) 

```

1.  定义`summarize`命令，带有此命令特有的参数。提供将填充`Namespace`对象的默认值：

```py
            summarize_parser = subparsers.add_parser('summarize') 
            summarize_parser.add_argument('-o', '--output', dest='summary_file') 
            summarize_parser.add_argument('game_files', nargs='*') 
            summarize_parser.set_defaults(command=Summarize) 

```

1.  定义`simsum`命令，并类似地提供一个独特的默认值，以便更轻松地处理`Namespace`：

```py
            simsum_parser = subparsers.add_parser('simsum') 
            simsum_parser.add_argument('-g', '--games', type=int, default=100000) 
            simsum_parser.add_argument('-o', '--output', dest='summary_file') 
            simsum_parser.set_defaults(command=SimSum) 

```

1.  解析命令行值。在这种情况下，`get_options()`函数的整体参数预期是`sys.argv[1:]`的值，其中包括 Python 命令的参数。我们可以覆盖参数值以进行测试：

```py
            options = parser.parse_args(argv) 
            if 'command' not in options: 
                parser.print_help() 
                sys.exit(2) 
            return options 

```

整体解析器包括三个子命令解析器。一个将处理`craps simulate`命令，另一个处理`craps summarize`，第三个处理`craps simsum`。每个子命令具有略有不同的选项组合。

`command`选项只能通过`set_defaults()`方法设置。这会发送有关要执行的命令的有用的附加信息。在这种情况下，我们提供了必须实例化的类。

1.  整体应用程序由以下`main()`函数定义：

```py
        def main(): 
            options = get_options(sys.argv[1:]) 
            command = options.command(options) 
            command.execute() 

```

选项将被解析。每个不同的子命令为`options.command`参数设置一个唯一的类值。这个类用于构建`Command`子类的实例。这个对象将有一个`execute()`方法，用于执行这个命令的真正工作。

1.  实现根命令的操作系统包装器。我们可能有一个名为`craps`的文件。该文件将具有 rx 权限，以便其他用户可以读取。文件的内容可能是这一行：

```py
 **python3.5 ch13_r08.py $*** 

```

这个小的 shell 脚本提供了一个方便的方式来输入一个`**craps**`命令，并使其正确执行一个具有不同名称的 Python 脚本。

我们可以这样创建一个 bash shell 别名：

```py
 **alias craps='python3.5 ch13_r08.py'** 

```

这可以放在`.bashrc`文件中以定义一个`**craps**`命令。

## 工作原理...

这个配方有两个部分：

+   使用`Command`设计模式来定义一组相关的多态类。有关更多信息，请参阅*使用命令设计模式组合多个应用程序*配方。

+   使用`argparse`模块的特性来处理子命令。

这里重要的`argparse`模块特性是解析器的`add_subparsers()`方法。此方法返回一个对象，用于构建每个不同的子命令解析器。我们将此对象分配给变量`subparsers`。

我们还在顶层解析器中定义了一个简单的`command`参数。这个参数只能由为每个子解析器定义的默认值填充。这提供了一个值，显示实际调用了哪个子命令。

每个子解析器都是使用子解析器对象的`add_parser()`方法构建的。然后返回的`parser`对象可以定义参数和默认值。

当执行整体解析器时，它将解析在子命令之外定义的任何参数。如果有子命令，这将用于确定如何解析剩余的参数。

看下面的命令：

```py
 **craps simulate -g 100 -o x.yaml** 

```

这个命令将被解析为创建一个像这样的`Namespace`对象：

```py
 **Namespace(command=<class '__main__.Simulate'>, game_file='x.yaml', games=100)** 

```

`Namespace`对象中的`command`属性是作为子命令定义的一部分提供的默认值。`game_file`和`games`的值来自`-o`和`-g`选项。

### 命令设计模式

为各种子命令创建可互换的多态类，创建一个易于重构或扩展的设计。`Command`设计模式强烈鼓励每个单独的子类具有相同的签名，以便可以创建和执行任何可用的命令类之一。

SOLID 设计原则之一是 Liskov 替换原则。命令抽象类的任何子类都可以用于替换父类。

每个`Command`都有一个一致的接口：

+   `__init__()`方法期望由参数解析器创建的命名空间对象。每个类将只从这个命名空间中选择所需的值，忽略其他任何值。这允许全局参数被不需要它的子命令忽略。

+   `execute()`方法执行处理并写入任何输出。这完全基于初始化时提供的值。

命令设计模式的使用使得很容易确保它们可以相互替换。替换原则意味着`main()`函数可以简单地创建一个实例，然后执行对象的`execute()`方法。

## 还有更多...

我们可以考虑将子命令解析器的细节下推到每个类定义中。例如，“模拟”类定义了两个参数：

```py
    simulate_parser.add_argument('-g', '--games', type=int, default=100000) 
    simulate_parser.add_argument('-o', '--output', dest='game_file') 

```

`get_option()`函数似乎不应该定义关于实现类的这些细节。一个适当封装的设计似乎应该将这些细节分配给每个`Command`子类。

我们需要添加一个配置给定解析器的静态方法。新的类定义将如下所示：

```py
    import ch13_r05 
    class Simulate(Command): 
        def __init__(self, options, *, seed=None): 
            self.games = options.games 
            self.game_file = options.game_file 
            self.seed = seed 
        def execute(self): 
            data = ch13_r05.roll_iter(self.games, self.seed) 
            ch13_r05.write_rolls(self.game_file, data) 
        @staticmethod 
        def configure(simulate_parser): 
            simulate_parser.add_argument('-g', '--games', type=int, default=100000) 
            simulate_parser.add_argument('-o', '--output', dest='game_file') 

```

我们添加了一个`configure()`方法来配置解析器。这个改变使得很容易看到`__init__()`参数将如何通过解析命令行值来创建。这使我们能够重写`get_option()`函数，如下：

```py
    import argparse 
    def get_options(argv): 
        parser = argparse.ArgumentParser(prog='craps') 
        subparsers = parser.add_subparsers() 

        simulate_parser = subparsers.add_parser('simulate') 
        Simulate.configure(simulate_parser) 
        simulate_parser.set_defaults(command=Simulate) 

        # etc. for each class 

```

这将利用静态的`configure()`方法来提供参数细节。命令参数的默认值可以由整体的`get_options()`处理，因为它不涉及内部细节。

## 另请参阅

+   请参阅*为组合设计脚本*，*使用日志记录进行控制和审计输出*和*将两个应用程序合并为一个*的方法，了解组件的背景

+   在第五章的*使用 argparse 获取命令行输入*方法中，了解更多关于参数解析的背景

# 包装和组合 CLI 应用程序

一种常见的自动化类型涉及运行几个程序，这些程序实际上都不是 Python 应用程序。由于这些程序不是用 Python 编写的，因此不可能重写每个程序以创建一个复合的 Python 应用程序。我们无法遵循*将两个应用程序合并为一个*的方法。

与聚合功能不同，另一种选择是在 Python 中包装其他程序以提供更高级的构造。使用情况与编写 shell 脚本的使用情况非常相似。不同之处在于使用 Python 而不是 shell 语言。使用 Python 有一些优势：

+   Python 拥有丰富的数据结构集合。而 shell 只有字符串和字符串数组。

+   Python 拥有出色的单元测试框架。这可以确保 Python 版本的 shell 脚本可以正常工作，而不会使广泛使用的服务崩溃的风险。

我们如何从 Python 中运行其他应用程序？

## 准备工作

在*为组合设计脚本*的方法中，我们确定了一个应用程序，该应用程序进行了一些处理，导致了一个相当复杂的结果。对于这个方法，我们假设该应用程序不是用 Python 编写的。

我们想要运行这个程序几百次，但我们不想将必要的命令复制粘贴到脚本中。此外，由于 shell 很难测试并且数据结构很少，我们希望避免使用 shell。

对于这个方法，我们假设`ch13_r05`应用程序是一个本地二进制应用程序；它可能是用 C++或 Fortran 编写的。这意味着我们不能简单地导入包含应用程序的 Python 模块。相反，我们将不得不通过运行一个单独的操作系统进程来处理这个应用程序。

我们将使用`subprocess`模块在操作系统级别运行应用程序。从 Python 中运行另一个二进制程序有两种常见的用例：

+   没有输出，或者我们不想在我们的 Python 程序中收集它。第一种情况是当 OS 实用程序在成功或失败时返回状态码时的典型情况。第二种情况是当许多子程序都在写入标准错误日志时的典型情况；父 Python 程序只是启动子进程。

+   我们需要捕获并可能分析输出以检索信息或确定成功的级别。

在这个配方中，我们将看看第一种情况——输出不是我们需要捕获的东西。在*包装程序并检查输出*配方中，我们将看看第二种情况，即 Python 包装程序对输出进行了审查。

## 如何做...

1.  导入 `subprocess` 模块：

```py
        import subprocess 

```

1.  设计命令行。通常，应该在操作系统提示符下进行测试，以确保它执行正确的操作：

```py
 **slott$ python3 ch13_r05.py --samples 10 --output x.yaml** 

```

输出文件名需要灵活，这样我们可以运行程序数百次。这意味着创建名称为 `game_{n}.yaml` 的文件。

1.  编写一个语句，通过适当的命令进行迭代。每个命令可以构建为一系列单词的序列。从工作的 shell 命令开始，并在空格上拆分该行，以创建适当的单词序列：

```py
        files = 100 
        for n in range(files): 
            filename = 'game_{n}.yaml'.format_map(vars()) 
            command = ['python3', 'ch13_r05.py', 
                '--samples', '10', '--output', filename] 

```

这将创建各种命令。我们可以使用 `print()` 函数显示每个命令，并确认文件名是否定义正确。

1.  评估 `subprocess` 模块中的 `run()` 函数。这将执行给定的命令。提供 `check=True`，这样如果有任何问题，它将引发 `subprocess.CalledProcessError` 异常：

```py
        subprocess.run(command, check=True)

```

1.  为了正确测试这一点，整个序列应该转换为一个适当的函数。如果将来会有更多相关的命令，它应该是 `Command` 类层次结构中的子类的方法。参见*在复合应用程序中管理参数和配置*配方。

## 它是如何工作的...

`subprocess` 模块是 Python 程序运行计算机上其他程序的方式。`run()` 函数为我们做了很多事情。

在 POSIX（如 Linux 或 Mac OS X）环境中，步骤类似于以下序列：

+   为子进程准备 `stdin`、`stdout` 和 `stderr` 文件描述符。在这种情况下，我们接受了默认值，这意味着子进程继承了父进程正在使用的文件。如果子进程打印到 `stdout`，它将出现在父进程使用的同一个控制台上。

+   调用 `os.fork()` 函数将当前进程分成父进程和子进程。父进程将获得子进程的进程 ID；然后它可以等待子进程完成。

+   在子进程中，执行 `os.execl()` 函数（或类似的函数）以提供子进程将执行的命令路径和参数。

+   然后子进程运行，使用给定的 `stdin`、`stdout` 和 `stderr` 文件。

+   同时，父进程使用诸如 `os.wait()` 的函数等待子进程完成并返回最终状态。

+   由于我们使用了 `check=True` 选项，`run()` 函数将非零状态转换为异常。

OS shell（如 bash）会向应用程序开发人员隐藏这些细节。`subprocess.run()` 函数同样隐藏了创建和等待子进程的细节。

Python 的 `subprocess` 模块提供了许多类似于 shell 的功能。最重要的是，Python 提供了几组额外的功能：

+   更丰富的数据结构。

+   异常用于识别出现的问题。这比在 shell 脚本中插入 `if` 语句来检查状态码要简单得多且更可靠。

+   一种在不使用操作系统资源的情况下对脚本进行单元测试的方法。

## 还有更多...

我们将向这个脚本添加一个简单的清理功能。想法是所有的输出文件应该作为一个原子操作创建。我们希望所有文件都存在，或者没有文件存在。我们不希望有不完整的数据文件集。

这符合 ACID 属性：

+   **原子性**：整个数据集要么可用，要么不可用。集合是一个单一的、不可分割的工作单元。

+   **一致性**：文件系统应该从一个内部一致的状态转移到另一个一致的状态。任何摘要或索引都应该正确反映实际文件。

+   **隔离性**：如果我们想要并行处理数据，那么多个并行进程应该可以工作。并发操作不应该相互干扰。

+   **持久性**：一旦文件被写入，它们应该保留在文件系统上。对于文件来说，这个属性几乎是不言而喻的。对于更复杂的数据库，需要考虑可能被数据库客户端确认但实际上尚未写入服务器的事务数据。

使用操作系统进程和单独的工作目录可以相对简单地实现大多数这些特性。然而，原子性属性导致需要进行清理操作。

为了清理，我们需要用 `try:` 块包装核心处理。整个函数看起来像这样：

```py
    import subprocess 
    from pathlib import Path 

    def make_files(files=100): 
        try: 
            for n in range(files): 
                filename = 'game_{n}.yaml'.format_map(vars()) 
                command = ['python3', 'ch13_r05.py', 
                    '--samples', '10', '--output', filename] 
                subprocess.run(command, check=True) 
        except subprocess.CalledProcessError as ex: 
            for partial in Path('.').glob("game_*.yaml"): 
                partial.unlink() 
            raise 

```

异常处理块有两个作用。首先，它会从当前工作目录中删除任何不完整的文件。其次，它会重新引发原始异常，以便故障传播到客户端应用程序。

由于处理失败，提高异常是很重要的。在某些情况下，应用程序可能会定义一个新的异常，特定于该应用程序。可以引发这个新的异常，而不是重新引发原始的 `CalledProcessError` 异常。

### 单元测试

为了对这个进行单元测试，我们需要模拟两个外部对象。我们需要模拟 `subprocess` 模块中的 `run()` 函数。我们不想实际运行其他进程，但我们想确保 `run()` 函数从 `make_files()` 函数中被适当地调用。

我们还需要模拟 `Path` 类和生成的 `Path` 对象。这些提供文件名，并将调用 `unlink()` 方法。我们需要为此创建模拟，以确保真实应用程序只取消链接适当的文件。

使用模拟对象进行测试意味着我们永远不会在测试时意外删除有用的文件。这是使用 Python 进行这种自动化的重要好处。

这是我们定义各种模拟对象的设置：

```py
    import unittest 
    from unittest.mock import * 

    class GIVEN_make_files_exception_WHEN_call_THEN_run(unittest.TestCase): 
        def setUp(self): 
            self.mock_subprocess_run = Mock( 
                side_effect = [ 
                    None, 
                    subprocess.CalledProcessError(2, 'ch13_r05')] 
            ) 
            self.mock_path_glob_instance = Mock() 
            self.mock_path_instance = Mock( 
                glob = Mock( 
                    return_value = [self.mock_path_glob_instance] 
                ) 
            ) 
            self.mock_path_class = Mock( 
                return_value = self.mock_path_instance 
            ) 

```

我们已经定义了 `self.mock_subprocess_run`，它将表现得有点像 `run()` 函数。我们使用了 `side_effect` 属性为这个函数提供多个返回值。第一个响应将是 `None` 对象。然而，第二个响应将是一个 `CalledProcessError` 异常。这个异常需要两个参数，一个进程返回代码，和原始命令。

`self.mock_path_class`，最后显示，响应对 `Path` 类请求的调用。这将返回一个模拟的类实例。`self.mock_path_instance` 对象是 `Path` 的模拟实例。

创建的第一个路径实例将评估 `glob()` 方法。为此，我们使用了 `return_value` 属性来返回要删除的 `Path` 实例列表。在这种情况下，返回值将是一个我们期望被取消链接的单个 `Path` 对象。

`self.mock_path_glob_instance` 对象是从 `glob()` 返回的。如果算法操作正确，这应该被取消链接。

这是这个单元测试的 `runTest()` 方法：

```py
    def runTest(self): 
        with patch('__main__.subprocess.run', self.mock_subprocess_run), \ 
            patch('__main__.Path', self.mock_path_class): 
            self.assertRaises( 
                subprocess.CalledProcessError, make_files, files=3) 
        self.mock_subprocess_run.assert_has_calls( 
            [call( 
                ['python3', 'ch13_r05.py', '--samples', '10', 
                 '--output', 'game_0.yaml'], 
                check=True), 
             call( 
                ['python3', 'ch13_r05.py', '--samples', '10', 
                '--output', 'game_1.yaml'], 
                check=True), 
             ] 
         ) 
         self.assertEqual(2, self.mock_subprocess_run.call_count) 
         self.mock_path_class.assert_called_once_with('.') 
         self.mock_path_instance.glob.assert_called_once_with('game_*.yaml') 
         self.mock_path_glob_instance.unlink.assert_called_once_with() 

```

我们应用了两个补丁：

+   在 `__main__` 模块中，对 `subprocess` 的引用将使用 `self.mock_subprocess_run` 对象替换 `run()` 函数。这将允许我们跟踪 `run()` 被调用的次数。它将允许我们确认 `run()` 是否以正确的参数被调用。

+   在 `__main__` 模块中，对 `Path` 的引用将被替换为 `self.mock_path_class` 对象。这将返回已知的值，并允许我们确认只有预期的调用被执行。

`self.assertRaises`方法用于确认在调用`make_files()`方法时，在这个特定的修补上下文中正确引发了`CalledProcessError`异常。`run()`方法的模拟版本将引发异常——我们期望确切的异常是停止处理的异常。

模拟的`run()`函数只被调用两次。第一次调用将成功。第二次调用将引发异常。我们可以使用`Mock`对象的`call_count`属性来确认确实调用了两次`run()`。

`self.mock_path_instance` 方法是`Path('.')`对象的模拟，该对象作为异常处理的一部分创建。这个对象必须评估`glob()`方法。测试断言检查参数值，以确保使用了`'game_*.yaml'`。

最后，`self.mock_path_glob_instance`是`Path('.').glob('game_*.yaml')`创建的`Path`对象的模拟。这个对象将评估`unlink()`方法。这将导致删除文件。

这个单元测试提供了算法将按照广告运行的信心。测试是在不占用大量计算资源的情况下进行的。最重要的是，测试是在不小心删除错误文件的情况下进行的。

## 另请参阅

+   这种自动化通常与其他 Python 处理结合使用。请参阅*为组合设计脚本*配方。

+   目标通常是创建一个复合应用程序；参见*在复合应用程序中管理参数和配置*配方。

+   有关此配方的变体，请参阅*包装程序并检查输出*配方。

# 包装程序并检查输出

一种常见的自动化类型涉及运行几个程序，其中没有一个实际上是 Python 应用程序。在这种情况下，不可能重写每个程序以创建一个复合的 Python 应用程序。为了正确地聚合功能，其他程序必须被包装为 Python 类或模块，以提供一个更高级的构造。

这种用例与编写 shell 脚本的用例非常相似。不同之处在于 Python 可能是比操作系统内置的 shell 语言更好的编程语言。

在某些情况下，Python 提供的优势是能够分析输出文件。Python 程序可能会转换、过滤或总结子进程的输出。

我们如何从 Python 中运行其他应用程序并处理它们的输出？

## 准备工作

在*为组合设计脚本*配方中，我们确定了一个应用程序进行了一些处理，导致了一个相当复杂的结果。我们想运行这个程序几百次，但我们不想复制和粘贴必要的命令到一个脚本中。此外，由于 shell 很难测试并且数据结构很少，我们想避免使用 shell。

对于这个配方，我们假设`ch13_r05`应用程序是用 Fortran 或 C++编写的本机二进制应用程序。这意味着我们不能简单地导入包含应用程序的 Python 模块。相反，我们将不得不通过运行一个单独的操作系统进程来处理这个应用程序。

我们将使用`subprocess`模块在操作系统级别运行应用程序。从 Python 中运行另一个二进制程序有两种常见的用例：

+   没有任何输出，或者我们不想在我们的 Python 程序中收集它。

+   我们需要捕获并可能分析输出以检索信息或确定成功的级别。我们可能需要转换、过滤或总结日志输出。

在这个配方中，我们将看看第二种情况——输出必须被捕获和总结。在*包装和组合 CLI 应用程序*配方中，我们将看看第一种情况，即输出被简单地忽略。

这是运行`ch13_r05`应用程序的一个例子：

```py
 **slott$ python3 ch13_r05.py --samples 10 --output=x.yaml** 

 **Namespace(output='x.yaml', output_path=PosixPath('x.yaml'), samples=10, seed=None)** 

 **Counter({5: 7, 6: 7, 7: 7, 8: 5, 4: 4, 9: 4, 11: 3, 10: 1, 12: 1})** 

```

有两行输出写入操作系统标准输出文件。第一行有选项的摘要。第二行的输出是一个带有文件摘要的`Counter`对象。我们想要捕获这些`'Counter'`行的细节。

## 如何操作...

1.  导入`subprocess`模块：

```py
        import subprocess 

```

1.  设计命令行。通常，这应该在操作系统提示符下进行测试，以确保它执行正确的操作。我们展示了一个命令的示例。

1.  为要执行的各种命令定义一个生成器。每个命令都可以作为一个单词序列构建。原始的 shell 命令被拆分成单词序列。

```py
        def command_iter(files): 
            for n in range(files): 
                filename = 'game_{n}.yaml'.format_map(vars()) 
                command = ['python3', 'ch13_r05.py', 
                    '--samples', '10', '--output', filename] 
                yield command 

```

这个生成器将产生一系列命令字符串。客户端可以使用`for`语句来消耗生成的每个命令。

1.  定义一个执行各种命令并收集输出的函数：

```py
        def command_output_iter(iterable): 
            for command in iterable: 
                process = subprocess.run(command, stdout=subprocess.PIPE, check=True) 
                output_bytes = process.stdout 
                output_lines = list(l.strip() for l in output_bytes.splitlines()) 
                yield output_lines 

```

使用`stdout=subprocess.PIPE`的参数值意味着父进程将收集子进程的输出。创建一个操作系统级的管道，以便父进程可以读取子进程的输出。

这个生成器将产生一系列行列表。每个行列表将是`ch13_r05.py`应用程序的输出行。通常每个列表中会有两行。第一行是参数摘要，第二行是`Counter`对象。

1.  定义一个整体流程，将这两个生成器结合起来，以便执行生成的每个命令：

```py
        command_sequence = command_iter(100) 
        output_lines_sequence = command_output_iter(command_sequence) 
        for batch in output_lines_sequence: 
            for line in batch: 
                if line.startswith('Counter'): 
                    batch_counter = eval(line) 
                    print(batch_counter) 

```

`command_sequence`变量是一个生成器，将产生多个命令。这个序列是由`command_iter()`函数构建的。

`output_lines_sequence`是一个生成器，将产生多个输出行列表。这是由`command_output_iter()`函数构建的，它将使用给定的`command_sequence`对象，运行多个命令，收集输出。

`output_lines_sequence`中的每个批次将是一个包含两行的列表。以`Counter`开头的行表示一个`Counter`对象。

我们使用`eval()`函数从文本表示中重新创建原始的`Counter`对象。我们可以使用这些`Counter`对象进行分析或总结。

大多数实际应用程序将不得不使用比内置的`eval()`更复杂的函数来解释输出。有关处理复杂行格式的信息，请参阅第一章中的*使用正则表达式解析字符串*，*数字、字符串和元组*，以及第九章中的*使用正则表达式读取复杂格式*，*输入/输出、物理格式和逻辑布局*。

## 工作原理...

`subprocess`模块是 Python 程序运行在给定计算机上的其他程序的方式。`run()`函数为我们做了很多事情。

在 POSIX（如 Linux 或 Mac OS X）环境中，步骤类似于以下步骤：

+   为子进程准备`stdin`，`stdout`和`stderr`文件描述符。在这种情况下，我们安排父进程从子进程收集输出。子进程将`stdout`文件产生到一个共享缓冲区（在 Linux 术语中是一个管道），由父进程消耗。另一方面，`stderr`输出保持不变——子进程继承了父进程的相同连接，错误消息将显示在父进程使用的同一个控制台上。

+   调用`os.fork()`和`os.execl()`函数将当前进程分成父进程和子进程，然后启动子进程。

+   然后子进程运行，使用给定的`stdin`，`stdout`和`stderr`。

+   同时，父进程正在从子进程的管道中读取，同时等待子进程完成。

+   由于我们使用了`check=True`选项，非零状态被转换为异常。

## 还有更多...

我们将向这个脚本添加一个简单的总结功能。每个样本批次产生两行输出。输出文本通过表达式`list(l.strip() for l in output_bytes.splitlines())`分割成两行的序列。这将文本分割成行，并从每行中去除前导和尾随空格，留下稍微容易处理的文本。

总体脚本过滤了这些行，寻找以'Counter'开头的行。这些行中的每一行都是`Counter`对象的文本表示。在行上使用`eval()`函数将重建原始的`Counter`的副本。许多 Python 类定义都是这样的——`repr()`和`eval()`函数是彼此的反函数。`repr()`函数将对象转换为文本，`eval()`函数可以将文本转换回对象。这并不适用于所有类，但对于许多类来说是正确的。

我们可以创建各种`Counter`对象的总结。为了做到这一点，有助于有一个生成器来处理批次并产生最终的总结。

函数应该是这样的：

```py
    def process_batches(): 
        command_sequence = command_iter(2) 
        output_lines_sequence = command_output_iter(command_sequence) 
        for batch in output_lines_sequence: 
            for line in batch: 
                if line.startswith('Counter'): 
                    batch_counter = eval(line) 
                    yield batch_counter 

```

这将使用`command_iter()`函数创建处理命令。`command_output_iter()`将处理每个单独的命令，收集整个输出行集。

嵌套的`for`语句将检查每个批次的行列表。在每个列表中，它将检查每一行。以`Counter`开头的行将使用`eval()`函数进行评估。`Counter`对象的结果序列是这个生成器的输出。

我们可以使用这样的流程来总结`Counter`实例的序列：

```py
    total_counter = Counter() 
    for batch_counter in process_batches(): 
        print(batch_counter) 
        total_counter.update(batch_counter) 
    print("Total") 
    print(total_counter) 

```

我们将创建`Counter`来保存总数，`total_counter`。`process_batches()`将从处理的每个文件中产生单独的`Counter`实例。这些批次级别的对象用于更新`total_counter`。然后我们可以打印总数，显示所有文件中数据的聚合分布。

## 另请参阅

+   参见*包装和组合 CLI 应用程序*食谱，了解这个食谱的另一种方法。

+   这种自动化通常与其他 Python 处理结合在一起。请参见*为组合设计脚本*食谱。

+   目标通常是创建一个组合应用程序；参见*管理组合应用程序中的参数和配置*食谱。

![](img/614271.jpg)

# 控制复杂的步骤序列

在*将两个应用程序合并为一个*食谱中，我们探讨了将多个 Python 脚本合并为一个更长、更复杂操作的方法。在*包装和组合 CLI 应用程序*和*包装程序并检查输出*食谱中，我们探讨了使用 Python 包装非 Python 程序的方法。

我们如何有效地结合这些技术？我们能否使用 Python 创建更长、更复杂的操作序列？

## 准备工作

在*为组合设计脚本*食谱中，我们创建了一个应用程序，进行了一些处理，导致了一个相当复杂的结果的产生。在*使用日志进行控制和审计输出*食谱中，我们看了第二个应用程序，它建立在这些结果的基础上，创建了一个复杂的统计摘要。

总体流程如下：

1.  运行`ch13_r05`程序 100 次，创建 100 个中间文件。

1.  运行`ch13_r06`程序总结这些中间文件。

我们保持这个简单，这样就可以专注于涉及的 Python 编程。

对于这个食谱，我们假设这两个应用程序都不是用 Python 编写的。我们假装它们是用 Fortran 或 Ada 或其他与 Python 不直接兼容的语言编写的。

在*将两个应用程序合并为一个*食谱中，我们看了如何可以组合 Python 应用程序。当应用程序是用 Python 编写时，这是首选的方法。当应用程序不是用 Python 编写时，需要额外的工作。

这个配方使用了命令设计模式；这支持命令序列的扩展和修改。

## 如何做...

1.  我们将定义一个抽象的`Command`类。其他命令将被定义为子类。我们将将子进程处理推入此类定义以简化子类：

```py
        import subprocess 
        class Command: 
            def execute(self, options): 
                self.command = self.create_command(options) 
                results = subprocess.run(self.command, 
                    check=True, stdout=subprocess.PIPE) 
                self.output = results.stdout 
                return self.output 
            def create_command(self, options): 
                return ['echo', self.__class__.__name__, repr(self.options)] 

```

`execute()`方法首先通过创建 OS 级别的要执行的命令来工作。每个子类将为包装的命令提供不同的规则。一旦命令构建完成，`subprocess`模块的`run()`函数将处理此命令。

`create_command()` 方法构建由操作系统执行的命令的单词序列。通常，选项将用于自定义创建的命令参数。此方法的超类实现提供了一些调试信息。每个子类必须重写此方法以产生有用的输出。

1.  我们可以使用`Command`超类来定义一个命令来模拟游戏并创建样本：

```py
        import ch13_r05 

        class Simulate(Command): 
            def __init__(self, seed=None): 
                self.seed = seed 
            def execute(self, options): 
                if self.seed: 
                    os.environ['RANDOMSEED'] = str(self.seed) 
                super().execute(options) 
            def create_command(self, options): 
                return ['python3', 'ch13_r05.py`, 
                    '--samples', str(options.samples), 
                    '-o', options.game_file] 

```

在这种情况下，我们提供了对`execute()`方法的重写，以便这个类可以更改环境变量。这允许集成测试设置特定的随机种子，并确认结果与固定的预期值匹配。

`create_command()` 方法发出了用于执行`ch13_r05`命令的命令行的单词。这将数字值`options.samples`转换为字符串。

1.  我们还可以使用`Command`超类来定义一个命令来总结各种模拟过程：

```py
        import ch13_r06 

        class Summarize(Command): 
            def create_command(self, options): 
                return ['python3', 'ch13_r06.py', 
                    '-o', options.summary_file, 
                    ] + options.game_files 

```

在这种情况下，我们只实现了`create_command()`。此实现为`ch13_r06`命令提供了参数。

1.  鉴于这两个命令，整个主程序可以遵循*为组合设计脚本*配方的设计模式。我们需要收集选项，然后使用这些选项来执行这两个命令：

```py
        from argparse import Namespace 

        def demo(): 
            options = Namespace(samples=100, 
                game_file='x12.yaml', game_files=['x12.yaml'], 
                summary_file='y12.yaml') 
            step1 = Simulate() 
            step2 = Summarize() 
            step1.execute(options) 
            step2.execute(options) 

```

此演示函数`demo()`创建了一个带有可能来自命令行的参数的`Namespace`实例。它构建了两个处理步骤。最后，它执行每个步骤。

这种函数提供了一个执行一系列应用程序的高级脚本。它比 shell 要灵活得多，因为我们可以利用 Python 丰富的数据结构。因为我们使用 Python，我们也可以包括单元测试。

## 工作原理...

在这个配方中有两个相互交织的设计模式：

+   `Command`类层次结构

+   使用`subprocess.run()`函数包装外部命令

`Command`类层次结构的想法是将每个单独的步骤或操作变成一个共同的、抽象的超类的子类。在这种情况下，我们称这个超类为`Command`。这两个操作是`Command`类的子类。这确保我们可以为所有类提供共同的特性。

包装外部命令有几个考虑因素。一个主要问题是如何构建所需的命令行选项。在这种情况下，`run()`函数将使用一个单词列表，非常容易将文字字符串、文件名和数值组合成一个程序的有效选项集。另一个主要问题是如何处理 OS 定义的标准输入、标准输出和标准错误文件。在某些情况下，这些文件可以显示在控制台上。在其他情况下，应用程序可能会捕获这些文件以进行进一步的分析和处理。

这里的基本思想是分开两个考虑因素：

1.  执行命令的概述。这包括关于顺序、迭代、条件处理和可能对顺序进行更改的问题。这些是与用户故事相关的高级考虑因素。

1.  执行每个命令的详细信息。这包括命令行选项、使用的输出文件和其他 OS 级别的考虑因素。这些是更多关于实现细节的技术考虑因素。

将两者分开使得更容易实现或修改用户故事。对操作系统级别的考虑的更改不应该改变用户故事；处理可能会更快或使用更少的内存，但其他方面是相同的。同样，对用户故事的更改不应该破坏操作系统级别的考虑。

## 还有更多...

一系列复杂的步骤可能涉及一个或多个步骤的迭代。由于高级脚本是用 Python 编写的，添加迭代是用`for`语句完成的：

```py
    def process_i(options): 
        step1 = Simulate() 
        options.game_files = [] 
        for i in range(options.simulations): 
            options.game_file = 'game_{i}.yaml'.format_map(vars()) 
            options.game_files.append(options.game_file) 
            step1.execute(options) 
        step2 = Summarize() 
        step2.execute(options) 

```

此`process_i()`函数将多次处理`Simulate`步骤。它使用`simulations`选项来指定要运行多少次模拟。每次模拟将产生预期数量的样本。

此函数将为处理的每次迭代设置`game_file`选项的不同值。每个生成的文件名都将是唯一的，导致产生多个样本文件。文件列表也被收集到`game_files`选项中。

当执行下一步`Summarize`类时，它将具有适当的文件列表进行处理。分配给`options`变量的`Namespace`对象可用于跟踪全局状态变化，并将此信息提供给后续处理步骤。

### 构建有条件的处理。

由于高级编程是用 Python 编写的，因此很容易添加不基于封装的两个应用程序的附加处理。一个功能可能是可选的总结步骤。

例如，如果选项没有`summary_file`选项，则可以跳过处理。这可能会导致`process()`函数的一个版本看起来像这样：

```py
    def process_c(options): 
        step1 = Simulate() 
        step1.execute(options) 
        if 'summary_file' in options: 
            step2 = Summarize() 
            step2.execute(options) 

```

此`procees_c()`函数将有条件地处理`Summarize`步骤。如果有`summary_file`选项，它将执行第二步。否则，它将跳过总结步骤。

在这种情况下，以及前面的例子中，我们已经使用了 Python 编程功能来增强这两个应用程序。

## 另请参阅

+   通常，这些类型的处理步骤是为更大或更复杂的应用程序完成的。有关与更大更复杂的复合应用程序一起使用的更多食谱，请参阅*将两个应用程序合并为一个*和*在复合应用程序中管理参数和配置*。
