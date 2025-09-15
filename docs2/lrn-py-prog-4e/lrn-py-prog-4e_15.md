# 15

# 命令行应用程序

> 用户界面就像一个笑话。如果你不得不解释它，那就不是很好。
> 
> – 马丁·勒布兰

在本章中，我们将学习如何在 Python 中创建**命令行界面**（**CLI**）应用程序，也称为**命令行应用程序**。CLI 是一种用户界面，用户可以在控制台或终端中输入命令。显著的例子包括 macOS、Linux 和其他基于 UNIX 操作系统的**Bash**和**Zsh**外壳，以及 Windows 的**命令提示符**和**PowerShell**。CLI 应用程序是在这种命令行外壳环境中主要使用的应用程序。通过在 shell 中输入一个命令（可能后跟一些参数）来执行 CLI 应用程序。

虽然图形用户界面（**GUIs**）和 Web 应用程序更为流行，但命令行应用程序仍然有其位置。它们在开发者、系统管理员、网络管理员和其他技术用户中特别受欢迎。这种受欢迎的原因有几个。一旦你熟悉了所需的命令，你通常可以通过在 CLI 中输入命令而不是在 GUI 的菜单和按钮中点击来更快地完成任务。大多数 shell 还允许将一个命令的输出直接连接到另一个命令的输入。这被称为管道，它允许用户将简单的命令组合成数据处理管道以执行更复杂的任务。命令序列可以保存在脚本中，从而实现可重复性和自动化。通过提供确切的要输入的命令来执行任务，而不是解释如何导航 GUI 或 Web 界面，也更容易记录执行任务的说明。

命令行应用程序比图形或 Web 界面更快、更容易开发和维护。因此，开发团队有时更愿意将工具作为命令行应用程序来实现，以便减少构建内部工具的时间和精力，并更多地关注面向客户的功能。学习如何构建命令行应用程序也是学习如何构建更复杂软件（如 GUI 应用程序或分布式应用程序）的绝佳跳板。

在本章中，我们将创建一个命令行应用程序，用于与我们在上一章中学习的铁路 API 进行交互。我们将利用这个项目来探讨以下主题：

+   解析命令行参数

+   通过分解为子命令来构建 CLI 应用程序的结构

+   安全处理密码

我们将在本章结束时提供一些进一步资源建议，你可以在那里了解更多关于命令行应用程序的信息。

# 命令行参数

命令行应用程序的主要用户界面由可以传递给它的命令行参数组成。在我们开始探索铁路 CLI 项目之前，让我们简要地了解一下命令行参数以及 Python 提供用于处理它们的机制。

大多数应用程序接受各种**选项**（或**标志**）以及**位置参数**。一些应用程序由几个**子命令**组成，每个子命令都有自己的独特选项和位置参数。

## 位置参数

位置参数表示应用程序应操作的主要数据或对象。它们必须按特定顺序提供，通常不是可选的。例如，考虑以下命令：

```py
$ cp original.txt copy.txt 
```

此命令将创建一个名为 `copy.txt` 的 `original.txt` 文件的副本。必须提供两个位置参数（`original.txt` 和 `copy.txt`），改变它们的顺序将改变命令的含义。

## 选项

选项用于修改应用程序的行为。它们通常是可选的，通常由一个带有一个连字符的单个字母或带有两个连字符的单词组成。选项不需要出现在命令行的任何特定顺序或位置。它们甚至可以放在位置参数之后或之间。例如，许多应用程序接受 `-v` 或 `--verbose` 选项以启用详细输出。一些选项的行为类似于开关，仅通过其存在（或不存在）来简单地打开（或关闭）某些功能。其他选项需要额外的参数作为值。例如，考虑以下命令：

```py
$ grep -r --exclude '*.txt' hello . 
```

这将递归地进入当前目录，并在所有不以 `.txt` 结尾的文件中搜索字符串 `hello`。`-r` 选项使 grep 递归地搜索目录。如果没有此选项，当被要求搜索目录而不是常规文件时，它会退出并显示错误。`--exclude` 选项需要一个文件名模式（`'*.txt'`）作为参数，并导致 grep 排除与模式匹配的文件从搜索中。

在 Windows 上，选项传统上以正斜杠字符（`/`）作为前缀，而不是连字符。然而，许多现代和跨平台的应用程序使用连字符以与其他操作系统保持一致性。

## 子命令

复杂的应用程序通常被分为几个子命令。**Git** 版本控制系统是这一点的绝佳例子。例如，考虑以下命令

```py
$ git commit -m "Fix some bugs" 
```

以及

```py
$ git ls-files -m 
```

在这里，`commit` 和 `ls-files` 是 `git` 应用程序的子命令。`commit` 子命令创建一个新的提交，使用传递给 `-m` 选项（`"Fix some bugs"`）的文本作为提交信息。`ls-files` 命令显示 Git 仓库中文件的信息。`-m` 选项对 `ls-files` 指示命令仅显示尚未提交的修改的文件。

子命令的使用有助于结构化和组织应用程序界面，使用户更容易找到他们需要的功能。每个子命令也可以有自己的帮助信息，这意味着用户可以更容易地学习如何使用一个功能，而无需阅读整个应用程序的完整文档。它还促进了代码的模块化，这提高了可维护性，并允许在不修改现有代码的情况下添加新命令。

## 参数解析

Python 应用程序可以通过`sys.argv`访问传递给它们的命令行参数。让我们编写一个简单的脚本，只打印`sys.argv`的值：

```py
# argument_parsing/argv.py
import sys
print(sys.argv) 
```

当我们不传递任何参数运行此脚本时，输出如下所示：

```py
$ python argument_parsing/argv.py
['argument_parsing/argv.py'] 
```

如果我们传递一些参数，我们会得到以下结果：

```py
$ python argument_parsing/argv.py your lucky number is 13
['argument_parsing/argv.py', 'your', 'lucky', 'number', 'is', '13'] 
```

如您所见，`sys.argv`是一个字符串列表。第一个元素是运行应用程序使用的命令。其余元素包含命令行参数。

不接受任何选项的简单应用程序可以直接从`sys.argv`中提取位置参数。然而，对于接受选项的应用程序，参数解析逻辑可能会变得复杂。幸运的是，Python 标准库中的`argparse`模块提供了一个命令行参数解析器，这使得解析参数变得容易，而无需编写任何复杂的逻辑。

有几个第三方库可以作为`argparse`的替代方案。在本章中，我们不会介绍这些库，但我们将提供一些链接到一些最受欢迎的库。

例如，我们编写了一个脚本，它接受`name`和`age`作为位置参数，并打印出问候语。给定名字`Heinrich`和年龄`42`，它应该打印出`"Hi Heinrich. You are 42 years old."`。它接受一个自定义的问候语来代替`"Hi"`，通过`-g`选项。在命令行中添加`-r`或`--reverse`会导致在打印之前反转信息。

```py
# argument_parsing/greet.argparse.py
import argparse
def main():
    args = parse_arguments()
    print(args)
    msg = "{greet} {name}. You are {age} years old.".format(
        **vars(args)
    )
    if args.reverse:
        msg = msg[::-1]
    print(msg)
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Your name")
    parser.add_argument("age", type=int, help="Age")
    parser.add_argument("-r", "--reverse", action="store_true")
    parser.add_argument(
        "-g", default="Hi", help="Custom greeting", dest="greet"
    )
    return parser.parse_args()
if __name__ == "__main__":
    main() 
```

让我们更仔细地看看`parse_arguments()`函数。我们首先创建`ArgumentParser`类的实例。然后，我们通过调用`add_argument()`方法来定义我们接受的参数。我们从`name`和`age`位置参数开始，为每个参数提供帮助字符串，并指定`age`必须是一个整数。如果没有指定`type`，则参数将被解析为字符串。下一个参数是一个选项，可以在命令行上指定为`-r`或`--reverse`。最后一个参数是`"-g"`选项，默认值为`"``Hi"`。最后，我们调用解析器的`parse_args()`方法，以解析命令行参数。这将返回一个包含从命令行解析的参数值的`Namespace`对象。

`add_argument()` 函数的 `action` 关键字参数定义了解析器应该如何处理相应的命令行参数。如果没有指定，默认的行为是 `"store"`，这意味着将命令行提供的值存储为解析参数时返回的对象的属性。`"store_true"` 行为意味着该选项将被视为一个开关。如果它在命令行上存在，解析器将存储值 `True`；如果不存在，我们得到 `False`。`dest` 关键字参数指定了将存储值的属性的名称。如果没有指定 `dest`，解析器默认使用位置参数的名称，或者选项参数的第一个长选项字符串（去除前导 `--`）。如果只提供了一个短选项字符串，则使用该字符串（去除前导 `-`）。

让我们看看运行此脚本会发生什么。

```py
$ python argument_parsing/greet.argparse.py Heinrich -r 42
Namespace(name='Heinrich', age=42, reverse=True, greet='Hi')
.dlo sraey 24 era uoY .hcirnieH iH 
```

如果我们提供错误的参数，我们会得到一个 `usage` 消息，指示预期的参数是什么，以及一个错误消息告诉我们我们做错了什么：

```py
$ python argument_parsing/greet.argparse.py -g -r Heinrich 42
usage: greet.argparse.py [-h] [-r] [-g GREET] name age
greet.argparse.py: error: argument -g: expected one argument 
```

`usage` 消息提到了一个 `-h` 选项，我们没有添加。让我们看看它做了什么：

```py
$ python argument_parsing/greet.argparse.py -h
usage: greet.argparse.py [-h] [-r] [-g GREET] name age
positional arguments:
  name           Your name
  age            Age
options:
  -h, --help     show this help message and exit
  -r, --reverse
  -g GREET       Custom greeting 
```

解析器自动添加一个 `help` 选项，它显示了详细的用法信息，包括我们传递给 `add_argument()` 方法的 `help` 字符串。

为了帮助您欣赏 `argparse` 的强大功能，我们在本章的源代码中添加了一个不使用 `argparse` 的问候脚本版本。您可以在 `argument_parsing/greet.argv.py` 文件中找到它。

在本节中，我们只是刚刚触及了 `argparse` 的功能。在下一节中，我们将展示一些更高级的功能，当我们探索铁路 CLI 应用程序时。

# 为铁路 API 构建 CLI 客户端

现在我们已经涵盖了命令行参数解析的基础，我们可以开始着手构建一个更复杂的 CLI 应用程序了。您可以在本章源代码的项目目录下找到该应用程序的代码。让我们先看看 `project` 目录的内容。

```py
$ tree -a project
project
├── .env.example
├── railway_cli
│   ├── __init__.py
│   ├── __main__.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── schemas.py
│   ├── cli.py
│   ├── commands
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── base.py
│   │   └── stations.py
│   ├── config.py
│   └── exceptions.py
└── secrets
    ├── railway_api_email
    └── railway_api_password 
```

`.env.example` 文件是为创建铁路应用程序的 `.env` 配置文件而创建的模板。`secrets` 目录中的文件包含作为管理员用户与铁路 API 进行身份验证所需的凭证。

要成功运行本节中的示例，您需要运行 *第十四章，API 开发简介* 中的 API。您还需要创建一个名为 `.env` 的 `.env.example` 文件副本，并确保它包含 API 的正确 URL。

`railway_cli` 目录是铁路 CLI 应用的 Python 包。`api` 子包包含与铁路 API 交互的代码。在 `commands` 子包中，你可以找到应用程序子命令的实现。`exceptions.py` 模块定义了应用程序中可能发生的错误异常。`config.py` 包含处理全局配置设置的代码。驱动 CLI 应用的主函数位于 `cli.py` 中。`__main__.py` 模块是一个特殊的文件，使得包可执行。当使用类似以下命令执行包时

```py
$ python -m railway_cli 
```

Python 将加载并执行 `__main__.py` 模块。其内容如下：

```py
# project/railway_cli/__main__.py
from . import cli
cli.main() 
```

该模块所做的一切就是导入 `cli` 模块并调用 `cli.main()` 函数，这是 CLI 应用程序的主要入口点。

## 与铁路 API 交互

在我们深入研究命令行界面代码之前，我们想简要地讨论一下 API 客户端代码。我们不会详细查看代码，而是只提供一个高级概述。我们将深入研究代码作为你的练习。

在 `api` 子包中，你可以找到两个模块，`client.py` 和 `schemas.py`：

+   `schemas.py` 定义了 `pydantic` 模型来表示我们期望从 API 收到的对象（我们只为车站和火车定义了模型）。

+   `client.py` 包含三个类和一些辅助函数：

    +   `HTTPClient` 是一个用于发送 **HTTP** 请求的通用类。它是 `requests` 库中的 `Session` 对象的包装器。它具有与 API 使用的 HTTP 动词（`get`、`post`、`put` 和 `delete`）相对应的方法。这个类负责错误处理和从 API 响应中提取数据。

    +   `StationClient` 是一个用于与 API 的车站端点交互的更高级客户端。

    +   `AdminClient` 是一个用于处理管理端点的更高级客户端。它有一个使用 `users/authenticate` 端点进行用户认证的方法，以及一个通过 `admin/stations/{station_id}` 端点删除车站的方法。

## 创建命令行界面

我们将从一个名为 `cli.py` 的模块开始探索代码。我们将分块检查它，从 `main()` 函数开始。

```py
# project/railway_cli/cli.py
import argparse
import sys
from . import __version__, commands, config, exceptions
from .commands.base import Command
def main(cmdline: list[str] | None = None) -> None:
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args(cmdline)
    try:
        # args.command is expected to be a Command class
        command: Command = args.command(args)
        command.execute()
    except exceptions.APIError as exc:
        sys.exit(f"API error: {exc}")
    except exceptions.CommandError as exc:
        sys.exit(f"Command error: {exc}")
    except exceptions.ConfigurationError as exc:
        sys.exit(f"Configuration error: {exc}") 
```

我们首先导入标准库模块 `argparse` 和 `sys`。我们还从当前包中导入 `__version__`、`config`、`commands` 和 `exceptions`，以及从 `commands.base` 模块导入 `Command` 类。

对于 Python 模块和包来说，将版本号暴露在 `__version__` 名称下是一个常见的约定。它通常是一个字符串，并且通常定义在包的顶级 `__init__.py` 文件中。

在`main()`函数中，我们调用`get_arg_parser()`来获取一个`ArgumentParser`实例，并调用其`parse_args()`方法来解析命令行参数。我们期望返回的`Namespace`对象具有一个`command`属性，它应该是一个`Command`类。我们创建这个类的实例，并将解析后的参数传递给它。最后，我们调用命令的`execute()`方法。

我们通过调用`sys.exit()`来处理`APIError`、`CommandError`和`ConfigurationError`异常，以退出并显示针对发生异常类型的错误消息。这些是我们应用程序代码中抛出的唯一异常。如果发生任何其他意外错误，Python 将终止应用程序并向用户的控制台打印完整的异常跟踪信息。这可能看起来不太友好，但它将使调试变得容易得多。CLI 应用程序的用户也往往更技术熟练，因此他们不太可能因为异常跟踪信息而感到沮丧，与 GUI 或 Web 应用程序的用户相比。

注意，`main()`函数有一个可选参数`cmdline`，我们将其传递给`parse_args()`方法。如果`cmdline`是`None`，`parse_args()`将默认解析来自`sys.argv`的参数。然而，如果我们传递一个字符串列表，`parse_args()`将解析这个列表。以这种方式结构化代码对于单元测试特别有用，因为它允许我们在测试中避免操作全局的`sys.argv`。

我们将很快查看`Command`类以及如何设置参数解析器以返回`Command`类来执行。不过，让我们首先关注`get_arg_parser()`函数。

```py
# project/railway_cli/cli.py
def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=__package__,
        description="Commandline interface for the Railway API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    config.configure_arg_parser(parser)
    commands.configure_parsers(parser)
    return parser 
```

`get_arg_parser()`函数为应用程序创建并配置一个`ArgumentParser`实例。`prog`参数指定用于帮助信息的程序名称。通常，`argparse`从`sys.argv[0]`获取这个值；然而，对于通过`python -m package_name`执行的包，它是`__main__.py`，所以我们用包的名称覆盖它。`description`参数提供了程序在帮助信息中显示的简要描述。`formatter_class`确定帮助信息的格式化方式（`ArgumentDefaultsHelpFormatter`将所有选项的默认值添加到帮助信息中）。我们使用`"version"`操作添加一个`"-V"`或`"--version"`选项，如果命令行中遇到此选项，将打印版本信息并退出。最后，我们调用`config.configure_arg_parser()`和`commands.configure_parsers()`函数来进一步配置解析器，然后再返回它。

Python 的导入系统将每个导入模块的`__package__`属性设置为它所属的包名。

在接下来的几节中，我们将查看`config`和`commands`模块的命令行参数配置，首先是`config`。

## 配置文件和秘密

除了命令行参数之外，许多 CLI 应用程序也会从配置文件中读取设置。配置文件通常用于 API URL 等设置，这些设置通常不会从一个应用程序的调用改变到下一个调用。每次运行应用程序时都要求用户在命令行上提供这些设置会相当繁琐。

配置文件的另一个常见用途是提供密码和其他秘密。将密码作为命令行参数提供并不被认为是好的安全实践，因为在大多数操作系统中，任何已登录的用户都可以看到任何正在运行的应用程序的完整命令行。大多数 shell 也具有命令历史功能，这可能会暴露作为命令行参数传递的密码。在配置文件或专门的秘密文件中提供密码要安全得多，这些文件用于配置秘密，例如密码。文件名对应于秘密的名称，文件内容是秘密本身。

非常重要的是要记住，将秘密与我们的代码一起存储永远是不安全的。特别是如果您使用版本控制系统，如 Git 或 Mercurial，请务必小心，不要将任何秘密与源代码一起提交。

在`railway_cli`应用程序中，`config`模块负责处理配置文件和秘密。我们使用`pydantic-settings`库，我们在*第十四章，API 开发简介*中已经遇到过的库，来管理配置。让我们分块查看代码。

```py
# project/railway_cli/config.py
import argparse
from getpass import getpass
from pydantic import EmailStr, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from .exceptions import ConfigurationError
class Settings(BaseSettings):
    url: str
    secrets_dir: str | None = None
class AdminCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="railway_api_")
    email: EmailStr
    password: SecretStr = Field(
        default_factory=lambda: SecretStr(
            getpass(prompt="Admin Password: ")
        )
    ) 
```

在文件顶部的导入之后，我们有两个类：`Settings`和`AdminCredentials`。两者都继承自`pydantic_settings.BaseSettings`。`Settings`类定义了`railway_cli`应用程序的一般配置：

+   `url`：用于配置 railway API URL。

+   `secrets_dir`：可以用来配置包含秘密文件的目录的路径。API 管理员凭据将从该目录中的秘密文件中加载。

`AdminCredentials`类定义了作为管理员用户进行 API 身份验证所需的凭据。`SettingsConfigDict`的`env_prefix`参数将在查找秘密目录中的值时添加到字段名称前。例如，`password`将在名为`railway_api_password`的文件中查找。`AdminCredentials`类包含以下字段：

+   `email`：将包含用于身份验证的管理员电子邮件地址。我们使用`pydantic.EmailStr`类型来确保它包含有效的电子邮件地址。

+   `password`：将包含管理员密码。我们使用 `pydantic.SecretStr` 类型来确保当打印（例如，在应用程序日志中）时，值将被星号屏蔽。如果类实例化时（通过秘密文件或类构造函数的参数）没有提供值，`pydantic` 将调用通过 `Field` 函数的 `default_factory` 参数提供的函数。我们使用这个来调用标准库中的 `getpass` 函数，以安全地提示用户输入管理员密码。

在这些类定义下方，你可以找到 `configure_arg_parser()` 函数。现在让我们看看它：

```py
# project/railway_cli/config.py
def configure_arg_parser(parser: argparse.ArgumentParser) -> None:
    config_group = parser.add_argument_group(
        "configuration",
        description="""The API URL must be set in the
        configuration file. The admin email and password should be
        configured via secrets files named email and password in a
        secrets directory.""",
    )
    config_group.add_argument(
        "--config-file",
        help="Load configuration from a file",
        default=".env",
    )
    config_group.add_argument(
        "--secrets-dir",
        help="""The secrets directory. Can also be set via the
        configuration file.""",
    ) 
```

我们使用参数解析器的 `add_argument_group()` 方法创建一个名为 `"configuration"` 的参数组，并给它一个 `description` 描述。我们向这个组添加了允许用户指定配置文件名和秘密目录的选项。请注意，参数组不会影响参数的解析或返回方式。它仅仅意味着这些参数将在帮助信息中的公共标题下分组。

为了简化起见，我们在这个示例中将默认配置文件路径设置为 `.env`。然而，使用应用程序运行的平台的标准配置文件位置被认为是最佳实践。`platformdirs` 库（[`platformdirs.readthedocs.io`](https://platformdirs.readthedocs.io)）在这方面特别有帮助。

`config.py` 模块的最后一部分包括用于检索设置和管理员凭证的辅助函数：

```py
# project/railway_cli/config.py
def get_settings(args: argparse.Namespace) -> Settings:
    try:
        return Settings(_env_file=args.config_file)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc
def get_admin_credentials(
    args: argparse.Namespace, settings: Settings
) -> AdminCredentials:
    secrets_dir = args.secrets_dir
    if secrets_dir is None:
        secrets_dir = settings.secrets_dir
    try:
        return AdminCredentials(_secrets_dir=secrets_dir)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc 
```

`get_settings()` 函数创建并返回 `Settings` 类的一个实例。`_env_file=args.config_file` 参数告诉 `pydantic-settings` 从通过 `--config-file` 命令行选项指定的文件（默认为 `.env`）中加载设置。`get_admin_credentials()` 函数创建并返回 `AdminCredentials` 类的一个实例。类中的 `_secrets_dir` 参数指定了 `pydantic-settings` 将在其中查找凭证的秘密目录。如果命令行上设置了 `--secrets-dir` 选项，我们将使用该选项；否则，使用 `settings.secrets_dir`。如果这也是 `None`，则不会使用秘密目录。

## 创建子命令

铁路 API 有用于列出车站、创建车站、获取出发时间等单独的端点。在我们的应用程序中拥有类似的架构是有意义的。有许多方法可以组织子命令的代码。在这个应用程序中，我们选择使用面向对象的方法。每个命令都实现为一个类，其中包含一个配置参数解析器的方法，以及一个执行命令的方法。所有命令都是 `Command` 基类子类。你可以在 `commands/base.py` 模块中找到它：

```py
# project/railway_cli/commands/base.py
import argparse
from typing import ClassVar
from ..api.client import HTTPClient
from ..config import get_settings
class Command:
    name: ClassVar[str]
    help: ClassVar[str]
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.settings = get_settings(args)
        self.api_client = HTTPClient(self.settings.url)
    @classmethod
    def configure_arg_parser(
        cls, parser: argparse.ArgumentParser
    ) -> None:
        raise NotImplementedError
    def execute(self) -> None:
        raise NotImplementedError 
```

如您所见，`Command` 类是一个普通类。`name` 和 `help` 上的 `ClassVar` 注解表明这些是期望作为类属性，而不是实例属性。`__init__()` 方法接受一个 `argparse.Namespace` 对象，并将其分配给 `self.args`。它调用 `get_settings()` 来加载配置文件。在返回之前，它还创建了一个 `HTTPClient` 对象（来自 `api/client.py`）并将其分配给 `self.api_client`。

`configure_arg_parser()` 类方法和 `execute()` 方法在调用时都会抛出 `NotImplementedError`，这意味着子类需要用它们自己的实现覆盖这些方法。

为了设置子命令的参数解析，我们需要为每个子命令创建一个解析器，并通过调用 `Command` 类的 `configure_arg_parser()` 类方法来配置它。`commands.configure_parsers()` 函数负责这个过程。现在让我们看看这个。

```py
# project/railway_cli/commands/__init__.py
import argparse
from .admin import admin_commands
from .base import Command
from .stations import station_commands
def configure_parsers(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(
        description="Available commands", required=True
    )
    command: type[Command]
    for command in [*admin_commands, *station_commands]:
        command_parser = subparsers.add_parser(
            command.name, help=command.help
        )
        command.configure_arg_parser(command_parser)
        **command_parser.set_defaults(command=command)** 
```

`parser.add_subparsers()` 方法返回一个对象，可以用来将子命令解析器附加到主解析器。`description` 参数用于生成子命令的帮助文本，`required=True` 确保如果命令行上没有提供子命令，解析器将产生错误。

我们遍历 `admin_commands` 和 `station_commands` 列表，并为每个创建一个子解析器。`add_parser()` 方法期望子命令的 `name` 和可以传递给 `ArgumentParser` 类的任何参数。它返回一个新的 `ArgumentParser` 实例，我们将其传递给 `command.configure_arg_parser()` 类方法。请注意，`command: type[Command]` 类型注解表明我们期望 `admin_commands` 和 `station_commands` 的所有元素都是 `Command` 的子类。

`set_defaults()` 方法允许我们独立于命令行上的内容，在解析器返回的命名空间上设置属性。我们使用这个方法将每个子解析器的 `command` 属性设置为相应的 `Command` 子类。`parse_args()` 方法返回的 `Namespace` 对象将只包含来自恰好一个子解析器的属性（与命令行上提供的子命令相对应）。因此，当我们调用 `cli.main()` 函数中的 `args.command(args=args)` 时，我们保证会得到用户选择的命令类的实例。

现在我们已经将配置参数解析器的所有代码组合在一起，我们可以查看当我们使用 `-h` 选项运行应用程序时生成的帮助文本。

```py
$ python -m railway_cli -h
usage: railway_cli [-h] [-V] [--config-file CONFIG_FILE]
                   [--secrets-dir SECRETS_DIR]
                   {admin-delete-station,get-station,...}
                   ...
Commandline interface for the Railway API
options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
configuration:
  The API URL must be set in the configuration file. ...
  --config-file CONFIG_FILE
                        Load configuration from a file (default:
                        .env)
  --secrets-dir SECRETS_DIR
                        The secrets directory. Can also be set
                        via the configuration file. (default:
                        None)
subcommands:
  Available commands
  {admin-delete-station,get-station,list-stations,...}
    admin-delete-station
                        Delete a station
    get-station         Get a station
    list-stations       List stations
    create-station      Create a station
    update-station      Update an existing station
    get-arrivals        Get arrivals for a station
    get-departures      Get departures for a station 
```

我们已经缩减了一些输出并删除了空白行，但您可以看到有一个`usage`摘要显示了如何使用该命令，然后是我们在创建参数解析器时设置的`description`。接着是全局的`options`部分，包括`-h`或`--help`选项和`-V`或`--version`选项。接下来是`configuration`部分，其中包含了在`config.configure_arg_parser()`函数中配置的描述和选项。最后，我们有一个`subcommands`部分，其中包含了我们在`commands.configure_parsers()`中传递给参数解析器`add_subparsers()`方法的描述，以及所有可用子命令的列表和为每个子命令设置的帮助字符串。

我们已经看到了子命令的基类和配置参数解析器以与子命令一起工作的代码。现在让我们看看子命令的实现。

## 实现子命令

子命令解析器是完全独立的，因此我们可以实现子命令而无需担心它们的命令行选项可能会相互冲突。我们只需要确保命令名称是唯一的。这意味着我们可以通过添加命令来扩展应用程序，而无需修改任何现有代码。我们为这个应用程序选择基于类的处理方法，这使得添加命令变得容易。我们只需创建一个新的`Command`子类，定义其`name`和`help`文本，并实现`configure_arg_parser()`和`execute()`方法。作为一个例子，让我们看看`create-station`命令的代码。

```py
# project/railway_cli/commands/stations.py
class CreateStation(Command):
    name = "create-station"
    help = "Create a station"
    @classmethod
    def configure_arg_parser(
        cls, parser: argparse.ArgumentParser
    ) -> None:
        parser.add_argument(
            "--code", help="The station code", required=True
        )
        parser.add_argument(
            "--country", help="The station country", required=True
        )
        parser.add_argument(
            "--city", help="The station city", required=True
        )
    def execute(self) -> None:
        station_client = StationClient(self.api_client)
        station = station_client.create(
            code=self.args.code,
            country=self.args.country,
            city=self.args.city,
        )
        print(station) 
```

注意，我们没有在这里重现`commands/stations.py`模块顶部的导入。如您所见，命令的代码相当简单。`configure_arg_parser()`类方法为站点代码、城市和国家添加了选项。

注意，这三个选项都被标记为`required`。Python 的`argparse`文档不鼓励使用`required`选项；然而，在某些情况下，它可以导致更用户友好的界面。如果一个命令需要超过两个具有不同意义的参数，用户可能很难记住正确的顺序。使用选项意味着顺序不重要，并且每个参数的含义立即显而易见。

让我们看看运行此命令时会发生什么。首先，使用`-h`选项查看帮助信息：

```py
$ python -m railway_cli create-station -h
usage: railway_cli create-station [-h] --code CODE --country
                                  COUNTRY --city CITY
options:
  -h, --help         show this help message and exit
  --code CODE        The station code
  --country COUNTRY  The station country
  --city CITY        The station city 
```

帮助信息清楚地显示了如何使用该命令。现在我们可以创建一个站点：

```py
$ python -m railway_cli create-station --code LSB --city Lisbon \
    --country Portugal
id=12 code='LSB' country='Portugal' city='Lisbon' 
```

输出显示站点已成功创建并分配了`id` `12`。

这就带我们结束了对铁路 CLI 应用程序的探索。

# 其他资源和工具

我们将本章以一些链接结束，这些链接指向您可以了解更多信息和一些用于开发 CLI 应用程序的有用库：

+   尽管我们已经尽力使本章内容尽可能全面，但 `argparse` 模块的功能远不止我们在这里展示的。不过，官方文档在 [`docs.python.org/3/library/argparse.html`](https://docs.python.org/3/library/argparse.html) 上非常优秀。

+   如果 `argparse` 不符合您的喜好，还有几个第三方库可用于命令行参数解析。我们建议您都尝试一下：

    +   **Click** 是迄今为止最受欢迎的第三方 CLI 库。除了命令行解析外，它还提供了创建交互式应用程序（如输入提示）和生成彩色输出的功能。您可以在 [`click.palletsprojects.com`](https://click.palletsprojects.com) 了解更多信息。

    +   **Typer** 由与 FastAPI 相同的开发者创建。它的目标是将 FastAPI 应用于 API 开发的相同原则应用于 CLI 开发。您可以在 [`typer.tiangolo.com/`](https://typer.tiangolo.com/) 了解更多信息。

+   **Pydantic Settings**，我们在本章和*第十四章，API 开发简介*中用于配置管理，也支持解析命令行参数。有关更多信息，请参阅[`docs.pydantic.dev/latest/concepts/pydantic_settings/#command-line-support`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/#command-line-support)。

+   大多数现代 shell 都支持可编程的命令行自动补全。提供命令行补全可以使您的 CLI 应用程序更容易使用。`argcomplete` 库（[`kislyuk.github.io/argcomplete/`](https://kislyuk.github.io/argcomplete/)）为使用 `argparse` 处理命令行参数的应用程序提供了 `bash` 和 `zsh` shell 的命令行补全功能。

+   《*命令行界面指南*》（[`clig.dev/`](https://clig.dev/)）是一个全面的开源资源，提供了设计用户友好的命令行界面的优秀建议。

# 摘要

在本章中，我们通过为我们在*第十四章，API 开发简介*中创建的铁路 API 开发 CLI 客户端来学习命令行应用程序。我们学习了如何使用标准库 `argparse` 模块解析命令行参数。我们探讨了如何通过使用子命令来构建 CLI 应用程序界面，并看到这如何帮助我们构建易于维护和扩展的模块化应用程序。我们以一些链接到其他用于 Python CLI 应用程序开发的库以及一些您可以了解更多信息的资源来结束本章。

与命令行应用程序一起工作是练习本书所学技能的绝佳方式。我们鼓励您研究本章的代码，通过添加更多命令来扩展它，并通过添加日志和测试来改进它。

在下一章中，我们将学习如何打包和发布 Python 应用程序。

# 加入我们的 Discord 社区

加入我们的社区 Discord 空间，与作者和其他读者进行讨论：

`discord.com/invite/uaKmaz7FEC`

![img](img/QR_Code119001106417026468.png)
