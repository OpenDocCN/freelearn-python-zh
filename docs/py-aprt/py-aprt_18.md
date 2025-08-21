## 附录 A：虚拟环境

*虚拟环境*是一个轻量级的、独立的 Python 安装。虚拟环境的主要动机是允许不同的项目控制安装的 Python 包的版本，而不会干扰同一主机上安装的其他 Python 项目。虚拟环境包括一个目录，其中包含对现有 Python 安装的符号链接（Unix），或者是一个副本（Windows），以及一个空的`site-packages`目录，用于安装特定于该虚拟环境的 Python 包。虚拟环境的第二个动机是，用户可以在不需要系统管理员权限的情况下创建虚拟环境，这样他们可以轻松地在本地安装软件包。第三个动机是，不同的虚拟环境可以基于不同版本的 Python，这样可以更容易地在同一台计算机上测试代码，比如在 Python 3.4 和 Python 3.5 上。

如果你使用的是 Python 3.3 或更高版本，那么你的系统上应该已经安装了一个叫做`venv`的模块。你可以通过在命令行上运行它来验证：

```py
$ python3 -m venv
usage: venv [-h] [--system-site-packages] [--symlinks | --copies] [--clear]
            [--upgrade] [--without-pip]
            ENV_DIR [ENV_DIR ...]
venv: error: the following arguments are required: ENV_DIR

```

如果你没有安装`venv`，还有另一个工具叫做`virtualenv`，它的工作方式非常类似。你可以从[Python Package Index (PyPI)](https://pypi.python.org/pypi/virtualenv)获取它。我们将在附录 C 中解释如何从 PyPI 安装软件包。你可以使用`venv`或`virtualenv`，不过我们将在这里使用`venv`，因为它已经内置在最新版本的 Python 中。

### 创建虚拟环境

使用`venv`非常简单：你指定一个目录的路径，该目录将包含新的虚拟环境。该工具会创建新目录并填充它的安装内容：

```py
$ python3 -m venv my_python_3_5_project_env

```

### 激活虚拟环境

创建环境后，你可以通过在环境的`bin`目录中使用`activate`脚本来*激活*它。在 Linux 或 macOS 上，你需要`source`该脚本：

```py
$ source my_python_3_5_project_env/bin/activate

```

在 Windows 上运行它：

```py
> my_python_3_5_project_env\bin\activate

```

一旦你这样做，你的提示符将会改变，提醒你当前处于虚拟环境中：

```py
(my_python_3_5_project_env) $

```

运行`python`时执行的 Python 来自虚拟环境。实际上，使用虚拟环境是获得可预测的 Python 版本的最佳方式，而不是记住要使用`python`来运行 Python 2，`python3`来运行 Python 3。

一旦进入虚拟环境，你可以像平常一样工作，放心地知道包安装与系统 Python 和其他虚拟环境是隔离的。

### 退出虚拟环境

要离开虚拟环境，请使用`deactivate`命令，这将使你返回到激活虚拟环境的父 shell：

```py
(my_python_3_5_project_env) $ deactivate
$

```

### 其他用于虚拟环境的工具

如果你经常使用虚拟环境——我们建议你几乎总是在其中工作——管理大量的环境本身可能会变得有些繁琐。集成开发环境，比如*JetBrains’ PyCharm*，提供了出色的支持来创建和使用虚拟环境。在命令行上，我们推荐一个叫做[virtualenv wrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)的工具，它可以使在依赖不同虚拟环境的项目之间切换几乎变得轻而易举，一旦你做了一些初始配置。
