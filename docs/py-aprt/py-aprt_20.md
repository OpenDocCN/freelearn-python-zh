## 附录 C：安装第三方软件包

Python 的打包历史曾经饱受困扰和混乱。幸运的是，情况已经稳定下来，一个名为`pip`的工具已经成为通用 Python 使用中包安装工具的明确领先者。对于依赖*Numpy*或*Scipy*软件包的数值或科学计算等更专业的用途，您应该考虑*Anaconda*作为`pip`的一个强大替代品。

### 介绍`pip`

在本附录中，我们将专注于`pip`，因为它是由核心 Python 开发人员正式认可的，并且具有开箱即用的支持。`pip`工具已包含在 Python 3.4 及以上版本中。对于较旧版本的 Python 3，您需要查找有关如何为您的平台安装`pip`的具体说明，因为您可能需要使用操作系统的软件包管理器，这取决于您最初安装 Python 的方式。开始的最佳地方是[Python 包装用户指南](https://packaging.python.org/tutorials/installing-packages/#install-pip-setuptools-and-wheel)。

`venv`模块还将确保`pip`安装到新创建的环境中。

`pip`工具是独立于标准库的其余部分开发的，因此通常有比随附 Python 分发的版本更近的版本可用。您可以使用`pip`来升级自身：

```py
$ pip install --upgrade pip

```

这是有用的，可以避免`pip`重复警告您不是最新版本。但请记住，这只会在当前 Python 环境中生效，这可能是一个虚拟环境。

### Python 包索引

`pip`工具可以在中央存储库（*Python 包索引*或*PyPI*，也被昵称为“奶酪店”）中搜索软件包，然后下载和安装它们以及它们的依赖项。您可以在[`pypi.python.org/pypi`](https://pypi.python.org/pypi)上浏览 PyPI。这是一种非常方便的安装 Python 软件的方式，因此了解如何使用它是很好的。

#### 使用`pip`安装

我们将演示如何使用`pip`来安装`nose`测试工具。`nose`是一种用于运行基于`unittest`的测试的强大工具，例如我们在第十章中开发的测试。它可以做的一个非常有用的事情是*发现*所有的测试并运行它们。这意味着您不需要将`unittest.main()`添加到您的代码中；您可以使用 nose 来查找和运行您的测试。

不过，首先我们需要做一些准备工作。让我们创建一个虚拟环境（参见附录 B），这样我们就不会意外地安装到系统 Python 安装中。使用`pyenv`创建一个虚拟环境，并激活它：

```py
$ python3 -m venv test_env
$ source activate test_env/bin/activate
(test_env) $

```

由于`pip`的更新频率远远超过 Python 本身，因此在任何新的虚拟环境中升级`pip`是一个良好的做法，所以让我们这样做。幸运的是，`pip`能够更新自身：

```py
(test_env) $ pip install --upgrade pip
Collecting pip
  Using cached pip-8.1.2-py2.py3-none-any.whl
Installing collected packages: pip
  Found existing installation: pip 8.1.1
    Uninstalling pip-8.1.1:
      Successfully uninstalled pip-8.1.1
Successfully installed pip-8.1.2

```

如果您不升级`pip`，每次使用它时都会收到警告，如果自上次升级以来已有新版本可用。

现在让我们使用`pip`来安装`nose`。`pip`使用子命令来决定要执行的操作，并且要安装模块，您可以使用`pip install package-name`：

```py
(test_env) $ pip install nose
Collecting nose
  Downloading nose-1.3.7-py3-none-any.whl (154kB)
    100% |████████████████████████████████| 163kB 2.1MB/s
Installing collected packages: nose
Successfully installed nose-1.3.7

```

如果成功，`nose`已准备好在我们的虚拟环境中使用。让我们通过尝试在 REPL 中导入它并检查安装路径来确认它是否可用：

```py
(test_env) $ python
Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 26 2016, 10:47:25)
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import nose
>>> nose.__file__
'/Users/sixty_north/.virtualenvs/test_env/lib/python3.5/site-packages/nose/__init__.p\
y'

```

除了安装模块外，`nose`还会在虚拟环境的`bin`目录中安装`nosetests`程序。为了真正锦上添花，让我们使用`nosetests`来运行第十一章中的`palindrome.py`中的测试：

```py
(test_env) $ cd palindrome
(test_env) $ nosetests palindrome.py
...
----------------------------------------------------------------------
Ran 3 tests in 0.001s

OK

```

### 使用`pip`安装本地软件包

您还可以使用`pip`从文件中安装本地软件包，而不是从 Python 包索引中安装。要做到这一点，请将打包分发的文件名传递给`pip install`。例如，在附录 B 中，我们展示了如何使用`distutils`构建所谓的源分发。要使用`pip`安装这个，做：

```py
(test_env) $ palindrome/dist
(test_env) $ pip install palindrome-1.0.zip

```

### 卸载软件包

使用`pip`安装软件包而不是直接调用源分发的`setup.py`的一个关键优势是，`pip`知道如何卸载软件包。要这样做，使用`uninstall`子命令：

```py
(test_env) $ pip uninstall palindrome-1.0.zip
Uninstalling palindrome-1.0:
Proceed (y/n)? y
  Successfully uninstalled palindrome-1.0

```
