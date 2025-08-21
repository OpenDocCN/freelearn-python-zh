## 附录 B：打包和分发

打包和分发你的 Python 代码可能是一个复杂的，有时令人困惑的任务，特别是如果你的项目有很多依赖项或涉及比纯 Python 代码更奇特的组件。然而，对于许多情况来说，以标准方式使你的代码对他人可访问是非常直接的，我们将在本节中看到如何使用标准的`distutils`模块来做到这一点。`distutils`的主要优势是它包含在 Python 标准库中。对于远非最简单的打包要求，你可能会想要使用`setuptools`，它具有超出`distutils`的功能，但相应地更加令人困惑。

`distutils`模块允许你编写一个简单的 Python 脚本，它知道如何将你的 Python 模块安装到任何 Python 安装中，包括托管在虚拟环境中的安装。按照惯例，这个脚本被称为`setup.py`，并且存在于项目结构的顶层。然后可以执行此脚本来执行实际安装。

### 使用`distutils`配置包

让我们看一个`distutils`的简单例子。我们将为我们在第十一章中编写的`palindrome`模块创建一个基本的`setup.py`安装脚本。

我们想要做的第一件事是创建一个目录来保存我们的项目。让我们称之为`palindrome`：

```py
$ mkdir palindrome
$ cd palindrome

```

让我们把我们的`palindrome.py`复制到这个目录中：

```py
"""palindrome.py - Detect palindromic integers"""

import unittest

def digits(x):
    """Convert an integer into a list of digits.

 Args:
 x: The number whose digits we want.

 Returns: A list of the digits, in order, of ``x``.

 >>> digits(4586378)
 [4, 5, 8, 6, 3, 7, 8]
 """

    digs = []
    while x != 0:
        div, mod = divmod(x, 10)
        digs.append(mod)
        x = div
    digs.reverse()
    return digs

def is_palindrome(x):
    """Determine if an integer is a palindrome.

 Args:
 x: The number to check for palindromicity.

 Returns: True if the digits of ``x`` are a palindrome,
 False otherwise.

 >>> is_palindrome(1234)
 False
 >>> is_palindrome(2468642)
 True
 """
    digs = digits(x)
    for f, r in zip(digs, reversed(digs)):
        if f != r:
            return False
    return True

class Tests(unittest.TestCase):
    "Tests for the ``is_palindrome()`` function."
    def test_negative(self):
        "Check that it returns False correctly."
        self.assertFalse(is_palindrome(1234))

    def test_positive(self):
        "Check that it returns True correctly."
        self.assertTrue(is_palindrome(1234321))

    def test_single_digit(self):
        "Check that it works for single digit numbers."
        for i in range(10):
            self.assertTrue(is_palindrome(i))

if __name__ == '__main__':
    unittest.main()

```

最后让我们创建`setup.py`脚本：

```py
from distutils.core import setup

setup(
    name = 'palindrome',
    version = '1.0',
    py_modules  = ['palindrome'],

    # metadata
    author = 'Austin Bingham',
    author_email = 'austin@sixty-north.com',
    description = 'A module for finding palindromic integers.',
    license = 'Public domain',
    keywords = 'palindrome',
    )

```

文件中的第一行从`distutils.core`模块导入我们需要的功能，即`setup()`函数。这个函数完成了安装我们代码的所有工作，所以我们需要告诉它我们正在安装的代码。当然，我们通过传递给函数的参数来做到这一点。

我们告诉`setup()`的第一件事是这个项目的名称。在这种情况下，我们选择了`palindrome`，但你可以选择任何你喜欢的名称。不过，一般来说，最简单的方法是将名称与项目名称保持一致。

我们传递给`setup()`的下一个参数是版本。同样，这可以是任何你想要的字符串。Python 不依赖于版本遵循任何规则。

下一个参数`py_modules`可能是最有趣的。我们使用它来指定我们想要安装的 Python 模块。列表中的每个条目都是模块的名称，不包括`.py`扩展名。`setup()`将查找匹配的`.py`文件并安装它。所以，在我们的例子中，我们要求`setup()`安装`palindrome.py`，当然，这是我们项目中的一个文件。

我们在这里使用的其余参数都相当不言自明，主要是为了帮助人们正确使用你的模块，并知道如果他们遇到问题应该联系谁。

在我们开始使用我们的`setup.py`之前，我们首先需要创建一个虚拟环境，我们将在其中安装我们的模块。在你的`palindrome`目录中，创建一个名为`palindrome_env`的虚拟环境：

```py
$ python3 -m venv palindrome_env

```

当这完成后，激活新的环境。在 Linux 或 macOS 上，执行激活脚本：

```py
$ source palindrome_env/bin/activate

```

或者在 Windows 上直接调用脚本：

```py
> palindrome_env\bin\activate

```

### 使用`distutils`安装

现在我们有了`setup.py`，我们可以用它来做一些有趣的事情。我们可以做的第一件事，也许是最明显的，就是将我们的模块安装到我们的虚拟环境中！我们通过向`setup.py`传递`install`参数来实现这一点：

```py
(palindrome_env)$ python setup.py install
running install
running build
running build_py
copying palindrome.py -> build/lib
running install_lib
copying build/lib/palindrome.py -> /Users/sixty_north/examples/palindrome/palindrome_\
env/lib/python3.5/site-packages
byte-compiling /Users/sixty_north/examples/palindrome/palindrome_env/lib/python3.5/si\
te-packages/palindrome.py to palindrome.cpython-35.pyc
running install_egg_info
Writing /Users/sixty_north/examples/palindrome/palindrome_env/lib/python3.5/site-pack\
ages/palindrome-1.0-py3.5.egg-info

```

当调用`setup()`时，它会打印出几行来告诉你它的进度。对我们来说最重要的一行是它实际将`palindrome.py`复制到安装文件夹的地方：

```py
copying build/lib/palindrome.py -> /Users/sixty_north/examples/palindrome/palindrome_\
env/lib/python3.5/site-packages

```

Python 安装的`site-packages`目录是第三方包通常安装的地方，就像我们的包看起来安装成功了一样。

让我们通过运行 Python 来验证这一点，并看到我们的模块可以被导入。请注意，在我们这样做之前，我们要改变目录，否则当我们导入`palindrome`时，Python 会加载我们当前目录中的源文件：

```py
(palindrome_env)$ cd ..
(palindrome_env)$ python
Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 26 2016, 10:47:25)
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import palindrome
>>> palindrome.__file__
'/Users/sixty_north/examples/palindrome/palindrome_env/lib/python3.5/site-packages/pa\
lindrome.py'

```

在这里，我们使用模块的`__file__`属性来查看它是从哪里导入的，我们看到我们是从我们的虚拟环境的`site-packages`中导入的，这正是我们想要的。

退出 Python REPL 后，不要忘记切换回你的源目录：

```py
(palindrome_env)$ cd palindrome

```

### 使用`distutils`进行打包

`setup()`的另一个有用的特性是它可以创建各种类型的“分发”格式。它将把你指定的所有模块打包成易于分发给他人的包。你可以使用`sdist`命令来实现这一点（这是“源分发”的缩写）：

```py
(palindrome_env)$ python setup.py sdist --format zip
running sdist
running check
warning: check: missing required meta-data: url

warning: sdist: manifest template 'MANIFEST.in' does not exist (using default file li\
st)

warning: sdist: standard file not found: should have one of README, README.txt

writing manifest file 'MANIFEST'
creating palindrome-1.0
making hard links in palindrome-1.0...
hard linking palindrome.py -> palindrome-1.0
hard linking setup.py -> palindrome-1.0
creating dist
creating 'dist/palindrome-1.0.zip' and adding 'palindrome-1.0' to it
adding 'palindrome-1.0/palindrome.py'
adding 'palindrome-1.0/PKG-INFO'
adding 'palindrome-1.0/setup.py'
removing 'palindrome-1.0' (and everything under it)

```

如果我们查看，我们会发现这个命令创建了一个新的目录`dist`，其中包含了新生成的分发文件：

```py
(palindrome_env) $ ls dist
palindrome-1.0.zip

```

如果我们解压缩该文件，我们会看到它包含了我们项目的源代码，包括`setup.py`：

```py
(palindrome_env)$ cd dist
(palindrome_env)$ unzip palindrome-1.0.zip
Archive:  palindrome-1.0.zip
  inflating: palindrome-1.0/palindrome.py
  inflating: palindrome-1.0/PKG-INFO
  inflating: palindrome-1.0/setup.py

```

现在你可以把这个 zip 文件发送给任何想要使用你的代码的人，他们可以使用`setup.py`将其安装到他们的系统中。非常方便！

请注意，`sdist`命令可以生成各种类型的分发。要查看可用的选项，可以使用`--help-formats`选项：

```py
(palindrome_env) $ python setup.py sdist --help-formats
List of available source distribution formats:
  --formats=bztar  bzip2'ed tar-file
  --formats=gztar  gzip'ed tar-file
  --formats=tar    uncompressed tar file
  --formats=zip    ZIP file
  --formats=ztar   compressed tar file

```

这一部分只是简单地介绍了`distutils`的基础知识。你可以通过向`setup.py`传递`--help`来了解更多关于如何使用`distutils`的信息：

```py
(palindrome_env) $ python setup.py --help
Common commands: (see '--help-commands' for more)

  setup.py build      will build the package underneath 'build/'
  setup.py install    will install the package

Global options:
  --verbose (-v)      run verbosely (default)
  --quiet (-q)        run quietly (turns verbosity off)
  --dry-run (-n)      don't actually do anything
  --help (-h)         show detailed help message
  --command-packages  list of packages that provide distutils commands

Information display options (just display information, ignore any commands)
  --help-commands     list all available commands
  --name              print package name
  --version (-V)      print package version
  --fullname          print <package name>-<version>
  --author            print the author's name
  --author-email      print the author's email address
  --maintainer        print the maintainer's name
  --maintainer-email  print the maintainer's email address
  --contact           print the maintainer's name if known, else the author's
  --contact-email     print the maintainer's email address if known, else the
                      author's
  --url               print the URL for this package
  --license           print the license of the package
  --licence           alias for --license
  --description       print the package description
  --long-description  print the long package description
  --platforms         print the list of platforms
  --classifiers       print the list of classifiers
  --keywords          print the list of keywords
  --provides          print the list of packages/modules provided
  --requires          print the list of packages/modules required
  --obsoletes         print the list of packages/modules made obsolete

usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
   or: setup.py --help [cmd1 cmd2 ...]
   or: setup.py --help-commands
   or: setup.py cmd --help

```

对于许多简单的项目，你会发现我们刚刚介绍的几乎就是你需要了解的全部内容。
