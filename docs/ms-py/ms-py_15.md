# 第十五章：包装-创建您自己的库或应用程序

到目前为止，这些章节已经涵盖了如何编写、测试和调试 Python 代码。有了这一切，只剩下一件事，那就是打包和分发您的 Python 库/和应用程序。为了创建可安装的包，我们将使用 Python 这些天捆绑的`setuptools`包。如果您以前创建过包，您可能还记得`distribute`和`distutils2`，但非常重要的是要记住，这些都已经被`setuptools`和`distutils`取代，您不应该再使用它们！

我们可以使用`setuptools`打包哪些类型的程序？我们将向您展示几种情况：

+   常规包

+   带有数据的包

+   安装可执行文件和自定义`setuptools`命令

+   在包上运行测试

+   包含 C/C++扩展的包

# 安装包

在我们真正开始之前，重要的是要知道如何正确安装包。至少有四种不同的选项可以安装包。第一种最明显的方法是使用普通的`pip`命令：

```py
pip install package

```

这也可以通过直接使用`setup.py`来实现：

```py
cd package
python setup.py install

```

这将在您的 Python 环境中安装包，如果您使用它，可能是`virtualenv`/`venv`，否则是全局环境。

然而，对于开发来说，这是不推荐的。要测试您的代码，您需要为每个测试重新安装包，或者修改 Python 的`site-packages`目录中的文件，这意味着它将位于您的修订控制系统之外。这就是开发安装的用途；它们不是将包文件复制到 Python 包目录中，而是在`site-packages`目录中安装到实际包位置的路径的链接。这使您可以修改代码，并立即在运行的脚本和应用程序中看到结果，而无需在每次更改后重新安装代码。

与常规安装一样，`pip`和`setup.py`版本都可用：

```py
pip install –e package_directory

```

以及`setup.py`版本：

```py
cd package_directory
python setup.py develop

```

# 设置参数

之前的章节实际上已经向我们展示了一些示例，但让我们重申和回顾最重要的部分实际上是做什么。在整个本章中，您将使用的核心功能是`setuptools.setup`。

### 注意

对于最简单的包，Python 捆绑的`distutils`包将足够，但无论如何我推荐`setuptools`。`setuptools`包具有许多`distutils`缺乏的出色功能，并且几乎所有 Python 环境都会有`setuptools`可用。

在继续之前，请确保您拥有最新版本的`pip`和`setuptools`：

```py
pip install -U pip setuptools

```

### 注意

`setuptools`和`distutils`包在过去几年中发生了重大变化，2014 年之前编写的文档/示例很可能已经过时。小心不要实现已弃用的示例，并跳过使用`distutils`的任何文档/示例。

既然我们已经具备了所有先决条件，让我们创建一个包含最重要字段的示例，并附带内联文档：

```py
import setuptools

if __name__ == '__main__':
    setuptools.setup(
        name='Name',
        version='0.1',

        # This automatically detects the packages in the specified
        # (or current directory if no directory is given).
        packages=setuptools.find_packages(),

        # The entry points are the big difference between
        # setuptools and distutils, the entry points make it
        # possible to extend setuptools and make it smarter and/or
        # add custom commands.
        entry_points={

            # The following would add: python setup.py
            # command_name
            'distutils.commands': [
                'command_name = your_package:YourClass',
            ],

            # The following would make these functions callable as
            # standalone scripts. In this case it would add the
            # spam command to run in your shell.
            'console_scripts': [
                'spam = your_package:SpamClass',
            ],
        },

        # Packages required to use this one, it is possible to
        # specify simply the application name, a specific version
        # or a version range. The syntax is the same as pip
        # accepts.
        install_requires=['docutils>=0.3'],

        # Extra requirements are another amazing feature of
        # setuptools, it allows people to install extra
        # dependencies if you are interested. In this example
        # doing a "pip install name[all]" would install the
        # python-utils package as well.
        extras_requires={
            'all': ['python-utils'],
        },

        # Packages required to install this package, not just for
        # running it but for the actual install. These will not be
        # installed but only downloaded so they can be used during
        # the install. The pytest-runner is a useful example:
        setup_requires=['pytest-runner'],

        # The requirements for the test command. Regular testing
        # is possible through: python setup.py test The Pytest
        # module installs a different command though: python
        # setup.py pytest
        tests_require=['pytest'],

        # The package_data, include_package_data and
        # exclude_package_data arguments are used to specify which
        # non-python files should be included in the package. An
        # example would be documentation files.  More about this
        # in the next paragraph
        package_data={
            # Include (restructured text) documentation files from
            # any directory
            '': ['*.rst'],
            # Include text files from the eggs package:
            'eggs': ['*.txt'],
        },

        # If a package is zip_safe the package will be installed
        # as a zip file. This can be faster but it generally
        # doesn't make too much of a difference and breaks
        # packages if they need access to either the source or the
        # data files. When this flag is omitted setuptools will
        # try to autodetect based on the existance of datafiles
        # and C extensions. If either exists it will not install
        # the package as a zip. Generally omitting this parameter
        # is the best option but if you have strange problems with
        # missing files, try disabling zip_safe.
        zip_safe=False,

        # All of the following fileds are PyPI metadata fields.
        # When registering a package at PyPI this is used as
        # information on the package page.
        author='Rick van Hattem',
        author_email='wolph@wol.ph',

        # This should be a short description (one line) for the
        # package
        description='Description for the name package',

        # For this parameter I would recommend including the
        # README.rst

        long_description='A very long description',
        # The license should be one of the standard open source
        # licenses: https://opensource.org/licenses/alphabetical
        license='BSD',

        # Homepage url for the package
        url='https://wol.ph/',
    )
```

这是相当多的代码和注释，但它涵盖了您在现实生活中可能遇到的大多数选项。这里讨论的最有趣和多功能的参数将在接下来的各个部分中单独介绍。

附加文档可以在`pip`和`setuptools`文档以及 Python 包装用户指南中找到：

+   [`pythonhosted.org/setuptools/`](http://pythonhosted.org/setuptools/)

+   [`pip.pypa.io/en/stable/`](https://pip.pypa.io/en/stable/)

+   [`python-packaging-user-guide.readthedocs.org/en/latest/`](http://python-packaging-user-guide.readthedocs.org/en/latest/)

# 包

在我们的例子中，我们只是使用`packages=setuptools.find_packages()`。在大多数情况下，这将工作得很好，但重要的是要理解它的作用。`find_packages`函数会查找给定目录中的所有目录，并在其中有`__init__.py`文件的情况下将其添加到列表中。因此，你通常可以使用`['your_package']`代替`find_packages()`。然而，如果你有多个包，那么这往往会变得乏味。这就是`find_packages()`有用的地方；只需指定一些包含参数（第二个参数）或一些排除参数（第三个参数），你就可以在项目中拥有所有相关的包。例如：

```py
packages = find_packages(exclude=['tests', 'docs'])

```

# 入口点

`entry_points`参数可以说是`setuptools`最有用的功能。它允许你向`setuptools`中的许多东西添加钩子，但最有用的两个是添加命令行和 GUI 命令的可能性，以及扩展`setuptools`命令。命令行和 GUI 命令甚至会在 Windows 上转换为可执行文件。第一节中的例子已经演示了这两个功能：

```py
entry_points={
    'distutils.commands': [
        'command_name = your_package:YourClass',
    ],
    'console_scripts': [
        'spam = your_package:SpamClass',
    ],
},
```

这个演示只是展示了如何调用函数，但没有展示实际的函数。

## 创建全局命令

第一个，一个简单的例子，没有什么特别的；只是一个作为常规`main`函数被调用的函数，在这里你需要自己指定`sys.argv`（或者更好的是使用`argparse`）。这是`setup.py`文件：

```py
import setuptools

if __name__ == '__main__':
    setuptools.setup(
        name='Our little project',
        entry_points={
            'console_scripts': [
                'spam = spam.main:main',
            ],
        },
    )
```

当然，这里有`spam/main.py`文件：

```py
import sys

def main():
    print('Args:', sys.argv)
```

一定不要忘记创建一个`spam/__init__.py`文件。它可以是空的，但它需要存在，以便 Python 知道它是一个包。

现在，让我们试着安装这个包：

```py
# pip install -e .
Installing collected packages: Our-little-project
 **Running setup.py develop for Our-little-project
Successfully installed Our-little-project
# spam 123 abc
Args: ['~/envs/mastering_python/bin/spam', '123', 'abc']

```

看，创建一个在常规命令行 shell 中安装的`spam`命令是多么简单！在 Windows 上，它实际上会给你一个可执行文件，该文件将被添加到你的路径中，但无论在哪个平台上，它都将作为一个可调用的独立可执行文件。

## 自定义 setup.py 命令

编写自定义的`setup.py`命令非常有用。一个例子是`sphinx-pypi-upload-2`，我在所有的包中都使用它，它是我维护的`unmaintained sphinx-pypi-upload`包的分支。这是一个使构建和上传 Sphinx 文档到 Python 包索引变得非常简单的包，当分发你的包时非常有用。使用`sphinx-pypi-upload-2`包，你可以做以下操作（我在分发我维护的任何包时都会这样做）：

```py
python setup.py sdist bdist_wheel upload build_sphinx upload_sphinx

```

这个命令会构建你的包并将其上传到 PyPI，并构建 Sphinx 文档并将其上传到 PyPI。

但你当然想看看这是如何工作的。首先，这是我们`spam`命令的`setup.py`：

```py
import setuptools

if __name__ == '__main__':
    setuptools.setup(
        name='Our little project',
        entry_points={
            'distutils.commands': [
                'spam = spam.command:SpamCommand',
            ],
        },
    )
```

其次，`SpamCommand`类。基本要点是继承`setuptools.Command`并确保实现所有需要的方法。请注意，所有这些方法都需要实现，但如果需要，可以留空。这是`spam/command.py`文件：

```py
import setuptools

class SpamCommand(setuptools.Command):
    description = 'Make some spam!'
# Specify the commandline arguments for this command here. This
# parameter uses the getopt module for parsing'
    user_options = [
        ('spam=', 's', 'Set the amount of spams'),
    ]

    def initialize_options(self):
# This method can be used to set default values for the
# options. These defaults can be overridden by
# command-line, configuration files and the setup script
# itself.
        self.spam = 3

    def finalize_options(self):
# This method allows you to override the values for the
# options, useful for automatically disabling
# incompatible options and for validation.
        self.spam = max(0, int(self.spam))

    def run(self):
        # The actual running of the command.
        print('spam' * self.spam)
```

执行它非常简单：

```py
# pip install -e .
Installing collected packages: Our-little-project
 **Running setup.py develop for Our-little-project
Successfully installed Our-little-project-0.0.0
# python setup.py --help-commands
[...]
Extra commands:
 **[...]
 **spam              Make some spam!
 **test              run unit tests after in-place build
 **[...]

usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
 **or: setup.py --help [cmd1 cmd2 ...]
 **or: setup.py --help-commands
 **or: setup.py cmd –help

# python setup.py --help spam
Common commands: (see '--help-commands' for more)

[...]

Options for 'SpamCommand' command:
 **--spam (-s)  Set the amount of spams

usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
 **or: setup.py --help [cmd1 cmd2 ...]
 **or: setup.py --help-commands
 **or: setup.py cmd --help

# python setup.py spam
running spam
spamspamspam
# python setup.py spam -s 5
running spam
spamspamspamspamspam

```

实际上只有很少的情况下你会需要自定义的`setup.py`命令，但这个例子仍然很有用，因为它目前是`setuptools`的一个未记录的部分。

# 包数据

在大多数情况下，你可能不需要包含包数据，但在需要数据与你的包一起的情况下，有一些不同的选项。首先，重要的是要知道默认情况下包含在你的包中的文件有哪些：

+   包目录中的 Python 源文件递归

+   `setup.py`和`setup.cfg`文件

+   测试：`test/test*.py`

+   在`examples`目录中的所有`*.txt`和`*.py`文件

+   在根目录中的所有`*.txt`文件

所以在默认值之后，我们有了第一个解决方案：`setup`函数的`package_data`参数。它的语法非常简单，一个字典，其中键是包，值是要包含的模式：

```py
package_data = {
    'docs': ['*.rst'],
}
```

第二种解决方案是使用`MANIFEST.in`文件。该文件包含要包括、排除和其他的模式。`include`和`exclude`命令使用模式进行匹配。这些模式是通配符样式的模式（请参阅`glob`模块的文档：[`docs.python.org/3/library/glob.html`](https://docs.python.org/3/library/glob.html)），并且对于包括和排除命令都有三种变体：

+   `include`/`exclude`: 这些命令仅适用于给定的路径，而不适用于其他任何内容

+   `recursive-include`/`recursive-exclude`: 这些命令类似于`include`/`exclude`命令，但是递归处理给定的路径

+   `global-include`/`global-exclude`: 对于这些命令要非常小心，它们将在源树中的任何位置包含或排除这些文件

除了`include`/`exclude`命令之外，还有另外两个命令；`graft`和`prune`命令，它们包括或排除包括给定目录下的所有文件的目录。这对于测试和文档可能很有用，因为它们可以包括非标准文件。除了这些例子之外，几乎总是最好明确包括您需要的文件并忽略所有其他文件。这是一个`MANIFEST.in`的例子：

```py
# Comments can be added with a hash tag
include LICENSE CHANGES AUTHORS

# Include the docs, tests and examples completely
graft docs
graft tests
graft examples

# Always exclude compiled python files
global-exclude *.py[co]

# Remove documentation builds
prune docs/_build
```

# 测试软件包

在第十章，“测试和日志-为错误做准备”，测试章节中，我们看到了 Python 的许多测试系统。正如您可能怀疑的那样，至少其中一些已经集成到了`setup.py`中。

## Unittest

在开始之前，我们应该为我们的包创建一个测试脚本。对于实际的测试，请参阅第十章，“测试和日志-为错误做准备”，测试章节。在这种情况下，我们将只使用一个无操作测试，`test.py`：

```py
import unittest

class Test(unittest.TestCase):

    def test(self):
        pass
```

标准的`python setup.py test`命令将运行常规的`unittest`命令：

```py
# python setup.py -v test
running test
running "unittest --verbose"
running egg_info
writing Our_little_project.egg-info/PKG-INFO
writing dependency_links to Our_little_project.egg-info/dependency_links.txt
writing top-level names to Our_little_project.egg-info/top_level.txt
writing entry points to Our_little_project.egg-info/entry_points.txt
reading manifest file 'Our_little_project.egg-info/SOURCES.txt'
writing manifest file 'Our_little_project.egg-info/SOURCES.txt'
running build_ext
test (test.Test) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.000s

OK

```

可以通过使用`--test-module`、`--test-suite`或`--test-runner`参数告诉`setup.py`使用不同的测试。虽然这些很容易使用，但我建议跳过常规的`test`命令，而是尝试使用`nose`或`py.test`。

## py.test

`py.test`软件包有几种集成方法：`pytest-runner`，您自己的测试命令，以及生成`runtests.py`脚本进行测试的已弃用方法。如果您的软件包中仍在使用`runtests.py`，我强烈建议切换到其他选项之一。

但在讨论其他选项之前，让我们确保我们有一些测试。所以让我们在我们的包中创建一个测试。我们将把它存储在`test_pytest.py`中：

```py
def test_a():
    pass

def test_b():
    pass
```

现在，其他测试选项。由于自定义命令实际上并没有增加太多内容，而且实际上使事情变得更加复杂，我们将跳过它。如果您想自定义测试的运行方式，请改用`pytest.ini`和`setup.cfg`文件。最好的选项是`pytest-runner`，它使运行测试变得非常简单：

```py
# pip install pytest-runner
Collecting pytest-runner
 **Using cached pytest_runner-2.7-py2.py3-none-any.whl
Installing collected packages: pytest-runner
Successfully installed pytest-runner-2.7
# python setup.py pytest
running pytest
running egg_info
writing top-level names to Our_little_project.egg-info/top_level.txt
writing dependency_links to Our_little_project.egg-info/dependency_links.txt
writing entry points to Our_little_project.egg-info/entry_points.txt
writing Our_little_project.egg-info/PKG-INFO
reading manifest file 'Our_little_project.egg-info/SOURCES.txt'
writing manifest file 'Our_little_project.egg-info/SOURCES.txt'
running build_ext
======================== test session starts =========================
platform darwin -- Python 3.5.1, pytest-2.8.7, py-1.4.31, pluggy-0.3.1
rootdir: h15, inifile: pytest.ini
collected 2 items

test_pytest.py ..

====================== 2 passed in 0.01 seconds ======================

```

为了正确地集成这种方法，我们应该对`setup.py`脚本进行一些更改。它们并不是严格需要的，但对于使用您的软件包的其他人来说，这会使事情变得更加方便，可能不知道您正在使用`py.test`，例如。首先，我们确保标准的`python setup.py test`命令实际上运行`pytest`命令，而不是通过修改`setup.cfg`来运行：

```py
[aliases]
test=pytest
```

其次，我们要确保`setup.py`命令安装我们运行`py.test`测试所需的软件包。为此，我们还需要修改`setup.py`：

```py
import setuptools

if __name__ == '__main__':
    setuptools.setup(
        name='Our little project',
        entry_points={
            'distutils.commands': [
                'spam = spam.command:SpamCommand',
            ],
        },
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
    )
```

这种方法的美妙之处在于常规的`python setup.py test`命令可以工作，并且在运行测试之前会自动安装所有所需的要求。但是，由于`pytest`要求仅在`tests_require`部分中，如果未运行测试命令，则它们将不会被安装。唯一始终会被安装的软件包是`pytest-runner`软件包，这是一个非常轻量级的软件包，因此安装和运行起来非常轻便。

## Nosetests

`nose`包只处理安装，并且与`py.test`略有不同。唯一的区别是`py.test`有一个单独的`pytest-runner`包用于测试运行器，而 nose 包有一个内置的`nosetests`命令。因此，以下是 nose 版本：

```py
# pip install nose
Collecting nose
 **Using cached nose-1.3.7-py3-none-any.whl
Installing collected packages: nose
Successfully installed nose-1.3.7
# python setup.py nosetests
running nosetests
running egg_info
writing top-level names to Our_little_project.egg-info/top_level.txt
writing entry points to Our_little_project.egg-info/entry_points.txt
writing Our_little_project.egg-info/PKG-INFO
writing dependency_links to Our_little_project.egg-info/dependency_lin
ks.txt
reading manifest file 'Our_little_project.egg-info/SOURCES.txt'
writing manifest file 'Our_little_project.egg-info/SOURCES.txt'
..
----------------------------------------------------------------------
Ran 2 tests in 0.006s

OK

```

# C/C++扩展

前一章已经在一定程度上涵盖了这一点，因为编译 C/C++文件是必需的。但是那一章并没有解释在这种情况下`setup.py`在做什么以及如何做。

为了方便起见，我们将重复`setup.py`文件：

```py
import setuptools

spam = setuptools.Extension('spam', sources=['spam.c'])

setuptools.setup(
    name='Spam',
    version='1.0',
    ext_modules=[spam],
)
```

在开始使用这些扩展之前，你应该学习以下命令：

+   `build`：这实际上不是一个特定于 C/C++的构建函数（尝试`build_clib`），而是一个组合构建函数，用于在`setup.py`中构建所有内容。

+   `clean`：这会清理`build`命令的结果。通常情况下不需要，但有时重新编译工作的文件检测是不正确的。因此，如果遇到奇怪或意外的问题，请尝试先清理项目。

## 常规扩展

`setuptools.Extension`类告诉`setuptools`一个名为`spam`的模块使用源文件`spam.c`。这只是一个扩展的最简单版本，一个名称和一个源列表，但在许多情况下，你需要的不仅仅是简单的情况。

一个例子是`pillow`库，它会检测系统上可用的库，并根据此添加扩展。但是因为这些扩展包括库，所以需要一些额外的编译标志。基本的 PIL 模块本身似乎并不太复杂，但是库实际上都是包含了所有自动检测到的库和匹配的宏定义：

```py
exts = [(Extension("PIL._imaging", files, libraries=libs,
                   define_macros=defs))]
```

`freetype`扩展有类似的东西：

```py
if feature.freetype:
    exts.append(Extension(
        "PIL._imagingft", ["_imagingft.c"], libraries=["freetype"]))
```

## Cython 扩展

`setuptools`库在处理扩展时实际上比常规的`distutils`库要聪明一些。它实际上向`Extension`类添加了一个小技巧。还记得第十二章中对性能的简要介绍吗？`setuptools`库使得编译这些变得更加方便。`Cython`手册建议你使用类似以下代码的东西：

```py
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("eggs.pyx")
)
```

这里的`eggs.pyx`包含：

```py
def make_eggs(int n):
    print('Making %d eggs: %s' % (n, n * 'eggs '))
```

这种方法的问题是，除非你安装了`Cython`，否则`setup.py`会出现问题：

```py
# python setup.py build
Traceback (most recent call last):
 **File "setup.py", line 2, in <module>
 **import Cython
ImportError: No module named 'Cython'

```

为了防止这个问题，我们只需要让`setuptools`处理这个问题：

```py
import setuptools

eggs = setuptools.Extension('eggs', sources=['eggs.pyx'])

setuptools.setup(
    name='Eggs',
    version='1.0',
    ext_modules=[eggs],
    setup_requires=['Cython'],
)
```

现在，如果需要，`Cython`将被自动安装，并且代码将正常工作：

```py
# python setup.py build
running build
running build_ext
cythoning eggs.pyx to eggs.c
building 'eggs' extension
...
# python setup.py develop
running develop
running egg_info
creating Eggs.egg-info
writing dependency_links to Eggs.egg-info/dependency_links.txt
writing top-level names to Eggs.egg-info/top_level.txt
writing Eggs.egg-info/PKG-INFO
writing manifest file 'Eggs.egg-info/SOURCES.txt'
reading manifest file 'Eggs.egg-info/SOURCES.txt'
writing manifest file 'Eggs.egg-info/SOURCES.txt'
running build_ext
skipping 'eggs.c' Cython extension (up-to-date)
copying build/... ->
Creating Eggs.egg-link (link to .)
Adding Eggs 1.0 to easy-install.pth file

Installed Eggs
Processing dependencies for Eggs==1.0
Finished processing dependencies for Eggs==1.0
# python -c 'import eggs; eggs.make_eggs(3)'
Making 3 eggs: eggs eggs eggs

```

然而，为了开发目的，`Cython`还提供了一种不需要手动构建的更简单的方法。首先，为了确保我们实际上正在使用这种方法，让我们安装`Cython`，并彻底卸载和清理`eggs`：

```py
# pip uninstall eggs -y
Uninstalling Eggs-1.0:
 **Successfully uninstalled Eggs-1.0
# pip uninstall eggs -y
Cannot uninstall requirement eggs, not installed
# python setup.py clean
# pip install cython

```

现在让我们尝试运行我们的`eggs.pyx`模块：

```py
>>> import pyximport
>>> pyximport.install()
(None, <pyximport.pyximport.PyxImporter object at 0x...>)
>>> import eggs
>>> eggs.make_eggs(3)
Making 3 eggs: eggs eggs eggs

```

这就是在没有显式编译的情况下运行`pyx`文件的简单方法。

# Wheels - 新的 eggs

对于纯 Python 包，`sdist`（源分发）命令一直足够了。但是对于 C/C++包来说，通常并不那么方便。C/C++包的问题在于，除非使用二进制包，否则需要进行编译。传统上，这些通常是`.egg`文件，但它们从未真正解决了问题。这就是为什么引入了`wheel`格式（PEP 0427），这是一种包含源代码和二进制代码的二进制包格式，可以在 Windows 和 OS X 上安装，而无需编译器。作为额外的奖励，它也可以更快地安装纯 Python 包。

实现起来幸运的是很简单。首先，安装`wheel`包：

```py
# pip install wheel

```

现在你可以使用`bdist_wheel`命令来构建你的包。唯一的小问题是，默认情况下 Python 3 创建的包只能在 Python 3 上运行，因此 Python 2 安装将退回到`sdist`文件。为了解决这个问题，你可以将以下内容添加到你的`setup.cfg`文件中：

```py
[bdist_wheel]
universal = 1
```

这里唯一需要注意的重要事项是，在 C 扩展的情况下，可能会出错。Python 3 的二进制 C 扩展与 Python 2 的不兼容。因此，如果您有一个纯 Python 软件包，并且同时针对 Python 2 和 3，启用该标志。否则，就将其保持为默认值。

## 分发到 Python Package Index

一旦您的一切都正常运行，经过测试和记录，就是时候将项目实际推送到**Python Package Index**（**PyPI**）了。在将软件包推送到 PyPI 之前，我们需要确保一切都井井有条。

首先，让我们检查`setup.py`文件是否有问题：

```py
# python setup.py check
running check
warning: check: missing required meta-data: url

warning: check: missing meta-data: either (author and author_email) or (maintainer and maintainer_email) must be supplied

```

看起来我们忘记了指定`url`和`author`或`maintainer`信息。让我们填写这些：

```py
import setuptools

eggs = setuptools.Extension('eggs', sources=['eggs.pyx'])

setuptools.setup(
    name='Eggs',
    version='1.0',
    ext_modules=[eggs],
    setup_requires=['Cython'],
    url='https://wol.ph/',
    author='Rick van Hattem (Wolph)',
    author_email='wolph@wol.ph',
)
```

现在让我们再次检查：

```py
# python setup.py check
running check

```

完美！没有错误，一切看起来都很好。

现在我们的`setup.py`已经井井有条了，让我们来尝试测试。由于我们的小测试项目几乎没有测试，这将几乎是空的。但是如果您正在启动一个新项目，我建议从一开始就尽量保持 100%的测试覆盖率。稍后实施所有测试通常更加困难，而在工作时进行测试通常会让您更多地考虑代码的设计决策。运行测试非常容易：

```py
# python setup.py test
running test
running egg_info
writing dependency_links to Eggs.egg-info/dependency_links.txt
writing Eggs.egg-info/PKG-INFO
writing top-level names to Eggs.egg-info/top_level.txt
reading manifest file 'Eggs.egg-info/SOURCES.txt'
writing manifest file 'Eggs.egg-info/SOURCES.txt'
running build_ext
skipping 'eggs.c' Cython extension (up-to-date)
copying build/... ->

---------------------------------------------------------------------
Ran 0 tests in 0.000s

OK

```

现在我们已经检查完毕，下一步是构建文档。如前所述，`sphinx`和`sphinx-pypi-upload-2`软件包可以在这方面提供帮助：

```py
# python setup.py build_sphinx
running build_sphinx
Running Sphinx v1.3.5
...

```

一旦我们确定一切都正确，我们就可以构建软件包并将其上传到 PyPI。对于纯 Python 版本的发布，您可以使用`sdist`（源分发）命令。对于使用本机安装程序的软件包，有一些选项可用，例如`bdist_wininst`和`bdist_rpm`。我个人几乎在所有我的软件包中使用以下命令：

```py
# python setup.py build_sphinx upload_sphinx sdist bdist_wheel upload

```

这将自动构建 Sphinx 文档，将文档上传到 PyPI，使用源构建软件包，并使用源上传软件包。

显然，只有在您是特定软件包的所有者并且被 PyPI 授权时，才能成功完成此操作。

### 注意

在上传软件包之前，您需要在 PyPI 上注册软件包。这可以使用`register`命令来完成，但由于这会立即在 PyPI 服务器上注册软件包，因此在测试时不应使用。

# 总结

阅读完本章后，您应该能够创建包含不仅是纯 Python 文件，还包括额外数据、编译的 C/C++扩展、文档和测试的 Python 软件包。有了这些工具，您现在可以制作高质量的 Python 软件包，这些软件包可以轻松地在其他项目和软件包中重复使用。

Python 基础设施使得创建新软件包并将项目拆分为多个子项目变得非常容易。这使您能够创建简单且可重用的软件包，因为一切都很容易进行测试。虽然您不应该过度拆分软件包，但是如果脚本或模块具有自己的目的，那么它就是可以单独打包的候选项。

通过本章，我们已经完成了本书。我真诚地希望您喜欢阅读，并了解了新颖有趣的主题。非常感谢您的任何反馈，所以请随时通过我的网站[`wol.ph/`](https://wol.ph/)与我联系。
