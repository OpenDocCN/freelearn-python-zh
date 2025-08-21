# 第十一章：与文件系统交互

在本章中，我们将涵盖许多与文件系统交互相关的教程。第一个教程将涉及如何在使用 Python 代码修改任何文件之前，对需要这样做的设备重新挂载文件系统。

然后，将涵盖列出、删除和创建文件的教程。还将涵盖更高级的主题，如计算磁盘使用量。本章的教程将为你提供将文件系统交互添加到嵌入式项目中所需的工具。当你想要将传感器数据记录到文件中，或者当你希望你的代码读取并加载一组文件到数据结构中时，这将非常有用。当你必须列出一组要在你的应用程序中显示的图像时，这也会很有帮助。

在本章中，我们将涵盖以下教程：

+   重新挂载文件系统

+   列出文件

+   删除文件

+   创建一个目录

+   读取文件的内容

+   写入文件的内容

+   计算磁盘使用量

# 技术要求

本章的代码文件可以在本书的 GitHub 存储库的`Chapter11`文件夹中找到，网址为[`github.com/PacktPublishing/MicroPython-Cookbook`](https://github.com/PacktPublishing/MicroPython-Cookbook)。

本章中的所有教程都使用了 CircuitPython 3.1.2。

# 重新挂载文件系统

这个教程将向你展示如何重新挂载文件系统，以便可以从你的 Python 脚本中写入数据。一些开发板，比如 Circuit Playground Express，默认情况下会将连接的设备显示为 USB 驱动器，以便轻松编辑和保存你的代码。然而，这种方法的折衷是，你的 Python 代码无法写入或更改开发板存储中的任何内容。在这些开发板上，你必须重新挂载文件系统，以允许你的脚本向其文件系统写入数据。

通过这个教程的最后，你将知道如何允许数据写入文件系统，以及如何恢复更改，这对于某些项目将变得至关重要。例如，如果你想要使用 Circuit Playground Express 来记录温度读数到日志文件，你将需要利用这种方法。

# 准备工作

你需要访问 Circuit Playground Express 上的 REPL 来运行本教程中提供的代码。

# 如何做...

按照以下步骤学习如何重新挂载文件系统：

1.  在 REPL 中运行以下代码行：

```py
>>> f = open('main.py')
```

1.  Python 代码可以打开一个文件进行读取。但是，如果你尝试打开一个文件进行写入，就像下面的代码块中所示，你将会得到一个`OSError`实例，因为文件系统处于只读模式：

```py
>>> f = open('hi.txt', 'w')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
OSError: 30
>>>
```

1.  我们现在将创建一个脚本，用于重新挂载文件系统以允许读写数据。

1.  以下代码应该保存到`boot.py`文件中。如果文件不存在，那么你将需要创建它：

```py
import storage
storage.remount('/', False)
```

1.  从计算机上弹出`CIRCUITPY`驱动器。

1.  从计算机上拔下开发板。

1.  重新连接开发板到计算机。

1.  在 REPL 中运行以下代码行，确认你的代码可以将数据写入到开发板的存储中：

```py
>>> f = open('hi.txt', 'w')
>>> f.write('hi there')
8
>>> f.close()
```

1.  当在 REPL 中运行以下代码块时，将删除`boot.py`文件：

```py
>>> import os
>>> os.remove('boot.py')
```

1.  要将这些更改应用到启动过程中，再次从计算机中弹出`CIRCUITPY`驱动器。

1.  从计算机上拔下开发板。

1.  重新连接开发板到计算机。

1.  你应该能够编辑并保存`main.py`文件的内容，就像之前做的那样。

# 工作原理...

Circuit Playground Express 为你提供了一种从 Python 脚本中启用对存储的读写的方法。我们将代码放在`boot.py`文件中，因为这个脚本将在启动过程的早期运行，在`main.py`文件（其中包含我们的主要代码库）之前运行。

在`boot.py`脚本中，我们导入`storage`模块，然后调用它的`remount`函数，第二个参数设置为`False`，表示文件系统应该以读写模式挂载。无论是创建还是删除文件，每当我们对`boot.py`文件进行更改，都必须对板子进行硬重置，即拔下并重新连接板子，更改才能生效。如本教程所示，恢复此更改的最简单方法是从 REPL 中删除`boot.py`文件。

# 还有更多...

一般来说，只有提供 USB 驱动器编辑功能的板子才需要这个额外的重新挂载文件系统的步骤。例如，ESP8266 没有 USB 驱动器功能，因此不需要这一步。还需要注意的是，一旦你在代码中启用了对文件系统的写入，你就无法在文本编辑器中编辑`main.py`文件。每当你想要回到编辑代码时，你都需要删除`boot.py`文件。如果你的项目只需要对文件系统进行只读访问，比如列出文件和读取文件内容，那么你可以在任何模式下安全运行。

# 另请参阅

以下是关于这个教程的一些参考资料：

+   有关`mount`函数的文档可以在[`circuitpython.readthedocs.io/en/3.x/shared-bindings/storage/__init__.html#storage.mount`](https://circuitpython.readthedocs.io/en/3.x/shared-bindings/storage/__init__.html#storage.mount)找到。

+   有关写入文件系统的文档可以在[`learn.adafruit.com/cpu-temperature-logging-with-circuit-python/writing-to-the-filesystem`](https://learn.adafruit.com/cpu-temperature-logging-with-circuit-python/writing-to-the-filesystem)找到。

# 列出文件

这个教程将向你展示如何在 MicroPython 中列出文件和目录。我们还将展示你可以使用的技术，以便对列表进行过滤，使其只包括文件或只包括目录。一旦你有了以这种方式与文件系统交互的能力，你就可以在自己的项目中使用它，你的代码将接受一个动态的文件列表，这个列表不需要在程序中硬编码。这样，这些文件可能代表一组可配置的音频文件，你想要播放，或者一组图像，你将在连接的屏幕上显示。

# 准备工作

你需要访问 Circuit Playground Express 上的 REPL 来运行本教程中提供的代码。

# 如何做...

按照以下步骤学习如何列出文件：

1.  在 REPL 中执行以下代码块：

```py
>>> import os
>>> os.listdir()
['.fseventsd', '.metadata_never_index', '.Trashes', 'boot_out.txt', 'main.py', 'lib']
```

1.  将生成顶级文件夹中所有文件和目录的列表。下面的代码块将生成相同的列表，但是是排序后的。

```py
>>> sorted(os.listdir())
['.Trashes', '.fseventsd', '.metadata_never_index', 'boot_out.txt', 'lib', 'main.py']
```

1.  我们还可以列出特定目录中的文件，如下面的代码块所示：

```py
>>> os.listdir('.fseventsd')
['no_log']
```

1.  下面的代码块将检查并显示`lib`路径不是一个文件：

```py
>>> FILE_CODE  = 0x8000
>>> 
>>> os.stat('lib')[0] == FILE_CODE
False
```

1.  现在，我们将确认`main.py`被检测为一个文件：

```py
>>> os.stat('main.py')[0] == FILE_CODE
True
```

1.  下面的代码块定义并调用`isfile`函数来验证两个路径的类型：

```py
>>> def isfile(path):
...     return os.stat(path)[0] == FILE_CODE
...     
...     
... 
>>> isfile('lib')
False
>>> isfile('main.py')
True
>>> 
```

1.  下面的代码块将列出根路径中的所有文件：

```py
>>> files = [i for i in sorted(os.listdir()) if isfile(i)]
>>> files
['.Trashes', '.metadata_never_index', 'boot_out.txt', 'main.py']
```

1.  现在，我们将列出根路径中的所有目录：

```py
>>> dirs = [i for i in sorted(os.listdir()) if not isfile(i)]
>>> dirs
['.fseventsd', 'lib']
```

1.  以下代码应该放入`main.py`文件中：

```py
import os

FILE_CODE = 0x8000

def isfile(path):
    return os.stat(path)[0] == FILE_CODE

def main():
    files = [i for i in sorted(os.listdir()) if isfile(i)]
    print('files:', files)
    dirs = [i for i in sorted(os.listdir()) if not isfile(i)]
    print('dirs:', dirs)

main()
```

当执行时，此脚本将打印出根路径中文件和目录的排序列表。

# 它是如何工作的...

在导入`os`模块后，定义了一个名为`isfile`的函数，它将根据提供的路径是文件还是目录返回`True`或`False`。定义并调用了`main`函数，之后它将生成一个路径名列表。第一个列表将检索排序后的路径列表，然后过滤列表，只保留文件。然后打印出这个列表。然后采取同样的方法来获取目录列表并打印出来。

# 还有更多...

本文介绍了处理文件时可以派上用场的一些技术。它表明，默认情况下，文件列表不会按字母顺序返回，因此如果需要，可以使用内置的`sorted`函数对文件列表进行排序。它还定义了一个名为`isfile`的函数，用于检查特定路径是否为文件。如果需要，您可以创建一个等效的`isdir`函数。本文还展示了使用列表推导的简单方法，以过滤默认列表以生成仅包含特定类型条目的路径的过滤列表，例如文件或目录。

# 另请参阅

有关本文的一些参考资料：

+   有关`listdir`函数的文档可在[`circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.listdir`](https://circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.listdir)找到。

+   有关`stat`函数的文档可在[`circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.stat`](https://circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.stat)找到。

+   有关`isfile`函数的文档可在[`docs.python.org/3/library/os.path.html#os.path.isfile`](https://docs.python.org/3/library/os.path.html#os.path.isfile)找到。

# 删除文件

本文将向您展示如何在 MicroPython 中删除文件和目录。有专门的函数用于删除文件和删除目录。我们将向您展示如何为每种类型的路径调用这些不同的函数。然后，我们将向您展示如何创建一个通用函数，可以自动删除任一类型的路径。

在您创建的项目中，有许多情况需要删除文件。您可能创建一个将数据记录到文件中的项目。日志轮换是一种机制，可以让您定期创建新的日志文件并自动删除旧文件。您将需要删除文件的功能来实现日志轮换。在许多 MicroPython 嵌入式项目中，由于这些板上的存储容量有限，删除文件以节省空间的问题变得更加重要。

# 准备工作

您需要访问 Circuit Playground Express 上的 REPL 来运行本文中提供的代码。确保您已经完成了本章的*重新挂载文件系统*配方，因为您需要对存储系统具有写访问权限才能删除文件。

# 如何做...

按照以下步骤学习如何删除文件：

1.  在根路径下创建一个名为`hi.txt`的文件。

1.  使用 REPL 运行以下代码行：

```py
>>> import os
>>> os.remove('hi.txt')
```

1.  `hi.txt`文件现在已从板的文件系统中删除。运行以下代码块。它应该会出现异常，因为文件不再存在：

```py
>>> os.remove('hi.txt')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
OSError: [Errno 2] No such file/directory
>>> 
```

1.  在根路径下创建一个名为`mydir`的目录。

1.  以下代码块将删除`mydir`目录：

```py
>>> os.rmdir('mydir')
```

1.  以下代码块定义了`isfile`函数：

```py
>>> FILE_CODE = 0x8000
>>> 
>>> def isfile(path):
...     return os.stat(path)[0] == FILE_CODE
...     
...     
... 
>>>
```

1.  现在我们可以定义一个名为`any_remove`的函数，它将删除任何类型的路径：

```py
>>> def any_remove(path):
...     func = os.remove if isfile(path) else os.rmdir
...     func(path)
...     
...     
... 
>>> 
```

1.  在根路径下创建一个名为`hi.txt`和一个名为`mydir`的目录。

1.  运行以下代码块：

```py
>>> any_remove('hi.txt')
>>> any_remove('mydir')
>>> 
```

前面的代码块现在使用相同的函数调用删除了此文件和目录。

# 工作原理...

首先定义的`any_remove`函数接受路径并设置一个名为`func`的变量。此变量将存储需要调用以删除提供的路径的可调用对象。检查路径的类型，并根据提供的路径类型设置`func`为`os.remove`或`os.rmdir`。然后使用提供的路径调用此函数以执行实际的删除。

# 更多信息...

本教程介绍了一种您可以使用的技术，用于创建一个方便的函数，该函数接受任何类型的路径，并调用正确的底层函数来删除它。需要记住的一件事是，您只能删除空目录。本教程中的函数和示例支持删除空目录，但如果使用具有文件的目录调用，则会失败。您可以扩展`delete`函数以执行递归目录列表，然后删除所有子文件夹和目录。

# 另请参阅

以下是关于本教程的一些参考资料：

+   有关`remove`函数的文档可以在[`circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.remove`](https://circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.remove)找到。

+   有关`rmdir`函数的文档可以在[`circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.rmdir`](https://circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.rmdir)找到。

# 创建目录

这个教程将向您展示如何在 MicroPython 中创建一个目录。我们还将向您展示如何创建一个可以多次调用相同路径的函数，只有在目录尚不存在时才会创建目录。然后，我们将定义一个函数，其行为与 Python 标准库中的`makedirs`函数相同，但在 MicroPython 中没有包含。

这组功能在您需要创建一个可能需要创建特定目录树并用一组特定文件填充的项目时非常有用。当您在像 ESP8266 这样的开发板上工作时，也有必要访问这些功能，因为它只能让您通过 REPL 和 Python 代码创建所需的目录。

# 准备工作

您需要访问 Circuit Playground Express 上的 REPL 才能运行本教程中提供的代码。确保您已经完成了本章中的*重新挂载文件系统*教程，因为本教程需要对存储系统进行写访问。

# 如何操作...

按照以下步骤学习如何创建目录：

1.  在 REPL 中运行以下代码行：

```py
>>> import os
>>> os.mkdir('mydir')
```

1.  现在已经创建了一个名为`mydir`的目录。当您运行以下代码块时，将引发异常，因为该目录已经存在：

```py
>>> os.mkdir('mydir')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
OSError: [Errno 17] File exists
>>> 
```

1.  以下代码块将定义一个根据路径是否存在返回`True`或`False`的函数：

```py
>>> def exists(path):
...     try:
...         os.stat(path)
...     except OSError:
...         return False
...     return True
...     
...     
... 
>>> 
```

1.  以下代码块将在两个不同的路径上调用`exists`函数，以验证其是否正常工作：

```py
>>> exists('main.py')
True
>>> exists('invalid_path')
False
```

1.  现在我们可以定义一个名为`mkdir_safe`的函数，它只在目录不存在时才创建目录：

```py
>>> def mkdir_safe(path):
...     if not exists(path):
...         os.mkdir(path)
...         
...         
... 
>>> 
```

1.  以下代码块将在相同路径上多次调用`mkdir_safe`函数，不会引发异常：

```py
>>> mkdir_safe('newdir')
>>> mkdir_safe('newdir')
```

1.  我们现在将定义一个可以递归创建目录的函数：

```py
>>> def makedirs(path):
...     items = path.strip('/').split('/')
...     count = len(items)
...     paths = ['/' + '/'.join(items[0:i + 1]) for i in 
...     range(count)]
...     for path in paths:
...         os.mkdir(path)
...         
...         
... 
>>> 
```

1.  运行以下代码块：

```py
>>> makedirs('/top/middle/bottom')
```

执行上述代码块将按正确顺序从上到下创建三个目录。

# 工作原理...

在本教程中，定义并使用了三个不同的函数，每个函数执行特定的功能。`exists`函数检查路径是否存在，并返回`True`或`False`。此检查尝试在路径上调用`stat`函数，并捕获可能引发的任何`OSError`。如果路径存在，则不会引发此异常，并返回`True`值；否则返回`False`值。

下一个函数`mkdir_safe`只是检查路径是否存在，并且仅在不存在路径时调用`mkdir`函数。最后，定义了`makedirs`函数，该函数接收具有多个级别的路径。路径被拆分为其各个部分，然后将要创建的路径列表按正确顺序保存在列表中，从最高路径到最低路径。通过循环遍历每个路径，并通过调用`mkdir`函数来创建每个路径。

# 还有更多...

这个教程介绍了三个通用函数，每个函数都有特定的目的。通过以这种方式创建代码片段，可以更容易地将一个项目的片段并入其他项目中。其中两个定义的函数——`exists`和`makedirs`——是 Python 标准库的一部分，但在 MicroPython 中找不到。这个教程演示了在许多情况下，即使在 Python 标准库中缺少某个函数，您也经常可以在 MicroPython 中创建自己的实现。

# 另请参阅

以下是关于这个教程的一些参考资料：

+   `mkdir`函数的文档可以在[`circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.mkdir`](https://circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.mkdir)找到。

+   `makedirs`函数的文档可以在[`docs.python.org/3/library/os.html#os.makedirs`](https://docs.python.org/3/library/os.html#os.makedirs)找到。

+   `exists`函数的文档可以在[`docs.python.org/3/library/os.path.html#os.path.exists`](https://docs.python.org/3/library/os.path.html#os.path.exists)找到。

# 读取文件的内容

这个教程将向您展示如何将文件内容读入脚本中的变量。这个教程将涵盖将文件内容作为字符串读取的方法，以及将其作为字节对象读取的方法。您创建的许多项目通常需要打开不同的数据文件，如音频文件、图像和文本文件。这个教程将为您提供基本的构建模块，以便您可以促进这些交互。

# 准备工作

您需要访问 Circuit Playground Express 上的 REPL 来运行本教程中提供的代码。

# 如何做...

按照以下步骤学习如何读取文件的内容：

1.  在根路径下创建一个名为`hi.txt`的文件，其中包含以下内容：

```py
hi there
```

1.  在 REPL 中执行以下代码块：

```py
>>> f = open('hi.txt')
>>> data = f.read()
>>> f.close()
>>> data
'hi there\n'
```

1.  名为`hi.txt`的文件的内容被读入一个名为`data`的变量中，然后作为输出显示。下面的代码块也将文件的内容读入一个变量中，但使用`with`语句：

```py
>>> with open('hi.txt') as f:
...     data = f.read()
...     
... 
>>> data
'hi there\n'
```

1.  可以通过一行代码将文件的内容读入一个变量中，如下例所示：

```py
>>> data = open('hi.txt').read()
>>> data
'hi there\n'
```

1.  执行以下代码块：

```py
>>> data = open('hi.txt', 'rb').read()
>>> data
b'hi there\n'
>>> type(data)
<class 'bytes'>
```

上述代码块在执行时，将以`bytes`对象而不是字符串读取文件内容。

# 它是如何工作的...

在这个教程中，我们探讨了从文件中读取数据的四种不同方法。第一种方法使用`open`函数获取文件对象。然后，从这个文件对象中读取数据并关闭。然后，我们可以改进这种较旧的文件处理方式，如第二个例子所示，使用`with`语句，它将在退出`with`块时自动关闭文件。第三个例子在一行中打开并读取所有内容。`open`函数接受文件模式作为第二个参数。如果我们传递`rb`值，那么它将以二进制模式打开文件。这将导致返回一个字节对象，而不是一个字符串。

# 还有更多...

您需要根据您希望与之交互的数据选择正确的读取文件数据的方法。如果您的数据文件是纯文本文件，那么默认的文本模式就足够了。但是，如果您要读取`.wav`文件格式的原始音频数据，您需要将数据读取为二进制数据，可能会出现异常，因为数据可能无法解码为字符串。

# 另请参阅

以下是关于这个教程的一些参考资料：

+   `open`函数的文档可以在[`docs.python.org/3/library/functions.html#open`](https://docs.python.org/3/library/functions.html#open)找到。

+   关于`with`语句的文档可以在[`docs.python.org/3/reference/compound_stmts.html#with`](https://docs.python.org/3/reference/compound_stmts.html#with)找到。

# 写入文件的内容

这个配方将向您展示如何将数据写入输出文件。我们将介绍如何将字符串以及字节写入文件。然后，我们将定义一种对象类型，以便更容易执行这些常见的操作，将文本和二进制数据写入文件。如果您想创建一个将传感器数据保存到日志文件或将一些用户生成的数据记录到板的存储器中的项目，那么您将需要使用我们在这个配方中描述的许多技术。

# 准备工作

您需要在 Circuit Playground Express 上访问 REPL 以运行本配方中提供的代码。确保您已经完成了本章中的*重新挂载文件系统*配方，因为这个配方需要对存储系统进行写访问。

# 如何做...

按照以下步骤学习如何写入文件的内容：

1.  使用 REPL 运行以下代码行：

```py
>>> with open('hi.txt', 'w') as f:
...     count = f.write('hi there')
...     
...     
... 
>>> count
8
```

1.  文本读取`hi there`已经使用文件对象的`write`方法写入到名为`hi.txt`的文件中。然后返回并显示写入的字节数。以下代码块将获取一个字节对象并将其传递给`write`方法，以便它可以将数据写入提供的文件中：

```py
>>> with open('hi.txt', 'wb') as f:
...     count = f.write(b'hi there')
...     
...     
... 
>>> count
8
>>>
```

1.  以下代码块将定义一个`Path`类，其中包含两个方法。一个方法将初始化新对象，而另一个方法将生成对象的表示：

```py
>>> class Path:
...     def __init__(self, path):
...         self._path = path
...         
...     def __repr__(self):
...         return "Path(%r)" % self._path
...         
... 
>>> 
>>> path = Path('hi.txt')
>>> path
Path('hi.txt')
>>> 
```

1.  执行以下代码块后，我们将得到一个`Path`类，其中包含两个额外的方法，以便我们可以将文本和二进制数据写入文件：

```py
>>> class Path:
...     def __init__(self, path):
...         self._path = path
...         
...     def __repr__(self):
...         return "Path(%r)" % self._path
...         
...     def write_bytes(self, data):
...         with open(self._path, 'wb') as f:
...             return f.write(data)
...             
...     def write_text(self, data):
...         with open(self._path, 'w') as f:
...             return f.write(data)
...             
... 
>>> 
>>> path = Path('hi.txt')
>>> path.write_text('hi there')
8
>>> path.write_bytes(b'hi there')
8
```

1.  将以下代码块保存到名为`pathlib.py`的文件中，以便可以导入： 

```py
class Path:
    def __init__(self, path):
        self._path = path

    def __repr__(self):
        return "Path(%r)" % self._path

    def write_bytes(self, data):
        with open(self._path, 'wb') as f:
            return f.write(data)

    def write_text(self, data):
        with open(self._path, 'w') as f:
            return f.write(data)
```

1.  以下代码应该放入`main.py`文件中：

```py
from pathlib import Path

Path('hi.txt').write_text('hi there')
```

当执行此脚本时，它将把文本`hi there`消息写入到`hi.txt`文件中。

# 它是如何工作的...

在这个配方中，我们首先展示了写入字节和文本数据到文件的两种最直接的方法。然后，我们创建了一个名为`Path`的类，这个类分为两个阶段构建。第一个版本让我们创建`Path`对象，这些对象可以跟踪它们的路径，并在请求时返回一个人类可读的表示。

然后，我们添加了辅助方法来帮助写入文本数据或二进制数据。类的名称和方法与 Python 的`pathlib`模块中的`Path`对象具有相同的命名和功能。最终的代码块展示了一个简单的示例，导入`pathlib`模块并调用其`write_text`方法将一些文本保存到文件中。

# 还有更多...

在一些项目中，您可能会发现自己处于一种情况中，需要与文件、它们的路径进行交互，并经常需要写入和读取数据。在这些情况下，拥有一个简化文件访问的类将非常有帮助。在这个配方中定义的`Path`对象非常适合这个目的。我们还遵循了 Python 标准库中的一个模块的相同命名和功能。这将使我们的代码在想要在具有完整 Python 安装的计算机上运行时更具可读性和可移植性。

# 另请参阅

以下是关于这个配方的一些参考资料：

+   关于`write_bytes`方法的文档可以在[`docs.python.org/3/library/pathlib.html#pathlib.Path.write_bytes`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.write_bytes)找到。

+   关于`write_text`方法的文档可以在[`docs.python.org/3/library/pathlib.html#pathlib.Path.write_text`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.write_text)找到。

# 计算磁盘使用量

这个配方将向您展示如何检查与存储系统相关的一些数字。我们将检索文件系统的块大小、总块数和空闲块数。然后我们可以使用这些数字来计算一些有用的数字，比如总磁盘容量以及磁盘上已使用和空闲的空间。

然后，我们将所有这些代码打包到一个函数中，以便在需要访问这些信息时更容易调用。您可以使用本配方中展示的技术来实现项目中的许多事情。例如，您可以使用它来查找设备上可用的总存储空间，因为这在不同的板之间有所不同。您甚至可以使用它来判断磁盘是否变得太满，以及您的脚本是否应该删除一些旧的日志文件。

# 准备就绪

您需要访问 Circuit Playground Express 上的 REPL 才能运行此配方中提供的代码。

# 如何做...

按照以下步骤学习如何计算磁盘使用情况：

1.  在 REPL 中运行以下代码行：

```py
>>> import os
>>> stats = os.statvfs('/')
>>> stats
(1024, 1024, 2024, 1040, 1040, 0, 0, 0, 0, 255)
```

1.  我们现在已经检索到了所有关键的文件系统信息，但它以元组的形式呈现，这使得很难知道哪个数字与什么相关。在以下代码块中，我们将把我们关心的值赋给更易读的变量名：

```py
>>> block_size = stats[0]
>>> total_blocks = stats[2]
>>> free_blocks = stats[3]
```

1.  现在我们已经将这些关键信息存储在易读的变量中，我们可以继续以下代码块来计算我们感兴趣的主要值：

```py
>>> stats = dict()
>>> stats['free'] = block_size * free_blocks
>>> stats['total'] = block_size * total_blocks
>>> stats['used'] = stats['total'] - stats['free']
>>> stats
{'free': 1064960, 'used': 1007616, 'total': 2072576}
```

1.  以下代码块将所有这些逻辑封装到一个单独的函数中：

```py
>>> def get_disk_stats():
...     stats = os.statvfs('/')
...     block_size = stats[0]
...     total_blocks = stats[2]
...     free_blocks = stats[3]
...     stats = dict()
...     stats['free'] = block_size * free_blocks
...     stats['total'] = block_size * total_blocks
...     stats['used'] = stats['total'] - stats['free']
...     return stats
... 
>>> 
>>> get_disk_stats()
{'free': 1064960, 'used': 1007616, 'total': 2072576}
>>> 
```

1.  以下函数将以更易读的方式格式化以字节表示的值：

```py
>>> def format_size(val):
...     val = int(val / 1024)               # convert bytes to KiB
...     val = '{:,}'.format(val)            # add thousand separator
...     val = '{0: >6} KiB'.format(val)     # right align amounts
...     return val
...     
...     
... 
>>> print('total space:', format_size(stats['total']))
total space:  2,024 KiB
>>> 
```

1.  我们现在可以创建一个函数，打印与总磁盘大小和使用情况相关的一些关键数字：

```py
>>> def print_stats():
...     stats = get_disk_stats()
...     print('free space: ', format_size(stats['free']))
...     print('used space: ', format_size(stats['used']))
...     print('total space:', format_size(stats['total']))
...     
...     
... 
>>> print_stats()
free space:   1,040 KiB
used space:     984 KiB
total space:  2,024 KiB
>>>
```

1.  以下代码应放入`main.py`文件中：

```py
import os

def get_disk_stats():
    stats = os.statvfs('/')
    block_size = stats[0]
    total_blocks = stats[2]
    free_blocks = stats[3]
    stats = dict()
    stats['free'] = block_size * free_blocks
    stats['total'] = block_size * total_blocks
    stats['used'] = stats['total'] - stats['free']
    return stats

def format_size(val):
    val = int(val / 1024)               # convert bytes to KiB
    val = '{:,}'.format(val)            # add thousand separator
    val = '{0: >6} KiB'.format(val)     # right align amounts
    return val

def print_stats():
    stats = get_disk_stats()
    print('free space: ', format_size(stats['free']))
    print('used space: ', format_size(stats['used']))
    print('total space:', format_size(stats['total']))

print_stats()
```

当执行此脚本时，它将打印有关磁盘空间的空闲、已使用和总空间的详细信息。

# 它是如何工作的...

`statvfs`函数返回与板上的文件系统相关的一些关键数字。在这个元组中，我们关心三个值，它们分别对应`block_size`、`total_blocks`和`free_blocks`变量。我们可以将这些值相乘，以字节为单位计算出空闲、已使用和总磁盘空间的数量。然后，`format_size`函数被定义为将字节值转换为`KiB`，添加千位分隔符，并右对齐这些值。`print_stats`函数简单地通过获取文件系统的`stats`并在每个值上调用`format_size`函数来组合所有这些代码。

# 还有更多...

Circuit Playground Express 配备了 2MB 的闪存，可以在此配方的输出中看到。MicroPython 使用 FAT 格式进行文件系统。您可以尝试在板上添加一些文件，然后重新运行脚本以查看文件系统使用情况的变化。请记住，要看到这些变化反映在多个板上，您必须弹出 USB 设备并重新插入以获取最新的文件系统使用情况。

# 另请参阅

以下是关于此配方的一些参考资料：

+   有关`statvfs`函数的文档可以在[`circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.statvfs`](https://circuitpython.readthedocs.io/en/3.x/shared-bindings/os/__init__.html#os.statvfs)找到。

+   有关`statvfs`返回的信息的详细信息可以在[`man7.org/linux/man-pages/man3/statvfs.3.html`](http://man7.org/linux/man-pages/man3/statvfs.3.html)找到。
