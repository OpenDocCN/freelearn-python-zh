# 处理文件，目录和数据

系统管理员执行处理各种文件，目录和数据等任务。在本章中，我们将学习`os`模块。`os`模块提供了与操作系统交互的功能。Python 程序员可以轻松使用此`os`模块执行文件和目录操作。`os`模块为处理文件，路径，目录和数据的程序员提供了工具。

在本章中，您将学习以下内容：

+   使用 os 模块处理目录

+   复制，移动，重命名和删除数据

+   处理路径，目录和文件

+   比较数据

+   合并数据

+   模式匹配文件和目录

+   元数据：关于数据的数据

+   压缩和恢复

+   使用`tarfile`模块创建 TAR 存档

+   使用`tarfile`模块检查 TAR 文件的内容

# 使用 os 模块处理目录

目录是文件和子目录的集合。`os`模块提供了各种函数，允许我们与操作系统交互。在本节中，我们将学习一些在处理目录时可以使用的函数。

# 获取工作目录

要开始处理目录，首先我们将获取当前工作目录的名称。`os`模块有一个`getcwd()`函数，使用它我们可以获取当前工作目录。启动`python3`控制台并输入以下命令以获取目录名称：

```py
$ python3 Python 3.6.5 (default, Apr  1 2018, 05:46:30) [GCC 7.3.0] on linux Type "help", "copyright", "credits" or "license" for more information. >>> import os >>> os.getcwd() '/home/student' **>>** 
```

# 更改目录

使用`os`模块，我们可以更改当前工作目录。为此，`os`模块具有`chdir()`函数，例如：

```py
>>> os.chdir('/home/student/work') >>> print(os.getcwd()) /home/student/work >>> 
```

# 列出文件和目录

在 Python 中列出目录内容很容易。我们将使用`os`模块，该模块具有一个名为`listdir()`的函数，该函数将返回工作目录中文件和目录的名称：

```py
>>> os.listdir() ['Public', 'python_learning', '.ICEauthority', '.python_history', 'work', '.bashrc', 'Pictures', '.gnupg', '.cache', '.bash_logout', '.sudo_as_admin_successful', '.bash_history', '.config', '.viminfo', 'Desktop', 'Documents', 'examples.desktop', 'Videos', '.ssh', 'Templates', '.profile', 'dir', '.pam_environment', 'Downloads', '.local', '.dbus', 'Music', '.mozilla'] >>> 
```

# 重命名目录

Python 中的`os`模块具有一个`rename()`函数，可帮助更改目录的名称：

```py
>>> os.rename('work', 'work1') >>> os.listdir() ['Public', 'work1', 'python_learning', '.ICEauthority', '.python_history', '.bashrc', 'Pictures', '.gnupg', '.cache', '.bash_logout', '.sudo_as_admin_successful', '.bash_history', '.config', '.viminfo', 'Desktop', 'Documents', 'examples.desktop', 'Videos', '.ssh', 'Templates', '.profile', 'dir', '.pam_environment', 'Downloads', '.local', '.dbus', 'Music', '.mozilla'] **>>** 
```

# 复制，移动，重命名和删除数据

我们将学习系统管理员在数据上执行的四种基本操作，即复制，移动，重命名和删除。Python 有一个名为`shutil`的内置模块，可以执行这些任务。使用`shutil`模块，我们还可以对数据执行高级操作。要在程序中使用`shutil`模块，只需编写`import shutil`导入语句。`shutil`模块提供了一些支持文件复制和删除操作的函数。让我们逐一了解这些操作。

# 复制数据

在本节中，我们将看到如何使用`shutil`模块复制文件。为此，首先我们将创建一个`hello.py`文件并在其中写入一些文本。

`hello.py`：

```py
print ("") print ("Hello World\n") print ("Hello Python\n")
```

现在，我们将编写将内容复制到`shutil_copy_example.py`脚本的代码。在其中写入以下内容：

```py
import shutil import os shutil.copy('hello.py', 'welcome.py') print("Copy Successful")
```

按照以下方式运行脚本：

```py
$ python3 shutil_copy_example.py Output: Copy Successful
```

检查`welcome.py`脚本的存在，并且您将发现`hello.py`的内容已成功复制到`welcome.py`中。

# 移动数据

在这里，我们将看到如何移动数据。我们将使用`shutil.move()`来实现这个目的。`shutil.move(source, destination)`将文件从源移动到目标。现在，我们将创建一个`shutil_move_example.py`脚本，并在其中写入以下内容：

```py
import shutil shutil.move('/home/student/sample.txt', '/home/student/Desktop/.')
```

按照以下方式运行脚本：

```py
$ python3 shutil_move_example.py
```

在此脚本中，我们要移动的文件是`sample.txt`，它位于`/home/student`目录中。`/home/student`是我们的源文件夹，`/home/student/Desktop`是我们的目标文件夹。因此，在运行脚本后，`sample.txt`将从`/home/student`移动到`/home/student/Desktop`目录。

# 重命名数据

在上一节中，我们学习了如何使用`shutil.move()`将文件从源移动到目标。使用`shutil.move()`，文件可以被重命名。创建一个`shutil_rename_example.py`脚本，并在其中写入以下内容：

```py
import shutil shutil.move('hello.py', 'hello_renamed.py')
```

按照以下方式运行脚本：

```py
$ python3 shutil_rename_example.py
```

输出：

现在，检查您的文件名是否已重命名为`hello_renamed.py`。

# 删除数据

我们将学习如何使用 Python 的`os`模块删除文件和文件夹。`os`模块的`remove()`方法将删除一个文件。如果您尝试使用此方法删除目录，它将给出一个`OSError`。要删除目录，请使用`rmdir()`。

现在，创建一个`os_remove_file_directory.py`脚本，并在其中写入以下内容：

```py
import os os.remove('sample.txt') print("File removed successfully") os.rmdir('work1') print("Directory removed successfully")
```

按照以下方式运行脚本：

```py
$ python3 os_remove_file_directory.py Output: File removed successfully Directory removed successfully
```

# 处理路径

现在，我们将学习关于`os.path()`的知识。它用于路径操作。在本节中，我们将看一些`os`模块为路径名提供的函数。

启动`python3`控制台：

```py
student@ubuntu:~$ python3
Python 3.6.6 (default, Sep 12 2018, 18:26:19)
[GCC 8.0.1 20180414 (experimental) [trunk revision 259383]] on linux
Type "help", "copyright", "credits" or "license" for more information.
<q>>></q>
```

+   `os.path.absname(path)`: 返回路径名的绝对版本。

```py
>>> import os >>> os.path.abspath('sample.txt') '/home/student/work/sample.txt'
```

+   `os.path.dirname(path)`: 返回路径的目录名。

```py
>>> os.path.dirname('/home/student/work/sample.txt') '/home/student/work'
```

+   `os.path.basename(path)`: 返回路径的基本名称。

```py
>>> os.path.basename('/home/student/work/sample.txt') 'sample.txt'
```

+   `os.path.exists(path)`: 如果路径指向现有路径，则返回`True`。

```py
>>> os.path.exists('/home/student/work/sample.txt') True
```

+   `os.path.getsize(path)`: 返回以字节为单位的输入路径的大小。

```py
>>> os.path.getsize('/home/student/work/sample.txt') 39
```

+   `os.path.isfile(path)`: 检查输入的路径是否为现有文件。如果是文件，则返回`True`。

```py
>>> os.path.isfile('/home/student/work/sample.txt') True
```

+   `os.path.isdir(path)`: 检查输入的路径是否为现有目录。如果是目录，则返回`True`。

```py
>>> os.path.isdir('/home/student/work/sample.txt') False
```

# 比较数据

在这里，我们将学习如何在 Python 中比较数据。我们将使用`pandas`模块来实现这个目的。

Pandas 是一个开源的数据分析库，提供了易于使用的数据结构和数据分析工具。它使导入和分析数据变得更容易。

在开始示例之前，请确保您的系统上已安装了`pandas`。您可以按照以下方式安装 pandas：

```py
pip3 install pandas     --- For Python3 
or
 pip install pandas       --- For python2
```

我们将学习使用 pandas 比较数据的一个例子。首先，我们将创建两个`csv`文件：`student1.csv`和`student2.csv`。我们将比较这两个`csv`文件的数据，并且输出应该返回比较结果。创建两个`csv`文件如下：

创建`student1.csv`文件内容如下：

```py
Id,Name,Gender,Age,Address 101,John,Male,20,New York 102,Mary,Female,18,London 103,Aditya,Male,22,Mumbai 104,Leo,Male,22,Chicago 105,Sam,Male,21,Paris 106,Tina,Female,23,Sydney
```

创建`student2.csv`文件内容如下：

```py
Id,Name,Gender,Age,Address 101,John,Male,21,New York 102,Mary,Female,20,London 103,Aditya,Male,22,Mumbai 104,Leo,Male,23,Chicago 105,Sam,Male,21,Paris 106,Tina,Female,23,Sydney
```

现在，我们将创建一个`compare_data.py`脚本，并在其中写入以下内容：

```py
import pandas as pd df1 = pd.read_csv("student1.csv") df2 = pd.read_csv("student2.csv") s1 = set([ tuple(values) for values in df1.values.tolist()]) s2 = set([ tuple(values) for values in df2.values.tolist()]) s1.symmetric_difference(s2) print (pd.DataFrame(list(s1.difference(s2))), '\n') print (pd.DataFrame(list(s2.difference(s1))), '\n')
```

按照以下方式运行脚本：

```py
$ python3 compare_data.py Output:
 0     1       2   3         4 0  102  Mary  Female  18    London 1  104   Leo    Male  22   Chicago 2  101  John    Male  20  New York

 0     1       2   3         4 0  101  John    Male  21  New York 1  104   Leo    Male  23   Chicago 2  102  Mary  Female  20    London
```

在前面的例子中，我们正在比较两个`csv`文件之间的数据：`student1.csv`和`student2.csv`。我们首先将我们的数据帧(`df1`，`df2`)转换为集合(`s1`，`s2`)。然后，我们使用`symmetric_difference()`集合。因此，它将检查`s1`和`s2`之间的对称差异，然后我们将打印结果。

# 合并数据

我们将学习如何在 Python 中合并数据。为此，我们将使用 Python 的 pandas 库。为了合并数据，我们将使用在上一节中已创建的两个`csv`文件，`student1.csv`和`student2.csv`。

现在，创建一个`merge_data.py`脚本，并在其中写入以下代码：

```py
import pandas as pd df1 = pd.read_csv("student1.csv") df2 = pd.read_csv("student2.csv") result = pd.concat([df1, df2]) print(result)
```

按如下方式运行脚本：

```py
$ python3 merge_data.py Output:
 Id    Name  Gender  Age   Address 0  101    John    Male   20  New York 1  102    Mary  Female   18    London 2  103  Aditya    Male   22    Mumbai 3  104     Leo    Male   22   Chicago 4  105     Sam    Male   21     Paris 5  106    Tina  Female   23    Sydney 0  101    John    Male   21  New York 1  102    Mary  Female   20    London 2  103  Aditya    Male   22    Mumbai 3  104     Leo    Male   23   Chicago 4  105     Sam    Male   21     Paris 5  106    Tina  Female   23    Sydney
```

# 模式匹配文件和目录

在本节中，我们将学习有关文件和目录的模式匹配。Python 有`glob`模块，用于查找与特定模式匹配的文件和目录的名称。

现在，我们将看一个例子。首先，创建一个`pattern_match.py`脚本，并在其中写入以下内容：

```py
import glob file_match = glob.glob('*.txt') print(file_match) file_match = glob.glob('[0-9].txt') print(file_match) file_match = glob.glob('**/*.txt', recursive=True) print(file_match) file_match = glob.glob('**/', recursive=True) print(file_match)
```

按照以下方式运行脚本：

```py
$ python3 pattern_match.py Output: ['file1.txt', 'filea.txt', 'fileb.txt', 'file2.txt', '2.txt', '1.txt', 'file.txt'] ['2.txt', '1.txt'] ['file1.txt', 'filea.txt', 'fileb.txt', 'file2.txt', '2.txt', '1.txt', 'file.txt', 'dir1/3.txt', 'dir1/4.txt'] ['dir1/']
```

在上一个例子中，我们使用了 Python 的`glob`模块进行模式匹配。`glob`(路径名)将返回与路径名匹配的名称列表。在我们的脚本中，我们在三个不同的`glob()`函数中传递了三个路径名。在第一个`glob()`中，我们传递了路径名`*.txt;`，这将返回所有具有`.txt`扩展名的文件名。在第二个`glob()`中，我们传递了`[0-9].txt`，这将返回以数字开头的文件名。在第三个`glob()`中，我们传递了`**/*.txt`，它将返回文件名以及目录名。它还将返回这些目录中的文件名。在第四个`glob()`中，我们传递了`**/`，它将仅返回目录名。

# 元数据：关于数据的数据

在本节中，我们将学习`pyPdf`模块，该模块有助于从`pdf`文件中提取元数据。但首先，什么是元数据？元数据是关于数据的数据。元数据是描述主要数据的结构化信息。元数据是数据的摘要。它包含有关实际数据的基本信息。它有助于找到数据的特定实例。

确保您的目录中有`pdf`文件，您想从中提取信息。

首先，我们必须安装`pyPdf`模块，如下所示：

```py
pip install pyPdf
```

现在，我们将编写一个`metadata_example.py`脚本，并查看如何从中获取元数据信息。我们将在 Python 2 中编写此脚本：

```py
import pyPdf def main():
 file_name = '/home/student/sample_pdf.pdf' pdfFile = pyPdf.PdfFileReader(file(file_name,'rb')) pdf_data = pdfFile.getDocumentInfo() print ("----Metadata of the file----") for md in pdf_data: print (md+ ":" +pdf_data[md]) if __name__ == '__main__':
 main()
```

按照以下方式运行脚本：

```py
student@ubuntu:~$ python metadata_example.py ----Metadata of the file---- /Producer:Acrobat Distiller Command 3.0 for SunOS 4.1.3 and later (SPARC) /CreationDate:D:19980930143358
```

在前面的脚本中，我们使用了 Python 2 的`pyPdf`模块。首先，我们创建了一个`file_name`变量，用于存储我们的`pdf`的路径。使用`PdfFileReader()`，数据被读取。`pdf_data`变量将保存有关您的`pdf`的信息。最后，我们编写了一个 for 循环来获取元数据信息。

# 压缩和恢复

在本节中，我们将学习`shutil`模块的`make_archive()`函数，该函数将压缩整个目录。为此，我们将编写一个`compress_a_directory.py`脚本，并在其中写入以下内容：

```py
import shutil shutil.make_archive('work', 'zip', 'work/')
```

按照以下方式运行脚本：

```py
$ python3 compress_a_directory.py
```

在前面的脚本中，在`shutil.make_archive()`函数中，我们将第一个参数作为我们压缩文件的名称。`zip`将是我们的压缩技术。然后，`work/`将是我们要压缩的目录的名称。

现在，要从压缩文件中恢复数据，我们将使用`shutil`模块的`unpack_archive()`函数。创建一个`unzip_a_directory.py`脚本，并在其中写入以下内容：

```py
import shutil shutil.unpack_archive('work1.zip')
```

按照以下方式运行脚本：

```py
$ python3 unzip_a_directory.py
```

现在，检查您的目录。解压目录后，您将获得所有内容。

# 使用 tarfile 模块创建 TAR 存档

本节将帮助您了解如何使用 Python 的`tarfile`模块创建 tar 存档。

`tarfile`模块用于使用`gzip`、`bz2`压缩技术读取和写入 tar 存档。确保必要的文件和目录存在。现在，创建一个`tarfile_example.py`脚本，并在其中写入以下内容：

```py
import tarfile tar_file = tarfile.open("work.tar.gz", "w:gz") for name in ["welcome.py", "hello.py", "hello.txt", "sample.txt", "sample1.txt"]:
 tar_file.add(name) tar_file.close()
```

按照以下方式运行脚本：

```py
$ python3 tarfile_example.py
```

现在，检查您当前的工作目录；您会看到`work.tar.gz`已经被创建。

# 使用 tarfile 模块检查 TAR 文件的内容

在本节中，我们将学习如何在不实际提取 tar 文件的情况下检查已创建的 tar 存档的内容。我们将使用 Python 的`tarfile`模块进行操作。

创建一个`examine_tar_file_content.py`脚本，并在其中写入以下内容：

```py
import tarfile tar_file = tarfile.open("work.tar.gz", "r:gz") print(tar_file.getnames())
```

按照以下方式运行脚本：

```py
$ python3 examine_tar_file_content.py Output: ['welcome.py', 'hello.py', 'hello.txt', 'sample.txt', 'sample1.txt']
```

在先前的示例中，我们使用了`tarfile`模块来检查创建的 tar 文件的内容。我们使用了`getnames()`函数来读取数据。

# 总结

在本章中，我们学习了处理文件和目录的 Python 脚本。我们还学习了如何使用`os`模块处理目录。我们学习了如何复制、移动、重命名和删除文件和目录。我们还学习了 Python 中的 pandas 模块，用于比较和合并数据。我们学习了如何创建 tar 文件并使用`tarfile`模块读取 tar 文件的内容。我们还在搜索文件和目录时进行了模式匹配。

在下一章中，我们将学习`tar`存档和 ZIP 创建。

# 问题

1.  如何处理不同路径，而不考虑不同的操作系统（Windows，Llinux）？

1.  Python 中`print()`的不同参数是什么？

1.  在 Python 中，`dir()`关键字的用途是什么？

1.  `pandas`中的数据框，系列是什么？

1.  什么是列表推导？

1.  我们可以使用集合推导和字典推导吗？如果可以，如何操作？

1.  如何使用 pandas dataframe 打印第一个/最后一个`N`行？

1.  使用列表推导编写一个打印奇数的程序

1.  `sys.argv` 的类型是什么？

1.  a) 集合

1.  b) 列表

1.  c) 元组

1.  d) 字符串

# 进一步阅读

+   `pathlib` 文档： [`docs.python.org/3/library/pathlib.html`](https://docs.python.org/3/library/pathlib.html)

+   [`pandas` 文档：](https://pandas.pydata.org/pandas-docs/stable/)[`pandas.pydata.org/pandas-docs/stable/`](https://pandas.pydata.org/pandas-docs/stable/)

+   `os` 模块文档：[`docs.python.org/3/library/os.html`](https://docs.python.org/3/library/os.html)
