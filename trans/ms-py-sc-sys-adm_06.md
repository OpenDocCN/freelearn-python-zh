# 文件存档、加密和解密

在上一章中，我们学习了如何处理文件、目录和数据。我们还学习了`tarfile`模块。在本章中，我们将学习文件存档、加密和解密。存档在管理文件、目录和数据方面起着重要作用。但首先，什么是存档？存档是将文件和目录存储到单个文件中的过程。Python有`tarfile`模块用于创建这样的存档文件。

在本章中，我们将涵盖以下主题：

+   创建和解压存档

+   Tar存档

+   ZIP创建

+   文件加密和解密

# 创建和解压存档

在本节中，我们将学习如何使用Python的`shutil`模块创建和解压存档。`shutil`模块有`make_archive()`函数，用于创建新的存档文件。使用`make_archive()`，我们可以存档整个目录及其内容。

# 创建存档

现在，我们将编写一个名为`shutil_make_archive.py`的脚本，并在其中编写以下内容：

```py
import tarfile
import shutil
import sys

shutil.make_archive(
 'work_sample', 'gztar',
 root_dir='..',
 base_dir='work',
)
print('Archive contents:')
with tarfile.open('work_sample.tar.gz', 'r') as t_file:
 for names in t_file.getnames():
 print(names)
```

运行程序，您将得到以下输出：

```py
$ python3 shutil_make_archive.py
Archive contents:
work
work/bye.py
work/shutil_make_archive.py
work/welcome.py
work/hello.py
```

在前面的例子中，为了创建一个存档文件，我们使用了Python的`shutil`和`tarfile`模块。在`shutil.make_archive()`中，我们指定了`work_sample`，这将是存档文件的名称，并且将以`gz`格式。我们在基本目录属性中指定了我们的工作目录名称。最后，我们打印了已存档的文件的名称。

# 解压存档

要解压缩存档，`shutil`模块有`unpack_archive()`函数。使用此函数，我们可以提取存档文件。我们传递了存档文件名和我们想要提取内容的目录。如果没有传递目录名称，则它将提取内容到您当前的工作目录中。

现在，创建一个名为`shutil_unpack_archive.py`的脚本，并在其中编写以下代码：

```py
import pathlib
import shutil
import sys
import tempfile
with tempfile.TemporaryDirectory() as d:
 shutil.unpack_archive('work_sample.tar.gz', extract_dir='/home/student/work',)
 prefix_len = len(d) + 1
 for extracted in pathlib.Path(d).rglob('*'):
 print(str(extracted)[prefix_len:])
```

按照以下方式运行脚本：

```py
student@ubuntu:~/work$ python3 shutil_unpack_archive.py
```

现在，检查您的`work/`目录，您将在其中找到`work/`文件夹，其中将有提取的文件。

# Tar存档

在本节中，我们将学习`tarfile`模块。我们还将学习如何测试输入的文件名，评估它是否是有效的存档文件。我们将看看如何将新文件添加到已存档的文件中，如何使用`tarfile`模块读取元数据，以及如何使用`extractall()`函数从存档中提取文件。

首先，我们将测试输入的文件名是否是有效的存档文件。为了测试这一点，`tarfile`模块有`is_tarfile()`函数，它返回一个布尔值。

创建一个名为`check_archive_file.py`的脚本，并在其中编写以下内容：

```py
import tarfile

for f_name in ['hello.py', 'work.tar.gz', 'welcome.py', 'nofile.tar', 'sample.tar.xz']:
 try:
 print('{:} {}'.format(f_name, tarfile.is_tarfile(f_name)))
 except IOError as err:
 print('{:} {}'.format(f_name, err))
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 check_archive_file.py
hello.py          False
work.tar.gz      True
welcome.py     False
nofile.tar         [Errno 2] No such file or directory: 'nofile.tar'
sample.tar.xz   True
```

因此，`tarfile.is_tarfile()`将检查列表中提到的每个文件名。`hello.py，welcome.py`文件不是tar文件，所以我们得到了一个布尔值`False`。`work.tar.gz`和`sample.tar.xz`是tar文件，所以我们得到了布尔值`True`。而我们的目录中没有`nofile.tar`这样的文件，所以我们得到了一个异常，因为我们在脚本中写了它。

现在，我们将在已创建的存档文件中添加一个新文件。创建一个名为`add_to_archive.py`的脚本，并在其中编写以下代码：

```py
import shutil
import os
import tarfile
print('creating archive')
shutil.make_archive('work', 'tar', root_dir='..', base_dir='work',)
print('\nArchive contents:')
with tarfile.open('work.tar', 'r') as t_file:
 for names in t_file.getnames():
 print(names)
os.system('touch sample.txt')
print('adding sample.txt')
with tarfile.open('work.tar', mode='a') as t:
 t.add('sample.txt')
print('contents:',)
with tarfile.open('work.tar', mode='r') as t:
 print([m.name for m in t.getmembers()])
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 add_to_archive.py
Output :
creating archive
Archive contents:
work
work/bye.py
work/shutil_make_archive.py
work/check_archive_file.py
work/welcome.py
work/add_to_archive.py
work/shutil_unpack_archive.py
work/hello.py
adding sample.txt
contents:
['work', 'work/bye.py', 'work/shutil_make_archive.py', 'work/check_archive_file.py', 'work/welcome.py', 'work/add_to_archive.py', 'work/shutil_unpack_archive.py', 'work/hello.py', 'sample.txt']
```

在这个例子中，我们首先使用`shutil.make_archive()`创建了一个存档文件，然后打印了存档文件的内容。然后我们在下一个语句中创建了一个`sample.txt`文件。现在，我们想要将`sample.txt`添加到已创建的`work.tar`中。在这里，我们使用了追加模式`a`。接下来，我们再次显示存档文件的内容。

现在，我们将学习如何从存档文件中读取元数据。`getmembers()`函数将加载文件的元数据。创建一个名为`read_metadata.py`的脚本，并在其中编写以下内容：

```py
import tarfile
import time
with tarfile.open('work.tar', 'r') as t:
 for file_info in t.getmembers():
 print(file_info.name)
 print("Size   :", file_info.size, 'bytes')
 print("Type   :", file_info.type)
 print()
```

运行脚本，你将得到以下输出：

```py
student@ubuntu:~/work$ python3 read_metadata.py
Output:
work/bye.py
Size : 30 bytes
Type : b'0' 
work/shutil_make_archive.py
Size : 243 bytes
Type : b'0'
work/check_archive_file.py
Size : 233 bytes
Type : b'0'

work/welcome.py
Size : 48 bytes
Type : b'0'

work/add_to_archive.py
Size : 491 bytes
Type : b'0'

work/shutil_unpack_archive.py
Size : 279 bytes
Type : b'0'
```

现在，我们将使用`extractall()`函数从存档中提取内容。为此，创建一个名为`extract_contents.py`的脚本，并在其中写入以下代码：

```py
import tarfile
import os
os.mkdir('work')
with tarfile.open('work.tar', 'r') as t:
 t.extractall('work')
print(os.listdir('work'))
```

运行脚本，你将得到以下输出：

```py
student@ubuntu:~/work$ python3 extract_contents.py
```

检查你的当前工作目录，你会发现`work/`目录。导航到该目录，你可以找到你提取的文件。

# ZIP创建

在本节中，我们将学习关于ZIP文件的知识。我们将学习`python`的`zipfile`模块，如何创建ZIP文件，如何测试输入的文件名是否是有效的`zip`文件名，读取元数据等等。

首先，我们将学习如何使用`shutil`模块的`make_archive()`函数创建一个`zip`文件。创建一个名为`make_zip_file.py`的脚本，并在其中写入以下代码：

```py
import shutil
shutil.make_archive('work', 'zip', 'work')
```

按如下方式运行脚本：

```py
student@ubuntu:~$ python3 make_zip_file.py
```

现在检查你的当前工作目录，你会看到`work.zip`。

现在，我们将测试输入的文件名是否是一个`zip`文件。为此，`zipfile`模块有`is_zipfile()`函数。

创建一个名为`check_zip_file.py`的脚本，并在其中写入以下内容：

```py
import zipfile
for f_name in ['hello.py', 'work.zip', 'welcome.py', 'sample.txt', 'test.zip']:
 try:
 print('{:}           {}'.format(f_name, zipfile.is_zipfile(f_name)))
 except IOError as err:
 print('{:}           {}'.format(f_name, err))
```

按如下方式运行脚本：

```py
student@ubuntu:~$ python3 check_zip_file.py
Output :
hello.py          False
work.zip         True
welcome.py     False
sample.txt       False
test.zip            True
```

在这个例子中，我们使用了一个`for`循环，我们在其中检查列表中的文件名。`is_zipfile()`函数将逐个检查文件名，并将布尔值作为结果。

现在，我们将看看如何使用Python的`zipfile`模块从存档的ZIP文件中读取元数据。创建一个名为`read_metadata.py`的脚本，并在其中写入以下内容：

```py
import zipfile

def meta_info(names):
 with zipfile.ZipFile(names) as zf:
 for info in zf.infolist():
 print(info.filename)
 if info.create_system == 0:
 system = 'Windows'
 elif info.create_system == 3:
 system = 'Unix'
 else:
 system = 'UNKNOWN'
 print("System         :", system)
 print("Zip Version    :", info.create_version)
 print("Compressed     :", info.compress_size, 'bytes')
 print("Uncompressed   :", info.file_size, 'bytes')
 print()

if __name__ == '__main__':
 meta_info('work.zip')
```

按如下方式执行脚本：

```py
student@ubuntu:~$ python3 read_metadata.py
Output:
sample.txt
System         : Unix
Zip Version    : 20
Compressed     : 2 bytes
Uncompressed   : 0 bytes

bye.py
System         : Unix
Zip Version    : 20
Compressed     : 32 bytes
Uncompressed   : 30 bytes

extract_contents.py
System         : Unix
Zip Version    : 20
Compressed     : 95 bytes
Uncompressed   : 132 bytes

shutil_make_archive.py
System         : Unix
Zip Version    : 20
Compressed     : 160 bytes
Uncompressed   : 243 bytes
```

为了获取`zip`文件的元数据信息，我们使用了`ZipFile`类的`infolist()`方法。

# 文件加密和解密

在本节中，我们将学习Python的`pyAesCrypt`模块。`pyAesCrypt`是一个文件加密模块，它使用`AES256-CBC`来加密/解密文件和二进制流。

按如下方式安装`pyAesCrypt`：

```py
pip3 install pyAesCrypt
```

创建一个名为`file_encrypt.py`的脚本，并在其中写入以下代码：

```py
import pyAesCrypt

from os import stat, remove
# encryption/decryption buffer size - 64K
bufferSize = 64 * 1024
password = "#Training"
with open("sample.txt", "rb") as fIn:
 with open("sample.txt.aes", "wb") as fOut:
 pyAesCrypt.encryptStream(fIn, fOut, password, bufferSize)
# get encrypted file size
encFileSize = stat("sample.txt.aes").st_size 
```

按如下方式运行脚本：

```py
student@ubuntu:~/work$ python3 file_encrypt.py
Output :
```

请检查你的当前工作目录。你会在其中找到加密文件`sample.txt.aes`。

在这个例子中，我们已经提到了缓冲区大小和密码。接下来，我们提到了要加密的文件名。在`encryptStream`中，我们提到了`fIn`，这是我们要加密的文件，以及`fOut`，这是我们加密后的文件名。我们将加密后的文件存储为`sample.txt.aes`。

现在，我们将解密`sample.txt.aes`文件以获取文件内容。创建一个名为`file_decrypt.py`的脚本，并在其中写入以下内容：

```py
import pyAesCrypt
from os import stat, remove
bufferSize = 64 * 1024
password = "#Training"
encFileSize = stat("sample.txt.aes").st_size
with open("sample.txt.aes", "rb") as fIn:
 with open("sampleout.txt", "wb") as fOut:
 try:
 pyAesCrypt.decryptStream(fIn, fOut, password, bufferSize, encFileSize)
 except ValueError:
 remove("sampleout.txt")
```

按如下方式运行脚本：

```py
student@ubuntu:~/work$ python3 file_decrypt.py
```

现在，检查你的当前工作目录。将会创建一个名为`sampleout.txt`的文件。那就是你的解密文件。

在这个例子中，我们提到了要解密的文件名，即`sample.txt.aes`。接下来，我们的解密文件将是`sampleout.txt`。在`decryptStream()`中，我们提到了`fIn`，这是我们要解密的文件，以及`fOut`，这是`解密`文件的名称。

# 总结

在本章中，我们学习了如何创建和提取存档文件。存档在管理文件、目录和数据方面起着重要作用。它还将文件和目录存储到一个单一文件中。

我们详细学习了Python模块`tarfile`和`zipfile`，它们使你能够创建、提取和测试存档文件。你将能够将一个新文件添加到已存档的文件中，读取元数据，从存档中提取文件。你还学习了使用`pyAescrypt`模块进行文件加密和解密。

在下一章中，你将学习Python中的文本处理和正则表达式。Python有一个非常强大的库叫做正则表达式，它可以执行搜索和提取数据等任务。

# 问题

1.  我们能使用密码保护来压缩数据吗？如果可以，怎么做？

1.  什么是Python中的上下文管理器？

1.  什么是pickling和unpickling？

1.  Python中有哪些不同类型的函数？

# 进一步阅读

+   数据压缩和归档：[https://docs.python.org/3/library/archiving.html](https://docs.python.org/3/library/archiving.html)

+   `tempfile`文档：[https://docs.python.org/2/library/tempfile.html](https://docs.python.org/2/library/tempfile.html)

+   密码学Python文档：[https://docs.python.org/3/library/crypto.html](https://docs.python.org/3/library/crypto.html)

+   `shutil`文档：[https://docs.python.org/3/library/shutil.html](https://docs.python.org/3/library/shutil.html)
