# 第四章：自动化常规管理活动

系统管理员执行各种管理活动。这些活动可能包括文件处理、日志记录、管理 CPU 和内存、处理密码，以及最重要的是进行备份。这些活动需要自动化。在本章中，我们将学习如何使用 Python 自动化这些活动。

在本章中，我们将涵盖以下主题：

+   通过重定向、管道和输入文件接受输入

+   在脚本中在运行时处理密码

+   执行外部命令并获取它们的输出

+   在运行时提示密码和验证

+   读取配置文件

+   向脚本添加日志和警告代码

+   限制 CPU 和内存使用

+   启动 Web 浏览器

+   使用`os`模块处理目录和文件

+   进行备份（使用`rsync`）

# 通过重定向、管道和输入文件接受输入

在本节中，我们将学习用户如何通过重定向、管道和外部输入文件接受输入。

接受重定向输入时，我们使用`stdin`。`pipe`是另一种重定向形式。这个概念意味着将一个程序的输出作为另一个程序的输入。我们可以通过外部文件和使用 Python 来接受输入。

# 通过重定向输入

`stdin`和`stdout`是`os`模块创建的对象。我们将编写一个脚本，其中我们将使用`stdin`和`stdout`。

创建一个名为`redirection.py`的脚本，并在其中编写以下代码：

```py
import sys class Redirection(object):
 def __init__(self, in_obj, out_obj): self.input = in_obj self.output = out_obj def read_line(self): res = self.input.readline() self.output.write(res) return resif __name__ == '__main__':
 if not sys.stdin.isatty(): sys.stdin = Redirection(in_obj=sys.stdin, out_obj=sys.stdout) a = input('Enter a string: ') b = input('Enter another string: ') print ('Entered strings are: ', repr(a), 'and', repr(b)) 
```

如下运行上述程序：

```py
$ python3 redirection.py 
```

我们将收到以下输出：

```py
Output: Enter a string: hello Enter another string: python Entered strings are:  'hello' and 'python'
```

每当程序在交互会话中运行时，`stdin`是键盘输入，`stdout`是用户的终端。`input()`函数用于从用户那里获取输入，`print()`是在终端（`stdout`）上写入的方式。

# 通过管道输入

管道是另一种重定向形式。这种技术用于将信息从一个程序传递到另一个程序。`|`符号表示管道。通过使用管道技术，我们可以以一种使一个命令的输出作为下一个命令的输入的方式使用两个以上的命令。

现在，我们将看看如何使用管道接受输入。为此，首先我们将编写一个返回`floor`除法的简单脚本。创建一个名为`accept_by_pipe.py`的脚本，并在其中编写以下代码：

```py
import sys for n in sys.stdin:
 print ( int(n.strip())//2 ) 
```

运行脚本，您将得到以下输出：

```py
$ echo 15 | python3 accept_by_pipe.py Output: 7 
```

在上面的脚本中，`stdin`是键盘输入。我们对我们在运行时输入的数字执行`floor`除法。floor 除法仅返回商的整数部分。当我们运行程序时，我们传递`15`，然后是管道`|`符号，然后是我们的脚本名称。因此，我们将`15`作为输入提供给我们的脚本。因此进行了 floor 除法，我们得到输出为`7`。

我们可以向我们的脚本传递多个输入。因此，在以下执行中，我们已经传递了多个输入值，如`15`、`45`和`20`。为了处理多个输入值，我们在脚本中编写了一个`for`循环。因此，它将首先接受`15`作为输入，然后是`45`，然后是`20`。输出将以新行打印出每个输入，因为我们在输入值之间写了`\n`。为了启用对反斜杠的这种解释，我们传递了`-e`标志：

```py
$ echo -e '15\n45\n20' | python3 accept_by_pipe.py Output: 7 22 10
```

运行后，我们得到了`15`、`45`和`20`的 floor 除法分别为`7`、`22`和`10`，显示在新行上。

# 通过输入文件输入

在本节中，我们将学习如何从输入文件中获取输入。在 Python 中，从输入文件中获取输入更容易。我们将看一个例子。但首先，我们将创建一个名为`sample.txt`的简单文本文件，并在其中编写以下代码：

`Sample.txt`：

```py
Hello World Hello Python
```

现在，创建一个名为`accept_by_input_file.py`的脚本，并在其中编写以下代码：

```py
i = open('sample.txt','r')
o = open('sample_output.txt','w')

a = i.read()
o.write(a)
```

运行程序，您将得到以下输出：

```py
$ python3 accept_by_input_file.py $ cat sample_output.txt Hello World Hello Python
```

# 在脚本中在运行时处理密码

在本节中，我们将看一个简单的处理脚本密码的例子。我们将创建一个名为`handling_password.py`的脚本，并在其中编写以下内容：

```py
import sys import paramiko import time ip_address = "192.168.2.106" username = "student" password = "training" ssh_client = paramiko.SSHClient() ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) ssh_client.load_system_host_keys() ssh_client.connect(hostname=ip_address,\
 username=username, password=password) print ("Successful connection", ip_address) ssh_client.invoke_shell() remote_connection = ssh_client.exec_command('cd Desktop; mkdir work\n') remote_connection = ssh_client.exec_command('mkdir test_folder\n') #print( remote_connection.read() ) ssh_client.close
```

运行上述脚本，您将收到以下输出：

```py
$ python3 handling_password.py  Output: Successful connection 192.168.2.106
```

在上述脚本中，我们使用了`paramiko`模块。`paramiko`模块是`ssh`的 Python 实现，提供了客户端-服务器功能。

按以下方式安装`paramiko`：

```py
pip3 install paramiko
```

在上述脚本中，我们远程连接到主机`192.168.2.106`。我们在脚本中提供了主机的用户名和密码。

运行此脚本后，在`192.168.2.106`桌面上，您将找到一个`work`文件夹，并且`test_folder`可以在`192.168.2.106`的`home/`目录中找到。

# 执行外部命令并获取它们的输出

在本节中，我们将学习 Python 的 subprocess 模块。使用`subprocess`，很容易生成新进程并获取它们的返回代码，执行外部命令并启动新应用程序。

我们将看一下如何在 Python 中执行外部命令并获取它们的输出，使用`subprocess`模块。我们将创建一个名为`execute_external_commands.py`的脚本，并在其中编写以下代码：

```py
import subprocess subprocess.call(["touch", "sample.txt"]) subprocess.call(["ls"]) print("Sample file created") subprocess.call(["rm", "sample.txt"]) 
```

```py
subprocess.call(["ls"]) print("Sample file deleted")
```

运行程序，您将得到以下输出：

```py
$ python3 execute_external_commands.py Output: 1.py     accept_by_pipe.py      sample_output.txt       sample.txt accept_by_input_file.py         execute_external_commands.py         output.txt        sample.py Sample.txt file created 1.py     accept_by_input_file.py         accept_by_pipe.py execute_external_commands.py  output.txt            sample_output.txt       sample.py Sample.txt file deleted
```

# 使用 subprocess 模块捕获输出

在本节中，我们将学习如何捕获输出。我们将为`stdout`参数传递`PIPE`以捕获输出。编写一个名为`capture_output.py`的脚本，并在其中编写以下代码：

```py
import subprocess res = subprocess.run(['ls', '-1'], stdout=subprocess.PIPE,) print('returncode:', res.returncode) print(' {} bytes in stdout:\n{}'.format(len(res.stdout), res.stdout.decode('utf-8'))) 
```

按以下方式执行脚本：

```py
student@ubuntu:~$ python3 capture_output.py 
```

在执行时，我们将收到以下输出：

```py
Output: returncode: 0 191 bytes in stdout: 1.py accept_by_input_file.py accept_by_pipe.py execute_external_commands.py getpass_example.py ouput.txt output.txt password_prompt_again.py sample_output.txt sample.py capture_output.py
```

在上述脚本中，我们导入了 Python 的 subprocess 模块，它有助于捕获输出。subprocess 模块用于创建新进程。它还有助于连接输入/输出管道并获取返回代码。`subprocess.run()`将运行作为参数传递的命令。`Returncode`将是子进程的退出状态。在输出中，如果返回代码为`0`，表示它成功运行。

# 运行时提示密码和验证

在本节中，我们将学习如何在运行时处理密码，使用`getpass 模块`。Python 中的`getpass()`模块提示用户输入密码而不回显。`getpass`模块用于在程序通过终端与用户交互时处理密码提示。

我们将看一些如何使用`getpass`模块的例子：

1.  创建一个名为`no_prompt.py`的脚本，并在其中编写以下代码：

```py
import getpass try:
 p = getpass.getpass() except Exception as error:
 print('ERROR', error) else:
 print('Password entered:', p)
```

在此脚本中，不为用户提供提示。因此，默认情况下，它设置为`Password`提示。

按以下方式运行脚本：

```py
$ python3 no_prompt.py Output : Password: Password entered: abcd
```

1.  我们将提供一个输入密码的提示。因此，在其中创建一个名为`with_prompt.py`的脚本，并在其中编写以下代码：

```py
import getpass try:
 p = getpass.getpass("Enter your password: ") except Exception as error:
 print('ERROR', error) else:
 print('Password entered:', p)
```

现在，我们已经编写了一个脚本，用于提供密码的提示。按以下方式运行程序：

```py
$ python3 with_prompt.py Output: Enter your password: Password entered: abcd
```

在这里，我们为用户提供了“输入密码”的提示。

现在，我们将编写一个脚本，如果输入错误的密码，它将只是打印一个简单的消息，但不会再提示输入正确的密码。

1.  编写一个名为`getpass_example.py`的脚本，并在其中编写以下代码：

```py
import getpass passwd = getpass.getpass(prompt='Enter your password: ') if passwd.lower() == '#pythonworld':
 print('Welcome!!') else:
 print('The password entered is incorrect!!')
```

按以下方式运行程序（这里我们输入了正确的密码，即`#pythonworld`）：

```py
$ python3 getpass_example.py Output: Enter your password: Welcome!!
```

现在，我们将输入一个错误的密码，并检查我们收到什么消息：

```py
$ python3 getpass_example.py Output: Enter your password: The password entered is incorrect!!
```

在这里，我们编写了一个脚本，如果我们输入错误的密码，就不会再要求输入密码。

现在，我们将编写一个脚本，当我们提供错误的密码时，它将再次要求输入正确的密码。使用`getuser()`来获取用户的登录名。`getuser()`函数将返回系统登录的用户。创建一个名为`password_prompt_again.py`的脚本，并在其中编写以下代码：

```py
import getpass user_name = getpass.getuser() print ("User Name : %s" % user_name) while True:
 passwd = getpass.getpass("Enter your Password : ") if passwd == '#pythonworld': print ("Welcome!!!") break else: print ("The password you entered is incorrect.")
```

运行程序，您将得到以下输出：

```py
student@ubuntu:~$ python3 password_prompt_again.py User Name : student Enter your Password : The password you entered is incorrect. Enter your Password : Welcome!!!
```

# 读取配置文件

在本节中，我们将学习 Python 的`configparser`模块。通过使用`configparser`模块，您可以管理应用程序的用户可编辑的配置文件。

这些配置文件的常见用途是，用户或系统管理员可以使用简单的文本编辑器编辑文件以设置应用程序默认值，然后应用程序将读取并解析它们，并根据其中写入的内容进行操作。

要读取配置文件，`configparser`有`read()`方法。现在，我们将编写一个名为`read_config_file.py`的简单脚本。在那之前，创建一个名为`read_simple.ini`的`.ini`文件，并在其中写入以下内容：`read_simple.ini`

```py
[bug_tracker] url = https://timesofindia.indiatimes.com/
```

创建`read_config_file.py`并输入以下内容：

```py
from configparser import ConfigParser p = ConfigParser() p.read('read_simple.ini') print(p.get('bug_tracker', 'url'))
```

运行`read_config_file.py`，您将获得以下输出：

```py
$ python3 read_config_file.py  Output: https://timesofindia.indiatimes.com/
```

`read()`方法接受多个文件名。每当扫描每个文件名并且该文件存在时，它将被打开并读取。现在，我们将编写一个用于读取多个文件名的脚本。创建一个名为`read_many_config_file.py`的脚本，并在其中编写以下代码：

```py
from configparser import ConfigParser import glob p = ConfigParser() files = ['hello.ini', 'bye.ini', 'read_simple.ini', 'welcome.ini'] files_found = p.read(files) files_missing = set(files) - set(files_found) print('Files found:  ', sorted(files_found)) print('Files missing:  ', sorted(files_missing))
```

运行上述脚本，您将获得以下输出：

```py
$ python3 read_many_config_file.py  Output Files found:   ['read_simple.ini'] Files missing:   ['bye.ini', 'hello.ini', 'welcome.ini']
```

在上面的示例中，我们使用了 Python 的`configparser`模块，该模块有助于管理配置文件。首先，我们创建了一个名为`files`的列表。`read()`函数将读取配置文件。在示例中，我们创建了一个名为`files_found`的变量，它将存储目录中存在的配置文件的名称。接下来，我们创建了另一个名为`files_missing`的变量，它将返回目录中不存在的文件名。最后，我们打印存在和缺失的文件名。

# 向脚本添加日志记录和警告代码

在本节中，我们将学习 Python 的日志记录和警告模块。日志记录模块将跟踪程序中发生的事件。警告模块警告程序员有关语言和库中所做更改的信息。

现在，我们将看一个简单的日志记录示例。我们将编写一个名为`logging_example.py`的脚本，并在其中编写以下代码：

```py
import logging LOG_FILENAME = 'log.txt' logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG,) logging.debug('This message should go to the log file') with open(LOG_FILENAME, 'rt') as f:
 prg = f.read() print('FILE:') print(prg)
```

运行以下程序：

```py
$ python3 logging_example.py Output: FILE: DEBUG:root:This message should go to the log file
```

检查`hello.py`，您会看到在该脚本中打印的调试消息：

```py
$ cat log.txt  Output: DEBUG:root:This message should go to the log file
```

现在，我们将编写一个名为`logging_warnings_codes.py`的脚本，并在其中编写以下代码：

```py
import logging import warnings logging.basicConfig(level=logging.INFO,) warnings.warn('This warning is not sent to the logs') logging.captureWarnings(True) warnings.warn('This warning is sent to the logs')
```

运行以下脚本：

```py
$ python3 logging_warnings_codes.py Output: logging_warnings_codes.py:6: UserWarning: This warning is not sent to the logs
 warnings.warn('This warning is not sent to the logs') WARNING:py.warnings:logging_warnings_codes.py:10: UserWarning: This warning is sent to the logs
 warnings.warn('This warning is sent to the logs')
```

# 生成警告

`warn()`用于生成警告。现在，我们将看一个生成警告的简单示例。编写一个名为`generate_warnings.py`的脚本，并在其中编写以下代码：

```py
import warnings warnings.simplefilter('error', UserWarning) print('Before') warnings.warn('Write your warning message here') print('After')
```

如下所示运行脚本：

```py
$ python3 generate_warnings.py Output: Before: Traceback (most recent call last):
 File "generate_warnings.py", line 6, in <module> warnings.warn('Write your warning message here') UserWarning: Write your warning message here
```

在上面的脚本中，我们通过`warn()`传递了一个警告消息。我们使用了一个简单的过滤器，以便您的警告将被视为错误，并且程序员将相应地解决该错误。

# 限制 CPU 和内存使用

在本节中，我们将学习如何限制 CPU 和内存使用。首先，我们将编写一个用于限制 CPU 使用的脚本。创建一个名为`put_cpu_limit.py`的脚本，并在其中编写以下代码：

```py
import resource import sys import signal import time def time_expired(n, stack):
 print('EXPIRED :', time.ctime()) raise SystemExit('(time ran out)') signal.signal(signal.SIGXCPU, time_expired) # Adjust the CPU time limit soft, hard = resource.getrlimit(resource.RLIMIT_CPU) print('Soft limit starts as  :', soft) resource.setrlimit(resource.RLIMIT_CPU, (10, hard)) soft, hard = resource.getrlimit(resource.RLIMIT_CPU) print('Soft limit changed to :', soft) print() # Consume some CPU time in a pointless exercise print('Starting:', time.ctime()) for i in range(200000):
 for i in range(200000): v = i * i # We should never make it this far print('Exiting :', time.ctime())
```

如下所示运行上述脚本：

```py
$ python3 put_cpu_limit.py Output: Soft limit starts as  : -1 Soft limit changed to : 10 Starting: Thu Sep  6 16:13:20 2018 EXPIRED : Thu Sep  6 16:13:31 2018 (time ran out)
```

在上面的脚本中，我们使用`setrlimit()`来限制 CPU 使用。因此，在我们的脚本中，我们将限制设置为 10 秒。

# 启动 webbrowser

在本节中，我们将学习 Python 的`webbrowser`模块。该模块具有在浏览器应用程序中打开 URL 的功能。我们将看一个简单的示例。创建一个名为`open_web.py`的脚本，并在其中编写以下代码：

```py
import webbrowser webbrowser.open('https://timesofindia.indiatimes.com/world')
```

如下所示运行脚本：

```py
$ python3 open_web.py Output:
Url mentioned in open() will be opened in your browser.
webbrowser – Command line interface
```

您还可以通过命令行使用 Python 的`webbrowser`模块，并且可以使用所有功能。要通过命令行使用`webbrowser`，运行以下命令：

```py
$ python3 -m webbrowser -n https://www.google.com/
```

在这里，[`www.google.com/`](https://www.google.com/)将在浏览器窗口中打开。您可以使用以下两个选项：

+   `-n`：打开一个新窗口

+   `-t`：打开一个新标签

# 使用 os 模块处理目录和文件

在本节中，我们将学习 Python 的`os`模块。Python 的`os`模块有助于实现操作系统任务。如果我们想执行操作系统任务，我们需要导入`os`模块。

我们将看一些与处理文件和目录相关的示例。

# 创建和删除目录

在本节中，我们将创建一个脚本，我们将看到可以用于处理文件系统上的目录的函数，其中将包括创建、列出和删除内容。创建一个名为`os_dir_example.py`的脚本，并在其中编写以下代码：

```py
import os directory_name = 'abcd' print('Creating', directory_name) os.makedirs(directory_name) file_name = os.path.join(directory_name, 'sample_example.txt') print('Creating', file_name) with open(file_name, 'wt') as f:
 f.write('sample example file') print('Cleaning up') os.unlink(file_name) os.rmdir(directory_name)       # Will delete the directory
```

按以下方式运行脚本：

```py
$ python3 os_dir_example.py Output: Creating abcd Creating abcd/sample_example.txt Cleaning up
```

使用`mkdir()`创建目录时，所有父目录必须已经创建。但是，使用`makedirs()`创建目录时，它将创建任何在路径中提到但不存在的目录。`unlink()`将删除文件路径，`rmdir()`将删除目录路径。

# 检查文件系统的内容

在本节中，我们将使用`listdir()`列出目录的所有内容。创建一个名为`list_dir.py`的脚本，并在其中编写以下代码：

```py
import os import sys print(sorted(os.listdir(sys.argv[1])))
```

按以下方式运行脚本：

```py
$ python3 list_dir.py /home/student/ ['.ICEauthority', '.bash_history', '.bash_logout', '.bashrc', '.cache', '.config', '.gnupg', '.local', '.mozilla', '.pam_environment', '.profile', '.python_history', '.ssh', '.sudo_as_admin_successful', '.viminfo', '1.sh', '1.sh.x', '1.sh.x.c', 'Desktop', 'Documents', 'Downloads', 'Music', 'Pictures', 'Public', 'Templates', 'Videos', 'examples.desktop', 'execute_external_commands.py', 'log.txt', 'numbers.txt', 'python_learning', 'work']
```

因此，通过使用`listdir()`，您可以列出文件夹的所有内容。

# 备份（使用 rsync）

这是系统管理员必须做的最重要的工作。在本节中，我们将学习如何使用`rsync`进行备份。`rsync`命令用于在本地和远程复制文件和目录，并使用`rsync`执行数据备份。为此，我们将编写一个名为`take_backup.py`的脚本，并在其中编写以下代码：

```py
import os import shutil import time from sh import rsync def check_dir(os_dir):
 if not os.path.exists(os_dir): print (os_dir, "does not exist.") exit(1) def ask_for_confirm():
 ans = input("Do you want to Continue? yes/no\n") global con_exit if ans == 'yes': con_exit = 0 return con_exit elif ans == "no": con_exit = 1 return con_exit else:1 print ("Answer with yes or no.") ask_for_confirm() def delete_files(ending):
 for r, d, f in os.walk(backup_dir): for files in f: if files.endswith("." + ending): os.remove(os.path.join(r, files)) backup_dir = input("Enter directory to backup\n")   # Enter directory name check_dir(backup_dir) print (backup_dir, "saved.") time.sleep(3) backup_to_dir= input("Where to backup?\n") check_dir(backup_to_dir) print ("Doing the backup now!") ask_for_confirm() if con_exit == 1:
 print ("Aborting the backup process!") exit(1) rsync("-auhv", "--delete", "--exclude=lost+found", "--exclude=/sys", "--exclude=/tmp", "--exclude=/proc", "--exclude=/mnt", "--exclude=/dev", "--exclude=/backup", backup_dir, backup_to_dir)
```

按以下方式运行脚本：

```py
student@ubuntu:~/work$ python3 take_backup.py Output : Enter directory to backup /home/student/work /home/student/work saved. Where to backup? /home/student/Desktop Doing the backup now! Do you want to Continue? yes/no yes
```

现在，检查`Desktop/directory`，您将在该目录中看到您的工作文件夹。`rsync`命令有一些选项，即以下选项：

+   `-a`：存档

+   `-u`：更新

+   `-h`：人类可读格式

+   `-v`：详细

+   `--delete`：从接收方删除多余的文件

+   `--exclude`：排除规则

# 摘要

在本章中，我们学习了如何自动化常规管理任务。我们学习了通过各种技术接受输入，运行时提示密码，执行外部命令，读取配置文件，在脚本中添加警告，通过脚本和命令行启动`webbrowser`，使用`os`模块处理文件和目录以及进行备份。

在下一章中，您将学习更多关于`os`模块和数据处理的知识。此外，您还将学习`tarfile`模块以及如何使用它。

# 问题

1.  如何使用`readline`模块？

1.  用于读取、创建新文件、删除文件、列出当前目录中文件的 Linux 命令是什么？

1.  有哪些包可用于在 Python 中运行 Linux/Windows 命令？

1.  如何读取或在配置`ini`文件中设置新值

1.  列出用于查找`cpu`使用情况的库？

1.  列出接受用户输入的不同方法？

1.  排序和排序有什么区别？

# 进一步阅读

+   学习 Linux 的基本命令：[`maker.pro/linux/tutorial/basic-linux-commands-for-beginners`](https://maker.pro/linux/tutorial/basic-linux-commands-for-beginners)

+   Selenium webdriver 文档：[`selenium-python.readthedocs.io/index.html`](https://selenium-python.readthedocs.io/index.html)
