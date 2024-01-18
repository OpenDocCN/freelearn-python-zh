# 使用Apache和其他日志文件

在本章中，您将学习有关日志文件的知识。您将学习如何解析日志文件。您还将了解为什么需要在程序中编写异常。解析不同文件的不同方法也很重要。您还将了解`ErrorLog`和`AccessLog`。最后，您将学习如何解析其他日志文件。

在本章中，您将学习以下内容：

+   解析复杂的日志文件

+   异常的需要

+   解析不同文件的技巧

+   错误日志

+   访问日志

+   解析其他日志文件

# 解析复杂的日志文件

首先，我们将研究解析复杂日志文件的概念。解析日志文件是一项具有挑战性的任务，因为大多数日志文件都是以纯文本格式，而且该格式没有遵循任何规则。这些文件可能会在不显示任何警告的情况下进行修改。用户可以决定他们将在日志文件中存储什么类型的数据以及以何种格式存储，以及谁将开发应用程序。

在进行日志解析示例或更改日志文件中的配置之前，我们首先必须了解典型日志文件中包含什么。根据这一点，我们必须决定我们将学习如何操作或从中获取信息。我们还可以在日志文件中查找常见术语，以便我们可以使用这些常见术语来获取数据。

通常，您会发现日志文件中生成的大部分内容是由应用程序容器生成的，还有系统访问状态的条目（换句话说，注销和登录）或通过网络访问的系统的条目。因此，当您的系统通过网络远程访问时，这种远程连接的条目将保存在日志文件中。让我们以这种情况为例。我们已经有一个名为`access.log`的文件，其中包含一些日志信息。

因此，让我们创建一个名为`read_apache_log.py`的脚本，并在其中写入以下内容：

```py
def read_apache_log(logfile):
 with open(logfile) as f: log_obj = f.read() print(log_obj) if __name__ == '__main__':
 read_apache_log("access.log")
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~$ python3 read_apache_log.py Output: 64.242.88.10 - - [07/Mar/2004:16:05:49 -0800] "GET /twiki/bin/edit/Main/Double_bounce_sender?topicparent=Main.ConfigurationVariables HTTP/1.1" 401 12846 64.242.88.10 - - [07/Mar/2004:16:06:51 -0800] "GET /twiki/bin/rdiff/TWiki/NewUserTemplate?rev1=1.3&rev2=1.2 HTTP/1.1" 200 4523 64.242.88.10 - - [07/Mar/2004:16:10:02 -0800] "GET /mailman/listinfo/hsdivision HTTP/1.1" 200 6291 64.242.88.10 - - [07/Mar/2004:16:11:58 -0800] "GET /twiki/bin/view/TWiki/WikiSyntax HTTP/1.1" 200 7352 64.242.88.10 - - [07/Mar/2004:16:20:55 -0800] "GET /twiki/bin/view/Main/DCCAndPostFix HTTP/1.1" 200 5253 64.242.88.10 - - [07/Mar/2004:16:23:12 -0800] "GET /twiki/bin/oops/TWiki/AppendixFileSystem?template=oopsmore&param1=1.12&param2=1.12 HTTP/1.1" 200 11382 64.242.88.10 - - [07/Mar/2004:16:24:16 -0800] "GET /twiki/bin/view/Main/PeterThoeny HTTP/1.1" 200 4924 64.242.88.10 - - [07/Mar/2004:16:29:16 -0800] "GET /twiki/bin/edit/Main/Header_checks?topicparent=Main.ConfigurationVariables HTTP/1.1" 401 12851 64.242.88.10 - - [07/Mar/2004:16:30:29 -0800] "GET /twiki/bin/attach/Main/OfficeLocations HTTP/1.1" 401 12851 64.242.88.10 - - [07/Mar/2004:16:31:48 -0800] "GET /twiki/bin/view/TWiki/WebTopicEditTemplate HTTP/1.1" 200 3732 64.242.88.10 - - [07/Mar/2004:16:32:50 -0800] "GET /twiki/bin/view/Main/WebChanges HTTP/1.1" 200 40520 64.242.88.10 - - [07/Mar/2004:16:33:53 -0800] "GET /twiki/bin/edit/Main/Smtpd_etrn_restrictions?topicparent=Main.ConfigurationVariables HTTP/1.1" 401 12851 64.242.88.10 - - [07/Mar/2004:16:35:19 -0800] "GET /mailman/listinfo/business HTTP/1.1" 200 6379 …..
```

在前面的示例中，我们创建了一个`read_apache_log`函数来读取Apache日志文件。在其中，我们打开了一个日志文件，然后打印了其中的日志条目。在定义了`read_apache_log()`函数之后，我们在主函数中调用了它，并传入了Apache日志文件的名称。在我们的案例中，Apache日志文件的名称是`access.log`。

在`access.log`文件中读取日志条目后，现在我们将从日志文件中解析IP地址。为此，请创建一个名为`parse_ip_address.py`的脚本，并在其中写入以下内容：

```py
import re from collections import Counter r_e = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}' with open("access.log") as f:
 print("Reading Apache log file") Apache_log = f.read() get_ip = re.findall(r_e,Apache_log) no_of_ip = Counter(get_ip) for k, v in no_of_ip.items(): print("Available IP Address in log file " + "=> " + str(k) + " " + "Count "  + "=> " + str(v))
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work/Chapter_15$ python3 parse_ip_address.py Output: Reading Apache log file Available IP Address in log file => 64.242.88.1 Count => 452 Available IP Address in log file => 213.181.81.4 Count => 1 Available IP Address in log file => 213.54.168.1 Count => 12 Available IP Address in log file => 200.160.249.6 Count => 2 Available IP Address in log file => 128.227.88.7 Count => 14 Available IP Address in log file => 61.9.4.6 Count => 3 Available IP Address in log file => 212.92.37.6 Count => 14 Available IP Address in log file => 219.95.17.5 Count => 1 3Available IP Address in log file => 10.0.0.1 Count => 270 Available IP Address in log file => 66.213.206.2 Count => 1 Available IP Address in log file => 64.246.94.1 Count => 2 Available IP Address in log file => 195.246.13.1 Count => 12 Available IP Address in log file => 195.230.181.1 Count => 1 Available IP Address in log file => 207.195.59.1 Count => 20 Available IP Address in log file => 80.58.35.1 Count => 1 Available IP Address in log file => 200.222.33.3 Count => 1 Available IP Address in log file => 203.147.138.2 Count => 13 Available IP Address in log file => 212.21.228.2 Count => 1 Available IP Address in log file => 80.58.14.2 Count => 4 Available IP Address in log file => 142.27.64.3 Count => 7 ……
```

在前面的示例中，我们创建了Apache日志解析器来确定服务器上一些特定IP地址及其请求次数。因此，很明显我们不想要Apache日志文件中的整个日志条目，我们只想从日志文件中获取IP地址。为此，我们必须定义一个模式来搜索IP地址，我们可以通过使用正则表达式来实现。因此，我们导入了`re`模块。然后我们导入了`Collection`模块作为Python内置数据类型`dict`、`list`、`set`和`tuple`的替代品。该模块具有专门的容器数据类型。在导入所需的模块后，我们使用正则表达式编写了一个模式，以匹配从日志文件中映射IP地址的特定条件。

在匹配模式中，`\d`可以是`0`到`9`之间的任何数字，`\r`代表原始字符串。然后，我们打开名为`access.log`的Apache日志文件并读取它。之后，我们在Apache日志文件上应用了正则表达式条件，然后使用`collection`模块的`counter`函数来获取我们根据`re`条件获取的每个IP地址的计数。最后，我们打印了操作的结果，如输出中所示。

# 异常的需要

在这一部分，我们将看看Python编程中异常的需要。正常的程序流程包括事件和信号。异常一词表明您的程序出了问题。这些异常可以是任何类型，比如零除错误、导入错误、属性错误或断言错误。这些异常会在指定的函数无法正常执行其任务时发生。一旦异常发生，程序执行就会停止，解释器将继续进行异常处理过程。异常处理过程包括在`try…except`块中编写代码。异常处理的原因是您的程序发生了意外情况。

# 分析异常

在这一部分，我们将了解分析异常。每个发生的异常都必须被处理。您的日志文件也应该包含一些异常。如果您多次遇到类似类型的异常，那么您的程序存在一些问题，您应该尽快进行必要的更改。

考虑以下例子：

```py
f = open('logfile', 'r') print(f.read()) f.close()
```

运行程序后，您将得到以下输出：

```py
Traceback (most recent call last):
 File "sample.py", line 1, in <module> f = open('logfile', 'r') FileNotFoundError: [Errno 2] No such file or directory: 'logfile'
```

在这个例子中，我们试图读取一个在我们目录中不存在的文件，结果显示了一个错误。因此，通过错误我们可以分析我们需要提供什么样的解决方案。为了处理这种情况，我们可以使用异常处理技术。所以，让我们看一个使用异常处理技术处理错误的例子。

考虑以下例子：

```py
try:
    f = open('logfile', 'r')
 print(f.read()) f.close()
except:
    print("file not found. Please check whether the file is present in your directory or not.") 
```

运行程序后，您将得到以下输出：

```py
file not found. Please check whether the file is present in your directory or not.
```

在这个例子中，我们试图读取一个在我们目录中不存在的文件。但是，在这个例子中，我们使用了文件异常技术，将我们的代码放在`try:`和`except:`块中。因此，如果在`try:`块中发生任何错误或异常，它将跳过该错误并执行`except:`块中的代码。在我们的情况下，我们只在`except:`块中放置了一个`print`语句。因此，在运行脚本后，当异常发生在`try:`块中时，它会跳过该异常并执行`except:`块中的代码。因此，在`except`块中的`print`语句会被执行，正如我们在之前的输出中所看到的。

# 解析不同文件的技巧

在这一部分，我们将学习解析不同文件时使用的技巧。在开始实际解析之前，我们必须先读取数据。您需要了解您将从哪里获取所有数据。但是，您也必须记住所有的日志文件大小都不同。为了简化您的任务，这里有一个要遵循的清单：

+   请记住，日志文件可以是纯文本或压缩文件。

+   所有日志文件都有一个`.log`扩展名的纯文本文件和一个`log.bz2`扩展名的`bzip2`文件。

+   您应该根据文件名处理文件集。

+   所有日志文件的解析必须合并成一个报告。

+   您使用的工具必须能够处理所有文件，无论是来自指定目录还是来自不同目录。所有子目录中的日志文件也应包括在内。

# 错误日志

在这一部分，我们将学习错误日志。错误日志的相关指令如下：

+   `ErrorLog`

+   `LogLevel`

服务器日志文件的位置和名称由`ErrorLog`指令设置。这是最重要的日志文件。Apache `httpd`发送的信息和处理过程中产生的记录都在其中。每当服务器出现问题时，这将是第一个需要查看的地方。它包含了出现问题的细节以及修复问题的过程。

错误日志被写入文件中。在Unix系统上，服务器可以将错误发送到`syslog`，或者您可以将它们传送到您的程序中。日志条目中的第一件事是消息的日期和时间。第二个条目记录了错误的严重程度。

`LogLevel`指令通过限制严重级别处理发送到错误日志的错误。第三个条目包含生成错误的客户端的信息。该信息将是IP地址。接下来是消息本身。它包含了服务器已配置为拒绝客户端访问的信息。服务器将报告所请求文档的文件系统路径。

错误日志文件中可能出现各种类型的消息。错误日志文件还包含来自CGI脚本的调试输出。无论信息写入`stderr`，都将直接复制到错误日志中。

错误日志文件是不可定制的。处理请求的错误日志中的条目将在访问日志中有相应的条目。您应该始终监视错误日志以解决测试期间的问题。在Unix系统上，您可以运行以下命令来完成这个任务：

```py
$ tail -f error_log
```

# 访问日志

在本节中，您将学习访问日志。服务器访问日志将记录服务器处理的所有请求。`CustomLog`指令控制访问日志的位置和内容。`LogFormat`指令用于选择日志的内容。

将信息存储在访问日志中意味着开始日志管理。下一步将是分析帮助我们获得有用统计信息的信息。Apache `httpd`有各种版本，这些版本使用了一些其他模块和指令来控制访问日志记录。您可以配置访问日志的格式。这个格式是使用格式字符串指定的。

# 通用日志格式

在本节中，我们将学习通用日志格式。以下语法显示了访问日志的配置：

```py
 LogFormat "%h %l %u %t \"%r\" %>s %b" nick_name
 CustomLog logs/access_log nick_name
```

这个字符串将定义一个昵称，然后将该昵称与日志格式字符串关联起来。日志格式字符串由百分比指令组成。每个百分比指令告诉服务器记录特定的信息。这个字符串可能包含文字字符。这些字符将直接复制到日志输出中。

`CustomLog`指令将使用定义的*昵称*设置一个新的日志文件。访问日志的文件名相对于`ServerRoot`，除非以斜杠开头。

我们之前提到的配置将以**通用日志格式**（**CLF**）写入日志条目。这是一种标准格式，可以由许多不同的Web服务器生成。许多日志分析程序读取这种日志格式。

现在，我们将看到每个百分比指令的含义：

+   `%h`：这显示了向Web服务器发出请求的客户端的IP地址。如果`HostnameLookups`打开，那么服务器将确定主机名并将其记录在IP地址的位置。

+   `%l`：这个术语用于指示所请求的信息不可用。

+   `%u`：这是请求文档的用户ID。相同的值将在`REMOTE_USER`环境变量中提供给CGI脚本。

+   `%t`：这个术语用于检测服务器处理请求完成的时间。格式如下：

```py
            [day/month/year:hour:minute:second zone]
```

对于`day`参数，需要两位数字。对于`month`，我们必须定义三个字母。对于年份，由于年份有四个字符，我们必须取四位数字。现在，在`day`、`month`和`year`之后，我们必须为`hour`、`minute`和`seconds`各取两位数字。

+   `\"%r\"`：这个术语用作请求行，客户端用双引号给出。这个请求行包含有用的信息。请求客户端使用`GET`方法，使用的协议是HTTP。

+   `%>s`：这个术语定义了客户端的状态代码。状态代码非常重要和有用，因为它指示了客户端发送的请求是否成功地发送到服务器。

+   `%b`：这个术语定义了对象返回给客户端时的总大小。这个总大小不包括响应头的大小。

# 解析其他日志文件

我们的系统中还有其他不同的日志文件，包括Apache日志。在我们的Linux发行版中，日志文件位于根文件系统中的`/var/log/`文件夹中，如下所示：

![](assets/42d14bf4-400a-417b-950f-1abb543b30f8.png)

在上面的屏幕截图中，我们可以很容易地看到不同类型的日志文件（例如，认证日志文件`auth.log`，系统日志文件`syslog`和内核日志`kern.log`）可用于不同的操作条目。当我们对Apache日志文件执行操作时，如前所示，我们也可以对本地日志文件执行相同类型的操作。让我们看一个以前解析日志文件的例子。在`simple_log.py`脚本中创建并写入以下内容：

```py
f=open('/var/log/kern.log','r') lines = f.readlines() for line in lines:
 kern_log = line.split() print(kern_log) f.close()
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~$ python3 simple_log.py Output:
 ['Dec', '26', '14:39:38', 'ubuntu', 'NetworkManager[795]:', '<info>', '[1545815378.2891]', 'device', '(ens33):', 'state', 'change:', 'prepare', '->', 'config', '(reason', "'none')", '[40', '50', '0]'] ['Dec', '26', '14:39:38', 'ubuntu', 'NetworkManager[795]:', '<info>', '[1545815378.2953]', 'device', '(ens33):', 'state', 'change:', 'config', '->', 'ip-config', '(reason', "'none')", '[50', '70', '0]'] ['Dec', '26', '14:39:38', 'ubuntu', 'NetworkManager[795]:', '<info>', '[1545815378.2997]', 'dhcp4', '(ens33):', 'activation:', 'beginning', 'transaction', '(timeout', 'in', '45', 'seconds)'] ['Dec', '26', '14:39:38', 'ubuntu', 'NetworkManager[795]:', '<info>', '[1545815378.3369]', 'dhcp4', '(ens33):', 'dhclient', 'started', 'with', 'pid', '5221'] ['Dec', '26', '14:39:39', 'ubuntu', 'NetworkManager[795]:', '<info>', '[1545815379.0008]', 'address', '192.168.0.108'] ['Dec', '26', '14:39:39', 'ubuntu', 'NetworkManager[795]:', '<info>', '[1545815379.0020]', 'plen', '24', '(255.255.255.0)'] ['Dec', '26', '14:39:39', 'ubuntu', 'NetworkManager[795]:', '<info>', '[1545815379.0028]', 'gateway', '192.168.0.1']
```

在上面的例子中，首先我们创建了一个简单的文件对象`f`，并以读模式在其中打开了`kern.log`文件。之后，我们在`file`对象上应用了`readlines()`函数，以便在`for`循环中逐行读取文件中的数据。然后我们对内核日志文件的每一行应用了**`split()`**函数，然后使用`print`函数打印整个文件，如输出所示。

像读取内核日志文件一样，我们也可以对其执行各种操作，就像我们现在要执行一些操作一样。现在，我们将通过索引访问内核日志文件中的内容。这是可能的，因为`split`函数将文件中的所有信息拆分为不同的迭代。因此，让我们看一个这样的条件的例子。创建一个`simple_log1.py`脚本，并将以下脚本放入其中：

```py
f=open('/var/log/kern.log','r') lines = f.readlines() for line in lines:
 kern_log = line.split()[1:3] print(kern_log)
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~$ python3 simple_log1.py Output: ['26', '14:37:20'] ['26', '14:37:20'] ['26', '14:37:32'] ['26', '14:39:38'] ['26', '14:39:38'] ['26', '14:39:38'] ['26', '14:39:38'] ['26', '14:39:38'] ['26', '14:39:38'] ['26', '14:39:38'] ['26', '14:39:38'] ['26', '14:39:38'] 
```

在上面的例子中，我们只是在`split`函数旁边添加了`[1:3]`，换句话说，切片。序列的子序列称为切片，提取子序列的操作称为切片。在我们的例子中，我们使用方括号（`[ ]`）作为切片运算符，并在其中有两个整数值，用冒号（`:`）分隔。操作符`[1:3]`返回序列的第一个元素到第三个元素的部分，包括第一个但不包括最后一个。当我们对任何序列进行切片时，我们得到的子序列始终与从中派生的原始序列具有相同的类型。

然而，列表（或元组）的元素可以是任何类型；无论我们如何对其进行切片，列表的派生切片都是列表。因此，在对日志文件进行切片后，我们得到了先前显示的输出。

# 摘要

在本章中，您学习了如何处理不同类型的日志文件。您还了解了解析复杂日志文件以及在处理这些文件时异常处理的必要性。解析日志文件的技巧将有助于顺利进行解析。您还了解了`ErrorLog`和`AccessLog`。

在下一章中，您将学习有关SOAP和REST通信的内容。

# 问题

1.  Python中运行时异常和编译时异常有什么区别？

1.  什么是正则表达式？

1.  探索Linux命令`head`，`tail`，`cat`和`awk`。

1.  编写一个Python程序，将一个文件的内容追加到另一个文件中。

1.  编写一个Python程序，以相反的顺序读取文件的内容。

1.  以下表达式的输出将是什么？

1.  `re.search(r'C\Wke', 'C@ke').group()`

1.  `re.search(r'Co+kie', 'Cooookie').group()`

1.  `re.match(r'<.*?>', '<h1>TITLE</h1>').group()`

# 进一步阅读

+   Python日志记录：[https://docs.python.org/3/library/logging.html](https://docs.python.org/3/library/logging.html)

+   正则表达式：[https://docs.python.org/3/howto/regex.html](https://docs.python.org/3/howto/regex.html)

+   异常处理：[https://www.pythonforbeginners.com/error-handling/python-try-and-except](https://www.pythonforbeginners.com/error-handling/python-try-and-except)
