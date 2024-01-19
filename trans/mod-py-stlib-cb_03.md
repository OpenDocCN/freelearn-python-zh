# 命令行

在本章中，我们将涵盖以下配方：

+   基本日志记录-日志记录允许您跟踪软件正在做什么，通常与其输出无关

+   记录到文件-当记录频繁时，有必要将日志存储在磁盘上

+   记录到 Syslog-如果您的系统有 Syslog 守护程序，则可能希望登录到 Syslog 而不是使用独立文件

+   解析参数-在使用命令行工具编写时，您需要为几乎任何工具解析选项

+   交互式 shell-有时选项不足，您需要一种交互式的 REPL 来驱动您的工具

+   调整终端文本大小-为了正确对齐显示的输出，我们需要知道终端窗口的大小

+   运行系统命令-如何将其他第三方命令集成到您的软件中

+   进度条-如何在文本工具中显示进度条

+   消息框-如何在文本工具中显示 OK/取消消息框

+   输入框-如何在文本工具中请求输入

# 介绍

编写新工具时，首先出现的需求之一是使其能够与周围环境进行交互-显示结果，跟踪错误并接收输入。

用户习惯于命令行工具与他们和系统交互的某些标准方式，如果从头开始遵循这个标准可能是耗时且困难的。

这就是为什么 Python 标准库提供了工具来实现能够通过 shell 和文本进行交互的软件的最常见需求。

在本章中，我们将看到如何实现某些形式的日志记录，以便我们的程序可以保留日志文件；我们将看到如何实现基于选项和交互式软件，然后我们将看到如何基于文本实现更高级的图形输出。

# 基本日志记录

控制台软件的首要要求之一是记录其所做的事情，即发生了什么以及任何警告或错误。特别是当我们谈论长期运行的软件或在后台运行的守护程序时。

遗憾的是，如果您曾经尝试使用 Python 的`logging`模块，您可能已经注意到除了错误之外，您无法获得任何输出。

这是因为默认启用级别是“警告”，因此只有警告和更严重的情况才会被跟踪。需要进行一些小的调整，使日志通常可用。

# 如何做...

对于这个配方，步骤如下：

1.  `logging`模块允许我们通过`basicConfig`方法轻松设置日志记录配置：

```py
>>> import logging, sys
>>> 
>>> logging.basicConfig(level=logging.INFO, stream=sys.stderr,
...                     format='%(asctime)s %(name)s %(levelname)s: %(message)s')
>>> log = logging.getLogger(__name__)
```

1.  现在我们的`logger`已经正确配置，我们可以尝试使用它：

```py
>>> def dosum(a, b, count=1):
...     log.info('Starting sum')
...     if a == b == 0:
...         log.warning('Will be just 0 for any count')
...     res = (a + b) * count
...     log.info('(%s + %s) * %s = %s' % (a, b, count, res))
...     print(res)
... 
>>> dosum(5, 3)
2018-02-11 22:07:59,870 __main__ INFO: Starting sum
2018-02-11 22:07:59,870 __main__ INFO: (5 + 3) * 1 = 8
8
>>> dosum(5, 3, count=2)
2018-02-11 22:07:59,870 __main__ INFO: Starting sum
2018-02-11 22:07:59,870 __main__ INFO: (5 + 3) * 2 = 16
16
>>> dosum(0, 1, count=5)
2018-02-11 22:07:59,870 __main__ INFO: Starting sum
2018-02-11 22:07:59,870 __main__ INFO: (0 + 1) * 5 = 5
5
>>> dosum(0, 0)
2018-02-11 22:08:00,621 __main__ INFO: Starting sum
2018-02-11 22:08:00,621 __main__ WARNING: Will be just 0 for any count
2018-02-11 22:08:00,621 __main__ INFO: (0 + 0) * 1 = 0
0
```

# 它是如何工作的...

`logging.basicConfig`配置`root`记录器（主记录器，如果找不到用于使用的记录器的特定配置，则 Python 将使用它）以在`INFO`级别或更高级别写入任何内容。这将允许我们显示除调试消息之外的所有内容。`format`参数指定了我们的日志消息应该如何格式化；在这种情况下，我们添加了日期和时间，记录器的名称，我们正在记录的级别以及消息本身。最后，`stream`参数告诉记录器将其输出写入标准错误。

一旦我们配置了`root`记录器，任何我们选择的日志记录，如果没有特定的配置，都将使用`root`记录器。

因此，下一行`logging.getLogger(__name__)`会获得一个与执行的 Python 模块类似命名的记录器。如果您将代码保存到文件中，则记录器的名称将类似于`dosum`（假设您的文件名为`dosum.py`）；如果没有，则记录器的名称将为`__main__`，就像前面的示例中一样。

Python 记录器在使用`logging.getLogger`检索时首次创建，并且对`getLogger`的任何后续调用只会返回已经存在的记录器。对于非常简单的程序，名称可能并不重要，但在更大的软件中，通常最好抓取多个记录器，这样您可以区分消息来自软件的哪个子系统。

# 还有更多...

也许你会想知道为什么我们配置`logging`将其输出发送到`stderr`，而不是标准输出。这样可以将我们软件的输出（通过打印语句写入`stdout`）与日志信息分开。这通常是一个好的做法，因为您的工具的用户可能需要调用您的工具的输出，而不带有日志消息生成的所有噪音，这样做可以让我们以以下方式调用我们的脚本：

```py
$ python dosum.py 2>/dev/null
8
16
5
0
```

我们只会得到结果，而不会有所有的噪音，因为我们将`stderr`重定向到`/dev/null`，这在 Unix 系统上会导致丢弃所有写入`stderr`的内容。

# 记录到文件

对于长时间运行的程序，将日志记录到屏幕并不是一个非常可行的选择。在运行代码数小时后，最旧的日志消息将丢失，即使它们仍然可用，也不容易阅读所有日志或搜索其中的内容。

将日志保存到文件允许无限长度（只要我们的磁盘允许）并且可以使用`grep`等工具进行搜索。

默认情况下，Python 日志配置为写入屏幕，但在配置日志时很容易提供一种方式来写入任何文件。

# 如何做到...

为了测试将日志记录到文件，我们将创建一个简短的工具，根据当前时间计算最多*n*个斐波那契数。如果是下午 3:01，我们只想计算 1 个数字，而如果是下午 3:59，我们想计算 59 个数字。

软件将提供计算出的数字作为输出，但我们还想记录计算到哪个数字以及何时运行：

```py
import logging, sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide logging file name as argument')
        sys.exit(1)

    logging_file = sys.argv[1]
    logging.basicConfig(level=logging.INFO, filename=logging_file,
                        format='%(asctime)s %(name)s %(levelname)s: %(message)s')

log = logging.getLogger(__name__)

def fibo(num):
    log.info('Computing up to %sth fibonacci number', num)
    a, b = 0, 1
    for n in range(num):
        a, b = b, a+b
        print(b, '', end='')
    print(b)

if __name__ == '__main__':
    import datetime
    fibo(datetime.datetime.now().second)
```

# 工作原理...

代码分为三个部分：初始化日志记录、`fibo`函数和我们工具的`main`函数。我们明确地以这种方式划分代码，因为`fibo`函数可能会在其他模块中使用，在这种情况下，我们不希望重新配置`logging`；我们只想使用程序提供的日志配置。因此，`logging.basicConfig`调用被包装在`__name__ == '__main__'`中，以便只有在模块被直接调用为工具时才配置`logging`，而不是在被其他模块导入时。

当调用多个`logging.basicConfig`实例时，只有第一个会被考虑。如果我们在其他模块中导入时没有将日志配置包装在`if`中，它可能最终会驱动整个软件的日志配置，这取决于模块导入的顺序，这显然是我们不想要的。

与之前的方法不同，`basicConfig`是使用`filename`参数而不是`stream`参数进行配置的。这意味着将创建`logging.FileHandler`来处理日志消息，并且消息将被追加到该文件中。

代码的核心部分是`fibo`函数本身，最后一部分是检查代码是作为 Python 脚本调用还是作为模块导入。当作为模块导入时，我们只想提供`fibo`函数并避免运行它，但当作为脚本执行时，我们想计算斐波那契数。

也许你会想知道为什么我使用了两个`if __name__ == '__main__'`部分；如果将两者合并成一个，脚本将继续工作。但通常最好确保在尝试使用日志之前配置`logging`，否则结果将是我们最终会使用`logging.lastResort`处理程序，它只会写入`stderr`直到日志被配置。

# 记录到 Syslog

类 Unix 系统通常提供一种通过`syslog`协议收集日志消息的方法，这使我们能够将存储日志的系统与生成日志的系统分开。

特别是在跨多个服务器分布的应用程序的情况下，这非常方便；您肯定不想登录到 20 个不同的服务器上收集您的 Python 应用程序的所有日志，因为它在多个节点上运行。特别是对于 Web 应用程序来说，这在云服务提供商中现在非常常见，因此能够在一个地方收集所有 Python 日志非常方便。

这正是使用`syslog`允许我们做的事情；我们将看到如何将日志消息发送到运行在我们系统上的守护程序，但也可以将它们发送到任何系统。

# 准备工作

虽然这个方法不需要`syslog`守护程序才能工作，但您需要一个来检查它是否正常工作，否则消息将无法被读取。在 Linux 或 macOS 系统的情况下，这通常是开箱即用的，但在 Windows 系统的情况下，您需要安装一个 Syslog 服务器或使用云解决方案。有许多选择，只需在 Google 上快速搜索，就可以找到一些便宜甚至免费的替代方案。

# 如何做...

当使用一个定制程度很高的日志记录解决方案时，就不再能依赖于`logging.basicConfig`，因此我们将不得不手动设置日志记录环境：

```py
import logging
import logging.config

# OSX logs through /var/run/syslog this should be /dev/log 
# on Linux system or a tuple ('ADDRESS', PORT) to log to a remote server
SYSLOG_ADDRESS = '/var/run/syslog'

logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s %(name)s: %(levelname)s %(message)s'
        },
    },
    'handlers': {
        'syslog': {
            'class': 'logging.handlers.SysLogHandler',
            'formatter': 'default',
            'address': SYSLOG_ADDRESS
        }
    },
    'root': {
        'handlers': ['syslog'],
        'level': 'INFO'
    }
})

log = logging.getLogger()
log.info('Hello Syslog!')
```

如果这样操作正常，您的消息应该被 Syslog 记录，并且在 macOS 上运行`syslog`命令或在 Linux 上作为`/var/log/syslog`的`tail`命令时可见：

```py
$ syslog | tail -n 2
Feb 18 17:52:43 Pulsar Google Chrome[294] <Error>: ... SOME CHROME ERROR MESSAGE ...
Feb 18 17:53:48 Pulsar 2018-02-18 17[4294967295] <Info>: 53:48,610 INFO root Hello Syslog!
```

`syslog`文件路径可能因发行版而异；如果`/var/log/syslog`不起作用，请尝试`/var/log/messages`或参考您的发行版文档。

# 还有更多...

由于我们依赖于`dictConfig`，您会注意到我们的配置比以前的方法更复杂。这是因为我们自己配置了日志基础设施的部分。

每当您配置日志记录时，都要使用记录器写入您的消息。默认情况下，系统只有一个记录器：`root`记录器（如果您调用`logging.getLogger`而不提供任何特定名称，则会得到该记录器）。

记录器本身不处理消息，因为写入或打印日志消息是处理程序的职责。因此，如果您想要读取您发送的日志消息，您需要配置一个处理程序。在我们的情况下，我们使用`SysLogHandler`，它写入到 Syslog。

处理程序负责写入消息，但实际上并不涉及消息应该如何构建/格式化。您会注意到，除了您自己的消息之外，当您记录某些内容时，还会得到日志级别、记录器名称、时间戳以及由日志系统为您添加的一些细节。将这些细节添加到消息中通常是格式化程序的工作。格式化程序获取记录器提供的所有信息，并将它们打包成应该由处理程序写入的消息。

最后但并非最不重要的是，您的日志配置可能非常复杂。您可以设置一些消息发送到本地文件，一些消息发送到 Syslog，还有一些应该打印在屏幕上。这将涉及多个处理程序，它们应该知道哪些消息应该处理，哪些消息应该忽略。允许这种知识是过滤器的工作。一旦将过滤器附加到处理程序，就可以控制哪些消息应该由该处理程序保存，哪些应该被忽略。

Python 日志系统现在可能看起来非常直观，这是因为它是一个非常强大的解决方案，可以以多种方式进行配置，但一旦您了解了可用的构建模块，就可以以非常灵活的方式将它们组合起来。

# 解析参数

当编写命令行工具时，通常会根据提供给可执行文件的选项来改变其行为。这些选项通常与可执行文件名称一起在`sys.argv`中可用，但解析它们并不像看起来那么容易，特别是当必须支持多个参数时。此外，当选项格式不正确时，通常最好提供一个使用消息，以便通知用户正确使用工具的方法。

# 如何做...

执行此食谱的以下步骤：

1.  `argparse.ArgumentParser`对象是负责解析命令行选项的主要对象：

```py
import argparse
import operator
import logging
import functools

parser = argparse.ArgumentParser(
    description='Applies an operation to one or more numbers'
)
parser.add_argument("number", 
                    help="One or more numbers to perform an operation on.",
                    nargs='+', type=int)
parser.add_argument('-o', '--operation', 
                    help="The operation to perform on numbers.",
                    choices=['add', 'sub', 'mul', 'div'], default='add')
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")

opts = parser.parse_args()

logging.basicConfig(level=logging.INFO if opts.verbose else logging.WARNING)
log = logging.getLogger()

operation = getattr(operator, opts.operation)
log.info('Applying %s to %s', opts.operation, opts.number)
print(functools.reduce(operation, opts.number))
```

1.  一旦我们的命令没有任何参数被调用，它将提供一个简短的使用文本：

```py
$ python /tmp/doop.py
usage: doop.py [-h] [-o {add,sub,mul,div}] [-v] number [number ...]
doop.py: error: the following arguments are required: number
```

1.  如果我们提供了`-h`选项，`argparse`将为我们生成一个完整的使用指南：

```py
$ python /tmp/doop.py -h
usage: doop.py [-h] [-o {add,sub,mul,div}] [-v] number [number ...]

Applies an operation to one or more numbers

positional arguments:
number                One or more numbers to perform an operation on.

optional arguments:
-h, --help            show this help message and exit
-o {add,sub,mul,div}, --operation {add,sub,mul,div}
                        The operation to perform on numbers.
-v, --verbose         increase output verbosity
```

1.  使用该命令将会得到预期的结果：

```py
$ python /tmp/dosum.py 1 2 3 4 -o mul
24
```

# 工作原理...

我们使用了`ArgumentParser.add_argument`方法来填充可用选项的列表。对于每个参数，还可以提供一个`help`选项，它将为该参数声明`help`字符串。

位置参数只需提供参数的名称：

```py
parser.add_argument("number", 
                    help="One or more numbers to perform an operation on.",
                    nargs='+', type=int)
```

`nargs`选项告诉`ArgumentParser`我们期望该参数被指定的次数，`+`值表示至少一次或多次。然后`type=int`告诉我们参数应该被转换为整数。

一旦我们有了要应用操作的数字，我们需要知道操作本身：

```py
parser.add_argument('-o', '--operation', 
                    help="The operation to perform on numbers.",
                    choices=['add', 'sub', 'mul', 'div'], default='add')
```

在这种情况下，我们指定了一个选项（以破折号`-`开头），可以提供`-o`或`--operation`。我们声明唯一可能的值是`'add'`、`'sub'`、`'mul'`或`'div'`（提供不同的值将导致`argparse`抱怨），如果用户没有指定默认值，则为`add`。

作为最佳实践，我们的命令只打印结果；能够询问一些关于它将要做什么的日志是很方便的。因此，我们提供了`verbose`选项，它驱动了我们为命令启用的日志级别：

```py
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
```

如果提供了该选项，我们将只存储`verbose`模式已启用（`action="store_true"`使得`True`被存储在`opts.verbose`中），并且我们将相应地配置`logging`模块，这样我们的`log.info`只有在`verbose`被启用时才可见。

最后，我们可以实际解析命令行选项并将结果返回到`opts`对象中：

```py
opts = parser.parse_args()
```

一旦我们有了可用的选项，我们配置日志，以便我们可以读取`verbose`选项并相应地配置它：

```py
logging.basicConfig(level=logging.INFO if opts.verbose else logging.WARNING)
```

一旦选项被解析并且`logging`被配置，剩下的就是在提供的数字集上执行预期的操作并打印结果：

```py
operation = getattr(operator, opts.operation)
log.info('Applying %s to %s', opts.operation, opts.number)
print(functools.reduce(operation, opts.number))
```

# 还有更多...

如果你将命令行选项与第一章*容器和数据结构*中的*带回退的字典*食谱相结合，你可以扩展工具的行为，不仅可以从命令行读取选项，还可以从环境变量中读取，当你无法完全控制命令的调用方式但可以设置环境变量时，这通常非常方便。

# 交互式 shell

有时，编写命令行工具是不够的，你需要能够提供某种交互。假设你想要编写一个邮件客户端。在这种情况下，必须要调用`mymail list`来查看你的邮件，或者从你的 shell 中读取特定的邮件，等等，这是不太方便的。此外，如果你想要实现有状态的行为，比如一个`mymail reply`实例，它应该回复你正在查看的当前邮件，这甚至可能是不可能的。

在这些情况下，交互式程序更好，Python 标准库通过`cmd`模块提供了编写这样一个程序所需的所有工具。

我们可以尝试为我们的`mymail`程序编写一个交互式 shell；它不会读取真实的电子邮件，但我们将伪造足够的行为来展示一个功能齐全的 shell。

# 如何做...

此示例的步骤如下：

1.  `cmd.Cmd`类允许我们启动交互式 shell 并基于它们实现命令：

```py
EMAILS = [
    {'sender': 'author1@domain.com', 'subject': 'First email', 
     'body': 'This is my first email'},
    {'sender': 'author2@domain.com', 'subject': 'Second email', 
     'body': 'This is my second email'},
]

import cmd
import shlex

class MyMail(cmd.Cmd):
    intro = 'Simple interactive email client.'
    prompt = 'mymail> '

    def __init__(self, *args, **kwargs):
        super(MyMail, self).__init__(*args, **kwargs)
        self.selected_email = None

    def do_list(self, line):
        """list

        List emails currently in the Inbox"""
        for idx, email in enumerate(EMAILS):
            print('[{idx}] From: {e[sender]} - 
                    {e[subject]}'.format(
                    idx=idx, e=email
            ))

    def do_read(self, emailnum):
        """read [emailnum]

        Reads emailnum nth email from those listed in the Inbox"""
        try:
            idx = int(emailnum.strip())
        except:
            print('Invalid email index {}'.format(emailnum))
            return

        try:
            email = EMAILS[idx]
        except IndexError:
            print('Email {} not found'.format(idx))
            return

        print('From: {e[sender]}\n'
              'Subject: {e[subject]}\n'
              '\n{e[body]}'.format(e=email))
        # Track the last read email as the selected one for reply.
        self.selected_email = idx

    def do_reply(self, message):
        """reply [message]

        Sends back an email to the author of the received email"""
        if self.selected_email is None:
            print('No email selected for reply.')
            return

        email = EMAILS[self.selected_email]
        print('Replied to {e[sender]} with: {message}'.format(
            e=email, message=message
        ))

    def do_send(self, arguments):
        """send [recipient] [subject] [message]

        Send a new email with [subject] to [recipient]"""
        # Split the arguments with shlex 
        # so that we allow subject or message with spaces. 
        args = shlex.split(arguments)
        if len(args) < 3:
            print('A recipient, a subject and a message are 
                  required.')
            return

        recipient, subject, message = args[:3]
        if len(args) >= 4:
            message += ' '.join(args[3:])

        print('Sending email {} to {}: "{}"'.format(
            subject, recipient, message
        ))

    def complete_send(self, text, line, begidx, endidx):
        # Provide autocompletion of recipients for send command.
        return [e['sender'] for e in EMAILS if e['sender'].startswith(text)]

    def do_EOF(self, line):
        return True

if __name__ == '__main__':
    MyMail().cmdloop()
```

1.  启动我们的脚本应该提供一个很好的交互提示：

```py
$ python /tmp/mymail.py 
Simple interactive email client.
mymail> help

Documented commands (type help <topic>):
========================================
help  list  read  reply  send

Undocumented commands:
======================
EOF
```

1.  如文档所述，我们应该能够读取邮件列表，阅读特定的邮件，并回复当前打开的邮件：

```py
mymail> list
[0] From: author1@domain.com - First email
[1] From: author2@domain.com - Second email
mymail> read 0
From: author1@domain.com
Subject: First email

This is my first email
mymail> reply Thanks for your message!
Replied to author1@domain.com with: Thanks for your message!
```

1.  然后，我们可以依赖更高级的发送命令，这些命令还为我们的新邮件提供了收件人的自动完成：

```py
mymail> help send
send [recipient] [subject] [message]

Send a new email with [subject] to [recipient]
mymail> send author
author1@domain.com  author2@domain.com  
mymail> send author2@domain.com "Saw your email" "I saw your message, thanks for sending it!"
Sending email Saw your email to author2@domain.com: "I saw your message, thanks for sending it!"
mymail> 
```

# 工作原理...

`cmd.Cmd`循环通过`prompt`类属性打印我们提供的`prompt`并等待命令。在`prompt`之后写的任何东西都会被分割，然后第一部分会被查找我们自己的子类提供的方法列表。

每当提供一个命令时，`cmd.Cmd.cmdloop`调用相关的方法，然后重新开始。

任何以`do_*`开头的方法都是一个命令，`do_`之后的部分是命令名称。如果在交互提示中使用`help`命令，则实现命令的方法的 docstring 将被报告在我们工具的文档中。

`Cmd`类不提供解析命令参数的功能，因此，如果您的命令有多个参数，您必须自己拆分它们。在我们的情况下，我们依赖于`shlex`，以便用户可以控制参数的拆分方式。这使我们能够解析主题和消息，同时提供了一种包含空格的方法。否则，我们将无法知道主题在哪里结束，消息从哪里开始。

`send`命令还支持自动完成收件人，通过`complete_send`方法。如果提供了`complete_*`方法，当按下*Tab*自动完成命令参数时，`Cmd`会调用它。该方法接收需要完成的文本以及有关整行文本和光标当前位置的一些详细信息。由于没有对参数进行解析，光标的位置和整行文本可以帮助提供不同的自动完成行为。在我们的情况下，我们只能自动完成收件人，因此无需区分各个参数。

最后但并非最不重要的是，`do_EOF`命令允许在按下*Ctrl* + *D*时退出命令行。否则，我们将无法退出交互式 shell。这是`Cmd`提供的一个约定，如果`do_EOF`命令返回`True`，则表示 shell 可以退出。

# 调整终端文本大小

我们在第二章的*文本管理*中看到了*对齐文本*的示例，其中展示了在固定空间内对齐文本的可能解决方案。可用空间的大小在`COLSIZE`常量中定义，选择适合大多数终端的三列（大多数终端适合 80 列）。

但是，如果用户的终端窗口小于 60 列会发生什么？我们的对齐会被严重破坏。此外，在非常大的窗口上，虽然文本不会被破坏，但与窗口相比会显得太小。

因此，每当显示应保持正确对齐属性的文本时，通常最好考虑用户终端窗口的大小。

# 如何做...

步骤如下：

1.  `shutil.get_terminal_size`函数可以指导终端窗口的大小，并为无法获得大小的情况提供后备。我们将调整`maketable`函数，以适应终端大小。

```py
import shutil
import textwrap, itertools

def maketable(cols):
    term_size = shutil.get_terminal_size(fallback=(80, 24))
    colsize = (term_size.columns // len(cols)) - 3
    if colsize < 1:
        raise ValueError('Column too small')
    return '\n'.join(map(' | '.join, itertools.zip_longest(*[
        [s.ljust(colsize) for s in textwrap.wrap(col, colsize)] for col in cols
    ], fillvalue=' '*colsize)))
```

1.  现在可以在多列中打印任何文本，并看到它适应您的终端窗口的大小：

```py
COLUMNS = 5
TEXT = ['Lorem ipsum dolor sit amet, consectetuer adipiscing elit. '
        'Aenean commodo ligula eget dolor. Aenean massa. '
        'Cum sociis natoque penatibus et magnis dis parturient montes, '
        'nascetur ridiculus mus'] * COLUMNS

print(maketable(TEXT))
```

如果尝试调整终端窗口大小并重新运行脚本，您会注意到文本现在总是以不同的方式对齐，以确保它适合可用的空间。

# 工作原理...

我们的`maketable`函数现在通过获取终端宽度(`term_size.columns`)并将其除以要显示的列数来计算列的大小，而不是依赖于列的大小的常量。

始终减去三个字符，因为我们要考虑`|`分隔符占用的空间。

终端的大小(`term_size`)通过`shutil.get_terminal_size`获取，它将查看`stdout`以检查连接终端的大小。

如果无法检索大小或连接的输出不是终端，则使用回退值。您可以通过将脚本的输出重定向到文件来检查回退值是否按预期工作：

```py
$ python myscript.py > output.txt
```

如果您打开`output.txt`，您应该会看到 80 个字符的回退值被用作文件没有指定宽度。

# 运行系统命令

在某些情况下，特别是在编写系统工具时，可能有一些工作需要转移到另一个命令。例如，如果你需要解压文件，在许多情况下，将工作转移到`gunzip`/`zip`命令可能更合理，而不是尝试在 Python 中复制相同的行为。

在 Python 中有许多处理这项工作的方法，它们都有微妙的差异，可能会让任何开发人员的生活变得困难，因此最好有一个通常有效的解决方案来解决最常见的问题。

# 如何做...

执行以下步骤：

1.  结合`subprocess`和`shlex`模块使我们能够构建一个在大多数情况下都可靠的解决方案：

```py
import shlex
import subprocess

def run(command):
    try:
        result = subprocess.check_output(shlex.split(command), 
                                         stderr=subprocess.STDOUT)
        return 0, result
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output
```

1.  很容易检查它是否按预期工作，无论是成功还是失败的命令：

```py
for path in ('/', '/should_not_exist'):
    status, out = run('ls "{}"'.format(path))
    if status == 0:
        print('<Success>')
    else:
        print('<Error: {}>'.format(status))
    print(out)
```

1.  在我的系统上，这样可以正确列出文件系统的根目录，并对不存在的路径进行抱怨：

```py
<Success>
Applications
Developer
Library
LibraryPreferences
Network
...

<Error: 2>
ls: cannot access /should_not_exist: No such file or directory
```

# 工作原理...

调用命令本身是由`subprocess.check_output`函数执行的，但在调用之前，我们需要正确地将命令拆分为包含命令本身及其参数的列表。依赖于`shlex`使我们能够驱动和区分参数应如何拆分。要查看其效果，可以尝试在任何类 Unix 系统上比较`run('ls / var')`和`run('ls "/ var"')`。第一个将打印很多文件，而第二个将抱怨路径不存在。这是因为在第一种情况下，我们实际上向`ls`发送了两个不同的参数（`/`和`var`），而在第二种情况下，我们发送了一个单一的参数（`"/ var"`）。如果我们没有使用`shlex`，就无法区分这两种情况。

传递`stderr=subprocess.STDOUT`选项，然后处理命令失败的情况（我们可以检测到，因为`run`函数将返回一个非零的状态），允许我们接收失败的描述。

调用我们的命令的繁重工作由`subprocess.check_output`执行，实际上，它是`subprocess.Popen`的包装器，将执行两件事：

1.  使用`subprocess.Popen`生成所需的命令，配置为将输出写入管道，以便父进程（我们自己的程序）可以从该管道中读取并获取输出。

1.  生成线程以持续从打开的管道中消耗内容，以与子进程通信。这确保它们永远不会填满，因为如果它们填满了，我们调用的命令将会被阻塞，因为它将无法再写入任何输出。

# 还有更多...

需要注意的一点是，我们的`run`函数将寻找一个可满足请求命令的可执行文件，但不会运行任何 shell 表达式。因此，无法将 shell 脚本发送给它。如果需要，可以将`shell=True`选项传递给`subprocess.check_output`，但这是极不鼓励的，因为它允许将 shell 代码注入到我们的程序中。

假设您想编写一个命令，打印用户选择的目录的内容；一个非常简单的解决方案可能是以下内容：

```py
import sys
if len(sys.argv) < 2:
    print('Please provide a directory')
    sys.exit(1)
_, out = run('ls {}'.format(sys.argv[1]))
print(out)
```

现在，如果我们在`run`中允许`shell=True`，并且用户提供了诸如`/var; rm -rf /`这样的路径，会发生什么？用户可能最终会删除整个系统磁盘，尽管我们仍然依赖于`shlex`来分割参数，但通过 shell 运行命令仍然不安全。

# 进度条

当进行需要大量时间的工作时（通常是需要 I/O 到较慢的端点，如磁盘或网络的任何工作），让用户知道您正在前进以及还有多少工作要做是一个好主意。进度条虽然不精确，但是是给我们的用户一个关于我们已经完成了多少工作以及还有多少工作要做的概览的很好的方法。

# 如何做...

配方步骤如下：

1.  进度条本身将由装饰器显示，这样我们就可以将其应用到任何我们想要以最小的努力报告进度的函数上。

```py
import shutil, sys

def withprogressbar(func):
    """Decorates ``func`` to display a progress bar while running.

    The decorated function can yield values from 0 to 100 to
    display the progress.
    """
    def _func_with_progress(*args, **kwargs):
        max_width, _ = shutil.get_terminal_size()

        gen = func(*args, **kwargs)
        while True:
            try:
                progress = next(gen)
            except StopIteration as exc:
                sys.stdout.write('\n')
                return exc.value
            else:
                # Build the displayed message so we can compute
                # how much space is left for the progress bar 
                  itself.
                message = '[%s] {}%%'.format(progress)
                # Add 3 characters to cope for the %s and %%
                bar_width = max_width - len(message) + 3  

                filled = int(round(bar_width / 100.0 * progress))
                spaceleft = bar_width - filled
                bar = '=' * filled + ' ' * spaceleft
                sys.stdout.write((message+'\r') % bar)
                sys.stdout.flush()

    return _func_with_progress
```

1.  然后我们需要一个实际执行某些操作并且可能想要报告进度的函数。在这个例子中，它将是一个简单的等待指定时间的函数。

```py
import time

@withprogressbar
def wait(seconds):
    """Waits ``seconds`` seconds and returns how long it waited."""
    start = time.time()
    step = seconds / 100.0
    for i in range(1, 101):
        time.sleep(step)
        yield i  # Send % of progress to withprogressbar

    # Return how much time passed since we started, 
    # which is in fact how long we waited for real.
    return time.time() - start
```

1.  现在调用被装饰的函数应该告诉我们它等待了多长时间，并在等待时显示一个进度条。

```py
print('WAITED', wait(5))
```

1.  当脚本运行时，您应该看到您的进度条和最终结果，看起来像这样：

```py
$ python /tmp/progress.py 
[=====================================] 100%
WAITED 5.308781862258911
```

# 工作原理...

所有的工作都由`withprogressbar`函数完成。它充当装饰器，因此我们可以使用`@withprogressbar`语法将其应用到任何函数上。

这非常方便，因为报告进度的代码与实际执行工作的代码是隔离的，这使我们能够在许多不同的情况下重用它。

为了创建一个装饰器，它在函数本身运行时与被装饰的函数交互，我们依赖于 Python 生成器。

```py
gen = func(*args, **kwargs)
while True:
    try:
        progress = next(gen)
    except StopIteration as exc:
        sys.stdout.write('\n')
        return exc.value
    else:
        # display the progressbar
```

当我们调用被装饰的函数（在我们的例子中是`wait`函数）时，实际上我们将调用装饰器中的`_func_with_progress`。该函数将要做的第一件事就是调用被装饰的函数。

```py
gen = func(*args, **kwargs)
```

由于被装饰的函数包含一个`yield progress`语句，每当它想显示一些进度（在`wait`中的`for`循环中的`yield i`），函数将返回`generator`。

每当生成器遇到`yield progress`语句时，我们将其作为应用于生成器的下一个函数的返回值收到。

```py
progress = next(gen)
```

然后我们可以显示我们的进度并再次调用`next(gen)`，这样被装饰的函数就可以继续前进并返回新的进度（被装饰的函数当前在`yield`处暂停，直到我们在其上调用`next`，这就是为什么我们的整个代码都包裹在`while True:`中的原因，让函数永远继续，直到它完成它要做的工作）。

当被装饰的函数完成了所有它要做的工作时，它将引发一个`StopIteration`异常，该异常将包含被装饰函数在`.value`属性中返回的值。

由于我们希望将任何返回值传播给调用者，我们只需自己返回该值。如果被装饰的函数应该返回其完成的工作的某些结果，比如一个`download(url)`函数应该返回对下载文件的引用，这一点尤为重要。

在返回之前，我们打印一个新行。

```py
sys.stdout.write('\n')
```

这确保了进度条后面的任何内容不会与进度条本身重叠，而是会打印在新的一行上。

然后我们只需显示进度条本身。配方中进度条部分的核心基于只有两行代码：

```py
sys.stdout.write((message+'\r') % bar)
sys.stdout.flush()
```

这两行将确保我们的消息在屏幕上打印，而不像`print`通常做的那样换行。相反，这将回到同一行的开头。尝试用`'\n'`替换`'\r'`，你会立即看到区别。使用`'\r'`，你会看到一个进度条从 0 到 100%移动，而使用`'\n'`，你会看到许多进度条被打印。

然后需要调用`sys.stdout.flush()`来确保进度条实际上被显示出来，因为通常只有在新的一行上才会刷新输出，而我们只是一遍又一遍地打印同一行，除非我们明确地刷新它，否则它不会被刷新。

现在我们知道如何绘制进度条并更新它，函数的其余部分涉及计算要显示的进度条：

```py
message = '[%s] {}%%'.format(progress)
bar_width = max_width - len(message) + 3  # Add 3 characters to cope for the %s and %%

filled = int(round(bar_width / 100.0 * progress))
spaceleft = bar_width - filled
bar = '=' * filled + ' ' * spaceleft
```

首先，我们计算`message`，这是我们想要显示在屏幕上的内容。消息是在没有进度条本身的情况下计算的，对于进度条，我们留下了一个`%s`占位符，以便稍后填充它。

我们这样做是为了知道在我们显示周围的括号和百分比后，进度条本身还有多少空间。这个值是`bar_width`，它是通过从屏幕宽度的最大值（在我们的函数开始时使用`shutil.get_terminal_size()`检索）中减去我们的消息的大小来计算的。我们必须添加的三个额外字符将解决在我们的消息中`%s`和`%%`消耗的空间，一旦消息显示到屏幕上，`%s`将被进度条本身替换，`%%`将解析为一个单独的`%`。

一旦我们知道了进度条本身有多少空间可用，我们就计算出应该用`'='`（已完成的部分）填充多少空间，以及应该用空格`' '`（尚未完成的部分）填充多少空间。这是通过计算要填充和匹配我们的进度的百分比的屏幕大小来实现的：

```py
filled = int(round(bar_width / 100.0 * progress))
```

一旦我们知道要用`'='`填充多少，剩下的就只是空格：

```py
spaceleft = bar_width - filled
```

因此，我们可以用填充的等号和`spaceleft`空格来构建我们的进度条：

```py
bar = '=' * filled + ' ' * spaceleft
```

一旦进度条准备好了，它将通过`%`字符串格式化操作符注入到在屏幕上显示的消息中：

```py
sys.stdout.write((message+'\r') % bar)
```

如果你注意到了，我混合了两种字符串格式化（`str.format`和`%`）。我这样做是因为我认为这样做可以更清楚地说明格式化的过程，而不是在每个格式化步骤上都要正确地进行转义。

# 消息框

尽管现在不太常见，但能够创建交互式基于字符的用户界面仍然具有很大的价值，特别是当只需要一个带有“确定”按钮的简单消息对话框或一个带有“确定/取消”对话框时；通过一个漂亮的文本对话框，可以更好地引导用户的注意力。

# 准备工作

`curses`库只包括在 Unix 系统的 Python 中，因此 Windows 用户可能需要一个解决方案，比如 CygWin 或 Linux 子系统，以便能够拥有包括`curses`支持的 Python 设置。

# 如何做到这一点...

对于这个配方，执行以下步骤：

1.  我们将制作一个`MessageBox.show`方法，我们可以在需要时用它来显示消息框。`MessageBox`类将能够显示只有确定或确定/取消按钮的消息框。

```py
import curses
import textwrap
import itertools

class MessageBox(object):
    @classmethod
    def show(cls, message, cancel=False, width=40):
        """Show a message with an Ok/Cancel dialog.

        Provide ``cancel=True`` argument to show a cancel button 
        too.
        Returns the user selected choice:

            - 0 = Ok
            - 1 = Cancel
        """
        dialog = MessageBox(message, width, cancel)
        return curses.wrapper(dialog._show)

    def __init__(self, message, width, cancel):
        self._message = self._build_message(width, message)
        self._width = width
        self._height = max(self._message.count('\n')+1, 3) + 6
        self._selected = 0
        self._buttons = ['Ok']
        if cancel:
            self._buttons.append('Cancel')

    def _build_message(self, width, message):
        lines = []
        for line in message.split('\n'):
            if line.strip():
                lines.extend(textwrap.wrap(line, width-4,                                             
                             replace_whitespace=False))
            else:
                lines.append('')
        return '\n'.join(lines)

    def _show(self, stdscr):
        win = curses.newwin(self._height, self._width, 
                            (curses.LINES - self._height) // 2, 
                            (curses.COLS - self._width) // 2)
        win.keypad(1)
        win.border()
        textbox = win.derwin(self._height - 1, self._width - 3, 
                             1, 2)
        textbox.addstr(0, 0, self._message)
        return self._loop(win)

    def _loop(self, win):
        while True:
            for idx, btntext in enumerate(self._buttons):
                allowedspace = self._width // len(self._buttons)
                btn = win.derwin(
                    3, 10, 
                    self._height - 4, 
                    (((allowedspace-10)//2*idx) + allowedspace*idx 
                       + 2)
                )
                btn.border()
                flag = 0
                if idx == self._selected:
                    flag = curses.A_BOLD
                btn.addstr(1, (10-len(btntext))//2, btntext, flag)
            win.refresh()

            key = win.getch()
            if key == curses.KEY_RIGHT:
                self._selected = 1
            elif key == curses.KEY_LEFT:
                self._selected = 0
            elif key == ord('\n'):
                return self._selected
```

1.  然后我们可以通过`MessageBox.show`方法来使用它：

```py
MessageBox.show('Hello World,\n\npress enter to continue')
```

1.  我们甚至可以用它来检查用户的选择：

```py
if MessageBox.show('Are you sure?\n\npress enter to confirm',
                   cancel=True) == 0:
    print("Yeah! Let's continue")
else:
    print("That's sad, hope to see you soon")
```

# 它是如何工作的...

消息框基于`curses`库，它允许我们在屏幕上绘制基于文本的图形。当我们使用对话框时，我们将进入全屏文本图形模式，一旦退出，我们将恢复先前的终端状态。

这使我们能够在更复杂的程序中交错使用`MessageBox`类，而不必用`curses`编写整个程序。这是由`curses.wrapper`函数允许的，该函数在`MessageBox.show`类方法中用于包装实际显示框的`MessageBox._show`方法。

消息显示是在`MessageBox`初始化程序中准备的，通过`MessageBox._build_message`方法，以确保当消息太长时自动换行，并正确处理多行文本。消息框的高度取决于消息的长度和结果行数，再加上我们始终包括的六行，用于添加边框（占用两行）和按钮（占用四行）。

然后，`MessageBox._show`方法创建实际的框窗口，为其添加边框，并在其中显示消息。消息显示后，我们进入`MessageBox._loop`，等待用户在 OK 和取消之间做出选择。

`MessageBox._loop`方法通过`win.derwin`函数绘制所有必需的按钮及其边框。每个按钮宽 10 个字符，高 3 个字符，并根据`allowedspace`的值显示自身，该值为每个按钮保留了相等的框空间。然后，一旦绘制了按钮框，它将检查当前显示的按钮是否为所选按钮；如果是，则使用粗体文本显示按钮的标签。这使用户可以知道当前选择的选项。

绘制了两个按钮后，我们调用`win.refresh()`来实际在屏幕上显示我们刚刚绘制的内容。

然后我们等待用户按任意键以相应地更新屏幕；左/右箭头键将在 OK/取消选项之间切换，*Enter*将确认当前选择。

如果用户更改了所选按钮（通过按左或右键），我们将再次循环并重新绘制按钮。我们只需要重新绘制按钮，因为屏幕的其余部分没有改变；窗口边框和消息仍然是相同的，因此无需覆盖它们。屏幕的内容始终保留，除非调用了`win.erase()`方法，因此我们永远不需要重新绘制不需要更新的屏幕部分。

通过这种方式，我们还可以避免重新绘制按钮本身。这是因为只有取消/确定文本在从粗体到普通体和反之时需要重新绘制。

用户按下*Enter*键后，我们退出循环，并返回当前选择的 OK 和取消之间的选择。这允许调用者根据用户的选择采取行动。

# 输入框

在编写基于控制台的软件时，有时需要要求用户提供无法通过命令选项轻松提供的长文本输入。

在 Unix 世界中有一些这样的例子，比如编辑`crontab`或一次调整多个配置选项。其中大多数依赖于启动一个完整的第三方编辑器，比如**nano**或**vim**，但是可以很容易地使用 Python 标准库滚动一个解决方案，这在许多情况下将足够满足我们的工具需要长或复杂的用户输入。

# 准备就绪

`curses`库仅包含在 Unix 系统的 Python 中，因此 Windows 用户可能需要一个解决方案，例如 CygWin 或 Linux 子系统，以便能够拥有包括`curses`支持的 Python 设置。

# 如何做...

对于这个示例，执行以下步骤：

1.  Python 标准库提供了一个`curses.textpad`模块，其中包含一个带有`emacs`的多行文本编辑器的基础，例如键绑定。我们只需要稍微扩展它以添加一些所需的行为和修复：

```py
import curses
from curses.textpad import Textbox, rectangle

class TextInput(object):
    @classmethod
    def show(cls, message, content=None):
        return curses.wrapper(cls(message, content)._show)

    def __init__(self, message, content):
        self._message = message
        self._content = content

    def _show(self, stdscr):
        # Set a reasonable size for our input box.
        lines, cols = curses.LINES - 10, curses.COLS - 40

        y_begin, x_begin = (curses.LINES - lines) // 2, 
                           (curses.COLS - cols) // 2
        editwin = curses.newwin(lines, cols, y_begin, x_begin)
        editwin.addstr(0, 1, "{}: (hit Ctrl-G to submit)"
         .format(self._message))
        rectangle(editwin, 1, 0, lines-2, cols-1)
        editwin.refresh()

        inputwin = curses.newwin(lines-4, cols-2, y_begin+2, 
        x_begin+1)
        box = Textbox(inputwin)
        self._load(box, self._content)
        return self._edit(box)

    def _load(self, box, text):
        if not text:
            return
        for c in text:
            box._insert_printable_char(c)

    def _edit(self, box):
        while True:
            ch = box.win.getch()
            if not ch:
                continue
            if ch == 127:
                ch = curses.KEY_BACKSPACE
            if not box.do_command(ch):
                break
            box.win.refresh()
        return box.gather()
```

1.  然后我们可以从用户那里读取输入：

```py
result = TextInput.show('Insert your name:')
print('Your name:', result)
```

1.  我们甚至可以要求它编辑现有文本：

```py
result = TextInput.show('Insert your name:', 
                        content='Some Text\nTo be edited')
print('Your name:', result)
```

# 工作原理...

一切都始于`TextInput._show`方法，该方法准备了两个窗口；第一个绘制帮助文本（在我们的示例中为'插入您的姓名：'），以及文本区域的边框框。

一旦绘制完成，它会创建一个专门用于`Textbox`的新窗口，因为文本框将自由地插入、删除和编辑该窗口的内容。

如果我们有现有的内容（`content=参数`），`TextInput._load`函数会负责在继续编辑之前将其插入到文本框中。提供的内容中的每个字符都通过`Textbox._insert_printable_char`函数注入到文本框窗口中。

然后我们最终可以进入编辑循环（`TextInput._edit`方法），在那里我们监听按键并做出相应反应。实际上，`Textbox.do_command`已经为我们完成了大部分工作，因此我们只需要将按下的键转发给它，以将字符插入到我们的文本中或对特殊命令做出反应。这个方法的特殊部分是我们检查字符 127，它是*Backspace*，并将其替换为`curses.KEY_BACKSPACE`，因为并非所有终端在按下*Backspace*键时发送相同的代码。一旦字符被`do_command`处理，我们就可以刷新窗口，以便任何新文本出现并再次循环。

当用户按下*Ctrl* + *G*时，编辑器将认为文本已完成并退出编辑循环。在这之前，我们调用`Textbox.gather`来获取文本编辑器的全部内容并将其发送回调用者。

需要注意的是，内容实际上是从`curses`窗口的内容中获取的。因此，它实际上包括您屏幕上看到的所有空白空间。因此，`Textbox.gather`方法将剥离空白空间，以避免将大部分空白空间包围您的文本发送回给您。如果您尝试编写包含多个空行的内容，这一点就非常明显；它们将与其余空白空间一起被剥离。
