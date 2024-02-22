# 第二章：简化任务自动化

在本章中，我们将涵盖以下内容：

+   准备一个任务

+   设置一个定时任务

+   捕获错误和问题

+   发送电子邮件通知

# 介绍

要正确自动化任务，我们需要一个平台，让它们在适当的时间自动运行。需要手动运行的任务并不真正实现了自动化。

但是，为了能够让它们在后台运行而不用担心更紧急的问题，任务需要适合以 *fire-and-forget* 模式运行。我们应该能够监控它是否正确运行，确保我们能够捕获未来的动作（比如在出现有趣的情况时接收通知），并知道在运行过程中是否出现了任何错误。

确保软件始终以高可靠性一致运行实际上是一件大事，这是一个需要专业知识和人员的领域，通常被称为系统管理员、运维或 **SRE**（**站点可靠性工程**）。像亚马逊和谷歌这样的网站需要巨大的投资来确保一切都能 24/7 正常运行。

这本书的目标要比那更加谦虚。你可能不需要每年低于几秒的停机时间。以合理的可靠性运行任务要容易得多。但是，要意识到还有维护工作要做，所以要有所准备。

# 准备一个任务

一切都始于准确定义需要运行的任务，并设计成不需要人工干预就能运行的方式。

一些理想的特点如下：

1.  **单一、明确的入口点**：不会对要运行的任务产生混淆。

1.  **清晰的参数**：如果有任何参数，它们应该非常明确。

1.  **无交互**：停止执行以请求用户信息是不可能的。

1.  **结果应该被存储**：可以在运行时以外的时间进行检查。

1.  **清晰的结果**：如果我们在交互中工作，我们会接受更详细的结果或进度报告。但是，对于自动化任务，最终结果应尽可能简洁明了。

1.  **错误应该被记录下来**：以便分析出错的原因。

命令行程序已经具备了许多这些特点。它有明确的运行方式，有定义的参数，并且结果可以被存储，即使只是以文本格式。但是，通过配置文件来澄清参数，并且输出到一个文件，可以进一步改进。

注意，第 6 点是 *捕获错误和问题* 配方的目标，并将在那里进行介绍。

为了避免交互，不要使用任何需要用户输入的命令，比如 `input`。记得删除调试时的断点！

# 准备工作

我们将按照一个结构开始，其中一个主函数作为入口点，并将所有参数提供给它。

这与第一章中 *添加命令行参数* 配方中呈现的基本结构相同，*让我们开始自动化之旅*。

定义一个主函数，包含所有明确的参数，涵盖了第 1 和第 2 点。第 3 点并不难实现。

为了改进第 2 和第 5 点，我们将研究如何从文件中检索配置并将结果存储在另一个文件中。另一个选项是发送通知，比如电子邮件，这将在本章后面介绍。

# 如何做...

1.  准备以下任务，并将其保存为 `prepare_task_step1.py`：

```py
import argparse

def main(number, other_number):
    result = number * other_number
    print(f'The result is {result}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n1', type=int, help='A number', default=1)
    parser.add_argument('-n2', type=int, help='Another number', default=1)

    args = parser.parse_args()

    main(args.n1, args.n2)
```

1.  更新文件以定义包含两个参数的配置文件，并将其保存为 `prepare_task_step2.py`。注意，定义配置文件会覆盖任何命令行参数：

```py
import argparse
import configparser

def main(number, other_number):
    result = number * other_number
    print(f'The result is {result}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n1', type=int, help='A number', default=1)
    parser.add_argument('-n2', type=int, help='Another number', default=1)

    parser.add_argument('--config', '-c', type=argparse.FileType('r'),
                        help='config file')

    args = parser.parse_args()
    if args.config:
        config = configparser.ConfigParser()
        config.read_file(args.config)
        # Transforming values into integers
        args.n1 = int(config['DEFAULT']['n1'])
        args.n2 = int(config['DEFAULT']['n2'])

    main(args.n1, args.n2)
```

1.  创建配置文件 `config.ini`：

```py
[ARGUMENTS]
n1=5
n2=7
```

1.  使用配置文件运行命令。注意，配置文件会覆盖命令行参数，就像第 2 步中描述的那样：

```py
$ python3 prepare_task_step2.py -c config.ini
The result is 35
$ python3 prepare_task_step2.py -c config.ini -n1 2 -n2 3
The result is 35
```

1.  添加一个参数来将结果存储在文件中，并将其保存为 `prepare_task_step5.py`：

```py
import argparse
import sys
import configparser

def main(number, other_number, output):
    result = number * other_number
    print(f'The result is {result}', file=output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n1', type=int, help='A number', default=1)
    parser.add_argument('-n2', type=int, help='Another number', default=1)

    parser.add_argument('--config', '-c', type=argparse.FileType('r'),
                        help='config file')
    parser.add_argument('-o', dest='output', type=argparse.FileType('w'),
                        help='output file',
                        default=sys.stdout)

    args = parser.parse_args()
    if args.config:
        config = configparser.ConfigParser()
        config.read_file(args.config)
        # Transforming values into integers
        args.n1 = int(config['DEFAULT']['n1'])
        args.n2 = int(config['DEFAULT']['n2'])

    main(args.n1, args.n2, args.output)
```

1.  运行结果以检查是否将输出发送到定义的文件。请注意，结果文件之外没有输出：

```py
$ python3 prepare_task_step5.py -n1 3 -n2 5 -o result.txt
$ cat result.txt
The result is 15
$ python3 prepare_task_step5.py -c config.ini -o result2.txt
$ cat result2.txt
The result is 35
```

# 工作原理...

请注意，`argparse`模块允许我们将文件定义为参数，使用`argparse.FileType`类型，并自动打开它们。这非常方便，如果文件无效，将会引发错误。

记得以正确的模式打开文件。在步骤 5 中，配置文件以读模式（`r`）打开，输出文件以写模式（`w`）打开，如果文件存在，将覆盖该文件。您可能会发现追加模式（`a`），它将在现有文件的末尾添加下一段数据。

`configparser`模块允许我们轻松使用配置文件。如步骤 2 所示，文件的解析就像下面这样简单：

```py
config = configparser.ConfigParser()
config.read_file(file)
```

然后，配置将作为由部分和值分隔的字典访问。请注意，值始终以字符串格式存储，需要转换为其他类型，如整数：

如果需要获取布尔值，请不要执行`value = bool(config[raw_value])`，因为无论如何都会转换为`True`；例如，字符串`False`是一个真字符串，因为它不是空的。相反，使用`.getboolean`方法，例如，`value = config.getboolean(raw_value)`。

Python3 允许我们向`print`函数传递一个`file`参数，它将写入该文件。步骤 5 展示了将所有打印信息重定向到文件的用法。

请注意，默认参数是`sys.stdout`，它将值打印到终端（标准输出）。这样做会使得在没有`-o`参数的情况下调用脚本将在屏幕上显示信息，这在调试时很有帮助：

```py
$ python3 prepare_task_step5.py -c config.ini
The result is 35
$ python3 prepare_task_step5.py -c config.ini -o result.txt
$ cat result.txt
The result is 35
```

# 还有更多...

请查看官方 Python 文档中`configparse`的完整文档：[`docs.python.org/3/library/configparser.html.`](https://docs.python.org/3/library/configparser.html)

在大多数情况下，这个配置解析器应该足够好用，但如果需要更多的功能，可以使用 YAML 文件作为配置文件。YAML 文件（[`learn.getgrav.org/advanced/yaml`](https://learn.getgrav.org/advanced/yaml)）作为配置文件非常常见，结构更好，可以直接解析，考虑到数据类型。

1.  将 PyYAML 添加到`requirements.txt`文件并安装它：

```py
PyYAML==3.12
```

1.  创建`prepare_task_yaml.py`文件：

```py
import yaml
import argparse
import sys

def main(number, other_number, output):
    result = number * other_number
    print(f'The result is {result}', file=output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n1', type=int, help='A number', default=1)
    parser.add_argument('-n2', type=int, help='Another number', default=1)

    parser.add_argument('-c', dest='config', type=argparse.FileType('r'),
 help='config file in YAML format',
 default=None)
    parser.add_argument('-o', dest='output', type=argparse.FileType('w'),
                        help='output file',
                        default=sys.stdout)

    args = parser.parse_args()
    if args.config:
        config = yaml.load(args.config)
        # No need to transform values
        args.n1 = config['ARGUMENTS']['n1']
        args.n2 = config['ARGUMENTS']['n2']

    main(args.n1, args.n2, args.output)
```

1.  定义配置文件`config.yaml`，可在 GitHub [`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter02/config.yaml`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter02/config.yaml) 中找到：

```py
ARGUMENTS:
    n1: 7
    n2: 4
```

1.  然后运行以下命令：

```py
$ python3 prepare_task_yaml.py -c config.yaml
The result is 28

```

还有设置默认配置文件和默认输出文件的可能性。这对于创建一个不需要输入参数的纯任务非常方便。

一般规则是，如果任务有一个非常具体的目标，请尽量避免创建太多的输入和配置参数。尝试将输入参数限制为任务的不同执行。一个永远不会改变的参数可能很好地被定义为**常量**。大量的参数将使配置文件或命令行参数变得复杂，并将在长期内增加更多的维护。另一方面，如果您的目标是创建一个非常灵活的工具，可以在非常不同的情况下使用，那么创建更多的参数可能是一个好主意。尝试找到适合自己的平衡！

# 另请参阅

+   第一章中的*命令行参数*配方，*让我们开始自动化之旅*

+   *发送电子邮件通知*配方

+   第十章中的*使用断点进行调试*配方，*调试技术*

# 设置 cron 作业

Cron 是一种老式但可靠的执行命令的方式。它自 Unix 的 70 年代以来就存在，并且是系统管理中常用的维护方式，比如释放空间、旋转日志、制作备份和其他常见操作。

这个配方是特定于 Unix 的，因此它将在 Linux 和 MacOS 中工作。虽然在 Windows 中安排任务是可能的，但非常不同，并且使用任务计划程序，这里不会描述。如果你有 Linux 服务器的访问权限，这可能是安排周期性任务的好方法。其主要优点如下：

+   它几乎存在于所有的 Unix 或 Linux 系统中，并配置为自动运行。

+   它很容易使用，尽管有点欺骗性。

+   这是众所周知的。几乎所有涉及管理任务的人都对如何使用它有一个大致的概念。

+   它允许轻松地周期性命令，精度很高。

但它也有一些缺点，如下：

+   默认情况下，它可能不会提供太多反馈。检索输出、记录执行和错误是至关重要的。

+   任务应尽可能自包含，以避免环境变量的问题，比如使用错误的 Python 解释器，或者应该执行的路径。

+   它是特定于 Unix 的。

+   只有固定的周期时间可用。

+   它不控制同时运行的任务数量。每次倒计时结束时，它都会创建一个新任务。例如，一个需要一个小时才能完成的任务，计划每 45 分钟运行一次，将有 15 分钟的重叠时间，两个任务将同时运行。

不要低估最新效果。同时运行多个昂贵的任务可能会对性能产生不良影响。昂贵的任务重叠可能导致竞争条件，使每个任务都无法完成！充分时间让你的任务完成并密切关注它们。

# 准备就绪

我们将生成一个名为`cron.py`的脚本：

```py
import argparse
import sys
from datetime import datetime
import configparser

def main(number, other_number, output):
    result = number * other_number
    print(f'[{datetime.utcnow().isoformat()}] The result is {result}', 
          file=output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', '-c', type=argparse.FileType('r'),
                        help='config file',
                        default='/etc/automate.ini')
    parser.add_argument('-o', dest='output', type=argparse.FileType('a'),
                        help='output file',
                        default=sys.stdout)

    args = parser.parse_args()
    if args.config:
        config = configparser.ConfigParser()
        config.read_file(args.config)
        # Transforming values into integers
        args.n1 = int(config['DEFAULT']['n1'])
        args.n2 = int(config['DEFAULT']['n2'])

    main(args.n1, args.n2, args.output)
```

注意以下细节：

1.  配置文件默认为`/etc/automate.ini`。重用上一个配方中的`config.ini`。

1.  时间戳已添加到输出中。这将明确显示任务运行的时间。

1.  结果将被添加到文件中，如使用`'a'`模式打开文件所示。

1.  `ArgumentDefaultsHelpFormatter`参数在使用`-h`参数打印帮助时会自动添加有关默认值的信息。

检查任务是否产生了预期的结果，并且你可以记录到一个已知的文件中：

```py
$ python3 cron.py
[2018-05-15 22:22:31.436912] The result is 35
$ python3 cron.py -o /path/automate.log
$ cat /path/automate.log
[2018-05-15 22:28:08.833272] The result is 35
```

# 如何做...

1.  获取 Python 解释器的完整路径。这是你的虚拟环境中的解释器：

```py
$ which python
/your/path/.venv/bin/python
```

1.  准备执行 cron。获取完整路径并检查是否可以无问题执行。执行几次：

```py
$ /your/path/.venv/bin/python /your/path/cron.py -o /path/automate.log
$ /your/path/.venv/bin/python /your/path/cron.py -o /path/automate.log

```

1.  检查结果是否正确地添加到结果文件中：

```py
$ cat /path/automate.log
[2018-05-15 22:28:08.833272] The result is 35
[2018-05-15 22:28:10.510743] The result is 35
```

1.  编辑 crontab 文件，以便每五分钟运行一次任务：

```py
$ crontab -e

*/5 * * * * /your/path/.venv/bin/python /your/path/cron.py -o /path/automate.log
```

请注意，这将使用默认的命令行编辑器打开一个编辑终端。

如果你还没有设置默认的命令行编辑器，默认情况下可能是 Vim。如果你对 Vim 没有经验，这可能会让你感到困惑。按*I*开始插入文本，*Esc*完成后退出。然后，在保存文件后退出，使用`:wq`。有关 Vim 的更多信息，请参阅此介绍：[`null-byte.wonderhowto.com/how-to/intro-vim-unix-text-editor-every-hacker-should-be-familiar-with-0174674`](https://null-byte.wonderhowto.com/how-to/intro-vim-unix-text-editor-every-hacker-should-be-familiar-with-0174674)。

有关如何更改默认命令行编辑器的信息，请参阅以下链接：[`www.a2hosting.com/kb/developer-corner/linux/setting-the-default-text-editor-in-linux.`](https://www.a2hosting.com/kb/developer-corner/linux/setting-the-default-text-editor-in-linux)

1.  检查 crontab 内容。请注意，这会显示 crontab 内容，但不会设置为编辑：

```py
$ contab -l
*/5 * * * * /your/path/.venv/bin/python /your/path/cron.py -o /path/automate.log
```

1.  等待并检查结果文件，看任务是如何执行的：

```py
$ tail -F /path/automate.log
[2018-05-17 21:20:00.611540] The result is 35
[2018-05-17 21:25:01.174835] The result is 35
[2018-05-17 21:30:00.886452] The result is 35
```

# 它的工作原理...

crontab 行由描述任务运行频率的行（前六个元素）和任务组成。初始的六个元素中的每一个代表不同的执行时间单位。它们大多数是星号，表示*任何*：

```py
* * * * * *
| | | | | | 
| | | | | +-- Year              (range: 1900-3000)
| | | | +---- Day of the Week   (range: 1-7, 1 standing for Monday)
| | | +------ Month of the Year (range: 1-12)
| | +-------- Day of the Month  (range: 1-31)
| +---------- Hour              (range: 0-23)
+------------ Minute            (range: 0-59)
```

因此，我们的行，`*/5 * * * * *`，意味着*每当分钟可被 5 整除时，在所有小时、所有天...所有年*。

以下是一些例子：

```py
30  15 * * * * means "every day at 15:30"
30   * * * * * means "every hour, at 30 minutes"
0,30 * * * * * means "every hour, at 0 minutes and 30 minutes"
*/30 * * * * * means "every half hour"
0    0 * * 1 * means "every Monday at 00:00"
```

不要试图猜测太多。使用像[`crontab.guru/`](https://crontab.guru/)这样的备忘单来获取示例和调整。大多数常见用法将直接在那里描述。您还可以编辑一个公式并获得有关其运行方式的描述性文本。

在描述如何运行 cron 作业之后，包括执行任务的行，如*如何操作…*部分的第 2 步中准备的那样。

请注意，任务的描述中包含了每个相关文件的完整路径——解释器、脚本和输出文件。这消除了与路径相关的所有歧义，并减少了可能出现错误的机会。一个非常常见的错误是无法确定其中一个（或多个）元素。

# 还有更多...

如果 crontab 执行时出现任何问题，您应该收到系统邮件。这将显示为终端中的消息，如下所示：

```py
You have mail.
$
```

这可以通过`mail`来阅读：

```py
$ mail
Mail version 8.1 6/6/93\. Type ? for help.
"/var/mail/jaime": 1 message 1 new
>N 1 jaime@Jaimes-iMac-5K Thu May 17 21:15 19/914 "Cron <jaime@Jaimes-iM"
? 1
Message 1:
...
/usr/local/Cellar/python/3.7.0/Frameworks/Python.framework/Versions/3.7/Resources/Python.app/Contents/MacOS/Python: can't open file 'cron.py': [Errno 2] No such file or directory
```

在下一个食谱中，我们将看到独立捕获错误的方法，以便任务可以顺利运行。

# 另请参阅

+   第一章《让我们开始自动化之旅》中的*添加命令行选项*食谱

+   *捕获错误和问题*食谱

# 捕获错误和问题

自动化任务的主要特点是其*fire-and-forget*质量。我们不会积极地查看结果，而是让它在后台运行。

此外，由于本书中大多数食谱涉及外部信息，如网页或其他报告，因此在运行时发现意外问题的可能性很高。这个食谱将呈现一个自动化任务，它将安全地将意外行为存储在一个日志文件中，以便以后检查。

# 准备工作

作为起点，我们将使用一个任务，该任务将按照命令行中的描述来除两个数字。

这个任务与*如何操作…*部分中的第 5 步中介绍的任务非常相似，但是我们将除法代替乘法。

# 如何操作...

1.  创建`task_with_error_handling_step1.py`文件，如下所示：

```py
import argparse
import sys

def main(number, other_number, output):
    result = number / other_number
    print(f'The result is {result}', file=output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n1', type=int, help='A number', default=1)
    parser.add_argument('-n2', type=int, help='Another number', default=1)      
    parser.add_argument('-o', dest='output', type=argparse.FileType('w'),
                        help='output file', default=sys.stdout)

    args = parser.parse_args()

    main(args.n1, args.n2, args.output)
```

1.  多次执行它，看看它是如何除以两个数字的：

```py
$ python3 task_with_error_handling_step1.py -n1 3 -n2 2
The result is 1.5
$ python3 task_with_error_handling_step1.py -n1 25 -n2 5
The result is 5.0
```

1.  检查除以`0`是否会产生错误，并且该错误是否未记录在结果文件中：

```py
$ python task_with_error_handling_step1.py -n1 5 -n2 1 -o result.txt
$ cat result.txt
The result is 5.0
$ python task_with_error_handling_step1.py -n1 5 -n2 0 -o result.txt
Traceback (most recent call last):
 File "task_with_error_handling_step1.py", line 20, in <module>
 main(args.n1, args.n2, args.output)
 File "task_with_error_handling_step1.py", line 6, in main
 result = number / other_number
ZeroDivisionError: division by zero
$ cat result.txt
```

1.  创建`task_with_error_handling_step4.py`文件：

```py
import logging
import sys
import logging

LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
LOG_LEVEL = logging.DEBUG

def main(number, other_number, output):
    logging.info(f'Dividing {number} between {other_number}')
    result = number / other_number
    print(f'The result is {result}', file=output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n1', type=int, help='A number', default=1)
    parser.add_argument('-n2', type=int, help='Another number', default=1)

    parser.add_argument('-o', dest='output', type=argparse.FileType('w'),
                        help='output file', default=sys.stdout)
    parser.add_argument('-l', dest='log', type=str, help='log file',
                        default=None)

    args = parser.parse_args()
    if args.log:
        logging.basicConfig(format=LOG_FORMAT, filename=args.log,
                            level=LOG_LEVEL)
    else:
        logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)

    try:
        main(args.n1, args.n2, args.output)
    except Exception as exc:
        logging.exception("Error running task")
        exit(1)
```

1.  运行它以检查它是否显示正确的`INFO`和`ERROR`日志，并且是否将其存储在日志文件中：

```py
$ python3 task_with_error_handling_step4.py -n1 5 -n2 0
2018-05-19 14:25:28,849 root INFO Dividing 5 between 0
2018-05-19 14:25:28,849 root ERROR division by zero
Traceback (most recent call last):
 File "task_with_error_handling_step4.py", line 31, in <module>
 main(args.n1, args.n2, args.output)
 File "task_with_error_handling_step4.py", line 10, in main
 result = number / other_number
ZeroDivisionError: division by zero
$ python3 task_with_error_handling_step4.py -n1 5 -n2 0 -l error.log
$ python3 task_with_error_handling_step4.py -n1 5 -n2 0 -l error.log
$ cat error.log
2018-05-19 14:26:15,376 root INFO Dividing 5 between 0
2018-05-19 14:26:15,376 root ERROR division by zero
Traceback (most recent call last):
 File "task_with_error_handling_step4.py", line 33, in <module>
 main(args.n1, args.n2, args.output)
 File "task_with_error_handling_step4.py", line 11, in main
 result = number / other_number
ZeroDivisionError: division by zero
2018-05-19 14:26:19,960 root INFO Dividing 5 between 0
2018-05-19 14:26:19,961 root ERROR division by zero
Traceback (most recent call last):
 File "task_with_error_handling_step4.py", line 33, in <module>
 main(args.n1, args.n2, args.output)
 File "task_with_error_handling_step4.py", line 11, in main
 result = number / other_number
ZeroDivisionError: division by zero
```

# 它是如何工作的...

为了正确捕获任何意外异常，主函数应该被包装到一个`try-except`块中，就像*如何操作…*部分中的第 4 步中所做的那样。将此与第 1 步中未包装代码的方式进行比较：

```py
    try:
        main(...)
    except Exception as exc:
        # Something went wrong
        logging.exception("Error running task")
        exit(1)
```

请注意，记录异常对于获取出了什么问题很重要。

这种异常被昵称为*宝可梦*，因为它可以*捕获所有*，因此它将在最高级别捕获任何意外错误。不要在代码的其他区域使用它，因为捕获所有可能会隐藏意外错误。至少，任何意外异常都应该被记录下来以便进行进一步分析。

使用`exit(1)`调用额外的步骤来以状态 1 退出通知操作系统我们的脚本出了问题。

`logging`模块允许我们记录。请注意基本配置，其中包括一个可选的文件来存储日志、格式和要显示的日志级别。

日志的可用级别从不太关键到更关键——`DEBUG`、`INFO`、`WARNING`、`ERROR`和`CRITICAL`。日志级别将设置记录消息所需的最小严重性。例如，如果将严重性设置为`WARNING`，则不会存储`INFO`日志。

创建日志很容易。您可以通过调用`logging.<logging level>`方法来实现（其中`logging level`是`debug`、`info`等）。例如：

```py
>>> import logging
>>> logging.basicConfig(level=logging.INFO)
>>> logging.warning('a warning message')
WARNING:root:a warning message
>>> logging.info('an info message')
INFO:root:an info message
>>> logging.debug('a debug message')
>>>
```

注意，低于`INFO`的严重性的日志不会显示。使用级别定义来调整要显示的信息量。例如，这可能会改变`DEBUG`日志仅在开发任务时使用，但在运行时不显示。请注意，`task_with_error_handling_step4.py`默认将日志级别定义为`DEBUG`。

良好的日志级别定义是显示相关信息的关键，同时减少垃圾邮件。有时设置起来并不容易，但特别是如果有多个人参与，尝试就`WARNING`与`ERROR`的确切含义达成一致，以避免误解。

`logging.exception()`是一个特殊情况，它将创建一个`ERROR`日志，但也将包括有关异常的信息，例如**堆栈跟踪**。

记得检查日志以发现错误。一个有用的提醒是在结果文件中添加一个注释，如下所示：

```py
try:
    main(args.n1, args.n2, args.output)
except Exception as exc:
    logging.exception(exc)
    print('There has been an error. Check the logs', file=args.output)
```

# 还有更多...

Python `logging`模块具有许多功能，例如以下内容：

+   进一步调整日志的格式，例如，包括生成日志的文件和行号。

+   定义不同的记录器对象，每个对象都有自己的配置，如日志级别和格式。这允许以不同的方式将日志发送到不同的系统，尽管通常不会出于简单起见而使用。

+   将日志发送到多个位置，例如标准输出和文件，甚至远程记录器。

+   自动旋转日志，创建新的日志文件，一段时间或大小后。这对于按天保持日志组织和允许压缩或删除旧日志非常方便。

+   从文件中读取标准日志配置。

与创建复杂规则相比，尝试进行广泛的日志记录，但使用适当的级别，然后进行过滤。

有关详细信息，请查看模块的 Python 文档[`docs.python.org/3.7/library/logging.html`](https://docs.python.org/3.7/library/logging.html)，或者查看教程[`docs.python.org/3.7/howto/logging.html`](https://docs.python.org/3.7/howto/logging.html)。

# 另请参阅

+   在第一章的*添加命令行选项*中，*让我们开始自动化之旅*中的*添加命令行选项*。

+   *准备任务*配方

# 发送电子邮件通知

电子邮件已成为每个人每天都使用的不可避免的工具。如果自动化任务检测到某些情况，它可能是发送通知的最佳位置。另一方面，电子邮件收件箱已经充斥着垃圾邮件，所以要小心。

垃圾邮件过滤器也是现实。小心选择发送电子邮件的对象和发送的电子邮件数量。电子邮件服务器或地址可能被标记为*垃圾邮件*，所有电子邮件都将被互联网悄悄丢弃。本示例将展示如何使用已有的电子邮件帐户发送单个电子邮件。

这种方法适用于发送给几个人的备用电子邮件，作为自动化任务的结果，但不要超过这个数量。

# 准备就绪

对于本示例，我们需要设置一个有效的电子邮件帐户，其中包括以下内容：

+   有效的电子邮件服务器

+   连接的端口

+   一个地址

+   密码

这四个元素应该足以发送电子邮件。

例如，Gmail 等一些电子邮件服务将鼓励您设置 2FA，这意味着仅密码不足以发送电子邮件。通常，它们允许您为应用程序创建一个特定的密码来使用，绕过 2FA 请求。查看您的电子邮件提供商的信息以获取选项。

要使用的电子邮件提供商应指示 SMTP 服务器和端口在其文档中使用。它们也可以从电子邮件客户端中检索，因为它们是相同的参数。查看您的提供商文档。在以下示例中，我们将使用 Gmail 帐户。

# 如何做...

1.  创建`email_task.py`文件，如下所示：

```py
import argparse
import configparser

import smtplib 
from email.message import EmailMessage

def main(to_email, server, port, from_email, password):
    print(f'With love, from {from_email} to {to_email}')

    # Create the message
    subject = 'With love, from ME to YOU'
    text = '''This is an example test'''
    msg = EmailMessage()
    msg.set_content(text)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    # Open communication and send
    server = smtplib.SMTP_SSL(server, port)
    server.login(from_email, password)
    server.send_message(msg)
    server.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('email', type=str, help='destination email')
    parser.add_argument('-c', dest='config', type=argparse.FileType('r'),
                        help='config file', default=None)

    args = parser.parse_args()
    if not args.config:
        print('Error, a config file is required')
        parser.print_help()
        exit(1)

    config = configparser.ConfigParser()
    config.read_file(args.config)

    main(args.email,
         server=config['DEFAULT']['server'],
         port=config['DEFAULT']['port'],
         from_email=config['DEFAULT']['email'],
         password=config['DEFAULT']['password'])
```

1.  创建一个名为`email_conf.ini`的配置文件，其中包含您的电子邮件账户的具体信息。例如，对于 Gmail 账户，请填写以下模板。该模板可在 GitHub [`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter02/email_conf.ini`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter02/email_conf.ini) 中找到，但请确保用您的数据填写它：

```py
[DEFAULT]
email = EMAIL@gmail.com
server = smtp.gmail.com
port = 465
password = PASSWORD
```

1.  确保文件不能被系统上的其他用户读取或写入，设置文件的权限只允许我们的用户。`600`权限意味着我们的用户有读写权限，其他人没有访问权限：

```py
$ chmod 600 email_config.ini
```

1.  运行脚本发送测试邮件：

```py
$ python3 email_task.py -c email_config.ini destination_email@server.com
```

1.  检查目标电子邮件的收件箱；应该收到一封主题为`With love, from ME to YOU`的电子邮件。

# 它是如何工作的...

脚本中有两个关键步骤——消息的生成和发送。

消息主要需要包含`To`和`From`电子邮件地址，以及`Subject`。如果内容是纯文本，就像在这种情况下一样，调用`.set_content()`就足够了。然后可以发送整个消息。

从一个与发送邮件的账户不同的邮箱发送邮件在技术上是可能的。尽管如此，这是不被鼓励的，因为你的电子邮件提供商可能会认为你试图冒充另一个邮箱。您可以使用`reply-to`头部来允许回复到不同的账户。

发送邮件需要连接到指定的服务器并启动 SMPT 连接。SMPT 是电子邮件通信的标准。

步骤非常简单——配置服务器，登录，发送准备好的消息，然后退出。

如果您需要发送多条消息，可以登录，发送多封电子邮件，然后退出，而不是每次都连接。

# 还有更多...

如果目标是更大规模的操作，比如营销活动，或者生产邮件，比如确认用户的电子邮件，请查看第八章，*处理通信渠道*

本步骤中使用的电子邮件消息内容非常简单，但电子邮件可能比这更复杂。

`To`字段可以包含多个收件人。用逗号分隔它们，就像这样：

```py
message['To'] = ','.join(recipients)
```

电子邮件可以以 HTML 格式定义，并附有纯文本和附件。基本操作是设置一个`MIMEMultipart`，然后附加组成邮件的每个 MIME 部分：

```py
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage message = MIMEMultipart()
part1 = MIMEText('some text', 'plain')
message.attach(part1)
with open('path/image', 'rb') as image:
 part2 = MIMEImage(image.read()) message.attach(part2)
```

最常见的 SMPT 连接是`SMPT_SSL`，它更安全，需要登录和密码，但也存在普通的未经身份验证的 SMPT；请查看您的电子邮件提供商的文档。

请记住，这个步骤是为简单的通知而设计的。如果附加不同的信息，电子邮件可能会变得非常复杂。如果您的目标是为客户或任何一般群体发送电子邮件，请尝试使用第八章，*处理通信渠道*中的想法。

# 另请参阅

+   在第一章，*让我们开始自动化之旅*中的*添加命令行选项*步骤

+   准备任务的步骤
