# 第十一章。调试和故障排除

|   | *"如果调试是移除软件错误的过程，那么编程就必须是引入它们的过程。"* |   |
| --- | --- | --- |
|   | --*埃德加·W·迪杰斯特拉* |

在专业程序员的生涯中，调试和故障排除占据了相当多的时间。即使你从事的是人类写过的最漂亮的代码库，其中仍然会有错误，这是肯定的。

我们花了很多时间阅读别人的代码，在我看来，一个优秀的软件开发者即使在阅读的不是报告为错误或存在错误的代码时，也能保持高度的关注。

能够高效快速地调试代码是任何程序员都需要不断改进的技能。有些人认为，因为他们已经阅读了手册，所以他们没问题，但现实是，游戏中的变量数量如此之大，以至于没有手册。有一些可以遵循的指南，但没有一本魔法书能教你成为调试高手所需知道的一切。

我觉得，在这个特定的问题上，我从我的同事那里学到了最多的东西。看到一个非常熟练的人攻击问题让我感到惊讶。我喜欢看到他们采取的步骤，他们验证的事情，以排除可能的原因，以及他们考虑的嫌疑人，最终引导他们找到问题的解决方案。

我们与每一个同事合作都能学到一些东西，或者用他们那惊人的猜测来让我们感到惊讶，而这些猜测最终被证明是正确的。当这种情况发生时，不要只是感到惊奇（或者更糟，感到嫉妒），而要抓住这个机会，问他们是如何得出这个猜测的，为什么。这个答案将让你看到，你能否在以后深入研究，以便下次，你将成为那个抓住错误的人。

有些错误很容易发现。它们来自粗心大意的错误，一旦你看到这些错误的影响，就很容易找到解决问题的解决方案。

但还有其他一些错误更加微妙，更加难以捉摸，需要真正的专业知识，以及大量的创造性和跳出思维，才能解决。

对于我来说，最糟糕的是那些非确定性的问题。这些有时会发生，有时不会。有些问题只发生在环境 A 中，而不发生在环境 B 中，尽管 A 和 B 应该完全相同。这些错误是真正的邪恶，它们可以让你发疯。

当然，错误不仅仅发生在沙盒中，对吧？当你的老板告诉你“*别担心！慢慢来修复这个问题，先吃午饭吧!*”。不。它们发生在周五下午五点半，那时你的大脑已经疲惫，你只想回家。就在那一刻，当每个人都瞬间变得焦躁不安，当你的老板在你耳边喘息时，你必须能够保持冷静。我确实是这么说的。如果你让大脑紧张，那么就告别创造性思维、逻辑推理以及你当时需要的所有东西。所以深呼吸，坐好，集中注意力。

在这一章中，我将尝试展示一些有用的技巧，你可以根据错误的严重程度来使用它们，以及一些希望有助于增强你对错误和问题的武器的建议。

# 调试技巧

在这部分，我将向你展示最常用的技术，我最常用的技术，然而，请不要认为这个列表是详尽的。

## 使用`print`进行调试

这可能是所有技巧中最简单的一个。它并不非常有效，不能在所有地方使用，并且需要访问源代码以及运行它的终端（因此可以显示`print`函数调用的结果）。

然而，在许多情况下，这仍然是一种快速且有用的调试方法。例如，如果你正在开发一个 Django 网站，页面上的情况并不是你所期望的，你可以在视图中添加`print`语句，并在重新加载页面时留意控制台。我可能已经做过无数次了。

当你在代码中分散`print`调用时，你通常最终会陷入一个重复大量调试代码的情况，要么是因为你在打印时间戳（就像我们测量列表解析和生成器速度时做的那样），要么是因为你必须以某种方式构建一个你想要显示的字符串。

另一个问题是在你的代码中很容易忘记调用`print`。

因此，与其使用裸露的`print`调用，我有时更喜欢编写一个自定义函数。让我们看看怎么做。

## 使用自定义函数进行调试

在代码片段中有一个自定义函数，你可以快速抓取并粘贴到代码中，然后用来调试，这可以非常有用。如果你动作快，你总是可以即兴编写一个。重要的是要以一种方式编写代码，这样在你最终移除调用和定义时不会留下任何东西，因此*重要的是要以一种完全自包含的方式编写代码*。这个要求的好另一个原因是它可以避免与代码中其他部分的潜在名称冲突。

让我们看看这样一个函数的例子。

`custom.py`

```py
def debug(*msg, print_separator=True):
    print(*msg)
    if print_separator:
        print('-' * 40)

debug('Data is ...')
debug('Different', 'Strings', 'Are not a problem')
debug('After while loop', print_separator=False)
```

在这种情况下，我使用关键字参数来能够打印一个分隔符，即一串 40 个短横线。

这个函数非常简单，我只是将`msg`中的任何内容重定向到`print`的调用，如果`print_separator`为`True`，我打印一个行分隔符。运行代码将显示：

```py
$ python custom.py 
Data is ...
----------------------------------------
Different Strings Are not a problem
----------------------------------------
After while loop

```

如你所见，最后一行后面没有分隔符。

这只是增加简单调用`print`函数的一种简单方法。让我们看看我们如何利用 Python 的一个巧妙特性来计算调用之间的时间差。

`custom_timestamp.py`

```py
from time import sleep

def debug(*msg, timestamp=[None]):
    print(*msg)
    from time import time  # local import
    if timestamp[0] is None:
        timestamp[0] = time()  #1
    else:
        now = time()
        print(' Time elapsed: {:.3f}s'.format(
            now - timestamp[0]))
        timestamp[0] = now  #2

debug('Entering nasty piece of code...')
sleep(.3)
debug('First step done.')
sleep(.5)
debug('Second step done.')
```

这有点棘手，但仍然相当简单。首先注意我们是从`debug`函数中导入`time`模块的`time`函数。这允许我们避免在函数外部添加该导入，也许会忘记它。

看看我是如何定义`timestamp`的。它当然是一个列表，但这里重要的是它是一个*可变*对象。这意味着当 Python 解析函数时，它将被设置，并且在整个不同的调用过程中保持其值。因此，如果我们每次调用后都在其中放入一个时间戳，我们就可以在不使用外部全局变量的情况下跟踪时间。我从对**闭包**的研究中借用了这个技巧，这是一个我鼓励你阅读的非常有意思的技术。

对了，所以，在打印了我们必须打印的任何消息并导入时间后，我们检查`timestamp`中唯一项的内容。如果它是`None`，那么我们没有先前的引用，因此我们将值设置为当前时间（`#1`）。

另一方面，如果我们有一个先前的引用，我们可以计算一个差值（我们将其格式化为三位小数），然后我们最终再次在`timestamp`中放入当前时间（`#2`）。这是一个很好的技巧，不是吗？

运行此代码会显示此结果：

```py
$ python custom_timestamp.py 
Entering nasty piece of code...
First step done.
 Time elapsed: 0.300s
Second step done.
 Time elapsed: 0.501s

```

无论你的情况如何，拥有这样一个自包含的函数非常有用。

## 检查 traceback

我们在第七章中简要介绍了 traceback，*测试、分析和处理异常*，当我们看到几种不同类型的异常时。traceback 提供了关于你的应用程序中发生错误的信息。阅读它将给你很大的帮助。让我们看一个非常小的例子：

`traceback_simple.py`

```py
d = {'some': 'key'}
key = 'some-other'
print(d[key])
```

我们有一个字典，我们尝试访问其中不存在的键。你应该记住这将引发一个`KeyError`异常。让我们运行代码：

```py
$ python traceback_simple.py 
Traceback (most recent call last):
 File "traceback_simple.py", line 3, in <module>
 print(d[key])
KeyError: 'some-other'

```

你可以看到我们得到了我们需要的所有信息：模块名称、导致错误的行（编号和指令），以及错误本身。有了这些信息，你可以回到源代码并尝试理解出了什么问题。

现在我们来创建一个更有趣的例子，这个例子基于这个例子，并练习一个仅在 Python 3 中可用的功能。想象一下，我们正在验证一个字典，处理必填字段，因此我们期望它们存在。如果不存在，我们需要引发一个自定义的 `ValidationError`，我们将在运行验证器的流程中进一步捕获它（这里没有展示，它可以是任何东西）。它应该像这样：

`traceback_validator.py`

```py
class ValidatorError(Exception):
    """Raised when accessing a dict results in KeyError. """

d = {'some': 'key'}
mandatory_key = 'some-other'
try:
    print(d[mandatory_key])
except KeyError:
    raise ValidatorError(
        '`{}` not found in d.'.format(mandatory_key))
```

我们定义一个自定义异常，当必填键不存在时引发。请注意，其主体由其文档字符串组成，因此我们不需要添加任何其他语句。

非常简单，我们定义一个虚拟字典并尝试使用 `mandatory_key` 访问它。当发生这种情况时，我们捕获 `KeyError` 并引发 `ValidatorError`。这样做的原因是，我们可能还希望在其他情况下引发 `ValidatorError`，而不仅仅是由于缺少必填键。这种技术允许我们在简单的 `try`/`except` 中运行验证，它只关心 `ValidatorError`。

问题是，在 Python 2 中，这段代码只会显示最后一个异常（`ValidatorError`），这意味着我们会失去关于其前面的 `KeyError` 的信息。在 Python 3 中，这种行为已经改变，异常现在是链式的，因此当发生问题时，你会得到更好的信息报告。代码产生以下结果：

```py
$ python traceback_validator.py 
Traceback (most recent call last):
 File "traceback_validator.py", line 7, in <module>
 print(d[mandatory_key])
KeyError: 'some-other'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
 File "traceback_validator.py", line 10, in <module>
 '`{}` not found in d.'.format(mandatory_key))
__main__.ValidatorError: `some-other` not found in d.

```

这很棒，因为我们可以看到导致我们引发 `ValidationError` 的异常的堆栈跟踪，以及 `ValidationError` 本身的堆栈跟踪。

我和我的一个审稿人关于从 `pip` 安装程序获得的堆栈跟踪进行了愉快的讨论。他在设置一切以便审查 第九章 的代码时遇到了麻烦，*数据科学*。他的新 Ubuntu 安装缺少一些库，这些库是 `pip` 包正常运行所必需的。

他受阻的原因是他试图从堆栈跟踪的顶部开始修复显示的错误。我建议他从底部开始修复。原因是，如果安装程序已经到达最后一行，我猜在那之前，无论发生什么错误，都有可能恢复。只有当 `pip` 决定无法继续进行时，才会到达最后一行，因此我开始修复那一行。一旦安装了修复该错误所需的库，其他一切都会顺利。

阅读堆栈跟踪可能很棘手，我的朋友缺乏处理这个问题的必要经验，因此，如果你陷入同样的情况，不要气馁，试着稍微改变一下，不要想当然。

Python 拥有一个庞大而精彩的社区，当你遇到问题时，你不太可能是第一个看到它的人，所以打开浏览器并搜索。通过这样做，你的搜索技巧也会提高，因为你必须将错误缩减到最小但最关键的细节集，这将使你的搜索变得有效。

如果你想要更好地玩耍和理解回溯，在标准库中有一个名为 `traceback` 的模块，你可以使用它。它提供了一个标准的接口来提取、格式化和打印 Python 程序的堆栈跟踪，这与 Python 解释器打印堆栈跟踪时的行为完全一致。

## 使用 Python 调试器

另一种非常有效的调试 Python 的方法是使用 Python 调试器：**pdb**。如果你像我一样沉迷于 IPython 控制台，那么你绝对应该检查一下 **ipdb** 库。*ipdb* 增强了标准的 *pdb* 接口，就像 IPython 对 Python 控制台所做的那样。

使用这个调试器的几种不同方法（无论哪个版本，这并不重要），但最常见的一种方法就是简单地设置一个断点并运行代码。当 Python 达到断点时，执行会暂停，你将获得对该点的控制台访问权限，以便你可以检查所有名称等。你还可以动态地更改数据以改变程序的流程。

作为一个小例子，让我们假装我们有一个解析器，它因为字典中缺少键而引发 `KeyError`。这个字典来自我们无法控制的 JSON 有效载荷，而我们目前只想暂时欺骗并传递这个控制权，因为我们对之后发生的事情感兴趣。让我们看看我们如何能够拦截这个时刻，检查数据，修复它，并使用 *ipdb* 到达底部。

`ipdebugger.py`

```py
# d comes from a JSON payload we don't control
d = {'first': 'v1', 'second': 'v2', 'fourth': 'v4'}
# keys also comes from a JSON payload we don't control
keys = ('first', 'second', 'third', 'fourth')

def do_something_with_value(value):
    print(value)

for key in keys:
    do_something_with_value(d[key])

print('Validation done.')
```

正如你所见，当 `key` 获取到字典中缺失的值 `'third'` 时，这段代码将会中断。记住，我们假装 `d` 和 `keys` 都是从我们无法控制的 JSON 有效载荷中动态获取的，因此我们需要检查它们以修复 `d` 并通过 `for` 循环。如果我们按原样运行代码，我们会得到以下结果：

```py
$ python ipdebugger.py 
v1
v2
Traceback (most recent call last):
 File "ipdebugger.py", line 10, in <module>
 do_something_with_value(d[key])
KeyError: 'third'

```

因此我们看到 `key` 在字典中缺失，但由于每次我们运行这段代码时我们可能得到不同的字典或 `keys` 元组，这个信息实际上并没有真正帮助我们。让我们注入一个对 *ipdb* 的调用。

`ipdebugger_ipdb.py`

```py
# d comes from a JSON payload we don't control
d = {'first': 'v1', 'second': 'v2', 'fourth': 'v4'}
# keys also comes from a JSON payload we don't control
keys = ('first', 'second', 'third', 'fourth')

def do_something_with_value(value):
    print(value)

import ipdb
ipdb.set_trace()  # we place a breakpoint here

for key in keys:
    do_something_with_value(d[key])

print('Validation done.')
```

如果我们现在运行这段代码，事情会变得有趣（请注意，你的输出可能略有不同，并且这个输出中的所有注释都是我添加的）：

```py
$ python ipdebugger_ipdb.py
> /home/fab/srv/l.p/ch11/ipdebugger_ipdb.py(12)<module>()
 11 
---> 12 for key in keys:  # this is where the breakpoint comes
 13     do_something_with_value(d[key])

ipdb> keys  # let's inspect the keys tuple
('first', 'second', 'third', 'fourth')
ipdb> !d.keys()  # now the keys of d
dict_keys(['first', 'fourth', 'second'])  # we miss 'third'
ipdb> !d['third'] = 'something dark side...'  # let's put it in
ipdb> c  # ... and continue
v1
v2
something dark side...
v4
Validation done.

```

这非常有趣。首先，请注意，当你达到断点时，你会得到一个控制台，告诉你你在哪里（Python 模块）以及下一行要执行的哪一行。此时，你可以执行一系列探索性操作，例如检查下一行之前和之后的代码，打印堆栈跟踪，与对象交互等。请查阅官方 Python 文档中的 *pdb* 了解更多。在我们的例子中，我们首先检查 `keys` 元组。之后，我们检查 `d` 的键。

你注意到我在 `d` 前面加上的感叹号了吗？这是必需的，因为 `d` 是 *pdb* 接口中的一个命令，用于将帧 (*d*)own 移动。

### 注意

我在 *ipdb* 壳中使用这种符号表示命令：每个命令通过一个字母激活，通常是命令名称的第一个字母。所以，*d* 对应 *down*，*n* 对应 *next*，*s* 对应 *step*，更简洁地表示为 (*d*)own，(*n*)ext 和 (*s*)tep。

我想这足够成为有一个更好的名字的理由，对吧？确实如此，但我需要向你展示这一点，所以我选择了使用 `d`。为了告诉 *pdb* 我们不是要执行 (*d*)own 命令，我们在 `d` 前面加上 "`!`"，这样我们就没问题了。

在查看 `d` 的键之后，我们发现 `'third'` 缺失，所以我们自己添加了它（这可能会很危险？想想看）。最后，现在所有的键都已经添加完毕，我们输入 `c`，这意味着 (*c*)ontinue。

*pdb* 还能让你使用 (*n*)ext 逐行执行代码，(*s*)tep 进入函数进行深入分析，或使用 (*b*)reak 处理中断。有关命令的完整列表，请参阅文档或在控制台中输入 (*h*)elp。

从输出中你可以看到，我们最终到达了验证的末尾。

*pdb*（或 *ipdb*）是我每天都会使用的无价工具，没有它们我无法生活。所以，去享受吧，在某个地方设置一个断点，尝试检查，遵循官方文档，并在你的代码中尝试命令以查看它们的效果并熟练掌握它们。

## 检查日志文件

另一种调试表现不佳的应用程序的方法是检查其日志文件。**日志文件**是特殊文件，其中应用程序记录了各种信息，通常与它内部发生的事情有关。如果启动了重要的程序，我通常会期望在日志中有一条记录。当它完成时，以及可能在其内部发生的事情，情况也是如此。

需要将错误记录下来，这样当出现问题的时候，我们可以通过查看日志文件中的信息来检查发生了什么。

在 Python 中设置记录器有许多不同的方法。日志记录非常灵活，你可以配置它。简而言之，游戏中通常有四个参与者：记录器、处理器、过滤器、和格式化器：

+   **记录器**暴露了应用程序代码直接使用的接口

+   **处理器**将日志记录（由记录器创建）发送到适当的目的地

+   **过滤器**提供了更细粒度的功能，用于确定要输出的日志记录

+   **格式化器**指定最终输出中日志记录的布局

日志是通过调用`Logger`类实例的方法来执行的。您记录的每一行都有一个级别。通常使用的级别有：`DEBUG`、`INFO`、`WARNING`、`ERROR`和`CRITICAL`。您可以从`logging`模块导入它们。它们按照严重性顺序排列，并且正确使用它们非常重要，因为它们将帮助您根据您正在搜索的内容过滤日志文件的内容。日志文件通常变得非常大，因此确保其中的信息被正确写入非常重要，这样在需要时您可以快速找到它们。

您可以将日志记录到文件中，但您也可以将日志记录到网络位置、队列、控制台等。一般来说，如果您有一个部署在单个机器上的架构，将日志记录到文件是可以接受的，但当您的架构跨越多个机器（例如在**面向服务的架构**的情况下），实现一个集中式的日志解决方案非常有用，这样所有来自每个服务的日志消息都可以存储和调查在一个地方。这非常有帮助，否则您真的会疯狂地尝试关联来自几个不同来源的巨大文件，以找出出了什么问题。

### 注意

**面向服务的架构**（**SOA**）是软件设计中的一个架构模式，其中应用程序组件通过通信协议（通常是网络）向其他组件提供服务。这个系统的美妙之处在于，当代码编写得当，每个服务都可以用最合适的语言来编写以实现其目的。唯一重要的是与其他服务的通信，这需要通过一个公共格式来实现，以便进行数据交换。

在这里，我将向您展示一个非常简单的日志示例。我们将记录几条消息到一个文件中：

`log.py`

```py
import logging

logging.basicConfig(
    filename='ch11.log',
    level=logging.DEBUG,  # minimum level capture in the file
    format='[%(asctime)s] %(levelname)s:%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')

mylist = [1, 2, 3]
logging.info('Starting to process `mylist`...')

for position in range(4):
    try:
        logging.debug('Value at position {} is {}'.format(
            position, mylist[position]))
    except IndexError:
        logging.exception('Faulty position: {}'.format(position))

logging.info('Done parsing `mylist`.')
```

我们一行一行地过一遍。首先，我们导入`logging`模块，然后设置基本配置。一般来说，生产环境的日志配置比这要复杂得多，但我希望尽可能保持简单。我们指定一个文件名，我们希望在文件中捕获的最小日志级别，以及消息格式。我们将记录日期和时间信息，日志级别和消息内容。

我将首先记录一个`info`消息，告诉我我们即将处理我们的列表。然后，我将记录（这次使用`DEBUG`级别，通过使用`debug`函数）某个位置上的值。我在这里使用`debug`是因为我希望将来能够过滤掉这些日志（通过将最小级别设置为`logging.INFO`或更高），因为我可能需要处理非常大的列表，并且我不想记录所有值。

如果我们得到一个`IndexError`（我们确实得到了，因为我正在遍历`range(4)`），我们调用`logging.exception()`，它与`logging.error()`相同，但它也会打印跟踪信息。

在代码的末尾，我记录了一条`info`消息，说明我们已经完成。结果是这个：

```py
[10/08/2015 04:17:06 PM] INFO:Starting to process `mylist`...
[10/08/2015 04:17:06 PM] DEBUG:Value at position 0 is 1
[10/08/2015 04:17:06 PM] DEBUG:Value at position 1 is 2
[10/08/2015 04:17:06 PM] DEBUG:Value at position 2 is 3
[10/08/2015 04:17:06 PM] ERROR:Faulty position: 3
Traceback (most recent call last):
 File "log.py", line 15, in <module>
 position, mylist[position]))
IndexError: list index out of range
[10/08/2015 04:17:06 PM] INFO:Done parsing `mylist`.

```

这正是我们能够调试在机器上运行的应用程序而不是在控制台上的应用程序所需要的。我们可以看到发生了什么，任何抛出的异常的跟踪信息，等等。

### 注意

这里提供的示例只是对日志记录的表面了解。对于更深入的解释，你可以在官方 Python 文档的如何（[`docs.python.org/3.4/howto/logging.html`](https://docs.python.org/3.4/howto/logging.html)）部分找到一个非常好的介绍。

日志记录是一种艺术，你需要在记录一切和记录什么都不记录之间找到一个好的平衡。理想情况下，你应该记录任何你需要确保应用程序正确运行的信息，以及可能的所有错误或异常。

## 其他技术

在本节的最后，我想简要演示一些你可能觉得有用的技术。

### 性能分析

我们在第七章中讨论了性能分析，*测试、性能分析和处理异常*，我在这里只是提到它，因为性能分析有时可以解释由于组件运行太慢而导致的奇怪错误。特别是当涉及到网络时，了解你的应用程序必须经历的时间和时间延迟对于理解问题出现时可能发生的情况非常重要，因此我建议你也从故障排除的角度熟悉性能分析技术。

### 断言

断言是确保你的假设得到验证的好方法。如果它们是，一切都会按常规进行，但如果它们不是，你会得到一个很好的异常，你可以处理它。有时，与其检查，不如在代码中添加几个断言来排除可能性。让我们看看一个例子：

`assertions.py`

```py
mylist = [1, 2, 3]  # this ideally comes from some place
assert 4 == len(mylist)  # this will break
for position in range(4):
    print(mylist[position])
```

这段代码模拟了一个情况，其中`mylist`当然不是由我们这样定义的，但我们假设它有四个元素。所以我们那里放了一个断言，结果是这个：

```py
$ python assertions.py 
Traceback (most recent call last):
 File "assertions.py", line 3, in <module>
 assert 4 == len(mylist)
AssertionError

```

这告诉我们问题确实在哪里。

## 去哪里寻找信息

在 Python 官方文档中，有一个专门介绍调试和性能分析的章节，你可以阅读有关`bdb`调试框架以及`faulthandler`、`timeit`、`trace`、`tracemalloc`和当然还有*pdb*等模块的内容。只需前往文档中的标准库部分，你就可以很容易地找到所有这些信息。

# 故障排除指南

在这个简短的章节中，我想分享一些来自我的故障排除经验的技巧。

## 使用控制台编辑器

首先，熟悉使用**vim**或**nano**作为编辑器，并学习控制台的基本知识。当事情变得糟糕时，你没有使用带有所有铃声和哨声的编辑器的奢侈。你必须连接到一个盒子并在那里工作。所以，熟悉使用控制台命令浏览你的生产环境，并能够使用基于控制台的编辑器（如 vi、vim 或 nano）编辑文件是非常好的主意。不要让你的常规开发环境宠坏你，因为如果你这样做，你将不得不付出代价。

## 检查的位置

我的第二个建议是关于在哪里放置调试断点。无论你是在使用`print`、自定义函数还是*ipdb*，你仍然需要选择放置提供信息的调用位置，对吧？

好吧，有些地方比其他地方更好，而且有一些处理调试进度的方法比其他方法更好。

我通常避免在`if`子句中放置断点，因为如果这个子句没有被执行，我就失去了获取所需信息的机会。有时候到达断点并不容易或快速，所以在放置它们之前要仔细思考。

另一个重要的事情是确定从哪里开始。想象一下，你有 100 行代码来处理你的数据。数据从第 1 行开始，不知为何在第 100 行出现了错误。你不知道错误在哪里，所以你该怎么办？你可以在第 1 行设置一个断点，并耐心地逐行检查你的数据。在最坏的情况下，99 行之后（以及许多咖啡杯）你才发现了错误。所以，考虑使用不同的方法。

你从第 50 行开始检查。如果数据是好的，这意味着错误发生在后面，在这种情况下，你将在第 75 行放置下一个断点。如果第 50 行的数据已经不好，你将继续通过在第 25 行放置断点。然后，你重复这个过程。每次，你要么向后移动，要么向前移动，移动的距离是上一次跳跃距离的一半。

在我们最坏的情况下，你的调试将从 1, 2, 3, ..., 99 变为 50, 75, 87, 93, 96, ..., 99，这要快得多。事实上，这是对数级的。这种搜索技术被称为**二分搜索**，它基于分而治之的方法，并且非常有效，所以尽量掌握它。

## 使用测试进行调试

你还记得第七章测试、性能分析和处理异常，关于测试的内容吗？嗯，如果我们有一个错误，并且所有测试都通过了，这意味着我们的测试代码库中存在问题或遗漏。所以，一种方法是对测试进行修改，使其能够应对新发现的边缘情况，然后逐步检查代码。这种方法非常有用，因为它确保了当你修复错误时，你的错误将被测试覆盖。

## 监控

监控也非常重要。当软件应用遇到边缘情况，如网络中断、队列满、外部组件无响应等情况时，它们可能会完全失控并出现非确定性的故障。在这些情况下，了解问题发生时的大致情况，并能以微妙、甚至神秘的方式将其与相关事物联系起来，这一点非常重要。

你可以监控 API 端点、进程、网页可用性和加载时间，以及基本上你可以编码的几乎所有东西。一般来说，当你从头开始构建一个应用程序时，考虑你想要如何监控它是非常有用的。

# 摘要

在这一简短的章节中，我们看到了调试和排查代码问题的不同技术和建议。调试是软件开发者工作中始终不可或缺的一部分，因此掌握它非常重要。

如果以正确的心态去对待，这可以是一件有趣且有益的事情。

我们看到了如何通过函数、日志、调试器、跟踪信息、性能分析和断言来检查我们的代码库。我们看到了其中大多数技术的简单示例，并且还讨论了一套有助于应对挑战的指导原则。

只需**记住始终保持冷静和专注**，调试就会变得更容易。这也是一种需要学习并认为最重要的技能。一个焦虑和紧张的大脑无法正常、逻辑和创造性地工作，因此，如果你不加强它，将很难将你的所有知识有效利用。

在下一章中，我们将以另一个小型项目结束本书，其目标是让你在跟随我的旅程开始时更加渴望。

准备好了吗？
