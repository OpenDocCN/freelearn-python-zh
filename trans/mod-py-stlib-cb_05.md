# 第五章：日期和时间

在本章中，我们将涵盖以下技巧：

+   时区感知的 datetime-检索当前 datetime 的可靠值

+   解析日期-如何根据 ISO 8601 格式解析日期

+   保存日期-如何存储 datetime

+   从时间戳到 datetime-转换为时间戳和从时间戳转换为 datetime

+   以用户格式显示日期-根据用户语言格式化日期

+   去明天-如何计算指向明天的 datetime

+   去下个月-如何计算指向下个月的 datetime

+   工作日-如何构建一个指向本月第*n*个星期一/星期五的日期

+   工作日-如何在时间范围内获取工作日

+   组合日期和时间-从日期和时间创建一个 datetime

# 介绍

日期是我们生活的一部分，我们习惯于处理时间和日期作为一个基本的过程。即使是一个小孩也知道现在是什么时间或者*明天*是什么意思。但是，试着和世界另一端的人交谈，突然之间*明天*、*午夜*等概念开始变得非常复杂。

当你说明天时，你是在说你的明天还是我的明天？如果你安排一个应该在午夜运行的进程，那么是哪一个午夜？

为了让一切变得更加困难，我们有闰秒，奇怪的时区，夏令时等等。当你尝试在软件中处理日期时，特别是在可能被世界各地的人使用的软件中，突然之间就会明白日期是一个复杂的事务。

本章包括一些短小的技巧，可以在处理用户提供的日期时避免头痛和错误。

# 时区感知的 datetime

Python datetimes 通常是*naive*，这意味着它们不知道它们所指的时区。这可能是一个主要问题，因为给定一个 datetime，我们无法知道它实际指的是什么时候。

在 Python 中处理日期最常见的错误是尝试通过`datetime.datetime.now()`获取当前 datetime，因为所有`datetime`方法都使用 naive 日期，所以无法知道该值代表的时间。

# 如何做到这一点...

执行以下步骤来完成这个技巧：

1.  检索当前 datetime 的唯一可靠方法是使用`datetime.datetime.utcnow()`。无论用户在哪里，系统如何配置，它都将始终返回 UTC 时间。因此，我们需要使其具有时区感知能力，以便能够将其拒绝到世界上的任何时区：

```py
import datetime

def now():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
```

1.  一旦我们有了一个具有时区感知能力的当前时间，就可以将其转换为任何其他时区，这样我们就可以向我们的用户显示他们自己时区的值：

```py
def astimezone(d, offset):
    return d.astimezone(datetime.timezone(datetime.timedelta(hours=offset)))
```

1.  现在，假设我目前在 UTC+01:00 时区，我可以获取 UTC 的具有时区感知能力的当前时间，然后在我的时区中显示它：

```py
>>> d = now()
>>> print(d)
2018-03-19 21:35:43.251685+00:00

>>> d = astimezone(d, 1)
>>> print(d)
2018-03-19 22:35:43.251685+01:00
```

# 它是如何工作的...

所有 Python datetimes，默认情况下都没有指定任何时区，但通过设置`tzinfo`，我们可以使它们意识到它们所指的时区。

如果我们只是获取当前时间（`datetime.datetime.now()`），我们无法轻松地从软件内部知道我们正在获取时间的时区。因此，我们唯一可以依赖的时区是 UTC。每当检索当前时间时，最好始终依赖于`datetime.datetime.utcnow()`。

一旦我们有了 UTC 的日期，因为我们知道它实际上是 UTC 时区的日期，我们可以轻松地附加`datetime.timezone.utc`时区（Python 提供的唯一时区）并使其具有时区感知能力。

`now`函数可以做到这一点：它获取 datetime 并使其具有时区感知能力。

由于我们的 datetime 现在具有时区感知能力，从那一刻起，我们可以依赖于`datetime.datetime.astimezone`方法将其转换为任何我们想要的时区。因此，如果我们知道我们的用户在 UTC+01:00，我们可以显示 datetime 的用户本地值，而不是显示 UTC 值。

这正是`astimezone`函数所做的。一旦提供了日期时间和与 UTC 的偏移量，它将返回一个日期，该日期是基于该偏移量的本地时区。

# 还有更多...

您可能已经注意到，虽然这个解决方案有效，但缺乏更高级的功能。例如，我目前在 UTC+01:00，但根据我的国家的夏令时政策，我可能在 UTC+02:00。此外，我们只支持基于整数小时的偏移量，虽然这是最常见的情况，但有一些时区，如印度或伊朗，有半小时的偏移量。

虽然我们可以扩展我们对时区的支持以包括这些奇怪的情况，但对于更复杂的情况，您可能应该依赖于`pytz`软件包，该软件包为完整的 IANA 时区数据库提供了时区。

# 解析日期

从另一个软件或用户那里接收日期时间时，它可能是以字符串格式。例如 JSON 等格式甚至不定义日期应该如何表示，但通常最好的做法是以 ISO 8601 格式提供这些日期。

ISO 8601 格式通常定义为`[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss]+-[TZ]`，例如`2018-03-19T22:00+0100`将指的是 UTC+01:00 时区的 3 月 19 日晚上 10 点。

ISO 8601 传达了表示日期和时间所需的所有信息，因此这是一种将日期时间编组并通过网络发送的好方法。

遗憾的是，它有许多奇怪之处（例如，`+00`时区也可以写为`Z`，或者您可以省略小时、分钟和秒之间的`:`），因此解析它有时可能会引起麻烦。

# 如何做...

以下是要遵循的步骤：

1.  由于 ISO 8601 允许所有这些变体，没有简单的方法将其传递给`datetime.datetime.strptime`，并为所有情况返回一个日期时间；我们必须将所有可能的格式合并为一个格式，然后解析该格式：

```py
import datetime

def parse_iso8601(strdate):
    date, time = strdate.split('T', 1)
    if '-' in time:
        time, tz = time.split('-')
        tz = '-' + tz
    elif '+' in time:
        time, tz = time.split('+')
        tz = '+' + tz
    elif 'Z' in time:
        time = time[:-1]
        tz = '+0000'
    date = date.replace('-', '')
    time = time.replace(':', '')
    tz = tz.replace(':', '')
    return datetime.datetime.strptime('{}T{}{}'.format(date, time, tz), 
                                      "%Y%m%dT%H%M%S%z")
```

1.  `parse_iso8601`的先前实现处理了大多数可能的 ISO 8601 表示：

```py
>>> parse_iso8601('2018-03-19T22:00Z')
datetime.datetime(2018, 3, 19, 22, 0, tzinfo=datetime.timezone.utc)
>>> parse_iso8601('2018-03-19T2200Z')
datetime.datetime(2018, 3, 19, 22, 0, tzinfo=datetime.timezone.utc)
>>> parse_iso8601('2018-03-19T22:00:03Z')
datetime.datetime(2018, 3, 19, 22, 0, 3, tzinfo=datetime.timezone.utc)
>>> parse_iso8601('20180319T22:00:03Z')
datetime.datetime(2018, 3, 19, 22, 0, 3, tzinfo=datetime.timezone.utc)
>>> parse_iso8601('20180319T22:00:03+05:00')
datetime.datetime(2018, 3, 19, 22, 0, 3, tzinfo=datetime.timezone(datetime.timedelta(0, 18000)))
>>> parse_iso8601('20180319T22:00:03+0500')
datetime.datetime(2018, 3, 19, 22, 0, 3, tzinfo=datetime.timezone(datetime.timedelta(0, 18000)))
```

# 它是如何工作的...

`parse_iso8601`的基本思想是，无论在解析之前收到 ISO 8601 的方言是什么，我们都将其转换为`[YYYY][MM][DD]T[hh][mm][ss]+-[TZ]`的形式。

最难的部分是检测时区，因为它可以由`+`、`-`分隔，甚至可以是`Z`。一旦提取了时区，我们可以摆脱日期中所有`-`的示例和时间中所有`:`的实例。

请注意，在提取时区之前，我们将时间与日期分开，因为日期和时区都可能包含`-`字符，我们不希望我们的解析器感到困惑。

# 还有更多...

解析日期可能变得非常复杂。虽然我们的`parse_iso8601`在与大多数以字符串格式提供日期的系统（如 JSON）交互时可以工作，但您很快就会面临它因日期时间可以表示的所有方式而不足的情况。

例如，我们可能会收到一个值，例如`2 周前`或`2013 年 7 月 4 日 PST`。尝试解析所有这些情况并不是很方便，而且可能很快变得复杂。如果您必须处理这些特殊情况，您可能应该依赖于外部软件包，如`dateparser`、`dateutil`或`moment`。

# 保存日期

迟早，我们都必须在某个地方保存一个日期，将其发送到数据库或将其保存到文件中。也许我们将其转换为 JSON 以将其发送到另一个软件。

许多数据库系统不跟踪时区。其中一些具有配置选项，指定它们应该使用的时区，但在大多数情况下，您提供的日期将按原样保存。

这会导致许多情况下出现意外的错误或行为。假设您是一个好童子军，并且正确地完成了接收保留其时区的日期时间所需的所有工作。现在您有一个`2018-01-15 15:30:00 UTC+01:00`的日期时间，一旦将其存储在数据库中，`UTC+01:00`将很容易丢失，即使您自己将其存储在文件中，存储和恢复时区通常是一项麻烦的工作。

因此，您应该始终确保在将日期时间存储在某个地方之前将其转换为 UTC，这将始终保证，无论日期时间来自哪个时区，当您将其加载回来时，它将始终表示正确的时间。

# 如何做到...

此食谱的步骤如下：

1.  要保存日期时间，我们希望有一个函数，确保日期时间在实际存储之前始终指的是 UTC：

```py
import datetime

def asutc(d):
    return d.astimezone(datetime.timezone.utc)
```

1.  `asutc`函数可用于任何日期时间，以确保在实际存储之前将其移至 UTC：

```py
>>> now = datetime.datetime.now().replace(
...    tzinfo=datetime.timezone(datetime.timedelta(hours=1))
... )
>>> now
datetime.datetime(2018, 3, 22, 0, 49, 45, 198483, 
                  tzinfo=datetime.timezone(datetime.timedelta(0, 3600)))
>>> asutc(now)
datetime.datetime(2018, 3, 21, 23, 49, 49, 742126, tzinfo=datetime.timezone.utc)
```

# 它是如何工作的...

此食谱的功能非常简单，通过`datetime.datetime.astimezone`方法，日期始终转换为其 UTC 表示。

这确保它将适用于存储跟踪时区的地方（因为日期仍将是时区感知的，但时区将是 UTC），以及当存储不保留时区时（因为没有时区的 UTC 日期仍然表示与零增量相同的 UTC 日期）。

# 从时间戳到日期时间

时间戳是从特定时刻开始的秒数的表示。通常，由于计算机可以表示的值在大小上是有限的，通常从 1970 年 1 月 1 日开始。

如果您收到一个值，例如`1521588268`作为日期时间表示，您可能想知道如何将其转换为实际的日期时间。

# 如何做到...

最近的 Python 版本引入了一种快速将日期时间与时间戳相互转换的方法：

```py
>>> import datetime
>>> ts = 1521588268

>>> d = datetime.datetime.utcfromtimestamp(ts)
>>> print(repr(d))
datetime.datetime(2018, 3, 20, 23, 24, 28)

>>> newts = d.timestamp()
>>> print(newts)
1521584668.0
```

# 还有更多...

正如食谱介绍中指出的，计算机可以表示的数字有一个限制。因此，重要的是要注意，虽然`datetime.datetime`可以表示几乎任何日期，但时间戳却不能。

例如，尝试表示来自`1300`的日期时间将成功，但将无法将其转换为时间戳：

```py
>>> datetime.datetime(1300, 1, 1)
datetime.datetime(1300, 1, 1, 0, 0)
>>> datetime.datetime(1300, 1, 1).timestamp()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
OverflowError: timestamp out of range
```

时间戳只能表示从 1970 年 1 月 1 日开始的日期。

对于遥远的日期，反向方向也是如此，而`253402214400`表示 9999 年 12 月 31 日的时间戳，尝试从该值之后的日期创建日期时间将失败：

```py
>>> datetime.datetime.utcfromtimestamp(253402214400)
datetime.datetime(9999, 12, 31, 0, 0)
>>> datetime.datetime.utcfromtimestamp(253402214400+(3600*24))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: year is out of range
```

日期时间只能表示从公元 1 年到 9999 年的日期。

# 以用户格式显示日期

在软件中显示日期时，如果用户不知道您将依赖的格式，很容易使用户感到困惑。

我们已经知道时区起着重要作用，并且在显示时间时我们总是希望将其显示为时区感知，但是日期也可能存在歧义。如果您写 3/4/2018，它是 4 月 3 日还是 3 月 4 日？

因此，通常有两种选择：

+   采用国际格式（2018-04-03）

+   本地化日期（2018 年 4 月 3 日）

可能的话，最好能够本地化日期格式，这样我们的用户将看到一个他们可以轻松识别的值。

# 如何做到...

此食谱需要以下步骤：

1.  Python 标准库中的`locale`模块提供了一种获取系统支持的本地化格式的方法。通过使用它，我们可以以目标系统允许的任何方式格式化日期：

```py
import locale
import contextlib

@contextlib.contextmanager
def switchlocale(name):
    prev = locale.getlocale()
    locale.setlocale(locale.LC_ALL, name)
    yield
    locale.setlocale(locale.LC_ALL, prev)

def format_date(loc, d):
    with switchlocale(loc):
        fmt = locale.nl_langinfo(locale.D_T_FMT)
        return d.strftime(fmt)
```

1.  调用`format_date`将正确地给出预期`locale`模块中日期的字符串表示：

```py
>>> format_date('de_DE', datetime.datetime.utcnow())
'Mi 21 Mär 00:08:59 2018'
>>> format_date('en_GB', datetime.datetime.utcnow())
'Wed 21 Mar 00:09:11 2018'
```

# 它是如何工作的...

`format_date`函数分为两个主要部分。

第一个由`switchlocale`上下文管理器提供，它负责启用请求的`locale`（locale 是整个进程范围的），并在包装的代码块中返回控制，然后恢复原始`locale`。这样，我们可以仅在上下文管理器中使用请求的`locale`，而不影响软件的任何其他部分。

第二个是上下文管理器内部发生的事情。使用`locale.nl_langinfo`，请求当前启用的`locale`的日期时间格式字符串（`locale.D_T_FMT`）。这会返回一个字符串，告诉我们如何在当前活动的`locale`中格式化日期时间。返回的字符串将类似于`'%a %e %b %X %Y'`。

然后日期本身根据通过`datetime.strftime`检索到的格式字符串进行格式化。

请注意，返回的字符串通常会包含`%a`和`%b`格式化符号，它们代表*当前星期*和*当前月份*的名称。由于星期几或月份的名称对每种语言都是不同的，Python 解释器将以当前启用的`locale`发出星期几或月份的名称。

因此，我们不仅按照用户的期望格式化了日期，而且结果输出也将是用户的语言。

# 还有更多...

虽然这个解决方案看起来非常方便，但重要的是要注意它依赖于动态切换`locale`。

切换`locale`是一个非常昂贵的操作，所以如果你有很多值需要格式化（比如`for`循环或成千上万的日期），这可能会太慢。

另外，切换`locale`也不是线程安全的，所以除非所有的`locale`切换发生在其他线程启动之前，否则你将无法在多线程软件中应用这个食谱。

如果你想以一种健壮且线程安全的方式处理本地化，你可能想要检查 babel 包。Babel 支持日期和数字的本地化，并且以一种不需要设置全局状态的方式工作，因此即使在多线程环境中也能正确运行。

# 前往明天

当你有一个日期时，通常需要对该日期进行数学运算。例如，也许你想要移动到明天或昨天。

日期时间支持数学运算，比如对它们进行加减，但涉及时间时，很难得到你需要添加或减去的确切秒数，以便移动到下一天或前一天。

因此，这个食谱将展示一种从任意给定日期轻松移动到下一个或上一个日期的方法。

# 如何做...

对于这个食谱，以下是步骤：

1.  `shiftdate`函数将允许我们按任意天数移动到一个日期：

```py
import datetime

def shiftdate(d, days):
    return (
        d.replace(hour=0, minute=0, second=0, microsecond=0) + 
        datetime.timedelta(days=days)
    )
```

1.  使用它就像简单地提供你想要添加或移除的天数一样简单：

```py
>>> now = datetime.datetime.utcnow()
>>> now
datetime.datetime(2018, 3, 21, 21, 55, 5, 699400)
```

1.  我们可以用它去到明天：

```py
>>> shiftdate(now, 1)
datetime.datetime(2018, 3, 22, 0, 0)
```

1.  或者前往昨天：

```py
>>> shiftdate(now, -1)
datetime.datetime(2018, 3, 20, 0, 0)
```

1.  甚至前往下个月：

```py
>>> shiftdate(now, 11)
datetime.datetime(2018, 4, 1, 0, 0)
```

# 它是如何工作的...

通常在移动日期时间时，我们想要去到一天的开始。假设你想要在事件列表中找到明天发生的所有事件，你真的想要搜索`day_after_tomorrow > event_time >= tomorrow`，因为你想要找到从明天午夜开始到后天午夜结束的所有事件。

因此，简单地改变日期本身是行不通的，因为我们的日期时间也与时间相关联。如果我们只是在日期上加一天，实际上我们会在明天包含的小时范围内结束。

这就是为什么`shiftdate`函数总是用午夜替换提供的日期时间的原因。

一旦日期被移动到午夜，我们只需添加一个等于指定天数的`timedelta`。如果这个数字是负数，我们将会向后移动时间，因为`D + -1 == D -1`。

# 前往下个月

在移动日期时，另一个经常需要的需求是能够将日期移动到下个月或上个月。

如果你阅读了*前往明天*的食谱，你会发现与这个食谱有很多相似之处，尽管在处理月份时需要一些额外的变化，而在处理天数时是不需要的，因为月份的持续时间是可变的。

# 如何做...

按照这个食谱执行以下步骤：

1.  `shiftmonth`函数将允许我们按任意月数前后移动我们的日期：

```py
import datetime

def shiftmonth(d, months):
    for _ in range(abs(months)):
        if months > 0:
            d = d.replace(day=5) + datetime.timedelta(days=28)
        else:
            d = d.replace(day=1) - datetime.timedelta(days=1)
    d = d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return d
```

1.  使用它就像简单地提供你想要添加或移除的月份一样简单：

```py
>>> now = datetime.datetime.utcnow()
>>> now
datetime.datetime(2018, 3, 21, 21, 55, 5, 699400)
```

1.  我们可以用它去到下个月：

```py
>>> shiftmonth(now, 1)
datetime.datetime(2018, 4, 1, 0, 0)
```

1.  或者回到上个月：

```py
>>> shiftmonth(now, -1)
datetime.datetime(2018, 2, 1, 0, 0)
```

1.  甚至可以按月份移动：

```py
>>> shiftmonth(now, 10)
datetime.datetime(2019, 1, 1, 0, 0)
```

# 它是如何工作的...

如果您尝试将此配方与*前往明天*进行比较，您会注意到，尽管其目的非常相似，但这个配方要复杂得多。

就像在移动天数时，我们有兴趣在一天中的特定时间点移动一样（通常是开始时），当移动月份时，我们不希望最终处于新月份的随机日期和时间。

这解释了我们配方的最后一部分，对于我们数学表达式产生的任何日期时间，我们将时间重置为该月的第一天的午夜：

```py
d = d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
```

就像对于天数配方一样，这使我们能够检查条件，例如`two_month_from_now > event_date >= next_month`，因为我们将捕捉从该月的第一天午夜到上个月的最后一天 23:59 的所有事件。

您可能想知道的部分是`for`循环。

与我们必须按天数移动（所有天数的持续时间相等为 24 小时）不同，当按月份移动时，我们需要考虑到每个月的持续时间不同的事实。

这就是为什么在向前移动时，我们将当前日期设置为月份的第 5 天，然后添加 28 天。仅仅添加 28 天是不够的，因为它只适用于 2 月，如果您在想，添加 31 天也不起作用，因为在 2 月的情况下，您将移动两个月而不是一个月。

这就是为什么我们将当前日期设置为月份的第 5 天，因为我们想要选择一个日期，我们确切地知道向其添加 28 天将使我们进入下一个月。

例如，选择月份的第一天将有效，因为 3 月 1 日+28 天=3 月 29 日，所以我们仍然在 3 月。而 3 月 5 日+28 天=4 月 2 日，4 月 5 日+28 天=5 月 3 日，2 月 5 日+28 天=3 月 5 日。因此，对于任何给定的月份，我们在将 5 日加 28 天时总是进入下一个月。

我们总是移动到不同的日期并不重要，因为该日期总是会被替换为该月的第一天。

由于我们无法移动确保我们总是准确地移动到下一个月的固定天数，所以我们不能仅通过添加`天数*月份`来移动，因此我们必须在`for`循环中执行此操作，并连续移动`月份`次数。

当向后移动时，事情变得容易得多。由于所有月份都从月份的第一天开始，我们只需移动到那里，然后减去一天。我们总是会在上个月的最后一天。

# 工作日

为月份的第 20 天或第 3 周构建日期非常简单，但如果您必须为月份的第 3 个星期一构建日期呢？

# 如何做...

按照以下步骤进行：

1.  为了解决这个问题，我们将实际生成所有与请求的工作日匹配的月份日期：

```py
import datetime

def monthweekdays(month, weekday):
    now = datetime.datetime.utcnow()
    d = now.replace(day=1, month=month, hour=0, minute=0, second=0, 
                    microsecond=0)
    days = []
    while d.month == month:
        if d.isoweekday() == weekday:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days
```

1.  然后，一旦我们有了这些列表，抓取*第 n 个*日期只是简单地索引结果列表。例如，要抓取 3 月的星期一：

```py
>>> monthweekdays(3, 1)
[datetime.datetime(2018, 3, 5, 0, 0), 
 datetime.datetime(2018, 3, 12, 0, 0), 
 datetime.datetime(2018, 3, 19, 0, 0), 
 datetime.datetime(2018, 3, 26, 0, 0)]
```

1.  所以抓取三月的第三个星期一将是：

```py
>>> monthweekdays(3, 1)[2]
datetime.datetime(2018, 3, 19, 0, 0)
```

# 它是如何工作的...

在配方的开始，我们为所请求的月份的第一天创建一个日期。然后我们每次向前移动一天，直到月份结束，并将所有与请求的工作日匹配的日期放在一边。

星期从星期一到星期日分别为 1 到 7。

一旦我们有了所有星期一、星期五或者月份的其他日期，我们只需索引结果列表，抓取我们真正感兴趣的日期。

# 工作日

在许多管理应用程序中，您只需要考虑工作日，星期六和星期日并不重要。在这些日子里，您不工作，所以从工作的角度来看，它们不存在。

因此，在计算项目管理或与工作相关的应用程序的给定时间跨度内包含的日期时，您可以忽略这些日期。

# 如何做...

我们想要获取两个日期之间的工作日列表：

```py
def workdays(d, end, excluded=(6, 7)):
    days = []
    while d.date() < end.date():
        if d.isoweekday() not in excluded:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days
```

例如，如果是 2018 年 3 月 22 日，这是一个星期四，我想知道工作日直到下一个星期一（即 3 月 26 日），我可以轻松地要求`workdays`：

```py
>>> workdays(datetime.datetime(2018, 3, 22), datetime.datetime(2018, 3, 26))
[datetime.datetime(2018, 3, 22, 0, 0), 
 datetime.datetime(2018, 3, 23, 0, 0)]
```

因此我们知道还剩下两天：星期四本身和星期五。

如果您在世界的某个地方工作日是星期日，可能不是星期五，`excluded`参数可以用来指示哪些日期应该从工作日中排除。

# 它是如何工作的...

这个方法非常简单，我们只是从提供的日期（`d`）开始，每次加一天，直到达到`end`。

我们认为提供的参数是日期时间，因此我们循环比较只有日期，因为我们不希望根据`d`和`end`中提供的时间随机包括和排除最后一天。

这允许`datetime.datetime.utcnow()`为我们提供第一个参数，而不必关心函数何时被调用。只有日期本身将被比较，而不包括它们的时间。

# 组合日期和时间

有时您会有分开的日期和时间。当它们由用户输入时，这种情况特别频繁。从交互的角度来看，通常更容易选择一个日期然后选择一个时间，而不是一起选择日期和时间。或者您可能正在组合来自两个不同来源的输入。

在所有这些情况下，您最终会得到一个日期和时间，您希望将它们组合成一个单独的`datetime.datetime`实例。

# 如何做到...

Python 标准库提供了对这些操作的支持，因此拥有其中的任何两个：

```py
>>> t = datetime.time(13, 30)
>>> d = datetime.date(2018, 1, 11)
```

我们可以轻松地将它们组合成一个单一的实体：

```py
>>> datetime.datetime.combine(d, t)
datetime.datetime(2018, 1, 11, 13, 30)
```

# 还有更多...

如果您的`time`实例有一个时区（`tzinfo`），将日期与时间组合也会保留它：

```py
>>> t = datetime.time(13, 30, tzinfo=datetime.timezone.utc)
>>> datetime.datetime.combine(d, t)
datetime.datetime(2018, 1, 11, 13, 30, tzinfo=datetime.timezone.utc)
```

如果您的时间没有时区，您仍然可以在组合这两个值时指定一个时区：

```py
>>> t = datetime.time(13, 30)
>>> datetime.datetime.combine(d, t, tzinfo=datetime.timezone.utc)
```

在组合时提供时区仅支持 Python 3.6+。如果您使用之前的 Python 版本，您将不得不将时区设置为时间值。
