# 第七章。测试驱动开发演练

在本章中，我们不会讨论 Python 的新测试技术，也不会花太多时间讨论测试的哲学。相反，我们将逐步演示一个实际的开发过程。谦逊且不幸易犯错的作者在开发个人日程安排程序的一部分时，记录了他的错误——以及测试如何帮助他修复这些错误。

在本章中，我们将涵盖以下主题：

+   编写可测试规范

+   编写驱动开发过程的单元测试

+   编写符合规范和单元测试的代码

+   使用可测试规范和单元测试来帮助调试

在阅读本章时，你将被提示设计和构建自己的模块，这样你就可以走自己的过程。

# 编写规范

如同往常，这个过程从一份书面规范开始。规范是一个我们在第二章和第三章中学习的`doctest`，即*使用 doctest*，所以计算机可以使用它来检查实现。尽管规范并不是一组单元测试，但为了使文档更易于人类读者理解，我们牺牲了单元测试的纪律（暂时如此）。这是一个常见的权衡，只要你也编写覆盖代码的单元测试来弥补，那就没问题。

本章项目的目标是创建一个能够表示个人时间管理信息的 Python 包。

以下代码放在一个名为`docs/outline.txt`的文件中：

```py
This project is a personal scheduling system intended to keep track of
a single person's schedule and activities. The system will store and
display two kinds of schedule information: activities and statuses.
Activities and statuses both support a protocol which allows them to
be checked for overlap with another object supporting the protocol.

>>> from planner.data import Activity, Status
>>> from datetime import datetime

Activities and statuses are stored in schedules, to which they can be
added and removed.

>>> from planner.data import Schedule
>>> activity = Activity('test activity',
..                      datetime(year = 2014, month = 6, day = 1,
..                               hour = 10, minute = 15),
..                      datetime(year = 2014, month = 6, day = 1,
..                               hour = 12, minute = 30))
>>> duplicate_activity = Activity('test activity',
..                      datetime(year = 2014, month = 6, day = 1,
..                               hour = 10, minute = 15),
..                      datetime(year = 2014, month = 6, day = 1,
..                               hour = 12, minute = 30))
>>> status = Status('test status',
...                 datetime(year = 2014, month = 7, day = 1,
...                          hour = 10, minute = 15),
...                 datetime(year = 2014, month = 7, day = 1,
...                          hour = 12, minute = 30))
>>> schedule = Schedule()
>>> schedule.add(activity)
>>> schedule.add(status)
>>> status in schedule
True
>>> activity in schedule
True
>>> duplicate_activity in schedule
True
>>> schedule.remove(activity)
>>> schedule.remove(status)
>>> status in schedule
False
>>> activity in schedule
False

Activities represent tasks that the person must actively engage in,
and they are therefore mutually exclusive: no person can have two
activities that overlap the same period of time.

>>> activity1 = Activity('test activity 1',
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 9, minute = 5),
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 12, minute = 30))
>>> activity2 = Activity('test activity 2',
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 10, minute = 15),
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 13, minute = 30))
>>> schedule = Schedule()
>>> schedule.add(activity1)
>>> schedule.add(activity2)
Traceback (most recent call last):
ScheduleError: "test activity 2" overlaps with "test activity 1"

Statuses represent tasks that a person engages in passively, and so
can overlap with each other and with activities.

>>> activity1 = Activity('test activity 1',
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 9, minute = 5),
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 12, minute = 30))
>>> status1 = Status('test status 1',
...                  datetime(year = 2014, month = 6, day = 1,
...                           hour = 10, minute = 15),
...                  datetime(year = 2014, month = 6, day = 1,
...                           hour = 13, minute = 30))
>>> status2 = Status('test status 2',
...                  datetime(year = 2014, month = 6, day = 1,
...                           hour = 8, minute = 45),
...                  datetime(year = 2014, month = 6, day = 1,
...                           hour = 15, minute = 30))
>>> schedule = Schedule()
>>> schedule.add(activity1)
>>> schedule.add(status1)
>>> schedule.add(status2)
>>> activity1 in schedule
True
>>> status1 in schedule
True
>>> status2 in schedule
True

Schedules can be saved to a sqlite database, and they can be reloaded
from that stored state.

>>> from planner.persistence import file
>>> storage = File(':memory:')
>>> schedule.store(storage)
>>> newsched = Schedule.load(storage)
>>> schedule == newsched
True
```

这个`doctest`将作为我项目的可测试规范，这意味着它将成为所有测试和将要构建的程序代码的基础。让我们更详细地看看每个部分：

```py
This project is a personal scheduling system intended to keep track of
a single person's schedule and activities. The system will store and
display two kinds of schedule information: activities and statuses.
Activities and statuses both support a protocol which allows them to
be checked for overlap with another object supporting the protocol.

>>> from planner.data import Activity, Status
>>> from datetime import datetime
```

上一段代码包含一些介绍性英文文本，以及几个`import`语句，它们引入了我们需要用于这些测试的代码。通过这样做，它们也告诉我们`planner`包的一些结构。它包含一个名为`data`的模块，该模块定义了`Activity`和`Status`。

```py
Activities and statuses are stored in schedules, to which they can be
added and removed.

>>> from planner.data import Schedule
>>> activity = Activity('test activity',
..                      datetime(year = 2014, month = 6, day = 1,
..                               hour = 10, minute = 15),
..                      datetime(year = 2014, month = 6, day = 1,
..                               hour = 12, minute = 30))
>>> duplicate_activity = Activity('test activity',
..                      datetime(year = 2014, month = 6, day = 1,
..                               hour = 10, minute = 15),
..                      datetime(year = 2014, month = 6, day = 1,
..                               hour = 12, minute = 30))
>>> status = Status('test status',
...                 datetime(year = 2014, month = 7, day = 1,
...                          hour = 10, minute = 15),
...                 datetime(year = 2014, month = 7, day = 1,
...                          hour = 12, minute = 30))
>>> schedule = Schedule()
>>> schedule.add(activity)
>>> schedule.add(status)
>>> status in schedule
True
>>> activity in schedule
True
>>> duplicate_activity in schedule
True
>>> schedule.remove(activity)
>>> schedule.remove(status)
>>> status in schedule
False
>>> activity in schedule
False
```

上一段测试描述了`Schedule`实例与`Activity`和`Status`对象交互时的一些期望行为。根据这些测试，`Schedule`实例必须接受`Activity`或`Status`对象作为其`add`和`remove`方法的参数；一旦添加，`in`运算符必须返回`True`，直到对象被移除。此外，具有相同参数的两个`Activity`实例必须被`Schedule`视为同一个对象：

```py
Activities represent tasks that the person must actively engage in,
and they are therefore mutually exclusive: no person can have two
activities that overlap the same period of time.

>>> activity1 = Activity('test activity 1',
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 9, minute = 5),
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 12, minute = 30))
>>> activity2 = Activity('test activity 2',
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 10, minute = 15),
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 13, minute = 30))
>>> schedule = Schedule()
>>> schedule.add(activity1)
>>> schedule.add(activity2)
Traceback (most recent call last):
ScheduleError: "test activity 2" overlaps with "test activity 1"
```

上一段测试代码描述了当将重叠活动添加到日程安排中时应该发生的情况。具体来说，应该抛出一个`ScheduleError`异常：

```py
Statuses represent tasks that a person engages in passively, and so
can overlap with each other and with activities.

>>> activity1 = Activity('test activity 1',
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 9, minute = 5),
...                      datetime(year = 2014, month = 6, day = 1,
...                               hour = 12, minute = 30))
>>> status1 = Status('test status 1',
...                  datetime(year = 2014, month = 6, day = 1,
...                           hour = 10, minute = 15),
...                  datetime(year = 2014, month = 6, day = 1,
...                           hour = 13, minute = 30))
>>> status2 = Status('test status 2',
...                  datetime(year = 2014, month = 6, day = 1,
...                           hour = 8, minute = 45),
...                  datetime(year = 2014, month = 6, day = 1,
...                           hour = 15, minute = 30))
>>> schedule = Schedule()
>>> schedule.add(activity1)
>>> schedule.add(status1)
>>> schedule.add(status2)
>>> activity1 in schedule
True
>>> status1 in schedule
True
>>> status2 in schedule
True
```

之前的测试代码描述了当将重叠状态添加到日程安排时应该发生什么：日程安排应该接受它们。此外，如果一个状态和一个活动重叠，它们仍然都可以被添加：

```py
Schedules can be saved to a sqlite database, and they can be reloaded
from that stored state.

>>> from planner.persistence import file
>>> storage = File(':memory:')
>>> schedule.store(storage)
>>> newsched = Schedule.load(storage)
>>> schedule == newsched
True
```

之前的代码描述了日程存储应该如何工作。它还告诉我们，`planner` 包需要包含一个 `persistence` 模块，该模块反过来应该包含 `File`。它还告诉我们，`Schedule` 实例应该有 `load` 和 `store` 方法，并且当它们包含相同的数据时，`==` 操作符应该返回 `True`。

## 尝试一下你自己——你打算做什么？

是时候你自己想出一个项目了，一个你可以自己工作的项目。我们逐步通过开发过程：

1.  想想一个与本章中描述的项目大致相同复杂性的项目。它应该是一个单独的模块或一个包中的几个模块。它还应该是你感兴趣的东西，这就是为什么我没有在这里给你一个具体的任务。

    想象一下项目已经完成，你需要编写一个描述你所做的工作，以及一些演示代码的描述。然后继续编写你的描述和演示代码，以 `doctest` 文件的形式。

1.  当你编写 `doctest` 文件时，要注意你的原始想法需要稍作改变以使演示更容易编写或工作得更好的地方。当你找到这样的案例时，请注意它们！在这个阶段，最好是稍微改变一下想法，并在整个过程中节省自己精力。

## 总结规范

我们现在为几个中等规模的项目——你的和我的——有了可测试的规范。这将帮助我们编写单元测试和代码，并让我们对每个项目作为一个整体完成的情况有一个整体的认识。

此外，将代码写入 `doctest` 的过程给了我们测试驱动我们想法的机会。尽管项目实现仍然只是想象中的，但我们可能通过具体使用它们来稍微改进了我们的项目。

一次又一次，我们在编写将要测试的代码之前编写这些测试是非常重要的。通过先编写测试，我们为自己提供了一个试金石，我们可以用它来判断我们的代码是否符合我们的意图。如果我们先编写代码，然后再编写测试，最终我们只是将代码实际执行的行为——而不是我们希望它执行的行为——嵌入到测试中。

# 编写初始单元测试

由于规范中不包含单元测试，在开始模块编码之前仍然需要单元测试。`planner.data` 类是实施的第一目标，因此它们是第一个接受测试的。

活动和状态被定义为非常相似，因此它们的测试模块也是相似的。尽管它们并不完全相同，也不需要具有任何特定的继承关系；因此测试仍然是独立的。

以下测试位于 `tests/test_activities.py`：

```py
from unittest import TestCase
from unittest.mock import patch, Mock
from planner.data import Activity, TaskError
from datetime import datetime

class constructor_tests(TestCase):
    def test_valid(self):
        activity = Activity('activity name',
                           datetime(year = 2012, month = 9, day = 11),
                           datetime(year = 2013, month = 4, day = 27))

        self.assertEqual(activity.name, 'activity name')
        self.assertEqual(activity.begins,
                         datetime(year = 2012, month = 9, day = 11))
        self.assertEqual(activity.ends,
                         datetime(year = 2013, month = 4, day = 27))

    def test_backwards_times(self):
        self.assertRaises(TaskError,
                          Activity,
                          'activity name',
                          datetime(year = 2013, month = 4, day = 27),
                          datetime(year = 2012, month = 9, day = 11))

    def test_too_short(self):
        self.assertRaises(TaskError,
                          Activity,
                          'activity name',
                          datetime(year = 2013, month = 4, day = 27,
                                   hour = 7, minute = 15),
                          datetime(year = 2013, month = 4, day = 27,
                                   hour = 7, minute = 15))

class utility_tests(TestCase):
    def test_repr(self):
        activity = Activity('activity name',
                           datetime(year = 2012, month = 9, day = 11),
                           datetime(year = 2013, month = 4, day = 27))

        expected = "<activity name 2012-09-11T00:00:00 2013-04-27T00:00:00>"

        self.assertEqual(repr(activity), expected)

class exclusivity_tests(TestCase):
    def test_excludes(self):
        activity = Mock()

        other = Activity('activity name',
                         datetime(year = 2012, month = 9, day = 11),
                         datetime(year = 2012, month = 10, day = 6))

        # Any activity should exclude any activity
        self.assertTrue(Activity.excludes(activity, other))

        # Anything not known to be excluded should be included
        self.assertFalse(Activity.excludes(activity, None))

class overlap_tests(TestCase):
    def test_overlap_before(self):
        activity = Mock(begins = datetime(year = 2012, month = 9, day = 11),
                        ends = datetime(year = 2012, month = 10, day = 6))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertFalse(Activity.overlaps(activity, other))

    def test_overlap_begin(self):
        activity = Mock(begins = datetime(year = 2012, month = 8, day = 11),
                        ends = datetime(year = 2012, month = 11, day = 27))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertTrue(Activity.overlaps(activity, other))

    def test_overlap_end(self):
        activity = Mock(begins = datetime(year = 2013, month = 1, day = 11),
                        ends = datetime(year = 2013, month = 4, day = 16))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertTrue(Activity.overlaps(activity, other))

    def test_overlap_inner(self):
        activity = Mock(begins = datetime(year = 2012, month = 10, day = 11),
                        ends = datetime(year = 2013, month = 1, day = 27))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertTrue(Activity.overlaps(activity, other))

    def test_overlap_outer(self):
        activity = Mock(begins = datetime(year = 2012, month = 8, day = 12),
                        ends = datetime(year = 2013, month = 3, day = 15))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertTrue(Activity.overlaps(activity, other))

    def test_overlap_after(self):
        activity = Mock(begins = datetime(year = 2013, month = 2, day = 6),
                        ends = datetime(year = 2013, month = 4, day = 27))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertFalse(Activity.overlaps(activity, other))
```

让我们一步一步地看看以下代码：

```py
    def test_valid(self):
        activity = Activity('activity name',
                            datetime(year = 2012, month = 9, day = 11),
                            datetime(year = 2013, month = 4, day = 27))

        self.assertEqual(activity.name, 'activity name')
        self.assertEqual(activity.begins,
                         datetime(year = 2012, month = 9, day = 11))
        self.assertEqual(activity.ends,
                         datetime(year = 2013, month = 4, day = 27))
```

`test_valid` 方法检查当所有参数都正确时构造函数是否工作正常。这是一个重要的测试，因为它定义了正常情况下应该是什么正确的行为。然而，我们还需要更多的测试来定义在异常情况下的正确行为：

```py
    def test_backwards_times(self):
        self.assertRaises(TaskError,
                          Activity,
                          'activity name',
                          datetime(year = 2013, month = 4, day = 27),
                          datetime(year = 2012, month = 9, day = 11))
```

在这里，我们确保不能创建一个在开始之前就结束的活动。这没有意义，并且很容易在实现过程中导致假设出错：

```py
    def test_too_short(self):
        self.assertRaises(TaskError,
                          Activity,
                          'activity name',
                          datetime(year = 2013, month = 4, day = 27,
                                   hour = 7, minute = 15),
                          datetime(year = 2013, month = 4, day = 27,
                                   hour = 7, minute = 15))
```

我们也不希望活动非常短。在现实世界中，耗时为零的活动是没有意义的，所以我们在这里有一个测试来确保不允许这种情况发生：

```py
class utility_tests(TestCase):
    def test_repr(self):
        activity = Activity('activity name',
                            datetime(year = 2012, month = 9, day = 11),
                            datetime(year = 2013, month = 4, day = 27))

        expected = "<activity name 2012-09-11T00:00:00 2013-04-27T00:00:00>"

        self.assertEqual(repr(activity), expected)
```

虽然 `repr(activity)` 在任何生产代码路径中可能不会被使用，但在开发和调试期间非常方便。这个测试定义了活动文本表示应该看起来是什么样子，以确保它包含所需的信息。

### 小贴士

`repr` 函数在调试期间通常很有用，因为它试图将任何对象转换成一个表示该对象的字符串。这与 `str` 函数不同，因为 `str` 尝试将对象转换成一个对人类阅读方便的字符串。另一方面，`repr` 函数试图创建一个包含代码的字符串，该代码可以重新创建对象。这是一个稍微有点难理解的概念，所以这里有一个对比 `str` 和 `repr` 的例子：

```py
>>> from decimal import Decimal
>>> x = Decimal('123.45678')
>>> str(x)
'123.45678'
>>> repr(x)
"Decimal('123.45678')"
```

```py
class exclusivity_tests(TestCase):
    def test_excludes(self):
        activity = Mock()

        other = Activity('activity name',
                         datetime(year = 2012, month = 9, day = 11),
                         datetime(year = 2012, month = 10, day = 6))

        # Any activity should exclude any activity
        self.assertTrue(Activity.excludes(activity, other))

        # Anything not known to be excluded should be included
        self.assertFalse(Activity.excludes(activity, None))
```

存储在日程表中的对象决定它们是否与其他它们重叠的对象互斥。具体来说，活动应该相互排除，所以我们在这里进行检查。我们使用一个模拟对象作为主要活动，但我们有点偷懒，使用一个真实的 `Activity` 实例进行比较，相信在这种情况下不会有问题。我们预计 `Activity.excludes` 不会做很多比将其参数应用于 `isinstance` 函数之外的事情，所以构造函数中的错误不会对事情造成太大影响。

```py
class overlap_tests(TestCase):
    def test_overlap_before(self):
        activity = Mock(begins = datetime(year = 2012, month = 9, day = 11),
                        ends = datetime(year = 2012, month = 10, day = 6))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertFalse(Activity.overlaps(activity, other))

    def test_overlap_begin(self):
        activity = Mock(begins = datetime(year = 2012, month = 8, day = 11),
                        ends = datetime(year = 2012, month = 11, day = 27))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertTrue(Activity.overlaps(activity, other))

    def test_overlap_end(self):
        activity = Mock(begins = datetime(year = 2013, month = 1, day = 11),
                        ends = datetime(year = 2013, month = 4, day = 16))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertTrue(Activity.overlaps(activity, other))

    def test_overlap_inner(self):         activity = Mock(begins = datetime(year = 2012, month = 10, day = 11),
                        ends = datetime(year = 2013, month = 1, day = 27))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertTrue(Activity.overlaps(activity, other))

    def test_overlap_outer(self):
        activity = Mock(begins = datetime(year = 2012, month = 8, day = 12),
                        ends = datetime(year = 2013, month = 3, day = 15))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertTrue(Activity.overlaps(activity, other))

    def test_overlap_after(self):
        activity = Mock(begins = datetime(year = 2013, month = 2, day = 6),
                        ends = datetime(year = 2013, month = 4, day = 27))

        other = Mock(begins = datetime(year = 2012, month = 10, day = 7),
                     ends = datetime(year = 2013, month = 2, day = 5))

        self.assertFalse(Activity.overlaps(activity, other))
```

这些测试描述了在第一个活动重叠的情况下检查活动是否重叠的代码行为。

+   在第二个活动之前

+   与第二个活动的开始重叠

+   与第二个活动的结束重叠

+   在第二个活动的范围内开始和结束

+   在第二个活动之前开始并在其之后结束

+   在第二个活动之后

这涵盖了任务之间可能存在的关系域。

在这些测试中没有使用任何实际的活动，只是给 `Mock` 对象赋予了 `Activity.overlaps` 函数应该查找的属性。一如既往，我们尽力确保不同的代码单元在测试期间不能相互交互。

### 小贴士

你可能已经注意到，我们通过传递构造函数所需的属性作为关键字参数来创建模拟对象，使用了一个快捷方式来创建模拟对象。大多数时候，这是一种节省一点工作的便捷方式，但它确实有一个问题，那就是它只适用于没有用作`Mock`构造函数实际参数的属性名称。值得注意的是，名为`name`的属性不能以这种方式分配，因为该参数对`Mock`有特殊含义。

`tests/test_statuses.py`中的代码几乎相同，只是它使用的是`Status`类而不是`Activity`类。尽管如此，有一个显著的区别：

```py
    def test_excludes(self):
        status = Mock()

        other = Status('status name',
                       datetime(year = 2012, month = 9, day = 11),
                       datetime(year = 2012, month = 10, day = 6))

        # A status shouldn't exclude anything
        self.assertFalse(Status.excludes(status, other))
        self.assertFalse(Status.excludes(status, None))
```

`Status`和`Activity`之间的定义性区别在于，状态不会排除与之重叠的其他任务。测试自然应该反映这种差异。

以下代码位于`tests/test_schedules.py`中。我们定义了几个模拟对象，它们表现得像状态或活动，并且支持重叠和排除协议。我们将在几个测试中使用这些模拟对象，以查看调度如何处理重叠和排除对象的组合：

```py
from unittest import TestCase
from unittest.mock import patch, Mock
from planner.data import Schedule, ScheduleError
from datetime import datetime

class add_tests(TestCase):
    overlap_exclude = Mock()
    overlap_exclude.overlaps = Mock(return_value = True)
    overlap_exclude.excludes = Mock(return_value = True)

    overlap_include = Mock()
    overlap_include.overlaps = Mock(return_value = True)
    overlap_include.excludes = Mock(return_value = False)

    distinct_exclude = Mock()
    distinct_exclude.overlaps = Mock(return_value = False)
    distinct_exclude.excludes = Mock(return_value = True)

    distinct_include = Mock()
    distinct_include.overlaps = Mock(return_value = False)
    distinct_include.excludes = Mock(return_value = False)

    def test_add_overlap_exclude(self):
        schedule = Schedule()
        schedule.add(self.distinct_include)
        self.assertRaises(ScheduleError,
                          schedule.add,
                          self.overlap_exclude)

    def test_add_overlap_include(self):
        schedule = Schedule()
        schedule.add(self.distinct_include)
        schedule.add(self.overlap_include)

    def test_add_distinct_exclude(self):
        schedule = Schedule()
        schedule.add(self.distinct_include)
        schedule.add(self.distinct_exclude)

    def test_add_distinct_include(self):
        schedule = Schedule()
        schedule.add(self.distinct_include)
        schedule.add(self.distinct_include)

    def test_add_over_overlap_exclude(self):
        schedule = Schedule()
        schedule.add(self.overlap_exclude)
        self.assertRaises(ScheduleError,
                          schedule.add,
                          self.overlap_include)

    def test_add_over_distinct_exclude(self):
        schedule = Schedule()
        schedule.add(self.distinct_exclude)
        self.assertRaises(ScheduleError,
                          schedule.add,
                          self.overlap_include)

    def test_add_over_overlap_include(self):
        schedule = Schedule()
        schedule.add(self.overlap_include)
        schedule.add(self.overlap_include)

    def test_add_over_distinct_include(self):
        schedule = Schedule()
        schedule.add(self.distinct_include)
        schedule.add(self.overlap_include)

class in_tests(TestCase):
    fake = Mock()
    fake.overlaps = Mock(return_value = True)
    fake.excludes = Mock(return_value = True)

    def test_in_before_add(self):
        schedule = Schedule()
        self.assertFalse(self.fake in schedule)

    def test_in_after_add(self):
        schedule = Schedule()
        schedule.add(self.fake)
        self.assertTrue(self.fake in schedule)
```

让我们仔细看看以下代码的一些部分：

```py
    overlap_exclude = Mock()
    overlap_exclude.overlaps = Mock(return_value = True)
    overlap_exclude.excludes = Mock(return_value = True)

    overlap_include = Mock()
    overlap_include.overlaps = Mock(return_value = True)
    overlap_include.excludes = Mock(return_value = False)

    distinct_exclude = Mock()
    distinct_exclude.overlaps = Mock(return_value = False)
    distinct_exclude.excludes = Mock(return_value = True)

    distinct_include = Mock()
    distinct_include.overlaps = Mock(return_value = False)
    distinct_include.excludes = Mock(return_value = False)
```

这些行将模拟对象作为`add_tests`类的属性创建。每个模拟对象都有模拟的`overlaps`和`excludes`方法，当被调用时总是返回`True`或`False`。这意味着每个模拟对象都认为自己是重叠的，要么是所有东西，要么什么都不是，并且排除要么是所有东西，要么什么都不是。在这四个模拟对象之间，我们涵盖了所有可能的组合。在接下来的测试中，我们将添加这些模拟对象的组合到调度中，并确保它做了正确的事情：

```py
    def test_add_overlap_exclude(self):
        schedule = Schedule()
        schedule.add(self.distinct_include)
        self.assertRaises(ScheduleError,
                          schedule.add,
                          self.overlap_exclude)

    def test_add_overlap_include(self):
        schedule = Schedule()
        schedule.add(self.distinct_include)
        schedule.add(self.overlap_include)

    def test_add_distinct_exclude(self):
        schedule = Schedule()
        schedule.add(self.distinct_include)
        schedule.add(self.distinct_exclude)

    def test_add_distinct_include(self):
        schedule = Schedule()
        schedule.add(self.distinct_include)
        schedule.add(self.distinct_include)
```

前面的四个测试涵盖了我们将非重叠对象添加到调度中的情况。所有这些测试都预期会接受非重叠对象，除了第一个。在这个测试中，我们之前添加了一个声称确实存在重叠的对象；此外，它排除了所有与之重叠的对象。这个测试表明，如果被添加的对象或已经在调度中的对象认为存在重叠，那么调度必须将其视为重叠。

```py
    def test_add_over_overlap_exclude(self):
        schedule = Schedule()
        schedule.add(self.overlap_exclude)
        self.assertRaises(ScheduleError,
                          schedule.add,
                          self.overlap_include)
```

在这个测试中，我们确保如果已经在调度中的对象与新的对象重叠并声称具有排他性，那么添加新的对象将失败。

```py
    def test_add_over_distinct_exclude(self):
        schedule = Schedule()
        schedule.add(self.distinct_exclude)
        self.assertRaises(ScheduleError,
                          schedule.add,
                          self.overlap_include)
```

在这个测试中，我们确保即使已经在调度中的对象认为它不会与新的对象重叠，它也会排除新的对象，因为新的对象认为存在重叠。

```py
    def test_add_over_overlap_include(self):
        schedule = Schedule()
        schedule.add(self.overlap_include)
        schedule.add(self.overlap_include)

    def test_add_over_distinct_include(self):
        schedule = Schedule()
        schedule.add(self.distinct_include)
        schedule.add(self.overlap_include)
```

这些测试确保包容性对象不会以某种方式干扰彼此添加到调度中。

```py
class in_tests(TestCase):
    fake = Mock()
    fake.overlaps = Mock(return_value = True)
    fake.excludes = Mock(return_value = True)

    def test_in_before_add(self):
        schedule = Schedule()
        self.assertFalse(self.fake in schedule)

    def test_in_after_add(self):
        schedule = Schedule()
        schedule.add(self.fake)
        self.assertTrue(self.fake in schedule)
```

这两个测试描述了与`in`操作符相关的调度行为。具体来说，当问题中的对象实际上在调度中时，它应该返回`True`。

## 亲自试试看——编写您的早期单元测试

即使是一个用`doctest`编写的可测试规范，仍然存在许多可以通过良好的单元测试来消除的歧义。再加上规范没有在不同测试之间保持分离，您就可以看到，是时候让您的项目获得一些单元测试了。执行以下步骤：

1.  找到您项目中由规范（或由规范暗示）描述的某个元素。

1.  编写一个单元测试，描述当给定正确输入时该元素的行为。

1.  编写一个单元测试，描述当给定错误输入时该元素的行为。

1.  编写单元测试，描述该元素在正确和错误输入之间的边界行为。

1.  如果您能找到程序中另一个未测试的部分，请回到步骤 1。

## 总结初始单元测试

这就是您真正将一个模糊不清的想法转化为您将要做的精确描述的地方。

最终结果可能相当长，这不应该让人感到惊讶。毕竟，在这个阶段，您的目标是完全定义您项目的行为；即使不考虑实现该行为的具体细节，这也是很多信息。

# 编码`planner.data`

是时候根据规范文档和单元测试编写一些代码了。具体来说，是时候编写`planner.data`模块了，该模块包含`Status`、`Activity`和`Schedule`。

为了创建这个包，我创建了一个名为`planner`的目录，并在该目录内创建了一个名为`__init__.py`的文件。不需要在`__init__.py`中放置任何内容，但该文件本身需要存在，以便告诉 Python`planner`目录是一个包。

以下代码位于`planner/data.py`：

```py
from datetime import timedelta

class TaskError(Exception):
    pass

class ScheduleError(Exception):
    pass

class Task:
    def __init__(self, name, begins, ends):
        if ends < begins:
            raise TaskError('The begin time must precede the end time')
        if ends - begins < timedelta(minutes = 5):
            raise TaskError('The minimum duration is 5 minutes')

        self.name = name
        self.begins = begins
        self.ends = ends

    def excludes(self, other):
        return NotImplemented

    def overlaps(self, other):
        if other.begins < self.begins:
            return other.ends > self.begins
        elif other.ends > self.ends:
            return other.begins < self.ends
        else:
            return True

    def __repr__(self):
        return '<{} {} {}>'.format(self.name,
                                   self.begins.isoformat(),
                                   self.ends.isoformat())

class Activity(Task):
    def excludes(self, other):
        return isinstance(other, Activity)

class Status(Task):
    def excludes(self, other):
        return False

class Schedule:
    def __init__(self):
        self.tasks = []

    def add(self, task):
        for contained in self.tasks:
            if task.overlaps(contained):
                if task.exclude(contained) or contained.exclude(task):
                    raise ScheduleError(task, containeed)

        self.tasks.append(task)

    def remove(self, task):
        try:
            self.tasks.remove(task)
        except ValueError:
            pass

    def __contains__(self, task):
        return task in self.tasks
```

这里的`Task`类包含了`Activity`类和`Status`类所需的大部分行为。由于它们所做的大部分事情都是共同的，因此编写一次代码并重用是有意义的。只有`excludes`方法在每个子类中都需要不同。这使得活动和状态类非常简单。`Schedule`类也相当简单。但是，这是正确的吗？我们的测试将告诉我们。

### 注意

在前面的代码中，我们使用了`timedelta`类和`datetime.isoformat`方法。这两个都是`datetime`模块的有用但有些晦涩的功能。`timedelta`实例表示两个时间点之间的持续时间。`isoformat`方法返回一个表示`datetime`模块的 ISO 8601 标准格式的字符串。

# 使用测试来确保代码正确

好吧，所以这段代码看起来相当不错。不幸的是，Nose 告诉我们有几个问题。实际上，Nose 报告了相当多的问题，但其中很多似乎与几个根本原因有关。

首先，让我们解决尽管`Activity`和`Status`类似乎没有`exclude`方法，但一些代码尝试调用该方法的问题。从 Nose 输出中看到的这个问题的典型报告看起来像跟踪回溯后：

```py
AttributeError: 'Activity' object has no attribute 'exclude'
```

看看我们的代码，我们看到它正确地被命名为`excludes`。Nose 错误报告中包含的跟踪回溯告诉我们问题出在`planner/data.py`的第 51 行，看起来是一个简单的修复。

我们只需将第 51 行的内容更改为以下内容：

```py
if task.exclude(contained) or contained.exclude(task):
```

变为：

```py
if task.excludes(contained) or contained.excludes(task):
```

然后再次运行 Nose。

类似地，我们的几个测试报告了以下输出：

```py
NameError: name 'containeed' is not defined
```

这显然是另一个打字错误。这一次是在`planner/data.py`的第 52 行。哎呀！我们也会修复这个问题，并再次运行 Nose 以查看还有什么问题。

继续我们优先处理低垂果实的趋势，让我们澄清以下报告的问题：

```py
SyntaxError: unexpected EOF while parsing
```

这又是另一个打字错误，这一次在`docs/outline.txt`中。这一次，问题不是测试中的代码问题，而是测试本身的问题。它仍然需要修复。

问题在于，在最初输入测试时，我显然只在几行的开头输入了两个点，而不是三个点，这告诉 doctest 表达式将继续到那一行。

修复那个问题后，事情开始变得不那么明显了。让我们接下来处理这个问题：

```py
File "docs/outline.txt", line 36, in outline.txt
Failed example:
 duplicate_activity in schedule
Expected:
 True
Got:
 False
```

为什么活动没有被看作是日程表的一部分？前面的例子通过了，这表明`in`操作符对我们实际添加到日程表中的活动是有效的。失败出现在我们尝试使用等效活动时；一旦我们意识到这一点，我们就知道我们需要修复什么。要么是我们的`__eq__`方法不起作用，要么（正如实际情况）我们忘记编写它。

我们可以通过向`Task`添加`__eq__`和`__ne__`方法来修复这个 bug，这些方法将被`Activity`和`Status`继承。

```py
    def __eq__(self, other):
        return (self.name == other.name and
                self.begins == other.begins and
                self.ends == other.ends)

    def __ne__(self, other):
        return (self.name != other.name or
                self.begins != other.begins or
                self.ends != other.ends)
```

现在，两个具有相同名称、开始时间和结束时间的任务将被视为等效，即使一个是`Status`而另一个是`Activity`。后者不一定正确，但它并没有导致我们的任何测试失败，所以我们暂时保留它。如果以后成为问题，我们将编写一个测试来检查它，然后修复它。

这一个是怎么回事？

```py
File "docs/outline.txt", line 61, in outline.txt
Failed example:
 schedule.add(activity2)
Expected:
 Traceback (most recent call last):
 ScheduleError: "test activity 2" overlaps with "test activity 1"
Got:
 Traceback (most recent call last):
 File "/usr/lib64/python3.4/doctest.py", line 1324, in __run
 compileflags, 1), test.globs)
 File "<doctest outline.txt[20]>", line 1, in <module>
 schedule.add(activity2)
 File "planner/data.py", line 62, in add
 raise ScheduleError(task, contained)
 planner.data.ScheduleError: (<test activity 2 2014-06-01T10:15:00 2014-06-01T13:30:00>, <test activity 1 2014-06-01T09:05:00 2014-06-01T12:30:00>)
```

嗯，看起来很丑陋，但如果你仔细看，你会发现`doctest`只是在抱怨抛出的异常没有按预期打印出来。它甚至是正确的异常；只是格式问题。

我们可以在`planner/data.py`的第 62 行修复这个问题，通过将这一行改为读取：

```py
raise ScheduleError('"{}" overlaps with "{}"'.format(task.name, contained.name))
```

这个 doctest 示例还有一个问题，那就是我们写下了期望的异常名为`ScheduleError`，这是 Python 2 打印异常的方式。然而，Python 3 使用限定名称打印异常，所以我们需要在 doctest 文件的第 63 行将其更改为`planner.data.ScheduleError`。

现在，如果你一直在跟随，所有的错误都应该已经修复，除了 `docs/outline.txt` 中的某些验收测试。基本上，这些失败的测试告诉我们我们还没有编写持久化代码，这是真的。

## 亲自尝试——编写和调试代码

基本步骤，正如我们之前讨论的，是编写一些代码，然后运行测试以查找代码中的问题，并重复。当你偶然遇到一个现有测试未涵盖的错误时，你需要编写一个新的测试并继续这个过程。执行以下步骤：

1.  编写满足至少一些测试的代码。

    运行你的测试。如果你使用了我们在前几章中提到的工具，你应该可以通过执行以下命令来运行所有内容：

    ```py
    $ python3 -m nose
    ```

1.  如果你在已经编写的代码中发现了错误，请使用测试输出帮助你定位和识别它们。一旦你理解了这些错误，尝试修复它们，然后回到步骤 2。

1.  一旦你修复了你编写的代码中的所有错误，如果你的项目还没有完成，选择一些新的测试来集中精力，然后回到步骤 1。

对这个过程的足够迭代将引导你拥有一个完整且经过测试的项目。当然，实际任务比简单地说是“它会工作”要困难得多，但最终它会工作。你将产生一个你可以有信心的代码库。这也会是一个比没有测试更容易的过程。

你的项目可能已经完成，但在个人调度器上还有更多的事情要做。在这一章的这个阶段，我还没有完成编写和调试过程。现在是时候去做这件事了。

# 编写持久化测试

由于我还没有实际的持久化代码单元测试，我将从编写一些测试开始。在这个过程中，我必须弄清楚持久化实际上是如何工作的。以下代码放在 `tests/test_persistence.py` 中：

```py
from unittest import TestCase
from planner.persistence import File

class test_file(TestCase):
    def test_basic(self):
        storage = File(':memory:')
        storage.store_object('tag1', ('some object',))
        self.assertEqual(tuple(storage.load_objects('tag1')),
                         (('some object',),))

    def test_multiple_tags(self):
        storage = File(':memory:')

        storage.store_object('tag1', 'A')
        storage.store_object('tag2', 'B')
        storage.store_object('tag1', 'C')
        storage.store_object('tag1', 'D')
        storage.store_object('tag3', 'E')
        storage.store_object('tag3', 'F')

        self.assertEqual(set(storage.load_objects('tag1')),
                         set(['A', 'C', 'D']))

        self.assertEqual(set(storage.load_objects('tag2')),
                         set(['B']))

        self.assertEqual(set(storage.load_objects('tag3')),
                         set(['E', 'F']))
```

观察测试代码的每个重要部分，我们看到以下内容：

```py
    def test_basic(self):
        storage = File(':memory:')
        storage.store_object('tag1', ('some object',))
        self.assertEqual(tuple(storage.load_objects('tag1')),
                         (('some object',),))
```

`test_basic` 测试创建 `File`，在名称 `'tag1'` 下存储一个单一的对象，然后从存储中重新加载该对象并检查它是否与原始对象相等。这确实是一个非常基础的测试，但它覆盖了简单的用例。

### 提示

在这里我们不需要测试夹具，因为我们实际上并没有在处理需要创建和删除的磁盘文件。特殊的文件名 `':memory:'` 告诉 SQLite 在内存中完成所有操作。这对于测试来说特别方便。

```py
    def test_multiple_tags(self):
        storage = File(':memory:')

        storage.store_object('tag1', 'A')
        storage.store_object('tag2', 'B')
        storage.store_object('tag1', 'C')
        storage.store_object('tag1', 'D')
        storage.store_object('tag3', 'E')
        storage.store_object('tag3', 'F')

        self.assertEqual(set(storage.load_objects('tag1')),
                         set(['A', 'C', 'D']))

        self.assertEqual(set(storage.load_objects('tag2')),
                         set(['B']))

        self.assertEqual(set(storage.load_objects('tag3')),
                         set(['E', 'F']))
```

`test_multiple_tags` 测试创建一个存储，然后在其中存储多个对象，其中一些具有重复的标签。然后它检查存储是否保留了所有给定标签的对象，并在请求时返回它们。

换句话说，所有这些测试都将持久化文件定义为从字符串键到对象值的 multimap。

### 注意

多映射是单个键和任意多个值之间的映射。换句话说，每个单独的键可能关联一个值，也可能是五十个。

# 完成个人计划

现在已经有了至少基本的单元测试覆盖持久化机制，是时候编写持久化代码本身了。以下内容位于 `planner/persistence.py` 文件中：

```py
import sqlite3
from pickle import loads, dumps

class File:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)

        try:
            self.connection.execute("""
                create table objects (tag, pickle)
            """)
        except sqlite3.OperationalError:
            pass

    def store_object(self, tag, object):
        self.connection.execute('insert into objects values (?, ?)',
                                (tag, dumps(object)))

    def load_objects(self, tag):
        cursor = self.connection.execute("""
                     select pickle from objects where tag like ?
                 """, (tag,))
        return [loads(row['pickle']) for row in cursor]
```

`store_object` 方法运行一个简短的 SQL 语句将对象存储到数据库字段中。对象序列化由 `pickle` 模块的 `dumps` 函数处理。

### 注意

`pickle` 模块整体处理存储和检索 Python 对象。特别是 `dumps` 函数将 Python 对象转换为字节字符串，这些字节字符串可以通过 `loads` 函数转换回 Python 对象。

`load_object` 方法使用 SQL 查询数据库以获取存储在给定标签下的每个对象的序列化版本，然后使用 `pickle.loads` 将这些序列化转换为要返回的实际对象。

现在我运行 Nose 来找出什么出了问题：

```py
ERROR: test_multiple_tags (test_persistence.test_file)
----------------------------------------------------------------------
Traceback (most recent call last):
 File "tests/test_persistence.py", line 21, in test_multiple_tags
 self.assertEqual(set(storage.load_objects('tag1')),
 File "planner/persistence.py", line 23, in load_objects
 return [loads(row['pickle']) for row in cursor]
 File "planner/persistence.py", line 23, in <listcomp>
 return [loads(row['pickle']) for row in cursor]
TypeError: tuple indices must be integers, not str
```

哎，是的。`sqlite3` 模块返回查询行作为元组，除非你告诉它否则。我想使用列名作为索引，所以我需要设置行工厂。我们将在 `File` 构造函数中添加以下行：

```py
self.connection.row_factory = sqlite3.Row
```

现在我运行 Nose，它告诉我的唯一问题是，我还没有实现 `Schedule.load` 和 `Schedule.store`。此外，还没有任何单元测试来检查这些方法。唯一的错误来自规范 doctest。是时候在 `tests/test_schedules.py` 中编写更多的单元测试了：

```py
class store_load_tests(TestCase):
    def setUp(self):
        fake_tasks = []
        for i in range(50):
            fake_task = Mock()
            fake_task.overlaps = Mock(return_value = False)
            fake_task.name = 'fake {}'.format(i)

        self.tasks = fake_tasks

    def tearDown(self):
        del self.tasks

    def test_store(self):
        fake_file = Mock()

        schedule = Schedule('test_schedule')

        for task in self.tasks:
            schedule.add(task)

        schedule.store(fake_file)

        for task in self.tasks:
            fake_file.store_object.assert_any_call('test_schedule', task)

    def test_load(self):
        fake_file = Mock()

        fake_file.load_objects = Mock(return_value = self.tasks)

        schedule = Schedule.load(fake_file, 'test_schedule')

        fake_file.load_objects.assert_called_once_with('test_schedule')

        self.assertEqual(set(schedule.tasks),
                         set(self.tasks))
```

现在我有一些测试要检查，是时候在 `planner/data.py` 中编写 `Schedule` 类的存储和加载方法了：

```py
    def store(self, storage):
        for task in self.tasks:
            storage.store_object(self.name, task)

    @staticmethod
    def load(storage, name = 'schedule'):
        value = Schedule(name)

        for task in storage.load_objects(name):
            value.add(task)

        return value
```

这些更改还意味着对 `Schedule` 构造函数的更改：

```py
    def __init__(self, name = 'schedule'):
        self.tasks = []
        self.name = name
```

好吧，现在，我运行 Nose，然后... 仍然有问题：

```py
File "docs/outline.txt", line 101, in outline.txt
Failed example:
 schedule == newsched
Expected:
 True
Got:
 False
```

看起来，日程表也需要根据其内容进行比较。这很容易做到：

```py
    def __eq__(self, other):
        return self.tasks == other.tasks
```

就像上次我们编写比较函数一样；这个函数有一些不寻常的行为，即它只有在任务以相同的顺序添加到日程表中时才认为两个日程表相等。再次强调，尽管这有点奇怪，但它并没有导致任何测试失败，而且它并不明显错误；所以我们将其留到它变得重要的时候再处理。

# 摘要

在本章中，我们学习了如何将本书前面部分介绍过的技能应用于实践中。我们通过逐步分析你谦逊的作者实际编写包的过程的录音来做到这一点。同时，你也有机会处理自己的项目，做出自己的决定，并设计自己的测试。你已经在一个测试驱动型项目中担任了领导角色，你应该能够在任何时候再次做到这一点。

现在我们已经涵盖了 Python 测试的核心，我们准备讨论集成和系统级别的测试，我们将在下一章中这样做。
