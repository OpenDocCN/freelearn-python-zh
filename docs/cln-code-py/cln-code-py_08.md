# 第八章：单元测试和重构

本章探讨的思想是本书全局背景中的基本支柱，因为它们对我们的最终目标至关重要：编写更好、更易维护的软件。

单元测试（以及任何形式的自动测试）对于软件的可维护性至关重要，因此是任何优质项目中不可或缺的东西。正因为如此，本章专门致力于自动化测试作为一个关键策略，以安全地修改代码，并在逐步改进的版本中进行迭代。

在本章之后，我们将对以下内容有更深入的了解：

+   为什么自动化测试对于采用敏捷软件开发方法论的项目至关重要

+   单元测试作为代码质量的一种启发方式

+   可用于开发自动化测试和设置质量门限的框架和工具

+   利用单元测试更好地理解领域问题并记录代码

+   与单元测试相关的概念，比如测试驱动开发

# 设计原则和单元测试

在本节中，我们首先从概念角度来看一下单元测试。我们将重新审视我们在之前讨论过的一些软件工程原则，以了解这与清晰代码的关系。

之后，我们将更详细地讨论如何将这些概念付诸实践（在代码层面），以及我们可以利用哪些框架和工具。

首先，我们快速定义一下单元测试的内容。单元测试是负责验证代码的其他部分的代码。通常，任何人都会倾向于说单元测试验证应用程序的“核心”，但这样的定义将单元测试视为次要的，这并不是本书中对单元测试的思考方式。单元测试是核心，是软件的关键组成部分，应该像业务逻辑一样受到同等的考虑。

单元测试是一段代码，它导入代码的部分业务逻辑，并运行其逻辑，断言几种情景，以保证特定条件。单元测试必须具有一些特征，比如：

+   隔离：单元测试应该完全独立于任何其他外部代理，并且它们必须只关注业务逻辑。因此，它们不连接到数据库，不执行 HTTP 请求等。隔离还意味着测试在彼此之间是独立的：它们必须能够以任何顺序运行，而不依赖于任何先前的状态。

+   性能：单元测试必须运行快速。它们旨在多次重复运行。

+   自我验证：单元测试的执行决定了其结果。不需要额外的步骤来解释单元测试（更不用说手动了）。

更具体地说，在 Python 中，这意味着我们将有新的`*.py`文件，我们将在其中放置我们的单元测试，并且它们将被某个工具调用。这些文件将有`import`语句，以从我们的业务逻辑中获取我们需要的内容（我们打算测试的内容），并在这个文件中编写测试本身。之后，一个工具将收集我们的单元测试并运行它们，给出一个结果。

这最后一部分实际上就是自我验证的含义。当工具调用我们的文件时，将启动一个 Python 进程，并在其上运行我们的测试。如果测试失败，进程将以错误代码退出（在 Unix 环境中，这可以是任何不等于`0`的数字）。标准是工具运行测试，并为每个成功的测试打印一个点（`.`），如果测试失败，则打印一个`F`（测试条件未满足），如果出现异常，则打印一个`E`。

# 关于其他形式的自动化测试的说明

单元测试旨在验证非常小的单元，例如函数或方法。我们希望通过单元测试达到非常详细的粒度，尽可能测试更多的代码。为了测试一个类，我们不想使用单元测试，而是使用测试套件，这是一组单元测试。每一个单元测试将测试更具体的内容，比如该类的一个方法。

这并不是唯一的单元测试形式，也不能捕捉到每一个可能的错误。还有验收测试和集成测试，都超出了本书的范围。

在集成测试中，我们希望一次测试多个组件。在这种情况下，我们希望验证它们是否集体按预期工作。在这种情况下，有副作用是可以接受的（甚至是可取的），并且可以忽略隔离，这意味着我们希望发出 HTTP 请求，连接到数据库等。

验收测试是一种自动化的测试形式，试图从用户的角度验证系统，通常执行用例。

这两种测试方式失去了单元测试的另一个优点：速度。正如你可以想象的，它们将需要更多时间来运行，因此它们将运行得更少。

在一个良好的开发环境中，程序员将拥有整个测试套件，并且在进行代码更改、迭代、重构等过程中，会一直运行单元测试。一旦更改准备就绪，并且拉取请求已经打开，持续集成服务将对该分支运行构建，其中将运行单元测试，以及可能存在的集成或验收测试。不用说，在合并之前，构建的状态应该是成功的（绿色），但重要的是测试类型之间的差异：我们希望一直运行单元测试，并且较少频繁地运行那些需要更长时间的测试。因此，我们希望有很多小的单元测试，以及一些自动化测试，策略性地设计来尽可能覆盖单元测试无法达到的地方（例如数据库）。

最后，智者之言。请记住，本书鼓励实用主义。除了本书中给出的定义和关于单元测试的观点之外，读者必须牢记，根据您的标准和背景，最佳解决方案应该占主导地位。没有人比您更了解您的系统。这意味着，如果由于某种原因，您必须编写一个需要启动 Docker 容器来针对数据库进行测试的单元测试，那就去做吧。正如我们在整本书中反复提醒的那样，实用性胜过纯粹性。

# 单元测试和敏捷软件开发

在现代软件开发中，我们希望不断地以尽可能快的速度交付价值。这些目标背后的理念是，我们获得反馈的越早，影响就越小，改变就越容易。这些并不是新的想法；其中一些类似于几十年前的制造原则，而其他一些（比如尽快从利益相关者那里获得反馈并对其进行迭代的想法）可以在《大教堂与集市》等文章中找到。

因此，我们希望能够有效地应对变化，为此，我们编写的软件将不得不改变。就像我们在前几章中提到的，我们希望我们的软件是适应性强、灵活和可扩展的。

单单代码（无论它写得多么好和设计得多么好）不能保证它足够灵活以便进行更改。假设我们按照 SOLID 原则设计了一款软件，并且在某个部分实际上有一组符合开闭原则的组件，这意味着我们可以很容易地扩展它们而不会影响太多现有的代码。进一步假设代码是以有利于重构的方式编写的，因此我们可以根据需要进行更改。当我们进行这些更改时，有什么可以证明我们没有引入任何错误？我们怎么知道现有的功能被保留了？你会对向用户发布这个新版本感到有信心吗？他们会相信新版本的工作方式与预期一样吗？

对所有这些问题的答案是，除非我们有正式的证明，否则我们无法确定。而单元测试就是这样，它是程序按照规范工作的正式证明。

因此，单元（或自动）测试作为一个安全网，给了我们在代码上工作的信心。有了这些工具，我们可以高效地工作在我们的代码上，因此这最终决定了团队在软件产品上的速度（或能力）。测试越好，我们就越有可能快速交付价值，而不会因为不时出现的错误而停滞不前。

# 单元测试和软件设计

当涉及主代码和单元测试之间的关系时，这是另一面的问题。除了在前一节中探讨的实用原因之外，它归结为良好的软件是可测试的软件。**可测试性**（决定软件易于测试程度的质量属性）不仅仅是一种美好的东西，而是对清晰代码的驱动。

单元测试不仅仅是主代码库的补充，而是对代码编写方式有直接影响和真正影响的东西。从最初意识到我们想要为代码的某些部分添加单元测试时，我们必须对其进行更改（从而得到更好的版本），到其最终表达（在本章末尾附近探讨）时，整个代码（设计）是由它将如何通过**测试驱动设计**进行测试而驱动的。

从一个简单的例子开始，我们将向您展示一个小的用例，其中测试（以及测试我们的代码的需要）导致我们编写代码的方式得到改进。

在以下示例中，我们将模拟一个需要向外部系统发送关于每个特定任务获得的结果的指标的过程（和往常一样，只要我们专注于代码，细节就不重要）。我们有一个代表领域问题上某个任务的`Process`对象，并且它使用一个`metrics`客户端（一个外部依赖，因此我们无法控制）来将实际的指标发送到外部实体（这可能是发送数据到`syslog`或`statsd`，例如）：

```py
class MetricsClient:
    """3rd-party metrics client"""

    def send(self, metric_name, metric_value):
        if not isinstance(metric_name, str):
            raise TypeError("expected type str for metric_name")

        if not isinstance(metric_value, str):
            raise TypeError("expected type str for metric_value")

        logger.info("sending %s = %s", metric_name, metric_value)

class Process:

    def __init__(self):
        self.client = MetricsClient() # A 3rd-party metrics client

    def process_iterations(self, n_iterations):
        for i in range(n_iterations):
            result = self.run_process()
            self.client.send("iteration.{}".format(i), result)
```

在第三方客户端的模拟版本中，我们规定提供的参数必须是字符串类型。因此，如果`run_process`方法的`result`不是字符串，我们可能期望它会失败，而事实上确实如此：

```py
Traceback (most recent call last):
...
    raise TypeError("expected type str for metric_value")
TypeError: expected type str for metric_value
```

记住，这种验证不在我们的控制之内，我们无法改变代码，因此在继续之前，我们必须为方法提供正确类型的参数。但由于这是我们发现的一个错误，我们首先想要编写一个单元测试，以确保它不会再次发生。我们这样做实际上是为了证明我们修复了问题，并且保护免受这个错误的影响，无论代码被重构多少次。

通过模拟`Process`对象的`client`，我们可以测试代码，但这样做会运行比需要的更多的代码（注意我们想要测试的部分嵌套在代码中）。此外，方法相对较小是件好事，因为如果不是这样，测试将不得不运行更多不需要的部分，我们可能也需要模拟。这是另一个良好设计的例子（小而紧密的函数或方法），与可测试性相关。

最后，我们决定不费太多力气，只测试我们需要的部分，所以我们不直接在`main`方法上与`client`交互，而是委托给一个`wrapper`方法，新的类看起来是这样的：

```py
class WrappedClient:

    def __init__(self):
        self.client = MetricsClient()

    def send(self, metric_name, metric_value):
        return self.client.send(str(metric_name), str(metric_value))

class Process:
    def __init__(self):
        self.client = WrappedClient()

    ... # rest of the code remains unchanged
```

在这种情况下，我们选择为指标创建我们自己的版本的`client`，也就是说，一个围绕我们以前使用的第三方库的包装器。为此，我们放置了一个类（具有相同的接口），将根据需要转换类型。

这种使用组合的方式类似于适配器设计模式（我们将在下一章中探讨设计模式，所以现在只是一个信息性的消息），而且由于这是我们领域中的一个新对象，它可以有其相应的单元测试。拥有这个对象将使测试变得更简单，但更重要的是，现在我们看到，我们意识到这可能是代码应该一开始就应该编写的方式。尝试为我们的代码编写单元测试使我们意识到我们完全错过了一个重要的抽象！

既然我们已经将方法分离出来，让我们为其编写实际的单元测试。在本例中使用的`unittest`模块的详细信息将在我们探讨测试工具和库的章节中更详细地探讨，但现在阅读代码将给我们一个关于如何测试的第一印象，并且会使之前的概念变得不那么抽象：

```py
import unittest
from unittest.mock import Mock

class TestWrappedClient(unittest.TestCase):
    def test_send_converts_types(self):
        wrapped_client = WrappedClient()
        wrapped_client.client = Mock()
        wrapped_client.send("value", 1)

        wrapped_client.client.send.assert_called_with("value", "1")
```

`Mock`是`unittest.mock`模块中可用的一种类型，它是一个非常方便的对象，可以询问各种事情。例如，在这种情况下，我们将其用于替代第三方库（模拟成系统边界，如下一节所述），以检查它是否按预期调用（再次强调，我们不测试库本身，只测试它是否被正确调用）。注意我们运行了一个类似于我们的`Process`对象的调用，但我们期望参数被转换为字符串。

# 定义要测试的边界

测试需要付出努力。如果我们在决定要测试什么时不小心，我们将永远无法结束测试，因此浪费了大量的精力而没有取得多少成果。

我们应该将测试范围限定在我们的代码边界内。如果不这样做，我们将不得不测试依赖项（外部/第三方库或模块）或我们的代码，然后测试它们各自的依赖项，依此类推，永无止境。我们不负责测试依赖关系，因此我们可以假设这些项目有自己的测试。只需测试对外部依赖的正确调用是否使用了正确的参数（这甚至可能是对补丁的可接受使用），但我们不应该投入更多的精力。

这是另一个良好软件设计的实例。如果我们在设计时小心谨慎，并清晰地定义了系统的边界（也就是说，我们设计时朝向接口，而不是会改变的具体实现，从而颠倒了对外部组件的依赖关系以减少时间耦合），那么在编写单元测试时，模拟这些接口将会更容易得多。

在良好的单元测试中，我们希望在系统的边界上打补丁，并专注于要执行的核心功能。我们不测试外部库（例如通过`pip`安装的第三方工具），而是检查它们是否被正确调用。当我们在本章后面探讨`mock`对象时，我们将回顾执行这些类型的断言的技术和工具。

# 测试框架和工具

有很多工具可以用于编写单元测试，它们都有各自的优缺点并且服务于不同的目的。但在所有工具中，有两种最有可能覆盖几乎所有场景，因此我们将本节限制在这两种工具上。

除了测试框架和测试运行库之外，通常还可以找到配置代码覆盖率的项目，它们将其用作质量指标。由于覆盖率（作为指标使用时）是误导性的，因此在了解如何创建单元测试之后，我们将讨论为什么不应轻视它。

# 单元测试的框架和库

在本节中，我们将讨论两个编写和运行单元测试的框架。第一个是`unittest`，它在 Python 的标准库中可用，而第二个`pytest`必须通过`pip`外部安装。

+   `unittest`: [`docs.python.org/3/library/unittest.html`](https://docs.python.org/3/library/unittest.html)

+   `pytest`: [`docs.pytest.org/en/latest/`](https://docs.pytest.org/en/latest/)

当涉及到为我们的代码覆盖测试场景时，`unittest`可能就足够了，因为它有很多辅助功能。然而，对于我们有多个依赖项、连接到外部系统并且可能需要打补丁对象以及定义固定参数化测试用例的更复杂的系统，`pytest`看起来更完整。

我们将使用一个小程序作为示例，以展示如何使用这两种选项进行测试，最终将帮助我们更好地了解它们之间的比较。

演示测试工具的示例是一个支持合并请求中的代码审查的版本控制工具的简化版本。我们将从以下标准开始：

+   如果至少有一个人不同意更改，合并请求将被拒绝

+   如果没有人反对，并且至少有其他两个开发人员认为合并请求是好的，它就会被批准

+   在其他情况下，它的状态是`pending`

代码可能如下所示：

```py
from enum import Enum

class MergeRequestStatus(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"

class MergeRequest:
    def __init__(self):
        self._context = {
            "upvotes": set(),
            "downvotes": set(),
        }

    @property
    def status(self):
        if self._context["downvotes"]:
            return MergeRequestStatus.REJECTED
        elif len(self._context["upvotes"]) >= 2:
            return MergeRequestStatus.APPROVED
        return MergeRequestStatus.PENDING

    def upvote(self, by_user):
        self._context["downvotes"].discard(by_user)
        self._context["upvotes"].add(by_user)

    def downvote(self, by_user):
        self._context["upvotes"].discard(by_user)
        self._context["downvotes"].add(by_user)
```

# unittest

`unittest`模块是一个很好的选择，可以开始编写单元测试，因为它提供了丰富的 API 来编写各种测试条件，并且由于它在标准库中可用，因此它非常灵活和方便。

`unittest`模块基于 JUnit（来自 Java）的概念，而 JUnit 又基于来自 Smalltalk 的单元测试的原始思想，因此它是面向对象的。因此，测试是通过对象编写的，其中检查由方法验证，并且通常通过类将测试分组到场景中。

要开始编写单元测试，我们必须创建一个从`unittest.TestCase`继承的测试类，并定义我们想要在其方法中强调的条件。这些方法应该以`test_*`开头，并且可以在内部使用从`unittest.TestCase`继承的任何方法来检查必须成立的条件。

我们可能想要验证我们的情况的一些条件的示例包括：

```py
class TestMergeRequestStatus(unittest.TestCase):

    def test_simple_rejected(self):
        merge_request = MergeRequest()
        merge_request.downvote("maintainer")
        self.assertEqual(merge_request.status, MergeRequestStatus.REJECTED)

    def test_just_created_is_pending(self):
        self.assertEqual(MergeRequest().status, MergeRequestStatus.PENDING)

    def test_pending_awaiting_review(self):
        merge_request = MergeRequest()
        merge_request.upvote("core-dev")
        self.assertEqual(merge_request.status, MergeRequestStatus.PENDING)

    def test_approved(self):
        merge_request = MergeRequest()
        merge_request.upvote("dev1")
        merge_request.upvote("dev2")

        self.assertEqual(merge_request.status, MergeRequestStatus.APPROVED)
```

单元测试的 API 提供了许多有用的比较方法，其中最常见的是`assertEquals(<actual>, <expected>[, message])`，它可以用来比较操作的结果与我们期望的值，可选地使用在错误情况下显示的消息。

另一个有用的测试方法允许我们检查是否引发了某个异常。当发生异常情况时，我们在代码中引发异常，以防止在错误的假设下进行持续处理，并且通知调用者调用的方式有问题。这是应该进行测试的逻辑的一部分，这就是这个方法的作用。

假设我们现在正在进一步扩展我们的逻辑，以允许用户关闭他们的合并请求，一旦发生这种情况，我们就不希望再进行更多的投票（在合并请求已经关闭后评估合并请求是没有意义的）。为了防止这种情况发生，我们扩展我们的代码，并在不幸的事件发生时引发异常，当有人试图对已关闭的合并请求进行投票时。

在添加了两个新状态（`OPEN`和`CLOSED`）和一个新的`close()`方法之后，我们修改了之前的投票方法，以处理此检查：

```py
class MergeRequest:
    def __init__(self):
        self._context = {
            "upvotes": set(),
            "downvotes": set(),
        }
        self._status = MergeRequestStatus.OPEN

    def close(self):
        self._status = MergeRequestStatus.CLOSED

    ...
    def _cannot_vote_if_closed(self):
        if self._status == MergeRequestStatus.CLOSED:
            raise MergeRequestException("can't vote on a closed merge 
            request")

    def upvote(self, by_user):
        self._cannot_vote_if_closed()

        self._context["downvotes"].discard(by_user)
        self._context["upvotes"].add(by_user)

    def downvote(self, by_user):
        self._cannot_vote_if_closed()

        self._context["upvotes"].discard(by_user)
        self._context["downvotes"].add(by_user)
```

现在，我们想要检查这个验证是否有效。为此，我们将使用`asssertRaises`和`assertRaisesRegex`方法：

```py
    def test_cannot_upvote_on_closed_merge_request(self):
        self.merge_request.close()
        self.assertRaises(
            MergeRequestException, self.merge_request.upvote, "dev1"
        )

    def test_cannot_downvote_on_closed_merge_request(self):
        self.merge_request.close()
        self.assertRaisesRegex(
            MergeRequestException,
            "can't vote on a closed merge request",
            self.merge_request.downvote,
            "dev1",
        )
```

前者期望在调用第二个参数中的可调用对象时引发提供的异常，使用函数的其余部分的参数（`*args`和`**kwargs`），如果不是这种情况，它将失败，并表示预期引发的异常未被引发。后者也是如此，但它还检查引发的异常是否包含与提供的正则表达式匹配的消息。即使引发了异常，但消息不同（不匹配正则表达式），测试也会失败。

尝试检查错误消息，因为异常不仅会更准确地进行额外检查，确保实际上触发了我们想要的异常，还会检查是否另一个相同类型的异常偶然发生。

# 参数化测试

现在，我们想要测试合并请求的阈值接受如何工作，只需提供`context`的数据样本，而不需要整个`MergeRequest`对象。我们想要测试`status`属性的部分，即在检查它是否关闭之后的部分，但是独立地。

实现这一目标的最佳方法是将该组件分离为另一个类，使用组合，然后继续使用自己的测试套件测试这个新的抽象：

```py
class AcceptanceThreshold:
    def __init__(self, merge_request_context: dict) -> None:
        self._context = merge_request_context

    def status(self):
        if self._context["downvotes"]:
            return MergeRequestStatus.REJECTED
        elif len(self._context["upvotes"]) >= 2:
            return MergeRequestStatus.APPROVED
        return MergeRequestStatus.PENDING

class MergeRequest:
    ...
    @property
    def status(self):
        if self._status == MergeRequestStatus.CLOSED:
            return self._status

        return AcceptanceThreshold(self._context).status()
```

有了这些变化，我们可以再次运行测试并验证它们是否通过，这意味着这次小的重构没有破坏当前功能（单元测试确保回归）。有了这一点，我们可以继续实现编写特定于新类的测试的目标：

```py
class TestAcceptanceThreshold(unittest.TestCase):
    def setUp(self):
        self.fixture_data = (
            (
                {"downvotes": set(), "upvotes": set()},
                MergeRequestStatus.PENDING
            ),
            (
                {"downvotes": set(), "upvotes": {"dev1"}},
                MergeRequestStatus.PENDING,
            ),
            (
                {"downvotes": "dev1", "upvotes": set()},
                MergeRequestStatus.REJECTED
            ),
            (
                {"downvotes": set(), "upvotes": {"dev1", "dev2"}},
                MergeRequestStatus.APPROVED
            ),
        )

    def test_status_resolution(self):
        for context, expected in self.fixture_data:
            with self.subTest(context=context):
                status = AcceptanceThreshold(context).status()
                self.assertEqual(status, expected)
```

在`setUp()`方法中，我们定义了要在整个测试中使用的数据装置。在这种情况下，实际上并不需要，因为我们可以直接放在方法中，但是如果我们希望在执行任何测试之前运行一些代码，这就是写入的地方，因为这个方法在每次运行测试之前都会被调用一次。

通过编写代码的新版本，被测试代码下的参数更清晰更紧凑，并且在每种情况下都会报告结果。

为了模拟我们正在运行所有参数，测试会遍历所有数据，并对每个实例执行代码。这里一个有趣的辅助方法是使用`subTest`，在这种情况下，我们使用它来标记被调用的测试条件。如果其中一个迭代失败，`unittest`会报告相应的变量值，这些变量被传递给`subTest`（在这种情况下，它被命名为`context`，但任何一系列关键字参数都可以起到同样的作用）。例如，一个错误可能看起来像这样：

```py
FAIL: (context={'downvotes': set(), 'upvotes': {'dev1', 'dev2'}})
----------------------------------------------------------------------
Traceback (most recent call last):
  File "" test_status_resolution
    self.assertEqual(status, expected)
AssertionError: <MergeRequestStatus.APPROVED: 'approved'> != <MergeRequestStatus.REJECTED: 'rejected'>
```

如果选择参数化测试，请尽量提供每个参数实例的上下文信息，以便更容易进行调试。

# pytest

Pytest 是一个很棒的测试框架，可以通过`pip install pytest`进行安装。与`unittest`相比的一个区别是，虽然仍然可以将测试场景分类为类，并创建我们测试的面向对象模型，但这并不是强制性的，也可以通过使用`assert`语句来写更少的样板代码进行单元测试。

默认情况下，使用`assert`语句进行比较就足以让`pytest`识别单元测试并相应地报告其结果。还可以使用包中的特定函数进行更高级的用法，但这需要使用特定的函数。

一个很好的特性是命令`pytests`将运行它能够发现的所有测试，即使它们是用`unittest`编写的。这种兼容性使得逐渐从`unittest`过渡到`pytest`变得更容易。

# 使用 pytest 进行基本测试用例

我们在上一节中测试的条件可以用`pytest`中的简单函数重写。

一些简单断言的示例如下：

```py
def test_simple_rejected():
    merge_request = MergeRequest()
    merge_request.downvote("maintainer")
    assert merge_request.status == MergeRequestStatus.REJECTED

def test_just_created_is_pending():
    assert MergeRequest().status == MergeRequestStatus.PENDING

def test_pending_awaiting_review():
    merge_request = MergeRequest()
    merge_request.upvote("core-dev")
    assert merge_request.status == MergeRequestStatus.PENDING
```

布尔相等比较不需要更多的简单断言语句，而其他类型的检查，比如异常的检查需要我们使用一些函数：

```py
def test_invalid_types():
    merge_request = MergeRequest()
    pytest.raises(TypeError, merge_request.upvote, {"invalid-object"})

def test_cannot_vote_on_closed_merge_request():
    merge_request = MergeRequest()
    merge_request.close()
    pytest.raises(MergeRequestException, merge_request.upvote, "dev1")
    with pytest.raises(
        MergeRequestException,
        match="can't vote on a closed merge request",
    ):
        merge_request.downvote("dev1")
```

在这种情况下，`pytest.raises`相当于`unittest.TestCase.assertRaises`，它也接受作为方法和上下文管理器调用。如果我们想检查异常的消息，而不是使用不同的方法（如`assertRaisesRegex`），则必须使用相同的函数，但作为上下文管理器，并提供`match`参数与我们想要识别的表达式。

`pytest`还会将原始异常包装成一个自定义异常，可以通过检查其属性（例如`.value`）来预期，以便在需要检查更多条件时使用，但这个函数的使用覆盖了绝大多数情况。

# 参数化测试

使用`pytest`运行参数化测试更好，不仅因为它提供了更清晰的 API，而且因为每个测试与其参数的组合都会生成一个新的测试用例。

为了使用这个，我们必须在我们的测试上使用`pytest.mark.parametrize`装饰器。装饰器的第一个参数是一个字符串，指示要传递给`test`函数的参数的名称，第二个参数必须是可迭代的，包含这些参数的相应值。

注意测试函数的主体如何被简化为一行（在移除内部`for`循环和其嵌套的上下文管理器后），并且每个测试用例的数据都正确地与函数的主体隔离开来，这样更容易扩展和维护：

```py
@pytest.mark.parametrize("context,expected_status", (
    (
        {"downvotes": set(), "upvotes": set()},
        MergeRequestStatus.PENDING
    ),
    (
        {"downvotes": set(), "upvotes": {"dev1"}},
        MergeRequestStatus.PENDING,
    ),
    (
        {"downvotes": "dev1", "upvotes": set()},
        MergeRequestStatus.REJECTED
    ),
    (
        {"downvotes": set(), "upvotes": {"dev1", "dev2"}},
        MergeRequestStatus.APPROVED
    ),
))
def test_acceptance_threshold_status_resolution(context, expected_status):
    assert AcceptanceThreshold(context).status() == expected_status
```

使用`@pytest.mark.parametrize`来消除重复，尽可能使测试主体保持内聚，并明确指定代码必须支持的参数（测试输入或场景）。

# Fixture

`pytest`的一个很棒的功能是它如何促进创建可重用的功能，这样我们可以有效地测试数据或对象，而不需要重复。

例如，我们可能想要创建一个处于特定状态的`MergeRequest`对象，并在多个测试中使用该对象。我们通过创建一个函数并应用`@pytest.fixture`装饰器来将我们的对象定义为 fixture。想要使用该 fixture 的测试将必须具有与定义的函数相同名称的参数，`pytest`将确保提供它： 

```py
@pytest.fixture
def rejected_mr():
    merge_request = MergeRequest()

    merge_request.downvote("dev1")
    merge_request.upvote("dev2")
    merge_request.upvote("dev3")
    merge_request.downvote("dev4")

    return merge_request

def test_simple_rejected(rejected_mr):
    assert rejected_mr.status == MergeRequestStatus.REJECTED

def test_rejected_with_approvals(rejected_mr):
    rejected_mr.upvote("dev2")
    rejected_mr.upvote("dev3")
    assert rejected_mr.status == MergeRequestStatus.REJECTED

def test_rejected_to_pending(rejected_mr):
    rejected_mr.upvote("dev1")
    assert rejected_mr.status == MergeRequestStatus.PENDING

def test_rejected_to_approved(rejected_mr):
    rejected_mr.upvote("dev1")
    rejected_mr.upvote("dev2")
    assert rejected_mr.status == MergeRequestStatus.APPROVED
```

记住，测试也会影响主要代码，因此干净代码的原则也适用于它们。在这种情况下，我们在之前章节中探讨过的**不要重复自己**（**DRY**）原则再次出现，我们可以借助`pytest`的 fixture 来实现它。

除了创建多个对象或公开将在整个测试套件中使用的数据之外，还可以使用它们来设置一些条件，例如全局修补一些不希望被调用的函数，或者当我们希望使用修补对象时。

# 代码覆盖率

测试运行器支持覆盖插件（通过`pip`安装）将提供有关测试运行时执行了代码的哪些行的有用信息。这些信息对我们非常有帮助，以便我们知道代码的哪些部分需要被测试覆盖，并确定需要进行的改进（无论是在生产代码中还是在测试中）。其中最广泛使用的库之一是`coverage`（[`pypi.org/project/coverage/`](https://pypi.org/project/coverage/)）。

虽然它们非常有帮助（我们强烈建议您使用它们并配置您的项目在运行测试时在 CI 中运行覆盖），但它们也可能会产生误导；特别是在 Python 中，如果我们不仔细阅读覆盖报告，就会产生错误的印象。

# 设置其余覆盖

在`pytest`的情况下，我们必须安装`pytest-cov`软件包（在撰写本书时，本书使用的是版本`2.5.1`）。安装后，当运行测试时，我们必须告诉`pytest`运行器也将运行`pytest-cov`，以及应该覆盖哪个软件包（以及其他参数和配置）。

该软件包支持多种配置，如不同类型的输出格式，并且很容易将其与任何 CI 工具集成，但在所有这些功能中，一个强烈推荐的选项是设置标志，告诉我们哪些行尚未被测试覆盖，因为这将帮助我们诊断我们的代码，并允许我们开始编写更多的测试。

为了向您展示这是什么样子，使用以下命令：

```py
pytest \
    --cov-report term-missing \
    --cov=coverage_1 \
    test_coverage_1.py
```

这将产生类似以下的输出：

```py
test_coverage_1.py ................ [100%]

----------- coverage: platform linux, python 3.6.5-final-0 -----------
Name         Stmts Miss Cover Missing
---------------------------------------------
coverage_1.py 38      1  97%    53
```

在这里，它告诉我们有一行没有单元测试，因此我们可以查看并了解如何为其编写单元测试。这是一个常见的情况，我们意识到为了覆盖这些缺失的行，我们需要通过创建更小的方法来重构代码。结果，我们的代码看起来会好得多，就像我们在本章开头看到的例子一样。

问题在于相反的情况——我们能相信高覆盖率吗？这是否意味着我们的代码是正确的？不幸的是，拥有良好的测试覆盖率是必要的，但不足以保证代码的清洁。对代码的某些部分没有测试显然是不好的。拥有测试实际上是非常好的（我们可以说对于已经存在的测试），并且实际上断言了它们是代码质量的保证。然而，我们不能说这就是所有需要的；尽管覆盖率很高，但仍需要更多的测试。

这些是测试覆盖率的注意事项，我们将在下一节中提到。

# 测试覆盖的注意事项

Python 是解释性的，而覆盖工具利用这一点来识别在测试运行时被解释（运行）的行。然后它会在最后报告这一点。一行被解释并不意味着它被正确测试了，这就是为什么我们应该仔细阅读最终的覆盖报告并信任它所说的内容。

这实际上对于任何语言都是正确的。执行了一行代码并不意味着它已经经历了所有可能的组合。所有分支在提供的数据下成功运行只意味着代码支持了该组合，但这并不能告诉我们任何其他可能导致程序崩溃的参数组合。

使用覆盖作为发现代码中盲点的工具，而不是作为度量标准或目标。

# 模拟对象

有些情况下，我们的代码不是在测试环境中唯一存在的东西。毕竟，我们设计和构建的系统必须做一些真实的事情，这通常意味着连接到外部服务（数据库、存储服务、外部 API、云服务等）。因为它们需要具有这些副作用，它们是不可避免的。尽管我们抽象我们的代码，朝着接口编程，并且隔离代码以最小化副作用，但它们会出现在我们的测试中，我们需要一种有效的方式来处理它们。

`模拟`对象是防止不良副作用的最佳策略之一。我们的代码可能需要执行 HTTP 请求或发送通知电子邮件，但我们肯定不希望这些事件发生在我们的单元测试中。此外，单元测试应该运行得很快，因为我们希望经常运行它们（实际上是一直），这意味着我们不能承受延迟。因此，真正的单元测试不使用任何实际服务——它们不连接到任何数据库，不发出 HTTP 请求，基本上除了执行生产代码的逻辑之外什么都不做。

我们需要执行这些操作的测试，但它们不是单元测试。集成测试应该以更广泛的视角测试功能，几乎模仿用户的行为。但它们不快。因为它们连接到外部系统和服务，所以运行时间更长，成本更高。通常，我们希望有大量的单元测试能够快速运行，以便一直运行它们，而集成测试则较少运行（例如，在任何新的合并请求上）。

虽然模拟对象很有用，但滥用它们的使用范围介于代码异味和反模式之间是我们在深入讨论之前想要提到的第一个警告。

# 关于修补和模拟的公平警告

我们之前说过，单元测试帮助我们编写更好的代码，因为我们想要开始测试代码的部分时，通常必须编写可测试的代码，这通常意味着它们也是内聚的、细粒度的和小的。这些都是软件组件中具有的良好特性。

另一个有趣的收获是，测试将帮助我们注意到代码中存在代码异味的地方。我们的代码存在代码异味的主要警告之一是，我们发现自己试图 monkey-patch（或模拟）许多不同的东西，只是为了覆盖一个简单的测试用例。

`unittest`模块提供了一个在`unittest.mock.patch`中修补对象的工具。修补意味着原始代码（由导入时指定其位置的字符串给出）将被其他东西替换，而不是其原始代码，默认情况下是模拟对象。这会在运行时替换代码，并且有一个缺点，即我们失去了原始代码的联系，使我们的测试变得更加肤浅。它还带来了性能考虑，因为在运行时修改对象会带来开销，并且如果我们重构代码并移动事物，这可能会导致更新。

在我们的测试中使用 monkey-patching 或模拟可能是可以接受的，而且本身并不代表一个问题。另一方面，滥用 monkey-patching 确实是一个标志，表明我们的代码需要改进。

# 使用模拟对象

在单元测试术语中，有几种对象属于名为**测试替身**的类别。测试替身是一种对象，它将以不同种类的原因在我们的测试套件中代替真实对象（也许我们不需要实际的生产代码，而只需要一个虚拟对象，或者我们不能使用它，因为它需要访问服务或者它具有我们不希望在单元测试中出现的副作用等）。

有不同类型的测试替身，例如虚拟对象、存根、间谍或模拟。模拟是最一般的对象类型，由于它们非常灵活和多功能，因此适用于所有情况，而无需详细了解其他情况。正因为如此，标准库还包括了这种类型的对象，并且在大多数 Python 程序中都很常见。这就是我们将在这里使用的：`unittest.mock.Mock`。

**模拟**是一种根据规范创建的对象类型（通常类似于生产类的对象）和一些配置的响应（也就是说，我们可以告诉模拟在某些调用时应该返回什么，并且它的行为应该是什么）。然后，“模拟”对象将记录其内部状态的一部分，例如它是如何被调用的（使用了什么参数，多少次等），我们可以使用该信息在以后的阶段验证我们应用程序的行为。

在 Python 的情况下，标准库中提供的`Mock`对象提供了一个很好的 API，可以进行各种行为断言，例如检查模拟调用了多少次，使用了什么参数等。

# 模拟的类型

标准库在`unittest.mock`模块中提供了`Mock`和`MagicMock`对象。前者是一个可以配置为返回任何值并将跟踪对其进行的调用的测试替身。后者也是如此，但它还支持魔术方法。这意味着，如果我们编写了使用魔术方法的成语代码（并且我们正在测试的代码的某些部分将依赖于它），那么我们可能必须使用`MagicMock`实例而不仅仅是`Mock`。

当我们的代码需要调用魔术方法时，尝试使用`Mock`将导致错误。请参阅以下代码，以了解此示例：

```py
class GitBranch:
    def __init__(self, commits: List[Dict]):
        self._commits = {c["id"]: c for c in commits}

    def __getitem__(self, commit_id):
        return self._commits[commit_id]

    def __len__(self):
        return len(self._commits)

def author_by_id(commit_id, branch):
    return branch[commit_id]["author"]
```

我们想测试这个函数；但是，另一个测试需要调用`author_by_id`函数。由于某种原因，因为我们没有测试该函数，提供给该函数（并返回）的任何值都将是好的：

```py
def test_find_commit():
    branch = GitBranch([{"id": "123", "author": "dev1"}])
    assert author_by_id("123", branch) == "dev1"

def test_find_any():
    author = author_by_id("123", Mock()) is not None
    # ... rest of the tests..
```

正如预期的那样，这不起作用：

```py
def author_by_id(commit_id, branch):
    > return branch[commit_id]["author"]
    E TypeError: 'Mock' object is not subscriptable
```

使用`MagicMock`将起作用。我们甚至可以配置此类型模拟的魔术方法，以返回我们需要的内容，以便控制我们测试的执行：

```py
def test_find_any():
    mbranch = MagicMock()
    mbranch.__getitem__.return_value = {"author": "test"}
    assert author_by_id("123", mbranch) == "test"
```

# 测试替身的用例

为了看到模拟的可能用途，我们需要向我们的应用程序添加一个新组件，该组件将负责通知“构建”“状态”的合并请求。当“构建”完成时，将使用合并请求的 ID 和“构建”的“状态”调用此对象，并通过向特定的固定端点发送 HTTP`POST`请求来使用此信息更新合并请求的“状态”：

```py
# mock_2.py

from datetime import datetime

import requests
from constants import STATUS_ENDPOINT

class BuildStatus:
    """The CI status of a pull request."""

    @staticmethod
    def build_date() -> str:
        return datetime.utcnow().isoformat()

    @classmethod
    def notify(cls, merge_request_id, status):
        build_status = {
            "id": merge_request_id,
            "status": status,
            "built_at": cls.build_date(),
        }
        response = requests.post(STATUS_ENDPOINT, json=build_status)
        response.raise_for_status()
        return response

```

这个类有很多副作用，但其中一个是一个重要的难以克服的外部依赖。如果我们试图在不修改任何内容的情况下对其进行测试，那么它将在尝试执行 HTTP 连接时立即失败并出现连接错误。

作为测试目标，我们只想确保信息被正确组成，并且库请求是使用适当的参数进行调用的。由于这是一个外部依赖项，我们不测试请求；只需检查它是否被正确调用就足够了。

当尝试比较发送到库的数据时，我们将面临另一个问题，即该类正在计算当前时间戳，这在单元测试中是不可能预测的。直接修补`datetime`是不可能的，因为该模块是用 C 编写的。有一些外部库可以做到这一点（例如`freezegun`），但它们会带来性能损耗，并且对于这个例子来说会过度。因此，我们选择将我们想要的功能封装在一个静态方法中，以便我们可以修补它。

现在我们已经确定了代码中需要替换的要点，让我们编写单元测试：

```py
# test_mock_2.py

from unittest import mock

from constants import STATUS_ENDPOINT
from mock_2 import BuildStatus

@mock.patch("mock_2.requests")
def test_build_notification_sent(mock_requests):
    build_date = "2018-01-01T00:00:01"
    with mock.patch("mock_2.BuildStatus.build_date", 
    return_value=build_date):
        BuildStatus.notify(123, "OK")

    expected_payload = {"id": 123, "status": "OK", "built_at": 
    build_date}
    mock_requests.post.assert_called_with(
        STATUS_ENDPOINT, json=expected_payload
    )
```

首先，我们使用`mock.patch`作为装饰器来替换`requests`模块。这个函数的结果将创建一个`mock`对象，将作为参数传递给测试（在这个例子中命名为`mock_requests`）。然后，我们再次使用这个函数，但这次作为上下文管理器，来改变计算“构建”日期的类的方法的返回值，用我们控制的值替换它，我们将在断言中使用。

一旦我们把所有这些都放在那里，我们就可以用一些参数调用类方法，然后我们可以使用`mock`对象来检查它是如何被调用的。在这种情况下，我们使用这个方法来查看`requests.post`是否确实以我们想要的参数被调用。

这是模拟的一个很好的特性——它们不仅限制了所有外部组件的范围（在这种情况下，以防止实际发送一些通知或发出 HTTP 请求），而且还提供了一个有用的 API 来验证调用及其参数。

在这种情况下，我们能够通过设置相应的“模拟”对象来测试代码，但事实上，与主要功能的总代码行数相比，我们不得不进行相当多的补丁。关于被测试的纯生产代码与我们必须模拟的代码部分之间的比例没有明确的规则，但是通过运用常识，我们可以看到，如果我们不得不在相同的部分进行相当多的补丁，那么某些东西并没有被清晰地抽象出来，看起来像是代码异味。

在下一节中，我们将探讨如何重构代码来解决这个问题。

# 重构

**重构**是软件维护中的一个关键活动，但如果没有单元测试，就不能做到（至少是正确的）。我们时不时需要支持一个新功能或以意想不到的方式使用我们的软件。我们需要意识到，满足这些要求的唯一方法是首先重构我们的代码，使其更通用。只有这样，我们才能继续前进。

通常，在重构我们的代码时，我们希望改进其结构，使其更好，有时更通用，更可读，或更灵活。挑战在于在实现这些目标的同时保持与修改之前完全相同的功能。这意味着，在我们重构的组件的客户眼中，可能根本没有发生任何事情。

必须支持与之前相同的功能，但使用不同版本的代码这一约束意味着我们需要对修改过的代码运行回归测试。运行回归测试的唯一经济有效的方法是自动化。自动化测试的最经济有效的版本是单元测试。

# 改进我们的代码

在前面的例子中，我们能够将代码的副作用与我们无法在单元测试中控制的部分分离出来，通过对依赖于这些部分的代码进行补丁，使其可测试。这是一个很好的方法，因为毕竟，`mock.patch`函数对于这些任务来说非常方便，可以替换我们告诉它的对象，给我们一个`Mock`对象。

这样做的缺点是，我们必须提供我们将要模拟的对象的路径，包括模块，作为一个字符串。这有点脆弱，因为如果我们重构我们的代码（比如说我们重命名文件或将其移动到其他位置），所有的补丁位置都必须更新，否则测试将会失败。

在这个例子中，`notify()`方法直接依赖于一个实现细节（`requests`模块），这是一个设计问题，也就是说，它也对单元测试产生了上述的脆弱性。

我们仍然需要用双重对象（模拟）替换这些方法，但如果我们重构代码，我们可以以更好的方式来做。让我们将这些方法分开成更小的方法，最重要的是注入依赖，而不是固定它。现在代码应用了依赖反转原则，并且期望与支持接口的东西一起工作（在这个例子中是隐式的），比如`requests`模块提供的接口：

```py
from datetime import datetime

from constants import STATUS_ENDPOINT

class BuildStatus:

    endpoint = STATUS_ENDPOINT

    def __init__(self, transport):
        self.transport = transport

    @staticmethod
    def build_date() -> str:
        return datetime.utcnow().isoformat()

    def compose_payload(self, merge_request_id, status) -> dict:
        return {
            "id": merge_request_id,
            "status": status,
            "built_at": self.build_date(),
        }

    def deliver(self, payload):
        response = self.transport.post(self.endpoint, json=payload)
        response.raise_for_status()
        return response

    def notify(self, merge_request_id, status):
        return self.deliver(self.compose_payload(merge_request_id, status))
```

我们将方法分开（不再是 notify，而是 compose + deliver），创建了一个新的`compose_payload()`方法（这样我们可以替换，而不需要打补丁类），并要求注入`transport`依赖。现在`transport`是一个依赖项，更容易更改该对象为我们想要的任何双重对象。

甚至可以暴露这个对象的一个 fixture，并根据需要替换双重对象：

```py
@pytest.fixture
def build_status():
    bstatus = BuildStatus(Mock())
    bstatus.build_date = Mock(return_value="2018-01-01T00:00:01")
    return bstatus

def test_build_notification_sent(build_status):

    build_status.notify(1234, "OK")

    expected_payload = {
        "id": 1234,
        "status": "OK",
        "built_at": build_status.build_date(),
    }

```

```py
    build_status.transport.post.assert_called_with(
        build_status.endpoint, json=expected_payload
    )
```

# 生产代码并不是唯一在演变的东西

我们一直在说单元测试和生产代码一样重要。如果我们对生产代码足够小心以创建最佳的抽象，为什么我们不为单元测试做同样的事呢？

如果单元测试的代码和主要代码一样重要，那么设计时一定要考虑可扩展性，并尽可能使其易于维护。毕竟，这段代码将由原作者以外的工程师来维护，因此必须易读。

我们如此重视代码的灵活性的原因是，我们知道需求会随着时间的推移而改变和演变，最终随着领域业务规则的变化，我们的代码也将不得不改变以支持这些新需求。由于生产代码已经改变以支持新需求，测试代码也将不得不改变以支持生产代码的新版本。

在我们最初的示例中，我们为合并请求对象创建了一系列测试，尝试不同的组合并检查合并请求的状态。这是一个很好的第一步，但我们可以做得更好。

一旦我们更好地理解了问题，我们就可以开始创建更好的抽象。首先想到的是，我们可以创建一个检查特定条件的更高级抽象。例如，如果我们有一个专门针对`MergeRequest`类的测试套件对象，我们知道其功能将局限于这个类的行为（因为它应该符合 SRP），因此我们可以在这个测试类上创建特定的测试方法。这些方法只对这个类有意义，但可以帮助减少大量样板代码。

我们可以创建一个封装这一结构的断言的方法，并在所有测试中重复使用它，而不是重复断言：

```py
class TestMergeRequestStatus(unittest.TestCase):
    def setUp(self):
        self.merge_request = MergeRequest()

    def assert_rejected(self):
        self.assertEqual(
            self.merge_request.status, MergeRequestStatus.REJECTED
        )

    def assert_pending(self):
        self.assertEqual(
            self.merge_request.status, MergeRequestStatus.PENDING
        )

    def assert_approved(self):
        self.assertEqual(
            self.merge_request.status, MergeRequestStatus.APPROVED
        )

    def test_simple_rejected(self):
        self.merge_request.downvote("maintainer")
        self.assert_rejected()

    def test_just_created_is_pending(self):
        self.assert_pending()
```

如果合并请求的状态检查发生变化（或者我们想要添加额外的检查），只有一个地方（`assert_approved()`方法）需要修改。更重要的是，通过创建这些更高级的抽象，最初只是单元测试的代码开始演变成可能最终成为具有自己 API 或领域语言的测试框架，使测试更具有声明性。

# 更多关于单元测试

通过我们迄今为止重新审视的概念，我们知道如何测试我们的代码，考虑我们的设计将如何进行测试，并配置项目中的工具来运行自动化测试，这将使我们对所编写软件的质量有一定程度的信心。

如果我们对代码的信心是由编写在其上的单元测试所决定的，那么我们如何知道它们足够了？我们怎么能确定我们已经在测试场景上经历了足够多的测试，而且没有漏掉一些测试？谁说这些测试是正确的？也就是说，谁来测试这些测试？

关于我们编写的测试是否彻底的问题的第一部分，通过基于属性的测试来超越我们的测试努力来回答。

问题的第二部分可能会有不同的观点给出多个答案，但我们将简要提到变异测试作为确定我们的测试确实是正确的手段。在这方面，我们认为单元测试检查我们的主要生产代码，这也对单元测试起到了控制作用。

# 基于属性的测试

基于属性的测试包括生成测试用例的数据，目的是找到会使代码失败的情景，而这些情景在我们之前的单元测试中没有涵盖。

这个主要的库是`hypothesis`，它与我们的单元测试一起配置，将帮助我们找到会使我们的代码失败的问题数据。

我们可以想象这个库的作用是找到我们代码的反例。我们编写我们的生产代码（以及针对它的单元测试！），并声称它是正确的。现在，通过这个库，我们定义了一些必须满足我们代码的`hypothesis`，如果有一些情况下我们的断言不成立，`hypothesis`将提供一组导致错误的数据。

单元测试最好的一点是它让我们更加深入地思考我们的生产代码。`hypothesis`最好的一点是它让我们更加深入地思考我们的单元测试。

# 变异测试

我们知道测试是我们确保代码正确的正式验证方法。那么是什么确保测试是正确的呢？你可能会想到生产代码，是的，在某种程度上这是正确的，我们可以将主要代码视为对我们测试的一个平衡。

编写单元测试的重点在于我们正在保护自己免受错误的侵害，并测试我们真的不希望在生产中发生的失败场景。测试通过是好事，但如果它们通过了错误的原因就不好了。也就是说，我们可以将单元测试用作自动回归工具——如果有人在代码中引入了错误，我们期望我们的至少一个测试能够捕捉到并失败。如果这没有发生，要么是缺少了一个测试，要么是我们已有的测试没有进行正确的检查。

这就是变异测试的理念。使用变异测试工具，代码将被修改为新版本（称为变异体），这些变异体是原始代码的变体，但其中一些逻辑被改变了（例如，操作符被交换，条件被倒置等）。一个良好的测试套件应该能够捕捉到这些变异体并将其消灭，这意味着我们可以依赖这些测试。如果一些变异体在实验中幸存下来，通常这是一个不好的迹象。当然，这并不是完全精确的，所以有一些中间状态我们可能想要忽略。

为了快速向您展示这是如何工作的，并让您对此有一个实际的想法，我们将使用一个不同版本的代码来计算合并请求的状态，这是基于批准和拒绝的数量。这一次，我们已经改变了代码，改为一个简单版本，根据这些数字返回结果。我们已经将包含状态常量的枚举移到一个单独的模块中，所以现在看起来更加紧凑：

```py
# File mutation_testing_1.py
from mrstatus import MergeRequestStatus as Status

def evaluate_merge_request(upvote_count, downvotes_count):
    if downvotes_count > 0:
        return Status.REJECTED
    if upvote_count >= 2:
        return Status.APPROVED
    return Status.PENDING
```

现在我们将添加一个简单的单元测试，检查其中一个条件及其预期的“结果”：

```py
# file: test_mutation_testing_1.py
class TestMergeRequestEvaluation(unittest.TestCase):
    def test_approved(self):
        result = evaluate_merge_request(3, 0)
        self.assertEqual(result, Status.APPROVED)
```

现在，我们将安装`mutpy`，一个用于 Python 的变异测试工具，使用`pip install mutpy`，并告诉它使用这些测试运行此模块的变异测试：

```py
$ mut.py \
    --target mutation_testing_$N \
    --unit-test test_mutation_testing_$N \
    --operator AOD `# delete arithmetic operator` \
    --operator AOR `# replace arithmetic operator` \
    --operator COD `# delete conditional operator` \
    --operator COI `# insert conditional operator` \
    --operator CRP `# replace constant` \
    --operator ROR `# replace relational operator` \
    --show-mutants
```

结果将会看起来类似于这样：

```py
[*] Mutation score [0.04649 s]: 100.0%
 - all: 4
 - killed: 4 (100.0%)
 - survived: 0 (0.0%)
 - incompetent: 0 (0.0%)
 - timeout: 0 (0.0%)
```

这是一个好迹象。让我们拿一个特定的实例来分析发生了什么。输出中的一行显示了以下变异体：

```py
 - [# 1] ROR mutation_testing_1:11 : 
------------------------------------------------------
 7: from mrstatus import MergeRequestStatus as Status
 8: 
 9: 
 10: def evaluate_merge_request(upvote_count, downvotes_count):
~11:     if downvotes_count < 0:
 12:         return Status.REJECTED
 13:     if upvote_count >= 2:
 14:         return Status.APPROVED
 15:     return Status.PENDING
------------------------------------------------------
[0.00401 s] killed by test_approved (test_mutation_testing_1.TestMergeRequestEvaluation)
```

请注意，这个变异体由原始版本和第 11 行中操作符改变（`>`改为`<`）组成，结果告诉我们这个变异体被测试杀死了。这意味着使用这个代码版本（假设有人错误地进行了这个更改），函数的结果将是`APPROVED`，而测试期望它是`REJECTED`，所以测试失败，这是一个好迹象（测试捕捉到了引入的错误）。

变异测试是确保单元测试质量的一种好方法，但它需要一些努力和仔细的分析。在复杂的环境中使用这个工具，我们将不得不花一些时间分析每个场景。同样，运行这些测试是昂贵的，因为它需要运行不同版本的代码，这可能会占用太多资源并且可能需要更长的时间来完成。然而，手动进行这些检查会更加昂贵，并且需要更多的努力。不进行这些检查可能会更加危险，因为我们会危及测试的质量。

# 测试驱动开发简介

有一些专门讲述 TDD 的书籍，所以在这本书中全面涵盖这个话题是不现实的。然而，这是一个非常重要的话题，必须提到。

TDD 的理念是在编写生产代码之前编写测试，以便生产代码只是为了响应由于功能缺失而失败的测试而编写的。

我们希望先编写测试，然后编写代码的原因有多个。从实用的角度来看，我们会相当准确地覆盖我们的生产代码。由于所有的生产代码都是为了响应单元测试而编写的，很少会有功能缺失的测试（当然这并不意味着有 100%的覆盖率，但至少所有的主要函数、方法或组件都会有各自的测试，即使它们并不完全覆盖）。

这个工作流程很简单，高层次上包括三个步骤。首先，我们编写一个描述需要实现的单元测试。当我们运行这个测试时，它会失败，因为这个功能还没有被实现。然后，我们开始实现满足条件的最小代码，并再次运行测试。这次，测试应该通过。现在，我们可以改进（重构）代码。

这个循环被称为著名的**红-绿-重构**，意思是一开始测试失败（红色），然后我们让它们通过（绿色），然后我们进行重构并迭代。

# 总结

单元测试是一个非常有趣和深刻的话题，但更重要的是，它是清晰代码的关键部分。最终，单元测试决定了代码的质量。单元测试通常作为代码的镜子——当代码易于测试时，它是清晰和正确设计的，这将反映在单元测试中。

单元测试的代码和生产代码一样重要。所有适用于生产代码的原则也适用于单元测试。这意味着它们应该以同样的努力和深思熟虑来设计和维护。如果我们不关心我们的单元测试，它们将开始出现问题并变得有缺陷（或有问题），结果就是无用的。如果发生这种情况，它们很难维护，就会成为一个负担，这会使情况变得更糟，因为人们会倾向于忽视它们或完全禁用它们。这是最糟糕的情况，因为一旦发生这种情况，整个生产代码就会受到威胁。盲目前进（没有单元测试）是一种灾难。

幸运的是，Python 提供了许多用于单元测试的工具，无论是在标准库中还是通过`pip`可用。它们非常有帮助，花时间配置它们确实会在长远来看得到回报。

我们已经看到单元测试作为程序的正式规范以及软件按照规范工作的证明，我们也了解到在发现新的测试场景时，总是有改进的空间，我们总是可以创建更多的测试。在这个意义上，用不同的方法（比如基于属性的测试或变异测试）扩展我们的单元测试是一个很好的投资。

# 参考资料

以下是您可以参考的信息列表：

+   Python 标准库的`unittest`模块包含了如何开始构建测试套件的全面文档（[`docs.python.org/3/library/unittest.html`](https://docs.python.org/3/library/unittest.html)）

+   Hypothesis 官方文档（[`hypothesis.readthedocs.io/en/latest/`](https://hypothesis.readthedocs.io/en/latest/)）

+   `pytest`官方文档（[`docs.pytest.org/en/latest/`](https://docs.pytest.org/en/latest/)）

+   《大教堂与集市：关于 Linux 和开源的思考》（*CatB*），作者 Eric S. Raymond（出版商 O'Reilly Media，1999）
