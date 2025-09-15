

# 第八章：单元测试和重构

本章探讨的思想在全球范围内是本书的基本支柱，因为它们对我们最终目标的重要性：编写更好、更易于维护的软件。

单元测试（以及任何形式的自动测试）对于软件的可维护性至关重要，因此任何质量项目都不能缺少。正因为如此，本章专门致力于自动化测试作为关键策略的各个方面，以确保安全地修改代码，并逐步迭代出更好的版本。

在本章之后，我们将对以下内容有更深入的了解：

+   为什么自动化测试对于项目的成功至关重要

+   单元测试如何作为代码质量的启发式方法

+   可用于开发自动化测试和设置质量门框架和工具

+   利用单元测试更好地理解领域问题并记录代码

+   与单元测试相关的概念，例如测试驱动开发

在前面的章节中，我们看到了 Python 特定的特性以及我们如何利用它们来实现更易于维护的代码。我们还探讨了软件工程的通用设计原则如何应用于 Python，利用其特性。在这里，我们也将回顾软件工程的一个重要概念，如自动化测试，但使用工具，其中一些是标准库中可用的（如`unittest`模块），还有一些是外部包（如`pytest`）。我们开始这段旅程，通过探索软件设计如何与单元测试相关联。

# 设计原则和单元测试

在本节中，我们首先将从概念上审视单元测试。我们将回顾上一章中讨论的一些软件工程原则，以了解这与清洁代码的关系。

之后，我们将更详细地讨论如何将这些概念付诸实践（在代码层面），以及我们可以利用哪些框架和工具。

首先，我们快速定义单元测试是什么。单元测试是负责验证其他代码部分的代码。通常，任何人都会倾向于说单元测试验证应用的“核心”，但这种定义将单元测试视为次要的，而这并不是本书中对它们的看法。单元测试是核心，是软件的一个关键组成部分，它们应该与业务逻辑一样受到同样的考虑。

单元测试是一段代码，它导入包含业务逻辑的部分代码，并对其逻辑进行练习，通过断言几个场景来确保某些条件。单元测试必须具备一些特性，例如：

+   隔离：单元测试应该完全独立于任何其他外部代理，并且它们必须只关注业务逻辑。因此，它们不会连接到数据库，不会执行 HTTP 请求等。隔离还意味着测试之间是独立的：它们必须能够以任何顺序运行，而不依赖于任何先前的状态。

+   性能：单元测试必须运行得快。它们旨在多次重复运行。

+   可重复性：单元测试应该能够以确定性的方式客观评估软件的状态。这意味着测试产生的结果应该是可重复的。单元测试评估代码的状态：如果测试失败，它必须持续失败，直到代码被修复。如果测试通过，并且代码没有变化，它应该继续通过。测试不应该是不稳定的或随机的。

+   自验证：单元测试的执行决定了其结果。不应需要额外的步骤来解释单元测试（更不用说手动干预）。

更具体地说，在 Python 中，这意味着我们将有新的`*.py`文件，我们将在这里放置我们的单元测试，并且它们将被某些工具调用。这些文件将包含`import`语句，以从我们的业务逻辑（我们打算测试的内容）中获取所需的内容，并在该文件内部，我们编写测试本身。之后，一个工具将收集我们的单元测试并运行它们，给出结果。

这最后部分就是自验证的实际意义。当工具调用我们的文件时，将启动一个 Python 进程，我们的测试将在其上运行。如果测试失败，进程将以错误代码退出（在 Unix 环境中，这可以是除`0`以外的任何数字）。标准是工具运行测试，并为每个成功的测试打印一个点（`.`）；如果测试失败（测试条件未满足），则打印`F`；如果有异常，则打印`E`。

## 关于其他形式自动测试的注意事项

单元测试旨在验证非常小的代码单元，例如，一个函数或一个方法。我们希望我们的单元测试达到非常详细粒度，尽可能多地测试代码。要测试更大的东西，比如一个类，我们不想只使用单元测试，而应该使用测试套件，这是一个单元测试的集合。每个测试都将测试更具体的东西，比如那个类的方法。

单元测试不是唯一的自动测试机制，我们不应该期望它们捕获所有可能的错误。还有*验收*和*集成*测试，这两者都不在本书的范围之内。

在集成测试中，我们希望同时测试多个组件。在这种情况下，我们希望验证它们是否集体地按预期工作。在这种情况下，允许（甚至更希望）有副作用，并忘记隔离，这意味着我们希望发出 HTTP 请求，连接到数据库等。虽然我们希望我们的集成测试实际上像生产代码那样运行，但还有一些依赖关系我们仍然希望避免。例如，如果你的服务通过互联网连接到另一个外部依赖项，那么这部分确实会被省略。

假设你有一个使用数据库并连接到一些其他内部服务的应用程序。该应用程序将为不同的环境有不同的配置文件，当然，在生产环境中，你将设置用于真实服务的配置。然而，对于集成测试，你将希望使用专门为这些测试构建的 Docker 容器来模拟数据库，这将在特定的配置文件中进行配置。至于依赖项，你将希望尽可能使用 Docker 服务来模拟它们。

在本章稍后部分将介绍将模拟作为单元测试的一部分。当涉及到对组件进行测试时，模拟依赖关系的内容将在第十章“清洁架构”中介绍，那时我们将从软件架构的角度提到组件。

接受测试是一种自动化的测试形式，试图从用户的角度验证系统，通常执行用例。

与单元测试相比，这两种测试形式失去了一个很好的特性：速度。正如你可以想象的那样，它们将需要更多的时间来运行，因此它们将运行得较少。

在一个好的开发环境中，程序员将拥有整个测试套件，并在修改代码、迭代、重构等过程中不断重复运行单元测试。一旦更改准备就绪，并且拉取请求已打开，持续集成服务将为该分支运行构建，其中单元测试将一直运行，直到存在集成或接受测试。不用说，构建的状态在合并之前应该是成功的（绿色），但重要的是测试类型之间的差异：我们希望一直运行单元测试，而那些运行时间较长的测试则运行得较少。

因此，我们希望拥有大量的单元测试，以及一些策略性地设计的自动化测试，以尽可能覆盖单元测试无法触及的地方（例如数据库的使用）。

最后，给明智的人一句话。记住，这本书鼓励实用主义。除了这些定义和本节开头关于单元测试的要点之外，读者必须记住，根据您的标准和环境，最佳解决方案应该占主导地位。没有人比您更了解您的系统，这意味着如果出于某种原因，您必须编写一个需要启动 Docker 容器以测试数据库的单元测试，那就去做吧。正如我们在本书中反复提醒的那样，*实用性胜过纯粹性*。

## 单元测试和敏捷软件开发

在现代软件开发中，我们希望不断交付价值，并且尽可能快地交付。这些目标背后的逻辑是，我们越早得到反馈，影响就越小，改变就越容易。这些根本不是新想法；其中一些类似于几十年前的原则，而另一些（如尽快从利益相关者那里获得反馈并在此基础上迭代的思想）你可以在像**《大教堂与市集》**（缩写为**CatB**）这样的文章中找到。

因此，我们希望能够有效地应对变化，为此，我们编写的软件将必须发生变化。正如我在前面的章节中提到的，我们希望我们的软件具有适应性、灵活性和可扩展性。

代码本身（无论编写和设计得有多好）不能保证我们它足够灵活以进行更改，如果没有正式的证明它在修改后仍然可以正确运行。

假设我们按照 SOLID 原则设计一款软件，在某个部分我们实际上有一组符合开闭原则的组件，这意味着我们可以轻松地扩展它们，而不会对现有代码造成太大影响。进一步假设代码是以有利于重构的方式编写的，因此我们可以根据需要对其进行更改。那么，当我们进行这些更改时，我们是否在引入任何错误呢？我们如何知道现有功能是否得到保留（并且没有回归）？您是否足够自信将此版本发布给用户？他们会相信新版本能按预期工作吗？

所有这些问题的答案是我们不能确定，除非我们有正式的证明。而单元测试正是这样：正式证明程序按照规格工作。

因此，单元测试（或自动化测试）就像一个安全网，它给了我们信心去修改代码。有了这些工具，我们可以高效地工作，因此这最终决定了软件产品团队的工作速度（或容量）。测试越好，我们能够快速交付价值而不被错误频繁阻止的可能性就越大。

## 单元测试和软件设计

当涉及到主代码和单元测试之间的关系时，这是硬币的另一面。除了上一节中探讨的实用主义原因之外，这归结于好的软件是可测试的软件的事实。

**可测试性**（决定软件测试难易程度的质量属性）不仅是一个好东西，而且是编写干净代码的驱动力。

单元测试不仅仅是主代码库的补充，而是一种对代码编写方式有直接影响和实际影响的因素。这有很多层次，从一开始，当我们意识到我们想要为代码的某些部分添加单元测试时，我们必须对其进行更改（从而得到一个更好的版本），到其最终的表达（在本章末尾附近探讨）时，整个代码（设计）都是通过将要进行的测试方式（**测试驱动设计**）来驱动的。

从一个简单的例子开始，我将向您展示一个小的用例，其中测试（以及测试我们代码的需要）导致我们代码编写方式的改进。

在以下示例中，我们将模拟一个需要将每个特定任务获得的结果发送到外部系统的过程（正如通常一样，只要我们专注于代码，细节就不会有任何影响）。我们有一个`Process`对象，它代表领域问题上的一个任务，并使用`metrics`客户端（一个外部依赖项，因此我们无法控制）将实际指标发送到外部实体（这可能是指向`syslog`或`statsd`发送数据等）：

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
            self.client.send(f"iteration.{i}", str(result)) 
```

在第三方客户端的模拟版本中，我们设定了必须提供字符串类型参数的要求。因此，如果`run_process`方法的`result`不是字符串，我们可能会预期它将失败，而且确实如此：

```py
Traceback (most recent call last):
...
    raise TypeError("expected type str for metric_value")
TypeError: expected type str for metric_value 
```

记住，这种验证超出了我们的控制范围，我们无法更改代码，所以在继续之前，我们必须提供正确类型的参数。但既然这是我们发现的错误，我们首先想编写一个单元测试来确保它不会再次发生。我们这样做是为了证明我们修复了问题，并且为了防止未来再次出现这个错误，无论代码更改多少次。

可以通过模拟`Process`对象的客户端来测试代码（我们将在探讨单元测试工具的*模拟对象*部分中看到如何这样做），但这样做会运行比所需的更多代码（注意我们想要测试的部分是如何嵌套在代码中的）。此外，方法相对较小是个好事，因为如果不是这样，测试将不得不运行更多我们不希望运行的未指定部分，我们可能也需要对这些部分进行模拟。这是另一个关于良好设计（小而内聚的函数或方法）的例子，它与可测试性相关。

最后，我们决定不必过于麻烦，只测试我们需要测试的部分，所以不是直接在`main`方法上与`client`交互，而是委托给一个`wrapper`方法，新的类看起来是这样的：

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

在这种情况下，我们选择创建我们自己的`client`版本用于指标，即围绕我们曾经使用过的第三方库的一个包装器。为此，我们放置一个具有相同接口的类，它将相应地进行类型转换。

这种使用组合的方式类似于适配器设计模式（我们将在下一章探讨设计模式，所以现在就先作为一个信息提示），由于这是我们领域中的新对象，它可以有自己的相应单元测试。拥有这个对象将使测试变得更加简单，但更重要的是，现在我们来看它，我们意识到这可能是代码最初就应该编写的方式。尝试为我们的代码编写单元测试让我们意识到我们完全遗漏了一个重要的抽象！

现在我们已经将方法分离成应该的样子，让我们为它编写实际的单元测试。关于本例中使用的`unittest`模块的详细信息将在探讨测试工具和库的部分进行更详细的探讨，但就现在而言，阅读代码将给我们一个如何测试的第一印象，并且会使之前的概念更加具体：

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

`Mock`是`unittest.mock`模块中的一个类型，它是一个方便的对象，可以询问各种各样的事情。例如，在这种情况下，我们用它来代替第三方库（如下一节注释所述，模拟到系统的边界）以检查它是否按预期调用（并且再次，我们不是测试库本身，只是测试它是否正确调用）。注意我们运行了一个像我们的`Process`对象中的调用，但我们期望参数被转换为字符串。

这是一个单元测试如何帮助我们改进代码设计的例子：通过尝试测试代码，我们得到了一个更好的版本。我们可以更进一步地说，这个测试还不够好，因为单元测试在第二行中覆盖了包装器客户端的内部协作者。为了解决这个问题，我们可能会说，实际的客户端必须通过参数（使用依赖注入）提供，而不是在初始化方法中创建它。而且，单元测试再次让我们想到了一个更好的实现。

之前例子的推论应该是，代码的可测试性也反映了其质量。换句话说，如果代码难以测试，或者其测试复杂，那么它可能需要改进。

> “编写测试没有技巧；只有编写可测试代码的技巧”
> 
> – 米什科·赫维

## 定义要测试的边界

测试需要付出努力。如果我们决定测试什么时不小心，我们永远不会结束测试，从而浪费了大量努力而没有取得多少成果。

我们应该将测试范围限定在我们的代码边界内。如果我们不这样做，我们就必须测试代码中的依赖项（外部/第三方库或模块），然后是它们各自的依赖项，如此等等，形成一个永无止境的旅程。测试依赖项不是我们的责任，因此我们可以假设这些项目有自己的测试。只需测试正确的外部依赖项是否以正确的参数调用（这可能甚至可以接受使用修补），但我们不应该投入比这更多的努力。

这又是一个良好的软件设计带来回报的例子。如果我们已经谨慎地进行了设计，并清楚地定义了系统的边界（也就是说，我们设计的是接口，而不是将改变的具体实现，从而将外部组件的依赖关系反转以减少时间耦合），那么在编写单元测试时模拟这些接口将会容易得多。

在良好的单元测试中，我们希望针对系统的边界进行修补，并关注要测试的核心功能。我们不测试外部库（例如通过`pip`安装的第三方工具），而是检查它们是否被正确调用。当我们在本章后面探索`mock`对象时，我们将回顾执行这些类型断言的技术和工具。

# 测试工具

我们可以用来编写单元测试的工具有很多，它们各有优缺点，服务于不同的目的。我将介绍 Python 中用于单元测试的两个最常见库。它们涵盖了大多数（如果不是所有）用例，并且非常受欢迎，因此了解如何使用它们非常有用。

除了测试框架和测试运行库之外，通常还会发现配置代码覆盖率的项目，它们将其用作质量指标。由于覆盖率（当用作指标时）具有误导性，在了解如何创建单元测试之后，我们将讨论为什么它不应被轻视。

下一节将从介绍本章中我们将要使用的用于单元测试的主要库开始。

## 单元测试的框架和库

在本节中，我们将讨论两个用于编写和运行单元测试的框架。第一个框架是`unittest`，它包含在 Python 的标准库中，而第二个框架`pytest`则需要通过`pip`外部安装：

+   `unittest`: [`docs.python.org/3/library/unittest.html`](https://docs.python.org/3/library/unittest.html)

+   `pytest`: [`docs.pytest.org/en/latest/`](https://docs.pytest.org/en/latest/)

当涉及到为我们的代码覆盖测试场景时，仅使用`unittest`可能就足够了，因为它有大量的辅助工具。然而，对于具有多个依赖项、与外部系统连接以及可能需要修补对象、定义固定值和参数化测试用例的更复杂系统，`pytest`看起来是一个更完整的选项。

我们将使用一个小程序作为示例，展示如何使用两种选项进行测试，这最终将帮助我们更好地了解这两个库的比较。

展示测试工具的示例是一个支持合并请求中代码审查的版本控制工具的简化版本。我们将从以下标准开始：

+   如果至少有一个人反对更改，合并请求将被“拒绝”。

+   如果没有人反对，并且合并请求至少对其他两位开发者来说是好的，它就是“批准”的。

+   在任何其他情况下，其状态是“挂起”。

下面是代码可能的样子：

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

以此代码为基础，让我们看看如何使用本章中介绍的两种库进行单元测试。这个想法不仅是为了了解如何使用每个库，而且是为了识别一些差异。

### unittest

`unittest`模块是一个很好的起点，用于编写单元测试，因为它提供了一个丰富的 API 来编写各种测试条件，并且由于它包含在标准库中，因此它非常灵活和方便。

`unittest` 模块基于 JUnit（来自 Java）的概念，而 JUnit 又基于来自 Smalltalk 的单元测试的原始想法（这可能是这个模块上方法命名惯例背后的原因），因此它本质上是面向对象的。因此，测试是通过类编写的，检查是通过方法验证的，通常在类中按场景分组测试。

要开始编写单元测试，我们必须创建一个继承自`unittest.TestCase`的测试类，并定义我们想要在其方法上施加的条件。这些方法应该以`test_`开头，并且可以内部使用从`unittest.TestCase`继承的任何方法来检查必须成立的条件。

我们可能想要验证的一些条件示例如下：

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

单元测试的 API 提供了许多有用的比较方法，最常见的是`assertEqual(<实际>, <预期>[, message])`，它可以用来比较操作的结果与我们期望的值，可选地使用在出错时显示的消息。

我使用顺序（`<actual>, <expected>`）命名参数，因为在我的经验中，大多数情况下我找到的都是这种顺序。尽管我相信这可能是最常见的形式（作为一种约定）在 Python 中使用，但没有任何建议或指南。事实上，一些项目（如 gRPC）使用相反的形式（`<expected>, <actual>`），这实际上在其他语言中（例如 Java 和 Kotlin）是一种约定。关键是保持一致并尊重项目中已经使用的格式。

另一种有用的测试方法允许我们检查是否抛出了特定的异常（`assertRaises`）。

当发生异常情况时，我们在代码中抛出异常以防止在错误假设下进行进一步处理，并通知调用者调用过程中存在问题。这是逻辑中应该被测试的部分，这也是这个方法的目的。

想象一下，我们现在将我们的逻辑扩展一点，以允许用户关闭他们的合并请求，一旦发生这种情况，我们就不希望再进行任何投票（一旦已经关闭，评估合并请求就没有意义了）。为了防止这种情况发生，我们扩展了我们的代码，并在有人试图对一个已关闭的合并请求进行投票的不幸事件上抛出异常。

在添加了两个新的状态（`OPEN` 和 `CLOSED`）以及一个新的 `close()` 方法后，我们修改了之前的投票方法，以便首先处理这个检查：

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
            raise MergeRequestException(
                "can't vote on a closed merge request"
            )
    def upvote(self, by_user):
        self._cannot_vote_if_closed()
        self._context["downvotes"].discard(by_user)
        self._context["upvotes"].add(by_user)
    def downvote(self, by_user):
        self._cannot_vote_if_closed()
        self._context["upvotes"].discard(by_user)
        self._context["downvotes"].add(by_user) 
```

现在，我们想要检查这个验证确实有效。为此，我们将使用 `assertRaises` 和 `assertRaisesRegex` 方法：

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

前者期望在调用第二个参数中的可调用对象时抛出提供的异常，以及函数其余部分的参数（`*args` 和 `**kwargs`），如果没有这样做，它将失败，表示预期抛出的异常没有发生。后者做的是同样的事情，但它还检查抛出的异常是否包含与提供的参数匹配的正则表达式消息。即使异常被抛出，但消息不同（不匹配正则表达式），测试也会失败。

尝试检查错误信息，因为除了异常作为额外的检查将更加准确并确保触发的是我们想要的异常之外，它还会检查是否偶然触发了同一类型的另一个异常。

注意这些方法也可以用作上下文管理器。在其第一种形式（在之前的示例中使用的那种形式）中，该方法接收异常，然后是可调用对象，最后是用于该可调用对象的参数列表）。但我们也可以将异常作为方法的参数传递，将其用作上下文管理器，并在该上下文管理器的块内评估我们的代码，格式如下：

```py
with self.assertRaises(MyException):
   test_logic() 
```

这种第二种形式通常更有用（有时，是唯一的选择）；例如，如果我们需要测试的逻辑不能表示为一个单一的调用函数。

在某些情况下，你会注意到我们需要运行相同的测试用例，但使用不同的数据。与其重复并生成重复的测试，我们可以构建一个单一的测试用例，并使用不同的值来测试其条件。这被称为**参数化测试**，我们将在下一节开始探索这些内容。稍后，我们将使用`pytest`重新审视参数化测试。

#### 参数化测试

现在，我们想要测试合并请求的阈值接受度是如何工作的，只需提供`context`看起来像什么的数据样本，而不需要整个`MergeRequest`对象。我们想要测试`status`属性中检查是否关闭之后的那个部分，但独立地。

实现这一点的最佳方式是将该组件分离成另一个类，使用组合，然后继续使用自己的测试套件测试这个新的抽象：

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

通过这些更改，我们可以再次运行测试并验证它们是否通过，这意味着这个小重构没有破坏当前功能（单元测试确保回归）。有了这个，我们可以继续我们的目标，编写针对新类的特定测试：

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
                MergeRequestStatus.REJECTED,
            ),
            (
                {"downvotes": set(), "upvotes": {"dev1", "dev2"}},
                MergeRequestStatus.APPROVED,
            ),
        )
    def test_status_resolution(self):
        for context, expected in self.fixture_data:
            with self.subTest(context=context):
                status = AcceptanceThreshold(context).status()
                self.assertEqual(status, expected) 
```

在这里，在`setUp()`方法中，我们定义了在整个测试中要使用的数据固定值。在这种情况下，实际上并不需要，因为我们可以直接将其放在方法上，但如果我们期望在执行任何测试之前运行一些代码，这就是我们编写它的地方，因为此方法在每次运行测试之前只调用一次。

在这个特定的情况下，我们可以将这个元组定义为类的属性，因为它是一个常量（静态）值。如果我们需要运行一些代码，并执行一些计算（例如构建对象或使用工厂），那么`setUp()`方法就是我们的唯一选择。

通过编写这个新版本的代码，被测试代码下的参数更清晰、更紧凑。

为了模拟我们正在运行所有参数，测试会遍历所有数据，并使用每个实例来测试代码。这里有一个有趣的辅助工具是使用`subTest`，在这种情况下，我们使用它来标记被调用的测试条件。如果这些迭代中的任何一个失败了，`unittest`会报告它，并带有传递给`subTest`的变量的相应值（在这种情况下，它被命名为`context`，但任何一系列关键字参数都可以正常工作）。例如，一个错误发生可能看起来像这样：

```py
FAIL: (context={'downvotes': set(), 'upvotes': {'dev1', 'dev2'}})
----------------------------------------------------------------------
Traceback (most recent call last):
  File "" test_status_resolution
    self.assertEqual(status, expected)
AssertionError: <MergeRequestStatus.APPROVED: 'approved'> != <MergeRequestStatus.REJECTED: 'rejected'> 
```

如果你选择参数化测试，尽量提供每个参数实例的上下文信息，尽可能多，以便更容易调试。

参数化测试背后的思想是在不同的数据集上运行相同的测试条件。这个想法是首先确定要测试的数据的等价类，然后选择每个类的值代表（关于这一点，本章后面将详细介绍）。然后，你可能会想知道你的测试在哪个等价类上失败了，而 `subTest` 上下文管理器提供的上下文在这种情况下很有帮助。

### pytest

Pytest 是一个优秀的测试框架，可以通过 `pip install pytest` 安装。与 `unittest` 相比，有一个区别是，虽然我们仍然可以在类中分类测试场景并创建面向对象的测试模型，但这实际上不是必需的，并且我们可以通过在简单函数中使用 `assert` 语句来检查我们想要验证的条件，从而以更少的样板代码编写单元测试。

默认情况下，使用 `assert` 语句进行比较就足以让 `pytest` 识别单元测试并相应地报告其结果。更高级的使用，如前节中看到的，也是可能的，但它们需要使用该包中的特定函数。

一个很好的特性是，`pytests` 命令将运行它能够发现的所有测试，即使它们是用 `unittest` 编写的。这种兼容性使得从 `unittest` 到 `pytest` 的逐步过渡变得更容易。

#### 使用 pytest 的基本测试用例

我们在上一节测试的条件可以用 `pytest` 重新编写成简单的函数。

一些使用简单断言的例子如下：

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

布尔相等比较不需要比简单的 `assert` 语句更多，而其他类型的检查，如异常检查，则需要我们使用一些函数：

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

在这种情况下，`pytest.raises` 等同于 `unittest.TestCase.assertRaises`，并且它也接受作为方法或上下文管理器调用。如果我们想检查异常的消息，而不是使用不同的方法（例如 `assertRaisesRegex`），则必须使用相同的函数，但作为上下文管理器，并通过提供我们想要识别的表达式的 `match` 参数来实现。

`pytest` 还会将原始异常包装成一个可预期的自定义异常（通过检查一些属性，例如 `.value`，例如），如果我们想检查更多条件，但这个函数的使用涵盖了绝大多数情况。

#### 参数化测试

使用 `pytest` 运行参数化测试更好，不仅因为它提供了一个更干净的 API，而且还因为每个测试及其参数的组合都会生成一个新的测试用例（一个新的函数）。

为了使用这个，我们必须在我们的测试上使用 `pytest.mark.parametrize` 装饰器。装饰器的第一个参数是一个字符串，表示要传递给 `test` 函数的参数名称，第二个必须是可迭代的，包含这些参数的相应值。

注意到测试函数的主体被简化为了一行（在移除内部 `for` 循环及其嵌套上下文管理器之后），并且每个测试用例的数据都正确地从函数的主体中隔离出来，这使得扩展和维护更容易：

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
        MergeRequestStatus.REJECTED,
    ),
    (
        {"downvotes": set(), "upvotes": {"dev1", "dev2"}},
        MergeRequestStatus.APPROVED,
    ),
),)
def test_acceptance_threshold_status_resolution(context, expected_status):
    assert AcceptanceThreshold(context).status() == expected_status 
```

使用 `@pytest.mark.parametrize` 来消除重复，尽可能使测试的主体保持一致，并明确代码必须支持的参数（测试输入或场景）。

使用参数化时的重要建议是，每个参数（每个迭代）应仅对应一个测试场景。这意味着你不应该将不同的测试条件混合到同一个参数中。如果你需要测试不同参数的组合，那么使用不同的参数化堆叠。堆叠这个装饰器将创建与装饰器中所有值的笛卡尔积一样多的测试条件。

例如，一个配置如下测试：

```py
@pytest.mark.parametrize("x", (1, 2))
@pytest.mark.parametrize("y", ("a", "b"))
def my_test(x, y):
   … 
```

将为 `(x=1, y=a)`、`(x=1, y=b)`、`(x=2, y=a)` 和 `(x=2, y=b)` 这些值运行。

这是一个更好的方法，因为每个测试都更小，每个参数化更具体（一致）。这将允许你以更简单的方式通过所有可能的组合的爆炸来对代码进行压力测试。

当你有需要测试的数据，或者你知道如何轻松构建它时，数据参数工作得很好，但在某些情况下，你需要为测试构建特定的对象，或者你发现自己反复编写或构建相同的对象。为了帮助解决这个问题，我们可以使用 fixtures，正如我们将在下一节中看到的。

#### Fixtures

`pytest` 的一个优点是它如何促进创建可重用功能，这样我们就可以用数据或对象来测试，更有效地避免重复。

例如，我们可能希望在特定状态下创建一个 `MergeRequest` 对象，并在多个测试中使用该对象。我们通过创建一个函数并应用 `@pytest.fixture` 装饰器来定义我们的对象作为 fixture。想要使用该 fixture 的测试必须有一个与定义的函数同名参数，`pytest` 将确保它被提供：

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

记住，测试也会影响主代码，因此清洁代码的原则也适用于它们。在这种情况下，我们在前几章中探讨的**不要重复自己**（**DRY**）原则再次出现，我们可以借助 `pytest` fixtures 来实现它。

除了创建多个对象或公开将在整个测试套件中使用的数据之外，还可以使用它们来设置一些条件，例如，全局修补我们不希望被调用的函数，或者当我们想要使用修补对象时。

### 代码覆盖率

测试运行器支持覆盖率插件（通过 `pip` 安装），这些插件可以提供有关代码中哪些行在测试运行时被执行的有用信息。这些信息非常有帮助，使我们知道哪些代码部分需要被测试，以及需要改进的地方（包括生产代码和测试）。我的意思是，检测到我们生产代码中未覆盖的行将迫使我们必须为该代码部分编写测试（因为请记住，没有测试的代码应被视为有缺陷的）。在尝试覆盖代码的过程中，可能会发生几件事情：

+   我们可能会意识到我们完全遗漏了一个测试场景。

+   我们将尝试编写更多的单元测试或覆盖更多代码行的单元测试。

+   我们将尝试简化我们的生产代码，去除冗余，使其更加紧凑，这意味着更容易被覆盖。

+   我们甚至可能会意识到我们试图覆盖的代码行是不可到达的（可能是在逻辑中犯了错误），并且可以安全地删除。

请记住，尽管这些都是积极的一面，但覆盖率永远不应该是一个目标，而只是一个指标。这意味着试图达到高覆盖率，只是为了达到 100%，将不会富有成效或有效。我们应该将代码覆盖率视为一个单位，以识别需要测试的明显代码部分，并了解我们如何可以改进这一点。然而，我们可以设定一个最低阈值，比如 80%（一个普遍接受的价值），作为期望覆盖率的最低水平，以了解项目有合理数量的测试。

此外，认为高程度的代码覆盖率是健康代码库的标志也是危险的：请记住，大多数覆盖率工具都会报告已执行的生产代码行。一行被调用并不意味着它已经被正确测试（只是它运行了）。一个单独的语句可能封装了多个逻辑条件，每个条件都需要单独测试。

不要被高程度的代码覆盖率误导，并继续思考测试代码的方法，包括那些已经覆盖的行。

在此方面最广泛使用的库之一是 `coverage` ([`pypi.org/project/coverage/`](https://pypi.org/project/coverage/))。我们将在下一节中探讨如何设置这个工具。

#### 设置 rest 覆盖率

在 `pytest` 的情况下，我们可以安装 `pytest-cov` 包。一旦安装，当运行测试时，我们必须告诉 `pytest` 运行器 `pytest-cov` 也会运行，以及哪些包（或包）应该被覆盖（以及其他参数和配置）。

此软件包支持多种配置，包括不同类型的输出格式，并且很容易将其与任何持续集成工具集成，但在所有这些功能中，一个高度推荐的选择是设置一个标志，它会告诉我们哪些行还没有被测试覆盖，因为这将帮助我们诊断代码并允许我们开始编写更多的测试。

为了向您展示这会是什么样子，请使用以下命令：

```py
PYTHONPATH=src pytest \
    --cov-report term-missing \
    --cov=coverage_1 \
    tests/test_coverage_1.py 
```

这将产生类似于以下输出的结果：

```py
test_coverage_1.py ................ [100%]
----------- coverage: platform linux, python 3.6.5-final-0 -----------
Name         Stmts Miss Cover Missing
---------------------------------------------
coverage_1.py 39      1  97%    44 
```

这里，它告诉我们有一行没有单元测试，因此我们可以看看如何为它编写单元测试。这是一个常见的场景，我们意识到要覆盖这些缺失的行，我们需要通过创建更小的方法来重构代码。结果，我们的代码将看起来好得多，就像我们在本章开头看到的例子一样。

问题在于相反的情况——我们能否相信高覆盖率？这难道意味着我们的代码是正确的吗？不幸的是，良好的测试覆盖率是干净代码的必要但不充分条件。代码的部分没有测试显然是件坏事。实际上有测试是非常好的，但我们只能对存在的测试这样说。然而，我们对缺失的测试知之甚少，即使代码覆盖率很高，我们可能仍然遗漏了许多条件。

这些是测试覆盖率的一些注意事项，我们将在下一节中提及。

#### 测试覆盖率的注意事项

Python 是解释型语言，在非常高的层面上，覆盖率工具利用这一点来识别在测试运行期间被解释（运行）的行。然后它将在结束时报告这一点。一行被解释的事实并不意味着它得到了适当的测试，这就是为什么我们应该小心阅读最终的覆盖率报告并相信它所说的内容。

这实际上适用于任何语言。一行被执行的事实并不意味着它被所有可能的组合所压力测试。所有分支在提供的数据上成功运行的事实仅意味着代码支持该组合，但它并没有告诉我们关于任何其他可能导致程序崩溃的参数组合的信息（模糊测试）。

将覆盖率用作发现代码盲点的工具，而不是作为指标或目标。

为了用简单的例子来说明这一点，考虑以下代码：

```py
def my_function(number: int):
    return "even" if number % 2 == 0 else "odd" 
```

现在，让我们假设我们为它编写以下测试：

```py
@pytest.mark.parametrize("number,expected", [(2, "even")])
def test_my_function(number, expected):
    assert my_function(number) == expected 
```

如果我们运行带有覆盖率的测试，报告将给出令人眼花缭乱的 100%覆盖率。不用说，我们遗漏了对单条语句一半条件进行的测试。更令人不安的是，由于该语句的`else`子句没有运行，我们不知道我们的代码可能会以哪些方式出错（为了使这个例子更加夸张，想象一下有一个错误的语句，比如`1/0`而不是字符串`"odd"`，或者有一个函数调用）。

可以说，我们可能更进一步地认为这仅仅是“成功路径”，因为我们向函数提供了良好的值。但是，对于不正确的类型呢？函数应该如何防御这种情况？

正如你所见，即使是单个看似无辜的语句也可能引发许多问题和测试条件，我们需要为这些问题做好准备。

检查我们的代码覆盖率是一个好主意，甚至可以将代码覆盖率阈值配置为 CI 构建的一部分，但我们必须记住，这仅仅是我们工具箱中的另一个工具。就像我们之前探索过的其他工具（代码检查器、格式化工具等）一样，它只有在更多工具和为干净代码库准备的良好环境中才有用。

另一个有助于我们测试工作的工具是使用模拟对象。我们将在下一节中探讨这些内容。

### 模拟对象

有时候，我们的代码并不是测试环境中唯一存在的东西。毕竟，我们设计和构建的系统必须做一些真实的事情，这通常意味着连接到外部服务（数据库、存储服务、外部 API、云服务等等）。因为它们需要那些副作用，所以它们是不可避免的。尽管我们抽象代码、面向接口编程、将代码与外部因素隔离以最小化副作用，但它们仍然会出现在我们的测试中，我们需要一种有效的方式来处理这种情况。

模拟对象是我们用来保护单元测试免受不期望的副作用影响（如本章前面所述）的最好策略之一。我们的代码可能需要执行 HTTP 请求或发送通知电子邮件，但我们肯定不希望在单元测试中发生这种情况。单元测试应该针对我们代码的逻辑，并且运行快速，因为我们希望非常频繁地运行它们，这意味着我们无法承受延迟。因此，真正的单元测试不会使用任何实际的服务——它们不会连接到任何数据库，不会发起 HTTP 请求，基本上，它们除了锻炼生产代码的逻辑之外，什么都不做。

我们需要能够执行这些操作的测试，但它们并不是单元测试。集成测试应该从更广泛的角度测试功能，几乎模仿用户的行为。但它们并不快。因为它们连接到外部系统和服务，所以它们需要更长的时间，并且运行成本更高。一般来说，我们希望有大量的单元测试可以快速运行，以便随时运行，而集成测试则运行得较少（例如，在每次新的合并请求时）。

虽然模拟对象很有用，但滥用它们介于代码异味或反模式之间。这是我们将在下一节讨论的第一个问题，在讨论使用模拟的细节之前。

#### 关于补丁和模拟的一个公正警告

我之前说过，单元测试帮助我们编写更好的代码，因为当我们开始思考如何测试我们的代码时，我们会意识到如何改进它以使其可测试。通常，随着代码的可测试性提高，它也会变得更干净（更一致、更细粒度、分解成更小的组件等）。

另一个有趣的收获是，测试将帮助我们注意到我们认为代码正确的地方的代码异味。我们的代码有代码异味的一个主要警告是我们是否发现自己试图猴子补丁（或模拟）很多不同的事情，只是为了覆盖一个简单的测试用例。

`unittest`模块提供了一个工具，可以在`unittest.mock.patch`中修补我们的对象。

补丁意味着原始代码（由一个表示其在导入时位置的字符串给出）将被替换为其他东西，而不是其原始代码。如果没有提供替换对象，默认是一个标准的模拟对象，它将简单地接受所有方法调用或询问的属性。

补丁函数在运行时替换代码，其缺点是我们失去了与最初存在的原始代码的联系，使我们的测试变得略微肤浅。它还因为修改解释器中对象的额外开销而考虑性能问题，这可能是如果我们重构代码并移动事物时可能需要未来更改的事情（因为补丁函数中声明的字符串将不再有效）。

在我们的测试中使用猴子补丁或模拟可能是可以接受的，并且本身并不代表问题。另一方面，猴子补丁的滥用确实是一个红旗，告诉我们我们的代码中有些地方需要改进。

例如，就像在测试一个函数时遇到困难可能会让我们想到这个函数可能太大，应该分解成更小的部分一样，试图测试需要非常侵入性猴子补丁的代码片段应该告诉我们，也许代码过于依赖硬依赖，应该使用依赖注入代替。

#### 使用模拟对象

在单元测试术语中，有几个类型的对象属于名为**测试双工**的类别。测试双工是一种对象，由于不同的原因（可能我们不需要实际的生成代码，只是一个哑对象就足够了，或者可能我们无法使用它，因为它需要访问服务或它有我们不希望在单元测试中出现的副作用等），将在我们的测试套件中取代真实对象。

测试双工有不同类型，如哑对象、存根、间谍或 mock。

Mocks 是最通用的对象类型，由于它们非常灵活和多功能，因此适用于所有情况，无需深入了解其他类型。正因为如此，标准库也包含这种类型的对象，这在大多数 Python 程序中很常见。这里我们将使用的就是：`unittest.mock.Mock`。

**mock**是一种根据特定规格（通常类似于生产类对象）和一些配置的响应（即我们可以告诉 mock 在特定调用时应返回什么，以及其行为应如何）创建的对象。`Mock`对象将记录其内部状态的一部分，包括其被调用的方式（使用什么参数、调用了多少次等），我们可以使用这些信息来验证我们应用程序在后续阶段的行为。

在 Python 的情况下，标准库中可用的`Mock`对象提供了一个很好的 API 来执行各种行为断言，例如检查 mock 对象被调用的次数、使用什么参数等。

##### Mock 类型

标准库在`unittest.mock`模块中提供了`Mock`和`MagicMock`对象。前者是一种可以配置为返回任何值的测试双工，并将跟踪对其的调用。后者执行相同的功能，但它还支持魔法方法。这意味着，如果我们已经编写了使用魔法方法的惯用代码（并且我们正在测试的代码将依赖于它），我们可能需要使用`MagicMock`实例而不是仅仅使用`Mock`。

当我们的代码需要调用魔法方法时尝试使用`Mock`会导致错误。以下代码是这种情况的一个示例：

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

我们想测试这个函数；然而，另一个测试需要调用`author_by_id`函数。由于我们没有测试那个函数，任何提供给该函数（并返回）的值都是好的：

```py
def test_find_commit():
    branch = GitBranch([{"id": "123", "author": "dev1"}])
    assert author_by_id("123", branch) == "dev1"
def test_find_any():
    author = author_by_id("123", Mock()) is not None
    # ... rest of the tests.. 
```

如预期的那样，这不会工作：

```py
def author_by_id(commit_id, branch):
    > return branch[commit_id]["author"]
    E TypeError: 'Mock' object is not subscriptable 
```

使用`MagicMock`代替将工作。我们甚至可以配置这种类型 mock 的魔法方法，以返回我们需要以控制测试执行的内容：

```py
def test_find_any():
    mbranch = MagicMock()
    mbranch.__getitem__.return_value = {"author": "test"}
    assert author_by_id("123", mbranch) == "test" 
```

##### 测试双工的使用案例

要看到 mocks 的可能的用法，我们需要向我们的应用程序添加一个新的组件，该组件将负责通知合并请求的`build`的`status`。当一个`build`完成时，这个对象将被调用，带上合并请求的 ID 和`build`的`status`，并通过向特定的固定端点发送 HTTP `POST`请求来更新合并请求的`status`：

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

这个类有许多副作用，但其中之一是一个重要的外部依赖项，很难克服。如果我们尝试在不修改任何内容的情况下编写关于它的测试，它将因为连接错误而失败，一旦它尝试执行 HTTP 连接。

作为测试目标，我们只想确保信息被正确组合，并且库请求是以适当的参数被调用的。由于这是一个外部依赖项，我们不想测试`requests`模块；只需检查它是否被正确调用就足够了。

当我们尝试比较发送给库的数据时，我们将会遇到另一个问题，即这个类正在计算当前的时间戳，这在单元测试中是无法预测的。直接修补`datetime`是不可能的，因为这个模块是用 C 语言编写的。有一些外部库可以做到这一点（例如`freezegun`），但它们会带来性能上的惩罚，并且对于这个例子来说，这将是过度的。因此，我们选择将我们想要的功能包装在一个静态方法中，这样我们就可以修补它。

既然我们已经确定了代码中需要替换的点，让我们编写单元测试：

```py
# test_mock_2.py
from unittest import mock
from constants import STATUS_ENDPOINT
from mock_2 import BuildStatus
@mock.patch("mock_2.requests")
def test_build_notification_sent(mock_requests):
    build_date = "2018-01-01T00:00:01"
    with mock.patch(
        "mock_2.BuildStatus.build_date", 
        return_value=build_date
    ):
        BuildStatus.notify(123, "OK")
    expected_payload = {
        "id": 123, 
        "status": "OK", 
        "built_at": build_date
    }
    mock_requests.post.assert_called_with(
        STATUS_ENDPOINT, json=expected_payload
    ) 
```

首先，我们使用`mock.patch`作为装饰器来替换`requests`模块。这个函数的结果将创建一个`mock`对象，该对象将被作为参数传递给测试（在这个例子中命名为`mock_requests`）。然后，我们再次使用这个函数，但这次作为上下文管理器来改变计算`build`日期的类的方法的返回值，用我们控制的值替换它，我们将在断言中使用这个值。

一旦我们把这些都设置好了，我们就可以用一些参数调用类方法，然后我们可以使用`mock`对象来检查它是如何被调用的。在这种情况下，我们使用这个方法来查看`requests.post`是否确实以我们想要的方式组合了参数被调用。

这是 mocks 的一个很好的特性——它们不仅为所有外部组件（在这种情况下是为了防止发送一些通知或发出 HTTP 请求）设定了边界，而且还提供了一个有用的 API 来验证调用及其参数。

虽然在这种情况下，我们能够通过设置相应的`mock`对象来测试代码，但这也意味着我们必须针对主要功能的总代码行数进行大量的修补。关于纯生产代码被测试的比例与我们必须模拟的代码部分的比例没有规则，但当然，通过使用常识，我们可以看到，如果我们必须在相同的部分修补很多东西，那么某些东西可能没有清晰地抽象出来，这看起来像是一个代码异味。

可以将外部依赖项的修补与固定值结合使用，以应用一些全局配置。例如，通常一个好的做法是防止所有单元测试执行 HTTP 调用，因此我们可以在单元测试的子目录中，在`pytest`的配置文件中添加一个固定值（`tests/unit/conftest.py`）：

```py
@pytest.fixture(autouse=True)
def no_requests():
    with patch("requests.post"):
        yield 
```

此函数将在所有单元测试中自动调用（因为`autouse=True`），当它这样做时，它将修补`requests`模块中的`post`函数。这只是一个你可以适应你项目的想法，以添加一些额外的安全性和确保你的单元测试没有副作用。

在下一节中，我们将探讨如何重构代码以克服这个问题。

# 重构

重构意味着通过重新排列其内部表示来改变代码的结构，而不修改其外部行为。

一个例子是，如果你发现一个具有许多职责和非常长的方法的类，然后决定通过使用更小的方法、创建新的内部协作者和将职责分配到新的、更小的对象来改变它。在这个过程中，你小心不要改变该类的原始接口，保持所有公共方法与之前相同，并且不更改任何签名。对于该类的外部观察者来说，可能看起来什么都没发生（但我们知道并非如此）。

**重构**是软件维护中的关键活动，但如果没有单元测试（至少不能正确地完成）就无法进行。这是因为，随着每次更改的进行，我们需要知道我们的代码仍然是正确的。从某种意义上说，你可以把我们的单元测试看作是我们代码的“外部观察者”，确保合同没有破裂。

不时地，我们需要支持新的功能或以未预料到的方式使用我们的软件。满足此类需求唯一的方法是首先重构我们的代码，使其更加通用或灵活。

通常，当我们重构代码时，我们希望改进其结构，使其更好，有时更通用、更易读或更灵活。挑战是在保持修改前代码的精确功能的同时实现这些目标。必须支持与之前相同的功能的约束意味着我们需要在修改过的代码上运行回归测试。运行回归测试的唯一经济有效的方式是如果这些测试是自动的。最经济有效的自动测试版本是单元测试。

## 代码的演变

在上一个示例中，我们能够将副作用从代码中分离出来，通过修补那些依赖于我们无法在单元测试中控制的代码部分，从而使代码可测试。这是一个很好的方法，因为毕竟，`mock.patch` 函数在这些任务中非常有用，它可以替换我们告诉它的对象，并返回一个 `Mock` 对象。

这样做的缺点是我们必须以字符串的形式提供将要模拟的对象的路径，包括模块。这有点脆弱，因为如果我们重构代码（比如说我们重命名文件或将其移动到其他位置），所有带有补丁的地方都需要更新，否则测试会失败。

在这个例子中，`notify()` 方法直接依赖于实现细节（`requests` 模块）的事实是一个设计问题；也就是说，它对单元测试也有影响，上述的脆弱性意味着。

我们仍然需要用双份（模拟）来替换这些方法，但如果我们对代码进行重构，我们可以做得更好。让我们将这些方法分成更小的部分，最重要的是注入依赖而不是保持固定。现在的代码应用了依赖倒置原则，并期望与支持接口（在这个例子中，是一个隐式接口）的东西一起工作，例如 `requests` 模块提供的：

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

我们将方法分开（注意现在 notify 是组合 `+` 交付），将 `compose_payload()` 作为新方法（这样我们就可以替换，而无需修补类），并要求注入 `transport` 依赖。现在 `transport` 是一个依赖项，替换成我们想要的任何双份都变得容易得多。

甚至可以公开这个对象的固定配置，按照需要替换双份：

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
    build_status.transport.post.assert_called_with(
        build_status.endpoint, json=expected_payload
    ) 
```

如第一章所述，编写干净代码的目标是编写可维护的代码，我们可以重构它，使其能够根据更多需求进行演变和扩展。为此，测试非常有帮助。但鉴于测试如此重要，我们还需要重构它们，以便它们也能随着代码的演变保持其相关性和有用性。这是下一节讨论的主题。

## 生产代码并非唯一会演变的东西

我们一直说单元测试和产品代码一样重要。如果我们对产品代码足够小心，以创建最佳的可能抽象，那么为什么不对单元测试也这样做呢？

如果单元测试代码和主代码一样重要，那么考虑到可扩展性并尽可能使其易于维护是明智的。毕竟，这是除了其原始作者之外的其他工程师必须维护的代码，所以它必须易于阅读。

我们之所以如此关注代码的灵活性，是因为我们知道需求会随着时间的变化而变化和发展，最终，随着领域业务规则的变化，我们的代码也必须改变以支持这些新需求。由于产品代码为了支持新需求而改变，反过来，测试代码也必须改变以支持产品代码的新版本。

在我们使用的第一个例子中，我们为合并请求对象创建了一系列测试，尝试了不同的组合并检查了合并请求留下的状态。这是一个好的起点，但我们能做得更好。

一旦我们更好地理解了问题，我们就可以开始创建更好的抽象。在这方面，首先想到的是我们可以创建一个更高层次的抽象来检查特定条件。例如，如果我们有一个针对`MergeRequest`类的特定测试套件的对象，我们知道它的功能将仅限于这个类的行为（因为它应该符合 SRP），因此我们可以在这个测试类上创建特定的测试方法。这些方法只对这个类有意义，但将有助于减少大量的样板代码。

而不是重复相同的结构中的断言，我们可以创建一个封装这个并跨所有测试重用的方法：

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

如果我们检查合并请求状态的方式发生变化（或者让我们说我们想要添加额外的检查），那么只有一个地方（`assert_approved()`方法）需要修改。更重要的是，通过创建这些高级抽象，最初仅仅是单元测试的代码开始演变成可能最终成为一个具有自己 API 或领域语言的测试框架，使测试更加声明式。

# 更多关于测试的内容

通过我们至今为止回顾的概念，我们知道如何测试我们的代码，从测试的角度思考我们的设计，并配置项目中的工具以运行自动测试，这些测试将给我们一些关于我们所编写的软件质量的信心。

如果我们对代码的信心取决于其上的单元测试，我们如何知道它们足够吗？我们如何确保我们已经足够全面地覆盖了测试场景，并且没有遗漏任何测试？谁说这些测试是正确的？也就是说，谁测试测试？

关于我们编写的测试的彻底性，问题的第一部分通过超越我们的测试努力，通过基于属性的测试来回答。

问题的第二部分可能会有来自不同观点的多个答案，但我们将简要提及变异测试作为一种确定我们的测试确实正确的方法。在这种情况下，我们认为单元测试检查我们的主要生产代码，这也作为单元测试的一个控制。

## 基于属性的测试

基于属性的测试包括为测试用例生成数据，以找到之前单元测试未覆盖的会导致代码失败的场景。

这个功能的主要库是 `hypothesis`，配置后与我们的单元测试一起，将帮助我们找到会导致代码失败的问题数据。

我们可以想象这个库的作用是找到我们代码的反例。我们编写我们的生产代码（以及为其编写的单元测试！），并声称它是正确的。现在，有了这个库，我们定义了一个必须适用于我们代码的 `hypothesis`，如果有一些情况下我们的断言不成立，`hypothesis` 将会提供一组导致错误的数据。

单元测试的最好之处在于它们让我们更深入地思考我们的生产代码。`hypothesis` 的最好之处在于它让我们更深入地思考我们的单元测试。

## 变异测试

我们知道测试是我们确保代码正确性的正式验证方法。那么，确保测试正确的是什么？你可能认为，生产代码，是的，从某种意义上说这是正确的。我们可以将主要代码视为我们测试的平衡。

编写单元测试的目的是保护我们免受错误的影响，并测试我们不想在生产中发生的失败场景。测试通过是好事，但如果它们通过的原因不正确，那就不好了。也就是说，我们可以将单元测试用作自动回归工具——如果有人在代码中引入了错误，我们期望至少有一个测试能够捕获它并失败。如果这种情况没有发生，要么是缺少测试，要么是我们已有的测试没有进行正确的检查。

这就是变异测试背后的想法。使用变异测试工具，代码将被修改为新的版本（称为 **变异体**），这些版本是原始代码的变体，但其中一些逻辑被改变（例如，运算符被交换，条件被反转）。

一个好的测试套件应该捕捉这些变异体并将它们消灭，在这种情况下，这意味着我们可以依赖测试。如果有些变异体在实验中幸存下来，这通常是一个坏兆头。当然，这并不完全准确，所以我们可能会忽略一些中间状态。

为了快速展示这是如何工作的，并让你对这一过程有一个实际的概念，我们将使用一个不同的代码版本，该版本根据批准和拒绝的数量计算合并请求的状态。这次，我们更改了代码以使其成为一个简单的版本，该版本基于这些数字返回结果。我们将枚举与状态常量一起移动到单独的模块中，使其看起来更紧凑：

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

现在我们将添加一个简单的单元测试，检查一个条件及其预期的`result`：

```py
# file: test_mutation_testing_1.py
class TestMergeRequestEvaluation(unittest.TestCase):
    def test_approved(self):
        result = evaluate_merge_request(3, 0)
        self.assertEqual(result, Status.APPROVED) 
```

现在，我们将使用`pip install mutpy`安装`mutpy`，一个 Python 的突变测试工具，并告诉它运行这个模块的突变测试。以下代码针对不同的情况运行，这些情况通过更改`CASE`环境变量来区分：

```py
$ PYTHONPATH=src mut.py \
    --target src/mutation_testing_${CASE}.py \
    --unit-test tests/test_mutation_testing_${CASE}.py \
    --operator AOD `# delete arithmetic operator`\
    --operator AOR `# replace arithmetic operator` \
    --operator COD `# delete conditional operator` \
    --operator COI `# insert conditional operator` \
    --operator CRP `# replace constant` \
    --operator ROR `# replace relational operator` \
    --show-mutants 
```

如果你运行上一个命令针对案例 2（也可以通过`make mutation CASE=2`来运行），结果将类似于以下内容：

```py
[*] Mutation score [0.04649 s]: 100.0%
   - all: 4
   - killed: 4 (100.0%)
   - survived: 0 (0.0%)
   - incompetent: 0 (0.0%)
   - timeout: 0 (0.0%) 
```

这是一个好兆头。让我们分析一个特定实例来了解发生了什么。输出中的一行显示了以下突变体：

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

注意，这个突变体由原始版本组成，第 11 行中的运算符已更改（`>`改为`<`），结果告诉我们这个突变体被测试杀死了。这意味着在这个代码版本中（让我们想象有人不小心做了这个更改），函数的结果将是`APPROVED`，由于测试期望它是`REJECTED`，所以测试失败，这是一个好兆头（测试捕捉到了引入的 bug）。

突变测试是确保单元测试质量的好方法，但它需要一些努力和仔细的分析。在复杂环境中使用这个工具时，我们将不得不花时间分析每个场景。同样，运行这些测试的成本也很高，因为它需要运行不同版本的代码的多次运行，这可能会消耗过多的资源，并且可能需要更长的时间来完成。然而，如果需要手动进行这些检查，成本将更高，并且需要更多的努力。完全不进行这些检查可能风险更大，因为我们可能会危及测试的质量。

## 测试中的常见主题

我想简要地提及一些在思考如何测试我们的代码时通常值得记住的话题，因为它们是反复出现的并且很有帮助。

当你试图为代码编写测试时，你通常会想要考虑这些点，因为它们会导致无情的测试。当你编写单元测试时，你的心态必须全部集中在破坏代码上：你想要确保你找到错误以便修复它们，并且它们不会滑入生产环境（这将更糟）。

### 边界或极限值

边界值通常是代码中的麻烦之源，所以这可能是一个好的起点。查看代码并检查围绕某些值设置的条件。然后，添加测试以确保你包括了这些值。

例如，在如下代码行中：

```py
if remaining_days > 0: ... 
```

明确为 `0` 添加测试，因为这看起来是代码中的一个特殊情况。

更普遍地，在一个检查值范围的条件下，检查区间的两端。如果代码处理数据结构（如列表或栈），检查空列表或满栈，并确保索引始终设置正确，即使是对于极限值。

### 等价类

等价类是在集合上的一个划分，使得该划分中的所有元素在某个函数下是等价的。因为该划分内的所有元素都是等价的，所以我们只需要其中一个作为代表来测试该条件。

为了给出一个简单的例子，让我们回顾一下在演示代码覆盖率的章节中使用的上一段代码：

```py
def my_function(number: int):
    return "even" if number % 2 == 0 else "odd" 
```

在这里，函数有一个单独的 `if` 语句，根据该条件返回不同的数据。

如果我们想通过规定输入测试值集 `S` 是整数集来简化这个函数的测试，我们可以争论它可以分为两个部分：偶数和奇数。

因为这段代码对偶数执行某些操作，对奇数执行其他操作，所以我们可以说这些是我们的测试条件。也就是说，我们只需要每个子集的一个元素来测试整个条件，不需要更多。换句话说，用 2 进行测试与用 4 进行测试相同（两种情况下都执行了相同的逻辑），所以我们不需要两者，只需要其中一个（任何）即可。同样适用于 1 和 3（或任何其他奇数）。

我们可以将这些代表性元素分开成不同的参数，并通过使用 `@pytest.mark.parametrize` 装饰器运行相同的测试。重要的是要确保我们覆盖了所有情况，并且我们没有重复元素（也就是说，我们不会添加两个具有相同分区元素的不同的参数化，因为这不会增加任何价值）。

通过等价类进行测试有两个好处：一方面，我们通过不重复那些对我们的测试场景没有增加任何东西的新值来有效地测试，另一方面，如果我们耗尽了所有等价类，那么我们对要测试的场景就有很好的覆盖率。

### 边缘情况

最后，尝试添加所有你能想到的边缘情况的特定测试。这很大程度上取决于你编写的业务逻辑和代码的特异之处，并且与测试边界值的概念有所重叠。

例如，如果你的代码部分处理日期，确保你测试闰年、2 月 29 日以及新年前后。

到目前为止，我们假设我们在编写代码之后编写测试。这是一个典型的情况。毕竟，大多数时候，你会发现自己在处理一个已经存在的代码库，而不是从头开始。

有一种替代方案，即在编写代码之前先编写测试。这可能是因为你正在启动一个新的项目或功能，并且希望在编写实际的生产代码之前看到它的样子。或者，可能是因为代码库中存在缺陷，你首先想要编写一个测试来重现它，然后再着手修复。这被称为**测试驱动设计**（**TDD**），将在下一节中进行讨论。

## 测试驱动开发的简要介绍

有整本书只专注于 TDD，所以在这个书中全面覆盖这个主题是不现实的。然而，这是一个如此重要的主题，以至于它必须被提及。

TDD 背后的想法是，测试应该在生产代码之前编写，以便生产代码只编写来响应由于缺少功能实现而失败的测试。

我们之所以想要先编写测试再编写代码，有多个原因。从实用主义的角度来看，我们将非常准确地覆盖我们的生产代码。由于所有生产代码都是为响应单元测试而编写的，因此不太可能有测试遗漏了功能（当然，这并不意味着有 100%的覆盖率，但至少所有主要功能、方法或组件都将有相应的测试，即使它们并没有完全被覆盖）。

工作流程简单，从高层次来看，包括三个步骤：

1.  编写一个单元测试来描述代码应该如何表现。这可能是指尚未存在的新功能，或者当前有问题的代码，在这种情况下，测试描述了期望的场景。第一次运行此测试必须失败。

1.  对代码进行最小更改，使其通过那个测试。现在测试应该通过了。

1.  改进（重构）代码，并再次运行测试，确保它仍然有效。

这个周期被普及为著名的**红-绿-重构**，意味着一开始测试会失败（红色），然后我们让它们通过（绿色），然后我们继续重构代码并迭代它。

# 摘要

单元测试是一个非常有趣且深入的主题，但更重要的是，它是干净代码的关键部分。最终，单元测试决定了代码的质量。单元测试通常充当代码的镜子——当代码易于测试时，它清晰且设计正确，这将在单元测试中得到反映。

单元测试的代码与生产代码一样重要。适用于生产代码的所有原则也适用于单元测试。这意味着它们应该以同样的努力和细致来设计和维护。如果我们不关心我们的单元测试，它们将开始出现问题并变得有缺陷（或有问题），结果变得无用。如果发生这种情况，并且难以维护，它们就变成了负担，使事情变得更糟，因为人们往往会忽略它们或完全禁用它们。这是最糟糕的情况，因为一旦发生这种情况，整个生产代码就处于危险之中。盲目地前进（没有单元测试）是灾难的配方。

幸运的是，Python 提供了许多单元测试工具，这些工具既包含在标准库中，也通过`pip`可用。它们非常有帮助，投入时间来配置它们从长远来看是值得的。

我们已经看到单元测试是如何作为程序的正式规范，以及证明软件按照规范工作的证据，我们还了解到，在发现新的测试场景时，总有改进的空间，我们总是可以创建更多的测试。从这个意义上说，通过不同的方法（如基于属性的测试或突变测试）扩展我们的单元测试是一个好的投资。

在下一章中，我们将学习设计模式及其在 Python 中的应用。

# 参考文献

这里有一份你可以参考的信息列表：

+   Python 标准库中的`unittest`模块提供了关于如何开始构建测试套件的全面文档：[`docs.python.org/3/library/unittest.html`](https://docs.python.org/3/library/unittest.html)

+   Hypothesis：[`hypothesis.readthedocs.io/en/latest/`](https://hypothesis.readthedocs.io/en/latest/)

+   `Pytest`的官方文档：[`docs.pytest.org/en/latest/`](https://docs.pytest.org/en/latest/)

+   *《大教堂与市集：一位意外革命家的 Linux 与开源沉思（CatB）》（The Cathedral and the Bazaar: Musings on Linux and Open Source by an Accidental Revolutionary (CatB)）*，由埃里克·S·雷蒙德（Eric S. Raymond）撰写（出版社：O'Reilly Media，1999 年）

+   代码重构：[`refactoring.com/`](https://refactoring.com/)

+   *《软件测试的艺术》*，由*Glenford J. Myers*撰写（出版社：Wiley；第 3 版，2011 年 11 月 8 日）

+   编写可测试的代码：[`testing.googleblog.com/2008/08/by-miko-hevery-so-you-decided-to.html`](https://testing.googleblog.com/2008/08/by-miko-hevery-so-you-decided-to.html)
