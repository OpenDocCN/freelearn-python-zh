

# 第十五章：Flask 单元测试

**单元测试**是软件开发的一个关键阶段，它保证了应用程序每个组件的正确运行。在*第七章*“React 单元测试”中，我们讨论了与 React 组件相关的单元测试，这是构建可靠用户界面以构建网络应用程序前端部分的过程。在后台开发中，单元测试的原则相似，只是你使用的是不同的编程语言——或者更确切地说，你仍然在与后台技术栈一起工作。

单元测试确保软件应用程序的每个组件或模块在与其他应用程序部分隔离的情况下正确工作。通过单独和彻底地测试每个单元，开发人员可以在开发周期的早期识别和修复问题，这可以从长远来看节省时间和精力。

单元测试有助于早期发现缺陷，并为重构代码提供安全网，使得随着时间的推移维护和演进应用程序变得更加容易。最终，单元测试的目标是产生符合用户需求和期望的高质量软件。

在本章中，我们将简要讨论单元测试在 Flask 中的重要性，并探讨使用 pytest 作为 Flask 应用程序的测试框架的好处。我们还将涵盖 pytest 的安装和设置过程，以及测试驱动开发（TDD）的基本原理。

此外，我们还将深入探讨编写基本测试和断言以及处理异常。在本章结束时，你将能够理解单元测试在 Flask 应用程序中的重要性，描述 pytest 是什么以及它与其他测试框架的区别，以及如何将 pytest 集成到现有的项目中。

你还将学习如何使用 pytest 测试 JSON API，了解如何向 API 端点发送请求并验证响应数据。最后，你将能够应用测试驱动开发（TDD）原则，在编写实际代码之前编写测试，并使用测试来指导开发过程。

在本章中，我们将涵盖以下主题：

+   Flask 应用程序中的单元测试

+   介绍 pytest

+   pytest 的设置

+   pytest 的基本语法、结构和功能

+   编写单元测试

+   测试 JSON API

+   使用 Flask 进行测试驱动开发

+   处理异常

# 技术要求

本章的完整代码可在 GitHub 上找到：[`github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter15`](https://github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter15)

# Flask 应用程序中的单元测试

**Flask**就像厨师的刀一样，对于网络开发者来说——它是一个多才多艺的工具，可以帮助你迅速制作出可扩展和灵活的应用程序。然而，随着 Flask 应用程序复杂性的增加，确保应用程序的所有组件正确协同工作变得越来越困难。这就是单元测试发挥作用的地方。

单元测试是一种软件测试技术，它涉及在隔离于应用程序其余部分的情况下测试应用程序的每个组件或模块。通过单独和彻底地测试每个单元，开发人员可以在开发过程的早期阶段识别和修复问题。单元测试的实践有助于快速发现缺陷，并在进行更改或修改代码时提供保障，从而使得随着时间的推移维护和演进应用程序变得更加容易。

使用 Flask 应用程序，单元测试有助于确保所有路由、视图和其他组件按预期工作。单元测试还可以帮助捕捉与数据库交互、外部 API 调用和其他外部依赖项的问题。

测试启发式方法或原则如下：

+   **首次**：快速、独立、可重复、自我验证和及时

+   **RITE**：可读性、隔离性、全面性和明确性

+   **3A**：安排、行动、断言

这些原则可以作为开发者的指南和最佳实践，确保他们的单元测试工作有效。这些测试原则可以提高代码质量，最小化错误和缺陷，并最终向应用程序用户提供更优质的软件产品。通过遵守这些原则，开发人员和测试人员可以提高代码库的整体可靠性和可维护性。

让我们简要地审视这些测试原则，以了解它们如何指导你编写出色的单元测试。

## FIRST

FIRST 强调单元测试运行快速、不依赖于外部因素、可重复运行而不产生副作用、自我检查和及时编写的重要性：

+   `pytest_mock`插件。

+   **独立**：单元测试应该设计为相互独立运行，以便一个测试的失败不会影响其他测试的执行。在 Flask 中，我们可以通过在每个测试之前使用 Flask 测试客户端重置应用程序状态来实现测试之间的独立性。

+   **可重复**：单元测试应该设计为每次运行时都能产生相同的结果，无论它们在哪个环境中执行。这意味着正在测试的单元不应该依赖于外部因素，如系统时间或随机数生成器，这些因素可能会引入测试结果的可变性。

+   **自检性**：单元测试应该设计成能够检查其结果并报告失败，而不需要人工干预。这意味着单元测试应该包含断言，比较预期的结果与测试的实际结果。在 Flask 中，我们可以使用内置的断言语句来检查测试结果。

+   **及时性**：单元测试应该设计成能够及时编写，理想情况下在它们所测试的代码编写之前。这意味着它们应该是开发过程的一部分，而不是事后考虑。在 Flask 中，我们可以遵循 TDD（测试驱动开发）方法来确保测试在代码编写之前完成。

接下来，我们将探讨 RITE（可重复、隔离、全面和可扩展）这一测试原则，它可以提高单元测试的有效性并提升代码质量。

## RITE

RITE 强调单元测试易于阅读和理解的重要性，它们应该与其它组件隔离，覆盖所有可能的场景，并且在断言中明确：

+   **可重复性**：测试应该能够在不同的系统和环境中重复。这意味着测试不应该依赖于外部因素，如网络连接、时间或其他系统资源。通过确保测试可以在不同的环境中一致运行，开发者可以确信他们的代码按预期工作。

+   **隔离性**：测试应该相互独立，不共享任何状态。这意味着每个测试都应该从一个干净的状态开始，不依赖于任何之前的测试结果或全局状态。通过隔离测试，开发者可以确保每个测试都在测试特定的功能部分，并且不受系统其他部分的影响。

+   **全面性**：测试应该测试系统的所有方面，包括边缘情况和错误条件。这意味着开发者应该努力创建尽可能覆盖代码库的测试，包括所有可能的输入和输出。

+   **可扩展性**：测试应该易于扩展和维护，随着系统的演变而发展。这意味着测试应该设计成能够适应代码库的变化，例如新功能或系统架构的变化。

简而言之，RITE 原则是有益的，因为它们可以帮助你提高代码的质量、可靠性和可维护性。

在前进的过程中，我们将探讨 3A（安排、行动和断言）这一单元测试方法，它可以使你的单元测试更易于阅读和维护。

## 3A

3A 是一个简单的单元测试结构指南，包括三个步骤——安排（Arrange）、行动（Act）和断言（Assert）。安排阶段设置测试场景，行动阶段执行被测试的操作，断言阶段检查预期的结果。3A 原则是设计和编写有效单元测试的最佳实践：

+   **安排（Arrange）**：在这个步骤中，你通过初始化对象、设置变量和其他必要操作来设置测试的条件。这确保了测试环境得到正确配置，并且被测试的系统处于预期的状态。

+   **行动（Act）**：在这个步骤中，你执行被测试的动作或方法调用。这可能包括向函数传递参数、在对象上调用方法或向 API 端点发出请求。关键是确保所采取的行动是具体且针对被测试的功能的。

+   **断言（Assert）**：在这个步骤中，你验证动作的结果是否与预期结果相符。这通常涉及到检查函数返回的值、比较方法调用前后对象的状态，或者确保 API 端点返回正确的响应状态码和数据。

接下来，我们将探讨 Pytest 作为一个广泛使用的测试框架，它能够无缝地与 Flask 集成。Pytest 赋予开发者高效创建和执行单元测试、集成测试等能力，确保 Flask Web 应用的健壮性和可靠性。

# 介绍 Pytest

**Pytest**是一个开源的 Python 测试框架，它简化了编写和执行简洁、易读测试的过程。Pytest 提供了一种简单灵活的方式来编写测试，并自带广泛的支持测试选项，包括功能测试、单元测试和集成测试。

由于其易用性、强大的固定系统以及与其他 Python 测试工具的集成，Pytest 在 Python 开发者中得到了广泛的应用。Pytest 具有自动发现并运行项目中所有测试的`-test`发现能力。Pytest 生成详细的报告，为开发者提供了对测试结果的宝贵见解。

这些报告包括关于执行测试的数量、每个测试的运行时间以及发生的任何失败或错误的信息。这些信息可以帮助开发者迅速定位并解决问题，从而提高代码库的整体质量。Pytest 拥有一个庞大的用户和贡献者社区，他们积极开发和维护扩展 Pytest 功能的插件。

有趣的是，Pytest 与其他测试框架（如`unittest`、`nose`、`doctest`、`tox`、`hypothesis library`和`robot framework`）相比，以其简洁和强大、多功能性和社区支持而不同，提供了易于使用的测试功能以及详细的报告。Pytest 无疑是 Python 开发者进行单元测试和其他测试需求的热门选择。

接下来，我们将逐步介绍如何设置 Pytest 并创建我们的第一个测试。

# 设置 Pytest

测试 Python 代码是开发过程中的一个重要部分，Pytest 是实现强大测试环境的有力工具。在本节中，我们将向您介绍设置 Pytest 的步骤，并将您的 Python 代码测试体验从业余水平提升到专业水平，提供高级功能和能力，使测试更快、更简单、更有效。

要设置 Pytest，您可以按照以下步骤操作：

1.  `pip`是 Python 的包安装器。在`bizza/backend/`项目目录中打开您的终端或命令提示符，并运行以下命令：

    ```py
    pip install pytest
    ```

    前一行安装了 Pytest 及其所有依赖项。

1.  在您的项目目录中的`test_addition.py` – 即`bizza/backend/tests/test_addition.py`。这是一个简单的示例测试文件，用于热身。

1.  在`test_addition.py`中，使用以下格式编写一个简单的测试函数：

    ```py
    def test_function_name():    assert expression
    ```

    让我们讨论前面的简短格式片段：

    +   `test_function_name`代表测试函数的名称。

    +   `expression`代表您想要测试的代码。

    +   `assert`语句检查表达式是否为真，如果表达式为假，则引发错误。

注意

在 Pytest 中，测试函数通过其名称识别，并且应该以`test_`前缀开头。使用这种命名约定，Pytest 可以识别您的函数作为测试并自动运行它们。当您在终端中运行 Pytest 时，Pytest 会在您的代码库中搜索任何以`test_`开头的函数。然后，Pytest 执行这些函数并报告测试结果。

现在，让我们描述一个测试函数，该函数测试添加两个数是否产生预期的结果：

```py
def test_addition():    assert 1 + 1 == 2
```

前面的代码显示了一个简单的 Pytest 测试函数，该函数测试两个数的相加。函数的名称以`test_`开头，这告诉 Pytest 它是一个测试函数。

函数体包含一个断言，检查`1 + 1`是否等于`2`。如果断言为`true`，则测试通过。如果断言为`false`，则测试失败，Pytest 会报告错误。

1.  在`bizza/backend/`。运行以下命令以运行您的测试：

    ```py
    pytest
    (venv) C:\bizza\backend>pytest========================================================================= test session starts =========================================================================platform win32 -- Python 3.10.1, pytest-7.3.1, pluggy-1.0.0rootdir: C:\bizza\backendplugins: Faker-16.6.0collected 1 itemtests\test_addition.py [100%]========================================================================= 1 passed in 21.61s ==========================================================================
    ```

    让我们看一下前面的输出：

    1.  上一段代码的第一行显示了有关平台和 Python、Pytest 以及其他相关插件的版本信息。

    1.  第二行指示测试的根目录。在这种情况下，它是`C:\bizza\backend`。

    1.  第三行显示 Pytest 已收集一个测试项，该测试项存储在`tests\test_addition.py`文件中。

    1.  第四行显示了测试结果：一个单独的点表示测试通过。如果测试失败，这将显示为`"F"`。

    1.  第五行显示了一些摘要信息，包括通过测试的数量和运行测试所需的时间。

    1.  最后，命令提示符返回，表示测试已运行完成。

假设`test_addition.py`函数的输出已更改为`5`而不是`2`。我们应该期待测试失败吗？当然，是的！测试应该失败。以下为失败的测试输出：

```py
(venv) C:\bizza\backend>pytest================================================= test session starts =================================================
collected 1 item
tests\test_addition.py F                              [100%]
====================================================== FAILURES =======================================================
____________________________________________________ test_addition ____________________________________________________
    def test_addition():
>      assert 1 + 1 == 5
E      assert (1 + 1) == 5
tests\test_addition.py:3: AssertionError
```

前面的输出表明名为`test_addition.py`的测试失败了。断言`1 + 1 == 5`失败，因为 1 + 1 的实际结果是 2，而不是 5。

准备好下一步了吗？让我们来检查 Pytest 的基本语法和结构。然后，我们将深入探讨使用 Pytest 进行单元测试。

# Pytest 的基本语法、结构和功能

Pytest 测试函数的基本语法和结构可以表示如下：

```py
def test_function_name():    # Arrange: set up the necessary test data or
      environment
    # Act: execute the code being tested
    result = some_function()
    # Assert: check that the expected behavior is observed
    assert result == expected_result
```

`test_function_name`应该是一个描述性的名称，传达测试的目的：

+   `Arrange`部分设置必要的测试数据或环境，例如初始化对象或连接到数据库

+   `Act`部分执行被测试的代码，例如调用一个函数或执行特定的操作

+   `Assert`部分检查是否观察到了预期的行为，使用断言来验证代码的输出或行为是否符合预期

Pytest 支持广泛的断言，包括`assert x == y, assert x != y, assert x in y,`等等。Pytest 还支持使用 fixtures，可以用来管理测试依赖关系和设置测试数据和环境。

Pytest 测试函数的基本语法和结构旨在使编写清晰、简洁的测试变得容易，以验证代码按预期工作。使用 Pytest 的结构和 fixtures 的使用，你可以编写可靠、可重复且易于维护的测试。

接下来，我们将探讨 Pytest 的一个关键特性：** fixtures**。

## 使用 fixtures

在软件测试中，**fixture**是为测试运行所需定义的状态或数据集。本质上，fixtures 是帮助管理和提供一致资源（如数据、配置或对象）的函数，这些资源用于测试套件中的不同测试用例。Fixtures 使你能够为测试建立稳定且受控的环境。

它们确保每个测试用例都能访问所需的资源，而不会在多个测试中重复设置和清理方法。你可能想知道设置和清理方法是什么。让我们暂停一下，更详细地了解一下测试 Flask 应用程序中的这对组合。

在单元测试中，设置和清理方法的概念是准备和清理测试环境的关键技术，用于在每个测试用例执行前后。在深入测试用例之前，设置过程开始发挥作用。设置方法在每个测试用例之前执行，其目的是建立测试所需的条件。

例如，让我们考虑一个 Flask 单元测试场景；设置方法可以被设计成模拟 Flask 应用程序实例并配置测试客户端，从而为模拟 HTTP 请求和响应提供必要的测试基础设施。

相反，还有拆卸阶段。拆卸过程在每个测试用例执行后进行，涉及清理在设置操作期间最初建立的资源。回到 Flask 单元测试的例子，拆卸方法可能被编程为优雅地终止测试客户端并关闭 Flask 应用程序实例。这确保了没有残留的资源保持活跃，可能会干扰后续的测试。

这对设置和拆卸通常位于封装测试用例套件的类的范围内。为了更好地理解，考虑以下代码片段，它展示了如何将设置和拆卸方法结合到一个类中，以验证 Flask 应用程序：

```py
class FlaskTestCase:    def setup(self):
        self.app = create_app()
        self.client = app.test_client()
    def teardown(self):
        self.app = None
        self.client = None
    def test_index_page(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert response.content == b"Bizza Web Application"
```

在前面的代码中，设置方法创建了一个 Flask 应用程序实例和一个测试客户端。另一方面，拆卸方法优雅地结束测试客户端并处理 Flask 应用程序实例。结果是，一旦测试结束，资源就得到了整洁有序的关闭。

然而，在 pytest 中，可以使用固定装置来模拟设置和拆卸范式。固定装置充当为多个测试用例提供共享资源的函数。固定装置允许你定义和管理测试依赖项。这就是 pytest 中固定装置的工作方式。你使用`@pytest.fixture`装饰器定义一个固定装置。然后，这个函数可以作为测试函数的参数使用，允许测试函数访问固定装置的数据或环境。

当运行测试函数时，pytest 会自动检测任何定义为参数的固定装置，并首先运行这些固定装置函数，将它们的返回值作为参数传递给测试函数。这确保了测试函数可以访问它运行正确所需的数据或环境。

以下代码片段展示了可以用来生成 Flask 应用程序实例和测试客户端的固定装置：

```py
import pytest@pytest.fixture()
def app():
    app = create_app()
    return app
@pytest.fixture()
def client(app):
    client = app.test_client()
    return client
```

上述代码显示，`app`固定装置创建了一个 Flask 应用程序实例，而`client`固定装置创建了一个测试客户端。这些固定装置然后可以在测试套件中的测试用例中使用，以获取对 Flask 应用程序和测试客户端的访问。

值得注意的是，采用固定装置进行设置和拆卸的一个明显优势是它们的可重用性。通过使用固定装置，设置和拆卸逻辑可以高效地在多个测试用例之间共享。这无疑将确保测试代码更加易于维护，并且通过扩展，提高测试用例的重用性。

测试中的固定装置可以提供以下明确的好处：

+   **可重用性**：你可以定义一个 fixture 一次，并在多个测试中使用它。这可以节省时间并减少重复。

+   **可读性**：通过将设置代码分离到 fixture 函数中，你的测试函数可以更加专注且易于阅读。

+   **可维护性**：Fixtures 确保即使你的代码库在演变过程中，你的测试也是一致的和可重复的。

pytest 中的 Fixtures 提供了一个强大且灵活的机制来管理测试依赖关系并简化你的测试工作流程。

现在，让我们深入探讨 pytest 中的参数化。使用 pytest 中的参数化测试可以让你用更少的代码重复来更彻底地测试你的代码。

## pytest 中的参数化

在 pytest 中**参数化**测试是一个功能，它允许你编写一个可以执行不同输入参数集的单个测试函数。当你想用各种输入或配置测试一个函数或方法时，这非常有用。

要在 pytest 中参数化一个测试函数，你可以使用`@pytest.mark.parametrize`装饰器。这个装饰器接受两个参数：参数的名称和表示要测试的不同参数集的值或元组列表。

让我们探索 pytest 中的参数化测试函数：

```py
import pytestdef add(a, b):
    return a + b
@pytest.mark.parametrize("a, b, expected_result", [
    (1, 2, 3),
    (10, 20, 30),
    (0, 0, 0),
    (-1, 1, 0), ids=["1+2=3", "10+20=30", "0+0=0",
        "-1+1=0"]
])
def test_addition(a, b, expected_result):
    assert add(a, b) == expected_result
```

上述代码是 pytest 中参数化测试的演示，用于测试具有多个输入值的函数。

被测试的函数是`add(a, b)`，它接受两个参数`a`和`b`，并返回它们的和。`@pytest.mark.parametrize`装饰器用于提供输入值列表及其对应的预期结果。

装饰器接受三个参数：

+   一个以逗号分隔的参数名称字符串——在本例中为`"a, b, expected_result"`。

+   表示参数集及其预期结果的元组列表。在本例中，我们有四个参数集：`(1, 2, 3)`、`(10, 20, 30)`、`(0, 0, 0)`和`(-1, 1, 0)`。

+   一个可选的`ids`参数，它为测试用例提供自定义名称。

对于列表中的每个参数集，pytest 将使用相应的`a`、`b`和`expected_result`值执行`test_addition()`函数。测试函数中的`assert`语句检查`add(a, b)`的实际结果是否与预期结果匹配。

当测试函数执行时，pytest 将为每个参数集生成一个单独的报告，这样你可以确切地看到哪些案例通过了，哪些失败了：

+   第一个参数集`(1, 2, 3)`测试`add()`函数是否正确地将`1`和`2`相加，结果为`3`

+   第二个参数集`(10, 20, 30)`测试`add()`是否正确地将`10`和`20`相加，结果为`30`

+   第三个参数集`(0, 0, 0)`测试`add()`是否正确地将两个零相加，结果为`0`

+   第四个参数集`(-1, 1, 0)`测试`add()`是否正确地将`-1`和`1`相加，结果为`0`

参数化测试可以通过减少测试函数中的重复代码量以及更容易测试广泛的输入和配置，帮助你编写更简洁和有效的测试代码。

这还不是 pytest 功能的全部。接下来，我们将探索 pytest 中的外部依赖模拟。

## pytest 中的外部依赖模拟

**模拟外部依赖**是一种测试技术，它涉及创建外部依赖的模拟版本，如 API 或数据库，以隔离你的测试代码从这些依赖中。当你编写单元测试时，你通常只想测试测试范围内的代码，而不是它所依赖的任何外部服务或库。

这种做法有助于保持你的测试集中且快速，同时避免由于依赖可能不可用或行为不可预测的外部依赖而产生的不正确或错误的测试结果。

要创建一个模拟对象，你必须使用一个模拟框架，例如 `unittest.mock` 或 `pytest-mock`，来创建一个模仿真实对象行为的假对象。然后，你可以使用这个模拟对象在你的测试中代替真实对象，这允许你在受控环境中测试你的代码。

例如，假设你正在测试一个从外部 API 获取数据的函数。你可以使用模拟框架来创建一个模仿 API 行为的模拟对象，然后在测试中使用这个模拟对象而不是实际调用 API。这允许你在受控环境中测试你的函数行为，而不必担心网络连接或外部 API 的行为。

在你的测试中使用模拟策略可以帮助你编写更全面的测试，因为它允许你模拟错误条件或难以或无法通过真实外部依赖复制的边缘情况。例如，你可以使用模拟对象来模拟网络超时或数据库错误，然后验证你的测试代码是否正确处理了这些条件。

假设在我们的项目中有一个 `Speaker` 类，它依赖于外部的 `email_service` 模块来向演讲者发送电子邮件通知。我们想要为 `Speaker` 类编写一个测试，以验证当添加新演讲者时，`Speaker` 类会发送预期的电子邮件通知。为了实现这一点，我们可以使用 `pytest-mock` 插件来模拟 `email_service` 模块并检查是否执行了预期的调用。

让我们深入到代码实现中。

在 `bizza/backend/tests` 目录下添加 `test_speaker.py` 文件：

```py
# test_speaker.pyfrom bizza.backend.speaker import Speaker
def test_speaker_notification(mocker):
    # Arrange
    email_mock = mocker.patch(
        "bizza.backend.email_service.send_email")
    speaker = Speaker("John Darwin", "john@example.com")
    # Act
    speaker.register()
    # Assert
    email_mock.assert_called_once_with(
        "john@example.com",
        "Thank you for registering as a speaker",
        "Hello John, \n\nThank you for registering as a
        speaker. We look forward to your talk!\n\nBest
        regards,\nThe Conference Team"
    )
```

在前面的代码中，我们使用 `mocker.patch` 为 `email_service.send_email` 函数创建了一个模拟对象。然后，我们创建了一个新的 `Speaker` 对象并调用了 `Speaker` 对象的 `register()` 方法，这应该会触发发送电子邮件通知。

然后，我们使用了模拟对象的`assert_called_once_with`方法来检查预期的电子邮件是否以正确的参数发送。如果`send_email`函数以不同的参数被调用，测试将失败。

通过使用`pytest-mock`来模拟外部依赖项，我们可以将我们的测试从任何潜在的网络问题或其他`email_service`模块的依赖项中隔离出来。这使得我们的测试更加可靠，并且随着时间的推移更容易维护。

模拟外部依赖项是一种强大的技术，可以将测试代码从外部服务或库中隔离出来，并创建受控环境，允许你编写全面、可靠的测试。

# 编写单元测试

使用 pytest 编写测试涉及创建验证代码功能的测试函数。这些测试函数由 pytest 执行，可以组织成测试模块和测试包。除了测试函数之外，pytest 还提供了其他测试功能，如 fixtures、参数化和模拟，这些可以帮助你编写更健壮和高效的测试。

在本节中，我们将介绍使用 pytest 编写测试的基础知识，包括创建测试函数，使用断言来检查预期行为，以及将测试组织成测试套件。

现在，让我们集中精力编写一个应用程序用户注册组件的单元测试。

## 单元测试用户注册

单元测试是软件开发过程中的关键部分。正如之前所述，单元测试无疑允许开发者验证他们的代码是否正确且可靠地工作。单元测试特别重要的一个领域是用户注册，这是许多应用的一个关键部分。

用户注册功能通常涉及收集用户输入，验证输入，将其存储在数据库中，并向用户发送确认电子邮件。彻底测试这些功能对于确保其按预期工作以及用户可以成功且安全地注册非常重要。

在这个上下文中，单元测试可以用来验证注册功能是否正确处理各种场景，例如有效和无效的输入、重复的用户名和电子邮件确认。

让我们检查一个用户注册的单元测试实现。

### 用户创建单元测试

让我们来测试新用户是否可以被创建并保存到数据库中。在`tests`目录下创建`test_user_login_creation.py`：

```py
def test_create_user(db):    # Create a new user
    user = User(username='testuser',
        password='testpassword',
            email='test@example.com')
    #Add the user to the database
    db.session.add(user)
    db.session.commit()
    # Retrieve the user from the database
    retrieved_user = db.session.query(User)
        .filter_by(username='testuser').first()
    # Assert that the retrieved user matches the original
      user
    assert retrieved_user is not None
    assert retrieved_user.username == 'testuser'
    assert retrieved_user.email == 'test@example.com'
```

在前面的测试片段中，我们创建了一个具有特定`username`、`password`和`email address`的新用户。然后，我们将用户添加到数据库中并提交更改。最后，我们使用查询从数据库中检索用户，并断言检索到的用户在所有字段上与原始用户匹配。这个测试确保新用户可以成功创建并保存到数据库中。

### 输入验证单元测试

让我们来测试注册表单是否正确验证用户输入并返回适当的错误消息：

```py
def test_user_registration_input_validation(client, db):    # Attempt to register a new user with an invalid
      username
    response = client.post('/register',
        data={'username': 'a'*51,
            'password': 'testpassword',
                'email': 'test@example.com'})
    # Assert that the response status code is 200 OK
    assert response.status_code == 200
    # Assert that an error message is displayed for the
      invalid username
    assert b'Invalid username. Must be between 1 and 50
        characters.' in response.data
    # Attempt to register a new user with an invalid email
      address
    response = client.post('/register',
        data={'username': 'testuser',
            'password': 'testpassword',
                'email': 'invalid-email'})
    # Assert that the response status code is 200 OK
    assert response.status_code == 200
    # Assert that an error message is displayed for the
      invalid email address
    assert b'Invalid email address.' in response.data
    # Attempt to register a new user with a password that
      is too short
    response = client.post('/register',
        data={'username': 'testuser',
            'password': 'short',
                'email': 'test@example.com'})
    # Assert that the response status code is 200 OK
    assert response.status_code == 200
    # Assert that an error message is displayed for the
      short password
    assert b'Password must be at least 8 characters long.'
        in response.data
```

在前面的测试中，我们模拟了使用各种无效输入尝试注册新用户的情况，例如无效的`username`、`email address`或`password`属性过短。我们使用无效输入数据向`'/register'`端点发送`POST`请求，并断言响应状态码为`200 OK`，表示注册表单已成功提交，但存在错误。

然后，我们断言页面为每个无效输入显示了适当的错误消息。这个测试确保注册表单正确验证用户输入，并为无效输入返回适当的错误消息。

接下来，我们将检查`login`组件的单元测试。

## 单元测试用户登录

单元测试用户登录涉及测试负责验证尝试登录应用程序的用户代码的功能。这通常涉及验证用户凭证是否正确，并根据认证是否成功返回适当的响应。

在这种情况下，单元测试可以帮助确保登录过程可靠且安全，对无效登录尝试进行适当的错误处理。此外，单元测试还可以帮助识别登录过程中的潜在漏洞，例如注入攻击或密码猜测尝试。

### 有效凭证用户单元测试

让我们来测试一个使用有效凭证的用户可以成功登录并访问应用程序：

```py
def test_user_login(client, user):    # Login with valid credentials
    response = client.post('/login',
        data={'username': user.username,
            'password': user.password},
        follow_redirects=True)
    # Check that the response status code is 200 OK
    assert response.status_code == 200
    # Check that the user is redirected to the home page
      after successful login
    assert b'Welcome to the application!' in response.data
```

在前面的测试中，我们使用客户端固定值模拟用户通过向登录端点发送带有有效凭证的`POST`请求来登录。我们还使用用户固定值创建一个具有有效凭证的测试用户。在发送登录请求后，我们检查响应状态码是否为`200 OK`，以及用户是否被重定向到主页，这表明登录成功。

### 无效凭证用户单元测试

让我们来测试一个使用无效凭证的用户无法登录，并收到适当的错误信息：

```py
def test_login_invalid_credentials(client):    # Try to log in with invalid credentials
    response = client.post('/login',
        data={'username': 'nonexistentuser',
        'password': 'wrongpassword'})
    # Check that the response status code is 401
      Unauthorized
    assert response.status_code == 401
    # Check that the response contains the expected error
      message
    assert b'Invalid username or password' in response.data
```

在前面的测试中，我们尝试使用无效的用户名和密码登录，并期望服务器响应`401 Unauthorized`状态码和指示凭证无效的错误消息。

### 测试 SQL 注入攻击

让我们来测试代码是否正确验证用户输入以防止 SQL 注入攻击：

```py
def test_sql_injection_attack_login(client):    # Attempt to login with a username that contains SQL
      injection attack code
    response = client.post('/login',
        data={'username': "'; DROP TABLE users; --",
            'password': 'password'})
    # Check that the response status code is 401
      Unauthorized
    assert response.status_code == 401
    # Check that the user was not actually logged in
    assert current_user.is_authenticated == False
```

在前面的测试中，我们尝试使用 SQL 注入攻击代码作为登录表单中的`username`输入。测试检查响应状态码是否为`401 Unauthorized`，这表明攻击未成功，用户未登录。

它还检查`current_user.is_authenticated`属性是否为`False`，确认用户未认证。这个测试有助于确保代码正确验证用户输入，以防止 SQL 注入攻击。

### 测试密码强度

让我们测试代码是否正确验证用户密码以确保它们满足最小复杂度要求（例如，最小长度、特殊字符的要求等）：

```py
def test_password_strength():    # Test that a password with valid length and characters
      is accepted
    assert check_password_strength("abc123XYZ!") == True
    # Test that a password with an invalid length is rejected
    assert check_password_strength("abc") == False
    # Test that a password without any special characters
      is rejected
    assert check_password_strength("abc123XYZ") == False
    # Test that a password without any lowercase letters is
      rejected
    assert check_password_strength("ABC123!") == False
    # Test that a password without any uppercase letters is
      rejected
    assert check_password_strength("abc123!") == False
    # Test that a password without any numbers is rejected
    assert check_password_strength("abcXYZ!") == False
```

在前面的测试中，`check_password_strength()` 是一个接受密码字符串作为输入的函数，如果它满足最小复杂度要求则返回 `True`，否则返回 `False`。这个单元测试通过测试各种场景来验证该函数按预期工作。

使用测试框架 Pytest 和编写有效的单元测试，开发者可以尽早捕捉到错误和缺陷，降低生产中的错误风险，并提高代码库的整体质量和可靠性。

注意

前面的测试假设你已经设置了一个 Flask 应用程序，其中包含用户注册和登录的路由，以及一个带有用户模型的 `SQLAlchemy` 数据库。我们还假设你已经配置了一个带有 Pytest 的 Flask 测试客户端固定装置（client）的测试客户端。

接下来，我们将查看测试 JSON API 以确保 API 端点按预期工作。

# 测试 JSON API

测试 JSON API 是开发任何与外部客户端通信的 Web 应用程序的重要部分。API 提供了一种简单灵活的方式在服务器和客户端之间交换数据。在将 API 暴露给外部用户之前，确保 API 按预期工作至关重要。

单元测试 JSON API 涉及验证 API 端点对于不同类型的输入数据返回预期的结果，并处理错误情况。此外，确保 API 遵循行业标准协议并对常见 Web 漏洞具有安全性也是至关重要的。这样，开发者可以确保 Web 应用程序的可靠性和安全性，并最大限度地减少错误或安全漏洞的风险。

让我们通过一个包含四个测试的测试套件来过一遍——`test_get_all_speakers`、`test_create_speaker`、`test_update_speaker` 和 `test_delete_speaker`：

```py
import pytestimport requests
# Define the base URL for the speakers API
BASE_URL = 'https://localhost:5000/v1/api/speakers/'
def test_get_all_speakers():
    # Send a GET request to the speakers API to retrieve
      all speakers
    response = requests.get(BASE_URL)
    # Check that the response has a status code of 200 OK
    assert response.status_code == 200
    # Check that the response contains a JSON object with a
      list of speakers
    assert isinstance(response.json(), list)
```

前面的测试，`test_get_all_speakers`，向演讲者 API 发送一个 `GET` 请求检索所有演讲者，然后检查响应状态码为 `200 OK` 并包含一个包含演讲者列表的 JSON 对象。

## 测试创建演讲者数据

以下测试，`test_create_speaker`，定义了一个要创建的演讲者数据对象，向演讲者 API 发送一个 `POST` 请求使用这些数据创建一个新的演讲者，然后检查响应状态码为 `201 CREATED` 并包含一个包含新创建的演讲者数据的 JSON 对象：

```py
def test_create_speaker():    # Define the speaker data to be created
    speaker_data = {
        'name': 'John Darwin',
        'topic': 'Python',
        'email': 'john@example.com',
        'phone': '555-555-5555'
    }
    # Send a POST request to the speakers API to create a
      new speaker
    response = requests.post(BASE_URL, json=speaker_data)
    # Check that the response has a status code of 201
      CREATED
    assert response.status_code == 201
    # Check that the response contains a JSON object with
      the newly created speaker data
    assert response.json()['name'] == 'John Darwin'
    assert response.json()['topic'] == 'Python'
    assert response.json()['email'] == 'john@example.com'
    assert response.json()['phone'] == '555-555-5555'
```

## 更新演讲者数据对象

以下测试代码，`test_update_speaker`，定义了一个要更新的演讲者数据对象，向演讲者 API 发送一个 `PUT` 请求使用这些数据更新 `id 1` 的演讲者，然后检查响应状态码为 `200` 表示更新成功：

```py
def test_update_speaker():    # Define the speaker data to be updated
    speaker_data = {
        'name': 'John Doe',
        'topic': 'Python for Data Science',
        'email': 'johndoe@example.com',
        'phone': '555-555-5555'
    }
    # Send a PUT request to the speakers API to update the
      speaker data
    response = requests.put(BASE_URL + '1',
        json=speaker_data)
    # Check that the response has a status code of 200 OK
    assert response.status_code == 200
    # Check that the response contains a JSON object with
      the updated speaker data
    assert response.json()['name'] == 'John Darwin'
    assert response.json()['topic'] == 'Python for Data
        Science'
    assert response.json()['email'] == 'john@example.com'
    assert response.json()['phone'] == '555-555-5555'
```

## 测试删除演讲者数据对象

以下代码片段向 Speakers API 发送一个`DELETE`请求来删除`ID 1`的演讲者。测试函数检查响应的状态码为`204 NO CONTENT`。如果成功从 API 中删除了`ID 1`的演讲者，API 的响应应该有状态码`204 NO CONTENT`。如果找不到演讲者或删除请求中存在错误，响应状态码将不同，测试将失败：

```py
def test_delete_speaker():    # Send a DELETE request to the speakers API to delete
      the speaker with ID 1
    response = requests.delete(BASE_URL + '1')
    # Check that the response has a status code of 204 NO
      CONTENT
    assert response.status_code == 204
```

在这一点上，你可能想知道，为什么在我们应用程序中一旦出现错误，我们就需要投入时间和资源来纠正它们，而完全有可能从一开始就主动预防它们的发生？

接下来，我们将讨论使用 Flask 作为软件开发的重要主动方法！

# 测试驱动开发与 Flask

TDD 是一种软件开发方法，你需要在编写实际代码之前编写自动化测试。这个过程包括为特定的功能或功能编写测试用例，然后编写必要的最少代码以使测试通过。一旦测试通过，你将编写额外的测试来覆盖不同的边缘情况和功能，直到你完全实现了所需的功能。

以使用 Flask 的参与者端点作为案例研究，TDD 过程可能看起来是这样的：

1.  **定义功能**：第一步是定义你想要实现的功能。在这种情况下，功能是一个允许用户查看活动参与者列表的端点。

1.  **编写测试用例**：接下来，你必须编写一个测试用例来定义端点的预期行为。例如，你可能编写一个测试来检查端点返回包含参与者列表的 JSON 响应。

1.  **运行测试**：然后你运行测试，由于你还没有实现端点，所以测试将失败。

1.  **编写最少代码**：你编写必要的最少代码以使测试通过。在这种情况下，你会编写参与者端点的代码。

1.  **再次运行测试**：然后，你必须再次运行测试，这次应该会通过，因为你已经实现了端点。

1.  如果活动不存在，将出现`404`错误。

现在，让我们使用 TDD 方法实现参与者端点，从编写失败的测试用例开始，因为我们还没有实现端点。

## 定义功能

第一步是定义你想要实现的功能。在这种情况下，功能是一个允许用户查看活动参与者列表的端点。

## 编写失败的测试用例

下一步是编写一个测试用例，检查参与者端点是否返回预期的数据。这个测试最初应该会失败，因为我们还没有实现端点。

在`tests`目录内创建`test_attendees.py`，并将以下代码添加到`bizza/backend/tests/test_attendees.py`中：

```py
from flask import Flask, jsonifyimport pytest
app = Flask(__name__)
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client
def test_attendees_endpoint_returns_correct_data(client):
    response = client.get('/events/123/attendees')
    expected_data = [{'name': 'John Darwin',
        'email': 'john@example.com'},
            {'name': 'Jane Smith',
                'email': 'jane@example.com'}]
    assert response.json == expected_data
```

## 实现通过测试的最少代码

现在，我们可以实现参会者端点函数以返回硬编码的数据。这是使测试通过所需的最小代码量：

```py
# Define the attendee endpoint@app.route('/events/<int:event_id>/attendees')
def get_attendees(event_id):
    # Return a hardcoded list of attendees as a JSON
      response
    attendees = [{'name': 'John Darwin',
        'email': 'john@example.com'},
            {'name': 'Jane Smith',
                'email': 'jane@example.com'}]
    return jsonify(attendees)
```

## 运行测试并确保其通过

再次运行测试以确保它现在可以通过：

```py
$ pytest test_attendees.py----------------------------------------------------------------------
Ran 1 test in 0.001s
OK
```

## 代码重构

现在我们有了通过测试，我们可以重构代码以提高其可维护性、效率和可读性。例如，我们可以用从数据库或外部 API 检索的数据替换硬编码的数据。

## 编写额外的测试用例

最后，我们可以编写额外的测试用例以确保端点在不同场景下的行为正确。例如，我们可能会编写测试以确保端点正确处理无效输入，或者在没有找到给定活动的参会者时返回空列表。

使用 TDD（测试驱动开发）流程，你可以确保你的代码经过彻底测试，并且你已经实现了所有期望的功能。这种方法可以帮助你在开发早期阶段捕捉到错误，并使未来维护和重构代码变得更加容易。

到目前为止，我们已经讨论了 TDD 作为一种软件开发方法，其中测试是在实际代码实现之前创建的。这种方法鼓励开发者编写定义代码预期行为的测试，然后编写代码本身以使测试通过。接下来，我们将深入探讨 Flask 测试套件中的异常处理。

# 处理异常

使用单元测试处理异常是一种软件开发技术，它涉及测试代码在运行时可能遇到的不同类型的异常的处理方式。异常可能由各种因素触发，例如无效输入、意外输入或代码运行环境中的问题。

单元测试是编写小型、自动化的测试以确保单个代码单元按预期工作的实践。在处理异常方面，单元测试可以帮助确保代码能够适当地响应各种错误条件。作为开发者，你需要测试你的代码能够优雅地处理异常。你可以在受控环境中模拟这些错误条件，以便你对代码处理可能发生的异常的能力更有信心。

例如，在一个具有`attendees`端点的 Flask 应用程序中，你可能想测试应用程序如何处理没有参会者的活动请求。通过编写一个单元测试，向端点发送一个没有参会者的活动的请求，我们可以确保应用程序返回适当的错误响应代码和消息，而不是崩溃或提供不准确响应。

让我们深入探讨如何处理参会者端点的异常的代码实现：

```py
from flask import Flask, jsonifyapp = Flask(__name__)
class Event:
    def __init__(self, name):
        self.name = name
        self.attendees = []
    def add_attendee(self, name):
        self.attendees.append(name)
    def get_attendees(self):
        if not self.attendees:
            raise Exception("No attendees found for event")
        return self.attendees
@app.route('/event/<event_name>/attendees')
def get_attendees(event_name):
    try:
        event = Event(event_name)
        attendees = event.get_attendees()
    except Exception as e:
        return jsonify({'error': str(e)}), 404
    return jsonify(attendees)
```

在先前的实现中，我们向`Event`类添加了一个自定义异常，名为`Exception("No attendees found for event")`。在`get_attendees`方法中，如果没有参与者，我们将抛出此异常。在 Flask 端点函数中，我们将`Event`实例化和`get_attendees`调用包裹在`try/except`块中。

如果抛出异常，我们将返回一个包含错误信息和`404`状态码的 JSON 响应，以指示请求的资源未找到。

让我们检查测试函数：

```py
def test_get_attendees_empty():    event_name = 'test_event'
    app = create_app()
    with app.test_client() as client:
        response =
            client.get(f'/event/{event_name}/attendees')
        assert response.status_code == 404
        assert response.json == {'error': 'No attendees
            found for event'}
def test_get_attendees():
    event_name = 'test_event'
    attendee_name = 'John Doe'
    event = Event(event_name)
    event.add_attendee(attendee_name)
    app = create_app()
    with app.test_client() as client:
        response =
            client.get(f'/event/{event_name}/attendees')
        assert response.status_code == 200
        assert response.json == [attendee_name]
```

在第一个测试函数`test_get_attendees_empty()`中，我们期望端点返回`404`状态码和错误信息 JSON 响应，因为没有参与者参加该活动。在第二个测试`test_get_attendees()`中，我们向活动添加一个参与者，并期望端点返回`200`状态码和包含参与者姓名的 JSON 响应。

当你在代码中对预期的异常进行测试并优雅地处理它们时，你可以确保你的应用程序按预期行为，并在需要时向用户提供有用的错误信息。

# 摘要

单元测试作为 Flask 应用程序开发的关键方面，确保了应用软件的可靠性和功能性。在本章中，我们学习了如何为 Flask 应用程序的各个组件构建和实施有效的单元测试。我们探讨了 Pytest 如何简化测试过程并提高开发者的生产力。

本章涵盖了 Pytest 的基础知识，包括其介绍、设置过程、基本语法和功能。我们发现了设置和清理方法的重要性，这些方法有助于创建受控的测试环境，并在每个测试用例之后确保资源的适当处置。

通过应用这些技术，我们能够创建更健壮和隔离的单元测试，这些测试反映了现实世界的场景。此外，我们提供了如何编写单元测试、测试 JSON API、应用 TDD 以及处理 Flask 应用程序中的异常的指南。通过采用这些实践，开发者可以提高其 Flask 应用程序的整体质量，并最大限度地减少错误和缺陷的风险。

随着我们继续前进并结束构建健壮且可扩展的 Flask 应用程序的旅程，下一章将深入探讨容器化和部署的世界。我们将探讨如何容器化 Flask 应用程序，使我们能够复制开发环境，并轻松地将我们的应用程序部署到各种平台。

我们还将深入研究将 Flask 应用程序部署到云服务，利用 Docker 和 AWS 等平台的力量进行高效和可扩展的部署。
