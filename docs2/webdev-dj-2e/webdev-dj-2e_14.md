# 第十四章：14. 测试

概述

本章向您介绍了测试 Django Web 应用程序的概念。您将了解测试在软件开发中的重要性，尤其是在构建 Web 应用程序方面。您将为 Django 应用程序的组件编写单元测试，例如**视图**、**模型**和**端点**。完成本章后，您将具备为 Django Web 应用程序编写测试用例的技能。这样，您可以确保您的应用程序代码按预期工作。

# 简介

在前面的章节中，我们通过编写不同的组件，如数据库模型、视图和模板，来专注于构建我们的 Django Web 应用程序。我们这样做是为了提供一个交互式应用程序，让用户可以创建个人资料并为他们读过的书籍撰写评论。

除了构建和运行应用程序之外，确保应用程序代码按预期工作还有一个重要的方面。这是通过一种称为**测试**的技术来保证的。在测试中，我们运行 Web 应用程序的不同部分，并检查执行组件的输出是否与预期的输出匹配。如果输出匹配，我们可以说该组件已成功测试，如果输出不匹配，我们则说该组件未能按预期工作。

在本章中，随着我们浏览不同的部分，我们将了解测试的重要性，了解测试 Web 应用程序的不同方法，以及我们如何构建一个强大的测试策略，以确保我们构建的 Web 应用程序是健壮的。让我们从了解测试的重要性开始我们的旅程。

# 测试的重要性

确保应用程序按预期设计的方式工作是开发工作的重要方面，否则，我们的用户可能会不断遇到奇怪的行为，这通常会驱使他们远离与应用程序的互动。

我们在测试上投入的努力帮助我们确保我们打算解决的问题确实被正确解决。想象一下，一个开发者正在构建一个在线活动调度平台。在这个平台上，用户可以根据他们的本地时区在日历上安排活动。现在，如果在这个平台上，用户可以按预期安排活动，但由于一个错误，活动被安排在了错误的时间区域？这类问题往往会驱使许多用户离开。

正因如此，许多公司花费大量资金确保他们构建的应用程序已经经过彻底的测试。这样，他们可以确保不会发布有缺陷的产品或远未满足用户需求的产品。

简而言之，测试帮助我们实现以下目标：

+   确保应用程序的组件按规范工作

+   确保与不同基础设施平台的互操作性：如果一个应用程序可以部署在不同的操作系统上，例如 Linux、Windows 等

+   在重构应用程序代码时降低引入错误的可能性

现在，许多人关于测试的常见假设是他们必须手动测试所有组件，以确保每个组件按照其规范工作，每次更改或向应用程序添加新组件时都重复此操作。虽然这是真的，但这并不提供完整的测试图景。随着时间的推移，测试作为一种技术已经变得非常强大，作为开发者，你可以通过实现**自动化测试用例**来减少大量的测试工作。那么，这些自动化测试用例是什么？或者说，什么是**自动化测试**？让我们来了解一下。

# 自动化测试

当单个组件被修改时，重复测试整个应用程序可能是一项具有挑战性的任务，尤其是如果该应用程序包含大量的代码库。代码库的大小可能是由于功能数量庞大或解决的问题的复杂性。

随着我们开发应用程序，确保对这些应用程序所做的更改可以轻松测试非常重要，这样我们就可以验证是否有破坏性的东西。这就是自动化测试概念派上用场的地方。自动化测试的重点是将测试编写为代码，这样应用程序的各个组件就可以在隔离状态下以及它们相互交互的情况下进行测试。

从这个角度来看，现在对我们来说，定义可以为应用程序执行的不同类型的自动化测试变得很重要。

自动化测试可以大致分为五种不同类型：

+   **单元测试**：在这种测试类型中，代码的各个独立单元被单独测试。例如，单元测试可以针对单个方法或单个独立的 API。这种测试的目的是确保应用程序的基本单元按照其规范工作。

+   **集成测试**：在这种测试类型中，代码的各个独立单元被合并成一个逻辑分组。一旦形成这种分组，就会对这个逻辑组进行测试，以确保该组按预期的方式工作。

+   **功能测试**：在这种测试中，测试应用程序不同组件的整体功能。这可能包括不同的 API、用户界面等。

+   **冒烟测试**：在这种测试中，测试已部署应用程序的稳定性，以确保应用程序在用户与其交互时继续保持功能，而不会导致崩溃。

+   **回归测试**：这种测试是为了确保对应用程序所做的更改不会降低应用程序先前构建的功能。

如我们所见，测试是一个庞大的领域，需要时间来掌握，关于这个主题已经写出了整本书。为了确保我们突出测试的重要方面，我们将在本章中专注于单元测试的方面。

# Django 中的测试

Django 是一个功能丰富的框架，旨在使 Web 应用程序开发快速。它提供了一种全面的方式来测试应用程序。它还提供了一个良好集成的模块，允许应用程序开发者为其应用程序编写单元测试。此模块基于大多数 Python 发行版附带的 Python `unittest`库。

让我们开始了解如何在 Django 中编写基本的测试用例，以及如何利用框架提供的模块来测试我们的应用程序代码。

## 实现测试用例

当你在实现测试代码的机制时，首先需要理解的是如何逻辑地分组这种实现，以便相互紧密相关的模块可以在一个逻辑单元中进行测试。

这可以通过实现一个**测试用例**来简化。测试用例不过是一个逻辑单元，它将逻辑上相似的测试组合在一起，这样所有用于初始化测试用例环境的公共逻辑都可以组合在同一个地方，从而在实现应用程序测试代码时避免重复工作。

## Django 中的单元测试

现在，随着我们对测试的基本理解已经清楚，让我们看看我们如何在 Django 中进行单元测试。在 Django 的上下文中，一个单元测试由两个主要部分组成：

+   一个`TestCase`类，它封装了为给定模块分组的不同测试用例

+   需要执行以测试特定组件流程的实际测试用例

实现单元测试的类应该继承自 Django 的`test`模块提供的`TestCase`类。默认情况下，Django 在应用程序目录中提供了一个`tests.py`文件，可以用来存储应用程序模块的测试用例。

一旦编写了这些单元测试，它们也可以通过直接运行`manage.py`中提供的`test`命令来轻松执行，如下所示：

```py
python manage.py test
```

## 利用断言

编写测试的一个重要部分是验证测试是否通过或失败。通常，为了在测试环境中实现这样的决策，我们使用一种称为**断言**的东西。

断言是软件测试中的一个常见概念。它们接受两个操作数，并验证左边的操作数（LHS）的值是否与右边的操作数（RHS）的值匹配。如果左边的值与右边的值匹配，则认为断言成功，而如果值不同，则认为断言失败。

一个评估为`False`的断言实际上会导致测试用例被评估为失败，然后报告给用户。

Python 中的断言实现相当简单，它们使用一个简单的关键字`assert`。例如，以下代码片段展示了一个非常简单的断言：

```py
assert 1 == 1
```

前面的断言接受一个表达式，该表达式评估为`True`。如果这个断言是测试用例的一部分，那么测试就会成功。

现在，让我们看看如何使用 Python 的`unittest`库实现测试用例。这样做相当简单，可以按照以下几个易于遵循的步骤完成：

1.  导入`unittest`模块，它允许我们构建测试用例：

    ```py
    import unittest
    ```

1.  一旦模块被导入，你可以创建一个以`Test`开头的新类，该类继承自`unittest`模块提供的`TestCase`类：

    ```py
    class TestMyModule(unittest.TestCase):
        def test_method_a(self):
            assert <expression>
    ```

    只有当`TestMyModule`类继承自`TestCase`类时，Django 才能自动运行它，并且与框架完全集成。一旦类被定义，我们就可以在类内部实现一个新的方法，命名为`test_method_a()`，该方法验证断言。

    注意

    这里需要注意的一个重要部分是测试用例和测试函数的命名方案。所实现的测试用例应该以`test`为前缀，这样测试执行模块可以检测它们作为有效的测试用例并执行它们。同样的规则也适用于测试方法的命名。

1.  一旦编写了测试用例，就可以简单地通过运行以下命令来执行它：

    ```py
    python manage.py test
    ```

    现在，随着我们对实现测试用例的基本理解已经明确，让我们编写一个非常简单的单元测试来查看单元测试框架在 Django 中的行为。

## 练习 14.01：编写简单的单元测试

在这个练习中，你将编写一个简单的单元测试来了解 Django 单元测试框架的工作方式，并使用这些知识来实现你的第一个测试用例，该测试用例验证几个简单的表达式。

1.  要开始，打开`Bookr`项目下`reviews`应用的`tests.py`文件。默认情况下，该文件将只包含一行，导入 Django 的`TestCase`类。如果文件已经包含几个测试用例，你可以删除文件中除导入`TestCase`类的行之外的所有行，如下所示：

    ```py
    from django.test import TestCase
    ```

1.  在你刚刚打开的`tests.py`文件中添加以下代码行：

    ```py
    class TestSimpleComponent(TestCase):
        def test_basic_sum(self):
            assert 1+1 == 2
    ```

    在这里，你创建了一个名为 `TestSimpleComponent` 的新类，它继承自 Django 的 `test` 模块提供的 `TestCase` 类。`assert` 语句将比较左侧的表达式（`1 + 1`）与右侧的表达式（`2`）。

1.  一旦你编写了测试用例，导航回项目文件夹，并运行以下命令：

    ```py
    python manage.py test
    ```

    应该生成以下输出：

    ```py
    % ./manage.py test
    Creating test database for alias 'default'...
    System check identified no issues (0 silenced).
    .
    ----------------------------------------------------------------------
    Ran 1 test in 0.001s
    OK
    Destroying test database for alias 'default'...
    ```

    前面的输出表明 Django 的测试运行器执行了一个测试用例，该测试用例成功通过了评估。

1.  在确认测试用例正常工作并通过测试后，现在尝试在 `test_basic_sum()` 方法的末尾添加另一个断言，如下面的代码片段所示：

    ```py
        assert 1+1 == 3
    ```

1.  在 `tests.py` 文件中添加了 `assert` 语句后，现在可以从项目文件夹中运行以下命令来执行测试用例：

    ```py
    python manage.py test
    ```

在这一点上，你会注意到 Django 报告测试用例的执行失败了。

通过这种方式，你现在已经了解了如何在 Django 中编写测试用例以及如何使用断言来验证测试方法调用生成的输出是否正确。

### 断言类型

在 *练习 14.01*，*编写简单的单元测试* 中，当我们遇到以下 `assert` 语句时，我们对断言有了一个简短的接触：

```py
assert 1+1 == 2
```

这些断言语句很简单，使用了 Python 的 `assert` 关键字。在使用 `unittest` 库进行单元测试时，有几种不同的断言类型可以进行测试。让我们看看那些：

+   `assertIsNone`：这个断言用于检查一个表达式是否评估为 `None`。例如，这种类型的断言可以在查询数据库返回 `None` 的情况下使用，因为没有找到指定过滤条件下的记录。

+   `assertIsInstance`：这个断言用于验证提供的对象是否评估为提供的类型的实例。例如，我们可以验证方法返回的值是否确实为特定的类型，如列表、字典、元组等。

+   `assertEquals`：这是一个非常基础的函数，它接受两个参数并检查提供给它的参数是否在值上相等。这在你计划比较那些不保证排序的数据结构的值时可能很有用。

+   `assertRaises`：这个方法用于验证当调用它时，提供给它的方法名称是否引发了指定的异常。这在编写测试用例时很有用，其中需要测试引发异常的代码路径。例如，这种断言在确保执行数据库查询的方法（例如，让我们知道数据库连接是否尚未建立）引发异常时可能很有用。

这些只是我们可以在测试用例中做出的一小部分有用的断言。Django 测试库建立在 `unittest` 模块之上，它提供了更多可以进行测试的断言。

## 在每个测试用例运行后执行预测试设置和清理

在编写测试用例时，有时我们可能需要执行一些重复性任务；例如，设置一些测试所需的变量。一旦测试完成，我们希望清理所有对测试变量的更改，以便任何新的测试都从一个全新的实例开始。

幸运的是，`unittest`库提供了一种有用的方法，通过它可以自动化我们在每个测试用例运行之前设置环境以及在测试用例完成后清理环境时的重复性工作。这是通过以下两个方法实现的，我们可以在`TestCase`中实现这些方法。

`setUp()`：此方法在`TestCase`类中每个`test`方法执行之前调用。它实现了在测试执行之前设置测试用例环境的代码。此方法可以是一个设置任何本地数据库实例或测试变量的好地方，这些变量可能对测试用例是必需的。

注意

`setUp()`方法仅适用于在`TestCase`类内部编写的测试用例。

例如，以下示例说明了如何在`TestCase`类内部使用`setUp()`方法的一个简单定义：

```py
class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Do some initialization work
    def test_method_a(self):
        # code for testing method A
    def test_method_b(self):
        # code for testing method B
```

在前面的示例中，当我们尝试执行测试用例时，我们定义的`setUp()`方法将在每次`test`方法执行之前调用。换句话说，`setUp()`方法将在`test_method_a()`调用之前调用，然后它将在`test_method_b()`调用之前再次调用。

`tearDown()`：此方法在`test`函数执行完毕后调用，并在测试用例执行完毕后清理变量及其值。无论测试用例评估结果为`True`还是`False`，都会执行此方法。下面将展示如何使用`tearDown()`方法的示例：

```py
class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Do some initialization work
    def test_method_a(self):
        # code for testing method A
    def test_method_b(self):
        # code for testing method B
    def tearDown(self):
        # perform cleanup
```

在前面的示例中，`tearDown()`方法将在每次`test`方法执行完毕时调用，即`test_method_a()`执行完毕后，再次在`test_method_b()`执行完毕后。

现在，我们已经了解了编写测试用例的不同组件。让我们看看如何使用提供的测试框架来测试 Django 应用程序的不同方面。

# 测试 Django 模型

Django 中的模型是数据将在应用程序数据库中存储的对象表示。它们提供了可以帮助我们验证给定记录提供的数据输入的方法，以及在数据插入数据库之前对数据进行任何处理的方法。

就像在 Django 中创建模型一样容易，测试它们也同样简单。现在，让我们看看如何使用 Django 测试框架来测试 Django 模型。

## 练习 14.02：测试 Django 模型

在这个练习中，您将创建一个新的 Django 模型，并为它编写测试用例。这个测试用例将验证您的模型是否能够正确地将数据插入和从数据库中检索。这类在数据库模型上运行的测试用例在团队开发大型项目时可能非常有用，因为同一个数据库模型可能会随着时间的推移被多个开发者修改。为数据库模型实现测试用例允许开发者预先识别他们可能在不经意间引入的潜在破坏性更改：

注意

为了确保我们能够熟练地从零开始在新创建的应用程序上运行测试，我们将创建一个新的应用程序，名为 `bookr_test`。这个应用程序的代码与主 `bookr` 应用程序独立，因此，我们不会将这个应用程序的文件包含在 `final/bookr` 文件夹中。完成本章内容后，我们建议您通过为 `bookr` 应用程序的各种组件编写类似的测试来练习您所学的知识。

1.  创建一个新的应用程序，您将使用它来完成本章的练习。为此，运行以下命令，这将为您的情况设置一个新的应用程序：

    ```py
    python manage.py startapp bookr_test
    ```

1.  为了确保 `bookr_test` 应用程序的行为与 Django 项目中的任何其他应用程序相同，将此应用程序添加到 `bookr` 项目的 `INSTALLED_APPS` 部分中。为此，打开您的 `bookr` 项目的 `settings.py` 文件，并将以下代码追加到 `INSTALLED_APPS` 列表中：

    ```py
    INSTALLED_APPS = [….,\
                      ….,\
                      'bookr_test']
    ```

1.  现在，随着应用程序设置的完成，创建一个新的数据库模型，您将使用它来进行测试。对于这个练习，您将创建一个名为 `Publisher` 的新模型，该模型将存储有关书籍出版商的详细信息。要创建模型，打开 `bookr_test` 目录下的 `models.py` 文件，并将以下代码添加到其中：

    ```py
    from django.db import models
    class Publisher(models.Model):
        """A company that publishes books."""
        name = models.CharField\
               (max_length=50,\
                help_text="The name of the Publisher.")
        website = models.URLField\
                  (help_text="The Publisher's website.")
        email = models.EmailField\
                (help_text="The Publisher's email address.")
        def __str__(self):
            return self.name
    ```

    在前面的代码片段中，您创建了一个名为 `Publisher` 的新类，该类继承自 Django 的 `models` 模块中的 `Model` 类，将类定义为 Django 模型，该模型将用于存储有关出版商的数据：

    ```py
    class Publisher(models.Model)
    ```

    在这个模型内部，您添加了三个字段，这些字段将作为模型的属性：

    `name`: 出版商的名称

    `website`: 出版商的网站

    `email`: 出版商的电子邮件地址

    完成此操作后，您创建了一个类方法 `__str__()`，它定义了模型字符串表示的形式。

1.  现在，模型已经创建，您首先需要迁移此模型，然后才能在它上面运行测试。为此，运行以下命令：

    ```py
    python manage.py makemigrations
    python manage.py migrate
    ```

1.  现在模型已经设置好了，编写测试用例来测试在 *步骤 3* 中创建的模型。为此，打开 `bookr_test` 目录下的 `tests.py` 文件，并将以下代码添加到其中：

    ```py
    from django.test import TestCase
    from .models import Publisher
    class TestPublisherModel(TestCase):
        """Test the publisher model."""
        def setUp(self):
            self.p = Publisher(name='Packt', \
                               website='www.packt.com', \
                               email='contact@packt.com')
        def test_create_publisher(self):
            self.assertIsInstance(self.p, Publisher)
        def test_str_representation(self):
            self.assertEquals(str(self.p), "Packt")
    ```

    在前面的代码片段中，有几个值得探讨的地方。

    在开始时，在从 Django 的`test`模块导入`TestCase`类之后，你从`bookr_test`目录导入了`Publisher`模型，该模型将被用于测试。

    在导入所需的库之后，你创建了一个名为`TestPublisherModel`的新类，它继承自`TestCase`类，并用于组织与`Publisher`模型相关的单元测试：

    ```py
    class TestPublisherModel(TestCase):
    ```

    在这个类中，你定义了一些方法。首先，你定义了一个名为`setUp()`的新方法，并在其中添加了`Model`对象创建的代码，这样每次在这个测试用例中执行新的`test`方法时，都会创建一个`Model`对象。这个`Model`对象被存储为类成员，这样就可以在其他方法中无问题地访问它：

    ```py
    def setUp(self):
        self.p = Publisher(name='Packt', \
                           website='www.packt.com', \
                           email='contact@packt.com')
    ```

    第一个测试用例验证`Publisher`模型的`Model`对象是否创建成功。为此，你创建了一个名为`test_create_publisher()`的新方法，在其中检查创建的模型对象是否指向`Publisher`类型的对象。如果这个`Model`对象没有成功创建，你的测试将失败：

    ```py
        def test_create_publisher(self):
            self.assertIsInstance(self.p, Publisher)
    ```

    如果你仔细检查，你在这里使用的是`unittest`库的`assertIsInstance()`方法来断言`Model`对象是否属于`Publisher`类型。

    下一个测试验证模型的字符串表示是否与预期相同。从代码定义来看，`Publisher`模型的字符串表示应该输出出版者的名称。为了测试这一点，你创建了一个名为`test_str_representation()`的新方法，并检查生成的模型字符串表示是否与预期匹配：

    ```py
    def test_str_representation(self):
        self.assertEquals(str(self.p), "Packt")
    ```

    为了执行这个验证，你使用`unittest`库的`assertEquals`方法，该方法验证提供的两个值是否相等。

1.  现在测试用例已经就绪，你可以运行它们来检查会发生什么。要运行这些测试用例，请运行以下命令：

    ```py
    python manage.py test
    ```

    一旦命令执行完成，你将看到类似于以下输出的输出（你的输出可能略有不同）：

    ```py
    % python manage.py test
    Creating test database for alias 'default'...
    System check identified no issues (0 silenced).
    ..
    ----------------------------------------------------------------------
    Ran 2 tests in 0.002s
    OK
    Destroying test database for alias 'default'...
    ```

    如前所述的输出所示，测试用例执行成功，从而验证了诸如创建新的`Publisher`对象及其在检索时的字符串表示等操作是否正确执行。

通过这个练习，我们看到了如何轻松编写 Django 模型的测试用例并验证其功能，包括对象的创建、检索和表示。

此外，在这个练习的输出中还有一行重要的内容需要注意：

```py
"Destroying test database for alias 'default'..."
```

这是因为当存在需要将数据持久化存储在数据库中的测试用例时，Django 不会使用生产数据库，而是为测试用例创建一个新的空数据库，它使用这个数据库来持久化测试用例的值。

# 测试 Django 视图

Django 中的视图控制用户基于在 Web 应用程序中访问的 URL 渲染 HTTP 响应。在本节中，我们将了解如何测试 Django 中的视图。想象一下，您正在开发一个需要大量**应用程序编程接口**（**API**）端点的网站。一个有趣的问题可能是，您将如何验证每个新的端点？如果手动完成，每次添加新端点时，您都必须首先部署应用程序，然后在浏览器中手动访问端点以验证其是否正常工作。当端点数量较少时，这种方法可能可行，但如果端点有数百个，这种方法可能会变得极其繁琐。

Django 提供了一种非常全面的测试应用程序视图的方法。这是通过使用 Django 的`test`模块提供的测试客户端类来实现的。这个类可以用来访问映射到视图的 URL，并捕获访问 URL 端点时生成的输出。然后我们可以使用捕获的输出来测试 URL 是否生成了正确的响应。此客户端可以通过从 Django 的`test`模块导入`Client`类，然后按照以下代码片段初始化它来使用：

```py
from django.test import Client
c = Client()
```

客户端对象支持多种方法，这些方法可以用来模拟用户可以发起的不同 HTTP 调用，例如，`GET`、`POST`、`PUT`、`DELETE`等。发起此类请求的示例将如下所示：

```py
response = c.get('/welcome')
```

视图生成的响应随后被客户端捕获，并作为`response`对象暴露出来，然后可以查询以验证视图的输出。

带着这些知识，现在让我们看看我们如何为我们的 Django 视图编写测试用例。

## 练习 14.03：为 Django 视图编写单元测试

在这个练习中，您将使用 Django 测试客户端编写针对您的 Django 视图的测试用例，该视图将被映射到特定的 URL。这些测试用例将帮助您验证当使用其映射的 URL 访问视图函数时，是否生成了正确的响应：

1.  对于这个练习，您将使用在*练习 14.02*的*步骤 1*中创建的`bookr_test`应用程序，即*测试 Django 模型*。要开始，打开 bookr_test 目录下的`views.py`文件，并将以下代码添加到其中：

    ```py
    from django.http import HttpResponse
    def greeting_view(request):
        """Greet the user."""
        return HttpResponse("Hey there, welcome to Bookr!")\
                           ("Your one stop place")\
                           ("to review books.")
    ```

    在这里，您创建了一个简单的 Django 视图，该视图将在用户访问映射到提供的视图的端点时，用欢迎消息问候用户。

1.  一旦创建了此视图，您需要将其映射到 URL 端点，然后可以在浏览器或测试客户端中访问它。为此，打开`bookr_test`目录下的`urls.py`文件，并将高亮代码添加到`urlpatterns`列表中：

    ```py
    from django.urls import path
    from . import views
    urlpatterns = [greeting_view to the 'test/greeting' endpoint for the application by setting the path in the urlpatterns list.
    ```

1.  一旦设置了此路径，你需要确保它也被你的项目识别。为此，你需要将此条目添加到`bookr`项目的 URL 映射中。为了实现这一点，打开`bookr`目录下的`urls.py`文件，并将以下突出显示的行追加到`urlpatterns`列表的末尾，如下所示：

    ```py
    urlpatterns = [….,\
                   ….,\
                   urls.py file should look like this now: http://packt.live/3nF8Sdb.
    ```

1.  一旦设置好视图，请验证其是否正确工作。通过运行以下命令来完成此操作：

    ```py
    python manage.py runserver localhost:8080
    ```

    然后在你的网络浏览器中访问`http://localhost:8080/test/greeting`。一旦页面打开，你应该看到以下文本，这是你在*步骤 1*中添加到问候视图中的，并在浏览器中显示：

    ```py
    Hey there, welcome to Bookr! Your one stop place to review books.
    ```

1.  现在，你准备好为`greeting_view`编写测试用例。在这个练习中，你将编写一个测试用例，检查在访问`/test/greeting`端点时，你是否得到一个成功的结果。为了实现这个测试用例，打开`bookr_test`目录下的`tests.py`文件，并在文件末尾添加以下代码：

    ```py
    from django.test import TestCase, Client
    class TestGreetingView(TestCase):
        """Test the greeting view."""
        def setUp(self):
            self.client = Client()
        def test_greeting_view(self):
            response = self.client.get('/test/greeting')
            self.assertEquals(response.status_code, 200)
    ```

    在前面的代码片段中，你定义了一个测试用例，有助于验证问候视图是否正常工作。

    这是通过首先导入 Django 的测试客户端来完成的，它允许通过对其调用并分析生成的响应来测试映射到 URL 的视图：

    ```py
    from django.test import TestCase, Client
    ```

    完成导入后，你现在创建一个名为`TestGreetingView`的新类，该类将分组与你在*步骤 2*中创建的问候视图相关的测试用例：

    ```py
    class TestGreetingView(TestCase):
    ```

    在此测试用例中，你定义了两个方法，`setUp()`和`test_greeting_view()`。`test_greeting_view()`方法实现了你的测试用例。在这个方法中，你首先对映射到问候视图的 URL 进行 HTTP `GET`调用，然后将视图生成的响应存储在创建的`response`对象中：

    ```py
    response = self.client.get('/test/greeting')
    ```

    一旦这个调用完成，你将在`response`变量内部获得其 HTTP 响应代码、内容和头信息。接下来，使用以下代码，你将进行断言以验证调用生成的状态码是否与成功 HTTP 调用（`HTTP 200`）的状态码匹配：

    ```py
    self.assertEquals(response.status_code, 200)
    ```

    通过这种方式，你现在可以运行测试了。

1.  编写测试用例后，让我们看看运行测试用例时会发生什么：

    ```py
    python manage.py test
    ```

    一旦命令执行，你可以预期看到以下片段所示的输出：

    ```py
    % python manage.py test
    Creating test database for alias 'default'...
    System check identified no issues (0 silenced).
    ...
    ----------------------------------------------------------------------
    Ran 3 tests in 0.006s
    OK
    Destroying test database for alias 'default'...
    ```

    如输出所示，你的测试用例执行成功，从而验证了`greeting_view()`方法生成的响应符合你的预期。

在这个练习中，你学习了如何为 Django 视图函数实现测试用例，并使用 Django 提供的`TestClient`断言视图函数生成的输出与开发者应看到的输出相匹配。

## 使用身份验证测试视图

在上一个示例中，我们探讨了如何在 Django 中测试视图。关于这个视图的一个重要方面是，我们创建的视图可以被任何人访问，并且没有任何身份验证或登录检查来保护。现在想象一个场景，其中视图只有在用户登录时才可访问。例如，想象实现一个视图函数，用于渲染我们网络应用的注册用户的个人资料页面。为了确保只有登录用户可以查看其账户的个人资料页面，您可能希望将视图限制为仅对登录用户开放。

通过这种方式，我们现在有一个重要的问题：*我们如何测试需要认证的视图？*

幸运的是，Django 的测试客户端提供了这项功能，我们可以通过它登录到我们的视图并对其运行测试。这可以通过使用 Django 的测试客户端 `login()` 方法来实现。当此方法被调用时，Django 的测试客户端会对服务执行身份验证操作，如果身份验证成功，它将在内部存储登录 cookie，然后可以使用它进行进一步的测试运行。以下代码片段显示了如何设置 Django 的测试客户端来模拟已登录用户：

```py
login = self.client.login(username='testuser', password='testpassword')
```

`login` 方法需要测试用户的用户名和密码，正如将在下一个练习中展示的那样。因此，让我们看看如何测试需要用户身份验证的流程。

## 练习 14.04：编写测试用例以验证已认证用户

在这个练习中，您将编写测试用例来测试需要用户进行身份验证的视图。作为这部分，您将验证当未登录用户尝试访问页面以及当登录用户尝试访问映射到视图函数的页面时，视图方法生成的输出。

1.  对于这个练习，您将使用在 *练习 14.02* 的 *步骤 1* 中创建的 `bookr_test` 应用程序。要开始，打开 `bookr_test` 应用程序下的 `views.py` 文件，并向其中添加以下代码：

    ```py
    from django.http import HttpResponse
    from django.contrib.auth.decorators import login_required
    ```

    一旦添加了前面的代码片段，请在文件的末尾创建一个新的函数 `greeting_view_user()`，如下面的代码片段所示：

    ```py
    @login_required
    def greeting_view_user(request):
        """Greeting view for the user."""
        user = request.user
        return HttpResponse("Welcome to Bookr! {username}"\
                            .format(username=user))
    ```

    通过这种方式，您已经创建了一个简单的 Django 视图，该视图将在用户访问映射到提供的视图的端点时，用欢迎信息问候登录用户。

1.  一旦创建了此视图，您需要将其映射到可以在浏览器或测试客户端中访问的 URL 端点。为此，打开 `bookr_test` 目录下的 `urls.py` 文件，并向其中添加以下突出显示的代码：

    ```py
    from django.urls import path
    from . import views
    urlpatterns = [greeting_view_user to the 'test/greet_user' endpoint for the application by setting the path in the urlpatterns list. If you have followed the previous exercises, this URL should already be set up for detection in the project and no further steps are required to configure the URL mapping.
    ```

1.  一旦设置了视图，接下来您需要做的就是验证它是否正确工作。为此，运行以下命令：

    ```py
    python manage.py runserver localhost:8080
    ```

    然后在您的网页浏览器中访问 `http://localhost:8080/test/greet_user`。

    如果您尚未登录，通过访问前面的 URL，您将被重定向到项目的登录页面。

1.  现在，为`greeting_view_user`编写测试用例，该测试用例检查在访问`/test/greet_user`端点时是否得到成功的结果。为了实现此测试用例，打开`bookr_test`目录下的`tests.py`文件，并向其中添加以下代码：

    ```py
    from django.contrib.auth.models import User
    class TestLoggedInGreetingView(TestCase):
        """Test the greeting view for the authenticated users."""
        def setUp(self):
            test_user = User.objects.create_user\
                        (username='testuser', \
                         password='test@#628password')
            test_user.save()
            self.client = Client()
        def test_user_greeting_not_authenticated(self):
            response = self.client.get('/test/greet_user')
            self.assertEquals(response.status_code, 302)
        def test_user_authenticated(self):
            login = self.client.login\
                    (username='testuser', \
                     password='test@#628password')
            response = self.client.get('/test/greet_user')
            self.assertEquals(response.status_code, 200)
    ```

    在前面的代码片段中，您实现了一个测试用例，该用例检查在内容可见之前是否启用了视图的认证。

    因此，您首先导入了将用于定义测试用例和初始化测试客户端的必需类和方法：

    ```py
    from django.test import TestCase, Client
    ```

    您接下来需要的是 Django `auth`模块中的`User`模型：

    ```py
    from django.contrib.auth.models import User
    ```

    此模型是必需的，因为对于需要认证的测试用例，您需要初始化一个新的测试用户。接下来，您创建了一个名为`TestLoggedInGreetingView`的新类，该类封装了与`greeting_user`视图（需要认证）相关的测试。在这个类内部，您定义了三个方法，分别是：`setUp()`、`test_user_greeting_not_authenticated()`和`test_user_authenticated()`。`setUp()`方法用于首先初始化一个测试用户，您将使用它进行认证。这是一个必需的步骤，因为 Django 测试环境是一个完全隔离的环境，它不使用生产应用程序中的数据，因此所有必需的模型和对象都必须在测试环境中单独实例化。

    然后，您使用以下代码创建了测试用户并初始化了测试客户端：

    ```py
    test_user = User.objects.create_user\
                (username='testuser', \
                 password='test@#628password')
    test_user.save()
    self.client = Client()
    ```

    接下来，您为用户未认证时的`greet_user`端点编写了测试用例。在这个测试用例中，您应该期望 Django 将用户重定向到登录端点。可以通过检查响应的 HTTP 状态码来检测此重定向，状态码应设置为`HTTP 302`，表示重定向操作：

    ```py
    def test_user_greeting_not_authenticated(self):
        response = self.client.get('/test/greet_user')
        self.assertEquals(response.status_code, 302)
    ```

    接下来，您又编写了另一个测试用例，以检查当用户认证时`greet_user`端点是否成功渲染。为了认证用户，您首先调用测试客户端的`login()`方法，并通过提供在`setUp()`方法中创建的测试用户的用户名和密码进行认证，如下所示：

    ```py
    login = self.client.login\
            (username='testuser', \
             password='test@#628password')
    ```

    登录完成后，您向`greet_user`端点发送一个`HTTP GET`请求，并通过检查返回响应的 HTTP 状态码来验证端点是否生成正确的结果：

    ```py
    response = self.client.get('/test/greet_user')
    self.assertEquals(response.status_code, 200)
    ```

1.  测试用例编写完成后，是时候检查它们如何运行了。为此，运行以下命令：

    ```py
    python manage.py test
    ```

    执行完成后，您应该会看到以下类似响应：

    ```py
    % python manage.py test
    Creating test database for alias 'default'...
    System check identified no issues (0 silenced).
    .....
    ----------------------------------------------------------------------
    Ran 5 tests in 0.366s
    OK
    Destroying test database for alias 'default'...
    ```

    如前所述的输出所示，我们的测试用例已成功通过，验证了我们创建的视图在用户未认证时重定向用户到网站，并在用户认证时允许用户查看页面。

在这个练习中，我们只是实现了一个测试用例，我们可以测试视图函数生成的关于用户认证状态的输出。

# Django 请求工厂

到目前为止，我们一直在使用 Django 的测试客户端来测试我们为应用程序创建的视图。测试客户端类模拟了一个浏览器，并使用这种模拟来调用所需的 API。但如果我们不想使用测试客户端及其作为浏览器的相关模拟，而是想直接通过传递请求参数来测试视图函数，我们应该怎么做？

为了帮助我们处理这种情况，我们可以利用 Django 提供的`RequestFactory`类。`RequestFactory`类帮助我们提供`request`对象，我们可以将其传递给我们的视图函数以评估其工作。可以通过以下方式创建`RequestFactory`的实例：

```py
factory = RequestFactory()
```

因此创建的`factory`对象仅支持 HTTP 方法，如`get()`、`post()`、`put()`等，以模拟对任何 URL 端点的调用。让我们看看我们如何修改我们在*练习 14.04*，*编写测试用例以验证已认证用户*中编写的测试用例，以使用`RequestFactory`。

## 练习 14.05：使用请求工厂测试视图

在这个练习中，你将使用请求工厂来测试 Django 中的视图函数：

1.  对于这个练习，你将使用在*练习 14.04*，*编写测试用例以验证已认证用户*的第 1 步中创建的现有的`greeting_view_user`视图函数，如下所示：

    ```py
    @login_required
    def greeting_view_user(request):
        """Greeting view for the user."""
        user = request.user
        return HttpResponse("Welcome to Bookr! {username}"\
                            .format(username=user))
    ```

1.  接下来，修改在`bookr_test`目录下的`tests.py`文件中定义的现有测试用例`TestLoggedInGreetingView`。打开`tests.py`文件并做出以下更改。

    首先，你需要添加以下导入以在测试用例中使用`RequestFactory`：

    ```py
    from django.test import RequestFactory
    ```

    下一步你需要的是从 Django 的`auth`模块导入`AnonymousUser`类和从`views`模块导入的`greeting_view_user`视图方法。这是为了测试使用模拟未登录用户的视图函数所必需的。这可以通过添加以下代码来完成：

    ```py
    from django.contrib.auth.models import AnonymousUser
    from .views import greeting_view_user
    ```

1.  一旦添加了`import`语句，修改`TestLoggedInGreetingView`类的`setUp()`方法，并更改其内容以类似于以下所示：

    ```py
    def setUp(self):
        self.test_user = User.objects.create_user\
                         (username='testuser', \
                          password='test@#628password')
        self.test_user.save()
        self.factory = RequestFactory()
    ```

    在这个方法中，你首先创建了一个`user`对象，并将其存储为类成员，以便你可以在测试中稍后使用它。一旦创建了`user`对象，然后实例化一个新的`RequestFactory`类的新实例，以便用于测试我们的视图函数。

1.  现在定义了`setUp()`方法后，修改现有的测试以使用`RequestFactory`实例。对于对视图函数的非认证调用测试，将`test_user_greeting_not_authenticated`方法修改为以下内容：

    ```py
    def test_user_greeting_not_authenticated(self):
        request = self.factory.get('/test/greet_user')
        request.user = AnonymousUser()
        response = greeting_view_user(request)
        self.assertEquals(response.status_code, 302)
    ```

    在这个方法中，你首先使用在`setUp()`方法中定义的`RequestFactory`实例创建了一个`request`对象。一旦完成，你将一个`AnonymousUser()`实例分配给`request.user`属性。将`AnonymousUser()`实例分配给该属性使得视图函数认为发起请求的用户未登录：

    ```py
    request.user = AnonymousUser()
    ```

    一旦完成，你调用了`greeting_view_user()`视图方法，并将你创建的`request`对象传递给它。一旦调用成功，你使用以下代码在`response`变量中捕获方法的输出：

    ```py
    response = greeting_view_user(request)
    ```

    对于未认证的用户，你期望得到一个重定向响应，可以通过检查响应的 HTTP 状态码来测试，如下所示：

    ```py
    self.assertEquals(response.status_code, 302)
    ```

1.  一旦完成这个步骤，继续修改其他方法，例如`test_user_authenticated()`，同样使用`RequestFactory`实例，如下所示：

    ```py
    def test_user_authenticated(self):
        request = self.factory.get('/test/greet_user')
        request.user = self.test_user
        response = greeting_view_user(request)
        self.assertEquals(response.status_code, 200)
    ```

    如你所见，大部分代码与你在`test_user_greeting_not_authenticated`方法中编写的代码相似，只是有一点小的变化：在这个方法中，我们不是使用`AnonymousUser`来设置`request.user`属性，而是使用你在`setUp()`方法中创建的`test_user`。

    ```py
    request.user = self.test_user
    ```

    完成这些更改后，是时候运行测试了。

1.  要运行测试并验证请求工厂是否按预期工作，请运行以下命令：

    ```py
    python manage.py test
    ```

    命令执行后，你可以期待看到类似于以下输出的结果：

    ```py
    % python manage.py test   
    Creating test database for alias 'default'...
    System check identified no issues (0 silenced).
    ......
    ----------------------------------------------------------------------
    Ran 6 tests in 0.248s
    OK
    Destroying test database for alias 'default'...
    ```

    从输出中我们可以看到，我们编写的测试用例已经成功通过，从而验证了`RequestFactory`类的行为。

通过这个练习，我们学习了如何利用`RequestFactory`编写测试用例，并将`request`对象直接传递给视图函数，而不是使用测试客户端方法模拟 URL 访问，从而允许更直接的测试。

## 测试基于类的视图

在上一个练习中，我们看到了如何测试定义为方法的视图。但对于基于类的视图呢？我们该如何测试它们？

结果表明，测试基于类的视图相当简单。例如，如果我们有一个名为`ExampleClassView(View)`的基于类的视图，要测试这个视图，我们只需要使用以下语法：

```py
response = ExampleClassView.as_view()(request)
```

就这么简单。

Django 应用程序通常由几个不同的组件组成，这些组件可以独立工作，例如模型，以及一些需要与 URL 映射和其他框架部分交互才能工作的其他组件。测试这些不同的组件可能需要一些只有这些组件共有的步骤。例如，在测试模型时，我们可能首先想要在开始测试之前创建某些`Model`类的对象，或者对于视图，我们可能首先想要使用用户凭据初始化测试客户端。

事实上，Django 还提供了一些基于`TestCase`类的其他类，可以用来编写关于所使用组件类型的特定类型的测试用例。让我们看看 Django 提供这些不同的类。

# Django 中的测试用例类

除了 Django 提供的基类`TestCase`，它可以用来为不同的组件定义多种测试用例之外，Django 还提供了一些从`TestCase`类派生的专用类。这些类根据它们提供给开发者的功能，用于特定类型的测试用例。

让我们快速看一下它们。

## SimpleTestCase

这个类是从 Django 的`test`模块提供的`TestCase`类派生出来的，应该用于编写测试视图函数的简单测试用例。通常，当你的测试用例涉及进行数据库查询时，不推荐使用这个类。该类还提供了许多有用的功能，例如以下功能：

+   检查由视图函数引发的异常的能力

+   测试表单字段的能力

+   内置的测试客户端

+   验证视图函数引起的重定向的能力

+   匹配由视图函数生成的两个 HTML、JSON 或 XML 输出的相等性

现在，让我们先了解一下`SimpleTestCase`是什么，然后尝试理解另一种类型的测试用例类，它有助于编写涉及与数据库交互的测试用例。

## TransactionTestCase

这个类是从`SimpleTestCase`类派生出来的，应该在编写涉及数据库交互的测试用例时使用，例如数据库查询、模型对象创建等。

该类提供了以下附加功能：

+   在测试用例运行之前将数据库重置到默认状态的能力

+   根据数据库功能跳过测试 - 如果用于测试的数据库不支持生产数据库的所有功能，这个功能可能会很有用

## LiveServerTestCase

这个类类似于`TransactionTestCase`类，但有一个小的区别，即该类中编写的测试用例使用 Django 创建的实时服务器（而不是使用默认的测试客户端）。

当编写测试用例以测试渲染的网页及其任何交互时，这种运行实时服务器进行测试的能力会很有用，这是在使用默认测试客户端时无法实现的。

这样的测试用例可以利用像**Selenium**这样的工具，它可以用来构建通过与之交互来修改渲染页面状态的交互式测试用例。

## 测试代码模块化

在前面的练习中，我们已经看到了如何为项目中的不同组件编写测试用例。但需要注意的一个重要方面是，到目前为止，我们一直在单个文件中为所有组件编写测试用例。当应用程序没有很多视图和模型时，这种方法是可以的。但随着应用程序的增长，这可能会变得有问题，因为现在我们的单个 `tests.py` 文件将难以维护。

为了避免遇到此类场景，我们应该尝试模块化我们的测试用例，使得模型测试用例与视图相关的测试用例等保持分离。为了实现这种模块化，我们只需要执行两个简单的步骤：

1.  通过运行以下命令，在您的应用程序目录内创建一个名为 `tests` 的新目录：

    ```py
    mkdir tests
    ```

1.  通过运行以下命令，在您的测试目录中创建一个名为 `__init__.py` 的新空文件：

    ```py
    touch __init__.py
    ```

    这个 `__init__.py` 文件是 Django 所需的，以便正确检测我们创建的 `tests` 目录作为一个模块而不是普通目录。

完成前面的步骤后，您可以继续为应用程序中的不同组件创建新的测试文件。例如，要为您的模型编写测试用例，您可以在测试目录中创建一个名为 `test_models.py` 的新文件，并将与模型测试相关的任何代码添加到此文件中。

此外，您不需要采取任何其他额外步骤来运行您的测试。相同的命令将完美适用于您的模块化测试代码库：

```py
python manage.py test
```

通过这种方式，我们现在已经了解了如何为我们的项目编写测试用例。那么，我们何不通过为正在进行的 Bookr 项目编写测试用例来评估我们的知识呢？

## 活动 14.01：在 Bookr 中测试模型和视图

在这个活动中，您将为 Bookr 项目实现测试用例。您将实现验证 `reviews` 应用程序中创建的模型的测试用例，然后您将实现一个简单的测试用例来验证 `reviews` 应用程序。

以下步骤将帮助您完成这个活动：

1.  在 `reviews` 应用程序目录中创建一个名为 `tests` 的目录，以便我们可以将所有针对 `reviews` 应用程序的测试用例进行模块化。

1.  创建一个空的 `__init__.py` 文件，这样目录就被视为不是普通目录，而是一个 Python 模块目录。

1.  创建一个新文件，名为 `test_models.py`，用于实现测试模型的代码。在此文件中，导入您想要测试的模型。

1.  在 `test_models.py` 中，创建一个新的类，该类继承自 `django.tests` 模块的 `TestCase` 类，并实现验证 `Model` 对象创建和读取的方法。

1.  要测试视图函数，在 `tests` 目录中（在步骤 1 中创建的）创建一个名为 `test_views.py` 的新文件。

1.  在 `test_views.py` 文件中，从 `django.tests` 模块导入测试 `Client` 类，以及从 `reviews` 应用程序的 `views.py` 文件中导入 `index` 视图函数。

1.  在 *步骤 5* 中创建的 `test_views.py` 文件中，创建一个新的 `TestCase` 类，并实现方法来验证索引视图。

1.  在 *步骤 7* 中创建的 `TestCase` 类中，创建一个新的函数 `setUp()`，在其中你应该初始化一个 `RequestFactory` 实例，该实例将用于创建一个可以直接传递给视图函数进行测试的 `request` 对象。

1.  完成前面的步骤并编写了测试用例后，通过执行 `python manage.py test` 来运行测试用例，以验证测试用例是否通过。

完成此活动后，所有测试用例都应成功通过。

注意

此活动的解决方案可以在 [`packt.live/2Nh1NTJ`](http://packt.live/2Nh1NTJ) 找到。

# 摘要

在本章中，我们探讨了如何使用 Django 为我们的 Web 应用程序项目编写不同组件的测试用例。我们了解了为什么测试在任何 Web 应用程序的开发中都起着至关重要的作用，以及行业中采用的不同测试技术，以确保他们发布的应用程序代码是稳定且无错误的。

我们接着探讨了如何使用 Django 的 `test` 模块提供的 `TestCase` 类来实现我们的单元测试，这些测试可以用来测试模型以及视图。我们还探讨了如何使用 Django 的 `test` 客户端来测试需要或不需要用户认证的视图函数。我们还简要介绍了使用 `RequestFactory` 来测试方法视图和基于类的视图的另一种方法。

我们通过了解 Django 提供的预定义类以及它们应该在哪里使用，并探讨了如何模块化我们的测试代码库，使其看起来更整洁，来结束本章。

随着我们进入下一章，我们将尝试理解如何通过将第三方库集成到我们的项目中来使我们的 Django 应用程序更加强大。然后，我们将使用此功能将第三方身份验证集成到我们的 Django 应用程序中，从而允许用户使用 Google Sign-In、Facebook 登录等流行服务登录应用程序。
