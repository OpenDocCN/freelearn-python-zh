# 总结

在上一章中，我们学习了一些技术，可以用来将基于`unittest`的测试套件转换为 pytest，从简单地将其用作运行器，一直到将复杂的现有功能转换为更符合 pytest 风格的方式。

这是本快速入门指南的最后一章，我们将讨论以下主题：

+   我们学到了什么

+   pytest 社区

+   下一步

+   最终总结

# 我们学到了什么

接下来的章节将总结我们在本书中学到的内容。

# 介绍

+   您应该考虑编写测试作为您的安全网。这将使您对自己的工作更有信心，允许您放心地进行重构，并确保您没有破坏系统的其他部分。

+   如果您正在将 Python 2 代码库转换为 Python 3，测试套件是必不可少的，因为任何指南都会告诉您，([`docs.python.org/3/howto/pyporting.html#have-good-test-coverage`](https://docs.python.org/3/howto/pyporting.html#have-good-test-coverage))。

+   如果您依赖的**外部 API**没有自动化测试，为其编写测试是一个好主意。

+   pytest 之所以是初学者的绝佳选择之一，是因为它很容易上手；使用简单的函数和`assert`语句编写您的测试。

# 编写和运行测试

+   始终使用**虚拟环境**来管理您的软件包和依赖关系。这个建议适用于任何 Python 项目。

+   pytest 的**内省功能**使得表达您的检查变得简洁；可以直接比较字典、文本和列表。

+   使用`pytest.raises`检查异常和`pytest.warns`检查警告。

+   使用`pytest.approx`比较浮点数和数组。

+   测试组织；您可以将您的测试**内联**到应用程序代码中，也可以将它们保存在一个单独的目录中。

+   使用`-k`标志选择测试：`-k test_something`。

+   使用`-x`在**第一个失败**时停止。

+   记住了**重构二人组**：`--lf -x`。

+   使用`-s`禁用**输出捕获**。

+   使用`-ra`显示测试失败、xfails 和跳过的**完整摘要**。

+   使用`pytest.ini`进行**每个存储库的配置**。

# 标记和参数化

+   在测试函数和类中使用`@pytest.mark`装饰器创建**标记**。要应用到**模块**，请使用`pytestmark`特殊变量。

+   使用`@pytest.mark.skipif`、`@pytest.mark.skip`和`pytest.importorskip("module")`来跳过**当前环境**不适用的测试。

+   使用`@pytest.mark.xfail(strict=True)`或`pytest.xfail("reason")`来标记**预期失败**的测试。

+   使用`@pytest.mark.xfail(strict=False)`来标记**不稳定的测试**。

+   使用`@pytest.mark.parametrize`快速测试**多个输入**的代码和测试**相同接口的不同实现**。

# Fixture

+   **Fixture**是 pytest 的主要特性之一，用于**共享资源**并提供易于使用的**测试辅助工具**。

+   使用`conftest.py`文件在测试模块之间**共享 fixtures**。记得优先使用本地导入以加快测试收集速度。

+   使用**autouse** fixture 确保层次结构中的每个测试都使用某个 fixture 来执行所需的设置或拆卸操作。

+   Fixture 可以假定**多个范围**：`function`、`class`、`module`和`session`。明智地使用它们来减少测试套件的总时间，记住高级 fixture 实例在测试之间是共享的。

+   可以使用`@pytest.fixture`装饰器的`params`参数对**fixture 进行参数化**。使用参数化 fixture 的所有测试将自动进行参数化，使其成为一个非常强大的工具。

+   使用`tmpdir`和`tmpdir_factory`创建空目录。

+   使用`monkeypatch`临时更改对象、字典和环境变量的属性。

+   使用`capsys`和`capfd`来捕获和验证发送到标准输出和标准错误的输出。

+   fixture 的一个重要特性是它们**抽象了依赖关系**，在使用**简单函数与 fixture**之间存在平衡。

# 插件

+   使用`plugincompat` ([`plugincompat.herokuapp.com/`](http://plugincompat.herokuapp.com/)) 和 PyPI ([`pypi.org/`](https://pypi.org/)) 搜索新插件。

+   插件**安装简单**：使用`pip`安装，它们会自动激活。

+   有大量的插件可供使用，满足各种需求。

# 将 unittest 套件转换为 pytest

+   你可以从切换到**pytest 作为运行器**开始。通常情况下，这可以在现有代码中**不做任何更改**的情况下完成。

+   使用`unittest2pytest`将`self.assert*`方法转换为普通的`assert`。

+   现有的**设置**和**拆卸**代码可以通过**autouse** fixtures 进行小的重构后重复使用。

+   可以将复杂的测试工具**层次结构**重构为更**模块化的 fixture**，同时保持现有的测试工作。

+   有许多方法可以进行迁移：一次性转换**所有**内容，转换现有测试时逐步转换测试，或者仅在**新**测试中使用 pytest。这取决于你的测试套件大小和时间预算。

# pytest 社区

我们的社区位于 GitHub 的`pytest-dev`组织（[`github.com/pytest-dev`](https://github.com/pytest-dev)）和 BitBucket（[`bitbucket.org/pytest-dev`](https://bitbucket.org/pytest-dev)）。pytest 仓库（[`github.com/pytest-dev/pytest`](https://github.com/pytest-dev/pytest)）本身托管在 GitHub 上，而 GitHub 和 Bitbucket 都托管了许多插件。成员们努力使社区对来自各个背景的新贡献者尽可能友好和欢迎。我们还在`pytest-dev@python.org`上有一个邮件列表，欢迎所有人加入（[`mail.python.org/mailman/listinfo/pytest-dev`](https://mail.python.org/mailman/listinfo/pytest-dev)）。

大多数 pytest-dev 成员居住在西欧，但我们有来自全球各地的成员，包括阿联酋、俄罗斯、印度和巴西（我就住在那里）。

# 参与其中

因为所有 pytest 的维护完全是自愿的，我们一直在寻找愿意加入社区并帮助改进 pytest 及其插件的人，与他人诚信合作。有许多参与的方式：

+   提交功能请求；我们很乐意听取用户对于他们希望在 pytest 或插件中看到的新功能的意见。确保将它们报告为问题以开始讨论（[`github.com/pytest-dev/pytest/issues`](https://github.com/pytest-dev/pytest/issues)）。

+   报告错误：如果你遇到错误，请报告。我们会尽力及时修复错误。

+   更新文档；我们有许多与文档相关的未解决问题（[`github.com/pytest-dev/pytest/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22status%3A+easy%22+label%3A%22type%3A+docs%22+`](https://github.com/pytest-dev/pytest/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22status%3A+easy%22+label%3A%22type%3A+docs%22+))。如果你喜欢帮助他人并撰写良好的文档，这是一个帮助他人的绝佳机会。

+   实现新功能；尽管代码库对新手来说可能看起来令人生畏，但有许多标有易标签的功能或改进（[`github.com/pytest-dev/pytest/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22status%3A+easy%22`](https://github.com/pytest-dev/pytest/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22status%3A+easy%22)），这对新贡献者很友好。此外，如果你不确定，可以随时询问！

+   修复错误；尽管 pytest 对自身进行了 2000 多次测试，但像任何软件一样，它也存在已知的错误。我们非常乐意审查已知错误的拉取请求（[`github.com/pytest-dev/pytest/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22type%3A+bug%22`](https://github.com/pytest-dev/pytest/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22type%3A+bug%22)）。

+   在推特上使用`#pytest`标签或提及`@pytestdotorg`来传播你的爱。我们也喜欢阅读关于你使用 pytest 的经验的博客文章。

+   在许多会议上，社区的成员组织研讨会、冲刺活动或发表演讲。一定要打个招呼！

成为贡献者很容易；你只需要贡献一个关于相关代码更改、文档或错误修复的拉取请求，如果愿意，你就可以成为`pytest-dev`组织的成员。作为成员，你可以帮助回答、标记和关闭问题，并审查和合并拉取请求。

另一种贡献方式是向`pytest-dev`提交新的插件，可以在 GitHub 或 BitBucket 上进行。我们喜欢当新的插件被添加到组织中，因为这会提供更多的可见性，并帮助与其他成员分享维护工作。

你可以在 pytest 网站上阅读我们的完整贡献指南（[`docs.pytest.org/en/latest/contributing.html`](https://docs.pytest.org/en/latest/contributing.html)）。

# 2016 年冲刺活动

2016 年 6 月，核心团队在德国弗莱堡举办了一次大规模的冲刺活动。超过 20 名参与者参加了为期六天的活动；活动主题围绕着实施新功能和解决问题。我们进行了大量的小组讨论和闪电演讲，并休息一天去美丽的黑森林徒步旅行。

团队成功发起了一次成功的 Indiegogo 活动（[`www.indiegogo.com/projects/python-testing-sprint-mid-2016#/`](https://www.indiegogo.com/projects/python-testing-sprint-mid-2016#/)），旨在筹集 11000 美元以偿还参与者的旅行费用、冲刺场地和餐饮费用。最终，我们筹集了超过 12000 美元，这显示了使用 pytest 的用户和公司的赞赏。

这真是太有趣了！我们一定会在未来重复这样的活动，希望能有更多的参与者。

# 下一步

在学到所有这些知识之后，你可能迫不及待地想要开始使用 pytest，或者更频繁地使用它。

以下是你可以采取的一些下一步的想法：

+   在工作中使用它；如果你已经在日常工作中使用 Python 并有大量的测试，那是开始的最佳方式。你可以慢慢地使用 pytest 作为测试运行器，并以你感到舒适的速度使用更多的 pytest 功能。

+   在你自己的开源项目中使用它：如果你是一个开源项目的成员或所有者，这是获得一些 pytest 经验的好方法。如果你已经有了一个测试套件，那就更好了，但如果没有，当然从 pytest 开始将是一个很好的选择。

+   为开源项目做贡献；你可以选择一个具有`unittest`风格测试的开源项目，并决定提供更改以使用 pytest。2015 年 4 月，pytest 社区组织了所谓的 Adopt pytest 月活动（[`docs.pytest.org/en/latest/adopt.html`](https://docs.pytest.org/en/latest/adopt.html)），开源项目与社区成员配对，将他们的测试套件转换为 pytest。这个活动取得了成功，大多数参与者都玩得很开心。这是参与另一个开源项目并同时学习 pytest 的好方法。

+   为 pytest 本身做出贡献；如前所述，pytest 社区对新贡献者非常欢迎。我们很乐意欢迎你！

本书故意省略了一些主题，因为它们被认为对于快速入门来说有点高级，或者因为由于空间限制，我们无法将它们纳入书中。

+   tox（https://tox.readthedocs.io/en/latest/）是一个通用的虚拟环境管理器和命令行工具，可用于测试具有多个 Python 版本和依赖项的项目。如果您维护支持多个 Python 版本和环境的项目，它就是一个救星。pytest 和`tox`是兄弟项目，它们在一起工作得非常好，尽管它们是独立的，并且对于它们自己的目的非常有用。

+   插件：本书不涵盖如何使用插件扩展 pytest，所以如果您感兴趣，请务必查看 pytest 文档的插件部分（https://docs.pytest.org/en/latest/fixture.html），并寻找其他可以作为示例的插件。此外，请务必查看示例部分（https://docs.pytest.org/en/latest/example/simple.html）以获取高级 pytest 自定义的片段。

+   日志记录和警告是两个 Python 功能，pytest 内置支持，本书没有详细介绍，但如果您经常使用这些功能，它们确实值得一看。

# 最终总结

所以，我们已经完成了快速入门指南。在本书中，我们从在命令行上使用 pytest 到将现有测试套件转换为利用强大的 pytest 功能的技巧和窍门，进行了全面的概述。您现在应该能够每天轻松使用 pytest，并在需要时帮助他人。

您已经走到了这一步，所以祝贺您！希望您在学习的过程中学到了一些东西，并且玩得开心！
