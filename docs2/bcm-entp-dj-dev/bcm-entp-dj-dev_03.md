# *第二章*: 项目配置

源代码在任何软件中都被认为是肉和骨头，或者说是房屋的框架。在本章中，我们将构建一个包含源代码所在文件的工程。我们将讨论一些在开发者直接与源代码工作时会派上用场的工具。当使用 Django 时，虽然任何工具都可以用来编辑源代码，但有些工具比其他工具更高效。在本章中，我们将探索一些无数的现有工具，并讨论为什么**集成开发环境**（**IDE**）也可能被使用。

我们还将了解与项目中的 Django `settings.py` 文件（夹）一起工作的意义。当然，软件还需要一个数据库来存储和检索用户输入和创建的数据，我们将为项目的每个环境安装本地和远程数据库。我们将介绍可用的各种数据库类型，然后专注于本书中示例所使用的最流行的类型。不需要使用我们将要使用的相同数据库类型，但如果您使用不同的类型，可能会遇到一些细微的差异；请谨慎行事。在阅读本书的其余部分之前，阅读本章至关重要。

在本章中，我们将介绍以下内容：

+   选择开发工具

+   启动项目

+   创建虚拟环境

+   项目配置

+   使用基本数据库设置

+   为 Heroku 准备 PostgreSQL

# 技术要求

要在本章中处理代码，以下工具需要在您的本地机器上安装：

+   Python 版本 3.9 – 作为项目的底层编程语言

+   Django 版本 4.0 – 作为项目的后端框架

+   pip 软件包管理器 – 用于管理第三方 Python/Django 软件包

接下来，您需要一种方法来编辑本章以及本书其余部分中将要编写的代码。本章的第一部分将向您提供从文本编辑器到 IDE 的几个开发工具选择。相同的 IDE 将用于演示一些操作，但并不需要使用相同的工具。您可以使用您喜欢的任何 IDE，或者根本不使用 IDE，而是使用终端或命令行窗口。还需要一个数据库，本章的第三部分将提供几个选项供您选择。任何数据库类型都可以与您的项目一起工作，并将提供配置示例，但我们将仅使用 PostgreSQL 来与我们的项目和 Heroku 一起工作。

本书将专注于 Django 和企业开发的概念，而不是指导您如何使用 Git 操作。如果您想快速学习如何使用 Git 操作，可以观看以下视频：[`www.packtpub.com/product/git-and-github-the-complete-git-and-github-course-video/9781800204003`](https://www.packtpub.com/product/git-and-github-the-complete-git-and-github-course-video/9781800204003)。

本章中创建的所有代码都可以在本书的 GitHub 仓库中找到：[`github.com/PacktPublishing/Becoming-an-Enterprise-Django-Developer`](https://github.com/PacktPublishing/Becoming-an-Enterprise-Django-Developer)。本章中使用的代码将与项目的核心中的每个文件相关。从*第三章*“模型、关系和继承”开始，每一章的代码都将位于其自己的 Django 应用文件夹中。请参阅本章标题为“创建 Django 应用”的小节，以了解更多相关信息。

查看以下视频以查看*代码的实际应用*：[`bit.ly/3NqNuFG`](https://bit.ly/3NqNuFG)

# 选择开发工具

配置我们的项目指的是我们将如何构建和安排构成应用程序的文件。这也指的是我们如何在协作团队中共享这些文件。一些工具可以创建可以在团队成员之间共享的文件，例如与开发和调试功能相关的预配置设置。这些文件有时被称为配置或解决方案文件。这意味着您可以预先配置一组开发工具，以帮助团队快速上手。确保所有成员使用类似工具可以使调试和查看非自己编写的代码变得容易得多。这种一致性也使得团队成员在同步工作流程时的口头和书面沟通更加高效。

虽然共享项目配置文件有其好处，但并不是团队中的每个人都必须使用相同的工具。实际上，甚至可以在同一个仓库中创建多种不同配置文件，以适应各种开发工具，为开发者提供多种预配置的项目文件供选择。即使我们在仓库中提供了配置文件，开发者仍然可以使用基本的文本编辑器，通过终端或命令行窗口进行编辑并本地运行项目。让我们比较几种这些文本编辑器工具，并讨论它们提供的某些好处。然后，我们将讨论 IDE 是什么以及它们与文本编辑器的区别。

## 文本编辑器

**文本编辑器**非常简单，正如其名称所示，是一种编辑文本的方式，或者在我们的情况下，是源代码。由于一些流行编辑器包含了许多功能，因此有时将这些编辑器称为轻量级 IDE。

很可能，今天被认为最流行的三个文本编辑器如下：

+   **Atom**

+   **Notepad++**

+   **Sublime Text**

### Atom

Atom 旨在成为一个完全可编辑和可定制的文本编辑器。该工具的官方网站在描述他们的软件时使用了“可黑客化”这个术语。此编辑器允许您安装帮助增强您代码编写能力的包。它具有智能代码自动完成和代码样式，让您输入更少，看到更多。此工具具有内置的文件浏览器，允许您在您选择的文件夹和文件中查找和替换代码。Atom 也适用于所有三个主要操作系统，Windows、Mac 和 Linux。它是免费使用的，也是开源的，允许您修改编辑器的源代码。要下载 Atom 并开始使用它，请访问[`atom.io/`](https://atom.io/)。

### Notepad++

Notepad++是另一个免费的开源编辑器，它配备了时尚的代码语法高亮显示器，并提供代码自动完成建议。它是为了与运行大量不同的编程语言而构建的。它甚至允许您直接从编辑器编写宏和运行自定义脚本。像 Atom 一样，它也有安装插件的能力。它非常轻量级，可以设置为操作系统的默认编辑器。要下载 Notepad++，请访问[`notepad-plus-plus.org/downloads/`](https://notepad-plus-plus.org/downloads/)。

### Sublime Text

Sublime Text 是另一个受欢迎的选择。此编辑器允许您创建项目文件，并包括构建各种编程语言的功能。一个酷炫的功能是它将使用您在`.html`文档内部存在的`<script type="text/javascript"></script>`来以不同的方式样式化和显示，与该文件中找到的 HTML 不同。

Sublime 家族中的另一个应用称为 Sublime Merge，它是一种简单的方式来合并代码并执行 Git 操作。我最喜欢的功能是使用它来查看历史状态和提交日志。Sublime Text 可以免费评估一段时间，但最终，在试用期满后，它会提示您购买许可证。要开始使用 Sublime Text，请访问[`www.sublimetext.com/`](https://www.sublimetext.com/)。

如果不是几十个，也有可能是数百个其他文本编辑器可以选择，但本章没有提及。不要将自己限制在提供的流行选择中，并自由探索许多其他工具。一些文本编辑器提供的功能和能力有时可以被称为轻量级 IDE；接下来让我们讨论一下什么是 IDE。

## 集成开发环境

IDE 是我们所说的将许多不同的编程工具组合成一个单一的桌面应用程序的软件，开发者可以使用它来构建其他应用程序。它是你的团队可以使用并从中受益以保持生产力的开发工具。IDE 主要包含一种查看和编辑源代码的方式，自动化本地构建过程，并提供帮助你调试和分析代码的方式。该工具包含样式化和格式化代码的方式，在你键入时显示错误，并提供代码完成建议以减少你工作时的按键次数。这些环境还提供了搜索项目中包含的其他代码文件的方式，以及将你的代码推送到外部仓库的方式。例如，如果使用具有内置 Git 功能的 IDE，那么在*第一章*中提到的 Sourcetree 应用程序将不再需要。

同样适用于具有 Git 功能的文本编辑器选择。类似于一些具有花哨功能的文本编辑器，IDE 将创建可以与你的团队共享的常见配置文件。其中一些文件我们不想共享，例如存储断点和其他仅针对该开发者和本地实例的特定本地调试设置的文件。然而，与你的团队共享配置设置可以让成员们更容易、更快地将他们的 IDE 设置在他们的机器上，所有工具都准备好了或几乎准备好了可以使用。

一个简单的文本编辑器就足够了。一个项目可以从这种方式开始，然后你的团队可以在之后引入 IDE 的使用。有时，在使用 IDE 时，你可以在子目录中组织你的项目文件，这些目录在其他情况下可能不存在。如果你已经知道你的团队从一开始就需要这些生产力工具，你可以从 IDE 开始，并通过 IDE 本身创建项目，让 IDE 以对 IDE 自然的方式组织你的文件。这种方式有时更容易，因为一些 IDE 可能配置的文件结构可能与你仅使用终端或命令行窗口中的 Django `startproject`命令创建的结构略有不同。

让我们讨论一下开发者今天与 Django 一起使用的流行 IDE 选择，然后从中选择一个来展示使用 IDE 的概念。本章将展示在 IDE 和终端中与 Django 管理命令一起工作的方法，以展示使用一个与另一个相比的优势。在本章中，使用 IDE 演示的每个操作，都将提供该操作的命令行驱动的命令等效。这将还允许你在需要时完全绕过 IDE。从现在开始，所有未来的章节都只提供标准的命令行驱动的 Django 管理命令，继续使用 IDE 将是可选的。

### PyDev with Eclipse

**PyDev** 实际上是一个针对 **Eclipse IDE** 的插件，但它也可以作为一个独立的 IDE 使用。PyDev 可以直接下载并安装，因为它会预装 **LiClipse**，这是 Eclipse IDE 的轻量级版本。Eclipse IDE 提供了一个完全集成的开发体验。它允许以多种方式调试代码。性能分析只是其中之一，这是一种帮助开发者理解事件时间、内存使用、磁盘使用或其他诊断的工具。CPU 性能分析器这个术语常用于讨论可以帮助你找出哪个特定进程拖累了系统的工具。它会告诉你它挂起的时间有多长，并给你一些关于如何修复它的想法。所有之前提到的文本编辑器都附带在 Eclipse 中。PyDev 拥有大量的库，可供选择用于今天存在的许多不同语言。Eclipse 和 PyDev 都支持所有主流操作系统，如 Windows、Mac 和 Linux。要下载 Eclipse，请访问 [`www.eclipseclp.org/download.html`](https://www.eclipseclp.org/download.html)。

Eclipse 和 PyDev 都是免费使用的，并且都是开源许可，允许你修改 IDE 软件。与今天的其他桌面应用程序相比，Eclipse IDE 的安装稍微困难一些。它的安装程序需要你阅读很多内容，才能知道如何开始安装它。Eclipse 还需要你的机器上安装并运行 **Java**。Java 是一种高级编程语言和计算平台，通常不需要与 Django 和 Python 一起工作，除非你正在使用 Eclipse。因此，我们不会在本章中使用这个 IDE。要下载 PyDev，请访问 [`www.pydev.org/download.html`](https://www.pydev.org/download.html)。

### PyCharm

**PyCharm** 可能是当今 Python 和 Django 开发者中最受欢迎的选择。它易于使用，比 PyDev 更容易安装，因为它提供了适用于 Windows、Mac 或 Linux 机器的简单可执行文件。它还提供了免费社区版和付费专业版。付费版提供了更多功能，并提供了更多专业的科学和 Web 开发工具，例如直接在 IDE 中进行高级数据库集成和数据库开发，以及其他调试和性能分析工具。免费社区版对于大多数 Django 开发者来说已经足够了。这个版本允许开发者与 Django 项目集成，并在连接到远程 Git 仓库的同时运行虚拟环境。要下载和安装 PyCharm，请访问 [`www.jetbrains.com/pycharm/download/`](https://www.jetbrains.com/pycharm/download/)。

虽然这可能是 Django 开发者中最受欢迎的 IDE，但 Visual Studio 可能是任何语言或框架开发者中最受欢迎的 IDE。因此，我们将使用 Visual Studio 在本章中演示示例。

### Visual Studio

**Visual Studio** 已经是许多 .NET、C#、C++ 和其他开发者过去 20 多年来的首选工具。它是一个非常强大的 IDE，内置了您能想到的所有工具，甚至更多。它有多种不同的版本和风味，并且多年来它只以付费形式提供。然后，大约在 2013 年，微软开始向公众免费提供 Visual Studio Community Edition。对于一些人来说，缺点是 Visual Studio 直到 2017 年才在 Mac 平台上可用，在此之前它只适用于 Windows 平台。目前，Linux 平台不受支持。然而，轻量级的 IDE Visual Studio Code 可在 Windows、Mac 和 Linux 三种平台上使用。

**Visual Studio Code** 可能已经成为史上最受欢迎的开发者工具。Visual Studio 和 Visual Studio Code 都支持使用 Python 和 Django。在安装这些工具中的任何一个之后，您需要选择要包含在安装中的相关 Python/Django 软件包，或者单独安装它们。对于在 Linux 机器上工作或不想使用 IDE 的您，请参考每个创建项目涉及的动作的 IDE 展示之后的管理命令。如果您在 Windows 或 Mac 系统上，并希望使用 IDE 进行操作，请下载此处提供的 **Visual Studio 2019 – 社区版** 安装程序：[`visualstudio.microsoft.com/downloads/`](https://visualstudio.microsoft.com/downloads/)。在安装过程中，请确保选择它提供的任何 Python 开发扩展，如下面的截图所示：

![图 2.1 – Visual Studio – Python 开发扩展](img/Figure_2.01_B17243.jpg)

图 2.1 – Visual Studio – Python 开发扩展

您可以在该界面的 **单独组件** 选项卡下找到其他有用的工具。在继续安装之前，包括您想要包含的任何其他工具。现在我们已经安装了 **Visual Studio 2019 – 社区版**，让我们为可以在仓库中与其他开发者共享的项目构建一个解决方案文件。

注意

虽然 Visual Studio 2019 是微软提供的最新产品，但 Visual Studio 2022 将在本书出版时大约同期发布。如果您使用的是较新版本的 Visual Studio，您应该能够执行本章中描述的所有相同操作。Visual Studio IDE 的截图可能不会完全相同，并且可能还需要进行一些代码调整。

# 开始一个项目

有两种启动项目的方式，本章将允许你选择想要遵循的方法。我们鼓励你使用 IDE，因为熟练使用这个工具对你的团队来说在长远来看是有益的。然而，如果你的团队使用的是除 Visual Studio 之外的 IDE，或者你只使用文本编辑器与代码一起工作，每个步骤的命令行等效操作也提供给了任何人，以便他们可以完成本章。本书中的其他所有章节都将仅关注代码，这些代码可以与或无需 IDE 一起使用。

## 使用 IDE

打开 Visual Studio IDE，选择`django`关键词，并在结果列表中选择**空白 Django Web 项目**，如图所示：

![Figure 2.2 – Visual Studio – Create a new project

![img/Figure_2.02_B17243.jpg]

Figure 2.2 – Visual Studio – Create a new project

在下一屏中，输入`becoming_a_django_entdev`作为`manage.py`文件，这是项目的根目录。这将使使用 IDE 内提供的终端等工具变得稍微容易一些：

![Figure 2.3 – Visual Studio – Creating a Django project

![img/Figure_2.03_B17243.jpg]

Figure 2.3 – Visual Studio – Creating a Django project

注意

当使用 Visual Studio IDE 创建项目时，位于`/becoming_a_django_entdev/becoming_a_django_entdev/`文件夹中的文件，例如`settings.py`和`urls.py`文件，将自动使用 Django 2.1.2 版本生成。这些文件在使用 Django 的后续版本时仍然有效。此外，当我们进入本章的*创建虚拟环境*部分时，我们将实际安装 Django 4.0 包，这是本书中使用的版本。尽管最初创建一些项目文件时使用了 2.1.2 版本，但项目将始终使用 4.0 版本并成功运行。当使用终端或命令行窗口创建项目时，这种情况不会发生。稍后，在*第九章* *Django 测试*中，你将学习如何验证实际安装和使用的版本。

## 使用命令行

Django 终端或命令行窗口启动 Django 项目的等效命令是`startproject`。使用此命令创建项目有两种方式。第一种方法是使用您机器上全局安装的 Django 版本创建项目，然后接下来构建您的虚拟环境。

另一种方式是首先创建虚拟环境，然后激活你的环境，安装你想要的 Django 版本，然后使用虚拟环境中安装的 Django 版本构建项目。IDE 为我们做的第一件事是使用 IDE 提供的 Django 版本创建项目，然后当我们创建虚拟环境时，Django 的版本更新到了`requirements.txt`文件中指定的版本。

当虚拟环境中某个包更新时，旧版本会被卸载，新版本会被全新安装。为了这个练习，我们将卸载可能存在的任何版本的 Django，然后安装本书编写时最新的版本。然后，我们将在下一节使用命令行创建虚拟环境，尽可能接近使用 IDE 提供的示例。

按照以下步骤创建您的项目文件：

1.  打开您的终端或命令行窗口，导航到在*第一章*“承担一个巨大的项目”中创建的本地仓库文件夹。确保您此时不在虚拟环境中。然后，执行以下命令以卸载任何现有的 Django 版本，然后安装我们将要在机器上全局使用的正确版本的 Django：

    ```py
    PS C:\Projects\Packt\Repo> pip uninstall django
    PS C:\Projects\Packt\Repo> pip install django==4.0
    ```

1.  执行基于 Django 4.0 版本的 Django 命令，该命令将创建一个项目以及所有必要的核心文件，以便与 Django 一起工作。我们将把这个项目命名为与 IDE 示例中相同：`becoming_a_django_entdev`。`startproject`命令创建一个`manage.py`、`wsgi.py`、`asgi.py`文件以及几个其他作为所有 Django 项目基础的样板文件。`becoming_a_django_entdev`选项是项目的名称和项目将被放置的文件夹名称。执行以下命令以创建项目：

    ```py
    PS C:\Projects\Packt\Repo> python -m django startproject becoming_a_django_entdev
    ```

前面的命令是一个经过测试并证明在 Windows 机器上可以正常工作的友好命令。传统上，开发者会使用以下命令来启动项目；然而，这个命令在 Windows 操作系统上无法工作：

```py
PS C:\Projects\Packt\Repo> django-admin startproject becoming_a_django_entdev
```

接下来，让我们创建和配置一个虚拟环境，这对于使用项目中包含的任何第三方包是必需的。

# 创建虚拟环境

目前我们不应该为我们的项目有一个虚拟环境。如果您已经有了，请继续并忽略它，为这个接下来的练习创建一个新的虚拟环境。无论您是使用 Visual Studio IDE 创建项目，还是在前面的练习中使用终端或命令行窗口中的 Django 命令，您的仓库中的文件结构应该看起来像以下树结构：

`├── .git`

`├── readme.md`

`├── requirements.txt`

`├── becoming_a_django_entdev`

`│ ├── .vs`

`│ ├── becoming_a_django_entdev.sln`

`│ ├── db.sqlite3`

`│ ├── manage.py`

`│ ├── obj`

`│ ├── requirements.txt`

`│ ├── staticfiles`

`│ └── becoming_a_django_entdev`

`│ ├── __init__.py`

`│ ├── asgi.py`

`│ ├── settings.py`

`│ ├── urls.py`

`│ └── wsgi.py`

现在我们有两个 `requirements.txt` 文件和两个名为 `/becoming_a_django_entdev/becoming_a_django_entdev/` 的文件夹，其中存放着各种文件。我们将保持文件夹结构不变，稍后配置额外的设置以允许 Heroku 与此文件夹结构协同工作。我们这样做是因为通过 IDE 创建项目只会产生这种结果，请注意，其他 IDE 或甚至轻量级 IDE 文本编辑器可能创建的文件夹结构可能与前面的树形结构不同。如果您决定使用 Django 命令创建新项目，您确实有选项指定一个额外的选项以防止在子目录中创建项目。这将把 `manage.py` 文件放在存储库根目录中，并导致只有一个 `requirements.txt` 文件。这样做将需要使用以下命令示例：

```py
PS C:\Projects\Packt\Repo> python -m django startproject becoming_a_django_entdev ./
```

在这里，我们在前面的命令末尾添加了额外的 `./` 选项，表示将项目放在我们现在所在的文件夹中。如果没有这个选项，默认情况下将创建一个额外的子文件夹，文件夹名称与提供的项目名称相同。然而，这样做将导致没有解决方案文件，现在也没有通过 Visual Studio IDE 运行项目的方法。由于本章的目标是演示 IDE 的使用，这就是我们保持前面树形结构的原因。还有其他选项可以做到这一点，这将引导您在 Visual Studio 中使用存储库根目录中的 `requirements.txt` 文件创建新项目以使用此配置。

我们将在下一节讨论如何做。

## 配置 requirements.txt 文件

在 `manage.py` 文件所在的同一文件夹中，Visual Studio 为我们创建了一个 `requirements.txt` 文件。在 *第一章*，*承担一个庞大的项目*，我们已经创建了一个空的 `requirements.txt` 文件，只是为了满足 Heroku 在部署到我们创建的每个环境时的需求。如果存储库根目录中没有 `requirements.txt` 文件，Heroku 将无法部署。Heroku 需要这个副本来识别一个项目为 Python 项目；这正是他们的测试脚本编写的方式。这就是我们现在有两个 `requirements.txt` 文件可以工作的原因。另一个文件位于 `/becoming_a_django_entdev/` 中的一级文件夹内，这是为了启用 Visual Studio IDE 提供的所有功能和服务。这个 `requirements.txt` 文件是我们将写入所有必需依赖的地方。

在您本地存储库根目录中的 `requirements.txt` 文件中，该文件夹与您的 `.git` 文件夹位于同一位置，添加以下代码：

```py
# ././requirements.txt in Root Directory
```

```py
# Path to Visual Studio requirements.txt below
```

```py
-r becoming_a_django_entdev/requirements.txt
```

前面的代码声明了 `requirements.txt` 文件的位置，该文件嵌套在我们现在将称之为项目根目录的 `/becoming_a_django_entdev/` 中。在你的项目根目录中打开 `requirements.txt` 文件，该目录也是 `manage.py` 文件所在的同一文件夹，然后添加以下项：

```py
# requirements.txt
```

```py
django~=4.0
```

```py
django-extensions
```

```py
django-heroku
```

```py
dj-database-url
```

```py
gunicorn
```

```py
python-dotenv
```

```py
pydotplus
```

```py
psycopg2
```

```py
psycopg2-binary==2.8.6
```

```py
whitenoise
```

```py
secret-key-generator
```

确保每个项目都位于单独的一行上。你项目所需的任何包都将始终放在这个文件中。你可以更加模块化，将这些文件添加到其他导入中，称为 `-r becoming_a_django_entdev/requirements.txt` 以指定文件路径。

在前面的例子中，使用了 `~=` 操作符，表示大于或等于指定的版本号。例如，它将提供 4.0 版本的 Django，这意味着如果存在 4.0.9 版本，它将使用该版本，但不会提供 4.1 或更高版本。它只会提供 4.0.X 范围的最高版本。`==` 操作符表示包含与指定确切版本号匹配的包。完全不指定版本意味着它将抓取安装、重新安装或升级时存在的该包的最新版本。你可以在 PyPI 包库中找到的任何内容都可以放在这个文件中，要求它必须安装在你的虚拟环境中。

`django` 包是最重要的，因为它是我们使用框架的基础。`django-heroku` 包是由 Heroku 创建的包，它包含了一系列 Heroku 本身依赖的其他包。当父包安装时，这些 Heroku 依赖项将自动为你安装。之前列出的其他包将在下一节中使用，以帮助我们正确配置项目，以便我们在本书的其余部分使用 Django。

现在我们已经定义了要安装到虚拟环境中的多个包，让我们使用 Visual Studio IDE 创建我们的虚拟环境。

## 使用 IDE

对于那些希望使用命令行的人来说，请跳转到标题为 *使用命令行* 的子节。

在你的 Visual Studio IDE 中，导航到 IDE 的 **解决方案资源管理器** 部分，右键单击 **Python 环境**，然后选择 **添加环境**。打开的窗口如下所示：

![图 2.4 – Visual Studio – 添加环境![图片 2.04_B17243.jpg](img/Figure_2.04_B17243.jpg)

图 2.4 – Visual Studio – 添加环境

当窗口弹出时，将虚拟环境名称输入为 `virtual_env` 并选择您的基解释器。这个名字可以是您选择的任何名字，但为了与本书中的示例保持一致，请命名为 `virtual_env`。在创建此练习时选择了 Python 版本 3.9 作为基解释器。位置非常重要，因为它会询问您是否想要 `requirements.txt` 文件与您的 `manage.py` 文件位于同一文件夹中。更改此位置可能会破坏使用您的 IDE 的一些功能，例如与您的虚拟环境一起工作的操作，并导致您遇到未解决的导入问题。未解决的导入可能会允许您的项目仍然运行，但代码高亮和格式化通常会中断。现在您应该在您的**解决方案资源管理器**和 IDE 中的**Python 环境**窗口中看到您的虚拟环境，如图所示：

![图 2.5 – Visual Studio – 虚拟环境成功

![图片 2.05_B17243.jpg]

图 2.5 – Visual Studio – 虚拟环境成功

如果您以后想安装包，例如，如果有人在 2 个月后将包添加到 `requirements.txt` 文件中，开发者可以在**解决方案资源管理器**中右键单击虚拟环境，然后选择**从 requirements.txt 安装**。Visual Studio 将更新任何版本并安装任何缺失的包。Visual Studio 和 pip 通常不会删除未使用的包；如果您遇到冲突，可能需要手动卸载包。

### 运行/激活项目

到目前为止，我们所做的一切都应该允许我们在本地运行项目，因为 Visual Studio 为我们做了很多繁重的工作。如果您决定不使用 IDE，您目前无法运行您的项目，并且必须在本章的大部分内容中工作才能达到这个结果。在任何情况下，您目前都无法在 Heroku 上运行您的项目。使用 IDE 顶部的**Play/Run Web Server**按钮，选择您想要的浏览器（如果它提供了选择下拉菜单），然后按播放，如以下截图所示：

![图 2.6 – Visual Studio – 运行项目

![图片 2.06_B17243.jpg]

图 2.6 – Visual Studio – 运行项目

当按下“播放”按钮时，将打开一个命令窗口，您可以在其中查看打印到屏幕上的任何控制台消息，并且浏览器中选中的新标签页也将打开。地址栏中的地址将指向 `http://localhost:#####/`，并且您的项目现在应该正在成功运行。端口号通常是一个随机数字。Visual Studio 使用的是它在您的机器上当前未使用的端口号，并且也不是默认或保留端口号，例如 `8000`。这是 Visual Studio 为同时运行多个项目的人提供的内置功能，因为我经常发现自己这样做。

如果您看不到此按钮，您可以通过在 Visual Studio 工具栏区域的空白处右键单击来将其添加到工具栏。在出现的下拉菜单中，从选择列表中选择**标准**。如果您从列表中选择**Python**，您可以直接从工具栏中使用与虚拟环境一起工作的工具。

除了打开的命令行窗口外，您选择的浏览器也会在新标签页中打开，指向`http://127.0.0.1:#####/`，其中`#####`是 Visual Studio 使用的随机端口。在这里，您将看到成功安装 Django 的标志性登录页面，如下面的截图所示：

![图 2.7 – Django 安装成功](img/Figure_2.07_B17243.jpg)

图 2.7 – Django 安装成功

要在 Visual Studio 中使用标准端口`8000`并指向`http://localhost:8000/`的播放按钮，请按照下一小节的步骤操作。

### 手动设置端口

我们可以在 Visual Studio 中非常容易地指定端口号，以控制每个项目使用哪个端口。在 IDE 顶部的**项目**选项卡中，从下拉菜单中选择**成为 Django 企业开发者属性**。在打开的编辑器选项卡中，点击内部的**调试**选项卡，并在标题为**端口号**的字段中指定端口号。

接下来，我们将看到如何使用命令行创建虚拟环境。

## 使用命令行

对于许多更喜欢终端或命令行窗口的您，一个流行的模块，可用于在 Windows、Mac 和 Linux 上创建虚拟环境，称为`venv`。

按照以下步骤创建您的虚拟环境：

1.  打开您的终端或命令行窗口，导航到我们在*第一章*“承担一个巨大项目”中创建的本地 Git 仓库。

1.  导航到您第一个名为`becoming_a_django_entdev`的文件夹。这个文件夹应该已经存在，并且包含在您使用命令行或 IDE 在此章早期启动项目时创建的文件：

    ```py
    PS C:\Projects\Packt\Repo> cd becoming_a_django_entdev
    ```

1.  在此目录中，即`manage.py`文件所在的目录，运行以下命令以创建一个名为`virtual_env`的虚拟环境：

    ```py
    PS C:\Projects\Packt\Repo\becoming_a_django_entdev> python -m venv virtual_env
    ```

`venv`模块应该与所有 Python 安装一起提供，但如果您在三个主要平台上运行此命令时遇到问题，请访问以下文档以帮助您调试问题：[`docs.python.org/3/library/venv.html`](https://docs.python.org/3/library/venv.html)。

1.  对于 Windows，激活您的虚拟环境：

    ```py
    PS C:\Projects\Packt\Repo\becoming_a_django_entdev> virtual_env/Scripts/activate
    ```

Mac 和 Linux 用户应跳转到*激活虚拟环境*小节，了解如何在那些平台上激活虚拟环境。

1.  接下来，通过运行以下命令安装项目根目录中`requirements.txt`文件定义的包，其中`manage.py`文件位于：

    ```py
    PS C:\Projects\Packt\Repo\becoming_a_django_entdev> pip install -r requirements.txt
    ```

在 Windows、Mac 和 Linux 上创建虚拟环境的另一种方法是使用`virtualenv`，如下所示：

```py
PS C:\Projects\Packt\Repo\becoming_a_django_entdev> pip install virtualenv
PS C:\Projects\Packt\Repo\becoming_a_django_entdev> virtualenv virtual_env
```

现在我们已经为我们的项目创建了一个虚拟环境，让我们激活这个虚拟环境并运行项目。

### 激活虚拟环境

我们可以不使用 IDE，而是激活一个虚拟环境，直接从命令行运行项目。如果您已经在上一小节中激活了虚拟环境，您可以跳过这一小节。以下是一些示例，展示了如何为每个主要平台（Windows、Mac 和 Linux）激活虚拟环境：

按照以下步骤激活您的虚拟环境：

1.  对于 Windows 用户，导航到项目的根目录，其中包含`manage.py`文件，然后使用以下命令激活您的虚拟环境：

    ```py
    PS C:\Projects\Packt\Repo\becoming_a_django_entdev> virtual_env/Scripts/activate
    ```

1.  Mac 和 Linux 用户需要运行以下命令：

    ```py
    PS C:\Projects\Packt\Repo\becoming_a_django_entdev> source virtual_env/bin/activate
    ```

如果成功，您现在将在终端中看到以下提示，等待在虚拟环境中执行下一个命令：

```py
(virtual_env) PS C:\Projects\Packt\Repo\becoming_a_django_entdev>
```

只有在激活了虚拟环境后，您才能执行标准的 Django 管理命令。

注意

从现在起，为了节省空间、去除杂乱和防止在提供活动虚拟环境中的终端或命令行示例时产生混淆，以下示例将按照以下方式展示：

`(virtual_env) PS >`

现在虚拟环境已经激活，让我们运行我们的项目。

### 运行项目

如果您决定使用命令行而不是 Visual Studio IDE 来创建项目，您目前将无法运行项目。这是因为 Visual Studio 创建了一个本地 SQLite 数据库，对任何必要的迁移进行了处理，并自动为我们在`requirements.txt`文件中包含的每个包进行了迁移。作为参考，运行项目的命令如下所示。您需要完成本章*PostgreSQL*部分中的练习来配置数据库，然后才能成功执行以下命令。完成这些操作后，您可以回到这一节。

确保您仍然位于与`manage.py`文件相同的文件夹中，并且您的虚拟环境处于激活状态，然后执行以下`runserver`命令：

```py
(virtual_env) PS > python manage.py runserver
```

如果运行成功，您将在终端中看到以下信息。每次您加载页面并且项目正在运行时，您都会在这个终端中看到所有打印到屏幕上的消息：

```py
Watching for file changes with StatReloader
Performing system checks...
System check identified no issues (0 silenced).
Django version 2.2.24, using settings 'becoming_a_django_entdev.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

### 手动设置端口号

使用终端或命令行窗口手动指定端口号很简单。然而，每次我们运行项目时都必须这样做，这与在 IDE 中设置项目配置不同。以下示例将端口号选项添加到上一节中使用的相同`runserver`命令，指定使用端口号`8000`来运行项目：

```py
(virtual_env) PS > python manage.py runserver 8000
```

# 项目配置

所有项目都需要以某种方式配置，以便与所有涉及的包和主机一起工作。我们将使用我们刚刚创建的项目，并配置 `settings.py`、`.env`、`procfile` 和 `.gitignore` 等文件。以 `settings.py` 文件为例——此文件或文件将存储项目中使用的所有全局常量。当使用包时，它们通常提供一种方式，可以从主设置文件中自定义该包的行为。其他文件，如 `.env` 和 `procfile`，将用于防止在与 Heroku 作为主机工作时出现部署问题。目前，由于我们创建了项目文件，我们无法成功部署到 Heroku 环境。在尝试成功部署之前，请先完成以下配置部分。

## Django settings.py 文件

在我们创建项目时自动为我们生成的 `settings.py` 文件中，我们需要添加特定于我们项目的设置。

按照以下步骤配置您的项目设置：

1.  在 `settings.py` 文件的顶部，在存在的前两个导入下面添加 `import django_heroku`。还要添加 `import dotenv` 和 `dj_database_url`。这两个包将用于建立数据库连接。您的 `settings.py` 文件顶部应类似于以下示例：

    ```py
    # /becoming_a_django_entdev/settings.py
    import os
    import posixpath
    import django_heroku
    import dj_database_url
    import dotenv
    ```

注意，某些系统可能在其文件顶部显示 `from pathlib import Path` 而不是 `import os` 和 `import posixpath`。

在 `settings.py` 文件的底部，添加以下代码：

```py
# /becoming_a_django_entdev/settings.py
...
django_heroku.settings(locals())
```

此声明将从 `django_heroku` 包导入特定的 Heroku 设置。

1.  为了使您的 DNS 正确工作，您需要告诉 Django 您允许主机访问此站点。这是 Django 的内置安全功能，旨在阻止常见的 `ALLOWED_HOSTS`，应类似于以下示例：

    ```py
    # /becoming_a_django_entdev/settings.py
    ...
    ALLOWED_HOSTS = [
        'your-domain.com',
        'www.your-domain.com',
        'dev.your-domain.com',
        'staging.your-domain.com',
        'becoming-an-entdev.herokuapp.com',
        'mighty-sea-09431.herokuapp.com',
        'pure-atoll-19670.herokuapp.com',
    ]
    ```

可用的选项包括通配符，如星号字符（`*`），这将允许任何内容。然而，这并不被视为最佳实践，并且非常不安全。如果一个域名以单个点开始，就像以下示例中的第二个条目，它也将作为通配符，允许所有对应父域的子域。如果您决定使用这些通配符选项，请谨慎使用：

```py
# /becoming_a_django_entdev/settings.py
...
ALLOWED_HOSTS = [
    '*',
    '.your-domain.com',
]
```

1.  从 Django 3.2 版本开始，我们必须在设置文件中添加一个名为 `DEFAULT_AUTO_FIELD` 的变量。在此版本之前，这并不是必要的。此设置告诉 Django 如何处理和处理所有对象的全部主键。如果没有此设置，我们否则必须为每个我们创建的模型类添加一个名为 `id = models.AutoField(primary_key=True)` 的字段。由于这是一项艰巨的任务，我们可以通过在 `settings.py` 文件的任何位置放置以下示例来避免它：

    ```py
    # /becoming_a_django_entdev/settings.py
    ...
    DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'
    ```

目前，我们只需保留默认的 `DATABASES = {...}` 设置。在本章的 *为 Heroku 准备 PostgreSQL* 部分，我们将讨论根据我们的特定用途更改此设置。还将提供所有支持数据库类型的示例，以便在处理不遵循本书中每个示例的项目时提供快速参考。

让我们创建我们的环境文件。

## 创建环境文件

Heroku 将使用名为 `.env` 的环境文件来存储在各个环境中运行项目时使用的环境相关变量。我们使用它来告诉 Heroku 诸如每个环境使用什么数据库或是否应该开启/关闭调试等信息。Heroku 建议我们在本地使用 SQLite，但在他们的环境应用中不使用。它可以在本地配置为使用 PostgreSQL，我们将在本章末尾进行演示。即使您在本地使用 SQLite3，您仍然需要在您的机器上安装 PostgreSQL 软件套件，以便驱动程序可以与所有远程连接一起工作。在我们讨论在本地设置 PostgreSQL 之前，我们需要完成配置项目的其余部分。

### 本地变量

要创建您的本地变量，请按照以下步骤操作：

1.  从您的项目根目录运行以下命令，其中包含您的 `manage.py` 文件：

    ```py
    (virtual_env) PS > echo 'DATABASE_URL=sqlite:///db.sqlite3' > .env
    ```

`echo` 语句用于创建一个名为 `.env` 的新文件，其中包含一行内容，即 `DATABASE_URL=sqlite:///db.sqlite3`。

Windows 用户在尝试运行上述命令时可能会遇到类似 `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte` 的错误消息。这是在 Windows 机器上经常出现的问题，会导致此类错误。为了克服这个问题，请使用 Notepad++ 打开您的 `.env` 文件，转到 **编码** 下拉菜单，从选择列表中选择 **UTF-8**，然后保存。如果这失败了，这是可能的，只需删除文件并使用 Notepad++ 从头开始重新创建它，使用正确的编码创建文件。

1.  无论此文件是如何创建的，请确保以下代码行存在于您的本地 `.env` 文件中：

    ```py
    # /becoming_a_django_entdev/.env
    DATABASE_URL = sqlite:///db.sqlite3
    ```

我们放置在此文件中的代码指向名为 `db.spqlite3` 的 SQLite3 数据库文件的位置。如果您没有使用 Visual Studio IDE 创建项目，此文件尚不存在，并且您的项目目前无法运行。

1.  我们还可以向此文件添加其他变量，然后使用 `settings.py` 文件中的 `python-dotenv` 包来访问这些变量。要访问这些变量，我们必须使用以下示例在 `settings.py` 文件中加载 `.env` 文件：

    ```py
    # /becoming_a_django_entdev/settings.py
    ...
    import dotenv
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    dotenv_file = os.path.join(BASE_DIR, ".env")
    if os.path.isfile(dotenv_file):
        dotenv.load_dotenv(dotenv_file)
    SECRET_KEY = os.getenv('SECRET_KEY')
    ```

将高亮代码放置在`settings.py`文件顶部附近，位于现有的`BASE_DIR`变量下方。在此示例中，我们还用`os.getenv('SECRET_KEY'),`替换了`SECRET_KEY`变量的字符串值，通过`os.getenv()`我们可以访问`.env`文件中找到的任何变量。

1.  为了使我们的项目在本地运行，我们需要将`settings.py`文件中`SECRET_KEY`变量的字符串值添加到`.env`文件中。使用以下示例编写您的变量，不要在此文件中为任何字符串值添加引号：

    ```py
    # /becoming_a_django_entdev/.env
    ...
    SECRET_KEY = my_randomly_generated_key
    ```

就这样，我们的本地变量现在已配置。接下来，让我们配置我们的远程变量。

### 远程变量

远程变量与我们在本地创建的`.env`变量相同，但现在具有与每个远程环境相关的值。由于我们将在我们的仓库中忽略`.env`文件，我们将在名为“创建.gitignore 文件”的子节中讨论这一点，因此我们需要为每个存在的 Heroku 环境手动在`.env`文件中创建`SECRET_KEY`变量。由于 Heroku 已经为我们创建了这些`.env`文件，我们将借此机会添加所需的变量。这些步骤也可以在任何时候在 SDLC（软件开发生命周期）中根据需要添加变量。

要添加您的远程变量，请使用 Heroku**命令行界面**（**CLI**）登录您的 Heroku 账户，并将以下代码块中提供的应用程序名称替换为您的 Heroku 应用程序名称：

```py
(virtual_env) PS > heroku login
(virtual_env) PS > heroku config:add SECRET_KEY=my_randomly_generated_key --app becoming-an-entdev
(virtual_env) PS > heroku config:add SECRET_KEY=my_randomly_generated_key --app mighty-sea-09431
(virtual_env) PS > heroku config:add SECRET_KEY=my_randomly_generated_key --app pure-atoll-19670
```

在前面的代码中，为存在的三个环境中的每一个提供了一个示例。为每个环境逐个执行它们。

最好为每个环境提供一个不同的`SECRET_KEY`值。在*第三章*，“模型、关系和继承”一节中，我们在名为“生成`SECRET_KEY`变量”的子节中讨论了与 Django shell 一起工作后，将探讨如何以更安全的方式生成`SECRET_KEY`变量。将此`SECRET_KEY`变量添加到每个 Heroku 环境中的操作仍将如本节所述进行。

接下来，让我们创建一个**Procfile**，即**进程文件**。

## 创建 Procfile

存储在仓库根目录（`.git`文件夹所在位置）的`procfile`。此文件告诉 Heroku 其余项目文件的存放位置，具体来说，是`procfile`文件内`wsgi.py`或`asgi.py`文件的存放位置，使用标准的 Python 路径语法。`wsgi.py`文件通常位于任何 Django 项目的`project_name`文件夹中，创建时；该文件夹位于`manage.py`文件所在的同一文件夹中。如果您想使用异步服务器网关接口，则应指定`asgi.py`文件的存放位置。

假设`manage.py`文件和 Django 项目文件夹与仓库的根目录位于同一文件夹中，我们将在该 procfile 中包含以下路径：

```py
# ./Procfile in Root of Repository
```

```py
web: gunicorn becoming_a_django_entdev.wsgi
```

我们的项目文件嵌套在 `/becoming_a_django_entdev/` 文件夹中一层，就像在 IDE 中操作或在本书早期使用 `startproject` 命令创建项目时那样。如果我们尝试使用标准的 Python 路径语法添加此目录，我们将在部署到 Heroku 时遇到问题。相反，在一行代码中，告诉 Heroku 首先更改目录，然后执行前面的命令，使用以下代码示例：

```py
# ./Procfile in Root of Repository
```

```py
web: sh -c 'cd ./becoming_a_django_entdev/ && exec gunicorn becoming_a_django_entdev.wsgi --log-file -'
```

请使用第二个示例。只有当你已经将你的项目结构设置为存储在仓库的根目录，并且不太可能使用 Heroku 作为你的主机时，才使用第一个示例。如果没有之前显示的 `--log-file –` 参数，Heroku 也无法部署。使用 `log` 参数，你可以从每个应用的 Heroku 仪表板中读取部署错误。

接下来，我们需要控制当 Django 与 Heroku 一起使用时如何管理静态文件。

## Django 静态文件

传统上，我们不需要修改设置文件以允许 Django 在项目中使用静态文件。在我们的情况下，我们需要添加到现有的设置中，以便 Heroku 能够与它们一起工作。Heroku 使用一个名为 `whitenoise` 的包来处理你的静态文件，该包在 `requirements.txt` 文件中安装。这些文件是 `.css`、`.js`、图像或字体文件，它们位于任何 Django 应用程序的 `static` 文件夹中。

按照以下步骤配置你的项目以使用 `whitenoise` 包：

1.  将以下行添加到你的 `settings.py` 文件中找到的 `MIDDLEWARE` 设置。添加到该列表中任何现有项的下方：

    ```py
    # /becoming_a_django_entdev/settings.py
    ...
    MIDDLEWARE = [
        ...,
        'whitenoise.middleware.WhiteNoiseMiddleware',
    ]
    ```

1.  Heroku 还需要我们添加一个名为 `STATICFILES_STORAGE` 的变量。将此变量添加到你的 `settings.py` 文件中，位于你的 `STATIC_URL` 和 `STATIC_ROOT` 变量之上，如下所示：

    ```py
    # /becoming_a_django_entdev/settings.py
    ...
    STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
    STATIC_URL = '/staticfiles/'
    STATIC_ROOT = posixpath.join(
        *(BASE_DIR.split(os.path.sep) + ['staticfiles'])
    )
    ```

1.  如果你的值尚未更改，将你的 `STATIC_URL` 和 `STATIC_ROOT` 变量的值更改为前面代码中显示的值。

我们静态文件现在应该已经正确连接。到目前为止，我们已经在本地配置了一切，以便成功部署到我们的任何远程 Heroku 环境。

注意

在 Django 中，当项目在本地运行时，如果 `DEBUG = False`，即使静态文件出现在你的远程环境中，它们也无法加载。这种情况发生的原因是服务器通常配置为处理如何提供你的静态文件，而不是 Django 控制这一点，类似于 Heroku 使用 `whitenoise` 包来提供其静态文件的方式。当 `DEBUG` 设置为 `False` 时，`whitenoise` 包也用于在本地提供静态文件。

让我们接下来连接我们的媒体文件。

## Django 媒体文件

与静态文件类似，媒体文件路径需要配置，以便与 `whitenoise` 包和 Heroku 一起使用。媒体文件被认为是用户上传到你的系统中的任何内容，例如图片、音频文件或另一个文档。这两个变量，`MEDIA_URL` 和 `MEDIA_ROOT`，在 Django 中默认定义为空字符串；我们只需设置它们的值，指向我们想要放置它们的媒体文件夹。除了这些设置之外，还可能需要额外的步骤才能与 Heroku 和媒体文件一起使用。请参考这里找到的详细 Python 指南以了解更多信息：[`devcenter.heroku.com/articles/s3-upload-python`](https://devcenter.heroku.com/articles/s3-upload-python)。

要定义你的媒体相关变量，在你的 `settings.py` 文件中，在 `STATIC_URL` 和 `STATIC_ROOT` 变量下方，包括以下两个变量：

```py
# /becoming_a_django_entdev/settings.py
```

```py
…
```

```py
MEDIA_URL = '/media/'
```

```py
MEDIA_ROOT = posixpath.join(
```

```py
    *(BASE_DIR.split(os.path.sep) ['media'])
```

```py
)
```

就这些了，这就是我们需要在我们的 `settings.py` 文件中配置的，以连接静态和媒体文件。在 *第四章* 的 *URLs, Views, and Templates* 小节中，标题为 *Function – static()* 的部分，我们将在这些内容完全集成到我们的项目中之前，需要配置额外的 URL 模式。

在我们尝试部署之前，我们需要创建前面提到的 `.gitignore` 文件，以及完成本章的 *为 Heroku 准备 PostgreSQL* 部分的操作。

## 创建一个 .gitignore 文件

在部署之前，我们想要创建的最后一个文件是一个 `.gitignore` 文件，这样我们就可以只分享我们想要分享的内容，并排除像现在在 `virtual_env` 和 `.vs` 文件夹中看到的那些庞大的文件。这个文件将用于确保我们不会意外地将任何不想要的代码推送到我们的远程仓库。一旦文件被推送到仓库，它将始终被跟踪，直到它被删除；因此，我们想要确保我们首先这样做。忽略由 Visual Studio 创建的 `.suo` 文件也是一个好主意。这些文件包含用户特定的信息，例如断点和调试器监视器。你还会想要忽略任何不需要与其他开发者共享的 `build`、`bin` 和 `log` 文件。这也是我们将定义一个模式来忽略定义环境特定变量的 `.env` 文件的地方。当我们到达本章的 *PostgreSQL* 小节时，我们将很快创建开发、预发布和生产的 `.env` 文件，这是成功部署所必需的。

在你的仓库根目录中创建一个名为 `.gitignore` 的文件，其中你的 `.git` 文件夹位于。然后，添加这里显示的项目。它们将作为你的忽略模式：

```py
# ./.gitignore in Root of Repository
```

```py
# Keep Rule
```

```py
!gitkeep.txt
```

```py
# Django #
```

```py
db.sqlite3
```

```py
*.log
```

```py
*.pyc
```

```py
__pycache__
```

```py
# Media – User Generated Content #
```

```py
media/
```

```py
# Environments #
```

```py
.env
```

```py
virtual_env/
```

```py
# Visual Studio and Visual Studio Code #
```

```py
*.suo
```

```py
*.pyproj
```

```py
*.pyproj.user
```

```py
*.history
```

```py
.vs/
```

```py
obj/
```

```py
# Heroku
```

```py
staticfiles/
```

这些只是与我们的特定 Heroku/Django 和 Visual Studio 配置相关的几个示例。您的 `.gitignore` 文件可以包含更多的忽略模式。在这本书提供的源代码中，已经提供了许多其他忽略模式的示例，并按类别划分，如果您需要，欢迎使用。

忽略模式接受通配符，例如前面示例中使用的星号（`*`）。它们还接受一个排除操作符，用感叹号（`!`）符号表示。`!gitkeep.txt` 模式在需要空文件夹存在于仓库中的情况下常用，例如媒体文件夹。如果文件夹内没有任何内容，Git 会自动忽略该文件夹；如果我们把一个 `gitkeep.txt` 文件放在那个媒体文件夹里，我们就可以绕过忽略该文件夹内所有内容的困境。媒体文件夹用于用户生成的内容，但我们不希望跟踪放入其中的文件。实际上，一些空文件夹是必需的，以防止运行时出错，例如当上传图片到系统而媒体文件夹尚未存在时。有时，这甚至可以防止一个仓库的新克隆版本在第一次运行时完全无法正常运行，这取决于该文件夹的存在与否。

您现在可以根据需要将代码推送到开发、预发布或生产环境。只需在执行 Git 操作时注意您正在推送到或从哪个远程环境和分支，当完成这些操作后，您可以访问之前讨论过的 Heroku 仪表板*第一章*，*承担一个巨大的项目*，以查看部署是否成功，如果不成功，请查看日志以了解失败的原因。

## 创建 Django 应用

在 Django 中，您的 `manage.py` 文件所在的文件夹中的所有文件集被认为是您的项目。您的项目可以包含许多应用，其中 Django 中的 **应用** 被视为在项目中执行某些操作的过程，例如日志记录、琐事或记录保存，仅举几个例子。它们也可以是简单的东西，例如来自 PyPI 包的特殊表单字段或事件监听器，其中该特定包基本上被视为我们正在安装的 Django 应用。

应用是我们编写模型、视图、测试用例、表单、管理类、HTML 模板和与该应用相关的静态文件的地方。应用也是项目中大部分代码存在的地方。它们也被设计成模块化的，如果我们想的话，一个应用可以在多个项目中共享。接下来，我们将为自己创建一个应用，并将其命名为 `chapter_2`。每一章都将遵循相同的命名约定。我们将尝试根据本书每一章的内容来组织应用。有些章节可能涉及项目中的每个文件，本书的这一章就是这样。这是因为在本章中，我们正在处理项目的全局文件。选择您想要使用的创建应用的方法，无论是使用 IDE 还是命令行驱动的方法。

### 使用 IDE

要轻松创建应用，请在此处右键单击 `chapter_2` 中的项目名称，如图所示：

![图 2.8 – Visual Studio – 添加应用](img/Figure_2.08_B17243.jpg)

图 2.8 – Visual Studio – 添加应用

在 Visual Studio 中，您可以选择项目中的任何文件夹或子文件夹来创建 Django 应用。如果您希望以不同的方式组织项目结构，只需右键单击您想要创建该应用的文件夹，而不是之前描述的文件夹。

注意

Visual Studio 将使用虚拟环境中安装或更新的 Django 版本来安装应用。没有像之前通过 IDE 创建项目时那样的特殊版本 2.1.2 用例；要安装的应用文件版本将是 Django 4.0。此外，请注意，您可以使用 IDE 在项目根目录中的任何目录中创建应用。您不必将它们安装在本练习中使用的目录中。

让我们使用命令行创建一个 Django 应用。

### 使用命令行

要使用命令行创建 Django 应用，您首先需要为新应用创建文件夹结构。在这里，我们希望它完全模仿之前练习中 IDE 创建的文件夹结构，以确保每种方法都能产生相同的结果，并与本书的其他部分协同工作。从 `manage.py` 文件所在的同一文件夹开始，确保您的虚拟环境已激活，并运行以下创建文件夹命令。然后，执行下面的传统 `startapp` 命令，以创建名为 `chapter_2` 的应用：

```py
(virtual_env) PS > mkdir becoming_a_django_entdev/chapter_2
(virtual_env) PS > python manage.py startapp chapter_2 becoming_a_django_entdev/chapter_2
```

在前面的示例中，我们首先使用 `mkdir` 命令创建了 `chapter_2` 文件夹，然后执行了 `startapp` 命令。我们还提供了一个参数，指定了应用将被安装到的文件夹，也就是 Visual Studio 为我们放置的同一个文件夹。再次强调，您不必将应用安装在这个目录中——如果您以不同的方式创建了文件夹结构，在整个书籍中根据需要调整即可。

### 激活新的 Django 应用

Django 应用创建后，除非你将其作为已安装的应用包含在 Django 中，否则它不会自动在你的项目中工作。

要激活你的应用，请按照以下步骤操作：

1.  所有 Django 应用都必须包含在`INSTALLED_APPS`列表中。添加你的章节应用，如下所示：

    ```py
    # /becoming_a_django_entdev/settings.py
    ...
    INSTALLED_APPS = [
        ...
     'django_extensions',
      'becoming_a_django_entdev.chapter_2',
    ]
    DEFAULT_AUTO_FIELD'= 'django.db.models.AutoField'
    ```

1.  有时候，有必要告诉 Django 在你的目录树中查找你的应用的位置。在你刚刚创建的应用的`apps.py`文件中，你可以使用标准的 Python 路径语法指定你的应用的位置。修改以下示例中`name =`变量的值：

    ```py
    # /becoming_a_django_entdev/chapter_2/apps.py
    from django.apps import AppConfig
    class chapter_2Config(AppConfig):
        name= 'becoming_a_django_entdev.chapter_2'
    ```

注意，你必须为所有未来创建的章节应用都这样做。

项目文件结构现在应该看起来像以下树，其中突出显示的项目是我们从开始配置虚拟环境以来所做的添加：

`├── .git`

`│ ├── .gitignore`

`│ ├── procfile`

`├── requirements.txt`

`├── readme.md`

`├── becoming_a_django_entdev`

`│ ├── .vs`

`│ ├── becoming_a_django_entdev.sln`

`│ ├── db.sqlite3`

`│ ├── manage.py`

`│ ├── media`

`│ ├── obj`

`│ ├── requirements.txt`

`│ ├── virtual_env`

`│ ├── staticfiles`

`│ └── becoming_a_django_entdev`

`│ ├── chapter_2`

`│ ├── __init__.py`

`│ ├── asgi.py`

`│ ├── settings.py`

`│ ├── urls.py`

`│ └── wsgi.py`

接下来，让我们讨论 Django 支持的不同类型的数据库。

# 使用基本的数据库设置

一个网站本身如果没有数据库与之交互就远远不够有用；这就是为什么 Visual Studio IDE 附带了一个轻量级且便携的 SQLite3 数据库。每当执行`startproject`命令时，Visual Studio 都会在`manage.py`文件所在的同一文件夹中创建一个名为`db.sqlite3`的文件。如果你使用终端或命令行窗口创建了项目，那么你将不会在下面的配置示例中使用 SQLite 数据库，如果你尝试在没有此数据库的情况下运行项目，它将失败。这是 Django 直接支持的五种标准数据库类型之一。除了 Django 直接支持的五种类型之外，还可以使用其他数据库类型。我们还将提供一个如何配置其他类型数据库的示例，例如 Microsoft SQL Server 数据库。不属于五种标准类型的类型将需要使用 Django 提供的引擎之外的引擎，这意味着你需要找到一个 PyPI 包来安装或为你要支持的数据库类型编写自己的。

Django 的五种标准数据库类型如下：

+   **SQLite**

+   **MySQL**

+   **MariaDB**

+   **Oracle**

+   **PostgreSQL**

以下示例将帮助您配置您选择的现有数据库。最后一个示例，PostgreSQL，将是本章后续示例中使用的类型。您可以使用本书剩余章节中的任何类型。使用数据库还有更多内容，例如创建表和执行查询，这些内容将在后续章节中讨论。这些示例仅涉及为建立与数据库的连接而编写的设置，无论该连接是本地还是远程。

## SQLite

SQLite 是一种基于 C 语言的 **关系数据库管理系统**（**RDBMS**）。它非常轻量级且易于携带，在数据库管理方面有时被称为首选的 *磁盘存储* 方法。这是在极短的时间内将 *概念验证* 项目启动起来的首选方法。这种数据库类型甚至可以在驱动器和存储库中共享，以便快速传输和携带项目及其数据。如果安全性对您来说和它应有的重要性一样重要，则不建议这样做。

在实际应用中使用此数据库类型以及与 Heroku（如我们用于托管应用程序的方式）结合使用时，存在许多问题。您可能会发现您可以在远程 Heroku 环境中暂时使其工作。然而，在应用程序的生命周期中，每次部署都会导致数据的完全丢失。因此，我们将不得不偏离 Django 附带的标准数据库，并依赖我们环境中更健壮的数据库系统。

一个标准的 SQLite3 数据库配置示例如下：

```py
# /becoming_a_django_entdev/settings.py
```

```py
...
```

```py
DATABASES = {
```

```py
    'default': {
```

```py
        'ENGINE': 'django.db.backends.sqlite3',
```

```py
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
```

```py
    }
```

```py
}
```

根据需要添加额外的 `default` 参数。

## MySQL

MySQL 是一种更健壮的 SQL RDBMS。它是开源的，就像 SQLite 一样，它与 Windows、Mac 和 Linux 系统兼容。它被设计成作为客户端-服务器模型工作，这意味着软件安装在客户端机器上以执行服务器端将监听并响应的请求操作。MySQL 已经成为有史以来最受欢迎和最常用的数据库类型之一，尤其是如果您考虑了所有使用 MySQL 分支构建的其他数据库选择。

按照以下步骤配置您的 MySQL 数据库连接：

1.  一个标准的 MySQL 数据库连接示例如下：

    ```py
    # /becoming_a_django_entdev/settings.py
    ...
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql', 
            'NAME': 'database_name',
            'USER': 'database_user',
            'PASSWORD': 'database_password',
            'HOST': 'localhost',
            'PORT': '3306',
        }
    }
    ```

1.  为了使这种数据库类型正常工作，您还需要安装以下包并将其包含在您的 `requirements.txt` 文件中：

    ```py
    # requirements.txt
    ...
    Mysqlclient
    ```

根据您的项目需要调整默认参数。

## MariaDB

MariaDB 实际上是 MySQL 的一个分支；它是一个社区开发的版本，随着时间的推移发展成现在的样子。MariaDB 和 MySQL 数据库类型都提供技术支持。由于两者都是开源的，因此在线上也有大量的信息和资源免费提供。尽管 MariaDB 的使用不如 MySQL 那么广泛，但它仍然非常受欢迎。由于 MariaDB 实质上只是另一个 MySQL 安装，因此你的 `settings.py` 文件中 `DATABASES` 配置的 `ENGINE` 将与上一小节中标题为 *MySQL* 的示例相同。两者也将需要你安装相同的 `mysqlclient` 包，并将其包含在 `requirements.txt` 文件中。

## Oracle

**Oracle 数据库**是由 Oracle 创建的多模型、对象关系型数据库管理系统。它主要用于**在线事务处理**（**OLTP**）和**数据仓库**。与其他网络开发数据库相比，这个数据库在使用和功能方面要复杂一些。它旨在用于企业网格计算，这是一种将一系列网络计算机组合起来作为一台更大的虚拟超级计算机的方法。Oracle 开发了一个数据库专门用于这些网络集群配置。它被认为是当今最复杂和最健壮的数据库系统之一，超出了本书的范围。本书中的大部分代码仍然可以与这种数据库类型一起使用，但可能需要进行一些修改。

按照以下步骤配置你的 Oracle 数据库连接：

1.  可以使用以下示例建立典型的 Oracle 数据库连接：

    ```py
    # /becoming_a_django_entdev/settings.py
    ...
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.oracle',
            'NAME': 'xe',
            'USER': 'database_user',
            'PASSWORD': 'database_password,
            'HOST': 'database_host',
            'PORT': 'database_port_#',
        }
    }
    ```

1.  Oracle 数据库类型还需要安装以下包，并将其包含在 `requirements.txt` 文件中：

    ```py
    # requirements.txt
    ...
    cx_Oracle
    ```

Oracle 还需要遵循额外的设置和配置步骤，并且它们在其网站上提供了指导：[`cx-oracle.readthedocs.io/en/latest/user_guide/installation.html`](https://cx-oracle.readthedocs.io/en/latest/user_guide/installation.html)。

## SQL Server

这是一种 Django 支持的五种标准数据库类型之外的数据库类型，我们将提供一个示例。由微软开发的微软 SQL Server 数据库类型也是 MySQL 的一个分支，并且目前在微软的企业数据库系统中得到广泛应用。许多系统今天都依赖于这种数据库类型，它通常与 Azure 托管的应用程序结合使用。

按照以下步骤配置你的微软 SQL Server 数据库连接：

1.  可以使用以下连接设置建立微软 SQL Server 数据库类型：

    ```py
    # /becoming_a_django_entdev/settings.py
    ...
    DATABASES = {
        'default': {
            'ENGINE': 'sql_server.pyodbc',
            'NAME': 'database_name',
            'USER': 'database_user',
            'PASSWORD': 'database_password',
            'HOST': 'database_host',
            'PORT': '1433',
            'OPTIONS': {
                'driver': 'ODBC Driver 17 for SQL Server',
            },
        }
    }
    ```

1.  此数据库类型还需要安装以下包，并将它们包含在你的 `requirements.txt` 文件中。根据你的设置，你可能只需要以下三个包中的一个或两个；然而，添加所有三个也不会有害：

    ```py
    # requirements.txt
    ...
    pyodbc
    django-pyodbc
    django-pyodbc-azure
    ```

接下来，我们将安装本书其余部分使用的类型，即 PostgreSQL。

## PostgreSQL

**PostgreSQL**已成为大多数 Django 开发者首选的数据库，除非有特定原因需要使用其他选择。它被认为是一种关系型数据库管理系统（RDBMS），这是一种在相关表中存储面向对象数据的常用工具，这些表被称为对象。其主要特性包括**ANSI SQL 兼容性**和可扩展性，这意味着它是一种旨在让人们能够扩展的数据库类型。它将在每个主要操作系统上运行，如 Windows、Mac 和 Linux，使其具有多功能性。

### 标准 PostgreSQL settings.py

如果您在与 Heroku 之外的正常 Django 项目工作，请配置您的 PostgreSQL 数据库连接。

将以下设置添加到您的`settings.py`文件中：

```py
# /becoming_a_django_entdev/settings.py
```

```py
…
```

```py
DATABASES = {
```

```py
 '  'defa'lt': {
```

```py
     '  'ENG'NE': 'django.db.backends.postgre'ql',
```

```py
     '  'N'ME': 'database_n'me',
```

```py
     '  'H'ST': 'localh'st',
```

```py
     '  'U'ER': 'database_u'er',
```

```py
     '  'PASSW'RD': 'database_passw'rd',
```

```py
     '  'P'RT': '5'32',
```

```py
    }
```

```py
}
```

由于我们正在使用 Heroku，尽管我们仍然在每个远程 Heroku 环境中使用 PostgreSQL，但我们不能使用前面的设置。下一节将为我们提供特定于 Heroku 环境的设置和工具，以及如何在本地上使用 PostgreSQL 的其余说明。

# 为 Heroku 准备 PostgreSQL

本节专门用于配置每个环境，无论您是使用 IDE、命令行窗口还是终端。要安装和使用 PostgreSQL，我们需要在本地安装它，然后在每个远程环境中再次安装。

## 安装 PostgreSQL

本节将指导您在本地和远程机器上安装所需的 PostgreSQL 软件。

### 本地安装

要使用 PostgreSQL，我们需要在每台机器上安装一系列软件和驱动程序。在 Heroku 仪表板中，我们需要包括附加组件以在这些环境中安装 PostgreSQL。要在您的开发机器上安装它，您可以选择直接从出版商的网站下载您平台的安装程序，网站地址为：[`www.postgresql.org/download/`](https://www.postgresql.org/download/)。在安装过程中，请注意您设置的端口号、创建的密码以及是否要求您将任何内容添加到环境 PATH 变量中，请这样做！与手动配置相比，在安装期间检查/启用这些选项总是更容易。

在安装过程中，安装程序将询问您是否想要安装任何开发工具，例如数据库命令行工具或数据库管理工具。在安装您的 PostgreSQL 驱动程序时选择**PgAdmin**；我们将使用此工具在本章中演示几个示例。PgAdmin 管理工具用于访问和查看数据库表结构及其中的数据。PgAdmin 也与每个主要操作系统兼容，并且可以从 PostgreSQL 安装中单独下载和安装；您可以在他们的网站上找到它：[`www.pgadmin.org/download/`](https://www.pgadmin.org/download/)。

在我们之前创建的本地存储库中找到的`.env`文件中，将`DATABASE_URL=sqlite:///db.sqlite3`替换为以下值：

```py
# .env
```

```py
DATABASE_URL=postgres://postgres:your_password@localhost:5432/local_postgresql
```

将`your_password`占位符替换为您在安装 PostgreSQL 和 PgAdmin 时输入的密码。如有需要，替换任何其他占位符。默认情况下，用户名通常是`postgres`，但在安装过程中可以更改为其他名称。在我们可以实际运行项目之前，我们需要使用数据库管理工具 PgAdmin 创建本地数据库。即使您之前使用 Visual Studio IDE 创建项目，现在也会出现错误。

本书代码附带了一个名为`example.env`的文件，您可以将其用作参考，以确保您的文件包含适合我们软件堆栈的正确设置。

### 远程安装 – Heroku 仪表板

在 Heroku 中，我们必须逐个将 PostgreSQL 添加到我们所有的三个远程环境中。有两种方法可以做到这一点：通过 Heroku 仪表板或 Heroku CLI。我们将演示这两种方法。导航到以下位置的 Heroku 仪表板：[`dashboard.heroku.com/apps`](https://dashboard.heroku.com/apps)。我们应该看到所有三个环境，也称为 Heroku 应用。点击每个应用，并在提供的搜索字段中导航到`Heroku Postgres`。接下来，它将询问您选择哪种计划；您可以使用名为**Hobby-Dev**的免费计划，这是本书演示中将要使用的计划。稍后，您可以决定是否升级并支付新的计划费用。对于更技术性的需求，这些计划包括增加流量支持、同时连接数、RAM 量、加密措施、增加安全性以及更多。您可以访问他们的计划页面了解更多他们提供的内容：[`elements.heroku.com/addons/heroku-postgresql`](https://elements.heroku.com/addons/heroku-postgresql)。

如果成功，您将看到 Heroku 为我们自动创建了一个**DATABASE_URL**配置变量。您可以通过转到**设置**选项卡，然后在页面下方滚动到**配置变量**下找到这个变量。点击**显示配置变量**后，它也会出现。以下截图还显示，我们之前在本章中使用的 CLI 创建的相同**SECRET_KEY**变量也可以在这里找到。任何其他敏感的环境变量都可以以相同的方式创建并在这里找到。点击**添加**按钮可以消除使用 CLI 创建环境变量操作的需求：

![图 2.9 – Heroku – DATABASE_URL](img/Figure_2.09_B17243.jpg)

图 2.9 – Heroku – DATABASE_URL

让我们通过使用 Heroku CLI 而不是 Heroku 仪表板来安装 PostgreSQL。

### 远程安装 – Heroku CLI

使用 Heroku CLI，我们首先需要在终端或命令行窗口中登录我们的账户。以下示例显示了如何在我们的每个远程环境中安装 PostgreSQL，从生产环境开始：

```py
(virtual_env) PS > heroku login
(virtual_env) PS > heroku addons:create heroku-postgresql:hobby-dev
(virtual_env) PS > heroku addons:create heroku-postgresql:hobby-dev --app mighty-sea-09431
(virtual_env) PS > heroku addons:create heroku-postgresql:hobby-dev --app pure-atoll-19670
```

我们必须指定 Heroku 在我们创建这些环境时为我们生成的随机应用名称。如果我们需要指定我们想要使用的 PostgreSQL 版本，我们可以使用以下示例。如果不指定版本号，将使用可用的最新版本：

```py
(virtual_env) PS > heroku addons:create heroku-postgresql:hobby-dev --version=10
```

而不是去您的 Heroku 仪表板查看这些变量，执行以下命令也会显示每个环境存在的变量列表。将以下示例中的每个应用名称替换为您自己的 Heroku 应用名称：

```py
(virtual_env) PS > heroku config --app becoming_an_entdev
(virtual_env) PS > heroku config --app mighty-sea-09431
(virtual_env) PS > heroku config --app pure-atoll-19670
```

## 使用 PgAdmin 工具

PgAdmin 是我们将要使用的数据库管理工具，用于管理所有本地和远程数据库，以及创建我们需要连接的本地数据库。这个工具应该在 *安装 PostgreSQL* 子节中更早地安装过。如果没有安装，请重新查看该子节以获取更多信息。

### 创建本地数据库

在 `postgres`。在代码示例中引用我的本地数据库时，我给定的数据库名称是 `local_postgresql`。

![图 2.10 – PgAdmin – 创建本地数据库](img/Figure_2.10_B17243.jpg)

图 2.10 – PgAdmin – 创建本地数据库

接下来，我们将连接到远程数据库。

### 连接到远程数据库

对于每个远程数据库，我们都需要在 PgAdmin 工具中添加一个新的服务器连接。

按照以下步骤连接到远程数据库：

1.  首先从您的 Heroku 账户收集一些信息。在您的 Heroku 仪表板中，在每个 Heroku 应用下，导航到 **资源** 选项卡，并在 **附加组件** 下，点击如下截图所示的 **在新标签页中打开** 图标：

![图 2.11 – Heroku – 在新标签页中打开数据库](img/Figure_2.11_B17243.jpg)

图 2.11 – Heroku – 在新标签页中打开数据库

1.  在新打开的浏览器标签页中，导航到 **设置** 选项卡，并记下提供的信息。以下截图中的高亮字段是我们将在 PgAdmin 工具内部建立连接所需的重要信息。请记下它们：

![图 2.12 – Heroku – 数据库信息](img/Figure_2.12_B17243.jpg)

图 2.12 – Heroku – 数据库信息

1.  打开 PgAdmin，并从该应用的顶部导航栏中，在 **对象** 选项卡下，选择 **创建** | **服务器**。在打开的窗口中，在 **常规** 选项卡下，输入您选择的任何名称。这是为了仅用于本地参考来命名您的服务器连接。命名它们，以便您知道哪个环境是哪个，如下面的截图所示：

![图 2.13 – PgAdmin – 创建 - 服务器](img/Figure_2.13_B17243.jpg)

图 2.13 – PgAdmin – 创建 - 服务器

1.  此任务仅在 PgAdmin 工具中创建您的服务器连接；它并不是创建数据库服务器本身。使用在*图 2.12*中找到的信息，在**连接**选项卡下填写相应的字段，如下面的截图所示：

![图 2.14 – PgAdmin – 创建服务器连接](img/Figure_2.14_B17243.jpg)

图 2.14 – PgAdmin – 创建服务器连接

1.  将标记为**维护数据库**的字段中的值替换为 Heroku 提供的数据库名称；对**用户名**也做同样的操作。

1.  如果成功，您现在将在**PgAdmin**应用的**浏览器**面板中看到您的新服务器连接。**浏览器**面板通常位于 PgAdmin 程序的左侧。由于我们使用的是名为**Hobby-Dev**的免费计划，我们将在这个列表中看到成百上千的其他数据库。除了一个之外，所有其他数据库都将被灰色显示，就像以下截图所示：

![图 2.15 – PgAdmin – 查看表](img/Figure_2.15_B17243.jpg)

图 2.15 – PgAdmin – 查看表

在*图 2.12*中标记为颜色的数据库就是您的数据库名称，如*图 2.12*所示。展开您的数据库，您应该能够访问；尝试打开任何其他数据库将导致**权限被拒绝**。您可以通过购买 Heroku 提供的专用数据库托管计划来避免这种共享计划，这要安全得多。

现在，将您的文件更改推送到您的 Git 开发仓库，您应该会看到您的 Heroku 应用成功部署。

接下来，我们将调整我们的环境连接设置以在本地使用 PostgreSQL。

## 环境连接设置

对于我们的 Heroku 配置，我们需要在`settings.py`文件中使用以下示例定义一个默认数据库，而不是使用标准的 PostgreSQL 连接设置。

要为 Heroku 和本地使用配置 PostgreSQL，请按照以下步骤操作：

1.  将您的`DATABASES`变量更改为使用以下代码：

    ```py
    # /becoming_a_django_entdev/settings.py
    ...
    DATABASES = {
        'default': dj_database_url.config(
            conn_max_age=600
        )
    }
    ```

您也可以通过调整`conn_max_age`的值来增加已建立数据库连接允许存在的时长。默认值是`600`秒，相当于 10 分钟。由于`dj_database_url`模块将尝试使用`settings.py`文件登录 Heroku。

1.  在我们之前在这个文件底部放置的`django_heroku.settings(locals())`行下面放置以下代码；它应该看起来像以下示例：

    ```py
    # /becoming_a_django_entdev/settings.py
    ...
    django_heroku.settings(locals())
    options = DATABASES['default'].get('OPTIONS', {})
    options.pop('sslmode', None)
    ```

就这些了。让我们接下来在本地构建初始表结构。

## 构建初始表结构

接下来，我们需要创建与项目中的模型和任何第三方包相关的表结构。这些操作可以使用 Visual Studio IDE 或通过终端或命令行窗口执行。从以下选项中选择您的方法。

### 使用 IDE

在你的 IDE 中，你需要执行三个操作：创建迁移、迁移和创建超级用户，按照这个顺序。在 Visual Studio IDE 中，转到你的 **解决方案资源管理器**，然后右键单击你的项目名称。在弹出的菜单中，在 **Python** 下选择 **Django 创建迁移**，然后选择 **Django 迁移…**，如图所示。然后，选择 **Django 创建超级用户** 操作：

![图 2.16 – Visual Studio – manage.py 命令](img/Figure_2.16_B17243.jpg)

![图 2.16 – Visual Studio – manage.py 命令](img/Figure_2.16_B17243.jpg)

图 2.16 – Visual Studio – manage.py 命令

接下来，我们将使用命令行驱动的方法。

### 命令 – makemigrations 和 migrate

如果你没有使用 IDE 或者决定在 IDE 之外运行你的项目，这些命令就是为你准备的。确保你已经激活了你的虚拟环境，并且你位于与你的 `manage.py` 文件相同的目录中。使用终端或命令行窗口，通过以下两个示例在你的本地数据库中创建你的表结构：

```py
(virtual_env) PS > python manage.py makemigrations
(virtual_env) PS > python manage.py migrate
```

由于我们还没有创建任何模型，这两个命令将只创建默认的 `User` `Auth` 和其他所有 Django 安装的标准 Django 管理模型。

### 命令 – createsuperuser

第一次创建数据库表时，我们需要创建一个超级用户，这样我们才能成功访问管理面板。额外的用户可以在 Django 管理面板内部创建，或者通过再次执行以下代码片段中的命令来创建。现在运行此命令以创建超级用户：

```py
(virtual_env) PS > python manage.py createsuperuser
```

接下来，当被提示这样做时，输入它要求你提供的用户名、电子邮件地址和密码。记住这些信息，因为你将需要它们来访问在 *第六章* 中介绍的 Django 管理站点，*探索 Django 管理站点*。

## 远程数据迁移

执行 **远程数据迁移**意味着运行与我们为本地数据库执行的相同迁移命令，但针对每个远程数据库。

为了手动为每个我们创建的远程环境运行迁移命令，我们首先需要为每个环境激活 **Bash shell**。按照以下步骤进行操作：

1.  登录你的 Heroku 账户，然后逐个执行以下命令，每个环境一次。注意，在启动每个 shell 之后，你应该遵循以下步骤，然后再启动下一个 shell：

    ```py
    (virtual_env) PS > heroku login
    (virtual_env) PS > heroku run bash --app becoming-an-entdev
    (virtual_env) PS > heroku run bash --app mighty-sea-09431
    (virtual_env) PS > heroku run bash --app pure-atoll-19670
    ```

1.  对于每个你为它激活了 Bash shell 的环境，一旦它加载完成，你会看到你的命令行现在以美元符号字符（`$`）开头。你也会注意到，无论你之前在哪个目录中，它都会把你带到你的 `.git` 文件夹现在所在的目录。使用 `cd` 命令导航到你的 `manage.py` 文件所在的位置，然后运行以下迁移命令，就像我们为本地数据库所做的那样：

    ```py
    $ cd becoming_a_django_entdev
    $ python manage.py makemigrations
    $ python manage.py migrate
    ```

1.  输入 `exit` 并等待它从每个 shell 中退出，然后进入下一个环境。

请记住，所有这些命令所做的只是迁移您的表结构；它们实际上并没有迁移那些表中的数据。

接下来，我们将练习使用 Heroku 提供的推送/拉取操作来迁移数据和表结构，从而无需在 Bash shell 中运行这些命令。

## Heroku 数据库推送/拉取操作

Heroku 内置了将数据推送到和从指定数据库的功能。这些命令将允许我们合并数据和表结构。有了这个，我们可以在本地运行`makemigrations`和`migrate`命令，用示例数据填充我们的数据库，然后将它推送到我们的其他环境，从而无需在每个远程环境中单独运行这些命令。您应该非常小心地执行这些任务，因为执行错误的操作时，完全有可能覆盖或完全丢失数据。好消息是，每当执行这些命令时，Heroku 都会自动生成数据库当前状态的数据备份，让您能够撤销任何已执行的操作。

按照以下步骤执行您的操作：

1.  要直观地看到正在发生的变化，请导航到包含添加或删除表的`push`或`pull`操作。

1.  使用我们从*图 2.12*页面收集的信息，执行以下命令以推送您的数据库，包括一个参数来指示我们正在推送到的环境：

    ```py
    (virtual_env) PS > heroku login
    (virtual_env) PS > PGUSER=postgres PGPASSWORD=password heroku pg:push local_postgresql postgresql-tetrahedral-62413 --app mighty-sea-09431
    ```

确保您使用的是创建本地 PostgreSQL 数据库时指定的用户名和密码。将`postgresql-tetrahedral-62413`替换为`--app`参数标签的内容。`local_postgresql`是我为我的本地数据库使用的名称。

您现在应该能够看到`Auth`和 Django 管理表中有总共 10 个表，这些表是在我们首次在本地运行迁移命令时创建的。

### Windows 用户注意事项

Windows 用户可能难以执行前面的命令。请确保您的 Windows 环境变量设置中包含了`bin`文件夹的路径。该文件夹通常位于 Windows 10 安装的`C:\Program Files\PostgreSQL\##\bin`，其中`##`是您机器上安装的 PostgreSQL 的数字版本号。这将使`heroku pg:`命令和`psql`命令在您的 CLI 中正常工作。在更改环境变量后，通常建议重启您的 CLI 或操作系统。以下命令将在您的 CLI 终端窗口打开期间设置变量。在 Windows 中将它们一起运行在一行上通常会导致错误。当这些变量设置好并且您的窗口保持打开状态时，您将能够根据需要运行您的`push`或`pull`命令：

```py
(virtual_env) PS > heroku login
(virtual_env) PS > set PGHOST=localhost
(virtual_env) PS > set PGUSER=postgres
(virtual_env) PS > set PGPASSWORD=password
(virtual_env) PS > set PGPORT=5432
(virtual_env) PS > heroku pg:push local_postgresql postgresql-tetrahedral-62413 --app mighty-sea-09431
```

如果我们想要拉取数据而不是推送数据，那么这个命令与将`push`改为`pull`一样简单，如下面的示例所示：

```py
(virtual_env) PS > heroku pg:pull local_postgresql postgresql-tetrahedral-62413 --app mighty-sea-09431
```

要查看有关如何使用 Heroku 命令行操作针对 PostgreSQL 的完整指南，请查看他们的知识库文章：[`devcenter.heroku.com/articles/heroku-postgresql`](https://devcenter.heroku.com/articles/heroku-postgresql)。

# 摘要

到目前为止，我们已经做了很多工作，但我们还没有真正开始为项目构建任何 Django 应用。到目前为止所做的一切工作都可以被视为在将项目交给开发团队之前必要的初步工作。已经提供了两种创建项目的方法：一种方法使用一个工具来帮助简化生产，称为 IDE，另一种方法使用终端或命令行窗口中的命令。我们在仓库中跟踪一个解决方案文件，以便我们可以在团队内部共享它，但我们不会跟踪在运行项目时自动创建的个人设置和调试文件。即使与使用 IDE 的人共享项目配置文件，不使用 IDE 的开发者仍然可以与代码库一起工作。在那之后，我们在项目级别和数据库级别配置了 Django 以与主机提供商 Heroku 一起工作。最后，我们激活了允许开发者在本地或远程数据库中查看和编辑数据的工具。

现在，你可以将这个解决方案交给你的团队，并开始将任务委派给该团队中的每个成员。在下一章中，我们将开始创建在数据库内部构建表的模型。这些模型可以被视为在本书中*第三章*，“模型、关系和继承”之后构建其余章节的元素。
