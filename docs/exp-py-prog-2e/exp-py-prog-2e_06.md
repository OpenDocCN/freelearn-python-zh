# 第六章：部署代码

即使完美的代码（如果存在的话）如果不被运行，也是无用的。因此，为了发挥作用，我们的代码需要安装到目标机器（计算机）并执行。将特定版本的应用程序或服务提供给最终用户的过程称为部署。

对于桌面应用程序来说，这似乎很简单——你的工作就是提供一个可下载的包，并在必要时提供可选的安装程序。用户有责任在自己的环境中下载并安装它。你的责任是尽可能地使这个过程简单和方便。适当的打包仍然不是一项简单的任务，但一些工具已经在上一章中进行了解释。

令人惊讶的是，当你的代码不是产品本身时，情况会变得更加复杂。如果你的应用程序只提供向用户出售的服务，那么你有责任在自己的基础设施上运行它。这种情况对于 Web 应用程序或任何“X 作为服务”产品都很典型。在这种情况下，代码被部署到远程机器上，通常开发人员几乎无法物理接触到这些机器。如果你已经是云计算服务的用户，比如亚马逊网络服务（AWS）或 Heroku，这一点尤其真实。

在本章中，我们将集中讨论代码部署到远程主机的方面，因为 Python 在构建各种与网络相关的服务和产品领域非常受欢迎。尽管这种语言具有很高的可移植性，但它没有特定的特性，可以使其代码易于部署。最重要的是你的应用程序是如何构建的，以及你用什么流程将其部署到目标环境中。因此，本章将重点讨论以下主题：

+   部署代码到远程环境的主要挑战是什么

+   如何构建易于部署的 Python 应用程序

+   如何在没有停机的情况下重新加载 Web 服务

+   如何利用 Python 打包生态系统进行代码部署

+   如何正确监控和调试远程运行的代码

# 十二要素应用

无痛部署的主要要求是以确保这个过程简单和尽可能流畅的方式构建你的应用程序。这主要是关于消除障碍和鼓励良好的做法。在只有特定人员负责开发（开发团队或简称为 Dev）的组织中，以及不同的人负责部署和维护执行环境（运维团队或简称为 Ops）的组织中，遵循这样的常见做法尤为重要。

与服务器维护、监控、部署、配置等相关的所有任务通常被放在一个袋子里，称为运维。即使在没有专门的运维团队的组织中，通常也只有一些开发人员被授权执行部署任务和维护远程服务器。这种职位的通用名称是 DevOps。此外，每个开发团队成员都负责运维并不是一种不寻常的情况，因此在这样的团队中，每个人都可以被称为 DevOps。无论你的组织结构如何，每个开发人员都应该知道运维工作以及代码如何部署到远程服务器，因为最终，执行环境及其配置是你正在构建的产品的隐藏部分。

以下的常见做法和约定主要是出于以下原因：

+   在每家公司，员工会离职，新员工会入职。通过使用最佳方法，你可以让新团队成员更容易地加入项目。你永远无法确定新员工是否已经熟悉了系统配置和可靠运行应用程序的常见做法，但你至少可以让他们更有可能快速适应。

+   在只有一些人负责部署的组织中，它简单地减少了运维和开发团队之间的摩擦。

鼓励构建易于部署应用程序的实践的一个很好的来源是一个名为**十二要素应用**的宣言。它是一个通用的、与语言无关的构建软件即服务应用程序的方法论。它的目的之一是使应用程序更容易部署，但它也强调了其他主题，比如可维护性和使应用程序更容易扩展。

正如其名称所示，十二要素应用由 12 条规则组成：

+   **代码库**：一个代码库在版本控制中跟踪，多次部署

+   **依赖关系**：明确声明和隔离依赖关系

+   **配置**：将配置存储在环境中

+   **后端服务**：将后端服务视为附加资源

+   **构建、发布、运行**：严格区分构建和运行阶段

+   **进程**：将应用程序作为一个或多个无状态进程执行

+   **端口绑定**：通过端口绑定导出服务

+   **并发**：通过进程模型进行扩展

+   **可处置性**：通过快速启动和优雅关闭来最大化健壮性

+   **开发/生产一致性**：尽量使开发、演示和生产环境尽可能相似

+   **日志**：将日志视为事件流

+   **管理进程**：将管理任务作为一次性进程运行

在这里扩展每个规则有点无意义，因为十二要素应用方法论的官方页面（[`12factor.net/`](http://12factor.net/)）包含了每个应用要素的广泛原理，以及不同框架和环境的工具示例。

本章试图与上述宣言保持一致，因此我们将在必要时详细讨论其中一些。所呈现的技术和示例有时可能略微偏离这 12 个要素，但请记住，这些规则并非铁板一块。只要能达到目的，它们就是好的。最终，重要的是工作的应用程序（产品），而不是与某种任意方法论兼容。

# 使用 Fabric 进行部署自动化

对于非常小的项目，可能可以手动部署代码，也就是通过远程 shell 手动输入必要的命令序列来安装新版本的代码并在远程 shell 上执行。然而，即使对于一个中等大小的项目，这种方法容易出错，繁琐，并且应该被视为浪费你最宝贵的资源，也就是你自己的时间。

解决这个问题的方法是自动化。一个简单的经验法则是，如果你需要手动执行相同的任务至少两次，你应该自动化它，这样你就不需要第三次手动执行了。有各种工具可以让你自动化不同的事情：

+   远程执行工具如 Fabric 用于按需在多个远程主机上自动执行代码。

+   诸如 Chef、Puppet、CFEngine、Salt 和 Ansible 等配置管理工具旨在自动配置远程主机（执行环境）。它们可以用于设置后端服务（数据库、缓存等）、系统权限、用户等。它们大多也可以用作像 Fabric 这样的远程执行工具，但根据它们的架构，这可能更容易或更困难。

配置管理解决方案是一个复杂的话题，值得单独写一本书。事实上，最简单的远程执行框架具有最低的入门门槛，并且是最受欢迎的选择，至少对于小型项目来说是这样。事实上，每个配置管理工具都提供了一种声明性地指定机器配置的方式，深层内部都实现了远程执行层。

此外，根据某些工具的设计，由于它们的设计，它可能不适合实际的自动化代码部署。一个这样的例子是 Puppet，它确实不鼓励显式运行任何 shell 命令。这就是为什么许多人选择同时使用这两种类型的解决方案来相互补充：配置管理用于设置系统级环境，按需远程执行用于应用程序部署。

Fabric ([`www.fabfile.org/`](http://www.fabfile.org/))到目前为止是 Python 开发人员用来自动化远程执行的最流行的解决方案。它是一个用于简化使用 SSH 进行应用程序部署或系统管理任务的 Python 库和命令行工具。我们将重点关注它，因为它相对容易上手。请注意，根据您的需求，它可能不是解决问题的最佳方案。无论如何，它是一个很好的工具，可以为您的操作添加一些自动化，如果您还没有的话。

### 提示

**Fabric 和 Python 3**

本书鼓励您只在 Python 3 中开发（如果可能的话），并提供有关旧语法特性和兼容性注意事项的注释，只是为了使最终版本切换更加轻松。不幸的是，在撰写本书时，Fabric 仍未正式移植到 Python 3。这个工具的爱好者们被告知至少有几年的时间正在开发 Fabric 2，将带来一个兼容性更新。据说这是一个完全重写，带有许多新功能，但目前还没有 Fabric 2 的官方开放存储库，几乎没有人看到过它的代码。核心 Fabric 开发人员不接受当前项目的 Python 3 兼容性的任何拉取请求，并关闭对其的每个功能请求。这种对流行开源项目的开发方式至少是令人不安的。这个问题的历史并不让我们看到 Fabric 2 的官方发布的机会很高。这种秘密开发新 Fabric 版本的做法引发了许多问题。

不管任何人的观点，这个事实并不会减少 Fabric 在当前状态下的实用性。因此，如果您已经决定坚持使用 Python 3，有两个选择：使用一个完全兼容且独立的分支（[`github.com/mathiasertl/fabric/`](https://github.com/mathiasertl/fabric/)）或者在 Python 3 中编写您的应用程序，并在 Python 2 中维护 Fabric 脚本。最好的方法是在一个单独的代码存储库中进行。

当然，您可以只使用 Bash 脚本来自动化所有工作，但这非常繁琐且容易出错。Python 有更方便的字符串处理方式，并鼓励代码模块化。事实上，Fabric 只是一个通过 SSH 粘合命令执行的工具，因此仍然需要一些关于命令行界面及其实用程序在您的环境中如何工作的知识。

使用 Fabric 开始工作，您需要安装`fabric`包（使用`pip`），并创建一个名为`fabfile.py`的脚本，通常位于项目的根目录中。请注意，`fabfile`可以被视为项目配置的一部分。因此，如果您想严格遵循十二要素应用程序方法论，您不应该在部署的应用程序源树中维护其代码。事实上，复杂的项目通常是由维护为单独代码库的各种组件构建而成，因此，将所有项目组件配置和 Fabric 脚本放在一个单独的存储库中是一个很好的方法。这样可以使不同服务的部署更加一致，并鼓励良好的代码重用。

一个定义了简单部署过程的示例`fabfile`将如下所示：

```py
# -*- coding: utf-8 -*-
import os

from fabric.api import *  # noqa
from fabric.contrib.files import exists

# Let's assume we have private package repository created
# using 'devpi' project
PYPI_URL = 'http://devpi.webxample.example.com'

# This is arbitrary location for storing installed releases.
# Each release is a separate virtual environment directory
# which is named after project version. There is also a
# symbolic link 'current' that points to recently deployed
# version. This symlink is an actual path that will be used
# for configuring the process supervision tool e.g.:
# .
# ├── 0.0.1
# ├── 0.0.2
# ├── 0.0.3
# ├── 0.1.0
# └── current -> 0.1.0/

REMOTE_PROJECT_LOCATION = "/var/projects/webxample"

env.project_location = REMOTE_PROJECT_LOCATION

# roledefs map out environment types (staging/production)
env.roledefs = {
    'staging': [
        'staging.webxample.example.com',
    ],
    'production': [
        'prod1.webxample.example.com',
        'prod2.webxample.example.com',
    ],
}

def prepare_release():
    """ Prepare a new release by creating source distribution and uploading to out private package repository
    """
    local('python setup.py build sdist upload -r {}'.format(
        PYPI_URL
    ))

def get_version():
    """ Get current project version from setuptools """
    return local(
        'python setup.py --version', capture=True
    ).stdout.strip()

def switch_versions(version):
    """ Switch versions by replacing symlinks atomically """
    new_version_path = os.path.join(REMOTE_PROJECT_LOCATION, version)
    temporary = os.path.join(REMOTE_PROJECT_LOCATION, 'next')
    desired = os.path.join(REMOTE_PROJECT_LOCATION, 'current')

    # force symlink (-f) since probably there is a one already
    run(
        "ln -fsT {target} {symlink}"
        "".format(target=new_version_path, symlink=temporary)
    )
    # mv -T ensures atomicity of this operation
    run("mv -Tf {source} {destination}"
        "".format(source=temporary, destination=desired))

@task
def uptime():
    """
    Run uptime command on remote host - for testing connection.
    """
    run("uptime")

@task
def deploy():
    """ Deploy application with packaging in mind """
    version = get_version()
    pip_path = os.path.join(
        REMOTE_PROJECT_LOCATION, version, 'bin', 'pip'
    )

    prepare_release()

    if not exists(REMOTE_PROJECT_LOCATION):
        # it may not exist for initial deployment on fresh host
        run("mkdir -p {}".format(REMOTE_PROJECT_LOCATION))

    with cd(REMOTE_PROJECT_LOCATION):
        # create new virtual environment using venv
        run('python3 -m venv {}'.format(version))

        run("{} install webxample=={} --index-url {}".format(
            pip_path, version, PYPI_URL
        ))

    switch_versions(version)
    # let's assume that Circus is our process supervision tool
    # of choice.
    run('circusctl restart webxample')
```

每个使用`@task`装饰的函数都被视为`fabric`包提供的`fab`实用程序的可用子命令。您可以使用`-l`或`--list`开关列出所有可用的子命令：

```py
$ fab --list
Available commands:

 **deploy  Deploy application with packaging in mind
 **uptime  Run uptime command on remote host - for testing connection.

```

现在，您可以只需一个 shell 命令将应用程序部署到给定的环境类型：

```py
$ fab –R production deploy

```

请注意，前面的`fabfile`仅用于举例说明。在您自己的代码中，您可能希望提供全面的故障处理，并尝试重新加载应用程序，而无需重新启动 Web 工作进程。此外，此处介绍的一些技术现在可能很明显，但稍后将在本章中进行解释。这些是：

+   使用私有软件包存储库部署应用程序

+   在远程主机上使用 Circus 进行进程监控

# 您自己的软件包索引或索引镜像

有三个主要原因您可能希望运行自己的 Python 软件包索引：

+   官方 Python 软件包索引没有任何可用性保证。它由 Python 软件基金会运行，感谢众多捐赠。因此，这往往意味着该站点可能会关闭。您不希望由于 PyPI 中断而在中途停止部署或打包过程。

+   即使对于永远不会公开发布的闭源代码，也有用处，因为它可以使用 Python 编写的可重用组件得到适当打包。这简化了代码库，因为公司中用于不同项目的软件包不需要被打包。您可以直接从存储库安装它们。这简化了这些共享代码的维护，并且如果公司有许多团队在不同项目上工作，可能会减少整个公司的开发成本。

+   使用`setuptools`对整个项目进行打包是非常好的做法。然后，部署新应用程序版本通常只需运行`pip install --update my-application`。

### 提示

**代码打包**

代码打包是将外部软件包的源代码包含在其他项目的源代码（存储库）中的做法。当项目的代码依赖于某个特定版本的外部软件包时，通常会这样做，该软件包也可能被其他软件包（以完全不同的版本）所需。例如，流行的`requests`软件包在其源代码树中打包了`urllib3`的某个版本，因为它与之紧密耦合，并且几乎不太可能与`urllib3`的其他版本一起使用。一些特别经常被其他人打包的模块的例子是`six`。它可以在许多流行项目的源代码中找到，例如 Django（`django.utils.six`），Boto（`boto.vedored.six`）或 Matplotlib（`matplotlib.externals.six`）。

尽管一些大型和成功的开源项目甚至也会使用打包，但如果可能的话应该避免。这只在某些情况下才有合理的用途，并且不应被视为软件包依赖管理的替代品。

## PyPI 镜像

PyPI 中断的问题可以通过允许安装工具从其镜像之一下载软件包来在一定程度上得到缓解。事实上，官方 Python 软件包索引已经通过**CDN**（**内容传送网络**）提供服务，因此它本质上是镜像的。这并不改变这样的事实，即它似乎偶尔会出现一些糟糕的日子，当任何尝试下载软件包失败时。在这里使用非官方镜像不是一个解决方案，因为这可能会引发一些安全顾虑。

最好的解决方案是拥有自己的 PyPI 镜像，其中包含您需要的所有软件包。唯一使用它的一方是您自己，因此更容易确保适当的可用性。另一个优势是，每当此服务关闭时，您无需依赖其他人来重新启动它。PyPA 维护和推荐的镜像工具是**bandersnatch**（[`pypi.python.org/pypi/bandersnatch`](https://pypi.python.org/pypi/bandersnatch)）。它允许您镜像 Python Package Index 的全部内容，并且可以作为`.pypirc`文件中存储库部分的`index-url`选项提供（如前一章中所述）。此镜像不接受上传，也没有 PyPI 的 Web 部分。无论如何，要小心！完整的镜像可能需要数百千兆字节的存储空间，并且其大小将随着时间的推移而继续增长。

但是，为什么要停留在一个简单的镜像上，而我们有一个更好的选择呢？您几乎不太可能需要整个软件包索引的镜像。即使是具有数百个依赖项的项目，它也只是所有可用软件包的一小部分。此外，无法上传自己的私有软件包是这种简单镜像的巨大局限性。似乎使用 bandersnatch 的附加价值与其高昂的价格相比非常低。在大多数情况下，这是正确的。如果软件包镜像仅用于单个或少数项目，那么使用**devpi**（[`doc.devpi.net/`](http://doc.devpi.net/)）将是一个更好的方法。它是一个与 PyPI 兼容的软件包索引实现，提供以下两种功能：

+   上传非公共软件包的私有索引

+   索引镜像

devpi 相对于 bandersnatch 的主要优势在于它如何处理镜像。它当然可以像 bandersnatch 一样对其他索引进行完整的通用镜像，但这不是它的默认行为。它不是对整个存储库进行昂贵的备份，而是为已被客户端请求的软件包维护镜像。因此，每当安装工具（`pip`、`setuptools`和`easyinstall`）请求软件包时，如果在本地镜像中不存在，devpi 服务器将尝试从镜像索引（通常是 PyPI）下载并提供。软件包下载后，devpi 将定期检查其更新，以保持镜像的新鲜状态。

镜像方法在您请求尚未被镜像的新软件包并且上游软件包索引中断时留下了轻微的失败风险。无论如何，由于在大多数部署中，您将仅依赖于已在索引中镜像的软件包，因此这种风险得到了减少。对于已经请求的软件包，镜像状态与 PyPI 具有最终一致性，并且新版本将自动下载。这似乎是一个非常合理的权衡。

## 使用软件包进行部署

现代 Web 应用程序有很多依赖项，并且通常需要许多步骤才能在远程主机上正确安装。例如，对于远程主机上的应用程序的新版本的典型引导过程包括以下步骤：

+   为隔离创建新的虚拟环境

+   将项目代码移动到执行环境

+   安装最新的项目要求（通常来自`requirements.txt`文件）

+   同步或迁移数据库架构

+   从项目源和外部软件包收集静态文件到所需位置

+   为可用于不同语言的应用程序编译本地化文件

对于更复杂的网站，可能会有许多与前端代码相关的附加任务：

+   使用预处理器（如 SASS 或 LESS）生成 CSS 文件

+   对静态文件（JavaScript 和 CSS 文件）进行缩小、混淆和/或合并

+   编译用 JavaScript 超集语言（CoffeeScript、TypeScript 等）编写的代码到本机 JS

+   预处理响应模板文件（缩小、内联样式等）

所有这些步骤都可以使用诸如 Bash、Fabric 或 Ansible 之类的工具轻松自动化，但在安装应用程序的远程主机上做所有事情并不是一个好主意。原因如下：

+   一些用于处理静态资产的流行工具可能是 CPU 密集型或内存密集型。在生产环境中运行它们可能会使应用程序执行不稳定。

+   这些工具通常需要额外的系统依赖项，这些依赖项可能不是项目的正常运行所必需的。这些主要是额外的运行时环境，如 JVM、Node 或 Ruby。这增加了配置管理的复杂性，并增加了整体维护成本。

+   如果您将应用程序部署到多个服务器（十个、百个、千个），那么您只是在重复很多工作，这些工作本来可以只做一次。如果您有自己的基础设施，那么您可能不会经历巨大的成本增加，特别是如果您在低流量时段进行部署。但如果您在计费模型中运行云计算服务，该模型会额外收费用于负载峰值或一般执行时间，那么这些额外成本可能在适当的规模上是相当可观的。

+   大多数这些步骤只是花费了很多时间。您正在将代码安装到远程服务器上，所以您最不希望的是在部署过程中由于某些网络问题而中断连接。通过保持部署过程快速，您可以降低部署中断的几率。

出于明显的原因，上述部署步骤的结果不能包含在应用程序代码存储库中。简单地说，有些事情必须在每个发布中完成，你无法改变这一点。显然这是一个适当自动化的地方，但关键是在正确的地方和正确的时间做。

大部分静态收集和代码/资产预处理等工作可以在本地或专用环境中完成，因此部署到远程服务器的实际代码只需要进行最少量的现场处理。在构建分发或安装包的过程中，最显著的部署步骤是：

+   安装 Python 依赖项和传输静态资产（CSS 文件和 JavaScript）到所需位置可以作为`setup.py`脚本的`install`命令的一部分来处理

+   预处理代码（处理 JavaScript 超集、资产的缩小/混淆/合并，以及运行 SASS 或 LESS）和诸如本地化文本编译（例如 Django 中的`compilemessages`）等工作可以作为`setup.py`脚本的`sdist`/`bdist`命令的一部分

包括除 Python 以外的预处理代码可以很容易地通过适当的`MANIFEST.in`文件处理。依赖项当然最好作为`setuptools`包的`setup()`函数调用的`install_requires`参数提供。

当然，打包整个应用程序将需要您进行一些额外的工作，比如提供自己的自定义`setuptools`命令或覆盖现有的命令，但这将为您带来许多优势，并使项目部署更快速和可靠。

让我们以一个基于 Django 的项目（在 Django 1.9 版本中）为例。我选择这个框架是因为它似乎是这种类型的最受欢迎的 Python 项目，所以你很有可能已经对它有所了解。这样的项目中文件的典型结构可能如下所示：

```py
$ tree . -I __pycache__ --dirsfirst
.
├── webxample
│   ├── conf
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── locale
│   │   ├── de
│   │   │   └── LC_MESSAGES
│   │   │       └── django.po
│   │   ├── en
│   │   │   └── LC_MESSAGES
│   │   │       └── django.po
│   │   └── pl
│   │       └── LC_MESSAGES
│   │           └── django.po
│   ├── myapp
│   │   ├── migrations
│   │   │   └── __init__.py
│   │   ├── static
│   │   │   ├── js
│   │   │   │   └── myapp.js
│   │   │   └── sass
│   │   │       └── myapp.scss
│   │   ├── templates
│   │   │   ├── index.html
│   │   │   └── some_view.html
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── tests.py
│   │   └── views.py
│   ├── __init__.py
│   └── manage.py
├── MANIFEST.in
├── README.md
└── setup.py

15 directories, 23 files

```

请注意，这与通常的 Django 项目模板略有不同。默认情况下，包含 WSGI 应用程序、设置模块和 URL 配置的包与项目名称相同。因为我们决定采用打包的方法，这将被命名为`webxample`。这可能会引起一些混淆，所以最好将其重命名为`conf`。

不要深入可能的实现细节，让我们只做一些简单的假设：

+   我们的示例应用程序有一些外部依赖。在这里，将是两个流行的 Django 软件包：`djangorestframework` 和 `django-allauth`，以及一个非 Django 软件包：`gunicorn`。

+   `djangorestframework` 和 `django-allauth` 被提供为 `webexample.webexample.settings` 模块中的 `INSTALLED_APPS`。

+   该应用程序在三种语言（德语、英语和波兰语）中进行了本地化，但我们不希望将编译的 `gettext` 消息存储在存储库中。

+   我们厌倦了普通的 CSS 语法，所以我们决定使用更强大的 SCSS 语言，我们使用 SASS 将其转换为 CSS。

了解项目的结构后，我们可以编写我们的 `setup.py` 脚本，使 `setuptools` 处理：

+   在 `webxample/myapp/static/scss` 下编译 SCSS 文件

+   从 `.po` 格式编译 `webexample/locale` 下的 `gettext` 消息到 `.mo` 格式

+   安装要求

+   提供软件包的入口点的新脚本，这样我们将有自定义命令而不是 `manage.py` 脚本

我们在这里有点运气。 `libsass` 的 Python 绑定是 SASS 引擎的 C/C++端口，它与 `setuptools` 和 `distutils` 提供了很好的集成。只需进行少量配置，它就可以为运行 SASS 编译提供自定义的 `setup.py` 命令：

```py
from setuptools import setup

setup(
    name='webxample',
    setup_requires=['libsass >= 0.6.0'],
    sass_manifests={
        'webxample.myapp': ('static/sass', 'static/css')
    },
)
```

因此，我们可以通过键入 `python setup.py build_scss` 来将我们的 SCSS 文件编译为 CSS，而不是手动运行 `sass` 命令或在 `setup.py` 脚本中执行子进程。这还不够。这让我们的生活变得更容易，但我们希望整个分发过程完全自动化，因此只需一个步骤即可创建新版本。为了实现这个目标，我们不得不稍微覆盖一些现有的 `setuptools` 分发命令。

处理一些项目准备步骤的 `setup.py` 文件示例可能如下所示：

```py
import os

from setuptools import setup
from setuptools import find_packages
from distutils.cmd import Command
from distutils.command.build import build as _build

try:
    from django.core.management.commands.compilemessages \
        import Command as CompileCommand
except ImportError:
    # note: during installation django may not be available
    CompileCommand = None

# this environment is requires
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "webxample.conf.settings"
)

class build_messages(Command):
    """ Custom command for building gettext messages in Django
    """
    description = """compile gettext messages"""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):

        pass

    def run(self):
        if CompileCommand:
            CompileCommand().handle(
                verbosity=2, locales=[], exclude=[]
            )
        else:
            raise RuntimeError("could not build translations")

class build(_build):
    """ Overriden build command that adds additional build steps
    """
    sub_commands = [
        ('build_messages', None),
        ('build_sass', None),
    ] + _build.sub_commands

setup(
    name='webxample',
    setup_requires=[
        'libsass >= 0.6.0',
        'django >= 1.9.2',
    ],
    install_requires=[
        'django >= 1.9.2',
        'gunicorn == 19.4.5',
        'djangorestframework == 3.3.2',
        'django-allauth == 0.24.1',
    ],
    packages=find_packages('.'),
    sass_manifests={
        'webxample.myapp': ('static/sass', 'static/css')
    },
    cmdclass={
        'build_messages': build_messages,
        'build': build,
    },
    entry_points={
        'console_scripts': {
            'webxample = webxample.manage:main',
        }
    }
)
```

通过这种实现，我们可以使用这个单一的终端命令构建所有资产并为 `webxample` 项目创建源分发的软件包：

```py
$ python setup.py build sdist

```

如果您已经拥有自己的软件包索引（使用 `devpi` 创建），则可以添加 `install` 子命令或使用 `twine`，这样该软件包将可以在您的组织中使用 `pip` 进行安装。如果我们查看使用我们的 `setup.py` 脚本创建的源分发结构，我们可以看到它包含了从 SCSS 文件生成的编译的 `gettext` 消息和 CSS 样式表：

```py
$ tar -xvzf dist/webxample-0.0.0.tar.gz 2> /dev/null
$ tree webxample-0.0.0/ -I __pycache__ --dirsfirst
webxample-0.0.0/
├── webxample
│   ├── conf
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── locale
│   │   ├── de
│   │   │   └── LC_MESSAGES
│   │   │       ├── django.mo
│   │   │       └── django.po
│   │   ├── en
│   │   │   └── LC_MESSAGES
│   │   │       ├── django.mo
│   │   │       └── django.po
│   │   └── pl
│   │       └── LC_MESSAGES
│   │           ├── django.mo
│   │           └── django.po
│   ├── myapp
│   │   ├── migrations
│   │   │   └── __init__.py
│   │   ├── static
│   │   │   ├── css
│   │   │   │   └── myapp.scss.css
│   │   │   └── js
│   │   │       └── myapp.js
│   │   ├── templates
│   │   │   ├── index.html
│   │   │   └── some_view.html
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── tests.py
│   │   └── views.py
│   ├── __init__.py
│   └── manage.py
├── webxample.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   └── top_level.txt
├── MANIFEST.in
├── PKG-INFO
├── README.md
├── setup.cfg
└── setup.py

16 directories, 33 files

```

使用这种方法的额外好处是，我们能够在 Django 的默认 `manage.py` 脚本的位置提供我们自己的项目入口点。现在我们可以使用这个入口点运行任何 Django 管理命令，例如：

```py
$ webxample migrate
$ webxample collectstatic
$ webxample runserver

```

这需要在 `manage.py` 脚本中进行一些小的更改，以便与 `setup()` 中的 `entry_points` 参数兼容，因此它的主要部分的代码被包装在 `main()` 函数调用中：

```py
#!/usr/bin/env python3
import os
import sys

def main():
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE", "webxample.conf.settings"
    )

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)

if __name__ == "__main__":
    main()
```

不幸的是，许多框架（包括 Django）并不是以打包项目的方式设计的。这意味着根据应用程序的进展，将其转换为包可能需要进行许多更改。在 Django 中，这通常意味着重写许多隐式导入并更新设置文件中的许多配置变量。

另一个问题是使用 Python 打包创建的发布的一致性。如果不同的团队成员被授权创建应用程序分发，那么在相同可复制的环境中进行此过程至关重要，特别是当您进行大量资产预处理时；即使从相同的代码库创建，可能在两个不同的环境中创建的软件包看起来也不一样。这可能是由于在构建过程中使用了不同版本的工具。最佳实践是将分发责任移交给持续集成/交付系统，如 Jenkins 或 Buildbot。额外的好处是您可以断言软件包在分发之前通过了所有必需的测试。您甚至可以将自动部署作为这种持续交付系统的一部分。

尽管如此，使用`setuptools`将您的代码分发为 Python 软件包并不简单和轻松；它将极大简化您的部署，因此绝对值得一试。请注意，这也符合十二要素应用程序的第六条详细建议：将应用程序执行为一个或多个无状态进程（[`12factor.net/processes`](http://12factor.net/processes)）。

# 常见的惯例和做法

有一套部署的常见惯例和做法，不是每个开发人员都可能知道，但对于任何曾经进行过一些操作的人来说都是显而易见的。正如在章节介绍中所解释的那样，即使您不负责代码部署和操作，至少了解其中一些对于在开发过程中做出更好的设计决策是至关重要的。

## 文件系统层次结构

您可能会想到的最明显的惯例可能是关于文件系统层次结构和用户命名的。如果您在这里寻找建议，那么您会感到失望。当然有一个**文件系统层次结构标准**，它定义了 Unix 和类 Unix 操作系统中的目录结构和目录内容，但真的很难找到一个完全符合 FHS 的实际操作系统发行版。如果系统设计师和程序员不能遵守这样的标准，那么很难期望管理员也能做到。根据我的经验，我几乎在可能的任何地方看到应用程序代码部署，包括在根文件系统级别的非标准自定义目录。几乎总是，做出这样决定的人都有非常充分的理由。在这方面我能给你的唯一建议是：

+   明智选择，避免惊喜

+   在项目的所有可用基础设施中保持一致

+   尽量在您的组织（您所在的公司）中保持一致

真正有帮助的是为您的项目记录惯例。只需确保这些文件对每个感兴趣的团队成员都是可访问的，并且每个人都知道这样的文件存在。

## 隔离

隔离的原因以及推荐的工具已经在第一章中讨论过，*Python 的当前状态*。对于部署，只有一件重要的事情要补充。您应该始终为应用程序的每个发布版本隔离项目依赖关系。在实践中，这意味着每当您部署应用程序的新版本时，您应该为此版本创建一个新的隔离环境（使用`virtualenv`或`venv`）。旧环境应该在您的主机上保留一段时间，以便在出现问题时可以轻松地回滚到旧版本之一。

为每个发布创建新的环境有助于管理其干净状态并符合提供的依赖项列表。通过新环境，我们指的是在文件系统中创建一个新的目录树，而不是更新已经存在的文件。不幸的是，这可能会使一些事情变得更加困难，比如优雅地重新加载服务，如果环境是就地更新的话，这将更容易实现。

## 使用进程监控工具

远程服务器上的应用程序通常不会意外退出。如果是 Web 应用程序，其 HTTP 服务器进程将无限期地等待新的连接和请求，并且只有在发生一些无法恢复的错误时才会退出。

当然，无法在 shell 中手动运行它并保持一个永久的 SSH 连接。使用`nohup`、`screen`或`tmux`来半守护化进程也不是一个选择。这样做就像是在设计您的服务注定要失败。

您需要的是一些进程监控工具，可以启动和管理您的应用程序进程。在选择合适的工具之前，您需要确保它：

+   如果服务退出，则重新启动服务

+   可靠地跟踪其状态

+   捕获其`stdout`/`stderr`流以进行日志记录

+   以特定用户/组权限运行进程

+   配置系统环境变量

大多数 Unix 和 Linux 发行版都有一些内置的进程监控工具/子系统，比如`initd`脚本、`upstart`和`runit`。不幸的是，在大多数情况下，它们不适合运行用户级应用程序代码，并且非常难以维护。特别是编写可靠的`init.d`脚本是一个真正的挑战，因为它需要大量的 Bash 脚本编写，这很难做到正确。一些 Linux 发行版，比如 Gentoo，对`init.d`脚本有了重新设计的方法，因此编写它们变得更容易。无论如何，为了一个单一的进程监控工具而将自己锁定到特定的操作系统发行版并不是一个好主意。

Python 社区中管理应用程序进程的两种流行工具是 Supervisor ([`supervisord.org`](http://supervisord.org))和 Circus ([`circus.readthedocs.org/en/latest/`](https://circus.readthedocs.org/en/latest/))。它们在配置和使用上都非常相似。Circus 比 Supervisor 稍微年轻一些，因为它是为了解决后者的一些弱点而创建的。它们都可以使用简单的 INI 格式进行配置。它们不仅限于运行 Python 进程，还可以配置为管理任何应用程序。很难说哪一个更好，因为它们都提供非常相似的功能。

无论如何，Supervisor 不支持 Python 3，因此我们不会推荐它。虽然在 Supervisor 的控制下运行 Python 3 进程不是问题，但我将以此为借口，只展示 Circus 配置的示例。

假设我们想要在 Circus 控制下使用`gunicorn` web 服务器运行 webxample 应用程序（在本章前面介绍过）。在生产环境中，我们可能会在适用的系统级进程监控工具（`initd`、`upstart`和`runit`）下运行 Circus，特别是如果它是从系统软件包存储库安装的。为了简单起见，我们将在虚拟环境内本地运行。允许我们在 Circus 中运行应用程序的最小配置文件（这里命名为`circus.ini`）如下所示：

```py
[watcher:webxample]
cmd = /path/to/venv/dir/bin/gunicorn webxample.conf.wsgi:application
numprocesses = 1
```

现在，`circus`进程可以使用这个配置文件作为执行参数来运行：

```py
$ circusd circus.ini
2016-02-15 08:34:34 circus[1776] [INFO] Starting master on pid 1776
2016-02-15 08:34:34 circus[1776] [INFO] Arbiter now waiting for commands
2016-02-15 08:34:34 circus[1776] [INFO] webxample started
[2016-02-15 08:34:34 +0100] [1778] [INFO] Starting gunicorn 19.4.5
[2016-02-15 08:34:34 +0100] [1778] [INFO] Listening at: http://127.0.0.1:8000 (1778)
[2016-02-15 08:34:34 +0100] [1778] [INFO] Using worker: sync
[2016-02-15 08:34:34 +0100] [1781] [INFO] Booting worker with pid: 1781

```

现在，您可以使用`circusctl`命令来运行一个交互式会话，并使用简单的命令来控制所有受管进程。以下是这样一个会话的示例：

```py
$ circusctl
circusctl 0.13.0
webxample: active
(circusctl) stop webxample
ok
(circusctl) status
webxample: stopped
(circusctl) start webxample
ok
(circusctl) status
webxample: active

```

当然，上述两种工具都有更多功能可用。它们的所有功能都在它们的文档中有解释，因此在做出选择之前，您应该仔细阅读它们。

## 应用代码应该在用户空间中运行

您的应用程序代码应始终在用户空间中运行。这意味着它不得以超级用户权限执行。如果您按照 Twelve-Factor App 设计应用程序，可以在几乎没有特权的用户下运行应用程序。拥有文件并且不属于特权组的用户的传统名称是`nobody`，但实际建议是为每个应用程序守护进程创建一个单独的用户。原因是系统安全性。这是为了限制恶意用户在控制应用程序进程后可能造成的损害。在 Linux 中，同一用户的进程可以相互交互，因此在用户级别上将不同的应用程序分开是很重要的。

## 使用反向 HTTP 代理

多个 Python 符合 WSGI 标准的 Web 服务器可以轻松地自行提供 HTTP 流量，无需在其上方使用任何其他 Web 服务器。然而，通常还是很常见将它们隐藏在 Nginx 等反向代理后面，原因有很多：

+   TLS/SSL 终止通常最好由顶级 Web 服务器（如 Nginx 和 Apache）处理。然后，Python 应用程序只能使用简单的 HTTP 协议（而不是 HTTPS），因此安全通信通道的复杂性和配置留给了反向代理。

+   非特权用户无法绑定低端口（0-1000 范围内），但 HTTP 协议应该在端口 80 上为用户提供服务，HTTPS 应该在端口 443 上提供服务。为此，必须以超级用户权限运行进程。通常，更安全的做法是让应用程序在高端口上提供服务，或者在 Unix 域套接字上提供服务，并将其用作在更特权用户下运行的反向代理的上游。

+   通常，Nginx 可以比 Python 代码更有效地提供静态资产（图像、JS、CSS 和其他媒体）。如果将其配置为反向代理，那么只需几行配置就可以通过它提供静态文件。

+   当单个主机需要从不同域中的多个应用程序提供服务时，Apache 或 Nginx 是不可或缺的，用于为在同一端口上提供服务的不同域创建虚拟主机。

+   反向代理可以通过添加额外的缓存层来提高性能，也可以配置为简单的负载均衡器。

一些 Web 服务器实际上建议在代理后运行，例如 Nginx。例如，`gunicorn`是一个非常强大的基于 WSGI 的服务器，如果其客户端速度很快，可以提供出色的性能结果。另一方面，它不能很好地处理慢速客户端，因此很容易受到基于慢速客户端连接的拒绝服务攻击的影响。使用能够缓冲慢速客户端的代理服务器是解决这个问题的最佳方法。

## 优雅地重新加载进程

Twelve-Factor App 方法论的第九条规则涉及进程的可处置性，并指出您应该通过快速启动时间和优雅的关闭来最大程度地提高鲁棒性。虽然快速启动时间相当不言自明，但优雅的关闭需要一些额外的讨论。

在 Web 应用程序范围内，如果以非优雅的方式终止服务器进程，它将立即退出，没有时间完成处理请求并向连接的客户端回复适当的响应。在最佳情况下，如果使用某种反向代理，那么代理可能会向连接的客户端回复一些通用的错误响应（例如 502 Bad Gateway），即使这并不是通知用户您已重新启动应用程序并部署新版本的正确方式。

根据 Twelve-Factor App，Web 服务器进程应能够在接收到 Unix `SIGTERM`信号（例如`kill -TERM <process-id>`）时优雅地退出。这意味着服务器应停止接受新连接，完成处理所有挂起的请求，然后在没有其他事情可做时以某种退出代码退出。

显然，当所有服务进程退出或开始其关闭过程时，您将无法再处理新请求。这意味着您的服务仍然会经历停机，因此您需要执行额外的步骤-启动新的工作进程，这些工作进程将能够在旧的工作进程优雅退出时接受新的连接。各种 Python WSGI 兼容的 Web 服务器实现允许在没有任何停机时间的情况下优雅地重新加载服务。最流行的是 Gunicorn 和 uWSGI：

+   Gunicorn 的主进程在接收到`SIGHUP`信号（`kill -HUP <process-pid>`）后，将启动新的工作进程（带有新的代码和配置），并尝试在旧的工作进程上进行优雅的关闭。

+   uWSGI 至少有三种独立的方案来进行优雅的重新加载。每一种都太复杂，无法简要解释，但它的官方文档提供了所有可能选项的完整信息。

优雅的重新加载在部署 Web 应用程序中已经成为标准。Gunicorn 似乎有一种最容易使用但也给您留下最少灵活性的方法。另一方面，uWSGI 中的优雅重新加载允许更好地控制重新加载，但需要更多的努力来自动化和设置。此外，您如何处理自动部署中的优雅重新加载也受到您使用的监视工具以及其配置方式的影响。例如，在 Gunicorn 中，优雅的重新加载就像这样简单：

```py
kill -HUP <gunicorn-master-process-pid>

```

但是，如果您想通过为每个发布分离虚拟环境并使用符号链接配置进程监视来正确隔离项目分发（如之前在`fabfile`示例中提出的），您很快会注意到这并不像预期的那样工作。对于更复杂的部署，目前还没有可用的解决方案可以直接为您解决问题。您总是需要进行一些黑客攻击，有时这将需要对低级系统实现细节有相当高的了解。

# 代码仪器和监控

我们的工作并不仅仅是编写应用程序并将其部署到目标执行环境。可能编写一个应用程序后，部署后将不需要任何进一步的维护，尽管这是非常不太可能的。实际上，我们需要确保它被正确地观察以发现错误和性能问题。

为了确保我们的产品按预期工作，我们需要正确处理应用程序日志并监视必要的应用程序指标。这通常包括：

+   监控 Web 应用程序访问日志以获取各种 HTTP 状态代码

+   可能包含有关运行时错误和各种警告的进程日志的收集

+   监控远程主机上的系统资源（CPU 负载、内存和网络流量），应用程序运行的地方

+   监控业务绩效和指标的应用级性能（客户获取、收入等）

幸运的是，有很多免费的工具可用于仪器化您的代码并监视其性能。其中大多数都很容易集成。

## 记录错误-哨兵/乌鸦

无论您的应用程序经过多么精确的测试，事实是痛苦的。您的代码最终会在某个时候失败。这可能是任何事情-意外的异常、资源耗尽、某些后台服务崩溃、网络中断，或者只是外部库中的问题。一些可能的问题，如资源耗尽，可以通过适当的监控来预测和防止，但无论您如何努力，总会有一些事情会越过您的防线。

您可以做的是为这种情况做好充分准备，并确保没有错误被忽视。在大多数情况下，应用程序引发的任何意外故障场景都会导致异常，并通过日志系统记录。这可以是`stdout`、`sderr`、“文件”或您为日志记录配置的任何输出。根据您的实现，这可能会导致应用程序退出并带有一些系统退出代码，也可能不会。

当然，您可以仅依赖于存储在文件中的这些日志来查找和监视应用程序错误。不幸的是，观察文本日志中的错误非常痛苦，并且在除了在开发中运行代码之外的任何更复杂的情况下都无法很好地扩展。您最终将被迫使用一些专为日志收集和分析而设计的服务。适当的日志处理对于稍后将要解释的其他原因非常重要，但对于跟踪和调试生产错误并不起作用。原因很简单。错误日志的最常见形式只是 Python 堆栈跟踪。如果您仅停留在那里，您很快就会意识到这不足以找到问题的根本原因-特别是当错误以未知模式或在某些负载条件下发生时。

您真正需要的是尽可能多的关于错误发生的上下文信息。拥有在生产环境中发生的错误的完整历史记录，并且可以以某种便捷的方式浏览和搜索，也非常有用。提供这种功能的最常见工具之一是 Sentry（[`getsentry.com`](https://getsentry.com)）。它是一个经过实战考验的用于跟踪异常和收集崩溃报告的服务。它作为开源软件提供，是用 Python 编写的，并起源于用于后端 Web 开发人员的工具。现在它已经超出了最初的野心，并支持了许多其他语言，包括 PHP、Ruby 和 JavaScript，但仍然是大多数 Python Web 开发人员的首选工具。

### 提示

**Web 应用程序中的异常堆栈跟踪**

通常，Web 应用程序不会在未处理的异常上退出，因为 HTTP 服务器有义务在发生任何服务器错误时返回一个 5XX 组的状态代码的错误响应。大多数 Python Web 框架默认情况下都会这样做。在这种情况下，实际上是在较低的框架级别处理异常。无论如何，在大多数情况下，这仍将导致异常堆栈跟踪被打印（通常在标准输出上）。

Sentry 以付费软件即服务模式提供，但它是开源的，因此可以免费托管在您自己的基础设施上。提供与 Sentry 集成的库是`raven`（可在 PyPI 上获得）。如果您尚未使用过它，想要测试它但无法访问自己的 Sentry 服务器，那么您可以轻松在 Sentry 的本地服务站点上注册免费试用。一旦您可以访问 Sentry 服务器并创建了一个新项目，您将获得一个称为 DSN 或数据源名称的字符串。这个 DSN 字符串是集成应用程序与 sentry 所需的最小配置设置。它以以下形式包含协议、凭据、服务器位置和您的组织/项目标识符：

```py
'{PROTOCOL}://{PUBLIC_KEY}:{SECRET_KEY}@{HOST}/{PATH}{PROJECT_ID}'
```

一旦您获得了 DSN，集成就非常简单：

```py
from raven import Client

client = Client('https://<key>:<secret>@app.getsentry.com/<project>')

try:
    1 / 0
except ZeroDivisionError:
    client.captureException()
```

Raven 与最流行的 Python 框架（如 Django，Flask，Celery 和 Pyramid）有许多集成，以使集成更容易。这些集成将自动提供特定于给定框架的附加上下文。如果您选择的 Web 框架没有专门的支持，`raven`软件包提供了通用的 WSGI 中间件，使其与任何基于 WSGI 的 Web 服务器兼容：

```py
from raven import Client
from raven.middleware import Sentry

# note: application is some WSGI application object defined earlier
application = Sentry(
    application,
    Client('https://<key>:<secret>@app.getsentry.com/<project>')
)
```

另一个值得注意的集成是跟踪通过 Python 内置的`logging`模块记录的消息的能力。启用此类支持仅需要几行额外的代码：

```py
from raven.handlers.logging import SentryHandler
from raven.conf import setup_logging

client = Client('https://<key>:<secret>@app.getsentry.com/<project>')
handler = SentryHandler(client)
setup_logging(handler)
```

捕获`logging`消息可能会有一些不明显的注意事项，因此，如果您对此功能感兴趣，请确保阅读官方文档。这应该可以避免令人不快的惊喜。

最后一点是关于运行自己的 Sentry 以节省一些钱的方法。 "没有免费的午餐。"最终，您将支付额外的基础设施成本，而 Sentry 将只是另一个需要维护的服务。*维护=额外工作=成本*！随着您的应用程序增长，异常的数量也会增长，因此您将被迫在扩展产品的同时扩展 Sentry。幸运的是，这是一个非常强大的项目，但如果负载过重，它将无法为您提供任何价值。此外，保持 Sentry 准备好应对灾难性故障场景，其中可能会发送数千个崩溃报告，是一个真正的挑战。因此，您必须决定哪个选项对您来说真正更便宜，以及您是否有足够的资源和智慧来自己完成所有这些。当然，如果您的组织的安全政策禁止向第三方发送任何数据，那么就在自己的基础设施上托管它。当然会有成本，但这绝对是值得支付的成本。

## 监控系统和应用程序指标

在监控性能方面，可供选择的工具数量可能令人不知所措。如果您期望很高，那么可能需要同时使用其中的几个。

Munin（[`munin-monitoring.org`](http://munin-monitoring.org)）是许多组织使用的热门选择之一，无论它们使用什么技术栈。它是一个很好的工具，用于分析资源趋势，并且即使在默认安装时也提供了许多有用的信息，而无需额外配置。它的安装包括两个主要组件：

+   Munin 主机从其他节点收集指标并提供指标图

+   Munin 节点安装在受监视的主机上，它收集本地指标并将其发送到 Munin 主机

主机、节点和大多数插件都是用 Perl 编写的。还有其他语言的节点实现：`munin-node-c`是用 C 编写的（[`github.com/munin-monitoring/munin-c`](https://github.com/munin-monitoring/munin-c)），`munin-node-python`是用 Python 编写的（[`github.com/agroszer/munin-node-python`](https://github.com/agroszer/munin-node-python)）。Munin 附带了大量插件，可在其`contrib`存储库中使用。这意味着它提供了对大多数流行的数据库和系统服务的开箱即用支持。甚至还有用于监视流行的 Python Web 服务器（如 uWSGI 和 Gunicorn）的插件。

Munin 的主要缺点是它将图形呈现为静态图像，并且实际的绘图配置包含在特定插件配置中。这并不利于创建灵活的监控仪表板，并在同一图表中比较来自不同来源的度量值。但这是我们为简单安装和多功能性所付出的代价。编写自己的插件非常简单。有一个`munin-python`包（[`python-munin.readthedocs.org/en/latest/`](http://python-munin.readthedocs.org/en/latest/)），它可以帮助用 Python 编写 Munin 插件。

很遗憾，Munin 的架构假设每个主机上都有一个单独的监控守护进程负责收集指标，这可能不是监控自定义应用程序性能指标的最佳解决方案。编写自己的 Munin 插件确实非常容易，但前提是监控进程已经以某种方式报告其性能统计数据。如果您想收集一些自定义应用程序级别的指标，可能需要将它们聚合并存储在某些临时存储中，直到报告给自定义的 Munin 插件。这使得创建自定义指标变得更加复杂，因此您可能需要考虑其他解决方案。

另一个特别容易收集自定义指标的流行解决方案是 StatsD（[`github.com/etsy/statsd`](https://github.com/etsy/statsd)）。它是一个用 Node.js 编写的网络守护程序，监听各种统计数据，如计数器、计时器和量规。由于基于 UDP 的简单协议，它非常容易集成。还可以使用名为`statsd`的 Python 包将指标发送到 StatsD 守护程序：

```py
>>> import statsd
>>> c = statsd.StatsClient('localhost', 8125)
>>> c.incr('foo')  # Increment the 'foo' counter.
>>> c.timing('stats.timed', 320)  # Record a 320ms 'stats.timed'.

```

由于 UDP 是无连接的，它对应用程序代码的性能开销非常低，因此非常适合跟踪和测量应用程序代码内的自定义事件。

不幸的是，StatsD 是唯一的指标收集守护程序，因此它不提供任何报告功能。您需要其他进程能够处理来自 StatsD 的数据，以查看实际的指标图。最受欢迎的选择是 Graphite（[`graphite.readthedocs.org`](http://graphite.readthedocs.org)）。它主要做两件事：

+   存储数字时间序列数据

+   根据需要呈现此数据的图形

Graphite 提供了保存高度可定制的图形预设的功能。您还可以将许多图形分组到主题仪表板中。与 Munin 类似，图形呈现为静态图像，但还有 JSON API 允许其他前端读取图形数据并以其他方式呈现。与 Graphite 集成的一个很棒的仪表板插件是 Grafana（[`grafana.org`](http://grafana.org)）。它真的值得一试，因为它比普通的 Graphite 仪表板具有更好的可用性。Grafana 提供的图形是完全交互式的，更容易管理。

不幸的是，Graphite 是一个有点复杂的项目。它不是一个单一的服务，而是由三个独立的组件组成：

+   **Carbon**：这是一个使用 Twisted 编写的守护程序，用于监听时间序列数据

+   **whisper**：这是一个简单的数据库库，用于存储时间序列数据

+   **graphite webapp**：这是一个 Django Web 应用程序，根据需要呈现静态图像（使用 Cairo 库）或 JSON 数据

当与 StatsD 项目一起使用时，`statsd`守护程序将其数据发送到`carbon`守护程序。这使得整个解决方案成为一个相当复杂的各种应用程序堆栈，每个应用程序都是使用完全不同的技术编写的。此外，没有预配置的图形、插件和仪表板可用，因此您需要自己配置所有内容。这在开始时需要很多工作，很容易忽略一些重要的东西。这就是为什么即使决定将 Graphite 作为核心监控服务，使用 Munin 作为监控备份也可能是一个好主意。

## 处理应用程序日志

虽然像 Sentry 这样的解决方案通常比存储在文件中的普通文本输出更强大，但日志永远不会消失。向标准输出或文件写入一些信息是应用程序可以做的最简单的事情之一，这绝对不应被低估。有可能 raven 发送到 Sentry 的消息不会被传递。网络可能会失败。Sentry 的存储可能会耗尽，或者可能无法处理传入的负载。在任何消息被发送之前，您的应用程序可能会崩溃（例如，出现分段错误）。这只是可能的情况之一。不太可能的是您的应用程序无法记录将要写入文件系统的消息。这仍然是可能的，但让我们诚实一点。如果您面临日志记录失败的情况，可能您有更多紧迫的问题，而不仅仅是一些丢失的日志消息。

记住，日志不仅仅是关于错误。许多开发人员过去认为日志只是在调试问题时有用的数据来源，或者可以用来进行某种取证。肯定有更少的人尝试将其用作生成应用程序指标的来源或进行一些统计分析。但是日志可能比这更有用。它们甚至可以成为产品实现的核心。一个很好的例子是亚马逊的一篇文章，介绍了一个实时竞价服务的示例架构，其中一切都围绕访问日志收集和处理。请参阅[`aws.amazon.com/blogs/aws/real-time-ad-impression-bids-using-dynamodb/`](https://aws.amazon.com/blogs/aws/real-time-ad-impression-bids-using-dynamodb/)。

### 基本的低级日志实践

十二要素应用程序表示日志应被视为事件流。因此，日志文件本身并不是日志，而只是一种输出格式。它们是流的事实意味着它们代表按时间顺序排列的事件。在原始状态下，它们通常以文本格式呈现，每个事件一行，尽管在某些情况下它们可能跨越多行。这对于与运行时错误相关的任何回溯都是典型的。

根据十二要素应用程序方法论，应用程序不应知道日志存储的格式。这意味着写入文件，或者日志轮换和保留不应由应用程序代码维护。这些是应用程序运行的环境的责任。这可能令人困惑，因为许多框架提供了用于管理日志文件以及轮换、压缩和保留实用程序的函数和类。诱人的是使用它们，因为一切都可以包含在应用程序代码库中，但实际上这是一个应该真正避免的反模式。

处理日志的最佳约定可以归结为几条规则：

+   应用程序应始终将日志无缓冲地写入标准输出（`stdout`）

+   执行环境应负责将日志收集和路由到最终目的地

所提到的执行环境的主要部分通常是某种进程监控工具。流行的 Python 解决方案，如 Supervisor 或 Circus，是处理日志收集和路由的第一责任方。如果日志要存储在本地文件系统中，那么只有它们应该写入实际的日志文件。

Supervisor 和 Circus 也能够处理受管进程的日志轮换和保留，但您确实应该考虑是否要走这条路。成功的操作大多是关于简单性和一致性。您自己应用程序的日志可能不是您想要处理和存档的唯一日志。如果您使用 Apache 或 Nginx 作为反向代理，您可能希望收集它们的访问日志。您可能还希望存储和处理缓存和数据库的日志。如果您正在运行一些流行的 Linux 发行版，那么每个这些服务都有它们自己的日志文件被名为`logrotate`的流行实用程序处理（轮换、压缩等）。我强烈建议您忘记 Supervisor 和 Circus 的日志轮换能力，以便与其他系统服务保持一致。`logrotate`更加可配置，还支持压缩。

### 提示

**logrotate 和 Supervisor/Circus**

在使用`logrotate`与 Supervisor 或 Circus 时，有一件重要的事情需要知道。日志的轮换将始终发生在 Supervisor 仍然具有对已轮换日志的打开描述符时。如果您不采取适当的对策，那么新事件仍将被写入已被`logrotate`删除的文件描述符。结果，文件系统中将不再存储任何内容。解决这个问题的方法非常简单。使用`copytruncate`选项为 Supervisor 或 Circus 管理的进程的日志文件配置`logrotate`。在旋转后，它将复制日志文件并在原地将原始文件截断为零大小。这种方法不会使任何现有的文件描述符无效，已经运行的进程可以不间断地写入日志文件。Supervisor 还可以接受`SIGUSR2`信号，这将使其重新打开所有文件描述符。它可以作为`logrotate`配置中的`postrotate`脚本包含在内。这种第二种方法在 I/O 操作方面更经济，但也更不可靠，更难维护。

### 日志处理工具

如果您没有处理大量日志的经验，那么当使用具有实质负载的产品时，您最终会获得这种经验。您很快会注意到，基于将它们存储在文件中并在某些持久存储中备份的简单方法是不够的。没有适当的工具，这将变得粗糙和昂贵。像`logrotate`这样的简单实用程序只能确保硬盘不会被不断增加的新事件所溢出，但是拆分和压缩日志文件只有在数据归档过程中才有帮助，但并不会使数据检索或分析变得更简单。

在处理跨多个节点的分布式系统时，很好地拥有一个单一的中心点，从中可以检索和分析所有日志。这需要一个远远超出简单压缩和备份的日志处理流程。幸运的是，这是一个众所周知的问题，因此有许多可用的工具旨在解决它。

许多开发人员中的一个受欢迎的选择是**Logstash**。这是一个日志收集守护程序，可以观察活动日志文件，解析日志条目并以结构化形式将它们发送到后端服务。后端的选择几乎总是相同的——**Elasticsearch**。Elasticsearch 是建立在 Lucene 之上的搜索引擎。除了文本搜索功能外，它还具有一个独特的数据聚合框架，非常适合用于日志分析的目的。

这对工具的另一个补充是**Kibana**。它是一个非常多才多艺的监控、分析和可视化平台，适用于 Elasticsearch。这三种工具如何相互补充的方式，是它们几乎总是作为单一堆栈一起用于日志处理的原因。

现有服务与 Logstash 的集成非常简单，因为它可以监听现有日志文件的更改，以便通过最小的日志配置更改获取新事件。它以文本形式解析日志，并且预先配置了对一些流行日志格式（如 Apache/Nginx 访问日志）的支持。Logstash 唯一的问题是它不能很好地处理日志轮换，这有点令人惊讶。通过发送已定义的 Unix 信号（通常是`SIGHUP`或`SIGUSR1`）来强制进程重新打开其文件描述符是一个非常成熟的模式。似乎每个处理日志的应用程序都应该知道这一点，并且能够处理各种日志文件轮换场景。遗憾的是，Logstash 不是其中之一，因此如果您想使用`logrotate`实用程序管理日志保留，请记住要大量依赖其`copytruncate`选项。Logstash 进程无法处理原始日志文件被移动或删除的情况，因此在没有`copytruncate`选项的情况下，它将无法在日志轮换后接收新事件。当然，Logstash 可以处理不同的日志流输入，例如 UDP 数据包、TCP 连接或 HTTP 请求。

另一个似乎填补了一些 Logstash 空白的解决方案是 Fluentd。它是一种替代的日志收集守护程序，可以与 Logstash 在提到的日志监控堆栈中互换使用。它还有一个选项，可以直接监听和解析日志事件，所以最小的集成只需要一点点努力。与 Logstash 相比，它处理重新加载非常出色，甚至在日志文件轮换时也不需要信号。无论如何，最大的优势来自于使用其替代的日志收集选项，这将需要对应用程序中的日志配置进行一些重大更改。

Fluentd 真的将日志视为事件流（正如《十二要素应用程序》所推荐的）。基于文件的集成仍然是可能的，但它只是对将日志主要视为文件的传统应用程序的向后兼容性。每个日志条目都是一个事件，应该是结构化的。Fluentd 可以解析文本日志，并具有多个插件选项来处理：

+   常见格式（Apache、Nginx 和 syslog）

+   使用正则表达式指定的任意格式，或者使用自定义解析插件处理

+   结构化消息的通用格式，例如 JSON

Fluentd 的最佳事件格式是 JSON，因为它增加的开销最少。 JSON 中的消息也可以几乎不经过任何更改地传递到 Elasticsearch 或数据库等后端服务。

Fluentd 的另一个非常有用的功能是能够使用除了写入磁盘的日志文件之外的其他传输方式传递事件流。最值得注意的内置输入插件有：

+   `in_udp`：使用此插件，每个日志事件都作为 UDP 数据包发送

+   `in_tcp`：使用此插件，事件通过 TCP 连接发送

+   `in_unix`：使用此插件，事件通过 Unix 域套接字（命名套接字）发送

+   `in_http`：使用此插件，事件作为 HTTP POST 请求发送

+   `in_exec`：使用此插件，Fluentd 进程会定期执行外部命令，以 JSON 或 MessagePack 格式获取事件

+   `in_tail`：使用此插件，Fluentd 进程会监听文本文件中的事件

对于日志事件的替代传输可能在需要处理机器存储的 I/O 性能较差的情况下特别有用。在云计算服务中，通常默认磁盘存储的 IOPS（每秒输入/输出操作次数）非常低，您需要花费大量资金以获得更好的磁盘性能。如果您的应用程序输出大量日志消息，即使数据量不是很大，也可能轻松饱和您的 I/O 能力。通过替代传输，您可以更有效地使用硬件，因为您只需将数据缓冲的责任留给单个进程——日志收集器。当配置为在内存中缓冲消息而不是磁盘时，甚至可以完全摆脱日志的磁盘写入，尽管这可能会大大降低收集日志的一致性保证。

使用不同的传输方式似乎略微违反了十二要素应用程序方法的第 11 条规则。详细解释时，将日志视为事件流表明应用程序应始终仅通过单个标准输出流（`stdout`）记录日志。仍然可以在不违反此规则的情况下使用替代传输方式。写入`stdout`并不一定意味着必须将此流写入文件。您可以保留应用程序以这种方式记录日志，并使用外部进程将其捕获并直接传递给 Logstash 或 Fluentd，而无需涉及文件系统。这是一种高级模式，可能并不适用于每个项目。它具有更高复杂性的明显缺点，因此您需要自行考虑是否真的值得这样做。

# 总结

代码部署并不是一个简单的话题，阅读本章后您应该已经知道这一点。对这个问题的广泛讨论很容易占据几本书。即使我们的范围仅限于 Web 应用程序，我们也只是触及了表面。本章以十二要素应用程序方法为基础。我们只详细讨论了其中的一些内容：日志处理、管理依赖关系和分离构建/运行阶段。

阅读本章后，您应该知道如何正确自动化部署过程，考虑最佳实践，并能够为在远程主机上运行的代码添加适当的仪器和监视。
