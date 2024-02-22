# 第七章：云计算

*云计算*是通过互联网（*云*）分发计算服务，如服务器、存储资源、数据库、网络、软件、分析和智能。本章的目的是概述与 Python 编程语言相关的主要云计算技术。

首先，我们将描述 PythonAnywhere 平台，通过它我们将在云上部署 Python 应用程序。在云计算的背景下，将确定两种新兴技术：容器和无服务器技术。

*容器*代表资源虚拟化的新方法，*无服务器*技术代表了云服务领域的一大进步，因为它们可以加快应用程序的发布。

实际上，您不必担心供应、服务器或基础架构配置。您只需要创建可以独立于应用程序运行的函数（即 Lambda 函数）。

在本章中，我们将涵盖以下内容：

+   什么是云计算？

+   了解*云计算架构

+   使用 PythonAnywhere 开发 Web 应用程序

+   将 Python 应用程序容器化

+   介绍无服务器计算

我们还将看到如何利用*AWS Lambda*框架开发 Python 应用程序。

# 什么是云计算？

云计算是一种基于一组资源的计算模型，例如虚拟处理、大容量存储和网络，可以动态聚合和激活为运行应用程序的平台，满足适当的服务水平并优化资源使用效率。

这可以通过最少的管理工作或与服务提供商的交互快速获取和释放。这种云模型由五个基本特征、三种服务模型和四种部署模型组成。

特别是，五个基本特征如下：

+   **免费和按需访问**：这使用户可以通过*用户友好*的界面访问提供商提供的服务，无需人工干预。

+   **网络的无处不在的访问**：资源可以通过网络随时访问，并且可以通过标准设备（如*智能手机*、*平板电脑*和*个人电脑*）访问。

+   **快速弹性**：这是云快速和自动增加或减少分配的资源的能力，使其对用户来说似乎是无限的。这为系统提供了很大的可伸缩性。

+   **可测量的服务**：云系统不断监视提供的资源，并根据估计的使用自动优化它们。这样，客户只支付在特定会话中实际使用的资源。

+   **资源共享**：提供商通过多租户模型提供其资源，以便可以根据客户的请求动态分配和重新分配，并由多个消费者使用：

![](img/2ada1f6c-31da-4700-aa33-d72e0da304a4.png)

云计算的主要特点

然而，云计算有许多定义，每个定义都有不同的解释和含义。国家标准与技术研究所（**NIST**）试图提供详细和官方的解释（[`csrc.nist.gov/publications/detail/sp/800-145/final`](https://csrc.nist.gov/publications/detail/sp/800-145/final)）。

另一个特性（未列在 NIST 定义中，但是云计算的基础）是虚拟化的概念。这是在相同的物理资源上执行多个*操作系统*的可能性，保证了许多优势，如可伸缩性、成本降低和向客户提供新资源的速度更快。

虚拟化的最常见方法如下：

+   容器

+   虚拟机

这两种解决方案在隔离应用程序方面几乎具有相同的优势，但它们在不同的虚拟化级别上工作，因为容器虚拟化操作系统，而虚拟机虚拟化硬件。这意味着容器更具可移植性和效率。

通过容器进行虚拟化的最常见应用是 Docker。我们将简要介绍这个框架，并看看如何将 Python 应用程序容器化（或 dockerize）。

# 了解云计算架构

云计算架构指的是构成系统结构的一系列组件和子组件。通常，它可以分为*前端*和*后端*两个主要部分：

![](img/ee53b3a7-0361-4d4e-a48a-b3e8ec8f6bb3.png)

云计算架构

每个部分都有非常具体的含义和范围，并通过虚拟网络或互联网网络与其他部分相连。

*前端*指的是用户可见的云计算系统部分，通过一系列界面和应用程序实现，允许消费者访问云系统。不同的云计算系统有不同的用户界面。

*后端*是客户看不到的部分。该部分包含所有资源，允许提供商提供云计算服务，如服务器、存储系统和虚拟机。创建后端的想法是将整个系统的管理委托给单个中央服务器，因此必须不断监视流量和用户请求，执行访问控制，并实施通信协议。

在这种架构的各个组件中，最重要的是 Hypervisor，也称为*虚拟机管理器*。这是一种固件，可以动态分配资源，并允许您在多个用户之间共享单个实例。简而言之，这是实现虚拟化的程序，这是云计算的主要属性之一。

在提供云计算的定义并解释基本特性之后，我们将介绍云计算服务可以提供的*服务模型*。

# 服务模型

提供商提供的云计算服务可分为三大类：

+   **S**软件即服务（SaaS）

+   **P**平台即服务（PaaS）

+   **I**基础设施即服务（IaaS）

这种分类导致了一个名为**SPI**模型的方案的定义（请参阅前面列表中的**粗体**首字母）。有时它被称为云计算堆栈，因为这些类别是基于彼此的。

现在将详细描述每个级别，采用自上而下的方法。

# SaaS

SaaS 提供商为用户提供按需的软件应用程序，可以通过任何互联网设备（如 Web 浏览器）访问。此外，提供商托管软件应用程序和基础架构，减轻了客户管理和维护活动的负担，如软件更新和安全补丁的应用。

使用这种模型对用户和提供商都有许多优势。对于用户来说，管理成本大大降低，对于提供商来说，他们对流量有更多的控制，从而避免任何过载。SaaS 的一个例子是任何基于 Web 的电子邮件服务，如**Gmail**，**Outlook**，**Salesforce**和**Yahoo!**。

# PaaS

与 SaaS 不同，这项服务指的是应用程序的整个开发环境，而不仅仅是其使用。因此，PaaS 解决方案提供了一个通过 Web 浏览器访问的云平台，用于开发、测试、分发和管理软件应用程序。此外，提供商提供基于 Web 的界面、多租户架构和通信工具，以便让开发人员更简单地创建应用程序。这支持软件的整个生命周期，也有利于合作。

PaaS 的例子有**微软 Azure 服务**、**谷歌应用引擎**和**亚马逊网络服务**。

# IaaS

IaaS 是一种以按需服务提供计算基础设施的模型。因此，您可以购买虚拟机，在其上运行自己的软件，存储资源（根据实际需求迅速增加或减少存储容量），网络和操作系统，并根据实际使用情况付费。这种动态基础设施增加了更大的可扩展性，同时也大大降低了成本。

这种模型既被小型新兴公司使用，因为它们没有大量资金进行投资，也被寻求简化其硬件架构的成熟公司使用。IaaS 卖家的范围非常广泛，包括**亚马逊网络服务**、**IBM**和**甲骨文**。

# 分发模型

事实上，云计算架构并非都是一样的。实际上，有四种不同的分发模型：

+   **公共云**

+   ****私有云****

+   **云社区**

+   **混合云**

# 公共云

这种分发模型对所有人开放，包括个人用户和公司。通常，公共云在由服务提供商拥有的数据中心中运行，处理硬件、软件和其他支持基础设施。这样，用户就不必进行任何维护活动/费用。

# 私有云

也被称为*内部云*，私有云提供与公共云相同的优势，但对数据和流程提供更大的控制。这种模型被呈现为一种专门为公司工作的云基础设施，因此在给定公司的边界内进行管理和托管。显然，使用它的组织可以将其架构扩展到与其有业务关系的任何群体。

通过采用这种解决方案，可以避免涉及敏感数据违规和工业间谍活动的可能问题，同时也不忽视使用简化、可配置和高性能的工作配置系统的可能性。正因为如此，近年来使用私有云的公司数量显著增加。

# 云社区

从概念上讲，这种模型描述了由几家具有共同利益的公司实施和管理的共享基础设施。这种类型的解决方案很少被使用，因为在各个社区成员之间分享责任和管理活动可能变得复杂。

# 混合云

NIST 将其定义为前面提到的三种实施模型（私有云、公共云和社区云）的组合结果，试图利用每种云的优势来弥补其他云的不足之处。使用的云保持独立实体，这可能导致操作一致性的缺失。因此，采用这种模型的公司有责任通过专有技术来保证其服务器的互操作性，使其能够优化其必须扮演的特定角色。

混合云与其他所有云的一个特点是云爆发，或者在出现大量峰值需求时，能够动态地将私有云中的过多流量转移到公共云中的可能性。

这种实施模型是由那些打算在保留内部云中的敏感数据的同时共享其软件应用程序的公司采用的。

# 云计算平台

云计算平台是一组软件和技术，可以在云中交付资源（按需，可扩展和虚拟化资源）。最受欢迎的平台包括谷歌的平台，当然还有云计算的里程碑：**亚马逊网络服务**（**AWS**）。两者都支持 Python 作为开发语言。

然而，在下一个教程中，我们将专注于 PythonAnywhere，这是专门用于部署 Python 编程语言的 Web 应用程序的云平台。

# 使用 PythonAnywhere 开发 Web 应用程序

PythonAnywhere 是基于 Python 编程语言的在线托管开发和服务环境。一旦在网站上注册，您将被引导到包含完全由 HTML 代码制作的高级 shell 和文本编辑器的仪表板。通过这样，您可以创建，修改和执行自己的脚本。

此外，这个开发环境还允许您选择要使用的 Python 版本。在这方面，一个简单的向导帮助我们预配置应用程序。

# 准备就绪

首先让我们看看如何获取网站的登录凭据。

以下屏幕截图显示了各种订阅类型，以及获得免费帐户的可能性（请转到[`www.pythonanywhere.com/registration/register/beginner/`](https://www.pythonanywhere.com/registration/register/beginner/)）：

![](img/4696816c-2cbe-451c-869b-2f40d7795d20.png)

PythonAnywhere：注册页面

一旦获得了对网站的访问权（建议您创建一个初学者帐户），我们登录。鉴于集成到浏览器中的 Python shell 对于初学者和入门编程课程来说非常有用，它们在技术上当然不是新鲜事物。

相反，PythonAnywhere 的附加值在您登录并访问个人仪表板时立即被感知：

![](img/1154438e-ea7b-4dd5-aa09-ddb777bdf822.png)

PythonAnywhere：仪表板

通过个人仪表板，我们可以选择在 2.7 和 3.7 之间运行的 Python 版本，还可以选择是否使用 IPython 界面：

![](img/e36f1f2e-0d04-4b54-88d6-6dbe567056e9.png)

PythonAnywhere：控制台视图

可以使用的控制台数量根据您拥有的订阅类型而变化。在我们的情况下，我们使用了初学者帐户，最多可以使用两个 Python 控制台。选择 Python shell，例如版本 3.5，应该在 Web 浏览器上打开以下视图：

![](img/53465f5a-186d-489c-9eab-fa193e4199c4.png)

PythonAnywhere：Python shell

在接下来的部分，我们想向您展示如何使用 PythonAnywhere 编写一个简单的 Web 应用程序。

# 如何做到... 

让我们看看以下步骤：

1.  在仪表板上，打开 Web 选项卡：

![](img/688c7c96-c296-42e7-8da8-aefc386aff11.png)

PythonAnywhere：Web 应用程序视图

1.  界面告诉我们我们还没有 Web 应用程序。通过选择添加新的 Web 应用程序，将打开以下视图。它告诉我们我们的应用程序将具有以下 Web 地址：[loginname.pythonanywhere.com](http://loginname.pythonanywhere.com)（例如，应用程序的 Web 地址将是[giazax.pythonanywhere.com](http://giazax.pythonanywhere.com)）：

![](img/db5b64c1-b102-47f4-b7f2-1eb5d8a79800.png)

PythonAnywhere：Web 应用程序向导

1.  当我们单击“下一步”时，我们可以选择我们想要使用的 Python Web 框架：

![](img/c2d207f8-18c7-4def-a938-e7528ea75290.png)

PythonAnywhere：Web 框架向导

1.  我们选择 Flask 作为 Web 框架，然后单击“下一步”来选择我们想要使用的 Python 版本，如下所示：

![](img/e70bb841-e4bc-44ea-a3c1-6409acbcbf73.png)

PythonAnywhere：Web 框架向导 Flask 是一个易于安装和使用的 Python 微框架，被 Pinterest 和 LinkedIn 等公司使用。

如果您不知道用于创建 Web 应用程序的框架是什么，那么您可以想象一组旨在简化 Web 服务（如 Web 服务器和 API）创建的程序。有关 Flask 的更多信息，请访问[`flask.pocoo.org/docs/1.0/`](http://flask.pocoo.org/docs/1.0/)。

1.  在上一个屏幕截图中，我们选择了 Flask 1.0.2 的 Python 3.5，然后点击“下一步”以输入用于保存 Flask 应用程序的 Python 文件的路径。在这里，选择了默认文件：

![](img/cf31fec4-6cf0-4ff9-ad1c-214ce6194f2a.png)

PythonAnywhere：Flask 项目定义

1.  当我们最后一次点击“下一步”时，将显示以下屏幕，其中总结了 Web 应用程序的配置参数：

![](img/c3ca51d3-dc99-44e5-99bb-5c5196be8342.png)

PythonAnywhere：giazax.pythonanywhere.com 的配置页面

现在，让我们看看这会发生什么。

# 它是如何工作的...

在 Web 浏览器的地址栏中，键入我们的 Web 应用程序的 URL，例如`https://giazax.pythonanywhere.com/`。该站点显示一个简单的欢迎短语：

![](img/5e9035b6-5b9d-421d-8aa0-3c6ecb2b0dec.png)

giazax.pythonanywhere.com 站点页面

通过选择“转到目录”可以查看此应用程序的源代码，与“源代码”标签对应。

![](img/df75e642-d6d0-48a7-8f8c-ed458c70f4dd.png)

PythonAnywhere：配置页面

在这里，可以分析构成 Web 应用程序的文件：

![](img/9d75d76e-4bb5-4a18-b904-3aabf59b9158.png)

PythonAnywhere：项目站点存储库

还可以上传新文件并可能修改内容。在这里，我们选择了我们第一个 Web 应用程序的`flask_app.py`文件。内容看起来像一个最小的 Flask 应用程序：

```py
# A very simple Flask Hello World app for you to get started with...

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
 return 'Hello from Flask!'
```

`route()`装饰器由 Flask 用于定义应触发`hello_world`函数的 URL。这个简单的函数返回在 Web 浏览器中显示的消息。

# 还有更多...

PythonAnywhere shell 是用 HTML 制作的，几乎可以在多个平台和浏览器上使用，包括苹果的移动版本。可以保持多个 shell 打开（根据所选的帐户配置文件选择不同数量），与其他用户共享它们，或根据需要终止它们。

PythonAnywhere 具有一个相当先进的文本编辑器，具有语法着色和自动缩进功能，通过它可以创建，修改和执行自己的脚本。文件存储在存储区域中，其大小取决于帐户的配置文件，但如果空间不足或者希望更流畅地与 PC 的文件系统集成，那么 PythonAnywhere 允许您使用 Dropbox 帐户，在流行的存储服务上访问您的共享文件夹。

每个 shell 可以包含与特定 URL 对应的 WSGI 脚本。还可以启动一个 bash shell，从中调用 Git 并与文件系统交互。最后，正如我们所看到的，有一个可用的向导，允许我们预配置**Django**和**web2py**或 Flask 应用程序。

此外，还有利用**MySQL**数据库的可能性，这是一系列允许我们定期执行某些脚本的 cron 作业。因此，我们将获得 PythonAnywhere 的真正本质：以光速部署 Web 应用程序。

*PythonAnywhere* 完全依赖于**Amazon EC2**基础设施，因此没有理由不信任该服务。因此，强烈建议那些考虑个人使用的人使用。初学者账户提供的资源比**Heroku**上的对应资源更多（[`www.heroku.com/`](https://www.heroku.com/)），部署比**OpenShift**（[`www.openshift.com/`](https://www.openshift.com/)）更简单，整个系统通常比**Google App Engine**（[`cloud.google.com/appengine/`](https://cloud.google.com/appengine/)）更灵活。

# 另请参阅

+   PythonAnywhere 的主要资源可以在这里找到：[`www.pythonanywhere.com`](https://www.pythonanywhere.com)。

+   对于通过 Python 进行 Web 编程，PythonAnywhere 支持**Django**（[`www.djangoproject.com/`](https://www.djangoproject.com/)）和**web2py**（[`www.web2py.com/`](http://www.web2py.com/)），以及**Flask**。

与**Flask**一样，建议您访问这些网站以获取有关如何使用这些库的信息。

# 将 Python 应用程序容器化

容器是虚拟化环境。它们包括软件所需的一切，即库、依赖项、文件系统和网络接口。与经典的虚拟机不同，所有上述元素与它们运行的机器共享内核。这样，对主机节点资源的使用影响大大减少。

这使得容器在可扩展性、性能和隔离方面成为一种非常有吸引力的技术。容器并不是一种新技术；它们在 2013 年 Docker 推出时就取得了成功。从那时起，它们彻底改变了应用开发和管理所使用的标准。

Docker 是一个基于**Linux 容器**（**LXC**）实现的容器平台，它通过管理容器作为自包含映像，并添加额外的工具来协调其生命周期和保存其状态，扩展了这项技术的功能。

容器化的想法恰恰是允许给定的应用程序在任何类型的系统上执行，因为所有其依赖项已经包含在容器本身中。

这样，应用程序变得高度可移植，并且可以在任何类型的环境上轻松测试和部署，无论是本地还是云端。

现在，让我们看看如何使用 Docker 将 Python 应用程序容器化。

# 准备工作

Docker 团队的直觉是采用容器的概念并构建一个围绕它的生态系统，简化其使用。这个生态系统包括一系列工具：

+   Docker 引擎（[`www.docker.com/products/docker-engine`](https://www.docker.com/products/docker-engine)）

+   Docker 工具箱（[`docs.docker.com/toolbox/`](https://docs.docker.com/toolbox/)）

+   Swarm（[`docs.docker.com/engine/swarm/`](https://docs.docker.com/engine/swarm/)）

+   Kitematic（[`kitematic.com/`](https://kitematic.com/)）

# 安装 Windows 版 Docker

安装非常简单：一旦下载了安装程序（[`docs.docker.com/docker-for-windows/install/`](https://docs.docker.com/docker-for-windows/install/)），只需运行它，就完成了。安装过程通常非常线性。唯一需要注意的是安装的最后阶段，可能需要启用 Hyper-V 功能。如果是这样，我们就接受并重新启动机器。

计算机重新启动后，Docker 图标应该出现在屏幕右下角的系统托盘中。

打开命令提示符或 PowerShell 控制台，并通过执行`docker version`命令来检查一切是否正常：

```py
C:\>docker version
Client: Docker Engine - Community
 Version: 18.09.2
 API version: 1.39
 Go version: go1.10.8
 Git commit: 6247962
 Built: Sun Feb 10 04:12:31 2019
 OS/Arch: windows/amd64
 Experimental: false

Server: Docker Engine - Community
 Engine:
 Version: 18.09.2
 API version: 1.39 (minimum version 1.12)
 Go version: go1.10.6
 Git commit: 6247962
 Built: Sun Feb 10 04:13:06 2019
 OS/Arch: linux/amd64
 Experimental: false
```

输出中最有趣的部分是在客户端和服务器之间进行的细分。客户端是我们的本地 Windows 系统，而服务器是 Docker 在幕后实例化的 Linux 虚拟机。这些部分通过 API 层进行通信，正如本教程开头提到的那样。

现在，让我们看看如何容器化（或 dockerize）一个简单的 Python 应用程序。

# 如何做...

让我们想象我们想要部署以下 Python 应用程序，我们称之为`dockerize.py`：

```py
from flask import Flask
app = Flask(__name__)
@app.route("/")
def hello():
 return "Hello World!"
if __name__ == "__main__":
 app.run(host="0.0.0.0", port=int("5000"), debug=True)
```

示例应用程序使用`Flask`模块。它在本地地址`5000`实现了一个简单的 Web 应用程序。

第一步是创建以下文本文件，扩展名为`.py`，我们将其称为`Dockerfile.py`：

```py
FROM python:alpine3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python ./dockerize.py
```

前面代码中列出的指令执行以下任务：

+   `FROM python:alpine3.7`指示 Docker 使用 Python 版本 3.7。

+   `COPY`将应用程序复制到容器镜像中。

+   `WORKDIR`设置工作目录（`WORKDIR`）。

+   `RUN`指令调用`pip`安装程序，指向`requirements.txt`文件。它包含应用程序必须执行的依赖项列表（在我们的情况下，唯一的依赖是`flask`）。

+   `EXPOSE`指令公开了 Flask 使用的端口。

因此，总结一下，我们已经编写了三个文件：

+   要容器化的应用程序：`dockerize.py`

+   `Dockerfile`

+   依赖列表文件

因此，我们需要创建`dockerize.py`应用程序的镜像：

```py
docker build --tag dockerize.py
```

这将标记`my-python-app`镜像并构建它。

# 它是如何工作的...

`my-python-app`镜像构建完成后，可以将其作为容器运行：

```py
docker run -p 5000:5000 dockerize.py
```

然后启动应用程序作为容器，之后，名称参数发送名称到容器，`-p`参数将`5000`主机端口映射到容器端口`5000`。

接下来，您需要打开您的 Web 浏览器，然后在地址栏中输入`localhost:5000`。如果一切顺利，您应该看到以下网页：

![](img/bbba23ab-e0cf-4174-9c0e-63b7b8a520b6.png)

Docker 应用程序

Docker 使用`run`命令运行`dockerize.py`容器，结果是一个 Web 应用程序。镜像包含了容器运行所需的指令。

容器和镜像之间的关联可以通过将镜像与类关联，将容器与类实例关联来理解面向对象编程范式。

当我们创建容器实例时，有必要总结发生了什么：

+   容器的镜像（如果尚未存在）将在本地卸载。

+   创建一个启动容器的环境。

+   屏幕上打印出一条消息。

+   然后放弃先前创建的环境。

所有这些都在几秒钟内以简单、直观和可读的命令完成。

# 还有更多...

显然，容器和虚拟机似乎是非常相似的概念。但尽管这两种解决方案具有共同的特点，它们是根本不同的技术，就像我们必须开始思考我们的应用程序架构有何不同一样。我们可以在容器中创建我们的单片应用程序，但这样做将无法充分利用容器的优势，因此也无法充分利用 Docker 的优势。

适用于容器基础架构的可能软件架构是经典的微服务架构。其思想是将应用程序分解为许多小组件，每个组件都有自己特定的任务，能够交换消息并相互合作。这些组件的部署将以许多容器的形式单独进行。

使用微服务可以处理的场景在虚拟机中是绝对不切实际的，因为每个新实例化的虚拟机都需要主机机器大量的能源开支。另一方面，容器非常轻便，因为它们执行与虚拟机完全不同的虚拟化：

![](img/f52f2b34-c17f-4138-80f4-e59f1fc1210e.png)

虚拟机和 Docker 实现中的微服务架构

在虚拟机中，一个称为**Hypervisor**的工具负责从主机操作系统中静态或动态地保留一定数量的资源，以便专门用于一个或多个称为**guests**或**hosts**的操作系统。客用操作系统将完全与主机操作系统隔离。这种机制在资源方面非常昂贵，因此将微服务与虚拟机结合的想法是完全不可能的。

另一方面，容器对这个问题提供了完全不同的解决方案。隔离性要弱得多，所有运行的容器共享与底层操作系统相同的内核。Hypervisor 的开销完全消失，一个主机可以承载数百个容器。

当我们要求 Docker 从其镜像运行容器时，它必须存在于本地磁盘上，否则 Docker 将警告我们有问题（显示消息“无法在本地找到图像'hello-world: latest'”），并将自动下载它。

要查看在我们的计算机上从 Docker 下载了哪些镜像，我们使用`docker images`命令：

```py
C:\>docker images
REPOSITORY TAG IMAGE ID CREATED SIZE
dockerize.py latest bc3d70b05ed4 23 hours ago 91.8MB
<none> <none> ca18efb44b3c 24 hours ago 91.8MB
python alpine3.7 00be2573e9f7 2 months ago 81.3MB
```

存储库是相关图像的容器。例如，dockerize 存储库包含 dockerize 图像的各种版本。在 Docker 世界中，术语**标签**更正确地用于表示图像版本的概念。在前面的代码示例中，图像已被标记为最新版本，并且是 dockerize 存储库唯一可用的标签。

最新标签是默认标签：每当我们引用一个存储库而没有指定标签名称时，Docker 将隐式地引用最新标签，如果不存在，则会显示错误。因此，作为最佳实践，存储库标签形式更可取，因为它允许更大的可预测性，避免容器之间的可能冲突和由于缺少最新标签而导致的错误。

# 另请参见

容器技术是一个非常广泛的概念，可以通过查阅网上的许多文章和应用示例来探索。然而，在开始这段漫长而艰难的旅程之前，建议从完整且充分信息的网站（[`www.docker.com/`](https://www.docker.com/)）开始。

在下一节中，我们将研究无服务器计算的主要特点，其主要目标是使软件开发人员更容易地编写设计用于在云平台上运行的代码。

# 介绍无服务器计算

近年来，出现了一种名为**函数即服务**（**FaaS**）的新服务模型，也被称为**无服务器计算**。

无服务器计算是一种云计算范式，允许执行应用程序而不必担心与底层基础设施相关的问题。术语**无服务器**可能会产生误导；事实上，可以认为这种模型不预见使用处理服务器。实际上，它表明应用程序的提供、可伸缩性和管理是自动进行的，对于开发人员来说是完全透明的。这一切都得益于一种称为**无服务器**的新架构模型。

第一个 FaaS 模型可以追溯到 2014 年发布的 AWS Lambda 服务。随着时间的推移，其他替代方案被添加到亚马逊解决方案中，这些替代方案由其他主要供应商开发，例如微软的 Azure Functions，IBM 和 Google 的 Cloud Functions。还有有效的开源解决方案：其中最常用的是 IBM 在其无服务器提供的 Bluemix 上使用的 Apache OpenWhisk，但也有 OpenLambda 和 IronFunctions，后者基于 Docker 的容器技术。

在这个教程中，我们将看到如何通过 AWS Lambda 实现无服务器 Python 函数。

# 准备就绪

AWS 是一整套通过共同接口提供和管理的云服务。提供 AWS 网络控制台中的服务的共同接口可在[`console.aws.amazon.com/`](https://console.aws.amazon.com/)上访问。

这种类型的服务是收费的。但是，在第一年，提供了*免费套餐*。这是一组使用最少资源并且可以免费用于评估服务和应用程序开发的服务。

有关如何在 AWS 创建免费账户的详细信息，请参阅官方亚马逊文档[`aws.amazon.com`](https://aws.amazon.com/)。

在这些部分，我们将概述在 AWS Lambda 中运行代码的基础知识，而无需预配或管理任何服务器。我们将展示如何使用 AWS Lambda 控制台在 Lambda 中创建`Hello World`函数。我们还将解释如何使用示例事件数据手动调用 Lambda 函数以及如何解释输出参数。本教程中显示的所有操作都可以作为免费计划的一部分执行[`aws.amazon.com/free`](https://aws.amazon.com/free)。

# 如何做...

让我们看看以下步骤：

1.  首先要做的是登录 Lambda 控制台([`console.aws.amazon.com/console/home`](https://console.aws.amazon.com/console/home))。然后，您需要定位并选择 Lambda 以在计算下打开 AWS Lambda 控制台（在以下截图中以绿色突出显示）：

![](img/e9d3929c-8fd7-49e4-ad1f-2f8a4f35040c.png)

AWS：选择 Lambda 服务

1.  然后，在 AWS Lambda 控制台中，选择立即开始，然后创建 Lambda 函数：

![](img/320f0ad3-408b-4960-9122-c930eb52b466.png)

AWS：Lambda 启动页面

1.  在筛选框中，输入`hello-world-python`，然后选择 hello-world-python 蓝图。

1.  现在我们需要配置 Lambda 函数。以下列表显示了配置并提供了示例值：

+   **配置函数**：

+   **名称**：在这里输入函数的名称。对于本教程，请输入`hello-world-python`。

+   **描述**：在这里，您可以输入函数的简要描述。此框中预填有短语 A starter AWS Lambda Function。

+   运行时：目前，可以使用 Java，Node.js 和 Python 2.7，3.6 和 3.7 编写 Lambda 函数的代码。对于本教程，请设置 Python 2.7 作为运行时。

+   Lambda 函数代码：

+   如下截图所示，可以查看 Python 示例代码。

+   **Lambda 函数处理程序和角色**：

+   处理程序：您可以指定 AWS Lambda 启动执行代码的方法。AWS Lambda 将事件数据作为处理程序的输入，然后处理事件。在此示例中，Lambda 从示例代码中识别事件，因此该字段将使用 lambda_function.lambda_handler 进行编译。

+   角色：单击下拉菜单，然后选择基本执行角色：

![](img/7743d286-d524-4125-9c76-5b27e8bb07ed.png)

AWS 配置函数页面

1.  在这一点上，有必要创建一个执行角色（名为 IAM 角色），该角色具有必要的授权，以便由 AWS Lambda 解释为 Lambda 函数的执行者。点击允许后，将返回配置函数页面，并选择 lambda_basic_execution 函数：

![](img/6a8575e9-23f0-41ac-952d-67f44104f573.png)

AWS：角色摘要页面

1.  控制台将代码保存在一个压缩文件中，该文件代表分发包。然后，控制台将分发包加载到 AWS Lambda 中以创建 Lambda 函数：

![](img/60142c8f-cb80-4dc5-9fa2-cb8761f5164b.png)

AWS：Lambda 审查页面

现在可以测试函数，检查结果并显示日志：

1.  要运行我们的第一个 Lambda 函数，请点击测试：

![](img/1731dab3-8c0f-4dcd-8550-ecc283af3b75.png)

AWS：Lambda 测试页面

1.  在弹出编辑器中输入事件以测试函数。

1.  在输入测试事件页面的示例事件模板列表中选择 Hello World：

![](img/58d60050-f25e-4828-a340-6059c227b0ac.png)

AWS：Lambda 模板

点击保存并测试。然后，AWS Lambda 将代表您执行该函数。

# 它是如何工作的...

执行完成后，可以在控制台中看到结果：

+   执行结果部分记录了函数的正确执行。

+   摘要部分显示了日志输出部分报告的最重要信息。

+   日志输出部分显示了 Lambda 函数执行生成的日志：

![](img/12d7eb67-8089-4f9f-b9b1-9a75ab4becdd.png)

AWS：执行结果

# 还有更多...

**AWS Lambda**监视函数并通过**Amazon CloudWatch**自动生成参数报告（请参见以下截图）。为了简化在执行期间对代码的监视，AWS Lambda 会自动跟踪请求数、每个请求的延迟以及带有错误的请求数量，并发布相关参数：

![](img/66a850ce-0883-45ed-ab46-5a2eb89214fb.png)

# 什么是 Lambda 函数？

Lambda 函数包含开发人员希望响应某些事件执行的代码。开发人员负责在参考提供程序的控制台中配置此代码并指定资源方面的要求。其他所有内容，包括资源的大小，都是由提供程序自动管理的，根据所需的工作负载。

# 为什么选择无服务器？

无服务器计算的好处如下：

+   **无需管理基础设施：**开发人员可以专注于构建产品，而不是运行时服务器的操作和管理。

+   **自动可伸缩性：**资源会自动重新校准以应对任何类型的工作负载，无需进行缩放配置，而是根据实时事件做出反应。

+   **资源使用优化：**由于处理和存储资源是动态分配的，因此不再需要提前投资于过量的容量。

+   **成本降低：**在传统的云计算中，即使实际上没有使用，也会预期支付运行资源的费用。在无服务器情况下，应用程序是事件驱动的，这意味着当应用程序代码未运行时，不会收取任何费用，因此您不必为未使用的资源付费。

+   **高可用性：**管理基础设施和应用程序的服务保证高可用性和容错性。

+   **市场推出时间改善：**消除基础设施管理费用使开发人员能够专注于产品质量，并更快地将代码投入生产。

# 可能的问题和限制

在评估采用无服务器计算时，需要考虑一些缺点：

+   **可能的性能损失：** 如果代码不经常使用，那么在执行过程中可能会出现延迟问题。与在服务器、虚拟机或容器上连续执行的情况相比，这些问题更加突出。这是因为（与使用自动缩放策略相反），在无服务器模型中，如果代码未被使用，云提供商通常会完全取消分配资源。这意味着如果运行时需要一些时间来启动，那么在初始启动阶段必然会产生额外的延迟。

+   **无状态模式：** 无服务器函数以无状态模式运行。这意味着，如果要添加逻辑以保存某些元素，例如作为参数传递给不同函数的参数，则需要向应用程序流添加持久存储组件并将事件相互关联。例如，亚马逊提供了一个名为**AWS Step Functions**的附加工具，用于协调和管理无服务器应用程序的所有微服务和分布式组件的状态。

+   **资源限制：** 无服务器计算不适用于某些工作负载或用例，特别是高性能工作负载和云提供商强加的资源使用限制（例如，AWS 限制 Lambda 函数的并发运行次数）。这两者都是由于在有限和固定时间内提供所需服务器数量的困难。

+   **调试和监控：** 如果依赖于非开源解决方案，开发人员将依赖供应商来调试和监控应用程序，因此将无法使用额外的分析器或调试器详细诊断任何问题。因此，他们将不得不依赖于各自提供商提供的工具。

# 另请参阅

正如我们所见，使用无服务器架构的参考点是 AWS 框架（[`aws.amazon.com/`](https://aws.amazon.com/)）。在上述网址中，您可以找到大量信息和教程，包括本节中描述的示例。