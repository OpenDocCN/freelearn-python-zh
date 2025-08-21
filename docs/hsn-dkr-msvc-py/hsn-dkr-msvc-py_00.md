# 前言

软件的演进意味着系统变得越来越庞大和复杂，使得一些传统的处理技术变得无效。近年来，微服务架构作为一种有效的处理复杂 Web 服务的技术，获得了广泛的认可，使更多的人能够在同一个系统上工作而不会相互干扰。简而言之，它创建了小型的 Web 服务，每个服务解决一个特定的问题，并通过明确定义的 API 进行协调。

在本书中，我们将详细解释微服务架构以及如何成功运行它，使您能够在技术层面上理解架构以及理解架构对团队和工作负载的影响。

对于技术方面，我们将使用精心设计的工具，包括以下内容：

+   **Python**，用于实现 RESTful Web 服务

+   **Git**源代码控制，跟踪实现中的更改，以及**GitHub**，共享这些更改

+   **Docker**容器，以标准化每个微服务的操作

+   **Kubernetes**，用于协调多个服务的执行

+   **云服务**，如 Travis CI 或 AWS，利用现有的商业解决方案来解决问题

我们还将涵盖在微服务导向环境中有效工作的实践和技术，其中最突出的是以下内容：

+   **持续集成**，以确保服务具有高质量并且可以部署

+   **GitOps**，用于处理基础设施的变更

+   **可观察性**实践，以正确理解实时系统中发生的情况

+   **旨在改善团队合作的实践和技术**，无论是在单个团队内还是跨多个团队之间

本书围绕一个传统的单体架构需要转移到微服务架构的示例场景展开。这个示例在《第一章》*进行迁移-设计、计划、执行*中有描述，并贯穿整本书。

# 本书适合对象

本书面向与复杂系统打交道并希望能够扩展其系统开发的开发人员或软件架构师。

它还面向通常处理已经发展到难以添加新功能并且难以扩展开发的单体架构的开发人员。本书概述了传统单体系统向微服务架构的迁移，提供了覆盖所有不同阶段的路线图。

# 本书涵盖的内容

《第一部分》*微服务简介*介绍了微服务架构和本书中将使用的概念。它还介绍了一个示例场景，贯穿全书。

《第一章》*进行迁移-设计、计划、执行*，探讨了单体架构和微服务之间的差异，以及如何设计和规划从前者到后者的迁移。

《第二部分》*设计和操作单个服务-创建 Docker 容器*，讨论了构建和操作微服务，涵盖了从设计和编码到遵循良好实践以确保其始终高质量的完整生命周期。

《第二章》*使用 Python 创建 REST 服务*，介绍了使用 Python 和高质量模块实现单个 Web RESTful 微服务。

《第三章》*使用 Docker 构建、运行和测试您的服务*，向您展示如何使用 Docker 封装微服务，以创建标准的、不可变的容器。

第四章《创建管道和工作流程》教你如何自动运行测试和其他操作，以确保容器始终具有高质量并且可以立即使用。

第三部分《使用多个服务：通过 Kubernetes 操作系统》转向下一个阶段，即协调每个单独的微服务，使它们作为一个整体在一致的 Kubernetes 集群中运行。

第五章《使用 Kubernetes 协调微服务》介绍了 Kubernetes 的概念和对象，包括如何安装本地集群。

第六章《使用 Kubernetes 进行本地开发》让您在本地 Kubernetes 集群中部署和操作您的微服务。

第七章《配置和保护生产系统》深入探讨了在 AWS 云中部署的生产 Kubernetes 集群的设置和操作。

第八章《使用 GitOps 原则》详细描述了如何使用 Git 源代码控制来控制 Kubernetes 基础设施定义。

第九章《管理工作流程》解释了如何在微服务中实现新功能，从设计和实施到部署到向世界开放的现有 Kubernetes 集群系统。

第四部分《生产就绪系统：使其在现实环境中运行》讨论了在现实集群中成功操作的技术和工具。

第十章《监控日志和指标》是关于监控活动集群的行为，以主动检测问题和改进。

第十一章《处理系统中的变更、依赖和秘密》关注如何有效地处理在集群中共享的配置，包括正确管理秘密值和依赖关系。

第十二章《跨团队协作和沟通》关注独立团队之间的团队合作挑战以及如何改善协作。

# 为了充分利用本书

本书使用 Python 进行编码，并假定读者能够熟练阅读这种编程语言，尽管不需要专家级水平。

本书中始终使用 Git 和 GitHub 进行源代码控制和跟踪更改。假定读者熟悉使用它们。

熟悉 Web 服务和 RESTful API 对于理解所呈现的不同概念是有用的。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packtpub.com/support](https://www.packtpub.com/support)并注册，以便直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packt.com](http://www.packt.com)。

1.  选择“支持”选项卡。

1.  单击“代码下载”。

1.  在搜索框中输入书名并按照屏幕上的说明操作。

下载文件后，请确保使用以下最新版本解压或提取文件夹：

+   Windows 系统使用 WinRAR/7-Zip

+   Mac 系统使用 Zipeg/iZip/UnRarX

+   Linux 系统使用 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python`](https://github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有其他代码包，来自我们丰富的图书和视频目录，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 上找到。去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。您可以在这里下载：[`static.packt-cdn.com/downloads/9781838823818_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781838823818_ColorImages.pdf)。

# 代码实例

您可以在此处查看本书的代码实例视频：[`bit.ly/34dP0Fm`](http://bit.ly/34dP0Fm)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这是一个例子：“这将生成两个文件：`key.pem` 和 `key.pub`，带有私钥/公钥对。”

代码块设置如下：

```py
class ThoughtModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50))
    text = db.Column(db.String(250))
    timestamp = db.Column(db.DateTime, server_default=func.now())
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目会以粗体显示：

```py
# Create a new thought
new_thought = ThoughtModel(username=username, text=text, timestamp=datetime.utcnow())
db.session.add(new_thought)
```

任何命令行输入或输出都以以下方式书写：

```py
$ openssl rsa -in key.pem -outform PEM -pubout -out key.pub
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这种方式出现在文本中。这是一个例子：“从管理面板中选择系统信息。”

警告或重要说明会以这种方式出现。

提示和技巧会以这种方式出现。
