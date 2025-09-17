# 第十章：10. 部署

## 学习目标

到本章结束时，你将能够：

+   解释将应用程序部署到云的过程

+   解释 SaaS、PaaS 和 IaaS 之间的区别

+   在开发和生产环境之间设置不同的配置

+   设置 Heroku 云平台

+   安装和配置 Heroku Postgres

+   使用 Heroku 命令行界面（Heroku CLI）部署应用程序

+   设置 Postman 环境变量

在本章中，我们将部署我们的应用程序到 Heroku，并使用 Postman 进行测试。

## 简介

在上一章中，我们向我们的 Smilecook 应用程序添加了缓存和速率限制功能。这两个功能非常有用，尤其是在我们处理大量流量时。缓存和速率限制可以提高响应速度，也可以提高安全性。

在本章中，我们将讨论如何将我们的应用程序部署到云服务器。部署应用程序就像出版一本书或发布一部电影。这就像将我们的应用程序推向市场。如今，许多云服务提供商提供免费使用配额。只要资源使用量低于一定阈值，它们允许开发者免费将其应用程序部署到其云平台。对于我们的 Smilecook 应用程序，我们只需要对代码和配置文件进行一些小的修改。其他一切将由云平台处理。你很快就会看到这是多么简单。

我们首先将对应用程序代码进行一些小的修改，以区分生产环境和开发环境配置。然后，我们将讨论 Heroku 云服务平台，我们将在该平台上部署 Smilecook 应用程序。我们将向您介绍 Heroku 云服务平台中的账户注册、配置和部署过程。

部署完成后，我们将使用 Postman 直接在生产环境中测试 API。这不是很令人兴奋吗？！无需多言，让我们开始吧。

## 部署

**部署**的目的是什么？我们之前编写的 API 应用程序只是在本地机器上运行代码。我们可以使用本地机器上的一个端口，从客户端向本地服务器发送请求。这对于开发目的来说很好。我们可以在开发环境中快速测试和调整我们的应用程序。然而，我们的本地机器并不是作为服务器来使用的；其他人无法访问它。他们也不能向托管在我们本地机器上的 API 发送 HTTP 请求。

如果我们想将此 API 服务对外开放，我们需要将其托管在服务器上。服务器应连接到互联网，拥有域名和 URL，以便其他人可以访问它。

将应用程序从本地机器迁移到运行在互联网上的服务器称为部署。这涉及到环境设置、依赖包安装和构建 Web 服务器等工作。

## 比较 SaaS、PaaS 和 IaaS

在过去，建立自己的网络服务器成本高昂。需要考虑的因素很多，包括网络连接、存储、服务器配置和操作系统设置。如今，云计算服务已经出现，提供所有基础设施服务，这显著降低了成本，尤其是对于个人开发者和中小型企业。云计算服务主要分为三大类。这些是**软件即服务**（**SaaS**）、**平台即服务**（**PaaS**）和**基础设施即服务**（**IaaS**）。每种服务都有其优缺点，这些将在本节中讨论。

**IaaS**：用户无需购买自己的服务器、软件、网络设备等。这些基础设施作为服务提供，用户无需关心设置和维护。他们仍然有能力配置这些服务，例如安装软件和设置防火墙。IaaS 的例子包括**AWS EC2**和**Google Compute Engine**（**GCE**）。

与过去相比，这种 IaaS 模型可以大大降低硬件和网络设置成本，以及与空间和资源相关的所有其他成本。个人开发者或中小型企业通常不需要那么多的系统资源。因此，这种模式允许他们按需租用基础设施作为服务；他们只需支付所需的资源费用。

+   优点：开发者拥有更大的灵活性。IaaS 为应用程序运行提供必要的计算资源。开发者可以根据应用程序的需求轻松请求额外的资源，或减少资源。这是易于定制的。

+   缺点：开发者需要花费时间学习如何根据他们的需求配置云平台。

**PaaS**：PaaS 位于 SaaS 和 IaaS 之间。用户无需管理和维护基础设施。服务提供商已经将这些基础设施和相关服务打包成一个平台，并以服务的形式出租给用户。用户无需担心后端设置，也不必担心扩展服务器数量和负载均衡等方面。用户（开发者）只需专注于他们的开发，并根据云平台相应地部署他们的工作。PaaS 的例子包括 Heroku、Windows Azure 和 AWS Elastic Beanstalk。

+   优点：减少了设置时间。通过利用平台提供的服务，开发者可以专注于开发。

+   缺点：可能会产生不必要的费用。与 IaaS 相比，PaaS 在基础设施设置和配置方面灵活性较低，因为你对基础设施的控制较少。由于整个平台都被打包成服务，一些未使用的打包资源可能会浪费。在这种情况下，费用可能比 IaaS 高。

**SaaS**：SaaS 基本上是指互联网上可用的 Web 应用。用户不需要维护软件。软件作为服务提供。一个非常典型的例子是 Gmail。SaaS 的例子包括 Dropbox、Salesforce 和 Slack。

+   优点：成本较低，因为我们不需要关心硬件购买和其他设置成本。如果用户有可以通过此服务解决的问题，SaaS 可能是最简单、最有效的解决方案。

+   缺点：由于大量用户数据将存储在云端平台，可能会对数据安全产生一些担忧。此外，一旦应用部署，我们还需要考虑服务的可用性。

作为个人开发者，我们需要一个稳定且可扩展的服务器来部署我们的应用。PaaS 是最佳选择。它为应用提供运行的计算平台，开发者无需担心硬件维护，因为服务提供商会处理所有这些。因此，这是一个节省时间和成本的开发者解决方案。开发者可以专注于开发优秀的软件。

## Heroku 平台

Heroku 是一个流行的 PaaS。我们可以在那里部署我们的 API，以便全世界任何人都可以访问。而且它不仅支持 Python，还支持其他编程语言，包括 Ruby 和 Go。

Heroku 为开发者提供免费计划，以便他们在那里部署和测试他们的应用。当然，他们也有付费计划，以及许多更强大的功能，可以使我们的 API 更加安全和高效。稍后，如果您需要这些强大的功能和系统资源来支持您的应用，您可以考虑这一点。但现在，出于教学目的，免费计划已经足够好。

#### 注意

除了 Heroku，还有其他云服务提供商。云服务市场的一些领导者包括**亚马逊网络服务**（**AWS**）、**谷歌云平台**（**GCP**）、IBM Cloud、Microsoft Azure 和 Rackspace Cloud。

## Smilecook 中的配置处理

大多数应用都需要多个配置；至少需要一个用于生产服务器，另一个用于开发使用。它们之间会有所不同，例如调试模式、密钥和数据库 URL。我们可以使用一个始终加载的默认配置，以及为生产服务器和开发环境分别创建的配置，以便根据环境继承默认配置。对于特定环境的配置，我们将创建两个新的类——`DevelopmentConfig`和`ProductionConfig`。

### 练习 63：生产环境和开发环境的配置处理

在这个练习中，我们将把应用程序配置在开发和生产环境之间分开。对于如`DEBUG`这样的配置，我们将需要两个环境的不同值。数据库 URL 也是如此。因此，我们将创建两组配置，`DevelopmentConfig`和`ProductionConfig`。前者用于开发环境和系统增强，而后者将在生产环境中运行。按照以下步骤完成练习：

1.  首先，在`config.py`中添加一个默认配置，它将在所有环境中使用：

    ```py
    import os
    class Config:
        DEBUG = False

        SQLALCHEMY_TRACK_MODIFICATIONS = False

        JWT_ERROR_MESSAGE_KEY = 'message'

        JWT_BLACKLIST_ENABLED = True
        JWT_BLACKLIST_TOKEN_CHECKS = ['access', 'refresh']

        UPLOADED_IMAGES_DEST = 'static/images'

        CACHE_TYPE = 'simple'
        CACHE_DEFAULT_TIMEOUT = 10 * 60

        RATELIMIT_HEADERS_ENABLED = True
    ```

1.  在`Config`类之后添加`DevelopmentConfig`：

    ```py
    class DevelopmentConfig(Config):
        DEBUG = True
        SECRET_KEY = 'super-secret-key'
        SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://your_name:your_password@localhost:5432/smilecook'
    ```

    新的`DevelopmentConfig`类扩展了父类`Config`。将`DEBUG`值设置为`True`。这样我们就可以在开发过程中看到错误信息。

1.  在`Development Config`类之后添加`ProductionConfig`：

    ```py
    class ProductionConfig(Config):
        SECRET_KEY = os.environ.get('SECRET_KEY')
        SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    ```

    这里的`ProductionConfig`类也扩展了父类`Config`。与`DevelopmentConfig`类类似，我们在这里设置了`SECRET_KEY`和`SQLALCHEMY_DATABASE_URI`。在生产环境中，这些值是从环境变量中获取的。我们将在稍后教你如何在云平台上设置这些值。

1.  在`app.py`中导入`os`：

    ```py
    import os
    ```

1.  在`app.py`中，进行以下更改以动态获取配置：

    ```py
    def create_app():
        env = os.environ.get('ENV', 'Development')
        if env == 'Production':
            config_str = 'config.ProductionConfig'
        else:
            config_str = 'config.DevelopmentConfig'
        app = Flask(__name__)
        app.config.from_object(config_str)
        ...

        return app
    ```

    `ENV`环境变量将通过`os.environ.get`获取。如果是`Production`，则使用生产环境配置。此外，还将使用开发环境配置。

1.  右键点击 PyCharm 并运行应用程序。因为我们没有在本地机器上设置`ENV`环境变量，所以 Flask 将选择`config.DevelopmentConfig`并执行它。我们可以从输出中看到**调试模式：开启**：![图 10.1：在开发环境中运行应用程序    ![图片](img/C15309_10_01.jpg)

###### 图 10.1：在开发环境中运行应用程序

因此，我们已经将生产环境和开发环境之间的配置分开。将来，如果有两个环境共享的通用配置，我们将它们放在`Config`类中。否则，它们应该放在相应的`DevelopmentConfig`或`ProductionConfig`类下。

### 练习 64：添加一个预发布配置类

为了便于内部测试，在这个练习中，我们需要添加一个`StagingConfig`类。这个配置将扩展通用`Config`类。预发布环境与生产环境不会相差太多，因为它主要是为了测试而设计，以模仿生产环境。并且我们将从环境变量中获取密钥和数据库 URI：

1.  在`config.py`中创建一个扩展`Config`的`StagingConfig`类：

    ```py
    class StagingConfig(Config):
        SECRET_KEY = os.environ.get('SECRET_KEY')
        SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    ```

1.  在`app.py`中修改`StagingConfig`的条件语句：

    ```py
        if env == 'Production':
            config_str = 'config.ProductionConfig'
        elif env == 'Staging':
            config_str = 'config.StagingConfig'
        else:
            config_str = 'config.DevelopmentConfig'
    ```

因此，我们已经为预发布环境设置了配置。但尚未完成，因为需要从云服务器获取环境变量。接下来，我们将开始处理云平台，**Heroku**。

## Heroku 应用

在我们将应用部署到 Heroku（云平台）之前，我们首先会在那里创建一个账户并设置环境。我们将创建一个新的 Heroku 应用。然后，我们需要在 Heroku 上安装 Postgres 数据库。安装过程可以在 Heroku 平台上完成；一切都是集成的。最后，我们设置虚拟环境变量，例如数据库 URL 和密钥。一旦所有这些前期准备工作完成，我们就会开始部署过程。

### 练习 65：在 Heroku 创建新应用

在这个练习中，我们首先注册 Heroku 账户。然后，我们将在上面创建一个新应用。Heroku 提供了一个用户界面友好、易于遵循的设置流程。我们只需点击几个按钮就完成了。由于 Heroku 是 PaaS，我们不需要管理任何硬件或设置操作系统。这些都是 Heroku 负责的：

1.  访问 Heroku 网站，[`www.heroku.com/`](https://www.heroku.com/)，并点击**注册**：![图 10.2：访问 Heroku 网站

    ![img/C15309_10_02.jpg]

    ###### 图 10.2：访问 Heroku 网站

1.  签到过程完成后，点击**登录**并访问仪表板。点击**创建新应用**以在 Heroku 中创建新应用：![图 10.3：登录并访问 Heroku 仪表板

    ![img/C15309_10_03.jpg]

    ###### 图 10.3：登录并访问 Heroku 仪表板

1.  输入应用名称，然后选择服务器区域（目前，唯一的选择是美国和欧洲；请选择离目标用户更近的一个）。然后，点击**创建应用**继续：

![图 10.4：输入应用名称并选择服务器区域

![img/C15309_10_04.jpg]

###### 图 10.4：输入应用名称并选择服务器区域

#### 注意

应用名称将用于 Heroku 提供的应用 URL 中，例如，`https://{app_name}.herokuapp.com/`。用户可以使用此 URL 访问我们的 API。

应用创建后，我们可以看到应用管理界面，大致如下：

![图 10.5：Heroku 应用管理界面

![img/C15309_10_05.jpg]

###### 图 10.5：Heroku 应用管理界面

应用管理界面提供了我们了解应用状态的信息：

**概览**：显示我们产生的费用或其他协作者的活动

**资源**：用于管理附加组件和**Procfile**设置

**部署**：用于选择部署方法

**指标**：用于显示应用的指标

**活动**：用于跟踪用户活动

**访问**：用于管理协作者的访问权限

**设置**：包括环境变量配置、构建包设置和其他高级功能

#### 注意

Heroku 平台的核心是能够使用轻量级容器 Dynos 运行应用程序。容器化是将您的应用程序代码、配置和依赖项打包成一个单一对象的标准方式。容器化可以减少对管理硬件、虚拟机或环境设置的负担。

一旦应用程序创建完成，我们将在 Heroku 中安装 Postgres 仓库，并通过 Heroku 扩展直接安装它。

## Heroku 扩展

Heroku 拥有一个丰富的扩展库。扩展类似于插件，为开发、扩展和运行您的应用程序提供工具和服务，包括数据存储、监控、日志记录、分析和安全。对于我们的 Smilecook 应用程序，我们将使用 Heroku 提供的 Heroku Postgres，这是一个基于 PostgreSQL 的可靠且强大的数据库服务。入门级免费，提供 10,000 行限制，并保证 99.5% 的正常运行时间。这适合开发爱好应用。

### 练习 66：安装 Heroku Postgres

在这个练习中，我们将安装 Heroku Postgres。与从 Postgres 官方网站安装相比，从 Heroku 安装 Postgres 更方便。我们只需在 **Heroku 扩展** 的 **数据存储** 类别中直接选择 **Heroku Postgres** 来安装。Heroku 提供了一个后端管理界面，使我们能够一目了然地查看数据库状态：

1.  切换到 Heroku 的 **资源** 选项卡，然后右键单击 **查找更多扩展** 按钮：![图 10.6：切换到 Heroku 的资源选项卡    ![图 10.06](img/C15309_10_06.jpg)

    ###### 图 10.6：切换到 Heroku 的资源选项卡

1.  在 **扩展** 页面上，点击 **数据存储** 并选择 **Heroku Postgres**：![图 10.7：Heroku 的扩展页面    ![图 10.07](img/C15309_10_07.jpg)

    ###### 图 10.7：Heroku 的扩展页面

1.  然后，点击 **安装 Heroku Postgres** 以在我们的云服务器上安装扩展：![图 10.8：安装 Heroku Postgres 扩展    ![图 10.08](img/C15309_10_08.jpg)

    ###### 图 10.8：安装 Heroku Postgres 扩展

1.  选择默认的 **Hobby Dev - Free Plan**。此计划免费。在 **要部署的应用** 中输入我们在上一个练习中使用的应用程序名称，然后点击 **部署扩展**：![图 10.9：选择 Heroku Postgres 扩展计划    ![图 10.09](img/C15309_10_09.jpg)

    ###### 图 10.9：选择 Heroku Postgres 扩展计划

1.  完成这些操作后，我们可以在 **扩展** 页面上检查 **Heroku Postgres** 是否已安装：![图 10.10：检查 Heroku Postgres 是否已安装    ![图 10.10](img/C15309_10_10.jpg)

    ###### 图 10.10：检查 Heroku Postgres 是否已安装

1.  然后，点击 **Heroku Postgres 扩展** 以进入管理页面：

![图 10.11：Heroku Postgres 管理页面![图 10.11](img/C15309_10_11.jpg)

###### 图 10.11：Heroku Postgres 管理页面

**概览**允许我们检查数据库状态、利用率等。**耐用性**允许我们管理数据安全和备份。**设置**存储数据库凭据和其他高级设置。**数据片段**允许您使用 SQL 命令在线查询数据库数据。您可以在那里导出或分享结果。

如您所见，在 Heroku 上安装 Postgres 非常简单；只需几个步骤。接下来，我们将着手在云平台上设置环境变量。

## 为 Heroku 应用设置环境变量

我们之前修改了`config.py`并在其中添加了`ProductionConfig`。现在我们必须在 Heroku 中添加环境变量，包括密钥和数据库 URL。除此之外，别忘了 Mailgun API 密钥和 Mailgun 域名。我们将在下一个练习中将所有这些一起设置。

### 练习 67：设置应用环境变量

在这个练习中，我们将设置生产环境中的环境变量。幸运的是，因为我们使用的是 Heroku Postgres，数据库 URL 环境变量已经为我们设置好了。我们只需要设置`ENV`、`SECRET_KEY`、`MAILGUN KEY`和`DOMAIN`。然后，一旦设置完成，在`Deploy`代码完成后，应用程序将读取`App config`中新增的环境变量：

1.  在 PyCharm 的 Python 控制台中，使用以下两行代码生成密钥：

    ```py
    >>>import os
    >>>os.urandom(24)
    ```

    #### 注意

    密钥应该尽可能随机。有很多随机生成器我们可以利用。但可能最简单的方法是在 PyCharm 的 Python 控制台中生成它。

1.  前往**设置**选项卡，并设置**ENV**、**MAILGUN_API_KEY**、**MAILGUN_DOMAIN**和**SECRET_KEY**环境变量，如下所示：

![图 10.12：在 Heroku 中设置环境变量![图片](img/C15309_10_12.jpg)

###### 图 10.12：在 Heroku 中设置环境变量

现在我们已经在 Heroku 上完成了必要的准备工作，我们将直接进入部署过程。

### 使用 Heroku Git 进行部署

Heroku 提供了一份关于如何部署我们的应用的指南。该指南可以在**部署**选项卡中找到。它主要分为三个部分。它们是**安装 Heroku CLI**、**创建一个新的 Git 仓库**和**部署您的应用**。具体细节如下：

![图 10.13：使用 Heroku Git 指南进行部署![图片](img/C15309_10_13.jpg)

###### 图 10.13：使用 Heroku Git 指南进行部署

**部署**选项卡中的指南有三个部分：

**安装 Heroku CLI**

+   `heroku login` – 使用 Heroku CLI 工具登录 Heroku。

**创建一个新的 Git 仓库**

+   `cd my-project/` – 切换到`my-project`文件夹。

+   `git init` – 初始化`git`，这是一个版本控制系统。我们很快就会讨论这个问题。

+   `heroku git:remote -a smilecook` – 将应用程序（Smilecook）仓库添加到本地 Git 的远程仓库列表中。

**部署你的应用程序**

+   `git add .` – 将所有文件和文件夹添加到当前目录及其子目录到 Git。

+   `git commit -am "make it better"` – 提交一个更改并插入提交信息`make it better`。

+   `git push heroku master` – 这将上传本地仓库内容到远程仓库，即 Heroku 中的仓库。一旦推送，Heroku 将运行应用程序启动程序。

在我们开始部署应用程序之前，还有一些术语需要解释。

### Git 是什么？

**Git** 是一个分布式版本控制系统。版本控制系统主要是一个可以跟踪你源代码每个版本的系统。源代码中的任何更改都将被记录在系统中。它允许开发者轻松地恢复到之前的版本。无需手动备份。

Git 还支持协作和其他高级功能。如果您感兴趣，可以访问官方 Git 网站了解更多信息：[`git-scm.com`](https://git-scm.com)。

### gitignore 是什么？

**gitignore** 是一个包含 Git 应该忽略的文件和文件夹列表的文件。列表中的文件和文件夹将不会存储在 Git 中。通常，我们会将环境配置、日志等内容包含在这个列表中。

### Procfile 是什么？

**Procfile** 是一个在 Heroku 应用程序启动过程中执行的文件。开发者将在此处放入他们希望在应用程序启动过程中让 Heroku 运行的命令。通常，我们会将设置脚本和服务器启动脚本放在这里。

### Gunicorn 是什么？

**Gunicorn** 是一个兼容各种 Web 应用程序的 Python WSGI HTTP 服务器。它可以作为 Web 服务器和 Web 应用程序之间的接口。Gunicorn 可以与多个 Web 服务器或启动多个 Web 应用程序进行通信。它是一个强大且快速的 HTTP 服务器。

现在我们已经了解了部署流程以及一些关键概念和术语，我们将在下一项练习中一起进行部署。

### 练习 68：设置 Git 和 Heroku CLI

在这个练习中，我们将部署我们的 Smilecook 应用程序到生产环境。我们首先下载并安装 Heroku CLI 和 Git，以便我们可以在本地机器上运行部署命令。然后，我们将添加`gitignore`文件以确保某些文件不会被上传到 Heroku。最后，我们将`main.py`和`Procfile`添加到项目的根目录中，然后将其部署到 Heroku：

1.  从 [`devcenter.heroku.com/articles/heroku-cli`](https://devcenter.heroku.com/articles/heroku-cli) 安装**Heroku CLI**。选择适合您操作系统的版本并下载它：![图 10.14：安装 Heroku CLI    ![图片](img/C15309_10_14.jpg)

    ###### 图 10.14：安装 Heroku CLI

1.  如果您还没有安装 Git，请从 [`git-scm.com/`](https://git-scm.com/) 安装：![图 10.15：安装 Git    ](img/C15309_10_15.jpg)

    ###### 图 10.15：安装 Git

1.  在 PyCharm 底部打开终端。运行 `git --version` 命令以确认 Git 已成功安装：

    ```py
    $ git --version
     git version 2.19.1 // You may see a different value inside the brackets depending on your OS
    ```

1.  右键点击在项目中创建一个 `.gitignore` 文件。此文件将包含我们不想添加到 Git 中的文件或文件夹列表：

    ```py
    static/images/avatars/*
    static/images/recipes/*
    .idea/
    venv/
    ```

    `static/images/avatars/*` – 我们不希望将之前章节中创建的所有测试图像上传到生产环境。

    `static/images/recipes/*` – 我们不希望将之前章节中创建的所有测试图像上传到生产环境。

    `.idea/` – 这是 IDE 项目特定设置文件夹。在生产环境中我们不需要它。

    `venv/` – 这是虚拟环境。

1.  登录您的 Heroku 账户：

    ```py
    $ heroku login
    ```

1.  然后，输入以下 `git init` 命令以初始化 Git。这是为了将版本控制添加到我们的项目中：

    ```py
    $ git init
    ```

1.  将 Heroku 仓库添加到 Git 远程仓库（请将 `your-heroku-app` 替换为您的 Heroku 应用名称）。

    ```py
    $ heroku git:remote -a your-heroku-app
    ```

    #### 注意

    在添加远程仓库之前，我们所有的更改只能提交到本地仓库。

1.  在 `requirements.txt` 文件中添加 `gunicorn` 包，它将成为我们的 HTTP 服务器：

    ```py
    gunicorn==19.9.0
    ```

1.  在项目根目录下创建 `main.py`。这将由 Gunicorn 执行以启动我们的 web 应用程序：

    ```py
    from app import create_app
    app = create_app()
    ```

1.  右键点击在项目根目录下创建一个文件。命名为 `Procfile`（不带扩展名），然后插入以下两个命令：

    ```py
    release: flask db upgrade
    web: gunicorn main:app
    ```

    此 `Procfile` 文件是用于 Heroku 在应用程序启动过程中运行的。第一行是要求 Heroku 在每次部署后运行 `flask db upgrade`。这是为了确保我们的数据库模式始终保持最新。

    第二行是为了让 Heroku 识别它为启动 web 服务器的任务。

1.  在 PyCharm 的 Python 控制台中运行 `git add .` 命令。这将把我们的源代码添加到 Git 中，用于版本控制和部署：

    ```py
    $ git add .
    ```

1.  运行 `git commit` 命令以提交我们的源代码。`-a` 参数告诉 Git 将已修改或删除的文件放入暂存区。`-m` 参数用于包含提交信息：

    ```py
    $ git commit -am "first commit"
    ```

1.  使用 `git push` 将源代码推送到 Heroku 仓库以部署应用程序：

    ```py
    $ git push heroku master
    ```

    Heroku 将自动设置环境。我们可以看到以下输出：

    ![图 10.16：将应用程序部署到 Heroku    ](img/C15309_10_16.jpg)

###### 图 10.16：将应用程序部署到 Heroku

#### 注意

在部署过程中，如果我们想了解更多关于幕后发生的事情，可以通过点击右上角的 **更多** 按钮，然后点击 **查看日志** 来检查应用程序日志。

![图 10.17：将应用程序部署到 Heroku](img/C15309_10_17.jpg)

###### 图 10.17：将应用程序部署到 Heroku

从前面的日志中，我们可以看到数据库升级后，将运行 Gunicorn。最后，你可以看到消息 **状态从启动变为运行**。

我们已成功将 Smilecook 应用程序部署到 Heroku，这意味着它已准备好为公众提供服务。稍后，我们将使用 Postman 进行测试。

#### 注意

在未来，如果有新版本，我们只需要使用三个命令来重新部署应用程序。首先，使用 `git add .` 将我们的源代码添加到 Git 中，然后使用 `git commit -am "make it better"`。最后，使用 `git push heroku master` 将源代码推送到 Heroku。

### 练习 69：在 pgAdmin 中检查 Heroku Postgres 表

在上一个练习中，我们完成了部署。现在我们需要检查数据库中是否已创建了表。因此，在这个练习中，我们将使用 `pgAdmin` 连接到 Heroku Postgres：

1.  获取 Heroku Postgres 数据库的凭据，转到 **Add-ons** > **Settings**，然后点击 **View Credentials**，你将看到以下屏幕：![图 10.18：获取 Heroku Postgres 数据库的凭据

    ![img/C15309_10_18.jpg]

    ###### 图 10.18：获取 Heroku Postgres 数据库的凭据

1.  在 **Servers** 上右键单击，然后在 pgAdmin 中创建新服务器：![图 10.19：在 pgAdmin 中创建新服务器

    ![img/C15309_10_19.jpg]

    ###### 图 10.19：在 pgAdmin 中创建新服务器

1.  在 **General** 选项卡中，将服务器命名为 **Heroku**：![图 10.20：在常规选项卡中输入服务器的名称

    ![img/C15309_10_20.jpg]

    ###### 图 10.20：在常规选项卡中输入服务器的名称

1.  在 **Connection** 选项卡中，输入凭据，包括 **主机名/地址**、**端口**、**维护数据库**、**用户名** 和 **密码**，然后点击 **保存**：![图 10.21：将凭据添加到连接选项卡

    ![img/C15309_10_21.jpg]

    ###### 图 10.21：将凭据添加到连接选项卡

1.  现在，在 pgAdmin 中检查数据库表。转到 **Heroku** >> **Databases** >> (你的数据库名称) >> **Schemas** >> **Public** >> **Tables** 以验证此操作：![图 10.22：在 pgAdmin 中检查数据库表

    ![img/C15309_10_22.jpg]

###### 图 10.22：在 pgAdmin 中检查数据库表

现在，我们可以查看数据库中是否已创建了表。如果你可以看到表已成功创建，我们可以继续到下一步，即使用 Postman 测试我们的 API。

## 在 Postman 中设置变量

我们已成功将项目部署到 Heroku。现在您可以使用我们之前设置的保存请求在 Postman 中测试它们。然而，我们之前在 Postman 中保存的请求都是针对 localhost 运行的。我们不必逐个更改 URL 到生产 URL，我们可以利用 Postman 中的变量。我们可以在 Postman 中设置一个`url`变量，并将生产 URL 分配给它，然后从保存的请求中将 URL 替换为`{{url}}`。然后 Postman 将动态地为我们替换`{{url}}`为生产 URL。

### 练习 70：在 Postman 中设置变量

在这个练习中，我们将设置 Postman 中的变量，以便根据环境动态地结合适当的值。我们将设置 URL 作为一个变量，这样当我们处于开发环境测试时，我们只需更改 URL 变量为`http://localhost:5000`。如果我们处于生产环境测试，我们可以将其更改为`https://your_heroku_app.herokuapp.com`：

1.  将环境名称设置为`Smilecook`。然后，创建一个名为`url`的变量，其值为`https://your_heroku_app.herokuapp.com`。如果当前值未设置，它将自动采用初始值。请将`your_heroku_app`替换为您的 Heroku 应用名称，然后点击**更新**：![图 10.23：在 Postman 中添加环境变量    ](img/C15309_10_23.jpg)

    ###### 图 10.23：在 Postman 中添加环境变量

1.  一旦添加，通过点击右上角的眼睛图标来验证变量：![图 10.24：在 Postman 中验证环境变量    ](img/C15309_10_24.jpg)

    ###### 图 10.24：在 Postman 中验证环境变量

1.  在`UserList`请求中，将 URL 更新为`{{url}}/users`，然后在请求发送时点击`https://your_heroku_app.herokuapp.com/users`）：![图 10.25：在 URL 中使用环境变量    ](img/C15309_10_25.jpg)

###### 图 10.25：在 URL 中使用环境变量

Postman 是一个非常强大的测试工具。它甚至允许我们有效地在不同的环境中测试我们的 API 端点。在未来，如果您想在生产环境中测试其他 API 端点，您只需更改之前保存的请求中的 URL。在下一个活动中，我们将测试您的这一知识。

### 活动 18：在 Postman 中将 access_token 更改为变量

在上一个练习中，您学习了如何将 URL 更改为变量。在这个活动中，我们希望您对`access_token`也做同样的操作：

1.  通过使用之前保存的**POST Token**请求来获取访问令牌。

1.  在 Postman 中将`access_token`作为一个变量添加。

1.  测试一个需要访问令牌的 Smilecook API 端点。

    #### 注意

    本活动的解决方案可以在第 345 页找到。

太棒了。当你完成这个活动时，这意味着你已经部署并测试了 Smilecook API 的生产环境。这是本书的最后一项活动，我们很高兴你走到了这一步！

现在，我们将设置 Smilecook 前端网站，它将使用你刚刚开发的 API。

## 设置前端界面以与 Smilecook API 一起工作

请从 [`github.com/TrainingByPackt/Python-API-Development-Fundamentals/tree/master/Lesson10/Frontend`](https://github.com/TrainingByPackt/Python-API-Development-Fundamentals/tree/master/Lesson10/Frontend) 下载包含前端网站源代码的 `smilecook-vuejs` 文件夹。

1.  在 Heroku 平台上创建一个新应用，用于部署我们的前端网页界面：![图 10.26：在 Heroku 平台上创建新应用

    ![img/C15309_10_26.jpg]

    ###### 图 10.26：在 Heroku 平台上创建新应用

1.  应用程序创建后，我们转到 **设置** 选项卡，然后是 **配置变量**。在这里，我们将设置一个环境变量，该变量将用于存储后端 API URL：![图 10.27：设置环境变量

    ![img/C15309_10_27.jpg]

    ###### 图 10.27：设置环境变量

1.  将变量名设置为 `VUE_APP_API_URL`，并将后端 Smilecook API URL 插入此处。

1.  在 PyCharm 中打开 `smilecook-vuejs` 项目。

1.  在 PyCharm 控制台中，输入以下命令以登录到 Heroku CLI：

    ```py
    $ heroku login
    ```

1.  然后，初始化 `git` 并将 Heroku 仓库添加到 `git:remote` 仓库：

    ```py
    $ git init
    $ heroku git:remote -a your_heroku_app_name
    ```

1.  然后，将源代码添加到 `git` 中，提交，并将它们推送到 Heroku。

    ```py
    $ git add .
    $ git commit -am "make it better"
    $ git push heroku master
    ```

1.  部署完成后，你应该在屏幕上看到以下信息：

    ```py
    remote: -----> Compressing...
    remote:        Done: 30M
    remote: -----> Launching...
    remote:        Released v1
    remote:        https://your_heroku_app_name.herokuapp.com/ deployed to Heroku
    remote: 
    remote: Verifying deploy... done.
    To https://git.heroku.com/your_heroku_app_name.git
       59c4f7f..57c0642  master -> master
    ```

1.  在浏览器中输入 `https://your_heroku_app_name.herokuapp.com/`；我们可以看到前端界面已成功设置：![图 10.28：前端设置成功

    ![img/C15309_10_28.jpg]

###### 图 10.28：前端设置成功

现在，你可以使用这个前端网站界面与 Smilecook API 进行交互。

## 摘要

在本章中，我们成功地将 Smilecook API 部署到了 Heroku 云服务器。由于我们利用了 Heroku 提供的服务，部署过程非常简单。我们不需要担心购买硬件、设置服务器操作系统、将服务器连接到互联网等等。一切都有 Heroku 提供。云平台服务可以快速帮助开发者将他们的应用程序/API 部署到互联网上。这种简单的部署过程允许开发者专注于开发，而不是基础设施/平台设置。一旦 API 部署完成，互联网上的数百万用户可以通过他们的客户端应用程序连接到该 API。

当然，Heroku 只是众多云服务中的一种。至于应该选择哪种云服务，您应该考虑诸如成本、提供的附加服务和我们应用程序的规模等重要因素。我们不会限制您使用特定的平台。实际上，我们希望这本书能成为您作为专业开发者的旅程的起点。凭借您所学的基础知识，您应该能够探索和进一步发展新技能，并使用新工具构建更高级的 API。

恭喜！我们已经完成了整本书。您不仅学习了 API 是什么，而且还自己开发和部署了一个真实的 API 服务，Smilecook。在整个书中，您学习了如何设置开发环境、构建 API、与数据库交互、对象序列化、安全令牌、与第三方 API 交互、缓存，以及最终的部署。我们横向覆盖了许多不同的主题，同时也深入探讨了每个主题。除了学习理论外，您还在练习和活动中进行了实际的编码，并对您的作品进行了彻底的测试。

您的下一步应该包括通过参与开发项目来继续学习。最重要的是要有实际的开发经验，并保持好奇心。每当遇到问题时，都要寻找更好的解决方案。您不应该只满足于完成任务。相反，您应该致力于把事情做对。这正是将您带到下一个层次的关键。

我们希望您与我们一同享受了学习之旅。谢谢！
