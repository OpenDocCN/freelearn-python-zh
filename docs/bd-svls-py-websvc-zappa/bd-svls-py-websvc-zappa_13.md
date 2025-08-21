# 第十三章：评估

# 第一章，无服务器的亚马逊网络服务

1.  在无服务器架构上部署应用程序只是将应用程序交给亚马逊基础设施。因此，有以下好处：

+   亚马逊将负责自动扩展

+   不需要服务器管理流程

+   在成本方面也有很大的差异，您按照基于执行时间的使用量付费。

+   它提供高可用性

1.  **Amazon Simple Storage Service** (**S3**)是亚马逊提供的存储服务。AWS Lambda 支持内联代码执行，您可以直接从其 Web 界面编写代码。它还支持从 Amazon S3 存储桶中获取代码库，您可以将代码库放入 ZIP 格式的构建包中。Zappa 有一个命令来生成应用程序的 ZIP 包。

# 第二章，开始使用 Zappa

1.  这是由 gun.io ([`www.gun.io/`](https://www.gun.io/))开发的开源工具，用于自动化在 AWS 基础设施上创建无服务器环境的手动过程。

1.  Zappa 通过在`zappa_setttings.json`中添加 AWS VPC 子网和安全组 ID 来提供配置 AWS **VPC** (**虚拟私有云**)的简单方法。

# 第三章，使用 Zappa 构建 Flask 应用

1.  亚马逊 API Gateway 是一个连接其他 AWS 服务的服务。API Gateway 为移动和 Web 应用程序提供了与其他 AWS 服务连接的 RESTful 应用程序接口。在我们的案例中，Zappa 配置了 API Gateway 接口，以代理请求来调用 AWS Lambda。

1.  Zappa 根据`zappa_settings.json`文件的配置执行部署操作。Zappa 使用`function_name`指向 Flask 应用对象，以便在 AWS Lambda 和 API Gateway 上配置应用程序。

# 第四章，使用 Zappa 构建基于 Flask 的 REST API

1.  **JWT** (**JSON Web Token**)提供了一种简单的方式来保护应用程序免受未经授权的访问。根据身份验证标头中提供的 JWT 令牌，可以授权对 API 的访问。

1.  `function_name`指示了 Flask 应用对象的模块路径。它帮助 Zappa 配置 Flask 应用程序及其路由与 API Gateway。

# 第五章，使用 Zappa 构建 Django 应用

1.  Amazon CloudFront 是一个托管的网络服务，可以通过互联网高速传送静态和动态网络内容。亚马逊在全球各地有各种数据中心，这些数据中心被称为边缘位置，因此 AWS CloudFront 使用这些边缘位置以最小的延迟传送网络内容，并提高应用程序的性能。

1.  Pipenv 是一个用于管理 Python 包的打包工具。它也被**Python.org** ([`www.python.org/`](https://www.python.org/))推荐。它有助于维护包和依赖项以及它们的版本。因此，它有助于开发和维护稳定版本的应用程序。

# 第六章，使用 Zappa 构建 Django REST API

1.  Django Rest Framework 是一个用于开发基于 Django 应用的 RESTful API 的库。它有一个标准模式来在 Django 模型上实现 API。它为开发人员提供了许多功能，以简单的方式实现和管理 API。

1.  Django-storage 是一个用于实现 Django 应用程序的自定义存储的库。它遵循 Django 的标准以持久化数据。

# 第七章，使用 Zappa 构建 Falcon 应用

1.  与其他 Python 框架相比，Falcon 框架具有很高的基准。它旨在以非常优化的方式编写 RESTful API。

1.  Peewee 库遵循 Django 的模式来创建数据库表并执行 CRUD 操作。它提供许多功能，如高性能、轻量级和较少的复杂性。SQLAlchemy 有一点学习曲线和复杂性。Peewee 可以被认为是一个小型/中型的应用程序或微服务。

1.  调度是在特定时间段执行程序的定义机制。因此，它与许多场景一起使用，其中我们需要执行程序或脚本以执行特定时间。例如，更新天气信息，发送通知警报等。

# 第八章，带 SSL 的自定义域

1.  AWS Route 53 是亚马逊的托管服务。它提供域名注册服务，将互联网流量路由到特定域的 AWS 资源，并为正在运行的 AWS 资源创建健康检查点。

1.  **域名服务器**（**DNS**）是一种维护和转换域名到**Internet Protocol**（**IP**）的机制，因为计算机通过 IP 地址进行通信并且很难记住。因此 DNS 有助于管理域名与 IP 地址的对应关系。

1.  ACM 根据域名生成 SSL 证书。如果您在域名上使用 SSL 证书，它将启用 HTTPS 协议，用于通过您的域进行过渡。HTTPS 是一种安全协议，它加密了互联网上传输的数据，并为通过您的域传输的机密信息提供了安全性。

# 第九章，在 AWS Lambda 上执行异步任务

1.  AWS SNS 是一个 Web 服务，提供发布和订阅模式的消息实现。它支持各种资源订阅通道并获取发布的消息。它可以用于管理和实现应用程序的通知服务。还有许多其他功能，可以考虑将 AWS SNS 服务用于应用程序开发。

1.  AWS SNS 用于发布和订阅模式。它支持将 AWS Lambda 注册为订阅者。它可以使用发布的消息上下文调用 AWS Lambda 函数。

# 第十章，高级 Zappa 设置

1.  AWS Lambda 旨在提供无服务器基础架构。它在调用请求时实例化上下文，然后在提供请求后销毁自身。AWS Lambda 会为初始启动和上下文设置添加一点时间延迟。为了避免这种情况，您可以通过使用 AWS CloudWatch 设置计划触发器来保持 Lambda 实例处于热状态。Zappa 默认提供了这个功能，您可以通过将`keep_warm`属性设置为`false`来关闭此功能。

1.  **跨域资源共享**（**CORS**）是一种机制，允许一个域从不同的域访问受限资源。

1.  Zappa 提供了一种管理大型项目的简单方法，因为 AWS Lambda 在上传构建包时有 50MB 的限制，但也有一个选项可以从 Amazon S3 服务更大的构建包。在 Zappa 设置中，您可以将`slim_handler`设置为`true`，这将在 Amazon S3 上上传构建包，然后针对在 Amazon S3 上上传的构建包配置 AWS Lambda。

# 第十一章，使用 Zappa 保护无服务器应用程序

1.  API Gateway 授权者是一种保护 API 资源的机制。API Gateway 授权者生成一个 API 密钥，可以绑定到任何资源。一旦绑定了 API 资源，API Gateway 将限制任何带有`x-api-key`头的 HTTP 请求。

1.  AWS Lambda 具有**死信队列**（**DLQ**）的功能，它使开发人员能够监视未知的失败。它可以配置为 AWS Lambda 函数中的 AWS SNS 或 SQS 的 DLQ。AWS Lambda 将在配置的 AWS SNS 或 SQS ARN 上发布失败事件。

1.  AWS 虚拟私有云创建了一个隔离的虚拟网络层，您可以在其中配置所有 AWS 资源。AWS VPC 将限制从 VPC 网络外部的访问，并启用安全层。

# 第十二章，Zappa 与 Docker

1.  Docker 容器是基本 Linux 系统的虚拟实例，它使您能够在隔离的环境中执行操作。Docker 容器具有所有基本配置，如网络、文件系统和操作系统级实用程序。

1.  一个 Docker 镜像是一个带有所需软件包的实际操作系统镜像。您也可以创建自己的镜像并将其发布到 Docker 仓库。一个 Docker 容器是 Docker 镜像的一个实例。您可以使用 Docker 镜像创建*N*个容器。
