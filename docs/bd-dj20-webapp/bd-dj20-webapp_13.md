# 第十三章：部署 Mail Ape

在本章中，我们将在**亚马逊网络服务**（**AWS**）云中的虚拟机上部署 Mail Ape。AWS 由许多不同的服务组成。我们已经讨论过使用 S3 和在 AWS 中启动容器。在本章中，我们将使用更多的 AWS 服务。我们将使用**关系数据库服务（RDS）**来运行 PostgreSQL 数据库服务器。我们将使用**简单队列服务（SQS）**来运行 Celery 消息队列。我们将使用**弹性计算云（EC2）**在云中运行虚拟机。最后，我们将使用 CloudFormation 来定义我们的基础设施为代码。

在本章中，我们将做以下事情：

+   分离生产和开发设置

+   使用 Packer 创建我们发布的 Amazon Machine Image

+   使用 CloudFormation 定义基础设施为代码

+   使用命令行将 Mail Ape 部署到 AWS

让我们首先分离我们的生产开发设置。

# 分离开发和生产

到目前为止，我们保留了一个需求文件和一个`settings.py`文件。这使得开发很方便。然而，我们不能在生产中使用我们的开发设置。

当前的最佳实践是每个环境使用单独的文件。然后每个环境的文件导入一个具有共享值的通用文件。我们将为我们的需求和设置文件使用这种模式。

让我们首先分离我们的需求文件。

# 分离我们的需求文件

为了分离我们的需求，我们将删除现有的`requirements.txt`文件，并用通用、开发和生产需求文件替换它。在删除`requirements.txt`之后，让我们在项目的根目录下创建`requirements.common.txt`：

```py
django<2.1
psycopg2<2.8
django-markdownify==0.3.0
django-crispy-forms==1.7.0
celery<4.2
django-celery-results<2.0
djangorestframework<3.8
factory_boy<3.0
```

接下来，让我们为`requirements.development.txt`创建一个需求文件：

```py
-r requirements.common.txt
celery[redis]
```

由于我们只在开发设置中使用 Redis，我们将在开发需求文件中保留该软件包。

我们将把我们的生产需求放在项目的根目录下的`requirements.production.txt`中：

```py
-r requirements.common.txt
celery[sqs]
boto3
pycurl
```

为了让 Celery 与 SQS（AWS 消息队列服务）配合工作，我们需要安装 Celery SQS 库（`celery[sqs]`）。我们还将安装`boto3`，Python AWS 库，和`pycurl`，Python 的`curl`实现。

接下来，让我们分离我们的 Django 设置文件。

# 创建通用、开发和生产设置

与我们之前的章节一样，在我们将设置分成三个文件之前，我们将通过将当前的`settings.py`重命名为`common_settings.py`然后进行一些更改来创建`common_settings.py`。

让我们将`DEBUG = False`更改为，以便没有新的设置文件可以*意外*处于调试模式。然后，让我们通过更新`SECRET_KEY = os.getenv('DJANGO_SECRET_KEY')`从环境变量中获取密钥。

在数据库配置中，我们可以删除所有凭据，但保留`ENGINE`（以明确表明我们打算在所有地方使用 Postgres）：

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
    }
}
```

接下来，让我们在`django/config/development_settings.py`中创建一个开发设置文件：

```py
from .common_settings import *

DEBUG = True

SECRET_KEY = 'secret key'

DATABASES['default']['NAME'] = 'mailape'
DATABASES['default']['USER'] = 'mailape'
DATABASES['default']['PASSWORD'] = 'development'
DATABASES['default']['HOST'] = 'localhost'
DATABASES['default']['PORT'] = '5432'

MAILING_LIST_FROM_EMAIL = 'mailape@example.com'
MAILING_LIST_LINK_DOMAIN = 'http://localhost'

EMAIL_HOST = 'smtp.example.com'
EMAIL_HOST_USER = 'username'
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_PORT = 587
EMAIL_USE_TLS = True

CELERY_BROKER_URL = 'redis://localhost:6379/0'
```

记得你需要将你的`MAILING_LIST_FROM_EMAIL`，`EMAIL_HOST`和`EMAIL_HOST_USER`更改为正确的开发数值。

接下来，让我们将我们的生产设置放在`django/config/production_settings.py`中：

```py
from .common_settings import *

DEBUG = False

assert SECRET_KEY is not None, (
    'Please provide DJANGO_SECRET_KEY environment variable with a value')

ALLOWED_HOSTS += [
    os.getenv('DJANGO_ALLOWED_HOSTS'),
]

DATABASES['default'].update({
    'NAME': os.getenv('DJANGO_DB_NAME'),
    'USER': os.getenv('DJANGO_DB_USER'),
    'PASSWORD': os.getenv('DJANGO_DB_PASSWORD'),
    'HOST': os.getenv('DJANGO_DB_HOST'),
    'PORT': os.getenv('DJANGO_DB_PORT'),
})

LOGGING['handlers']['main'] = {
    'class': 'logging.handlers.WatchedFileHandler',
    'level': 'DEBUG',
    'filename': os.getenv('DJANGO_LOG_FILE')
}

MAILING_LIST_FROM_EMAIL = os.getenv('MAIL_APE_FROM_EMAIL')
MAILING_LIST_LINK_DOMAIN = os.getenv('DJANGO_ALLOWED_HOSTS')

EMAIL_HOST = os.getenv('EMAIL_HOST')
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD')
EMAIL_PORT = os.getenv('EMAIL_HOST_PORT')
EMAIL_USE_TLS = os.getenv('EMAIL_HOST_TLS', 'false').lower() == 'true'

CELERY_BROKER_TRANSPORT_OPTIONS = {
    'region': 'us-west-2',
    'queue_name_prefix': 'mailape-',
CELERY_BROKER_URL = 'sqs://'
}
```

我们的生产设置文件大部分数值都来自环境变量，这样我们就不会将生产数值提交到服务器中。有三个设置我们需要审查，如下：

+   `MAILING_LIST_LINK_DOMAIN`：这是我们邮件中链接的域。在我们的情况下，在前面的代码片段中，我们使用了与我们添加到`ALLOWED_HOSTS`列表中的相同域，确保我们正在为链接指向的域提供服务。

+   `CELERY_BROKER_TRANSPORT_OPTIONS`：这是一个配置 Celery 使用正确的 SQS 队列的选项字典。我们需要将区域设置为`us-west-2`，因为我们整个生产部署将在该区域。默认情况下，Celery 将希望使用一个名为`celery`的队列。然而，我们不希望该名称与我们可能部署的其他 Celery 项目发生冲突。为了防止名称冲突，我们将配置 Celery 使用`mailape-`前缀。

+   `CELERY_BROKER_URL`：这告诉 Celery 要使用哪个代理。在我们的情况下，我们使用 SQS。我们将使用 AWS 的基于角色的授权为我们的虚拟机提供对 SQS 的访问权限，这样我们就不必提供任何凭据。

现在我们已经创建了我们的生产设置，让我们在 AWS 云中创建我们的基础设施。

# 在 AWS 中创建基础设施堆栈

为了在 AWS 上托管应用程序，我们需要确保我们已经设置了一些基础设施。我们需要以下内容：

+   一个 PostgreSQL 服务器

+   安全组，以打开网络端口，以便我们可以访问我们的数据库和 Web 服务器

+   一个 InstanceProfile，为我们部署的虚拟机提供对 SQS 的访问权限

我们可以使用 AWS Web 控制台或使用命令行界面创建所有这些。然而，随着时间的推移，如果我们依赖运行时调整，很难跟踪我们的基础设施是如何配置的。如果我们能够描述我们需要的基础设施在文件中，就像我们跟踪我们的代码一样，那将会更好。

AWS 提供了一个名为 CloudFormation 的服务，它让我们可以将基础设施视为代码。我们将使用 YAML（也可以使用 JSON，但我们将使用 YAML）在 CloudFormation 模板中定义我们的基础设施。然后，我们将执行我们的 CloudFormation 模板来创建一个 CloudFormation 堆栈。CloudFormation 堆栈将与 AWS 云中的实际资源相关联。如果我们删除 CloudFormation 堆栈，相关资源也将被删除。这使我们可以简单地控制我们对 AWS 资源的使用。

让我们在`cloudformation/infrastructure.yaml`中创建我们的 CloudFormation 模板。每个 CloudFormation 模板都以`Description`和模板格式版本信息开始。让我们从以下内容开始我们的文件：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape Infrastructure
```

我们的 CloudFormation 模板将包括以下三个部分：

+   `Parameters`：这是我们将在运行时传递的值。这个块是可选的，但很有用。在我们的情况下，我们将传递主数据库密码，而不是在我们的模板中硬编码它。

+   `Resources`：这是我们将描述的堆栈中包含的具体资源。这将描述我们的数据库服务器、SQS 队列、安全组和 InstanceProfile。

+   `Outputs`：这是我们将描述的值，以便更容易引用我们创建的资源。这个块是可选的，但很有用。我们将提供我们的数据库服务器地址和我们创建的 InstanceProfile 的 ID。

让我们从创建 CloudFormation 模板的`Parameters`块开始。

# 在 CloudFormation 模板中接受参数

为了避免在 CloudFormation 模板中硬编码值，我们可以接受参数。这有助于我们避免在模板中硬编码敏感值（如密码）。

让我们添加一个参数来接受数据库服务器主用户的密码：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape Infrastructure
Parameters:
  MasterDBPassword:
    Description: Master Password for the RDS instance
    Type: String
```

这为我们的模板添加了一个`MasterDBPassword`参数。我们以后将能够引用这个值。CloudFormation 模板让我们为参数添加两个信息：

+   `Description`：这不被 CloudFormation 使用，但对于必须维护我们的基础设施的人来说是有用的。

+   `Type`：CloudFormation 在执行我们的模板之前使用这个来检查我们提供的值是否有效。在我们的情况下，密码是一个`String`。

接下来，让我们添加一个`Resources`块来定义我们基础设施中需要的 AWS 资源。

# 列出我们基础设施中的资源

接下来，我们将在`cloudformation/infrastructure.yaml`中的 CloudFormation 模板中添加一个`Resources`块。我们的基础设施模板将定义五个资源：

+   安全组，将打开网络端口，允许我们访问数据库和 Web 服务器

+   我们的数据库服务器

+   我们的 SQS 队列

+   允许访问 SQS 的角色

+   InstanceProfile，让我们的 Web 服务器假定上述角色

让我们首先创建安全组，这将打开我们将访问数据库和 Web 服务器的网络端口。

# 添加安全组

在 AWS 中，SecurityGroup 定义了一组网络访问规则，就像网络防火墙一样。默认情况下，启动的虚拟机可以*发送*数据到任何网络端口，但不能在任何网络端口上*接受*连接。这意味着我们无法使用 SSH 或 HTTP 进行连接；让我们解决这个问题。

让我们在`cloudformation/infrastructure.yaml`中的 CloudFormation 模板中更新三个新的安全组：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape Infrastructure
Parameters:
  ...
Resources:
  SSHSecurityGroup:
    Type: "AWS::EC2::SecurityGroup"
    Properties:
      GroupName: ssh-access
      GroupDescription: permit ssh access
      SecurityGroupIngress:
        -
          IpProtocol: "tcp"
          FromPort: "22"
          ToPort: "22"
          CidrIp: "0.0.0.0/0"
  WebSecurityGroup:
    Type: "AWS::EC2::SecurityGroup"
    Properties:
      GroupName: web-access
      GroupDescription: permit http access
      SecurityGroupIngress:
        -
          IpProtocol: "tcp"
          FromPort: "80"
          ToPort: "80"
          CidrIp: "0.0.0.0/0"
  DatabaseSecurityGroup:
    Type: "AWS::EC2::SecurityGroup"
    Properties:
      GroupName: db-access
      GroupDescription: permit db access
      SecurityGroupIngress:
        -
          IpProtocol: "tcp"
          FromPort: "5432"
          ToPort: "5432"
          CidrIp: "0.0.0.0/0"
```

在前面的代码块中，我们定义了三个新的安全组，以打开端口`22`（SSH），`80`（HTTP）和`5432`（默认的 Postgres 端口）。

让我们更仔细地看一下 CloudFormation 资源的语法。每个资源块必须具有`Type`和`Properties`属性。`Type`属性告诉 CloudFormation 这个资源描述了什么。`Properties`属性描述了这个特定资源的设置。

我们使用以下属性的安全组：

+   `GroupName`：这提供了人性化的名称。这是可选的，但建议使用。 CloudFormation 可以为我们生成名称。安全组名称必须对于给定帐户是唯一的（例如，我不能有两个`db-access`组，但您和我每个人都可以有一个`db-access`组）。

+   `GroupDescription`：这是组用途的人性化描述。它是必需的。

+   `SecurityGroupIngress`：这是一个端口列表，用于接受此组中虚拟机的传入连接。

+   `FromPort`/`ToPort`：通常，这两个设置将具有相同的值，即您希望能够连接的网络端口。 `FromPort`是我们将连接的端口。 `ToPort`是服务正在监听的 VM 端口。

+   `CidrIp`：这是一个 IPv4 范围，用于接受连接。 `0.0.0.0/0`表示接受所有连接。

接下来，让我们将数据库服务器添加到我们的资源列表中。

# 添加数据库服务器

AWS 提供关系数据库服务器作为一种称为**关系数据库服务**（**RDS**）的服务。要在 AWS 上创建数据库服务器，我们将创建一个新的 RDS 虚拟机（称为*实例*）。一个重要的事情要注意的是，当我们启动一个 RDS 实例时，我们可以连接到服务器上的 PostgreSQL 数据库，但我们没有 shell 访问权限。我们必须在不同的虚拟机上运行 Django。

让我们在`cloudformation/infrastructure.yaml`中的 CloudFormation 模板中添加一个 RDS 实例：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape Infrastructure
Parameters:
  ...
Resources:
  ...
  DatabaseServer:
    Type: AWS::RDS::DBInstance
    Properties:
      DBName: mailape
      DBInstanceClass: db.t2.micro
      MasterUsername: master
      MasterUserPassword: !Ref MasterDBPassword
      Engine: postgres
      AllocatedStorage: 20
      PubliclyAccessible: true
      VPCSecurityGroups: !GetAtt DatabaseSecurityGroup.GroupId
```

我们的新 RDS 实例条目是`AWS::RDS::DBInstance`类型。让我们回顾一下我们设置的属性：

+   `DBName`：这是*服务器*的名称，而不是其中运行的任何数据库的名称。

+   `DBInstanceClass`：这定义了服务器虚拟机的内存和处理能力。在撰写本书时，`db.t2.micro`是首年免费套餐的一部分。

+   `MasterUsername`：这是服务器上特权管理员帐户的用户名。

+   `MasterUserPassword`：这是特权管理员帐户的密码

+   `!Ref MasterDBPassword`：这是引用`MasterDBPassword`参数的快捷语法。这样可以避免硬编码数据库服务器的管理员密码。

+   `Engine`：这是我们想要的数据库服务器类型；在我们的情况下，`postgres`将为我们提供一个 PostgreSQL 服务器。

+   `AllocatedStorage`：这表示服务器应该具有多少存储空间，以 GB 为单位。

+   `PubliclyAccessible`：这表示服务器是否可以从 AWS 云外部访问。

+   `VPCSecurityGroups`：这是一个 SecurityGroups 列表，指示哪些端口是打开和可访问的。

+   `!GetAtt DatabaseSecurityGroup.GroupId`: 这返回`DatabaseSecurityGroup`安全组的`GroupID`属性。

这个块还向我们介绍了 CloudFormation 的`Ref`和`GetAtt`函数。这两个函数让我们能够引用我们 CloudFormation 堆栈的其他部分，这是非常重要的。`Ref`让我们使用我们的`MasterDBPassword`参数作为我们数据库服务器的`MasterUserPassword`的值。`GetAtt`让我们在我们的数据库服务器的`VPCSercurityGroups`列表中引用我们 AWS 生成的`DatabaseSecurityGroup`的`GroupId`属性。

AWS CloudFormation 提供了各种不同的函数，以使构建模板更容易。它们在 AWS 在线文档中有记录（[`docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/intrinsic-function-reference.html`](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/intrinsic-function-reference.html)）。

接下来，让我们创建 Celery 将使用的 SQS 队列。

# 为 Celery 添加队列

SQS 是 AWS 消息队列服务。使用 SQS，我们可以创建一个与 Celery 兼容的消息队列，而无需维护。SQS 可以快速扩展以处理我们发送的任何请求数量。

要定义我们的队列，请将其添加到`cloudformation/infrastructure.yaml`中的`Resources`块中：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape Infrastructure
Parameters:
  ...
Resources:
  ...
  MailApeQueue:
    Type: "AWS::SQS::Queue"
    Properties:
      QueueName: mailape-celery
```

我们的新资源是`AWS::SQS::Queue`类型，并且有一个属性`QueueName`。

接下来，让我们创建一个角色和 InstanceProfile，让我们的生产服务器访问我们的 SQS 队列。

# 为队列访问创建角色

早些时候，在*添加安全组*部分，我们讨论了创建 SecurityGroups 以打开网络端口，以便我们可以进行网络连接。为了管理 AWS 资源之间的访问，我们需要使用基于角色的授权。在基于角色的授权中，我们定义一个角色，可以被分配该角色的人（假定该角色），以及该角色可以执行哪些操作。为了使我们的 Web 服务器使用该角色，我们需要创建一个与该角色关联的 EC2 实例配置文件。

让我们首先在`cloudformation/infrastructure.yaml`中添加一个角色：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape Infrastructure
Parameters:
  ...
Resources:
  ...
   SQSAccessRole:
    Type: "AWS::IAM::Role"
    Properties:
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          -
            Effect: "Allow"
            Principal:
              Service:
                - "ec2.amazonaws.com"
            Action:
              - "sts:AssumeRole"
      Policies:
        -
          PolicyName: "root"
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              -
                Effect: Allow
                Action: "sqs:*"
                Resource: !GetAtt MailApeQueue.Arn
              -
                Effect: Allow
                Action: sqs:ListQueues
                Resource: "*"
```

我们的新块是`AWS::IAM::Role`类型。IAM 是 AWS 身份和访问管理服务的缩写。我们的角色由以下两个属性组成：

+   `AssumeRolePolicyDocument`：这定义了谁可以被分配这个角色。在我们的情况下，我们说这个角色可以被亚马逊的 EC2 服务中的任何对象假定。稍后，我们将在我们的 EC2 实例中使用它。

+   `Policies`：这是该角色允许（或拒绝）的操作列表。在我们的情况下，我们允许在我们之前定义的 SQS 队列上执行所有 SQS 操作（`sqs:*`）。我们通过使用`GetAtt`函数引用我们的队列来获取其`Arn`，Amazon 资源名称（ARN）。ARN 是亚马逊为亚马逊云上的每个资源提供全局唯一 ID 的方式。

现在我们有了我们的角色，我们可以将其与一个`InstanceProfile`资源关联起来，该资源可以与我们的 Web 服务器关联起来：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape Infrastructure
Parameters:
  ...
Resources:
  ...
  SQSClientInstance:
    Type: "AWS::IAM::InstanceProfile"
    Properties:
      Roles:
        - !Ref SQSAccessRole
```

我们的新 InstanceProfile 是`AWS::IAM::InstanceProfile`类型，并且需要一个关联角色的列表。在我们的情况下，我们只需使用`Ref`函数引用我们之前创建的`SQSAccessRole`。

现在我们已经创建了我们的基础设施资源，让我们输出我们的数据库的地址和我们的`InstanceProfile`资源的 ARN。

# 输出我们的资源信息

CloudFormation 模板可以有一个输出块，以便更容易地引用创建的资源。在我们的情况下，我们将输出我们的数据库服务器的地址和`InstanceProfile`的 ARN。

让我们在`cloudformation/infrastructure.yaml`中更新我们的 CloudFormation 模板：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape Infrastructure
Parameters:
  ...
Resources:
  ...
Outputs:
  DatabaseDNS:
    Description: Public DNS of RDS database
    Value: !GetAtt DatabaseServer.Endpoint.Address
  SQSClientProfile:
    Description: Instance Profile for EC2 instances that need SQS Access
    Value: !GetAtt SQSClientInstance.Arn
```

在上述代码中，我们使用`GetAtt`函数返回我们的`DatabaseServer`资源的地址和我们的`SQSClientInstance` `InstanceProfile`资源的 ARN。

# 执行我们的模板以创建我们的资源

现在我们已经创建了我们的`CloudFormation`模板，我们可以创建一个`CloudFormation`堆栈。当我们告诉 AWS 创建我们的`CloudFormation`堆栈时，它将在我们的模板中创建所有相关资源。

要创建我们的模板，我们需要以下两件事情：

+   AWS 命令行界面（CLI）

+   AWS 访问密钥/秘密密钥对

我们可以使用`pip`安装 AWS CLI：

```py
$ pip install awscli
```

要获取（或创建）您的访问密钥/秘密密钥对，您需要访问 AWS 控制台的安全凭据部分。

然后我们需要使用我们的密钥和区域配置 AWS 命令行工具。`aws`命令提供了一个交互式的`configure`子命令来完成这个任务。让我们在命令行上运行它：

```py
$ aws configure
AWS Access Key ID [None]: <Your ACCESS key>
AWS Secret Access Key [None]: <Your secret key>
Default region name [None]: us-west-2
Default output format [None]: json
```

`aws configure`命令将您输入的值存储在主目录中的`.aws`目录中。

有了这些设置，我们现在可以创建我们的堆栈：

```py
$ aws cloudformation create-stack \
    --stack-name "infrastructure" \
    --template-body "file:///path/to/mailape/cloudformation/infrastrucutre.yaml" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters \
      "ParameterKey=MasterDBPassword,ParameterValue=password" \
    --region us-west-2
```

创建堆栈可能需要一些时间，因此该命令在等待成功时返回。让我们更仔细地看看我们的`create-stack`命令：

+   `--stack-name`：这是我们正在创建的堆栈的名称。堆栈名称必须在每个帐户中是唯一的。

+   `--template-body`：这要么是模板本身，要么是我们的情况下模板文件的`file://` URL。请记住，`file://` URL 需要文件的绝对路径。

+   `--capabilities CAPABILITY_NAMED_IAM`：这对于创建或影响**Identity and Access Management**（**IAM**）服务的模板是必需的。这可以防止意外影响访问管理服务。

+   `--parameters`：这允许我们传递模板参数的值。在我们的案例中，我们将数据库的主密码设置为`password`，这不是一个安全的值。

+   `--region`：AWS 云组织为世界各地的一组区域。在我们的案例中，我们使用的是位于美国俄勒冈州一系列数据中心的`us-west-2`。

请记住，您需要为数据库设置一个安全的主密码。

要查看堆栈创建的进度，我们可以使用 AWS Web 控制台（[`us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2`](https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2)）或使用命令行进行检查：

```py
$ aws cloudformation describe-stacks \
    --stack-name "infrastructure" \
    --region us-west-2
```

当堆栈完成创建相关资源时，它将返回类似于这样的结果：

```py
{
    "Stacks": [
        {
            "StackId": "arn:aws:cloudformation:us-west-2:XXX:stack/infrastructure/NNN",
            "StackName": "infrastructure",
            "Description": "Mail Ape Infrastructure",
            "Parameters": [
                {
                    "ParameterKey": "MasterDBPassword",
                    "ParameterValue": "password"
                }
            ],
            "StackStatus": "CREATE_COMPLETE",
            "Outputs": [
                {
                    "OutputKey": "SQSClientProfile",
                    "OutputValue": "arn:aws:iam::XXX:instance-profile/infrastructure-SQSClientInstance-XXX",
                    "Description": "Instance Profile for EC2 instances that need SQS Access"
                },
                {
                    "OutputKey": "DatabaseDNS",
                    "OutputValue": "XXX.XXX.us-west-2.rds.amazonaws.com",
                    "Description": "Public DNS of RDS database"
                }
            ],
        }
    ]
}
```

在`describe-stack`结果中特别注意的两件事是：

+   `Parameters`键下的对象将以明文显示我们的主数据库密码

+   `Outputs`对象键显示了我们的`InstanceProfile`资源的 ARN 和数据库服务器的地址

在所有先前的代码中，我已经用 XXX 替换了特定于我的帐户的值。您的输出将有所不同。

如果您想要删除与您的堆栈关联的资源，您可以直接删除该堆栈：

```py
$ aws cloudformation delete-stack --stack-name "infrastructure"
```

接下来，我们将构建一个 Amazon Machine Image，用于在 AWS 中运行 Mail Ape。

# 使用 Packer 构建 Amazon Machine Image

现在我们的基础设施在 AWS 中运行，让我们构建我们的 Mail Ape 服务器。在 AWS 中，我们可以启动一个官方的 Ubuntu VM，按照第九章中的步骤，*部署 Answerly*，并让我们的 Mail Ape 运行。但是，AWS 将 EC2 实例视为*临时*。如果 EC2 实例被终止，那么我们将不得不启动一个新实例并重新配置它。有几种方法可以缓解这个问题。我们将通过为我们的发布构建一个新的**Amazon Machine Image**（**AMI**）来解决临时 EC2 实例的问题。然后，每当我们使用该 AMI 启动 EC2 实例时，它将已经完美地配置好。

我们将使用 HashiCorp 的 Packer 工具自动构建我们的 AMI。 Packer 为我们提供了一种从 Packer 模板创建 AMI 的方法。 Packer 模板是一个定义了配置 EC2 实例到我们期望状态并保存 AMI 所需步骤的 JSON 文件。为了运行我们的 Packer 模板，我们还将编写一系列 shell 脚本来配置我们的 AMI。使用 Packer 这样的工具，我们可以自动构建一个新的发布 AMI。

让我们首先在我们的机器上安装 Packer。

# 安装 Packer

从[`www.packer.io`](https://www.packer.io)下载页面获取 Packer。 Packer 适用于所有主要平台。

接下来，我们将创建一个脚本来创建我们在生产中依赖的目录。

# 创建一个脚本来创建我们的目录结构

我们将编写的第一个脚本将为我们的所有代码创建目录。让我们在`scripts/make_aws_directories.sh`中添加以下脚本到我们的项目中：

```py
#!/usr/bin/env bash
set -e

sudo mkdir -p \
    /mailape/ubuntu \
    /mailape/apache \
    /mailape/django \
    /var/log/celery \
    /etc/mailape \
    /var/log/mailape

sudo chown -R ubuntu /mailape
```

在上述代码中，我们使用`mkdir`来创建目录。接下来，我们希望让`ubuntu`用户可以写入`/mailape`目录，所以我们递归地`chown`了`/mailape`目录。

所以，让我们创建一个脚本来安装我们需要的 Ubuntu 软件包。

# 创建一个脚本来安装我们所有的软件包

在我们的生产环境中，我们将不仅需要安装 Ubuntu 软件包，还需要安装我们已经列出的 Python 软件包。首先，让我们在`ubuntu/packages.txt`中列出所有我们的 Ubuntu 软件包：

```py
python3
python3-pip
python3-dev
virtualenv
apache2
libapache2-mod-wsgi-py3
postgresql-client
libcurl4-openssl-dev
libssl-dev
```

接下来，让我们创建一个脚本来安装`scripts/install_all_packages`中的所有软件包：

```py
#!/usr/bin/env bash
set -e

sudo apt-get update
sudo apt install -y $(cat /mailape/ubuntu/packages.txt | grep -i '^[a-z]')

virtualenv -p $(which python3) /mailape/virtualenv
source /mailape/virtualenv/bin/activate

pip install -r /mailape/requirements.production.txt

sudo chown -R www-data /var/log/mailape \
    /etc/mailape \
    /var/run/celery \
    /var/log/celery
```

在上述脚本中，我们将安装我们上面列出的 Ubuntu 软件包，然后创建一个`virtualenv`来隔离我们的 Mail Ape Python 环境和软件包。最后，我们将一些目录的所有权交给 Apache（`www-data`用户），以便它可以写入这些目录。我们无法给`www-data`用户所有权，因为直到我们安装`apache2`软件包之前，它们可能并不存在。

接下来，让我们配置 Apache2 使用 mod_wsgi 来运行 Mail Ape。

# 配置 Apache

现在，我们将添加 Apache mod_wsgi 配置，就像我们在第九章中所做的那样，*部署 Answerly*。 mod_wsgi 配置不是本章的重点，所以请参考第九章，*部署 Answerly*，了解这个配置的工作原理。

让我们为 Mail Ape 在`apache/mailape.apache.conf`中创建一个虚拟主机配置文件：

```py
LogLevel info
WSGIRestrictEmbedded On

<VirtualHost *:80>

    WSGIDaemonProcess mailape \
        python-home=/mailape/virtualenv \
        python-path=/mailape/django \
        processes=2 \
        threads=2

    WSGIProcessGroup mailape

    WSGIScriptAlias / /mailape/django/config/wsgi.py
    <Directory /mailape/django/config>
        <Files wsgi.py>
            Require all granted
        </Files>
    </Directory>

    Alias /static/ /mailape/django/static_root
    <Directory /mailape/django/static_root>
        Require all granted
    </Directory>
    ErrorLog ${APACHE_LOG_DIR}/error.log
    CustomLog ${APACHE_LOG_DIR}/access.log combined

</VirtualHost>
```

正如我们在第九章中所讨论的，*部署 Answerly*，我们无法将环境变量传递给我们的 mod_wsgi Python 进程，因此我们需要像在第九章中所做的那样更新项目的`wsgi.py`。

这是我们的新`django/config/wsgi.py`：

```py
import os
import configparser

from django.core.wsgi import get_wsgi_application

if not os.environ.get('DJANGO_SETTINGS_MODULE'):
    parser = configparser.ConfigParser()
    parser.read('/etc/mailape/mailape.ini')
    for name, val in parser['mod_wsgi'].items():
        os.environ[name.upper()] = val

application = get_wsgi_application()
```

我们在第九章*部署 Answerly*中讨论了上述脚本。这里唯一的区别是我们解析的文件，即`/etc/mailape/mailape.ini`。

接下来，我们需要将我们的虚拟主机配置添加到 Apache 的`sites-enabled`目录中。让我们在`scripts/configure_apache.sh`中创建一个脚本来做到这一点：

```py
#!/usr/bin/env bash

sudo rm /etc/apache2/sites-enabled/*
sudo ln -s /mailape/apache/mailape.apache.conf /etc/apache2/sites-enabled/000-mailape.conf
```

现在我们有了一个在生产环境中配置 Apache 的脚本，让我们配置我们的 Celery 工作进程开始。

# 配置 Celery

现在我们已经让 Apache 运行 Mail Ape，我们需要配置 Celery 来启动并处理我们的 SQS 队列。为了启动我们的 Celery 工作进程，我们将使用 Ubuntu 的 systemd 进程管理工具。

首先，让我们创建一个 Celery 服务文件，告诉 SystemD 如何启动 Celery。我们将在`ubuntu/celery.service`中创建服务文件：

```py
[Unit]
Description=Mail Ape Celery Service
After=network.target

[Service]
Type=forking
User=www-data
Group=www-data
EnvironmentFile=/etc/mailape/celery.env
WorkingDirectory=/mailape/django
ExecStart=/bin/sh -c '/mailape/virtualenv/bin/celery multi start worker \
    -A "config.celery:app" \
    --logfile=/var/log/celery/%n%I.log --loglevel="INFO" \
    --pidfile=/run/celery/%n.pid'
ExecStop=/bin/sh -c '/mailape/virtualenv/bin/celery multi stopwait worker \
    --pidfile=/run/celery/%n.pid'
ExecReload=/bin/sh -c '/mailape/virtualenv/bin/celery multi restart worker \
   -A "config.celery:app" \
   --logfile=/var/log/celery/%n%I.log --loglevel="INFO" \
   --pidfile=/run/celery/%n.pid'

[Install]
WantedBy=multi-user.target
```

让我们仔细看看这个文件中的一些选项：

+   `After=network.target`：这意味着 SystemD 在服务器连接到网络之前不会启动这个服务。

+   `Type=forking`：这意味着`ExecStart`命令最终将启动一个新进程，该进程将继续在自己的进程 ID（PID）下运行。

+   `User`: 这表示将拥有 Celery 进程的用户。在我们的情况下，我们将重用 Apache 的`www-data`用户。

+   `EnvironmentFile`: 这列出了一个将用于环境变量和所有`Exec`命令设置的值的文件。我们列出了一个与我们的 Celery 配置（`/mailape/ubuntu/celery.systemd.conf`）和一个与我们的 Mail Ape 配置（`/etc/mailape/celery.env`）的文件。

+   `ExecStart`: 这是将要执行的命令，用于启动 Celery。在我们的情况下，我们启动多个 Celery 工作者。我们所有的 Celery 命令将基于它们创建的进程 ID 文件来操作我们的工作者。Celery 将用工作者的 ID 替换`%n`。

+   `ExecStop`: 这是将根据它们的 PID 文件执行的命令，用于停止我们的 Celery 工作者。

+   `ExecReload`: 这是将执行的命令，用于重新启动我们的 Celery 工作者。Celery 支持`restart`命令，因此我们将使用它来执行重新启动。但是，此命令必须接收与我们的`ExecStart`命令相同的选项。

我们将把我们的 PID 文件放在`/var/run/celery`中，但我们需要确保该目录已创建。`/var/run`是一个特殊目录，不使用常规文件系统。我们需要创建一个配置文件，告诉 Ubuntu 创建`/var/run/celery`。让我们在`ubuntu/tmpfiles-celery.conf`中创建这个文件：

```py
d    /run/celery   0755 www-data www-data - -
```

这告诉 Ubuntu 创建一个由 Apache 用户（`www-data`）拥有的目录`/run/celery`。

最后，让我们创建一个脚本，将所有这些文件放在服务器的正确位置。我们将把这个脚本命名为`scripts/configure_celery.sh`：

```py
#!/usr/bin/env bash

sudo ln -s /mailape/ubuntu/celery.service /etc/systemd/system/celery.service
sudo ln -s /mailape/ubuntu/celery.service /etc/systemd/system/multi-user.target.wants/celery.service
sudo ln -s /mailape/ubuntu/tmpfiles-celery.conf /etc/tmpfiles.d/celery.conf
```

现在 Celery 和 Apache 已配置好，让我们确保它们具有正确的环境配置来运行 Mail Ape

# 创建环境配置文件

我们的 Celery 和 mod_wsgi Python 进程都需要从环境中提取配置信息，以连接到正确的数据库、SQS 队列和许多其他服务。这些是我们不想在版本控制系统中检查的设置和值（例如密码）。但是，我们仍然需要在生产环境中设置它们。为了创建定义我们的进程将在其中运行的环境的文件，我们将在`scripts/make_mailape_environment_ini.sh`中制作脚本：

```py
#!/usr/bin/env bash

ENVIRONMENT="
DJANGO_ALLOWED_HOSTS=${WEB_DOMAIN}
DJANGO_DB_NAME=mailape
DJANGO_DB_USER=mailape
DJANGO_DB_PASSWORD=${DJANGO_DB_PASSWORD}
DJANGO_DB_HOST=${DJANGO_DB_HOST}
DJANGO_DB_PORT=5432
DJANGO_LOG_FILE=/var/log/mailape/mailape.log
DJANGO_SECRET_KEY=${DJANGO_SECRET}
DJANGO_SETTINGS_MODULE=config.production_settings
MAIL_APE_FROM_EMAIL=admin@blvdplatform.com
EMAIL_HOST=${EMAIL_HOST}
EMAIL_HOST_USER=mailape
EMAIL_HOST_PASSWORD=${EMAIL_HOST_PASSWORD}
EMAIL_HOST_PORT=587
EMAIL_HOST_TLS=true

INI_FILE="[mod_wsgi]
${ENVIRONMENT}
"

echo "${INI_FILE}" | sudo tee "/etc/mailape/mailape.ini"
echo "${ENVIRONMENT}" | sudo tee "/etc/mailape/celery.env"
```

我们的`make_mailape_environment_ini.sh`脚本中有一些值是硬编码的，但引用了其他值（例如密码）作为环境变量。我们将在运行时将这些变量的值传递给 Packer。然后 Packer 将这些值传递给我们的脚本。

接下来，让我们制作 Packer 模板来构建我们的 AMI。

# 制作 Packer 模板

Packer 根据 Packer 模板文件中列出的指令创建 AMI。Packer 模板是一个由三个顶级键组成的 JSON 文件：

+   `variables`: 这将允许我们在运行时设置值（例如密码）

+   `builders`: 这指定了特定于云平台的详细信息，例如 AWS 凭据

+   `provisioners`: 这些是 Packer 将执行的指令，以制作我们的映像

让我们从`packer/web_worker.json`中创建我们的 Packer 模板，从`variables`部分开始：

```py
{
  "variables": {
    "aws_access_key": "",
    "aws_secret_key": "",
    "django_db_password":"",
    "django_db_host":"",
    "django_secret":"",
    "email_host":"",
    "email_host_password":"",
    "mail_ape_aws_key":"",
    "mail_ape_secret_key":"",
    "sqs_celery_queue":"",
    "web_domain":""
  }
}
```

在`variables`键下，我们将列出我们希望模板作为 JSON 对象键接受的所有变量。如果变量有默认值，那么我们可以将其作为该变量键的值提供。

接下来，让我们添加一个`builders`部分来配置 Packer 使用 AWS：

```py
{
  "variables": {...},
  "builders": [
    {
      "type": "amazon-ebs",
      "access_key": "{{user `aws_access_key`}}",
      "secret_key": "{{user `aws_secret_key`}}",
      "region": "us-west-2",
      "source_ami": "ami-78b82400",
      "instance_type": "t2.micro",
      "ssh_username": "ubuntu",
      "ami_name": "mailape-{{timestamp}}",
      "tags": {
        "project": "mailape"
      }
    }
  ]
}
```

`builders`是一个数组，因为我们可以使用相同的模板在多个平台上构建机器映像（例如 AWS 和 Google Cloud）。让我们详细看看每个选项：

+   `"type": "amazon-ebs"`: 告诉 Packer 我们正在创建一个带有弹性块存储的亚马逊机器映像。这是首选配置，因为它提供了灵活性。

+   `"access_key": "{{user aws_access_key }}"`: 这是 Packer 应该使用的访问密钥，用于与 AWS 进行身份验证。Packer 包含自己的模板语言，以便可以在运行时生成值。`{{ }}`之间的任何值都是由 Packer 模板引擎生成的。模板引擎提供了一个`user`函数，它接受用户提供的变量的名称并返回其值。例如，当运行 Packer 时，`{{user aws_access_key }}`将被用户提供给`aws_access_key`的值替换。

+   `"secret_key": "{{user aws_secret_key }}"`: 这与 AWS 秘钥相同。

+   `"region": "us-west-2"`: 这指定了 AWS 区域。我们所有的工作都将在`us-west-2`中完成。

+   `"source_ami": "ami-78b82400"`: 这是我们要定制的镜像，以制作我们的镜像。在我们的情况下，我们使用官方的 Ubuntu AMI。Ubuntu 提供了一个 EC2 AMI 定位器（[`cloud-images.ubuntu.com/locator/ec2/`](http://cloud-images.ubuntu.com/locator/ec2/)）来帮助找到他们的官方 AMI。

+   `"instance_type": "t2.micro"`: 这是一个小型廉价的实例，在撰写本书时，属于 AWS 免费套餐。

+   `"ssh_username": "ubuntu"`: Packer 通过 SSH 在虚拟机上执行所有操作。这是它应该用于身份验证的用户名。Packer 将为身份验证生成自己的密钥对，因此我们不必担心指定密码或密钥。

+   `"ami_name": "mailape-{{timestamp}}"`: 结果 AMI 的名称。`{{timestamp}}`是一个返回自 Unix 纪元以来的 UTC 时间的函数。

+   `"tags": {...}`: 标记资源可以更容易地在 AWS 中识别资源。这是可选的，但建议使用。

现在我们已经指定了我们的 AWS 构建器，我们将需要指定我们的配置程序。

Packer 配置程序是定制服务器的指令。在我们的情况下，我们将使用以下两种类型的配置程序：

+   `file`配置程序用于将我们的代码上传到服务器。

+   `shell`配置程序用于执行我们的脚本和命令

首先，让我们添加我们的`make_aws_directories.sh`脚本，因为我们需要它首先运行：

```py
{
  "variables": {...},
  "builders": [...],
  "provisioners": [
    {
      "type": "shell",
      "script": "{{template_dir}}/../scripts/make_aws_directories.sh"
    }
  ]
}
```

具有`script`属性的`shell`配置程序将上传，执行和删除脚本。Packer 提供了`{{template_dir}}`函数，它返回模板目录的目录。这使我们可以避免硬编码绝对路径。我们执行的第一个配置程序将执行我们在本节前面创建的`make_aws_directories.sh`脚本。

现在我们的目录存在了，让我们使用`file`配置程序将我们的代码和文件复制过去：

```py
{
  "variables": {...},
  "builders": [...],
  "provisioners": [
    ...,
    {
      "type": "file",
      "source": "{{template_dir}}/../requirements.common.txt",
      "destination": "/mailape/requirements.common.txt"
    },
    {
      "type": "file",
      "source": "{{template_dir}}/../requirements.production.txt",
      "destination": "/mailape/requirements.production.txt"
    },
    {
      "type": "file",
      "source": "{{template_dir}}/../ubuntu",
      "destination": "/mailape/ubuntu"
    },
    {
      "type": "file",
      "source": "{{template_dir}}/../apache",
      "destination": "/mailape/apache"
    },
    {
      "type": "file",
      "source": "{{template_dir}}/../django",
      "destination": "/mailape/django"
    },
  ]
}
```

`file`配置程序将本地文件或由`source`定义的目录上传到`destination`服务器上。

由于我们从工作目录上传了 Python 代码，我们需要小心旧的`.pyc`文件是否还存在。让我们确保在我们的生产服务器上删除这些文件：

```py
{
  "variables": {...},
  "builders": [...],
  "provisioners": [
    ...,
   {
      "type": "shell",
      "inline": "find /mailape/django -name '*.pyc' -delete"
   },
   ]
}
```

`shell`配置程序可以接收`inline`属性。然后，配置程序将在服务器上执行`inline`命令。

最后，让我们执行我们创建的其余脚本：

```py
{
  "variables": {...},
  "builders": [...],
  "provisioners": [
    ...,
    {
      "type": "shell",
      "scripts": [
        "{{template_dir}}/../scripts/install_all_packages.sh",
        "{{template_dir}}/../scripts/configure_apache.sh",
        "{{template_dir}}/../scripts/make_mailape_environment_ini.sh",
        "{{template_dir}}/../scripts/configure_celery.sh"
        ],
      "environment_vars": [
        "DJANGO_DB_HOST={{user `django_db_host`}}",
        "DJANGO_DB_PASSWORD={{user `django_db_password`}}",
        "DJANGO_SECRET={{user `django_secret`}}",
        "EMAIL_HOST={{user `email_host`}}",
        "EMAIL_HOST_PASSWORD={{user `email_host_password`}}",
        "WEB_DOMAIN={{user `web_domain`}}"
      ]
}
```

在这种情况下，`shell`配置程序已收到`scripts`和`environment_vars`。`scripts`是指向 shell 脚本的路径数组。数组中的每个项目都将被上传和执行。在执行每个脚本时，此`shell`配置程序将添加`environment_vars`中列出的环境变量。`environment_vars`参数可选地提供给所有`shell`配置程序，以提供额外的环境变量。

随着我们的最终配置程序添加到我们的文件中，我们现在已经完成了我们的 Packer 模板。让我们使用 Packer 来执行模板并构建我们的 Mail Ape 生产服务器。

# 运行 Packer 来构建 Amazon Machine Image

安装了 Packer 并创建了 Mail Ape 生产服务器 Packer 模板，我们准备构建我们的**Amazon Machine Image** (**AMI**)。

让我们运行 Packer 来构建我们的 AMI：

```py
$ packer build \
    -var "aws_access_key=..." \
    -var "aws_secret_key=..." \
    -var "django_db_password=..." \
    -var "django_db_host=A.B.us-west-2.rds.amazonaws.com" \
    -var "django_secret=..." \
    -var "email_host=smtp.example.com" \
    -var "email_host_password=..." \
    -var "web_domain=mailape.example.com" \
    packer/web_worker.json
Build 'amazon-ebs' finished.

==> Builds finished. The artifacts of successful builds are:
--> amazon-ebs: AMIs were created:
us-west-2: ami-XXXXXXXX
```

Packer 将输出我们新 AMI 镜像的 AMI ID。我们将能够使用这个 AMI 在 AWS 云中启动 EC2 实例。

如果您的模板由于缺少 Ubuntu 软件包而失败，请重试构建。在撰写本书时，Ubuntu 软件包存储库并不总是能够成功更新。

现在我们有了 AMI，我们可以部署它了。

# 在 AWS 上部署可扩展的自愈 Web 应用程序

现在我们有了基础架构和可部署的 AMI，我们可以在 AWS 上部署 Mail Ape。我们将使用 CloudFormation 定义一组资源，让我们根据需要扩展我们的应用程序。我们将定义以下三个资源：

+   一个弹性负载均衡器来在我们的 EC2 实例之间分发请求

+   一个 AutoScaling Group 来启动和终止 EC2 实例

+   一个 LaunchConfig 来描述要启动的 EC2 实例的类型

首先，让我们确保如果需要访问任何 EC2 实例来排除部署后出现的任何问题，我们有一个 SSH 密钥。

# 创建 SSH 密钥对

要在 AWS 中创建 SSH 密钥对，我们可以使用以下 AWS 命令行：

```py
$ aws ec2 create-key-pair --key-name mail_ape_production --region us-west-2
{
    "KeyFingerprint": "XXX",
    "KeyMaterial": "-----BEGIN RSA PRIVATE KEY-----\nXXX\n-----END RSA PRIVATE KEY-----",
    "KeyName": "tom-cli-test"
}
```

确保将`KeyMaterial`的值复制到您的 SSH 客户端的配置目录（通常为`~/.ssh`）-记得用实际的新行替换`\n`。

接下来，让我们开始我们的 Mail Ape 部署 CloudFormation 模板。

# 创建 Web 服务器 CloudFormation 模板

接下来，让我们创建一个 CloudFormation 模板，将 Mail Ape 服务器部署到云中。我们将使用 CloudFormation 告诉 AWS 如何扩展我们的服务器并在灾难发生时重新启动它们。我们将告诉 CloudFormation 创建以下三个资源：

+   一个**弹性负载均衡器**（**ELB**），它将能够在我们的服务器之间分发请求

+   一个 LaunchConfig，它将描述我们想要使用的 EC2 实例的 AMI、实例类型和其他细节。

+   一个自动扩展组，它将监视以确保我们拥有正确数量的健康 EC2 实例。

这三个资源是构建任何类型的可扩展自愈 AWS 应用程序的核心。

让我们从`cloudformation/web_worker.yaml`开始构建我们的 CloudFormation 模板。我们的新模板将与`cloudformation/infrastracture.yaml`具有相同的三个部分：`Parameters`、`Resources`和`Outputs`。

让我们从添加`Parameters`部分开始。

# 在 web worker CloudFormation 模板中接受参数

我们的 web worker CloudFormation 模板将接受 AMI 和 InstanceProfile 作为参数进行启动。这意味着我们不必在 Packer 和基础架构堆栈中分别硬编码我们创建的资源的名称。

让我们在`cloudformation/web_worker.yaml`中创建我们的模板：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape web worker
Parameters:
  WorkerAMI:
    Description: Worker AMI
    Type: String
  InstanceProfile:
    Description: the instance profile
    Type: String
```

现在我们有了 AMI 和 InstanceProfile 用于我们的 EC2 实例，让我们创建我们的 CloudFormation 堆栈的资源。

# 在我们的 web worker CloudFormation 模板中创建资源

接下来，我们将定义**弹性负载均衡器**（**ELB**）、启动配置和自动扩展组。这三个资源是大多数可扩展的 AWS Web 应用程序的核心。在构建模板时，我们将看看它们是如何交互的。

首先，让我们添加我们的负载均衡器：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape web worker
Parameters:
  ...
Resources:
  LoadBalancer:
    Type: "AWS::ElasticLoadBalancing::LoadBalancer"
    Properties:
      LoadBalancerName: MailApeLB
      Listeners:
        -
          InstancePort: 80
          LoadBalancerPort: 80
          Protocol: HTTP
```

在上述代码中，我们正在添加一个名为`LoadBalancer`的新资源，类型为`AWS::ElasticLoadBalancing::LoadBalancer`。ELB 需要一个名称（`MailApeLB`）和一个`Listeners`列表。每个`Listeners`条目应定义我们的 ELB 正在监听的端口（`LoadBalancerPort`）、请求将被转发到的实例端口（`InstancePort`）以及端口将使用的协议（在我们的情况下是`HTTP`）。

一个 ELB 将负责在我们启动来处理负载的任意数量的 EC2 实例之间分发 HTTP 请求。

接下来，我们将创建一个 LaunchConfig，告诉 AWS 如何启动一个新的 Mail Ape web worker：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape web worker
Parameters:
  ...
Resources:
  LoadBalancer:
    ...
  LaunchConfig:
    Type: "AWS::AutoScaling::LaunchConfiguration"
    Properties:
      ImageId: !Ref WorkerAMI
      KeyName: mail_ape_production
      SecurityGroups:
        - ssh-access
        - web-access
      InstanceType: t2.micro
      IamInstanceProfile: !Ref InstanceProfile
```

Launch Config 是`AWS::AutoScaling::LaunchConfiguration`类型的，描述了自动扩展组应该启动的新 EC2 实例的配置。让我们逐个查看所有的`Properties`，以确保我们理解它们的含义：

+   `ImageId`：这是我们希望实例运行的 AMI 的 ID。在我们的情况下，我们使用`Ref`函数从`WorkerAMI`参数获取 AMI ID。

+   `KeyName`：这是将添加到此机器的 SSH 密钥的名称。如果我们需要实时排除故障，这将非常有用。在我们的情况下，我们使用了本章早期创建的 SSH 密钥对的名称。

+   `SecurityGroups`：这是一个定义 AWS 要打开哪些端口的安全组名称列表。在我们的情况下，我们列出了我们在基础架构堆栈中创建的 web 和 SSH 组的名称。

+   `InstanceType`：这表示我们的 EC2 实例的实例类型。实例类型定义了可用于我们的 EC2 实例的计算和内存资源。在我们的情况下，我们使用的是一个非常小的经济实惠的实例，（在撰写本书时）在第一年内由 AWS 免费使用。

+   `IamInstanceProfile`：这表示我们的 EC2 实例的`InstanceProfile`。在这里，我们使用`Ref`函数来引用`InstanceProfile`参数。当我们创建我们的堆栈时，我们将使用我们早期创建的 InstanceProfile 的 ARN，该 ARN 为我们的 EC2 实例访问 SQS 提供了访问权限。

接下来，我们将定义启动由 ELB 转发的请求的 EC2 实例的 AutoScaling 组：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape web worker
Parameters:
  ...
Resources:
  LoadBalancer:
    ...
  LaunchConfig:
    ...
  WorkerGroup:
    Type: "AWS::AutoScaling::AutoScalingGroup"
    Properties:
      LaunchConfigurationName: !Ref LaunchConfig
      MinSize: 1
      MaxSize: 3
      DesiredCapacity: 1
      LoadBalancerNames:
        - !Ref LoadBalancer
```

我们的新**自动扩展组**（**ASG**）是`AWS::AutoScaling::AutoScalingGroup`类型。让我们来看看它的属性：

+   `LaunchConfigurationName`：这是此 ASG 在启动新实例时应该使用的`LaunchConfiguration`的名称。在我们的情况下，我们使用`Ref`函数来引用我们上面创建的`LaunchConfig`，即启动配置。

+   `MinSize`/`MaxSize`：这些是所需的属性，设置此组可能包含的实例的最大和最小数量。这些值可以保护我们免受意外部署太多实例可能对我们的系统或每月账单产生负面影响。在我们的情况下，我们确保至少有一个（`1`）实例，但不超过三（`3`）个。

+   `DesiredCapacity`：这告诉我们的系统应该运行多少 ASG 和多少健康的 EC2 实例。如果一个实例失败并将健康实例的数量降到`DesiredCapacity`值以下，那么 ASG 将使用其启动配置来启动更多实例。

+   `LoadBalancerNames`：这是一个 ELB 的列表，可以将请求路由到由此 ASG 启动的实例。当新的 EC2 实例成为此 ASG 的一部分时，它也将被添加到命名 ELB 路由请求的实例列表中。在我们的情况下，我们使用`Ref`函数来引用我们在模板中早期定义的 ELB。

这三个工具共同帮助我们快速而顺利地扩展我们的 Django 应用程序。ASG 为我们提供了一种说出我们希望运行多少 Mail Ape EC2 实例的方法。启动配置描述了如何启动新的 Mail Ape EC2 实例。然后 ELB 将把请求分发到 ASG 启动的所有实例。

现在我们有了我们的资源，让我们输出一些最相关的数据，以使我们的部署其余部分变得容易。

# 输出资源名称

我们将添加到我们的 CloudFormation 模板的最后一部分是`Outputs`，以便更容易地记录我们的 ELB 的地址和我们的 ASG 的名称。我们需要我们 ELB 的地址来向`mailape.example.com`添加 CNAME 记录。如果我们需要访问我们的实例（例如，运行我们的迁移），我们将需要我们 ASG 的名称。

让我们用一个`Outputs`部分更新`cloudformation/web_worker.yaml`：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape web worker
Parameters:
  ...
Resources:
  LoadBalancer:
    ...
  LaunchConfig:
    ...
  WorkerGroup:
    ...
Outputs:
  LoadBalancerDNS:
    Description: Load Balancer DNS name
    Value: !GetAtt LoadBalancer.DNSName
  AutoScalingGroupName:
    Description: Auto Scaling Group name
    Value: !Ref WorkerGroup
```

`LoadBalancerDNS`的值将是我们上面创建的 ELB 的 DNS 名称。`AutoScalingGroupName`的值将是我们的 ASG，返回 ASG 的名称。

接下来，让我们为我们的 Mail Ape 1.0 版本创建一个堆栈。

# 创建 Mail Ape 1.0 版本堆栈

现在我们有了我们的 Mail Ape web worker CloudFormation 模板，我们可以创建一个 CloudFormation 堆栈。创建堆栈时，堆栈将创建其相关资源，如 ELB、ASG 和 Launch Config。我们将使用 AWS CLI 来创建我们的堆栈：

```py
$ aws cloudformation create-stack \
    --stack-name "mail_ape_1_0" \
    --template-body "file:///path/to/mailape/cloudformation/web_worker.yaml" \
    --parameters \
      "ParameterKey=WorkerAMI,ParameterValue=AMI-XXX" \
      "ParameterKey=InstanceProfile,ParameterValue=arn:aws:iam::XXX:instance-profile/XXX" \
    --region us-west-2
```

前面的命令看起来与我们执行创建基础设施堆栈的命令非常相似，但有一些区别：

+   --stack-name：这是我们正在创建的堆栈的名称。

+   --template-body "file:///path/..."：这是一个`file://` URL，其中包含我们的 CloudFormation 模板的绝对路径。由于路径前缀以两个`/`和 Unix 路径以`/`开头，因此这里会出现一个奇怪的三重`/`。

+   --parameters：这个模板需要两个参数。我们可以以任何顺序提供它们，但必须同时提供。

+   `"ParameterKey=WorkerAMI, ParameterValue=`：对于`WorkerAMI`，我们必须提供 Packer 给我们的 AMI ID。

+   `"ParameterKey=InstanceProfile,ParameterValue`：对于 InstanceProfile，我们必须提供我们的基础设施堆栈输出的 Instance Profile ARN。

+   --region us-west-2：我们所有的工作都在`us-west-2`地区进行。

要查看我们堆栈的输出，我们可以使用 AWS CLI 的`describe-stack`命令：

```py
$ aws cloudformation describe-stacks \
    --stack-name mail_ape_1_0 \
    --region us-west-2
```

结果是一个大的 JSON 对象；这里是一个略有缩短的示例版本：

```py
{
    "Stacks": [
        {
            "StackId": "arn:aws:cloudformation:us-west-2:XXXX:stack/mail_ape_1_0/XXX",
            "StackName": "mail_ape_1_0",
            "Description": "Mail Ape web worker",
            "Parameters": [
                {
                    "ParameterKey": "InstanceProfile",
                    "ParameterValue": "arn:aws:iam::XXX:instance-profile/XXX"
                },
                {
                    "ParameterKey": "WorkerAMI",
                    "ParameterValue": "ami-XXX"
                }
            ],
            "StackStatus": "CREATE_COMPLETE",
            "Outputs": [
                {
                    "OutputKey": "AutoScalingGroupName",
                    "OutputValue": "mail_ape_1_0-WebServerGroup-XXX",
                    "Description": "Auto Scaling Group name"
                },
                {
                    "OutputKey": "LoadBalancerDNS",
                    "OutputValue": "MailApeLB-XXX.us-west-2.elb.amazonaws.com",
                    "Description": "Load Balancer DNS name"
                }
            ],
        }
    ]
}
```

我们的资源（例如 EC2 实例）直到`StackStatus`为`CREATE_COMPLETE`时才会准备就绪。创建所有相关资源可能需要几分钟。

我们特别关注`Outputs`数组中的对象：

+   第一个值给出了我们的 ASG 的名称。有了我们 ASG 的名称，我们就能够找到该 ASG 中的 EC2 实例，以防需要 SSH 到其中一个。

+   第二个值给出了我们 ELB 的 DNS 名称。我们将使用我们 ELB 的 DNS 来为我们的生产 DNS 记录创建 CNAME 记录，以便将我们的流量重定向到这里（例如，为`mailape.example.com`创建一个 CNAME 记录，将流量重定向到我们的 ELB）。

让我们看看如何 SSH 到我们的 ASG 启动的 EC2 实例。

# SSH 到 Mail Ape EC2 实例

AWS CLI 为我们提供了许多获取有关我们 EC2 实例信息的方法。让我们找到我们启动的 EC2 实例的地址：

```py
$ aws ec2 describe-instances \
 --region=us-west-2 \
 --filters='Name=tag:aws:cloudformation:stack-name,Values=mail_ape_1_0' 
```

`aws ec2 describe-instances`命令将返回关于所有 EC2 实例的大量信息。我们可以使用`--filters`命令来限制返回的 EC2 实例。当我们创建一个堆栈时，许多相关资源都带有堆栈名称的标记。这使我们可以仅筛选出我们`mail_ape_1_0`堆栈中的 EC2 实例。

以下是输出的（大大）缩短版本：

```py
{
  "Reservations": [
    {
      "Groups": [],
      "Instances": [
        {
          "ImageId": "ami-XXX",
          "InstanceId": "i-XXX",
          "InstanceType": "t2.micro",
          "KeyName": "mail_ape_production",
          "PublicDnsName": "ec2-XXX-XXX-XXX-XXX.us-west-2.compute.amazonaws.com",
          "PublicIpAddress": "XXX",
          "State": {
            "Name": "running"
          },
          "IamInstanceProfile": {
            "Arn": "arn:aws:iam::XXX:instance-profile/infrastructure-SQSClientInstance-XXX"
          },
          "SecurityGroups": [
            {
              "GroupName": "ssh-access"
            },
            {
              "GroupName": "web-access"
            }
          ],
          "Tags": [
            {
              "Key": "aws:cloudformation:stack-name",
              "Value": "mail_ape_1_0"
            } ] } ] } ] }
```

在前面的输出中，请注意`PublicDnsName`和`KeyName`。由于我们在本章前面创建了该密钥，我们可以 SSH 到这个实例：

```py
$ ssh -i /path/to/saved/ssh/key ubuntu@ec2-XXX-XXX-XXX-XXX.us-west-2.compute.amazonaws.com
```

请记住，您在前面的输出中看到的`XXX`将在您的系统中被实际值替换。

现在我们可以 SSH 到系统中，我们可以创建和迁移我们的数据库。

# 创建和迁移我们的数据库

对于我们的第一个发布，我们首先需要创建我们的数据库。为了创建我们的数据库，我们将在`database/make_database.sh`中创建一个脚本：

```py
#!/usr/bin/env bash

psql -v ON_ERROR_STOP=1 postgresql://$USER:$PASSWORD@$HOST/postgres <<-EOSQL
    CREATE DATABASE mailape;
    CREATE USER mailape;
    GRANT ALL ON DATABASE mailape to "mailape";
    ALTER USER mailape PASSWORD '$DJANGO_DB_PASSWORD';
    ALTER USER mailape CREATEDB;
EOSQL
```

此脚本使用其环境中的三个变量：

+   $USER：Postgres 主用户用户名。我们在`cloudformation/infrastructure.yaml`中将其定义为`master`。

+   $PASSWORD：Postgres 主用户的密码。我们在创建`infrastructure`堆栈时将其作为参数提供。

+   $DJANGO_DB_PASSWORD：这是 Django 数据库的密码。我们在创建 AMI 时将其作为参数提供给 Packer。

接下来，我们将通过提供变量来在本地执行此脚本：

```py
$ export USER=master
$ export PASSWORD=...
$ export DJANGO_DB_PASSWORD=...
$ bash database/make_database.sh
```

我们的 Mail Ape 数据库现在已经创建。

接下来，让我们 SSH 到我们的新 EC2 实例并运行我们的数据库迁移：

```py
$ ssh -i /path/to/saved/ssh/key ubuntu@ec2-XXX-XXX-XXX-XXX.us-west-2.compute.amazonaws.com
$ source /mailape/virtualenv/bin/activate
$ cd /mailape/django
$ export DJANGO_DB_NAME=mailape
$ export DJANGO_DB_USER=mailape
$ export DJANGO_DB_PASSWORD=...
$ export DJANGO_DB_HOST=XXX.XXX.us-west-2.rds.amazonaws.com
$ export DJANGO_DB_PORT=5432
$ export DJANGO_LOG_FILE=/var/log/mailape/mailape.log
$ export DJANGO_SECRET_KEY=...
$ export DJANGO_SETTINGS_MODULE=config.production_settings
$ python manage.py migrate
```

我们的`manage.py migrate`命令与我们在以前章节中使用的非常相似。这里的主要区别在于我们需要首先 SSH 到我们的生产 EC2 实例。

当`migrate`返回成功时，我们的数据库已经准备好，我们可以发布我们的应用程序了。

# 发布 Mail Ape 1.0

现在我们已经迁移了我们的数据库，我们准备更新`mailape.example.com`的 DNS 记录，指向我们 ELB 的 DNS 记录。一旦 DNS 记录传播，Mail Ape 就会上线。

恭喜！

# 使用 update-stack 进行扩展和缩小

使用 CloudFormation 和 Auto Scaling Groups 的一个很棒的地方是，很容易扩展我们的系统。在本节中，让我们更新我们的系统，使用两个运行 Mail Ape 的 EC2 实例。

我们可以在`cloudformation/web_worker.yaml`中更新我们的 CloudFormation 模板：

```py
AWSTemplateFormatVersion: "2010-09-09"
Description: Mail Ape web worker
Parameters:
  ..
Resources:
  LoadBalancer:
    ...
  LaunchConfig:
    ...
  WorkerGroup:
    Type: "AWS::AutoScaling::AutoScalingGroup"
    Properties:
      LaunchConfigurationName: !Ref LaunchConfig
      MinSize: 1
      MaxSize: 3
      DesiredCapacity: 2
      LoadBalancerNames:
        - !Ref LoadBalancer
Outputs:
  ..
```

我们已经将`DesiredCapacity`从 1 更新为 2。现在，我们不再创建新的堆栈，而是更新现有的堆栈：

```py
$ aws cloudformation update-stack \
    --stack-name "mail_ape_1_0" \
    --template-body "file:///path/to/mailape/cloudformation/web_worker.yaml" \
    --parameters \
      "ParameterKey=WorkerAMI,UsePreviousValue=true" \
      "ParameterKey=InstanceProfile,UsePreviousValue=true" \
    --region us-west-2
```

前面的命令看起来很像我们的`create-stack`命令。一个方便的区别是我们不需要再次提供参数值 - 我们可以简单地通知`UsePreviousValue=true`告诉 AWS 重用之前的相同值。

同样，`describe-stack`会告诉我们更新何时完成：

```py
aws cloudformation describe-stacks \
    --stack-name mail_ape_1_0 \
    --region us-west-2
```

结果是一个大型的 JSON 对象 - 这里是一个截断的示例版本：

```py
{
    "Stacks": [
        {
            "StackId": "arn:aws:cloudformation:us-west-2:XXXX:stack/mail_ape_1_0/XXX",
            "StackName": "mail_ape_1_0",
            "Description": "Mail Ape web worker",
            "StackStatus": "UPDATE_COMPLETE"
        }
    ]
}
```

一旦我们的`StackStatus`为`UPDATE_COMPLETE`，我们的 ASG 将使用新的设置进行更新。ASG 可能需要几分钟来启动新的 EC2 实例，但我们可以使用我们之前创建的`describe-instances`命令来查找它：

```py
$ aws ec2 describe-instances \
 --region=us-west-2 \
 --filters='Name=tag:aws:cloudformation:stack-name,Values=mail_ape_1_0'
```

最终，它将返回两个实例。以下是输出的高度截断版本：

```py
{
  "Reservations": [
    {
      "Groups": [],
      "Instances": [
        {
          "ImageId": "ami-XXX",
          "InstanceId": "i-XXX",
          "PublicDnsName": "ec2-XXX-XXX-XXX-XXX.us-west-2.compute.amazonaws.com",
          "State": { "Name": "running" }
        },
        {
          "ImageId": "ami-XXX",
          "InstanceId": "i-XXX",
          "PublicDnsName": "ec2-XXX-XXX-XXX-XXX.us-west-2.compute.amazonaws.com",
          "State": { "Name": "running" }
        } ] } ] }
```

要缩小到一个实例，只需更新您的`web_worker.yaml`模板并再次运行`update-stack`。

恭喜！您现在知道如何将 Mail Ape 扩展到处理更高的负载，然后在非高峰时期缩小规模。

请记住，亚马逊的收费是基于使用情况的。如果您在阅读本书的过程中进行了扩展，请记住要缩小规模，否则您可能会被收取比预期更多的费用。确保您阅读关于 AWS 免费套餐限制的信息[`aws.amazon.com/free/`](https://aws.amazon.com/free/)。

# 总结

在本章中，我们将我们的 Mail Ape 应用程序并在 AWS 云中的生产环境中启动。我们使用 AWS CloudFormation 将我们的 AWS 资源声明为代码，使得跟踪我们需要的内容和发生了什么变化就像在我们的代码库的其余部分一样容易。我们使用 Packer 构建了我们的 Mail Ape 服务器运行的镜像，再次使我们能够将我们的服务器配置作为代码进行跟踪。最后，我们将 Mail Ape 启动到云中，并学会了如何进行扩展和缩小。

现在我们已经完成了学习构建 Django Web 应用程序的旅程，让我们回顾一下我们学到的一些东西。在三个项目中，我们看到了 Django 如何将代码组织成模型、视图和模板。我们学会了如何使用 Django 的表单类和 Django Rest Framework 的序列化器类进行输入验证。我们审查了安全最佳实践、缓存以及如何发送电子邮件。我们看到了如何将我们的代码部署到 Linux 服务器、Docker 容器和 AWS 云中。

您已经准备好使用 Django 来实现您的想法了！加油！
