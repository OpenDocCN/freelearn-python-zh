# 第十六章：使用 Boto3 自动化 AWS

在之前的章节中，我们探讨了如何使用 Python 自动化 OpenStack 和 VMware 私有云。我们将继续通过自动化最受欢迎的公共云之一——亚马逊网络服务（AWS）来继续我们的云自动化之旅。在本章中，我们将探讨如何使用 Python 脚本创建 Amazon Elastic Compute Cloud（EC2）和 Amazon Simple Storage Systems（S3）。

本章将涵盖以下主题：

+   AWS Python 模块

+   管理 AWS 实例

+   自动化 AWS S3 服务

# AWS Python 模块

Amazon EC2 是一个可扩展的计算系统，用于为托管不同虚拟机（例如 OpenStack 生态系统中的 nova-compute 项目）提供虚拟化层。它可以与其他服务（如 S3、Route 53 和 AMI）通信，以实例化实例。基本上，您可以将 EC2 视为其他在虚拟基础设施管理器上设置的虚拟化程序（如 KVM 和 VMware）之上的抽象层。EC2 将接收传入的 API 调用，然后将其转换为适合每个虚拟化程序的调用。

Amazon Machine Image（AMI）是一个打包的镜像系统，其中包含了启动虚拟机所需的操作系统和软件包（类似于 OpenStack 中的 Glance）。您可以从现有的虚拟机创建自己的 AMI，并在需要在其他基础设施上复制这些机器时使用它，或者您可以简单地从互联网或亚马逊市场上选择公开可用的 AMI。我们需要从亚马逊网络控制台获取 AMI ID，并将其添加到我们的 Python 脚本中。

AWS 设计了一个名为 Boto3 的 SDK（[`github.com/boto/boto3`](https://github.com/boto/boto3)），允许 Python 开发人员编写与不同服务的 API 进行交互和消费的脚本和软件，如 Amazon EC2 和 Amazon S3。该库是为提供对 Python 2.6.5、2.7+和 3.3 的本地支持而编写的。

Boto3 的主要功能在官方文档中有描述，网址为[`boto3.readthedocs.io/en/latest/guide/new.html`](https://boto3.readthedocs.io/en/latest/guide/new.html)，以下是一些重要功能：

+   资源：高级、面向对象的接口。

+   集合：用于迭代和操作资源组的工具。

+   客户端：低级服务连接。

+   分页器：自动分页响应。

+   等待者：一种暂停执行直到达到某种状态或发生故障的方式。每个 AWS 资源都有一个等待者名称，可以使用`<resource_name>.waiter_names`访问。

# Boto3 安装

在连接到 AWS 之前需要一些东西：

1.  首先，您需要一个具有创建、修改和删除基础设施权限的亚马逊管理员帐户。

1.  其次，安装用于与 AWS 交互的`boto3` Python 模块。您可以通过转到 AWS 身份和访问管理（IAM）控制台并添加新用户来创建一个专用于发送 API 请求的用户。您应该在“访问类型”部分下看到“编程访问”选项。 

1.  现在，您需要分配一个允许在亚马逊服务中具有完全访问权限的策略，例如 EC2 和 S3。通过单击“附加现有策略到用户”并将 AmazonEC2FullAccess 和 AmazonS3FullAccess 策略附加到用户名来实现。

1.  最后，点击“创建用户”以添加具有配置选项和策略的用户。

您可以在 AWS 上注册免费的基础套餐帐户，这将使您在 12 个月内获得亚马逊提供的许多服务。免费访问可以在[`aws.amazon.com/free/`](https://aws.amazon.com/free/)上获得。

在使用 Python 脚本管理 AWS 时，访问密钥 ID 用于发送 API 请求并从 API 服务器获取响应。我们不会使用用户名或密码发送请求，因为它们很容易被他人捕获。此信息是通过下载创建用户名后出现的文本文件获得的。重要的是将此文件放在安全的位置并为其提供适当的 Linux 权限，以打开和读取文件内容。

另一种方法是在您的家目录下创建一个`.aws`目录，并在其中放置两个文件：`credentials`和`config`。第一个文件将同时包含访问密钥 ID 和秘密访问 ID。

`~/.aws/credentials`如下所示：

```py
[default]
aws_access_key_id=AKIAIOSFODNN7EXAMPLE
aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

第二个文件将保存用户特定的配置，例如首选数据中心（区域），用于托管创建的虚拟机。在下面的示例中，我们指定要在`us-west-2`数据中心托管我们的机器。

配置文件`~/.aws/config`如下所示：

```py
[default]
region=us-west-2
```

现在，安装`boto3`需要使用通常的`pip`命令来获取最新的`boto3`版本：

```py
pip install boto3
```

![](img/00203.jpeg)

要验证模块是否成功安装，请在 Python 控制台中导入`boto3`，您不应该看到任何导入错误报告：

![](img/00204.jpeg)

# 管理 AWS 实例

现在，我们准备使用`boto3`创建我们的第一个虚拟机。正如我们所讨论的，我们需要 AMI，我们将从中实例化一个实例。将 AMI 视为 Python 类；创建一个实例将从中创建一个对象。我们将使用 Amazon Linux AMI，这是由 Amazon 维护的特殊 Linux 操作系统，用于部署 Linux 机器而不收取任何额外费用。您可以在每个区域找到完整的 AMI ID，网址为[`aws.amazon.com/amazon-linux-ami/`](https://aws.amazon.com/amazon-linux-ami/)：

![](img/00205.jpeg)

```py
import boto3
ec2 = boto3.resource('ec2') instance = ec2.create_instances(ImageId='ami-824c4ee2', MinCount=1, MaxCount=1, InstanceType='m5.xlarge',
  Placement={'AvailabilityZone': 'us-west-2'},
  ) print(instance[0])   
```

在上面的示例中，以下内容适用：

1.  我们导入了之前安装的`boto3`模块。

1.  然后，我们指定了要与之交互的资源类型，即 EC2，并将其分配给`ec2`对象。

1.  现在，我们有资格使用`create_instance()`方法，并为其提供实例参数，例如`ImageID`和`InstanceType`（类似于 OpenStack 中的 flavor，它确定了计算和内存方面的实例规格），以及我们应该在`AvailabilityZone`中创建此实例。

1.  `MinCount`和`MaxCount`确定 EC2 在扩展我们的实例时可以走多远。例如，当一个实例发生高 CPU 时，EC2 将自动部署另一个实例，以分享负载并保持服务处于健康状态。

1.  最后，我们打印了要在下一个脚本中使用的实例 ID。

输出如下：

![](img/00206.jpeg)您可以在以下链接中检查所有有效的 Amazon EC2 实例类型；请仔细阅读，以免因选择错误的类型而被过度收费：[`aws.amazon.com/ec2/instance-types/`](https://aws.amazon.com/ec2/instance-types/)

# 实例终止

打印的 ID 用于 CRUD 操作，以便稍后管理或终止实例。例如，我们可以使用之前创建的`ec2`资源提供的`terminate()`方法来终止实例：

```py
import boto3
ec2 = boto3.resource('ec2') instance_id = "i-0a81k3ndl29175220" instance = ec2.Instance(instance_id) instance.terminate() 
```

请注意，在前面的代码中我们硬编码了`instance_id`（当您需要创建一个可以在不同环境中使用的动态 Python 脚本时，这并不总是适用）。我们可以使用 Python 中可用的其他输入方法，例如`raw_input()`，从用户那里获取输入或查询我们帐户中可用的实例，并让 Python 提示我们需要终止哪些实例。另一个用例是创建一个 Python 脚本，检查我们实例的最后登录时间或资源消耗；如果它们超过特定值，我们将终止该实例。这在实验室环境中非常有用，您不希望因为恶意或设计不良的软件而被收取额外资源的费用。

# 自动化 AWS S3 服务

AWS **简单存储系统**（**S3**）提供了安全和高度可扩展的对象存储服务。您可以使用此服务存储任意数量的数据，并从任何地方恢复它。系统为您提供了版本控制选项，因此您可以回滚到文件的任何先前版本。此外，它提供了 REST Web 服务 API，因此您可以从外部应用程序访问它。

当数据传入 S3 时，S3 将为其创建一个`对象`，并将这些对象存储在`存储桶`中（将它们视为文件夹）。您可以为每个创建的存储桶提供复杂的用户权限，并且还可以控制其可见性（公共、共享或私有）。存储桶访问可以是策略或**访问控制列表**（**ACL**）。

存储桶还存储有描述键值对中对象的元数据，您可以通过 HTTP `POST`方法创建和设置。元数据可以包括对象的名称、大小和日期，或者您想要的任何其他自定义键值对。用户帐户最多可以拥有 100 个存储桶，但每个存储桶内托管的对象大小没有限制。

# 创建存储桶

与 AWS S3 服务交互时，首先要做的事情是创建一个用于存储文件的存储桶。在这种情况下，我们将`S3`提供给`boto3.resource()`。这将告诉`boto3`开始初始化过程，并加载与 S3 API 系统交互所需的命令：

```py
import boto3
s3_resource = boto3.resource("s3")   bucket = s3_resource.create_bucket(Bucket="my_first_bucket", CreateBucketConfiguration={
  'LocationConstraint': 'us-west-2'}) print(bucket)
```

在前面的例子中，以下内容适用：

1.  我们导入了之前安装的`boto3`模块。

1.  然后，我们指定了我们想要与之交互的资源类型，即`s3`，并将其分配给`s3_resource`对象。

1.  现在，我们可以在资源内部使用`create_bucket()`方法，并为其提供所需的参数来创建存储桶，例如`Bucket`，我们可以指定其名称。请记住，存储桶名称必须是唯一的，且之前不能已经使用过。第二个参数是`CreateBucketConfiguration`字典，我们在其中设置了创建存储桶的数据中心位置。

# 将文件上传到存储桶

现在，我们需要利用创建的存储桶并将文件上传到其中。请记住，存储桶中的文件表示为对象。因此，`boto3`提供了一些包含对象作为其一部分的方法。我们将从使用`put_object()`开始。此方法将文件上传到创建的存储桶并将其存储为对象：

```py
import boto3
s3_resource = boto3.resource("s3") bucket = s3_resource.Bucket("my_first_bucket")   with open('~/test_file.txt', 'rb') as uploaded_data:
  bucket.put_object(Body=uploaded_data) 
```

在前面的例子中，以下内容适用：

1.  我们导入了之前安装的`boto3`模块。

1.  然后，我们指定了我们想要与之交互的资源类型，即`s3`，并将其分配给`s3_resource`对象。

1.  我们通过`Bucket()`方法访问了`my_first_bucket`并将返回的值分配给了存储桶变量。

1.  然后，我们使用`with`子句打开了一个文件，并将其命名为`uploaded_data`。请注意，我们以二进制数据的形式打开了文件，使用了`rb`标志。

1.  最后，我们使用存储桶空间中提供的`put_object()`方法将二进制数据上传到我们的存储桶。

# 删除存储桶

要完成对存储桶的 CRUD 操作，我们需要做的最后一件事是删除存储桶。这是通过在我们的存储桶变量上调用`delete()`方法来实现的，前提是它已经存在，并且我们通过名称引用它，就像我们创建它并向其中上传数据一样。然而，当存储桶不为空时，`delete()`可能会失败。因此，我们将使用`bucket_objects.all().delete()`方法获取存储桶内的所有对象，然后对它们应用`delete()`操作，最后删除存储桶：

```py
import boto3
s3_resource = boto3.resource("s3") bucket = s3_resource.Bucket("my_first_bucket") bucket.objects.all().delete() bucket.delete()
```

# 总结

在本章中，我们学习了如何安装亚马逊弹性计算云（EC2），以及学习了 Boto3 及其安装。我们还学习了如何自动化 AWS S3 服务。

在下一章中，我们将学习 SCAPY 框架，这是一个强大的 Python 工具，用于构建和制作数据包并将其发送到网络上。
