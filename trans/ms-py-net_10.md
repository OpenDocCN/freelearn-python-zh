# AWS 云网络

云计算是当今计算领域的主要趋势之一。公共云提供商已经改变了高科技行业，以及从零开始推出服务的含义。我们不再需要构建自己的基础设施；我们可以支付公共云提供商租用他们资源的一部分来满足我们的基础设施需求。如今，在任何技术会议或聚会上，我们很难找到一个没有了解、使用或构建基于云的服务的人。云计算已经到来，我们最好习惯与之一起工作。

云计算有几种服务模型，大致分为软件即服务（SaaS）（[`en.wikipedia.org/wiki/Software_as_a_service`](https://en.wikipedia.org/wiki/Software_as_a_service)）、平台即服务（PaaS）（[`en.wikipedia.org/wiki/Cloud_computing#Platform_as_a_service_(PaaS)`](https://en.wikipedia.org/wiki/Cloud_computing#Platform_as_a_service_(PaaS)）和基础设施即服务（IaaS）（[`en.wikipedia.org/wiki/Infrastructure_as_a_service`](https://en.wikipedia.org/wiki/Infrastructure_as_a_service)）。每种服务模型从用户的角度提供了不同的抽象级别。对我们来说，网络是基础设施即服务提供的一部分，也是本章的重点。

亚马逊云服务（AWS）是第一家提供 IaaS 公共云服务的公司，也是 2018 年市场份额方面的明显领导者。如果我们将“软件定义网络”（SDN）定义为一组软件服务共同创建网络结构 - IP 地址、访问列表、网络地址转换、路由器 - 我们可以说 AWS 是世界上最大的 SDN 实现。他们利用全球网络、数据中心和主机的大规模来提供令人惊叹的各种网络服务。

如果您有兴趣了解亚马逊的规模和网络，我强烈建议您观看 James Hamilton 在 2014 年 AWS re:Invent 的演讲：[`www.youtube.com/watch?v=JIQETrFC_SQ`](https://www.youtube.com/watch?v=JIQETrFC_SQ)。这是一个罕见的内部人员对 AWS 规模和创新的视角。

在本章中，我们将讨论 AWS 云服务提供的网络服务以及如何使用 Python 与它们一起工作：

+   AWS 设置和网络概述

+   虚拟私有云

+   直接连接和 VPN

+   网络扩展服务

+   其他 AWS 网络服务

# AWS 设置

如果您还没有 AWS 账户并希望跟随这些示例，请登录[`aws.amazon.com/`](https://aws.amazon.com/)并注册。这个过程非常简单明了；您需要一张信用卡和某种形式的验证。AWS 在免费套餐中提供了许多服务（[`aws.amazon.com/free/`](https://aws.amazon.com/free/)），在一定水平上可以免费使用一些最受欢迎的服务。

列出的一些服务在第一年是免费的，其他服务在一定限额内是免费的，没有时间限制。请查看 AWS 网站获取最新的优惠。

![](img/c63f6039-b3d3-41b9-9654-c91aa3b51537.png)AWS 免费套餐

一旦您有了账户，您可以通过 AWS 控制台（[`console.aws.amazon.com/`](https://console.aws.amazon.com/)）登录并查看 AWS 提供的不同服务。控制台是我们可以配置所有服务并查看每月账单的地方。

![](img/40bd402d-f546-4d6c-b4bc-157cff3d978a.png)AWS 控制台

# AWS CLI 和 Python SDK

我们也可以通过命令行界面管理 AWS 服务。AWS CLI 是一个可以通过 PIP 安装的 Python 包（[`docs.aws.amazon.com/cli/latest/userguide/installing.html`](https://docs.aws.amazon.com/cli/latest/userguide/installing.html)）。让我们在 Ubuntu 主机上安装它：

```py
$ sudo pip3 install awscli
$ aws --version
aws-cli/1.15.59 Python/3.5.2 Linux/4.15.0-30-generic botocore/1.10.58
```

安装了 AWS CLI 后，为了更轻松和更安全地访问，我们将创建一个用户并使用用户凭据配置 AWS CLI。让我们回到 AWS 控制台，选择 IAM 进行用户和访问管理：

![](img/5fc1647e-cd89-424f-81af-e674415d622c.png) AWS IAM

我们可以在左侧面板上选择“用户”来创建用户：

![](img/697ef15f-61f3-445d-a5ef-3f24332c0a84.png)

选择编程访问并将用户分配给默认管理员组：

![](img/b6f17d3f-d822-41e8-a054-4089af2a9e9d.png)

最后一步将显示访问密钥 ID 和秘密访问密钥。将它们复制到文本文件中并保存在安全的地方：

![](img/3a53eaff-674a-4c07-8d58-3e256e86bfab.png)

我们将通过终端中的`aws configure`完成 AWS CLI 身份验证凭据设置。我们将在接下来的部分中介绍 AWS 地区；现在我们将使用`us-east-1`，但随时可以返回并更改这个值：

```py
$ aws configure
AWS Access Key ID [None]: <key>
AWS Secret Access Key [None]: <secret>
Default region name [None]: us-east-1
Default output format [None]: json
```

我们还将安装 AWS Python SDK，Boto3 ([`boto3.readthedocs.io/en/latest/`](https://boto3.readthedocs.io/en/latest/))：

```py
$ sudo pip install boto3
$ sudo pip3 install boto3

# verification
$ python3
Python 3.5.2 (default, Nov 23 2017, 16:37:01)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import boto3
>>> exit()
```

我们现在准备继续进行后续部分，从介绍 AWS 云网络服务开始。

# AWS 网络概述

当我们讨论 AWS 服务时，我们需要从地区和可用性区开始。它们对我们所有的服务都有重大影响。在撰写本书时，AWS 列出了 18 个地区、55 个可用性区和一个全球范围的本地地区。用 AWS 全球基础设施的话来说，([`aws.amazon.com/about-aws/global-infrastructure/`](https://aws.amazon.com/about-aws/global-infrastructure/))：

“AWS 云基础设施建立在地区和可用性区（AZ）周围。AWS 地区提供多个物理上分离和隔离的可用性区，这些区域通过低延迟、高吞吐量和高度冗余的网络连接在一起。”

AWS 提供的一些服务是全球性的，但大多数服务是基于地区的。对我们来说，这意味着我们应该在最接近我们预期用户的地区建立基础设施。这将减少服务对客户的延迟。如果我们的用户在美国东海岸，如果服务是基于地区的，我们应该选择`us-east-1`（北弗吉尼亚）或`us-east-2`（俄亥俄）作为我们的地区：

![](img/8255a3ce-7fed-4297-aa14-49c62b673c65.png)AWS 地区

并非所有地区都对所有用户可用，例如，GovCloud 和中国地区默认情况下对美国用户不可用。您可以通过`aws ec2 describe-regions`列出对您可用的地区：

```py
$ aws ec2 describe-regions
{
 "Regions": 
 {
 "RegionName": "ap-south-1",
 "Endpoint": "ec2.ap-south-1.amazonaws.com"
 },
 {
 "RegionName": "eu-west-3",
 "Endpoint": "ec2.eu-west-3.amazonaws.com"
 },
...
```

所有地区都是完全独立的。大多数资源不会在地区之间复制。如果我们有多个地区，比如`US-East`和`US-West`，并且需要它们之间的冗余，我们将需要自己复制必要的资源。选择地区的方式是在控制台右上角：

![如果服务是基于地区的，例如 EC2，只有在选择正确的地区时，门户才会显示该服务。如果我们的 EC2 实例在`us-east-1`，而我们正在查看 us-west-1 门户，则不会显示任何 EC2 实例。我犯过这个错误几次，并且想知道我的所有实例都去哪了！在前面的 AWS 地区截图中，地区后面的数字代表每个地区的 AZ 数量。每个地区有多个可用性区。每个可用性区都是隔离的，但地区中的可用性区通过低延迟的光纤连接在一起：![](img/9d72cf3c-5508-4524-966e-84554a9fa937.png)AWS 地区和可用性区

我们构建的许多资源都会在可用性区复制。AZ 的概念非常重要，它的约束对我们构建的网络服务非常重要。

AWS 独立地为每个账户将可用区映射到标识符。例如，我的可用区 us-eas-1a 可能与另一个账户的`us-east-1a`不同。

我们可以使用 AWS CLI 检查一个区域中的可用区：

```py
$ aws ec2 describe-availability-zones --region us-east-1
{
 "AvailabilityZones": [
 {
 "Messages": [],
 "RegionName": "us-east-1",
 "State": "available",
 "ZoneName": "us-east-1a"
 },
 {
 "Messages": [],
 "RegionName": "us-east-1",
 "State": "available",
 "ZoneName": "us-east-1b"
 },
...
```

为什么我们如此关心区域和可用区？正如我们将在接下来的几节中看到的，网络服务通常受区域和可用区的限制。例如，**虚拟私有云（VPC）**需要完全位于一个区域，每个子网需要完全位于一个可用区。另一方面，**NAT 网关**是与可用区相关的，因此如果我们需要冗余，就需要为每个可用区创建一个。我们将更详细地介绍这两项服务，但它们的用例在这里作为 AWS 网络服务提供的基础的例子。

**AWS 边缘位置**是**AWS CloudFront**内容传递网络的一部分，分布在 26 个国家的 59 个城市。这些边缘位置用于以低延迟分发内容，比整个数据中心的占地面积小。有时，人们会误将边缘位置的出现地点误认为是完整的 AWS 区域。如果占地面积仅列为边缘位置，那么 AWS 服务，如 EC2 或 S3，将不会提供。我们将在*AWS CloudFront*部分重新讨论边缘位置。

**AWS Transit Centers**是 AWS 网络中最少有文档记录的方面之一。它在 James Hamilton 的 2014 年**AWS re:Invent**主题演讲中提到（[`www.youtube.com/watch?v=JIQETrFC_SQ`](https://www.youtube.com/watch?v=JIQETrFC_SQ)），作为该区域不同可用区的聚合点。公平地说，我们不知道转换中心是否仍然存在并且在这些年后是否仍然起作用。然而，对于转换中心的位置以及它与我们将在本章后面看到的**AWS Direct Connect**服务的相关性，做出一个合理的猜测是公平的。

James Hamilton 是 AWS 的副总裁和杰出工程师之一，是 AWS 最有影响力的技术专家之一。如果有人在 AWS 网络方面具有权威性，那就是他。您可以在他的博客 Perspectives 上阅读更多关于他的愿景，网址为[`perspectives.mvdirona.com/`](https://perspectives.mvdirona.com/)。

在一个章节中不可能涵盖所有与 AWS 相关的服务。有一些与网络直接相关的相关服务我们没有空间来涵盖，但我们应该熟悉：

+   **身份和访问管理**（**IAM**）服务，[`aws.amazon.com/iam/`](https://aws.amazon.com/iam/)，是使我们能够安全地管理对 AWS 服务和资源的访问的服务。

+   **Amazon 资源名称**（**ARNs**），[`docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html`](https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html)，在整个 AWS 中唯一标识 AWS 资源。当我们需要识别需要访问我们的 VPC 资源的服务时，这个资源名称是重要的，比如 DynamoDB 和 API Gateway。

+   **Amazon 弹性计算云**（**EC2**），[`aws.amazon.com/ec2/`](https://aws.amazon.com/ec2/)，是使我们能够通过 AWS 接口获取和配置计算能力，如 Linux 和 Windows 实例的服务。我们将在本章的示例中使用 EC2 实例。

为了学习的目的，我们将排除 AWS GovCloud（美国）和中国，它们都不使用 AWS 全球基础设施，并且有自己的限制。

这是对 AWS 网络概述的一个相对较长的介绍，但是非常重要。这些概念和术语将在本书的其余章节中被引用。在接下来的章节中，我们将看一下 AWS 网络中最重要的概念（在我看来）：虚拟私有云。

# 虚拟私有云

亚马逊虚拟私有云（Amazon VPC）使客户能够将 AWS 资源启动到专门为客户账户提供的虚拟网络中。这是一个真正可定制的网络，允许您定义自己的 IP 地址范围，添加和删除子网，创建路由，添加 VPN 网关，关联安全策略，将 EC2 实例连接到自己的数据中心等等。在 VPC 不可用的早期，AZ 中的所有 EC2 实例都在一个共享的单一平面网络上。客户将把他们的信息放在云中会有多舒服呢？我想不会很舒服。从 2007 年 EC2 推出到 2009 年 VPC 推出之前，VPC 功能是 AWS 最受欢迎的功能之一。

在 VPC 中离开您的 EC2 主机的数据包将被 Hypervisor 拦截。Hypervisor 将使用了解我们 VPC 结构的映射服务对其进行检查。离开您的 EC2 主机的数据包将使用 AWS 真实服务器的源和目的地地址进行封装。封装和映射服务允许 VPC 的灵活性，但也有一些 VPC 的限制（多播，嗅探）。毕竟，这是一个虚拟网络。

自 2013 年 12 月以来，所有 EC2 实例都是 VPC-only。如果我们使用启动向导创建 EC2 实例，它将自动放入具有虚拟互联网网关以进行公共访问的默认 VPC。在我看来，除了最基本的用例，所有情况都应该使用默认 VPC。对于大多数情况，我们需要定义我们的非默认自定义 VPC。

让我们在`us-east-1`使用 AWS 控制台创建以下 VPC：

![](img/afa05231-3c87-47f3-a238-273e37860134.png)我们在美国东部的第一个 VPC

如果您还记得，VPC 是 AWS 区域绑定的，子网是基于可用性区域的。我们的第一个 VPC 将基于`us-east-1`；三个子网将分配给 1a、1b 和 1c 中的三个不同的可用性区域。

使用 AWS 控制台创建 VPC 和子网非常简单，AWS 在网上提供了许多很好的教程。我已经在 VPC 仪表板上列出了相关链接的步骤：

![](img/c1fe117d-9253-45cb-bb5a-934f48d39aef.png)

前两个步骤是点对点的过程，大多数网络工程师甚至没有先前的经验也可以完成。默认情况下，VPC 只包含本地路由`10.0.0.0/16`。现在，我们将创建一个互联网网关并将其与 VPC 关联：

![](img/4d402e8e-89ff-4a1d-82f2-9ac736fffdca.png)

然后，我们可以创建一个自定义路由表，其中包含指向互联网网关的默认路由。我们将把这个路由表与我们在`us-east-1a`的子网`10.0.0.0/24`关联，从而使其可以面向公众：

![](img/04ccec30-9248-4b9d-8f58-ea68cbbcd6d8.png)路由表

让我们使用 Boto3 Python SDK 来查看我们创建了什么；我使用标签`mastering_python_networking_demo`作为 VPC 的标签，我们可以将其用作过滤器：

```py
$ cat Chapter10_1_query_vpc.py
#!/usr/bin/env python3

import json, boto3

region = 'us-east-1'
vpc_name = 'mastering_python_networking_demo'

ec2 = boto3.resource('ec2', region_name=region)
client = boto3.client('ec2')

filters = [{'Name':'tag:Name', 'Values':[vpc_name]}]

vpcs = list(ec2.vpcs.filter(Filters=filters))
for vpc in vpcs:
    response = client.describe_vpcs(
                 VpcIds=[vpc.id,]
                )
    print(json.dumps(response, sort_keys=True, indent=4))
```

此脚本将允许我们以编程方式查询我们创建的 VPC 的区域：

```py
$ python3 Chapter10_1_query_vpc.py
{
 "ResponseMetadata": {
 "HTTPHeaders": {
 "content-type": "text/xml;charset=UTF-8",
 ...
 },
 "HTTPStatusCode": 200,
 "RequestId": "48e19be5-01c1-469b-b6ff-9c45f2745483",
 "RetryAttempts": 0
 },
 "Vpcs": [
 {
 "CidrBlock": "10.0.0.0/16",
 "CidrBlockAssociationSet": [
 {
 "AssociationId": "...",
 "CidrBlock": "10.0.0.0/16",
 "CidrBlockState": {
 "State": "associated"
 }
 }
 ],
 "DhcpOptionsId": "dopt-....",
 "InstanceTenancy": "default",
 "IsDefault": false,
 "State": "available",
 "Tags": [
 {
 "Key": "Name",
 "Value": "mastering_python_networking_demo"
 }
 ],
 "VpcId": "vpc-...."
 }
 ]
}

```

Boto3 VPC API 文档可以在[`boto3.readthedocs.io/en/latest/reference/services/ec2.html#vpc`](https://boto3.readthedocs.io/en/latest/reference/services/ec2.html#vpc)找到。

您可能想知道 VPC 中的子网如何相互到达。在物理网络中，网络需要连接到路由器才能到达其本地网络之外。在 VPC 中也是如此，只是它是一个具有本地网络默认路由表的*隐式路由器*，在我们的示例中是`10.0.0.0/16`。当我们创建 VPC 时，将创建此隐式路由器。

# 路由表和路由目标

路由是网络工程中最重要的主题之一。值得更仔细地研究它。我们已经看到在创建 VPC 时有一个隐式路由器和主路由表。从上一个示例中，我们创建了一个互联网网关，一个默认路由指向互联网网关的自定义路由表，并将自定义路由表与子网关联。

路由目标的概念是 VPC 与传统网络有些不同的地方。总之：

+   每个 VPC 都有一个隐式路由器

+   每个 VPC 都有一个带有本地路由的主路由表

+   您可以创建自定义路由表

+   每个子网可以遵循自定义路由表或默认的主路由表

+   路由表路由目标可以是互联网网关、NAT 网关、VPC 对等连接等

我们可以使用 Boto3 查看自定义路由表和子网的关联：

```py
$ cat Chapter10_2_query_route_tables.py
#!/usr/bin/env python3

import json, boto3

region = 'us-east-1'
vpc_name = 'mastering_python_networking_demo'

ec2 = boto3.resource('ec2', region_name=region)
client = boto3.client('ec2')

response = client.describe_route_tables()
print(json.dumps(response['RouteTables'][0], sort_keys=True, indent=4))
```

我们只有一个自定义路由表：

```py
$ python3 Chapter10_2_query_route_tables.py
{
 "Associations": [
 {
 ....
 }
 ],
 "PropagatingVgws": [],
 "RouteTableId": "rtb-6bee5514",
 "Routes": [
 {
 "DestinationCidrBlock": "10.0.0.0/16",
 "GatewayId": "local",
 "Origin": "CreateRouteTable",
 "State": "active"
 },
 {
 "DestinationCidrBlock": "0.0.0.0/0",
 "GatewayId": "igw-...",
 "Origin": "CreateRoute",
 "State": "active"
 }
 ],
 "Tags": [
 {
 "Key": "Name",
 "Value": "public_internet_gateway"
 }
 ],
 "VpcId": "vpc-..."
}
```

通过点击左侧子网部分并按照屏幕上的指示进行操作，创建子网非常简单。对于我们的目的，我们将创建三个子网，`10.0.0.0/24`公共子网，`10.0.1.0/24`和`10.0.2.0/24`私有子网。

现在我们有一个带有三个子网的工作 VPC：一个公共子网和两个私有子网。到目前为止，我们已经使用 AWS CLI 和 Boto3 库与 AWS VPC 进行交互。让我们看看另一个自动化工具**CloudFormation**。

# 使用 CloudFormation 进行自动化

AWS CloudFomation ([`aws.amazon.com/cloudformation/`](https://aws.amazon.com/cloudformation/))，是我们可以使用文本文件描述和启动所需资源的一种方式。我们可以使用 CloudFormation 在`us-west-1`地区配置另一个 VPC：

![](img/d37b0edc-2d29-4752-a72b-62c4184507a2.png)美国西部的 VPC

CloudFormation 模板可以是 YAML 或 JSON；我们将使用 YAML 来创建我们的第一个配置模板：

```py
$ cat Chapter10_3_cloud_formation.yml
AWSTemplateFormatVersion: '2010-09-09'
Description: Create VPC in us-west-1
Resources:
 myVPC:
 Type: AWS::EC2::VPC
 Properties:
 CidrBlock: '10.1.0.0/16'
 EnableDnsSupport: 'false'
 EnableDnsHostnames: 'false'
 Tags:
 - Key: Name
 Value: 'mastering_python_networking_demo_2'
```

我们可以通过 AWS CLI 执行模板。请注意，在我们的执行中指定了`us-west-1`地区：

```py
$ aws --region us-west-1 cloudformation create-stack --stack-name 'mpn-ch10-demo' --template-body file://Chapter10_3_cloud_formation.yml
{
 "StackId": "arn:aws:cloudformation:us-west-1:<skip>:stack/mpn-ch10-demo/<skip>"
}
```

我们可以通过 AWS CLI 验证状态：

```py
$ aws --region us-west-1 cloudformation describe-stacks --stack-name mpn-ch10-demo
{
 "Stacks": [
 {
 "CreationTime": "2018-07-18T18:45:25.690Z",
 "Description": "Create VPC in us-west-1",
 "DisableRollback": false,
 "StackName": "mpn-ch10-demo",
 "RollbackConfiguration": {},
 "StackStatus": "CREATE_COMPLETE",
 "NotificationARNs": [],
 "Tags": [],
 "EnableTerminationProtection": false,
 "StackId": "arn:aws:cloudformation:us-west-1<skip>"
 }
 ]
}
```

为了演示目的，最后一个 CloudFormation 模板创建了一个没有任何子网的 VPC。让我们删除该 VPC，并使用以下模板创建 VPC 和子网。请注意，在 VPC 创建之前我们将没有 VPC-id，因此我们将使用特殊变量来引用子网创建中的 VPC-id。这是我们可以用于其他资源的相同技术，比如路由表和互联网网关：

```py
$ cat Chapter10_4_cloud_formation_full.yml
AWSTemplateFormatVersion: '2010-09-09'
Description: Create subnet in us-west-1
Resources:
 myVPC:
 Type: AWS::EC2::VPC
 Properties:
 CidrBlock: '10.1.0.0/16'
 EnableDnsSupport: 'false'
 EnableDnsHostnames: 'false'
 Tags:
 - Key: Name
 Value: 'mastering_python_networking_demo_2'

 mySubnet:
 Type: AWS::EC2::Subnet
 Properties:
 VpcId: !Ref myVPC
 CidrBlock: '10.1.0.0/24'
 AvailabilityZone: 'us-west-1a'
 Tags:
 - Key: Name
 Value: 'mpn_demo_subnet_1'
```

我们可以执行并验证资源的创建如下：

```py
$ aws --region us-west-1 cloudformation create-stack --stack-name mpn-ch10-demo-2 --template-body file://Chapter10_4_cloud_formation_full.yml
{
 "StackId": "arn:aws:cloudformation:us-west-1:<skip>:stack/mpn-ch10-demo-2/<skip>"
}

$ aws --region us-west-1 cloudformation describe-stacks --stack-name mpn-ch10-demo-2
{
 "Stacks": [
 {
 "StackStatus": "CREATE_COMPLETE",
 ...
 "StackName": "mpn-ch10-demo-2",
 "DisableRollback": false
 }
 ]
}
```

我们还可以从 AWS 控制台验证 VPC 和子网信息。我们将首先从控制台验证 VPC：

![](img/c08d36c4-58bc-4669-8468-0e1e363b3f7b.png)VPC 在 us-west-1

我们还可以查看子网：

![](img/68f0b4e2-69b7-4e6c-a7ca-c5d6003ac4d3.png)us-west-1 的子网

现在我们在美国两个海岸有两个 VPC。它们目前的行为就像两个孤立的岛屿。这可能是您期望的操作状态，也可能不是。如果您希望 VPC 能够相互连接，我们可以使用 VPC 对等连接（[`docs.aws.amazon.com/AmazonVPC/latest/PeeringGuide/vpc-peering-basics.html`](https://docs.aws.amazon.com/AmazonVPC/latest/PeeringGuide/vpc-peering-basics.html)）来允许直接通信。

VPC 对等连接不限于同一帐户。只要请求被接受并且其他方面（安全性、路由、DNS 名称）得到处理，您就可以连接不同帐户的 VPC。

在接下来的部分，我们将看一下 VPC 安全组和网络访问控制列表。

# 安全组和网络 ACL

AWS 安全组和访问控制列表可以在 VPC 的安全部分找到：

![](img/8ecdda0a-5132-499c-bd48-8b417eed3499.png)VPC 安全

安全组是一个有状态的虚拟防火墙，用于控制资源的入站和出站访问。大多数情况下，我们将使用安全组来限制对我们的 EC2 实例的公共访问。当前限制是每个 VPC 中有 500 个安全组。每个安全组最多可以包含 50 个入站和 50 个出站规则。您可以使用以下示例脚本创建一个安全组和两个简单的入站规则：

```py
$ cat Chapter10_5_security_group.py
#!/usr/bin/env python3

import boto3

ec2 = boto3.client('ec2')

response = ec2.describe_vpcs()
vpc_id = response.get('Vpcs', [{}])[0].get('VpcId', '')

# Query for security group id
response = ec2.create_security_group(GroupName='mpn_security_group',
 Description='mpn_demo_sg',
 VpcId=vpc_id)
security_group_id = response['GroupId']
data = ec2.authorize_security_group_ingress(
 GroupId=security_group_id,
 IpPermissions=[
 {'IpProtocol': 'tcp',
 'FromPort': 80,
 'ToPort': 80,
 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]},
 {'IpProtocol': 'tcp',
 'FromPort': 22,
 'ToPort': 22,
 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}
 ])
print('Ingress Successfully Set %s' % data)

# Describe security group
#response = ec2.describe_security_groups(GroupIds=[security_group_id])
print(security_group_id)
```

我们可以执行脚本并收到有关创建可与其他 AWS 资源关联的安全组的确认：

```py
$ python3 Chapter10_5_security_group.py
Ingress Successfully Set {'ResponseMetadata': {'RequestId': '<skip>', 'HTTPStatusCode': 200, 'HTTPHeaders': {'server': 'AmazonEC2', 'content-type': 'text/xml;charset=UTF-8', 'date': 'Wed, 18 Jul 2018 20:51:55 GMT', 'content-length': '259'}, 'RetryAttempts': 0}}
sg-<skip>
```

网络访问控制列表（ACL）是一个无状态的额外安全层。VPC 中的每个子网都与一个网络 ACL 相关联。由于 ACL 是无状态的，您需要指定入站和出站规则。

安全组和 ACL 之间的重要区别如下：

+   安全组在网络接口级别操作，而 ACL 在子网级别操作

+   对于安全组，我们只能指定允许规则，而 ACL 支持允许和拒绝规则

+   安全组是有状态的；返回流量会自动允许。返回流量需要在 ACL 中明确允许

让我们来看看 AWS 网络中最酷的功能之一，弹性 IP。当我最初了解弹性 IP 时，我对动态分配和重新分配 IP 地址的能力感到震惊。

# 弹性 IP

弹性 IP（EIP）是一种使用可以从互联网访问的公共 IPv4 地址的方式。它可以动态分配给 EC2 实例、网络接口或其他资源。弹性 IP 的一些特点如下：

+   弹性 IP 与账户关联，并且是特定于地区的。例如，`us-east-1`中的 EIP 只能与`us-east-1`中的资源关联。

+   您可以取消与资源的弹性 IP 关联，并将其重新关联到不同的资源。这种灵活性有时可以用于确保高可用性。例如，您可以通过将相同的 IP 地址从较小的 EC2 实例重新分配到较大的 EC2 实例来实现迁移。

+   弹性 IP 有与之相关的小额每小时费用。

您可以从门户请求弹性 IP。分配后，您可以将其与所需的资源关联：

![](img/a40704c5-33fd-426b-b1be-f69f3075f380.png)弹性 IP 不幸的是，弹性 IP 在每个地区有默认限制，[`docs.aws.amazon.com/vpc/latest/userguide/amazon-vpc-limits.html`](https://docs.aws.amazon.com/vpc/latest/userguide/amazon-vpc-limits.html)。

在接下来的部分，我们将看看如何使用 NAT 网关允许私有子网与互联网通信。

# NAT 网关

为了允许我们的 EC2 公共子网中的主机从互联网访问，我们可以分配一个弹性 IP 并将其与 EC2 主机的网络接口关联。然而，在撰写本书时，每个 EC2-VPC 最多只能有五个弹性 IP 的限制([`docs.aws.amazon.com/AmazonVPC/latest/UserGuide/VPC_Appendix_Limits.html#vpc-limits-eips`](https://docs.aws.amazon.com/AmazonVPC/latest/UserGuide/VPC_Appendix_Limits.html#vpc-limits-eips))。有时，当需要时，允许私有子网中的主机获得出站访问权限而不创建弹性 IP 和 EC2 主机之间的永久一对一映射会很好。

这就是 NAT 网关可以帮助的地方，它允许私有子网中的主机通过执行网络地址转换（NAT）临时获得出站访问权限。这个操作类似于我们通常在公司防火墙上执行的端口地址转换（PAT）。要使用 NAT 网关，我们可以执行以下步骤：

+   通过 AWS CLI、Boto3 库或 AWS 控制台在具有对互联网网关访问权限的子网中创建 NAT 网关。NAT 网关将需要分配一个弹性 IP。

+   将私有子网中的默认路由指向 NAT 网关。

+   NAT 网关将遵循默认路由到互联网网关以进行外部访问。

这个操作可以用下图来说明：

NAT 网关操作

NAT 网关通常围绕着 NAT 网关应该位于哪个子网的最常见问题之一。经验法则是要记住 NAT 网关需要公共访问。因此，它应该在具有公共互联网访问权限的子网中创建，并分配一个可用的弹性 IP：

NAT 网关创建

在接下来的部分中，我们将看一下如何将我们在 AWS 中闪亮的虚拟网络连接到我们的物理网络。

# 直接连接和 VPN

到目前为止，我们的 VPC 是驻留在 AWS 网络中的一个自包含网络。它是灵活和功能齐全的，但要访问 VPC 内部的资源，我们需要使用它们的面向互联网的服务，如 SSH 和 HTTPS。

在本节中，我们将看一下 AWS 允许我们从私人网络连接到 VPC 的两种方式：IPSec VPN 网关和直接连接。

# VPN 网关

将我们的本地网络连接到 VPC 的第一种方式是使用传统的 IPSec VPN 连接。我们需要一个可以与 AWS 的 VPN 设备建立 VPN 连接的公共可访问设备。客户网关需要支持基于路由的 IPSec VPN，其中 VPN 连接被视为可以在虚拟链路上运行路由协议的连接。目前，AWS 建议使用 BGP 交换路由。

在 VPC 端，我们可以遵循类似的路由表，可以将特定子网路由到**虚拟私有网关**目标：

VPC VPN 连接（来源：[`docs.aws.amazon.com/AmazonVPC/latest/UserGuide/VPC_VPN.html`](https://docs.aws.amazon.com/AmazonVPC/latest/UserGuide/VPC_VPN.html)）

除了 IPSec VPN，我们还可以使用专用电路进行连接。

# 直接连接

我们看到的 IPSec VPN 连接是提供本地设备与 AWS 云资源连接的简单方法。然而，它遭受了 IPSec 在互联网上总是遭受的相同故障：它是不可靠的，我们对它几乎没有控制。性能监控很少，直到连接到我们可以控制的互联网部分才有**服务级别协议**（SLA）。

出于所有这些原因，任何生产级别的、使命关键的流量更有可能通过亚马逊提供的第二个选项，即 AWS 直接连接。AWS 直接连接允许客户使用专用虚拟电路将他们的数据中心和机房连接到他们的 AWS VPC。这个操作通常比较困难的部分通常是将我们的网络带到可以与 AWS 物理连接的地方，通常是在一个承载商酒店。您可以在这里找到 AWS 直接连接位置的列表：[`aws.amazon.com/directconnect/details/`](https://aws.amazon.com/directconnect/details/)。直接连接链接只是一个光纤补丁连接，您可以从特定的承载商酒店订购，将网络连接到网络端口并配置 dot1q 干线的连接。

还有越来越多的通过第三方承运商使用 MPLS 电路和聚合链路进行直接连接的连接选项。我发现并使用的最实惠的选择之一是 Equinix Cloud Exchange ([`www.equinix.com/services/interconnection-connectivity/cloud-exchange/`](https://www.equinix.com/services/interconnection-connectivity/cloud-exchange/))。通过使用 Equinix Cloud Exchange，我们可以利用相同的电路并以较低成本连接到不同的云提供商：

![](img/966cabb6-fb23-4921-96f6-290e979d6c9f.png)Equinix Cloud Exchange（来源：[`www.equinix.com/services/interconnection-connectivity/cloud-exchange/`](https://www.equinix.com/services/interconnection-connectivity/cloud-exchange/)）

在接下来的部分，我们将看一下 AWS 提供的一些网络扩展服务。

# 网络扩展服务

在本节中，我们将看一下 AWS 提供的一些网络服务。许多服务没有直接的网络影响，比如 DNS 和内容分发网络。由于它们与网络和应用性能的密切关系，它们与我们的讨论相关。

# 弹性负载均衡

**弹性负载均衡**（**ELB**）允许来自互联网的流量自动分布到多个 EC2 实例。就像物理世界中的负载均衡器一样，这使我们能够在减少每台服务器负载的同时获得更好的冗余和容错。ELB 有两种类型：应用和网络负载均衡。

应用负载均衡器通过 HTTP 和 HTTPS 处理 Web 流量；网络负载均衡器在 TCP 层运行。如果您的应用程序在 HTTP 或 HTTPS 上运行，通常最好选择应用负载均衡器。否则，使用网络负载均衡器是一个不错的选择。

可以在[`aws.amazon.com/elasticloadbalancing/details/`](https://aws.amazon.com/elasticloadbalancing/details/)找到应用和网络负载均衡器的详细比较：

![](img/af58c911-8764-46d9-a3e3-71e90df6e39b.png)弹性负载均衡器比较（来源：[`aws.amazon.com/elasticloadbalancing/details/`](https://aws.amazon.com/elasticloadbalancing/details/)）

弹性负载均衡器提供了一种在资源进入我们地区后平衡流量的方式。AWS Route53 DNS 服务允许在地区之间进行地理负载平衡。

# Route53 DNS 服务

我们都知道域名服务是什么；Route53 是 AWS 的 DNS 服务。Route53 是一个全功能的域名注册商，您可以直接从 AWS 购买和管理域名。关于网络服务，DNS 允许通过在地理区域之间以轮询方式服务域名来实现负载平衡。

在我们可以使用 DNS 进行负载平衡之前，我们需要以下项目：

+   每个预期的负载平衡地区中都有一个弹性负载均衡器。

+   注册的域名。我们不需要 Route53 作为域名注册商。

+   Route53 是该域的 DNS 服务。

然后我们可以在两个弹性负载均衡器之间的主动-主动环境中使用 Route 53 基于延迟的路由策略和健康检查。

# CloudFront CDN 服务

CloudFront 是亚马逊的**内容分发网络**（**CDN**），通过在物理上为客户提供更接近的内容，减少了内容交付的延迟。内容可以是静态网页内容、视频、应用程序、API，或者最近的 Lambda 函数。CloudFront 边缘位置包括现有的 AWS 区域，还有全球许多其他位置。CloudFront 的高级操作如下：

+   用户访问您的网站以获取一个或多个对象

+   DNS 将请求路由到距用户请求最近的 Amazon CloudFront 边缘位置

+   CloudFront 边缘位置将通过缓存提供内容或从源请求对象

AWS CloudFront 和 CDN 服务通常由应用程序开发人员或 DevOps 工程师处理。但是，了解它们的运作方式总是很好的。

# 其他 AWS 网络服务

还有许多其他 AWS 网络服务，我们没有空间来介绍。一些更重要的服务列在本节中：

+   **AWS Transit VPC** ([`aws.amazon.com/blogs/aws/aws-solution-transit-vpc/`](https://aws.amazon.com/blogs/aws/aws-solution-transit-vpc/))：这是一种连接多个虚拟私有云到一个作为中转中心的公共 VPC 的方式。这是一个相对较新的服务，但它可以最小化您需要设置和管理的连接。这也可以作为一个工具，当您需要在不同的 AWS 账户之间共享资源时。

+   **Amazon GuardDuty** ([`aws.amazon.com/guardduty/`](https://aws.amazon.com/guardduty/))：这是一个托管的威胁检测服务，持续监视恶意或未经授权的行为，以帮助保护我们的 AWS 工作负载。它监视 API 调用或潜在的未经授权的部署。

+   **AWS WAF**([`aws.amazon.com/waf/`](https://aws.amazon.com/waf/))：这是一个 Web 应用程序防火墙，可以帮助保护 Web 应用程序免受常见的攻击。我们可以定义定制的 Web 安全规则来允许或阻止 Web 流量。

+   **AWS Shield** ([`aws.amazon.com/shield/`](https://aws.amazon.com/shield/))：这是一个托管的**分布式拒绝服务**（**DDoS**）保护服务，可保护在 AWS 上运行的应用程序。基本级别的保护服务对所有客户免费；AWS Shield 的高级版本是一项收费服务。

# 总结

在本章中，我们深入了解了 AWS 云网络服务。我们讨论了 AWS 网络中区域、可用区、边缘位置和中转中心的定义。通过了解整体的 AWS 网络，这让我们对其他 AWS 网络服务的一些限制和内容有了一个很好的了解。在本章的整个过程中，我们使用了 AWS CLI、Python Boto3 库以及 CloudFormation 来自动化一些任务。

我们深入讨论了 AWS 虚拟私有云，包括路由表和路由目标的配置。关于安全组和网络 ACL 控制我们 VPC 的安全性的示例。我们还讨论了弹性 IP 和 NAT 网关，以允许外部访问。

连接 AWS VPC 到本地网络有两种方式：直接连接和 IPSec VPN。我们简要地介绍了每种方式以及使用它们的优势。在本章的最后，我们了解了 AWS 提供的网络扩展服务，包括弹性负载均衡、Route53 DNS 和 CloudFront。

在第十一章中，*使用 Git*，我们将更深入地了解我们一直在使用的版本控制系统：Git。
