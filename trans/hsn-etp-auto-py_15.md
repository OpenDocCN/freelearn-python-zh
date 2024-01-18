# 与OpenStack API交互

长期以来，IT基础设施依赖于商业软件（来自VMWare、Microsoft和Citrix等供应商）提供运行工作负载和管理资源（如计算、存储和网络）的虚拟环境。然而，IT行业正在迈向云时代，工程师正在将工作负载和应用程序迁移到云（无论是公共还是私有），这需要一个能够管理所有应用程序资源的新框架，并提供一个开放和强大的API接口，以与其他应用程序的外部调用进行交互。

OpenStack提供了开放访问和集成，以管理所有计算、存储和网络资源，避免在构建云时出现供应商锁定。它可以控制大量的计算节点、存储阵列和网络设备，无论每个资源的供应商如何，并在所有资源之间提供无缝集成。OpenStack的核心思想是将应用于底层基础设施的所有配置抽象为一个负责管理资源的*项目*。因此，您将找到一个管理计算资源的项目（称为Nova），另一个提供实例网络的项目（neutron），以及与不同存储类型交互的项目（Swift和Cinder）。

您可以在此链接中找到当前OpenStack项目的完整列表

[https://www.OpenStack.org/software/project-navigator/](https://www.openstack.org/software/project-navigator/)

此外，OpenStack为应用程序开发人员和系统管理员提供统一的API访问，以编排资源创建。

在本章中，我们将探索OpenStack的新开放世界，并学习如何利用Python和Ansible与其交互。

本章将涵盖以下主题：

+   了解RESTful web服务

+   设置环境

+   向OpenStack发送请求

+   从Python创建工作负载

+   使用Ansible管理OpenStack实例

# 了解RESTful web服务

**表述状态转移**（**REST**）依赖于HTTP协议在客户端和服务器之间传输消息。HTTP最初设计用于在请求时从Web服务器（服务器）向浏览器（客户端）传递HTML页面。页面代表用户想要访问的一组资源，并由**统一资源标识符**（**URI**）请求。

HTTP请求通常包含一个方法，该方法指示需要在资源上执行的操作类型。例如，当从浏览器访问网站时，您可以看到（在下面的屏幕截图中）方法是`GET`：

![](../images/00195.jpeg)

以下是最常见的HTTP方法及其用法：

| HTTP方法 | 操作 |
| `GET` | 客户端将要求服务器检索资源。 |
| `POST` | 客户端将指示服务器创建新资源。 |
| `PUT` | 客户端将要求服务器修改/更新资源。 |
| `DELETE` | 客户端将要求服务器删除资源。 |

应用程序开发人员可以公开其应用程序的某些资源，以供外部世界的客户端使用。携带请求从客户端到服务器并返回响应的传输协议是HTTP。它负责保护通信并使用服务器接受的适当数据编码机制对数据包进行编码，并且在两者之间进行无状态通信。

另一方面，数据包有效载荷通常以XML或JSON编码，以表示服务器处理的请求结构以及客户端偏好的响应方式。

世界各地有许多公司为开发人员提供其数据的公共访问权限，实时提供。例如，Twitter API（[https://developer.twitter.com/](https://developer.twitter.com/)）提供实时数据获取，允许其他开发人员在第三方应用程序中使用数据，如广告、搜索和营销。谷歌（[https://developers.google.com/apis-explorer/#p/discovery/v1/](https://developers.google.com/apis-explorer/#p/discovery/v1/)）、LinkedIn（[https://developer.linkedin.com/](https://developer.linkedin.com/)）和Facebook（[https://developers.facebook.com/](https://developers.facebook.com/)）等大公司也是如此。

对API的公共访问通常限制为特定数量的请求，无论是每小时还是每天，对于单个应用程序，以免过度使用公共资源。

Python提供了大量的工具和库来消耗API、编码消息和解析响应。例如，Python有一个`requests`包，可以格式化并发送HTTP请求到外部资源。它还有工具来解析JSON格式的响应并将其转换为Python中的标准字典。

Python还有许多框架可以将您的资源暴露给外部世界。`Django`和`Flask`是最好的之一，可以作为全栈框架。

# 设置环境

OpenStack是一个免费的开源项目，用于**基础设施即服务**（**IaaS**），可以控制CPU、内存和存储等硬件资源，并为许多供应商构建和集成插件提供一个开放的框架。

为了设置我们的实验室，我将使用最新的`OpenStack-rdo`版本（在撰写时），即Queens，并将其安装到CentOS 7.4.1708上。安装步骤非常简单，可以在[https://www.rdoproject.org/install/packstack/](https://www.rdoproject.org/install/packstack/)找到。

我们的环境包括一台具有100GB存储、12个vCPU和32GB RAM的机器。该服务器将包含OpenStack控制器、计算和neutron角色在同一台服务器上。OpenStack服务器连接到具有我们自动化服务器的相同交换机和相同子网。请注意，这在生产环境中并不总是这样，但您需要确保运行Python代码的服务器可以访问OpenStack。

实验室拓扑如下：

![](../images/00196.jpeg)

# 安装rdo-OpenStack软件包

在RHEL 7.4和CentOS上安装rdo-OpenStack的步骤如下：

# 在RHEL 7.4上

首先确保您的系统是最新的，然后从网站安装`rdo-release.rpm`以获取最新版本。最后，安装`OpenStack-packstack`软件包，该软件包将自动化OpenStack安装，如下段所示：

```py
$ sudo yum install -y https://www.rdoproject.org/repos/rdo-release.rpm
$ sudo yum update -y
$ sudo yum install -y OpenStack-packstack
```

# 在CentOS 7.4上

首先确保您的系统是最新的，然后安装rdoproject以获取最新版本。最后，安装`centos-release-OpenStack-queens`软件包，该软件包将自动化OpenStack安装，如下段所示：

```py
$ sudo yum install -y centos-release-OpenStack-queens
$ sudo yum update -y
$ sudo yum install -y OpenStack-packstack
```

# 生成答案文件

现在，您需要生成包含部署参数的答案文件。这些参数中的大多数都是默认值，但我们将更改一些内容：

```py
# packstack --gen-answer-file=/root/EnterpriseAutomation
```

# 编辑答案文件

使用您喜欢的编辑器编辑`EnterpriseAutomtion`文件，并更改以下内容：

```py
CONFIG_DEFAULT_PASSWORD=access123 CONFIG_CEILOMETER_INSTALL=n CONFIG_AODH_INSTALL=n CONFIG_KEYSTONE_ADMIN_PW=access123 CONFIG_PROVISION_DEMO=n 
```

`CELIOMETER`和`AODH`是OpenStack生态系统中的可选项目，可以在实验室环境中忽略。

我们还设置了一个用于生成临时令牌以访问API资源并访问OpenStack GUI的`KEYSTONE`密码

# 运行packstack

保存文件并通过`packstack`运行安装：

```py
# packstack answer-file=EnterpriseAutomation
```

此命令将从Queens存储库下载软件包并安装OpenStack服务，然后启动它们。安装成功完成后，将在控制台上打印以下消息：

```py
 **** Installation completed successfully ******

Additional information:
 * Time synchronization installation was skipped. Please note that unsynchronized time on server instances might be problem for some OpenStack components.
 * File /root/keystonerc_admin has been created on OpenStack client host 10.10.10.150\. To use the command line tools you need to source the file.
 * To access the OpenStack Dashboard browse to http://10.10.10.150/dashboard .
Please, find your login credentials stored in the keystonerc_admin in your home directory.
 * The installation log file is available at: /var/tmp/packstack/20180410-155124-CMpsKR/OpenStack-setup.log
 * The generated manifests are available at: /var/tmp/packstack/20180410-155124-CMpsKR/manifests
```

# 访问OpenStack GUI

现在您可以使用`http://<server_ip_address>/dashboard`访问OpenStack GUI。凭证将是admin和access123（取决于您在之前步骤中在`CONFIG_KEYSTONE_ADMIN_PW`中写入了什么）：

![](../images/00197.gif)

我们的云现在已经启动运行，准备接收请求。

# 向OpenStack keystone发送请求

OpenStack包含一系列服务，这些服务共同工作以管理虚拟机的创建、读取、更新和删除（CRUD）操作。每个服务都可以将其资源暴露给外部请求进行消费。例如，`nova`服务负责生成虚拟机并充当一个hypervisor层（虽然它本身不是一个hypervisor，但可以控制其他hypervisors，如KVM和vSphere）。另一个服务是`glance`，负责以ISO或qcow2格式托管实例镜像。`neutron`服务负责为生成的实例提供网络服务，并确保位于不同租户（项目）上的实例相互隔离，而位于相同租户上的实例可以通过覆盖网络（VxLAN或GRE）相互访问。

为了访问上述每个服务的API，您需要具有用于特定时间段的经过身份验证的令牌。这就是`keystone`的作用，它提供身份服务并管理每个用户的角色和权限。

首先，我们需要在自动化服务器上安装Python绑定。这些绑定包含用于访问每个服务并使用从KEYSTONE生成的令牌进行身份验证的Python代码。此外，绑定包含每个项目的支持操作（如创建/删除/更新/列出）：

```py
yum install -y gcc openssl-devel python-pip python-wheel
pip install python-novaclient
pip install python-neutronclient
pip install python-keystoneclient
pip install python-glanceclient
pip install python-cinderclient
pip install python-heatclient
pip install python-OpenStackclient
```

请注意，Python客户端名称为`python-<service_name>client`

您可以将其下载到站点的全局包或Python `virtualenv`环境中。然后，您将需要OpenStack管理员权限，这些权限可以在OpenStack服务器内的以下路径中找到：

```py
cat /root/keystonerc_admin
unset OS_SERVICE_TOKEN
export OS_USERNAME=admin
export OS_PASSWORD='access123'
export OS_AUTH_URL=http://10.10.10.150:5000/v3
export PS1='[\u@\h \W(keystone_admin)]\$ '

export OS_PROJECT_NAME=admin
export OS_USER_DOMAIN_NAME=Default
export OS_PROJECT_DOMAIN_NAME=Default
export OS_IDENTITY_API_VERSION=3
```

请注意，当我们与OpenStack keystone服务通信时，我们将在`OS_AUTH_URL`和`OS_IDENTITY_API_VERSION`参数中使用keystone版本3。大多数Python客户端与旧版本兼容，但需要您稍微更改脚本。在令牌生成期间还需要其他参数，因此请确保您可以访问`keystonerc_admin`文件。还可以在同一文件中的`OS_USERNAME`和`OS_PASSWORD`中找到访问凭证。

我们的Python脚本将如下所示：

```py
from keystoneauth1.identity import v3
from keystoneauth1 import session

auth = v3.Password(auth_url="http://10.10.10.150:5000/v3",
  username="admin",
  password="access123",
  project_name="admin",
  user_domain_name="Default",
  project_domain_name="Default")
sess = session.Session(auth=auth, verify=False)
print(sess) 
```

在上述示例中，以下内容适用：

+   `python-keystoneclient`使用`v3`类（反映了keystone API版本）向keystone API发出请求。此类可在`keystoneayth1.identity`内使用。

+   然后，我们将从`keystonerc_admin`文件中获取的完整凭证提供给`auth`变量。

+   最后，我们建立了会话，使用keystone客户端内的会话管理器。请注意，我们将`verify`设置为`False`，因为我们不使用证书来生成令牌。否则，您可以提供证书路径。

+   生成的令牌可以用于任何服务，并将持续一个小时，然后过期。此外，如果更改用户角色，令牌将立即过期，而不必等待一个小时。

OpenStack管理员可以在`/etc/keystone/keystone.conf`文件中配置`admin_token`字段，该字段永不过期。但出于安全原因，这在生产环境中不被推荐。

如果您不想将凭证存储在Python脚本中，可以将它们存储在`ini`文件中，并使用`configparser`模块加载它们。首先，在自动化服务器上创建一个`creds.ini`文件，并赋予适当的Linux权限，以便只能使用您自己的帐户打开它。

```py
#vim /root/creds.ini [os_creds]  auth_url="http://10.10.10.150:5000/v3" username="admin" password="access123" project_name="admin" user_domain_name="Default" project_domain_name="Default"
```

修改后的脚本如下：

```py
from keystoneauth1.identity import v3
from keystoneauth1 import session
import ConfigParser
config = ConfigParser.ConfigParser() config.read("/root/creds.ini") auth = v3.Password(auth_url=config.get("os_creds","auth_url"),
  username=config.get("os_creds","username"),
  password=config.get("os_creds","password"),
  project_name=config.get("os_creds","project_name"),
  user_domain_name=config.get("os_creds","user_domain_name"),
  project_domain_name=config.get("os_creds","project_domain_name")) sess = session.Session(auth=auth, verify=False) print(sess)   
```

`configparser`模块将解析`creds.ini`文件并查看文件内部的`os_creds`部分。然后，它将使用`get()`方法获取每个参数前面的值。

`config.get()`方法将接受两个参数。第一个参数是`.ini`文件内的部分名称，第二个是参数名称。该方法将返回与参数关联的值。

此方法应该为您的云凭据提供额外的安全性。保护文件的另一种有效方法是使用Linux的`source`命令将`keystonerc_admin`文件加载到环境变量中，并使用`os`模块内的`environ()`方法读取凭据。

# 从Python创建实例

要使实例运行起来，OpenStack实例需要三个组件。由`glance`提供的引导镜像，由`neutron`提供的网络端口，最后是由`nova`项目提供的定义分配给实例的CPU数量、RAM数量和磁盘大小的计算flavor。

# 创建图像

我们将首先下载一个`cirros`图像到自动化服务器。`cirros`是一个轻量级的基于Linux的图像，被许多OpenStack开发人员和测试人员用来验证OpenStack服务的功能：

```py
#cd /root/ ; wget http://download.cirros-cloud.net/0.4.0/cirros-0.4.0-x86_64-disk.img
```

然后，我们将使用`glanceclient`将图像上传到OpenStack图像存储库。请注意，我们需要首先具有keystone令牌和会话参数，以便与`glance`通信，否则，`glance`将不接受我们的任何API请求。

脚本将如下所示：

```py
from keystoneauth1.identity import v3
from keystoneauth1 import session
from glanceclient import client as gclient
from pprint import pprint

auth = v3.Password(auth_url="http://10.10.10.150:5000/v3",
  username="admin",
  password="access123",
  project_name="admin",
  user_domain_name="Default",
  project_domain_name="Default")     sess = session.Session(auth=auth, verify=False)    #Upload the image to the Glance  glance = gclient.Client('2', session=sess)   image = glance.images.create(name="CirrosImage",
  container_format='bare',
  disk_format='qcow2',
  )   glance.images.upload(image.id, open('/root/cirros-0.4.0-x86_64-disk.img', 'rb'))   
```

在上面的示例中，适用以下内容：

+   由于我们正在与`glance`（图像托管项目）通信，因此我们将从安装的`glanceclient`模块导入`client`。

+   使用相同的keystone脚本生成包含keystone令牌的`sess`。

+   我们创建了glance参数，该参数使用`glance`初始化客户端管理器，并提供版本（`版本2`）和生成的令牌。

+   您可以通过访问OpenStack GUI | API Access选项卡来查看所有支持的API版本，如下面的屏幕截图所示。还要注意每个项目的支持版本。

![](../images/00198.jpeg)

+   glance客户端管理器旨在在glance OpenStack服务上运行。指示管理器使用名称`CirrosImage`创建一个磁盘类型为`qcow2`格式的图像。

+   最后，我们将以二进制形式打开下载的图像，使用'rb'标志，并将其上传到创建的图像中。现在，`glance`将图像导入到图像存储库中新创建的文件中。

您可以通过两种方式验证操作是否成功：

1.  执行`glance.images.upload()`后如果没有打印出错误，这意味着请求格式正确，并已被OpenStack `glance` API接受。

1.  运行`glance.images.list()`。返回的输出将是一个生成器，您可以遍历它以查看有关上传图像的更多详细信息：

```py
print("==========================Image Details==========================") for image in glance.images.list(name="CirrosImage"):
  pprint(image) 
{u'checksum': u'443b7623e27ecf03dc9e01ee93f67afe',
 u'container_format': u'bare',
 u'created_at': u'2018-04-11T03:11:58Z',
 u'disk_format': u'qcow2',
 u'file': u'/v2/images/3c2614b0-e53c-4be1-b99d-bbd9ce14b287/file',
 u'id': u'3c2614b0-e53c-4be1-b99d-bbd9ce14b287',
 u'min_disk': 0,
 u'min_ram': 0,
 u'name': u'CirrosImage',
 u'owner': u'8922dc52984041af8fe22061aaedcd13',
 u'protected': False,
 u'schema': u'/v2/schemas/image',
 u'size': 12716032,
 u'status': u'active',
 u'tags': [],
 u'updated_at': u'2018-04-11T03:11:58Z',
 u'virtual_size': None,
 u'visibility': u'shared'}
```

# 分配flavor

Flavors用于确定实例的CPU、内存和存储大小。OpenStack带有一组预定义的flavors，具有从微小到超大的不同大小。对于`cirros`图像，我们将使用小型flavor，它具有2GB RAM，1个vCPU和20GB存储。访问flavors没有独立的API客户端；而是作为`nova`客户端的一部分。

您可以在OpenStack GUI | Admin | Flavors中查看所有可用的内置flavors：

![](../images/00199.gif)

脚本将如下所示：

```py
from keystoneauth1.identity import v3
from keystoneauth1 import session
from novaclient import client as nclient
from pprint import pprint

auth = v3.Password(auth_url="http://10.10.10.150:5000/v3",
  username="admin",
  password="access123",
  project_name="admin",
  user_domain_name="Default",
  project_domain_name="Default")   sess = session.Session(auth=auth, verify=False)   nova = nclient.Client(2.1, session=sess) instance_flavor = nova.flavors.find(name="m1.small") print("==========================Flavor Details==========================") pprint(instance_flavor)
```

在上述脚本中，适用以下内容：

+   由于我们将与`nova`（计算服务）通信以检索flavor，因此我们将导入`novaclient`模块作为`nclient`。

+   使用相同的keystone脚本生成包含keystone令牌的`sess`。

+   我们创建了`nova`参数，用它来初始化具有`nova`的客户端管理器，并为客户端提供版本（版本2.1）和生成的令牌。

+   最后，我们使用`nova.flavors.find()`方法来定位所需的规格，即`m1.small`。名称必须与OpenStack中的名称完全匹配，否则将抛出错误。

# 创建网络和子网

为实例创建网络需要两件事：网络本身和将子网与之关联。首先，我们需要提供网络属性，例如ML2驱动程序（Flat、VLAN、VxLAN等），区分在同一接口上运行的网络之间的分段ID，MTU和物理接口，如果实例流量需要穿越外部网络。其次，我们需要提供子网属性，例如网络CIDR、网关IP、IPAM参数（如果定义了DHCP/DNS服务器）以及与子网关联的网络ID，如下面的屏幕截图所示：

![](../images/00200.jpeg)

现在我们将开发一个Python脚本来与neutron项目进行交互，并创建一个带有子网的网络

```py
from keystoneauth1.identity import v3
from keystoneauth1 import session
import neutronclient.neutron.client as neuclient

auth = v3.Password(auth_url="http://10.10.10.150:5000/v3",
  username="admin",
  password="access123",
  project_name="admin",
  user_domain_name="Default",
  project_domain_name="Default")   sess = session.Session(auth=auth, verify=False)   neutron = neuclient.Client(2, session=sess)   # Create Network   body_network = {'name': 'python_network',
  'admin_state_up': True,
 #'port_security_enabled': False,
  'shared': True,
  # 'provider:network_type': 'vlan|vxlan',
 # 'provider:segmentation_id': 29 # 'provider:physical_network': None, # 'mtu': 1450,  } neutron.create_network({'network':body_network}) network_id = neutron.list_networks(name="python_network")["networks"][0]["id"]     # Create Subnet   body_subnet = {
  "subnets":[
  {
  "name":"python_network_subnet",
  "network_id":network_id,
  "enable_dhcp":True,
  "cidr": "172.16.128.0/24",
  "gateway_ip": "172.16.128.1",
  "allocation_pools":[
  {
  "start": "172.16.128.10",
  "end": "172.16.128.100"
  }
  ],
  "ip_version": 4,
  }
  ]
  } neutron.create_subnet(body=body_subnet) 
```

在上述脚本中，以下内容适用：

+   由于我们将与`neutron`（网络服务）通信来创建网络和关联子网，我们将导入`neutronclient`模块作为`neuclient`。

+   相同的keystone脚本用于生成`sess`，该`sess`保存后来用于访问neutron资源的keystone令牌。

+   我们将创建`neutron`参数，用它来初始化具有neutron的客户端管理器，并为其提供版本（版本2）和生成的令牌。

+   然后，我们创建了两个Python字典，`body_network`和`body_subnet`，它们分别保存了网络和子网的消息主体。请注意，字典键是静态的，不能更改，而值可以更改，并且通常来自外部门户系统或Excel表格，具体取决于您的部署。此外，我对在网络创建过程中不必要的部分进行了评论，例如`provider:physical_network`和`provider:network_type`，因为我们的`cirros`镜像不会与提供者网络（在OpenStack域之外定义的网络）通信，但这里提供了参考。

+   最后，通过`list_networks()`方法获取`network_id`，并将其作为值提供给`body_subnet`变量中的`network_id`键，将子网和网络关联在一起。

# 启动实例

最后一部分是将所有内容粘合在一起。我们有引导镜像、实例规格和连接机器与其他实例的网络。我们准备使用`nova`客户端启动实例（记住`nova`负责虚拟机的生命周期和VM上的CRUD操作）：

```py

print("=================Launch The Instance=================")   image_name = glance.images.get(image.id)   network1 = neutron.list_networks(name="python_network") instance_nics = [{'net-id': network1["networks"][0]["id"]}]   server = nova.servers.create(name = "python-instance",
  image = image_name.id,
  flavor = instance_flavor.id,
  nics = instance_nics,) status = server.status
while status == 'BUILD':
  print("Sleeping 5 seconds till the server status is changed")
  time.sleep(5)
  instance = nova.servers.get(server.id)
  status = instance.status
    print(status) print("Current Status is: {0}".format(status))
```

在上述脚本中，我们使用了`nova.servers.create()`方法，并传递了生成实例所需的所有信息（实例名称、操作系统、规格和网络）。此外，我们实现了一个轮询机制，用于轮询nova服务的服务器当前状态。如果服务器仍处于`BUILD`阶段，则脚本将休眠五秒，然后再次轮询。当服务器状态更改为`ACTIVE`或`FAILURE`时，循环将退出，并在最后打印服务器状态。

脚本的输出如下：

```py
Sleeping 5 seconds till the server status is changed
Sleeping 5 seconds till the server status is changed
Sleeping 5 seconds till the server status is changed
Current Status is: ACTIVE
```

此外，您可以从OpenStack GUI | 计算 | 实例中检查实例：

![](../images/00201.gif)

# 从Ansible管理OpenStack实例

Ansible提供了可以管理OpenStack实例生命周期的模块，就像我们使用API一样。您可以在[http://docs.ansible.com/ansible/latest/modules/list_of_cloud_modules.html#OpenStack](http://docs.ansible.com/ansible/latest/modules/list_of_cloud_modules.html#openstack)找到支持的模块的完整列表。

所有OpenStack模块都依赖于名为`shade`的Python库（[https://pypi.python.org/pypi/shade](https://pypi.python.org/pypi/shade)），该库提供了对OpenStack客户端的包装。

一旦您在自动化服务器上安装了`shade`，您将可以访问`os-*`模块，这些模块可以操作OpenStack配置，比如`os_image`（处理OpenStack镜像），`os_network`（创建网络），`os_subnet`（创建并关联子网到创建的网络），`os_nova_flavor`（根据RAM、CPU和磁盘创建flavors），最后是`os_server`模块（启动OpenStack实例）。

# 安装Shade和Ansible

在自动化服务器上，使用Python的`pip`来下载和安装`shade`，以及所有依赖项：

```py
pip install shade
```

安装完成后，您将在Python的正常`site-packages`下拥有`shade`，但我们将使用Ansible。

此外，如果您之前没有在自动化服务器上安装Ansible，您将需要安装Ansible：

```py
# yum install ansible -y
```

通过从命令行查询Ansible版本来验证Ansible是否已成功安装：

```py
[root@AutomationServer ~]# ansible --version
ansible 2.5.0
 config file = /etc/ansible/ansible.cfg
 configured module search path = [u'/root/.ansible/plugins/modules', u'/usr/share/ansible/plugins/modules']
 ansible python module location = /usr/lib/python2.7/site-packages/ansible
 executable location = /usr/bin/ansible
 python version = 2.7.5 (default, Aug  4 2017, 00:39:18) [GCC 4.8.5 20150623 (Red Hat 4.8.5-16)]
```

# 构建Ansible playbook

正如我们在[第13章](part0168.html#506UG0-9cfcdc5beecd470bbeda046372f0337f)中所看到的，*用于管理的Ansible*，依赖于一个YAML文件，其中包含了您需要针对清单中的主机执行的一切。在这种情况下，我们将指示playbook在自动化服务器上建立与`shade`库的本地连接，并提供`keystonerc_admin`凭据，以帮助`shade`向我们的OpenStack服务器发送请求。

playbook脚本如下：

```py
--- - hosts: localhost
  vars:
 os_server: '10.10.10.150'
  gather_facts: yes
  connection: local
  environment:
 OS_USERNAME: admin
  OS_PASSWORD: access123
  OS_AUTH_URL: http://{{ os_server }}:5000/v3
  OS_TENANT_NAME: admin
  OS_REGION_NAME: RegionOne
  OS_USER_DOMAIN_NAME: Default
  OS_PROJECT_DOMAIN_NAME: Default    tasks:
  - name: "Upload the Cirros Image"
  os_image:
 name: Cirros_Image
  container_format: bare
  disk_format: qcow2
  state: present
  filename: /root/cirros-0.4.0-x86_64-disk.img
  ignore_errors: yes    - name: "CREATE CIRROS_FLAVOR"
  os_nova_flavor:
 state: present
  name: CIRROS_FLAVOR
  ram: 2048
  vcpus: 4
  disk: 35
  ignore_errors: yes    - name: "Create the Cirros Network"
  os_network:
 state: present
  name: Cirros_network
  external: True
  shared: True
  register: Cirros_network
  ignore_errors: yes      - name: "Create Subnet for The network Cirros_network"
  os_subnet:
 state: present
  network_name: "{{ Cirros_network.id }}"
  name: Cirros_network_subnet
  ip_version: 4
  cidr: 10.10.128.0/18
  gateway_ip: 10.10.128.1
  enable_dhcp: yes
  dns_nameservers:
  - 8.8.8.8
  register: Cirros_network_subnet
  ignore_errors: yes      - name: "Create Cirros Machine on Compute"
  os_server:
 state: present
  name: ansible_instance
  image: Cirros_Image
  flavor: CIRROS_FLAVOR
  security_groups: default
  nics:
  - net-name: Cirros_network
  ignore_errors: yes 
```

在playbook中，我们使用`os_*`模块将镜像上传到OpenStack的`glance`服务器，创建一个新的flavor（而不是使用内置的flavor），并创建与子网关联的网络；然后，我们在`os_server`中将所有内容粘合在一起，该模块与`nova`服务器通信以生成机器。

请注意，主机将是本地主机（或托管`shade`库的机器名称），同时我们在环境变量中添加了OpenStack keystone凭据。

# 运行playbook

将playbook上传到自动化服务器并执行以下命令来运行它：

```py
ansible-playbook os_playbook.yml
```

playbook的输出将如下所示：

```py
 [WARNING]: No inventory was parsed, only implicit localhost is available

 [WARNING]: provided hosts list is empty, only localhost is available. Note that the implicit localhost does not match 'all'

PLAY [localhost] ****************************************************************************

TASK [Gathering Facts] **********************************************************************
ok: [localhost]

TASK [Upload the Cirros Image] **************************************************************
changed: [localhost]

TASK [CREATE CIRROS_FLAVOR] *****************************************************************
ok: [localhost]

TASK [Create the Cirros Network] ************************************************************
changed: [localhost]

TASK [Create Subnet for The network Cirros_network] *****************************************
changed: [localhost]

TASK [Create Cirros Machine on Compute] *****************************************************
changed: [localhost]

PLAY RECAP **********************************************************************************
localhost                  : ok=6    changed=4    unreachable=0    failed=0   
```

您可以访问OpenStack GUI来验证实例是否是从Ansible playbook创建的：

![](../images/00202.gif)

# 摘要

如今，IT行业正在尽可能地避免供应商锁定，转向开源世界。OpenStack为我们提供了窥视这个世界的窗口；许多大型组织和电信运营商正在考虑将其工作负载迁移到OpenStack，以在其数据中心构建私有云。然后，他们可以构建自己的工具来与OpenStack提供的开源API进行交互。

在下一章中，我们将探索另一个（付费的）公共亚马逊云，并学习如何利用Python来自动化实例创建。
