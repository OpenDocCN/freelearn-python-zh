# 第九章。连接到云

由于需要许多功能来呈现一个针对云提供商的统一工具，云模块可能看起来是最令人生畏的 Salt 模块类型。幸运的是，一旦你知道如何操作，连接到大多数云提供商都很简单。在本章中，我们将讨论：

+   理解云组件如何协同工作

+   学习所需的函数以及它们的使用方法

+   比较基于 Libcloud 的模块与直接 REST 模块

+   编写通用的云模块

+   云模块的故障排除

# 理解云组件

近年来，“云”这个词遭受了过度使用和误用的不幸，所以在我们谈论组件看起来像什么之前，我们首先需要定义我们真正在谈论的是什么。

Salt Cloud 旨在与*计算云*提供商一起运行。这意味着它们提供计算资源，通常以虚拟机的形式。许多云提供商还提供其他资源，如存储空间、DNS 和负载均衡。虽然 Salt Cloud 并非明确设计来管理这些资源，但可以添加对这些资源的支持。

对于我们的目的，我们将讨论创建云驱动程序，重点是管理虚拟机。其中一些技术可以用于添加其他资源，所以如果你打算朝那个方向发展，本章对你仍然有用。

## 观察拼图碎片

Salt Cloud 的主要目标是轻松在云提供商上创建虚拟机，在该机器上安装 Salt Minion，然后自动在 Master 上接受该 Minion 的密钥。当你深入挖掘时，你会发现许多部件协同工作以实现这一目标。

### 连接机制

大多数云提供商都提供 API 来管理账户中的资源。此 API 包括一个身份验证方案，以及一组用于类似目的的 URL。几乎每个云提供商都支持基于`GET`和`POST`方法的 URL，但一些支持其他方法，如`PATCH`和`DELETE`。

很频繁地，这些 URL 将包括多达四个组件：

+   资源名称

+   在该资源上要执行的操作

+   要管理的资源的 ID

+   定义如何管理资源的参数

这些组件可以与身份验证方案结合使用，创建一个用于执行所有可用管理功能的单一工具。

### 列出资源

大多数资源都有一种从 API 中列出它们的方式。这包括由云提供商定义的选项以及属于你的账户且可以由你管理的资源。通常可以从 API 中列出的资源包括：

+   操作系统镜像

+   可以创建的虚拟机大小

+   用户账户中的现有虚拟机

+   特定虚拟机的详细信息

+   由账户管理的非计算资源

Salt Cloud 模块应该提供几种列出资源的方法，无论是创建新的虚拟机还是管理现有的虚拟机

### 创建虚拟机

大多数云模块中最复杂的组件是`create()`函数，它协调请求虚拟机、等待其可用、登录并安装 Salt 以及接受该虚拟机的 Minion 密钥在 Master 上的任务。许多这些任务已经抽象成可以从云模块中调用的辅助函数，这大大简化了`create()`函数的开发

### 管理其他资源

一旦将前面的组件组合在一起，创建其他用于创建、列出、修改和删除其他资源的函数通常不会花费太多精力

## Libcloud 与 SDK 与直接 REST API 的比较

Salt 附带三种类型的云模块。第一种和原始类型的模块使用名为 Libcloud 的库与云服务提供商通信。使用此类库有一些明显的优点：

+   Libcloud 支持大量的云服务提供商

+   Libcloud 在各个提供商之间提供了一个标准且相对一致的接口

+   Salt Cloud 为 Libcloud 构建了一些内置的功能

+   Libcloud 正在积极开发，并频繁发布新版本

使用 Libcloud 也有一些缺点：

+   并非每个云中的每个功能都由 Libcloud 支持

+   新的云服务提供商可能尚未得到支持

+   一些旧的、不为人知的和专有的驱动程序可能永远不会得到支持

一些云服务提供商还提供了他们自己的库来连接到他们的基础设施。这可能证明是连接到他们的最快、最简单或最可靠的方式。使用提供商自己的 SDK 的一些优点是：

+   开发者可能对 API 有最全面的知识

+   当新功能发布时，SDK 通常是第一个支持它们的库

一些缺点是：

+   一些 SDK 仍然不支持该云服务提供商的所有功能

+   一些 SDK 可能难以使用

与云服务提供商通信的另一种选项是直接与他们通信 REST API。这种方法的一些优点是：

+   您可以控制模块的维护方式

+   您可以在不等待库的新版本的情况下添加自己的功能

但使用直接 REST API 有一些明显的缺点：

+   您必须维护该模块

+   您必须自己添加任何新的功能

+   您可能没有云服务提供商那么多的资源来使用驱动程序

您将需要决定哪种选项最适合您的具体情况。幸运的是，一旦您设置了要使用的连接机制（无论您是自己编写还是使用他人的），使用这些连接的函数之间实际上并没有真正的区别

# 编写通用的云模块

我们将设置一个非常通用的模块，该模块使用直接 REST API 与云提供商进行通信。如果您花了很多时间与不同的 API 交互，您会发现这里使用的风格非常常见。

## 检查所需配置

为了使用云提供商，您需要一个 `__virtual__()` 函数来检查所需配置，并在必要时检查任何依赖项。您还需要一个名为 `get_configured_provider()` 的函数，该函数检查确保连接到您的云提供商所需的配置（至少是身份验证，有时还有其他连接参数）已被指定。我们还需要定义 `__virtualname__`，它包含驱动程序的名称，Salt Cloud 将知道它。让我们从这里开始我们的云模块：

```py
'''
Generic Salt Cloud module

This module is not designed for any specific cloud provider, but is generic
enough that only minimal changes may be required for some providers.

This file should be saved as salt/cloud/clouds/generic.py

Set up the cloud configuration at ``/etc/salt/cloud.providers`` or
``/etc/salt/cloud.providers.d/generic.conf``:

.. code-block:: yaml

    my-cloud-config:
      driver: generic
      # The login user
      user: larry
      # The user's password
      password: 123pass
      # The user's API key
      api_key: 0123456789abcdef
'''
__virtualname__ = 'generic'

def __virtual__():
    '''
    Check for cloud configs
    '''
    # No special libraries required

    if get_configured_provider() is False:
        return False

    return __virtualname__

def get_configured_provider():
    '''
    Make sure configuration is correct
    '''
    return config.is_provider_configured(
        __opts__,
        __active_provider_name__ or __virtualname__,
        ('user', 'password', 'apikey')
    )
```

我们从一个包含有关我们驱动程序所需配置信息的 `docstring` 开始。我们将坚持使用简单的身份验证方案，该方案使用 API 密钥作为 URL 的一部分，以及 HTTP 用户名和密码。

`__virtual__()` 函数首先应该确保安装了所有必需的库。在我们的例子中，我们不需要任何特殊的东西，所以我们将跳过这一部分。然后我们调用 `get_configured_provider()` 来确保所有必需的配置都已就绪，如果一切顺利，我们返回 `__virtualname__`。

`get_configured_provider()` 函数将不会改变，除了模块工作所必需的绝对必需的参数列表之外。如果您打算接受任何可选参数，请不要将它们包含在这个函数中。

### 注意

`get_configured_provider()` 函数提到了另一个内置变量，称为 `__active_provider_name__`。这个变量包含用户在他们的提供者配置中为该模块设置的名称（例如 `my-cloud-config`）以及实际驱动程序的名称（在我们的例子中是 `generic`），两者之间用冒号（`:`）分隔。如果您要使用我们文档字符串中的示例配置，那么 `__active_provider_name__` 将被设置为 `my-cloud-config:generic`。

## 使用 http.query()

Salt 自带了一个用于通过 HTTP 通信的库。这个库本身不是一个连接库；相反，它允许您使用 `urllib2`（Python 的一部分），Tornado（Salt 自身的依赖项），或 `requests`（Python 中非常流行且功能强大的 HTTP 库）。像 Libcloud 一样，Salt 的 HTTP 库力求在所有可用库之间提供一致的接口。如果您需要在该库中使用特定功能，您可以指定要使用的库，但默认情况下使用 Tornado。

这个库位于 `salt.utils` 中，包含了许多与 HTTP 相关的函数。其中最常用的是 `query()` 函数。它不仅支持所有三个后端库，还包括将返回数据从 JSON 或 XML 自动转换为 Python 字典的机制。

`http.query()`的调用通常看起来像这样：

```py
import salt.utils.http
result = salt.utils.http.query(
    'https://api.example.com/v1/resource/action/id',
    'POST',
    data=post_data_dict,
    decode=True,
    decode_type='json',
    opts=__opts__
)
print(result['dict'])
```

## 常见的 REST API

在我们连接到 REST API 之前，我们需要知道它的样子。URL 的结构通常包含以下组件：

```py
https://<hostname>/<version>/<resource>[/<action>[/<id>]]
```

从技术上讲，URL 方案可以是 HTTP，但如果这是你唯一的选择，我建议切换到另一个云服务提供商。

主机名通常包含一些表明它属于 API 的提示，例如`api.example.com`。你的云服务提供商的文档将告诉你这里应该使用哪个主机名。主机名也可能包含有关你正在与之通信的数据中心的信息，例如`eu-north.api.example.com`。

大多数提供商还要求你指定你正在使用的 API 版本。这可能包含在 URL 中，或在`POST`数据中，甚至在客户端请求头中。除非你有非常充分的理由不这样做，否则你应该始终使用最新版本，但云服务提供商通常会支持旧版本，即使只是暂时性的。

资源指的是你实际监控的内容。这可能类似于虚拟机的`instance`或`nodes`，用于引用磁盘的`storage`或`volumes`，或者用于引用预构建操作系统镜像或模板的`images`。我希望我能在这里更加具体，但这将取决于你的云服务提供商。

动作可能出现在也可能不出现在 URL 中。一些云服务提供商将包括`create`、`list`、`modify`、`delete`等动作，后面跟着要管理的资源的 ID，如果需要的话。

然而，使用 HTTP 方法来确定动作正变得越来越普遍。以下方法通常由 REST API 使用：

### GET

这用于仅显示但永远不会更改资源的调用。如果没有提供 ID，则通常会提供一个资源列表。如果使用了 ID，则将返回该特定资源的详细信息。

### POST

这通常用于创建数据的调用，并且经常用于修改数据的调用。如果没有声明 ID，则通常将创建一个新的资源。如果提供了 ID，则将修改现有的资源。

### PATCH

此方法最近被添加用于修改现有资源。如果云服务提供商使用此方法，那么他们不太可能允许使用`POST`来修改现有数据。相反，`POST`将仅用于应用新数据，而`PATCH`将用于更新现有数据。

### DELETE

使用`DELETE`方法的调用通常包括资源类型和要删除的资源的 ID。此方法永远不会用于创建或修改数据；仅用于删除。

## 设置一个`query()`函数

现在我们知道了 API 将是什么样子，让我们创建一个函数来与之通信。我们将使用`http.query()`来与之通信，但我们还需要将一些其他项目包裹在里面。我们从一个函数声明开始：

```py
def _query(
    resource=None,
    action=None,
    method='GET',
    location=None,
    data=None,
):
```

注意，我们已经将此函数设为私有。没有理由允许此函数直接从命令行调用，因此我们需要将其隐藏。我们允许任何参数保持未指定，因为我们不一定总是需要所有这些参数。

让我们继续设置我们的 `_query()` 函数，然后逐一检查其中的各个组件：

```py
import json
import salt.utils.http
import salt.config as config

def _query(
        resource=None,
        action=None,
        params=None,
        method='GET',
        data=None
    ):
    '''
    Make a web call to the cloud provider
    '''
    user = config.get_cloud_config_value(
        'user', get_configured_provider(), __opts__,
    )

    password = config.get_cloud_config_value(
        'password', get_configured_provider(), __opts__,
    )

    api_key = config.get_cloud_config_value(
        'api_key', get_configured_provider(), __opts__,
    )

    location = config.get_cloud_config_value(
        'location', get_configured_provider(), __opts__, default=None
    )

    if location is None:
        location = 'eu-north'

    url = 'https://{0}.api.example.com/v1'.format(location)

    if resource:
        url += '/{0}'.format(resource)

    if action:
        url += '/{0}'.format(action)

    if not isinstance(params, dict):
        params = {}

    params['api_key'] = api_key

    if data is not None:
        data = json.dumps(data)

    result = salt.utils.http.query(
        url,
        method,
        params=params,
        data=data,
        decode=True,
        decode_type='json',
        hide_fields=['api_key'],
        opts=__opts__,
    )

    return result['dict']
```

我们首先收集我们云服务提供商所需的连接参数。`salt.config` 库包含一个名为 `get_cloud_config_value()` 的函数，该函数会在云配置中搜索请求的值。它可以搜索主云配置（通常位于 `/etc/salt/cloud`），以及任何提供者或配置文件配置。在这种情况下，所有配置都应位于提供者配置中，正如我们在文档字符串中所指定的。

一旦收集了 `user`、`password` 和 `api_key`，我们就将注意力转向 `location`。您可能记得，许多云提供商使用主机名来区分不同的数据中心。许多也设有默认数据中心。在我们的通用驱动程序中，我们将假设 `eu-north` 是默认的，并使用它创建一个 URL。我们的 URL 还包含了一个版本，正如我们之前提到的。

然后，我们查看将要使用的资源以及将要对其执行的操作。如果找到，这些操作将被附加到 URL 路径上。有了这些，我们就查看将要添加到 URL 中的任何参数。

`params` 变量指的是将被添加到 URL 中的 `<name>=<value>` 对。这些将以问号（`?`）开头，然后通过 ampersand（`&`）分隔，例如：

```py
http://example.com/form.cgi?name1=value1&name2=value2&name3=value3
```

我们不会自己将这些内容附加到 URL 上，而是让 `http.query()` 函数来处理。如果指定了数据，它将正确地编码这些数据，并将其附加到 URL 的末尾。

如果使用，`params` 需要指定为一个字典。我们知道 `api_key` 将会是其中一个 `params`，因此我们在类型检查之后添加它。

最后，我们需要查看将要发送到云提供商的任何数据。许多提供商要求将 `POST` 数据作为 JSON 字符串发送，而不是作为 URL 编码的数据，因此如果提供了任何数据，我们将在发送之前将其转换为 JSON。

一切准备就绪后，我们使用 `http.query()`（作为 `salt.utils.http.query()`）来实际发起调用。您可以看到 `url`、`method`（在函数声明中指定）、`params` 和 `data`。我们还设置了 `decode` 为 `True` 和 `decode_type` 为 `json`，这样云提供商返回的数据将自动为我们转换为字典。

我们还传递了一个字段列表，以隐藏在 `http.query()` 函数内部可能发生的任何日志记录。这将确保我们的 `api_key` 等数据在生成任何日志时保持私密。而不是记录一个 URL，例如：

```py
https://example.com/?api_key=0123456789abcdef
```

将会记录一个清理过的 URL：

```py
https://example.com/?api_key=XXXXXXXXXX
```

最后，我们传递一个`__opts__`的副本，这样`http.query()`就能访问从`master`或`minion`配置文件中需要的任何变量。

`http.query()`函数将返回一个字典，其中包含一个名为`dict`的项，它包含从云提供商返回的数据，已转换为字典格式。这是我们将其传递回调用我们的`_query()`函数的任何函数的内容。

## 获取配置文件详情

一旦我们能够连接到云提供商，我们就需要能够收集可用于在该提供商上创建虚拟机的信息。这几乎总是包括虚拟机镜像和虚拟机大小的列表。如果一个云提供商有多个数据中心（大多数都有），那么你还需要一个函数来返回这些数据中心的列表。

这三个函数分别称为`avail_images()`、`avail_sizes()`和`avail_locations()`。它们分别通过`salt-cloud`命令使用`--list-images`、`--list-sizes`和`--list-locations`选项访问。

### 列出镜像

镜像指的是预构建的根虚拟机卷。对于 Windows 镜像，这将是指`C:\`磁盘卷。在其他操作系统上，这将是指`/`卷。非常常见的是，云提供商将提供多种不同的操作系统和每种操作系统的多个不同版本。

例如，云提供商可能为 Ubuntu 14.04、Ubuntu 14.10、Ubuntu 15.04 等提供单个镜像，或者它们可能提供每个镜像捆绑 WordPress、MediaWiki、MariaDB 或其他流行的软件包。

在我们的通用云提供商的情况下，可以通过请求`images`资源简单地返回一系列镜像。

```py
def avail_images():
    '''
    Get list of available VM images
    '''
    return _query(resource='images')
```

在配置文件中，使用`image`参数指定镜像。

### 列出大小

大小是云提供商特有的一个概念，实际上并非每个云提供商都支持它们。根据提供商的不同，大小通常指的是处理器数量、处理器速度、RAM 大小、磁盘空间、磁盘类型（硬盘驱动器与 SSD）等的组合。

再次强调，我们的通用云提供商将在`sizes`资源下返回一系列大小。

```py
def avail_sizes():
    '''
    Get list of available VM sizes
    '''
    return _query(resource='sizes')
```

在配置文件中，使用`size`参数指定大小。

### 列出位置

根据云提供商的不同，位置可能指一个具体的数据中心，世界上某个地区的区域，甚至是一个包含多个数据中心的区域内的特定数据中心。

正如我们之前所说的，位置通常会被添加到与 API 通信所使用的 URL 之前。在我们的通用云提供商的情况下，位置是通过`regions`资源进行查询的。

```py
def avail_locations():
    '''
    Get list of available locations
    '''
    return _query(resource='locations')
```

在配置文件中，使用`location`参数指定位置。

## 列出节点

下一步是显示该云提供商账户中当前存在的节点。有三个`salt-cloud`参数可以显示节点数据：`-Q`或`--query`、`-F`或`--full-query`和`-S`或`--select-query`。每个选项都会查询每个配置的云提供商，并一次性返回所有信息。

### 查询标准节点数据

对于每个节点，应该始终提供六条信息。当使用`salt-cloud`与`-Q`参数时，这些数据会被显示：

+   `id`：此虚拟机由云提供商使用的 ID。

+   `image`：创建此虚拟机使用的镜像。如果此数据不可用，应设置为`None`。

+   `size`：创建此虚拟机使用的尺寸。如果此数据不可用，应设置为`None`。

+   `state`：此虚拟机的当前运行状态。这通常是`RUNNING`、`STOPPED`、`PENDING`（虚拟机仍在启动中）或`TERMINATED`（虚拟机已被销毁，但尚未清理）。如果此数据不可用，应设置为 None。

+   `private_ips`：在云提供商的内部网络上使用的任何私有 IP 地址。这些应作为列表返回。如果此数据不可用，列表应为空。

+   `public_ips`：此虚拟机可用的任何公网 IP 地址。应包括任何 IPv6 地址。这些 IP 应作为列表返回。如果此数据不可用，列表应为空。

用户应该能够访问所有这些变量，即使它们为空或设置为 None。这也是`-Q`参数应该返回的唯一数据。为了返回这些数据，我们使用一个名为`list_nodes()`的函数：

```py
def list_nodes():
    '''
    List of nodes, with standard query data
    '''
    ret = {}
    nodes = _query(resource='instances')
    for node in nodes:
        ret[node] = {
            'id': nodes[node]['id'],
            'image': nodes[node].get('image', None),
            'size': nodes[node].get('size', None),
            'state': nodes[node].get('state', None),
            'private_ips': nodes[node].get('private_ips', []),
            'public_ips': nodes[node].get('public_ips', []),
        }
    return ret
```

### 查询完整节点数据

虚拟机通常包含比`-Q`返回的信息多得多的信息。如果你想查看云提供商愿意并且能够显示给你的所有信息，请使用`-F`标志。这对应于一个名为`list_nodes_full()`的函数：

```py
def list_nodes_full():
    '''
    List of nodes, with full node data
    '''
    return _query(resource='instances')
```

有时，你可能只对一组非常具体的数据感兴趣。例如，你可能只想显示虚拟机的 ID、公网 IP 和状态。`-S`选项允许你执行一个查询，只返回完整查询中可用的字段的选择。这个选择本身是在主云配置文件中定义的列表（通常为`/etc/salt/cloud`）：

```py
query.selection:
  - id
  - public_ips
  - state
```

查询本身是由一个名为`list_nodes_select()`的函数执行的。一些提供商可能需要做一些特殊操作来分离这些数据，但大多数情况下，你可以直接使用`salt.utils.cloud`库中提供的`list_nodes_select()`函数：

```py
import salt.utils.cloud

def list_nodes_select():
    '''
    Return a list of the VMs that are on the provider, with select fields
    '''
    return salt.utils.cloud.list_nodes_select(
        list_nodes_full('function'), __opts__['query.selection'],
    )
```

## 创建虚拟机

任何云模块最复杂的部分传统上一直是`create()`函数。这是因为这个函数不仅仅是启动一个虚拟机。它的任务通常可以分解为以下组件：

+   请求云提供商创建虚拟机

+   等待虚拟机可用

+   登录到该虚拟机并安装 Salt

+   接受该虚拟机的 Minion 密钥在 Master 上

一些更复杂的云服务提供商可能包括额外的步骤，例如根据配置文件请求不同类型的虚拟机，或将卷附加到虚拟机上。此外，`create()`函数应该在 Salt 的事件总线上触发事件，让主服务器知道创建过程的进度。

在我们进入`create()`函数之前，我们应该准备另一个名为`request_instance()`的函数。这个函数将为我们做两件事：

+   它可以直接从`create()`函数中调用，这将简化`create()`函数

+   它可以在`create()`函数外部调用，当需要非 Salt 虚拟机时

这个函数不需要做太多。正如其名称所暗示的，它只需要请求云服务提供商创建一个虚拟机。但需要收集一些信息来构建 HTTP 请求：

```py
def request_instance(vm_):
    '''
    Request that a VM be created
    '''
    request_kwargs = {
        'name': vm_['name'],
        'image': vm_['image'],
        'size': vm_['size'],
        'location': vm_['location']
    }

    salt.utils.cloud.fire_event(
        'event',
        'requesting instance',
        'salt/cloud/{0}/requesting'.format(vm_['name']),
        {'kwargs': request_kwargs},
        transport=__opts__['transport']
    )

    return _query(
        resource='instances',
        method='POST',
        data=request_kwargs,
    )
```

你可能已经注意到了在这个函数中调用`salt.utils.cloud.fire_event()`。每次你在`create()`函数（或由`create()`调用的函数）中做重大操作时，都应该触发一个事件，提供一些关于你即将做什么的信息。这些事件将被事件反应器捕获，允许主服务器跟踪进度，并在配置为这样做的情况下，在正确的时间执行额外任务。

我们还将创建一个名为`query_instance()`的函数。这个函数将监视新请求的虚拟机，等待 IP 地址变得可用。这个 IP 地址将用于登录虚拟机并配置它。

```py
def query_instance(vm_):
    '''
    Query a VM upon creation
    '''
    salt.utils.cloud.fire_event(
        'event',
        'querying instance',
        'salt/cloud/{0}/querying'.format(vm_['name']),
        transport=__opts__['transport']
    )

    def _query_ip_address():
        nodes = list_nodes_full()
        data = nodes.get(vm_['name'], None)
        if not data:
            return False

        if 'public_ips' in data:
            return data['public_ips']
        return None

    data = salt.utils.cloud.wait_for_ip(
        _query_ip_address,
        timeout=config.get_cloud_config_value(
            'wait_for_ip_timeout', vm_, __opts__, default=10 * 60),
        interval=config.get_cloud_config_value(
            'wait_for_ip_interval', vm_, __opts__, default=10),
        interval_multiplier=config.get_cloud_config_value(
            'wait_for_ip_interval_multiplier', vm_, __opts__, default=1),
    )

    return data
```

这个函数使用了 Salt 附带的一个名为`salt.utils.cloud.wait_for_ip()`的函数。该函数接受一个回调，我们将其定义为嵌套函数，称为`_query_ip_address()`。这个嵌套函数会检查 IP 地址是否存在。如果存在，则`salt.utils.cloud.wait_for_ip()`将停止等待并继续执行。如果尚未存在，它将继续等待。

我们还传递了三个其他参数。`timeout`定义了等待 IP 地址出现的时间长度（在我们的案例中是十分钟）；`interval`告诉 Salt Cloud 在查询之间等待多长时间（我们的默认值是十秒）。

你可能会想使用更短的间隔，但许多云服务提供商如果账户似乎在滥用其权限，会限制请求。在此方面，`interval_multiplier`会在每次请求后增加`interval`。例如，如果`interval`设置为 1 且`interval_multiplier`设置为 2，那么请求将间隔 1 秒，然后是 2 秒、4 秒、8 秒、16 秒、32 秒，以此类推。

在这两个函数就位后，我们最终可以设置我们的`create()`函数。它需要一个参数，即一个包含配置文件、提供者和主要云配置数据的字典：

```py
def create(vm_):
    '''
    Create a single VM
    '''
    salt.utils.cloud.fire_event(
        'event',
        'starting create',
        'salt/cloud/{0}/creating'.format(vm_['name']),
        {
            'name': vm_['name'],
            'profile': vm_['profile'],
            'provider': vm_['driver'],
        },
        transport=__opts__['transport']
    )

    create_data = request_instance(vm_)
    query_data = query_instance(vm_)

    vm_['key_filename'] = config.get_cloud_config_value(
        'private_key', vm_, __opts__, search_global=False, default=None
    )
    vm_['ssh_host'] = query_data['public_ips'][0]

    salt.utils.cloud.bootstrap(vm_, __opts__)

    salt.utils.cloud.fire_event(
        'event',
        'created instance',
        'salt/cloud/{0}/created'.format(vm_['name']),
        {
            'name': vm_['name'],
            'profile': vm_['profile'],
            'provider': vm_['driver'],
        },
        transport=__opts__['transport']
    )

    return query_data
```

我们的功能开始于触发一个事件，声明创建过程正在开始。然后我们允许`request_instance()`和`query_instance()`执行它们的工作，从配置数据中提取 SSH 密钥文件名，然后从虚拟机数据中抓取用于从虚拟机登录到盒子的 IP 地址。

下一步涉及等待虚拟机变得可用，然后登录并配置它。但由于这个过程在所有云服务提供商之间都是相同的，所以它已经被整合到`salt.utils.cloud`中的另一个辅助函数`bootstrap()`中。`bootstrap()`函数甚至会为我们触发额外的事件，让事件反应器了解其自身状态。

最后，我们触发一个最后的事件，声明虚拟机的信息，并将虚拟机的数据返回给用户。

### 小贴士

你可能已经注意到，我们触发的事件都包含一个以`salt/cloud/`开头的标签，然后是虚拟机的名称，然后是我们当前执行步骤的简称。如果你在与更复杂的云服务提供商一起工作，并希望触发针对它们的特定事件，请保持标签看起来相同，尽可能简单。这将帮助你的用户跟踪所有你的云标签。

## 销毁虚拟机

能够销毁虚拟机与能够创建虚拟机一样重要，但过程幸运地要简单得多。请注意，在销毁时也应该触发事件：一次在发生之前，一次在发生之后：

```py
def destroy(name):
    '''
    Destroy a machine by name
    '''
    salt.utils.cloud.fire_event(
        'event',
        'destroying instance',
        'salt/cloud/{0}/destroying'.format(name),
        {'name': name},
        transport=__opts__['transport']
    )

    nodes = list_nodes_full()
    ret = _query(
        resource='instances/{0}'.format(nodes[name]['id']),
        location=node['location'],
        method='DELETE'
    )

    salt.utils.cloud.fire_event(
        'event',
        'destroyed instance',
        'salt/cloud/{0}/destroyed'.format(name),
        {'name': name},
        transport=__opts__['transport']
    )

    if __opts__.get('update_cachedir', False) is True:
        salt.utils.cloud.delete_minion_cachedir(
            name, __active_provider_name__.split(':')[0], __opts__
        )

    return ret
```

在这个函数中，我们做了另一件重要的事情。Salt Cloud 有维护虚拟机信息缓存的能力。我们之前没有看到这一点，因为`bootstrap()`函数在创建虚拟机时处理填充缓存。然而，由于没有销毁机器的通用方法，我们需要手动处理这一点。

## 使用动作和函数

到目前为止，我们编写的所有函数都是通过特殊的命令行参数（如`--query`或`--provision`）直接调用的。然而，云服务提供商可能能够执行的操作并不一定像我们之前看到的那么标准化。

例如，大多数云服务提供商都有`start`、`stop`和`restart`的 API 方法。但有些提供商并不支持所有这些；`start`和`stop`可能可用，但`restart`不可用。或者`start`和`restart`可用，但`stop`不可用。其他操作，如列出 SSH 密钥，可能在一个云服务提供商上可用，但在另一个提供商上不可用。

当涉及到对云服务提供商的操作时，主要有两种类型的操作可以进行。针对虚拟机（VM）的特定操作（如`stop`、`start`、`restart`等）在 Salt Cloud 中被称为**动作**。与云服务提供商的组件交互，但不特定于虚拟机的操作（如列出 SSH 密钥、修改用户等），在 Salt Cloud 中被称为**函数**。

### 使用动作

使用`--action`参数通过`salt-cloud`命令调用操作。因为它们作用于特定的虚拟机，所以传递给它们的第一个参数是一个名称。如果从命令行传递其他参数，它们将出现在名为`kwargs`的字典中。还有一个额外的参数，称为`call`，它告诉函数是否使用`--action`或`--function`调用。你可以使用这个来通知用户他们是否错误地调用了操作或函数：

```py
def rename(name, kwargs, call=None):
    '''
    Properly rename a node. Pass in the new name as "newname".
    '''
    if call != 'action':
        raise SaltCloudSystemExit(
            'The rename action must be called with -a or --action.'
        )

    salt.utils.cloud.rename_key(
        __opts__['pki_dir'], name, kwargs['newname']
    )

    nodes = list_nodes_full()
    return _query(
        resource='instances/{0}'.format(nodes[name]['id']),
        action='rename',
        method='POST',
        data={'name': kwargs['newname']}
    )
```

即使你并不打算向用户发出警告，你也必须接受`call`参数；无论是否传递给它，它都会被传递，如果没有提供，将会引发错误。

再次提醒，我又给你带来了一个惊喜。由于这个操作将重命名虚拟机，我们需要通知 Salt。如果我们不这样做，那么 Master 将无法联系 Minion。通常，有一个辅助函数（`salt.utils.cloud.rename_key()`）会为我们完成这项工作。

### 使用函数

因为函数不作用于特定的虚拟机，所以它们不需要名称参数。然而，它们确实需要`kwargs`和`call`参数，即使你并不打算使用它们。

```py
def show_image(kwargs, call=None):
    '''
    Show the details for a VM image
    '''
    if call != 'function':
        raise SaltCloudSystemExit(
            'The show_image function must be called with -f or --function.'
        )

    return _query(resource='images/{0}'.format(kwargs['image']))
```

如果你将`call`参数添加到模块中的各个函数中，你将能够直接使用`--action`或`--function`参数调用它们。这对于像`list_nodes()`这样的函数非常有用，当你只想一次查看一个云提供商的虚拟机，而不是一次性查看所有虚拟机时。

唯一不能这样调用的公共函数是`create()`函数。可以使用`--action`参数调用`destroy()`，而我们迄今为止添加的几乎所有其他内容都可以使用`--function`参数调用。我们将继续添加这些功能到我们的最终云模块中。

# 最终的云模块

当我们完成时，最终的云模块将看起来像这样：

```py
'''
Generic Salt Cloud module

This module is not designed for any specific cloud provider, but is generic
enough that only minimal changes may be required for some providers.

This file should be saved as salt/cloud/clouds/generic.py

Set up the cloud configuration at ``/etc/salt/cloud.providers`` or
``/etc/salt/cloud.providers.d/generic.conf``:

.. code-block:: yaml

    my-cloud-config:
      driver: generic
      # The login user
      user: larry
      # The user's password
      password: 123pass
'''
import json
import salt.utils.http
import salt.utils.cloud
import salt.config as config
from salt.exceptions import SaltCloudSystemExit

__virtualname__ = 'generic'

def __virtual__():
    '''
    Check for cloud configs
    '''
    if get_configured_provider() is False:
        return False

    return __virtualname__

def get_configured_provider():
    '''
    Make sure configuration is correct
    '''
    return config.is_provider_configured(
        __opts__,
        __active_provider_name__ or __virtualname__,
        ('user', 'password')
    )

def request_instance(vm_):
    '''
    Request that a VM be created
    '''
    request_kwargs = {
        'name': vm_['name'],
        'image': vm_['image'],
        'size': vm_['size'],
        'location': vm_['location']
    }

    salt.utils.cloud.fire_event(
        'event',
        'requesting instance',
        'salt/cloud/{0}/requesting'.format(vm_['name']),
        {'kwargs': request_kwargs},
        transport=__opts__['transport']
    )

    return _query(
        resource='instances',
        method='POST',
        data=request_kwargs,
    )

def query_instance(vm_):
    '''
    Query a VM upon creation
    '''
    salt.utils.cloud.fire_event(
        'event',
        'querying instance',
        'salt/cloud/{0}/querying'.format(vm_['name']),
        transport=__opts__['transport']
    )

    def _query_ip_address():
        nodes = list_nodes_full()
        data = nodes.get(vm_['name'], None)
        if not data:
            log.error('There was an empty response from the cloud provider')
            return False

        log.debug('Returned query data: {0}'.format(data))

        if 'public_ips' in data:
            return data['public_ips']
        return None

    data = salt.utils.cloud.wait_for_ip(
        _query_ip_address,
        timeout=config.get_cloud_config_value(
            'wait_for_ip_timeout', vm_, __opts__, default=10 * 60),
        interval=config.get_cloud_config_value(
            'wait_for_ip_interval', vm_, __opts__, default=10),
        interval_multiplier=config.get_cloud_config_value(
            'wait_for_ip_interval_multiplier', vm_, __opts__, default=1),
    )

    return data

def create(vm_):
    '''
    Create a single VM
    '''
    salt.utils.cloud.fire_event(
        'event',
        'starting create',
        'salt/cloud/{0}/creating'.format(vm_['name']),
        {
            'name': vm_['name'],
            'profile': vm_['profile'],
            'provider': vm_['driver'],
        },
        transport=__opts__['transport']
    )

    create_data = request_instance(vm_)
    query_data = query_instance(vm_)

    vm_['key_filename'] = config.get_cloud_config_value(
        'private_key', vm_, __opts__, search_global=False, default=None
    )
    vm_['ssh_host'] = query_data['public_ips'][0]

    salt.utils.cloud.bootstrap(vm_, __opts__)

    salt.utils.cloud.fire_event(
        'event',
        'created instance',
        'salt/cloud/{0}/created'.format(vm_['name']),
        {
            'name': vm_['name'],
            'profile': vm_['profile'],
            'provider': vm_['driver'],
        },
        transport=__opts__['transport']
    )

    return query_data

def destroy(name, call=None):
    '''
    Destroy a machine by name
    '''
    salt.utils.cloud.fire_event(
        'event',
        'destroying instance',
        'salt/cloud/{0}/destroying'.format(name),
        {'name': name},
        transport=__opts__['transport']
    )

    nodes = list_nodes_full()
    ret = _query(
        resource='instances/{0}'.format(nodes[name]['id']),
        location=node['location'],
        method='DELETE'
    )

    salt.utils.cloud.fire_event(
        'event',
        'destroyed instance',
        'salt/cloud/{0}/destroyed'.format(name),
        {'name': name},
        transport=__opts__['transport']
    )

    if __opts__.get('update_cachedir', False) is True:
        salt.utils.cloud.delete_minion_cachedir(
            name, __active_provider_name__.split(':')[0], __opts__
        )

    return ret

def rename(name, kwargs, call=None):
    '''
    Properly rename a node. Pass in the new name as "newname".
    '''
    if call != 'action':
        raise SaltCloudSystemExit(
            'The rename action must be called with -a or --action.'
        )

    salt.utils.cloud.rename_key(
        __opts__['pki_dir'], name, kwargs['newname']
    )

    nodes = list_nodes_full()
    return _query(
        resource='instances/{0}'.format(nodes[name]['id']),
        action='rename',
        method='POST',
        data={'name': kwargs['newname']}
    )

def show_image(kwargs, call=None):
    '''
    Show the details for a VM image
    '''
    if call != 'function':
        raise SaltCloudSystemExit(
            'The show_image function must be called with -f or --function.'
        )

    return _query(resource='images/{0}'.format(kwargs['image']))

def list_nodes(call=None):
    '''
    List of nodes, with standard query data
    '''
    ret = {}
    nodes = _query(resource='instances')
    for node in nodes:
        ret[node] = {
            'id': nodes[node]['id'],
            'image': nodes[node].get('image', None),
            'size': nodes[node].get('size', None),
            'state': nodes[node].get('state', None),
            'private_ips': nodes[node].get('private_ips', []),
            'public_ips': nodes[node].get('public_ips', []),
        }
    return ret

def list_nodes_full(call=None):
    '''
    List of nodes, with full node data
    '''
    return _query(resource='instances')

def list_nodes_select(call=None):
    '''
    Return a list of the VMs that are on the provider, with select fields
    '''
    return salt.utils.cloud.list_nodes_select(
        list_nodes_full('function'), __opts__['query.selection'], call,
    )

def avail_images(call=None):
    '''
    Get list of available VM images
    '''
    return _query(resource='images')

def avail_sizes(call=None):
    '''
    Get list of available VM sizes
    '''
    return _query(resource='sizes')

def avail_locations(call=None):
    '''
    Get list of available locations
    '''
    return _query(resource='locations')

def _query(
        resource=None,
        action=None,
        params=None,
        method='GET',
        location=None,
        data=None
    ):
    '''
    Make a web call to the cloud provider
    '''
    user = config.get_cloud_config_value(
        'user', get_configured_provider(), __opts__, search_global=False
    )
    password = config.get_cloud_config_value(
        'password', get_configured_provider(), __opts__,
    )
    api_key = config.get_cloud_config_value(
        'api_key', get_configured_provider(), __opts__,
    )
    location = config.get_cloud_config_value(
        'location', get_configured_provider(), __opts__, default=None
    )

    if location is None:
        location = 'eu-north'

    url = 'https://{0}.api.example.com/v1'.format(location)

    if resource:
        url += '/{0}'.format(resource)

    if action:
        url += '/{0}'.format(action)

    if not isinstance(params, dict):
        params = {}

    params['api_key'] = api_key

    if data is not None:
        data = json.dumps(data)

    result = salt.utils.http.query(
        url,
        method,
        params=params,
        data=data,
        decode=True,
        decode_type='json',
        hide_fields=['api_key'],
        opts=__opts__,
    )

    return result['dict']
```

# 云模块故障排除

云模块可能看起来令人畏惧，因为要制作一个连贯的代码块需要许多组件。但是，如果你以小块的方式处理模块，它将更容易处理。

## 首先编写`avail_sizes()`或`avail_images()`

每次我编写一个新的云模块时，我首先会尝试让一些示例代码运行起来，以执行一个小查询。因为图像和大小对于创建虚拟机至关重要，而且这些调用通常非常简单，所以它们通常是实现起来最简单的。

一旦这些函数中的一个开始工作，将其拆分为一个`_query()`函数（如果你不是从那里开始的）和一个调用它的函数。然后编写另一个调用它的函数。你可能需要为前几个函数调整`_query()`，但之后它将稳定下来，几乎不需要任何更改。

## 使用快捷方式

我无法告诉你我花了多少小时等待虚拟机启动，只是为了测试一小段代码。如果你将`create()`函数分解成许多更小的函数，那么你可以根据需要临时硬编码虚拟机数据，并跳过那些会浪费太多时间的操作。只是确保在完成时移除这些捷径！

# 摘要

Salt Cloud 旨在处理计算资源，尽管可以根据需要添加额外的云功能。可以使用 Libcloud、SDK 或直接 REST API 编写云模块；每种方法都有其优缺点。现代 REST API 通常非常相似且易于使用。一个连贯的云模块需要几个功能，但大多数都不复杂。操作是在单个虚拟机上执行的，而功能是在云提供商本身上执行的。

现在我们已经了解了云模块，是时候开始监控我们的资源了。接下来是：信标。
