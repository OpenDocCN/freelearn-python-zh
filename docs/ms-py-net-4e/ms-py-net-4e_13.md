

# 第十三章：使用 Elastic Stack 进行网络数据分析

在 *第七章*，*使用 Python 进行网络监控（第一部分*）和 *第八章*，*使用 Python 进行网络监控（第二部分*）中，我们讨论了监控网络的各种方法。在这两章中，我们探讨了两种不同的网络数据收集方法：我们可以从网络设备（如 SNMP）检索数据，或者我们可以通过基于流的导出监听网络设备发送的数据。收集数据后，我们需要将数据存储在数据库中，然后分析数据以获得洞察力，从而决定数据的意义。大多数情况下，分析结果以图表的形式显示，无论是折线图、柱状图还是饼图。我们可以为每个步骤使用单独的工具，如 PySNMP、Matplotlib 和 Pygal，或者我们可以利用一体化的工具，如 Cacti 或 ntop 进行监控。这两章中介绍的工具为我们提供了基本的监控和网络理解。

然后，我们继续到 *第九章*，*使用 Python 构建网络 Web 服务*，以构建 API 服务来抽象我们的网络，使其从高级工具中分离出来。在 *第十一章*，*AWS 云网络* 和 *第十二章*，*Azure 云网络* 中，我们使用 AWS 和 Azure 将本地网络扩展到云端。在这些章节中，我们覆盖了大量的内容，并拥有了一套坚实的工具来帮助我们使网络可编程。

从本章开始，我们将基于前几章的工具集，并查看我在熟悉了前几章中介绍的工具后，在旅程中找到的其他有用工具和项目。在本章中，我们将探讨一个开源项目，Elastic Stack ([`www.elastic.co`](https://www.elastic.co))，它可以帮助我们在之前所见的范围之外分析和监控我们的网络。

在本章中，我们将探讨以下主题：

+   什么是 Elastic（或 ELK）Stack？

+   Elastic Stack 安装

+   使用 Logstash 进行数据摄取

+   使用 Beats 进行数据摄取

+   使用 Elasticsearch 进行搜索

+   使用 Kibana 进行数据可视化

让我们先回答一个问题：Elastic Stack 究竟是什么？

# 什么是 Elastic Stack？

Elastic Stack 也被称为“ELK” Stack。那么，它究竟是什么呢？让我们看看开发者们用自己的话是如何描述的 ([`www.elastic.co/what-is/elk-stack`](https://www.elastic.co/what-is/elk-stack)):

> “ELK”是三个开源项目的缩写：Elasticsearch、Logstash 和 Kibana。Elasticsearch 是一个搜索和分析引擎。Logstash 是一个服务器端数据处理管道，可以同时从多个来源摄取数据，对其进行转换，然后将数据发送到类似 Elasticsearch 的“储藏室”。Kibana 允许用户在 Elasticsearch 中使用图表和图形可视化数据。Elastic Stack 是 ELK Stack 的下一进化阶段。

![包含图形用户界面的图片 自动生成描述](img/B18403_13_01.png)

图 13.1：Elastic Stack（来源：https://www.elastic.co/what-is/elk-stack）

从声明中我们可以看出，Elastic Stack 是由不同项目组成的集合，这些项目协同工作，覆盖了数据收集、存储、检索、分析和可视化的整个范围。这个堆栈的优点在于它紧密集成，但每个组件也可以单独使用。如果我们不喜欢 Kibana 用于可视化，我们可以轻松地插入 Grafana 用于图表。如果我们想使用其他数据摄取工具呢？没问题，我们可以使用 RESTful API 将我们的数据发布到 Elasticsearch。堆栈的中心是 Elasticsearch，这是一个开源的分布式搜索引擎。其他项目都是为了增强和支持搜索功能而创建的。一开始这可能听起来有点令人困惑，但当我们更深入地了解项目的组件时，它将变得更加清晰。

他们为什么将 ELK Stack 的名字改为 Elastic Stack？2015 年，Elastic 引入了一系列轻量级、单用途的数据传输工具，称为 Beats。它们立刻受到欢迎，并且继续非常受欢迎，但创造者无法为“B”想出一个好的首字母缩略词，因此决定将整个堆栈重命名为 Elastic Stack。

我们将重点关注 Elastic Stack 的网络监控和数据分析方面。尽管如此，这个堆栈有许多用例，包括风险管理、电子商务个性化、安全分析、欺诈检测等。它被各种组织使用，从思科、Box 和 Adobe 这样的网络公司，到美国宇航局喷气推进实验室、美国人口普查局等政府机构（[`www.elastic.co/customers/`](https://www.elastic.co/customers/))。

当我们谈论 Elastic 时，我们指的是 Elastic Stack 背后的公司。这些工具是开源的，公司通过销售支持、托管解决方案和围绕开源项目的咨询服务来赚钱。公司的股票在纽约证券交易所上市，股票代码为 ESTC。

现在我们对 ELK Stack 有了一个更好的了解，让我们看看本章的实验室拓扑。

# 实验室拓扑

对于网络实验室，我们将重用我们在 *第八章*，*Python 网络监控第二部分* 中使用的网络拓扑。网络设备将具有位于 `192.168.2.0/24` 管理网络的管理接口，以及位于 `10.0.0.0/8` 网络的互连和 `/30s` 子网。

我们在哪里可以安装 ELK Stack 到实验室？在生产环境中，我们应该在专用集群中运行 ELK Stack。然而，在我们的实验室中，我们可以通过 Docker 容器快速启动一个测试实例。如果需要 Docker 的复习，请参阅 *第五章*，*网络工程师的 Docker 容器*。

下面是我们网络实验室拓扑的图形表示：

![图描述自动生成](img/B18403_13_02.png)

图 13.2：实验室拓扑

| **设备** | **管理 IP** | **环回 IP** |
| --- | --- | --- |
| r1 | `192.168.2.218` | `192.168.0.1` |
| r2 | `192.168.2.219` | `192.168.0.2` |
| r3 | `192.168.2.220` | `192.168.0.3` |
| r5 | `192.168.2.221` | `192.168.0.4` |
| r6 | `192.168.2.222` | `192.168.0.5` |

Ubuntu 主机信息如下：

| **设备名称** | **外部链接 Eth0** | **内部 IP Eth1** |
| --- | --- | --- |
| 客户端 | `192.168.2.211` | `10.0.0.9` |
| 服务器 | `192.168.2.212` | `10.0.0.5` |

要运行多个容器，我们应该至少为宿主机分配 4 GB RAM 或更多。如果尚未启动，请先启动 Docker Engine，然后从 Docker Hub 拉取镜像：

```py
$ sudo service docker start
$ docker network create elastic
$ docker pull docker.elastic.co/elasticsearch/elasticsearch:8.4.2
$ docker run --name elasticsearch –-rm -it --network elastic -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -t docker.elastic.co/elasticsearch/elasticsearch:8.4.2 
```

当 Docker 容器运行时，生成的默认 Elastic 用户密码和 Kibana 注册令牌将输出到终端；请记住它们，因为我们稍后会用到。你可能需要稍微向上滚动屏幕以找到它们：

```py
->  Password for the elastic user (reset with 'bin/elasticsearch-reset-password -u elastic'):
  <password>
->  Configure Kibana to use this cluster:
* Run Kibana and click the configuration link in the terminal when Kibana starts.
* Copy the following enrollment token and paste it into Kibana in your browser (valid for the next 30 minutes):
  <token> 
```

一旦 Elasticsearch 容器启动，我们可以通过浏览到`https://<你的 IP>:9200`来测试实例：

![文本  自动生成的描述](img/B18403_13_03.png)

图 13.3：Elasticsearch 初始结果

然后，我们可以从另一个终端拉取并运行 Kibana 容器镜像：

```py
$ docker pull docker.elastic.co/kibana/kibana:8.4.2
$ docker run --name kibana –-rm -it --network elastic -p 5601:5601 docker.elastic.co/kibana/kibana:8.4.2 
```

一旦 Kibana 启动，我们就可以通过端口 5601 访问它：

![图形用户界面，文本，应用程序，聊天或文本消息  自动生成的描述](img/B18403_13_04.png)

图 13.4：Kibana 启动页面

注意它正在请求我们之前记下的注册令牌。我们可以将其粘贴并点击**配置 Elastic**。它将提示我们输入令牌，该令牌现在显示在 Kibana 终端上。一旦认证通过，Kibana 将开始配置 Elastic：

![图形用户界面，文本，应用程序  自动生成的描述](img/B18403_13_05.png)

图 13.5：配置 Elastic

最后，我们应该能够通过`http://<ip>:5601`访问 Kibana 界面。目前我们不需要任何集成；我们将选择**自行探索**：

![](img/B18403_13_06.png)

图 13.6：

我们将看到一个选项来加载一些示例数据。这是了解这个工具的好方法，所以让我们导入这些数据：

![图形用户界面，应用程序，网站  自动生成的描述](img/B18403_13_07.png)

图 13.7：Kibana 主页

我们将选择**尝试示例数据**并添加示例电子商务订单、示例航班数据和示例网络日志：

![](img/B18403_13_08.png)

图 13.8：添加示例数据

总结一下，我们现在已经在管理主机上以转发端口的形式运行了 Elasticsearch 和 Kibana 容器：

```py
$ docker ps
CONTAINER ID   IMAGE                                                 COMMAND                  CREATED              STATUS              PORTS                                                                                  NAMES
f7d6d8842060   docker.elastic.co/kibana/kibana:8.4.2                 "/bin/tini -- /usr/l…"   42 minutes ago       Up 42 minutes       0.0.0.0:5601->5601/tcp, :::5601->5601/tcp                                              kibana
dc2a1fa15e3b   docker.elastic.co/elasticsearch/elasticsearch:8.4.2   "/bin/tini -- /usr/l…"   46 minutes ago       Up 46 minutes       0.0.0.0:9200->9200/tcp, :::9200->9200/tcp, 0.0.0.0:9300->9300/tcp, :::9300->9300/tcp   elasticsearch 
```

太好了！我们几乎完成了。最后一部分是 Logstash。由于我们将使用不同的 Logstash 配置文件、模块和插件，我们将使用软件包而不是 Docker 容器在管理主机上安装它。Logstash 需要 Java 来运行：

```py
$ sudo apt install openjdk-11-jre-headless
$ java --version
openjdk 11.0.16 2022-07-19
OpenJDK Runtime Environment (build 11.0.16+8-post-Ubuntu-0ubuntu122.04)
OpenJDK 64-Bit Server VM (build 11.0.16+8-post-Ubuntu-0ubuntu122.04, mixed mode, sharing) 
```

我们可以下载 Logstash 捆绑包：

```py
$ wget https://artifacts.elastic.co/downloads/logstash/logstash-8.4.2-linux-x86_64.tar.gz
$ tar -xvzf logstash-8.4.2-linux-x86_64.tar.gz
$ cd logstash-8.4.2/ 
```

我们将修改 Logstash 配置文件中的几个字段：

```py
$ vim config/logstash.yml 
# change the following fields
node.name: mastering-python-networking
api.http.host: <your host ip>
api.http.port: 9600-9700 
```

我们现在不会启动 Logstash。我们将等待在本章的后面安装与网络相关的插件并创建必要的配置文件后，再启动 Logstash 进程。

让我们在下一节中花点时间看看如何将 ELK Stack 作为托管服务进行部署。

# 作为服务的 Elastic Stack

Elasticsearch 是一种流行的服务，由 Elastic.co 和其他云提供商提供托管选项。Elastic Cloud ([`www.elastic.co/cloud/`](https://www.elastic.co/cloud/)) 没有自己的基础设施，但它提供了在 AWS、Google Cloud Platform 或 Azure 上启动部署的选项。由于 Elastic Cloud 是基于其他公共云 VM 提供的，因此成本将略高于直接从云提供商（如 AWS）获取：

![时间线描述自动生成](img/B18403_13_09.png)

图 13.9：Elastic Cloud 产品

AWS 提供了与现有 AWS 产品紧密集成的托管 OpenSearch 产品 ([`aws.amazon.com/opensearch-service/`](https://aws.amazon.com/opensearch-service/))。例如，AWS CloudWatch 日志可以直接流式传输到 AWS OpenSearch 实例 ([`docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_OpenSearch_Stream.html`](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_OpenSearch_Stream.html) )。

从我的个人经验来看，尽管 Elastic Stack 因其优势而具有吸引力，但它是一个容易上手但如果没有陡峭的学习曲线则难以扩展的项目。如果我们不每天处理 Elasticsearch，学习曲线将更加陡峭。如果你像我一样，想利用 Elastic Stack 提供的功能，但又不想成为全职的 Elastic 工程师，我强烈建议使用托管选项进行生产。

选择哪个托管提供商取决于你对云提供商锁定和是否想使用最新功能的偏好。由于 Elastic Cloud 是由 Elastic Stack 项目背后的团队开发的，因此他们通常比 AWS 更快地提供最新功能。另一方面，如果你的基础设施完全建立在 AWS 云中，拥有一个紧密集成的 OpenSearch 实例可以节省你维护单独集群所需的时间和精力。

让我们在下一节中查看一个从数据摄取到可视化的端到端示例。

# 第一个端到端示例

新手对 Elastic Stack 最常见的反馈之一是需要了解多少细节才能开始。要在 Elastic Stack 中获得第一个可用的记录，用户需要构建一个集群，分配主节点和数据节点，摄取数据，创建索引，并通过网页或命令行界面进行管理。多年来，Elastic Stack 简化了安装过程，改进了其文档，并为新用户创建了示例数据集，以便在使用堆栈进行生产之前熟悉工具。

在 Docker 容器中运行组件有助于减轻安装的一些痛苦，但增加了维护的复杂性。在虚拟机与容器之间运行它们是一种权衡。

在我们深入探讨 Elastic Stack 的不同组件之前，查看一个跨越 Logstash、Elasticsearch 和 Kibana 的示例是有帮助的。通过回顾这个端到端示例，我们将熟悉每个组件提供的功能。当我们在本章的后面更详细地查看每个组件时，我们可以将特定组件在整体图景中的位置进行分类。

让我们首先将我们的日志数据放入 Logstash。我们将配置每个路由器将日志数据导出到 Logstash 服务器：

```py
r[1-6]#sh run | i logging
logging host <logstash ip> vrf Mgmt-intf transport udp port 5144 
```

在我们的 Elastic Stack 主机上，所有组件都已安装完毕后，我们将创建一个简单的 Logstash 配置，该配置监听 UDP 端口 `5144`，并将数据以 JSON 格式输出到控制台以及 Elasticsearch 主机：

```py
echou@elk-stack-mpn:~$ cd logstash-8.4.2/
echou@elk-stack-mpn:~/logstash-8.4.2$ mkdir network_configs
echou@elk-stack-mpn:~/logstash-8.4.2$ touch network_configs/simple_config.cfg
echou@elk-stack-mpn:~/logstash-8.4.2$ cat network_configs/simple_config.conf 
input {
  udp {
    port => 5144
    type => "syslog-ios"
  }
}
output {
  stdout { codec => json }
  elasticsearch {
    hosts => ["https://<elasticsearch ip>:9200"]
    ssl => true
    ssl_certificate_verification => false
    user => "elastic"
    password => "<password>"
    index => "cisco-syslog-%{+YYYY.MM.dd}"
  }
} 
```

配置文件仅包含一个输入部分和一个输出部分，不修改数据。类型 `syslog-ios` 是我们选择的用于识别此索引的名称。在 `output` 部分中，我们使用代表今天日期的变量配置索引名称。我们可以直接从二进制目录在前台运行 Logstash 进程：

```py
$ ./bin/logstash -f network_configs/simple_config.conf 
Using bundled JDK: /home/echou/Mastering_Python_Networking_Fourth_Edition/logstash-8.4.2/jdk
[2022-09-23T13:46:25,876][INFO ][logstash.inputs.udp      ][main][516c12046954cb8353b87ba93e5238d7964349b0fa7fa80339b72c6baca637bb] UDP listener started {:address=>"0.0.0.0:5144", :receive_buffer_bytes=>"106496", :queue_size=>"2000"} 
<skip> 
```

默认情况下，Elasticsearch 允许在向其发送数据时自动生成索引。我们可以在路由器上通过重置接口、重新加载 BGP 或简单地进入配置模式并退出来生成一些日志数据。一旦生成了一些新的日志，我们将看到创建的 `cisco-syslog-<date>` 索引：

```py
{"@timestamp":"2022-09-23T20:48:31.354Z", "log.level": "INFO", "message":"[cisco-syslog-2022.09.23/B7PH3hxNSHqAegikXyp9kg] create_mapping", "ecs.version": "1.2.0","service.name":"ES_ECS","event.dataset":"elasticsearch.server","process.thread.name":"elasticsearch[24808013b64b][masterService#updateTask][T#1]","log.logger":"org.elasticsearch.cluster.metadata.MetadataMappingService","elasticsearch.cluster.uuid":"c-j9Dg8YTh2PstO3JFP9AA","elasticsearch.node.id":"Pa4x3YJ-TrmFn5Pb2tObVw","elasticsearch.node.name":"24808013b64b","elasticsearch.cluster.name":"docker-cluster"} 
```

在这一点上，我们可以快速执行 `curl` 命令来查看在 Elasticsearch 上创建的索引。`curl` 命令使用 `insecure` 标志来适应自签名证书。URL 格式为“`https://<username>:<password>@<ip><port>/<path`>`。`"_cat/indices/cisco*"` 显示索引类别，然后匹配索引名称：

```py
$ curl -X GET --insecure "https://elastic:-Rel0twWMUk8L-ZtZr=I@192.168.2.126:9200/_cat/indices/cisco*"
yellow open cisco-syslog-2022.09.23 B7PH3hxNSHqAegikXyp9kg 1 1 9 0 21kb 21kb 
```

我们现在可以通过转到 **菜单 -> 管理 -> 堆栈管理** 来使用 Kibana 创建索引：

![图形用户界面，应用程序，团队  自动生成的描述](img/B18403_13_10.png)

图 13.10：堆栈管理

在 **数据 -> 索引管理** 下，我们可以看到新创建的 **cisco-syslog** 索引：

![图形用户界面，应用程序，团队  自动生成的描述](img/B18403_13_11.png)

图 13.11：索引管理

我们现在可以转到 **堆栈管理 -> Kibana -> 数据视图** 来创建数据视图。

![图形用户界面，文本，应用程序，电子邮件  自动生成的描述](img/B18403_13_12.png)

图 13.12：创建新数据视图步骤 1

由于索引已经在 Elasticsearch 中，我们只需匹配索引名称。请记住，我们的索引名称是基于时间的变量；我们可以使用星号通配符 (`*`) 来匹配所有以 **cisco-syslog** 开头的当前和未来索引：

![图形用户界面，文本，应用程序，电子邮件  自动生成的描述](img/B18403_13_13.png)

图 13.13：创建新数据视图步骤 2

我们的索引是基于时间的，也就是说，我们有一个可以用作时间戳的字段，并且我们可以根据时间进行搜索。我们应该指定我们指定的用作时间戳的字段。在我们的例子中，Elasticsearch 已经足够智能，可以从我们的 syslog 中选择一个字段作为时间戳；我们只需要在第二步中选择它从下拉菜单中。

在索引模式创建后，我们可以使用 **菜单 -> 发现**（在 **分析** 下）选项卡来查看条目。确保你选择了正确的索引和时间范围：

![图形用户界面，文本，应用程序  自动生成的描述](img/B18403_13_14.png)

图 13.14：Elasticsearch 索引文档发现

在我们收集了一些更多的日志信息之后，我们可以通过在 Logstash 进程上使用 *Ctrl + C* 来停止 Logstash 进程。这个第一个例子展示了我们如何利用 Elastic Stack 管道从数据摄取、存储和可视化。在 Logstash（或 Beats）中使用的数据摄取是一个连续的数据流，它会自动流入 Elasticsearch。Kibana 可视化工具为我们提供了一种更直观地分析 Elasticsearch 中的数据的方法，一旦我们对结果满意，就可以创建一个永久的可视化。我们可以使用 Kibana 创建更多的可视化图表，我们将在本章后面看到更多示例。

即使只有一个例子，我们也可以看到工作流程中最重要的部分是 Elasticsearch。它简单的 RESTful 接口、存储可伸缩性、自动索引和快速搜索结果赋予了堆栈适应我们网络分析需求的能力。

在下一节中，我们将探讨如何使用 Python 与 Elasticsearch 进行交互。

# 使用 Python 客户端连接到 Elasticsearch

我们可以通过 Python 库与 Elasticsearch 进行交互。例如，在下面的例子中，我们将使用 `requests` 库执行 `GET` 操作以从 Elasticsearch 主机检索信息。例如，我们知道以下 URL 端点的 `HTTP GET` 可以检索以 `kibana` 开头的当前索引：

```py
$ curl -X GET --insecure "https://elastic:-Rel0twWMUk8L-ZtZr=I@192.168.2.126:9200/_cat/indices/kibana*"
green open kibana_sample_data_ecommerce QcLgMu7CTEKNjeJeBxaD3w 1 0  4675 0 4.2mb 4.2mb
green open kibana_sample_data_logs      KPcJfMoSSaSs-kyqkuspKg 1 0 14074 0 8.1mb 8.1mb
green open kibana_sample_data_flights   q8MkYKooT8C5CQzbMMNTpg 1 0 13059 0 5.8mb 5.8mb 
```

我们可以使用 `requests` 库在 Python 脚本中创建类似的功能，`Chapter13_1.py`：

```py
#!/usr/bin/env python3
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
# disable https verification check warning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
def current_indices_list(es_host, index_prefix):
    current_indices = []
    http_header = {'content-type': 'application/json'}
    response = requests.get(es_host + "/_cat/indices/" + index_prefix + "*", headers=http_header, verify=False)
    for line in response.text.split('\n'):
        if line:
            current_indices.append(line.split()[2])
    return current_indices
if __name__ == "__main__":
    username = 'elastic'
    password = '-Rel0twWMUk8L-ZtZr=I'
    es_host = 'https://'+username+':'+password+'@192.168.2.126:9200'
    indices_list = current_indices_list(es_host, 'kibana')
    print(indices_list) 
```

执行脚本将给我们一个以 `kibana` 开头的索引列表：

```py
$ python Chapter13_1.py 
['kibana_sample_data_ecommerce', 'kibana_sample_data_logs', 'kibana_sample_data_flights'] 
```

我们还可以使用 Python Elasticsearch 客户端，[`elasticsearch-py.readthedocs.io/en/master/`](https://elasticsearch-py.readthedocs.io/en/master/)。它被设计为一个围绕 Elasticsearch RESTful API 的薄包装，以允许最大灵活性。让我们安装它并运行一个简单的示例：

```py
(venv) $ pip install elasticsearch 
```

示例 `Chapter13_2` 简单地连接到 Elasticsearch 集群，并搜索以 `kibana` 开头的索引：

```py
#!/usr/bin/env python3
from elasticsearch import Elasticsearch
es_host = Elasticsearch(["https://elastic:-Rel0twWMUk8L-ZtZr=I@192.168.2.126:9200/"],
                        ca_certs=False, verify_certs=False)
res = es_host.search(index="kibana*", body={"query": {"match_all": {}}})
print("Hits Total: " + str(res['hits']['total']['value'])) 
```

默认情况下，结果将返回前 10,000 条条目：

```py
$ python Chapter13_2.py
Hits Total: 10000 
```

使用简单的脚本，客户端库的优势并不明显。然而，当我们需要创建更复杂的搜索操作，例如需要使用每次查询返回的令牌来继续执行后续查询直到所有结果返回的滚动操作时，客户端库非常有帮助。客户端还可以帮助处理更复杂的行政任务，例如当我们需要重新索引现有索引时。我们将在本章剩余部分看到更多使用客户端库的示例。

在下一节中，我们将查看更多来自我们的 Cisco 设备 syslog 的数据摄取示例。

# 使用 Logstash 进行数据摄取

在最后一个示例中，我们使用 Logstash 从网络设备摄取日志数据。让我们在此基础上构建示例，并在`network_config/config_2.cfg`中添加一些额外的配置更改：

```py
input {
  udp {
    port => 5144
    type => "syslog-core"
  }
  udp {
    port => 5145
    type => "syslog-edge"
  }
}
filter {
 if [type] == "syslog-edge" {
    grok {
      match => { "message" => ".*" }
      add_field => [ "received_at", "%{@timestamp}" ]
    }
  } 
}
output {
  stdout { codec => json }
  elasticsearch {
    hosts => ["https://192.168.2.126:9200"]
    <skip>
  }
} 
```

在输入部分，我们将监听两个 UDP 端口，`5144`和`5145`。当接收到日志时，我们将使用`syslog-core`或`syslog-edge`对日志条目进行标记。我们还将向配置中添加一个过滤器部分，以特别匹配`syslog-edge`类型，并在消息部分应用正则表达式部分`Grok`。在这种情况下，我们将匹配所有内容，并添加一个额外的字段`received_at`，其值为时间戳。

关于 Grok 的更多信息，请参阅以下文档：[`www.elastic.co/guide/en/logstash/current/plugins-filters-grok.html`](https://www.elastic.co/guide/en/logstash/current/plugins-filters-grok.html).

我们将把`r5`和`r6`改为将 syslog 信息发送到 UDP 端口`5145`：

```py
r[5-6]#sh run | i logging
logging host 192.168.2.126 vrf Mgmt-intf transport udp port 5145 
```

当我们启动 Logstash 服务器时，我们会看到两个端口现在都在监听：

```py
$ ./bin/logstash -f network_configs/config_2.conf
<skip>
[2022-09-23T14:50:42,097][INFO ][logstash.inputs.udp      ][main][212f078853a453d3d8a5d8c1df268fd628577245cd1b66acb06b9e1cb1ff8a10] UDP listener started {:address=>"0.0.0.0:5144", :receive_buffer_bytes=>"106496", :queue_size=>"2000"}
[2022-09-23T14:50:42,106][INFO ][logstash.inputs.udp      ][main][6c3825527b168b167846f4ca7dea5ef55e1437753219866bdcc2eb51aee53c84] UDP listener started {:address=>"0.0.0.0:5145", :receive_buffer_bytes=>"106496", :queue_size=>"2000"} 
```

通过使用不同类型分离条目，我们可以在 Kibana **Discover**仪表板中特别搜索这些类型：

![图形用户界面，文本，应用程序，电子邮件  自动生成的描述](img/B18403_13_15.png)

图 13.15：Syslog 索引

如果我们扩展具有`syslog-edge`类型的条目，我们可以看到我们添加的新字段：

![图形用户界面，文本，应用程序  自动生成的描述](img/B18403_13_16.png)

图 13.16：Syslog 时间戳

Logstash 配置文件在输入、过滤和输出部分提供了许多选项。特别是，**过滤**部分提供了我们通过选择性匹配数据并在输出到 Elasticsearch 之前进一步处理数据来增强数据的方法。Logstash 可以通过模块进行扩展；每个模块都提供了一个快速端到端解决方案，用于摄取数据和可视化，并具有专门构建的仪表板。

关于 Logstash 模块的更多信息，请参阅以下文档：[`www.elastic.co/guide/en/logstash/8.4/logstash-modules.html`](https://www.elastic.co/guide/en/logstash/8.4/logstash-modules.html) .

Elastic Beats 与 Logstash 模块类似。它们是单一用途的数据传输工具，通常作为代理安装，用于收集主机上的数据并将输出数据直接发送到 Elasticsearch 或 Logstash 进行进一步处理。

有数百种不同的可下载 Beats，例如 Filebeat、Metricbeat、Packetbeat、Heartbeat 等。在下一节中，我们将看到如何使用 Filebeat 将 Syslog 数据摄取到 Elasticsearch 中。

# 使用 Beats 进行数据摄取

尽管 Logstash 功能强大，但数据摄取过程可能会变得复杂且难以扩展。如果我们以网络日志为例进行扩展，我们可以看到，即使只是网络日志，尝试解析来自 IOS 路由器、NXOS 路由器、ASA 防火墙、Meraki 无线控制器等不同日志格式也可能变得复杂。如果我们需要摄取来自 Apache 网络日志、服务器主机健康和安全信息的日志数据呢？至于像 NetFlow、SNMP 和计数器这样的数据格式呢？我们需要聚合的数据越多，情况可能就越复杂。

虽然我们无法完全摆脱聚合和数据摄取的复杂性，但当前的趋势是向更轻量级、单一用途的代理移动，该代理尽可能靠近数据源。例如，我们可以在我们的 Apache 服务器上安装专门收集网络日志数据的数据收集代理；或者我们可以有一个只收集、聚合和组织 Cisco IOS 日志的主机。Elastic Stack 将这些轻量级数据传输工具统称为 Beats：[`www.elastic.co/products/beats`](https://www.elastic.co/products/beats)。

Filebeat 是 Elastic Beats 软件的一个版本，旨在转发和集中日志数据。它寻找配置中指定的要收集的日志文件；一旦处理完成，它将新的日志数据发送到聚合事件并输出到 Elasticsearch 的底层进程。在本节中，我们将探讨如何使用 Filebeat 与 Cisco 模块收集网络日志数据。

让我们安装 Filebeat 并设置带有捆绑可视化模板和索引的 Elasticsearch 主机：

```py
$ $ curl -L -O https://artifacts.elastic.co/downloads/beats/filebeat/filebeat-8.4.2-amd64.deb
$ sudo dpkg -i filebeat-8.4.2-amd64.deb 
```

目录布局可能会让人困惑，因为它们被安装在不同的`/usr`、`/etc`和`/var`位置：

![图形用户界面，应用程序描述自动生成](img/B18403_13_17.png)

图 13.17：Elastic Filebeat 文件位置（来源：https://www.elastic.co/guide/en/beats/filebeat/8.4/directory-layout.html）

我们将对配置文件`/etc/filebeat/filebeat.yml`进行一些更改，以设置 Elasticsearch 和 Kibana 的位置：

```py
output.elasticsearch:
  # Array of hosts to connect to.
  hosts: ["192.168.2.126:9200"]
  # Protocol - either 'http' (default) or 'https'.
  protocol: "https"
  # Authentication credentials - either API key or username/password.
  username: "elastic"
  password: "changeme"
  ssl.verification_mode: none
setup.kibana:
  host: "192.168.2.126:5601" 
```

可以使用 Filebeat 来设置索引模板和示例 Kibana 仪表板：

```py
$ sudo filebeat setup --index-management -E output.logstash.enabled=false -E 'output.elasticsearch.hots=["https://elastic:-Rel0twWMUk8L-ZtZr=I@192.168.2.126:9200/"]'
$ sudo filebeat setup –dashboards 
```

让我们为 Filebeat 启用`cisco`模块：

```py
$ sudo filebeat modules enable cisco
Enabled cisco 
```

让我们先为`syslog`配置`cisco`模块。该文件位于`/etc/filebeat/modules.d/cisco.yml`下。在我们的例子中，我还指定了一个自定义的日志文件位置：

```py
- module: cisco
  ios:
    enabled: true
    var.input: syslog
    var.syslog_host: 0.0.0.0
    var.syslog_port: 514
    var.paths: ['/home/echou/syslog/my_log.log'] 
```

我们可以使用常见的 Ubuntu Linux 命令`service Filebeat`来启动、停止和检查 Filebeat 服务的状态`start`|`stop`|`status`：

```py
$ sudo service filebeat start
$ sudo service filebeat status
● filebeat.service - Filebeat sends log files to Logstash or directly to Elasticsearch.
     Loaded: loaded (/lib/systemd/system/filebeat.service; disabled; vendor preset: enabled)
     Active: active (running) since Fri 2022-09-23 16:06:09 PDT; 3s ago
<skip> 
```

在我们的设备上修改或添加 UDP 端口`514`用于 syslog。我们应该能够在**filebeat-***索引搜索下看到 syslog 信息：

![图形用户界面，文本，应用程序，电子邮件  自动生成的描述](img/B18403_13_18.png)

图 13.18：Elastic Filebeat 索引

如果我们将它与之前的 syslog 示例进行比较，我们可以看到与每条记录关联的字段和元信息要多得多，例如`agent.version`、`event.code`和`event.severity`：

![图形用户界面，应用程序  自动生成的描述](img/B18403_13_19.png)

图 13.19：Elastic Filebeat Cisco 日志

为什么额外的字段很重要？除了其他优点外，字段使得搜索聚合更加容易，而这反过来又使我们能够更好地绘制结果。我们将在下一节中看到绘制示例，其中我们将讨论 Kibana。

除了`cisco`模块外，还有针对 Palo Alto Networks、AWS、Google Cloud、MongoDB 等许多模块。最新的模块列表可以在[`www.elastic.co/guide/en/beats/filebeat/8.4/filebeat-modules.html`](https://www.elastic.co/guide/en/beats/filebeat/8.4/filebeat-modules.html)查看。

如果我们想监控 NetFlow 数据呢？没问题，有一个模块可以做到这一点！我们将通过启用模块并设置仪表板来运行与 Cisco 模块相同的流程：

```py
$ sudo filebeat modules enable netflow
$ sudo filebeat setup -e 
```

然后，配置模块配置文件，`/etc/filebeat/modules.d/netflow.yml`：

```py
- module: netflow
  log:
    enabled: true
    var:
      netflow_host: 0.0.0.0
      netflow_port: 2055 
```

我们将配置设备将 NetFlow 数据发送到端口`2055`。如果您需要复习，请阅读*第八章*，*使用 Python 进行网络监控 – 第二部分*中的相关配置。我们应该能够看到新的`netflow`数据输入类型：

![图形用户界面，文本，网站  自动生成的描述](img/B18403_13_20.png)

图 13.20：Elastic NetFlow 输入

记住每个模块都预装了可视化模板吗？不要过多地跳入可视化，但如果我们点击左侧面板上的**可视化**选项卡，然后搜索**netflow**，我们可以看到为我们创建的一些可视化：

![](img/B18403_13_21.png)

图 13.21：Kibana 可视化

点击**会话伙伴 [Filebeat Netflow]**选项，这将给我们一个很好的表格，我们可以通过每个字段重新排序顶级对话者：

![表格  自动生成的描述](img/B18403_13_22.png)

图 13.22：Kibana 表格

在下一节中，我们将专注于 ELK 堆栈的 Elasticsearch 部分。

# 使用 Elasticsearch 进行搜索

我们需要在 Elasticsearch 中添加更多数据，以便使搜索和图形更加有趣。我建议重新加载几个实验室设备，以便有接口重置、BGP 和 OSPF 建立以及设备启动消息的日志条目。否则，请随意使用本章开头导入的样本数据来处理本节。

如果我们回顾一下`Chapter13_2.py`脚本示例，当我们搜索时，有两项信息可能会因每个查询而异：索引和查询体。我通常喜欢将这些信息分解为可以在运行时动态更改的输入变量，以分离搜索逻辑和脚本本身。让我们创建一个名为`query_body_1.json`的文件：

```py
{
  "query": {
    "match_all": {}
  }
} 
```

我们将创建一个名为`Chapter13_3.py`的脚本，该脚本使用`argparse`在命令行接收用户输入：

```py
import argparse
parser = argparse.ArgumentParser(description='Elasticsearch Query Options')
parser.add_argument("-i", "--index", help="index to query")
parser.add_argument("-q", "--query", help="query file")
args = parser.parse_args() 
```

我们可以使用这两个输入值以与我们之前相同的方式构建搜索：

```py
# load elastic index and query body information
query_file = args.query
with open(query_file) as f:
    query_body = json.loads(f.read())
# Elasticsearch instance
es_host = Elasticsearch(["https://elastic:<pass> @192.168.2.126:9200/"],
              ca_certs=False, verify_certs=False) 
# Query both index and put into dictionary
index = args.index
res = es.search(index=index, body=query_body)
print(res['hits']['total']['value']) 
```

我们可以使用`help`选项查看应该与脚本一起提供的参数。以下是当我们使用相同的查询针对我们创建的两个不同索引时的结果：

```py
$ python Chapter13_3.py --help
usage: Chapter12_3.py [-h] [-i INDEX] [-q QUERY]
Elasticsearch Query Options
optional arguments:
  -h, --help            show this help message and exit
  -i INDEX, --index INDEX
                        index to query
  -q QUERY, --query QUERY
                        query file
$ python3 Chapter13_3.py -q query_body_1.json -i "cisco*"
50
$ python3 Chapter13_3.py -q query_body_1.json -i "filebeat*"
10000 
```

在开发我们的搜索时，通常需要尝试几次才能得到我们想要的结果。Kibana 提供的一个工具是开发者控制台，它允许我们在同一页面上玩转搜索条件并查看搜索结果。该工具位于菜单部分*Management for Dev Tools*。

例如，在下面的图中，我们执行了我们现在所做的相同搜索，并能够看到返回的 JSON 结果。这是我在 Kibana 界面上的最爱工具之一：

![图形用户界面，文本描述自动生成](img/B18403_13_23.png)

图 13.23：Kibana Dev Tools

大多数网络数据都是基于时间的，例如我们收集的日志和 NetFlow 数据。这些值是在某个时间点拍摄的快照，我们可能会在时间范围内分组这些值。例如，我们可能想知道，“在过去的 7 天里，谁是 NetFlow 顶级通信者？”或者“在过去 1 小时内，哪个设备有最多的 BGP 重置消息？”这些问题大多与聚合和时间范围有关。让我们看看一个限制时间范围的查询，`query_body_2.json`：

```py
{
  "query": {
    "bool": {
    "filter": [
      {
        "range": {
          "@timestamp": {
            "gte": "now-10m"
          }
        }
      }
    ]
    }
  }
} 
```

这是一个布尔查询，[`www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html`](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html)，这意味着它可以组合其他查询。在我们的查询中，我们使用过滤器将时间范围限制为过去 10 分钟。我们将`Chapter13_3.py`脚本复制到`Chapter13_4.py`，并修改输出以获取命中次数以及遍历实际返回的结果列表：

```py
<skip>
res = es.search(index=index, body=query_body)
print("Total hits: " + str(res['hits']['total']['value']))
for hit in res['hits']['hits']:
    pprint(hit) 
```

执行脚本将显示我们在过去 10 分钟内只有`23`次命中：

```py
$ python Chapter13_4.py -i "filebeat*" -q query_body_2.json
Total hits: 23 
```

我们可以在查询中添加另一个过滤器选项，通过`query_body_3.json`限制源 IP：

```py
{
  "query": {
    "bool": {
      "must": {
        "term": {
          "source.ip": "192.168.0.1"
        }
      },
<skip> 
```

结果将受到 r1 的 loopback IP 在过去 10 分钟内的源 IP 的限制：

```py
$ python Chapter12_4.py -i "filebeat*" -q query_body_3.json
Total hits: 18 
```

让我们再次修改搜索主体，添加一个聚合，[`www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-bucket.html`](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-bucket.html)，它从我们之前的搜索中计算所有网络字节的总和：

```py
{
  "aggs": {
        "network_bytes_sum": {
          "sum": {
            "field": "network.bytes"
        }
    }
  },
  <skip>
} 
```

每次运行脚本 `Chapter13_5.py` 时，结果都会不同。当我连续运行脚本时，当前结果大约为 1 MB：

```py
$ python Chapter13_5.py -i "filebeat*" -q query_body_4.json
1089.0
$ python Chapter13_5.py -i "filebeat*" -q query_body_4.json
990.0 
```

如您所见，构建搜索查询是一个迭代的过程；您通常从一个广泛的网络开始，并逐渐缩小标准以微调结果。一开始，您可能会花很多时间阅读文档并搜索确切的语法和过滤器。随着您经验的积累，搜索语法将变得更加容易。回到我们从 NetFlow 模块设置中看到的上一个可视化，我们可以使用检查工具来查看 **请求** 主体：

![图形用户界面，文本，应用程序，自动生成的描述](img/B18403_13_24.png)

图 13.24：Kibana 请求

我们可以将它放入查询 JSON 文件 `query_body_5.json` 中，并使用 `Chapter13_6.py` 文件执行它。我们将收到图表基于的原始数据：

```py
$ python Chapter13_6.py -i "filebeat*" -q query_body_5.json
{'1': {'value': 8156040.0}, 'doc_count': 8256, 'key': '10.0.0.5'}
{'1': {'value': 4747596.0}, 'doc_count': 103, 'key': '172.16.1.124'}
{'1': {'value': 3290688.0}, 'doc_count': 8256, 'key': '10.0.0.9'}
{'1': {'value': 576446.0}, 'doc_count': 8302, 'key': '192.168.0.2'}
{'1': {'value': 576213.0}, 'doc_count': 8197, 'key': '192.168.0.1'}
{'1': {'value': 575332.0}, 'doc_count': 8216, 'key': '192.168.0.3'}
{'1': {'value': 433260.0}, 'doc_count': 6547, 'key': '192.168.0.5'}
{'1': {'value': 431820.0}, 'doc_count': 6436, 'key': '192.168.0.4'} 
```

在下一节中，让我们更深入地了解 Elastic Stack 的可视化部分：Kibana。

# 使用 Kibana 进行数据可视化

到目前为止，我们已经使用 Kibana 来发现数据，管理 Elasticsearch 中的索引，使用开发者工具来开发查询，以及使用一些其他功能。我们还看到了来自 NetFlow 的预填充可视化图表，它为我们提供了数据中的顶级对话对。在本节中，我们将逐步介绍创建我们自己的图表的步骤。我们将从创建一个饼图开始。

饼图非常适合可视化组件的一部分相对于整体的部分。让我们基于 Filebeat 索引创建一个饼图，根据记录计数绘制前 10 个源 IP 地址。我们将选择 **仪表板 -> 创建仪表板 -> 创建可视化 -> 饼图**：

![图形用户界面，应用程序，自动生成的描述](img/B18403_13_25.png)

图 13.25：Kibana 饼图

然后，我们在搜索栏中输入 **netflow** 以选择我们的 **[Filebeat NetFlow]** 索引：

![图形用户界面，应用程序，团队，自动生成的描述](img/B18403_13_26.png)

图 13.26：Kibana 饼图来源

默认情况下，我们得到了默认时间范围内所有记录的总数。时间范围可以动态更改：

![](img/B18403_13_27.png)

图 13.27：Kibana 时间范围

我们可以为图表分配一个自定义标签：

![图标，自动生成的描述](img/B18403_13_28.png)

图 13.28：Kibana 图表标签

让我们点击**添加**选项来添加更多桶。我们将选择分割切片，选择聚合的术语，并从下拉菜单中选择**source.ip**字段。我们将保持**顺序**为**降序**，但将**大小**增加到**10**。

只有当你点击顶部**应用**按钮时，更改才会生效。在使用现代网站时，期望更改实时发生而不是通过点击**应用**按钮是一种常见的错误：

![图形用户界面，应用程序  自动生成描述](img/B18403_13_29.png)

图 13.29：Kibana 播放按钮

我们可以点击顶部的**选项**来关闭**环形图**并打开**显示标签**：

![图形用户界面，应用程序，团队  自动生成描述](img/B18403_13_30.png)

图 13.30：Kibana 图表选项

最终的图表是一个展示基于文档计数数量的顶级 IP 来源的饼图：

![图表，饼图  自动生成描述](img/B18403_13_31.png)

图 13.31：Kibana 饼图

与 Elasticsearch 一样，Kibana 的图表也是一个迭代过程，通常需要尝试几次才能正确设置。如果我们把结果分成不同的图表而不是同一图表上的切片呢？是的，这并不非常直观：

![图形用户界面，应用程序，团队  自动生成描述](img/B18403_13_32.png)

图 13.32：Kibana 分割图表

让我们坚持在同一饼图上分割切片，并将时间范围更改为**过去 1 小时**，然后保存图表以便稍后返回：

注意，我们还可以通过嵌入的 URL（如果 Kibana 可以从共享位置访问）或快照来共享图表：

![图表，饼图  自动生成描述](img/B18403_13_33.png)

图 13.33：Kibana 保存图表

我们还可以使用度量操作做更多的事情。例如，我们可以选择数据表图表类型，并使用源 IP 重复之前的桶分解。但我们还可以通过添加每个桶的网络字节总数来添加第二个度量：

![图形用户界面，应用程序  自动生成描述](img/B18403_13_34.png)

图 13.34：Kibana 度量

结果是一个表格，显示了文档计数数量以及网络字节的总和。这可以以 CSV 格式下载以供本地存储：

![图形用户界面，表格  中度自信自动生成](img/B18403_13_35.png)

图 13.35：Kibana 表格

Kibana 是 Elastic Stack 中一个非常强大的可视化工具。我们只是触及了其可视化功能的表面。除了许多其他图表选项，以更好地讲述您数据的故事外，我们还可以将多个可视化组合到一个仪表板上进行展示。我们还可以使用 Timelion ([`www.elastic.co/guide/en/kibana/8.4/timelion.html`](https://www.elastic.co/guide/en/kibana/8.4/timelion.html)) 将独立的数据源组合成单个可视化，或者使用 Canvas ([`www.elastic.co/guide/en/kibana/current/canvas.html`](https://www.elastic.co/guide/en/kibana/current/canvas.html)) 作为基于 Elasticsearch 数据的演示工具。

Kibana 通常用于工作流程的末端，以有意义的方式展示我们的数据。我们在本章范围内涵盖了从数据摄取到存储、检索和可视化的基本工作流程。令人惊讶的是，借助如 Elastic Stack 这样的集成开源堆栈，我们可以在短时间内完成这么多工作。

# 摘要

在本章中，我们使用了 Elastic Stack 来摄取、分析和可视化网络数据。我们使用了 Logstash 和 Beats 来摄取网络系统日志和 NetFlow 数据。然后我们使用 Elasticsearch 对数据进行索引和分类，以便于检索。最后，我们使用 Kibana 来可视化数据。我们使用 Python 与该堆栈交互，帮助我们更深入地了解我们的数据。Logstash、Beats、Elasticsearch 和 Kibana 一起构成了一个强大的全能型项目，可以帮助我们更好地理解我们的数据。

在下一章中，我们将探讨如何使用 Python 结合 Git 进行网络开发。

# 加入我们的书籍社区

要加入这本书的社区——在那里您可以分享反馈、向作者提问，并了解新书发布——请扫描下面的二维码：

[`packt.link/networkautomationcommunity`](https://packt.link/networkautomationcommunity)

![](img/QR_Code2903617220506617062.png)
