# 使用 Python 进行网络监控-第 2 部分

在第七章中，*使用 Python 进行网络监控-第 1 部分*，我们使用 SNMP 从网络设备查询信息。我们通过使用 SNMP 管理器查询驻留在网络设备上的 SNMP 代理来实现这一点。SNMP 信息以层次结构格式化，具有特定的对象 ID 来表示对象的值。大多数时候，我们关心的值是一个数字，比如 CPU 负载、内存使用率或接口流量。这是我们可以根据时间绘制图表，以便让我们了解值随时间的变化。

我们通常可以将 SNMP 方法归类为“拉”方法，因为我们不断地向设备请求特定的答案。这种方法会给设备增加负担，因为它需要在控制平面上花费 CPU 周期从子系统中找到答案，将答案打包成一个 SNMP 数据包，并将答案传输回轮询器。如果你曾经参加过家庭聚会，有一个家庭成员一遍又一遍地问你同样的问题，那就相当于 SNMP 管理器不断轮询受管节点。

随着时间的推移，如果我们有多个 SNMP 轮询器每 30 秒查询同一个设备（你会惊讶地发现这种情况经常发生），管理开销将变得相当大。在我们给出的家庭聚会的例子中，想象一下不是一个家庭成员，而是许多其他人每 30 秒打断你问你一个问题。我不知道你怎么想，但我知道即使是一个简单的问题（或者更糟糕的是，如果所有人都问同样的问题），我也会感到非常恼火。

我们可以提供更有效的网络监控的另一种方法是将管理站与设备之间的关系从拉模型转变为推模型。换句话说，信息可以以约定的格式从设备推送到管理站。这个概念是基于基于流的监控。在基于流的模型中，网络设备将流量信息流向管理站。格式可以是思科专有的 NetFlow（版本 5 或版本 9），行业标准 IPFIX，或开源 sFlow 格式。在本章中，我们将花一些时间用 Python 来研究 NetFlow、IPFIX 和 sFlow。

并非所有的监控都以时间序列数据的形式出现。如果你真的愿意，你可以将网络拓扑和 Syslog 等信息表示为时间序列格式，但这并不理想。我们可以使用 Python 来检查网络拓扑信息，并查看拓扑是否随时间发生了变化。我们可以使用 Graphviz 等工具与 Python 包装器来说明拓扑。正如在第六章中已经看到的，*使用 Python 进行网络安全*，Syslog 包含安全信息。在本章中，我们将研究使用 ELK 堆栈（Elasticsearch、Logstash、Kibana）作为收集和索引网络日志信息的有效方法。

具体来说，在本章中，我们将涵盖以下主题：

+   Graphviz，这是一个开源的图形可视化软件，可以帮助我们快速高效地绘制网络图

+   基于流的监控，如 NetFlow、IPFIX 和 sFlow

+   使用 ntop 来可视化流量信息

+   使用 Elasticsearch 来索引和分析我们收集的数据

让我们首先看看如何使用 Graphviz 作为监控网络拓扑变化的工具。

# Graphviz

Graphviz 是一种开源的图形可视化软件。想象一下，如果我们不用图片的好处来描述我们的网络拓扑给同事。我们可能会说，我们的网络由三层组成：核心、分发和接入。核心层包括两台路由器用于冗余，并且这两台路由器都对四台分发路由器进行全网状连接；分发路由器也对接入路由器进行全网状连接。内部路由协议是 OSPF，外部使用 BGP 与服务提供商进行对等连接。虽然这个描述缺少一些细节，但对于您的同事来说，这可能足够绘制出您网络的一个相当不错的高层图像。

Graphviz 的工作方式类似于通过描述 Graphviz 可以理解的文本格式来描述图形，然后我们可以将文件提供给 Graphviz 程序来为我们构建图形。在这里，图形是用一种称为 DOT 的文本格式描述的（[`en.wikipedia.org/wiki/DOT_(graph_description_language)`](https://en.wikipedia.org/wiki/DOT_(graph_description_language)）），Graphviz 根据描述渲染图形。当然，因为计算机缺乏人类的想象力，语言必须非常精确和详细。

对于 Graphviz 特定的 DOT 语法定义，请查看[`www.graphviz.org/doc/info/lang.html`](http://www.graphviz.org/doc/info/lang.html)。

在本节中，我们将使用**链路层发现协议**（**LLDP**）来查询设备邻居，并通过 Graphviz 创建网络拓扑图。完成这个广泛的示例后，我们将看到如何将新的东西，比如 Graphviz，与我们已经学到的东西结合起来解决有趣的问题。

让我们开始构建我们将要使用的实验室。

# 实验室设置

我们将使用 VIRL 来构建我们的实验室。与前几章一样，我们将组建一个包括多个路由器、一个服务器和一个客户端的实验室。我们将使用五个 IOSv 网络节点以及两个服务器主机：

![](img/3166a522-47db-4a53-bd7f-0e15ad415b04.png)

如果您想知道我们选择 IOSv 而不是 NX-OS 或 IOS-XR 以及设备数量的原因，在构建自己的实验室时，请考虑以下几点：

+   由 NX-OS 和 IOS-XR 虚拟化的节点比 IOS 更占用内存

+   我使用的 VIRL 虚拟管理器有 8GB 的 RAM，似乎足够支持九个节点，但可能会有点不稳定（节点随机从可达到不可达）

+   如果您希望使用 NX-OS，请考虑使用 NX-API 或其他 API 调用来返回结构化数据

对于我们的示例，我们将使用 LLDP 作为链路层邻居发现的协议，因为它是与厂商无关的。请注意，VIRL 提供了自动启用 CDP 的选项，这可以节省一些时间，并且在功能上类似于 LLDP；但是，它是一种思科专有技术，因此我们将在我们的实验室中禁用它：

![](img/7ffca5b5-9a64-4fdd-b214-e5af0d08b772.png)

实验室建立完成后，继续安装必要的软件包。

# 安装

可以通过`apt`获取 Graphviz：

```py
$ sudo apt-get -y install graphviz
```

安装完成后，请注意使用`dot`命令进行验证：

```py
$ dot -V
dot - graphviz version 2.38.0 (20140413.2041)~
```

我们将使用 Graphviz 的 Python 包装器，所以让我们现在安装它：

```py
$ sudo pip install graphviz #Python 2
$ sudo pip3 install graphviz

$ python3
Python 3.5.2 (default, Nov 23 2017, 16:37:01)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import graphviz
>>> graphviz.__version__
'0.8.4'
>>> exit() 
```

让我们看看如何使用这个软件。

# Graphviz 示例

像大多数流行的开源项目一样，Graphviz 的文档（[`www.graphviz.org/Documentation.php`](http://www.graphviz.org/Documentation.php)）是非常广泛的。对于新手来说，挑战通常在于从何处开始。对于我们的目的，我们将专注于绘制有向图的 dot 图，这是一种层次结构（不要与 DOT 语言混淆，DOT 语言是一种图描述语言）。

让我们从一些基本概念开始：

+   节点代表我们的网络实体，如路由器、交换机和服务器

+   边缘代表网络实体之间的链接

+   图表、节点和边都有可以调整的属性([`www.graphviz.org/doc/info/attrs.html`](https://www.graphviz.org/doc/info/attrs.html))

+   描述网络后，我们可以将网络图([`www.graphviz.org/doc/info/output.html`](https://www.graphviz.org/doc/info/output.html))输出为 PNG、JPEG 或 PDF 格式

我们的第一个例子是一个无向点图，由四个节点(`core`、`distribution`、`access1`和`access2`)组成。边由破折号`-`符号表示，将核心节点连接到分布节点，以及将分布节点连接到两个访问节点：

```py
$ cat chapter8_gv_1.gv
graph my_network {
 core -- distribution;
 distribution -- access1;
 distribution -- access2;
}
```

图表可以在命令行中输出为`dot -T<format> source -o <output file>`：

```py
$ dot -Tpng chapter8_gv_1.gv -o output/chapter8_gv_1.png
```

生成的图表可以从以下输出文件夹中查看：

就像第七章中的*使用 Python 进行网络监控-第 1 部分*一样，当处理这些图表时，可能更容易在 Linux 桌面窗口中工作，这样你就可以立即看到图表。

请注意，我们可以通过将图表指定为有向图，并使用箭头(`->`)符号来表示边来使用有向图。在节点和边的情况下，有几个属性可以修改，例如节点形状、边标签等。同一个图表可以修改如下：

```py
$ cat chapter8_gv_2.gv
digraph my_network {
 node [shape=box];
 size = "50 30";
 core -> distribution [label="2x10G"];
 distribution -> access1 [label="1G"];
 distribution -> access2 [label="1G"];
}
```

这次我们将文件输出为 PDF：

```py
$ dot -Tpdf chapter8_gv_2.gv -o output/chapter8_gv_2.pdf
```

看一下新图表中的方向箭头：

![](img/9856fec1-2710-4661-8be9-99f9a65151de.png)

现在让我们看一下围绕 Graphviz 的 Python 包装器。

# Python 与 Graphviz 示例

我们可以使用我们安装的 Python Graphviz 包再次生成与之前相同的拓扑图：

```py
$ python3
Python 3.5.2 (default, Nov 17 2016, 17:05:23)
>>> from graphviz import Digraph
>>> my_graph = Digraph(comment="My Network")
>>> my_graph.node("core")
>>> my_graph.node("distribution")
>>> my_graph.node("access1")
>>> my_graph.node("access2")
>>> my_graph.edge("core", "distribution")
>>> my_graph.edge("distribution", "access1")
>>> my_graph.edge("distribution", "access2")
```

该代码基本上产生了您通常会用 DOT 语言编写的内容，但以更 Pythonic 的方式。您可以在生成图表之前查看图表的源代码：

```py
>>> print(my_graph.source)
// My Network
digraph {
 core
 distribution
 access1
 access2
 core -> distribution
 distribution -> access1
 distribution -> access2
} 
```

图表可以通过`render()`方法呈现；默认情况下，输出格式为 PDF：

```py
>>> my_graph.render("output/chapter8_gv_3.gv")
'output/chapter8_gv_3.gv.pdf'
```

Python 包装器紧密模仿了 Graphviz 的所有 API 选项。您可以在 Graphviz Read the Docs 网站([`graphviz.readthedocs.io/en/latest/index.html`](http://graphviz.readthedocs.io/en/latest/index.html))上找到有关选项的文档。您还可以在 GitHub 上查看源代码以获取更多信息([`github.com/xflr6/graphviz`](https://github.com/xflr6/graphviz))。我们现在准备使用这个工具来绘制我们的网络。

# LLDP 邻居图

在本节中，我们将使用映射 LLDP 邻居的示例来说明多年来帮助我的问题解决模式：

1.  如果可能的话，将每个任务模块化为更小的部分。在我们的例子中，我们可以合并几个步骤，但如果我们将它们分解成更小的部分，我们将能够更容易地重用和改进它们。

1.  使用自动化工具与网络设备交互，但将更复杂的逻辑保留在管理站。例如，路由器提供了一个有点混乱的 LLDP 邻居输出。在这种情况下，我们将坚持使用可行的命令和输出，并在管理站使用 Python 脚本来解析我们需要的输出。

1.  在面对相同任务的选择时，选择可以重复使用的选项。在我们的例子中，我们可以使用低级别的 Pexpect、Paramiko 或 Ansible playbooks 来查询路由器。在我看来，Ansible 是一个更可重用的选项，所以我选择了它。

要开始，因为路由器默认情况下未启用 LLDP，我们需要首先在设备上配置它们。到目前为止，我们知道我们有许多选择；在这种情况下，我选择了使用`ios_config`模块的 Ansible playbook 来完成任务。`hosts`文件包括五台路由器：

```py
$ cat hosts
[devices]
r1 ansible_hostname=172.16.1.218
r2 ansible_hostname=172.16.1.219
r3 ansible_hostname=172.16.1.220
r5-tor ansible_hostname=172.16.1.221
r6-edge ansible_hostname=172.16.1.222
```

`cisco_config_lldp.yml` playbook 包括一个 play，其中嵌入了用于配置 LLDP 的变量：

```py
<skip>
 vars:
   cli:
     host: "{{ ansible_hostname }}"
     username: cisco
     password: cisco
     transport: cli tasks:
  - name: enable LLDP run
       ios_config:
         lines: lldp run
         provider: "{{ cli }}"
<skip>
```

几秒钟后，为了允许 LLDP 交换，我们可以验证 LLDP 确实在路由器上处于活动状态：

```py
$ ansible-playbook -i hosts cisco_config_lldp.yml

PLAY [Enable LLDP] ***********************************************************
...
PLAY RECAP *********************************************************************
r1 : ok=2 changed=1 unreachable=0 failed=0
r2 : ok=2 changed=1 unreachable=0 failed=0
r3 : ok=2 changed=1 unreachable=0 failed=0
r5-tor : ok=2 changed=1 unreachable=0 failed=0
r6-edge : ok=2 changed=1 unreachable=0 failed=0

## SSH to R1 for verification
r1#show lldp neighbors

Capability codes: (R) Router, (B) Bridge, (T) Telephone, (C) DOCSIS Cable Device (W) WLAN Access Point, (P) Repeater, (S) Station, (O) Other

Device ID Local Intf Hold-time Capability Port ID
r2.virl.info Gi0/0 120 R Gi0/0
r3.virl.info Gi0/0 120 R Gi0/0
r5-tor.virl.info Gi0/0 120 R Gi0/0
r5-tor.virl.info Gi0/1 120 R Gi0/1
r6-edge.virl.info Gi0/2 120 R Gi0/1
r6-edge.virl.info Gi0/0 120 R Gi0/0

Total entries displayed: 6
```

在输出中，您将看到`G0/0`配置为 MGMT 接口；因此，您将看到 LLDP 对等方，就好像它们在一个平坦的管理网络上一样。我们真正关心的是连接到其他对等方的`G0/1`和`G0/2`接口。当我们准备解析输出并构建我们的拓扑图时，这些知识将派上用场。

# 信息检索

我们现在可以使用另一个 Ansible playbook，即`cisco_discover_lldp.yml`，在设备上执行 LLDP 命令，并将每个设备的输出复制到`tmp`目录中：

```py
<skip>
 tasks:
   - name: Query for LLDP Neighbors
     ios_command:
       commands: show lldp neighbors
       provider: "{{ cli }}"
<skip>
```

./tmp 目录现在包含所有路由器的输出（显示 LLDP 邻居）的文件：

```py
$ ls -l tmp/
total 20
-rw-rw-r-- 1 echou echou 630 Mar 13 17:12 r1_lldp_output.txt
-rw-rw-r-- 1 echou echou 630 Mar 13 17:12 r2_lldp_output.txt
-rw-rw-r-- 1 echou echou 701 Mar 12 12:28 r3_lldp_output.txt
-rw-rw-r-- 1 echou echou 772 Mar 12 12:28 r5-tor_lldp_output.txt
-rw-rw-r-- 1 echou echou 630 Mar 13 17:12 r6-edge_lldp_output.txt
```

`r1_lldp_output.txt`的内容是我们 Ansible playbook 中的`output.stdout_lines`变量：

```py
$ cat tmp/r1_lldp_output.txt

[["Capability codes:", " (R) Router, (B) Bridge, (T) Telephone, (C) DOCSIS Cable Device", " (W) WLAN Access Point, (P) Repeater, (S) Station, (O) Other", "", "Device ID Local Intf Hold-time Capability Port ID", "r2.virl.info Gi0/0 120 R Gi0/0", "r3.virl.info Gi0/0 120 R Gi0/0", "r5-tor.virl.info Gi0/0 120 R Gi0/0", "r5-tor.virl.info Gi0/1 120 R Gi0/1", "r6-edge.virl.info Gi0/0 120 R Gi0/0", "", "Total entries displayed: 5", ""]]
```

# Python 解析脚本

我们现在可以使用 Python 脚本解析每个设备的 LLDP 邻居输出，并从结果构建网络拓扑图。目的是自动检查设备，看看 LLDP 邻居是否由于链路故障或其他问题而消失。让我们看看`cisco_graph_lldp.py`文件，看看是如何做到的。

我们从包的必要导入开始：一个空列表，我们将用节点关系的元组填充它。我们也知道设备上的`Gi0/0`连接到管理网络；因此，我们只在`show LLDP neighbors`输出中搜索`Gi0/[1234]`作为我们的正则表达式模式：

```py
import glob, re
from graphviz import Digraph, Source
pattern = re.compile('Gi0/[1234]')
device_lldp_neighbors = []
```

我们将使用`glob.glob()`方法遍历`./tmp`目录中的所有文件，解析出设备名称，并找到设备连接的邻居。脚本中有一些嵌入的打印语句，我们可以在最终版本中注释掉；如果取消注释，我们可以看到解析的结果：

```py
device: r1
 neighbors: r5-tor
 neighbors: r6-edge
device: r5-tor
 neighbors: r2
 neighbors: r3
 neighbors: r1
device: r2
 neighbors: r5-tor
 neighbors: r6-edge
device: r3
 neighbors: r5-tor
 neighbors: r6-edge
device: r6-edge
 neighbors: r2
 neighbors: r3
 neighbors: r1
```

完全填充的边列表包含了由设备及其邻居组成的元组：

```py
Edges: [('r1', 'r5-tor'), ('r1', 'r6-edge'), ('r5-tor', 'r2'), ('r5-tor', 'r3'), ('r5-tor', 'r1'), ('r2', 'r5-tor'), ('r2', 'r6-edge'), ('r3', 'r5-tor'), ('r3', 'r6-edge'), ('r6-edge', 'r2'), ('r6-edge', 'r3'), ('r6-edge', 'r1')]
```

我们现在可以使用 Graphviz 包构建网络拓扑图。最重要的部分是解压代表边关系的元组：

```py
my_graph = Digraph("My_Network")
<skip>
# construct the edge relationships
for neighbors in device_lldp_neighbors:
    node1, node2 = neighbors
    my_graph.edge(node1, node2)
```

如果我们打印出结果的源 dot 文件，它将是我们网络的准确表示：

```py
digraph My_Network {
   r1 -> "r5-tor"
   r1 -> "r6-edge"
   "r5-tor" -> r2
   "r5-tor" -> r3
   "r5-tor" -> r1
   r2 -> "r5-tor"
   r2 -> "r6-edge"
   r3 -> "r5-tor"
   r3 -> "r6-edge"
   "r6-edge" -> r2
   "r6-edge" -> r3
   "r6-edge" -> r1
}
```

有时，看到相同的链接两次会让人困惑；例如，`r2`到`r5-tor`的链接在上一个图表中每个方向都出现了两次。作为网络工程师，我们知道有时物理链接故障会导致单向链接，我们希望看到这种情况。

如果我们按原样绘制图表，节点的放置会有点奇怪。节点的放置是自动渲染的。以下图表说明了默认布局以及`neato`布局的渲染，即有向图（`My_Network`，`engine='neato'`）：

![](img/dde9bc1b-9f98-4da2-ac4e-4bea01181aa1.png)

`neato`布局表示尝试绘制更少层次结构的无向图：

![](img/54d47a85-be7a-4294-acf5-056fae7ad784.png)

有时，工具提供的默认布局就很好，特别是如果你的目标是检测故障而不是使其视觉上吸引人。然而，在这种情况下，让我们看看如何将原始 DOT 语言旋钮插入源文件。通过研究，我们知道可以使用`rank`命令指定一些节点可以保持在同一级别。然而，在 Graphviz Python API 中没有提供这个选项。幸运的是，dot 源文件只是一个字符串，我们可以使用`replace()`方法插入原始 dot 注释，如下所示：

```py
source = my_graph.source
original_text = "digraph My_Network {"
new_text = 'digraph My_Network {n{rank=same Client "r6-edge"}n{rank=same r1 r2 r3}n'
new_source = source.replace(original_text, new_text)
new_graph = Source(new_source)new_graph.render("output/chapter8_lldp_graph.gv")
```

最终结果是一个新的源文件，我们可以从中渲染最终的拓扑图：

```py
digraph My_Network {
{rank=same Client "r6-edge"}
{rank=same r1 r2 r3}
                Client -> "r6-edge"
                "r5-tor" -> Server
                r1 -> "r5-tor"
                r1 -> "r6-edge"
                "r5-tor" -> r2
                "r5-tor" -> r3
                "r5-tor" -> r1
                r2 -> "r5-tor"
                r2 -> "r6-edge"
                r3 -> "r5-tor"
                r3 -> "r6-edge"
               "r6-edge" -> r2
               "r6-edge" -> r3
               "r6-edge" -> r1
}
```

图现在可以使用了：

![](img/b0444bef-47f9-44c4-a33c-e9b8aca1caee.png)

# 最终 playbook

我们现在准备将这个新的解析脚本重新整合到我们的 playbook 中。我们现在可以添加渲染输出和图形生成的额外任务到`cisco_discover_lldp.yml`中：

```py
  tasks:
    - name: Query for LLDP Neighbors
      ios_command:
        commands: show lldp neighbors
        provider: "{{ cli }}"

      register: output

    - name: show output
      debug:
        var: output

    - name: copy output to file
      copy: content="{{ output.stdout_lines }}" dest="./tmp/{{ inventory_hostname }}_lldp_output.txt"

    - name: Execute Python script to render output
      command: ./cisco_graph_lldp.py
```

这本 playbook 现在将包括四个任务，涵盖了在 Cisco 设备上执行`show lldp`命令的端到端过程，将输出显示在屏幕上，将输出复制到单独的文件，然后通过 Python 脚本呈现输出。

playbook 现在可以通过`cron`或其他方式定期运行。它将自动查询设备的 LLDP 邻居并构建图表，该图表将代表路由器所知的当前拓扑结构。

我们可以通过关闭`r6-edge`上的`Gi0/1`和`Go0/2`接口来测试这一点。当 LLDP 邻居超时时，它们将从`r6-edge`的 LLDP 表中消失。

```py
r6-edge#sh lldp neighbors
...
Device ID Local Intf Hold-time Capability Port ID
r2.virl.info Gi0/0 120 R Gi0/0
r3.virl.info Gi0/3 120 R Gi0/2
r3.virl.info Gi0/0 120 R Gi0/0
r5-tor.virl.info Gi0/0 120 R Gi0/0
r1.virl.info Gi0/0 120 R Gi0/0

Total entries displayed: 5
```

如果我们执行这个 playbook，图表将自动显示`r6-edge`只连接到`r3`，我们可以开始排查为什么会这样。

![](img/061f359c-1354-4204-a779-51bf15a11c9c.png)

这是一个相对较长的例子。我们使用了书中学到的工具——Ansible 和 Python——来模块化和将任务分解为可重用的部分。然后我们使用了一个新工具，即 Graphviz，来帮助监视网络的非时间序列数据，如网络拓扑关系。

# 基于流的监控

正如章节介绍中提到的，除了轮询技术（如 SNMP）之外，我们还可以使用推送策略，允许设备将网络信息推送到管理站点。NetFlow 及其密切相关的 IPFIX 和 sFlow 就是从网络设备向管理站点推送的信息的例子。我们可以认为`推送`方法更具可持续性，因为网络设备本身负责分配必要的资源来推送信息。例如，如果设备的 CPU 繁忙，它可以选择跳过流导出过程，而优先路由数据包，这正是我们想要的。

根据 IETF 的定义，流是从发送应用程序到接收应用程序的一系列数据包。如果我们回顾 OSI 模型，流就是构成两个应用程序之间通信的单个单位。每个流包括多个数据包；有些流有更多的数据包（如视频流），而有些只有几个（如 HTTP 请求）。如果你思考一下流，你会注意到路由器和交换机可能关心数据包和帧，但应用程序和用户通常更关心网络流。

基于流的监控通常指的是 NetFlow、IPFIX 和 sFlow：

+   **NetFlow**：NetFlow v5 是一种技术，网络设备会缓存流条目，并通过匹配元组集（源接口、源 IP/端口、目的 IP/端口等）来聚合数据包。一旦流完成，网络设备会导出流特征，包括流中的总字节数和数据包计数，到管理站点。

+   **IPFIX**：IPFIX 是结构化流的提议标准，类似于 NetFlow v9，也被称为灵活 NetFlow。基本上，它是一个可定义的流导出，允许用户导出网络设备了解的几乎任何内容。灵活性往往是以简单性为代价的，与 NetFlow v5 相比，IPFIX 的配置更加复杂。额外的复杂性使其不太适合初学者学习。但是，一旦你熟悉了 NetFlow v5，你就能够解析 IPFIX，只要你匹配模板定义。

+   sFlow：sFlow 实际上没有流或数据包聚合的概念。它对数据包进行两种类型的抽样。它随机抽样*n*个数据包/应用程序，并具有基于时间的抽样计数器。它将信息发送到管理站，管理站通过参考接收到的数据包样本类型和计数器来推导网络流信息。由于它不在网络设备上执行任何聚合，可以说 sFlow 比 NetFlow 和 IPFIX 更具可扩展性。

了解每个模块的最佳方法可能是直接进入示例。

# 使用 Python 解析 NetFlow

我们可以使用 Python 解析在线上传输的 NetFlow 数据报。这为我们提供了一种详细查看 NetFlow 数据包以及在其工作不如预期时排除任何 NetFlow 问题的方法。

首先，让我们在 VIRL 网络的客户端和服务器之间生成一些流量。我们可以使用 Python 的内置 HTTP 服务器模块快速在充当服务器的 VIRL 主机上启动一个简单的 HTTP 服务器：

```py
cisco@Server:~$ python3 -m http.server
Serving HTTP on 0.0.0.0 port 8000 ...
```

对于 Python 2，该模块的名称为`SimpleHTTPServer`；例如，`python2 -m SimpleHTTPServer`。

我们可以在 Python 脚本中创建一个简短的`while`循环，不断向客户端的 Web 服务器发送`HTTP GET`：

```py
sudo apt-get install python-pip python3-pip
sudo pip install requests
sudo pip3 install requests

$ cat http_get.py
import requests, time
while True:
 r = requests.get('http://10.0.0.5:8000')
 print(r.text)
 time.sleep(5)
```

客户端应该得到一个非常简单的 HTML 页面：

```py
cisco@Client:~$ python3 http_get.py
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN"><html>
<title>Directory listing for /</title>
<body>
...
</body>
</html>
```

我们还应该看到客户端每五秒不断发出请求：

```py
cisco@Server:~$ python3 -m http.server
Serving HTTP on 0.0.0.0 port 8000 ...
10.0.0.9 - - [15/Mar/2017 08:28:29] "GET / HTTP/1.1" 200 -
10.0.0.9 - - [15/Mar/2017 08:28:34] "GET / HTTP/1.1" 200 -
```

我们可以从任何设备导出 NetFlow，但由于`r6-edge`是客户端主机的第一跳，我们将使此路由器将 NetFlow 导出到端口`9995`的管理主机。

在此示例中，我们仅使用一个设备进行演示；因此，我们手动配置它所需的命令。在下一节中，当我们在所有设备上启用 NetFlow 时，我们将使用 Ansible playbook 一次性配置所有路由器。

在 Cisco IOS 设备上导出 NetFlow 需要以下配置：

```py
!
ip flow-export version 5
ip flow-export destination 172.16.1.173 9995 vrf Mgmt-intf
!
interface GigabitEthernet0/4
 description to Client
 ip address 10.0.0.10 255.255.255.252
 ip flow ingress
 ip flow egress
...
!
```

接下来，让我们看一下 Python 解析器脚本。

# Python socket 和 struct

脚本`netFlow_v5_parser.py`是从 Brian Rak 的博客文章[`blog.devicenull.org/2013/09/04/python-netflow-v5-parser.html`](http://blog.devicenull.org/2013/09/04/python-netflow-v5-parser.html)修改而来。修改主要是为了 Python 3 兼容性以及解析额外的 NetFlow 版本 5 字段。我们选择 NetFlow v5 而不是 NetFlow v9 的原因是 v9 更复杂，使用模板来映射字段，使得在入门会话中更难学习。但是，由于 NetFlow 版本 9 是原始 NetFlow 版本 5 的扩展格式，本节介绍的所有概念都适用于它。

因为 NetFlow 数据包在线上传输时以字节表示，我们将使用标准库中包含的 Python struct 模块将字节转换为本机 Python 数据类型。

您可以在[`docs.python.org/3.5/library/socket.html`](https://docs.python.org/3.5/library/socket.html)和[`docs.python.org/3.5/library/struct.html`](https://docs.python.org/3.5/library/struct.html)找到有关这两个模块的更多信息。

我们将首先使用 socket 模块绑定和监听 UDP 数据报。使用`socket.AF_INET`，我们打算监听 IPv4 地址套接字；使用`socket.SOCK_DGRAM`，我们指定将查看 UDP 数据报：

```py
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 9995))
```

我们将启动一个循环，并每次从线上检索 1,500 字节的信息：

```py
while True:
        buf, addr = sock.recvfrom(1500)
```

以下行是我们开始解构或解包数据包的地方。`!HH`的第一个参数指定了网络的大端字节顺序，感叹号表示大端字节顺序，以及 C 类型的格式（`H = 2`字节无符号短整数）：

```py
(version, count) = struct.unpack('!HH',buf[0:4])
```

前四个字节包括版本和此数据包中导出的流数。如果您没有记住 NetFlow 版本 5 标头（顺便说一句，这是一个玩笑；我只是在想要快速入睡时才会读标头），这里有一个快速浏览：

![](img/1cb08f1a-95f9-4402-a1b2-ee4753fc54b8.png)NetFlow v5 标头（来源：http://www.cisco.com/c/en/us/td/docs/net_mgmt/netflow_collection_engine/3-6/user/guide/format.html#wp1006108）

其余的标头可以根据字节位置和数据类型进行相应的解析：

```py
 (sys_uptime, unix_secs, unix_nsecs, flow_sequence) = struct.unpack('!IIII', buf[4:20])
 (engine_type, engine_id, sampling_interval) = struct.unpack('!BBH', buf[20:24])
```

接下来的`while`循环将使用流记录填充`nfdata`字典，解包源地址和端口、目的地址和端口、数据包计数和字节计数，并在屏幕上打印出信息：

```py
for i in range(0, count):
    try:
        base = SIZE_OF_HEADER+(i*SIZE_OF_RECORD)
        data = struct.unpack('!IIIIHH',buf[base+16:base+36])
        input_int, output_int = struct.unpack('!HH', buf[base+12:base+16])
        nfdata[i] = {}
        nfdata[i]['saddr'] = inet_ntoa(buf[base+0:base+4])
        nfdata[i]['daddr'] = inet_ntoa(buf[base+4:base+8])
        nfdata[i]['pcount'] = data[0]
        nfdata[i]['bcount'] = data[1]
...
```

脚本的输出允许您一目了然地查看标头以及流内容：

```py
Headers:
NetFlow Version: 5
Flow Count: 9
System Uptime: 290826756
Epoch Time in seconds: 1489636168
Epoch Time in nanoseconds: 401224368
Sequence counter of total flow: 77616
0 192.168.0.1:26828 -> 192.168.0.5:179 1 packts 40 bytes
1 10.0.0.9:52912 -> 10.0.0.5:8000 6 packts 487 bytes
2 10.0.0.9:52912 -> 10.0.0.5:8000 6 packts 487 bytes
3 10.0.0.5:8000 -> 10.0.0.9:52912 5 packts 973 bytes
4 10.0.0.5:8000 -> 10.0.0.9:52912 5 packts 973 bytes
5 10.0.0.9:52913 -> 10.0.0.5:8000 6 packts 487 bytes
6 10.0.0.9:52913 -> 10.0.0.5:8000 6 packts 487 bytes
7 10.0.0.5:8000 -> 10.0.0.9:52913 5 packts 973 bytes
8 10.0.0.5:8000 -> 10.0.0.9:52913 5 packts 973 bytes
```

请注意，在 NetFlow 版本 5 中，记录的大小固定为 48 字节；因此，循环和脚本相对简单。但是，在 NetFlow 版本 9 或 IPFIX 的情况下，在标头之后，有一个模板 FlowSet（[`www.cisco.com/en/US/technologies/tk648/tk362/technologies_white_paper09186a00800a3db9.html`](http://www.cisco.com/en/US/technologies/tk648/tk362/technologies_white_paper09186a00800a3db9.html)），它指定了字段计数、字段类型和字段长度。这使得收集器可以在不事先知道数据格式的情况下解析数据。

通过在脚本中解析 NetFlow 数据，我们对字段有了很好的理解，但这非常繁琐且难以扩展。正如您可能已经猜到的那样，还有其他工具可以帮助我们避免逐个解析 NetFlow 记录的问题。让我们在接下来的部分看看这样的一个工具，名为**ntop**。

# ntop 流量监控

就像第七章中的 PySNMP 脚本，以及本章中的 NetFlow 解析器脚本一样，我们可以使用 Python 脚本来处理线路上的低级任务。但是，也有一些工具，比如 Cacti，它是一个包含数据收集（轮询器）、数据存储（RRD）和用于可视化的 web 前端的一体化开源软件包。这些工具可以通过将经常使用的功能和软件打包到一个软件包中来节省大量工作。

在 NetFlow 的情况下，有许多开源和商业 NetFlow 收集器可供选择。如果您快速搜索前 N 个开源 NetFlow 分析器，您将看到许多不同工具的比较研究。它们每个都有自己的优势和劣势；使用哪一个实际上是一种偏好、平台和您对定制的兴趣。我建议选择一个既支持 v5 又支持 v9，可能还支持 sFlow 的工具。其次要考虑的是工具是否是用您能理解的语言编写的；我想拥有 Python 可扩展性会是一件好事。

我喜欢并以前使用过的两个开源 NetFlow 工具是 NfSen（后端收集器为 NFDUMP）和`ntop`（或`ntopng`）。在这两者中，`ntop`是更为知名的流量分析器；它可以在 Windows 和 Linux 平台上运行，并且与 Python 集成良好。因此，在本节中，让我们以`ntop`为例。

我们的 Ubuntu 主机的安装很简单：

```py
$ sudo apt-get install ntop
```

安装过程将提示输入必要的接口以进行监听，并设置管理员密码。默认情况下，`ntop` web 界面监听端口为`3000`，而探针监听 UDP 端口为`5556`。在网络设备上，我们需要指定 NetFlow 导出器的位置：

```py
!
ip flow-export version 5
ip flow-export destination 172.16.1.173 5556 vrf Mgmt-intf
!
```

默认情况下，IOSv 创建一个名为`Mgmt-intf`的 VRF，并将`Gi0/0`放在 VRF 下。

我们还需要在接口配置下指定流量导出的方向，比如入口或出口：

```py
!
interface GigabitEthernet0/0
...
 ip flow ingress
 ip flow egress
...
```

供您参考，我已经包含了 Ansible playbook，`cisco_config_netflow.yml`，用于配置实验设备进行 NetFlow 导出。

`r5-tor`和`r6-edge`比`r1`、`r2`和`r3`多两个接口。

执行 playbook 并确保设备上的更改已正确应用：

```py
$ ansible-playbook -i hosts cisco_config_netflow.yml

TASK [configure netflow export station] ****************************************
changed: [r1]
changed: [r3]
changed: [r2]
changed: [r5-tor]
changed: [r6-edge]

TASK [configure flow export on Gi0/0] ******************************************
changed: [r2]
changed: [r1]
changed: [r6-edge]
changed: [r5-tor]
changed: [r3]
...
PLAY RECAP *********************************************************************
r1 : ok=4 changed=4 unreachable=0 failed=0
r2 : ok=4 changed=4 unreachable=0 failed=0
r3 : ok=4 changed=4 unreachable=0 failed=0
r5-tor : ok=6 changed=6 unreachable=0 failed=0
r6-edge : ok=6 changed=6 unreachable=0 failed=0

##Checking r2 for NetFlow configuration
r2#sh run | i flow
 ip flow ingress
 ip flow egress
 ip flow ingress
 ip flow egress
 ip flow ingress
 ip flow egress
ip flow-export version 5
ip flow-export destination 172.16.1.173 5556 vrf Mgmt-intf 
```

一切都设置好后，您可以检查 ntop web 界面以查看本地 IP 流量：

![](img/e414a37d-aacc-43ef-a619-d95d026509cf.png)

ntop 最常用的功能之一是使用它来查看最活跃的对话者图表：

![](img/b5516589-b8af-4feb-b605-e603552db5bd.png)

ntop 报告引擎是用 C 编写的；它快速高效，但是需要对 C 有足够的了解才能做一些像改变 web 前端这样简单的事情，这并不符合现代敏捷开发的思维方式。

在 2000 年代中期，ntop 的人们在 Perl 上尝试了几次，最终决定将 Python 嵌入为可扩展的脚本引擎。让我们来看看。

# ntop 的 Python 扩展

我们可以使用 Python 通过 ntop web 服务器来扩展 ntop。ntop web 服务器可以执行 Python 脚本。在高层次上，脚本将执行以下操作：

+   访问 ntop 状态的方法

+   Python CGI 模块处理表单和 URL 参数

+   制作生成动态 HTML 页面的模板

+   每个 Python 脚本都可以从`stdin`读取并打印出`stdout/stderr`

+   `stdout`脚本是返回的 HTTP 页面

有几个资源对于 Python 集成非常有用。在 Web 界面下，您可以单击关于|显示配置，以查看 Python 解释器版本以及 Python 脚本的目录：

![](img/df42b61f-c658-4b7b-b622-3d0a1e38244d.png)Python 版本

您还可以检查 Python 脚本应该驻留的各个目录：

![](img/f76bf24a-6008-4e58-90a9-cd40ddd4b3e4.png)

插件目录

在关于|在线文档|Python ntop 引擎下，有 Python API 和教程的链接：

![](img/bb5f4a60-be4f-443f-b572-5b0cab30ba76.png)Python ntop 文档

如前所述，ntop web 服务器直接执行放置在指定目录下的 Python 脚本：

```py
$ pwd
/usr/share/ntop/python
```

我们将把我们的第一个脚本，即`chapter8_ntop_1.py`，放在目录中。Python `CGI`模块处理表单并解析 URL 参数：

```py
# Import modules for CGI handling
import cgi, cgitb
import ntop

# Parse URL
cgitb.enable();
```

`ntop`实现了三个 Python 模块；每个模块都有特定的目的：

+   `ntop`：此模块与`ntop`引擎交互

+   **主机**：此模块用于深入了解特定主机的信息

+   **接口**：此模块表示有关本地主机接口的信息

在我们的脚本中，我们将使用`ntop`模块来检索`ntop`引擎信息，并使用`sendString()`方法发送 HTML 正文文本：

```py
form = cgi.FieldStorage();
name = form.getvalue('Name', default="Eric")

version = ntop.version()
os = ntop.os()
uptime = ntop.uptime()

ntop.printHTMLHeader('Mastering Python Networking', 1, 0)
ntop.sendString("Hello, "+ name +"<br>")
ntop.sendString("Ntop Information: %s %s %s" % (version, os, uptime))
ntop.printHTMLFooter()
```

我们将使用`http://<ip>:3000/python/<script name>`来执行 Python 脚本。这是我们的`chapter8_ntop_1.py`脚本的结果：

![](img/869976ca-1508-4a7b-a9ba-4bdd8c66b83e.png)

我们可以看另一个与接口模块交互的示例，`chapter8_ntop_2.py`。我们将使用 API 来遍历接口：

```py
import ntop, interface, json

ifnames = []
try:
    for i in range(interface.numInterfaces()):
        ifnames.append(interface.name(i))

except Exception as inst:
    print type(inst) # the exception instance
    print inst.args # arguments stored in .args
    print inst # __str__ allows args to printed directly
...
```

生成的页面将显示 ntop 接口：

![](img/6140fbbb-7e30-4897-ad3f-fabcd4ce6fa6.png)

除了社区版本外，ntop 还提供了一些商业产品供您选择。凭借活跃的开源社区、商业支持和 Python 可扩展性，ntop 是您 NetFlow 监控需求的不错选择。

接下来，让我们来看看 NetFlow 的表兄弟：sFlow。

# sFlow

sFlow 最初由 InMon（[`www.inmon.com`](http://www.inmon.com)）开发，后来通过 RFC 进行了标准化。当前版本是 v5。行业内许多人认为 sFlow 的主要优势是其可扩展性。sFlow 使用随机的一种`n`数据包流样本以及计数器样本的轮询间隔来推导出流量的估计；这比网络设备的 NetFlow 更节省 CPU。sFlow 的统计采样与硬件集成，并提供实时的原始导出。

出于可扩展性和竞争原因，sFlow 通常比 NetFlow 更受新供应商的青睐，例如 Arista Networks、Vyatta 和 A10 Networks。虽然思科在其 Nexus 产品线上支持 sFlow，但通常*不*支持在思科平台上使用 sFlow。

# SFlowtool 和 sFlow-RT 与 Python

很遗憾，到目前为止，sFlow 是我们的 VIRL 实验室设备不支持的东西（即使是 NX-OSv 虚拟交换机也不支持）。您可以使用思科 Nexus 3000 交换机或其他支持 sFlow 的供应商交换机，例如 Arista。实验室的另一个好选择是使用 Arista vEOS 虚拟实例。我碰巧可以访问运行 7.0（3）的思科 Nexus 3048 交换机，我将在本节中使用它作为 sFlow 导出器。

思科 Nexus 3000 的 sFlow 配置非常简单：

```py
Nexus-2# sh run | i sflow
feature sflow
sflow max-sampled-size 256
sflow counter-poll-interval 10
sflow collector-ip 192.168.199.185 vrf management
sflow agent-ip 192.168.199.148
sflow data-source interface Ethernet1/48
```

摄取 sFlow 的最简单方法是使用`sflowtool`。有关安装说明，请参阅[`blog.sflow.com/2011/12/sflowtool.html`](http://blog.sflow.com/2011/12/sflowtool.html)上的文档：

```py
$ wget http://www.inmon.com/bin/sflowtool-3.22.tar.gz
$ tar -xvzf sflowtool-3.22.tar.gz
$ cd sflowtool-3.22/
$ ./configure
$ make
$ sudo make install
```

安装完成后，您可以启动`sflowtool`并查看 Nexus 3048 发送到标准输出的数据报：

```py
$ sflowtool
startDatagram =================================
datagramSourceIP 192.168.199.148
datagramSize 88
unixSecondsUTC 1489727283
datagramVersion 5
agentSubId 100
agent 192.168.199.148
packetSequenceNo 5250248
sysUpTime 4017060520
samplesInPacket 1
startSample ----------------------
sampleType_tag 0:4
sampleType COUNTERSSAMPLE
sampleSequenceNo 2503508
sourceId 2:1
counterBlock_tag 0:1001
5s_cpu 0.00
1m_cpu 21.00
5m_cpu 20.80
total_memory_bytes 3997478912
free_memory_bytes 1083838464
endSample ----------------------
endDatagram =================================
```

`sflowtool` GitHub 存储库（[`github.com/sflow/sflowtool`](https://github.com/sflow/sflowtool)）上有许多很好的用法示例；其中之一是使用脚本接收`sflowtool`输入并解析输出。我们可以使用 Python 脚本来实现这个目的。在`chapter8_sflowtool_1.py`示例中，我们将使用`sys.stdin.readline`接收输入，并使用正则表达式搜索仅打印包含单词`agent`的行当我们看到 sFlow 数据包时：

```py
import sys, re
for line in iter(sys.stdin.readline, ''):
    if re.search('agent ', line):
        print(line.strip())
```

该脚本可以通过管道传输到`sflowtool`：

```py
$ sflowtool | python3 chapter8_sflowtool_1.py
agent 192.168.199.148
agent 192.168.199.148
```

还有许多其他有用的输出示例，例如`tcpdump`，以 NetFlow 版本 5 记录输出，以及紧凑的逐行输出。这使得`sflowtool`非常灵活，以适应您的监控环境。

ntop 支持 sFlow，这意味着您可以直接将您的 sFlow 导出到 ntop 收集器。如果您的收集器只支持 NetFlow，您可以在 NetFlow 版本 5 格式中使用`sflowtool`输出的`-c`选项：

```py
$ sflowtool --help
...
tcpdump output:
   -t - (output in binary tcpdump(1) format)
   -r file - (read binary tcpdump(1) format)
   -x - (remove all IPV4 content)
   -z pad - (extend tcpdump pkthdr with this many zeros
                          e.g. try -z 8 for tcpdump on Red Hat Linux 6.2)

NetFlow output:
 -c hostname_or_IP - (netflow collector host)
 -d port - (netflow collector UDP port)
 -e - (netflow collector peer_as (default = origin_as))
 -s - (disable scaling of netflow output by sampling rate)
 -S - spoof source of netflow packets to input agent IP
```

或者，您也可以使用 InMon 的 sFlow-RT（[`www.sflow-rt.com/index.php`](http://www.sflow-rt.com/index.php)）作为您的 sFlow 分析引擎。sFlow-RT 从操作员的角度来看，其主要优势在于其庞大的 REST API，可以定制以支持您的用例。您还可以轻松地从 API 中检索指标。您可以在[`www.sflow-rt.com/reference.php`](http://www.sflow-rt.com/reference.php)上查看其广泛的 API 参考。

请注意，sFlow-RT 需要 Java 才能运行以下内容：

```py
$ sudo apt-get install default-jre
$ java -version
openjdk version "1.8.0_121"
OpenJDK Runtime Environment (build 1.8.0_121-8u121-b13-0ubuntu1.16.04.2-b13)
OpenJDK 64-Bit Server VM (build 25.121-b13, mixed mode)
```

安装完成后，下载和运行 sFlow-RT 非常简单（[`sflow-rt.com/download.php`](https://sflow-rt.com/download.php)）：

```py
$ wget http://www.inmon.com/products/sFlow-RT/sflow-rt.tar.gz
$ tar -xvzf sflow-rt.tar.gz
$ cd sflow-rt/
$ ./start.sh
2017-03-17T09:35:01-0700 INFO: Listening, sFlow port 6343
2017-03-17T09:35:02-0700 INFO: Listening, HTTP port 8008
```

我们可以将 Web 浏览器指向 HTTP 端口`8008`并验证安装：

![](img/ac593583-04c5-4b0b-8ade-4a925b583726.png)sFlow-RT about

一旦 sFlow-RT 接收到任何 sFlow 数据包，代理和其他指标将出现：

![](img/1a438b4c-45c7-459a-953f-54991e570b6e.png)sFlow-RT agents

以下是使用 Python 请求从 sFlow-RT 的 REST API 中检索信息的两个示例：

```py
>>> import requests
>>> r = requests.get("http://192.168.199.185:8008/version")
>>> r.text
'2.0-r1180'
>>> r = requests.get("http://192.168.199.185:8008/agents/json")
>>> r.text
'{"192.168.199.148": {n "sFlowDatagramsLost": 0,n "sFlowDatagramSource": ["192.168.199.148"],n "firstSeen": 2195541,n "sFlowFlowDuplicateSamples": 0,n "sFlowDatagramsReceived": 441,n "sFlowCounterDatasources": 2,n "sFlowFlowOutOfOrderSamples": 0,n "sFlowFlowSamples": 0,n "sFlowDatagramsOutOfOrder": 0,n "uptime": 4060470520,n "sFlowCounterDuplicateSamples": 0,n "lastSeen": 3631,n "sFlowDatagramsDuplicates": 0,n "sFlowFlowDrops": 0,n "sFlowFlowLostSamples": 0,n "sFlowCounterSamples": 438,n "sFlowCounterLostSamples": 0,n "sFlowFlowDatasources": 0,n "sFlowCounterOutOfOrderSamples": 0n}}'
```

咨询参考文档，了解可用于您需求的其他 REST 端点。接下来，我们将看看另一个工具，称为**Elasticsearch**，它正在成为 Syslog 索引和一般网络监控的相当流行的工具。

# Elasticsearch（ELK 堆栈）

正如我们在本章中所看到的，仅使用我们已经使用的 Python 工具就足以监控您的网络，并具有足够的可扩展性，适用于各种规模的网络，无论大小。然而，我想介绍一个名为**Elasticsearch**（[`www.elastic.co/`](https://www.elastic.co/)）的额外的开源、通用分布式搜索和分析引擎。它通常被称为**Elastic**或**ELK 堆栈**，用于将**Elastic**与前端和输入包**Logstash**和**Kibana**结合在一起。

如果您总体上看网络监控，实际上是分析网络数据并理解其中的意义。ELK 堆栈包含 Elasticsearch、Logstash 和 Kibana 作为完整的堆栈，使用 Logstash 摄取信息，使用 Elasticsearch 索引和分析数据，并通过 Kibana 呈现图形输出。它实际上是三个项目合而为一。它还具有灵活性，可以用其他输入替换 Logstash，比如**Beats**。或者，您可以使用其他工具，比如**Grafana**，而不是 Kibana 进行可视化。Elastic Co*.*的 ELK 堆栈还提供许多附加工具，称为**X-Pack**，用于额外的安全性、警报、监控等。

正如您可能从描述中可以看出，ELK（甚至仅是 Elasticsearch）是一个深入的主题，有许多关于这个主题的书籍。即使只涵盖基本用法，也会占用比我们在这本书中可以空出的更多空间。我曾考虑过将这个主题从书中删除，仅仅是因为它的深度。然而，ELK 已经成为我正在进行的许多项目中非常重要的工具，包括网络监控。我觉得不把它放在书中会对你造成很大的伤害。

因此，我将花几页时间简要介绍这个工具以及一些用例，以及一些信息，让您有兴趣深入了解。我们将讨论以下主题：

+   建立托管的 ELK 服务

+   Logstash 格式

+   Logstash 格式的 Python 辅助脚本

# 建立托管的 ELK 服务

整个 ELK 堆栈可以安装为独立服务器或分布在多台服务器上。安装步骤可在[`www.elastic.co/guide/en/elastic-stack/current/installing-elastic-stack.html`](https://www.elastic.co/guide/en/elastic-stack/current/installing-elastic-stack.html)上找到。根据我的经验，即使只有少量数据，运行 ELK 堆栈的单个虚拟机通常也会耗尽资源。我第一次尝试将 ELK 作为单个虚拟机运行，仅持续了几天，几乎只有两三个网络设备向其发送日志信息。在作为初学者运行自己的集群的几次不成功尝试之后，我最终决定将 ELK 堆栈作为托管服务运行，这也是我建议您开始使用的方式。

作为托管服务，有两个提供商可以考虑：

+   **Amazon Elasticsearch Service**（[`aws.amazon.com/elasticsearch-service/`](https://aws.amazon.com/elasticsearch-service/)）

+   **Elastic Cloud**（[`cloud.elastic.co/`](https://cloud.elastic.co/)）

目前，AWS 提供了一个免费的套餐，很容易开始使用，并且与当前的 AWS 工具套件紧密集成，例如身份服务（[`aws.amazon.com/iam/`](https://aws.amazon.com/iam/)）和 lambda 函数（[`aws.amazon.com/lambda/`](https://aws.amazon.com/lambda/)）。然而，与 Elastic Cloud 相比，AWS 的 Elasticsearch 服务没有最新的功能，也没有扩展的 x-pack 集成。然而，由于 AWS 提供了免费套餐，我的建议是您从 AWS Elasticsearch 服务开始。如果您后来发现需要比 AWS 提供的更多功能，您总是可以转移到 Elastic Cloud。

设置服务很简单；我们只需要选择我们的区域并为我们的第一个域名命名。设置完成后，我们可以使用访问策略来通过 IP 地址限制输入；确保这是 AWS 将看到的源 IP 地址（如果您的主机 IP 地址在 NAT 防火墙后面被转换，请指定您的公司公共 IP）：

![](img/75b38547-2d30-40f8-a951-7af6777231e8.png)

# Logstash 格式

Logstash 可以安装在您习惯发送网络日志的服务器上。安装步骤可在[`www.elastic.co/guide/en/logstash/current/installing-logstash.html`](https://www.elastic.co/guide/en/logstash/current/installing-logstash.html)找到。默认情况下，您可以将 Logstash 配置文件放在`/etc/logstash/conf.d/`下。该文件采用`input-filter-output`格式（[`www.elastic.co/guide/en/logstash/current/advanced-pipeline.html`](https://www.elastic.co/guide/en/logstash/current/advanced-pipeline.html)）。在下面的示例中，我们将输入指定为`网络日志文件`，并使用占位符过滤输入，输出为将消息打印到控制台以及将输出导出到我们的 AWS Elasticsearch 服务实例：

```py
input {
  file {
    type => "network_log"
    path => "path to your network log file"
 }
}
filter {
  if [type] == "network_log" {
  }
}
output {
  stdout { codec => rubydebug }
  elasticsearch {
  index => "logstash_network_log-%{+YYYY.MM.dd}"
  hosts => ["http://<instance>.<region>.es.amazonaws.com"]
  }
}
```

现在让我们来看看我们可以用 Python 和 Logstash 做的其他事情。

# 用于 Logstash 格式的 Python 辅助脚本

前面的 Logstash 配置将允许我们摄取网络日志并在 Elasticsearch 上创建索引。如果我们打算放入 ELK 的文本格式不是标准的日志格式，会发生什么？这就是 Python 可以帮助的地方。在下一个示例中，我们将执行以下操作：

1.  使用 Python 脚本检索 Spamhaus 项目认为是拒收列表的 IP 地址列表（[`www.spamhaus.org/drop/drop.txt`](https://www.spamhaus.org/drop/drop.txt)）

1.  使用 Python 日志模块以 Logstash 可以摄取的方式格式化信息

1.  修改 Logstash 配置文件，以便任何新输入都可以发送到 AWS Elasticsearch 服务

`chapter8_logstash_1.py`脚本包含我们将使用的代码。除了模块导入之外，我们将定义基本的日志配置。该部分直接配置输出，并且应该与 Logstash 格式匹配：

```py
#!/usr/env/bin python

#https://www.spamhaus.org/drop/drop.txt

import logging, pprint, re
import requests, json, datetime
from collections import OrderedDict

#logging configuration
logging.basicConfig(filename='./tmp/spamhaus_drop_list.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%b %d %I:%M:%S')
```

我们将定义一些更多的变量，并将请求中的 IP 地址列表保存在一个变量中：

```py
host = 'python_networking'
process = 'spamhause_drop_list'

r = requests.get('https://www.spamhaus.org/drop/drop.txt')
result = r.text.strip()

timeInUTC = datetime.datetime.utcnow().isoformat()
Item = OrderedDict()
Item["Time"] = timeInUTC
```

脚本的最后一部分是一个循环，用于解析输出并将其写入新的日志文件：

```py
for line in result.split('n'):
    if re.match('^;', line) or line == 'r': # comments
        next
    else:
       ip, record_number = line.split(";")
       logging.warning(host + ' ' + process + ': ' + 'src_ip=' + ip.split("/")[0] + ' record_number=' + record_number.strip())
```

以下是日志文件条目的示例：

```py
$ cat tmp/spamhaus_drop_list.log
...
Jul 14 11:35:26 python_networking spamhause_drop_list: src_ip=212.92.127.0 record_number=SBL352250
Jul 14 11:35:26 python_networking spamhause_drop_list: src_ip=216.47.96.0 record_number=SBL125132
Jul 14 11:35:26 python_networking spamhause_drop_list: src_ip=223.0.0.0 record_number=SBL230805
Jul 14 11:35:26 python_networking spamhause_drop_list: src_ip=223.169.0.0 record_number=SBL208009
...
```

然后我们可以相应地修改 Logstash 配置文件以适应我们的新日志格式，首先是添加输入文件位置：

```py
input {
  file {
    type => "network_log"
    path => "path to your network log file"
 }
  file {
    type => "spamhaus_drop_list"
    path => "/home/echou/Master_Python_Networking/Chapter8/tmp/spamhaus_drop_list.log"
 }
}
```

我们可以使用`grok`添加更多的过滤配置：

```py
filter { 
  if [type] == "spamhaus_drop_list" {
     grok {
       match => [ "message", "%{SYSLOGTIMESTAMP:timestamp} %{SYSLOGHOST:hostname} %{NOTSPACE:process} src_ip=%{IP:src_ip} %{NOTSPACE:record_number}.*"]
       add_tag => ["spamhaus_drop_list"]
     }
  }
}
```

我们可以将输出部分保持不变，因为额外的条目将存储在同一索引中。现在我们可以使用 ELK 堆栈来查询、存储和查看网络日志以及 Spamhaus IP 信息。

# 总结

在本章中，我们看了一些额外的方法，可以利用 Python 来增强我们的网络监控工作。我们首先使用 Python 的 Graphviz 包来创建实时 LLDP 信息报告的网络拓扑图。这使我们能够轻松地显示当前的网络拓扑，以及轻松地注意到任何链路故障。

接下来，我们使用 Python 来解析 NetFlow 版本 5 数据包，以增强我们对 NetFlow 的理解和故障排除能力。我们还研究了如何使用 ntop 和 Python 来扩展 ntop 以进行 NetFlow 监控。sFlow 是一种替代的数据包抽样技术，我们使用`sflowtool`和 sFlow-RT 来解释结果。我们在本章结束时介绍了一个通用的数据分析工具，即 Elasticsearch，或者 ELK 堆栈。

在第九章中，*使用 Python 构建网络 Web 服务*，我们将探讨如何使用 Python Web 框架 Flask 来构建网络 Web 服务。
