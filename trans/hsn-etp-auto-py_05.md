# 从网络设备中提取有用数据

在上一章中，我们已经看到了如何使用不同的方法和协议访问网络设备，然后在远程设备上执行命令，将输出返回到Python。现在，是时候从这个输出中提取一些有用的数据了。

在本章中，您将学习如何使用Python中的不同工具和库从返回的输出中提取有用的数据，并使用正则表达式对其进行操作。此外，我们将使用一个名为`CiscoConfParse`的特殊库来审计配置，然后学习如何使用`matplotlib`库可视化数据，生成视觉上吸引人的图形和报告。

在本章中，我们将涵盖以下主题：

+   理解解析器

+   正则表达式简介

+   使用`Ciscoconfparse`进行配置审计

+   使用`matplotlib`可视化返回的数据

# 技术要求

您的环境中应安装并可用以下工具：

+   Python 2.7.1x

+   PyCharm社区版或专业版

+   EVE-NG实验室

您可以在以下GitHub URL找到本章开发的完整脚本：

[https://github.com/TheNetworker/EnterpriseAutomation.git](https://github.com/TheNetworker/EnterpriseAutomation.git)

# 理解解析器

在上一章中，我们探讨了访问网络设备、执行命令并将输出返回到终端的不同方式。现在我们需要处理返回的输出，并从中提取一些有用的信息。请注意，从Python的角度来看，输出只是一个多行字符串，Python不区分IP地址、接口名称或节点主机名，因为它们都是字符串。因此，第一步是设计和开发我们自己的解析器，使用Python根据返回的输出中的重要信息对项目进行分类和区分。

之后，您可以处理解析后的数据，并生成有助于可视化的图形，甚至将它们存储到持久的外部存储或数据库中。

# 正则表达式简介

正则表达式是一种语言，用于通过跟随整个字符串的模式来匹配特定的字符串出现。当找到匹配时，将返回匹配的字符串，并将其保存在Python格式的结构中，如`tuple`、`list`或`dictionary`。以下表总结了正则表达式中最常见的模式：

![](../images/00090.jpeg)

此外，正则表达式中的一个重要规则是您可以编写自己的正则表达式，并用括号`()`括起来，这称为捕获组，它可以帮助您保存重要数据，以便稍后使用捕获组编号引用它：

```py
line = '30 acd3.b2c6.aac9 FastEthernet0/1' 
match = re.search('(\d+) +([0-9a-f.]+) +(\S+)', line)
print match.group(1)
print match.group(2)
```

PyCharm将自动对写成正则表达式的字符串进行着色，并可以帮助您在将其应用于数据之前检查正则表达式的有效性。请确保在设置中启用了Check RegExp意图，如下所示：![](../images/00091.jpeg)

# 在Python中创建正则表达式

您可以使用Python中的`re`模块构建正则表达式，该模块已经与Python安装一起原生地提供。该模块内部有几种方法，如`search()`、`sub()`、`split()`、`compile()`和`findall()`，它们将以正则表达式对象的形式返回结果。以下是每个函数的用法总结：

| **函数名称** | **用法** |
| --- | --- |
| `search()` | 搜索和匹配模式的第一个出现。 |
| `findall()` | 搜索和匹配模式的所有出现，并将结果作为列表返回。 |
| `Finditer()` | 搜索和匹配模式的所有出现，并将结果作为迭代器返回。 |
| `compile()` | 将正则表达式编译为具有各种操作方法的模式对象，例如搜索模式匹配或执行字符串替换。如果您在脚本中多次使用相同的正则表达式模式，这将非常有用。 |
| `sub()` | 用于用另一个字符串替换匹配的模式。 |
| `split()` | 用于在匹配模式上拆分并创建列表。 |

正则表达式很难阅读；因此，让我们从简单的开始，看一些最基本级别的简单正则表达式。

使用`re`模块的第一步是在Python代码中导入它

```py
import re
```

我们将开始探索`re`模块中最常见的函数，即`search()`，然后我们将探索`findall()`。当您需要在字符串中找到一个匹配项，或者当您编写正则表达式模式来匹配整个输出并需要使用`groups()`方法来获取结果时，`search()`函数是合适的，正如我们将在接下来的例子中看到的。

`re.search()`函数的语法如下：

```py
match = re.search('regex pattern', 'string')
```

第一个参数`'regex pattern'`是为了匹配`'string'`中的特定出现而开发的正则表达式。当找到匹配项时，`search()`函数将返回一个特殊的匹配对象，否则将返回`None`。请注意，`search()`将仅返回模式的第一个匹配项，并将忽略其余的匹配项。让我们看一些在Python中使用`re`模块的例子：

**示例1：搜索特定IP地址**

```py
import re
intf_ip = 'Gi0/0/0.911            10.200.101.242   YES NVRAM  up                    up' match = re.search('10.200.101.242', intf_ip)    if match:
  print match.group()
```

在这个例子中，我们可以看到以下内容：

+   `re`模块被导入到我们的Python脚本中。

+   我们有一个字符串，对应于接口详细信息，并包含名称、IP地址和状态。这个字符串可以在脚本中硬编码，也可以使用Netmiko库从网络设备中生成。

+   我们将这个字符串传递给`search()`函数，以及我们的正则表达式，即IP地址。

+   然后，脚本检查前一个操作是否返回了`match`对象；如果是，则会打印出来。

测试匹配的最基本方法是通过`re.match`函数，就像我们在前面的例子中所做的那样。`match`函数接受一个正则表达式模式和一个字符串值。

请注意，我们只在`intf_ip`参数内搜索特定的字符串，而不是每个IP地址模式。

**示例1输出**

![](../images/00092.jpeg)

**示例2：匹配IP地址模式**

```py
import re
intf_ip = '''Gi0/0/0.705            10.103.17.5      YES NVRAM  up                    up Gi0/0/0.900            86.121.75.31  YES NVRAM  up                    up Gi0/0/0.911            10.200.101.242   YES NVRAM  up                    up Gi0/0/0.7000           unassigned      YES unset  up                    up ''' match = re.search("\d+\.\d+\.\d+\.\d+", intf_ip)   if match:
  print match.group()
```

在这个例子中，我们可以看到以下内容：

+   `re`模块被导入到我们的Python脚本中。

+   我们有一个多行字符串，对应于接口详细信息，并包含名称、IP地址和状态。

+   我们将这个字符串传递给`search()`函数，以及我们的正则表达式，即使用`\d+`匹配一个或多个数字，以及`\.`匹配点的出现。

+   然后，脚本检查前一个操作是否返回了`match`对象；如果是，则会打印出来。否则，将返回`None`对象。

**示例2输出**

![](../images/00093.jpeg)

请注意，`search()`函数只返回模式的第一个匹配项，而不是所有匹配项。

**示例3：使用** **groups()正则表达式**

如果您有一个长输出，并且需要从中提取多个字符串，那么您可以用`()`括起提取的值，并在其中编写您的正则表达式。这称为**捕获组**，用于捕获长字符串中的特定模式，如下面的代码片段所示：

```py
import re
log_msg = 'Dec 20 12:11:47.417: %LINK-3-UPDOWN: Interface GigabitEthernet0/0/4, changed state to down' match = re.search("(\w+\s\d+\s\S+):\s(\S+): Interface (\S+), changed state to (\S+)", log_msg) if match:
  print match.groups() 
```

在这个例子中，我们可以看到以下内容：

+   `re`模块被导入到我们的Python脚本中。

+   我们有一个字符串，对应于路由器中发生的事件，并存储在日志中。

+   我们将这个字符串传递给`search()`函数，以及我们的正则表达式。请注意，我们将时间戳、事件类型、接口名称和捕获组的新状态都括起来，并在其中编写我们的正则表达式。

+   然后，脚本检查前一个操作是否返回了匹配对象；如果是，则会打印出来，但这次我们使用了`groups()`而不是`group()`，因为我们正在捕获多个字符串。

**示例3输出**

![](../images/00094.jpeg)

请注意，返回的数据是一个名为**tuple**的结构化格式。我们可以稍后使用此输出来触发事件，并且例如在冗余接口上启动恢复过程。

我们可以增强我们之前的代码，并使用`Named`组来为每个捕获组命名，以便稍后引用或用于创建字典。在这种情况下，我们在正则表达式前面加上了`?P<"NAME">`，就像下一个示例（GitHub存储库中的**示例4**）中一样：**示例4：命名组**![](../images/00095.jpeg)

**示例5-1：使用re.search()搜索多行**

假设我们的输出中有多行，并且我们需要针对正则表达式模式检查所有这些行。请记住，`search()`函数在找到第一个模式匹配时退出。在这种情况下，我们有两种解决方案。第一种是通过在`"\n"`上拆分整个字符串将每行输入到搜索函数中，第二种解决方案是使用`findall()`函数。让我们探讨这两种解决方案：

```py

import re

show_ip_int_br_full = """ GigabitEthernet0/0/0        110.110.110.1   YES NVRAM  up                    up GigabitEthernet0/0/1        107.107.107.1   YES NVRAM  up                    up GigabitEthernet0/0/2        108.108.108.1   YES NVRAM  up                    up GigabitEthernet0/0/3        109.109.109.1   YES NVRAM  up                    up GigabitEthernet0/0/4   unassigned      YES NVRAM  up                    up GigabitEthernet0/0/5             10.131.71.1     YES NVRAM  up                    up GigabitEthernet0/0/6          10.37.102.225   YES NVRAM  up                    up GigabitEthernet0/1/0            unassigned      YES unset  up                    up GigabitEthernet0/1/1           57.234.66.28   YES manual up                    up GigabitEthernet0/1/2           10.10.99.70   YES manual up                    up GigabitEthernet0/1/3           unassigned      YES manual deleted               down GigabitEthernet0/1/4           192.168.200.1   YES manual up                    up GigabitEthernet0/1/5   unassigned      YES manual down                  down GigabitEthernet0/1/6         10.20.20.1      YES manual down                  down GigabitEthernet0/2/0         10.30.40.1      YES manual down                  down GigabitEthernet0/2/1         57.20.20.1      YES manual down                  down  """ for line in show_ip_int_br_full.split("\n"):
  match = re.search(r"(?P<interface>\w+\d\/\d\/\d)\s+(?P<ip>\d+.\d+.\d+.\d+)", line)
  if match:
  intf_ip = match.groupdict()
  if intf_ip["ip"].startswith("57"):
  print "Subnet is configured on " + intf_ip["interface"] + " and ip is " + intf_ip["ip"]
```

上面的脚本将拆分`show ip interface brief`输出并搜索特定模式，即接口名称和配置在其上的IP地址。根据匹配的数据，脚本将继续检查每个IP地址并使用`start with 57`进行验证，然后脚本将打印相应的接口和完整的IP地址。

**示例5-1输出**

![](../images/00096.jpeg)如果您只搜索第一次出现，可以优化脚本，并且只需在找到第一个匹配项时中断外部`for`循环，但请注意，第二个匹配项将无法找到或打印。

**示例5-2：使用re.findall()搜索多行**

`findall()`函数在提供的字符串中搜索所有不重叠的匹配项，并返回与正则表达式模式匹配的字符串列表（与`search`函数不同，后者返回`match`对象），如果没有捕获组，则返回。如果您用捕获组括起您的正则表达式，那么`findall()`将返回一个元组列表。在下面的脚本中，我们有相同的多行输出，并且我们将使用`findall()`方法来获取所有配置了以57开头的IP地址的接口：

```py
import re
from pprint import pprint
show_ip_int_br_full = """ GigabitEthernet0/0/0        110.110.110.1   YES NVRAM  up                    up GigabitEthernet0/0/1        107.107.107.1   YES NVRAM  up                    up GigabitEthernet0/0/2        108.108.108.1   YES NVRAM  up                    up GigabitEthernet0/0/3        109.109.109.1   YES NVRAM  up                    up GigabitEthernet0/0/4   unassigned      YES NVRAM  up                    up GigabitEthernet0/0/5             10.131.71.1     YES NVRAM  up                    up GigabitEthernet0/0/6          10.37.102.225   YES NVRAM  up                    up GigabitEthernet0/1/0            unassigned      YES unset  up                    up GigabitEthernet0/1/1           57.234.66.28   YES manual up                    up GigabitEthernet0/1/2           10.10.99.70   YES manual up                    up GigabitEthernet0/1/3           unassigned      YES manual deleted               down GigabitEthernet0/1/4           192.168.200.1   YES manual up                    up GigabitEthernet0/1/5   unassigned      YES manual down                  down GigabitEthernet0/1/6         10.20.20.1      YES manual down                  down GigabitEthernet0/2/0         10.30.40.1      YES manual down                  down GigabitEthernet0/2/1         57.20.20.1      YES manual down                  down """    intf_ip = re.findall(r"(?P<interface>\w+\d\/\d\/\d)\s+(?P<ip>57.\d+.\d+.\d+)", show_ip_int_br_full) pprint(intf_ip) 
```

**示例5-2输出**：

![](../images/00097.jpeg)

请注意，这一次我们不必编写`for`循环来检查每行是否符合正则表达式模式。这将在`findall()`方法中自动完成。

# 使用CiscoConfParse进行配置审计

在网络配置上应用正则表达式以从输出中获取特定信息需要我们编写一些复杂的表达式来解决一些复杂的用例。在某些情况下，您只需要检索一些配置或修改现有配置而不深入编写正则表达式，这就是`CiscoConfParse`库诞生的原因（[https://github.com/mpenning/ciscoconfparse](https://github.com/mpenning/ciscoconfparse)）。

# CiscoConfParse库

正如官方GitHub页面所说，该库检查了一个类似iOS风格的配置，并将其分解成一组链接的父/子关系。您可以对这些关系执行复杂的查询：

![](../images/00098.jpeg)来源：[https://github.com/mpenning/ciscoconfparse](https://github.com/mpenning/ciscoconfparse)

因此，配置的第一行被视为父级，而后续行被视为父级的子级。`CiscoConfparse`库将父级和子级之间的关系构建成一个对象，因此最终用户可以轻松地检索特定父级的配置，而无需编写复杂的表达式。

非常重要的是，您的配置文件格式良好，以便在父级和子级之间建立正确的关系。

如果需要向文件中注入配置，也适用相同的概念。该库将搜索给定的父级，并将配置插入其下方，并保存到新文件中。这在您需要对多个文件运行配置审计作业并确保它们都具有一致的配置时非常有用。

# 支持的供应商

作为一个经验法则，任何具有制表符分隔配置的文件都可以被`CiscoConfParse`解析，并且它将构建父子关系。

以下是支持的供应商列表：

+   Cisco IOS，Cisco Nexus，Cisco IOS-XR，Cisco IOS-XE，Aironet OS，Cisco ASA，Cisco CatOS

+   Arista EOS

+   Brocade

+   HP交换机

+   Force10交换机

+   Dell PowerConnect交换机

+   Extreme Networks

+   Enterasys

+   ScreenOS

另外，从1.2.4版本开始，`CiscoConfParse`可以处理花括号分隔的配置，这意味着它可以处理以下供应商：

+   Juniper Network的Junos OS

+   Palo Alto Networks防火墙配置

+   F5 Networks配置

# CiscoConfParse安装

`CiscoConfParse`可以通过在Windows命令行或Linux shell上使用`pip`来安装：

```py
pip install ciscoconfparse
```

![](../images/00099.jpeg)

请注意，还安装了一些其他依赖项，例如`ipaddr`，`dnsPython`和`colorama`，这些依赖项被`CiscoConfParse`使用。

# 使用CiscoConfParse

我们将要处理的第一个示例是从名为`Cisco_Config.txt`的文件中提取关闭接口的示例Cisco配置。

![](../images/00100.jpeg)

在这个例子中，我们可以看到以下内容：

+   从`CiscoConfParse`模块中，我们导入了`CiscoConfParse`类。同时，我们导入了`pprint`模块，以便以可读格式打印输出以适应Python控制台输出。

+   然后，我们将`config`文件的完整路径提供给`CiscoConfParse`类。

+   最后一步是使用内置函数之一，例如`find_parents_w_child()`，并提供两个参数。第一个是父级规范，它搜索以`interface`关键字开头的任何内容，而子规范具有`shutdown`关键字。

正如您所看到的，在三个简单的步骤中，我们能够获取所有具有关闭关键字的接口，并以结构化列表输出。

**示例1输出**

![](../images/00101.jpeg)

**示例2：检查特定功能的存在**

第二个示例将检查配置文件中是否存在路由器关键字，以指示路由协议（例如`ospf`或`bgp`）是否已启用。如果模块找到它，则结果将为`True`。否则，将为`False`。这可以通过模块内的内置函数`has_line_with()`来实现：

![](../images/00102.jpeg)

这种方法可以用于设计`if`语句内的条件，我们将在下一个和最后一个示例中看到。

**示例2输出**

![](../images/00103.jpeg)

**示例3：从父级打印特定子项**：

![](../images/00104.jpeg)

在这个例子中，我们可以看到以下内容：

+   从`CiscoConfParse`模块中，我们导入了`CiscoConfParse`类。同时，我们导入了`pprint`模块，以便以可读格式打印输出以适应Python控制台输出。

+   然后，我们将`config`文件的完整路径提供给`CiscoConfParse`类。

+   我们使用了一个内置函数，例如`find_all_children()`，并且只提供了父级。这将指示`CiscoConfParse`类列出此父级下的所有配置行。

+   最后，我们遍历返回的输出（记住，它是一个列表），并检查字符串中是否存在网络关键字。如果是，则将其附加到网络列表中，并在最后打印出来。

**示例3输出：**

![](../images/00105.jpeg)

`CiscoConfParse`模块中还有许多其他可用的函数，可用于轻松从配置文件中提取数据并以结构化格式返回输出。以下是其他函数的列表：

+   `find_lineage`

+   查找行()

+   查找所有子级()

+   查找块()

+   查找有子级的父级()

+   查找有父级的子级()

+   查找没有子级的父级()

+   查找没有父级的子级()

# 使用matplotLib可视化返回的数据

俗话说，“一图胜千言”。可以从网络中提取大量信息，如接口状态、接口计数器、路由器更新、丢包、流量量等。将这些数据可视化并放入图表中将帮助您看到网络的整体情况。Python有一个名为**matplotlib**的优秀库（[https://matplotlib.org/](https://matplotlib.org/)），用于生成图表并对其进行自定义。

Matplotlib能够创建大多数类型的图表，如折线图、散点图、条形图、饼图、堆叠图、3D图和地理地图图表。

# Matplotlib安装

我们将首先使用`pip`从PYpI安装库。请注意，除了matplotlib之外，还将安装一些其他包，如`numpy`和`six`：

```py
pip install matplotlib
```

![](../images/00106.jpeg)

现在，尝试导入`matplotlib`，如果没有打印错误，则成功导入模块：

![](../images/00107.jpeg)

# Matplotlib实践

我们将从简单的示例开始，以探索matplotlib的功能。我们通常做的第一件事是将`matplotlib`导入到我们的Python脚本中：

```py
import matplotlib.pyplot as plt
```

请注意，我们将`pyplot`导入为一个简短的名称`plt`，以便在我们的脚本中使用。现在，我们将在其中使用`plot()`方法来绘制我们的数据，其中包括两个列表。第一个列表表示*x*轴的值，而第二个列表表示*y*轴的值：

```py
plt.plot([0, 1, 2, 3, 4], [0, 10, 20, 30, 40])
```

现在，这些值被放入了图表中。

最后一步是使用`show()`方法将该图表显示为窗口：

```py
plt.show()
```

![](../images/00108.jpeg)在Ubuntu中，您可能需要安装`Python-tk`才能查看图表。使用`apt install Python-tk`。

生成的图表将显示代表x轴和y轴输入值的线。在窗口中，您可以执行以下操作：

+   使用十字图标移动图表

+   调整图表大小

+   使用缩放图标放大特定区域

+   使用主页图标重置到原始视图

+   使用保存图标保存图表

您可以通过为图表添加标题和两个轴的标签来自定义生成的图表。此外，如果图表上有多条线，还可以添加解释每条线含义的图例：

```py
import matplotlib.pyplot as plt
plt.plot([0, 1, 2, 3, 4], [0, 10, 20, 30, 40]) plt.xlabel("numbers") plt.ylabel("numbers multiplied by ten") plt.title("Generated Graph\nCheck it out") plt.show()
```

![](../images/00109.jpeg)请注意，我们通常不会在Python脚本中硬编码绘制的值，而是会从网络外部获取这些值，这将在下一个示例中看到。

此外，您可以在同一图表上绘制多个数据集。您可以添加另一个代表先前图表数据的列表，`matplotlib`将绘制它。此外，您可以添加标签以区分图表上的数据集。这些标签的图例将使用`legend()`函数打印在图表上：

```py
import matplotlib.pyplot as plt
plt.plot([0, 1, 2, 3, 4], [0, 10, 20, 30, 40], label="First Line")
plt.plot([5, 6, 7, 8, 9], [50, 60, 70, 80, 90], label="Second Line") plt.xlabel("numbers") plt.ylabel("numbers multiplied by ten") plt.title("Generated Graph\nCheck it out") plt.legend() plt.show()
```

![](../images/00110.jpeg)

# 使用matplotlib可视化SNMP

在这个用例中，我们将利用`pysnmp`模块向路由器发送SNMP `GET`请求，检索特定接口的输入和输出流量速率，并使用`matplotlib`库对输出进行可视化。使用的OID是`.1.3.6.1.4.1.9.2.2.1.1.6`和`.1.3.6.1.4.1.9.2.2.1.1.8`，分别表示输入和输出速率：

```py
from pysnmp.entity.rfc3413.oneliner import cmdgen
import time
import matplotlib.pyplot as plt    cmdGen = cmdgen.CommandGenerator()   snmp_community = cmdgen.CommunityData('public') snmp_ip = cmdgen.UdpTransportTarget(('10.10.88.110', 161)) snmp_oids = [".1.3.6.1.4.1.9.2.2.1.1.6.3",".1.3.6.1.4.1.9.2.2.1.1.8.3"]   slots = 0 input_rates = [] output_rates = [] while slots <= 50:
  errorIndication, errorStatus, errorIndex, varBinds = cmdGen.getCmd(snmp_community, snmp_ip, *snmp_oids)    input_rate = str(varBinds[0]).split("=")[1].strip()
  output_rate = str(varBinds[1]).split("=")[1].strip()    input_rates.append(input_rate)
  output_rates.append(output_rate)    time.sleep(6)
  slots = slots + 1
  print slots

time_range = range(0, slots)   print input_rates
print output_rates
# plt.figure() plt.plot(time_range, input_rates, label="input rate") plt.plot(time_range, output_rates, label="output rate") plt.xlabel("time slot") plt.ylabel("Traffic Measured in bps") plt.title("Interface gig0/0/2 Traffic") 
```

```py
plt.legend() plt.show()
```

在这个例子中，我们可以看到以下内容：

+   我们从`pysnmp`模块导入了`cmdgen`，用于为路由器创建SNMP `GET`命令。我们还导入了`matplotlib`模块。

+   然后，我们使用`cmdgen`来定义Python和路由器之间的传输通道属性，并提供SNMP社区。

+   `pysnmp`将开始使用提供的OID发送SNMP GET请求，并将输出和错误（如果有）返回到`errorIndication`、`errorStatus`、`errorIndex`和`varBinds`。我们对`varBinds`感兴趣，因为它包含输入和输出流量速率的实际值。

+   注意，`varBinds` 的形式将是 `<oid> = <value>`，因此我们只提取了值，并将其添加到之前创建的相应列表中。

+   这个操作将在6秒的间隔内重复100次，以收集有用的数据。

+   最后，我们将收集到的数据提供给从 `matplotlib` 导入的 `plt`，并通过提供 `xlabel`、`ylabel`、标题和 `legends` 来自定义图表：

**脚本输出**：

![](../images/00111.jpeg)

# 总结

在本章中，我们学习了如何在Python中使用不同的工具和技术从返回的输出中提取有用的数据并对其进行操作。此外，我们使用了一个名为 `CiscoConfParse` 的特殊库来审计配置，并学习了如何可视化数据以生成吸引人的图表和报告。

在下一章中，我们将学习如何编写模板并使用它来使用 Jinja2 模板语言生成配置。
