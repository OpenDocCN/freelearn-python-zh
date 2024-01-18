# 使用 Python 和 Jinja2 生成配置

本章介绍了 YAML 格式，用于表示数据并从 Jinja2 语言创建的黄金模板生成配置。我们将在 Ansible 和 Python 中使用这两个概念来创建我们配置的数据模型存储。

在本章中，我们将涵盖以下主题：

+   什么是 YAML？

+   使用 Jinja2 构建黄金配置模板

# 什么是 YAML？

**YAML Ain’t Markup Language**（**YAML**）通常被称为数据序列化语言。它旨在是人类可读的，并将数据组织成结构化格式。编程语言可以理解 YAML 文件的内容（通常具有`.yml`或`.yaml`扩展名），并将其映射到内置数据类型。例如，当您在 Python 脚本中使用`.yaml`文件时，它将自动将内容转换为字典`{}`或列表`[]`，因此您可以对其进行处理和迭代。

YAML 规则有助于构建可读文件，因此了解它们以编写有效和格式良好的 YAML 文件非常重要。

# YAML 文件格式

在开发 YAML 文件时需要遵循一些规则。YAML 使用缩进（类似于 Python），它建立了项目之间的关系：

1.  因此，编写 YAML 文件的第一个规则是使缩进保持一致，使用空格或制表符，并且不要混合使用它们。

1.  第二条规则是在创建具有键和值的字典时使用冒号`:`（有时称为`yaml`中的关联数组）。冒号左侧的项目是键，而冒号右侧的项目是值。

1.  第三条规则是在列表中使用破折号`"-"`来分组项目。您可以在 YAML 文件中混合使用字典和列表，以有效地描述您的数据。左侧作为字典键，右侧作为字典值。您可以创建任意数量的级别以获得结构化数据：

![](img/00112.jpeg)

让我们举个例子并应用这些规则：

![](img/00113.jpeg)

有很多事情要看。首先，文件有一个顶级，`my_datacenter`，它作为顶级键，其值由它之后的所有缩进行组成，即`GW`，`switch1`和`switch2`。这些项目也作为键，并在其中有值，即`eve_port`，`device_template`，`hostname`，`mgmt_int`，`mgmt_ip`和`mgmt_subnet`，它们同时作为第 3 级键和第 2 级值。

另一件事要注意的是`enabled_ports`，它是一个键，但具有作为列表的值。我们知道这一点，因为下一级缩进是一个破折号。

请注意，所有接口都是同级元素，因为它们具有相同级别的缩进。

最后，不需要在字符串周围使用单引号或双引号。当我们将文件加载到 Python 中时，Python 会自动执行这些操作，并且还将根据缩进确定每个项目的数据类型和位置。

现在，让我们开发一个 Python 脚本，读取这个 YAML 文件，并使用`yaml`模块将其转换为字典和列表：

![](img/00114.jpeg)

在这个例子中，我们可以看到以下内容：

+   我们在 Python 脚本中导入了`yaml`模块，以处理 YAML 文件。此外，我们导入了`pprint`函数，以显示嵌套字典和列表的层次结构。

+   然后，我们使用`with`子句和`open（）`函数打开了`yaml_example.yml`文件作为`yaml_file`。

+   最后，我们使用`load（）`函数将文件加载到`yaml_data`变量中。在这个阶段，Python 解释器将分析`yaml`文件的内容并建立项目之间的关系，然后将它们转换为标准数据类型。输出可以使用`pprint（）`函数在控制台上显示。

**脚本输出**

![](img/00115.jpeg)

现在，使用标准 Python 方法访问任何信息都相当容易。例如，您可以通过使用`my_datacenter`后跟`switch1`键来访问`switch1`配置，如以下代码片段所示：

```py
pprint(yaml_data['my_datacenter']['switch1'])

{'device_template': 'vIOSL2_Template',
 'eve_port': 32769,
 'hostname': 'SW1',
 'mgmt_intf': 'gig0/0',
 'mgmt_ip': '10.10.88.111',
 'mgmt_subnet': '255.255.255.0'}    
```

此外，您可以使用简单的`for`循环迭代键，并打印任何级别的值：

```py
for device in yaml_data['my_datacenter']:
    print device

GW
switch2
switch1
```

作为最佳实践，建议您保持键名一致，仅在描述数据时更改值。例如，`hostname`，`mgmt_intf`和`mgmt_ip`项目在所有具有相同名称的设备上都存在，而它们在`.yaml`文件中的值不同。

# 文本编辑器提示

正确的缩进对于 YAML 数据非常重要。建议使用高级文本编辑器，如 Sublime Text 或 Notepad++，因为它们具有将制表符转换为特定数量的空格的选项。同时，您可以选择特定的制表符缩进大小为 2 或 4。因此，每当您点击*Tab*按钮时，您的编辑器将将制表符转换为静态数量的空格。最后，您可以选择在每个缩进处显示垂直线，以确保行缩进相同。

请注意，Microsoft Windows Notepad 没有此选项，这可能会导致 YAML 文件的格式错误。

以下是一个名为 Sublime Text 的高级编辑器的示例，可以配置为使用上述选项：

![](img/00116.jpeg)

屏幕截图显示了垂直线指南，确保当您点击 Tab 时，兄弟项目处于相同的缩进级别和空格数。

# 使用 Jinja2 构建黄金配置

大多数网络工程师都有一个文本文件，用作特定设备配置的模板。该文件包含许多值的网络配置部分。当网络工程师想要配置新设备或更改其配置时，他们基本上会用另一个文件中的特定值替换此文件中的特定值，以生成新的配置。

在本书的后面，我们将使用 Python 和 Ansible，使用 Jinja2 模板语言([`jinja.pocoo.org`](http://jinja.pocoo.org))高效地自动化此过程。 Jinja2 开发的核心概念和驱动程序是在特定网络/系统配置的所有模板文件中具有统一的语法，并将数据与实际配置分离。这使我们能够多次使用相同的模板，但使用不同的数据集。此外，正如 Jinja2 网页所示，它具有一些独特的功能，使其脱颖而出，与其他模板语言不同。

以下是官方网站上提到的一些功能：

+   强大的自动 HTML 转义系统，用于跨站点脚本预防。

+   高性能，使用即时编译到 Python 字节码。Jinja2 将在首次加载时将您的模板源代码转换为 Python 字节码，以获得最佳的运行时性能。

+   可选的提前编译。

+   易于调试，具有将模板编译和运行时错误集成到标准 Python 回溯系统的调试系统。

+   可配置的语法：例如，您可以重新配置 Jinja2 以更好地适应输出格式，例如 LaTeX 或 JavaScript。

+   模板设计帮助程序：Jinja2 附带了一系列有用的小助手，可帮助解决模板中的常见任务，例如将项目序列分成多列。

另一个重要的 Jinja 功能是*模板继承*，我们可以创建一个*基础/父模板*，为我们的系统或所有设备的 Day 0 初始配置定义基本结构。此初始配置将是基本配置，并包含通用部分，例如用户名、管理子网、默认路由和 SNMP 社区。其他*子模板*扩展基础模板并继承它。

在本章中，术语 Jinja 和 Jinja2 可以互换使用。

在我们深入研究 Jinja2 语言提供的更多功能之前，让我们先来看几个构建模板的例子：

1.  首先，我们需要确保 Jinja2 已经安装在您的系统中，使用以下命令：

```py
pip install jinja2 
```

该软件包将从 PyPi 下载，然后将安装在站点软件包中。

1.  现在，打开你喜欢的文本编辑器，并编写以下模板，它代表了一个简单的 Day 0（初始）配置，用于配置设备主机名、一些`aaa`参数、每个交换机上应存在的默认 VLAN 以及 IP 地址的管理：

```py
hostname {{ hostname }}

aaa new-model aaa session-id unique aaa authentication login default local aaa authorization exec default local none vtp mode transparent vlan 10,20,30,40,50,60,70,80,90,100,200   int {{ mgmt_intf }}
no switchport no shut ip address {{ mgmt_ip }} {{ mgmt_subnet }}
```

一些文本编辑器（如 Sublime Text 和 Notepad++）支持 Jinja2，并可以为您提供语法高亮和自动补全，无论是通过本地支持还是通过扩展。

请注意，在上一个模板中，变量是用双大括号`{{  }}`写的。因此，当 Python 脚本加载模板时，它将用所需的值替换这些变量：

```py
#!/usr/bin/python   from jinja2 import Template
template = Template(''' hostname {{hostname}}   aaa new-model aaa session-id unique aaa authentication login default local aaa authorization exec default local none vtp mode transparent vlan 10,20,30,40,50,60,70,80,90,100,200   int {{mgmt_intf}}
 no switchport no shut ip address {{mgmt_ip}} {{mgmt_subnet}} ''')   sw1 = {'hostname': 'switch1', 'mgmt_intf': 'gig0/0', 'mgmt_ip': '10.10.88.111', 'mgmt_subnet': '255.255.255.0'} print(template.render(sw1))
```

在这个例子中，我们可以看到以下内容：

+   首先，我们导入了`jinja2`模块中的`Template`类。这个类将验证和解析 Jinja2 文件。

+   然后，我们定义了一个变量`sw1`，它是一个带有与模板内变量名称相等的键的字典。字典值将是渲染模板的数据。

+   最后，我们在模板中使用了`render()`方法，该方法以`sw1`作为输入，将 Jinja2 模板与渲染值连接起来，并打印配置。

**脚本输出**

![](img/00117.jpeg)

现在，让我们改进我们的脚本，使用 YAML 来渲染模板，而不是在字典中硬编码值。这个概念很简单：我们将在 YAML 文件中建模我们实验室的`day0`配置，然后使用`yaml.load()`将该文件加载到我们的 Python 脚本中，并使用输出来填充 Jinja2 模板，从而生成每个设备的`day0`配置文件：

![](img/00118.jpeg)

首先，我们将扩展上次开发的 YAML 文件，并在保持每个节点层次结构不变的情况下，向其中添加其他设备：

```py
--- dc1:
 GW: eve_port: 32773
  device_template: vIOSL3_Template
  hostname: R1
  mgmt_intf: gig0/0
  mgmt_ip: 10.10.88.110
  mgmt_subnet: 255.255.255.0      switch1:
 eve_port: 32769
  device_template: vIOSL2_Template
  hostname: SW1
  mgmt_intf: gig0/0
  mgmt_ip: 10.10.88.111
  mgmt_subnet: 255.255.255.0    switch2:
 eve_port: 32770
  device_template: vIOSL2_Template
  hostname: SW2
  mgmt_intf: gig0/0
  mgmt_ip: 10.10.88.112
  mgmt_subnet: 255.255.255.0    switch3:
 eve_port: 32769
  device_template: vIOSL2_Template
  hostname: SW3
  mgmt_intf: gig0/0
  mgmt_ip: 10.10.88.113
  mgmt_subnet: 255.255.255.0    switch4:
 eve_port: 32770
  device_template: vIOSL2_Template
  hostname: SW4
  mgmt_intf: gig0/0
  mgmt_ip: 10.10.88.114
  mgmt_subnet: 255.255.255.0 
```

**以下是 Python 脚本：**

```py
#!/usr/bin/python __author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"   import yaml
from jinja2 import Template

with open('/media/bassim/DATA/GoogleDrive/Packt/EnterpriseAutomationProject/Chapter6_Configuration_generator_with_python_and_jinja2/network_dc.yml', 'r') as yaml_file:
  yaml_data = yaml.load(yaml_file)   router_day0_template = Template(""" hostname {{hostname}} int {{mgmt_intf}}
 no shutdown ip add {{mgmt_ip}} {{mgmt_subnet}}   lldp run   ip domain-name EnterpriseAutomation.net ip ssh version 2 ip scp server enable crypto key generate rsa general-keys modulus 1024   snmp-server community public RW snmp-server trap link ietf snmp-server enable traps snmp linkdown linkup snmp-server enable traps syslog snmp-server manager   logging history debugging logging snmp-trap emergencies logging snmp-trap alerts logging snmp-trap critical logging snmp-trap errors logging snmp-trap warnings logging snmp-trap notifications logging snmp-trap informational logging snmp-trap debugging   """)     switch_day0_template = Template(""" hostname {{hostname}}   aaa new-model aaa session-id unique aaa authentication login default local aaa authorization exec default local none vtp mode transparent vlan 10,20,30,40,50,60,70,80,90,100,200   int {{mgmt_intf}}
 no switchport no shut ip address {{mgmt_ip}} {{mgmt_subnet}}   snmp-server community public RW snmp-server trap link ietf snmp-server enable traps snmp linkdown linkup snmp-server enable traps syslog snmp-server manager   logging history debugging logging snmp-trap emergencies logging snmp-trap alerts logging snmp-trap critical logging snmp-trap errors logging snmp-trap warnings logging snmp-trap notifications logging snmp-trap informational logging snmp-trap debugging   """)   for device,config in yaml_data['dc1'].iteritems():
  if config['device_template'] == "vIOSL2_Template":
  device_template = switch_day0_template
    elif config['device_template'] == "vIOSL3_Template":
  device_template = router_day0_template

    print("rendering now device {0}" .format(device))
  Day0_device_config = device_template.render(config)    print Day0_device_config
    print "=" * 30 
```

在这个例子中，我们可以看到以下内容：

+   我们像往常一样导入了`yaml`和`Jinja2`模块

+   然后，我们指示脚本将`yaml`文件加载到`yaml_data`变量中，这将把它转换为一系列字典和列表

+   分别定义了路由器和交换机配置的两个模板，分别为`router_day0_template`和`switch_day0_template`

+   `for`循环将遍历`dc1`的设备，并检查`device_template`，然后为每个设备渲染配置

**脚本输出**

以下是路由器配置（输出已省略）：

![](img/00119.jpeg)

以下是交换机 1 的配置（输出已省略）：

![](img/00120.jpeg)

# 从文件系统中读取模板

Python 开发人员的一种常见方法是将静态的、硬编码的值和模板移出 Python 脚本，只保留脚本内的逻辑。这种方法可以使您的程序更加清晰和可扩展，同时允许其他团队成员通过更改输入来获得期望的输出，而对 Python 了解不多的人也可以使用这种方法。Jinja2 也不例外。您可以使用 Jinja2 模块中的`FileSystemLoader()`类从操作系统目录中加载模板。我们将修改我们的代码，将`router_day0_template`和`switch_day0_template`的内容从脚本中移到文本文件中，然后将它们加载到我们的脚本中。

**Python 代码**

```py
import yaml
from jinja2 import FileSystemLoader, Environment

with open('/media/bassim/DATA/GoogleDrive/Packt/EnterpriseAutomationProject/Chapter6_Configuration_generator_with_python_and_jinja2/network_dc.yml', 'r') as yaml_file:
  yaml_data = yaml.load(yaml_file)     template_dir = "/media/bassim/DATA/GoogleDrive/Packt/EnterpriseAutomationProject/Chapter6_Configuration_generator_with_python_and_jinja2"   template_env = Environment(loader=FileSystemLoader(template_dir),
  trim_blocks=True,
  lstrip_blocks= True
  )     for device,config in yaml_data['dc1'].iteritems():
  if config['device_template'] == "vIOSL2_Template":
  device_template = template_env.get_template("switch_day1_template.j2")
  elif config['device_template'] == "vIOSL3_Template":
  device_template = template_env.get_template("router_day1_template.j2")    print("rendering now device {0}" .format(device))
  Day0_device_config = device_template.render(config)    print Day0_device_config
    print "=" * 30 
```

在这个例子中，我们不再像之前那样从 Jinja2 模块中加载`Template()`类，而是导入`Environment()`和`FileSystemLoader()`，它们用于通过提供`template_dir`从特定操作系统目录中读取 Jinja2 文件，其中存储了我们的模板。然后，我们将使用创建的`template_env`对象，以及`get_template()`方法，获取模板名称并使用配置渲染它。

确保您的模板文件以`.j2`扩展名结尾。这将使 PyCharm 将文件中的文本识别为 Jinja2 模板，从而提供语法高亮和更好的代码完成。

# 使用 Jinja2 循环和条件

Jinja2 中的循环和条件用于增强我们的模板并为其添加更多功能。我们将首先了解如何在模板中添加`for`循环，以便迭代从 YAML 传递的值。例如，我们可能需要在每个接口下添加交换机配置，比如使用交换机端口模式并配置 VLAN ID，这将在访问端口下配置，或者在干线端口的情况下配置允许的 VLAN 范围。

另一方面，我们可能需要在路由器上启用一些接口并为其添加自定义配置，比如 MTU、速度和双工。因此，我们将使用`for`循环。

请注意，我们的脚本逻辑的一部分现在将从 Python 移动到 Jinja2 模板中。Python 脚本将只是从操作系统外部或通过脚本内部的`Template()`类读取模板，然后使用来自 YAML 文件的解析值渲染模板。

Jinja2 中`for`循环的基本结构如下：

```py
{% for key, value in var1.iteritems() %}
configuration snippets
{% endfor %}
```

请注意使用`{% %}`来定义 Jinja2 文件中的逻辑。

此外，`iteritems()`具有与迭代 Python 字典相同的功能，即迭代键和值对。循环将为`var1`字典中的每个元素返回键和值。

此外，我们可以有一个`if`条件来验证特定条件，如果条件为真，则配置片段将被添加到渲染文件中。基本的`if`结构如下所示：

```py
{% if enabled_ports %}
configuration snippet goes here and added to template if the condition is true
{% endif %}
```

现在，我们将修改描述数据中心设备的`.yaml`文件，并为每个设备添加接口配置和已启用的端口：

```py
--- dc1:
 GW: eve_port: 32773
  device_template: vIOSL3_Template
  hostname: R1
  mgmt_intf: gig0/0
  mgmt_ip: 10.10.88.110
  mgmt_subnet: 255.255.255.0
  enabled_ports:
  - gig0/0
  - gig0/1
  - gig0/2    switch1:
 eve_port: 32769
  device_template: vIOSL2_Template
  hostname: SW1
  mgmt_intf: gig0/0
  mgmt_ip: 10.10.88.111
  mgmt_subnet: 255.255.255.0
  interfaces:
 gig0/1: vlan: [1,10,20,200]
  description: TO_DSW2_1
  mode: trunk   gig0/2:
 vlan: [1,10,20,200]
  description: TO_DSW2_2
  mode: trunk   gig0/3:
 vlan: [1,10,20,200]
  description: TO_ASW3
  mode: trunk   gig1/0:
 vlan: [1,10,20,200]
  description: TO_ASW4
  mode: trunk
  enabled_ports:
  - gig0/0
  - gig1/1    switch2:
 eve_port: 32770
  device_template: vIOSL2_Template
  hostname: SW2
  mgmt_intf: gig0/0
  mgmt_ip: 10.10.88.112
  mgmt_subnet: 255.255.255.0
  interfaces:
 gig0/1: vlan: [1,10,20,200]
  description: TO_DSW1_1
  mode: trunk   gig0/2:
 vlan: [1,10,20,200]
  description: TO_DSW1_2
  mode: trunk
  gig0/3:
 vlan: [1,10,20,200]
  description: TO_ASW3
  mode: trunk   gig1/0:
 vlan: [1,10,20,200]
  description: TO_ASW4
  mode: trunk
  enabled_ports:
  - gig0/0
  - gig1/1    switch3:
 eve_port: 32769
  device_template: vIOSL2_Template
  hostname: SW3
  mgmt_intf: gig0/0
  mgmt_ip: 10.10.88.113
  mgmt_subnet: 255.255.255.0
  interfaces:
 gig0/1: vlan: [1,10,20,200]
  description: TO_DSW1
  mode: trunk   gig0/2:
 vlan: [1,10,20,200]
  description: TO_DSW2
  mode: trunk   gig1/0:
 vlan: 10
  description: TO_Client1
  mode: access   gig1/1:
 vlan: 20
  description: TO_Client2
  mode: access
  enabled_ports:
  - gig0/0    switch4:
 eve_port: 32770
  device_template: vIOSL2_Template
  hostname: SW4
  mgmt_intf: gig0/0
  mgmt_ip: 10.10.88.114
  mgmt_subnet: 255.255.255.0
  interfaces:
 gig0/1: vlan: [1,10,20,200]
  description: TO_DSW2
  mode: trunk   gig0/2:
 vlan: [1,10,20,200]
  description: TO_DSW1
  mode: trunk   gig1/0:
 vlan: 10
  description: TO_Client1
  mode: access   gig1/1:
 vlan: 20
  description: TO_Client2
  mode: access
  enabled_ports:
  - gig0/0
```

请注意，我们将交换机端口分类为干线端口或访问端口，并为每个端口添加 VLAN。

根据`yaml`文件，以交换机端口访问模式进入的数据包将被标记为 VLAN。在干线端口模式下，只有数据包的 VLAN ID 属于配置列表，才允许数据包进入。

现在，我们将为设备 Day 1（运行）配置创建两个额外的模板。第一个模板将是`router_day1_template`，第二个将是`switch_day1_template`，它们都将继承之前开发的相应 day0 模板：

**router_day1_template:**

```py
{% include 'router_day0_template.j2' %}   {% if enabled_ports %}
 {% for port in enabled_ports %} interface {{ port }}
    no switchport
 no shutdown mtu 1520 duplex auto speed auto  {% endfor %}   {% endif %}
```

**switch_day1_template:**

```py

{% include 'switch_day0_template.j2' %}   {% if enabled_ports %}
 {% for port in enabled_ports %} interface {{ port }}
    no switchport
 no shutdown mtu 1520 duplex auto speed auto    {% endfor %} {% endif %}   {% if interfaces %}
 {% for intf,intf_config in interfaces.items() %} interface {{ intf }}
 description "{{intf_config['description']}}"
 no shutdown duplex full  {% if intf_config['mode'] %}   {% if intf_config['mode'] == "access" %}
  switchport mode {{intf_config['mode']}}
 switchport access vlan {{intf_config['vlan']}}
   {% elif intf_config['mode'] == "trunk" %}
  switchport {{intf_config['mode']}} encapsulation dot1q
 switchport mode trunk switchport trunk allowed vlan {{intf_config['vlan']|join(',')}}
   {% endif %}
 {% endif %}
 {% endfor %} {% endif %} 
```

请注意使用`{% include <template_name.j2> %}`标签，它指的是设备的 day0 模板。

此模板将首先被渲染并填充来自 YAML 的传递值，然后填充下一个部分。

Jinja2 语言继承了许多写作风格和特性，来自 Python 语言。虽然在开发模板和插入标签时不是强制遵循缩进规则，但作者更喜欢在可读的 Jinja2 模板中使用缩进。

**脚本输出:**

```py
rendering now device GW
hostname R1
int gig0/0
  no shutdown
  ip add 10.10.88.110 255.255.255.0
lldp run
ip domain-name EnterpriseAutomation.net
ip ssh version 2
ip scp server enable
crypto key generate rsa general-keys modulus 1024
snmp-server community public RW
snmp-server trap link ietf
snmp-server enable traps snmp linkdown linkup
snmp-server enable traps syslog
snmp-server manager
logging history debugging
logging snmp-trap emergencies
logging snmp-trap alerts
logging snmp-trap critical
logging snmp-trap errors
logging snmp-trap warnings
logging snmp-trap notifications
logging snmp-trap informational
logging snmp-trap debugging
interface gig0/0
    no switchport
    no shutdown
    mtu 1520
    duplex auto
    speed auto
interface gig0/1
    no switchport
    no shutdown
    mtu 1520
    duplex auto
    speed auto
interface gig0/2
    no switchport
    no shutdown
    mtu 1520
    duplex auto
    speed auto
==============================
rendering now device switch1
hostname SW1
aaa new-model
aaa session-id unique
aaa authentication login default local
aaa authorization exec default local none
vtp mode transparent
vlan 10,20,30,40,50,60,70,80,90,100,200
int gig0/0
 no switchport
 no shut
 ip address 10.10.88.111 255.255.255.0
snmp-server community public RW
snmp-server trap link ietf
snmp-server enable traps snmp linkdown linkup
snmp-server enable traps syslog
snmp-server manager
logging history debugging
logging snmp-trap emergencies
logging snmp-trap alerts
logging snmp-trap critical
logging snmp-trap errors
logging snmp-trap warnings
logging snmp-trap notifications
logging snmp-trap informational
logging snmp-trap debugging
interface gig0/0
    no switchport
    no shutdown
    mtu 1520
    duplex auto
    speed auto
interface gig1/1
    no switchport
    no shutdown
    mtu 1520
    duplex auto
    speed auto
interface gig0/2
 description "TO_DSW2_2"
 no shutdown
 duplex full
 switchport trunk encapsulation dot1q
 switchport mode trunk
 switchport trunk allowed vlan 1,10,20,200
interface gig0/3
 description "TO_ASW3"
 no shutdown
 duplex full
 switchport trunk encapsulation dot1q
 switchport mode trunk
 switchport trunk allowed vlan 1,10,20,200
interface gig0/1
 description "TO_DSW2_1"
 no shutdown
 duplex full
 switchport trunk encapsulation dot1q
 switchport mode trunk
 switchport trunk allowed vlan 1,10,20,200
interface gig1/0
 description "TO_ASW4"
 no shutdown
 duplex full
 switchport trunk encapsulation dot1q
 switchport mode trunk
 switchport trunk allowed vlan 1,10,20,200
==============================

<switch2 output omitted>

==============================
rendering now device switch3
hostname SW3
aaa new-model
aaa session-id unique
aaa authentication login default local
aaa authorization exec default local none
vtp mode transparent
vlan 10,20,30,40,50,60,70,80,90,100,200
int gig0/0
 no switchport
 no shut
 ip address 10.10.88.113 255.255.255.0
snmp-server community public RW
snmp-server trap link ietf
snmp-server enable traps snmp linkdown linkup
snmp-server enable traps syslog
snmp-server manager
logging history debugging
logging snmp-trap emergencies
logging snmp-trap alerts
logging snmp-trap critical
logging snmp-trap errors
logging snmp-trap warnings
logging snmp-trap notifications
logging snmp-trap informational
logging snmp-trap debugging
interface gig0/0
    no switchport
    no shutdown
    mtu 1520
    duplex auto
    speed auto
interface gig0/2
 description "TO_DSW2"
 no shutdown
 duplex full
 switchport trunk encapsulation dot1q
 switchport mode trunk
 switchport trunk allowed vlan 1,10,20,200
interface gig1/1
 description "TO_Client2"
 no shutdown
 duplex full
 switchport mode access
 switchport access vlan 20
interface gig1/0
 description "TO_Client1"
 no shutdown
 duplex full
 switchport mode access
 switchport access vlan 10
interface gig0/1
 description "TO_DSW1"
 no shutdown
 duplex full
 switchport trunk encapsulation dot1q
 switchport mode trunk
 switchport trunk allowed vlan 1,10,20,200
==============================
<switch4 output omitted>
```

# 总结

在本章中，我们学习了 YAML 及其格式以及如何使用文本编辑器。我们还了解了 Jinja2 及其配置。然后，我们探讨了在 Jinja2 中使用循环和条件的方法。

在下一章中，我们将学习如何使用多进程同时实例化和执行 Python 代码。
