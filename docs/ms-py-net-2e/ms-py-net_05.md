# 第五章：Python 自动化框架-超越基础

在第一章中，*TCP/IP 协议套件和 Python 回顾*，我们看了一些基本结构，以使 Ansible 运行起来。我们使用 Ansible 清单文件、变量和 playbook。我们还看了一些使用 Cisco、Juniper 和 Arista 设备的网络模块的示例。

在本章中，我们将进一步建立在之前章节所学到的知识基础上，并深入探讨 Ansible 的更高级主题。关于 Ansible 已经写了很多书，而且 Ansible 的内容远不止我们可以在两章中涵盖的。这里的目标是介绍我认为作为网络工程师您需要的大部分 Ansible 功能和功能，并尽可能地缩短学习曲线。

需要指出的是，如果您对第四章中提出的一些观点不清楚，现在是回顾它们的好时机，因为它们是本章的先决条件。

在本章中，我们将研究以下主题：

+   Ansible 条件

+   Ansible 循环

+   模板

+   组和主机变量

+   Ansible Vault

+   Ansible 角色

+   编写自己的模块

我们有很多内容要涵盖，所以让我们开始吧！

# Ansible 条件

Ansible 条件类似于编程语言中的条件语句。在第一章中，*TCP/IP 协议套件和 Python 回顾*，我们看到 Python 使用条件语句只执行代码的一部分，使用`if.. then`或`while`语句。在 Ansible 中，它使用条件关键字只有在条件满足时才运行任务。在许多情况下，play 或任务的执行可能取决于事实、变量或上一个任务的结果。例如，如果您有一个升级路由器镜像的 play，您希望包括一步来确保新的路由器镜像在移动到下一个重启路由器的 play 之前已经在设备上。

在本节中，我们将讨论`when`子句，它支持所有模块，以及在 Ansible 网络命令模块中支持的独特条件状态。一些条件如下：

+   等于（`eq`）

+   不等于（`neq`）

+   大于（`gt`）

+   大于或等于（`ge`）

+   小于（`lt`）

+   小于或等于（`le`）

+   包含

# when 子句

`when`子句在您需要检查变量或 play 执行结果的输出并相应地采取行动时非常有用。我们在第四章中看到了`when`子句的一个快速示例，*Python 自动化框架- Ansible 基础*，当我们查看 Ansible 2.5 最佳实践结构时。如果您还记得，只有当设备的网络操作系统是 Cisco IOS 时，任务才会运行。让我们在`chapter5_1.yml`中看另一个使用它的例子：

```py
    ---
    - name: IOS Command Output
      hosts: "iosv-devices"
      gather_facts: false
      connection: local
      vars:
        cli:
          host: "{{ ansible_host }}"
          username: "{{ username }}"
          password: "{{ password }}"
          transport: cli
      tasks:
        - name: show hostname
          ios_command:
            commands:
              - show run | i hostname
                provider: "{{ cli }}"
            register: output
        - name: show output
          when: '"iosv-2" in "{{ output.stdout }}"'
          debug:
            msg: '{{ output }}'
```

我们在这个 playbook 中看到了之前在第四章中的所有元素，*Python 自动化框架- Ansible 基础**，直到第一个任务结束。在 play 的第二个任务中，我们使用`when`子句来检查输出是否包含`iosv-2`关键字。如果是，我们将继续执行任务，该任务使用 debug 模块来显示输出。当 playbook 运行时，我们将看到以下输出：

```py
    <skip>
    TASK [show output]  
    *************************************************************
    skipping: [ios-r1]
 ok: [ios-r2] => {
 "msg": {
 "changed": false,
 "stdout": [
 "hostname iosv-2"
 ],
 "stdout_lines": [
 [
 "hostname iosv-2"
 ]
 ],
 "warnings": []
 }
 }
    <skip>
```

我们可以看到`iosv-r1`设备被跳过了，因为条件没有通过。我们可以在`chapter5_2.yml`中进一步扩展这个例子，只有当条件满足时才应用某些配置更改：

```py
    <skip> 
    tasks:
      - name: show hostname
        ios_command:
          commands:
            - show run | i hostname
          provider: "{{ cli }}"
        register: output
      - name: config example
        when: '"iosv-2" in "{{ output.stdout }}"'
        ios_config:
          lines:
            - logging buffered 30000
          provider: "{{ cli }}"
```

我们可以在这里看到执行输出：

```py
 TASK [config example] 
 **********************************************************
 skipping: [ios-r1]
 changed: [ios-r2] 
 PLAY RECAP 
 ***********************************************************
 ios-r1 : ok=1 changed=0 unreachable=0 failed=0
 ios-r2 : ok=2 changed=1 unreachable=0 failed=0
```

再次注意执行输出中`ios-r2`是唯一应用的更改，而`ios-r1`被跳过。在这种情况下，日志缓冲区大小只在`ios-r2`上更改。

`when`子句在使用设置或事实模块时也非常有用-您可以根据最初收集的一些`事实`来采取行动。例如，以下语句将确保只有主要版本为`16`的 Ubuntu 主机将受到条件语句的影响：

```py
when: ansible_os_family == "Debian" and ansible_lsb.major_release|int >= 16
```

有关更多条件，请查看 Ansible 条件文档([`docs.ansible.com/ansible/playbooks_conditionals.html`](http://docs.ansible.com/ansible/playbooks_conditionals.html))。

# Ansible 网络事实

在 2.5 之前，Ansible 网络配送了许多特定于网络的事实模块。网络事实模块存在，但供应商之间的命名和使用方式不同。从 2.5 版本开始，Ansible 开始标准化其网络事实模块的使用。Ansible 网络事实模块从系统中收集信息，并将结果存储在以`ansible_net_`为前缀的事实中。这些模块收集的数据在模块文档中有记录的*返回值*中。这对于 Ansible 网络模块来说是一个相当重要的里程碑，因为它默认情况下可以为您抽象出事实收集过程的大部分繁重工作。

让我们使用在第四章中看到的相同结构，*Python 自动化框架- Ansible 基础*，Ansible 2.5 最佳实践，但扩展它以查看`ios_facts`模块如何用于收集事实。回顾一下，我们的清单文件包含两个 iOS 主机，主机变量驻留在`host_vars`目录中：

```py
$ cat hosts
[ios-devices]
iosv-1
iosv-2

$ cat host_vars/iosv-1
---
ansible_host: 172.16.1.20
ansible_user: cisco
ansible_ssh_pass: cisco
ansible_connection: network_cli
ansible_network_os: ios
ansbile_become: yes
ansible_become_method: enable
ansible_become_pass: cisco
```

我们的 playbook 将有三个任务。第一个任务将使用`ios_facts`模块为我们的两个网络设备收集事实。第二个任务将显示为每个设备收集和存储的某些事实。您将看到我们显示的事实是默认的`ansible_net`事实，而不是来自第一个任务的已注册变量。第三个任务将显示我们为`iosv-1`主机收集的所有事实：

```py
$ cat my_playbook.yml
---
- name: Chapter 5 Ansible 2.5 network facts
 connection: network_cli
 gather_facts: false
 hosts: all
 tasks:
 - name: Gathering facts via ios_facts module
 ios_facts:
 when: ansible_network_os == 'ios'

 - name: Display certain facts
 debug:
 msg: "The hostname is {{ ansible_net_hostname }} running {{ ansible_net_version }}"

 - name: Display all facts for a host
 debug:
 var: hostvars['iosv-1']
```

当我们运行 playbook 时，您会看到前两个任务的结果是我们预期的：

```py
$ ansible-playbook -i hosts my_playbook.yml

PLAY [Chapter 5 Ansible 2.5 network facts] *************************************

TASK [Gathering facts via ios_facts module] ************************************
ok: [iosv-2]
ok: [iosv-1]

TASK [Display certain facts] ***************************************************
ok: [iosv-2] => {
 "msg": "The hostname is iosv-2 running 15.6(3)M2"
}
ok: [iosv-1] => {
 "msg": "The hostname is iosv-1 running 15.6(3)M2"
}
```

第三个任务将显示为 iOS 设备收集的所有网络设备事实。已经收集了大量有关 iOS 设备的信息，可以帮助您进行网络自动化需求：

```py
TASK [Display all facts for a host] ********************************************
ok: [iosv-1] => {
 "hostvars['iosv-1']": {
 "ansbile_become": true,
 "ansible_become_method": "enable",
 "ansible_become_pass": "cisco",
 "ansible_check_mode": false,
 "ansible_connection": "network_cli",
 "ansible_diff_mode": false,
 "ansible_facts": {
 "net_all_ipv4_addresses": [
 "10.0.0.5",
 "172.16.1.20",
 "192.168.0.1"
 ],
 "net_all_ipv6_addresses": [],
 "net_filesystems": [
 "flash0:"
 ],
 "net_gather_subset": [
 "hardware",
 "default",
 "interfaces"
 ],
 "net_hostname": "iosv-1",
 "net_image": "flash0:/vios-adventerprisek9-m",
 "net_interfaces": {
 "GigabitEthernet0/0": {
 "bandwidth": 1000000,
 "description": "OOB Management",
 "duplex": "Full",
 "ipv4": [
 {
 "address": "172.16.1.20",
 "subnet": "24"
 }
[skip]
```

Ansible 2.5 中的网络事实模块是简化工作流程的重要一步，并使其与其他服务器模块齐头并进。

# 网络模块条件

让我们通过使用我们在本章开头看到的比较关键字来查看另一个网络设备条件示例。我们可以利用 IOSv 和 Arista EOS 都以 JSON 格式提供`show`命令的输出这一事实。例如，我们可以检查接口的状态：

```py
 arista1#sh interfaces ethernet 1/3 | json
 {
 "interfaces": {
 "Ethernet1/3": {
 "interfaceStatistics": {
 <skip>
 "outPktsRate": 0.0
 },
 "name": "Ethernet1/3",
 "interfaceStatus": "disabled",
 "autoNegotiate": "off",
 <skip>
 }
 arista1#
```

如果我们有一个操作要执行，并且它取决于`Ethernet1/3`被禁用以确保没有用户影响，比如确保没有用户连接到`Ethernet1/3`，我们可以在`chapter5_3.yml`剧本中使用以下任务。它使用`eos_command`模块来收集接口状态输出，并在继续下一个任务之前使用`waitfor`和`eq`关键字来检查接口状态：

```py
    <skip>
     tasks:
       - name: "sh int ethernet 1/3 | json"
         eos_command:
           commands:
             - "show interface ethernet 1/3 | json"
           provider: "{{ cli }}"
           waitfor:
             - "result[0].interfaces.Ethernet1/3.interfaceStatus eq 
    disabled"
         register: output
       - name: show output
         debug:
           msg: "Interface Disabled, Safe to Proceed"
```

在满足条件后，将执行第二个任务：

```py
 TASK [sh int ethernet 1/3 | json] 
 **********************************************
 ok: [arista1]

 TASK [show output] 
 *************************************************************
 ok: [arista1] => {
 "msg": "Interface Disabled, Safe to Proceed"
 }
```

如果接口处于活动状态，则将在第一个任务后给出错误如下：

```py
 TASK [sh int ethernet 1/3 | json] 
 **********************************************
 fatal: [arista1]: FAILED! => {"changed": false, "commands": ["show 
 interface ethernet 1/3 | json | json"], "failed": true, "msg": 
 "matched error in response: show interface ethernet 1/3 | json | 
 jsonrn% Invalid input (privileged mode required)rn********1>"}
 to retry, use: --limit 
 @/home/echou/Master_Python_Networking/Chapter5/chapter5_3.retry

 PLAY RECAP 
 ******************************************************************
 arista1 : ok=0 changed=0 unreachable=0 failed=1
```

查看其他条件，如`包含`，`大于`和`小于`，因为它们符合您的情况。

# Ansible 循环

Ansible 在 playbook 中提供了许多循环，例如标准循环，循环文件，子元素，do-until 等等。在本节中，我们将看两种最常用的循环形式：标准循环和循环哈希值。

# 标准循环

playbook 中的标准循环经常用于轻松多次执行类似任务。标准循环的语法非常简单：`{{ item }}`变量是在`with_items`列表上循环的占位符。例如，看一下`chapter5_4.yml` playbook 中的以下部分：

```py
      tasks:
        - name: echo loop items
          command: echo {{ item }}
          with_items: ['r1', 'r2', 'r3', 'r4', 'r5']   
```

它将使用相同的`echo`命令循环遍历五个列表项：

```py
TASK [echo loop items] *********************************************************
changed: [192.168.199.185] => (item=r1)
changed: [192.168.199.185] => (item=r2)
changed: [192.168.199.185] => (item=r3)
changed: [192.168.199.185] => (item=r4)
changed: [192.168.199.185] => (item=r5)
```

我们将在`chapter5_5.yml` playbook 中将标准循环与网络命令模块相结合，以向设备添加多个 VLAN：

```py
 tasks:
   - name: add vlans
     eos_config:
       lines:
           - vlan {{ item }}
       provider: "{{ cli }}"
     with_items:
         - 100
         - 200
         - 300
```

`with_items`列表也可以从变量中读取，这样可以更灵活地构建 playbook 的结构：

```py
vars:
  vlan_numbers: [100, 200, 300]
<skip>
tasks:
  - name: add vlans
    eos_config:
      lines:
          - vlan {{ item }}
      provider: "{{ cli }}"
    with_items: "{{ vlan_numbers }}"
```

标准循环在执行 playbook 中的冗余任务时是一个很好的时间节省器。它还通过减少任务所需的行数使 playbook 更易读。

在下一节中，我们将看看如何循环遍历字典。

# 循环遍历字典

循环遍历一个简单的列表很好。然而，我们经常有一个带有多个属性的实体。如果您考虑上一节中的`vlan`示例，每个`vlan`都会有一些独特的属性，比如`vlan`描述，网关 IP 地址，可能还有其他属性。通常，我们可以使用字典来表示实体，以将多个属性合并到其中。

让我们在上一节中的`vlan`示例中扩展为`chapter5_6.yml`中的字典示例。我们为三个`vlan`定义了字典值，每个值都有一个嵌套字典，用于描述和 IP 地址：

```py
    <skip> 
    vars:
       cli:
         host: "{{ ansible_host }}"
         username: "{{ username }}"
         password: "{{ password }}"
         transport: cli
       vlans: {
           "100": {"description": "floor_1", "ip": "192.168.10.1"},
           "200": {"description": "floor_2", "ip": "192.168.20.1"}
           "300": {"description": "floor_3", "ip": "192.168.30.1"}
       }
```

我们可以通过使用每个项目的键作为`vlan`号来配置第一个任务`add vlans`：

```py
     tasks:
       - name: add vlans
         nxos_config:
           lines:
             - vlan {{ item.key }}
           provider: "{{ cli }}"
         with_dict: "{{ vlans }}"
```

我们可以继续配置`vlan`接口。请注意，我们使用`parents`参数来唯一标识应该针对哪个部分检查命令。这是因为描述和 IP 地址都是在配置中的`interface vlan <number>`子部分下配置的：

```py
  - name: configure vlans
    nxos_config:
       lines:
         - description {{ item.value.name }}
         - ip address {{ item.value.ip }}/24
       provider: "{{ cli }}"
       parents: interface vlan {{ item.key }}
    with_dict: "{{ vlans }}"
```

执行时，您将看到字典被循环遍历：

```py
TASK [configure vlans] *********************************************************
changed: [nxos-r1] => (item={'key': u'300', 'value': {u'ip': u'192.168.30.1', u'name': u'floor_3'}})
changed: [nxos-r1] => (item={'key': u'200', 'value': {u'ip': u'192.168.20.1', u'name': u'floor_2'}})
changed: [nxos-r1] => (item={'key': u'100', 'value': {u'ip': u'192.168.10.1', u'name': u'floor_1'}})
```

让我们检查所需的配置是否应用到设备上：

```py
nx-osv-1# sh run | i vlan
<skip>
vlan 1,10,100,200,300
nx-osv-1#
```

```py
nx-osv-1# sh run | section "interface Vlan100"
interface Vlan100
 description floor_1
 ip address 192.168.10.1/24
nx-osv-1#
```

有关 Ansible 的更多循环类型，请随时查看文档（[`docs.ansible.com/ansible/playbooks_loops.html`](http://docs.ansible.com/ansible/playbooks_loops.html)）。

循环遍历字典在第一次使用时需要一些练习。但就像标准循环一样，循环遍历字典将成为您工具箱中的一个宝贵工具。

# 模板

就我所记，作为一名网络工程师，我一直在使用一种网络模板。根据我的经验，许多网络设备的网络配置部分是相同的，特别是如果这些设备在网络中担任相同的角色。

大多数情况下，当我们需要为新设备进行配置时，我们使用相同的模板形式的配置，替换必要的字段，并将文件复制到新设备上。使用 Ansible，您可以使用模板模块（[`docs.ansible.com/ansible/template_module.html`](http://docs.ansible.com/ansible/template_module.html)）自动化所有工作。

我们正在使用的基本模板文件利用了 Jinja2 模板语言（[`jinja.pocoo.org/docs/`](http://jinja.pocoo.org/docs/)）。我们在第四章中简要讨论了 Jinja2 模板语言，*Python 自动化框架- Ansible 基础*，我们将在这里更多地了解它。就像 Ansible 一样，Jinja2 有自己的语法和循环和条件的方法；幸运的是，我们只需要了解它的基础知识就足够了。Ansible 模板是我们日常任务中将要使用的重要工具，我们将在本节中更多地探索它。我们将通过逐渐从简单到更复杂地构建我们的 playbook 来学习语法。

模板使用的基本语法非常简单；你只需要指定源文件和要复制到的目标位置。

现在我们将创建一个空文件：

```py
$ touch file1
```

然后，我们将使用以下 playbook 将`file1`复制到`file2`。请注意，playbook 仅在控制机上执行。接下来，我们将为`template`模块的参数指定源文件和目标文件的路径：

```py
---
- name: Template Basic
  hosts: localhost

  tasks:
    - name: copy one file to another
      template:
        src=./file1
        dest=./file2
```

在 playbook 执行期间，我们不需要指定主机文件，因为默认情况下 localhost 是可用的。但是，你会收到一个警告：

```py
$ ansible-playbook chapter5_7.yml
 [WARNING]: provided hosts list is empty, only localhost is available
<skip>
TASK [copy one file to another] ************************************************

changed: [localhost]
<skip>
```

源文件可以有任何扩展名，但由于它们是通过 Jinja2 模板引擎处理的，让我们创建一个名为`nxos.j2`的文本文件作为模板源。模板将遵循 Jinja2 的惯例，使用双大括号来指定变量：

```py
    hostname {{ item.value.hostname }}
    feature telnet
    feature ospf
    feature bgp
    feature interface-vlan

    username {{ item.value.username }} password {{ item.value.password 
    }} role network-operator
```

# Jinja2 模板

让我们也相应地修改 playbook。在`chapter5_8.yml`中，我们将进行以下更改：

1.  将源文件更改为`nxos.j2`

1.  将目标文件更改为一个变量

1.  提供作为字典的变量值，我们将在模板中进行替换：

```py
    ---
    - name: Template Looping
      hosts: localhost

      vars:
        nexus_devices: {
          "nx-osv-1": {"hostname": "nx-osv-1", "username": "cisco", 
    "password": "cisco"}
        }

      tasks:
        - name: create router configuration files
          template:
            src=./nxos.j2
            dest=./{{ item.key }}.conf
          with_dict: "{{ nexus_devices }}"
```

运行 playbook 后，你会发现名为`nx-osv-1.conf`的目标文件已经填充好，可以使用了：

```py
$ cat nx-osv-1.conf
hostname nx-osv-1

feature telnet
feature ospf
feature bgp
feature interface-vlan

username cisco password cisco role network-operator
```

# Jinja2 循环

我们还可以在 Jinja2 中循环遍历列表和字典。我们将在`nxos.j2`中使用这两种循环：

```py
    {% for vlan_num in item.value.vlans %}
    vlan {{ vlan_num }}
    {% endfor %}

    {% for vlan_interface in item.value.vlan_interfaces %}
    interface {{ vlan_interface.int_num }}
      ip address {{ vlan_interface.ip }}/24
    {% endfor %}
```

在`chapter5_8.yml` playbook 中提供额外的列表和字典变量：

```py
   vars:
     nexus_devices: {
       "nx-osv-1": {
       "hostname": "nx-osv-1",
       "username": "cisco",
       "password": "cisco",
       "vlans": [100, 200, 300],
       "vlan_interfaces": [
          {"int_num": "100", "ip": "192.168.10.1"},
          {"int_num": "200", "ip": "192.168.20.1"},
          {"int_num": "300", "ip": "192.168.30.1"}
        ]
       }
     }
```

运行 playbook，你会看到路由器配置中`vlan`和`vlan_interfaces`的配置都已填写好。

# Jinja2 条件

Jinja2 还支持`if`条件检查。让我们在某些设备上打开 netflow 功能的字段中添加这个条件。我们将在`nxos.j2`模板中添加以下内容：

```py
    {% if item.value.netflow_enable %}
    feature netflow
    {% endif %}
```

我们将列出 playbook 中的差异：

```py
    vars:
      nexus_devices: {
      <skip>
             "netflow_enable": True
      <skip>
     }
```

我们将采取的最后一步是通过将`nxos.j2`放置在`true-false`条件检查中，使其更具可扩展性。在现实世界中，我们往往会有多个设备了解`vlan`信息，但只有一个设备作为客户端主机的网关：

```py
    {% if item.value.l3_vlan_interfaces %}
    {% for vlan_interface in item.value.vlan_interfaces %}
    interface {{ vlan_interface.int_num }}
     ip address {{ vlan_interface.ip }}/24
    {% endfor %}
    {% endif %}
```

我们还将在 playbook 中添加第二个设备，名为`nx-osv-2`：

```py
     vars:
       nexus_devices: {
       <skip>
         "nx-osv-2": {
           "hostname": "nx-osv-2",
           "username": "cisco",
           "password": "cisco",
           "vlans": [100, 200, 300],
           "l3_vlan_interfaces": False,
           "netflow_enable": False
         }
        <skip>
     }
```

我们现在准备运行我们的 playbook：

```py
$ ansible-playbook chapter5_8.yml
 [WARNING]: provided hosts list is empty, only localhost is available. Note
that the implicit localhost does not match 'all'

PLAY [Template Looping] ********************************************************

TASK [Gathering Facts] *********************************************************
ok: [localhost]

TASK [create router configuration files] ***************************************
ok: [localhost] => (item={'value': {u'username': u'cisco', u'password': u'cisco', u'hostname': u'nx-osv-2', u'netflow_enable': False, u'vlans': [100, 200, 300], u'l3_vlan_interfaces': False}, 'key': u'nx-osv-2'})
ok: [localhost] => (item={'value': {u'username': u'cisco', u'password': u'cisco', u'hostname': u'nx-osv-1', u'vlan_interfaces': [{u'int_num': u'100', u'ip': u'192.168.10.1'}, {u'int_num': u'200', u'ip': u'192.168.20.1'}, {u'int_num': u'300', u'ip': u'192.168.30.1'}], u'netflow_enable': True, u'vlans': [100, 200, 300], u'l3_vlan_interfaces': True}, 'key': u'nx-osv-1'})

PLAY RECAP *********************************************************************
localhost : ok=2 changed=0 unreachable=0 failed=0
```

让我们检查两个配置文件的差异，以确保条件性的更改正在发生：

```py
$ cat nx-osv-1.conf
hostname nx-osv-1

feature telnet
feature ospf
feature bgp
feature interface-vlan

feature netflow

username cisco password cisco role network-operator

vlan 100
vlan 200
vlan 300

interface 100
 ip address 192.168.10.1/24
interface 200
 ip address 192.168.20.1/24
interface 300
 ip address 192.168.30.1/24

$ cat nx-osv-2.conf
hostname nx-osv-2

feature telnet
feature ospf
feature bgp
feature interface-vlan

username cisco password cisco role network-operator

vlan 100
vlan 200
vlan 300
```

很整洁，对吧？这肯定可以为我们节省大量时间，以前需要重复复制和粘贴。对我来说，模板模块是一个重大的改变。几年前，这个模块就足以激励我学习和使用 Ansible。

我们的 playbook 变得有点长了。在下一节中，我们将看到如何通过将变量文件转移到组和目录中来优化 playbook。

# 组和主机变量

请注意，在之前的 playbook`chapter5_8.yml`中，我们在`nexus_devices`变量下的两个设备的用户名和密码变量中重复了自己：

```py
    vars:
      nexus_devices: {
        "nx-osv-1": {
          "hostname": "nx-osv-1",
          "username": "cisco",
          "password": "cisco",
          "vlans": [100, 200, 300],
        <skip>
        "nx-osv-2": {
          "hostname": "nx-osv-2",
          "username": "cisco",
          "password": "cisco",
          "vlans": [100, 200, 300],
        <skip>
```

这并不理想。如果我们需要更新用户名和密码的值，我们需要记住在两个位置更新。这增加了管理负担，也增加了出错的机会。作为最佳实践，Ansible 建议我们使用`group_vars`和`host_vars`目录来分离变量。

有关更多 Ansible 最佳实践，请查看[`docs.ansible.com/ansible/playbooks_best_practices.html`](http://docs.ansible.com/ansible/playbooks_best_practices.html)。

# 组变量

默认情况下，Ansible 将在与 playbook 同一目录中寻找组变量，称为`group_vars`，用于应用于组的变量。默认情况下，它将在清单文件中匹配组名的文件名。例如，如果我们在清单文件中有一个名为`[nexus-devices]`的组，我们可以在`group_vars`下有一个名为`nexus-devices`的文件，其中包含可以应用于该组的所有变量。

我们还可以使用名为`all`的特殊文件来包含应用于所有组的变量。

我们将利用此功能来处理我们的用户名和密码变量。首先，我们将创建`group_vars`目录：

```py
$ mkdir group_vars
```

然后，我们可以创建一个名为`all`的 YAML 文件来包含用户名和密码：

```py
$ cat group_vars/all
---
username: cisco
password: cisco
```

然后我们可以在 playbook 中使用变量：

```py
    vars:
      nexus_devices: {
       "nx-osv-1": {
          "hostname": "nx-osv-1",
          "username": "{{ username }}",
          "password": "{{ password }}",
          "vlans": [100, 200, 300],
        <skip>
         "nx-osv-2": {
          "hostname": "nx-osv-2",
          "username": "{{ username }}",
          "password": "{{ password }}",
          "vlans": [100, 200, 300],
        <skip>
```

# 主机变量

我们可以进一步以与组变量相同的格式分离主机变量。这就是我们能够在第四章中应用变量的 Ansible 2.5 playbook 示例以及本章前面部分的方法：

```py
$ mkdir host_vars
```

在我们的情况下，我们在本地主机上执行命令，因此`host_vars`下的文件应该相应地命名，例如`host_vars/localhost`。在我们的`host_vars/localhost`文件中，我们还可以保留在`group_vars`中声明的变量：

```py
$ cat host_vars/localhost
---
"nexus_devices":
 "nx-osv-1":
 "hostname": "nx-osv-1"
 "username": "{{ username }}"
 "password": "{{ password }}"
 "vlans": [100, 200, 300]
 "l3_vlan_interfaces": True
 "vlan_interfaces": [
 {"int_num": "100", "ip": "192.168.10.1"},
 {"int_num": "200", "ip": "192.168.20.1"},
 {"int_num": "300", "ip": "192.168.30.1"}
 ]
 "netflow_enable": True

 "nx-osv-2":
 "hostname": "nx-osv-2"
 "username": "{{ username }}"
 "password": "{{ password }}"
 "vlans": [100, 200, 300]
 "l3_vlan_interfaces": False
 "netflow_enable": False
```

在我们分离变量之后，playbook 现在变得非常轻量，只包含我们操作的逻辑：

```py
 $ cat chapter5_9.yml
 ---
 - name: Ansible Group and Host Variables
 hosts: localhost

 tasks:
 - name: create router configuration files
 template:
 src=./nxos.j2
 dest=./{{ item.key }}.conf
 with_dict: "{{ nexus_devices }}"
```

`group_vars`和`host_vars`目录不仅减少了我们的操作开销，还可以通过允许我们使用 Ansible Vault 加密敏感信息来帮助保护文件，接下来我们将看一下。

# Ansible Vault

从前一节中可以看出，在大多数情况下，Ansible 变量提供敏感信息，如用户名和密码。最好在变量周围采取一些安全措施，以便我们可以对其进行保护。Ansible Vault（[`docs.ansible.com/ansible/2.5/user_guide/vault.html`](https://docs.ansible.com/ansible/2.5/user_guide/vault.html)）为文件提供加密，使其呈现为明文。

所有 Ansible Vault 函数都以`ansible-vault`命令开头。您可以通过 create 选项手动创建加密文件。系统会要求您输入密码。如果您尝试查看文件，您会发现文件不是明文。如果您已经下载了本书的示例，我使用的密码只是单词*password*：

```py
$ ansible-vault create secret.yml
Vault password: <password>

$ cat secret.yml
$ANSIBLE_VAULT;1.1;AES256
336564626462373962326635326361323639323635353630646665656430353261383737623<skip>653537333837383863636530356464623032333432386139303335663262
3962
```

编辑或查看加密文件，我们将使用`edit`选项编辑或通过`view`选项查看文件：

```py
$ ansible-vault edit secret.yml 
Vault password:

$ ansible-vault view secret.yml 
Vault password:
```

让我们加密`group_vars/all`和`host_vars/localhost`变量文件：

```py
$ ansible-vault encrypt group_vars/all host_vars/localhost
Vault password:
Encryption successful
```

现在，当我们运行 playbook 时，我们将收到解密失败的错误消息：

```py
ERROR! Decryption failed on /home/echou/Master_Python_Networking/Chapter5/Vaults/group_vars/all
```

当我们运行 playbook 时，我们需要使用`--ask-vault-pass`选项：

```py
$ ansible-playbook chapter5_10.yml --ask-vault-pass
Vault password:
```

对于任何访问的 Vault 加密文件，解密将在内存中进行。

在 Ansible 2.4 之前，Ansible Vault 要求所有文件都使用相同的密码进行加密。自 Ansible 2.4 及以后版本，您可以使用 vault ID 来提供不同的密码文件（[`docs.ansible.com/ansible/2.5/user_guide/vault.html#multiple-vault-passwords`](https://docs.ansible.com/ansible/2.5/user_guide/vault.html#multiple-vault-passwords)）。

我们还可以将密码保存在文件中，并确保特定文件具有受限权限：

```py
$ chmod 400 ~/.vault_password.txt
$ ls -lia ~/.vault_password.txt 
809496 -r-------- 1 echou echou 9 Feb 18 12:17 /home/echou/.vault_password.txt
```

然后，我们可以使用`--vault-password-file`选项执行 playbook：

```py
$ ansible-playbook chapter5_10.yml --vault-password-file ~/.vault_password.txt
```

我们还可以仅加密一个字符串，并使用`encrypt_string`选项将加密的字符串嵌入到 playbook 中（[`docs.ansible.com/ansible/2.5/user_guide/vault.html#use-encrypt-string-to-create-encrypted-variables-to-embed-in-yaml`](https://docs.ansible.com/ansible/2.5/user_guide/vault.html#use-encrypt-string-to-create-encrypted-variables-to-embed-in-yaml)）：

```py
$ ansible-vault encrypt_string
New Vault password:
Confirm New Vault password:
Reading plaintext input from stdin. (ctrl-d to end input)
new_user_password
!vault |
 $ANSIBLE_VAULT;1.1;AES256
 616364386438393262623139623561613539656664383834643338323966623836343737373361326134663232623861313338383534613865303864616364380a626365393665316133616462643831653332663263643734363863666632636464636563616265303665626364636562316635636462323135663163663331320a62356361326639333165393962663962306630303761656435633966633437613030326633336438366264626464366138323666376239656633623233353832

Encryption successful
```

然后可以将字符串放置在 playbook 文件中作为变量。在下一节中，我们将使用`include`和`roles`进一步优化我们的 playbook。

# Ansible 包括和角色

处理复杂任务的最佳方法是将它们分解成更小的部分。当然，这种方法在 Python 和网络工程中都很常见。在 Python 中，我们将复杂的代码分解成函数、类、模块和包。在网络中，我们也将大型网络分成机架、行、集群和数据中心等部分。在 Ansible 中，我们可以使用`roles`和`includes`将大型 playbook 分割和组织成多个文件。拆分大型 Ansible playbook 简化了结构，因为每个文件都专注于较少的任务。它还允许 playbook 的各个部分被重复使用。

# Ansible 包含语句

随着 playbook 的规模不断增长，最终会显而易见，许多任务和操作可以在不同的 playbook 之间共享。Ansible`include`语句类似于许多 Linux 配置文件，只是告诉机器扩展文件的方式与直接编写文件的方式相同。我们可以在 playbook 和任务中使用 include 语句。在这里，我们将看一个扩展我们任务的简单示例。

假设我们想要显示两个不同 playbook 的输出。我们可以制作一个名为`show_output.yml`的单独的 YAML 文件作为附加任务：

```py
    ---
    - name: show output
        debug:
          var: output
```

然后，我们可以在多个 playbook 中重用此任务，例如在`chapter5_11_1.yml`中，它与上一个 playbook 几乎相同，只是在最后注册输出和包含语句方面有所不同：

```py
    ---
    - name: Ansible Group and Host Varibles
      hosts: localhost

      tasks:
        - name: create router configuration files
          template:
            src=./nxos.j2
            dest=./{{ item.key }}.conf
          with_dict: "{{ nexus_devices }}"
          register: output

        - include: show_output.yml
```

另一个 playbook，`chapter5_11_2.yml`，可以以相同的方式重用`show_output.yml`：

```py
    ---
    - name: show users
      hosts: localhost

      tasks:
        - name: show local users
          command: who
          register: output

        - include: show_output.yml
```

请注意，两个 playbook 使用相同的变量名`output`，因为在`show_output.yml`中，我们为简单起见硬编码了变量名。您还可以将变量传递到包含的文件中。

# Ansible 角色

Ansible 角色将逻辑功能与物理主机分开，以更好地适应您的网络。例如，您可以构建角色，如 spines、leafs、core，以及 Cisco、Juniper 和 Arista。同一物理主机可以属于多个角色；例如，设备可以同时属于 Juniper 和核心。这种灵活性使我们能够执行操作，例如升级所有 Juniper 设备，而不必担心设备在网络层中的位置。

Ansible 角色可以根据已知的文件基础结构自动加载某些变量、任务和处理程序。关键是这是一个已知的文件结构，我们会自动包含。实际上，您可以将角色视为 Ansible 预先制作的`include`语句。

Ansible playbook 角色文档（[`docs.ansible.com/ansible/playbooks_roles.html#roles`](http://docs.ansible.com/ansible/playbooks_roles.html#roles)）描述了我们可以配置的角色目录列表。我们不需要使用所有这些目录。在我们的示例中，我们只会修改“tasks 和 vars”文件夹。但是，了解 Ansible 角色目录结构中所有可用选项是很好的。

以下是我们将用作角色示例的内容：

```py
├── chapter5_12.yml
├── chapter5_13.yml
├── hosts
└── roles
 ├── cisco_nexus
 │   ├── defaults
 │   ├── files
 │   ├── handlers
 │   ├── meta
 │   ├── tasks
 │   │   └── main.yml
 │   ├── templates
 │   └── vars
 │       └── main.yml
 └── spines
 ├── defaults
 ├── files
 ├── handlers
 ├── tasks
 │   └── main.yml
 ├── templates
 └── vars
 └── main.yml
```

您可以看到，在顶层，我们有主机文件以及 playbooks。我们还有一个名为`roles`的文件夹。在文件夹内，我们定义了两个角色：`cisco_nexus`和`spines`。大多数角色下的子文件夹都是空的，除了“tasks 和 vars”文件夹。每个文件夹内都有一个名为`main.yml`的文件。这是默认行为：main.yml 文件是您在 playbook 中指定角色时自动包含的入口点。如果您需要拆分其他文件，可以在 main.yml 文件中使用 include 语句。

这是我们的情景：

+   我们有两个 Cisco Nexus 设备，`nxos-r1`和`nxos-r2`。我们将为它们所有配置日志服务器以及日志链路状态，利用`cisco_nexus`角色。

+   此外，nxos-r1 也是一个脊柱设备，我们将希望配置更详细的日志记录，也许是因为脊柱在我们的网络中处于更关键的位置。

对于我们的`cisco_nexus`角色，我们在`roles/cisco_nexus/vars/main.yml`中有以下变量：

```py
---
cli:
  host: "{{ ansible_host }}"
  username: cisco
  password: cisco
  transport: cli
```

我们在`roles/cisco_nexus/tasks/main.yml`中有以下配置任务：

```py
---
- name: configure logging parameters
  nxos_config:
    lines:
      - logging server 191.168.1.100
      - logging event link-status default
    provider: "{{ cli }}"
```

我们的 playbook 非常简单，因为它只需要指定我们想要根据`cisco_nexus`角色配置的主机：

```py
---
- name: playbook for cisco_nexus role
  hosts: "cisco_nexus"
  gather_facts: false
  connection: local

  roles:
    - cisco_nexus
```

当您运行 playbook 时，playbook 将包括在`cisco_nexus`角色中定义的任务和变量，并相应地配置设备。

对于我们的`spine`角色，我们将在`roles/spines/tasks/mail.yml`中有一个额外的更详细的日志记录任务：

```py
---
- name: change logging level
  nxos_config:
    lines:
      - logging level local7 7
    provider: "{{ cli }}"
```

在我们的 playbook 中，我们可以指定它包含`cisco_nexus`角色和`spines`角色：

```py
---
- name: playbook for spine role
  hosts: "spines"
  gather_facts: false
  connection: local

  roles:
    - cisco_nexus
    - spines
```

当我们按照这个顺序包括这两个角色时，`cisco_nexus`角色任务将被执行，然后是 spines 角色：

```py
TASK [cisco_nexus : configure logging parameters] ******************************
changed: [nxos-r1]

TASK [spines : change logging level] *******************************************
ok: [nxos-r1]
```

Ansible 角色是灵活和可扩展的，就像 Python 函数和类一样。一旦您的代码增长到一定程度，将其分解成更小的部分以便维护几乎总是一个好主意。

您可以在 Ansible 示例 Git 存储库中找到更多角色的示例，网址为[`github.com/ansible/ansible-examples`](https://github.com/ansible/ansible-examples)。

**Ansible Galaxy** ([`docs.ansible.com/ansible/latest/reference_appendices/galaxy.html`](https://docs.ansible.com/ansible/latest/reference_appendices/galaxy.html))是一个免费的社区网站，用于查找、共享和协作角色。您可以在 Ansible Galaxy 上查看由 Juniper 网络提供的 Ansible 角色的示例：

![](img/6809eca8-e5b1-4f7c-8fc6-f690fc747597.png)JUNOS Role on Ansible Galaxy ( [`galaxy.ansible.com/Juniper/junos`](https://galaxy.ansible.com/Juniper/junos))

在下一节中，我们将看一下如何编写我们自己的自定义 Ansible 模块。

# 编写您自己的自定义模块

到目前为止，您可能会感到 Ansible 中的网络管理在很大程度上取决于找到适合您任务的正确模块。这种逻辑中肯定有很多道理。模块提供了一种抽象管理主机和控制机之间交互的方式；它们允许我们专注于我们操作的逻辑。到目前为止，我们已经看到主要供应商为 Cisco、Juniper 和 Arista 提供了各种模块。

以 Cisco Nexus 模块为例，除了特定任务，如管理 BGP 邻居(`nxos_bgp`)和 aaa 服务器(`nxos_aaa_server`)。大多数供应商还提供了运行任意 show(`nxos_config`)和配置命令(`nxos_config`)的方法。这通常涵盖了我们大部分的用例。

从 Ansible 2.5 开始，还有网络事实模块的简化命名和用法。

如果您使用的设备当前没有您正在寻找的任务的模块怎么办？在本节中，我们将看一下几种方法，通过编写我们自己的自定义模块来解决这种情况。

# 第一个自定义模块

编写自定义模块并不需要复杂；实际上，它甚至不需要用 Python 编写。但是由于我们已经熟悉 Python，我们将使用 Python 来编写我们的自定义模块。我们假设该模块是我们自己和我们的团队将使用的，而不需要提交给 Ansible，因此我们将暂时忽略一些文档和格式。

如果您有兴趣开发可以提交到 Ansible 的模块，请参阅 Ansible 的模块开发指南([`docs.ansible.com/ansible/latest/dev_guide/developing_modules.html`](https://docs.ansible.com/ansible/latest/dev_guide/developing_modules.html))。

默认情况下，如果我们在与 playbook 相同的目录中创建一个名为`library`的文件夹，Ansible 将包括该目录在模块搜索路径中。因此，我们可以将我们的自定义模块放在该目录中，并且我们将能够在我们的 playbook 中使用它。自定义模块的要求非常简单：模块只需要返回 JSON 输出给 playbook。

回想一下，在第三章 *API 和意图驱动的网络*中，我们使用以下 NXAPI Python 脚本与 NX-OS 设备进行通信：

```py
    import requests
    import json

    url='http://172.16.1.142/ins'
    switchuser='cisco'
    switchpassword='cisco'

    myheaders={'content-type':'application/json-rpc'}
    payload=[
     {
       "jsonrpc": "2.0",
       "method": "cli",
       "params": {
         "cmd": "show version",
         "version": 1.2
       },
       "id": 1
     }
    ]
    response = requests.post(url,data=json.dumps(payload),   
    headers=myheaders,auth=(switchuser,switchpassword)).json()

    print(response['result']['body']['sys_ver_str'])
```

当我们执行它时，我们只是收到了系统版本。我们可以简单地修改最后一行为 JSON 输出，如下面的代码所示：

```py
    version = response['result']['body']['sys_ver_str']
    print json.dumps({"version": version})
```

我们将把这个文件放在`library`文件夹下：

```py
$ ls -a library/
. .. custom_module_1.py
```

在我们的剧本中，我们可以使用动作插件([`docs.ansible.com/ansible/dev_guide/developing_plugins.html`](https://docs.ansible.com/ansible/dev_guide/developing_plugins.html)) `chapter5_14.yml`来调用这个自定义模块：

```py
    ---
    - name: Your First Custom Module
      hosts: localhost
      gather_facts: false
      connection: local

      tasks:
        - name: Show Version
          action: custom_module_1
          register: output

        - debug:
            var: output
```

请注意，就像`ssh`连接一样，我们正在本地执行模块，并且模块正在进行 API 调用。当你执行这个剧本时，你将得到以下输出：

```py
$ ansible-playbook chapter5_14.yml
 [WARNING]: provided hosts list is empty, only localhost is available

PLAY [Your First Custom Module] ************************************************

TASK [Show Version] ************************************************************
ok: [localhost]

TASK [debug] *******************************************************************
ok: [localhost] => {
 "output": {
 "changed": false,
 "version": "7.3(0)D1(1)"
 }
}

PLAY RECAP *********************************************************************
localhost : ok=2 changed=0 unreachable=0 failed=0
```

正如你所看到的，你可以编写任何受 API 支持的模块，Ansible 将乐意接受任何返回的 JSON 输出。

# 第二个自定义模块

在上一个模块的基础上，让我们利用 Ansible 中的常见模块样板，该样板在模块开发文档中有说明([`docs.ansible.com/ansible/dev_guide/developing_modules_general.html`](http://docs.ansible.com/ansible/dev_guide/developing_modules_general.html))。我们将修改最后一个自定义模块，并创建`custom_module_2.py`来接收剧本中的输入。

首先，我们将从`ansible.module_utils.basic`导入样板代码：

```py
    from ansible.module_utils.basic import AnsibleModule

    if __name__ == '__main__':
        main()
```

然后，我们可以定义主要函数，我们将在其中放置我们的代码。`AnsibleModule`，我们已经导入了，提供了处理返回和解析参数的常见代码。在下面的示例中，我们将解析`host`、`username`和`password`三个参数，并将它们作为必填字段：

```py
    def main():
        module = AnsibleModule(
          argument_spec = dict(
          host = dict(required=True),
          username = dict(required=True),
          password = dict(required=True)
      )
    )
```

然后，可以检索这些值并在我们的代码中使用：

```py
     device = module.params.get('host')
     username = module.params.get('username')
     password = module.params.get('password')

     url='http://' + host + '/ins'
     switchuser=username
     switchpassword=password
```

最后，我们将跟踪退出代码并返回值：

```py
    module.exit_json(changed=False, msg=str(data))
```

我们的新剧本`chapter5_15.yml`将与上一个剧本相同，只是现在我们可以在剧本中为不同的设备传递值：

```py
     tasks:
       - name: Show Version
         *action: custom_module_1 host="172.16.1.142" username="cisco"* 
 *password="cisco"*
         register: output
```

当执行时，这个剧本将产生与上一个剧本完全相同的输出。但是，因为我们在自定义模块中使用了参数，所以现在可以将自定义模块传递给其他人使用，而不需要他们了解我们模块的细节。他们可以在剧本中写入自己的用户名、密码和主机 IP。

当然，这是一个功能齐全但不完整的模块。首先，我们没有进行任何错误检查，也没有为使用提供任何文档。但是，这是一个很好的演示，展示了构建自定义模块有多么容易。额外的好处是，我们看到了如何使用我们已经制作的现有脚本，并将其转换为自定义的 Ansible 模块。

# 总结

在本章中，我们涵盖了很多内容。基于我们之前对 Ansible 的了解，我们扩展到了更高级的主题，如条件、循环和模板。我们看了如何通过主机变量、组变量、包含语句和角色使我们的剧本更具可扩展性。我们还看了如何使用 Ansible Vault 保护我们的剧本。最后，我们使用 Python 制作了自己的自定义模块。

Ansible 是一个非常灵活的 Python 框架，可以用于网络自动化。它提供了另一个抽象层，与 Pexpect 和基于 API 的脚本分开。它在性质上是声明式的，更具表达性，符合我们的意图。根据你的需求和网络环境，它可能是你可以用来节省时间和精力的理想框架。

在第六章 *使用 Python 进行网络安全*中，我们将使用 Python 进行网络安全。
