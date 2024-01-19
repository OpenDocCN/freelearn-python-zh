# 系统管理的 Ansible

在本章中，我们将探索一种被成千上万的网络和系统工程师使用的流行自动化框架*Ansible*，Ansible 用于管理服务器和网络设备，通过多种传输协议如 SSH、Netconf 和 API 来提供可靠的基础设施。

我们首先将学习 ansible 中使用的术语，如何构建包含基础设施访问详细信息的清单文件，使用条件、循环和模板渲染等功能构建强大的 Ansible playbook。

Ansible 属于软件配置管理类别；它用于管理多个不同设备和服务器上的配置生命周期，确保所有设备上都应用相同的步骤，并帮助创建基础设施即代码（IaaC）环境。

本章将涵盖以下主题：

+   Ansible 及其术语

+   在 Linux 上安装 Ansible

+   在临时模式下使用 Ansible

+   创建您的第一个 playbook

+   理解 Ansible 的条件、处理程序和循环

+   使用 Ansible 事实

+   使用 Ansible 模板

# Ansible 术语

Ansible 是一个自动化工具和完整的框架，它提供了基于 Python 工具的抽象层。最初，它是设计用来处理任务自动化的。这个任务可以在单个服务器上执行，也可以在成千上万的服务器上执行，ansible 都可以毫无问题地处理；后来，Ansible 的范围扩展到了网络设备和云提供商。Ansible 遵循“幂等性”的概念，其中 Ansible 指令可以多次运行相同的任务，并始终在所有设备上给出相同的配置，最终达到期望的状态，变化最小。例如，如果我们运行 Ansible 将文件上传到特定组的服务器，然后再次运行它，Ansible 将首先验证文件是否已经存在于远程目的地，如果存在，那么 ansible 就不会再次上传它。

再次。这个功能叫做“幂等性”。

Ansible 的另一个方面是它是无代理的。在运行任务之前，Ansible 不需要在服务器上安装任何代理。它利用 SSH 连接和 Python 标准库在远程服务器上执行任务，并将输出返回给 Ansible 服务器。此外，它不会创建数据库来存储远程机器信息，而是依赖于一个名为`inventory`的平面文本文件来存储所有所需的服务器信息，如 IP 地址、凭据和基础设施分类。以下是一个简单清单文件的示例：

```py
[all:children] web-servers db-servers   [web-servers] web01 Ansible_ssh_host=192.168.10.10     [db-servers] db01 Ansible_ssh_host=192.168.10.11 db02 Ansible_ssh_host=192.168.10.12   [all:vars] Ansible_ssh_user=root Ansible_ssh_pass=access123   [db-servers:vars] Ansible_ssh_user=root Ansible_ssh_pass=access123   
```

```py
[local] 127.0.0.1 Ansible_connection=local Ansible_python_interpreter="/usr/bin/python"
```

请注意，我们将在我们的基础设施中执行相同功能的服务器分组在一起（比如数据库服务器，在一个名为`[db-servers]`的组中；同样的，对于`[web-servers]`也是如此）。然后，我们定义一个特殊的组，称为`[all]`，它结合了这两个组，以防我们有一个针对所有服务器的任务。

`children`关键字在`[all:children]`中的意思是组内的条目也是包含主机的组。

Ansible 的“临时”模式允许用户直接从终端向远程服务器执行任务。假设您想要在特定类型的服务器上更新特定的软件包，比如数据库或 Web 后端服务器，以解决一个新的 bug。与此同时，您不想要开发一个复杂的 playbook 来执行一个简单的任务。通过利用 Ansible 的临时模式，您可以在 Ansible 主机终端上输入命令来在远程服务器上执行任何命令。甚至一些模块也可以在终端上执行；我们将在“在临时模式下使用 Ansible”部分中看到这一点。

# 在 Linux 上安装 Ansible

Ansible 软件包在所有主要的 Linux 发行版上都可用。在本节中，我们将在 Ubuntu 和 CentOS 机器上安装它。在编写本书时使用的是 Ansible 2.5 版本，并且它支持 Python 2.6 和 Python 2.7。此外，从 2.2 版本开始，Ansible 为 Python 3.5+提供了技术预览。

# 在 RHEL 和 CentOS

在安装 Ansible 之前，您需要安装和启用 EPEL 存储库。要这样做，请使用以下命令：

```py
sudo yum install epel-release
```

然后，按照以下命令安装 Ansible 软件包：

```py
sudo yum install Ansible
```

# Ubuntu

首先确保您的系统是最新的，并添加 Ansible 通道。最后，安装 Ansible 软件包本身，如下面的代码片段所示：

```py
$ sudo apt-get update
$ sudo apt-get install software-properties-common
$ sudo apt-add-repository ppa:Ansible/Ansible
$ sudo apt-get update
$ sudo apt-get install Ansible
```

有关更多安装选项，请查看官方 Ansible 网站（[`docs.Ansible.com/Ansible/latest/installation_guide/intro_installation.html?#installing-the-control-machine`](http://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html?#installing-the-control-machine)）。

您可以通过运行`Ansible --version`来验证您的安装，以检查已安装的版本：

![](img/00164.jpeg)Ansible 配置文件通常存储在`/etc/Ansible`中，文件名为`Ansible.cfg`。

# 在临时模式下使用 Ansible

当您需要在远程机器上执行简单操作而不创建复杂和持久的任务时，可以使用 Ansible 临时模式。这通常是用户在开始使用 Ansible 时首先使用的地方，然后再执行 playbook 中的高级任务。

执行临时命令需要两件事。首先，您需要清单文件中的主机或组；其次，您需要要执行的针对目标机器的 Ansible 模块：

1.  首先，让我们定义我们的主机，并将 CentOS 和 Ubuntu 机器添加到一个单独的组中：

```py
[all:children] centos-servers ubuntu-servers   [centos-servers] centos-machine01 Ansible_ssh_host=10.10.10.193   [ubuntu-servers] ubuntu-machine01 Ansible_ssh_host=10.10.10.140   [all:vars] Ansible_ssh_user=root Ansible_ssh_pass=access123   [centos-servers:vars] Ansible_ssh_user=root Ansible_ssh_pass=access123   [ubuntu-servers:vars] Ansible_ssh_user=root Ansible_ssh_pass=access123

[routers]
gateway ansible_ssh_host = 10.10.88.110 ansible_ssh_user=cisco ansible_ssh_pass=cisco   [local] 127.0.0.1 Ansible_connection=local Ansible_python_interpreter="/usr/bin/python"
```

1.  将此文件保存为`hosts`，放在`/root/`或您的主目录中的`AutomationServer`下。

1.  然后，使用`ping`模块运行`Ansible`命令：

```py
# Ansible -i hosts all -m ping
```

`-i`参数将接受我们添加的清单文件，而`-m`参数将指定 Ansible 模块的名称。

运行命令后，您将得到以下输出，指示连接到远程机器失败：

```py
ubuntu-machine01 | FAILED! => {
 "msg": "Using a SSH password instead of a key is not possible because Host Key checking is enabled and sshpass does not support this.  Please add this host's fingerprint to your known_hosts file to manage this host."
}
centos-machine01 | FAILED! => {
 "msg": "Using a SSH password instead of a key is not possible because Host Key checking is enabled and sshpass does not support this.  Please add this host's fingerprint to your known_hosts file to manage this host."
}
```

这是因为远程机器不在 Ansible 服务器的`known_hosts`中；可以通过两种方法解决。

第一种方法是手动 SSH 到它们，这将将主机指纹添加到服务器。或者，您可以在 Ansible 配置中完全禁用主机密钥检查，如下面的代码片段所示：

```py
sed -i -e 's/#host_key_checking = False/host_key_checking = False/g' /etc/Ansible/Ansible.cfg

sed -i -e 's/#   StrictHostKeyChecking ask/   StrictHostKeyChecking no/g' /etc/ssh/ssh_config
```

重新运行`Ansible`命令，您应该从三台机器中获得成功的输出：

```py
127.0.0.1 | SUCCESS => {
 "changed": false, 
 "ping": "pong"
}
ubuntu-machine01 | SUCCESS => {
 "changed": false, 
 "ping": "pong"
}
centos-machine01 | SUCCESS => {
 "changed": false, 
 "ping": "pong"
}
```

Ansible 中的`ping`模块不执行针对设备的 ICMP 操作。它实际上尝试使用提供的凭据通过 SSH 登录到设备；如果登录成功，它将返回`pong`关键字给 Ansible 主机。

另一个有用的模块是`apt`或`yum`，用于管理 Ubuntu 或 CentOS 服务器上的软件包。以下示例将在 Ubuntu 机器上安装`apache2`软件包：

```py
# Ansible -i hosts ubuntu-servers -m apt -a "name=apache2 state=present" 
```

`apt`模块中的状态可以有以下值：

| **状态** | **操作** |
| --- | --- |
| `absent` | 从系统中删除软件包。 |
| `present` | 确保软件包已安装在系统上。 |
| `latest` | 确保软件包是最新版本。 |

您可以通过运行`Ansible-doc <module_name>`来访问 Ansible 模块文档；您将看到模块的完整选项和示例。

`service`模块用于管理服务的操作和当前状态。您可以在`state`选项中将服务状态更改为`started`、`restarted`或`stopped`，ansible 将运行适当的命令来更改状态。同时，您可以通过配置`enabled`来配置服务是否在启动时启用或禁用。

```py
#Ansible -i hosts centos-servers -m service -a "name=httpd state=stopped, enabled=no"
```

此外，您可以通过提供服务名称并将`state`设置为`restarted`来重新启动服务：

```py
#Ansible -i hosts centos-servers -m service -a "name=mariadb state=restarted"
```

以 adhoc 模式运行 Ansible 的另一种方法是直接将命令传递给 Ansible，而不是使用内置模块，而是使用`-a`参数：

```py
#Ansible -i hosts all -a "ifconfig"
```

您甚至可以通过运行`reboot`命令重新启动服务器；但这次，我们只会针对 CentOS 服务器运行它：

```py
#Ansible -i hosts centos-servers -a "reboot"
```

有时，您需要使用不同的用户运行命令（或模块）。当您在具有分配给不同于 SSH 用户的特定权限的远程服务器上运行脚本时，这将非常有用。在这种情况下，我们将添加`-u`，`--become`和`--ask-become-pass`（`-K`）开关。这将使 Ansible 使用提供的用户名运行命令，并提示您输入用户的密码：

```py
#Ansible -i hosts ubuntu-servers --become-user bassim  --ask-become-pass -a "cat /etc/sudoers"
```

# Ansible 的实际工作方式

Ansible 基本上是用 Python 编写的，但它使用自己的 DSL（领域特定语言）。您可以使用此 DSL 编写，ansible 将在远程机器上将其转换为 Python 以执行任务。因此，它首先验证任务语法并从 Ansible 主机复制模块到远程服务器，然后在远程服务器上执行它。

执行的结果以`json`格式返回到 Ansible 主机，因此您可以通过了解其键来匹配任何返回的值：

![](img/00165.jpeg)

在安装了 Python 的网络设备的情况下，Ansible 使用 API 或`netconf`（如果网络设备支持，例如 Juniper 和 Cisco Nexus）；或者，它只是使用 paramiko 的`exec_command()`函数执行命令，并将输出返回到 Ansible 主机。这可以通过使用`raw`模块来完成，如下面的代码片段所示：

```py
# Ansible -i hosts routers -m raw -a "show arp" 
gateway | SUCCESS | rc=0 >>

Sat Apr 21 01:33:58.391 CAIRO

Address         Age        Hardware Addr   State      Type  Interface
85.54.41.9         -          45ea.2258.d0a9  Interface  ARPA  TenGigE0/2/0/0
10.88.18.1      -          d0b7.428b.2814  Satellite  ARPA  TenGigE0/2/0/0
192.168.100.1   -          00a7.5a3b.4193  Interface  ARPA  GigabitEthernet100/0/0/9
192.168.100.2   02:08:03   fc5b.3937.0b00  Dynamic    ARPA  \
```

# 创建您的第一个剧本

现在魔术派对可以开始了。Ansible 剧本是一组需要按顺序执行的命令（称为任务），它描述了执行完成后主机的期望状态。将剧本视为包含一组指令的手册，用于更改基础设施的状态；每个指令都依赖于许多内置的 Ansible 模块来执行任务。例如，您可能有一个用于构建 Web 应用程序的剧本，其中包括 SQL 服务器，用作后端数据库和 nginx Web 服务器。剧本将有一系列任务针对每组服务器执行，以将它们的状态从`不存在`更改为`存在`，或者更改为`重新启动`或`不存在`，如果要删除 Web 应用程序。

剧本的强大之处在于您可以使用它在任何地方配置和设置基础设施。用于创建开发环境的相同过程将用于生产环境。剧本用于创建在您的基础设施上运行的自动化工作流程：

![](img/00166.jpeg)

剧本是用 YAML 编写的，我们在第六章中讨论过，*使用 Python 和 Jinja2 生成配置*。剧本由多个 play 组成，针对清单文件中定义的一组主机执行。主机将被转换为 Python `list`，列表中的每个项目将被称为`play`。在前面的示例中，`db-servers`任务是一些 play，并且仅针对`db-servers`执行。在剧本执行期间，您可以决定运行文件中的所有 play，仅特定 play 或具有特定标记的任务，而不管它们属于哪个 play。

现在，让我们看看我们的第一个剧本，以了解其外观和感觉：

```py
- hosts: centos-servers
  remote_user: root

  tasks:
    - name: Install openssh
      yum: pkg=openssh-server state=installed

    - name: Start the openssh
      service: name=sshd state=started enabled=yes
```

这是一个简单的剧本，有一个包含两个任务的`play`：

1.  安装`openssh-server`。

1.  安装后启动`sshd`服务，并确保它在启动时可用。

现在，我们需要将其应用于特定主机（或一组主机）。因此，我们将`hosts`设置为之前在 inventory 文件中定义的`CentOS-servers`，并且我们还将`remote_user`设置为 root，以确保之后的任务将以 root 权限执行。

任务将包括名称和 Ansible 模块。名称用于描述任务。为任务提供名称并不是强制性的，但建议这样做，以防需要从特定任务开始执行。

第二部分是 Ansible 模块，这是必需的。在我们的示例中，我们使用了核心模块`yum`来在目标服务器上安装`openssh-server`软件包。第二个任务具有相同的结构，但这次我们将使用另一个核心模块，称为`service`，来启动和启用`sshd`守护程序。

最后要注意 Ansible 中不同组件的缩进。例如，任务的名称应该在同一级别，而`tasks`应该与同一行上的`hosts`对齐。

让我们在我们的自动化服务器上运行 playbook 并检查输出：

```py
#Ansible-playbook -i hosts first_playbook.yaml 

PLAY [centos-servers] **********************************************************************

TASK [Gathering Facts] *********************************************************************
ok: [centos-machine01]

TASK [Install openssh] *********************************************************************
ok: [centos-machine01]

TASK [Start the openssh] *******************************************************************
ok: [centos-machine01]

```

```py
PLAY RECAP *********************************************************************************
centos-machine01           : ok=3    changed=0    unreachable=0    failed=0   
```

您可以看到 playbook 在`centos-machine01`上执行，并且任务按照 playbook 中定义的顺序依次执行。

YAML 要求保留缩进级别，并且不要混合制表符和空格；否则，将会出现错误。许多文本编辑器和 IDE 将制表符转换为一组空格。以下截图显示了该选项的示例，在 notepad++编辑器首选项中：![](img/00167.jpeg)

# 理解 Ansible 条件、处理程序和循环

在本章的这一部分，我们将看一些 Ansible playbook 中的高级功能。

# 设计条件

Ansible playbook 可以根据任务内部特定条件的结果执行任务（或跳过任务）——例如，当您想要在特定操作系统家族（Debian 或 CentOS）上安装软件包时，或者当操作系统是特定版本时，甚至当远程主机是虚拟机而不是裸机时。这可以通过在任务内部使用`when`子句来实现。

让我们增强先前的 playbook，并将`openssh-server`安装限制为仅适用于基于 CentOS 的系统，这样当它遇到使用`apt`模块而不是`yum`的 Ubuntu 服务器时，就不会出错。

首先，我们将在我们的`inventory`文件中添加以下两个部分，将 CentOS 和 Ubuntu 机器分组到`infra`部分中：

```py
[infra:children] centos-servers ubuntu-servers     [infra:vars] Ansible_ssh_user=root Ansible_ssh_pass=access123 
```

然后，我们将重新设计 playbook 中的任务，添加`when`子句，将任务执行限制为仅适用于基于 CentOS 的机器。这应该读作`如果远程机器是基于 CentOS 的，那么我将执行任务；否则，跳过`。

```py
- hosts: infra
  remote_user: root

  tasks:
    - name: Install openssh
      yum: pkg=openssh-server state=installed
      when: Ansible_distribution == "CentOS"

    - name: Start the openssh
      service: name=sshd state=started enabled=yes
  when: Ansible_distribution == "CentOS"
```

让我们运行 playbook：

```py
# Ansible-playbook -i hosts using_when.yaml 

PLAY [infra] *******************************************************************************

TASK [Gathering Facts] *********************************************************************
ok: [centos-machine01]
ok: [ubuntu-machine01]

TASK [Install openssh] *********************************************************************
skipping: [ubuntu-machine01]
ok: [centos-machine01]

TASK [Start the openssh] *******************************************************************
skipping: [ubuntu-machine01]
ok: [centos-machine01]

PLAY RECAP *********************************************************************************
centos-machine01           : ok=3    changed=0    unreachable=0    failed=0 
ubuntu-machine01           : ok=1    changed=0    unreachable=0    failed=0  
```

请注意，playbook 首先收集有关远程机器的信息（我们将在本章后面讨论），然后检查操作系统。当它遇到`ubuntu-machine01`时，任务将被跳过，并且在 CentOS 上将正常运行。

您还可以有多个条件需要满足才能运行任务。例如，您可以有以下 playbook，验证两件事情——首先，机器基于 Debian，其次，它是一个虚拟机，而不是裸机：

```py
- hosts: infra
  remote_user: root

  tasks:
    - name: Install openssh
      apt: pkg=open-vm-tools state=installed
      when:
        - Ansible_distribution == "Debian"
        - Ansible_system_vendor == "VMware, Inc."
```

运行此 playbook 将产生以下输出：

```py
# Ansible-playbook -i hosts using_when_1.yaml 

PLAY [infra] *******************************************************************************

TASK [Gathering Facts] *********************************************************************
ok: [centos-machine01]
ok: [ubuntu-machine01]

TASK [Install openssh] *********************************************************************
skipping: [centos-machine01]
ok: [ubuntu-machine01]

PLAY RECAP *********************************************************************************
centos-machine01           : ok=1    changed=0    unreachable=0    failed=0
ubuntu-machine01           : ok=2    changed=0    unreachable=0    failed=0 
```

Ansible 的`when`子句还接受表达式。例如，您可以检查返回的输出中是否存在特定关键字（使用注册标志保存），并根据此执行任务。

以下 playbook 将验证 OSPF 邻居状态。第一个任务将在路由器上执行`show ip ospf neighbor`并将输出注册到名为`neighbors`的变量中。接下来的任务将检查返回的输出中是否有`EXSTART`或`EXCHANGE`，如果找到，将在控制台上打印一条消息：

```py
hosts: routers

tasks:
  - name: "show the ospf neighbor status"
    raw: show ip ospf neighbor
    register: neighbors

  - name: "Validate the Neighbors"
    debug:
      msg: "OSPF neighbors stuck"
    when: ('EXSTART' in neighbors.stdout) or ('EXCHANGE' in neigbnors.stdout)
```

您可以在[`docs.Ansible.com/Ansible/latest/user_guide/playbooks_conditionals.html#commonly-used-facts`](http://docs.ansible.com/ansible/latest/user_guide/playbooks_conditionals.html#commonly-used-facts)中检查在`when`子句中常用的事实。

# 在 ansible 中创建循环

Ansible 提供了许多重复在 play 中执行相同任务的方法，但每次都有不同的值。例如，当您想在服务器上安装多个软件包时，您不需要为每个软件包创建一个任务。相反，您可以创建一个任务，安装一个软件包并向任务提供软件包名称的列表，Ansible 将对它们进行迭代，直到完成安装。为此，我们需要在包含列表的任务内使用`with_items`标志，并使用变量`{{ item }}`，它作为列表中项目的占位符。playbook 将利用`with_items`标志对一组软件包进行迭代，并将它们提供给`yum`模块，该模块需要软件包的名称和状态：

```py
- hosts: infra
  remote_user: root

  tasks:
    - name: "Modifying Packages"
  yum: name={{ item.name }} state={{ item.state }}
  with_items:
        - { name: python-keyring-5.0-1.el7.noarch, state: absent }
  - { name: python-django, state: absent }
  - { name: python-django-bash-completion, state: absent }
  - { name: httpd, state: present }
  - { name: httpd-tools, state: present }
  - { name: python-qpid, state: present }
  when: Ansible_distribution == "CentOS"
```

您可以将状态的值硬编码为`present`；在这种情况下，所有的软件包都将被安装。然而，在前一种情况下，`with_items`将向`yum`模块提供两个元素。

playbook 的输出如下：

![](img/00168.jpeg)

# 使用处理程序触发任务

好的；您已经在系统中安装和删除了一系列软件包。您已经将文件复制到/从服务器。并且您已经通过使用 Ansible playbook 在服务器上做了很多改变。现在，您需要重新启动一些其他服务，或者向文件中添加一些行，以完成服务的配置。所以，您应该添加一个新的任务，对吗？是的，这是正确的。然而，Ansible 提供了另一个很棒的选项，称为**handlers**，它不会在触发时自动执行（不像任务），而是只有在被调用时才会执行。这为您提供了灵活性，可以在 play 中的任务执行时调用它们。

处理程序与主机和任务具有相同的对齐方式，并位于每个 play 的底部。当您需要调用处理程序时，您可以在原始任务内使用`notify`标志，以确定将执行哪个处理程序；Ansible 将它们链接在一起。

让我们看一个例子。我们将编写一个 playbook，在 CentOS 服务器上安装和配置 KVM。KVM 在安装后需要进行一些更改，比如加载`sysctl`，启用`kvm`和`802.1q`模块，并在`boot`时加载`kvm`：

```py
- hosts: centos-servers
  remote_user: root

  tasks:
    - name: "Install KVM"
  yum: name={{ item.name }} state={{ item.state }}
  with_items:
        - { name: qemu-kvm, state: installed }
  - { name: libvirt, state: installed }
  - { name: virt-install, state: installed }
  - { name: bridge-utils, state: installed }    notify:
        - load sysctl
        - load kvm at boot
        - enable kvm

  handlers:
    - name: load sysctl
      command: sysctl -p

    - name: enable kvm
      command: "{{ item.name }}"
      with_items:
        - {name: modprobe -a kvm}
  - {name: modprobe 8021q}
  - {name: udevadm trigger}    - name: load kvm at boot
      lineinfile: dest=/etc/modules state=present create=True line={{ item.name }}
  with_items:
        - {name: kvm}   
```

注意安装任务后使用`notify`。当任务运行时，它将按顺序通知三个处理程序，以便它们将被执行。处理程序将在任务成功执行后运行。这意味着如果任务未能运行（例如，找不到`kvm`软件包，或者没有互联网连接来下载它），则系统不会发生任何更改，`kvm`也不会被启用。

处理程序的另一个很棒的特性是，它只在任务中有更改时才运行。例如，如果您重新运行任务，Ansible 不会安装`kvm`软件包，因为它已经安装；它不会调用任何处理程序，因为它在系统中没有检测到任何更改。

我们将在最后关于两个模块添加一个注释：`lineinfile`和`command`。第一个模块实际上是通过使用正则表达式向配置文件中插入或删除行；我们使用它来将`kvm`插入`/etc/modules`，以便在机器启动时自动启动 KVM。第二个模块`command`用于在设备上直接执行 shell 命令并将输出返回给 Ansible 主机。

# 使用 Ansible 事实

Ansible 不仅用于部署和配置远程主机。它可以用于收集有关它们的各种信息和事实。事实收集可能需要大量时间来从繁忙的系统中收集所有内容，但将为目标机器提供全面的视图。

收集到的事实可以在后续的 playbook 中使用，设计任务条件。例如，我们使用`when`子句将`openssh`安装限制为仅适用于基于 CentOS 的系统：

```py
when: Ansible_distribution == "CentOS"
```

您可以通过在与主机和任务相同级别上配置`gather_facts`来在 Ansible plays 中启用/禁用事实收集。

```py
- hosts: centos-servers
  gather_facts: yes
  tasks:
    <your tasks go here>
```

在 Ansible 中收集事实并打印它们的另一种方法是在 adhoc 模式中使用`setup`模块。返回的结果以嵌套的字典和列表的形式描述远程目标的事实，例如服务器架构、内存、网络设置、操作系统版本等：

```py
#Ansible -i hosts ubuntu-servers -m setup | less 
```

![](img/00169.jpeg)

您可以使用点表示法或方括号从事实中获取特定值。例如，要获取`eth0`的 IPv4 地址，可以使用`Ansible_eth0["ipv4"]["address"]`或`Ansible_eth0.ipv4.address`。

# 使用 Ansible 模板

与 Ansible 一起工作的最后一部分是了解它如何处理模板。Ansible 使用我们在第六章中讨论过的 Jinja2 模板，*使用 Python 和 Jinja2 生成配置*。它使用 Ansible 事实或在`vars`部分提供的静态值填充参数，甚至使用使用`register`标志存储的任务的结果。

在以下示例中，我们将构建一个 Ansible playbook，其中包含前面三个案例。首先，在`vars`部分中定义一个名为`Header`的变量，其中包含一个欢迎消息作为静态值。然后，我们启用`gather_facts`标志，以从目标机器获取所有可能的信息。最后，我们执行`date`命令，以获取服务器的当前日期并将输出存储在`date_now`变量中：

```py
- hosts: centos-servers
  vars:
    - Header: "Welcome to Server facts page generated from Ansible playbook"
 gather_facts: yes  tasks:
    - name: Getting the current date
      command: date
      register: date_now
    - name: Setup webserver
      yum: pkg=nginx state=installed
      when: Ansible_distribution == "CentOS"

      notify:
        - enable the service
        - start the service

    - name: Copying the index page
      template: src=index.j2 dest=/usr/share/nginx/html/index.html

  handlers:
    - name: enable the service
      service: name=nginx enabled=yes    - name: start the service
      service: name=nginx state=started
```

在前面的 playbook 中使用的模板模块将接受一个名为`index.j2`的 Jinja2 文件，该文件位于 playbook 的同一目录中；然后，它将从我们之前讨论过的三个来源中提供所有 jinj2 变量的值。然后，渲染后的文件将存储在模板模块提供的`dest`选项中的路径中。

`index.j2`的内容如下。它将是一个简单的 HTML 页面，利用 jinja2 语言生成最终的 HTML 页面：

```py
<html> <head><title>Hello world</title></head> <body>   <font size="6" color="green">{{ Header }}</font>   <br> <font size="5" color="#ff7f50">Facts about the server</font> <br> <b>Date Now is:</b> {{ date_now.stdout }}

<font size="4" color="#00008b"> <ul>
 <li>IPv4 Address: {{ Ansible_default_ipv4['address'] }}</li>
 <li>IPv4 gateway: {{ Ansible_default_ipv4['gateway'] }}</li>
 <li>Hostname: {{ Ansible_hostname }}</li>
 <li>Total Memory: {{ Ansible_memtotal_mb }}</li>
 <li>Operating System Family: {{ Ansible_os_family }}</li>
 <li>System Vendor: {{ Ansible_system_vendor }}</li> </ul> </font> </body> </html>
```

运行此 playbook 将在 CentOS 机器上安装 nginx web 服务器，并向其添加一个`index.html`页面。您可以通过浏览器访问该页面：

![](img/00170.gif)

您还可以利用模板模块生成网络设备配置。在[第六章](https://cdp.packtpub.com/hands_on_enterprise_automation_with_python/wp-admin/post.php?post=322&action=edit#post_33)中使用的 jinja2 模板，*使用 Python 和 Jinja2 生成配置*，为路由器生成了`day0`和`day1`配置，可以在 Ansible playbook 中重复使用。

# 总结

Ansible 是一个非常强大的工具，用于自动化 IT 基础设施。它包含许多模块和库，几乎涵盖了系统和网络自动化中的所有内容，使软件部署、软件包管理和配置管理变得非常容易。虽然 Ansible 可以在 adhoc 模式下执行单个模块，但 Ansible 的真正力量在于编写和开发 playbook。
