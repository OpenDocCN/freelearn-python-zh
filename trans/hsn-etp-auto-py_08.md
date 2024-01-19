# 第八章：准备实验室环境

在本章中，我们将使用两个流行的 Linux 发行版 CentOS 和 Ubuntu 来设置实验室。CentOS 是一个以社区驱动的 Linux 操作系统，面向企业服务器，并以其与**Red Hat Enterprise Linux**（**RHEL**）的兼容性而闻名。Ubuntu 是另一个基于著名的 Debian 操作系统的 Linux 发行版；目前由 Canonical Ltd.开发，并为其提供商业支持。

我们还将学习如何使用名为**Cobbler**的免费开源软件安装这两个 Linux 发行版，它将使用`kickstart`为 CentOS 自动引导服务器并使用 Anaconda 为基于 Debian 的系统进行自定义。

本章将涵盖以下主题：

+   获取 Linux 操作系统

+   在 hypervisor 上创建一个自动化机器

+   开始使用 Cobbler

# 获取 Linux 操作系统

在接下来的章节中，我们将在不同的 hypervisors 上创建两台 Linux 机器，CentOS 和 Ubuntu。这些机器将作为我们环境中的自动化服务器。

# 下载 CentOS

CentOS 二进制文件可以通过多种方法下载。您可以直接从世界各地的多个 FTP 服务器下载它们，也可以从种子人员那里以种子方式下载它们。此外，CentOS 有两种版本：

+   Minimal ISO：提供基本服务器和必要软件包

+   Everything ISO：提供服务器和主要存储库中的所有可用软件包

首先，前往 CentOS 项目链接（[`www.centos.org/`](https://www.centos.org/)）并单击获取 CentOS 现在按钮，如下截图所示：

![](img/00126.jpeg)

然后，选择最小的 ISO 镜像，并从任何可用的下载站点下载它。

CentOS 可用于多个云提供商，如 Google、Amazon、Azure 和 Oracle Cloud。您可以在[`cloud.centos.org/centos/7/images/`](https://cloud.centos.org/centos/7/images/)找到所有云镜像。

# 下载 Ubuntu

Ubuntu 以为为向最终用户提供良好的桌面体验而广为人知。Canonical（Ubuntu 开发者）与许多服务器供应商合作，以在不同的硬件上认证 Ubuntu。Canonical 还为 Ubuntu 提供了一个服务器版本，其中包括 16.04 中的许多功能，例如：

+   Canonical 将在 2021 年之前提供支持

+   能够在所有主要架构上运行-x86、x86-64、ARM v7、ARM64、POWER8 和 IBM s390x（LinuxONE）

+   ZFS 支持，这是一种适用于服务器和容器的下一代卷管理文件系统

+   LXD Linux 容器 hypervisor 增强，包括 QoS 和资源控制（CPU、内存、块 I/O 和存储配额）

+   安装 snaps，用于简单的应用程序安装和发布管理。

+   DPDK 的首个生产版本-线速内核网络

+   Linux 4.4 内核和`systemd`服务管理器

+   作为 AWS、Microsoft Azure、Joyent、IBM、Google Cloud Platform 和 Rackspace 上的客户进行认证

+   Tomcat（v8）、PostgreSQL（v9.5）、Docker v（1.10）、Puppet（v3.8.5）、QEMU（v2.5）、Libvirt（v1.3.1）、LXC（v2.0）、MySQL（v5.6）等的更新

您可以通过浏览至[`www.ubuntu.com/download/server`](https://www.ubuntu.com/download/server)并选择 Ubuntu 16.04 LTS 来下载 Ubuntu LTS：

![](img/00127.jpeg)

# 在 hypervisor 上创建一个自动化机器

下载 ISO 文件后，我们将在 VMware ESXi 和 KVM hypervisors 上创建一个 Linux 机器。

# 在 VMware ESXi 上创建一个 Linux 机器

我们将使用 VMware vSphere 客户端创建一个虚拟机。使用 root 凭据登录到可用的 ESXi 服务器之一。首先，您需要将 Ubuntu 或 CentOS ISO 上传到 VMware 数据存储中。然后，按照以下步骤创建机器：

1.  右键单击服务器名称，然后选择新的虚拟机：

![](img/00128.jpeg)

1.  选择自定义安装，这样您在安装过程中将有更多选项：

![](img/00129.gif)

1.  为 VM 提供一个名称：AutomationServer。

1.  选择机器版本：8。

1.  选择要创建机器的数据存储。

1.  选择客户操作系统：Ubuntu Linux（64 位）或 Red Hat 版本 6/7：

![](img/00130.gif)

1.  VM 规格不应少于 2 个 vCPU 和 4GB RAM，以便获得高效的性能。分别在 CPU 和内存选项卡中选择它们。

1.  在“网络”选项卡中，选择两个带有 E1000 适配器的接口。其中一个接口将连接到互联网，第二个接口将管理客户端：

![](img/00131.gif)

1.  选择系统的默认 SCSI 控制器。在我的情况下，它将是 LSI 逻辑并行。

1.  选择创建一个新的虚拟磁盘，并为 VM 提供 20GB 的磁盘大小。

1.  现在虚拟机已准备就绪，您可以开始 Linux 操作系统的安装。将上传的镜像关联到 CD/DVD 驱动器，并确保选择“开机时连接”选项：

![](img/00132.jpeg)

一旦它开始运行，您将被要求选择一种语言：

![](img/00133.jpeg)

按照通常的步骤完成 CentOS/Ubuntu 安装。

# 在 KVM 上创建 Linux 机器

我们将使用 KVM 中提供的`virt-manager`实用程序启动 KVM 的桌面管理。然后我们将创建一个新的 VM：

1.  在这里，我们将选择本地安装媒体（ISO 镜像或 CDROM）作为安装方法：

![](img/00134.jpeg)

1.  然后，我们将点击浏览并选择先前下载的镜像（CentOS 或 Ubuntu）。您将注意到 KVM 成功检测到操作系统类型和版本：

![](img/00135.jpeg)

1.  然后，我们将根据 CPU、内存和存储选择机器规格：

![](img/00136.jpeg)

1.  为您的机器选择适当的存储空间：

![](img/00137.jpeg)

1.  最后一步是选择一个名称，然后点击“在安装前自定义配置”选项，以添加一个额外的网络接口到自动化服务器。然后，点击“完成”：

![](img/00138.jpeg)

打开另一个窗口，其中包含机器的所有规格。点击“添加硬件”，然后选择“网络”：

![](img/00139.jpeg)

我们将添加另一个网络接口以与客户端通信。第一个网络接口使用 NAT 通过物理无线网卡连接到互联网：

![](img/00140.jpeg)

最后，在主窗口上点击“开始安装”，KVM 将开始分配硬盘并将 ISO 镜像附加到虚拟机上：

![](img/00141.jpeg)

一旦完成，您将看到以下屏幕：

![](img/00142.jpeg)

按照通常的步骤完成 CentOS/Ubuntu 安装。

# 开始使用 Cobbler

Cobbler 是一款用于无人值守网络安装的开源软件。它利用多个工具，如 DHCP、FTP、PXE 和其他开源工具（稍后我们将解释它们），以便您可以一站式自动安装操作系统。目标机器（裸机或虚拟机）必须支持从其**网络接口卡**（NIC）引导。此功能使机器能够发送一个 DHCP 请求，该请求会命中 Cobbler 服务器，后者将处理其余事宜。

您可以在其 GitHub 页面上阅读有关该项目的更多信息（[`github.com/cobbler/cobbler`](https://github.com/cobbler/cobbler)）。

# 了解 Cobbler 的工作原理

Cobbler 依赖于多个工具来为客户端提供**预引导执行环境**（PXE）功能。首先，它依赖于接收客户端开机时的 DHCP 广播消息的 DHCP 服务；然后，它会回复一个 IP 地址、子网掩码、下一个服务器（TFTP），最后是`pxeLinux.0`，这是客户端最初向服务器发送 DHCP 消息时请求的加载程序文件名。

第二个工具是 TFTP 服务器，它托管`pxeLinux.0`和不同的发行版镜像。

第三个工具是模板渲染实用程序。Cobbler 使用`cheetah`，这是一个由 Python 开发的开源模板引擎，并且有自己的 DSL（特定领域语言）格式。我们将使用它来生成`kickstart`文件。

Kickstart 文件用于自动安装基于 Red Hat 的发行版，如 CentOS、Red Hat 和 Fedora。它还有限的支持用于安装基于 Debian 系统的`Anaconda`文件的渲染。

还有其他附加工具。`reposync`用于将在线存储库从互联网镜像到 Cobbler 内的本地目录，使其对客户端可用。`ipmitools`用于远程管理不同服务器硬件的开关机：

![](img/00143.jpeg)

在以下拓扑中，Cobbler 托管在先前安装的自动化服务器上，并将连接到一对服务器。我们将通过 Cobbler 在它们上安装 Ubuntu 和 Red Hat。自动化服务器还有另一个接口直接连接到互联网，以便下载 Cobbler 所需的一些附加软件包，我们将在下一节中看到：

![](img/00144.jpeg)

| **服务器** | **IP 地址** |
| --- | --- |
| 自动化服务器（已安装 cobbler） | `10.10.10.130` |
| 服务器 1（CentOS 机器） | IP 范围为`10.10.10.5`-`10.10.10.10` |
| 服务器 2（Ubuntu 机器） | IP 范围为`10.10.10.5`-`10.10.10.10` |

# 在自动化服务器上安装 Cobbler

我们将首先在我们的自动化服务器（无论是 CentOS 还是 Ubuntu）上安装一些基本软件包，如`vim`、`tcpudump`、`wget`和`net-tools`。然后，我们将从`epel`存储库安装`cobbler`软件包。请注意，这些软件包对于 Cobbler 并不是必需的，但我们将使用它们来了解 Cobbler 的真正工作原理。

对于 CentOS，请使用以下命令：

```py
yum install vim vim-enhanced tcpdump net-tools wget git -y
```

对于 Ubuntu，请使用以下命令：

```py
sudo apt install vim tcpdump net-tools wget git -y
```

然后，我们需要禁用防火墙。Cobbler 与 SELinux 策略不兼容，建议禁用它，特别是如果您对它们不熟悉。此外，我们将禁用`iptables`和`firewalld`，因为我们在实验室中，而不是在生产环境中。

对于 CentOS，请使用以下命令：

```py
# Disable firewalld service
systemctl disable firewalld
systemctl stop firewalld

# Disable IPTables service
systemctl disable iptables.service
systemctl stop iptables.service

# Set SELinux to permissive instead of enforcing
sed -i s/^SELinux=.*$/SELinux=permissive/ /etc/seLinux/config
setenforce 0
```

对于 Ubuntu，请使用以下命令：

```py
# Disable ufw service
sudo ufw disable

# Disable IPTables service 
sudo iptables-save > $HOME/BeforeCobbler.txt 
sudo iptables -X 
sudo iptables -t nat -F 
sudo iptables -t nat -X 
sudo iptables -t mangle -F 
sudo iptables -t mangle -X 
sudo iptables -P INPUT ACCEPT 
sudo iptables -P FORWARD ACCEPT 
sudo iptables -P OUTPUT ACCEPT

# Set SELinux to permissive instead of enforcing
sed -i s/^SELinux=.*$/SELinux=permissive/ /etc/seLinux/config
setenforce 0
```

最后，重新启动自动化服务器机器以使更改生效：

```py
reboot
```

现在，我们将安装`cobbler`软件包。该软件在`epel`存储库中可用（但我们需要先安装它）在 CentOS 的情况下。Ubuntu 在上游存储库中没有该软件可用，因此我们将在该平台上下载源代码并进行编译。

对于 CentOS，请使用以下命令：

```py
# Download and Install EPEL Repo
yum install epel-release -y

# Install Cobbler
yum install cobbler -y

#Install cobbler Web UI and other dependencies
yum install cobbler-web dnsmasq fence-agents bind xinetd pykickstart -y
```

撰写本书时的 Cobbler 当前版本为 2.8.2，发布于 2017 年 9 月 16 日。对于 Ubuntu，我们将从 GIT 存储库克隆最新的软件包，并从源代码构建它：

```py
#install the dependencies as stated in (http://cobbler.github.io/manuals/2.8.0/2/1_-_Prerequisites.html)

sudo apt-get install createrepo apache2 mkisofs libapache2-mod-wsgi mod_ssl python-cheetah python-netaddr python-simplejson python-urlgrabber python-yaml rsync sysLinux atftpd yum-utils make python-dev python-setuptools python-django -y

#Clone the cobbler 2.8 from the github to your server (require internet)
git clone https://github.com/cobbler/cobbler.git
cd cobbler

#Checkout the release28 (latest as the developing of this book)
git checkout release28

#Build the cobbler core package
make install

#Build cobbler web
make webtest
```

成功在我们的机器上安装 Cobbler 后，我们需要自定义它以更改默认设置以适应我们的网络环境。我们需要更改以下内容：

+   选择`bind`或`dnsmasq`模块来管理 DNS 查询

+   选择`isc`或`dnsmaasq`模块来为客户端提供传入的 DHCP 请求

+   配置 TFTP Cobbler IP 地址（在 Linux 中通常是静态地址）。

+   提供为客户端提供 DHCP 范围

+   重新启动服务以应用配置

让我们逐步查看配置：

1.  选择`dnsmasq`作为 DNS 服务器：

```py
vim /etc/cobbler/modules.conf
[dns]
module = manage_dnsmasq
vim /etc/cobbler/settings
manage_dns: 1
restart_dns: 1
```

1.  选择`dnsmasq`来管理 DHCP 服务：

```py
vim /etc/cobbler/modules.conf

[dhcp]
module = manage_dnsmasq
vim /etc/cobbler/settings
manage_dhcp: 1
restart_dhcp: 1
```

1.  将 Cobbler IP 地址配置为 TFTP 服务器：

```py
vim /etc/cobbler/settings
server: 10.10.10.130
next_server: 10.10.10.130
vim /etc/xinetd.d/tftp
 disable                 = no
```

还要通过将`pxe_just_once`设置为`0`来启用 PXE 引导循环预防：

```py
pxe_just_once: 0
```

1.  在`dnsmasq`服务模板中添加客户端`dhcp-range`：

```py
vim /etc/cobbler/dnsmasq.template
dhcp-range=10.10.10.5,10.10.10.10,255.255.255.0
```

注意其中一行写着`dhcp-option=66,$next_server`。这意味着 Cobbler 将把之前在设置中配置为 TFTP 引导服务器的`next_server`传递给通过`dnsmasq`提供的 DHCP 服务请求 IP 地址的任何客户端。

1.  启用并重新启动服务：

```py
systemctl enable cobblerd
systemctl enable httpd
systemctl enable dnsmasq

systemctl start cobblerd
systemctl start httpd
systemctl start dnsmasq
```

# 通过 Cobbler 提供服务器

现在我们离通过 Cobbler 使我们的第一台服务器运行起来只有几步之遥。基本上，我们需要告诉 Cobbler 我们客户端的 MAC 地址以及它们使用的操作系统：

1.  导入 Linux ISO。Cobbler 将自动分析映像并为其创建一个配置文件：

```py

cobbler import --arch=x86_64 --path=/mnt/cobbler_images --name=CentOS-7-x86_64-Minimal-1708

task started: 2018-03-28_132623_import
task started (id=Media import, time=Wed Mar 28 13:26:23 2018)
Found a candidate signature: breed=redhat, version=rhel6
Found a candidate signature: breed=redhat, version=rhel7
Found a matching signature: breed=redhat, version=rhel7
Adding distros from path /var/www/cobbler/ks_mirror/CentOS-7-x86_64-Minimal-1708-x86_64:
creating new distro: CentOS-7-Minimal-1708-x86_64
trying symlink: /var/www/cobbler/ks_mirror/CentOS-7-x86_64-Minimal-1708-x86_64 -> /var/www/cobbler/links/CentOS-7-Minimal-1708-x86_64
creating new profile: CentOS-7-Minimal-1708-x86_64
associating repos
checking for rsync repo(s)
checking for rhn repo(s)
checking for yum repo(s)
starting descent into /var/www/cobbler/ks_mirror/CentOS-7-x86_64-Minimal-1708-x86_64 for CentOS-7-Minimal-1708-x86_64
processing repo at : /var/www/cobbler/ks_mirror/CentOS-7-x86_64-Minimal-1708-x86_64
need to process repo/comps: /var/www/cobbler/ks_mirror/CentOS-7-x86_64-Minimal-1708-x86_64
looking for /var/www/cobbler/ks_mirror/CentOS-7-x86_64-Minimal-1708-x86_64/repodata/*comps*.xml
Keeping repodata as-is :/var/www/cobbler/ks_mirror/CentOS-7-x86_64-Minimal-1708-x86_64/repodata
*** TASK COMPLETE ***
```

在将其导入到挂载点之前，您可能需要挂载 Linux ISO 映像，使用`mount -O loop /root/<image_iso>  /mnt/cobbler_images/`。

您可以运行`cobbler profile report`命令来检查创建的配置文件：

```py
cobbler profile report

Name                           : CentOS-7-Minimal-1708-x86_64
TFTP Boot Files                : {}
Comment                        : 
DHCP Tag                       : default
Distribution                   : CentOS-7-Minimal-1708-x86_64
Enable gPXE?                   : 0
Enable PXE Menu?               : 1
Fetchable Files                : {}
Kernel Options                 : {}
Kernel Options (Post Install)  : {}
Kickstart                      : /var/lib/cobbler/kickstarts/sample_end.ks
Kickstart Metadata             : {}
Management Classes             : []
Management Parameters          : <<inherit>>
Name Servers                   : []
Name Servers Search Path       : []
Owners                         : ['admin']
Parent Profile                 : 
Internal proxy                 : 
Red Hat Management Key         : <<inherit>>
Red Hat Management Server      : <<inherit>>
Repos                          : []
Server Override                : <<inherit>>
Template Files                 : {}
Virt Auto Boot                 : 1
Virt Bridge                    : xenbr0
Virt CPUs                      : 1
Virt Disk Driver Type          : raw
Virt File Size(GB)             : 5
Virt Path                      : 
Virt RAM (MB)                  : 512
Virt Type                      : kvm
```

您可以看到`import`命令自动填充了许多字段，如`Kickstart`、`RAM`、`操作系统`和`initrd/kernel`文件位置。

1.  向配置文件添加任何额外的存储库（可选）：

```py
cobbler repo add --mirror=https://dl.fedoraproject.org/pub/epel/7/x86_64/ --name=epel-local --priority=50 --arch=x86_64 --breed=yum

cobbler reposync 
```

现在，编辑配置文件，并将创建的存储库添加到可用存储库列表中：

```py
cobbler profile edit --name=CentOS-7-Minimal-1708-x86_64 --repos="epel-local"
```

1.  添加客户端 MAC 地址并将其链接到创建的配置文件：

```py
cobbler system add --name=centos_client --profile=CentOS-7-Minimal-1708-x86_64  --mac=00:0c:29:4c:71:7c --ip-address=10.10.10.5 --subnet=255.255.255.0 --static=1 --hostname=centos-client  --gateway=10.10.10.1 --name-servers=8.8.8.8 --interface=eth0
```

`--hostname`字段对应于本地系统名称，并使用`--ip-address`、`--subnet`和`--gateway`选项配置客户端网络。这将使 Cobbler 生成一个带有这些选项的`kickstart`文件。

如果您需要自定义服务器并添加额外的软件包、配置防火墙、ntp 以及配置分区和硬盘布局，那么您可以将这些设置添加到`kickstart`文件中。Cobbler 在`/var/lib/cobbler/kickstarts/sample.ks`下提供了示例文件，您可以将其复制到另一个文件夹，并在上一个命令中提供`--kickstart`参数。

您可以通过在`kickstart`文件中运行 Ansible 来将 Ansible 集成到其中，使用拉模式（而不是默认的推送模式）。Ansible 将从在线 GIT 存储库（如 GitHub 或 GitLab）下载 playbook，并在此之后执行它。

1.  通过以下命令指示 Cobbler 生成为我们的客户端提供服务所需的配置文件，并使用新信息更新内部数据库：

```py
#cobbler sync  task started: 2018-03-28_141922_sync
task started (id=Sync, time=Wed Mar 28 14:19:22 2018)
running pre-sync triggers
cleaning trees
removing: /var/www/cobbler/images/CentOS-7-Minimal-1708-x86_64
removing: /var/www/cobbler/images/Ubuntu_Server-x86_64
removing: /var/www/cobbler/images/Ubuntu_Server-hwe-x86_64
removing: /var/lib/tftpboot/pxeLinux.cfg/default
removing: /var/lib/tftpboot/pxeLinux.cfg/01-00-0c-29-4c-71-7c
removing: /var/lib/tftpboot/grub/01-00-0C-29-4C-71-7C
removing: /var/lib/tftpboot/grub/efidefault
removing: /var/lib/tftpboot/grub/grub-x86_64.efi
removing: /var/lib/tftpboot/grub/images
removing: /var/lib/tftpboot/grub/grub-x86.efi
removing: /var/lib/tftpboot/images/CentOS-7-Minimal-1708-x86_64
removing: /var/lib/tftpboot/images/Ubuntu_Server-x86_64
removing: /var/lib/tftpboot/images/Ubuntu_Server-hwe-x86_64
removing: /var/lib/tftpboot/s390x/profile_list
copying bootloaders
trying hardlink /var/lib/cobbler/loaders/grub-x86_64.efi -> /var/lib/tftpboot/grub/grub-x86_64.efi
trying hardlink /var/lib/cobbler/loaders/grub-x86.efi -> /var/lib/tftpboot/grub/grub-x86.efi
copying distros to tftpboot
copying files for distro: Ubuntu_Server-x86_64
trying hardlink /var/www/cobbler/ks_mirror/Ubuntu_Server-x86_64/install/netboot/ubuntu-installer/amd64/Linux -> /var/lib/tftpboot/images/Ubuntu_Server-x86_64/Linux
trying hardlink /var/www/cobbler/ks_mirror/Ubuntu_Server-x86_64/install/netboot/ubuntu-installer/amd64/initrd.gz -> /var/lib/tftpboot/images/Ubuntu_Server-x86_64/initrd.gz
copying files for distro: Ubuntu_Server-hwe-x86_64
trying hardlink /var/www/cobbler/ks_mirror/Ubuntu_Server-x86_64/install/hwe-netboot/ubuntu-installer/amd64/Linux -> /var/lib/tftpboot/images/Ubuntu_Server-hwe-x86_64/Linux
trying hardlink /var/www/cobbler/ks_mirror/Ubuntu_Server-x86_64/install/hwe-netboot/ubuntu-installer/amd64/initrd.gz -> /var/lib/tftpboot/images/Ubuntu_Server-hwe-x86_64/initrd.gz
copying files for distro: CentOS-7-Minimal-1708-x86_64
trying hardlink /var/www/cobbler/ks_mirror/CentOS-7-x86_64-Minimal-1708-x86_64/images/pxeboot/vmlinuz -> /var/lib/tftpboot/images/CentOS-7-Minimal-1708-x86_64/vmlinuz
trying hardlink /var/www/cobbler/ks_mirror/CentOS-7-x86_64-Minimal-1708-x86_64/images/pxeboot/initrd.img -> /var/lib/tftpboot/images/CentOS-7-Minimal-1708-x86_64/initrd.img
copying images
generating PXE configuration files
generating: /var/lib/tftpboot/pxeLinux.cfg/01-00-0c-29-4c-71-7c
generating: /var/lib/tftpboot/grub/01-00-0C-29-4C-71-7C
generating PXE menu structure
copying files for distro: Ubuntu_Server-x86_64
trying hardlink /var/www/cobbler/ks_mirror/Ubuntu_Server-x86_64/install/netboot/ubuntu-installer/amd64/Linux -> /var/www/cobbler/images/Ubuntu_Server-x86_64/Linux
trying hardlink /var/www/cobbler/ks_mirror/Ubuntu_Server-x86_64/install/netboot/ubuntu-installer/amd64/initrd.gz -> /var/www/cobbler/images/Ubuntu_Server-x86_64/initrd.gz
Writing template files for Ubuntu_Server-x86_64
copying files for distro: Ubuntu_Server-hwe-x86_64
trying hardlink /var/www/cobbler/ks_mirror/Ubuntu_Server-x86_64/install/hwe-netboot/ubuntu-installer/amd64/Linux -> /var/www/cobbler/images/Ubuntu_Server-hwe-x86_64/Linux
trying hardlink /var/www/cobbler/ks_mirror/Ubuntu_Server-x86_64/install/hwe-netboot/ubuntu-installer/amd64/initrd.gz -> /var/www/cobbler/images/Ubuntu_Server-hwe-x86_64/initrd.gz
Writing template files for Ubuntu_Server-hwe-x86_64
copying files for distro: CentOS-7-Minimal-1708-x86_64
trying hardlink /var/www/cobbler/ks_mirror/CentOS-7-x86_64-Minimal-1708-x86_64/images/pxeboot/vmlinuz -> /var/www/cobbler/images/CentOS-7-Minimal-1708-x86_64/vmlinuz
trying hardlink /var/www/cobbler/ks_mirror/CentOS-7-x86_64-Minimal-1708-x86_64/images/pxeboot/initrd.img -> /var/www/cobbler/images/CentOS-7-Minimal-1708-x86_64/initrd.img
Writing template files for CentOS-7-Minimal-1708-x86_64
rendering DHCP files
rendering DNS files
rendering TFTPD files
generating /etc/xinetd.d/tftp
processing boot_files for distro: Ubuntu_Server-x86_64
processing boot_files for distro: Ubuntu_Server-hwe-x86_64
processing boot_files for distro: CentOS-7-Minimal-1708-x86_64
cleaning link caches
running post-sync triggers
running python triggers from /var/lib/cobbler/triggers/sync/post/*
running python trigger cobbler.modules.sync_post_restart_services
running: service dnsmasq restart
received on stdout: 
received on stderr: Redirecting to /bin/systemctl restart dnsmasq.service

running shell triggers from /var/lib/cobbler/triggers/sync/post/*
running python triggers from /var/lib/cobbler/triggers/change/*
running python trigger cobbler.modules.scm_track
running shell triggers from /var/lib/cobbler/triggers/change/*
*** TASK COMPLETE ***
```

一旦您启动了 CentOS 客户端，您将注意到它进入 PXE 过程并通过`PXE_Network`发送 DHCP 消息。Cobbler 将以 MAC 地址分配一个 IP 地址、一个`PXELinux0`文件和所需的镜像来响应：

![](img/00145.jpeg)

在 Cobbler 完成 CentOS 安装后，您将看到主机名在机器中正确配置：

![](img/00146.jpeg)

您可以为 Ubuntu 机器执行相同的步骤。

# 摘要

在本章中，您学习了如何通过在虚拟化程序上安装两台 Linux 机器（CentOS 和 Ubuntu）来准备实验室环境。然后，我们探讨了自动化选项，并通过安装 Cobbler 加快了服务器部署速度。

在下一章中，您将学习如何从 Python 脚本直接向操作系统 shell 发送命令并调查返回的输出。
