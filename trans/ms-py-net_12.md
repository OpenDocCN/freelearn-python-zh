# 使用 Jenkins 进行持续集成

网络触及技术堆栈的每个部分；在我工作过的所有环境中，它总是一个零级服务。它是其他服务依赖的基础服务。在其他工程师、业务经理、运营商和支持人员的心目中，网络应该只是工作。它应该始终可访问并且功能正常——一个好的网络是一个没有人听说过的网络。

当然，作为网络工程师，我们知道网络和其他技术堆栈一样复杂。由于其复杂性，构成运行网络的构件有时可能很脆弱。有时，我看着一个网络，想知道它怎么可能工作，更不用说它是如何在数月甚至数年内运行而没有对业务产生影响的。

我们对网络自动化感兴趣的部分原因是为了找到可靠和一致地重复我们的网络变更流程的方法。通过使用 Python 脚本或 Ansible 框架，我们可以确保所做的变更保持一致并可靠地应用。正如我们在上一章中看到的，我们可以使用 Git 和 GitHub 可靠地存储流程的组件，如模板、脚本、需求和文件。构成基础设施的代码是经过版本控制、协作和对变更负责的。但我们如何将所有这些部分联系在一起呢？在本章中，我们将介绍一个流行的开源工具，可以优化网络管理流程，名为 Jenkins。

# 传统的变更管理流程

对于在大型网络环境中工作过的工程师来说，他们知道网络变更出错的影响可能很大。我们可以进行数百次变更而没有任何问题，但只需要一个糟糕的变更就能导致网络对业务产生负面影响。

关于网络故障导致业务痛苦的故事数不胜数。2011 年最显著和大规模的 AWS EC2 故障是由于我们在 AWS US-East 地区的正常扩展活动中的网络变更引起的。变更发生在 PDT 时间 00:47，并导致各种服务出现 12 小时以上的停机，给亚马逊造成了数百万美元的损失。更重要的是，这个相对年轻的服务的声誉受到了严重打击。IT 决策者将这次故障作为“不要”迁移到 AWS 云的理由。花了多年时间才重建了其声誉。您可以在[`aws.amazon.com/message/65648/`](https://aws.amazon.com/message/65648/)阅读更多关于事故报告的信息。

由于其潜在影响和复杂性，在许多环境中，都实施了网络变更咨询委员会（CAB）。典型的 CAB 流程如下：

1.  网络工程师将设计变更并详细列出所需的步骤。这可能包括变更的原因、涉及的设备、将要应用或删除的命令、如何验证输出以及每个步骤的预期结果。

1.  通常要求网络工程师首先从同行那里获得技术审查。根据变更的性质，可能需要不同级别的同行审查。简单的变更可能需要单个同行技术审查；复杂的变更可能需要高级指定工程师批准。

1.  CAB 会议通常按照固定时间安排，也可以临时召开紧急会议。

1.  工程师将变更提交给委员会。委员会将提出必要的问题，评估影响，并批准或拒绝变更请求。

1.  变更将在预定的变更窗口进行，由原始工程师或其他工程师执行。

这个过程听起来合理和包容，但在实践中证明有一些挑战：

+   **撰写文稿耗时**：设计工程师通常需要花费很多时间来撰写文档，有时写作过程所需时间比应用变更的时间还长。这通常是因为所有网络更改都可能产生影响，我们需要为技术和非技术 CAB 成员记录过程。

+   **工程师专业知识**：有不同水平的工程专业知识，有些经验更丰富，他们通常是最受欢迎的资源。我们应该保留他们的时间来解决最复杂的网络问题，而不是审查基本的网络更改。

+   **会议耗时**：组织会议和让每个成员出席需要很多精力。如果需要批准的人员正在度假或生病会发生什么？如果您需要在预定的 CAB 时间之前进行网络更改呢？

这些只是基于人的 CAB 流程的一些更大的挑战。就我个人而言，我非常讨厌 CAB 流程。我不否认对同行审查和优先级排序的需求；但是，我认为我们需要尽量减少潜在的开销。让我们看看在软件工程流程中采用的潜在流程。

# 持续集成简介

在软件开发中的**持续集成（CI）**是一种快速发布对代码库的小更改的方式，同时进行测试和验证。关键是对可以进行 CI 兼容的更改进行分类，即不过于复杂，并且足够小，以便可以轻松撤销。测试和验证过程是以自动化方式构建的，以获得对其将被应用而不会破坏整个系统的信心基线。

在 CI 之前，对软件的更改通常是以大批量进行的，并且通常需要一个漫长的验证过程。开发人员可能需要几个月才能看到他们的更改在生产中生效，获得反馈并纠正任何错误。简而言之，CI 流程旨在缩短从想法到变更的过程。

一般的工作流程通常包括以下步骤：

1.  第一位工程师获取代码库的当前副本并进行更改

1.  第一位工程师向仓库提交变更

1.  仓库可以通知需要的人员仓库的变化，以便一组工程师审查变化。他们可以批准或拒绝变更

1.  持续集成系统可以持续地从仓库中获取变更，或者当变更发生时，仓库可以向 CI 系统发送通知。无论哪种方式，CI 系统都将获取代码的最新版本

1.  CI 系统将运行自动化测试，以尝试捕捉任何故障

1.  如果没有发现故障，CI 系统可以选择将更改合并到主代码中，并可选择部署到生产系统

这是一个概括的步骤列表。对于每个组织，流程可能会有所不同；例如，可以在提交增量代码后立即运行自动化测试，而不是在代码审查后运行。有时，组织可能选择在步骤之间进行人工工程师参与进行理智检查。

在下一节中，我们将说明在 Ubuntu 16.04 系统上安装 Jenkins 的说明。

# 安装 Jenkins

在本章中我们将使用的示例中，我们可以在管理主机或单独的机器上安装 Jenkins。我个人偏好将其安装在单独的虚拟机上。到目前为止，虚拟机将具有与管理主机相似的网络设置，一个接口用于互联网连接，另一个接口用于 VMNet 2 连接到 VIRL 管理网络。

Jenkins 镜像和每个操作系统的安装说明可以在[`jenkins.io/download/`](https://jenkins.io/download/)找到。以下是我在 Ubuntu 16.04 主机上安装 Jenkins 所使用的说明：

```py
$ wget -q -O - https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo apt-key add -

# added Jenkins to /etc/apt/sources.list
$ cat /etc/apt/sources.list | grep jenkins
deb https://pkg.jenkins.io/debian-stable binary/

# install Java8
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt update; sudo apt install oracle-java8-installer

$ sudo apt-get update
$ sudo apt-get install jenkins

# Start Jenkins
$ /etc/init.d/jenkins start
```

在撰写本文时，我们必须单独安装 Java，因为 Jenkins 不适用于 Java 9；有关更多详细信息，请参阅[`issues.jenkins-ci.org/browse/JENKINS-40689`](https://issues.jenkins-ci.org/browse/JENKINS-40689)。希望在您阅读本文时，该问题已得到解决。

Jenkins 安装完成后，我们可以将浏览器指向端口`8080`的 IP 地址以继续该过程：

![](img/d3769c61-58a4-44da-971f-7bc86ec69c20.png)

解锁 Jenkins 屏幕

如屏幕上所述，从`/var/lib/jenkins/secrets/initialAdminPassword`获取管理员密码，并将输出粘贴到屏幕上。暂时，我们将选择“安装建议的插件”选项：

![](img/e2c7bdc2-7431-48ea-8b79-8388ba52f17a.png)

安装建议的插件

创建管理员用户后，Jenkins 将准备就绪。如果您看到 Jenkins 仪表板，则安装成功：

![](img/251e88e1-8c3e-466f-928e-509f710bd880.png)

Jenkins 仪表板

我们现在准备使用 Jenkins 来安排我们的第一个作业。

# Jenkins 示例

在本节中，我们将看一些 Jenkins 示例以及它们如何与本书中涵盖的各种技术联系在一起。Jenkins 之所以是本书的最后一章，是因为它将利用许多其他工具，例如我们的 Python 脚本、Ansible、Git 和 GitHub。如有需要，请随时参阅第十一章，*使用 Git*。

在示例中，我们将使用 Jenkins 主服务器来执行我们的作业。在生产中，建议添加 Jenkins 节点来处理作业的执行。

在我们的实验中，我们将使用一个简单的带有 IOSv 设备的两节点拓扑结构：

![](img/6c297d94-b1ca-47c6-a287-058426b828d9.png)

第十二章实验拓扑

让我们构建我们的第一个作业。

# Python 脚本的第一个作业

对于我们的第一个作业，让我们使用我们在第二章中构建的 Parmiko 脚本，*低级网络设备交互*，`chapter2_3.py`。如果您还记得，这是一个使用`Paramiko`对远程设备进行`ssh`并获取设备的`show run`和`show version`输出的脚本：

```py
$ ls
chapter12_1.py
$ python3 /home/echou/Chapter12/chapter12_1.py
...
$ ls
chapter12_1.py iosv-1_output.txt iosv-2_output.txt
```

我们将使用“创建新作业”链接来创建作业，并选择“自由风格项目”选项：

![](img/44832874-d06e-4afa-bfed-9389c6d67b28.png)

示例 1 自由风格项目

我们将保留所有默认设置和未选中的内容；选择“执行 shell”作为构建选项：

![](img/d1bb4d78-f774-4400-8eb6-47f54d4c10c8.png)

示例 1 构建步骤

当提示出现时，我们将输入与 shell 中使用的确切命令：

![](img/11251fcf-41d1-4c61-974e-fb29d3557e4b.png)

示例 1shell 命令

一旦我们保存了作业配置，我们将被重定向到项目仪表板。我们可以选择立即构建选项，作业将出现在构建历史下：

![](img/0127ca4d-f50f-4e23-9f6b-bf65a99f2d25.png)

示例 1 构建

您可以通过单击作业并在左侧面板上选择“控制台输出”来检查构建的状态：

![](img/124a9350-e1b1-4f6e-b92f-a577a029c6b7.png)

示例 1 控制台输出

作为可选步骤，我们可以按照固定间隔安排此作业，就像 cron 为我们所做的那样。作业可以在“构建触发器”下安排，选择“定期构建”并输入类似 cron 的计划。在此示例中，脚本将每天在 02:00 和 22:00 运行。

![](img/e62e60a1-3565-4665-8aa4-057ec7b5e2a8.png)

示例 1 构建触发器

我们还可以在 Jenkins 上配置 SMTP 服务器以允许构建结果的通知。首先，我们需要在主菜单下的“管理 Jenkins | 配置系统”中配置 SMTP 服务器设置：

![](img/445a095e-4b20-45df-a321-6d73b5152199.png)

示例 1 配置系统

我们将在页面底部看到 SMTP 服务器设置。单击“高级设置”以配置 SMTP 服务器设置以及发送测试电子邮件：

![](img/9f77d424-47bc-45ef-bee1-b5afee81c341.png)

示例 1 配置 SMTP

我们将能够配置电子邮件通知作为作业的后续操作的一部分：

![](img/50afeec2-613d-41a8-9ba2-1a26f5f87d90.png)

示例 1 电子邮件通知

恭喜！我们刚刚使用 Jenkins 创建了我们的第一个作业。从功能上讲，这并没有比我们的管理主机实现更多的功能。然而，使用 Jenkins 有几个优点：

+   我们可以利用 Jenkins 的各种数据库认证集成，比如 LDAP，允许现有用户执行我们的脚本。

+   我们可以使用 Jenkins 的基于角色的授权来限制用户。例如，一些用户只能执行作业而没有修改访问权限，而其他用户可以拥有完全的管理访问权限。

+   Jenkins 提供了一个基于 Web 的图形界面，允许用户轻松访问脚本。

+   我们可以使用 Jenkins 的电子邮件和日志服务来集中我们的作业并收到结果通知。

Jenkins 本身就是一个很好的工具。就像 Python 一样，它有一个庞大的第三方插件生态系统，可以用来扩展其功能和功能。

# Jenkins 插件

我们将安装一个简单的计划插件作为说明插件安装过程的示例。插件在“管理 Jenkins | 管理插件”下进行管理：

![](img/867ca9d1-9540-42a2-9a55-da6e7f887c5b.png)

Jenkins 插件

我们可以使用搜索功能在可用选项卡下查找计划构建插件：

![](img/79da400f-bb21-4fa6-8695-4f576a64bc83.png)

Jenkins 插件搜索

然后，我们只需点击“安装而不重启”，我们就能在接下来的页面上检查安装进度：

![](img/4cecebf1-d057-4bb2-a1b8-6bb21cdf1f30.png)

Jenkins 插件安装

安装完成后，我们将能够看到一个新的图标，允许我们更直观地安排作业：

![](img/d6b1fccc-091e-435a-9f7e-e24bcdafa7a3.png)

Jenkins 插件结果

作为一个流行的开源项目的优势之一是能够随着时间的推移而增长。对于 Jenkins 来说，插件提供了一种为不同的客户需求定制工具的方式。在接下来的部分，我们将看看如何将版本控制和批准流程集成到我们的工作流程中。

# 网络持续集成示例

在这一部分，让我们将我们的 GitHub 存储库与 Jenkins 集成。通过集成 GitHub 存储库，我们可以利用 GitHub 的代码审查和协作工具。

首先，我们将创建一个新的 GitHub 存储库，我将把这个存储库称为`chapter12_example2`。我们可以在本地克隆这个存储库，并将我们想要的文件添加到存储库中。在这种情况下，我正在添加一个将`show version`命令的输出复制到文件中的 Ansible playbook：

```py
$ cat chapter12_playbook.yml
---
- name: show version
  hosts: "ios-devices"
  gather_facts: false
  connection: local

  vars:
    cli:
      host: "{{ ansible_host }}"
      username: "{{ ansible_user }}"
      password: "{{ ansible_password }}"

  tasks:
    - name: show version
      ios_command:
        commands: show version
        provider: "{{ cli }}"

      register: output

    - name: show output
      debug:
        var: output.stdout

    - name: copy output to file
      copy: content="{{ output }}" dest=./output/{{ inventory_hostname }}.txt
```

到目前为止，我们应该已经非常熟悉了运行 Ansible playbook。我将跳过`host_vars`和清单文件的输出。然而，最重要的是在提交到 GitHub 存储库之前验证它在本地机器上运行：

```py
$ ansible-playbook -i hosts chapter12_playbook.yml

PLAY [show version] **************************************************************

TASK [show version] **************************************************************
ok: [iosv-1]
ok: [iosv-2]
...
TASK [copy output to file] *******************************************************
changed: [iosv-1]
changed: [iosv-2]

PLAY RECAP ***********************************************************************
iosv-1 : ok=3 changed=1 unreachable=0 failed=0
iosv-2 : ok=3 changed=1 unreachable=0 failed=0
```

我们现在可以将 playbook 和相关文件推送到我们的 GitHub 存储库：

![](img/ef05539d-35a8-4d2d-93e1-c394c8c672a9.png)

示例 2GitHub 存储库

让我们重新登录 Jenkins 主机安装`git`和 Ansible：

```py
$ sudo apt-get install git
$ sudo apt-get install software-properties-common
$ sudo apt-get update
$ sudo apt-get install ansible
```

一些工具可以在全局工具配置下安装；Git 就是其中之一。然而，由于我们正在安装 Ansible，我们可以在同一个命令提示符下安装 Git：

![](img/cbe499e5-4768-462b-8b9e-8ec7c4b341d1.png)

全局工具配置

我们可以创建一个名为`chapter12_example2`的新自由样式项目。在源代码管理下，我们将指定 GitHub 存储库作为源：

![](img/02120bae-0753-4b1b-96c0-395cbd62d7c7.png)

示例 2 源代码管理

在我们进行下一步之前，让我们保存项目并运行构建。在构建控制台输出中，我们应该能够看到存储库被克隆，索引值与我们在 GitHub 上看到的匹配：

![](img/6e43af16-88fb-4710-9e71-65561a16403a.png)

示例 2 控制台输出 1

现在我们可以在构建部分中添加 Ansible playbook 命令：

![](img/b0a9e905-9f58-4f88-92f0-1162b87d63d3.png)

示例 2 构建 shell

如果我们再次运行构建，我们可以从控制台输出中看到 Jenkins 将在执行 Ansible playbook 之前从 GitHub 获取代码：

![](img/7262af09-7d6c-4e5d-a11a-bc5cf0533bc1.png)

示例 2 构建控制台输出 2

将 GitHub 与 Jenkins 集成的好处之一是我们可以在同一个屏幕上看到所有 Git 信息：

![](img/6098f97c-31f5-4e7e-8ede-8b4375702f56.png)

示例 2 Git 构建数据

项目的结果，比如 Ansible playbook 的输出，可以在`workspace`文件夹中看到：

![](img/484f53e1-38ab-4e6a-a79c-72ac8c79dddd.png)

示例 2 工作空间

此时，我们可以按照之前的步骤使用周期性构建作为构建触发器。如果 Jenkins 主机是公开访问的，我们还可以使用 GitHub 的 Jenkins 插件将 Jenkins 作为构建的触发器。这是一个两步过程，第一步是在您的 GitHub 存储库上启用插件：

![](img/abaf72ff-5399-456e-8cf5-2e38279ae55d.png)

示例 2 GitHub Jenkins 服务

第二步是将 GitHub 挂钩触发器指定为我们项目的构建触发器：

![](img/34ffd332-eb02-4ab6-8ee4-0634823689b7.png)

示例 2 Jenkins 构建触发器

将 GitHub 存储库作为源，可以为处理基础设施提供全新的可能性。我们现在可以使用 GitHub 的分叉、拉取请求、问题跟踪和项目管理工具来高效地共同工作。一旦代码准备就绪，Jenkins 可以自动拉取代码并代表我们执行。

您会注意到我们没有提到任何关于自动化测试的内容。我们将在第十三章中讨论测试，*网络驱动开发*。

Jenkins 是一个功能齐全的系统，可能会变得复杂。我们在本章中只是浅尝辄止。Jenkins 流水线、环境设置、多分支流水线等都是非常有用的功能，可以适应最复杂的自动化项目。希望本章能为您进一步探索 Jenkins 工具提供有趣的介绍。

# 使用 Python 与 Jenkins

Jenkins 为其功能提供了完整的 REST API：[`wiki.jenkins.io/display/JENKINS/Remote+access+API`](https://wiki.jenkins.io/display/JENKINS/Remote+access+API)。还有许多 Python 包装器，使交互更加容易。让我们来看看 Python-Jenkins 包：

```py
$ sudo pip3 install python-jenkins
$ python3
>>> import jenkins
>>> server = jenkins.Jenkins('http://192.168.2.123:8080', username='<user>', password='<pass>')
>>> user = server.get_whoami()
>>> version = server.get_version()
>>> print('Hello %s from Jenkins %s' % (user['fullName'], version))
Hello Admin from Jenkins 2.121.2
```

我们可以与服务器管理一起工作，比如`插件`：

```py
>>> plugin = server.get_plugins_info()
>>> plugin
[{'supportsDynamicLoad': 'MAYBE', 'downgradable': False, 'requiredCoreVersion': '1.642.3', 'enabled': True, 'bundled': False, 'shortName': 'pipeline-stage-view', 'url': 'https://wiki.jenkins-ci.org/display/JENKINS/Pipeline+Stage+View+Plugin', 'pinned': False, 'version': 2.10, 'hasUpdate': False, 'deleted': False, 'longName': 'Pipeline: Stage View Plugin', 'active': True, 'backupVersion': None, 'dependencies': [{'shortName': 'pipeline-rest-api', 'version': '2.10', 'optional': False}, {'shortName': 'workflow-job', 'version': '2.0', 'optional': False}, {'shortName': 'handlebars', 'version': '1.1', 'optional': False}...
```

我们还可以管理 Jenkins 作业：

```py
>>> job = server.get_job_config('chapter12_example1')
>>> import pprint
>>> pprint.pprint(job)
("<?xml version='1.1' encoding='UTF-8'?>\n"
 '<project>\n'
 ' <actions/>\n'
 ' <description>Paramiko Python Script for Show Version and Show '
 'Run</description>\n'
 ' <keepDependencies>false</keepDependencies>\n'
 ' <properties>\n'
 ' <jenkins.model.BuildDiscarderProperty>\n'
 ' <strategy class="hudson.tasks.LogRotator">\n'
 ' <daysToKeep>10</daysToKeep>\n'
 ' <numToKeep>5</numToKeep>\n'
 ' <artifactDaysToKeep>-1</artifactDaysToKeep>\n'
 ' <artifactNumToKeep>-1</artifactNumToKeep>\n'
 ' </strategy>\n'
 ' </jenkins.model.BuildDiscarderProperty>\n'
 ' </properties>\n'
 ' <scm class="hudson.scm.NullSCM"/>\n'
 ' <canRoam>true</canRoam>\n'
 ' <disabled>false</disabled>\n'
 ' '
 '<blockBuildWhenDownstreamBuilding>false</blockBuildWhenDownstreamBuilding>\n'
 ' <blockBuildWhenUpstreamBuilding>false</blockBuildWhenUpstreamBuilding>\n'
 ' <triggers>\n'
 ' <hudson.triggers.TimerTrigger>\n'
 ' <spec>0 2,20 * * *</spec>\n'
 ' </hudson.triggers.TimerTrigger>\n'
 ' </triggers>\n'
 ' <concurrentBuild>false</concurrentBuild>\n'
 ' <builders>\n'
 ' <hudson.tasks.Shell>\n'
 ' <command>python3 /home/echou/Chapter12/chapter12_1.py</command>\n'
 ' </hudson.tasks.Shell>\n'
 ' </builders>\n'
 ' <publishers/>\n'
 ' <buildWrappers/>\n'
 '</project>')
>>>
```

使用 Python-Jenkins 使我们有一种以编程方式与 Jenkins 进行交互的方法。

# 网络连续集成

连续集成在软件开发领域已经被采用了一段时间，但在网络工程领域相对较新。我们承认，在网络基础设施中使用连续集成方面我们有些落后。毫无疑问，当我们仍在努力摆脱使用 CLI 来管理设备时，将我们的网络视为代码是一项挑战。

有许多很好的使用 Jenkins 进行网络自动化的例子。其中一个是由 Tim Fairweather 和 Shea Stewart 在 AnsibleFest 2017 网络跟踪中提出的：[`www.ansible.com/ansible-for-networks-beyond-static-config-templates`](https://www.ansible.com/ansible-for-networks-beyond-static-config-templates)。另一个用例是由 Dyn 的 Carlos Vicente 在 NANOG 63 上分享的：[`www.nanog.org/sites/default/files/monday_general_autobuild_vicente_63.28.pdf`](https://www.nanog.org/sites/default/files/monday_general_autobuild_vicente_63.28.pdf)。

即使持续集成对于刚开始学习编码和工具集的网络工程师来说可能是一个高级话题，但在我看来，值得努力学习和在生产中使用持续集成。即使在基本水平上，这种经验也会激发出更多创新的网络自动化方式，无疑会帮助行业向前发展。

# 总结

在本章中，我们研究了传统的变更管理流程，以及为什么它不适合当今快速变化的环境。网络需要与业务一起发展，变得更加敏捷，能够快速可靠地适应变化。

我们研究了持续集成的概念，特别是开源的 Jenkins 系统。Jenkins 是一个功能齐全、可扩展的持续集成系统，在软件开发中被广泛使用。我们安装并使用 Jenkins 来定期执行基于`Paramiko`的 Python 脚本，并进行电子邮件通知。我们还看到了如何安装 Jenkins 的插件来扩展其功能。

我们看了如何使用 Jenkins 与我们的 GitHub 存储库集成，并根据代码检查触发构建。通过将 Jenkins 与 GitHub 集成，我们可以利用 GitHub 的协作流程。

在第十三章中，《面向网络的测试驱动开发》，我们将学习如何使用 Python 进行测试驱动开发。
