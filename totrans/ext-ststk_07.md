# 第七章。使用运行器进行脚本编写

Unix 背后的设计原则之一是程序应该小巧，只做一件事，但要做好。执行模块遵循这一模式，使用通常只做一件事的函数，并将相关函数组合到模块中。当函数被执行时，它执行那个任务，然后返回。

在 Unix 中，这些小程序可以通过 shell 脚本组合在一起，将它们连接成一个更强大的工具。Salt 的运行器系统将脚本元素引入 Salt，使用与 Salt 本身编写相同的语言：Python。在本章中，我们将讨论：

+   连接到 Salt 的本地客户端

+   向执行模块添加额外逻辑

+   运行器的故障排除

# 使用 Salt 的本地客户端

运行器最初被设计在主节点上运行，以将多个作业在 Minions 之间合并成一个完整任务。为了与这些 Minions 通信，运行器需要使用`local_client`。与其他组件不同，这并不是直接集成到运行器中的；你需要自己初始化客户端。让我们快速设置一个示例：

```py
import salt.client
client = salt.client.get_local_client(__opts__['conf_file'])
minions = client.cmd('*', 'test.ping', timeout=__opts__['timeout'])
```

这三条线构成了设置和使用本地客户端的基础。首先，我们导入`salt.client`库。然后，我们实例化一个客户端对象，用于与 Salt 通信。在创建那个客户端对象时，你需要告诉它在哪里可以找到 Salt 的配置文件。幸运的是，这是我们在`__opts__`字典中免费获得的东西，我们不太可能需要更改它，所以你代码中的那一行可能总是看起来与我们在这里做的一模一样。

最后一行使用`client`对象向目标发出命令。从那返回的是在指定超时内响应的 Minions 列表。让我们继续将最后一行分解成组件，并讨论每一个：

```py
minions = client.cmd(
    '*',  # The target to use
    'test.ping',  # The command to issue
    timeout=__opts__['timeout']  # How long to wait for a response
)
```

到目前为止，你应该已经习惯了使用`'*'`作为目标，并且知道它指的是所有的 Minions。你也应该知道`test.ping`是一个标准命令，常用于检查并查看哪些 Minions 正在响应。超时也是必需的，但很少需要使用除配置的超时之外的其他超时，所以`__opts__['timeout']`几乎总是足够的。

## 使用本地客户端进行脚本编写

运行器，就像其他 Salt 模块一样，是基于模块内部的函数。前面的代码在技术上是对的，但它不是用作运行器的地方。让我们继续创建一个名为`scan`的运行器模块，我们将使用它来收集有关所有 Minions 的各种信息：

```py
'''
Scan Minions for various pieces of information

This file should be saved as salt/runners/scan.py
'''
import salt.client

__func_alias__ = {
	'up_': 'up'
}

def up_():
    '''
    Return a list of minions which are responding
    '''
    client = salt.client.get_local_client(__opts__['conf_file'])
    minions = client.cmd('*', 'test.ping', timeout=__opts__['timeout'])
    return sorted(minions.keys())
```

目前我们没有什么，但它作为一个运行器是功能性的。我们的第一个函数叫做`up`，但由于使用少于三个字符的函数名被认为是不好的做法，所以我们将其定义为`up_()`，并使用`__func_alias__`使其可调用为`up`。

此函数将连接到本地客户端，对所有 Minions 发出`test.ping`测试，然后返回一个列表，显示哪些 Minions 响应了。如果我们返回`minions`而不是`minions.keys()`，那么我们会得到一个所有响应的 Minions 及其响应内容的列表。由于我们知道`test.ping`总是会返回`True`（假设它首先返回），我们可以跳过返回这些数据。我们还对 Minions 列表进行了排序，以便于阅读。

要执行此函数，请使用`salt-run`命令：

```py
# salt-run scan.up
- achatz
- dufresne

```

### 注意

为什么不在模块顶部创建客户端连接，以便每个函数都可以访问它？由于加载器以这种方式向 Salt 展示模块，`__opts__`字典仅在函数内部可用，因此我们无法在模块顶部使用它。你可以硬编码正确的路径，但我们都知道，硬编码的数据也是不好的做法，应该避免。

如果你只想定义一次客户端，那么考虑使用一个名为`_get_conn()`的私有函数，它返回连接对象。然而，由于它只包含一行代码，而这行代码不太可能改变，所以可能不值得这么做。

我们创建的`scan.up`函数告诉我们哪些 Minions 正在响应，但你可能更感兴趣的是哪些没有响应。这些更有可能告诉你 Minions 何时出现连接问题。让我们继续添加一个名为`down()`的函数：

```py
import salt.key

def down():
    '''
    Return a list of minions which are NOT responding
    '''
    minions = up_()
    key = salt.key.Key(__opts__)
    keys = key.list_keys()
    return sorted(set(keys['minions']) – set(minions))
```

首先，我们需要知道哪些 Minions 已经响应，但我们已经有一个函数可以报告这一点，所以我们只需使用那个函数的响应。

我们还需要一个预期返回的 Minions 列表。我们可以通过创建一个`salt.key`对象，并请求它提供一个列表，其中包含其密钥已被 Master 接受的 Minions。

现在我们有了应该响应的 Minions 列表，我们就从列表中移除已经响应的 Minions，如果列表中还有剩余的 Minions，那么它们就是我们可以假设已经宕机的 Minions。和之前一样，我们在返回 Minions 列表时已经对它们进行了排序，以便于阅读：

```py
# salt-run scan.down
- adria
- trotter

```

## 使用不同的目标

将`salt-run`命令与`salt`命令区分开来的一个主要不同之处在于无法在命令行上指定目标。这是因为运行者被设计成能够自己确定自己的目标。

让我们继续更新`up_()`和`down()`函数，以便用户不仅可以指定自己的目标，还可以指定目标类型：

```py
def up_(tgt='*', tgt_type='glob'):
    '''
    Return a list of minions which are responding
    '''
    client = salt.client.get_local_client(__opts__['conf_file'])
    minions = client.cmd(
        tgt,
        'test.ping',
        expr_form=tgt_type,
        timeout=__opts__['timeout']
    )
    return sorted(minions.keys())

def down(tgt='*', tgt_type='glob'):
    '''
    Return a list of minions which are NOT responding
    '''
    minions = up_(tgt, tgt_type)

    key = salt.key.Key(__opts__)
    keys = key.list_keys()

    return sorted(set(keys['minions']) - set(minions))
```

在我们的函数中，`tgt`参数指的是目标。本地客户端无论如何都需要指定一个目标，所以我们只需在我们的函数中将`'*'`替换为`tgt`。`tgt_type`是要使用的目标类型。默认情况下，Salt 使用目标类型为`glob`，但用户可以根据需要指定其他类型（`pcre`、`list`等）。在本地客户端中，此参数的名称为`expr_form`。检查`salt --help`命令的输出中的“目标选择选项”，以查看您的 Salt 版本支持哪些选项。

## 结合作业以添加更多逻辑

运行器最强大的功能之一是能够从一个作业的输出中获取信息，并使用它来启动另一个作业。首先，让我们定义一些关于我们基础设施的内容：

+   我们正在使用 Salt Virt 来管理一些虚拟机。

+   一些 Minions 运行虚拟机管理程序；其他是运行在那些虚拟机管理程序内部的虚拟机。还有一些既不运行虚拟机管理程序，也不是虚拟机。

+   正在使用多种不同的操作系统，例如 Suse、CentOS 和 Ubuntu。

考虑到这一点，我们需要运行一个报告，以确定哪些虚拟机管理程序运行在哪些操作系统上。

我们可以使用这个 Salt 命令来发现哪些 Minions 正在运行哪些操作系统：

```py
# salt '*' grains.item os

```

我们可以运行以下命令来找出哪些 Minions 是虚拟化的：

```py
# salt '*' grains.item virtual

```

但是，仅仅因为 Minion 的`virtual`grain 设置为`physical`并不意味着它是一个虚拟机管理程序。我们可以运行以下命令来找出哪些 Minions 正在运行虚拟机管理程序：

```py
# salt '*' virt.is_hyper

```

然而，没有东西可以聚合这些数据并告诉我们哪些虚拟机管理程序正在运行哪些操作系统；因此，让我们编写一个可以做到这一点的函数：

```py
def hyper_os():
    '''
    Return a list of which operating system each hypervisor is running
    '''
    client = salt.client.get_local_client(__opts__['conf_file'])
    minions = client.cmd(
        '*',
        'virt.is_hyper',
        timeout=__opts__['timeout']
    )

    hypers = []
    for minion in minions:
        if minions[minion] is True:
            hypers.append(minion)

    return client.cmd(
        hypers,
        'grains.item',
        arg=('os',),
        expr_form='list',
        timeout=__opts__['timeout']
    )
```

在我们创建`client`对象之后，我们的第一个任务是查看哪些 Minions 实际上正在运行虚拟机管理程序。然后我们遍历该列表，并将它们保存在另一个名为`hypers`的列表中。因为我们以列表形式存储它，所以我们可以再次将`expr_form`为`list`的它传递给客户端。

我们还增加了一些新内容。`grains.item`函数期望一个单一参数，告诉它要查找哪个 grain。当你需要将一系列未命名的参数传递给一个函数时，请将其作为`arg`传递。当我们运行这个运行器时，我们的输出将类似于以下内容：

```py
# salt-run scan.hyper_os
dufresne:
 ----------
 os:
 Arch

```

假设我们想要能够在显示在虚拟机管理程序列表中的任何机器上运行任意的 Salt 命令。在我们的下一部分代码中，我们将做两件事。我们将把`hyper_os()`拆分成两个函数，分别称为`hypers()`和`hyper_os()`，然后添加一个名为`hyper_cmd()`的新函数，该函数将使用`hypers()`函数：

```py
def hypers(client=None):
    '''
    Return a list of Minions that are running hypervisors
    '''
    if client is None:
        client = salt.client.get_local_client(__opts__['conf_file'])

    minions = client.cmd(
        '*',
        'virt.is_hyper',
        timeout=__opts__['timeout']
    )

    hypers = []
    for minion in minions:
        if minions[minion] is True:
            hypers.append(minion)

    return hypers

def hyper_os():
    '''
    Return a list of which operating system each hypervisor is running
    '''
    client = salt.client.get_local_client(__opts__['conf_file'])

    return client.cmd(
        hypers(client),
        'grains.item',
        arg=('os',),
        expr_form='list',
        timeout=__opts__['timeout']
    )

def hyper_cmd(cmd, arg=None, kwarg=None):
    '''
    Execute an arbitrary command on Minions which run hypervisors
    '''
    client = salt.client.get_local_client(__opts__['conf_file'])

    if arg is None:
        arg = []

    if not isinstance(arg, list):
        arg = [arg]

    if kwarg is None:
        kwarg = {}

    return client.cmd(
        hypers(client),
        cmd,
        arg=arg,
        kwarg=kwarg,
        expr_form='list',
        timeout=__opts__['timeout']
    )
```

你可能会注意到每个函数都能够创建自己的`client`对象，包括`hypers()`。这允许我们单独使用`scan.hypers`。然而，它还允许我们从其他函数中传递一个`client`对象。这可以在创建每个 Salt 命令的单独`client`对象上节省大量时间。

`hyper_cmd()` 函数允许我们以多种方式传递参数，或者如果不需要，则不传递任何参数。不传递任何参数使用它将看起来像这样：

```py
# salt-run scan.hyper_cmd test.ping

```

使用未命名的参数时，它看起来像这样：

```py
# salt-run scan.hyper_cmd test.ping

```

当你传递一个参数列表时，事情开始变得复杂。默认情况下，Salt 能够将命令行中传递的 YAML 转换为 Salt 内部可以使用的数据结构。这意味着你可以运行这个命令：

```py
# salt-run scan.hyper_cmd test.arg [one,two]

```

Salt 将自动将 `[one,two]` 翻译成一个包含 `one` 字符串后跟 `two` 字符串的列表。然而，如果你运行这个命令，情况并非如此：

```py
# salt-run scan.hyper_cmd test.arg one,two

```

在这种情况下，Salt 会认为你传递了一个值为 `one,two` 的字符串。如果你想要允许用户输入这样的列表，你需要手动检测和解析它们。

如果你想要传递命名参数，事情会变得更加复杂。以下是一个有效的例子：

```py
salt-run scan.hyper_cmd network.interface kwarg="{'iface':'wlp3s0'}"

```

但要求用户输入这些内容是非常糟糕的。让我们使用 Python 自身的 `*` 和 `**` 工具来缩小我们的函数，这些工具允许我们从命令行接受任意列表和字典：

```py
def hyper_cmd(cmd, *arg, **kwarg):
    '''
    Execute an arbitrary command on Minions which run hypervisors
    '''
    client = salt.client.get_local_client(__opts__['conf_file'])

    return client.cmd(
        hypers(client),
        cmd,
        arg=arg,
        kwarg=kwarg,
        expr_form='list',
        timeout=__opts__['timeout']
    )
```

现在，我们可以运行以下命令：

```py
# salt-run scan.hyper_cmd test.kwarg iface='wlp3s0'

```

# 最终模块

在我们的代码就绪后，最终的模块将看起来像这样：

```py
'''
Scan Minions for various pieces of information

This file should be saved as salt/runners/scan.py
'''
import salt.client
import salt.key

__func_alias__ = {
    'up_': 'up'
}

def up_(tgt='*', tgt_type='glob'):
    '''
    Return a list of minions which are responding
    '''
    client = salt.client.get_local_client(__opts__['conf_file'])
    minions = client.cmd(
        tgt,
        'test.ping',
        expr_form=tgt_type,
        timeout=__opts__['timeout']
    '''
    Return a list of minions which are NOT responding
    '''
    minions = up_(tgt, tgt_type)

    key = salt.key.Key(__opts__)
    keys = key.list_keys()

    return sorted(set(keys['minions']) - set(minions))

def hypers(client=None):
    '''
    Return a list of Minions that are running hypervisors
    '''
    if client is None:
        client = salt.client.get_local_client(__opts__['conf_file'])

    minions = client.cmd(
        '*',
        'virt.is_hyper',
        timeout=__opts__['timeout']
    )

    hypers = []
    for minion in minions:
        if minions[minion] is True:
            hypers.append(minion)

    return hypers

def hyper_os():
    '''
    Return a list of which operating system each hypervisor is running
    '''
    client = salt.client.get_local_client(__opts__['conf_file'])

    return client.cmd(
        hypers(client),
        'grains.item',
        arg=('os',),
        expr_form='list',
        timeout=__opts__['timeout']
    )

def hyper_cmd(cmd, *arg, **kwarg):
    '''
    Execute an arbitrary command on Minions which run hypervisors
    '''
    client = salt.client.get_local_client(__opts__['conf_file'])

    return client.cmd(
        hypers(client),
        cmd,
        arg=arg,
        kwarg=kwarg,
        expr_form='list',
        timeout=__opts__['timeout']
    )
```

# 运行者的故障排除

在某种程度上，运行者比其他类型的模块更容易进行故障排除。例如，尽管它们在主服务器上运行，但它们不需要重启 `salt-master` 服务来获取新的更改。实际上，除非你使用本地客户端，否则你实际上不需要 `salt-master` 服务在运行。

## 与 salt-master 服务一起工作

如果你使用的是本地客户端，并且尝试在没有 `salt-master` 服务运行的情况下发出命令，你会得到一个看起来像这样的错误：

```py
# salt-run scan.hyper_os
Exception occurred in runner scan.hyper_os: Traceback (most recent call last):
 File "/usr/lib/python2.7/site-packages/salt/client/mixins.py", line 340, in low
 data['return'] = self.functionsfun
 File "/usr/lib/python2.7/site-packages/salt/runners/scan.py", line 68, in hyper_os
 hypers(client),
 File "/usr/lib/python2.7/site-packages/salt/runners/scan.py", line 50, in hypers
 timeout=__opts__['timeout']
 File "/usr/lib/python2.7/site-packages/salt/client/__init__.py", line 562, in cmd
 **kwargs)
 File "/usr/lib/python2.7/site-packages/salt/client/__init__.py", line 317, in run_job
 raise SaltClientError(general_exception)
SaltClientError: Salt request timed out. The master is not responding. If this error persists after verifying the master is up, worker_threads may need to be increased.

```

这是因为，尽管运行者本身不依赖于 `salt-master` 服务，但 Minion 依赖于它来接收命令，并将响应发送回主服务器。

## 超时问题

如果主服务器运行正常，但你没有收到预期的响应，考虑一下你正在触发的目标。对于运行者向所有 Minion 发出命令来说，这是非常常见的，但在大型基础设施中进行测试，或者你的主服务器上有属于无法访问或不再存在的 Minion 的密钥时，运行者命令可能需要很长时间才能返回。

在编写你的模块时，你可能想要考虑将目标从 `'*'` 改为特定的 Minion，或者可能是一个特定的 Minion 列表（`expr_form` 设置为 `'list'`，就像我们在 `hyper_os()` 和 `hyper_cmd()` 函数中所做的那样）。只是确保在生产前将其设置回原样。

# 摘要

运行者向 Salt 添加了一个脚本元素，使用 Python。它们设计在 Master 上运行，但不需要`salt-master`服务正在运行，除非它们正在使用本地客户端向仆从发布命令。运行者设计为能够独立管理目标，但你可以添加元素以允许用户指定目标。它们特别适用于使用一个作业的输出作为另一个作业的输入，这允许你在执行模块周围包装自己的逻辑。

在下一章中，我们将允许大师使用外部资源来存储它为它的仆从（Minions）提供的服务文件。接下来：添加外部文件服务器。
