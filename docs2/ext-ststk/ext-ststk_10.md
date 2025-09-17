# 第十章 监控信标

**信标**是 Salt 中的一种新型模块，旨在监视 Minion 上的资源，并在这些资源与您期望它们看起来不一致时向主节点报告。在本章中，我们将讨论：

+   使用 Salt 监控外部系统

+   信标故障排除

# 监视数据

监控服务有两种基本类型：那些记录数据的，以及基于那些数据触发警报的。表面上，信标可能看起来像是第二种类型。它们以常规间隔运行（默认情况下每秒运行一次）并且当它们发现重要的数据时，会将这些数据发送到主节点。

然而，由于信标可以访问它们运行的 Minion 上的执行模块，它们可以与 Minion 上任何执行模块可以交互的程序进行交互。

## 关注事物

让我们继续构建一个监控`nspawn`容器的信标。它不需要非常复杂；实际上，信标应该尽可能简单，因为它们预计会频繁运行。我们的信标只需要关注应该运行和应该不存在的容器。

### 注意

容器在现代数据中心中变得非常流行，这很大程度上归功于 Docker 和 LXC。systemd 有自己的容器系统，称为`nspawn`，它本身就是一个非常强大的系统。现在许多 Linux 发行版都预装了 systemd，这意味着您可能已经安装了`nspawn`。您可以在 Lennart Pottering 的博客上找到关于`nspawn`的更完整讨论：

[`0pointer.net/blog/systemd-for-administrators-part-xxi.html`](http://0pointer.net/blog/systemd-for-administrators-part-xxi.html)

首先，我们需要设置我们的`__virtual__()`函数。由于`nspawn`是`systemd`的一部分，并不是每个 Minion 都安装了`systemd`，因此我们需要对其进行检查。然而，由于我们将使用随 Salt 一起提供的`nspawn`执行模块，并且它已经包含了一个`__virtual__()`函数，我们真正需要做的只是确保它存在：

```py
'''
Send events covering nspawn containers

This beacon accepts a list of containers and whether they should be
running or absent:

beacons:
  nspawn:
    vsftpd: absent
    httpd: running

This file should be saved as salt/beacons/nspawn.py
'''
__virtualname__ = 'nspawn'

def __virtual__():
    '''
    Ensure that systemd-nspawn is available
    '''
    if 'nspawn.list_running' in __salt__:
        return __virtualname__
    return False
```

有针对性地检查`nspawn.list_running`是有意义的，因为这是我们在这里唯一会使用的函数。

## 验证配置

信标不知道要监视哪些数据时不会运行。您可能在前面的文档字符串中看到了配置示例。`validate()`函数检查传递给此信标的配置，以确保其格式正确。

如果我们要对此进行极简处理，那么我们只需检查确保已经传递了正确的数据类型。在我们的例子中，我们期望的是一个字典，所以我们只需检查这一点即可：

```py
def validate(config):
    '''
    Validate the beacon configuration
    '''
    if not isinstance(config, dict):
        return False
    return True
```

但我们将添加一点更多，以确保至少容器列表被设置为所需的值之一：`running`或`absent`：

```py
def validate(config):
    '''
    Validate the beacon configuration
    '''
    if not isinstance(config, dict):
        return False
    for key in config:
        if config[key] not in ('running', 'absent'):
            return False
    return True
```

如果你不需要这个函数，可以跳过它；如果没有它，Salt 会跳过它。然而，保留它是一个好主意，以帮助防止不良配置导致信标崩溃并带有堆栈跟踪。

## beacon() 函数

与其他一些类型的模块一样，信标有一个必需的函数，因为 Salt 在尝试使用模块时会查找它。不出所料，这个函数叫做 `beacon()`。它传递与 `validate()` 函数相同的 `config` 数据。

我们信标的唯一任务是使用 `machinectl` 报告当前在 Minion 上运行的容器。它的输出看起来像以下这样：

```py
# machinectl list
MACHINE       CLASS     SERVICE 
vsftpd         container systemd-nspawn

1 machines listed.

```

我们可以手动调用它并解析输出，但正如我之前说的，Salt 已经附带了一个 `nspawn` 执行模块，它有一个 `list_running()` 函数，可以为我们做所有这些事情。

我们接下来需要做的就是获取报告为正在运行的节点列表，然后将其与 `config` 字典中的节点列表进行匹配：

```py
def beacon(config):
    '''
    Scan for nspawn containers and fire events
    '''
    nodes = __salt__['nspawn.list_running']()
    ret = []
    for name in config:
        if config[name] == 'running':
            if name not in nodes:
                ret.append({name: 'Absent'})
        elif config[name] == 'absent':
            if name in nodes:
                ret.append({name: 'Running'})
        else:
            if name not in nodes:
                ret.append({name: False})

    return ret
```

我们不是逐个检查正在运行的节点列表，而是遍历已配置的节点列表。如果一个本应不存在的节点出现在运行列表中，我们就将其标记为正在运行。如果一个节点应该运行但未出现，我们就将其标记为不存在。

最后的 `else` 语句会通知我们如果列表中出现了一些未被标记为正在运行或不存在的东西。由于我们已经在 `validate()` 函数中进行了这个检查，所以这不应该需要。但保留这种检查并不是一个坏主意，以防你的 `validate()` 函数错过了什么。如果你开始看到这个模块的事件，节点被设置为 `False`，那么你就知道你需要回去检查 `validate()` 函数。

如果你一直在跟进并已经开始了这个模块的测试，你可能注意到了一些，嗯，令人讨厌的事情。默认情况下，信标每秒执行一次。你可以根据每个模块来更改这个间隔：

```py
beacons:
  nspawn:
    vsftpd: present
    httpd: absent
    interval: 30
```

使用这种配置，`nspawn` 信标将每五秒执行一次，而不是每秒执行一次。这将减少噪音，但也意味着你的信标不一定能像你希望的那样频繁地监视。

让我们添加一些代码，这将允许信标以你想要的频率运行，但以不那么规律的频率发送更新。假设你的信标已经与监控服务（通过事件反应器）绑定，并且你想要实时的监控，但不需要每五分钟被告知一次，“哦，顺便说一下，容器仍然处于关闭状态”：

```py
import time
def beacon(config):
    '''
    Scan for nspawn containers and fire events
    '''
    interval = __salt__'config.get'
    now = int(time.time())

    nodes = __salt__['nspawn.list_running']()
    ret = []
    for name in config:
        lasttime = __grains__.get('nspawn_last_notify', {}).get(name, 0)
        if config[name] == 'running':
            if name not in nodes:
                if now - lasttime >= interval:
                    ret.append({name: 'Absent'})
                    __salt__'grains.setval'
        elif config[name] == 'absent':
            if name in nodes:
                if now - lasttime >= interval:
                    ret.append({name: 'Running'})
                    __salt__'grains.setval'
        else:
            if name not in nodes:
                if now - lasttime >= interval:
                    ret.append({name: False})
                        __salt__'grains.setval'

    return ret
```

首先，我们设置了一个名为 `nspawn_alert_interval` 的警报间隔，并将其默认设置为 `360` 秒（或者说，每五分钟一次）。因为我们使用了 `config.get` 来查找它，所以我们可以在这 `master` 或 `minion` 配置文件中，或者在 Minion 的一个 grain 或 pillar 中进行配置。

然后，我们使用 Python 自带的`time.time()`函数记录当前时间。这个函数报告自纪元以来的秒数，这对于我们的目的来说非常完美，因为我们的警报间隔也是以秒为单位的。

当我们遍历配置的节点列表时，我们检查最后一次发送通知的时间。这存储在一个名为`nspawn_last_notify`的 grain 中。这不是用户会更新的 grain；这是信标会跟踪的。

事实上，你会在`if`语句的每个分支中看到这种情况发生。每当信标检测到应该发送警报时，它首先检查在指定的时间间隔内是否已经发送了警报。如果没有，那么它设置一个要返回的事件。

## 监视信标

信标使用 Salt 的事件总线向 Master 发送通知。您可以使用`state`运行器中的`event`函数来监视事件总线上的信标。这个特定信标模块的返回值将如下所示：

```py
salt/beacon/alton/nspawn/	{
    "_stamp": "2016-01-17T17:48:48.986662",
    "data": {
        "vsftpd": "Present",
        "id": "alton"
    },
    "tag": "salt/beacon/alton/nspawn/"
}
```

注意标签，它包含`salt/beacon/`，然后是触发信标的 Minion（`alton`）的 ID，然后是信标本身的名称（`nspawn`）。

# 最后一个信标模块

一切结束后，我们的最终信标模块将如下所示：

```py
'''
Send events covering nspawn containers

This beacon accepts a list of containers and whether they should be
running or absent:

    .. code-block:: yaml

        beacons:
          nspawn:
            vsftpd: running
            httpd: absent

This file should be saved as salt/beacons/nspawn.py
'''
import time

__virtualname__ = 'nspawn'

def __virtual__():
    '''
    Ensure that systemd-nspawn is available
    '''
    if 'nspawn.list_running' in __salt__:
        return __virtualname__
    return False

def validate(config):
    '''
    Validate the beacon configuration
    '''
    if not isinstance(config, dict):
        return False
    for key in config:
        if config[key] not in ('running', 'absent'):
            return False
    return True

def beacon(config):
    '''
    Scan for nspawn containers and fire events
    '''
    interval = __salt__'config.get'
    now = int(time.time())

    nodes = __salt__['nspawn.list_running']()
    ret = []
    for name in config:
        lasttime = __grains__.get('nspawn_last_notify', {}).get(name, 0)
        if config[name] == 'running':
            if name not in nodes:
                if now - lasttime >= interval:
                    ret.append({name: 'Absent'})
                    __salt__'grains.setval'
        elif config[name] == 'absent':
            if name in nodes:
                if now - lasttime >= interval:
                    ret.append({name: 'Running'})
                    __salt__'grains.setval'
        else:
            if name not in nodes:
                if now - lasttime >= interval:
                    ret.append({name: False})
                    __salt__'grains.setval'

    return ret
```

# 信标故障排除

信标是一种需要运行中的 Master 和运行中的 Minion 的模块。在前景运行`salt-master`服务不会给你太多洞察力，因为代码将在 Minion 上运行，但在前景运行`salt-minion`服务将非常有帮助：

```py
# salt-minion -l debug

```

预留一个只配置了信标而没有其他配置的 Minion。默认情况下，这些信标每秒运行一次，这确实会生成非常嘈杂的日志：

```py
[INFO    ] Executing command 'machinectl --no-legend --no-pager list' in directory '/root'
[DEBUG   ] stdout: vsftpd container systemd-nspawn
[INFO    ] Executing command 'machinectl --no-legend --no-pager list' in directory '/root'
[DEBUG   ] stdout: vsftpd container systemd-nspawn
[INFO    ] Executing command 'machinectl --no-legend --no-pager list' in directory '/root'
[DEBUG   ] stdout: vsftpd container systemd-nspawn

```

想象一下同时运行多个信标，每个信标都记录它当前正在做什么的数据。这会很快变得无聊。

你还希望在 Master 上保持一个事件监听器打开：

```py
# salt-r
un state.event pretty=True

```

```py
salt/beacon/alton/nspawn/	{
    "_stamp": "2016-01-17T17:48:48.986662",
    "data": {
        "ftp-container": "Present",
        "id": "alton"
    },
    "tag": "salt/beacon/alton/nspawn/"
}
```

幸运的是，信标不是那种你需要等待的东西；只需让机器表现出你想要的行为，然后启动`salt-minion`进程。只需确保测试你期望找到的任何行为的变化，无论它是否预期返回一个事件。

# 摘要

信标赋予 Minion 根据监控条件触发事件的能力。一个`validate()`函数有助于确保配置正确，但不是必需的。一个`beacon()`函数是必需的，因为它是执行实际监控的函数。尽可能使用执行模块来执行繁重的工作。信标可以在非常短的时间间隔内运行，但通过让它们在 grains 中存储数据，您可以设置更长的时间间隔的通知。

现在我们已经将书中所有的 Minion 端模块处理完毕，让我们回到 Master 端模块，完成剩余的工作。接下来：扩展 Master。
