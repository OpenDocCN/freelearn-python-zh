# 第六章：处理返回数据

当 Salt Master 向 Minion 发出命令并且任务成功完成时，总会存在返回数据。`salt`命令通常会监听返回数据，并且如果它及时返回，它将使用输出器显示。但无论是否发生这种情况，Minion 总会将返回数据发送回 Master，以及任何配置为 Returner 的其他目的地。

本章全部关于处理返回数据，使用 Returner 和 Outputter 模块。我们将讨论：

+   数据如何返回给 Master

+   编写 Returner 模块

+   将 Returners 扩展为用作外部作业缓存

+   Returners 故障排除

+   编写 Outputter 模块

+   Outputters 故障排除

# 将数据返回到外部目的地

处理返回数据最重要的模块类型称为 Returner。当 Master 向目标发布任务（称为作业）时，它会为它分配一个作业 ID（或 JID）。当 Minion 完成该作业时，它会将结果数据连同与其关联的 JID 一起发送回 Master。

# 将数据返回给 Master

Salt 的架构基于发布-订阅模式，俗称 pub/sub。在这个设计中，一个或多个客户端订阅一个消息队列。当消息发布到队列时，任何当前订阅者都会收到一个副本，他们通常会以某种方式处理它。

事实上，Salt 使用了两个消息队列，这两个队列都由 Master 管理。第一个队列由 Master 用于向其 Minions 发布命令。每个 Minion 都可以看到发布到这个队列的消息，但只有当 Minions 包含在目标中时，它们才会做出反应。针对`'*'`的目标消息将由所有连接的 Minions 处理，而使用`-s`命令行选项针对`192.168.0.0/16`的目标消息将只会被以`192.168`开头的 IP 地址的 Minions 处理。

第二个消息队列也由 Master 托管，但消息是由 Minions 发布的，Master 本身是订阅者。这些消息通常存储在 Master 的作业缓存中。Returners 可以被配置为将这些消息发送到其他目的地，并且一些 Returners 也可以使用这些目的地作为作业缓存本身。如果当收到这些消息时`salt`命令仍在监听，那么它也会将这些数据发送到输出器。

# 监听事件数据

每次将消息发布到队列时，Salt 的事件总线也会触发一个事件。您可以使用`state.event`运行器来监听事件总线并实时显示这些消息。

确保您有`salt-master`服务正在运行，并且至少有一台连接到它的机器上的`salt-minion`服务。在 Master 上运行以下命令：

```py
# salt-run state.event

```

在另一个终端中，向一个或多个 Minions 发出命令：

```py
# salt '*' test.ping

```

在运行事件监听器的终端中，您将看到作业发送到 Minions：

```py
Event fired at Sun Dec 20 12:04:15 2015
*************************
Tag: 20151220120415357444
Data:
{'_stamp': '2015-12-20T19:04:15.387417',
 'minions': ['trotter',
             'achatz']}
```

本事件包含的信息仅限于一个时间戳，表示作业创建的时间，以及一个列表，列出了指定的目标（在我们的例子中，所有目标）应执行作业并从中返回数据的 Minions。

这是一个非常小的任务，所以你几乎可以立即看到来自 Minions 的返回数据。因为每个 Minion 都单独响应，所以你会看到每个 Minion 的条目：

```py
Event fired at Sun Dec 20 12:04:15 2015
*************************
Tag: salt/job/20151220120415357444/ret/dufresne
Data:
{'_stamp': '2015-12-20T19:04:15.618340',
 'cmd': '_return',
 'fun': 'test.ping',
 'fun_args': [],
 'id': 'dufresne',
 'jid': '20151220120415357444',
 'retcode': 0,
 'return': True,
 'success': True}
```

注意每个事件使用的标签。当 Master 创建作业时创建的事件有一个只包含 JID 的标签。每个返回事件都包含一个以`salt/job/<JID>/ret/<Minion ID>`命名的命名空间标签。

几秒钟后，salt 命令也将返回，并通知你哪些 Minions 完成了分配给它们的作业，哪些没有完成：

```py
# salt '*' test.ping
achatz:
 True
trotter:
 Minion did not return. [Not connected]

```

在我们的例子中，`achatz`是活跃的，并且能够按照要求返回`True`。不幸的是，`trotter`已经不再存在，所以无法完成我们需要的操作。

## 当返回者监听 Minions

每次 Master 从 Minion 收到响应时，它将调用返回者。如果一个作业针对的是，比如说，400 个 Minions，那么你应该预期返回者将被执行 400 次，每个 Minion 一次。

这通常不是问题。如果一个返回者连接到数据库，那么这个数据库很可能能够快速处理 400 个响应。然而，如果您创建了一个发送消息给人类的返回者，比如 Salt 附带的 SMTP 返回者，那么您可以预期会发送 400 封单独的电子邮件；每个 Minion 一封。

还有一点需要注意：返回者最初是为了在 Minions 上执行而设计的。背后的想法是将工作卸载到 Minions 上，这样在一个大型环境中，Master 就不需要处理所有必要的工作，比如每个 Minion 每次作业连接数据库。

返回者现在可以由 Master 或 Minion 运行，当编写自己的返回者时，你应该预期这两种可能性。我们将在本章后面讨论此配置，当我们谈到作业缓存时。

让我们看看这个动作的一个例子。连接到您的其中一个 Minion 并停止`salt-minion`服务。然后以`info`日志级别在前台启动它：

```py
# salt-minion --log-level info

```

然后连接到 Master 并直接向其发出作业：

```py
# salt dufresne test.ping
dufresne:
 True

```

切换回 Minion，你将看到一些关于作业的信息：

```py
[INFO    ] User sudo_techhat Executing command test.ping with jid 20151220124647074029
[INFO    ] Starting a new job with PID 25016
[INFO    ] Returning information for job: 20151220124647074029
```

现在再次发出命令，但将`--return`标志设置为`local`。此返回者将直接在本地控制台显示返回数据：

```py
# salt dufresne --return local test.ping
dufresne:
 True

```

再次切换回 Minion 以检查返回数据：

```py
[INFO    ] User sudo_techhat Executing command test.ping with jid 20151220124658909637
[INFO    ] Starting a new job with PID 25066
[INFO    ] Returning information for job: 20151220124658909637
{'fun_args': [], 'jid': '20151220124658909637', 'return': True, 'retcode': 0, 'success': True, 'fun': 'test.ping', 'id': 'dufresne'}
```

# 您的第一个返回者

打开`salt/returners/local.py`。这里没有多少内容，但我们感兴趣的是`returner()`函数。它非常非常小：

```py
def returner(ret):
    '''
    Print the return data to the terminal to verify functionality
    '''
    print(ret)
```

实际上，它所做的只是接受返回的数据作为`ret`，然后将其打印到控制台。它甚至不尝试进行任何形式的格式化打印；它只是原样输出。

这实际上是一个返回器所需的最基本内容：一个接受字典的 `returner()` 函数，然后对其进行处理。让我们创建我们自己的返回器，它以 JSON 格式将作业信息本地存储。

```py
'''
Store return data locally in JSON format

This file should be saved as salt/returners/local_json.py
'''
import json
import salt.utils

def returner(ret):
    '''
    Open new file, and save return data to it in JSON format
    '''
    path = '/tmp/salt-{0}-{1}.json'.format(ret['jid'], ret['id'])
    with salt.utils.fopen(path, 'w') as fp_:
        json.dump(ret, fp_)
```

在 Minion 上保存此文件，然后向其发出作业。无论是否重新启动 `salt-minion` 服务，返回器模块都使用 `LazyLoader`。但我们将继续使用 `salt-call`：

```py
# salt-call --local --return local_json test.ping
local:
 True

```

现在请查看 `/tmp/` 目录：

```py
# ls -l /tmp/salt*
-rw-r--r-- 1 root  root  132 Dec 20 13:03 salt-20151220130309936721-dufresne.json

```

如果你查看该文件，你会看到看起来与从本地返回器接收到的数据非常相似，但它是 JSON 格式：

```py
# cat /tmp/salt-20151220130309936721-dufresne.json
{"fun_args": [], "jid": "20151220130309936721", "return": true, "retcode": 0, "success": true, "fun": "test.ping", "id": "dufresne"}

```

## 使用作业缓存

从某种意义上说，我们的 JSON 返回器是一个作业缓存，因为它缓存了返回数据。不幸的是，它不包含任何处理已保存数据的代码。通过更新逻辑并添加一些函数，我们可以扩展其功能。

目前，我们的返回器表现得就像一组日志文件。让我们将其改为更像一个平面文件数据库。我们将使用 JID 作为访问密钥，并根据 JID 中的日期格式化目录结构：

```py
import json
import os.path
import salt.utils
import salt.syspaths

def _job_path(jid):
    '''
    Return the path for the requested JID
    '''
    return os.path.join(
        salt.syspaths.CACHE_DIR,
        'master',
        'json_cache',
        jid[:4],
        jid[4:6],
        jid[6:],
    )

def returner(ret):
    '''
    Open new file, and save return data to it in JSON format
    '''
    path = os.path.join(_job_path(ret['jid']), ret['id']) + '/'
    __salt__'file.makedirs'
    ret_file = os.path.join(path, 'return.json')
    with salt.utils.fopen(ret_file, 'w') as fp_:
        json.dump(ret, fp_)
```

我们没有改变任何东西，除了目录结构及其处理方式。私有函数 `_job_path()` 将标准化目录结构，并可以被未来的函数使用。我们还使用了 `salt.syspaths` 来检测 Salt 在这台机器上配置的缓存文件位置。当针对名为 `dufresne` 的 Minion 运行时，用于存储返回数据的路径将看起来像：

```py
/var/cache/salt/master/json_cache/2015/12/21134608721496/dufresne/return.json
```

我们还需要存储有关作业本身的信息。`return.json` 文件包含一些关于作业的信息，但不是全部。

让我们添加一个保存作业元数据的函数。这个元数据被称为负载，包含一个 `jid`，一个名为 `clear_load` 的字典，它包含大部分元数据，以及一个名为 `minions` 的列表，它将包含所有包含在目标中的 Minions：

```py
def save_load(jid, clear_load, minions=None):
    '''
    Save the load to the specified JID
    '''
    path = os.path.join(_job_path(jid)) + '/'
    __salt__'file.makedirs'

    load_file = os.path.join(path, 'load.json')
    with salt.utils.fopen(load_file, 'w') as fp_:
        json.dump(clear_load, fp_)

    if 'tgt' in clear_load:
        if minions is None:
            ckminions = salt.utils.minions.CkMinions(__opts__)
            # Retrieve the minions list
            minions = ckminions.check_minions(
                    clear_load['tgt'],
                    clear_load.get('tgt_type', 'glob')
                    )
        minions_file = os.path.join(path, 'minions.json')
        with salt.utils.fopen(minions_file, 'w') as fp_:
            json.dump(minions, fp_)
```

再次强调，我们生成数据将被写入的路径。`clear_load` 字典将被写入该路径内的 `load.json` 文件。Minions 的列表有点棘手，因为它可能包含一个空列表。如果是这样，我们使用 `salt.utils.minions` 内的一个名为 `CkMinions` 的类来生成该列表，基于用于作业的目标。一旦我们有了这个列表，我们就将其写入为 `minions.json`。

测试这一点也有点棘手，因为它需要一个由 Master 生成的工作来生成所需的所有元数据。我们还需要让 Master 知道我们正在使用外部作业缓存。

首先，编辑主配置文件并添加一个 `ext_job_cache` 行，将其设置为 `local_json`：

```py
ext_job_cache: local_json
```

### 注意

**外部作业缓存与 Master 作业缓存**

当主节点设置为使用外部工作缓存（使用 `ext_job_cache` 设置）时，返回代码将在从节点上执行。这将减轻主节点的负载，因为每个从节点将记录自己的工作数据，而不是请求主节点。然而，连接到工作缓存（例如，如果使用了数据库）所需的任何凭证都需要从节点可以访问。

当主节点设置为使用主节点工作缓存（使用 `master_job_cache` 设置）时，返回代码将在主节点上执行。这将增加主节点的负载，但可以节省您向从节点提供凭证的麻烦。

一旦您打开了工作缓存，让我们先重启主节点和从节点，然后尝试一下：

```py
# systemctl restart salt-master
# systemctl restart salt-minion
# salt dufresne test.ping
dufresne:
 True
# find /var/cache/salt/master/json_cache/
/var/cache/salt/master/json_cache/2015/12/
/var/cache/salt/master/json_cache/2015/12/21184312454127
/var/cache/salt/master/json_cache/2015/12/21184312454127/load.json
/var/cache/salt/master/json_cache/2015/12/21184312454127/dufresne
/var/cache/salt/master/json_cache/2015/12/21184312454127/dufresne/return.json
/var/cache/salt/master/json_cache/2015/12/21184312454127/minions.json
# cat /var/cache/salt/master/json_cache/2015/12/21184312454127/load.json
{"tgt_type": "glob", "jid": "20151221184312454127", "cmd": "publish", "tgt": "dufresne", "kwargs": {"delimiter": ":", "show_timeout": true, "show_jid": false}, "ret": "local_json", "user": "sudo_larry", "arg": [], "fun": "test.ping"}
# cat /var/cache/salt/master/json_cache/2015/12/21184312454127/minions.json
["dufresne"]

```

现在我们有了保存的信息，但我们没有检索它的方法，除了手动查看文件之外。让我们继续完善我们的返回器，添加一些可以读取数据的函数。

首先，我们需要一个只返回工作负载信息的函数：

```py
def get_load(jid):
    '''
    Return the load data for a specified JID
    '''
    path = os.path.join(_job_path(jid), 'load.json')
    with salt.utils.fopen(path, 'r') as fp_:
        return json.load(fp_)
```

我们还需要一个函数来获取每个工作的工作数据。这两个函数将由 `jobs` 运行者一起使用：

```py
def get_jid(jid):
    '''
    Return the information returned when the specified JID was executed
    '''
    minions_path = os.path.join(_job_path(jid), 'minions.json')
    with salt.utils.fopen(minions_path, 'r') as fp_:
        minions = json.load(fp_)

    ret = {}
    for minion in minions:
        data_path = os.path.join(_job_path(jid), minion, 'return.json')
        with salt.utils.fopen(data_path, 'r') as fp_:
            ret[minion] = json.load(fp_)

    return ret
```

我们不需要重新启动主节点来测试这个功能，因为工作负载运行者不需要主节点正在运行：

```py
# salt-run jobs.print_job 20151221184312454127
20151221184312454127:
 ----------
 Arguments:
 Function:
 test.ping
 Result:
 ----------
 dufresne:
 ----------
 fun:
 test.ping
 fun_args:
 id:
 dufresne
 jid:
 20151221184312454127
 retcode:
 0
 return:
 True
 success:
 True
 StartTime:
 2015, Dec 21 18:43:12.454127
 Target:
 dufresne
 Target-type:
 glob
 User:
 sudo_techhat

```

我们还需要一个函数，该函数返回一个 JIDs 列表，以及它们相关联的工作的一些基本信息。这个函数将使用另一个导入，我们将使用它来快速定位 `load.json` 文件：

```py
import salt.utils.find

def get_jids():
    '''
    Return a dict mapping all JIDs to job information
    '''
    path = os.path.join(
        salt.syspaths.CACHE_DIR,
        'master',
        'json_cache'
    )

    ret = {}
    finder = salt.utils.find.Finder({'name': 'load.json'})
    for file_ in finder.find(path):
        with salt.utils.fopen(file_) as fp_:
            data = json.load(fp_)
        if 'jid' in data:
            ret[data['jid']] = {
                'Arguments': data['arg'],
                'Function': data['fun'],
                'StartTime': salt.utils.jid.jid_to_time(data['jid']),
                'Target': data['tgt'],
                'Target-type': data['tgt_type'],
                'User': data['user'],
            }

    return ret
```

再次使用 `jobs` 运行者测试这个功能：

```py
# salt-run jobs.list_jobs
20151221184312454127:
 ----------
 Arguments:
 Function:
 test.ping
 StartTime:
 2015, Dec 21 18:43:12.454127
 Target:
 dufresne
 Target-type:
 glob
 User:
 sudo_techhat

```

# 最终模块

一旦我们将所有代码编译在一起，最终的模块将看起来像这样：

```py
'''
Store return data locally in JSON format

This file should be saved as salt/returners/local_json.py
'''
import json
import os.path
import salt.utils
import salt.utils.find
import salt.utils.jid
import salt.syspaths

def _job_path(jid):
    '''
    Return the path for the requested JID
    '''
    return os.path.join(
        salt.syspaths.CACHE_DIR,
        'master',
        'json_cache',
        jid[:4],
        jid[4:6],
        jid[6:],
    )

def returner(ret):
    '''
    Open new file, and save return data to it in JSON format
    '''
    path = os.path.join(_job_path(ret['jid']), ret['id']) + '/'
    __salt__'file.makedirs'
    ret_file = os.path.join(path, 'return.json')
    with salt.utils.fopen(ret_file, 'w') as fp_:
        json.dump(ret, fp_)

def save_load(jid, clear_load, minions=None):
    '''
    Save the load to the specified JID
    '''
    path = os.path.join(_job_path(jid)) + '/'
    __salt__'file.makedirs'

    load_file = os.path.join(path, 'load.json')
    with salt.utils.fopen(load_file, 'w') as fp_:
        json.dump(clear_load, fp_)
            minions = ckminions.check_minions(
                    clear_load['tgt'],
                    clear_load.get('tgt_type', 'glob')
                    )
        minions_file = os.path.join(path, 'minions.json')
        with salt.utils.fopen(minions_file, 'w') as fp_:
            json.dump(minions, fp_)

def get_load(jid):
    '''
    Return the load data for a specified JID
    '''
    path = os.path.join(_job_path(jid), 'load.json')
    with salt.utils.fopen(path, 'r') as fp_:
        return json.load(fp_)

def get_jid(jid):
    '''
    Return the information returned when the specified JID was executed
    '''
    minions_path = os.path.join(_job_path(jid), 'minions.json')
    with salt.utils.fopen(minions_path, 'r') as fp_:
        minions = json.load(fp_)

    ret = {}
    for minion in minions:
        data_path = os.path.join(_job_path(jid), minion, 'return.json')
        with salt.utils.fopen(data_path, 'r') as fp_:
            ret[minion] = json.load(fp_)

    return ret

def get_jids():
    '''
    Return a dict mapping all JIDs to job information
    '''
    path = os.path.join(
        salt.syspaths.CACHE_DIR,
        'master',
        'json_cache'
    )

    ret = {}
    finder = salt.utils.find.Finder({'name': 'load.json'})
    for file_ in finder.find(path):
        with salt.utils.fopen(file_) as fp_:
            data = json.load(fp_)
        if 'jid' in data:
            ret[data['jid']] = {
                'Arguments': data['arg'],
                'Function': data['fun'],
                'StartTime': salt.utils.jid.jid_to_time(data['jid']),
                'Target': data['tgt'],
                'Target-type': data['tgt_type'],
                'User': data['user'],
            }

    return ret
```

# 返回器的故障排除

正如您所看到的，有许多不同的 Salt 组件使用不同的返回器部分。其中一些需要主节点正在运行，这使得它们稍微有点难以调试。以下是一些可以帮助的策略。

## 使用 salt-call 进行测试

可以使用 `salt-call` 命令测试 `returner()` 函数。在这种情况下，简单的 `print` 语句可以用来向您的控制台显示信息。如果有拼写错误，Python 将显示错误消息。如果问题涉及技术上有效但仍然有缺陷的代码，那么可以使用 `print` 语句来追踪问题。

## 在主节点运行时进行测试

`save_load()` 函数需要在主节点上生成一个工作负载，到一个或多个从节点。这当然需要主节点和至少一个从节点正在运行。您可以在不同的终端中前台运行它们，以便看到 `print` 语句的输出：

```py
# salt-master --log-level debug
# salt-minion --log-level debug

```

如果您使用 `ext_job_cache`，那么您将想要监控的是从节点。如果您使用 `master_job_cache`，那么请监控主节点。

## 使用运行者进行测试

`get_load()`、`get_jid()` 和 `get_jids()` 函数都是由 `jobs` 运行器使用的。这个运行器不需要 Master 或 Minions 运行；它只要求被返回者使用的数据库可用。再次强调，这些函数内部的 `print` 语句会在使用 `jobs` 运行器时显示信息。

# 编写输出器模块

当使用 `salt` 命令时，在等待期间接收到的任何返回数据都会显示给用户。在这种情况下，输出器模块用于将数据显示到控制台（或者更准确地说，到 `STDOUT`），通常以某种用户友好的格式。

## 序列化我们的输出

因为 Salt 已经自带了一个 `json` 输出器，我们将利用输出数据实际上会输出到 `STDOUT` 的这一事实，并创建一个使用序列化器（`pickle`）可能输出二进制数据的 `outputter`：

```py
'''
Pickle outputter

This file should be saved as salt/output/pickle.py
'''
from __future__ import absolute_import
import pickle

def output(data):
    '''
    Dump out data in pickle format
    '''
    return pickle.dumps(data)
```

这个 `outputter` 的实现非常简单。唯一需要的函数叫做 `output()`，它接受一个字典。字典的名称无关紧要，只要函数定义了一个即可。

`pickle` 库是 Python 内置的，正如你在 `pickle` 渲染器中看到的，它非常容易使用：我们只需告诉它将数据输出到一个字符串，然后返回给 Salt。

如同往常，我们可以使用 `salt-call` 来测试这个 `outputter`：

```py
# salt-call --local test.ping --out pickle
(dp0
S'local'
p1
I01
s.

```

如果你查看一些 Salt 附带的其它输出器，你会发现它们同样简单。甚至 `json` 输出器也没有做任何额外的工作，除了格式化输出。大多数执行模块默认会使用 `nested` 输出器。`nested` 使用基于 YAML 的格式，但带有彩色数据。然而，`state` 函数使用的是 `highstate` 输出器，它基于 `nested` 返回数据的聚合版本，包括关于状态运行成功率的统计信息。

# 输出器的故障排除

输出器可能是最容易调试的模块之一。你应该能够使用 `salt-call` 命令测试任何输出器。

在测试时，先从简单的 `test.ping` 开始，以确保首先得到一些输出。一旦你满意你的 `output()` 函数返回的是看起来正确的基本数据，查看 `grains.items`，它将使用列表和字典。

你可能会发现测试你的输出与另一个已知工作良好的输出器很有用。我发现 `pprint` 输出器在以易于阅读的格式显示数据时通常是最简洁的，但占用的屏幕空间最少：

```py
# salt-call --local grains.items --out pickle
# salt-call --local grains.items --out pprint

```

# 摘要

返回数据命令始终发送到主节点，即使在`salt`命令完成监听之后。事件总线拾取这些消息并将它们存储在外部作业缓存中。如果`salt`命令仍在监听，那么它将通过`outputter`显示。但指定返回者总会将返回数据发送到某个地方进行处理，只要主节点本身仍在运行。

可以使用`--return`标志指定返回者，或者可以通过`ext_job_cache` `master`配置选项在 Minion 上默认运行，或者在主节点上使用`master_job_cache` `master`配置选项来设置。

现在我们有了处理返回数据的方法，是时候创建更智能的过程来执行我们的命令了。接下来是运行者。
