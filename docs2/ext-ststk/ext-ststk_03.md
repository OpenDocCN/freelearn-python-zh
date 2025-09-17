# 第三章：扩展 Salt 配置

到现在为止，您已经知道如何从 Salt 的各个部分访问配置变量，除了 SDB 模块，这将在本章中介绍。但在设置静态配置的同时，能够从外部源提供这些数据非常有用。在本章中，您将学习以下内容：

+   编写动态 grains 和外部的 pillars

+   故障排除 grains 和 pillars

+   编写和使用 SDB 模块

+   故障排除 SDB 模块

# 动态设置 grains

正如您已经知道的，grains 包含描述 Minion 某些方面的变量。这可能包括有关操作系统、硬件、网络等信息。它还可以包含静态定义的用户数据，这些数据配置在`/etc/salt/minion`或`/etc/salt/grains`中。还可以使用 grains 模块动态定义 grains。

## 设置一些基本的 grains

Grains 模块很有趣，只要模块被加载，所有公共函数都会被执行。随着每个函数的执行，它将返回一个字典，其中包含要合并到 Minion 的 grains 中的项。

让我们继续设置一个新的 grains 模块来演示。我们将返回数据的名称前面加上一个`z`，以便于查找。

```py
'''
Test module for Extending SaltStack

This module should be saved as salt/grains/testdata.py
'''

def testdata():
    '''
    Return some test data
    '''
    return {'ztest1': True}
```

继续将此文件保存为`salt/grains/testdata.py`，然后使用`salt-call`显示所有 grains，包括这个：

```py
# salt-call --local grains.items
local:
 ----------
...
 virtual:
 physical
 zmqversion:
 4.1.3
 ztest1:
 True

```

请记住，您也可以使用 `grains.item` 来仅显示单个 grain：

```py
# salt-call --local grains.item ztest
local:
 ----------
 ztest1:
 True

```

这个模块可能看起来并不怎么有用，因为这只是静态数据，这些数据可以在`minion`或`grains`文件中定义。但请记住，与其他模块一样，grains 模块可以使用`__virtual__()`函数进行控制。让我们继续设置它，以及一个决定此模块是否首先加载的某种类型的标志：

```py
import os.path

def __virtual__():
    '''
    Only load these grains if /tmp/ztest exists
    '''
    if os.path.exists('/tmp/ztest'):
        return True
    return False
```

继续运行以下命令以查看此功能的作用：

```py
# salt-call --local grains.item ztest
local:
 ----------
 ztest:
# touch /tmp/ztest
# salt-call --local grains.item ztest
local:
 ----------
 ztest:
 True

```

这对于控制整个模块的返回数据非常有用，无论是动态的还是，如这个模块目前所是，静态的。

您可能想知道为什么那个例子检查了文件的存在，而不是检查现有的 Minion 配置。这是为了说明检测某些系统属性可能会决定如何设置 grains。如果您只想在`minion`文件中设置一个标志，您可以从`__opts__`中提取它。让我们继续将其添加到`__virtual__()`函数中：

```py
def __virtual__():
    '''
    Only load these grains if /tmp/ztest exists
    '''
    if os.path.exists('/tmp/ztest'):
        return True
    if __opts__.get('ztest', False):
        return True
    return False
```

继续删除旧的标志，并设置新的标志：

```py
# rm /tmp/ztest
# echo 'ztest: True' >> /etc/salt/minion
# salt-call --local grains.item ztest
local:
 ----------
 ztest:
 True

```

让我们继续设置这个模块，使其也能返回动态数据。由于 YAML 在 Salt 中非常普遍，让我们设置一个函数，返回 YAML 文件的内容：

```py
import yaml
import salt.utils

def yaml_test():
    '''
    Return sample data from /etc/salt/test.yaml
    '''
    with salt.utils.fopen('/etc/salt/yamltest.yaml', 'r') as fh_:
        return yaml.safe_load(fh_)
```

您可能会注意到，我们使用了`salt.utils.fopen()`而不是标准的 Python `open()`。Salt 的`fopen()`函数用一些额外的处理包装了 Python 的`open()`，以确保在 Minions 上正确关闭文件。

保存你的模块，然后输入以下命令以查看结果：

```py
# echo 'yamltest: True' > /etc/salt/yamltest.yaml
# salt-call --local grains.item yamltest
local:
 ----------
 yamltest:
 True

```

## （不）跨调用执行模块

你可能会尝试从谷物模块内部跨调用执行模块。不幸的是，这是不可能的。许多执行模块中的`__virtual__()`函数严重依赖于谷物。如果允许谷物在 Salt 决定是否首先启用执行模块之前就跨调用执行模块，将会导致循环依赖。

只需记住，谷物首先加载，然后是柱子，然后是执行模块。如果你打算使用两种或更多这类模块的代码，考虑在`salt/utils/`目录下为它设置一个库。

## 最终的谷物模块

我们已经编写了所有这些代码，生成的模块应该看起来如下：

```py
'''
Test module for Extending SaltStack.

This module should be saved as salt/grains/testdata.py
'''
import os.path
import yaml
import salt.utils

def __virtual__():
    '''
    Only load these grains if /tmp/ztest exists
    '''
    if os.path.exists('/tmp/ztest'):
        return True
    if __opts__.get('ztest', False):
        return True
    return False

def testdata():
    '''
    Return some test data
    '''
    return {'ztest1': True}

def yaml_test():
    '''
    Return sample data from /etc/salt/test.yaml
    '''
    with salt.utils.fopen('/etc/salt/yamltest.yaml', 'r') as fh_:
        return yaml.safe_load(fh_)
```

# 创建外部柱子

正如你所知，柱子就像谷物，有一个关键的区别：谷物是在 Minion 上定义的，而柱子是为单个 Minion 定义的，从 Master 那里定义。

对于用户来说，这里没有太多区别，除了柱子必须映射到 Master 上的目标，使用`pillar_roots`中的`top.sls`文件。这样的映射可能看起来像这样：

```py
# cat /srv/pillar/top.sls
base:
 '*':
 - test

```

在这个例子中，我们会定义一个名为 test 的柱子，它可能看起来像这样：

```py
# cat /srv/pillar/test.sls
test_pillar: True

```

动态柱子在`top.sls`文件中仍然被映射，但在配置方面，相似之处到此为止。

## 配置外部柱子

与动态谷物不同，只要它们的`__virtual__()`函数允许它们这样做，就会一直运行，柱子必须在`master`配置文件中显式启用。或者，如果我们像我们这样在本地模式下运行，在`minion`配置文件中。让我们继续在`/etc/salt/minion`的末尾添加以下行：

```py
ext_pillar:
  - test_pillar: True
```

如果我们在 Master 上测试这个，我们需要重新启动`salt-master`服务。然而，由于我们在 Minion 上以本地模式测试，这不会是必需的。

## 添加外部柱子

我们还需要创建一个简单的外部柱子来开始。创建`salt/pillar/test_pillar.py`并包含以下内容：

```py
'''
This is a test external pillar
'''

def ext_pillar(minion_id, pillar, config):
    '''
    Return the pillar data
    '''
    return {'test_pillar': minion_id}
```

保存你的工作，然后测试以确保它工作正常：

```py
# salt-call --local pillar.item test_pillar
local:
 ----------
 test_pillar:
 dufresne

```

让我们回顾一下这里发生了什么。首先，我们有一个名为`ext_pillar()`的函数。这个函数在所有外部柱子中都是必需的。它也是唯一必需的函数。任何其他函数，无论是否以前置下划线命名，都将仅限于这个模块。

这个函数将始终传递三份数据。第一份是请求此柱子的 Minion 的 ID。你可以在我们的例子中看到这一点：之前示例中运行的`minion_id`是`dufresne`。第二份是为这个 Minion 定义的静态柱子的副本。第三份是在`master`（或在这种情况下，`minion`）配置文件中传递给这个外部柱子的额外数据。

让我们继续更新我们的支柱，以显示每个组件的外观。将你的`ext_pillar()`函数更改为如下所示：

```py
def ext_pillar(minion_id, pillar, command):
    '''
    Return the pillar data
    '''
    return {'test_pillar': {
        'minion_id': minion_id,
        'pillar': pillar,
        'config': config,
    }}
```

保存它，然后修改你的`minion`（或`master`）文件中的`ext_pillar`配置：

```py
ext_pillar:
  - test_pillar: Alas, poor Yorik. I knew him, Horatio.
```

再次查看你的支柱数据：

```py
# salt-call --local pillar.item test_pillar
local:
 ----------
 test_pillar:
 ----------
 config:
 Alas, poor Yorik. I knew him, Horatio.
 minion_id:
 dufresne
 pillar:
 ----------
 test_pillar:
 True

```

你可以看到我们之前提到的`test_pillar`。当然，你也可以看到`minion_id`，就像之前一样。这里的重要部分是`config`。

这个例子被选出来是为了清楚地说明`config`参数是从哪里来的。当一个外部支柱被添加到`ext_pillar`列表中时，它作为一个字典被输入，其值只有一个条目。指定的条目可以是字符串、布尔值、整数或浮点数。它不能是字典或列表。

这个参数通常用于从配置文件传递参数到支柱。例如，Salt 附带的`cmd_yaml`支柱使用它来定义一个预期以 YAML 格式返回数据的命令：

```py
ext_pillar:
- cmd_yaml: cat /etc/salt/testyaml.yaml
```

如果你的支柱只需要被启用，那么你只需将其设置为 True，然后忽略它即可。然而，你仍然必须设置它！Salt 会期望那里有数据，如果没有，你会收到这样的错误：

```py
[CRITICAL] The "ext_pillar" option is malformed
```

### 小贴士

虽然`minion_id`、`pillar`和`config`都按顺序传递到`ext_pillar()`函数中，但 Salt 实际上并不关心你在函数定义中如何命名变量。如果你想叫它们 Emeril、Mario 和 Alton 也行（虽然你不会这么做）。但无论你叫它们什么，它们都必须都在那里。

## 另一个外部支柱

让我们再组合另一个外部支柱，这样它就不会和我们的第一个支柱混淆。这个支柱的工作是检查网络服务的状态。首先，让我们编写我们的支柱代码：

```py
'''
Get status from HTTP service in JSON format.

This file should be saved as salt/pillar/http_status.py
'''
import salt.utils.http

def ext_pillar(minion_id, pillar, config):
    '''
    Call a web service which returns status in JSON format
    '''
    comps = config.split()
    key = comps[0]
    url = comps[1]
    status = salt.utils.http.query(url, decode=True)
    return {key: status['dict']}
```

你可能已经注意到，我们的`docstring`声明说这个文件应该保存为`salt/pillar/http_status.py`。当你检查 Salt 代码库时，有一个名为`salt/`的目录包含实际的代码。这就是`docstring`中提到的目录。你将在本书中的代码示例中继续看到这些注释。

将此文件保存为`salt/pillar/http_status.py`。然后继续更新你的`ext_pillar`配置，使其指向它。现在，我们将使用 GitHub 的状态 URL：

```py
ext_pillar
  - http_status: github https://status.github.com/api/status.json
```

继续保存配置，然后测试支柱：

```py
# salt-call --local pillar.item github
local:
 ----------
 github:
 ----------
 last_updated:
 2015-12-02T05:22:16Z
 status:
 good

```

如果你需要能够检查多个服务的状态，你可以多次使用同一个外部支柱，但使用不同的配置。尝试更新你的`ext_pillar`定义，包含两个条目：

```py
ext_pillar
  - http_status: github https://status.github.com/api/status.json
  - http_status: github2 https://status.github.com/api/status.json
```

现在，这可能会迅速成为一个问题。如果你不断地调用 GitHub 的状态 API，GitHub 不会高兴。所以，虽然获取实时状态更新很好，但你可能想做一些限制你的查询。让我们将状态保存到文件中，并从那里返回。我们将检查文件的最后修改时间戳，以确保它不会在一分钟内更新多次。

让我们继续更新整个外部 pillar：

```py
'''
Get status from HTTP service in JSON format.

This file should be saved as salt/pillar/http_status.py
'''
import json
import time
import datetime
import os.path
import salt.utils.http

def ext_pillar(minion_id,  # pylint: disable=W0613
               pillar,  # pylint: disable=W0613
               config):
    '''
    Return the pillar data
    '''
    comps = config.split()

    key = comps[0]
    url = comps[1]

    refresh = False
    status_file = '/tmp/status-{0}.json'.format(key)
    if not os.path.exists(status_file):
        refresh = True
    else:
        stamp = os.path.getmtime(status_file)
        now = int(time.mktime(datetime.datetime.now().timetuple()))
        if now - 60 >= stamp:
            refresh = True

    if refresh:
        salt.utils.http.query(url, decode=True, decode_out=status_file)

    with salt.utils.fopen(status_file, 'r') as fp_:
        return {key: json.load(fp_)}
```

现在我们设置了一个名为 `refresh` 的标志，并且只有当该标志为 `True` 时才会访问 URL。我们还定义了一个将缓存从该 URL 获取的内容的文件。该文件将包含 pillar 的名称，因此最终会有一个像 `/tmp/status-github.json` 这样的名称。以下两行将检索文件的最后修改时间和当前时间（以秒为单位）：

```py
        stamp = os.path.getmtime(status_file)
        now = int(time.mktime(datetime.datetime.now().timetuple()))
```

通过比较这两个，我们可以确定文件是否超过 60 秒。如果我们想使 pillar 更具可配置性，甚至可以将那个 `60` 移动到 `config` 参数，并从 `comps[2]` 中获取它。

# 故障排除 grains 和 pillars

在编写 grains 和 pillars 时，你可能会遇到一些困难。让我们看看你可能会遇到的最常见问题。

## 动态 grains 未显示

你可能会发现，当你从 Master 发出 `grains.items` 命令时，你的动态 grains 没有显示。这可能很难追踪，因为 grains 在 Minion 上评估，任何错误都不太可能通过线路返回给你。

当你发现动态 grains 没有按照预期显示时，通常最简单的方法是直接登录到 Minion 进行故障排除。打开一个 shell 并尝试执行一个 `salt-call` 命令，看看是否有错误出现。如果它们没有立即出现，尝试在命令中添加 `--log-level=debug` 来查看是否有错误隐藏在那个级别。使用 `trace` 日志级别可能也是必要的。

## 外部 pillars 未显示

这些可能有点难以找出。使用 `salt-call` 在 grains 中查找错误是有效的，因为所有代码都可以在不启动或联系服务的情况下执行。但是，pillars 来自 Master，除非你在 `local` 模式下运行 `salt-call`。

如果你能够在一个 Minion 上安装你的外部 pillar 代码进行测试，那么步骤与检查 grains 错误相同。但是，如果你发现自己处于 Master 的环境无法在 Minion 上复制的情形，你需要使用不同的策略。

在 Master 上停止 `salt-master` 服务，然后以调试日志级别重新启动它：

```py
# salt-master --log-level debug

```

然后打开另一个 shell 并检查受影响的 Minion 的 pillars：

```py
# salt <minionid> pillar.items

```

pillar 代码中的任何错误都应该在 `salt-master` 以前台运行时在窗口中显示出来。

# 编写 SDB 模块

SDB 是一种相对较新的模块类型，非常适合开发。它代表 Simple Database，它旨在允许数据以非常简短的 URI 查询。底层配置可以像必要的那么复杂，只要用于查询它的 URI 尽可能简单。

SDB 的另一个设计目标是 URI 可以隐藏敏感信息，使其不会直接存储在配置文件中。例如，密码通常用于其他类型的模块，如`mysql`模块。但是，将密码存储在随后存储在版本控制系统（如 Git）中的文件中是一种不良做法。

使用 SDB 即时查找密码允许存储密码的引用，但不是密码本身。这使得在版本控制系统中存储引用敏感数据的文件变得更加安全。

有一种假设的功能可能会让人想要使用 SDB：在 Minion 上存储加密数据，这些数据不能被 Master 读取。可以在 Minion 上运行需要本地认证的代理，例如从 Minion 的键盘输入密码，或者使用硬件加密设备。可以创建利用这些代理的 SDB 模块，由于它们的本质，认证凭证本身不能被 Master 获取。

问题在于 Master 可以访问任何订阅它的 Minion 可以访问的内容。尽管数据可能存储在 Minion 上的加密数据库中，并且尽管其传输到 Master 时肯定被加密，但一旦到达 Master，它仍然可以以明文形式读取。

## 获取 SDB 数据

SDB 只使用了两个公共函数：`get`和`set`。实际上，其中最重要的一个是`get`，因为`set`通常可以在 Salt 之外完成。让我们先看看`get`函数。

对于我们的示例，我们将创建一个模块，它读取 JSON 文件，然后从中返回请求的键。首先，让我们设置我们的 JSON 文件：

```py
{
    "user": "larry",
    "password": "123pass"
}
```

将该文件保存为`/root/mydata.json`。然后编辑`minion`配置文件并添加一个配置配置文件：

```py
myjson:
    driver: json
    json_file: /root/mydata.json
```

准备好这两样东西后，我们就可以开始编写我们的模块了。JSON 有一个非常简单的接口，所以这里不会有太多内容：

```py
'''
SDB module for JSON

This file should be saved as salt/sdb/json.py
'''
from __future__ import absolute_import
import salt.utils
import json

def get(key, profile=None):
    '''
    Get a value from a JSON file
    '''
    with salt.utils.fopen(profile['json_file'], 'r') as fp_:
        json_data = json.load(fp_)
    return json_data.get(key, None)
```

你可能已经注意到，我们在必要的 JSON 代码之外添加了一些额外的东西。首先，我们导入了一个名为`absolute_import`的东西。这是因为这个文件叫做`json.py`，它正在导入另一个名为`json`的库。如果没有`absolute_import`，该文件将尝试导入自己，并且无法从实际的`json`库中找到必要的函数。

`get()`函数接受两个参数：`key`和`profile`。`key`指的是将用于访问所需数据的键。`profile`是我们保存到`minion`配置文件中的`myjson`配置文件数据的副本。

SDB URI 使用了这两个项目。当我们构建该 URI 时，它将被格式化为：

```py
sdb://<profile_name>/<key>
```

例如，如果我们使用`sdb`执行模块来检索`key1`的值，我们的命令将如下所示：

```py
# salt-call --local sdb.get sdb://myjson/user
local:
 larry

```

在此模块和配置文件就绪的情况下，我们现在可以向 minion 配置文件（或 grains 或 pillars，甚至 `master` 配置文件）中添加类似以下内容的行：

```py
username: sdb://myjson/user
password: sdb://myjson/password

```

当一个使用 `config.get` 的模块遇到 SDB URI 时，它将自动即时将其转换为适当的数据。

在我们继续之前，让我们稍微更新一下这个函数，以便进行一些错误处理。如果用户在配置文件中输入了错误（例如 `json_fle` 而不是 `json_file`），或者引用的文件不存在，或者 JSON 格式不正确，那么这个模块将开始输出跟踪回执消息。让我们继续处理所有这些问题，使用 Salt 的 `CommandExecutionError`：

```py
from __future__ import absolute_import
from salt.exceptions import CommandExecutionError
import salt.utils
import json

def get(key, profile=None):
    '''
    Get a value from a JSON file
    '''
    try:
        with salt.utils.fopen(profile['json_file'], 'r') as fp_:
            json_data = json.load(fp_)
        return json_data.get(key, None)
    except IOError as exc:
        raise CommandExecutionError (exc)
    except KeyError as exc:
        raise CommandExecutionError ('{0} needs to be configured'.format(exc))
    except ValueError as exc:
        raise CommandExecutionError (
            'There was an error with the JSON data: {0}'.format(exc)
        )
```

`IOError` 会捕获指向非实际文件的路径的问题。`KeyError` 会捕获缺少配置文件（如果其中一个项拼写错误）的错误。`ValueError` 会捕获格式不正确的 JSON 文件的问题。这将使错误变成：

```py
Traceback (most recent call last):
  File "/usr/bin/salt-call", line 11, in <module>
    salt_call()
  File "/usr/lib/python2.7/site-packages/salt/scripts.py", line 333, in salt_call
    client.run()
  File "/usr/lib/python2.7/site-packages/salt/cli/call.py", line 58, in run
    caller.run()
  File "/usr/lib/python2.7/site-packages/salt/cli/caller.py", line 133, in run
    ret = self.call()
  File "/usr/lib/python2.7/site-packages/salt/cli/caller.py", line 196, in call
    ret['return'] = func(*args, **kwargs)
  File "/usr/lib/python2.7/site-packages/salt/modules/sdb.py", line 28, in get
    return salt.utils.sdb.sdb_get(uri, __opts__)
  File "/usr/lib/python2.7/site-packages/salt/utils/sdb.py", line 37, in sdb_get
    return loaded_dbfun
  File "/usr/lib/python2.7/site-packages/salt/sdb/json_sdb.py", line 49, in get
    with salt.utils.fopen(profile['json_fil']) as fp_:
KeyError: 'json_fil'
```

...变成这样的错误：

```py
Error running 'sdb.get': 'json_fil' needs to be configured
```

## 设置 SDB 数据

用于 `set` 的函数可能看起来很奇怪，因为 `set` 是 Python 的内置函数。这意味着该函数可能不被称为 `set()`；它必须被命名为其他名称，然后使用 `__func_alias__` 字典给出别名。让我们继续创建一个除了返回要设置的 `value` 之外什么都不做的函数：

```py
__func_alias__ = {
    'set_': 'set'
}

def set_(key, value, profile=None):
    '''
    Set a key/value pair in a JSON file
    '''
    return value
```

对于只读数据，这对你来说已经足够了，但在这个案例中，我们将修改 JSON 文件。首先，让我们看看传递给我们的函数的参数。

你已经知道，数据的关键点是要引用的，并且配置文件包含 Minion 配置文件中的配置数据副本。你可能也能猜到，值包含要应用的数据副本。

值不会改变实际的 URI；无论你是获取还是设置数据，它始终相同。执行模块本身接受要设置的数据，然后设置它。你可以通过以下方式看到这一点：

```py
# salt-call --local sdb.set sdb://myjson/password 321pass
local:
 321pass

```

考虑到这一点，让我们继续让我们的模块读取 JSON 文件，应用新值，然后再将其写回。目前，我们将跳过错误处理，以便更容易阅读：

```py
def set_(key, value, profile=None):
    '''
    Set a key/value pair in a JSON file
    '''
    with salt.utils.fopen(profile['json_file'], 'r') as fp_:
        json_data = json.load(fp_)

    json_data[key] = value

    with salt.utils.fopen(profile['json_file'], 'w') as fp_:
        json.dump(json_data, fp_)

    return get(key, profile)
```

这个函数与之前一样读取 JSON 文件，然后更新特定的值（如果需要则创建它），然后写回文件。完成时，它使用 `get()` 函数返回数据，这样用户就知道是否设置正确。如果返回错误的数据，那么用户就会知道出了问题。它不一定能告诉他们出了什么问题，但会拉响一个红旗。

让我们继续添加一些错误处理来帮助用户了解出了什么问题。我们将继续添加来自 `get()` 函数的错误处理：

```py
def set_(key, value, profile=None):
    '''
    Set a key/value pair in a JSON file
    '''
    try:
        with salt.utils.fopen(profile['json_file'], 'r') as fp_:
            json_data = json.load(fp_)
    except IOError as exc:
        raise CommandExecutionError (exc)
    except KeyError as exc:
        raise CommandExecutionError ('{0} needs to be configured'.format(exc))
    except ValueError as exc:
        raise CommandExecutionError (
            'There was an error with the JSON data: {0}'.format(exc)
        )

    json_data[key] = value

    try:
        with salt.utils.fopen(profile['json_file'], 'w') as fp_:
            json.dump(json_data, fp_)
    except IOError as exc:
        raise CommandExecutionError (exc)

    return get(key, profile)
```

由于我们在读取文件时进行了所有错误处理，当我们再次写入时，我们已经知道路径是有效的，JSON 是有效的，并且没有配置错误。然而，保存文件时仍然可能出错。尝试以下操作：

```py
# chattr +i /root/mydata.json
# salt-call --local sdb.set sdb://myjson/password 456pass
Error running 'sdb.set': [Errno 13] Permission denied: '/root/mydata.json'

```

我们已将文件的属性更改为不可变（只读），因此我们不能再写入文件。如果没有`IOError`，我们就会得到一个像以前一样的难看的跟踪回信息。移除不可变属性将允许我们的函数正常运行：

```py
# chattr -i /root/mydata.json
# salt-call --local sdb.set sdb://myjson/password 456pass
local:
 456pass

```

## 使用描述性的`docstring`

使用 SDB 模块时，添加一个演示如何配置和使用模块的`docstring`比以往任何时候都更重要。没有它，用户几乎不可能弄清楚如何使用模块，尝试修改模块的情况甚至更糟。

`docstring`不需要是新颖的。它应该包含足够的信息来使用模块，但不要太多，以免弄清楚事情变得令人困惑和沮丧。你应该包括配置数据的示例，以及与该模块一起使用的 SDB URI：

```py
'''
SDB module for JSON

Like all sdb modules, the JSON module requires a configuration profile to
be configured in either the minion or master configuration file. This profile
requires very little. In the example:

.. code-block:: yaml

    myjson:
      driver: json
      json_file: /root/mydata.json

The ``driver`` refers to the json module and json_file is the path to the JSON
file that contains the data.

.. code-block:: yaml

    password: sdb://myjson/somekey
'''
```

## 使用更复杂的配置

可能会诱使人们创建使用更复杂 URI 的 SDB 模块。例如，完全有可能创建一个支持如下 URI 的模块：

```py
sdb://mydb/user=curly&group=ops&day=monday
```

使用前面的 URI，传入的`key`将是：

```py
user=curly&group=ops&day=monday
```

到那时，将取决于你解析出密钥并将其转换为代码可用的东西。然而，我强烈反对这样做！

你使 SDB URI 越复杂，它就越不像简单的数据库查找。你还可能以意想不到的方式暴露数据。再次看看前面的`key`。它揭示了以下关于存储敏感信息的数据库的信息：

+   存在一个被称为用户的字段（抽象的或真实的）。由于用户往往比他们想象的要懒惰，这很可能是指向一个名为 user 的真实数据库字段。如果是这样，那么这暴露了数据库模式的一部分。

+   有一个名为 ops 的组。这意味着还有其他组。由于*ops*通常指的是执行服务器操作任务的团队，那么这意味着还有一个名为*dev*的组吗？如果 dev 组被攻破，攻击者能窃取到哪些有价值的资料？

+   指定了一天。这家公司是否每天轮换密码？指定*monday*的事实意味着最多只有七个密码：一周中每一天一个。

与其将所有这些信息放入 URL 中，通常更好的做法是将它们隐藏在配置文件中。可以假设`mydb`指的是一个数据库连接（如果我们把配置文件命名为`mysql`，我们就会暴露数据库连接的类型）。跳过任何可能存在的数据库凭证，我们可以使用如下配置文件：

```py
mydb:
  driver: <some SDB module>
  fields:
    user: sdbkey
    group: ops
    day: monday
```

假设相关的模块能够将这些`field`s 转换成查询，并在内部将`sdbkey`更改为实际传入的任何`key`，我们可以使用如下看起来像的 URI：

```py
sdb://mydb/curly
```

你仍然可以猜测`curly`指的是用户名，当 URI 与配置参数如一起使用时，这可能是更加明显的：

```py
username: sdb://mydb/curly
```

然而，它并没有暴露数据库中的字段名称。

## 最终的 SDB 模块

在我们编写的所有代码中，生成的模块应该看起来像以下这样：

```py
'''
SDB module for JSON

Like all sdb modules, the JSON module requires a configuration profile to
be configured in either the minion or master configuration file. This profile
requires very little. In the example:

.. code-block:: yaml

    myjson:
      driver: json
      json_file: /root/mydata.json

The ``driver`` refers to the json module and json_file is the path to the JSON
file that contains the data.

.. code-block:: yaml

    password: sdb://myjson/somekey
'''
from __future__ import absolute_import
from salt.exceptions import CommandExecutionError
import salt.utils
import json

__func_alias__ = {
    'set_': 'set'
}

def get(key, profile=None):
    '''
    Get a value from a JSON file
    '''
    try:
        with salt.utils.fopen(profile['json_file'], 'r') as fp_:
            json_data = json.load(fp_)
        return json_data.get(key, None)
    except IOError as exc:
        raise CommandExecutionError (exc)
    except KeyError as exc:
        raise CommandExecutionError ('{0} needs to be configured'.format(exc))
    except ValueError as exc:
        raise CommandExecutionError (
            'There was an error with the JSON data: {0}'.format(exc)
        )

def set_(key, value, profile=None):  # pylint: disable=W0613
    '''
    Set a key/value pair in a JSON file
    '''
    try:
        with salt.utils.fopen(profile['json_file'], 'r') as fp_:
            json_data = json.load(fp_)
    except IOError as exc:
        raise CommandExecutionError (exc)
    except KeyError as exc:
        raise CommandExecutionError ('{0} needs to be configured'.format(exc))
    except ValueError as exc:
        raise CommandExecutionError (
            'There was an error with the JSON data: {0}'.format(exc)
        )

    json_data[key] = value

    try:
        with salt.utils.fopen(profile['json_file'], 'w') as fp_:
            json.dump(json_data, fp_)
    except IOError as exc:
        raise CommandExecutionError (exc)

    return get(key, profile)
```

# 使用 SDB 模块

有许多地方可以使用 SDB 模块。因为 SDB 检索被集成在`config`执行模块中的`config.get`函数中，以下位置可以用来为 Minion 设置一个值：

+   Minion 配置文件

+   粒子

+   柱子

+   主配置文件

SDB 也由 Salt Cloud 支持，因此你还可以在以下位置设置 SDB URI：

+   主要云配置文件

+   云配置文件

+   云提供商

+   云映射

无论你在哪里设置 SDB URI，格式都是相同的：

```py
<setting name>: sdb://<profile name>/<key>
```

这对于云提供商特别有用，所有这些都需要凭证，但其中许多也使用更复杂的配置块，这些配置块应该被纳入版本控制。

以`openstack`云提供商为例：

```py
my-openstack-config:
  identity_url: https://keystone.example.com:35357/v2.0/
  compute_region: intermountain
  compute_name: Compute
  tenant: sdb://openstack_creds/tenant
  user: sdb://openstack_creds/username
  ssh_key_name: sdb://openstack_creds/keyname
```

在这个组织中，`compute_region`和`compute_name`可能是公开的。而`identity_url`肯定也是（否则，你怎么进行认证呢？）。但其他信息可能应该保持隐藏。

如果你曾经在使用 Salt Cloud 设置 OpenStack，你可能已经使用了许多其他参数，其中许多可能不是敏感的。然而，一个复杂的配置文件可能应该保存在版本控制系统。使用 SDB URI，你可以这样做而不必担心暴露敏感数据。

# SDB 模块故障排除

我们已经介绍了一些可以添加到我们的 SDB 模块中的错误处理，但你可能仍然会遇到问题。像粒子和柱子一样，最常见的问题是在预期中数据没有显示出来。

## SDB 数据未显示

你可能会发现，当你将 SDB URI 包含在配置中时，它并没有像你想象的那样解析。如果你在早期的 SDB 代码中犯了拼写错误，你可能已经发现`sdb.get`在存在语法错误时会非常乐意抛出跟踪回溯。但如果使用`salt-call`在`sdb.get`上没有引发你可以看到的任何错误，那么可能不是你的代码中的问题。

在开始责怪其他服务之前，最好确保你不是问题所在。开始记录关键信息，以确保它以你期望的方式显示。确保在模块顶部添加以下行：

```py
import logging
log = logging.getLogger(__name__)
```

然后，你可以使用`log.debug()`来记录这些信息。如果你正在记录敏感信息，你可能想使用`log.trace()`代替，以防你忘记取出日志消息。

你可能想从记录每个函数接收到的信息开始，以确保它看起来符合你的预期。让我们先看看之前提到的 `get()` 示例，并添加一些日志记录：

```py
def get(key, profile=None):
    '''
    Get a value from a JSON file
    '''
    import pprint
    log.debug(key)
    log.debug(pprint.pformat(profile))
    with salt.utils.fopen(profile['json_file'], 'r') as fp_:
        json_data = json.load(fp_)
    return json_data.get(key, None)
```

我们在这里只添加了几行日志，但使用了 Python 的 `pprint` 库来格式化其中之一。`pprint.pformat()` 函数用于格式化打算存储在字符串中或传递给函数的文本，而不是像 `pprint.pprint()` 那样直接输出到 `STDOUT`。

如果你的 SDB 模块连接到某个服务，你可能会发现该服务本身不可用。这可能是由于未知的或意外的防火墙规则、网络错误，或者服务本身的实际停机。在代码中分散日志消息将帮助你发现它失败的地方，这样你就可以在那里解决问题。

# 摘要

可以通过加载系统连接到 Salt 配置的三个区域是动态 grains、外部 pillars 和 SDB。Grains 在 Minion 上生成，pillars 在 Master 上生成，SDB URI 可以在这两个地方配置。

SDB 模块允许配置存储在外部，但可以从 Salt 配置的各个部分引用。当从执行模块访问时，它们在 Minion 上解析。当从 Salt-Cloud 访问时，它们在运行 Salt Cloud 的任何系统上解析。

现在我们已经处理好了配置，是时候深入配置管理了，通过在执行模块周围包装状态模块来实现。
