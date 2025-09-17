# 第二章。编写执行模块

执行模块构成了 Salt 执行的工作负载的骨干。它们也易于编写，编写它们的技巧是编写其他所有类型 Salt 模块的基础。对执行模块的工作原理有扎实的理解后，其他模块类型的功能也将被打开。

在本章中，我们将讨论：

+   编写 Salt 模块的基本知识

+   利用 Salt 内置功能

+   使用良好实践

+   执行模块的故障排除

# 编写 Salt 模块

在所有 Salt 模块中，有一些项目是一致的。这些组件在所有模块类型中通常以相同的方式工作，尽管在少数地方你可以期望至少有一点偏差。我们将在其他章节中介绍这些偏差。现在，让我们谈谈那些通常相同的事情。

## 隐藏对象

很久以前，程序员就习惯于在函数、变量等前面加上下划线，如果它们只打算在同一个模块内部使用。在许多语言中，这样使用的对象被称为**私有对象**。

一些环境通过不允许外部代码直接引用这些内容来强制执行私有行为。其他环境允许这样做，但使用是不被鼓励的。Salt 模块属于强制执行私有函数行为的列表；如果一个 Salt 模块中的函数以下划线开头，它甚至不会被暴露给尝试调用它的其他模块。

在 Python 中，存在一种特殊的对象，其名称以两个下划线开头和结尾。这些被称为“魔法方法”（magic methods）的东西被昵称为**dunder**（意为双下划线）。Python 通常如何处理它们超出了本书的范围，但重要的是要知道 Salt 添加了一些自己的。其中一些是内置的，通常在（几乎）所有模块类型中都可以使用，而另一些则是用户定义的对象，Salt 会对它们进行特殊处理。

## __virtual__()函数

这是一个可以出现在任何模块中的函数。如果没有`__virtual__()`函数，那么该模块将始终在每个系统上可用。如果该模块存在，那么它的任务是确定该模块的要求是否得到满足。这些要求可能包括配置设置到软件包依赖的任何数量。

如果要求没有得到满足，那么`__virtual__()`函数将返回`False`。在 Salt 的较新版本中，可以返回一个包含`False`值和无法加载模块的原因的元组。如果它们得到满足，那么它可以返回两种类型的值。这里事情变得有点棘手。

假设我们正在开发的模块位于 `salt/modules/mymodule.py`。如果满足要求，并且该模块将被引用为 `mymodule`，那么 `__virtual__()` 函数将返回 `True`。假设该模块中还有一个名为 `test()` 的函数，它将使用以下命令调用：

```py
#salt-call mymodule.test

```

如果满足要求，但此模块将被引用为 `testmodule`，那么 `__virtual__()` 函数将返回字符串 `testmodule`。然而，而不是直接返回该字符串，你应该在所有函数之前使用 `__virtualname__` 变量定义它。

让我们开始编写一个模块，使用 `__virtual__()` 函数和 `__virtualname__` 变量。我们目前不会检查任何要求：

```py
'''
This module should be saved as salt/modules/mysqltest.py
'''
__virtualname__ = 'mysqltest'

def __virtual__():
    '''
    For now, just return the __virtualname__
    '''
    return __virtualname__

def ping():
    '''
    Returns True

    CLI Example:
        salt '*' mysqltest.ping
    '''
    return True
```

## 代码格式化

在我们继续前进之前，我想指出一些你应该现在就注意的重要事项，这样你就不会养成需要以后改正的坏习惯。

模块以一种特殊的注释开始，称为 `docstring`。在 Salt 中，它以三行单引号开始和结束，所有引号都在一行上，单独成行。不要使用双引号。不要将文本放在引号所在的同一行上。所有公共函数也必须包含 `docstring`，遵循相同的规则。这些 `docstrings` 由 Salt 内部使用，为 `sys.doc` 等函数提供帮助文本。

### 注意

请记住，这些指南是针对 Salt 的；Python 本身遵循不同的风格。有关更多信息，请参阅附录 B 中的 *Understanding the Salt style guide*。

注意，`ping()` 函数的 `docstring` 包含一个 `CLI Example`。你应该总是包含足够的信息，以便清楚地了解函数的用途，并且至少包含一个（或更多，根据需要）命令行示例，以演示如何使用该函数。私有函数不包括 `CLI Example`。

你应该在顶部和下面的函数之间，以及所有函数之间，始终包含两个空行，任何导入和变量声明之间也应如此。文件末尾不应有空格。

## 虚拟模块

`__virtual__()` 函数的主要动机不仅仅是重命名模块。使用此函数允许 Salt 不仅检测有关系统的某些信息，而且还可以使用这些信息适当地加载特定模块，使某些任务更加通用。

第一章，*从基础开始*，提到了这些例子中的一些。`salt/modules/aptpkg.py` 包含了多个测试，以确定它是否运行在类似 Debian 的操作系统上，该操作系统使用 `apt` 工具集进行软件包管理。`salt/modules/yumpkg.py`、`salt/modules/pacman.py`、`salt/modules/solarispkg.py` 以及其他一些模块中也有类似的测试。如果这些模块中的任何一个都通过了所有测试，那么它将被加载为 `pkg` 模块。

如果您正在构建这样的模块集，重要的是要记住，它们应该尽可能相似。例如，所有的 `pkg` 模块都包含一个名为 `install()` 的函数。每个 `install()` 函数都接受相同的参数，执行相同的任务（适用于该平台），然后以完全相同的格式返回数据。

可能存在这样的情况，某个函数适用于一个平台，但不适用于另一个平台。例如，`salt/modules/aptpkg.py` 包含一个名为 `autoremove()` 的函数，它调用 `apt-get autoremove`。在 `yum` 中没有这样的功能，因此 `salt/modules/yumpkg.py` 中不存在该函数。如果存在，那么这个函数在两个文件中应该以相同的方式表现。

## 使用 salt.utils 库

前一个模块总是会运行，因为它不会检查系统上的需求。现在让我们继续添加一些检查。

在 `salt/utils/` 目录内有一套丰富的工具可用于导入。其中许多直接位于 `salt.utils` 命名空间下，包括一个非常常用的函数 `salt.utils.which()`。当给出命令名时，此函数将报告该命令在系统上的位置。如果不存在，则返回 `False`。

让我们重新设计 `__virtual__()` 函数，以查找名为 `mysql` 的命令：

```py
'''
This module should be saved as salt/modules/mysqltest.py
'''
import salt.utils

__virtualname__ = 'mysqltest'

def __virtual__():
    '''
    Check for MySQL
    '''
    if not salt.utils.which('mysql'):
        return False
    return __virtualname__

def ping():
    '''
    Returns True

    CLI Example:
        salt '*' mysqltest.ping
    '''
    return True
```

`salt.utils` 库与 Salt 一起提供，但您需要显式导入它们。对于 Python 开发者来说，只导入函数的一部分是很常见的。您可能会发现使用以下导入行很有吸引力：

```py
from salt.utils import which
```

然后使用以下行：

```py
if which('myprogram'):
```

虽然在 Salt 中没有明确禁止，但除非必要，否则不建议这样做。虽然这可能会需要更多的输入，尤其是如果您在特定模块中多次使用特定函数，这样做可以更容易地一眼看出特定函数来自哪个模块。

## 使用 __salt__ 字典进行跨调用

有时候，能够调用另一个模块中的另一个函数是有帮助的。例如，调用外部 shell 命令是 Salt 的一个重要部分。实际上，它如此重要，以至于在`cmd`模块中进行了标准化。最常用的发布 shell 命令的命令是`cmd.run`。以下 Salt 命令演示了在 Windows Minion 上使用`cmd.run`：

```py
#salt winminon cmd.run 'dir C:\'

```

如果你需要你的执行模块从这样的命令中获取输出，你会使用以下 Python 代码：

```py
__salt__'cmd.run'
```

`__salt__`对象是一个字典，其中包含对该 Minion 上所有可用函数的引用。如果一个模块存在，但它的`__virtual__()`函数返回`False`，那么它将不会出现在这个列表中。作为一个函数引用，它需要在末尾使用括号，并在括号内放置任何参数。

让我们创建一个函数，告诉我们`sshd`守护进程是否在 Linux 系统上运行，并且监听某个端口：

```py
def check_mysqld():
    '''
    Check to see if sshd is running and listening

    CLI Example:
        salt '*' testmodule.check_mysqld
    '''
    output = __salt__'cmd.run'
    if 'tcp' not in output:
        return False
    return True
```

如果`sshd`正在运行并监听某个端口，`netstat -tulpn | grep sshd`命令的输出应该看起来像这样：

```py
tcp        0      0 0.0.0.0:3306              0.0.0.0:*               LISTEN      426/mysqld
tcp6       0      0 :::3306                   :::*                    LISTEN      426/mysqld
```

如果`mysqld`正在运行，并且监听 IPv4 或 IPv6（或两者），那么这个函数将返回`True`。

这个函数远非完美。有许多因素可能导致这个命令返回一个假阳性。例如，假设你正在寻找`sshd`而不是`mysqld`。再假设你是一位美式足球迷，并且自己编写了一个高清足球视频流服务，你称之为`passhd`。这可能不太可能，但绝对不是不可能的。这提出了一个重要观点：在处理从用户或计算机接收到的数据时，**要信任但也要验证**。实际上，你应该始终假设有人会试图做坏事，你应该寻找阻止他们这样做的方法。

## 获取配置参数

虽然一些软件可以在没有任何特殊配置的情况下访问，但还有很多软件需要设置一些信息。执行模块可以从四个地方获取其配置：Minion 配置文件、grain 数据、pillar 数据和主配置文件。

### 注意

这就是 Salt 内置函数行为不同的地方之一。Grain 和 pillar 数据对执行和状态模块可用，但对其他类型的模块不可用。这是因为 grain 和 pillar 数据是特定于运行模块的 Minion 的。例如，Runners 无法访问这些数据，因为 Runners 是在 Master 上使用的，而不是直接在 Minions 上。

我们可以查找配置的第一个地方是`__opts__`字典。当在 Minion 上执行模块工作时，这个字典将包含来自 Minion 配置文件的数据副本。它还可能包含一些 Salt 在运行时自己生成的一些信息。当从在 Master 上执行的模块访问时，这些数据将来自主配置文件。

还可以在 grain 或 pillar 数据中设置配置值。这些信息分别通过 `__grains__` 和 `__pillar__` 字典访问。以下示例显示了从这些位置拉取的不同配置值：

```py
username = __opts__['username']
hostname = __grains__['host']
password = __pillar__['password']
```

由于这些值可能实际上不存在，最好使用 Python 的 `dict.get()` 方法，并提供一个默认值：

```py
username = __opts__.get('username', 'salt')
hostname = __grains__.get('host', 'localhost')
password = __pillar__.get('password', None)
```

我们可以存储配置数据的最后一个地方是在主配置文件中。Master 的所有配置都可以存储在一个名为 `master` 的 pillar 字典中。默认情况下，这不会对 Minions 可用。但是，可以通过在 `master` 配置文件中将 `pillar_opts` 设置为 `True` 来启用它。

一旦 `pillar_opts` 被启用，你可以使用如下命令访问 `master` 配置中的值：

```py
master_interface = __pillar__['master']['interface']
master_sock_dir = __pillar__.get('master', {}).get('sock_dir', None)
```

最后，你可以让 Salt 依次在每个位置搜索特定的变量。当你不在乎哪个组件携带你需要的信息，只要你能从某个地方获取它时，这非常有价值。

为了搜索这些区域，需要跨调用 `config.get()` 函数：

```py
username = __salt__'config.get'
```

这将按以下顺序搜索配置参数：

1.  `__opts__`（在 Minion 上）。

1.  `__grains__`。

1.  `__pillar__`。

1.  `__opts__`（在 Master 上）。

请记住，当使用 `config.get()` 时，将使用找到的第一个值。如果你要查找的值在 `__grains__` 和 `__pillar__` 中都定义了，那么将使用 `__grains__` 中的值。

使用 `config.get()` 的另一个优点是，此函数将自动解析使用 `sdb://` URI 引用的数据。当直接访问这些字典时，任何 `sdb://` URI 都需要手动处理。编写和使用 SDB 模块将在 第三章，*扩展 Salt 配置* 中介绍。

让我们继续设置一个模块，该模块获取配置数据并使用它来连接到服务：

```py
'''
This module should be saved as salt/modules/mysqltest.py
'''
import MySQLdb

def version():
    '''
    Returns MySQL Version

    CLI Example:
        salt '*' mysqltest.version
    '''
    user = __salt__'config.get'
    passwd = __salt__'config.get'
    host = __salt__'config.get'
    port = __salt__'config.get'
    db_ = __salt__'config.get'
    dbc = MySQLdb.connect(
        connection_user=user,
        connection_pass=passwd,
        connection_host=host,
        connection_port=port,
        connection_db=db_,
    )
    cur = dbc.cursor()
    return cur.execute('SELECT VERSION()')
```

此执行模块将在 Minion 上运行，但它可以使用在四个配置区域中定义的任何配置连接到任何 MySQL 数据库。然而，这个功能相当有限。如果 `MySQLdb` 驱动未安装，那么在 Minion 启动时日志文件中会出现错误。如果你需要执行其他类型的查询，你将需要每次都获取配置值。让我们依次解决这些问题。

### 小贴士

你注意到我们使用了名为 `db_` 的变量而不是 `db` 吗？在 Python 中，被认为更好的做法是使用至少三个字符长的变量名。Salt 也认为这是必需的。对于通常较短的变量，在变量名末尾附加一个或两个下划线是一个非常常见的实现方式。

## 处理导入

许多 Salt 模块需要安装第三方 Python 库。如果其中任何一个库没有安装，那么`__virtual__()`函数应该返回`False`。但你是如何事先知道这些库是否可以导入的呢？

在 Salt 模块中，一个非常常见的技巧是尝试导入库，并记录导入是否成功。这通常是通过一个名为`HAS_LIBS`的变量来完成的：

```py
try:
    import MySQLdb
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

def __virtual__():
    '''
    Check dependencies
    '''
    return HAS_LIBS
```

在这种情况下，Python 将尝试导入`MySQLdb`。如果成功，则将`HAS_LIBS`设置为`True`。否则，将其设置为`False`。由于这直接关联到`__virtual__()`函数需要返回的值，我们只需将其原样返回即可，只要我们不更改`__virtualname__`。如果我们更改了，那么函数将看起来像这样：

```py
def __virtual__():
    '''
    Check dependencies
    '''
    if HAS_LIBS:
        return __virtualname__
    return False
```

## 代码重用

仍然存在消除同一模块中不同函数之间冗余代码的问题。在代码中使用连接对象（如数据库游标或云提供商身份验证）的模块中，通常会有特定的函数被留出以收集配置和建立连接。

这些云模块的一个非常常见的名称是`_get_conn()`，所以让我们在我们的例子中使用它：

```py
def _get_conn():
    '''
    Get a database connection object
    '''
    user = __salt__'config.get'
    passwd = __salt__'config.get'
    host = __salt__'config.get'
    port = __salt__'config.get'
    db_ = __salt__'config.get'
    return MySQLdb.connect(
        connection_user=user,
        connection_pass=passwd,
        connection_host=host,
        connection_port=port,
        connection_db=db_,
    )

def version():
    '''
    Returns MySQL Version

    CLI Example:
        salt '*' mysqltest.version
    '''
    dbc = _get_conn()
    cur = dbc.cursor()
    return cur.execute('SELECT VERSION()')
```

这大大简化了我们的代码，通过将每个函数中的大量行转换为单行。当然，这可以进一步扩展。实际上，与 Salt 一起提供的`salt/modules/mysql.py`模块使用了一个名为`_connect()`的函数而不是`_get_conn()`，并且它还将`cur.execute()`抽象到它自己的`_execute()`函数中。你可以在 Salt 的 GitHub 页面上看到这些：

[`github.com/saltstack/salt`](https://github.com/saltstack/salt)

## 日志消息

非常常见，你会执行需要记录某种消息的操作。当编写新代码时，这尤其常见；能够记录调试信息是非常好的。

Salt 内置了一个基于 Python 自己的`logging`库的日志系统。要启用它，你需要在模块顶部添加两行：

```py
import logging
log = logging.getLogger(__name__)
```

在这些设置到位后，你可以使用如下命令来记录日志：

```py
log.debug('This is a log message')
```

在 Salt 中通常使用五个级别的日志记录：

1.  `log.info()`: 此级别的信息被认为是所有用户都认为重要的事情。这并不意味着有任何错误发生，但像所有日志消息一样，其输出将被发送到`STDERR`而不是`STDOUT`（只要 Salt 在前台运行，并且没有配置到其他地方进行日志记录）。

1.  `log.warn()`: 从这里记录的消息应该向用户表明某些事情没有按预期进行。然而，它并没有到足以停止代码运行的程度。

1.  `log.error()`: 这表示出了问题，Salt 无法继续运行直到问题得到修复。

1.  `log.debug()`: 这不仅是有助于确定程序思考什么的信息，而且也是为了使程序的用户（如故障排除）有用。

1.  `log.trace()`: 这与调试消息类似，但这里的信息更有可能只对开发者有用。

现在，我们将在 `_get_conn()` 函数中添加一个 `log.trace()`，这样我们就可以知道何时成功连接到数据库：

```py
def _get_conn():
    '''
    Get a database connection object
    '''
    user = __salt__'config.get'
    passwd = __salt__'config.get'
    host = __salt__'config.get'
    port = __salt__'config.get'
    db_ = __salt__'config.get'
    dbc = MySQLdb.connect(
        connection_user=user,
        connection_pass=passwd,
        connection_host=host,
        connection_port=port,
        connection_db=db_,
    )
    log.trace('Connected to the database')
    return dbc
```

### 小贴士

有一些地方可能会诱使人们使用日志消息，但应该避免这样做。具体来说，日志消息可以在任何函数中使用，除了 `__virtual__()`。在函数外部以及 `__virtual__()` 函数中使用的日志消息会生成混乱的日志文件，难以阅读和导航。

## 使用 __func_alias__ 字典

在 Python 中有一些保留字。不幸的是，其中一些词对于像函数名这样的用途也非常有用。例如，许多模块都有一个函数，其任务是列出与该模块相关的数据，因此将这样的函数命名为 `list()` 似乎是自然的。但这会与 Python 的内置 `list` 冲突。这会引发问题，因为函数名直接暴露在 Salt 命令中。

对于这个问题有一个解决方案。可以在模块顶部声明一个 `__func_alias__` 字典，它将在命令行中使用的别名与函数的实际名称之间创建一个映射。例如：

```py
__func_alias__ = {
    'list_': 'list'
}

def list_(type_):
    '''
    List different resources in MySQL
    CLI Examples:
        salt '*' mysqltest.list tables
        salt '*' mysqltest.list databases
    '''
    dbc = _get_conn()
    cur = dbc.cursor()
    return cur.execute('SHOW {0}()'.format(type_))
```

这样一来，`list_` 函数将被调用为 `mysqltest.list`（如 CLI 示例所示），而不是 `mysqltest.list_`。

### 小贴士

为什么我们使用变量 `type_` 而不是 `type`？因为 `type` 是 Python 的内置函数。但因为这个函数只有一个参数，所以预计用户不需要在 Salt 命令中使用 `type_=<something>`。

## 验证数据

从最后一部分代码来看，此时可能有许多读者脑海中响起了警钟。它允许一种非常常见的安全漏洞，称为注入攻击。因为这个函数没有对 `type_` 变量进行任何形式的验证，用户可以传递可能导致破坏或获取他们不应拥有的数据的代码。

有些人可能会认为在 Salt 中这并不是一个必然的问题，因为在许多环境中，只有受信任的用户应该有权访问。然而，由于 Salt 可以被各种用户类型使用，他们可能只被授权有限访问，因此存在许多场景，注入攻击可能会造成灾难性的后果。想象一下，一个用户正在运行以下 Salt 命令：

```py
#salt myminion mysqltest.list 'tables; drop table users;'

```

这通常很容易修复，只需在用户输入中添加一些简单的检查（记住：**信任但验证**）：

```py
from salt.exceptions import CommandExecutionError

def list_(type_):
    '''
    List different resources in MySQL
    CLI Examples:
        salt '*' mysqltest.list tables
        salt '*' mysqltest.list databases
    '''
    dbc = _get_conn()
    cur = dbc.cursor()
    valid_types = ['tables', 'databases']
    if type_ not in valid_types:
        err_msg = 'A valid type was not specified'
        log.error(err_msg)
        raise CommandExecutionError(err_msg)
    return cur.execute('SHOW {0}()'.format(type_))
```

在这种情况下，我们在允许它们传递到 SQL 查询之前已经声明了哪些类型是有效的。即使是一个坏字符也会导致 Salt 拒绝完成命令。这种类型的数据验证通常更好，因为它不试图修改输入数据以使其安全运行。这样做被称为 *验证用户输入*。

我们还添加了另一段代码：一个 Salt 异常。`salt.exceptions` 库中有许多这样的异常，但 `CommandExecutionError` 是你在验证数据时可能会经常使用的一个。

## 字符串格式化

关于字符串格式化的一点说明：较老的 Python 开发者可能已经注意到，我们选择使用 `str.format()` 而不是旧的 `printf` 风格的字符串处理。以下两行代码在 Python 中做的是同样的事情：

```py
'The variable's value is {0}'.format(myvar)
'The variable's value is %s' % myvar
```

使用 `str.format()` 进行字符串格式化在 Python 中稍微快一点，但在 Salt 中除了不合适的地方外都是必需的。

不要被 Python 2.7.x 中可用的以下快捷方式所诱惑：

```py
'The variable's value is {}'.format(myvar)
```

由于 Salt 仍然需要在 Python 2.6 上运行，而 Python 2.6 不支持使用 `{}` 代替 `{0}`，这将为旧平台上的用户带来问题。

# 最后一个模块

当我们将所有前面的代码放在一起时，我们最终得到以下模块：

```py
'''
This module should be saved as salt/modules/mysqltest.py
'''
import salt.utils

try:
    import MySQLdb
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

import logging
log = logging.getLogger(__name__)

__func_alias__ = {
    'list_': 'list'
}

__virtualname__ = 'mysqltest'

def __virtual__():
    '''
    Check dependencies, using both methods from the chapter
    '''
    if not salt.utils.which('mysql'):
        return False

    if HAS_LIBS:
        return __virtualname__

    return False

def ping():
    '''
    Returns True

    CLI Example:
        salt '*' mysqltest.ping
    '''
    return True

def check_mysqld():
    '''
    Check to see if sshd is running and listening

    CLI Example:
        salt '*' testmodule.check_mysqld
    '''
    output = __salt__'cmd.run'
    if 'tcp' not in output:
        return False
    return True

def _get_conn():
    '''
    Get a database connection object
    '''
    user = __salt__'config.get'
    passwd = __salt__'config.get'
    host = __salt__'config.get'
    port = __salt__'config.get'
    db_ = __salt__'config.get'
    dbc = MySQLdb.connect(
        connection_user=user,
        connection_pass=passwd,
        connection_host=host,
        connection_port=port,
        connection_db=db_,
    )
    log.trace('Connected to the database')
    return dbc

def version():
    '''
    Returns MySQL Version

    CLI Example:
        salt '*' mysqltest.version
    '''
    dbc = _get_conn()
    cur = dbc.cursor()
    return cur.execute('SELECT VERSION()')

def list_(type_):
    '''
    List different resources in MySQL
    CLI Examples:
        salt '*' mysqltest.list tables
        salt '*' mysqltest.list databases
    '''
    dbc = _get_conn()
    cur = dbc.cursor()
    valid_types = ['tables', 'databases']
    if type_ not in valid_types:
        err_msg = 'A valid type was not specified'
        log.error(err_msg)
        raise CommandExecutionError(err_msg)
    return cur.execute('SHOW {0}()'.format(type_))
```

# 调试执行模块

就像任何编程一样，你花在编写执行模块上的时间越多，你遇到问题的可能性就越大。让我们花点时间来谈谈如何调试和调试你的代码。

## 使用 salt-call

`salt-call` 命令始终是测试和调试代码的有价值工具。没有它，每次你想测试新代码时，都需要重新启动 `salt-minion` 服务；相信我，这很快就会变得很无聊。

由于 `salt-call` 不会启动服务，它将始终运行 Salt 代码的最新副本。它确实做了 `salt-minion` 服务的大部分工作：加载粒度、连接到 Master（除非被告知不要连接）以获取 pillar 数据、通过加载器过程决定要加载哪些模块，然后执行请求的命令。几乎唯一不做的就是保持运行。

使用 `salt-call` 发送命令与使用 `salt` 命令相同，只是不需要指定目标（因为目标是 `salt-call` 运行的 Minion）：

```py
#salt '*' mysqltest.ping
#salt-call mysqltest.ping

```

你可能会注意到，尽管你正在同一台机器上发出 `salt-call` 命令，该机器将执行操作，但它通常会运行得慢一点。这有两个原因。首先，你仍然基本上每次都在启动 `salt-minion` 服务，而没有真正让它保持运行。这意味着检测粒度、加载模块等操作将不得不每次都进行。

要了解这实际上需要多少时间，尝试比较带有和不带有粒度检测的执行时间：

```py
# time salt-call test.ping
local:
 True
real	0m3.257s
user	0m0.863s
sys	0m0.197s
# time salt-call --skip-grains test.ping
local:
 True
real	0m0.675s
user	0m0.507s
sys	0m0.080s

```

当然，如果你正在测试一个使用 grains 的模块，这不是一个可接受的战略。命令运行速度减慢的第二件事是必须连接到 Master。这并不像 grain 检测那样耗时，但它确实会减慢速度：

```py
# time salt-call --local test.ping
local:
 True
real	0m2.820s
user	0m0.797s
sys	0m0.120s

```

`--local`标志不仅告诉`salt-call`不要与 Master 通信。实际上，它告诉`salt-call`使用自身作为 Master（这意味着，以本地模式运行）。如果你的模块使用了 Master 上的 pillar 或其他资源，那么你只需在本地提供服务即可。

主机配置文件中你需要配置的任何内容都可以直接复制到`Minion`文件中。如果你只是使用默认设置，你甚至不需要这样做：只需从 Master 复制必要的文件到 Minion：

```py
# scp -r saltmaster:/srv/salt /srv
# scp -r saltmaster:/srv/pillar /srv

```

一切准备就绪后，使用`--local`标志启动`salt-call`并开始故障排除。

## <function>不可用

当我编写一个新的模块时，我遇到的第一大问题通常是让模块显示出来。这通常是因为代码明显有问题，比如打字错误。例如，如果我们把导入从`salt.utils`改为`salt.util`，我们的模块将无法加载：

```py
$ grep 'import salt' salt/modules/mysqltest.py
import salt.util
# salt-call --local mysqltest.ping
'mysqltest.ping' is not available.

```

在这种情况下，我们可以通过以`debug`模式运行`salt-call`来找到问题：

```py
# salt-call --local -l debug mysqltest.ping
...
[DEBUG   ] Failed to import module mysqltest:
Traceback (most recent call last):
 File "/usr/lib/python2.7/site-packages/salt/loader.py", line 1217, in _load_module
 ), fn_, fpath, desc)
 File "/usr/lib/python2.7/site-packages/salt/modules/mysqltest.py", line 4, in <module>
 import salt.util
ImportError: No module named util
...
'mysqltest.ping' is not available.

```

另一种可能是`__virtual__()`函数存在问题。这是我唯一一次建议向该函数添加日志消息的情况：

```py
def __virtual__():
    '''
    Check dependencies, using both methods from the chapter
    '''
    log.debug('Checking for mysql command')
    if not salt.utils.which('mysql'):
        return False

    log.debug('Checking for libs')
    if HAS_LIBS:
        return __virtualname__

    return False
```

然而，确保在生产之前将它们移除，否则你迟早会有一批非常不满意的用户。

### 小贴士

**下载示例代码**

你可以从[`www.packtpub.com`](http://www.packtpub.com)的账户下载本书的示例代码文件。如果你在其他地方购买了这本书，你可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给你。

你可以按照以下步骤下载代码文件：

+   使用您的电子邮件地址和密码登录或注册我们的网站。

+   将鼠标指针悬停在顶部的 SUPPORT 标签上。

+   点击代码下载与勘误表。

+   在搜索框中输入书籍名称。

+   选择你想要下载代码文件的书籍。

+   从下拉菜单中选择你购买这本书的地方。

+   点击代码下载。

文件下载完成后，请确保使用最新版本的软件解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

# 摘要

学习如何编写执行模块为编写其他 Salt 模块奠定了良好的基础。Salt 包含了许多内置功能，其中许多功能适用于所有模块类型。一些库也随 Salt 一起打包在`salt/utils/`目录中。使用`salt-call`命令（尤其是在本地模式下）进行故障排除时，Salt 模块的调试最为容易。

接下来，我们将讨论可以用来处理配置的各种类型的 Salt 模块。
