# 第八章。添加外部文件服务器

Salt 主服务器通常将其资源存储在托管它的机器上。这包括其他许多事情，比如为从服务器提供文件。文件服务器加载器允许你使用外部资源来存储这些文件，并将它们视为主服务器本地的文件。在本章中，我们将讨论：

+   理解 Salt 如何使用文件

+   抽象外部源以向 Salt 提供文件

+   使用 Salt 的缓存系统

+   故障排除外部文件服务器

# Salt 如何使用文件

Salt 内置的文件服务器在与从服务器通信时使用文件有两种方式。它们可以完整地提供服务，或者可以通过模板引擎进行处理，使用第五章中讨论的渲染模块，即*渲染数据*。

在任何情况下，这些文件都存储在一个或多个目录集中，这些目录是通过主配置文件中的`file_roots`指令配置的。这些目录按环境分组。当 Salt 寻找文件时，它将按照列出的顺序搜索这些目录。默认环境`base`通常使用`/srv/salt/`来存储文件。这样的配置看起来可能像：

```py
file_roots:
  base:
    - /srv/salt/
```

许多用户没有意识到，`file_roots`指令实际上是一个特定于名为`roots`的文件服务器模块的配置选项。这个模块，以及所有其他文件服务器模块，都是通过`fileserver_backend`指令进行配置的：

```py
fileserver_backend:
  - roots
```

这是你配置 Salt 中使用的任何其他文件服务器模块的地方。再次强调，模块的配置顺序与它们的使用顺序一致。当主服务器请求一个文件给从服务器时，Salt 会检查这些模块中的每一个，直到找到匹配项。找到后，它将停止搜索，并服务找到的文件。这意味着如果你有以下配置：

```py
fileserver_backend:
  - git
  - roots
```

如果 Salt 在 Git 中找到请求的文件，它将忽略在本地文件系统中找到的任何文件。

## 模拟文件系统

如果你以前编写过 FUSE 文件系统，你会在 Salt 文件服务器模块内部识别到一些函数。许多用于从操作系统请求文件的操作与 Salt 请求文件时使用的文件非常相似。归根结底，Salt 文件服务器模块实际上是一个虚拟文件系统，但它的 API 是专门为 Salt 设计的，而不是为操作系统设计的。

当你使用文件服务器模块进行开发时，你可能会注意到另一个趋势。虽然使用的数据可能存储在远程位置，但反复检索这些文件可能会在资源上造成成本。因此，许多文件服务器模块将从远程位置检索文件，并在主服务器上本地缓存它们，仅在必要时更新。

在这方面，当你编写文件服务器模块时，你通常只是在实现检索和缓存文件以及从缓存中提供文件的手段。这并不总是最好的做法；一个完全基于数据库查询的真正动态文件服务器可能通过始终执行查找来表现最佳。你需要从一开始就决定最合适的策略。

# 查看每个函数

我们将要编写的文件服务器将基于 SFTP。由于 SFTP 调用可能很昂贵，我们将使用一个依赖于流行的 Python 库 Paramiko 的缓存实现来检索文件。为了简单起见，我们只允许配置一个 SFTP 服务器，但如果你发现自己在使用这个模块，你可能想要考虑允许配置多个端点。

## 设置我们的模块

在我们介绍使用的函数之前，我们开始设置模块本身。我们将实现一些提供我们将在整个模块中使用的对象的函数：

```py
'''
The backend for serving files from an SFTP account.

To enable, add ``sftp`` to the :conf_master:`fileserver_backend` option in the
Master config file.

.. code-block:: yaml

    fileserver_backend:
      - sftp

Each environment is configured as a directory inside the SFTP account. The name
of the directory must match the name of the environment.

.. code-block:: yaml

    sftpfs_host: sftp.example.com
    sftpfs_port: 22
    sftpfs_username: larry
    sftpfs_password: 123pass
    sftpfs_root: /srv/sftp/salt/
'''
import os
import os.path
import logging
import time
import salt.fileserver
import salt.utils
import salt.syspaths

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import paramiko
    from paramiko import AuthenticationException
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

__virtualname__ = 'sftp'

log = logging.getLogger()

transport = None
client = None

def __virtual__():
    '''
    Only load if proper conditions are met
    '''
    if __virtualname__ not in __opts__['fileserver_backend']:
        return False

    if not HAS_LIBS:
        return False

    if __opts__.get('sftpfs_root', None) is None:
        return False

    global client
    global transport

    host = __opts__.get('sftpfs_host')
    port = __opts__.get('sftpfs_port', 22)
    username = __opts__.get('sftpfs_username')
    password = __opts__.get('sftpfs_password')
    try:
        transport = paramiko.Transport((host, port))
        transport.connect(username=username, password=password)
        client = paramiko.SFTPClient.from_transport(transport)
    except AuthenticationException:
        return False

    return True
```

已经有很多内容了！幸运的是，你现在应该已经认识到了大部分内容，所以这部分应该会很快过去。

我们包含了一个比平常更长的文档字符串，它解释了如何配置 Salt 使用我们的模块。当我们到达`__virtual__()`函数时，我们将看到这些参数的使用。

接下来，我们设置我们的导入。大多数这些导入的使用将在我们通过单个函数进行时进行说明，但有两个我们将其包裹在`try/except`块中。第一个是`fcntl`，这是一个 Unix 系统调用，用于处理文件描述符。这个库在 Unix 和 Linux 中用于锁定文件，但在 Windows 中不存在。然而，我们模块的其余部分在 Windows 中是可用的，所以我们现在设置一个标志，稍后当我们需要锁定文件时可以使用。

第二个导入是 Paramiko。这是 Python 中用于 SSH 和 SFTP 的最受欢迎的连接库之一，对于我们的目的来说简单易用。如果没有安装，我们可以在`__virtual__()`函数中返回`False`。

我们添加了`__virtualname__`，尽管它不是严格必要的，只是为了有一个中央且易于找到的地方来命名我们的模块。我们将在`__virtual__()`函数中使用这个变量。我们还添加了一些日志记录，我们将利用它们。

在加载`__virtual__()`函数之前，我们已经定义了两个变量，用于连接到 SFTP 服务器。我们将在`__virtual__()`内部将连接分配给它们，并且它们将在整个模块中使用。

最后，我们有我们的 `__virtual__()` 函数。首先，我们检查这个模块是否已经被配置用于使用。如果没有，就没有继续下去的必要了。我们还检查确保 Paramiko 已经安装。然后我们确保已经指定了 SFTP 服务器的根目录。现在可能还不明显，但这个目录在其他地方也将是必需的。如果它不存在，那么我们甚至不会尝试去连接服务器。

如果它已被定义，那么我们可以继续尝试建立连接。如果我们的其他参数定义不正确，Paramiko 将会抛出 `AuthenticationException`。在这种情况下，当然我们会认为这个模块不可用，并返回 `False`。但如果所有这些条件都满足，那么我们就准备好开始工作了！

让我们回顾一下在任何文件服务器模块中应该找到的函数。在每个部分，我们将实现并解释那个函数。

## envs()

我们首先报告哪些环境已经为这个文件服务器配置。至少，`base` 环境应该被支持并报告，但最好提供一个机制来支持其他环境。因为我们实际上是在抽象文件管理机制，所以通常最简单的方法就是通过将环境分离到目录中来实现：

```py
def envs():
    '''
    Treat each directory as an environment
    '''
    ret = []
    root = __opts__.get('sftpfs_root')
    for entry in client.listdir_attr(root):
        if str(oct(entry.st_mode)).startswith('04'):
            ret.append(entry.filename)
    return ret
```

这个函数需要返回一个列表。因为我们已经将环境分离到它们自己的目录中，所以我们的模块只需要返回我们配置的根目录下的目录列表。

这个函数很难测试，因为在任何 Salt 模块中都没有直接接口。然而，一旦下两个函数就位，就可以对其进行测试。

## file_list() 和 dir_list()

这两个函数相当直观；它们连接到远程端点，并返回该环境下的所有文件和目录列表：

```py
def file_list(load):
    '''
    Return a list of all files on the file server in a specified environment
    '''
    root = __opts__.get('sftpfs_root')
    path = os.path.join(root, load['saltenv'], load['prefix'])
    return _recur_path(path, load['saltenv'])

def dir_list(load):
    '''
    Return a list of all directories on the master
    '''
    root = __opts__.get('sftpfs_root')
    path = os.path.join(root, load['saltenv'], load['prefix'])
    return _recur_path(path, load['saltenv'], True)

def _recur_path(path, saltenv, only_dirs=False):
    '''
    Recurse through the remote directory structure
    '''
    root = __opts__.get('sftpfs_root')
    ret = []
    try:
        for entry in client.listdir_attr(path):
            full = os.path.join(path, entry.filename)
            if str(oct(entry.st_mode)).startswith('04'):
                ret.append(full)
                ret.extend(_recur_path(full, saltenv, only_dirs))
            else:
                if only_dirs is False:
                    ret.append(full)
        return ret
    except IOError:
        return []
```

这两个函数所需的东西完全相同，只是是否包含文件。因为递归通常总是需要的，所以我们添加了一个名为 `_recur_path()` 的递归函数，它可以报告目录或文件和目录。你可能注意到了对 `entry.st_mode` 的检查。你可能把 Unix 文件模式看作是一组权限，这些权限可以使用 `chmod` （**ch**ange **mod**e）命令来更改。然而，模式还存储了文件类型：

```py
0100755  # This is a file, with 0755 permissions
040755  # This is a directory, with 0755 permissions
```

我们可以使用另一个 try/except 块来检查是否可以进入一个目录。但检查模式会更省事。如果它以 `04` 开头，那么我们知道它是一个目录。

这些函数都需要一个 `load` 参数。如果你查看内部，你会找到一个看起来像这样的字典：

```py
{'cmd': '_file_list', 'prefix': '', 'saltenv': 'base'}
```

`cmd` 字段存储了使用了哪种命令。`prefix` 将包含包含任何请求文件的目录路径，在环境中，`saltenv` 告诉你请求的环境本身的名称。你将在整个模块中看到这个参数，但它的外观大致相同。

让我们来看看几个 Salt 命令：

```py
# salt-call --local cp.list_master
local:
 - testdir
 - testfile
# salt-call --local cp.list_master_dirs
local:
 - testdir

```

请记住，`--local` 将告诉 `salt-call` 假装它是它自己的 Master。在这种情况下，它将查找 `minion` 配置文件以获取连接参数。

## find_file()

与 `file_list()` 和 `dir_list()` 类似，此功能检查请求的路径。然后报告指定的文件是否存在：

```py
'''
def find_file(path, saltenv='base', **kwargs):
    '''
    Search the environment for the relative path
    '''
    fnd = {'path': '',
           'rel': ''}

    full = os.path.join(salt.syspaths.CACHE_DIR, 'sftpfs', saltenv, path)

    if os.path.isfile(full) and not salt.fileserver.is_file_ignored(__opts__, full):
        fnd['path'] = full
        fnd['rel'] = path

    return fnd
```

你可能已经注意到在此函数中没有进行任何 SFTP 调用。这是因为我们正在使用缓存文件服务器，我们现在需要检查的是文件是否已被缓存。如果是，那么 Salt 将直接从缓存中提供文件。

如果你正在编写一个不保留本地缓存的文件服务器模块，那么此功能应该检查远程端点以确保文件存在。

说到缓存，此函数中更重要的一行是定义 `full` 变量的那一行。这设置了用于此缓存文件服务器的目录结构。它使用 `salt.syspaths` 确定您平台上的正确目录；通常，这将是在 `/var/cache/salt/`。

注意，此函数中没有传递 `load`，但 `saltenv`（通常在 `load` 中）被传递。Salt 的早期版本将 `saltenv` 传递为 `env`，并将 `**kwargs` 函数作为通配符来防止 Python 在旧实现上崩溃。

再次强调，无法直接测试此功能。它将在本节后面的 `update()` 函数中使用。

## serve_file()

使用 `find_file()` 找到文件后，其数据将传递给此函数以返回实际的文件内容：

```py
def serve_file(load, fnd):
    '''
    Return a chunk from a file based on the data received
    '''
    ret = {'data': '',
           'dest': ''}

    if 'path' not in load or 'loc' not in load or 'saltenv' not in load:
        return ret

    if not fnd['path']:
        return ret

    ret['dest'] = fnd['rel']
    gzip = load.get('gzip', None)

    full = os.path.join(salt.syspaths.CACHE_DIR, 'sftpfs', fnd['path'])

    with salt.utils.fopen(fnd['path'], 'rb') as fp_:
        fp_.seek(load['loc'])
        data = fp_.read(__opts__['file_buffer_size'])
        if gzip and data:
            data = salt.utils.gzip_util.compress(data, gzip)
            ret['gzip'] = gzip
        ret['data'] = data
    return ret
```

此功能直接由 Salt 的内部文件服务器使用，在将文件分块传递给 Minions 之前将文件分割成块。如果在主配置文件中将 `gzip` 标志设置为 `True`，则每个这些块都将单独压缩。

由于在我们的情况下，此功能是从缓存中提供文件，因此你可能会使用这里打印的此功能，除了定义 `full` 变量的那一行。如果你没有使用缓存文件服务器，那么你需要有访问和提供文件每个块的方法，正如请求的那样。

你可以使用 `cp.get_file` 函数测试此功能。此功能需要下载的文件名和保存文件到本地的完整路径：

```py
# salt-call --local cp.get_file salt://testfile /tmp/testfile
local:
 /tmp/testfile

```

## update()

在固定的时间间隔内，Salt 将要求外部文件服务器对其进行维护。此功能将比较本地文件缓存（如果正在使用）与远程端点，并使用新信息更新 Salt：

```py
def update():
    '''
    Update the cache, and reap old entries
    '''
    base_dir = os.path.join(salt.syspaths.CACHE_DIR, 'sftpfs')
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    try:
        salt.fileserver.reap_fileserver_cache_dir(
            os.path.join(base_dir, 'hash'),
            find_file
        )
    except (IOError, OSError):
        # Hash file won't exist if no files have yet been served up
        pass

    # Find out what the latest file is, so that we only update files more
    # recent than that, and not the entire filesystem
    if os.listdir(base_dir):
        all_files = []
        for root, subFolders, files in os.walk(base_dir):
            for fn_ in files:
                full_path = os.path.join(root, fn_)
                all_files.append([
                    os.path.getmtime(full_path),
                    full_path,
                ])

    # Pull in any files that have changed
    for env in envs():
        path = os.path.join(__opts__['sftpfs_root'], env)
        result = client.listdir_attr(path)
        for fileobj in result:
            file_name = os.path.join(base_dir, env, fileobj.filename)

            # Make sure the directory exists first
            comps = file_name.split('/')
            file_path = '/'.join(comps[:-1])
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            if str(oct(fileobj.st_mode)).startswith('04'):
                # Create the directory
                if not os.path.exists(file_name):
                    os.makedirs(file_name)
            else:
                # Write out the file
                if fileobj.st_mtime > all_files[file_name]:
                    client.get(os.path.join(path, fileobj.filename), file_name)
            os.utime(file_name, (fileobj.st_atime, fileobj.st_mtime))
```

呼呼！这是一个很长的函数！首先，我们定义缓存目录，如果它不存在，则创建它。这对于缓存文件服务器来说很重要。然后我们要求 Salt 使用内置的`salt.fileserver.reap_fileserver_cache_dir()`函数清理旧条目。这传递了`find_file()`的引用以帮助工作。

下一节将介绍剩余的文件，以检查它们的最后修改时间戳。只有在文件尚未下载，或者远程 SFTP 服务器上有更新的副本时，才会下载文件。

最后，我们遍历每个环境，查看哪些文件已更改，并在必要时下载它们。如果本地缓存中不存在任何目录，则会创建它们。无论我们创建文件还是目录，我们都会确保更新其时间戳，以便缓存与服务器上的内容相匹配。

这个函数将由 Salt Master 定期运行，但您可以通过手动从本地缓存中删除文件并请求副本来强制它运行：

```py
# rm /var/cache/salt/sftpfs/base/testfile
# salt-call --local cp.get_file salt://testfile /tmp/testfile
local:
 /tmp/testfile

```

## file_hash()

Salt 知道文件已被更改的一种方式是跟踪文件的哈希签名。如果哈希值发生变化，那么 Salt 将知道是时候从缓存中提供文件的新副本了：

```py
def file_hash(load, fnd):
    '''
    Return a file hash, the hash type is set in the master config file
    '''
    path = fnd['path']
    ret = {}

    # if the file doesn't exist, we can't get a hash
    if not path or not os.path.isfile(path):
        return ret

    # set the hash_type as it is determined by config
    ret['hash_type'] = __opts__['hash_type']

    # Check if the hash is cached
    # Cache file's contents should be 'hash:mtime'
    cache_path = os.path.join(
        salt.syspaths.CACHE_DIR,
        'sftpfs',
        'hash',
        load['saltenv'],
        '{0}.hash.{1}'.format(
            fnd['rel'],
            ret['hash_type']
        )
    )

    # If we have a cache, serve that if the mtime hasn't changed
    if os.path.exists(cache_path):
        try:
            with salt.utils.fopen(cache_path, 'rb') as fp_:
                try:
                    hsum, mtime = fp_.read().split(':')
                except ValueError:
                    log.debug(
                        'Fileserver attempted to read incomplete cache file. Retrying.'
                    )
                    file_hash(load, fnd)
                    return ret
                if os.path.getmtime(path) == mtime:
                    # check if mtime changed
                    ret['hsum'] = hsum
                    return ret
        except os.error:
            # Can't use Python select() because we need Windows support
            log.debug(
                'Fileserver encountered lock when reading cache file. Retrying.'
            )
            file_hash(load, fnd)
            return ret

    # If we don't have a cache entry-- lets make one
    ret['hsum'] = salt.utils.get_hash(path, __opts__['hash_type'])
    cache_dir = os.path.dirname(cache_path)

    # Make cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Save the cache object 'hash:mtime'
    if HAS_FCNTL:
        with salt.utils.flopen(cache_path, 'w') as fp_:
            fp_.write('{0}:{1}'.format(ret['hsum'], os.path.getmtime(path)))
            fcntl.flock(fp_.fileno(), fcntl.LOCK_UN)
        return ret
    else:
        with salt.utils.fopen(cache_path, 'w') as fp_:
            fp_.write('{0}:{1}'.format(ret['hsum'], os.path.getmtime(path)))
        return ret
```

这是我们示例中最长的函数，但幸运的是，它也需要最少的修改，对于一个缓存文件服务器来说。正如本书中的其他示例一样，您可以从 Packt Publishing 的网站上下载此模块的副本。一旦下载完成，您可能只需要更改`cache_path`的值。然而，我们仍然会简要地介绍这个函数。

在设置了一些基本设置，包括正在散列的文件的路径，检查该路径是否存在，并定义在缓存中保存哈希副本的位置之后。在我们的例子中，我们在缓存中设置了一个额外的目录结构，与原始结构相似，但文件名后附加了`.hash.<hash_type>`。生成的文件将具有如下名称：

```py
/var/cache/salt/sftpfs/hash/base/testfile.hash.md5
```

下一节将检查哈希文件是否已创建，以及是否与本地副本的时间戳匹配。如果现有哈希文件的时间戳太旧，则将生成新的哈希值。

如果我们通过了所有这些，那么我们就知道是时候生成新的哈希值了。在确定要使用的哈希类型并设置存放它的目录之后，我们到达了实际将哈希写入磁盘的部分。还记得模块开头对`fcntl`的检查吗？在繁忙的 Salt Master 上，可能同时尝试对同一文件进行多次操作。有了`fcntl`，我们可以在写入之前锁定该文件，以避免损坏。

# 最终模块

在所有函数就绪后，最终的模块将看起来像这样：

```py
'''
The backend for serving files from an SFTP account.

To enable, add ``sftp`` to the :conf_master:`fileserver_backend` option in the
Master config file.

.. code-block:: yaml

    fileserver_backend:
      - sftp

Each environment is configured as a directory inside the SFTP account. The name
of the directory must match the name of the environment.

.. code-block:: yaml

    sftpfs_host: sftp.example.com
    sftpfs_port: 22
    sftpfs_username: larry
    sftpfs_password: 123pass
    sftpfs_root: /srv/sftp/salt/
'''
import os
import os.path
import logging
import time

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # fcntl is not available on windows
    HAS_FCNTL = False

import salt.fileserver
import salt.utils
import salt.syspaths

try:
    import paramiko
    from paramiko import AuthenticationException
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

__virtualname__ = 'sftp'

log = logging.getLogger() 
transport = None
client = None

def __virtual__():
    '''
    Only load if proper conditions are met
    '''
    if __virtualname__ not in __opts__['fileserver_backend']:
        return False

    if not HAS_LIBS:
        return False

    if __opts__.get('sftpfs_root', None) is None:
        return False

    global client
    global transport

    host = __opts__.get('sftpfs_host')
    port = __opts__.get('sftpfs_port', 22)
    username = __opts__.get('sftpfs_username')
    password = __opts__.get('sftpfs_password')
    try:
        transport = paramiko.Transport((host, port))
        transport.connect(username=username, password=password)
        client = paramiko.SFTPClient.from_transport(transport)
    except AuthenticationException:
        return False

    return True

def envs():
    '''
    Treat each directory as an environment
    '''
    ret = []
    root = __opts__.get('sftpfs_root')
    for entry in client.listdir_attr(root):
        if str(oct(entry.st_mode)).startswith('04'):
            ret.append(entry.filename)
    return ret

def file_list(load):
    '''
    Return a list of all files on the file server in a specified environment
    '''
    root = __opts__.get('sftpfs_root')
    path = os.path.join(root, load['saltenv'], load['prefix'])
    return _recur_path(path, load['saltenv'])

def dir_list(load):
    '''
    Return a list of all directories on the master
    '''
    root = __opts__.get('sftpfs_root')
    path = os.path.join(root, load['saltenv'], load['prefix'])
    return _recur_path(path, load['saltenv'], True)

def _recur_path(path, saltenv, only_dirs=False):
    '''
    Recurse through the remote directory structure
    '''
    root = __opts__.get('sftpfs_root')
    ret = []
    try:
        for entry in client.listdir_attr(path):
            full = os.path.join(path, entry.filename)
            if str(oct(entry.st_mode)).startswith('04'):
                ret.append(full)
                ret.extend(_recur_path(full, saltenv, only_dirs))
            else:
                if only_dirs is False:
                    ret.append(full)
        return ret
    except IOError:
        return []

def find_file(path, saltenv='base', env=None, **kwargs):
    '''
    Search the environment for the relative path
    '''
    fnd = {'path': '',
           'rel': ''}

    full = os.path.join(salt.syspaths.CACHE_DIR, 'sftpfs', saltenv, path)

    if os.path.isfile(full) and not salt.fileserver.is_file_ignored(__opts__, full):
        fnd['path'] = full
        fnd['rel'] = path

    return fnd

def serve_file(load, fnd):
    '''
    Return a chunk from a file based on the data received
    '''
    ret = {'data': '',
           'dest': ''}

    if 'path' not in load or 'loc' not in load or 'saltenv' not in load:
        return ret

    if not fnd['path']:
        return ret

    ret['dest'] = fnd['rel']
    gzip = load.get('gzip', None)

    full = os.path.join(salt.syspaths.CACHE_DIR, 'sftpfs', fnd['path'])

    with salt.utils.fopen(fnd['path'], 'rb') as fp_:
        fp_.seek(load['loc'])
        data = fp_.read(__opts__['file_buffer_size'])
        if gzip and data:
            data = salt.utils.gzip_util.compress(data, gzip)
            ret['gzip'] = gzip
        ret['data'] = data
    return ret

def update():
    '''
    Update the cache, and reap old entries
    '''
    base_dir = os.path.join(salt.syspaths.CACHE_DIR, 'sftpfs')
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    try:
        salt.fileserver.reap_fileserver_cache_dir(
            os.path.join(base_dir, 'hash'),
            find_file
        )
    except (IOError, OSError):
        # Hash file won't exist if no files have yet been served up
        pass

    # Find out what the latest file is, so that we only update files more
    # recent than that, and not the entire filesystem
    if os.listdir(base_dir):
        all_files = {}
        for root, subFolders, files in os.walk(base_dir):
            for fn_ in files:
                full_path = os.path.join(root, fn_)
                all_files[full_path] = os.path.getmtime(full_path)

    # Pull in any files that have changed
    for env in envs():
        path = os.path.join(__opts__['sftpfs_root'], env)
        result = client.listdir_attr(path)
        for fileobj in result:
            file_name = os.path.join(base_dir, env, fileobj.filename)

            # Make sure the directory exists first
            comps = file_name.split('/')
            file_path = '/'.join(comps[:-1])
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            if str(oct(fileobj.st_mode)).startswith('04'):
                # Create the directory
                if not os.path.exists(file_name):
                    os.makedirs(file_name)
            else:
                # Write out the file
                if fileobj.st_mtime > all_files[file_name]:
                    client.get(os.path.join(path, fileobj.filename), file_name)
            os.utime(file_name, (fileobj.st_atime, fileobj.st_mtime))

def file_hash(load, fnd):
    '''
    Return a file hash, the hash type is set in the master config file
    '''
    path = fnd['path']
    ret = {}

    # if the file doesn't exist, we can't get a hash
    if not path or not os.path.isfile(path):
        return ret

    # set the hash_type as it is determined by config
    # -- so mechanism won't change that
    ret['hash_type'] = __opts__['hash_type']

    # Check if the hash is cached
    # Cache file's contents should be 'hash:mtime'
    cache_path = os.path.join(
        salt.syspaths.CACHE_DIR,
        'sftpfs',
        'hash',
        load['saltenv'],
        '{0}.hash.{1}'.format(
            fnd['rel'],
            ret['hash_type']
        )
    )

    # If we have a cache, serve that if the mtime hasn't changed
    if os.path.exists(cache_path):
        try:
            with salt.utils.fopen(cache_path, 'rb') as fp_:
                try:
                    hsum, mtime = fp_.read().split(':')
                except ValueError:
                    log.debug(
                        'Fileserver attempted to read'
                        'incomplete cache file. Retrying.'
                    )
                    file_hash(load, fnd)
                    return ret
                if os.path.getmtime(path) == mtime:
                    # check if mtime changed
                    ret['hsum'] = hsum
                    return ret
        except os.error:
            # Can't use Python select() because we need Windows support
            log.debug(
                'Fileserver encountered lock when reading cache file. Retrying.'
            )
            file_hash(load, fnd)
            return ret

    # If we don't have a cache entry-- lets make one
    ret['hsum'] = salt.utils.get_hash(path, __opts__['hash_type'])
    cache_dir = os.path.dirname(cache_path)

    # Make cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Save the cache object 'hash:mtime'
    if HAS_FCNTL:
        with salt.utils.flopen(cache_path, 'w') as fp_:
            fp_.write('{0}:{1}'.format(ret['hsum'], os.path.getmtime(path)))
            fcntl.flock(fp_.fileno(), fcntl.LOCK_UN)
        return ret
    else:
        with salt.utils.fopen(cache_path, 'w') as fp_:
            fp_.write('{0}:{1}'.format(ret['hsum'], os.path.getmtime(path)))
        return ret
```

# 文件服务器故障排除

文件服务器模块可能难以调试，因为许多组件需要就位，其他组件才能使用。但有一些技巧你可以记住。

## 从小开始

我已经尝试以编写和调试最容易的顺序呈现必要的功能。虽然不能直接调用`envs()`，但它很容易编写，可以在处理`file_list()`和`dir_list()`时进行调试。而且，可以使用`cp.list_master`和`cp.list_master_dirs`函数分别轻松调试这两个功能。

## 在 Minion 上测试

虽然文件服务器模块是为在主服务器上使用而设计的，但它们也可以在 Minion 上进行测试。确保在`minion`配置文件中而不是在`master`文件中定义所有适当的配置。使用`salt-call --local`来发布命令，并定期清除本地缓存（在`/var/salt/cache/`中）以及使用`cp.get_file`下载的任何文件。

# 摘要

文件服务器模块可以用来在外部端点呈现资源，就像它们是位于主服务器上的文件一样。默认的文件服务器模块，名为`roots`，实际上确实使用了主服务器上的本地文件。许多文件服务器模块在主服务器上本地缓存文件，以避免对外部源进行过多的调用，但这并不总是合适的。

文件服务器模块内部有许多功能，它们协同工作以呈现类似文件服务器的接口。其中一些功能不能直接测试，但它们仍然可以与其他具有直接外部接口的功能一起测试。

尽管涉及许多功能，但文件服务器模块相对容易编写。在下一章中，我们将讨论云模块，它们有更多的必需功能，但编写起来却更加容易。
