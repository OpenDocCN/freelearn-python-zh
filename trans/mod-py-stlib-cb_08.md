# 第八章：密码学

本章中，我们将涵盖以下食谱：

+   要求密码-在终端软件中要求密码时，请确保不要泄漏它。

+   哈希密码-如何存储密码而不会泄漏风险？

+   验证文件的完整性-如何检查通过网络传输的文件是否已损坏。

+   验证消息的完整性-如何检查您发送给另一个软件的消息是否已被更改。

# 介绍

虽然加密通常被认为是一个复杂的领域，但它是我们作为软件开发人员日常生活的一部分，或者至少应该是，以确保我们的代码库具有最低的安全级别。

本章试图覆盖大多数您每天都必须面对的常见任务的食谱，这些任务可以帮助使您的软件对攻击具有抵抗力。

虽然用 Python 编写的软件很难受到利用，比如缓冲区溢出（除非解释器或您依赖的编译库中存在错误），但仍然有很多情况可能会泄露必须保密的信息。

# 要求密码

在基于终端的程序中，通常会向用户询问密码。通常不建议从命令选项中这样做，因为在类 Unix 系统上，可以通过运行`ps`命令获取进程列表的任何人都可以看到它们，并且可以通过运行`history`命令获取最近执行的命令列表。

虽然有方法可以调整命令参数以将其隐藏在进程列表中，但最好还是交互式地要求密码，以便不留下任何痕迹。

但是，仅仅交互地要求它们是不够的，除非您还确保在输入时不显示它们，否则任何看着您屏幕的人都可以获取您的所有密码。

# 如何做...

幸运的是，Python 标准库提供了一种从提示中输入密码而不显示它们的简单方法：

```py
>>> import getpass
>>> pwd = getpass.getpass()
Password: 
>>> print(pwd)
'HelloWorld'
```

# 它是如何工作的...

`getpass.getpass`函数将在大多数系统上使用`termios`库来禁用用户输入的字符的回显。为了避免干扰其他应用程序输入，它将在终端的新文件描述符中完成。

在不支持此功能的系统上，它将使用更基本的调用直接从`sys.stdin`读取字符而不回显它们。

# 哈希密码

避免以明文存储密码是一种已知的最佳实践，因为软件通常只需要检查用户提供的密码是否正确，并且可以存储密码的哈希值并与提供的密码的哈希值进行比较。如果两个哈希值匹配，则密码相等；如果不匹配，则提供的密码是错误的。

存储密码是一个非常标准的做法，通常它们被存储为哈希加一些盐。盐是一个随机生成的字符串，它在哈希之前与密码连接在一起。由于是随机生成的，它确保即使相同密码的哈希也会得到不同的结果。

Python 标准库提供了一套相当完整的哈希函数，其中一些非常适合存储密码。

# 如何做...

Python 3 引入了密钥派生函数，特别适用于存储密码。提供了`pbkdf2`和`scrypt`。虽然`scrypt`更加抗攻击，因为它既消耗内存又消耗 CPU，但它只能在提供 OpenSSL 1.1+的系统上运行。而`pbkdf2`可以在任何系统上运行，在最坏的情况下会使用 Python 提供的后备。

因此，从安全性的角度来看，`scrypt`更受青睐，但由于其更广泛的可用性以及自 Python 3.4 以来就可用的事实，我们将依赖于`pbkdf2`（`scrypt`仅在 Python 3.6+上可用）：

```py
import hashlib, binascii, os

def hash_password(password):
    """Hash a password for storing."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), 
                                salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password[:64]
    stored_password = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512', 
                                  provided_password.encode('utf-8'), 
                                  salt.encode('ascii'), 
                                  100000)
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')
    return pwdhash == stored_password
```

这两个函数可以用来对用户提供的密码进行哈希处理，以便存储在磁盘或数据库中（`hash_password`），并在用户尝试重新登录时验证密码是否与存储的密码匹配（`verify_password`）：

```py
>>> stored_password = hash_password('ThisIsAPassWord')
>>> print(stored_password)
cdd5492b89b64f030e8ac2b96b680c650468aad4b24e485f587d7f3e031ce8b63cc7139b18
aba02e1f98edbb531e8a0c8ecf971a61560b17071db5eaa8064a87bcb2304d89812e1d07fe
bfea7c73bda8fbc2204e0407766197bc2be85eada6a5
>>> verify_password(stored_password, 'ThisIsAPassWord')
True
>>> verify_password(stored_password, 'WrongPassword')
False
```

# 工作原理...

这里涉及两个函数：

+   `hash_password`：以安全的方式对提供的密码进行编码，以便存储在数据库或文件中

+   `verify_password`：给定一个编码的密码和用户提供的明文密码，它验证提供的密码是否与编码的（因此已保存的）密码匹配。

`hash_password`实际上做了多件事情；它不仅仅是对密码进行哈希处理。

它的第一件事是生成一些随机盐，应该添加到密码中。这只是从`os.urandom`读取的一些随机字节的`sha256`哈希。然后提取哈希盐的字符串表示形式作为一组十六进制数字（`hexdigest`）。

然后将盐提供给`pbkdf2_hmac`，与密码本身一起进行哈希处理，以随机化的方式哈希密码。由于`pbkdf2_hmac`需要字节作为输入，因此两个字符串（密码和盐）先前被编码为纯字节。盐被编码为纯 ASCII，因为哈希的十六进制表示只包含 0-9 和 A-F 字符。而密码被编码为`utf-8`，它可能包含任何字符。（有人的密码里有表情符号吗？）

生成的`pbkdf2`是一堆字节，因为我们想要将其存储到数据库中；我们使用`binascii.hexlify`将一堆字节转换为它们的十六进制表示形式的字符串格式。`hexlify`是一种方便的方法，可以将字节转换为字符串而不丢失数据。它只是将所有字节打印为两个十六进制数字，因此生成的数据将比原始数据大一倍，但除此之外，它与转换后的数据完全相同。

最后，该函数将哈希与其盐连接在一起。因为我们知道`sha256`哈希的`hexdigest`始终是 64 个字符长。通过将它们连接在一起，我们可以通过读取结果字符串的前 64 个字符来重新获取盐。

这将允许`verify_password`验证密码，并验证是否需要使用用于编码的盐。

一旦我们有了密码，`verify_password`就可以用来验证提供的密码是否正确。因此，它需要两个参数：哈希密码和应该被验证的新密码。

`verify_password`的第一件事是从哈希密码中提取盐（记住，我们将它放在`hash_password`结果字符串的前 64 个字符中）。

然后将提取的盐和密码候选者提供给`pbkdf2_hmac`，计算它们的哈希，然后将其转换为一个字符串，使用`binascii.hexlify`。如果生成的哈希与先前存储的密码的哈希部分匹配（盐后的字符），这意味着这两个密码匹配。

如果结果哈希不匹配，这意味着提供的密码是错误的。正如你所看到的，我们非常重要的是将盐和密码一起提供，因为我们需要它来验证密码，不同的盐会导致不同的哈希，因此我们永远无法验证密码。

# 验证文件的完整性

如果你曾经从公共网络下载过文件，你可能会注意到它们的 URL 经常是这种形式：`http://files.host.com/somefile.tar.gz#md5=3b3f5b2327421800ef00c38ab5ad81a6`。

这是因为下载可能出错，你得到的数据可能部分损坏。因此 URL 包含了一个 MD5 哈希，你可以使用`md5sum`工具来验证下载的文件是否正确。

当你从 Python 脚本下载文件时也是一样。如果提供的文件有一个 MD5 哈希用于验证，你可能想要检查检索到的文件是否有效，如果不是，那么你可以重新尝试下载它。

# 如何做到...

在`hashlib`中，有多种受支持的哈希算法，而且可能最常见的是`md5`，因此我们可以依靠`hashlib`来验证我们下载的文件：

```py
import hashlib

def verify_file(filepath, expectedhash, hashtype='md5'):
    with open(filepath, 'rb') as f:
        try:
            filehash = getattr(hashlib, hashtype)()
        except AttributeError:
            raise ValueError(
                'Unsupported hashing type %s' % hashtype
            ) from None

        while True:
            data = f.read(4096)
            if not data:
                break
            filehash.update(data)

    return filehash.hexdigest() == expectedhash
```

然后我们可以使用`verify_file`下载并验证我们的文件。

例如，我可能从**Python Package Index** (**PyPI**)下载`wrapt`分发包，并且我可能想要验证它是否已正确下载。

文件名将是`wrapt-1.10.11.tar.gz#sha256=d4d560d479f2c21e1b5443bbd15fe7ec4b37fe7e53d335d3b9b0a7b1226fe3c6`，我可以运行我的`verify_file`函数：

```py
>>> verify_file(
...     'wrapt-1.10.11.tar.gz', 
...     'd4d560d479f2c21e1b5443bbd15fe7ec4b37fe7e53d335d3b9b0a7b1226fe3c6',
...     'sha256
... )
True
```

# 工作原理...

该函数的第一步是以二进制模式打开文件。由于所有哈希函数都需要字节，而且我们甚至不知道文件的内容，因此以二进制模式读取文件是最方便的解决方案。

然后，它检查所请求的哈希算法是否在`hashlib`中可用。通过`getattr`通过尝试抓取`hashlib.md5`，`hashlib.sha256`等来完成。如果不支持该算法，它将不是有效的`hashlib`属性（因为它不会存在于模块中），并且将抛出`AttributeError`。为了使这些更容易理解，它们被捕获并引发了一个新的`ValueError`，清楚地说明该算法不受支持。

文件打开并验证算法后，将创建一个空哈希（请注意，在`getattr`之后，括号将导致返回的哈希的创建）。

我们从一个空的开始，因为文件可能非常大，我们不想一次性读取完整的文件并将其一次性传递给哈希函数。

相反，我们从一个空哈希开始，并且以 4 KB 的块读取文件，然后将每个块馈送到哈希算法以更新哈希。

最后，一旦我们计算出哈希，我们就会获取其十六进制数表示，并将其与函数提供的哈希进行比较。

如果两者匹配，那么文件就是正确下载的。

# 验证消息的完整性

在通过公共网络或对其他用户和系统可访问的存储发送消息时，我们需要知道消息是否包含原始内容，或者是否被任何人拦截和修改。

这是一种典型的中间人攻击形式，它可以修改我们内容中的任何内容，这些内容存储在其他人也可以阅读的地方，例如未加密的网络或共享系统上的磁盘。

HMAC 算法可用于保证消息未从其原始状态更改，并且经常用于签署数字文档以确保其完整性。

HMAC 的一个很好的应用场景可能是密码重置链接；这些链接通常包括有关应该重置密码的用户的参数：[`myapp.com/reset-password?user=myuser@email.net`](http://myapp.com/reset-password?user=myuser@email.net)。

但是，任何人都可以替换用户参数并重置其他人的密码。因此，我们希望确保我们提供的链接实际上没有被修改，因为它是通过附加 HMAC 发送的。

这将导致类似于以下内容：[`myapp.com/reset-password?user=myuser@email.net&signature=8efc6e7161004cfb09d05af69cc0af86bb5edb5e88bd477ba545a9929821f582`](http://myapp.com/reset-password?user=myuser@email.net&signature=8efc6e7161004cfb09d05af69cc0af86bb5edb5e88bd477ba545a9929821f582)。

此外，任何尝试修改用户都将使签名无效，从而使其无法重置其他人的密码。

另一个用例是部署 REST API 以验证和验证请求。亚马逊网络服务使用 HMAC 作为其网络服务的身份验证系统。注册时，会为您提供访问密钥和密钥。您发出的任何请求都必须使用 HMAC 进行哈希处理，使用密钥来确保您实际上是请求中所述的用户（因为您拥有其密钥），并且请求本身没有以任何方式更改，因为它的详细信息也使用 HMAC 进行了哈希处理。

HMAC 签名经常涉及到软件必须向自身发送消息或从拥有密钥的验证合作伙伴接收消息的情况。

# 如何做...

对于这个示例，需要执行以下步骤：

1.  标准库提供了一个 `hmac` 模块，结合 `hashlib` 提供的哈希函数，可以用于计算任何提供的消息的身份验证代码：

```py
import hashlib, hmac, time

def compute_signature(message, secret):
    message = message.encode('utf-8')
    timestamp = str(int(time.time()*100)).encode('ascii')

    hashdata = message + timestamp
    signature = hmac.new(secret.encode('ascii'), 
                         hashdata, 
                         hashlib.sha256).hexdigest()
    return {
        'message': message,
        'signature': signature,
        'timestamp': timestamp
    }

def verify_signature(signed_message, secret):
    timestamp = signed_message['timestamp']
    expected_signature = signed_message['signature']
    message = signed_message['message']

    hashdata = message + timestamp
    signature = hmac.new(secret.encode('ascii'), 
                         hashdata, 
                         hashlib.sha256).hexdigest()
    return signature == expected_signature
```

1.  然后，我们的函数可以用来计算签名消息，并且我们可以检查签名消息是否被以任何方式更改：

```py
>>> signed_msg = compute_signature('Hello World', 'very_secret')
>>> verify_signature(signed_msg, 'very_secret')
True
```

1.  如果尝试更改签名消息的消息字段，它将不再有效，只有真实的消息才能匹配签名：

```py
>>> signed_msg['message'] = b'Hello Boat'
>>> verify_signature(signed_msg, 'very_secret')
False
```

# 工作原理...

我们的目的是确保任何给定的消息都不能以任何方式更改，否则将使附加到消息的签名无效。

因此，`compute_signature` 函数在给定消息和私有密钥的情况下，返回发送到接收方时签名消息应包括的所有数据。发送的数据包括消息本身、签名和时间戳。时间戳包括在内，因为在许多情况下，确保消息是最近的消息是一个好主意。如果您收到使用 HMAC 签名的 API 请求或刚刚设置的 cookie，您可能希望确保您处理的是最近的消息，而不是一个小时前发送的消息。时间戳无法被篡改，因为它与消息一起包括在签名中，其存在使得攻击者更难猜测密钥，因为两个相同的消息将导致有两个不同的签名，这要归功于时间戳。

一旦消息和时间戳已知，`compute_signature` 函数将它们与密钥一起传递给 `hmac.new`，以计算签名本身。为了方便起见，签名被表示为组成十六进制数字的字符，这些数字表示签名由哪些字节组成。这确保它可以作为纯文本在 HTTP 标头或类似方式中传输。

一旦我们得到了由 `compute_signature` 返回的签名消息，可以将其存储在某个地方，并在加载时使用 `verify_signature` 来检查它是否被篡改。

`verify_signature` 函数执行与 `compute_signature` 相同的步骤。签名的消息包括消息本身、时间戳和签名。因此，`verify_signature` 获取消息和时间戳，并与密钥结合计算签名。如果计算得到的签名与签名消息中提供的签名匹配，这意味着消息没有被以任何方式更改。否则，即使对消息或时间戳进行微小更改，签名也将无效。
