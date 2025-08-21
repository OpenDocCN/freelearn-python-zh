# 第九章：加密和令牌

"三人可以保守一个秘密，如果其中两人已经死了。" – 本杰明·富兰克林，《穷查理年鉴》

在这一简短的章节中，我将简要概述 Python 标准库提供的加密服务。我还将涉及一种称为 JSON Web Token 的东西，这是一种非常有趣的标准，用于在两个方之间安全地表示声明。

特别是，我们将探讨以下内容：

+   Hashlib

+   秘密

+   HMAC

+   使用 PyJWT 的 JSON Web Tokens，这似乎是处理 JWTs 最流行的 Python 库。

让我们花点时间谈谈加密以及为什么它如此重要。

# 加密的需求

根据网上可以找到的统计数据，2019 年智能手机用户的估计数量将达到 25 亿左右。这些人中的每一个都知道解锁手机的 PIN 码，登录到我们所有用来做基本上所有事情的应用程序的凭据，从购买食物到找到一条街，从给朋友发消息到查看我们的比特币钱包自上次检查 10 秒钟前是否增值。

如果你是一个应用程序开发者，你必须非常、非常认真地对待安全性。无论你的应用程序有多小或者看似不重要：安全性应该始终是你关注的问题。

信息技术中的安全性是通过采用多种不同的手段来实现的，但到目前为止，最重要的手段是加密。你在电脑或手机上做的每件事情都应该包括一个加密发生的层面（如果没有，那真的很糟糕）。它用于用信用卡在线支付，以一种方式在网络上传输消息，即使有人截获了它们，他们也无法阅读，它用于在你将文件备份到云端时对文件进行加密（因为你会这样做，对吧？）。例子的列表是无穷无尽的。

现在，本章的目的并不是教你区分哈希和加密的区别，因为我可以写一本完全不同的书来讨论这个话题。相反，它的目的是向你展示如何使用 Python 提供的工具来创建摘要、令牌，以及在一般情况下，当你需要实现与加密相关的东西时，如何更安全地操作。

# 有用的指导方针

永远记住以下规则：

+   **规则一**：不要尝试创建自己的哈希或加密函数。真的不要。使用已经存在的工具和函数。要想出一个好的、稳固的算法来进行哈希或加密是非常困难的，所以最好将其留给专业的密码学家。

+   **规则二**：遵循规则一。

这就是你需要的唯一两条规则。除此之外，了解加密是非常有用的，所以你需要尽量多地了解这个主题。网上有大量的信息，但为了方便起见，我会在本章末尾放一些有用的参考资料。

现在，让我们深入研究我想向你展示的标准库模块中的第一个：`hashlib`。

# Hashlib

这个模块向许多不同的安全哈希和消息摘要算法公开了一个通用接口。这两个术语的区别只是历史上的：旧算法被称为**摘要**，而现代算法被称为**哈希**。

一般来说，哈希函数是指任何可以将任意大小的数据映射到固定大小数据的函数。它是一种单向加密，也就是说，不希望能够根据其哈希值恢复消息。

有几种算法可以用来计算哈希值，所以让我们看看如何找出你的系统支持哪些算法（注意，你的结果可能与我的不同）：

```py
>>> import hashlib
>>> hashlib.algorithms_available
{'SHA512', 'SHA256', 'shake_256', 'sha3_256', 'ecdsa-with-SHA1',
 'DSA-SHA', 'sha1', 'sha384', 'sha3_224', 'whirlpool', 'mdc2',
 'RIPEMD160', 'shake_128', 'MD4', 'dsaEncryption', 'dsaWithSHA',
 'SHA1', 'blake2s', 'md5', 'sha', 'sha224', 'SHA', 'MD5',
 'sha256', 'SHA384', 'sha3_384', 'md4', 'SHA224', 'MDC2',
 'sha3_512', 'sha512', 'blake2b', 'DSA', 'ripemd160'}
>>> hashlib.algorithms_guaranteed
{'blake2s', 'md5', 'sha224', 'sha3_512', 'shake_256', 'sha3_256',
 'shake_128', 'sha256', 'sha1', 'sha512', 'blake2b', 'sha3_384',
 'sha384', 'sha3_224'}
```

通过打开 Python shell，我们可以获取系统中可用的算法列表。如果我们的应用程序必须与第三方应用程序通信，最好从那些有保证的算法中选择一个，因为这意味着每个平台实际上都支持它们。注意到很多算法都以**sha**开头，这意味着**安全哈希算法**。让我们在同一个 shell 中继续：我们将为二进制字符串`b'Hash me now!'`创建一个哈希，我们将以两种方式进行：

```py
>>> h = hashlib.blake2b()
>>> h.update(b'Hash me')
>>> h.update(b' now!')
>>> h.hexdigest()
'56441b566db9aafcf8cdad3a4729fa4b2bfaab0ada36155ece29f52ff70e1e9d'
'7f54cacfe44bc97c7e904cf79944357d023877929430bc58eb2dae168e73cedf'
>>> h.digest()
b'VD\x1bVm\xb9\xaa\xfc\xf8\xcd\xad:G)\xfaK+\xfa\xab\n\xda6\x15^'
b'\xce)\xf5/\xf7\x0e\x1e\x9d\x7fT\xca\xcf\xe4K\xc9|~\x90L\xf7'
b'\x99D5}\x028w\x92\x940\xbcX\xeb-\xae\x16\x8es\xce\xdf'
>>> h.block_size
128
>>> h.digest_size
64
>>> h.name
'blake2b'
```

我们使用了`blake2b`加密函数，这是一个相当复杂的函数，它是在 Python 3.6 中添加的。创建哈希对象`h`后，我们以两步更新其消息。虽然我们不需要，但有时我们需要对不一次性可用的数据进行哈希，所以知道我们可以分步进行是很好的。

当消息符合我们的要求时，我们得到摘要的十六进制表示。这将使用每个字节两个字符（因为每个字符代表 4 位，即半个字节）。我们还得到摘要的字节表示，然后检查其细节：它有一个块大小（哈希算法的内部块大小，以字节为单位）为 128 字节，一个摘要大小（结果哈希的大小，以字节为单位）为 64 字节，还有一个名称。所有这些是否可以在一行中完成？是的，当然：

```py
>>> hashlib.blake2b(b'Hash me now!').hexdigest()
'56441b566db9aafcf8cdad3a4729fa4b2bfaab0ada36155ece29f52ff70e1e9d'
'7f54cacfe44bc97c7e904cf79944357d023877929430bc58eb2dae168e73cedf'
```

注意相同的消息产生相同的哈希，这当然是预期的。

让我们看看如果我们使用`sha256`而不是`blake2b`函数会得到什么：

```py
>>> hashlib.sha256(b'Hash me now!').hexdigest()
'10d561fa94a89a25ea0c7aa47708bdb353bbb062a17820292cd905a3a60d6783'
```

生成的哈希较短（因此不太安全）。

哈希是一个非常有趣的话题，当然，我们迄今为止看到的简单示例只是开始。`blake2b`函数允许我们在定制方面有很大的灵活性。这对于防止某些类型的攻击非常有用（有关这些威胁的完整解释，请参考标准文档：[`docs.python.org/3.7/library/hashlib.html`](https://docs.python.org/3.7/library/hashlib.html)中的`hashlib`模块）。让我们看另一个例子，我们通过添加`key`、`salt`和`person`来定制一个哈希。所有这些额外信息将导致哈希与我们没有提供它们时得到的哈希不同，并且在为我们系统处理的数据添加额外安全性方面至关重要：

```py
>>> h = hashlib.blake2b(
...   b'Important payload', digest_size=16, key=b'secret-key',
...   salt=b'random-salt', person=b'fabrizio'
... )
>>> h.hexdigest()
'c2d63ead796d0d6d734a5c3c578b6e41'
```

生成的哈希只有 16 字节长。在定制参数中，`salt`可能是最著名的一个。它是用作哈希数据的额外输入的随机数据。通常与生成的哈希一起存储，以便提供恢复相同哈希的手段，给定相同的消息。

如果你想确保正确地哈希一个密码，你可以使用`pbkdf2_hmac`，这是一种密钥派生算法，它允许你指定算法本身使用的`salt`和迭代次数。随着计算机变得越来越强大，增加随时间进行的迭代次数非常重要，否则随着时间的推移，成功的暴力破解攻击的可能性会增加。以下是你如何使用这样的算法：

```py
>>> import os
>>> dk = hashlib.pbkdf2_hmac(
...   'sha256', b'Password123', os.urandom(16), 100000
... )
>>> dk.hex()
'f8715c37906df067466ce84973e6e52a955be025a59c9100d9183c4cbec27a9e'
```

请注意，我已经使用`os.urandom`提供了一个 16 字节的随机盐，这是文档推荐的。

我鼓励你去探索和尝试这个模块，因为迟早你会不得不使用它。现在，让我们继续`secrets`。

# 秘密

这个小巧的模块用于生成密码强度的随机数，适用于管理密码、账户认证、安全令牌和相关秘密。它是在 Python 3.6 中添加的，基本上处理三件事：随机数、令牌和摘要比较。让我们快速地探索一下它们。

# 随机数

我们可以使用三个函数来处理随机数：

```py
# secrs/secr_rand.py
import secrets
print(secrets.choice('Choose one of these words'.split()))
print(secrets.randbelow(10 ** 6))
print(secrets.randbits(32))
```

第一个函数`choice`从非空序列中随机选择一个元素。第二个函数`randbelow`生成一个介于`0`和您调用它的参数之间的随机整数，第三个函数`randbits`生成一个具有*n*个随机位的整数。运行该代码会产生以下输出（始终不同）：

```py
$ python secr_rand.py
one
504156
3172492450
```

在需要在密码学环境中需要随机性时，您应该使用这些函数，而不是`random`模块中的函数，因为这些函数是专门为此任务设计的。让我们看看模块为我们提供了什么样的令牌。

# 令牌生成

同样，我们有三个函数，它们都以不同的格式生成令牌。让我们看一个例子：

```py
# secrs/secr_rand.py
print(secrets.token_bytes(16))
print(secrets.token_hex(32))
print(secrets.token_urlsafe(32))
```

第一个函数`token_bytes`简单地返回一个包含*n*个字节（在本例中为`16`）的随机字节字符串。另外两个函数也是如此，但`token_hex`以十六进制格式返回一个令牌，而`token_urlsafe`返回一个仅包含适合包含在 URL 中的字符的令牌。让我们看看输出（这是上一次运行的延续）：

```py
b'\xda\x863\xeb\xbb|\x8fk\x9b\xbd\x14Q\xd4\x8d\x15}'
9f90fd042229570bf633e91e92505523811b45e1c3a72074e19bbeb2e5111bf7
bl4qz_Av7QNvPEqZtKsLuTOUsNLFmXW3O03pn50leiY 
```

这一切都很好，那么为什么我们不用这些工具写一个随机密码生成器来玩一下呢？

```py
# secrs/secr_gen.py
import secrets
from string import digits, ascii_letters

def generate_pwd(length=8):
    chars = digits + ascii_letters
    return ''.join(secrets.choice(chars) for c in range(length))

def generate_secure_pwd(length=16, upper=3, digits=3):
    if length < upper + digits + 1:
        raise ValueError('Nice try!')
    while True:
        pwd = generate_pwd(length)
        if (any(c.islower() for c in pwd)
            and sum(c.isupper() for c in pwd) >= upper
            and sum(c.isdigit() for c in pwd) >= digits):
            return pwd

print(generate_secure_pwd())
print(generate_secure_pwd(length=3, upper=1, digits=1))
```

在前面的代码中，我们定义了两个函数。`generate_pwd`简单地通过从包含字母表（小写和大写）和 10 个十进制数字的字符串中随机选择`length`个字符，并将它们连接在一起来生成给定长度的随机字符串。

然后，我们定义另一个函数`generate_secure_pwd`，它简单地不断调用`generate_pwd`，直到我们得到的随机字符串符合要求，这些要求非常简单。密码必须至少有一个小写字符，`upper`个大写字符，`digits`个数字，和`length`长度。

在我们进入`while`循环之前，值得注意的是，如果我们将要求（大写、小写和数字）相加，而这个和大于密码的总长度，那么我们永远无法在循环内满足条件。因此，为了避免陷入无限循环，我在主体的第一行放了一个检查子句，并在需要时引发`ValueError`。你能想到如何为这种边缘情况编写测试吗？

`while`循环的主体很简单：首先我们生成随机密码，然后我们使用`any`和`sum`来验证条件。`any`如果可迭代的项目中有任何一个评估为`True`，则返回`True`。在这里，使用 sum 实际上稍微棘手一些，因为它利用了多态性。在继续阅读之前，你能看出我在说什么吗？

嗯，这很简单：在 Python 中，`True`和`False`是整数数字的子类，因此在`True`/`False`值的可迭代上求和时，它们将自动被`sum`函数解释为整数。这被称为**多态性**，我们在第六章中简要讨论过，*OOP，装饰器和迭代器*。

运行示例会产生以下结果：

```py
$ python secr_gen.py
nsL5voJnCi7Ote3F
J5e
```

第二个密码可能不太安全...

在我们进入下一个模块之前，最后一个例子。让我们生成一个重置密码的 URL：

```py
# secrs/secr_reset.py
import secrets

def get_reset_pwd_url(token_length=16):
    token = secrets.token_urlsafe(token_length)
    return f'https://fabdomain.com/reset-pwd/{token}'

print(get_reset_pwd_url())
```

这个函数非常简单，我只会向你展示输出：

```py
$ python secr_reset.py
https://fabdomain.com/reset-pwd/m4jb7aKgzTGuyjs9lTIspw
```

# 摘要比较

这可能相当令人惊讶，但在`secrets`中，您可以找到`compare_digest(a, b)`函数，它相当于通过简单地执行`a == b`来比较两个摘要。那么，为什么我们需要该函数呢？因为它旨在防止时序攻击。这种攻击可以根据比较失败所需的时间推断出两个摘要开始不同的位置。因此，`compare_digest`通过消除时间和失败之间的相关性来防止此类攻击。我认为这是一个很好的例子，说明了攻击方法可以有多么复杂。如果您因惊讶而挑起了眉毛，也许现在我说过永远不要自己实现加密函数的原因更加清楚了。

就是这样！现在，让我们来看看`hmac`。

# HMAC

该模块实现了 HMAC 算法，如 RFC 2104 所述（[`tools.ietf.org/html/rfc2104.html`](https://tools.ietf.org/html/rfc2104.html)）。由于它非常小，但仍然很重要，我将为您提供一个简单的示例：

```py
# hmc.py
import hmac
import hashlib

def calc_digest(key, message):
    key = bytes(key, 'utf-8')
    message = bytes(message, 'utf-8')
    dig = hmac.new(key, message, hashlib.sha256)
    return dig.hexdigest()

digest = calc_digest('secret-key', 'Important Message')
```

正如您所看到的，接口始终是相同或相似的。我们首先将密钥和消息转换为字节，然后创建一个`digest`实例，我们将使用它来获取哈希的十六进制表示。没有什么别的可说的，但我还是想添加这个模块，以保持完整性。

现在，让我们转向不同类型的令牌：JWT。

# JSON Web Tokens

**JSON Web Token**，或**JWT**，是用于创建断言某些声明的令牌的基于 JSON 的开放标准。您可以在网站上了解有关此技术的所有信息（[`jwt.io/`](https://jwt.io/)）。简而言之，这种类型的令牌由三个部分组成，用点分隔，格式为*A.B.C*。*B*是有效载荷，其中我们放置数据和声明。*C*是签名，用于验证令牌的有效性，*A*是用于计算签名的算法。*A*、*B*和*C*都使用 URL 安全的 Base64 编码（我将其称为 Base64URL）进行编码。

Base64 是一种非常流行的二进制到文本编码方案，它通过将二进制数据转换为基 64 表示形式来以 ASCII 字符串格式表示二进制数据。基 64 表示法使用字母*A-Z*、*a-z*和数字*0-9*，再加上两个符号*+*和*/*，总共共 64 个符号。因此，毫不奇怪，Base64 字母表由这 64 个符号组成。例如，Base64 用于编码电子邮件中附加的图像。这一切都是无缝进行的，因此绝大多数人完全不知道这一事实。

JWT 使用 Base64URL 进行编码的原因是因为在 URL 上下文中，字符`+`和`/`分别表示空格和路径分隔符。因此，在 URL 安全版本中，它们被替换为`-`和`_`。此外，任何填充字符（`=`），通常在 Base64 中使用，都被删除，因为在 URL 中它也具有特定含义。

因此，这种类型的令牌的工作方式与我们在处理哈希时习惯的方式略有不同。实际上，令牌携带的信息始终是可见的。您只需要解码*A*和*B*以获取算法和有效载荷。但是，安全性部分在于*C*，它是令牌的 HMAC 哈希。如果您尝试通过编辑有效载荷，将其重新编码为 Base64，并替换令牌中的有效载荷，那么签名将不再匹配，因此令牌将无效。

这意味着我们可以构建一个带有声明的有效载荷，例如*作为管理员登录*，或类似的内容，只要令牌有效，我们就知道我们可以信任该用户实际上是作为管理员登录的。

处理 JWT 时，您希望确保已经研究了如何安全处理它们。诸如不接受未签名的令牌，或限制您用于编码和解码的算法列表，以及其他安全措施等事项非常重要，您应该花时间调查和学习它们。

对于代码的这一部分，您需要安装`PyJWT`和`cryptography` Python 包。与往常一样，您将在本书源代码的要求中找到它们。

让我们从一个简单的例子开始：

```py
# tok.py
import jwt

data = {'payload': 'data', 'id': 123456789}

token = jwt.encode(data, 'secret-key')
data_out = jwt.decode(token, 'secret-key')
print(token)
print(data_out)
```

我们定义了包含 ID 和一些有效载荷数据的`data`有效载荷。然后，我们使用`jwt.encode`函数创建一个令牌，该函数至少需要有效载荷和一个用于计算签名的秘钥。用于计算令牌的默认算法是`HS256`。让我们看一下输出：

```py
$ python tok.py
b'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJwYXlsb2FkIjoiZGF0YSIsImlkIjoxMjM0NTY3ODl9.WFRY-uoACMoNYX97PXXjEfXFQO1rCyFCyiwxzOVMn40'
{'payload': 'data', 'id': 123456789}
```

因此，正如您所看到的，令牌是 Base64URL 编码的数据片段的二进制字符串。我们调用了`jwt.decode`，提供了正确的秘钥。如果我们做了其他操作，解码将会失败。

有时，您可能希望能够检查令牌的内容而不进行验证。您可以通过简单地调用`decode`来实现：

```py
# tok.py
jwt.decode(token, verify=False)
```

例如，当需要使用令牌有效载荷中的值来恢复秘钥时，这是很有用的，但是这种技术相当高级，所以在这种情况下我不会花时间讨论它。相反，让我们看看如何指定一个不同的算法来计算签名：

```py
# tok.py
token512 = jwt.encode(data, 'secret-key', algorithm='HS512')
data_out = jwt.decode(token512, 'secret-key', algorithm='HS512')
print(data_out)
```

输出是我们的原始有效载荷字典。如果您想在解码阶段允许多个算法，您甚至可以指定一个算法列表，而不仅仅是一个。

现在，虽然您可以在令牌有效载荷中放入任何您想要的内容，但有一些声明已经被标准化，并且它们使您能够对令牌有很大的控制权。

# 已注册的声明

在撰写本书时，这些是已注册的声明：

+   `iss`：令牌的*发行者*

+   `sub`：关于此令牌所携带信息的*主题*信息

+   `aud`：令牌的*受众*

+   `exp`：*过期时间*，在此时间之后，令牌被视为无效

+   `nbf`：*不早于（时间）*，或者在此时间之前，令牌被视为尚未有效

+   `iat`：令牌*发行*的时间

+   `jti`：令牌*ID*

声明也可以被归类为公共或私有：

+   **私有**：由 JWT 的用户（消费者和生产者）定义的声明。换句话说，这些是用于特定情况的临时声明。因此，必须小心防止碰撞。

+   **公共**：是在 IANA JSON Web Token 声明注册表中注册的声明（用户可以在其中注册他们的声明，从而防止碰撞），或者使用具有碰撞抵抗名称的名称（例如，通过在其名称前加上命名空间）。

要了解有关声明的所有内容，请参考官方网站。现在，让我们看一些涉及这些声明子集的代码示例。

# 与时间相关的声明

让我们看看如何使用与时间相关的声明：

```py
# claims_time.py
from datetime import datetime, timedelta
from time import sleep
import jwt

iat = datetime.utcnow()
nfb = iat + timedelta(seconds=1)
exp = iat + timedelta(seconds=3)
data = {'payload': 'data', 'nbf': nfb, 'exp': exp, 'iat': iat}

def decode(token, secret):
    print(datetime.utcnow().time().isoformat())
    try:
        print(jwt.decode(token, secret))
    except (
        jwt.ImmatureSignatureError, jwt.ExpiredSignatureError
    ) as err:
        print(err)
        print(type(err))

secret = 'secret-key'
token = jwt.encode(data, secret)

decode(token, secret)
sleep(2)
decode(token, secret)
sleep(2)
decode(token, secret)
```

在此示例中，我们将`iat`声明设置为当前的 UTC 时间（**UTC**代表**协调世界时**）。然后，我们将`nbf`和`exp`设置为分别从现在开始的`1`和`3`秒。然后，我们定义了一个解码辅助函数，它会对尚未有效或已过期的令牌做出反应，通过捕获适当的异常，然后我们调用它三次，中间隔着两次调用睡眠。这样，我们将尝试在令牌尚未有效时解码它，然后在它有效时解码，最后在它已经过期时解码。此函数还在尝试解密之前打印了一个有用的时间戳。让我们看看它是如何执行的（为了可读性已添加了空行）：

```py
$ python claims_time.py
14:04:13.469778
The token is not yet valid (nbf)
<class 'jwt.exceptions.ImmatureSignatureError'>

14:04:15.475362
{'payload': 'data', 'nbf': 1522591454, 'exp': 1522591456, 'iat': 1522591453}

14:04:17.476948
Signature has expired
<class 'jwt.exceptions.ExpiredSignatureError'>
```

正如您所看到的，一切都如预期执行。我们从异常中得到了很好的描述性消息，并且在令牌实际有效时得到了原始有效载荷。

# 与认证相关的声明

让我们看另一个涉及发行者（`iss`）和受众（`aud`）声明的快速示例。代码在概念上与上一个示例非常相似，我们将以相同的方式进行练习：

```py
# claims_auth.py
import jwt

data = {'payload': 'data', 'iss': 'fab', 'aud': 'learn-python'}
secret = 'secret-key'
token = jwt.encode(data, secret)

def decode(token, secret, issuer=None, audience=None):
    try:
        print(jwt.decode(
            token, secret, issuer=issuer, audience=audience))
    except (
        jwt.InvalidIssuerError, jwt.InvalidAudienceError
    ) as err:
        print(err)
        print(type(err))

decode(token, secret)
# not providing the issuer won't break
decode(token, secret, audience='learn-python')
# not providing the audience will break
decode(token, secret, issuer='fab')
# both will break
decode(token, secret, issuer='wrong', audience='learn-python')
decode(token, secret, issuer='fab', audience='wrong')

decode(token, secret, issuer='fab', audience='learn-python')
```

正如您所看到的，这一次我们指定了`issuer`和`audience`。事实证明，如果我们在解码令牌时不提供发行者，它不会导致解码失败。但是，提供错误的发行者将导致解码失败。另一方面，未提供受众，或提供错误的受众，都将导致解码失败。

与上一个示例一样，我编写了一个自定义解码函数，以响应适当的异常。看看您是否能跟上调用和随后的输出（我会在一些空行上帮助）：

```py
$ python claims_auth.py
Invalid audience
<class 'jwt.exceptions.InvalidAudienceError'>

{'payload': 'data', 'iss': 'fab', 'aud': 'learn-python'}

Invalid audience
<class 'jwt.exceptions.InvalidAudienceError'>

Invalid issuer
<class 'jwt.exceptions.InvalidIssuerError'>

Invalid audience
<class 'jwt.exceptions.InvalidAudienceError'>

{'payload': 'data', 'iss': 'fab', 'aud': 'learn-python'}
```

现在，让我们看一个更复杂的用例的最后一个例子。

# 使用非对称（公钥）算法

有时，使用共享密钥并不是最佳选择。在这种情况下，采用不同的技术可能会很有用。在这个例子中，我们将使用一对 RSA 密钥创建一个令牌（并解码它）。

公钥密码学，或非对称密码学，是使用公钥（可以广泛传播）和私钥（只有所有者知道）的密钥对的任何加密系统。如果您有兴趣了解更多关于这个主题的内容，请参阅本章末尾的推荐书目。

现在，让我们创建两对密钥。一对将没有密码，另一对将有密码。为了创建它们，我将使用 OpenSSH 的`ssh-keygen`工具（[`www.ssh.com/ssh/keygen/`](https://www.ssh.com/ssh/keygen/)）。在我为本章编写脚本的文件夹中，我创建了一个`rsa`子文件夹。在其中，运行以下命令：

```py
$ ssh-keygen -t rsa
```

将路径命名为`key`（它将保存在当前文件夹中），并在要求密码时简单地按下*Enter*键。完成后，再做一次相同的操作，但这次使用`keypwd`作为密钥的名称，并给它设置一个密码。我选择的密码是经典的`Password123`。完成后，切换回`ch9`文件夹，并运行以下代码：

```py
# token_rsa.py
import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

data = {'payload': 'data'}

def encode(data, priv_filename, priv_pwd=None, algorithm='RS256'):
    with open(priv_filename, 'rb') as key:
        private_key = serialization.load_pem_private_key(
            key.read(),
            password=priv_pwd,
            backend=default_backend()
        )
    return jwt.encode(data, private_key, algorithm=algorithm)

def decode(data, pub_filename, algorithm='RS256'):
    with open(pub_filename, 'rb') as key:
        public_key = key.read()
    return jwt.decode(data, public_key, algorithm=algorithm)

# no pwd
token = encode(data, 'rsa/key')
data_out = decode(token, 'rsa/key.pub')
print(data_out)

# with pwd
token = encode(data, 'rsa/keypwd', priv_pwd=b'Password123')
data_out = decode(token, 'rsa/keypwd.pub')
print(data_out)
```

在上一个示例中，我们定义了一对自定义函数来使用私钥/公钥对编码和解码令牌。正如您在`encode`函数的签名中所看到的，这次我们使用了`RS256`算法。我们需要使用特殊的`load_pem_private_key`函数打开私钥文件，该函数允许我们指定内容、密码和后端。`.pem`是我们的密钥创建的格式的名称。如果您查看这些文件，您可能会认出它们，因为它们非常流行。

逻辑非常简单，我鼓励您至少考虑一个使用这种技术可能比使用共享密钥更合适的用例。

# 有用的参考资料

在这里，您可以找到一些有用的参考资料，如果您想深入了解密码学的迷人世界：

+   密码学：[`en.wikipedia.org/wiki/Cryptography`](https://en.wikipedia.org/wiki/Cryptography)

+   JSON Web Tokens：[`jwt.io`](https://jwt.io)

+   哈希函数：[`en.wikipedia.org/wiki/Cryptographic_hash_function`](https://en.wikipedia.org/wiki/Cryptographic_hash_function)

+   HMAC：[`en.wikipedia.org/wiki/HMAC`](https://en.wikipedia.org/wiki/HMAC)

+   密码学服务（Python STD 库）：[`docs.python.org/3.7/library/crypto.html`](https://docs.python.org/3.7/library/crypto.html)

+   IANA JSON Web Token Claims Registry：[`www.iana.org/assignments/jwt/jwt.xhtml`](https://www.iana.org/assignments/jwt/jwt.xhtml)

+   PyJWT 库：[`pyjwt.readthedocs.io/`](https://pyjwt.readthedocs.io/)

+   密码学库：[`cryptography.io/`](https://cryptography.io/)

网络上还有更多内容，还有很多书籍可以学习，但我建议您从主要概念开始，然后逐渐深入研究您想更全面了解的具体内容。

# 总结

在这一短章中，我们探索了 Python 标准库中的密码学世界。我们学会了如何使用不同的密码学函数为消息创建哈希（或摘要）。我们还学会了如何在密码学上下文中创建令牌并处理随机数据。

然后，我们在标准库之外进行了小小的探索，了解了 JSON Web 令牌，这在现代系统和应用程序中的认证和声明相关功能中被广泛使用。

最重要的是要明白，在涉及密码学时，手动操作可能非常危险，因此最好还是把它交给专业人士，简单地使用我们现有的工具。

下一章将完全关于摆脱单行软件执行。我们将学习软件在现实世界中的运行方式，探索并发执行，并了解 Python 提供给我们的线程、进程和工具，以便同时执行*多项任务*，可以这么说。
