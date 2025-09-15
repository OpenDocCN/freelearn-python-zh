# 9

# 密码学和令牌

> “三个可以保守一个秘密，如果其中两个已经死了。”
> 
> ——本杰明·富兰克林，《穷理查年鉴》

在这一简短的章节中，我们将为您简要概述 Python 标准库提供的加密服务。我们还将涉及 JSON Web Tokens，这是一种用于在双方之间安全表示声明的有趣标准。

我们将探讨以下内容：

+   Hashlib

+   HMAC

+   秘密

+   使用 PyJWT 处理 JSON Web Tokens，这似乎是处理 JWT 最流行的 Python 库

让我们先花一点时间来谈谈密码学及其为什么如此重要。

# 密码学的必要性

据估计，截至 2024 年，全球大约有 53.5 亿到 54.4 亿人使用互联网。每年，使用在线银行服务、在线购物或只是在社交媒体上与朋友和家人交谈的人数都在增加。所有这些人期望他们的钱是安全的，他们的交易是安全的，他们的对话是私密的。

因此，如果您是应用程序开发者，您必须非常重视安全性。无论您的应用程序有多小或有多不重要：安全性始终应该是您关注的焦点。

信息技术的安全性是通过采用多种不同手段实现的，但迄今为止，最重要的手段是密码学。您使用计算机或手机做的几乎所有事情都应该包含一个密码学层。例如，密码学用于确保在线支付的安全，以在网络中传输消息，即使有人截获它们，也无法读取它们，以及当您在云中备份文件时加密您的文件。

本章的目的不是教您所有密码学的复杂性——有整本书是专门讨论这个主题的。相反，我们将向您展示如何使用 Python 提供的工具来创建摘要、令牌，并在需要实现与密码学相关的内容时，总体上确保更安全。在阅读本章时，请注意，密码学远不止加密和解密数据；实际上，在整个章节中，您将找不到任何加密或解密的示例！

## 有用的指南

总是记住以下规则：不要尝试创建自己的哈希或加密函数。简单地说，不要这样做。使用已经存在的工具和函数。发明一个良好、坚实、健壮的算法来进行哈希或加密是非常困难的，因此最好将其留给专业密码学家。

理解密码学很重要，所以尽量多了解这个主题。网上有大量信息，但为了您的方便，我们将在本章末尾提供一些有用的参考资料。

现在，让我们深入探讨我们想要向您展示的第一个标准库模块：`hashlib`。

# Hashlib

此模块提供了对各种加密哈希算法的访问。这些是数学函数，可以接受任何大小的消息并产生一个固定大小的结果，该结果被称为**哈希**或**摘要**。加密哈希有许多用途，从验证数据完整性到安全存储和验证密码。

理想情况下，加密哈希算法应该是：

+   **确定性**：相同的消息应该总是产生相同的哈希值。

+   **不可逆性**：从哈希值中确定原始消息应该是不可行的。

+   **抗碰撞性**：找到两个不同的消息，它们产生相同的哈希值应该是困难的。

这些属性对于哈希的安全应用至关重要。例如，将密码仅以哈希形式存储被认为是强制性的。

不可逆属性确保即使发生数据泄露，攻击者掌握了密码数据库，他们也无法获取原始密码。将密码仅存储为哈希意味着验证用户登录时密码的唯一方法是计算他们提供的密码的哈希值，并将其与存储的哈希值进行比较。当然，如果哈希算法不是确定的，这将不起作用。

抗碰撞性也很重要。它确保数据完整性，因为如果哈希用于为数据提供指纹，那么当数据发生变化时，指纹也应该发生变化。抗碰撞性防止攻击者用一个具有相同哈希值的不同文档替换文档。此外，许多安全协议依赖于抗碰撞哈希函数保证的唯一性。

通过`hashlib`可用的算法集取决于您平台使用的底层库。然而，某些算法在所有系统中都是保证存在的。让我们看看如何找出可用的内容（请注意，您的结果可能与我们的不同）：

```py
# hlib.txt
>>> import hashlib
>>> hashlib.algorithms_available
{'sha3_256', 'sha224', 'blake2b', 'sha512_224', 'ripemd160',
 'sha1', 'sha512_256', 'sha3_512', 'sha512', 'sha384', 'sha3_384',
'sha3_224', 'shake_256', 'shake_128', 'sm3', 'md5-sha1', 'sha256',
'md5', 'blake2s'}
>>> hashlib.algorithms_guaranteed
{'sha512', 'sha3_256', 'shake_128', 'sha224', 'blake2b',
 'shake_256', 'sha384', 'sha1', 'sha3_512', 'sha3_384', 'sha256',
 'sha3_224', 'md5', 'blake2s'} 
```

通过打开 Python shell，我们可以获取我们系统可用的算法集。如果我们的应用程序与第三方应用程序通信，最好从保证的集合中选择一个算法，因为这意味着每个平台都支持它们。注意，其中许多以*sha*开头，代表*安全哈希算法*。

让我们在同一个 shell 中继续；我们将为字节字符串`b"Hash me now!"`创建一个哈希：

```py
>>> h = hashlib.blake2b()
>>> h.update(b"Hash me")
>>> h.update(b" now!")
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

在这里，我们使用了`blake2b()`加密函数，它相当复杂，是在 Python 3.6 中添加的。在创建哈希对象`h`之后，我们分两步更新其消息。虽然我们不需要这样做，但有时我们需要对一次不可用的数据进行哈希处理，因此了解我们可以分步骤进行是很好的。

一旦我们添加了整个消息，我们就得到了摘要的十六进制表示。这将使用每个字节两个字符（因为每个字符代表四个位，即半个字节）。我们还得到了摘要的字节表示，然后我们检查其细节：它有一个块大小（散列算法的内部块大小，以字节为单位）为 128 字节，摘要大小（生成的散列大小，以字节为单位）为 64 字节，以及一个名称。

让我们看看如果我们使用`sha512()`而不是`blake2b()`函数，我们会得到什么：

```py
>>> hashlib.sha512(b"Hash me too!").hexdigest()
'a0d169ac9487fc6c78c7db64b54aefd01bd245bbd1b90b6fe5648c3c4eb0ea7d'
'93e1be50127164f21bc8ddb3dd45a6b4306dfe9209f2677518259502fed27686' 
```

生成的散列与`blake2b`的长度相同。请注意，我们可以用一行代码构造散列对象并计算摘要。

散列是一个有趣的话题，当然，我们迄今为止看到的简单示例只是开始。`blake2b()`函数由于可以调整的许多参数，为我们提供了很大的灵活性。这意味着它可以适应不同的应用程序或调整以防止特定类型的攻击。

在这里，我们将简要讨论这些参数中的一个；对于完整细节，请参阅官方文档[`docs.python.org/3/library/hashlib.html`](https://docs.python.org/3/library/hashlib.html)。`person`参数非常有趣。它用于**个性化**散列，迫使它对相同的消息产生不同的摘要。这有助于提高在同一个应用程序中为不同目的使用相同散列函数时的安全性：

```py
>>> import hashlib
>>> h1 = hashlib.blake2b(
...    b"Important data", digest_size=16, person=b"part-1")
>>> h2 = hashlib.blake2b(
...    b"Important data", digest_size=16, person=b"part-2")
>>> h3 = hashlib.blake2b(
...    b"Important data", digest_size=16)
>>> h1.hexdigest()
'c06b9af95d5aa6307e7e3fd025a15646'
>>> h2.hexdigest()
'9cb03be8f3114d0f06bddaedce2079c4'
>>> h3.hexdigest()
'7d35308ca3b042b5184728d2b1283d0d' 
```

在这里，我们还使用了`digest_size`参数来获取只有 16 字节长的散列。

通用散列函数，如`blake2b()`或`sha512()`，不适合安全存储密码。通用散列函数在现代计算机上计算速度很快，这使得攻击者可以通过**暴力破解**（每秒尝试数百万种可能性，直到找到匹配项）来反转散列。像`pbkdf2_hmac()`这样的密钥派生算法被设计得足够慢，以使这种暴力破解攻击变得不可行。`pbkdf2_hmac()`密钥派生算法通过使用许多重复应用通用散列函数（迭代次数可以作为参数指定）来实现这一点。随着计算机变得越来越强大，随着时间的推移增加迭代次数变得很重要；否则，随着时间的推移，对我们数据进行暴力破解攻击成功的可能性会增加。

好的密码散列函数也应该使用**盐**。盐是一段随机数据，用于初始化散列函数；这随机化了算法的输出，并保护了针对已知散列表的攻击。`pbkdf2_hmac()`函数通过必需的`salt`参数支持盐的使用。

这是您可以使用`pbkdf2_hmac()`来散列密码的方法：

```py
>>> import os
>>> dk = hashlib.pbkdf2_hmac("sha256", b"password123",
...     salt=os.urandom(16), iterations=200000)
>>> dk.hex()
'ac34579350cf6d05e01e745eb403fc50ac0e62fbeb553cbb895e834a77c37aed' 
```

注意，我们使用了`os.urandom()`来提供一个 16 字节的随机盐，正如文档中建议的那样。

通常，盐的值与哈希值一起存储。当用户尝试登录时，您的程序使用存储的盐来创建给定密码的哈希值，然后将其与存储的哈希值进行比较。使用相同的盐值确保当密码正确时，哈希值将是相同的。

我们鼓励您探索和实验这个模块，因为最终您将不得不使用它。现在，让我们继续讨论`hmac`模块。

# HMAC

此模块实现了 RFC 2104（[`datatracker.ietf.org/doc/html/rfc2104.html`](https://datatracker.ietf.org/doc/html/rfc2104.html)）中描述的**HMAC**算法。HMAC（根据询问对象的不同，代表**基于哈希的消息认证码**或**密钥哈希消息认证码**）是用于验证消息和验证它们没有被篡改的广泛使用机制。

该算法将消息与密钥结合并生成组合的哈希值。这个哈希值被称为**消息认证码**（**MAC**）或**签名**。签名与消息一起存储或传输。您可以通过使用相同的密钥重新计算签名并与之前计算的签名进行比较来验证消息没有被篡改。密钥必须被仔细保护；否则，能够访问密钥的攻击者将能够修改消息并替换签名，从而破坏认证机制。

让我们看看如何计算 MAC 的一个小例子：

```py
# hmc.py
import hmac
import hashlib
def calc_digest(key, message):
    key = bytes(key, "utf-8")
    message = bytes(message, "utf-8")
    dig = hmac.new(key, message, hashlib.sha256)
    return dig.hexdigest()
mac = calc_digest("secret-key", "Important Message") 
```

`hmac.new()`函数接受一个密钥、一个消息和要使用的哈希算法。它返回一个`hmac`对象，该对象具有与`hashlib`中的哈希对象类似的接口。密钥必须是一个`bytes`或`bytearray`对象，而消息可以是任何`bytes`-like 对象。因此，我们在创建`hmac`实例（`dig`）之前将我们的`key`和`message`转换为字节，我们使用它来获取哈希值的十六进制表示。

我们将在本章后面讨论 JWT 时，更详细地了解如何使用 HMAC 签名。在此之前，我们将快速查看`secrets`模块。

# 秘密

这个小模块是在 Python 3.6 中添加的，处理三件事：随机数、令牌和摘要比较。它使用底层操作系统提供的最安全的随机数生成器来生成适合在加密应用中使用令牌和随机数。让我们快速看看它提供了什么。

## 随机对象

我们可以使用三个函数来生成随机对象：

```py
# secrs/secr_rand.py
import secrets
print(secrets.choice("Choose one of these words".split()))
print(secrets.randbelow(10**6))
print(secrets.randbits(32)) 
```

第一个函数`choice()`从非空序列中随机返回一个元素。第二个函数`randbelow()`生成介于 0 和您调用它的参数之间的随机整数，第三个函数`randbits()`生成包含给定数量随机位的整数。运行此代码将产生以下输出（当然，每次运行都会不同）：

```py
$ python secr_rand.py
one
133025
1509555468 
```

当你在密码学环境中需要随机性时，你应该使用这些函数而不是`random`模块中的那些函数，因为这些函数是专门为此任务设计的。让我们看看这个模块为令牌提供了什么。

## 令牌生成

再次，我们有三个用于生成令牌的函数，每个函数的格式都不同。让我们看看示例：

```py
# secrs/secr_rand.py
import secrets
print(secrets.token_bytes(16))
print(secrets.token_hex(32))
print(secrets.token_urlsafe(32)) 
```

`token_bytes()`函数简单地返回一个包含指定字节数的随机字节字符串（在这个例子中是 16 个字节）。其他两个函数做的是同样的事情，但`token_hex()`返回一个以十六进制格式表示的令牌，而`token_urlsafe()`返回一个只包含适合包含在 URL 中的字符的令牌。以下是输出（这是前一个运行的延续）：

```py
b'\x0f\x8b\x8f\x0f\xe3\xceJ\xbc\x18\xf2\x1e\xe0i\xee1\x99'
98e80cddf6c371811318045672399b0950b8e3207d18b50d99d724d31d17f0a7
63eNkRalj8dgZqmkezjbEYoGddVcutgvwJthSLf5kho 
```

让我们看看我们如何使用这些工具来编写我们自己的随机密码生成器：

```py
# secrs/secr_gen.py
import secrets
from string import digits, ascii_letters
def generate_pwd(length=8):
    chars = digits + ascii_letters
    return "".join(secrets.choice(chars) for c in range(length))
def generate_secure_pwd(length=16, upper=3, digits=3):
    if length < upper + digits + 1:
        raise ValueError("Nice try!")
    while True:
        pwd = generate_pwd(length)
        if (
            any(c.islower() for c in pwd)
            and sum(c.isupper() for c in pwd) >= upper
            and sum(c.isdigit() for c in pwd) >= digits
        ):
            return pwd
print(generate_secure_pwd())
print(generate_secure_pwd(length=3, upper=1, digits=1)) 
```

`generate_pwd()`函数简单地通过将随机从包含所有字母（小写和大写）和 10 个十进制数字的字符串中随机选择的`length`个字符连接起来，生成指定长度的随机字符串。

然后，我们定义另一个函数，`generate_secure_pwd()`，它简单地不断调用`generate_pwd()`，直到我们得到的随机字符串符合一些基本要求。密码必须是`length`个字符长，至少包含一个小写字母，`upper`个大写字母，以及`digits`个数字。

如果参数指定的总大写字母、小写字母和数字的数量大于我们正在生成的密码的长度，我们就永远无法满足条件。我们在循环开始之前检查这一点，如果给定的参数会导致无限循环，则引发`ValueError`。

`while`循环的主体很简单：首先，我们生成随机密码，然后使用`any()`和`sum()`来验证条件。`any()`函数在其调用的可迭代对象中的任何项评估为`True`时返回`True`。`sum()`的使用实际上在这里稍微复杂一些，因为它利用了**多态性**。如您从*第二章*，*内置数据类型*中回忆的那样，`bool`类型是`int`的子类，因此当对`True`和`False`值的可迭代对象求和时，它们将被`sum()`函数自动解释为整数（值为 1 和 0）。这是一个多态性的例子，我们在*第六章*，*面向对象编程、装饰器和迭代器*中简要讨论了它。

运行示例产生以下结果：

```py
$ python secr_gen.py
mgQ3Hj57KjD1LI7M
b8G 
```

当然，你不会想使用长度为 3 的密码。

随机令牌的一个常见用途是在网站的密码重置 URL 中。以下是如何生成此类 URL 的示例：

```py
# secrs/secr_reset.py
import secrets
def get_reset_pwd_url(token_length=16):
    token = secrets.token_urlsafe(token_length)
    return f"https://example.com/reset-pwd/{token}"
print(get_reset_pwd_url()) 
```

运行上述代码产生了以下输出：

```py
$ python secr_reset.py
https://example.com/reset-pwd/ML_6_2wxDpXmDJLHrDnrRA 
```

## 摘要比较

这可能相当令人惊讶，但 `secrets` 模块还提供了一个 `compare_digest(a, b)` 函数，这是通过简单地执行 `a == b` 来比较两个摘要的等效函数。那么，我们为什么需要这个函数呢？因为它被设计用来防止时间攻击。这类攻击可以根据比较失败所需的时间推断出两个摘要开始不同的信息。因此，`compare_digest()` 通过消除时间和失败之间的相关性来防止这种攻击。我们认为这是一个复杂的攻击方法如何复杂的绝佳例子。如果您惊讶地扬起了眉毛，现在可能更清楚为什么我们说永远不要自己实现加密函数。

这使我们到达了 Python 标准库中加密服务的游览结束。现在，让我们继续探讨另一种类型的令牌：JWT。

# JSON Web Tokens

**JSON Web Token**（JWT），是一个基于 JSON 的开放标准，用于创建断言一系列 **声明** 的令牌。JWT 经常用作身份验证令牌。在这种情况下，声明通常是关于已验证用户身份和权限的陈述。这些令牌是经过加密签名的，这使得验证令牌内容自签发以来未被修改成为可能。您可以在网站上了解所有关于这项技术的信息（[`jwt.io`](https://jwt.io)）。

这种类型的令牌由三个部分组成，通过点连接在一起，格式为 *A.B.C* 。*B* 是有效载荷，这是我们放置声明的地方。*C* 是签名，用于验证令牌的有效性，而 *A* 是头部，它标识令牌为 JWT，并指示计算签名的算法。*A*、*B* 和 *C* 都使用 URL 安全的 Base64 编码（我们将其称为 Base64URL）。Base64URL 编码使得 JWT 可以作为 URL 的一部分使用（通常作为查询参数）；然而，JWT 也会出现在其他地方，包括 HTTP 头部。

`Base64` 是一种流行的二进制到文本编码方案，通过将其转换为 Radix-64 表示来以 ASCII 字符串格式表示二进制数据。Radix-64 表示使用字母 A-Z、a-z 和数字 0-9，以及两个符号 + 和 /，总共 64 个符号。`Base64` 用于例如编码电子邮件中附加的图像。它无缝发生，所以绝大多数用户对此一无所知。`Base64URL` 是 `Base64` 编码的一个变体，其中将具有特定 URL 上下文意义的 + 和 / 字符替换为 - 和 _。用于 `Base64` 填充的 = 字符在 URL 中也有特殊含义，并在 `Base64URL` 中省略。

这种类型的令牌的工作方式与我们在本章中迄今为止所看到的不同。事实上，令牌携带的信息始终是可见的。你只需要将 *A* 和 *B* 从 Base64URL 解码，以获取算法和有效载荷。安全性部分在于 *C* ，它是头部和有效载荷的 HMAC 签名。如果你尝试通过编辑头部或有效载荷来修改 *A* 或 *B* 部分，将其编码回 Base64URL，并在令牌中替换它，签名将不会匹配，因此令牌将无效。

这意味着我们可以构建包含诸如 *已登录为管理员* 或类似内容的有效载荷，只要令牌有效，我们就知道我们可以信任该用户已登录为管理员。

在处理 JWT 时，你想要确保你已经研究了如何安全地处理它们。像不接受未签名令牌或限制你用于编码和解码的算法列表，以及其他安全措施，这些都非常重要，你应该花时间调查和学习它们。

对于这段代码，你必须安装 `PyJWT` 和 `cryptography` Python 包。像往常一样，你可以在本章源代码的要求中找到它们。

让我们从简单的例子开始：

```py
# jwt/tok.py
import jwt
data = {"payload": "data", "id": 123456789}
algs = ["HS256", "HS512"]
token = jwt.encode(data, "secret-key")
data_out = jwt.decode(token, "secret-key", algorithms=algs)
print(token)
print(data_out) 
```

我们定义了包含 ID 和一些有效载荷数据的 `data` 有效载荷。我们使用 `jwt.encode()` 函数创建令牌，该函数接受有效载荷和一个密钥。密钥用于生成令牌头部和有效载荷的 HMAC 签名。接下来，我们再次解码令牌，指定我们愿意接受的签名算法。默认的算法用于计算令牌是 `HS256`；在这个例子中，我们在解码时接受 `HS256` 或 `HS512`（如果令牌是使用不同的算法生成的，它将被异常拒绝）。以下是输出：

```py
$ python jwt/tok.py
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJwYXlsb2FkIjoiZGF0YSIsIm...
{'payload': 'data', 'id': 123456789} 
```

如你所见，令牌是由 Base64URL 编码的数据片段组成的二进制字符串（为了适应一行而进行了缩写）。我们调用 `jwt.decode()` ，提供正确的密钥。如果我们提供了错误的密钥，我们会得到一个错误，因为签名只能用生成它的相同密钥进行验证。

JWT 通常用于在双方之间传输信息。例如，允许网站依赖第三方身份提供者来验证用户的身份验证协议通常使用 JWT。在这种情况下，用于签名令牌的密钥需要在双方之间共享。因此，它通常被称为 **共享密钥** 。

必须小心保护共享密钥，因为任何有权访问它的人都可以生成有效的令牌。

有时，你可能希望在验证签名之前先检查令牌的内容。你可以通过简单地调用 `decode()` 来这样做：

```py
# jwt/tok.py
jwt.decode(token, options={"verify_signature": False}) 
```

这很有用，例如，当需要在令牌有效载荷中恢复密钥时，但该技术相当高级，所以我们不会在本上下文中花费时间讨论它。相反，让我们看看我们如何指定用于计算签名的不同算法：

```py
# jwt/tok.py
token512 = jwt.encode(data, "secret-key", algorithm="HS512")
data_out = jwt.decode(
    token512, "secret-key", algorithms=["HS512"]
)
print(data_out) 
```

在这里，我们使用了`HS512`算法生成令牌，并在解码时指定我们只接受使用`HS512`算法生成的令牌。输出是原始的有效载荷字典。

现在，虽然你可以自由地将任何内容放入令牌有效载荷中，但有一些声明已被标准化；它们对于确保不同系统和应用程序之间的安全性、一致性和互操作性至关重要。

## 已注册的声明

JWT 标准定义了以下官方**已注册声明**：

+   `iss` : 令牌的发行者

+   `sub` : 关于此令牌所携带信息方的主题信息

+   `aud` : 令牌的受众

+   `exp` : 令牌过期时间，在此时间之后令牌无效

+   `nbf` : 不早于（时间），或令牌尚未被视为有效的之前的时间

+   `iat` : 令牌签发的时间

+   `jti` : 令牌 ID

未在标准中定义的声明可以归类为公共或私人：

+   **公共** : 为特定目的公开分配的声明。公共声明名称可以通过在 IANA JSON Web Token Claims Registry 中注册来保留。或者，声明应以确保它们不会与其他公共或官方声明名称冲突的方式命名（实现这一目标的一种方法是在声明名称前加上已注册的域名）。

+   **私人** : 不属于上述类别的任何其他声明被称为私人声明。此类声明的含义通常在特定应用的上下文中定义，并且在该上下文之外没有意义。为了防止歧义和混淆，必须小心避免名称冲突。

有关声明的信息，请参阅官方网站。现在，让我们看看几个涉及这些声明子集的代码示例。

### 与时间相关的声明

这是我们可能使用与时间相关的声明的方式：

```py
# jwt/claims_time.py
from datetime import datetime, timedelta, UTC
from time import sleep, time
import jwt
iat = datetime.now(tz=UTC)
nfb = iat + timedelta(seconds=1)
exp = iat + timedelta(seconds=3)
data = {"payload": "data", "nbf": nfb, "exp": exp, "iat": iat}
def decode(token, secret):
    print(f"{time():.2f}")
    try:
        print(jwt.decode(token, secret, algorithms=["HS256"]))
    except (
        jwt.ImmatureSignatureError,
        jwt.ExpiredSignatureError,
    ) as err:
        print(err)
        print(type(err))
secret = "secret-key"
token = jwt.encode(data, secret)
decode(token, secret)
sleep(2)
decode(token, secret)
sleep(2)
decode(token, secret) 
```

在本例中，我们将签发时间（`iat`）声明设置为当前的 UTC 时间（**UTC**代表**协调世界时**）。然后，我们将“不早于”（`nbf`）和“过期时间”（`exp`）声明分别设置为现在后的 1 秒和 3 秒。我们定义了一个`decode()`辅助函数，该函数通过捕获适当的异常来响应令牌尚未有效或已过期的行为，然后我们调用它三次，并在两次调用`sleep()`之间进行。

这样，我们将在令牌有效之前尝试解码令牌，然后在它有效时，最后在它过期后。此函数在尝试解码令牌之前还会打印一个有用的时间戳。让我们看看结果如何（为了可读性，添加了空白行）：

```py
$ python jwt/claims_time.py
1716674892.39
The token is not yet valid (nbf)
<class 'jwt.exceptions.ImmatureSignatureError'>
1716674894.39
{'payload': 'data', 'nbf': 1716674893, 'exp': 1716674895, 'iat': 1716674892}
1716674896.39
Signature has expired
<class 'jwt.exceptions.ExpiredSignatureError'> 
```

正如您所看到的，它按预期执行。当令牌有效时，我们从异常中获取描述性消息，并获取原始有效载荷。

### 身份验证相关声明

在这里，我们有一个快速示例，这次涉及到发行者（`iss`）和受众（`aud`）声明。代码在概念上与上一个示例非常相似，我们将以相同的方式练习它：

```py
# jwt/claims_auth.py
import jwt
data = {"payload": "data", "iss": "hein", "aud": "learn-python"}
secret = "secret-key"
token = jwt.encode(data, secret)
def decode(token, secret, issuer=None, audience=None):
    try:
        print(
            jwt.decode(
                token,
                secret,
                issuer=issuer,
                audience=audience,
                algorithms=["HS256"],
            )
        )
    except (
        jwt.InvalidIssuerError,
        jwt.InvalidAudienceError,
    ) as err:
        print(err)
        print(type(err))
# Not providing both the audience and issuer will fail
decode(token, secret)
# Not providing the issuer will succeed
decode(token, secret, audience="learn-python")
# Not providing the audience will fail
decode(token, secret, issuer="hein")
# Both will fail
decode(token, secret, issuer="wrong", audience="learn-python")
decode(token, secret, issuer="hein", audience="wrong")
# This will succeed
decode(token, secret, issuer="hein", audience="learn-python") 
```

正如您所看到的，这次，我们在创建令牌时指定了发行者（`iss`）和受众（`aud`）。即使我们省略了`jwt.decode()`的`issuer`参数，解码此令牌也能成功。然而，如果提供了`issuer`但与令牌中的 iss 字段不匹配，解码将失败。另一方面，如果省略了`audience`参数或与令牌中的`aud`字段不匹配，`jwt.decode()`将失败。

与上一个示例一样，我们编写了一个自定义的`decode()`函数来响应适当的异常。看看您是否能跟上调用和随后的相对输出（我们将用一些空白行来帮助您）：

```py
$ python jwt/claims_time.py
Invalid audience
<class 'jwt.exceptions.InvalidAudienceError'>
{'payload': 'data', 'iss': 'hein', 'aud': 'learn-python'}
Invalid audience
<class 'jwt.exceptions.InvalidAudienceError'>
Invalid issuer
<class 'jwt.exceptions.InvalidIssuerError'>
Audience doesn't match
<class 'jwt.exceptions.InvalidAudienceError'>
{'payload': 'data', 'iss': 'hein', 'aud': 'learn-python'} 
```

注意，在这个例子中，我们改变了`jwt.decode()`的参数，以向您展示各种场景下的行为。然而，在实际应用中，您通常会为`audience`和`issuer`使用固定值，并拒绝任何无法成功解码的令牌。在解码时省略`issuer`意味着您将接受来自任何发行者的令牌。省略`audience`意味着您将只接受未指定受众的令牌。

现在，让我们来看一个更复杂用例的最后一个示例。

## 使用非对称（公钥）算法

有时，使用共享密钥不是最佳选择。在这种情况下，可以使用非对称密钥对而不是 HMAC 来生成 JWT 签名。在这个例子中，我们将使用 RSA 密钥对创建一个令牌（并解码它）。

公钥加密，或非对称加密，是指使用一对密钥的任何加密系统：公钥，可以广泛分发；私钥，只有所有者知道。如果您想了解更多关于这个主题的信息，请参阅本章末尾的建议。可以使用私钥生成签名，而公钥可以用来验证签名。因此，双方可以交换 JWT，并且可以在不共享任何密钥的情况下验证签名。

首先，让我们创建一个 RSA 密钥对。我们将使用 OpenSSH 的`ssh-keygen`实用程序来完成这项工作。[`www.ssh.com/academy/ssh/keygen`](https://www.ssh.com/academy/ssh/keygen) 。在我们的脚本文件夹中，我们创建了一个`jwt/rsa`子文件夹。在其中，运行以下命令：

```py
$ ssh-keygen -t rsa –m PEM 
```

当被要求输入文件名时，请输入`key`（它将保存在当前文件夹中），并在被要求输入口令时简单地按*Enter*键。

生成我们的密钥后，我们现在可以切换回`ch09`文件夹并运行此代码：

```py
# jwt/token_rsa.py
import jwt
data = {"payload": "data"}
def encode(data, priv_filename, algorithm="RS256"):
    with open(priv_filename, "rb") as key:
        private_key = key.read()
    return jwt.encode(data, private_key, algorithm=algorithm)
def decode(data, pub_filename, algorithm="RS256"):
    with open(pub_filename, "rb") as key:
        public_key = key.read()
    return jwt.decode(data, public_key, algorithms=[algorithm])
token = encode(data, "jwt/rsa/key")
data_out = decode(token, "jwt/rsa/key.pub")
print(data_out)  # {'payload': 'data'} 
```

在这个例子中，我们定义了一些自定义函数来使用私钥/公钥对令牌进行编码和解码。正如你在`encode()`函数中看到的，我们这次使用的是`RS256`算法。请注意，当我们编码时，我们提供私钥，用于生成 JWT 签名。当我们解码 JWT 时，我们则提供公钥，用于验证签名。

逻辑很简单，我们鼓励你至少思考一个可能比使用共享密钥更合适的用例。

# 有用参考资料

在这里，如果你想要深入了解迷人的密码学世界，可以找到一份有用的参考资料列表：

+   密码学：[`en.wikipedia.org/wiki/Cryptography`](https://en.wikipedia.org/wiki/Cryptography)

+   JSON Web Tokens：[`jwt.io`](https://jwt.io)

+   JSON Web Tokens 的 RFC 标准：[`datatracker.ietf.org/doc/html/rfc7519`](https://datatracker.ietf.org/doc/html/rfc7519)

+   哈希函数：[`en.wikipedia.org/wiki/Cryptographic_hash_function`](https://en.wikipedia.org/wiki/Cryptographic_hash_function)

+   HMAC：[`en.wikipedia.org/wiki/HMAC`](https://en.wikipedia.org/wiki/HMAC)

+   密码学服务（Python STD 库）：[`docs.python.org/3/library/crypto.html`](https://docs.python.org/3/library/crypto.html)

+   IANA JSON Web Token 声明注册：[`www.iana.org/assignments/jwt/jwt.xhtml`](https://www.iana.org/assignments/jwt/jwt.xhtml)

+   PyJWT 库：[`pyjwt.readthedocs.io/`](https://pyjwt.readthedocs.io/)

+   密码学库：[`cryptography.io/`](https://cryptography.io/)

网上有很多信息，也有很多书籍可以学习，但我们建议你从主要概念开始，然后逐渐深入到你想要更彻底了解的特定细节。

# 摘要

在这一章中，我们探索了 Python 标准库中的密码学世界。我们学习了如何使用不同的密码学函数为消息创建哈希（或摘要）。我们还学习了在密码学背景下创建令牌和处理随机数据的方法。

我们随后对标准库外进行了一次小型的巡游，以了解 JSON Web Tokens，这些在现代系统和应用程序中常用于身份验证和声明相关功能。

最重要的就是要理解，在密码学方面，手动操作是非常危险的，因此最好将其留给专业人士，并简单地使用我们可用的工具。

下一章将介绍如何测试我们的代码，以确保它按预期的方式工作。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

`discord.com/invite/uaKmaz7FEC`

![img](img/QR_Code119001106417026468.png)
