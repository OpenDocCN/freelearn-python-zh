

# 保护你的服务

到目前为止，这本书中所有服务之间的交互都没有进行任何形式的身份验证或授权；每个 HTTP 请求都会愉快地返回结果。但在实际生产中，这不可能发生，有两个简单的原因：我们需要知道谁在调用服务（身份验证），并且我们需要确保调用者有权执行调用（授权）。例如，我们可能不希望匿名调用者删除数据库中的条目。

在单体 Web 应用程序中，简单的身份验证可以通过登录表单实现，一旦用户被识别，就会设置一个带有会话标识符的 cookie，以便客户端和服务器可以在所有后续请求上协作。在基于微服务的架构中，我们不能在所有地方使用这种方案，因为服务不是用户，也不会使用 Web 表单进行身份验证。我们需要一种自动接受或拒绝服务之间调用的方式。

OAuth2 授权协议为我们提供了在微服务中添加身份验证和授权的灵活性，这可以用来验证用户和服务。在本章中，我们将了解 OAuth2 的基本特性和如何实现一个身份验证微服务。这个服务将被用来保护服务之间的交互。

在代码层面可以做一些事情来保护你的服务，例如控制系统调用，或者确保 HTTP 重定向不会结束在敌对网页上。我们将讨论如何添加对不良格式数据的保护，一些常见的陷阱以及如何扫描你的代码以发现潜在的安全问题。

最后，保护服务还意味着我们希望在恶意网络流量到达我们的应用程序之前将其过滤掉。我们将探讨设置基本 Web 应用程序防火墙来保护我们的服务。

# OAuth2 协议

如果你正在阅读这本书，你很可能是那些使用用户名和密码登录网页的人。这是一个简单的模型来确认你是谁，但也有一些缺点。

许多不同的网站存在，每个网站都需要妥善处理某人的身份和密码。随着存储身份的地方增多，以及密码可以通过不同系统采取的路径增多，安全漏洞的可能性也会增加。这也使得攻击者更容易创建假冒网站，因为人们习惯于在多个可能略有不同的地方输入他们的用户名和密码。相反，你可能遇到过允许你“使用 Google”、“Microsoft”、“Facebook”或“GitHub”登录的网站。这个功能使用了 OAuth2，或者基于它的工具。

OAuth2 是一个广泛采用的标准，用于保护 Web 应用程序及其与用户和其他 Web 应用程序的交互。只有一个服务会被告知你的密码或多因素认证码，任何需要认证你的网站都会将你引导到那里。在这里我们将介绍两种认证类型，第一种是认证代码授权，它是由人类使用浏览器或移动应用程序发起的。

用户驱动的认证代码授权流程看起来很复杂，如*图 7.1*所示，但它发挥着重要的作用。按照图中的流程进行，当客户端请求一个资源——无论是网页还是某些数据，例如——他们必须登录才能查看时，应用程序会将`302`重定向发送到认证服务。在那个 URL 中会有另一个地址，认证服务可以使用它将客户端送回应用程序。

一旦客户端连接，认证服务就会执行你可能预期的事情——它会要求用户名、密码和多重因素认证码，有些人甚至还会显示图片或文本来证明你访问的是正确的位置。登录正确后，认证服务将客户端重定向回应用程序，这次带有用于展示的令牌。

应用程序可以使用认证服务验证令牌，并记住该结果直到令牌过期，或者对于某些可配置的时间长度，偶尔重新检查以确保令牌没有被撤销。这样，应用程序就永远不需要处理用户名或密码，只需要学习足够的信息来唯一标识客户端。

![图片](img/B17108_07_01.png)

图 7.1：OAuth2 认证流程

当为程序设置 OAuth2 以便使用，使一个服务能够连接到另一个服务时，有一个类似的过程称为**客户端凭证授权**（**CCG**），其中服务可以连接到认证微服务并请求一个它可以使用的令牌。你可以参考 OAuth2 授权框架中*第 4.4 节*描述的 CCG 场景以获取更多信息：[`tools.ietf.org/html/rfc6749#section-4.4`](https://tools.ietf.org/html/rfc6749#section-4.4)。

这与授权代码的工作方式类似，但服务不会像用户一样重定向到网页。相反，它通过一个可以交换为令牌的秘密密钥隐式授权。

对于基于微服务的架构，使用这两种类型的授权将使我们能够集中管理系统的每个方面的认证和授权。构建一个实现 OAuth2 协议一部分的微服务，用于认证服务和跟踪它们之间的交互，是减少安全问题的良好解决方案——所有内容都集中在一个地方。

在本章中，CCG 流程是迄今为止最有趣的部分，因为它允许我们独立于用户来保护我们的微服务交互。它还简化了权限管理，因为我们可以根据上下文发行具有不同作用域的令牌。应用程序仍然负责执行那些作用域可以做什么和不能做什么的强制措施。

如果您不想实现和维护应用程序的认证部分，并且可以信任第三方来管理此过程，那么 Auth0 是一个出色的商业解决方案，它为基于微服务的应用程序提供了所有所需的 API：[`auth0.com/`](https://auth0.com/)。

# 基于 X.509 证书的认证

`X.509` 标准 ([`datatracker.ietf.org/doc/html/rfc5280`](https://datatracker.ietf.org/doc/html/rfc5280)) 用于保护网络。每个使用 TLS 的网站——即带有 `https://` URL 的网站——在其网络服务器上都有一个 `X.509` 证书，并使用它来验证服务器的身份并设置连接将使用的加密。

当客户端面对这样的证书时，它是如何验证服务器身份的？每个正确发行的证书都是由受信任的机构进行加密签名的。**证书颁发机构**（**CA**）通常会向您颁发证书，并且是浏览器依赖的最终组织，以了解可以信任谁。当加密连接正在协商时，客户端将检查它所获得的证书，并检查谁签发了它。如果它是一个受信任的 CA 并且加密检查通过，那么我们可以假设该证书代表它所声称的。有时签发者是一个中间机构，因此此步骤应重复进行，直到客户端达到一个受信任的 CA。

可以创建一个自签名证书，这在测试套件或本地开发环境中可能很有用——尽管这在数字上等同于说，“相信我，因为我这么说。”生产服务不应使用自签名证书，如果浏览器发出警告，坐在它前面的人有理由对访问的网站保持警惕。

获取一个好的证书比以前容易得多，这要归功于 Let's Encrypt ([`letsencrypt.org/`](https://letsencrypt.org/))。仍然收取证书费用的组织仍然提供价值——例如，扩展验证等特性不容易自动化，有时浏览器中的额外显示（通常在地址栏中的绿色锁形图标）也是值得的。

让我们使用 Let's Encrypt 生成一个证书，并使用一些命令行工具来检查它。在 Let's Encrypt 网站上有安装名为 `certbot` 的实用程序的说明。根据所使用的平台，说明可能会有所不同，因此我们在这里不包括它们。一旦安装了 `certbot`，为 `nginx` 等网络服务器获取证书就变得简单：

```py
$ sudo certbot --nginx
No names were found in your configuration files. Please enter in your domain
name(s) (comma and/or space separated)  (Enter 'c' to cancel): certbot-test.mydomain.org
Requesting a certificate for certbot-test.mydomain.org
Performing the following challenges:
http-01 challenge for certbot-test.mydomain.org
Waiting for verification...
Cleaning up challenges
Deploying Certificate to VirtualHost /etc/nginx/sites-enabled/default
Redirecting all traffic on port 80 to ssl in /etc/nginx/sites-enabled/default
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Congratulations! You have successfully enabled https://certbot-test.mydomain.org
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
nginx configuration, we see the parts that certbot has added in order to secure the web service:
```

```py
listen [::]:443 ssl ipv6only=on; # managed by Certbot
listen 443 ssl; # managed by Certbot
ssl_certificate /etc/letsencrypt/live/certbot-test.mydomain.org/fullchain.pem; # managed by Certbot
ssl_certificate_key /etc/letsencrypt/live/certbot-test.mydomain.org/privkey.pem; # managed by Certbot
include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot 
```

我们可以使用 OpenSSL 工具包来检查我们的证书，无论是通过查看文件还是通过向 Web 服务器发送查询。检查证书将提供大量信息，尽管对我们来说，重要的部分包括**有效期**和**主题**部分。服务运行时证书未续期而过期是一个常见的错误条件；`certbot`包括帮助程序，可以自动刷新即将到期的证书，因此如果我们使用提供的工具，这应该不会成为问题。

证书主题描述了证书是为哪个实体创建的，在这个例子中，是一个主机名。这里展示的证书的主题**通用名称**（**CN**）为`certbot-test.mydomain.org`，但如果这不是我们使用的主机名，那么连接到我们服务的客户端将有权抱怨。

为了检查证书的详细信息，包括主题，我们可以使用`openssl`实用程序来显示证书：

```py
$ sudo openssl x509 -in /etc/letsencrypt/live/certbot-test.mydomain.org/fullchain.pem  -text -noout
Certificate:
	Data:
  	Version: 3 (0x2)
  	Serial Number:
    	04:92:e3:37:a4:83:77:4f:b9:d7:5c:62:24:74:7e:a4:5a:e0
  	Signature Algorithm: sha256WithRSAEncryption
  	Issuer: C = US, O = Let's Encrypt, CN = R3
  	Validity
    	Not Before: Mar 13 14:43:12 2021 GMT
    	Not After : Jun 11 14:43:12 2021 GMT
  	Subject: CN = certbot-test.mydomain.org
... 
```

还可以使用`openssl`实用程序连接到正在运行的 Web 服务器，这可能有助于确认正在使用正确的证书，运行即将到期的证书的监控脚本，或其他类似的诊断。使用我们上面配置的`nginx`实例，我们可以建立一个加密会话，通过这个会话我们可以发送 HTTP 命令：

```py
$ openssl s_client -connect localhost:443
CONNECTED(00000003)
Can't use SSL_get_servername
depth=2 O = Digital Signature Trust Co., CN = DST Root CA X3
verify return:1
depth=1 C = US, O = Let's Encrypt, CN = R3
verify return:1
depth=0 CN = certbot-test.mydomain.org
verify return:1
---
Certificate chain
 0 s:CN = certbot-test.mydomain.org
 i:C = US, O = Let's Encrypt, CN = R3
 1 s:C = US, O = Let's Encrypt, CN = R3
 i:O = Digital Signature Trust Co., CN = DST Root CA X3
---
Server certificate
-----BEGIN CERTIFICATE-----
MII  
# A really long certificate has been removed here
-----END CERTIFICATE-----
subject=CN = certbot-test.mydomain.org
issuer=C = US, O = Let's Encrypt, CN = R3
---
New, TLSv1.3, Cipher is TLS_AES_256_GCM_SHA384
Server public key is 2048 bit
Secure Renegotiation IS NOT supported
Compression: NONE
Expansion: NONE
No ALPN negotiated
Early data was not sent
Verify return code: 0 (ok)
--- 
```

我们可以轻松地读取这个交换中的公共证书，并确认这是我们期望服务器从其配置文件中使用的证书。我们还可以发现客户端和服务器之间协商了哪些加密套件，并识别出任何可能成为问题的套件，如果正在使用较旧的客户端库或 Web 浏览器。

到目前为止，我们只讨论了服务器使用证书来验证其身份并建立安全连接的情况。客户端也可以出示证书来验证自身。证书将允许我们的应用程序验证客户端是否是他们声称的身份，但我们应该小心，因为这并不意味着客户端被允许做某事——这种控制仍然掌握在我们自己的应用程序手中。管理这些证书、设置 CA 以向客户端颁发适当的证书以及如何正确分发文件，这些都超出了本书的范围。如果您正在创建的应用程序选择这样做，一个好的起点是查看`nginx`文档中的[`nginx.org/en/docs/http/ngx_http_ssl_module.html#ssl_verify_client`](http://nginx.org/en/docs/http/ngx_http_ssl_module.html#ssl_verify_client)。

让我们来看看如何验证使用我们服务的客户端，以及我们如何设置一个专门用于验证客户端访问的微服务。

# 基于令牌的认证

正如我们之前所说的，当一个服务想要在不进行任何用户干预的情况下访问另一个服务时，我们可以使用 CCG 流。CCG 的理念是，一个服务可以连接到身份验证服务并请求一个令牌，然后它可以使用这个令牌来对其他服务进行身份验证。

在需要不同权限集或身份不同的系统中，身份验证服务可以发行多个令牌。

令牌可以包含对身份验证和授权过程有用的任何信息。以下是一些例子：

+   如果与上下文相关，`username` 或 `ID`

+   范围，它表示调用者可以做什么（读取、写入等）

+   一个表示令牌签发时间的 `时间戳`

+   一个表示令牌有效期的 `过期时间戳`

令牌通常构建为一个完整的证明，表明你有权使用一项服务。它是完整的，因为可以在不知道其他任何信息或无需查询外部资源的情况下，通过身份验证服务验证令牌。根据实现方式，令牌还可以用来访问不同的微服务。

OAuth2 使用 JWT 标准作为其令牌。OAuth2 中没有要求必须使用 JWT 的内容——它只是恰好适合 OAuth2 想要实现的功能。

## JWT 标准

在 RFC 7519 中描述的 **JSON Web Token**（JWT）是一个常用的标准，用于表示令牌：[`tools.ietf.org/html/rfc7519`](https://tools.ietf.org/html/rfc7519)。

JWT 是由三个点分隔的长字符串组成：

+   **头部**：它提供了有关令牌的信息，例如使用了哪种哈希算法

+   **有效载荷**：这是实际数据

+   **签名**：这是头部和有效载荷的签名哈希，用于验证其合法性

JWTs 是 Base64 编码的，因此它们可以安全地用于查询字符串。以下是一个 JWT 的编码形式：

```py
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9
.
eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlNpbW9uIEZyYXNlciIsIm lhdCI6MTYxNjQ0NzM1OH0
.
K4ONCpK9XKtc4s56YCC-13L0JgWohZr5J61jrbZnt1M 
```

令牌上方的每个部分在显示时通过换行符分隔——原始令牌是一行。你可以使用 Auth0 提供的实用工具来实验 JWT 编码和解码，该实用工具位于 [`jwt.io/`](https://jwt.io/)。

如果我们使用 Python 来解码它，数据就是简单的 Base64：

```py
>>> import base64
>>> def decode(data):
... # adding extra = for padding if needed
... pad = len(data) % 4
... if pad > 0:
...     data += "=" * (4 - pad)
... return base64.urlsafe_b64decode(data)
...
>>> decode("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
b'{"alg":"HS256","typ":"JWT"}'
>>> import base64 
>>> decode("eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlNpbW9uIEZyYXNlciIsImlhdC I6MTYxNjQ0NzM1OH0")
b'{"sub":"1234567890","name":"Simon Fraser","iat":1616447358}'
>>> decode("K4ONCpK9XKtc4s56YCC-13L0JgWohZr5J61jrbZnt1M")
b"+\x83\x8d\n\x92\xbd\\\xab\\\xe2\xcez` \xbe\xd7r\xf4&\x05\xa8\x85\x9a\xf9'\xadc\xad\xb6g\xb7S" 
```

JWT 的每一部分都是一个 JSON 映射，除了签名。头部通常只包含 `typ` 和 `alg` 键：`typ` 键表示这是一个 JWT，而 `alg` 键指示使用了哪种哈希算法。在下面的头部示例中，我们有 `HS256`，代表 `HMAC-SHA256`：

```py
{"typ": "JWT",  "alg": "HS256"} 
```

有效载荷包含你需要的内容，每个字段在 RFC 7519 的术语中被称为 **JWT 断言**。RFC 有一个预定义的断言列表，令牌可能包含这些断言，称为 **注册的断言名称**。以下是一些子集：

+   `iss`：这是发行者，即生成令牌的实体的名称。通常是完全限定的主机名，因此客户端可以使用它通过请求 `/.well-known/jwks.json` 来发现其公钥。

+   `exp`: 这是过期时间，是一个令牌无效的戳记。

+   `nbf`: 这代表**不可用之前时间**，是一个令牌无效的戳记。

+   `aud`: 这表示受众，即令牌发行的接收者。

+   `iat`: 代表**发行于**，这是一个表示令牌发行时间的戳记。

在以下有效载荷示例中，我们提供了自定义的`user_id`值以及使令牌在发行后 24 小时内有效的时戳；一旦有效，该令牌可以用于 24 小时：

```py
{
  "iss": "https://tokendealer.mydomain.org", 
  "aud": "mydomain.org", 
  "iat": 1616447358, 
  "nbt": 1616447358, 
  "exp": 1616533757, 
  "user_id": 1234
} 
```

这些头部为我们提供了很多灵活性，以控制我们的令牌将保持有效的时间。根据微服务的性质，令牌的**生存时间**（**TTL**）可以是极短到无限。例如，与系统内其他服务交互的微服务可能需要依赖足够长的令牌，以避免不必要地多次重新生成令牌。另一方面，如果你的令牌在野外分发，或者它们与改变高度重要的事情相关，使它们短暂有效是一个好主意。

JWT 的最后部分是签名。它包含头部和有效载荷的签名哈希。用于签名哈希的算法有几种；一些基于密钥，而另一些基于公钥和私钥对。

## PyJWT

在 Python 中，`PyJWT`库提供了你生成和读取 JWT 所需的所有工具：[`pyjwt.readthedocs.io/`](https://pyjwt.readthedocs.io/)。

一旦你使用 pip 安装了`pyjwt`（和`cryptography`），你就可以使用`encode()`和`decode()`函数来创建令牌。在以下示例中，我们使用`HMAC-SHA256`创建 JWT 并读取它。在读取令牌时，通过提供密钥来验证签名：

```py
>>> import jwt
>>>  def create_token(alg="HS256", secret="secret", data=None):
        return jwt.encode(data, secret, algorithm=alg)
...
>>>
>>> def read_token(token, secret="secret", algs=["HS256"]):
...  return jwt.decode(token, secret, algorithms=algs)
...
>>>  token = create_token(data={"some": "data", "inthe": "token"})
>>> print(token)
eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzb21lIjoiZGF0YSIsImludGhlIjoidG9rZW4ifQ.vMHiSS_vk-Z3gMMxcM22Ssjk3vW3aSmJXQ8YCSCwFu4
>>> print(read_token(token))
{'some': 'data', 'inthe': 'token'} 
```

当执行此代码时，令牌会以压缩和未压缩两种形式显示。如果你使用已注册的声明之一，`PyJWT`将控制它们。例如，如果提供了`exp`字段且令牌已过期，库将引发错误。

使用密钥进行签名和验证签名在运行少量服务时很好，但很快可能会成为问题，因为它要求你需要在所有需要验证签名的服务之间共享密钥。因此，当需要更改密钥时，在堆栈中安全地更改它可能是一个挑战。基于你共享的密钥进行身份验证也是一种弱点。如果单个服务被破坏且密钥被盗，你的整个身份验证系统都会受到破坏。

一个更好的技术是使用由公钥和私钥组成的非对称密钥。私钥由令牌发行者用来签名令牌，而公钥可以被任何人用来验证签名是否由该发行者签名。当然，如果攻击者能够访问私钥，或者能够说服客户端伪造的公钥是合法的，你仍然会遇到麻烦。

但使用公私钥对确实减少了您认证过程的攻击面，通常足以阻止大多数攻击者；并且，由于认证微服务将是唯一包含私钥的地方，您可以专注于增加额外的安全性。例如，这样的明智服务通常部署在防火墙环境中，所有访问都受到严格控制。现在让我们看看我们如何在实践中创建非对称密钥。

## 使用 JWT 证书

为了简化这个例子，我们将使用之前为`nginx`生成的`letsencrypt`证书。如果您在笔记本电脑或无法从互联网访问的容器上开发，您可能需要使用云实例或`certbot` DNS 插件生成这些证书，并将它们复制到正确的位置。

如果`certbot`直接生成证书，它们将保存在`/etc/letsencrypt/live/your-domain/`。首先，我们关注以下两个文件：

+   `cert.pem`，其中包含证书

+   `privkey.pem`，其中包含 RSA 私钥

为了使用这些与 PyJWT，我们需要从证书中提取公钥：

```py
openssl x509 -pubkey -noout -in cert.pem  > pubkey.pem 
```

**RSA**代表**Rivest, Shamir, 和 Adleman**，这三位作者。RSA 加密算法生成的密钥可以长达 4,096 字节，被认为是安全的。

从那里，我们可以在我们的 PyJWT 脚本中使用`pubkey.pem`和`privkey.pem`来签名和验证令牌的签名，使用`RSASSA-PKCS1-v1_5`签名算法和`SHA-512`哈希算法：

```py
 import jwt
  with open("pubkey.pem") as f:
    PUBKEY = f.read()
  with open("privkey.pem") as f:
    PRIVKEY = f.read()
  def create_token(**data):
    return jwt.encode(data, PRIVKEY, algorithm="RS512")
  def read_token(token):
    return jwt.decode(token, PUBKEY, algorithms="RS512")
  token = create_token(some="data", inthe="token")
  print(token)
  read = read_token(token)
  print(read) 
```

结果与之前的运行相似，只是我们得到了一个更大的令牌：

```py
eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzUxMiJ9.eyJzb21lIjoiZGF0YSIsImludGh lIjoidG9rZW4ifQ.gi5p3k4PAErw8KKrghRjsi8g1IXnflivXiwwaZdFEh84zvgw9RJRa 50uJe778A1CBelnmo2iapSWOQ9Mq5U6gpv4VxoVYv6QR2zFNO13GB_tce6xQ OhjpAd-hRxouy3Ozj4oNmvwLpCT5dYPsCvIiuYrLt4ScK5S3q3a0Ny64VXy 3CcISNkyjs7fnxyMMkCMZq65Z7jOncf1RXpzNNIt546aJGsCcpCPGHR1cRj uvV_uxPAMd-dfy2d5AfiCXOgvmwQhNdaxYIM0gPgz9_yHPzgaPjtgYoJMc9iK ZdOLz2-8pLc1D3r_uP3P-4mfxP7mOhQHYBrY9nv5MTSwFC3JDA
{'some': 'data', 'inthe': 'token'} 
```

在每个请求中添加如此多的额外数据可能会对产生的网络流量产生影响，因此，基于密钥的 JWT 技术是一个可以考虑的选项，如果您需要减少网络开销。

# TokenDealer 微服务

在构建认证微服务的第一步，我们将实现执行 CCG 流程所需的一切。为此，应用程序接收来自需要令牌的服务请求，并在需要时生成它们，假设请求中包含已知的密钥。生成的令牌将有一个一天的寿命。这种方法具有最大的灵活性，没有生成我们自己的`X.509`证书的复杂性，同时允许我们有一个服务负责生成令牌。

这个服务将是唯一一个拥有用于签署令牌的私钥的服务，并将公开公钥供其他想要验证令牌的服务使用。这个服务也将是唯一一个保存所有客户端 ID 和密钥的地方。

我们将通过声明一旦服务获取到令牌，它就可以访问我们生态系统中的任何其他服务来大大简化实现。当服务使用令牌访问时，它可以本地验证该令牌或调用 TokenDealer 来执行验证。网络请求和微服务中的某些 CPU 使用之间的选择将取决于应用程序做什么以及它的瓶颈在哪里。在平衡安全和性能要求时，可能有必要最多每几分钟验证一次令牌，而不是每次都验证。然而，如果需要使令牌无效，这将会造成延迟，因此我们应该参考用户故事，并在必要时与将使用该服务的人讨论，以确定哪个最重要。

为了实现我们所描述的，这个微服务将创建三个端点：

+   `GET /.well-known/jwks.json`: 当其他微服务想要自行验证令牌时，这是以 RFC 7517 中描述的 **JSON Web Key** (**JWK**) 格式发布的公钥。有关更多信息，请参阅以下链接：[`tools.ietf.org/html/rfc7517`](https://tools.ietf.org/html/rfc7517)。

+   `POST /oauth/token`: 这个端点接受带有凭证的请求并返回一个令牌。添加 `/oauth` 前缀是一个广泛采用的约定，因为它在 OAuth RFC 中被使用。

+   `POST /verify_token`: 这个端点在给定一个令牌的情况下返回令牌的有效负载。如果令牌无效，它将返回 HTTP 400 错误代码。

使用微服务骨架，我们可以创建一个非常简单的 Quart 应用程序，该应用程序实现了这三个视图。骨架可在[`github.com/PacktPublishing/Python-Microservices-Development-2nd-Edition/`](https://github.com/PacktPublishing/Python-Microservices-Development-2nd-Edition/)找到。

让我们来看看这三个 OAuth 视图。

## OAuth 实现

对于 CCG 流程，需要令牌的服务会发送一个包含以下字段的 URL 编码体的 `POST` 请求：

+   `client_id`: 这是一个唯一字符串，用于标识请求者。

+   `client_secret`: 这是一个用于验证请求者的密钥。它应该是一个预先生成并注册到认证服务的随机字符串。

+   `grant_type`: 这是授权类型，在这里必须是 `client_credentials`。

我们将做出一些假设以简化实现。首先，为了演示目的，我们将保持秘密列表在 Python 数据结构中。在生产服务中，它们应该在静态时加密，并保存在具有弹性的数据存储中。我们还将假设`client_id`是调用微服务的名称，并且现在我们将使用`binascii.hexlify(os.urandom(16))`生成秘密。

第一个视图将是实际生成其他服务所需令牌的视图。在我们的例子中，我们每次创建令牌时都会读取私钥——对于实际服务来说，最好将其存储在应用程序配置中，以减少从磁盘读取文件所需的时间。我们确保客户端已经向我们发送了一个合理的请求，并且它想要一些`client_credentials`。错误处理函数和实用工具可以在本章的完整源代码示例中找到。

令牌本身是一个具有多个字段的复杂数据结构：令牌的发行者（`iss`），通常是服务的 URL；令牌的目标受众（`aud`），即令牌的目标对象；令牌签发的时间（`iat`）；以及其过期时间（`exp`）。然后我们使用`jwt.encode`方法对数据进行签名，并将其返回给请求客户端：

```py
@app.route("/oauth/token", methods=["POST"])
async def create_token():
    with open(current_app.config["PRIVATE_KEY_PATH"]) as f:
        key = f.read().strip()
    try:
        data = await request.form
        if data.get("grant_type") != "client_credentials":
            return bad_request(f"Wrong grant_type {data.get('grant_type')}")

        client_id = data.get("client_id")
        client_secret = data.get("client_secret")
        aud = data.get("audience", "")

        if not is_authorized_app(client_id, client_secret):
            return abort(401)

        now = int(time.time())

        token = {
            "iss": current_app.config["TOKENDEALER_URL"],
            "aud": aud,
            "iat": now,
            "exp": now + 3600 * 24,
        }
        token = jwt.encode(token, key, algorithm="RS512")
        return {"access_token": token}
    except Exception as e:
        return bad_request("Unable to create a token") 
```

接下来要添加的视图是一个返回我们令牌生成所使用的公钥的功能，这样任何客户端都可以验证令牌而无需进行进一步的 HTTP 请求。这通常位于一个众所周知的 URL——地址中实际上包含字符串`.well-known/`，这是 IETF 鼓励的做法，为客户端提供发现有关服务元数据的方式。在这里，我们响应的是 JWKS。

返回的数据中包含密钥类型（`kty`）、算法（`alg`）、公钥使用（`use`）——这里是一个签名——以及我们使用 RSA 算法生成的加密密钥所用的两个值：

```py
@app.route("/.well-known/jwks.json")
async def _jwks():
    """Returns the public key in the Json Web Key Set (JWKS) format"""
    with open(current_app.config["PUBLIC_KEY_PATH"]) as f:
        key = f.read().strip()
    data = {
        "alg": "RS512",
        "e": "AQAB",
        "n": key,
        "kty": "RSA",
        "use": "sig",
    }

    return jsonify({"keys": [data]}) 
```

最后一个视图允许客户端验证令牌而无需自己进行工作。与令牌生成相比，这要简单得多，我们只需从输入数据中提取正确的字段，并调用`jwt.decode`函数来提供值。请注意，此函数验证令牌是否有效，但并不验证令牌是否允许任何特定的访问——这部分取决于已经向其展示令牌的服务：

```py
@app.route("/verify_token", methods=["POST"])
async def verify_token():
    with open(current_app.config["PUBLIC_KEY_PATH"]) as f:
        key = f.read()
    try:
        input_data = await request.form
        token = input_data["access_token"]
        audience = input_data.get("audience", "")
        return jwt.decode(token, key, algorithms=["RS512"], audience=audience)
    except Exception as e:
        return bad_request("Unable to verify the token") 
```

TokenDealer 微服务的全部源代码可以在 GitHub 上找到：[`github.com/PacktPublishing/Python-Microservices-Development-2nd-Edition`](https://github.com/PacktPublishing/Python-Microservices-Development-2nd-Edition).

微服务可以提供更多关于令牌生成的功能。例如，管理作用域并确保微服务 A 不允许生成在微服务 B 中使用的令牌，或者管理一个授权请求某些令牌的服务白名单。客户端还可以请求一个仅用于只读用途的令牌。然而，尽管如此，我们已实现的模式是微服务环境中简单基于令牌的认证系统的基石，您可以在自己的基础上进行开发，同时它也足够好，适用于我们的 Jeeves 应用。

回顾我们的示例微服务，TokenDealer 现在作为生态系统中的一个独立微服务存在，创建和验证允许访问我们的数据服务的密钥，并授权访问我们查询其他网站所需的第三方令牌和 API 密钥：

![img/B17108_07_02.png](img/B17108_07_02.png)

图 7.2：带有 CCG TokenDealer 的微服务生态系统

那些需要 JWT 的服务可以通过调用 TokenDealer 微服务来验证它。*图 7.2*中的 Quart 应用需要代表其用户从 TokenDealer 获取令牌。

现在我们已经有一个实现了 CCG 的 TokenDealer 服务，让我们看看它如何在下一节中由我们的服务使用。

## 使用 TokenDealer

在 Jeeves 中，**数据服务**是一个需要认证的好例子。通过**数据服务**添加信息需要限制在授权的服务范围内：

![image1.png](img/B17108_07_03.png)

图 7.3：请求 CCG 工作流

为该链接添加认证分为四个步骤：

1.  **TokenDealer**为 Strava 工作者管理一个`client_id`和`client_secret`对，并与 Strava 工作者开发者共享

1.  Strava 工作者使用`client_id`和`client_secret`从**TokenDealer**检索令牌

1.  工作者将令牌添加到每个请求到**数据服务**的头部

1.  **数据服务**通过调用**TokenDealer**的验证 API 或执行本地**JWT**验证来验证令牌

在完整实现中，第一步可以部分自动化。生成客户端密钥通常是通过认证服务的 Web 管理面板完成的。然后，该密钥提供给客户端微服务开发者。现在，每个需要令牌的微服务都可以获取一个，无论是首次连接，还是因为它们已经获得的令牌已过期。他们要做的只是在使用时将令牌添加到调用数据服务的授权头中。

以下是一个使用`requests`库进行此类调用的示例——假设我们的 TokenDealer 已经在`localhost:5000`上运行：

```py
# fetch_token.py
import requests
TOKENDEALER_SERVER = "http://localhost:5000"
SECRET = "f0fdeb1f1584fd5431c4250b2e859457"
def get_token():
    data = {
        "client_id": "worker1",
        "client_secret": secret,
        "audience": "jeeves.domain",
        "grant_type": "client_credentials",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    url = tokendealer_server + "/oauth/token"
    response = requests.post(url, data=data, headers=headers)
    return response.json()["access_token"] 
```

`get_token()` 函数检索一个令牌，该令牌随后可以在代码调用数据服务时用于授权头，我们假设数据服务在本例中监听端口 `5001`：

```py
# auth_caller.py
_TOKEN = None
def get_auth_header(new=False):
    global _TOKEN
    if _TOKEN is None or new:
        _TOKEN = get_token()
    return "Bearer " + _TOKEN
_dataservice = "http://localhost:5001"
def _call_service(endpoint, token):
    # not using session and other tools, to simplify the code
    url = _dataservice + "/" + endpoint
    headers = {"Authorization": token}
    return requests.get(url, headers=headers)
def call_data_service(endpoint):
    token = get_auth_header()
    response = _call_service(endpoint, token)
    if response.status_code == 401:
        # the token might be revoked, let's try with a fresh one
        token = get_auth_header(new=True)
        response = _call_service(endpoint, token)
    return response 
```

`call_data_service()` 函数会在调用数据服务并返回 401 响应时尝试获取新的令牌。这种在 401 响应上刷新令牌的模式可以用于你所有的微服务来自动化令牌生成。

这包括服务间的身份验证。你可以在示例 GitHub 仓库中找到完整的实现，以尝试基于 JWT 的身份验证方案，并将其作为构建你的身份验证过程的基础。

下一个部分将探讨保护你的网络服务的重要方面之一，即保护代码本身。

# 保护你的代码

无论我们做什么，应用程序都必须接收数据并对其采取行动，否则它将不会非常有用。如果一个服务接收数据，那么一旦你将你的应用程序暴露给世界，它就会面临众多可能的攻击类型，你的代码需要考虑到这一点进行设计。

任何发布到网络上的内容都可能受到攻击，尽管我们有优势，即大多数微服务没有暴露在公共互联网上，这减少了它们可能被利用的方式。系统的预期输入和输出更窄，通常可以使用如 OpenAPI 之类的规范工具更好地定义。

攻击并不总是由于恶意意图。如果调用者有错误或者只是没有正确调用你的服务，预期的行为应该是发送回一个`4xx`响应，并向客户端解释为什么请求被拒绝。

**开放网络应用安全项目**（**OWASP**）（[`www.owasp.org`](https://www.owasp.org)）是一个学习如何保护你的网络应用程序免受不良行为侵害的优秀资源。让我们看看一些最常见的攻击形式：

+   **注入**：在一个接收数据的程序中，攻击者通过请求发送 SQL 语句、shell 命令或其他指令。如果你的应用程序在使用这些数据时不够小心，你可能会运行旨在损害应用程序的代码。在 Python 中，可以通过使用 SQLAlchemy 来避免 SQL 注入攻击，它会以安全的方式为你构造 SQL 语句。如果你直接使用 SQL，或者向 shell 脚本、LDAP 服务器或其他结构化查询提供参数，你必须确保每个变量都被正确地引用。

+   **跨站脚本**（**XSS**）：这种攻击只发生在显示 HTML 的网页上。攻击者使用一些查询属性尝试在页面上注入他们的 HTML 片段，以欺骗用户执行一系列操作，让他们以为自己在合法网站上。

+   **跨站请求伪造**（**XSRF**/**CSRF**）：这种攻击基于通过重用用户从另一个网站的用户凭据来攻击服务。典型的 CSRF 攻击发生在`POST`请求中。例如，一个恶意网站显示一个链接给用户，诱骗该用户使用他们现有的凭据在你的网站上执行`POST`请求。

像本地文件包含（**LFI**）、远程文件包含（**RFI**）或远程代码执行（**RCE**）这样的攻击都是通过客户端输入欺骗服务器执行某些操作或泄露服务器文件的攻击。当然，这些攻击可能发生在大多数语言和工具包编写的应用程序中，但我们将检查一些 Python 的工具来防止这些攻击。

安全代码背后的理念简单，但在实践中很难做好。两个基本的原则是：

+   在应用程序和数据中执行任何操作之前，都应该仔细评估来自外部世界的每个请求。

+   应用程序在系统上所做的每一件事都应该有一个明确和有限的作用域。

让我们看看如何在实践中实施这些原则。

## 限制应用程序的作用域

即使你信任认证系统，你也应该确保连接的人拥有完成工作所需的最小访问级别。如果有客户端连接到你的微服务并能够进行认证，这并不意味着他们应该被允许执行任何操作。如果他们只需要只读访问，那么他们应该只被授予这一点。

这不仅仅是保护免受恶意代码的侵害，还包括错误和意外。每次当你认为“客户端永远不应该调用这个端点”时，就应该有某种机制来积极阻止客户端使用它。

这种作用域限制可以通过 JWTs 通过定义角色（如读写）并在令牌中添加该信息来实现，例如，在权限或作用域键下。然后目标微服务将能够拒绝使用仅应读取数据的令牌进行的`POST`调用。

这就是当你授予 GitHub 账户或 Android 手机上的应用程序访问权限时会发生的情况。会显示应用程序想要执行的操作的详细列表，你可以授予或拒绝访问。

这是在网络级控制和防火墙的基础上。如果你控制着微服务生态系统的所有部分，你还可以在系统级别使用严格的防火墙规则来白名单允许与每个微服务交互的 IP 地址，但这种设置在很大程度上取决于你部署应用程序的位置。在**亚马逊网络服务**（**AWS**）云环境中，你不需要配置 Linux 防火墙；你只需要在 AWS 控制台中设置访问规则即可。*第十章*，*在 AWS 上部署*，涵盖了在亚马逊云上部署微服务的基本知识。

除了网络访问之外，任何其他应用程序可以访问的资源都应在可能的情况下进行限制。在 Linux 上以 root 用户运行应用程序不是一个好主意，因为如果你的应用程序拥有完整的行政权限，那么成功入侵的攻击者也会有。

从本质上讲，如果一层安全措施失败，后面应该还有另一层。如果一个应用程序的 web 服务器被成功攻击，任何攻击者理想情况下都应尽可能有限制，因为他们只能访问应用程序中服务之间定义良好的接口——而不是对运行代码的计算机拥有完整的行政控制。在现代部署中，系统根访问已成为一种间接威胁，因为大多数应用程序都在容器或一个 **虚拟机**（**VM**）中运行，但即使其能力被运行的 VM 限制，一个进程仍然可以造成很多损害。如果攻击者访问到您的其中一个 VM，他们已经实现了控制整个系统的第一步。为了减轻这个问题，您应该遵循以下两条规则：

1.  所有软件都应该以尽可能小的权限集运行

1.  在执行来自您的网络服务的进程时，要非常谨慎，并在可能的情况下避免

对于第一条规则，像 `nginx` 这样的 web 服务器的默认行为是使用 `www-data` 用户和组来运行其进程，这样标准用户控制可以防止服务器访问其他文件，并且账户本身可以设置成不允许运行 shell 或任何其他交互式命令。同样的规则也适用于您的 Quart 进程。我们将在 *第九章*，*打包和运行 Python* 中看到在 Linux 系统上以用户空间运行堆栈的最佳实践。

对于第二条规则，除非绝对必要，否则应避免使用任何 Python 对 `os.system()` 的调用，因为它在计算机上创建一个新的用户 shell，增加了运行不良命令的风险，并增加了对系统无控制访问的风险。`subprocess` 模块更好，尽管它也必须谨慎使用以避免不希望的结果——避免使用 `shell=True` 参数，这将导致与 `os.system()` 相同的问题，并避免使用输入数据作为参数和命令。这也适用于发送电子邮件或通过 FTP 连接到第三方服务器的高级网络模块，通过本地系统。

## 不受信任的传入数据

大多数应用程序接受数据作为输入：要查找哪个账户；为哪个城市获取天气预报；要将钱转入哪个账户，等等。问题是来自我们系统之外的数据不容易被信任。

之前，我们讨论了 SQL 注入攻击；现在让我们考虑一个非常简单的例子，其中我们使用 SQL 查询来查找用户。我们有一个函数，它将查询视为要格式化的字符串，并使用标准的 Python 语法填充它：

```py
import pymysql 
connection = pymysql.connect(host='localhost', db='book') 
def get_user(user_id): 
    query = f"select * from user where id = {user_id}"
        with connection.cursor() as cursor: 
        cursor.execute(query) 
        result = cursor.fetchone() 
        return result 
```

当 `user_id` 总是合理的值时，这看起来是正常的。然而，如果有人提供了一个精心制作的恶意值呢？如果我们允许人们为上面的 `get_user()` 函数输入数据，并且他们不是输入一个数字作为 `user_id`，而是输入：

```py
'1'; insert into user(id, firstname, lastname, password) values (999, 'pwnd', 'yup', 'somehashedpassword') 
```

现在我们的 SQL 语句实际上是两个语句：

```py
select * from user where id = '1'
insert into user(id, firstname, lastname, password) values (999, 'pwnd', 'yup', 'somehashedpassword') 
```

`get_user` 将执行预期的查询，以及一个将添加新用户的查询！它还可以删除表，或执行 SQL 语句可用的任何其他操作。如果认证客户端权限有限，则有一些限制措施，但仍然可能暴露大量数据。可以通过引用构建原始 SQL 查询时使用的任何值来防止这种情况。在 `PyMySQL` 中，您只需将值作为参数传递给 `execute` 参数以避免此问题：

```py
def get_user(user_id): 
    query = 'select * from user where id = %s' 
        with connection.cursor() as cursor: 
        cursor.execute(query, (user_id,)) 
        result = cursor.fetchone() 
        return result 
```

每个数据库库都有这个功能，所以只要您在构建原始 SQL 时正确使用这些库，就应该没问题。更好的做法是完全避免使用原始 SQL，而是通过 SQLAlchemy 使用数据库模型。

如果您有一个视图，它从传入的请求中获取 JSON 数据并将其用于向数据库推送数据，您应该验证传入的请求包含您期望的数据，而不是盲目地将其传递给您的数据库后端。这就是为什么使用 Swagger 将数据描述为模式并使用它们来验证传入数据可能很有趣。微服务通常使用 JSON，但如果你碰巧使用模板提供格式化输出，那么这也是你需要小心处理模板如何处理变量的另一个地方。

**服务器端模板注入**（**SSTI**）是一种可能的攻击，其中您的模板盲目执行 Python 语句。在 2016 年，在 Uber 网站的一个 Jinja2 模板上发现了一个这样的注入漏洞，因为原始格式化是在模板执行之前完成的。更多信息请参阅[`hackerone.com/reports/125980`](https://hackerone.com/reports/125980)。

代码类似于这个小应用程序：

```py
from quart import Quart, request, render_template_string
app = Quart(__name__)
SECRET = "oh no!"
_TEMPLATE = """
    Hello %s
    Welcome to my API!
    """
class Extra:
    def __init__(self, data):
    self.data = data
@app.route("/")
async def my_microservice():
    user_id = request.args.get("user_id", "Anonymous")
    tmpl = _TEMPLATE % user_id
    return await render_template_string(tmpl, extra=Extra("something"))
app.run() 
```

通过在模板中使用原始的 `%` 格式化语法进行预格式化，视图在应用程序中创建了一个巨大的安全漏洞，因为它允许攻击者在 Jinja 脚本执行之前注入他们想要的内容。在下面的示例中，`user_id` 变量的安全漏洞被利用来从模块中读取 `SECRET` 全局变量的值：

```py
# Here we URL encode the following:
# http://localhost:5000/?user_id={{extra.__class__.__init__.__globals__["SECRET"]}} 
$ curl http://localhost:5000/?user_id=%7B%7Bextra.__class__.__init__.__globals__%5B%22SECRET%22%5D%7D%7D
Hello oh no!
Welcome to my API! 
```

这就是为什么避免使用输入数据进行字符串格式化很重要，除非有模板引擎或其他提供保护的层。

如果您需要在模板中评估不受信任的代码，您可以使用 Jinja 的沙盒；请参阅[`jinja.pocoo.org/docs/latest/sandbox/`](http://jinja.pocoo.org/docs/latest/sandbox/)。这个沙盒将拒绝访问正在评估的对象的方法和属性。例如，如果您在模板中传递一个可调用对象，您将确保其属性，如 `;__class__`，不能被使用。

话虽如此，由于语言本身的性质，Python 沙盒很难配置正确。很容易误配置沙盒，而且沙盒本身也可能因为语言的新版本而被破坏。最安全的做法是完全避免评估不受信任的代码，并确保你不会直接依赖于传入数据用于模板。

## 重定向和信任查询

在处理重定向时，也适用相同的预防措施。一个常见的错误是创建一个登录视图，假设调用者将被重定向到内部页面，并使用一个普通的 URL 进行重定向：

```py
@app.route('/login') 
def login(): 
    from_url = request.args.get('from_url', '/') 
    # do some authentication 
    return redirect(from_url) 
```

这个视图可以将调用者重定向到任何网站，这是一个重大的威胁——尤其是在登录过程中。良好的做法是在调用`redirect()`时避免使用自由字符串，而是使用`url_for()`函数，这将创建一个指向你的应用域的链接。如果你需要重定向到第三方，你不能使用`url_for()`和`redirect()`函数，因为它们可能会将你的客户端发送到不受欢迎的地方。

一种解决方案是创建一个受限制的第三方域名列表，你的应用程序允许重定向到这些域名，并确保应用程序或底层第三方库执行的重定向都经过该列表的检查。

这可以通过在视图生成响应后、Quart 将响应发送回客户端之前调用的`after_request()`钩子来完成。如果应用程序尝试发送回一个`302`状态码，你可以检查其位置是否安全，给定一个域名和端口号列表：

```py
# quart_after_response.py
from quart import Quart, redirect
from quart.helpers import make_response
from urllib.parse import urlparse
app = Quart(__name__)
@app.route("/api")
async def my_microservice():
    return redirect("https://github.com:443/")
# domain:port
SAFE_DOMAINS = ["github.com:443", "google.com:443"]
@app.after_request
async def check_redirect(response):
    if response.status_code != 302:
        return response
    url = urlparse(response.location)
    netloc = url.netloc
    if netloc not in SAFE_DOMAINS:
        # not using abort() here or it'll break the hook
        return await make_response("Forbidden", 403)
    return response
if __name__ == "__main__":
    app.run(debug=True) 
```

## 清洗输入数据

除了处理不受信任数据的其他做法之外，我们可以确保字段本身符合我们的预期。面对上述示例，我们可能会想过滤掉任何分号，或者可能所有花括号，但这让我们处于必须考虑数据可能出现的所有错误格式的位置，并试图战胜恶意程序员和随机错误的独创性。

相反，我们应该专注于我们对我们数据外观的了解——而不是它不应该是什么样子。这是一个更窄的问题，答案通常更容易定义。例如，如果我们知道一个端点接受 ISBN 来查找一本书，那么我们知道我们只应该期望一个由 10 或 13 位数字组成的序列，可能带有分隔符。然而，对于人来说，数据清理要困难得多。

在[`github.com/kdeldycke/awesome-falsehood`](https://github.com/kdeldycke/awesome-falsehood)上，有一些关于程序员对各种主题的错误认识的精彩列表。这些列表并不旨在详尽或具有权威性，但它们有助于提醒我们，我们可能对人类信息工作方式存在错误观念。人类姓名、邮政地址、电话号码：我们不应假设这些数据的任何样子，它们有多少行，或元素排列的顺序。我们能做的最好的事情是确保输入信息的人有最好的机会检查其正确性，然后使用前面描述的引用和沙箱技术来避免任何事故。

即使是电子邮件地址的验证也非常复杂。允许的格式有很多不同的部分，并不是所有的电子邮件系统都支持这些部分。有一句经常引用的话是，验证电子邮件地址的最佳方式是尝试发送一封电子邮件，这种方法既被合法网站使用——发送电子邮件并通知你“已发送电子邮件以确认您的账户”——也被垃圾邮件发送者使用，他们向数百万个地址发送无意义的消息，并记录哪些地址没有返回错误。

总结来说，你应该始终将传入的数据视为潜在的威胁，将其视为可能注入你系统的攻击源。转义或删除任何特殊字符，避免在没有隔离层的情况下直接在数据库查询或模板中使用数据，并确保你的数据看起来是你预期的样子。

你还可以使用 Bandit 代码检查器持续检查代码中的潜在安全问题，这在下一节中进行了探讨。

## 使用 Bandit 代码检查器

由 Python 代码质量权威机构管理，Bandit（[`github.com/PyCQA/bandit`](https://github.com/PyCQA/bandit)）是另一个用于扫描源代码中潜在安全风险的工具。它可以在 CI 系统中运行，以在部署之前自动测试任何更改。该工具使用`ast`模块以与`flake8`和`pylint`相同的方式解析代码。Bandit 还将扫描你代码中的一些已知安全问题。一旦使用`pip install bandit`命令安装它，你就可以使用`bandit`命令针对你的 Python 模块运行它。

如第三章所述，将 Bandit 添加到与其他检查并行的持续集成管道中，即《编码、测试和文档：良性循环》，是捕捉代码中潜在安全问题的好方法。

## 依赖项

大多数项目都会使用其他库，因为程序员是在他人的工作上构建的，而且通常没有足够的时间密切关注那些其他项目的发展。如果我们的依赖项中存在安全漏洞，我们希望快速了解这一点，以便我们可以更新我们的软件，而无需手动检查。

Dependabot ([`dependabot.com/`](https://dependabot.com/)) 是一个会对你的项目依赖进行安全扫描的工具。Dependabot 是 GitHub 的内置组件，其报告应显示在你的项目 **安全** 选项卡中。在项目的 **设置** 页面上开启一些额外功能，可以让 Dependabot 自动创建需要进行的任何更改以保持安全的拉取请求。

PyUp 拥有一组类似的功能，但需要手动设置——如果你不使用 GitHub，Dependabot 也是如此。

# 网络应用防火墙

即使数据处理得再安全，我们的应用程序仍然可能容易受到攻击。当你向世界公开 HTTP 端点时，这始终是一个风险。你希望调用者按预期行事，每个 HTTP 会话都遵循你在服务中编程的场景。

一个客户端可以发送合法的请求，并不断地用这些请求轰炸你的服务，导致由于所有资源都用于处理来自攻击者的请求而出现 **服务拒绝** （**DoS**）。当使用数百或数千个客户端进行此类操作时，这被称为 **分布式拒绝服务** （**DDoS**） 攻击。当客户端具有自动回放相同 API 的功能时，这个问题有时会在分布式系统中发生。如果客户端侧没有采取措施来限制调用，你可能会遇到由合法客户端过载的服务。

在服务器端添加保护以使这些热情的客户退却通常并不困难，并且这可以大大保护你的微服务堆栈。一些云服务提供商还提供针对 DDoS 攻击的保护以及这里提到的许多功能。

在本章前面提到的 OWASP 提供了一套规则，可用于 `ModSecurity` 工具包的 WAF，以避免许多类型的攻击：[`github.com/coreruleset/coreruleset/`](https://github.com/coreruleset/coreruleset/)。

在本节中，我们将专注于创建一个基本的 WAF，该 WAF 将明确拒绝在我们的服务上请求过多的客户端。本节的目的不是创建一个完整的 WAF，而是让你更好地理解 WAF 的实现和使用方式。我们可以在 Python 微服务中构建我们的 WAF，但如果所有流量都必须通过它，这将增加很多开销。一个更好的解决方案是直接依赖 Web 服务器。

## OpenResty：Lua 和 nginx

OpenResty ([`openresty.org/en/`](http://openresty.org/en/)) 是一个嵌入 Lua ([`www.lua.org/`](http://www.lua.org/)) 解释器的 `nginx` 发行版，可以用来编写 Web 服务器脚本。然后我们可以使用脚本将规则和过滤器应用于流量。

Lua 是一种优秀、动态类型的编程语言，它拥有轻量级且快速的解释器。该语言提供了一套完整的特性，并内置了异步特性。你可以在纯 Lua 中直接编写协程。

如果你安装了 Lua（参考 [`www.lua.org/start.html`](http://www.lua.org/start.html)），你可以使用 Lua **读取-评估-打印循环**（**REPL**）来玩转这门语言，就像使用 Python 一样：

```py
$ lua 
Lua 5.4.2  Copyright (C) 1994-2020 Lua.org, PUC-Rio
> io.write("Hello world\n")
Hello world
file (0x7f5a66f316a0)
> mytable = {}
> mytable["user"] = "simon"
> = mytable["user"]
simon
> = string.upper(mytable["user"])
SIMON
> 
```

要了解 Lua 语言，这是你的起点页面：[`www.lua.org/docs.html`](http://www.lua.org/docs.html)。

Lua 经常是嵌入到编译应用程序中的首选语言。它的内存占用非常小，并且允许快速动态脚本功能——这就是在 `OpenResty` 中发生的事情。你不需要构建 `nginx` 模块，而是可以使用 Lua 脚本来扩展 Web 服务器，并通过 OpenResty 直接部署它们。

当你从你的 `nginx` 配置中调用一些 Lua 代码时，OpenResty 使用的 `LuaJIT`（[`luajit.org/`](http://luajit.org/)）解释器将运行它们，运行速度与 `nginx` 代码本身相同。一些性能基准测试发现，在某些情况下 Lua 的速度可能比 C 或 C++ 快；请参考：[`luajit.org/performance.html`](http://luajit.org/performance.html)。

Lua 函数是协程，因此将在 `nginx` 中异步运行。这导致即使服务器收到大量并发请求时，开销也很低，这正是 WAF 所需要的。

OpenResty 以 Docker 镜像和一些 Linux 发行版的软件包的形式提供。如果需要，也可以从源代码编译；请参考 [`openresty.org/en/installation.html`](http://openresty.org/en/installation.html)。

在 macOS 上，你可以使用 `Brew` 和 `brew install openresty` 命令。

安装 OpenResty 后，你将获得一个 `openresty` 命令，它可以像 `nginx` 一样使用来服务你的应用程序。在以下示例中，`nginx` 配置将代理请求到运行在端口 `5000` 上的 Quart 应用程序：

```py
# resty.conf
daemon off;
worker_processes  1;
error_log /dev/stdout info;
events {
worker_connections  1024;
}
http {
  access_log /dev/stdout;
  server {
    listen   8888;
    server_name  localhost;
    location / {
      proxy_pass http://localhost:5000;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
   }
  } 
```

此配置可以使用 `openresty` 命令行，并在端口 `8888` 上以前台（守护进程关闭）模式运行，以代理转发到运行在端口 `5000` 上的 Quart 应用程序：

```py
$ openresty -p $(pwd) -c resty.conf
2021/07/03 16:11:08 [notice] 44691#12779096: using the "kqueue" event method
2021/07/03 16:11:08 [warn] 44691#12779096: 1024 worker_connections exceed open file resource limit: 256
nginx: [warn] 1024 worker_connections exceed open file resource limit: 256
2021/07/03 16:11:08 [notice] 44691#12779096: openresty/1.19.3.2
2021/07/03 16:11:08 [notice] 44691#12779096: built by clang 12.0.0 (clang-1200.0.32.2)
2021/07/03 16:11:08 [notice] 44691#12779096: OS: Darwin 19.6.0
2021/07/03 16:11:08 [notice] 44691#12779096: hw.ncpu: 12
2021/07/03 16:11:08 [notice] 44691#12779096: net.inet.tcp.sendspace: 131072
2021/07/03 16:11:08 [notice] 44691#12779096: kern.ipc.somaxconn: 128
2021/07/03 16:11:08 [notice] 44691#12779096: getrlimit(RLIMIT_NOFILE): 256:9223372036854775807
2021/07/03 16:11:08 [notice] 44691#12779096: start worker processes
2021/07/03 16:11:08 [notice] 44691#12779096: start worker process 44692 
```

注意，此配置也可以用于普通的 `nginx` 服务器，因为我们还没有使用任何 Lua。这就是 OpenResty 的一个优点：它是 `nginx` 的直接替换品，可以运行你的现有配置文件。

本节中展示的代码和配置可以在 [`github.com/PacktPublishing/Python-Microservices-Development-2nd-Edition/tree/main/CodeSamples`](https://github.com/PacktPublishing/Python-Microservices-Development-2nd-Edition/tree/main/CodeSample) 找到。

Lua 可以在请求到来时被调用；本章中最吸引人的两个时刻是：

+   `access_by_lua_block`: 这在构建响应之前对每个传入请求进行调用，并且是我们构建 WAF 访问规则的地方。

+   `content_by_lua_block`: 这使用 Lua 生成响应

让我们看看如何对传入请求进行速率限制。

## 速率和并发限制

速率限制包括在给定时间段内统计服务器接受的请求数量，并在达到限制时拒绝新的请求。

并发限制包括统计由 Web 服务器为同一远程用户服务的并发请求数量，并在达到定义的阈值时拒绝新的请求。由于许多请求可以同时到达服务器，并发限制器需要在阈值中留有小的余量。

这些技术在我们知道应用可以同时响应多少请求的上限时，可以避免应用内部出现任何问题，并且这可能是跨多个应用实例进行负载均衡的一个因素。这两个功能都是使用相同的技巧实现的。让我们看看如何构建一个并发限制器。

OpenResty 附带了一个用 Lua 编写的速率限制库，名为`lua-resty-limit-traffic`；你可以在`access_by_lua_block`部分中使用它：[`github.com/openresty/lua-resty-limit-traffic`](https://github.com/openresty/lua-resty-limit-traffic)。

该函数使用 Lua 的*Shared Dict*，这是一个由同一进程内的所有`nginx`工作进程共享的内存映射。使用内存字典意味着速率限制将在进程级别上工作。

由于我们通常在每个服务节点上部署一个`nginx`，因此速率限制将按每个 Web 服务器进行。所以，如果你为同一个微服务部署了多个节点，我们的有效速率限制将是单个节点可以处理的连接数乘以节点数——这在决定整体速率限制和微服务可以处理的并发请求数量时将非常重要。

在以下示例中，我们添加了一个`lua_shared_dict`定义和一个名为`access_by_lua_block`的部分来激活速率限制。请注意，这个示例是项目文档中示例的简化版本：

```py
# resty_limiting.conf
daemon off;
worker_processes  1;
error_log /dev/stdout info;
events {
    worker_connections  1024;
}
http {
    lua_shared_dict my_limit_req_store 100m;

    server {
        listen   8888;
        server_name  localhost;
        access_log /dev/stdout;
        location / {
            access_by_lua_block {
                local limit_req = require "resty.limit.req"
                local lim, err = limit_req.new("my_limit_req_store", 200, 100)
                local key = ngx.var.binary_remote_addr
                local delay, err = lim:incoming(key, true)
                if not delay then
                    if err == "rejected" then
                        return ngx.exit(503)
                    end
                    ngx.log(ngx.ERR, "failed to limit req: ", err)
                    return ngx.exit(500)
                end

                if delay >= 0.001 then
                    local excess = err
                    ngx.sleep(delay)
                end
            }
            proxy_pass http://localhost:5000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
} 
```

`access_by_lua_block`部分可以被视为一个 Lua 函数，并且可以使用`OpenResty`公开的一些变量和函数。例如，`ngx.var`是一个包含所有`nginx`变量的表，而`ngx.exit()`是一个可以用来立即向用户返回响应的函数——在我们的例子中，当因为速率限制需要拒绝调用时，返回一个`503`。

该库使用传递给`resty.limit.req`函数的`my_limit_req_store`字典；每次请求到达服务器时，它都会使用`binary_remote_addr`值调用`incoming()`函数，这是客户端地址。

`incoming()`函数将使用共享字典来维护每个远程地址的活跃连接数，并在该数字达到阈值时返回一个拒绝值；例如，当并发请求超过`300`时。

如果连接被接受，`incoming()` 函数会发送回一个延迟值。Lua 将使用该延迟和异步的 `ngx.sleep()` 函数保持请求。当远程客户端未达到 `200` 的阈值时，延迟将为 `0`，当在 `200` 和 `300` 之间时，会有一个小的延迟，这样服务器就有机会处理所有挂起的请求。

这种设计非常高效，可以防止服务因过多请求而超负荷。设置这样的上限也是避免达到一个你知道你的微服务将开始崩溃的点的好方法。例如，如果你的基准测试得出结论，你的服务在开始崩溃之前无法处理超过 100 个并发请求，你可以适当地设置速率限制，这样 `nginx` 就会拒绝请求，而不是让你的 Quart 微服务尝试处理所有这些传入的连接，然后再拒绝它们。

在这个例子中，用于计算速率的关键是请求的远程地址头。如果你的 `nginx` 服务器本身位于代理后面，确保你使用包含真实远程地址的头。否则，你将对单个远程客户端和代理服务器进行速率限制。在这种情况下，通常在 `X-Forwarded-For` 头中。

如果你需要一个功能更丰富的 WAF，`lua-resty-waf` ([`github.com/p0pr0ck5/lua-resty-waf`](https://github.com/p0pr0ck5/lua-resty-waf)) 项目就像 `lua-resty-limit-traffic` 一样工作，但提供了很多其他保护。它还能够读取 `ModSecurity` 规则文件，因此你可以使用 OWASP 项目中的规则文件，而无需使用 `ModSecurity` 本身。

## 其他 OpenResty 功能

OpenResty 内置了许多 Lua 脚本，这些脚本可以用来增强 `nginx`。一些开发者甚至用它来直接提供数据。以下组件页面包含了一些有用的工具，用于让 `nginx` 与数据库、缓存服务器等交互：[`openresty.org/en/components.html`](http://openresty.org/en/components.html)。

此外，还有一个网站供社区发布 OpenResty 组件：[`opm.openresty.org/`](https://opm.openresty.org/)。

如果你正在你的 Quart 微服务前面使用 OpenResty，可能还有其他用例，你可以将 Quart 应用中的一些代码转移到 OpenResty 的几行 Lua 代码中。目标不应该是将应用的逻辑移动到 OpenResty，而应该是利用 Web 服务器在调用你的 Quart 应用之前或之后执行任何可以执行的操作。让 Python 专注于应用逻辑，而 OpenResty 则专注于一层保护。

例如，如果你正在使用 Redis 或 Memcached 服务器来缓存一些`GET`资源，你可以直接从 Lua 调用它们，为特定的端点添加或检索缓存的版本。`srcache-nginx-module`（[`github.com/openresty/srcache-nginx-module`](https://github.com/openresty/srcache-nginx-module)）就是这样一种行为的实现，如果你能缓存它们，它将减少对 Quart 应用程序的`GET`调用次数。

要总结关于 WAF 的这一部分：OpenResty 是一个强大的`nginx`发行版，可以用来创建一个简单的 WAF 来保护你的微服务。它还提供了超出防火墙功能的特性。实际上，如果你采用 OpenResty 来运行你的微服务，Lua 将打开一个全新的可能性世界。

# 摘要

在本章中，我们探讨了如何在基于微服务应用程序环境中使用 OAuth2 和 JWT 来集中式地处理认证和授权。令牌赋予我们限制调用者使用某个微服务的能力，以及他们可以持续使用多长时间。

当与公钥和私钥一起使用时，它还限制了攻击者在整个应用程序的一个组件被攻破时可能造成的损害。它还确保每个连接都经过加密验证。

安全的代码库是构建安全应用程序的第一步。你应该遵循良好的编码实践，并确保你的代码在与传入的用户数据和资源交互时不会做任何坏事。虽然像 Bandit 这样的工具不能保证你代码的安全性和安全性，但它会捕捉到最明显的潜在安全问题，因此你完全没有必要犹豫是否在你的代码库上持续运行它。

最后，WAF（Web 应用防火墙）也是防止端点上的某些欺诈和滥用的好方法，使用像 OpenResty 这样的工具来做这件事非常简单，这要归功于 Lua 编程语言的力量。

OpenResty 也是通过在 Web 服务器级别做一些事情来赋予和加速你的微服务的好方法，当这些事情不需要在 Quart 应用程序内部完成时。
