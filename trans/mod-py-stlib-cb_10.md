# 第十章：网络

在本章中，我们将涵盖以下内容：

+   发送电子邮件-从您的应用程序发送电子邮件

+   获取电子邮件-检查并阅读新收到的邮件

+   FTP-从 FTP 上传、列出和下载文件

+   套接字-基于 TCP/IP 编写聊天系统

+   AsyncIO-基于协程的异步 HTTP 服务器，用于静态文件

+   远程过程调用-通过 XMLRPC 实现 RPC

# 介绍

现代应用程序经常需要通过网络与用户或其他软件进行交互。我们的社会越向连接的世界发展，用户就越希望软件能够与远程服务或网络进行交互。

基于网络的应用程序依赖于几十年来稳定且经过广泛测试的工具和范例，Python 标准库提供了对从传输到应用程序协议的最常见技术的支持。

除了提供对通信通道本身（如套接字）的支持外，标准库还提供了实现基于事件的应用程序模型，这些模型是网络使用案例的典型，因为在大多数情况下，应用程序将不得不对来自网络的输入做出反应并相应地处理它。

在本章中，我们将看到如何处理一些最常见的应用程序协议，如 SMTP、IMAP 和 FTP。但我们还将看到如何通过套接字直接处理网络，并如何实现我们自己的 RPC 通信协议。

# 发送电子邮件

电子邮件是当今最广泛使用的通信工具，如果您在互联网上，几乎可以肯定您有一个电子邮件地址，它们现在也高度集成在智能手机中，因此可以随时随地访问。

出于所有这些原因，电子邮件是向用户发送通知、完成报告和长时间运行进程结果的首选工具。

发送电子邮件需要一些机制，如果您想自己支持 SMTP 和 MIME 协议，这两种协议都相当复杂。

幸运的是，Python 标准库内置支持这两种情况，我们可以依赖`smtplib`模块与 SMTP 服务器交互以发送我们的电子邮件，并且可以依赖`email`包来实际创建电子邮件的内容并处理所需的所有特殊格式和编码。

# 如何做...

发送电子邮件是一个三步过程：

1.  联系 SMTP 服务器并对其进行身份验证

1.  准备电子邮件本身

1.  向 SMTP 服务器提供电子邮件

Python 标准库中涵盖了所有三个阶段，我们只需要将它们包装起来，以便在更简单的接口中方便使用：

```py
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from smtplib import SMTP

class EmailSender:
    def __init__(self, host="localhost", port=25, login="", password=""):
        self._host = host
        self._port = int(port)
        self._login = login
        self._password = password

    def send(self, sender, recipient, subject, body):
        header_charset = 'UTF-8'
        body_charset = 'UTF-8'

        sender_name, sender_addr = parseaddr(sender)
        recipient_name, recipient_addr = parseaddr(recipient)

        sender_name = str(Header(sender_name, header_charset))
        recipient_name = str(Header(recipient_name, header_charset))

        msg = MIMEText(body.encode(body_charset), 'plain', body_charset)
        msg['From'] = formataddr((sender_name, sender_addr))
        msg['To'] = formataddr((recipient_name, recipient_addr))
        msg['Subject'] = Header(subject, header_charset)

        smtp = SMTP(self._host, self._port)
        try:
            smtp.starttls()
        except:
            pass
        smtp.login(self._login, self._password)
        smtp.sendmail(sender, recipient, msg.as_string())
        smtp.quit()
```

我们的`EmailSender`类可用于轻松通过我们的电子邮件提供商发送电子邮件。

```py
es = EmailSender('mail.myserver.it', 
                 login='amol@myserver.it', 
                 password='mymailpassword')
es.send(sender='Sender <no-reply@senders.net>', 
        recipient='amol@myserver.it',
        subject='Hello my friend!',
        body='''Here is a little email for you''')
```

# 它是如何工作的...

发送电子邮件需要连接到 SMTP 服务器，这需要数据，如服务器运行的主机、暴露的端口以及用于对其进行身份验证的用户名和密码。

每次我们想要发送电子邮件时，都需要所有这些细节，因为每封电子邮件都需要单独的连接。因此，这些都是我们负责发送电子邮件的类始终需要可用的所有细节，并且在创建实例时请求：

```py
class EmailSender:
    def __init__(self, host="localhost", port=25, login="", password=""):
        self._host = host
        self._port = int(port)
        self._login = login
        self._password = password
```

一旦知道连接到 SMTP 服务器所需的所有细节，我们类的唯一公开方法就是实际发送电子邮件的方法：

```py
def send(self, sender, recipient, subject, body):
```

这需要组成电子邮件所需的细节：发件人地址、接收电子邮件的地址、主题和电子邮件内容本身。

我们的方法必须解析提供的发件人和收件人。包含发件人和收件人名称的部分与包含地址的部分是分开的：

```py
sender_name, sender_addr = parseaddr(sender)
recipient_name, recipient_addr = parseaddr(recipient)
```

如果`sender`类似于`"Alessandro Molina <amol@myserver.it>"`，`sender_name`将是`"Alessandro Molina"`，`sender_addr`将是`"amol@myserver.it"`。

这是必需的，因为名称部分通常包含不受限于纯 ASCII 的名称，邮件可能会发送到中国、韩国或任何其他需要正确支持 Unicode 以处理收件人名称的地方。

因此，我们必须以一种邮件客户端在接收电子邮件时能够理解的方式正确编码这些字符，这是通过使用提供的字符集编码的`Header`类来完成的，在我们的情况下是`"UTF-8"`：

```py
sender_name = str(Header(sender_name, header_charset))
recipient_name = str(Header(recipient_name, header_charset))
```

一旦发件人和收件人的名称以电子邮件标题所期望的格式进行编码，我们就可以将它们与地址部分结合起来，以构建回一个完整的收件人和发件人，形式为`"Name <address>"`：

```py
msg['From'] = formataddr((sender_name, sender_addr))
msg['To'] = formataddr((recipient_name, recipient_addr))
```

相同的情况也适用于“主题”，作为邮件的一个标题字段，也需要进行编码：

```py
msg['Subject'] = Header(subject, header_charset)
```

相反，消息的正文不必作为标题进行编码，并且可以以任何编码的纯字节表示形式提供，只要指定了编码。

在我们的情况下，消息的正文也被编码为`UTF-8`：

```py
msg = MIMEText(body.encode(body_charset), 'plain', body_charset)
```

然后，一旦消息本身准备就绪，正文和标题都被正确编码，唯一剩下的部分就是实际与 SMTP 服务器取得联系并发送电子邮件。

这是通过创建一个已知地址和端口的`SMTP`对象来完成的：

```py
smtp = SMTP(self._host, self._port)
```

然后，如果 SMTP 服务器支持 TLS 加密，我们就启动它。如果不支持，我们就忽略错误并继续：

```py
try:
    smtp.starttls()
except:
    pass
```

一旦启用了加密（如果可用），我们最终可以对 SMTP 服务器进行身份验证，并将邮件本身发送给相关的收件人：

```py
smtp.login(self._login, self._password)
smtp.sendmail(sender, recipient, msg.as_string())
smtp.quit()
```

为了测试编码是否按预期工作，您可以尝试发送一封包含标准 ASCII 字符之外字符的电子邮件，以查看您的客户端是否正确理解了电子邮件：

```py
es.send(sender='Sender <no-reply@senders.net>', 
        recipient='amol@myserver.it',
        subject='Have some japanese here: ã“ã‚“ã«ã¡ã¯',
        body='''And some chinese here! ä½ å¥½''')
```

如果一切都按预期进行，您应该能够对 SMTP 提供程序进行身份验证，发送电子邮件，并在收件箱中看到具有适当内容的电子邮件。

# 获取电子邮件

经常情况下，应用程序需要对某种事件做出反应，它们接收来自用户或软件的消息，然后需要相应地采取行动。基于网络的应用程序的整体性质在于对接收到的消息做出反应，但这类应用程序的一个非常特定和常见的情况是需要对接收到的电子邮件做出反应。

典型情况是，当用户需要向您的应用程序发送某种文档（通常是身份证或签署的合同）时，您希望对该事件做出反应，例如在用户发送签署的合同后启用服务。

这要求我们能够访问收到的电子邮件并扫描它们以检测发件人和内容。

# 如何做...

这个食谱的步骤如下：

1.  使用`imaplib`和`email`模块，可以构建一个工作的 IMAP 客户端，从支持的 IMAP 服务器中获取最近的消息：

```py
import imaplib
import re
from email.parser import BytesParser

class IMAPReader:
    ENCODING = 'utf-8'
    LIST_PATTERN = re.compile(
        r'\((?P<flags>.*?)\) "(?P<delimiter>.*)" (?P<name>.*)'
    )

    def __init__(self, host, username, password, ssl=True):
        if ssl:
            self._imap = imaplib.IMAP4_SSL(host)
        else:
            self._imap = imaplib.IMAP4(host)
        self._imap.login(username, password)

    def folders(self):
        """Retrieve list of IMAP folders"""
        resp, lines = self._imap.list()
        if resp != 'OK':
            raise Exception(resp)

        entries = []
        for line in lines:
            flags, _, name = self.LIST_PATTERN.match(
                line.decode(self.ENCODING)
            ).groups()
            entries.append(dict(
                flags=flags,
                name=name.strip('"')
            ))
        return entries

    def messages(self, folder, limit=10, peek=True):
        """Return ``limit`` messages from ``folder``

        peek=False will also fetch message body
        """
        resp, count = self._imap.select('"%s"' % folder, readonly=True)
        if resp != 'OK':
            raise Exception(resp)

        last_message_id = int(count[0])
        msg_ids = range(last_message_id, last_message_id-limit, -1)

        mode = '(BODY.PEEK[HEADER])' if peek else '(RFC822)'

        messages = []
        for msg_id in msg_ids:
            resp, msg = self._imap.fetch(str(msg_id), mode)
            msg = msg[0][-1]

            messages.append(BytesParser().parsebytes(msg))
            if len(messages) >= limit:
                break
        return messages

    def get_message_body(self, message):
        """Given a message for which the body was fetched, returns it"""
        body = []
        if message.is_multipart():
            for payload in message.get_payload():
                body.append(payload.get_payload())
        else:
            body.append(message.get_payload())
        return body

    def close(self):
        """Close connection to IMAP server"""
        self._imap.close()
```

1.  然后可以使用`IMAPReader`访问兼容的邮件服务器以阅读最近的电子邮件：

```py
mails = IMAPReader('imap.gmail.com', 
                   YOUR_EMAIL, YOUR_PASSWORD,
                   ssl=True)

folders = mails.folders()
for msg in mails.messages('INBOX', limit=2, peek=True):
    print(msg['Date'], msg['Subject'])
```

1.  这返回了最近两封收到的电子邮件的标题和时间戳：

```py
Fri, 8 Jun 2018 00:07:16 +0200 Hello Python CookBook!
Thu, 7 Jun 2018 08:21:11 -0400 SSL and turbogears.org
```

如果我们需要实际的电子邮件内容和附件，我们可以通过使用`peek=False`来检索它们，然后在检索到的消息上调用`IMAPReader.get_message_body`。

# 它的工作原理是...

我们的类充当了`imaplib`和`email`模块的包装器，为从文件夹中获取邮件的需求提供了一个更易于使用的接口。

实际上，可以从`imaplib`创建两种不同的对象来连接到 IMAP 服务器，一种使用 SSL，一种不使用。根据服务器的要求，您可能需要打开或关闭它（例如，Gmail 需要 SSL），这在`__init__`中进行了抽象处理：

```py
def __init__(self, host, username, password, ssl=True):
    if ssl:
        self._imap = imaplib.IMAP4_SSL(host)
    else:
        self._imap = imaplib.IMAP4(host)
    self._imap.login(username, password)
```

`__init__`方法还负责登录到 IMAP 服务器，因此一旦创建了阅读器，它就可以立即使用。

然后我们的阅读器提供了列出文件夹的方法，因此，如果您想要从所有文件夹中读取消息，或者您想要允许用户选择文件夹，这是可能的：

```py
def folders(self):
    """Retrieve list of IMAP folders"""
```

我们的`folders`方法的第一件事是从服务器获取文件夹列表。`imaplib`方法已经在出现错误时报告异常，但作为安全措施，我们还检查响应是否为`OK`：

```py
resp, lines = self._imap.list()
if resp != 'OK':
    raise Exception(resp)
```

IMAP 是一种基于文本的协议，服务器应该始终响应`OK <response>`，如果它能够理解您的请求并提供响应。否则，可能会返回一堆替代响应代码，例如`NO`或`BAD`。如果返回了其中任何一个，我们认为我们的请求失败了。

一旦我们确保实际上有文件夹列表，我们需要解析它。列表由多行文本组成。每行包含有关一个文件夹的详细信息，这些详细信息：标志和文件夹名称。它们由一个分隔符分隔，这不是标准的。在某些服务器上，它是一个点，而在其他服务器上，它是一个斜杠，因此我们在解析时需要非常灵活。这就是为什么我们使用允许标志和名称由任何分隔符分隔的正则表达式来解析它：

```py
LIST_PATTERN = re.compile(
    r'\((?P<flags>.*?)\) "(?P<delimiter>.*)" (?P<name>.*)'
)
```

一旦我们知道如何解析响应中的这些行，我们就可以根据它们构建一个包含名称和这些文件夹的标志的字典列表：

```py
entries = []
for line in lines:
    flags, _, name = self.LIST_PATTERN.match(
        line.decode(self.ENCODING)
    ).groups()
    entries.append(dict(
        flags=flags,
        name=name.strip('"')
    ))
return entries
```

然后可以使用`imaplib.ParseFlags`类进一步解析这些标志。

一旦我们知道要获取消息的文件夹的名称，我们就可以通过`messages`方法检索消息：

```py
def messages(self, folder, limit=10, peek=True):
    """Return ``limit`` messages from ``folder``

    peek=False will also fetch message body
    """
```

由于 IMAP 是一种有状态的协议，我们需要做的第一件事是选择我们想要运行后续命令的文件夹：

```py
resp, count = self._imap.select('"%s"' % folder, readonly=True)
if resp != 'OK':
    raise Exception(resp)
```

我们提供一个`readonly`选项，这样我们就不会无意中销毁我们的电子邮件，并像往常一样验证响应代码。

然后`select`方法的响应内容实际上是上传到该文件夹的最后一条消息的 ID。

由于这些 ID 是递增的数字，我们可以使用它来生成要获取的最近消息的最后`limit`条消息的 ID：

```py
last_message_id = int(count[0])
msg_ids = range(last_message_id, last_message_id-limit, -1)
```

然后，根据调用者的选择，我们选择要下载的消息的内容。如果只有标题或整个内容：

```py
mode = '(BODY.PEEK[HEADER])' if peek else '(RFC822)'
```

模式将被提供给`fetch`方法，告诉它我们要下载什么数据：

```py
resp, msg = self._imap.fetch(str(msg_id), mode)
```

然后，消息本身被组合成一个包含两个元素的元组列表。第一个元素包含消息返回的大小和模式（由于我们自己提供了模式，所以我们并不真的在乎），元组的最后一个元素包含消息本身，所以我们只需抓取它：

```py
msg = msg[0][-1]
```

一旦我们有了可用的消息，我们将其提供给`BytesParser`，以便我们可以得到一个`Message`实例：

```py
BytesParser().parsebytes(msg)
```

我们循环遍历所有消息，解析它们，并添加到我们将返回的消息列表中。一旦达到所需数量的消息，我们就停止：

```py
messages = []
for msg_id in msg_ids:
    resp, msg = self._imap.fetch(str(msg_id), mode)
    msg = msg[0][-1]

    messages.append(BytesParser().parsebytes(msg))
    if len(messages) >= limit:
        break
return messages
```

从`messages`方法中，我们得到一个`Message`对象的列表，我们可以轻松访问除消息正文之外的所有数据。因为正文实际上可能由多个项目组成（想象一条带附件的消息 - 它包含文本、图像、PDF 文件或任何附件）。

因此，读取器提供了一个`get_message_body`方法，用于检索消息正文的所有部分（如果是多部分消息），并将它们返回：

```py
def get_message_body(self, message):
    """Given a message for which the body was fetched, returns it"""
    body = []
    if message.is_multipart():
        for payload in message.get_payload():
            body.append(payload.get_payload())
    else:
        body.append(message.get_payload())
    return body
```

通过结合`messages`和`get_message_body`方法，我们能够从邮箱中抓取消息及其内容，然后根据需要对其进行处理。

# 还有更多...

编写一个功能完备且完全运行的 IMAP 客户端是一个独立的项目，超出了本书的范围。

IMAP 是一个复杂的协议，包括对标志、搜索和许多其他功能的支持。大多数这些命令都由`imaplib`提供，还可以上传消息到服务器或创建工具来执行备份或将消息从一个邮件帐户复制到另一个邮件帐户。

此外，当解析复杂的电子邮件时，`email`模块将处理电子邮件相关的 RFCs 指定的各种数据表示，例如，我们的示例将日期返回为字符串，但`email.utils.parsedate`可以将其解析为 Python 对象。

# FTP

FTP 是保存和从远程服务器检索文件的最广泛使用的解决方案。它已经存在了几十年，是一个相当容易使用的协议，可以提供良好的性能，因为它在传输内容上提供了最小的开销，同时支持强大的功能，如传输恢复。

通常，软件需要接收由其他软件自动上传的文件；多年来，FTP 一直被频繁地用作这些场景中的强大解决方案。无论您的软件是需要上传内容的软件，还是需要接收内容的软件，Python 标准库都内置了对 FTP 的支持，因此我们可以依靠`ftplib`来使用 FTP 协议。

# 如何做到这一点...

`ftplib`是一个强大的基础，我们可以在其上提供一个更简单的 API 来与 FTP 服务器进行交互，用于存储和检索文件：

```py
import ftplib

class FTPCLient:
    def __init__(self, host, username='', password=''):
        self._client = ftplib.FTP_TLS(timeout=10)
        self._client.connect(host)

        # enable TLS
        try:
            self._client.auth()
        except ftplib.error_perm:
            # TLS authentication not supported
            # fallback to a plain FTP client
            self._client.close()
            self._client = ftplib.FTP(timeout=10)
            self._client.connect(host)

        self._client.login(username, password)

        if hasattr(self._client, 'prot_p'):
            self._client.prot_p()

    def cwd(self, directory):
        """Enter directory"""
        self._client.cwd(directory)

    def dir(self):
        """Returns list of files in current directory.

        Each entry is returned as a tuple of two elements,
        first element is the filename, the second are the
        properties of that file.
        """
        entries = []
        for idx, f in enumerate(self._client.mlsd()):
            if idx == 0:
                # First entry is current path
                continue
            if f[0] in ('..', '.'):
                continue
            entries.append(f)
        return entries

    def download(self, remotefile, localfile):
        """Download remotefile into localfile"""
        with open(localfile, 'wb') as f:
            self._client.retrbinary('RETR %s' % remotefile, f.write)

    def upload(self, localfile, remotefile):
        """Upload localfile to remotefile"""
        with open(localfile, 'rb') as f:
            self._client.storbinary('STOR %s' % remotefile, f)

    def close(self):
        self._client.close()
```

然后，我们可以通过上传和获取一个简单的文件来测试我们的类：

```py
with open('/tmp/hello.txt', 'w+') as f:
    f.write('Hello World!')

cli = FTPCLient('localhost', username=USERNAME, password=PASSWORD)
cli.upload('/tmp/hello.txt', 'hellofile.txt')    
cli.download('hellofile.txt', '/tmp/hello2.txt')

with open('/tmp/hello2.txt') as f:
    print(f.read())
```

如果一切按预期工作，输出应该是`Hello World!`

# 工作原理...

`FTPClient`类提供了一个初始化程序，负责设置与服务器的正确连接以及一堆方法来实际对连接的服务器进行操作。

`__init__`做了很多工作，尝试建立与远程服务器的正确连接：

```py
def __init__(self, host, username='', password=''):
    self._client = ftplib.FTP_TLS(timeout=10)
    self._client.connect(host)

    # enable TLS
    try:
        self._client.auth()
    except ftplib.error_perm:
        # TLS authentication not supported
        # fallback to a plain FTP client
        self._client.close()
        self._client = ftplib.FTP(timeout=10)
        self._client.connect(host)

    self._client.login(username, password)

    if hasattr(self._client, 'prot_p'):
        self._client.prot_p()
```

首先它尝试建立 TLS 连接，这可以保证加密，否则 FTP 是一种明文协议，会以明文方式发送所有数据。

如果我们的远程服务器支持 TLS，可以通过调用`.auth()`在控制连接上启用它，然后通过调用`prot_p()`在数据传输连接上启用它。

FTP 基于两种连接，控制连接用于发送和接收服务器的命令及其结果，数据连接用于发送上传和下载的数据。

如果可能的话，它们两者都应该加密。如果我们的服务器不支持它们，我们将退回到普通的 FTP 连接，并继续通过对其进行身份验证来进行操作。

如果您的服务器不需要任何身份验证，提供`anonymous`作为用户名，空密码通常足以登录。

一旦我们连接上了，我们就可以自由地在服务器上移动，可以使用`cwd`命令来实现：

```py
def cwd(self, directory):
    """Enter directory"""
    self._client.cwd(directory)
```

这个方法只是内部客户端方法的代理，因为内部方法已经很容易使用并且功能齐全。

但一旦我们进入一个目录，我们需要获取它的内容，这就是`dir()`方法发挥作用的地方：

```py
def dir(self):
    """Returns list of files in current directory.

    Each entry is returned as a tuple of two elements,
    first element is the filename, the second are the
    properties of that file.
    """
    entries = []
    for idx, f in enumerate(self._client.mlsd()):
        if idx == 0:
            # First entry is current path
            continue
        if f[0] in ('..', '.'):
            continue
        entries.append(f)
    return entries
```

`dir()`方法调用内部客户端的`mlsd`方法，负责返回当前目录中文件的列表。

这个列表被返回为一个包含两个元素的元组：

```py
('Desktop', {'perm': 'ceflmp', 
             'unique': 'BAAAAT79CAAAAAAA', 
             'modify': '20180522213143', 
             'type': 'dir'})
```

元组的第一个条目包含文件名，而第二个条目包含其属性。

我们自己的方法只做了两个额外的步骤，它跳过了第一个返回的条目——因为那总是当前目录（我们用`cwd()`选择的目录）——然后跳过了任何特殊的父目录或当前目录的条目。我们对它们并不感兴趣。

一旦我们能够在目录结构中移动，我们最终可以将文件`upload`和`download`到这些目录中：

```py
def download(self, remotefile, localfile):
    """Download remotefile into localfile"""
    with open(localfile, 'wb') as f:
        self._client.retrbinary('RETR %s' % remotefile, f.write)

def upload(self, localfile, remotefile):
    """Upload localfile to remotefile"""
    with open(localfile, 'rb') as f:
        self._client.storbinary('STOR %s' % remotefile, f)
```

这两种方法非常简单，当我们上传文件时，它们只是打开本地文件进行读取，当我们下载文件时，它们只是打开本地文件进行写入，并发送 FTP 命令来检索或存储文件。

当上传一个新的`remotefile`时，将创建一个具有与`localfile`相同内容的文件。当下载时，将打开`localfile`以在其中写入`remotefile`的内容。

# 还有更多...

并非所有的 FTP 服务器都支持相同的命令。多年来，该协议进行了许多扩展，因此一些命令可能缺失或具有不同的语义。

例如，`mlsd`函数可能会缺失，但您可能有`LIST`或`nlst`，它们可以执行类似的工作。

您可以参考 RFC 959 了解 FTP 协议应该如何工作，但经常通过明确与您要连接的 FTP 服务器进行实验是评估它将接受哪些命令和签名的最佳方法。

经常，FTP 服务器实现了一个`HELP`命令，您可以使用它来获取支持的功能列表。

# 套接字

套接字是您可以用来编写网络应用程序的最低级别概念之一。这意味着我们通常要自己管理整个连接，当直接依赖套接字时，您需要处理连接请求，接受它们，然后启动一个线程或循环来处理通过新创建的连接通道发送的后续命令或数据。

这几乎所有依赖网络的应用程序都必须实现的流程，通常您调用服务器时都有一个基础在上述循环中。

Python 标准库提供了一个很好的基础，避免每次必须处理基于网络的应用程序时手动重写该流程。我们可以使用`socketserver`模块，让它为我们处理连接循环，而我们只需专注于实现应用程序层协议和处理消息。

# 如何做...

对于这个配方，您需要执行以下步骤：

1.  通过混合`TCPServer`和`ThreadingMixIn`类，我们可以轻松构建一个通过 TCP 处理并发连接的多线程服务器：

```py
import socket
import threading
import socketserver

class EchoServer:
    def __init__(self, host='0.0.0.0', port=9800):
        self._host = host
        self._port = port
        self._server = ThreadedTCPServer((host, port), EchoRequestHandler)
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True

    def start(self):
        if self._thread.is_alive():
            # Already serving
            return

        print('Serving on %s:%s' % (self._host, self._port))
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

class ThreadedTCPServer(socketserver.ThreadingMixIn, 
                        socketserver.TCPServer):
    allow_reuse_address = True

class EchoRequestHandler(socketserver.BaseRequestHandler):
    MAX_MESSAGE_SIZE = 2**16  # 65k
    MESSAGE_HEADER_LEN = len(str(MAX_MESSAGE_SIZE))

    @classmethod
    def recv_message(cls, socket):
        data_size = int(socket.recv(cls.MESSAGE_HEADER_LEN))
        data = socket.recv(data_size)
        return data

    @classmethod
    def prepare_message(cls, message):
        if len(message) > cls.MAX_MESSAGE_SIZE:
            raise ValueError('Message too big'

        message_size = str(len(message)).encode('ascii')
        message_size = message_size.zfill(cls.MESSAGE_HEADER_LEN)
        return message_size + message

    def handle(self):
        message = self.recv_message(self.request)
        self.request.sendall(self.prepare_message(b'ECHO: %s' % message))
```

1.  一旦我们有一个工作的服务器，为了测试它，我们需要一个客户端向其发送消息。为了方便起见，我们将保持客户端简单，只需连接，发送消息，然后等待一个简短的回复：

```py
def send_message_to_server(ip, port, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    try:
        message = EchoRequestHandler.prepare_message(message)
        sock.sendall(message)
        response = EchoRequestHandler.recv_message(sock)
        print("ANSWER: {}".format(response))
    finally:
        sock.close()
```

1.  现在我们既有服务器又有客户端，我们可以测试我们的服务器是否按预期工作：

```py
server = EchoServer()
server.start()

send_message_to_server('localhost', server._port, b"Hello World 1")
send_message_to_server('localhost', server._port, b"Hello World 2")
send_message_to_server('localhost', server._port, b"Hello World 3")

server.stop()
```

1.  如果一切正常，您应该看到：

```py
Serving on 0.0.0.0:9800
ANSWER: b'ECHO: Hello World 1'
ANSWER: b'ECHO: Hello World 2'
ANSWER: b'ECHO: Hello World 3'
```

# 它是如何工作的...

服务器部分由三个不同的类组成。

`EchoServer`，它编排服务器并提供我们可以使用的高级 API。`EchoRequestHandler`，它管理传入的消息并提供服务。`ThreadedTCPServer`，它负责整个网络部分，打开套接字，监听它们，并生成线程来处理连接。

`EchoServer`允许启动和停止我们的服务器：

```py
class EchoServer:
    def __init__(self, host='0.0.0.0', port=9800):
        self._host = host
        self._port = port
        self._server = ThreadedTCPServer((host, port), EchoRequestHandler)
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True

    def start(self):
        if self._thread.is_alive():
            # Already serving
            return

        print('Serving on %s:%s' % (self._host, self._port))
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
```

它创建一个新的线程，服务器将在其中运行并启动它（如果尚未运行）。该线程将只运行`ThreadedTCPServer.serve_forever`方法，该方法循环运行，依次为每个请求提供服务。

当我们完成服务器时，我们可以调用`stop()`方法，它将关闭服务器并等待其完成（一旦完成所有当前运行的请求，它将退出）。

`ThreadedTCPServer`基本上是标准库提供的标准服务器，如果不是因为我们也继承自`ThreadingMixIn`。`Mixin`是一组附加功能，您可以通过继承它来注入类中，在这种特定情况下，它为套接字服务器提供了线程功能。因此，我们可以同时处理多个请求，而不是一次只能处理一个请求。

我们还设置了服务器的`allow_reuse_address = True`属性，以便在发生崩溃或超时的情况下，套接字可以立即重用，而不必等待系统关闭它们。

最后，`EchoRequestHandler`提供了整个消息处理和解析。每当`ThreadedTCPServer`接收到新连接时，它将在处理程序上调用`handle`方法，由处理程序来执行正确的操作。

在我们的情况下，我们只是实现了一个简单的服务器，它会回复发送给它的内容，因此处理程序必须执行两件事：

+   解析传入的消息以了解其内容

+   发送一个具有相同内容的消息

在使用套接字时的一个主要复杂性是它们实际上并不是基于消息的。它们是一连串的数据（好吧，UDP 是基于消息的，但就我们而言，接口并没有太大变化）。这意味着不可能知道新消息何时开始以及消息何时结束。

`handle`方法只告诉我们有一个新连接，但在该连接上，可能会连续发送多条消息，除非我们知道消息何时结束，否则我们会将它们读取为一条大消息。

为了解决这个问题，我们使用了一个非常简单但有效的方法，即给所有消息加上它们自己的大小前缀。因此，当接收到新消息时，我们总是知道我们只需要读取消息的大小，然后一旦知道大小，我们将读取由大小指定的剩余字节。

要读取这些消息，我们依赖于一个实用方法`recv_message`，它将能够从任何提供的套接字中读取以这种方式制作的消息：

```py
@classmethod
def recv_message(cls, socket):
    data_size = int(socket.recv(cls.MESSAGE_HEADER_LEN))
    data = socket.recv(data_size)
    return data
```

该函数的第一件事是从套接字中精确读取`MESSAGE_HEADER_LEN`个字节。这些字节将包含消息的大小。所有大小必须相同。因此，诸如`10`之类的大小将必须表示为`00010`。然后前缀的零将被忽略。然后，该大小使用`int`进行转换，我们将得到正确的数字。大小必须全部相同，否则我们将不知道需要读取多少字节来获取大小。

我们决定将消息大小限制为 65,000，这导致`MESSAGE_HEADER_LEN`为五，因为需要五位数字来表示最多 65,536 的数字：

```py
MAX_MESSAGE_SIZE = 2**16  # 65k
MESSAGE_HEADER_LEN = len(str(MAX_MESSAGE_SIZE))
```

大小并不重要，我们只选择了一个相当大的值。允许的消息越大，就需要更多的字节来表示它们的大小。

然后`recv_message`方法由`handle()`使用来读取发送的消息：

```py
def handle(self):
    message = self.recv_message(self.request)
    self.request.sendall(self.prepare_message(b'ECHO: %s' % message))
```

一旦消息知道，`handle()`方法还会以相同的方式准备发送回一条新消息，并且为了准备响应，它依赖于`prepare_message`，这也是客户端用来发送消息的方法：

```py
@classmethod
def prepare_message(cls, message):
    if len(message) > cls.MAX_MESSAGE_SIZE:
        raise ValueError('Message too big'

    message_size = str(len(message)).encode('ascii')
    message_size = message_size.zfill(cls.MESSAGE_HEADER_LEN)
    return message_size + message
```

该函数的作用是，给定一条消息，它确保消息不会超过允许的最大大小，然后在消息前面加上它的大小。

该大小是通过将消息的长度作为文本获取，然后使用`ascii`编码将其编码为字节来计算的。由于大小只包含数字，因此`ascii`编码已经足够表示它们了：

```py
message_size = str(len(message)).encode('ascii')
```

由于生成的字符串可以有任何大小（从一到五个字节），我们总是用零填充它，直到达到预期的大小：

```py
message_size = message_size.zfill(cls.MESSAGE_HEADER_LEN)
```

然后将生成的字节添加到消息前面，并返回准备好的消息。

有了这两个函数，服务器就能够接收和发送任意大小的消息。

客户端函数的工作方式几乎相同，因为它必须发送一条消息，然后接收答案：

```py
def send_message_to_server(ip, port, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    try:
        message = EchoRequestHandler.prepare_message(message)
        sock.sendall(message)
        response = EchoRequestHandler.recv_message(sock)
        print("ANSWER: {}".format(response))
    finally:
        sock.close()
```

它仍然使用`EchoRequestHandler.prepare_message`来准备发送到服务器的消息，以及`EchoRequestHandler.recv_message`来读取服务器的响应。

唯一的额外部分与连接到服务器有关。为此，我们实际上创建了一个类型为`AF_INET`、`SOCK_STREAM`的套接字，这实际上意味着我们要使用 TCP/IP。

然后我们连接到服务器运行的`ip`和`port`，一旦连接成功，我们就通过生成的套接字`sock`发送消息并在同一个套接字上读取答案。

完成后，我们必须记得关闭套接字，否则它们将一直泄漏，直到操作系统决定杀死它们，因为它们长时间不活动。

# AsyncIO

虽然异步解决方案已经存在多年，但这些天它们变得越来越普遍。主要原因是，拥有一个没有数千个并发用户的应用程序不再是一个不寻常的场景；对于一个小型/中型应用程序来说，这实际上是一个常态，而且我们可以通过全球范围内使用的主要服务扩展到数百万用户。

能够提供这样的服务量，使用基于线程或进程的方法并不适合。特别是当用户触发的许多连接大部分时间可能都在那里无所事事。想想 Facebook Messenger 或 WhatsApp 这样的服务。无论你使用哪一个，你可能偶尔发送一条消息，大部分时间你与服务器的连接都在那里无所事事。也许你是一个热络的聊天者，每秒收到一条消息，但这仍然意味着在你的计算机每秒钟可以做的数百万次操作中，大部分时间都在无所事事。这种应用程序中的大部分繁重工作是由网络部分完成的，因此有很多资源可以通过在单个进程中进行多个连接来共享。

异步技术正好允许这样做，编写一个网络应用程序，而不是需要多个单独的线程（这将浪费内存和内核资源），我们可以有一个由多个协程组成的单个进程和线程，直到实际有事情要做时才会执行。

只要协程需要做的事情非常快速（比如获取一条消息并将其转发给你的另一个联系人），大部分工作将在网络层进行，因此可以并行进行。

# 如何做...

这个配方的步骤如下：

1.  我们将复制我们的回显服务器，但不再使用线程，而是使用 AsyncIO 和协程来提供请求：

```py
import asyncio

class EchoServer:
    MAX_MESSAGE_SIZE = 2**16  # 65k
    MESSAGE_HEADER_LEN = len(str(MAX_MESSAGE_SIZE))

    def __init__(self, host='0.0.0.0', port=9800):
        self._host = host
        self._port = port
        self._server = None

    def serve(self, loop):
        coro = asyncio.start_server(self.handle, self._host, self._port,
                                    loop=loop)
        self._server = loop.run_until_complete(coro)
        print('Serving on %s:%s' % (self._host, self._port))
        loop.run_until_complete(self._server.wait_closed())
        print('Done')

    @property
    def started(self):
        return self._server is not None and self._server.sockets

    def stop(self):
        print('Stopping...')
        self._server.close()

    async def handle(self, reader, writer):
        data = await self.recv_message(reader)
        await self.send_message(writer, b'ECHO: %s' % data)
        # Signal we finished handling this request
        # or the server will hang.
        writer.close()

    @classmethod
    async def recv_message(cls, socket):
        data_size = int(await socket.read(cls.MESSAGE_HEADER_LEN))
        data = await socket.read(data_size)
        return data

    @classmethod
    async def send_message(cls, socket, message):
        if len(message) > cls.MAX_MESSAGE_SIZE:
            raise ValueError('Message too big')

        message_size = str(len(message)).encode('ascii')
        message_size = message_size.zfill(cls.MESSAGE_HEADER_LEN)
        data = message_size + message

        socket.write(data)
        await socket.drain()
```

1.  现在我们有了服务器实现，我们需要一个客户端来测试它。由于实际上客户端做的与我们之前的配方相同，我们只是要重用相同的客户端实现。因此，客户端不会是基于 AsyncIO 和协程的，而是一个使用`socket`的普通函数：

```py
import socket

def send_message_to_server(ip, port, message):
    def _recv_message(socket):
        data_size = int(socket.recv(EchoServer.MESSAGE_HEADER_LEN))
        data = socket.recv(data_size)
        return data

    def _prepare_message(message):
        if len(message) > EchoServer.MAX_MESSAGE_SIZE:
            raise ValueError('Message too big')

        message_size = str(len(message)).encode('ascii')
        message_size = message_size.zfill(EchoServer.MESSAGE_HEADER_LEN)
        return message_size + message

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    try:
        sock.sendall(_prepare_message(message))
        response = _recv_message(sock)
        print("ANSWER: {}".format(response))
    finally:
        sock.close()
```

1.  现在我们可以把这些部分放在一起。为了在同一个进程中运行客户端和服务器，我们将在一个单独的线程中运行`asyncio`循环。因此，我们可以同时启动客户端。这并不是为了服务多个客户端而必须的，只是为了方便，避免不得不启动两个不同的 Python 脚本来玩服务器和客户端。

1.  首先，我们为服务器创建一个将持续`3`秒的线程。3 秒后，我们将明确停止我们的服务器：

```py
server = EchoServer()
def serve_for_3_seconds():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.call_later(3, server.stop)
    server.serve(loop)
    loop.close()

import threading
server_thread = threading.Thread(target=serve_for_3_seconds)
server_thread.start()
```

1.  然后，一旦服务器启动，我们就创建三个客户端并发送三条消息：

```py
while not server.started:
    pass

send_message_to_server('localhost', server._port, b"Hello World 1")
send_message_to_server('localhost', server._port, b"Hello World 2")
send_message_to_server('localhost', server._port, b"Hello World 3")
```

1.  完成后，我们等待服务器退出，因为 3 秒后它应该停止并退出：

```py
server_thread.join()
```

1.  如果一切按预期进行，你应该看到服务器启动，为三个客户端提供服务，然后退出：

```py
Serving on 0.0.0.0:9800
ANSWER: b'ECHO: Hello World 1'
ANSWER: b'ECHO: Hello World 2'
ANSWER: b'ECHO: Hello World 3'
Stopping...
Done 
```

# 工作原理...

这个配方的客户端大部分是直接从套接字服务配方中取出来的。区别在于服务器端不再是多线程的，而是基于协程的。

给定一个`asyncio`事件循环（我们在`serve_for_3_seconds`线程中使用`asyncio.new_event_loop()`创建的），`EchoServer.serve`方法创建一个基于协程的新服务器，并告诉循环永远提供请求，直到服务器本身关闭为止：

```py
def serve(self, loop):
    coro = asyncio.start_server(self.handle, self._host, self._port,
                                loop=loop)
    self._server = loop.run_until_complete(coro)
    print('Serving on %s:%s' % (self._host, self._port))
    loop.run_until_complete(self._server.wait_closed())
    print('Done')
```

`loop.run_until_complete`将阻塞，直到指定的协程退出，而`self._server.wait_closed()`只有在服务器本身停止时才会退出。

为了确保服务器在短时间内停止，当我们创建循环时，我们发出了`loop.call_later(3, server.stop)`的调用。这意味着 3 秒后，服务器将停止，整个循环将退出。

同时，直到服务器真正停止，它将继续提供服务。每个请求都会生成一个运行`handle`函数的协程：

```py
async def handle(self, reader, writer):
    data = await self.recv_message(reader)
    await self.send_message(writer, b'ECHO: %s' % data)
    # Signal we finished handling this request
    # or the server will hang.
    writer.close()
```

处理程序将接收两个流作为参数。一个用于传入数据，另一个用于传出数据。

就像我们在使用线程套接字服务器的情况下所做的那样，我们从`reader`流中读取传入的消息。为此，我们将`recv_message`重新实现为一个协程，这样我们就可以同时读取数据和处理其他请求：

```py
@classmethod
async def recv_message(cls, socket):
    data_size = int(await socket.read(cls.MESSAGE_HEADER_LEN))
    data = await socket.read(data_size)
    return data
```

当消息的大小和消息本身都可用时，我们只需返回消息，以便`send_message`函数可以将其回显到客户端。

在这种情况下，与`socketserver`的唯一特殊更改是我们要写入流写入器，但然后我们必须将其排空：

```py
socket.write(data)
await socket.drain()
```

这是因为在我们写入套接字后，我们需要将控制权发送回`asyncio`循环，以便它有机会实际刷新这些数据。

三秒后，调用`server.stop`方法，这将停止服务器，唤醒`wait_closed()`函数，从而使`EchoServer.serve`方法退出，因为它已经完成。

# 远程过程调用

有数百种系统可以在 Python 中执行 RPC，但由于它具有强大的网络工具并且是一种动态语言，我们需要的一切都已经内置在标准库中。

# 如何做到...

您需要执行以下步骤来完成此操作：

1.  使用`xmlrpc.server`，我们可以轻松创建一个基于 XMLRPC 的服务器，该服务器公开多个服务：

```py
import xmlrpc.server

class XMLRPCServices:
    class ExposedServices:
        pass

    def __init__(self, **services):
        self.services = self.ExposedServices()
        for name, service in services.items():
            setattr(self.services, name, service)

    def serve(self, host='localhost', port=8000):
        print('Serving XML-RPC on {}:{}'.format(host, port))
        self.server = xmlrpc.server.SimpleXMLRPCServer((host, port))
        self.server.register_introspection_functions()
        self.server.register_instance(self.services, 
                                      allow_dotted_names=True)
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
```

1.  特别是，我们将公开两项服务：一个用于获取当前时间，另一个用于将数字乘以`2`：

```py
class MathServices:
    def double(self, v):
        return v**2

class TimeServices:
    def currentTime(self):
        import datetime
        return datetime.datetime.utcnow()
```

1.  一旦我们有了我们的服务，我们可以使用`xmlrpc.client.ServerProxy`来消费它们，它提供了一个简单的调用接口来对 XMLRPC 服务器进行操作。

1.  通常情况下，为了在同一进程中启动客户端和服务器，我们可以使用一个线程来启动服务器，并让服务器在该线程中运行，而客户端驱动主线程：

```py
xmlrpcserver = XMLRPCServices(math=MathServices(),
                              time=TimeServices())

import threading
server_thread = threading.Thread(target=xmlrpcserver.serve)
server_thread.start()

from xmlrpc.client import ServerProxy
client = ServerProxy("http://localhost:8000")
print(
    client.time.currentTime()
)

xmlrpcserver.stop()
server_thread.join()
```

1.  如果一切正常，您应该在终端上看到当前时间的打印：

```py
Serving XML-RPC on localhost:8000
127.0.0.1 - - [10/Jun/2018 23:41:25] "POST /RPC2 HTTP/1.1" 200 -
20180610T21:41:25
```

# 它是如何工作的...

`XMLRPCServices`类接受我们要公开的所有服务作为初始化参数并将它们公开：

```py
xmlrpcserver = XMLRPCServices(math=MathServices(),
                              time=TimeServices())
```

这是因为我们公开了一个本地对象（`ExposedServices`），默认情况下为空，但我们将提供的所有服务作为属性附加到其实例上：

```py
def __init__(self, **services):
    self.services = self.ExposedServices()
    for name, service in services.items():
        setattr(self.services, name, service)
```

因此，我们最终暴露了一个`self.services`对象，它有两个属性：`math`和`time`，它们分别指向`MathServices`和`TimeServices`类。

实际上是由`XMLRPCServices.serve`方法来提供它们的：

```py
def serve(self, host='localhost', port=8000):
    print('Serving XML-RPC on {}:{}'.format(host, port))
    self.server = xmlrpc.server.SimpleXMLRPCServer((host, port))
    self.server.register_introspection_functions()
    self.server.register_instance(self.services, 
                                  allow_dotted_names=True)
    self.server.serve_forever()
```

这创建了一个`SimpleXMLRPCServer`实例，它是负责响应 XMLRPC 请求的 HTTP 服务器。

然后，我们将`self.services`对象附加到该实例，并允许它访问子属性，以便嵌套的`math`和`time`属性可以作为服务公开：

```py
self.server.register_instance(self.services, 
                              allow_dotted_names=True)
```

在实际启动服务器之前，我们还启用了内省功能。这些都是允许我们访问公开服务列表并请求其帮助和签名的所有功能：

```py
self.server.register_introspection_functions()
```

然后我们实际上启动了服务器：

```py
self.server.serve_forever()
```

这将阻止`serve`方法并循环提供请求，直到调用`stop`方法为止。

这就是为什么在示例中，我们在单独的线程中启动服务器的原因；也就是说，这样就不会阻塞我们可以用于客户端的主线程。

`stop`方法负责停止服务器，以便`serve`方法可以退出。该方法要求服务器在完成当前请求后立即终止，然后关闭关联的网络连接：

```py
def stop(self):
    self.server.shutdown()
    self.server.server_close()
```

因此，只需创建`XMLRPCServices`并提供它就足以使我们的 RPC 服务器正常运行：

```py
xmlrpcserver = XMLRPCServices(math=MathServices(),
                              time=TimeServices())
xmlrpcserver.serve()
```

在客户端，代码基础要简单得多；只需创建一个针对服务器公开的 URL 的`ServerProxy`即可：

```py
client = ServerProxy("http://localhost:8000")
```

然后，服务器公开的服务的所有方法都可以通过点表示法访问：

```py
client.time.currentTime()
```

# 还有更多...

`XMLRPCServices`具有很大的安全性影响，因此您不应该在开放网络上使用`SimpleXMLRPCServer`。

最明显的问题是，您允许任何人执行远程代码，因为 XMLRPC 服务器未经身份验证。因此，服务器应仅在您可以确保只有受信任的客户端能够访问服务的私人网络上运行。

但即使您在服务前提供适当的身份验证（通过在其前面使用任何 HTTP 代理来实现），您仍希望确保信任客户端将要发送的数据，因为`XMLRPCServices`存在一些安全限制。

所提供的数据是以明文交换的，因此任何能够嗅探您网络的人都能够看到它。

可以通过一些努力绕过这个问题，通过对`SimpleXMLRPCServer`进行子类化，并用 SSL 包装的`socket`实例替换它（客户端也需要这样做才能连接）。

但是，即使涉及到通信渠道的加固，您仍需要信任将要发送的数据，因为解析器是天真的，可以通过发送大量递归数据来使其失效。想象一下，您有一个实体，它扩展到数十个实体，每个实体又扩展到数十个实体，依此类推，达到 10-20 个级别。这将迅速需要大量的 RAM 来解码，但只需要几千字节来构建并通过网络发送。

此外，我们暴露子属性意味着我们暴露了比我们预期的要多得多。

您肯定希望暴露`time`服务的`currentTime`方法：

```py
client.time.currentTime()
```

请注意，您正在暴露`TimeServices`中声明的每个不以`_`开头的属性或方法。

在旧版本的 Python（如 2.7）中，这实际上意味着也暴露了内部代码，因为您可以通过诸如以下方式访问所有公共变量：

```py
client.time.currentTime.im_func.func_globals.keys()
```

然后，您可以通过以下方式检索它们的值：

```py
client.time.currentTime.im_func.func_globals.get('varname')
```

这是一个重大的安全问题。

幸运的是，函数的`im_func`属性已更名为`__func__`，因此不再可访问。但是，对于您自己声明的任何属性，仍然存在这个问题。
