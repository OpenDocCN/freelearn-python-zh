# 第四章. 与电子邮件互动

电子邮件是数字通信最流行的方式之一。Python 有丰富的内置库用于处理电子邮件。在本章中，我们将学习如何使用 Python 来撰写、发送和检索电子邮件。本章将涵盖以下主题：

+   通过`smtplib`库使用 SMTP 发送电子邮件

+   使用 TLS 保护电子邮件传输

+   使用`poplib`通过 POP3 检索电子邮件

+   使用`imapclient`通过 IMAP 检索电子邮件

+   使用 IMAP 在服务器上操作电子邮件

+   使用`logging`模块发送电子邮件

# 电子邮件术语

在我们开始使用 Python 撰写第一封电子邮件之前，让我们重新审视一些电子邮件的基本概念。通常，最终用户使用软件或图形用户界面（GUI）来撰写、发送和接收电子邮件。这种软件称为电子邮件客户端，例如 Mozilla Thunderbird、Microsoft Outlook 等都是电子邮件客户端。同样的任务也可以通过 Web 界面完成，即 Web 邮件客户端界面。一些常见的例子包括：Gmail、Yahoo 邮件、Hotmail 等。

您从客户端界面发送的邮件不会直接到达接收者的计算机。您的邮件会经过多个专用电子邮件服务器。这些服务器运行一个名为**邮件传输代理**（**MTA**）的软件，其主要工作是通过分析邮件头等内容将电子邮件路由到适当的目的地。

还有许多其他事情发生在路上，然后邮件到达收件人的本地电子邮件网关。然后，收件人可以使用他或她的电子邮件客户端检索电子邮件。

上述过程涉及一些协议。其中最常见的已列在这里：

+   **简单邮件传输协议**（**SMTP**）：MTA 使用 SMTP 协议将您的电子邮件传递到收件人的电子邮件服务器。SMTP 协议只能用于从一个主机发送电子邮件到另一个主机。

+   **邮局协议 3**（**POP3**）：POP3 协议为用户提供了一种简单和标准化的方式，以便访问邮箱，然后将邮件下载到他们的计算机上。使用 POP3 协议时，您的电子邮件消息将从互联网服务提供商（ISP）的邮件服务器下载到本地计算机。您还可以将电子邮件的副本留在 ISP 服务器上。

+   **互联网消息访问协议**（**IMAP**）：IMAP 协议还提供了一种简单和标准化的方式，用于从 ISP 的本地服务器访问您的电子邮件。IMAP 是一种客户端/服务器协议，其中电子邮件由 ISP 接收并保存。由于这只需要进行少量数据传输，即使在较慢的连接（如手机网络）上，这种方案也能很好地工作。只有当您发送请求读取特定的电子邮件时，该电子邮件消息才会从 ISP 下载。您还可以做一些其他有趣的事情，比如在服务器上创建和操作文件夹或邮箱、删除消息等。

Python 有三个模块，`smtplib`、`poplib`和`imaplib`，分别支持 SMTP、POP3 和 IMAP 协议。每个模块都有选项，可以使用**传输层安全**（**TLS**）协议安全地传输信息。每个协议还使用某种形式的身份验证来确保数据的保密性。

# 使用 SMTP 发送电子邮件

我们可以使用`smtplib`和`e-mail`包从 Python 脚本发送电子邮件。`smtplib`模块提供了一个 SMTP 对象，用于使用 SMTP 或**扩展 SMTP**（**ESMTP**）协议发送邮件。`e-mail`模块帮助我们构造电子邮件消息，并使用各种标题信息和附件。该模块符合[`tools.ietf.org/html/rfc2822.html`](http://tools.ietf.org/html/rfc2822.html)中描述的**Internet Message Format**（**IMF**）。

## 撰写电子邮件消息

让我们使用`email`模块中的类构造电子邮件消息。`email.mime`模块提供了从头开始创建电子邮件和 MIME 对象的类。**MIME**是**多用途互联网邮件扩展**的缩写。这是原始互联网电子邮件协议的扩展。这被广泛用于交换不同类型的数据文件，如音频、视频、图像、应用程序等。

许多类都是从 MIME 基类派生的。我们将使用一个 SMTP 客户端脚本，使用`email.mime.multipart.MIMEMultipart()`类作为示例。它接受通过关键字字典传递电子邮件头信息。让我们看看如何使用`MIMEMultipart()`对象指定电子邮件头。多部分 mime 指的是在单个电子邮件中发送 HTML 和 TEXT 部分。当电子邮件客户端接收多部分消息时，如果可以呈现 HTML，它将接受 HTML 版本，否则它将呈现纯文本版本，如下面的代码块所示：

```py
    from email.mime.multipart import MIMEMultipart()
    msg = MIMEMultipart()
    msg['To'] = recipient
    msg['From'] = sender
    msg['Subject'] = 'Email subject..'
```

现在，将纯文本消息附加到此多部分消息对象。我们可以使用`MIMEText()`对象来包装纯文本消息。这个类的构造函数接受额外的参数。例如，我们可以将`text`和`plain`作为它的参数。可以使用`set_payload()`方法设置此消息的数据，如下所示：

```py
    part = MIMEText('text', 'plain')
    message = 'Email message ….'
    part.set_payload(message)
```

现在，我们将将纯文本消息附加到多部分消息中，如下所示：

```py
    msg.attach(part)
```

该消息已准备好通过一个或多个 SMTP MTA 服务器路由到目标邮件服务器。但是，显然，脚本只与特定的 MTA 通信，而该 MTA 处理消息的路由。

## 发送电子邮件消息

`smtplib`模块为我们提供了一个 SMTP 类，可以通过 SMTP 服务器套接字进行初始化。成功初始化后，这将为我们提供一个 SMTP 会话对象。SMTP 客户端将与服务器建立适当的 SMTP 会话。这可以通过为 SMTP`session`对象使用`ehlo()`方法来完成。实际的消息发送将通过将`sendmail()`方法应用于 SMTP 会话来完成。因此，典型的 SMTP 会话将如下所示：

```py
    session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    session.ehlo()
    session.sendmail(sender, recipient, msg.as_string())
    session.quit()
```

在我们的示例 SMTP 客户端脚本中，我们使用了谷歌的免费 Gmail 服务。如果您有 Gmail 帐户，那么您可以通过 SMTP 从 Python 脚本发送电子邮件到该帐户。您的电子邮件可能会被最初阻止，因为 Gmail 可能会检测到它是从不太安全的电子邮件客户端发送的。您可以更改 Gmail 帐户设置，并启用您的帐户以从不太安全的电子邮件客户端发送/接收电子邮件。您可以在 Google 网站上了解有关从应用程序发送电子邮件的更多信息，网址为[`support.google.com/a/answer/176600?hl=en`](https://support.google.com/a/answer/176600?hl=en)。

如果您没有 Gmail 帐户，则可以在典型的 Linux 框中使用本地 SMTP 服务器设置并运行此脚本。以下代码显示了如何通过公共 SMTP 服务器发送电子邮件：

```py
#!/usr/bin/env python3
# Listing 1 – First email client
import smtplib

from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_SERVER = 'aspmx.l.google.com'
SMTP_PORT = 25

def send_email(sender, recipient):
    """ Send email message """
    msg = MIMEMultipart()
    msg['To'] = recipient
    msg['From'] = sender
    subject = input('Enter your email subject: ')
    msg['Subject'] = subject
    message = input('Enter your email message. Press Enter when finished. ')
    part = MIMEText('text', "plain")
    part.set_payload(message)
    msg.attach(part)
    # create smtp session
    session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    session.ehlo()
    #session.set_debuglevel(1)
    # send mail
    session.sendmail(sender, recipient, msg.as_string())
    print("You email is sent to {0}.".format(recipient))
    session.quit()

if __name__ == '__main__':
    sender = input("Enter sender email address: ")
    recipient = input("Enter recipient email address: ")
    send_email(sender, recipient)
```

如果您运行此脚本，则可以看到输出与此处提到的类似。出于匿名性考虑，在以下示例中未显示真实的电子邮件地址：

```py
**$ python3 smtp_mail_sender.py** 
**Enter sender email address: <SENDER>@gmail.com** 
**Enter recipeint email address: <RECEIVER>@gmail.com**
**Enter your email subject: Test mail**
**Enter your email message. Press Enter when finished. This message can be ignored**
**You email is sent to <RECEIVER>@gmail.com.**

```

这个脚本将使用 Python 的标准库模块`smtplib`发送一个非常简单的电子邮件消息。为了构成消息，从`email.mime`子模块导入了`MIMEMultipart`和`MIMEText`类。这个子模块有各种类型的类，用于以不同类型的附件组成电子邮件消息，例如`MIMEApplication()`、`MIMEAudio()`、`MIMEImage()`等。

在这个例子中，`send_mail()`函数被调用了两个参数：发件人和收件人。这两个参数都是电子邮件地址。电子邮件消息是由`MIMEMultipart()`消息类构造的。这个类命名空间中添加了`To`、`From`和`Subject`等基本标头。消息的正文是由`MIMEText()`类的实例组成的。这是通过`set_payload()`方法完成的。然后，这个有效载荷通过`attach()`方法附加到主消息上。

为了与 SMTP 服务器通信，将通过实例化`smtplib`模块的`SMTP()`类创建与服务器的会话。服务器名称和端口参数将传递给构造函数。根据 SMTP 协议，客户端将通过`ehlo()`方法向服务器发送扩展的问候消息。消息将通过`sendmail()`方法发送。

请注意，如果在 SMTP 会话对象上调用`set_debuglevel()`方法，它将产生额外的调试消息。在前面的例子中，这行被注释掉了。取消注释该行将产生类似以下的调试消息：

```py
**$ python3 smtp_mail_sender.py** 
**Enter sender email address: <SENDER>@gmail.com**
**Enter recipeint email address: <RECEIVER>@gmail.com**
**Enter your** 
**email subject: Test email**
**Enter your email message. Press Enter when finished. This is a test email**
**send: 'mail FROM:<SENDER@gmail.com> size=339\r\n'**
**reply: b'250 2.1.0 OK hg2si4622244wib.38 - gsmtp\r\n'**
**reply: retcode (250); Msg: b'2.1.0 OK hg2si4622244wib.38 - gsmtp'**
**send: 'rcpt TO:<RECEIVER@gmail.com>\r\n'**
**reply: b'250 2.1.5 OK hg2si4622244wib.38 - gsmtp\r\n'**
**reply: retcode (250); Msg: b'2.1.5 OK hg2si4622244wib.38 - gsmtp'**
**send: 'data\r\n'**
**reply: b'354  Go ahead hg2si4622244wib.38 - gsmtp\r\n'**
**reply: retcode (354); Msg: b'Go ahead hg2si4622244wib.38 - gsmtp'**
**data: (354, b'Go ahead hg2si4622244wib.38 - gsmtp')**
**send: 'Content-Type: multipart/mixed; 
boundary="===============1431208306=="\r\nMIME-Version: 1.0\r\nTo: RECEIVER@gmail.com\r\nFrom: SENDER@gmail.com\r\nSubject: Test  email\r\n\r\n--===============1431208306==\r\nContent-Type: text/plain; charset="us-ascii"\r\nMIME-Version: 1.0\r\nContent- Transfer-Encoding: 7bit\r\n\r\nThis is a test email\r\n-- ===============1431208306==--\r\n.\r\n'**
**reply: b'250 2.0.0 OK 1414233177 hg2si4622244wib.38 - gsmtp\r\n'**
**reply: retcode (250); Msg: b'2.0.0 OK 1414233177 hg2si4622244wib.38 - gsmtp'**
**data: (250, b'2.0.0 OK 1414233177 hg2si4622244wib.38 - gsmtp')**
**You email is sent to RECEIVER@gmail.com.**
**send: 'quit\r\n'**
**reply: b'221 2.0.0 closing connection hg2si4622244wib.38 - gsmtp\r\n'**
**reply: retcode (221); Msg: b'2.0.0 closing connection hg2si4622244wib.38 - gsmtp'**

```

这很有趣，因为消息是通过逐步方式通过公共 SMTP 服务器发送的。

# 使用 TLS 安全地发送电子邮件

TLS 协议是 SSL 或安全套接字层的后继者。这确保了客户端和服务器之间的通信是安全的。这是通过以加密格式发送消息来实现的，以便未经授权的人无法看到消息。使用`smtplib`使用 TLS 并不困难。创建 SMTP 会话对象后，需要调用`starttls()`方法。在发送电子邮件之前，需要使用 SMTP 服务器凭据登录到服务器。

这是第二个电子邮件客户端的示例：

```py
#!/usr/bin/env python3
# Listing 2
import getpass
import smtplib

from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587 # ssl port 465, tls port 587

def send_email(sender, recipient):
    """ Send email message """
    msg = MIMEMultipart()
    msg['To'] = recipient
    msg['From'] = sender
    msg['Subject'] = input('Enter your email subject: ')
    message = input('Enter your email message. Press Enter when finished. ')
    part = MIMEText('text', "plain")
    part.set_payload(message)
    msg.attach(part)
    # create smtp session
    session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    session.set_debuglevel(1)
    session.ehlo()
    session.starttls()
    session.ehlo
    password = getpass.getpass(prompt="Enter you email password: ") 
    # login to server
    session.login(sender, password)
    # send mail
    session.sendmail(sender, recipient, msg.as_string())
    print("You email is sent to {0}.".format(recipient))
    session.quit()

if __name__ == '__main__':
    sender = input("Enter sender email address: ")
    recipient = input("Enter recipeint email address: ")
    send_email(sender, recipient)
```

前面的代码与我们的第一个例子类似，只是对服务器进行了身份验证。在这种情况下，SMTP 用户会被服务器验证。如果我们在打开 SMTP 调试后运行脚本，那么我们将看到类似以下的输出：

```py
**$ python3 smtp_mail_sender_tls.py** 
**Enter sender email address: SENDER@gmail.com**
**Enter recipeint email address: RECEPIENT@gmail.com**
**Enter your email subject: Test email**
**Enter your email message. Press Enter when finished. This is a test email that can be ignored.**

```

用户输入后，将开始与服务器的通信。它将通过`ehlo()`方法开始。作为对这个命令的响应，SMTP 服务器将发送几行带有返回代码`250`的响应。这个响应将包括服务器支持的特性。

这些响应的摘要将表明服务器准备好与客户端继续，如下所示：

```py
**send: 'ehlo debian6box.localdomain.loc\r\n'**
**reply: b'250-mx.google.com at your service, [77.233.155.107]\r\n'**
**reply: b'250-SIZE 35882577\r\n'**
**reply: b'250-8BITMIME\r\n'**
**reply: b'250-STARTTLS\r\n'**
**reply: b'250-ENHANCEDSTATUSCODES\r\n'**
**reply: b'250-PIPELINING\r\n'**
**reply: b'250-CHUNKING\r\n'**
**reply: b'250 SMTPUTF8\r\n'**
**reply: retcode (250); Msg: b'mx.google.com at your service, [77.233.155.107]\nSIZE 35882577\n8BITMIME\nSTARTTLS\nENHANCEDSTATUSCODES\nPIPELINING\ nCHUNKING\nSMTPUTF8'**

```

在初始命令之后，客户端将使用`starttls()`方法将连接升级到 TLS，如下所示：

```py
**send: 'STARTTLS\r\n'**
**reply: b'220 2.0.0 Ready to start TLS\r\n'**
**reply: retcode (220); Msg: b'2.0.0 Ready to start TLS'**
**Enter you email password:** 
**send: 'ehlo debian6box.localdomain.loc\r\n'**
**reply: b'250-mx.google.com at your service, [77.233.155.107]\r\n'**
**reply: b'250-SIZE 35882577\r\n'**
**reply: b'250-8BITMIME\r\n'**
**reply: b'250-AUTH LOGIN PLAIN XOAUTH XOAUTH2 PLAIN-CLIENTTOKEN OAUTHBEARER\r\n'**
**reply: b'250-ENHANCEDSTATUSCODES\r\n'**
**reply: b'250-PIPELINING\r\n'**
**reply: b'250-CHUNKING\r\n'**
**reply: b'250 SMTPUTF8\r\n'**
**reply: retcode (250); Msg: b'mx.google.com at your service, [77.233.155.107]\nSIZE 35882577\n8BITMIME\nAUTH LOGIN PLAIN XOAUTH XOAUTH2 PLAIN-CLIENTTOKEN OAUTHBEARER\nENHANCEDSTATUSCODES\nPIPELINING\nCHUNKING\nSMTPUTF8'**

```

在认证阶段，客户端脚本通过`login()`方法发送认证数据。请注意，认证令牌是一个 base-64 编码的字符串，用户名和密码之间用空字节分隔。还有其他支持的身份验证协议适用于复杂的客户端。以下是认证令牌的示例：

```py
**send: 'AUTH PLAIN A...dvXXDDCCD.......sscdsvsdvsfd...12344555\r\n'**
**reply: b'235 2.7.0 Accepted\r\n'**
**reply: retcode (235); Msg: b'2.7.0 Accepted'**

```

客户端经过认证后，可以使用`sendmail()`方法发送电子邮件消息。这个方法传递了三个参数，发件人、收件人和消息。示例输出如下：

```py
**send: 'mail FROM:<SENDER@gmail.com> size=360\r\n'**
**reply: b'250 2.1.0 OK xw9sm8487512wjc.24 - gsmtp\r\n'**
**reply: retcode (250); Msg: b'2.1.0 OK xw9sm8487512wjc.24 - gsmtp'**
**send: 'rcpt TO:<RECEPIENT@gmail.com>\r\n'**
**reply: b'250 2.1.5 OK xw9sm8487512wjc.24 - gsmtp\r\n'**
**reply: retcode (250); Msg: b'2.1.5 OK xw9sm8487512wjc.24 - gsmtp'**
**send: 'data\r\n'**
**reply: b'354  Go ahead xw9sm8487512wjc.24 - gsmtp\r\n'**
**reply: retcode (354); Msg: b'Go ahead xw9sm8487512wjc.24 - gsmtp'**
**data: (354, b'Go ahead xw9sm8487512wjc.24 - gsmtp')**
**send: 'Content-Type: multipart/mixed; boundary="===============1501937935=="\r\nMIME-Version: 1.0\r\n**
**To: <Output omitted>-===============1501937935==--\r\n.\r\n'**
**reply: b'250 2.0.0 OK 1414235750 xw9sm8487512wjc.24 - gsmtp\r\n'**
**reply: retcode (250); Msg: b'2.0.0 OK 1414235750 xw9sm8487512wjc.24 - gsmtp'**
**data: (250, b'2.0.0 OK 1414235750 xw9sm8487512wjc.24 - gsmtp')**
**You email is sent to RECEPIENT@gmail.com.**
**send: 'quit\r\n'**
**reply: b'221 2.0.0 closing connection xw9sm8487512wjc.24 - gsmtp\r\n'**
**reply: retcode (221); Msg: b'2.0.0 closing connection xw9sm8487512wjc.24 - gsmtp'**

```

# 使用 poplib 通过 POP3 检索电子邮件

存储的电子邮件消息可以通过本地计算机下载和阅读。 POP3 协议可用于从电子邮件服务器下载消息。 Python 有一个名为`poplib`的模块，可以用于此目的。 此模块提供了两个高级类，`POP()`和`POP3_SSL()`，它们分别实现了与 POP3/POP3S 服务器通信的 POP3 和 POP3S 协议。 它接受三个参数，主机、端口和超时。 如果省略端口，则可以使用默认端口（110）。 可选的超时参数确定服务器上的连接超时长度（以秒为单位）。

`POP3()`的安全版本是其子类`POP3_SSL()`。 它接受附加参数，例如 keyfile 和 certfile，用于提供 SSL 证书文件，即私钥和证书链文件。

编写 POP3 客户端也非常简单。 要做到这一点，通过初始化`POP3()`或`POP3_SSL()`类来实例化一个邮箱对象。 然后，通过以下命令调用`user()`和`pass_()`方法登录到服务器：

```py
  mailbox = poplib.POP3_SSL(<POP3_SERVER>, <SERVER_PORT>) 
  mailbox.user('username')
       mailbox.pass_('password')
```

现在，您可以调用各种方法来操作您的帐户和消息。 这里列出了一些有趣的方法：

+   `stat()`: 此方法根据两个整数的元组返回邮箱状态，即消息计数和邮箱大小。

+   `list`(): 此方法发送一个请求以获取消息列表，这在本节后面的示例中已经演示过。

+   `retr()`: 此方法给出一个参数消息编号，表示要检索的消息。 它还标记消息为已读。

+   `dele()`: 此方法提供了要删除的消息的参数。 在许多 POP3 服务器上，直到 QUIT 才执行删除操作。 您可以使用`rset()`方法重置删除标志。

+   `quit()`: 此方法通过提交一些更改并将您从服务器断开连接来使您脱离连接。

让我们看看如何通过访问谷歌的安全 POP3 电子邮件服务器来读取电子邮件消息。 默认情况下，POP3 服务器在端口`995`上安全监听。 以下是使用 POP3 获取电子邮件的示例：

```py
#!/usr/bin/env python3
import getpass
import poplib

GOOGLE_POP3_SERVER = 'pop.googlemail.com'
POP3_SERVER_PORT = '995'

def fetch_email(username, password): 
    mailbox = poplib.POP3_SSL(GOOGLE_POP3_SERVER, POP3_SERVER_PORT) 
    mailbox.user(username)
    mailbox.pass_(password) 
    num_messages = len(mailbox.list()[1])
    print("Total emails: {0}".format(num_messages))
    print("Getting last message") 
    for msg in mailbox.retr(num_messages)[1]:
        print(msg)
    mailbox.quit()

if __name__ == '__main__':
    username = input("Enter your email user ID: ")
    password = getpass.getpass(prompt="Enter your email password:    ") 
    fetch_email(username, password)
```

正如您在前面的代码中所看到的，`fetch_email()`函数通过调用`POP3_SSL()`以及服务器套接字创建了一个邮箱对象。 通过调用`user()`和`pass_()`方法在此对象上设置了用户名和密码。 成功验证后，我们可以通过使用`list()`方法调用 POP3 命令。 在此示例中，消息的总数已显示在屏幕上。 然后，使用`retr()`方法检索了单个消息的内容。

这里显示了一个示例输出：

```py
**$ python3 fetch_email_pop3.py** 
**Enter your email user ID: <PERSON1>@gmail.com**
**Enter your email password:** 
**Total emails: 330**
**Getting last message**
**b'Received: by 10.150.139.7 with HTTP; Tue, 7 Oct 2008 13:20:42 -0700** 
**(PDT)'**
**b'Message-ID: <fc9dd8650810...@mail.gmail.com>'**
**b'Date: Tue, 7 Oct 2008 21:20:42 +0100'**
**b'From: "Mr Person1" <PERSON1@gmail.com>'**
**b'To: "Mr Person2" <PERSON2@gmail.com>'**
**b'Subject: Re: Some subject'**
**b'In-Reply-To: <1bec119d...@mail.gmail.com>'**
**b'MIME-Version: 1.0'**
**b'Content-Type: multipart/alternative; '**
**b'\tboundary="----=_Part_63057_22732713.1223410842697"'**
**b'References: <fc9dd8650809270....@mail.gmail.com>'**
**b'\t <1bec119d0810060337p557bc....@mail.gmail.com>'**
**b'Delivered-To: PERSON1@gmail.com'**
**b''**
**b'------=_Part_63057_22732713.1223410842697'**
**b'Content-Type: text/plain; charset=ISO-8859-1'**
**b'Content-Transfer-Encoding: quoted-printable'**
**b'Content-Disposition: inline'**
**b''**
**b'Dear Person2,'**

```

# 使用 imaplib 通过 IMAP 检索电子邮件

正如我们之前提到的，通过 IMAP 协议访问电子邮件不一定会将消息下载到本地计算机或手机。 因此，即使在任何低带宽互联网连接上使用，这也可以非常高效。

Python 提供了一个名为`imaplib`的客户端库，可用于通过 IMAP 协议访问电子邮件。 这提供了实现 IMAP 协议的`IMAP4()`类。 它接受两个参数，即用于实现此协议的主机和端口。 默认情况下，`143`已被用作端口号。

派生类`IMAP4_SSL()`提供了 IMAP4 协议的安全版本。 它通过 SSL 加密套接字连接。 因此，您将需要一个 SSL 友好的套接字模块。 默认端口是`993`。 与`POP3_SSL()`类似，您可以提供私钥和证书文件路径。

可以在这里看到 IMAP 客户端的典型示例：

```py
  mailbox = imaplib.IMAP4_SSL(<IMAP_SERVER>, <SERVER_PORT>) 
      mailbox.login('username', 'password')
      mailbox.select('Inbox')
```

上述代码将尝试启动一个 IMAP4 加密客户端会话。在`login()`方法成功之后，您可以在创建的对象上应用各种方法。在上述代码片段中，使用了`select()`方法。这将选择用户的邮箱。默认邮箱称为`Inbox`。此邮箱对象支持的方法的完整列表可在 Python 标准库文档页面上找到，网址为[`docs.python.org/3/library/imaplib.html`](https://docs.python.org/3/library/imaplib.html)。

在这里，我们想演示如何使用`search()`方法搜索邮箱。它接受字符集和搜索条件参数。字符集参数可以是`None`，其中将向服务器发送不带特定字符的请求。但是，至少需要指定一个条件。为了执行高级搜索以对消息进行排序，可以使用`sort()`方法。

与 POP3 类似，我们将使用安全的 IMAP 连接来连接到服务器，使用`IMAP4_SSL()`类。以下是一个 Python IMAP 客户端的简单示例：

```py
#!/usr/bin/env python3
import getpass
import imaplib
import pprint

GOOGLE_IMAP_SERVER = 'imap.googlemail.com'
IMAP_SERVER_PORT = '993'

def check_email(username, password): 
    mailbox = imaplib.IMAP4_SSL(GOOGLE_IMAP_SERVER, IMAP_SERVER_PORT) 
    mailbox.login(username, password)
    mailbox.select('Inbox')
    tmp, data = mailbox.search(None, 'ALL')
    for num in data[0].split():
        tmp, data = mailbox.fetch(num, '(RFC822)')
        print('Message: {0}\n'.format(num))
        pprint.pprint(data[0][1])
        break
    mailbox.close()
    mailbox.logout()

if __name__ == '__main__':
    username = input("Enter your email username: ")
    password = getpass.getpass(prompt="Enter you account password: ")
    check_email(username, password)
```

在此示例中，创建了`IMPA4_SSL()`的实例，即邮箱对象。在其中，我们将服务器地址和端口作为参数。成功使用`login()`方法登录后，您可以使用`select()`方法选择要访问的邮箱文件夹。在此示例中，选择了`Inbox`文件夹。为了阅读消息，我们需要从收件箱请求数据。其中一种方法是使用`search()`方法。在成功接收一些邮件元数据后，我们可以使用`fetch()`方法检索电子邮件消息信封部分和数据。在此示例中，使用`fetch()`方法寻找了 RFC 822 类型的标准文本消息。我们可以使用 Python 的 pretty print 或 print 模块在屏幕上显示输出。最后，将`close()`和`logout()`方法应用于邮箱对象。

上述代码将显示类似以下内容的输出：

```py
$ python3 fetch_email_imap.py 
Enter your email username: RECIPIENT@gmail.comn
Enter you Google password: 
Message b'1'
b'X-Gmail-Received: 3ec65fa310559efe27307d4e37fdc95406deeb5a\r\nDelivered-To: RECIPIENT@gmail.com\r\nReceived: by 10.54.40.10 with SMTP id n10cs1955wrn;\r\n    [Message omitted]
```

# 发送电子邮件附件

在前面的部分中，我们已经看到如何使用 SMTP 协议发送纯文本消息。在本节中，让我们探讨如何通过电子邮件消息发送附件。我们可以使用我们的第二个示例，其中我们使用了 TLS 发送电子邮件。在撰写电子邮件消息时，除了添加纯文本消息，还包括附加附件字段。

在此示例中，我们可以使用`email.mime.image`子模块的`MIMEImage`类型。一个 GIF 类型的图像将附加到电子邮件消息中。假设可以在文件系统路径的任何位置找到 GIF 图像。该文件路径通常基于用户输入。

以下示例显示了如何在电子邮件消息中发送附件：

```py
#!/usr/bin/env python3

import os
import getpass
import re
import sys
import smtplib

from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_SERVER = 'aspmx.l.google.com'
SMTP_PORT = 25

def send_email(sender, recipient):
    """ Sends email message """
    msg = MIMEMultipart()
    msg['To'] = recipient
    msg['From'] = sender
    subject = input('Enter your email subject: ')
    msg['Subject'] = subject
    message = input('Enter your email message. Press Enter when     finished. ')
    part = MIMEText('text', "plain")
    part.set_payload(message)
    msg.attach(part)
    # attach an image in the current directory
    filename = input('Enter the file name of a GIF image: ')
    path = os.path.join(os.getcwd(), filename)
    if os.path.exists(path):
        img = MIMEImage(open(path, 'rb').read(), _subtype="gif")
        img.add_header('Content-Disposition', 'attachment', filename=filename)
        msg.attach(img)
    # create smtp session
    session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    session.ehlo()
    session.starttls()
    session.ehlo
    # send mail
    session.sendmail(sender, recipient, msg.as_string())
    print("You email is sent to {0}.".format(recipient))
    session.quit()

if __name__ == '__main__':
    sender = input("Enter sender email address: ")
    recipient = input("Enter recipeint email address: ")
    send_email(sender, recipient)
```

如果运行上述脚本，它将询问通常的内容，即电子邮件发送者、收件人、用户凭据和图像文件的位置。

```py
**$ python3 smtp_mail_sender_mime.py** 
**Enter sender email address: SENDER@gmail.com**
**Enter recipeint email address: RECIPIENT@gmail.com**
**Enter your email subject: Test email with attachment** 
**Enter your email message. Press Enter when finished. This is a test email with atachment.**
**Enter the file name of a GIF image: image.gif**
**You email is sent to RECIPIENT@gmail.com.**

```

# 通过日志模块发送电子邮件

在任何现代编程语言中，都提供了常见功能的日志记录设施。同样，Python 的日志模块在功能和灵活性上非常丰富。我们可以使用日志模块的不同类型的日志处理程序，例如控制台或文件日志处理程序。您可以最大化日志记录的好处的一种方法是在生成日志时将日志消息通过电子邮件发送给用户。Python 的日志模块提供了一种称为`BufferingHandler`的处理程序类型，它能够缓冲日志数据。

稍后显示了扩展`BufferingHandler`的示例。通过`BufferingHandler`定义了一个名为`BufferingSMTPHandler`的子类。在此示例中，使用日志模块创建了一个记录器对象的实例。然后，将`BufferingSMTPHandler`的实例绑定到此记录器对象。将日志级别设置为 DEBUG，以便记录任何消息。使用了一个包含四个单词的示例列表来创建四个日志条目。每个日志条目应类似于以下内容：

```py
**<Timestamp> INFO  First line of log**
**This accumulated log message will be emailed to a local user as set on top of the script.**

```

现在，让我们来看一下完整的代码。以下是使用日志模块发送电子邮件的示例：

```py
import logging.handlers
import getpass

MAILHOST = 'localhost'
FROM = 'you@yourdomain'
TO = ['%s@localhost' %getpass.getuser()] 
SUBJECT = 'Test Logging email from Python logging module (buffering)'

class BufferingSMTPHandler(logging.handlers.BufferingHandler):
    def __init__(self, mailhost, fromaddr, toaddrs, subject, capacity):
        logging.handlers.BufferingHandler.__init__(self, capacity)
        self.mailhost = mailhost
        self.mailport = None
        self.fromaddr = fromaddr
        self.toaddrs = toaddrs
        self.subject = subject
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s"))

    def flush(self):
        if len(self.buffer) > 0:
            try:
                import smtplib
                port = self.mailport
                if not port:
                    port = smtplib.SMTP_PORT
                    smtp = smtplib.SMTP(self.mailhost, port)
                    msg = "From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n" % (self.fromaddr, ",".join(self.toaddrs), self.subject)
                for record in self.buffer:
                    s = self.format(record)
                    print(s)
                    msg = msg + s + "\r\n"
                smtp.sendmail(self.fromaddr, self.toaddrs, msg)
                smtp.quit()
            except:
                self.handleError(None) # no particular record
            self.buffer = []

def test():
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(BufferingSMTPHandler(MAILHOST, FROM, TO, SUBJECT, 10))
    for data in ['First', 'Second', 'Third', 'Fourth']:
        logger.info("%s line of log", data)
    logging.shutdown()

if __name__ == "__main__":
    test()
```

如您所见，我们的`BufferingSMTPHandler`方法只覆盖了一个方法，即`flush()`。在构造函数`__init__()`中，设置了基本变量以及使用`setFormatter()`方法设置了日志格式。在`flush()`方法中，我们创建了一个`SMTP()`对象的实例。使用可用数据创建了 SMTP 消息头。将日志消息附加到电子邮件消息，并调用`sendmail()`方法发送电子邮件消息。`flush()`方法中的代码包裹在`try-except`块中。

所讨论的脚本的输出将类似于以下内容：

```py
**$ python3 logger_mail_send.py** 
**2014-10-25 13:15:07,124 INFO  First line of log**
**2014-10-25 13:15:07,127 INFO  Second line of log**
**2014-10-25 13:15:07,127 INFO  Third line of log**
**2014-10-25 13:15:07,129 INFO  Fourth line of log**

```

现在，当您使用电子邮件命令（Linux/UNIX 机器上的本机命令）检查电子邮件消息时，您可以期望本地用户已收到电子邮件，如下所示：

```py
**$ mail**
**Mail version 8.1.2 01/15/2001\.  Type ? for help.**
**"/var/mail/faruq": 1 message 1 new**
**>N  1 you@yourdomain     Sat Oct 25 13:15   20/786   Test Logging email from Python logging module (buffering)**

```

您可以通过在命令提示符上输入消息 ID 和`&`来查看消息的内容，如下输出所示：

```py
**& 1**
**Message 1:**
**From you@yourdomain Sat Oct 25 13:15:08 2014**
**Envelope-to: faruq@localhost**
**Delivery-date: Sat, 25 Oct 2014 13:15:08 +0100**
**Date: Sat, 25 Oct 2014 13:15:07 +0100**
**From: you@yourdomain**
**To: faruq@localhost**
**Subject: Test Logging email from Python logging module (buffering)**

**2014-10-25 13:15:07,124 INFO  First line of log**
**2014-10-25 13:15:07,127 INFO  Second line of log**
**2014-10-25 13:15:07,127 INFO  Third line of log**
**2014-10-25 13:15:07,129 INFO  Fourth line of log**

```

最后，您可以通过在命令提示符上输入快捷键`q`来退出邮件程序，如下所示：

```py
**& q**
**Saved 1 message in /home/faruq/mbox**

```

# 总结

本章演示了 Python 如何与三种主要的电子邮件处理协议交互：SMTP、POP3 和 IMAP。在每种情况下，都解释了客户端代码的工作方式。最后，展示了在 Python 的日志模块中使用 SMTP 的示例。

在下一章中，您将学习如何使用 Python 与远程系统一起执行各种任务，例如使用 SSH 进行管理任务，通过 FTP、Samba 等进行文件传输。还将简要讨论一些远程监控协议，如 SNMP，以及身份验证协议，如 LDAP。因此，请在下一章中享受编写更多的 Python 代码。
