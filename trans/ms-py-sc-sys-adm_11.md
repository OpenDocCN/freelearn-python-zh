# 使用Python脚本处理电子邮件

在本章中，您将学习如何使用Python脚本处理电子邮件。您将学习电子邮件消息格式。我们将探索`smtplib`模块用于发送和接收电子邮件。我们将使用Python电子邮件包发送带有附件和HTML内容的电子邮件。您还将学习用于处理电子邮件的不同协议。

在本章中，您将学习以下内容：

+   电子邮件消息格式

+   添加HTML和多媒体内容

+   POP3和IMAP服务器

# 电子邮件消息格式

在本节中，我们将学习电子邮件消息格式。电子邮件消息由三个主要组件组成：

+   接收者的电子邮件地址

+   发件人的电子邮件地址

+   消息

消息格式中还包括其他组件，如主题行、电子邮件签名和附件。

现在，我们将看一个简单的例子，从您的Gmail地址发送纯文本电子邮件，在其中您将学习如何编写电子邮件消息并发送它。现在，请创建一个名为`write_email_message.py`的脚本，并在其中编写以下内容：

```py
import smtplib import getpass host_name = "smtp.gmail.com" port = 465 sender = 'sender_emil_id'
receiver = 'receiver_email_id' password = getpass.getpass() msg = """\ Subject: Test Mail Hello from Sender !!""" s = smtplib.SMTP_SSL(host_name, port) s.login(sender, password) s.sendmail(sender, receiver, msg) s.quit() print("Mail sent successfully")
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work/Chapter_11$ python3 write_email_message.py Output: Password: Mail sent successfully
```

在上面的例子中，我们使用了`smtplib` Python模块来发送电子邮件。确保您从Gmail ID向接收者发送电子邮件。`sender`变量保存了发件人的电子邮件地址。在`password`变量中，您可以输入密码，或者您可以使用`getpass`模块提示输入密码。在这里，我们提示输入密码。接下来，我们创建了一个名为`msg`的变量，这将是我们实际的电子邮件消息。在其中，我们首先提到了一个主题，然后是我们想要发送的消息。然后，在`login()`中，我们提到了`sender`和`password`变量。接下来，在`sendmail()`中，我们提到了`sender`，`receivers`和`text`变量。因此，通过这个过程，我们成功地发送了电子邮件。

# 添加HTML和多媒体内容

在本节中，我们将看到如何将多媒体内容作为附件发送以及如何添加HTML内容。为此，我们将使用Python的`email`包。

首先，我们将看如何添加HTML内容。为此，请创建一个名为`add_html_content.py`的脚本，并在其中编写以下内容：

```py
import os import smtplib from email.mime.text import MIMEText from email.mime.multipart import MIMEMultipart import getpass host_name = 'smtp.gmail.com' port = 465 sender = '*sender_emailid*' password = getpass.getpass() receiver = '*receiver_emailid*' text = MIMEMultipart() text['Subject'] = 'Test HTML Content' text['From'] = sender text['To'] = receiver msg = """\ <html>
 <body> <p>Hello there, <br> Good day !!<br> <a href="http://www.imdb.com">Home</a> </p> </body> </html> """ html_content = MIMEText(msg, "html") text.attach(html_content) s = smtplib.SMTP_SSL(host_name, port) print("Mail sent successfully !!")  s.login(sender, password) s.sendmail(sender, receiver, text.as_string()) s.quit()
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work/Chapter_11$ python3 add_html_content.py Output: Password: Mail sent successfully !!
```

在上面的例子中，我们使用了电子邮件包通过Python脚本发送HTML内容作为消息。我们创建了一个`msg`变量，其中存储了HTML内容。

现在，我们将看如何添加附件并通过Python脚本发送它。为此，请创建一个名为`add_attachment.py`的脚本，并在其中编写以下内容：

```py
import os import smtplib from email.mime.text import MIMEText from email.mime.image import MIMEImage from email.mime.multipart import MIMEMultipart import getpass host_name = 'smtp.gmail.com' port = 465 sender = '*sender_emailid*' password = getpass.getpass() receiver = '*receiver_emailid*' text = MIMEMultipart() text['Subject'] = 'Test Attachment' text['From'] = sender text['To'] = receiver txt = MIMEText('Sending a sample image.') text.attach(txt) f_path = 'path_of_file' with open(f_path, 'rb') as f:
 img = MIMEImage(f.read()) img.add_header('Content-Disposition',
 'attachment', filename=os.path.basename(f_path)) text.attach(img) s = smtplib.SMTP_SSL(host_name, port) print("Attachment sent successfully !!") s.login(sender, password) s.sendmail(sender, receiver, text.as_string()) s.quit()
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work/Chapter_11$ python3 add_attachment.py Output: Password: Attachment sent successfully !!
```

在上面的例子中，我们将一张图片作为附件发送给接收者。我们提到了发件人和收件人的电子邮件ID。接下来，在`f_path`中，我们提到了我们要发送的图片的路径。接下来，我们将该图片作为附件发送给接收者。

# POP3和IMAP服务器

在本节中，您将学习如何通过POP和IMAP服务器接收电子邮件。Python提供了`poplib`和`imaplib`库，用于通过Python脚本接收电子邮件。

# 使用poplib库接收电子邮件

**POP3**代表**邮局协议第3版**。这个标准协议帮助您从远程服务器接收电子邮件到我们的本地计算机。POP3的主要优势在于它允许我们将电子邮件下载到本地计算机上，并离线阅读已下载的电子邮件。

POP3协议在两个端口上运行：

+   端口`110`：默认的非加密端口

+   端口`995`：加密端口

现在，我们将看一些例子。首先，我们将看一个例子，其中我们收到了一些电子邮件。为此，请创建一个名为`number_of_emails.py`的脚本，并在其中编写以下内容：

```py
import poplib import getpass pop3_server = 'pop.gmail.com' username = 'Emaild_address'
password = getpass.getpass()
email_obj = poplib.POP3_SSL(pop3_server) print(email_obj.getwelcome()) email_obj.user(username) email_obj.pass_(password) email_stat = email_obj.stat() print("New arrived e-Mails are : %s (%s bytes)" % email_stat)
```

运行脚本，如下所示：

```py
student@ubuntu:~$ python3 number_of_emails.py
```

作为输出，您将得到邮箱中存在的电子邮件数量。

在上面的示例中，首先我们导入了`poplib`库，该库用于Python的POP3协议，以安全地接收电子邮件。然后，我们声明了特定的电子邮件服务器和我们的电子邮件凭据，即我们的用户名和密码。之后，我们打印来自服务器的响应消息，并向POP3 SSL服务器提供用户名和密码。登录后，我们获取邮箱统计信息并将其以电子邮件数量的形式打印到终端。

现在，我们将编写一个脚本来获取最新的电子邮件。为此，请创建一个名为`latest_email.py`的脚本，并在其中编写以下内容：

```py
import poplib
import getpass pop3_server = 'pop.gmail.com' username = 'Emaild_address' password = getpass.getpass() email_obj = poplib.POP3_SSL(pop3_server) print(email_obj.getwelcome()) email_obj.user(username) email_obj.pass_(password) print("\nLatest Mail\n") latest_email = email_obj.retr(1) print(latest_email[1])
```

运行脚本，如下所示：

```py
student@ubuntu:~$ python3 latest_email.py
```

作为输出，您将获得您收件箱中收到的最新邮件。

在上面的示例中，我们导入了用于Python的`poplib`库，以安全地提供POP3协议以接收电子邮件。在声明了特定的电子邮件服务器和用户名和密码之后，我们打印了来自服务器的响应消息，并向POP3 SSL服务器提供了用户名和密码。然后，我们从邮箱中获取了最新的电子邮件。

现在，我们将编写一个脚本来获取所有的电子邮件。为此，请创建一个名为`all_emails.py`的脚本，并在其中编写以下内容：

```py
import poplib
import getpass pop3_server = 'pop.gmail.com' username = 'Emaild_address' password = getpass.getpass() email_obj = poplib.POP3_SSL(pop3_server) print(email_obj.getwelcome()) email_obj.user(username) email_obj.pass_(password) email_stat = email_obj.stat() NumofMsgs = email_stat[0] for i in range(NumofMsgs):
 for mail in email_obj.retr(i+1)[1]: print(mail)
```

运行脚本，如下所示：

```py
student@ubuntu:~$ python3 latest_email.py
```

作为输出，您将获得您收件箱中收到的所有电子邮件。

# 使用imaplib库接收电子邮件

IMAP代表Internet消息访问协议。它用于通过本地计算机访问远程服务器上的电子邮件。IMAP允许多个客户端同时访问您的电子邮件。当您通过不同位置访问电子邮件时，IMAP更适用。

IMAP协议在两个端口上运行：

+   端口`143`：默认非加密端口

+   端口`993`：加密端口

现在，我们将看到使用`imaplib`库的示例。创建一个名为`imap_email.py`的脚本，并在其中编写以下内容：

```py
import imaplib import pprint
import getpass imap_server = 'imap.gmail.com' username = 'Emaild_address'
password = getpass.getpass()imap_obj = imaplib.IMAP4_SSL(imap_server) imap_obj.login(username, password) imap_obj.select('Inbox') temp, data_obj = imap_obj.search(None, 'ALL') for data in data_obj[0].split():
 temp, data_obj = imap_obj.fetch(data, '(RFC822)') print('Message: {0}\n'.format(data)) pprint.pprint(data_obj[0][1]) break imap_obj.close()
```

运行脚本，如下所示：

```py
student@ubuntu:~$ python3 imap_email.py
```

作为输出，您将获得指定文件夹中的所有电子邮件。

在上面的示例中，首先我们导入了`imaplib`库，该库用于Python通过IMAP协议安全地接收电子邮件。然后，我们声明了特定的电子邮件服务器和我们的用户凭据，即我们的用户名和密码。之后，我们向IMAP SSL服务器提供了用户名和密码。我们使用`'select('Inbox')'`函数在`imap_obj`上显示收件箱中的消息。然后，我们使用`for`循环逐个显示已获取的消息。为了显示消息，我们使用“pretty print”——即`pprint.pprint()`函数——因为它会格式化您的对象，将其写入数据流，并将其作为参数传递。最后，连接被关闭。

# 摘要

在本章中，我们学习了如何在Python脚本中编写电子邮件消息。我们还学习了Python的`smtplib`模块，该模块用于通过Python脚本发送和接收电子邮件。我们还学习了如何通过POP3和IMAP协议接收电子邮件。Python提供了`poplib`和`imaplib`库，我们可以使用这些库执行任务。

在下一章中，您将学习有关Telnet和SSH的知识。

# 问题

1.  POP3和IMAP是什么？

1.  break和continue分别用于什么？给出一个适当的例子。

1.  pprint是什么？

1.  什么是负索引，为什么要使用它们？

1.  `pyc`和`py`文件扩展名之间有什么区别？

1.  使用循环生成以下模式：

```py
 1010101
 10101 
 101 
 1  
```
