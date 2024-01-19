# 第八章：处理通信渠道

在本章中，我们将涵盖以下配方：

+   使用电子邮件模板

+   发送单个电子邮件

+   阅读电子邮件

+   将订阅者添加到电子邮件通讯中

+   通过电子邮件发送通知

+   生成短信

+   接收短信

+   创建一个 Telegram 机器人

# 介绍

处理通信渠道是自动化事务可以产生巨大收益的地方。在本配方中，我们将看到如何处理两种最常见的通信渠道——电子邮件，包括新闻通讯，以及通过电话发送和接收短信。

多年来，交付方法中存在相当多的滥用，如垃圾邮件或未经请求的营销信息，这使得与外部工具合作以避免消息被自动过滤器自动拒绝成为必要。我们将在适用的情况下提出适当的注意事项。所有工具都有很好的文档，所以不要害怕阅读它。它们还有很多功能，它们可能能够做一些正是你所寻找的东西。

# 使用电子邮件模板

要发送电子邮件，我们首先需要生成其内容。在本配方中，我们将看到如何生成适当的模板，既以纯文本样式又以 HTML 样式。

# 准备就绪

我们应该首先安装`mistune`模块，它将 Markdown 文档编译为 HTML。我们还将使用`jinja2`模块将 HTML 与我们的文本组合在一起。

```py
$ echo "mistune==0.8.3" >> requirements.txt
$ echo "jinja2==2.20" >> requirements.txt
$ pip install -r requirements.txt
```

在 GitHub 存储库中，有一些我们将使用的模板——`email_template.md`在[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/email_template.md`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/email_template.md)和一个用于样式的模板，`email_styling.html`在[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/email_styling.html`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/email_styling.html)。

# 如何做...

1.  导入模块：

```py
>>> import mistune
>>> import jinja2
```

1.  从磁盘读取两个模板：

```py
>>> with open('email_template.md') as md_file:
...     markdown = md_file.read()

>>> with open('email_styling.html') as styling_file:
...     styling = styling_file.read()
```

1.  定义要包含在模板中的`data`。模板非常简单，只接受一个参数：

```py
>>> data = {'name': 'Seamus'}
```

1.  呈现 Markdown 模板。这会产生`data`的纯文本版本：

```py
>>> text = markdown.format(**data)
```

1.  呈现 Markdown 并添加样式：

```py
>>> html_content = mistune.markdown(text)
>>> html = jinja2.Template(styling).render(content=html_content)
```

1.  将文本和 HTML 版本保存到磁盘以进行检查：

```py
>>> with open('text_version.txt', 'w') as fp:
...     fp.write(text)
>>> with open('html_version.html', 'w') as fp:
...     fp.write(html)
```

1.  检查文本版本：

```py
$ cat text_version.txt
Hi Seamus:

This is an email talking about **things**

### Very important info

1\. One thing
2\. Other thing
3\. Some extra detail

Best regards,

  *The email team*
```

1.  在浏览器中检查 HTML 版本：

![](img/cc6cc1e0-aeaf-4e3a-911b-dacf333609ee.png)

# 它是如何工作的...

第 1 步获取稍后将使用的模块，第 2 步读取将呈现的两个模板。`email_template.md`是内容的基础，它是一个 Markdown 模板。`email_styling.html`是一个包含基本 HTML 环绕和 CSS 样式信息的 HTML 模板。

基本结构是以 Markdown 格式创建内容。这是一个可读的纯文本文件，可以作为电子邮件的一部分发送。然后可以将该内容转换为 HTML，并添加一些样式来创建 HTML 函数。`email_styling.html`有一个内容区域，用于放置从 Markdown 呈现的 HTML。

第 3 步定义了将在`email_template.md`中呈现的数据。这是一个非常简单的模板，只需要一个名为`name`的参数。

在第 4 步，Markdown 模板与`data`一起呈现。这会产生电子邮件的纯文本版本。

第 5 步呈现了`HTML`版本。使用`mistune`将纯文本版本呈现为`HTML`，然后使用`jinja2`模板将其包装在`email_styling.html`中。最终版本是一个独立的 HTML 文档。

最后，我们将两个版本，纯文本（作为`text`）和 HTML（作为`html`），保存到文件中的第 6 步。第 7 步和第 8 步检查存储的值。信息是相同的，但在`HTML`版本中，它是有样式的。

# 还有更多...

使用 Markdown 可以轻松生成包含文本和 HTML 的双重电子邮件。Markdown 在文本格式中非常易读，并且可以自然地呈现为 HTML。也就是说，可以生成完全不同的 HTML 版本，这将允许更多的自定义和利用 HTML 的特性。

完整的 Markdown 语法可以在[`daringfireball.net/projects/markdown/syntax`](https://daringfireball.net/projects/markdown/syntax)找到，最常用元素的好的速查表在[`beegit.com/markdown-cheat-sheet`](https://beegit.com/markdown-cheat-sheet)。

虽然制作电子邮件的纯文本版本并不是绝对必要的，但这是一个很好的做法，表明您关心谁阅读了电子邮件。大多数电子邮件客户端接受 HTML，但并非完全通用。

对于 HTML 电子邮件，请注意整个样式应该包含在电子邮件中。这意味着 CSS 需要嵌入到 HTML 中。避免进行可能导致电子邮件在某些电子邮件客户端中无法正确呈现，甚至被视为垃圾邮件的外部调用。

`email_styling.html`中的样式基于可以在[`markdowncss.github.io/`](http://markdowncss.github.io/)找到的`modest`样式。还有更多可以使用的 CSS 样式，可以在 Google 中搜索找到更多。请记住删除任何外部引用，如前面所讨论的。

可以通过以`base64`格式对图像进行编码，以便直接嵌入 HTML`img`标签中，而不是添加引用，将图像包含在 HTML 中。

```py
>>> import base64
>>> with open("image.png",'rb') as file:
... encoded_data = base64.b64encode(file) >>> print "<img src='data:image/png;base64,{data}'/>".format(data=encoded_data)
```

您可以在[`css-tricks.com/data-uris/`](https://css-tricks.com/data-uris/)的文章中找到有关此技术的更多信息。

`mistune`完整文档可在[`mistune.readthedocs.io/en/latest/`](http://mistune.readthedocs.io/en/latest/)找到，`jinja2`文档可在[`jinja.pocoo.org/docs/2.10/`](http://jinja.pocoo.org/docs/2.10/)找到。

# 另请参阅

+   第五章中的*在 Markdown 中格式化文本*食谱，*生成精彩的报告*

+   第五章中的*使用模板生成报告食谱*，*生成精彩的报告*

+   第五章中的*发送事务性电子邮件*食谱，*生成精彩的报告*

# 发送单个电子邮件

发送电子邮件的最基本方法是从电子邮件帐户发送单个电子邮件。这个选项只建议用于非常零星的使用，但对于简单的目的，比如每天向受控地址发送几封电子邮件，这可能足够了。

不要使用此方法向分发列表或具有未知电子邮件地址的客户批量发送电子邮件。您可能因反垃圾邮件规则而被服务提供商禁止。有关更多选项，请参阅本章中的其他食谱。

# 准备工作

对于这个示例，我们将需要一个带有服务提供商的电子邮件帐户。根据要使用的提供商有一些小的差异，但我们将使用 Gmail 帐户，因为它非常常见且免费访问。

由于 Gmail 的安全性，我们需要创建一个特定的应用程序密码，用于发送电子邮件。请按照这里的说明操作：[`support.google.com/accounts/answer/185833`](https://support.google.com/accounts/answer/185833)。这将有助于为此示例生成一个密码。记得为邮件访问创建它。您可以随后删除密码以将其删除。

我们将使用 Python 标准库中的`smtplib`模块。

# 如何做...

1.  导入`smtplib`和`email`模块：

```py
>>> import smtplib
>>> from email.mime.multipart import MIMEMultipart
>>> from email.mime.text import MIMEText
```

1.  设置凭据，用您自己的凭据替换这些。出于测试目的，我们将发送到相同的电子邮件，但请随意使用不同的地址：

```py
>>> USER = 'your.account@gmail.com'
>>> PASSWORD = 'YourPassword'
>>> sent_from = USER
>>> send_to = [USER]
```

1.  定义要发送的数据。注意两种选择，纯文本和 HTML：

```py
>>> text = "Hi!\nThis is the text version linking to https://www.packtpub.com/\nCheers!"
>>> html = """<html><head></head><body>
... <p>Hi!<br>
... This is the HTML version linking to <a href="https://www.packtpub.com/">Packt</a><br>
... </p>
... </body></html>
"""
```

1.  将消息组成为`MIME`多部分，包括`主题`，`收件人`和`发件人`：

```py
>>> msg = MIMEMultipart('alternative')
>>> msg['Subject'] = 'An interesting email'
>>> msg['From'] = sent_from
>>> msg['To'] = ', '.join(send_to)
```

1.  填写电子邮件的数据内容部分：

```py
>>> part_plain = MIMEText(text, 'plain')
>>> part_html = MIMEText(html, 'html')
>>> msg.attach(part_plain)
>>> msg.attach(part_html)
```

1.  使用`SMTP SSL`协议发送电子邮件：

```py
>>> with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
...     server.login(USER, PASSWORD)
...     server.sendmail(sent_from, send_to, msg.as_string())
```

1.  邮件已发送。检查您的电子邮件帐户是否收到了消息。检查*原始电子邮件*，您可以看到完整的原始电子邮件，其中包含 HTML 和纯文本元素。电子邮件已被编辑：

![](img/eedfbdbd-64da-4f0e-abd9-7fa23bbaf20d.png)

# 工作原理...

在第 1 步之后，从`stmplib`和`email`进行相关导入，第 2 步定义了从 Gmail 获取的凭据。

第 3 步显示了将要发送的 HTML 和文本。它们是替代方案，因此它们应该呈现相同的信息，但以不同的格式呈现。

基本的消息信息在第 4 步中设置。它指定了电子邮件的主题，以及*from*和*to*。第 5 步添加了多个部分，每个部分都有适当的`MIMEText`类型。

最后添加的部分是首选的替代方案，根据`MIME`格式，因此我们最后添加了`HTML`部分。

第 6 步建立与服务器的连接，使用凭据登录并发送消息。它使用`with`上下文来获取连接。

如果凭据出现错误，它将引发一个异常，显示用户名和密码不被接受。

# 还有更多...

请注意，`sent_to`是一个地址列表。您可以将电子邮件发送给多个地址。唯一的注意事项在第 4 步，需要将其指定为所有地址的逗号分隔值列表。

尽管可以将`sent_from`标记为与发送电子邮件时使用的地址不同，但并不建议这样做。这可能被解释为试图伪造电子邮件的来源，并引发检测为垃圾邮件来源的迹象。

此处使用的服务器`smtp.gmail.com`*是 Gmail 指定的服务器，并且`SMTPS`（安全`SMTP`）的定义端口为`465`。Gmail 还接受端口`587`，这是标准端口，但需要您通过调用`.starttls`指定会话的类型，如下面的代码所示：

```py
 with smtplib.SMTP('smtp.gmail.com', 587) as server:
    server.starttls()
    server.login(USER, PASSWORD)
    server.sendmail(sent_from, send_to, msg.as_string())
```

如果您对这些差异和两种协议的更多细节感兴趣，可以在这篇文章中找到更多信息：[`www.fastmail.com/help/technical/ssltlsstarttls.html`](https://www.fastmail.com/help/technical/ssltlsstarttls.html)。

完整的`smtplib`文档可以在[`docs.python.org/3/library/smtplib.html`](https://docs.python.org/3/library/smtplib.html)找到，`email`模块中包含有关电子邮件不同格式的信息，包括`MIME`类型的示例，可以在这里找到：[`docs.python.org/3/library/email.html`](https://docs.python.org/3/library/email.html)。

# 另请参阅

+   *使用电子邮件模板*示例

+   *发送单个电子邮件*示例

# 阅读电子邮件

在本示例中，我们将看到如何从帐户中读取电子邮件。我们将使用`IMAP4`标准，这是最常用的用于阅读电子邮件的标准。

一旦读取，电子邮件可以被自动处理和分析，以生成智能自动响应、将电子邮件转发到不同的目标、聚合结果进行监控等操作。选项是无限的！

# 准备就绪

对于此示例，我们将需要一个带有服务提供商的电子邮件帐户。基于要使用的提供商的小差异，但我们将使用 Gmail 帐户，因为它非常常见且免费访问。

由于 Gmail 的安全性，我们需要创建一个特定的应用程序密码来发送电子邮件。请按照这里的说明操作：[`support.google.com/accounts/answer/185833`](https://support.google.com/accounts/answer/185833)。这将为此示例生成一个密码。记得为邮件创建它。您可以在之后删除密码以将其删除。

我们将使用 Python 标准库中的`imaplib`模块。

该示例将读取最后收到的电子邮件，因此您可以使用它更好地控制将要读取的内容。我们将发送一封看起来像是发送给支持的简短电子邮件。

# 如何做...

1.  导入`imaplib`和`email`模块：

```py
>>> import imaplib
>>> import email
>>> from email.parser import BytesParser, Parser
>>> from email.policy import default
```

1.  设置凭据，用您自己的凭据替换这些：

```py
>>> USER = 'your.account@gmail.com'
>>> PASSWORD = 'YourPassword'
```

1.  连接到电子邮件服务器：

```py
>>> mail = imaplib.IMAP4_SSL('imap.gmail.com')
>>> mail.login(USER, PASSWORD)
```

1.  选择收件箱文件夹：

```py
>>> mail.select('inbox')
```

1.  读取所有电子邮件 UID 并检索最新收到的电子邮件：

```py
>>> result, data = mail.uid('search', None, 'ALL')
>>> latest_email_uid = data[0].split()[-1]
>>> result, data = mail.uid('fetch', latest_email_uid, '(RFC822)')
>>> raw_email = data[0][1]
```

1.  将电子邮件解析为 Python 对象：

```py
>>> email_message = BytesParser(policy=default).parsebytes(raw_email)
```

1.  显示电子邮件的主题和发件人：

```py
>>> email_message['subject']
'[Ref ABCDEF] Subject: Product A'
>>> email.utils.parseaddr(email_message['From'])
('Sender name', 'sender@gmail.com')
```

1.  检索文本的有效载荷：

```py
>>> email_type = email_message.get_content_maintype()
>>> if email_type == 'multipart':
...     for part in email_message.get_payload():
...         if part.get_content_type() == 'text/plain':
...             payload = part.get_payload()
... elif email_type == 'text':
...     payload = email_message.get_payload()
>>> print(payload)
Hi:

  I'm having difficulties getting into my account. What was the URL, again?

  Thanks!
    A confuser customer
```

# 工作原理...

导入将要使用的模块并定义凭据后，我们在第 3 步连接到服务器。

第 4 步连接到`inbox`。这是 Gmail 中包含收件箱的默认文件夹，其中包含收到的电子邮件。

当然，您可能需要阅读不同的文件夹。您可以通过调用`mail.list()`来获取所有文件夹的列表。

在第 5 步，首先通过调用`.uid('search', None, "ALL")`检索收件箱中所有电子邮件的 UID 列表。然后通过`fetch`操作和`.uid('fetch', latest_email_uid, '(RFC822)')`再次从服务器检索最新收到的电子邮件。这将以 RFC822 格式检索电子邮件，这是标准格式。请注意，检索电子邮件会将其标记为已读。

`.uid`命令允许我们调用 IMAP4 命令，返回一个带有结果（`OK`或`NO`）和数据的元组。如果出现错误，它将引发适当的异常。

`BytesParser`模块用于将原始的`RFC822`电子邮件转换为 Python 对象。这是在第 6 步完成的。

元数据，包括主题、发件人和时间戳等详细信息，可以像字典一样访问，如第 7 步所示。地址可以从原始文本格式解析为带有`email.utils.parseaddr`的部分。

最后，内容可以展开和提取。如果电子邮件的类型是多部分的，可以通过迭代`.get_payload()`来提取每个部分。最容易处理的是`plain/text`，因此假设它存在，第 8 步中的代码将提取它。

电子邮件正文存储在`payload`变量中。

# 还有更多...

在第 5 步，我们正在检索收件箱中的所有电子邮件，但这并非必要。搜索可以进行过滤，例如只检索最近一天的电子邮件：

```py
import datetime
since = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%d-%b-%Y")
result, data = mail.uid('search', None, f'(SENTSINCE {since})')
```

这将根据电子邮件的日期进行搜索。请注意，分辨率以天为单位。

还有更多可以通过`IMAP4`完成的操作。查看 RFC 3501  [`tools.ietf.org/html/rfc3501`](https://tools.ietf.org/html/rfc3501)和 RFC 6851 [`tools.ietf.org/html/rfc6851`](https://tools.ietf.org/html/rfc6851)以获取更多详细信息。

RFC 描述了 IMAP4 协议，可能有点枯燥。检查可能的操作将让您了解详细调查的可能性，可能通过 Google 搜索示例。

可以解析和处理电子邮件的主题和正文，以及日期、收件人、发件人等其他元数据。例如，本食谱中检索的主题可以按以下方式处理：

```py
>>> import re
>>> re.search(r'\[Ref (\w+)] Subject: (\w+)', '[Ref ABCDEF] Subject: Product A').groups()
('ABCDEF', 'Product') 
```

有关正则表达式和其他解析信息的更多信息，请参见第一章，*让我们开始自动化之旅*。

# 另请参阅

+   第一章中的*介绍正则表达式*食谱，*让我们开始自动化之旅*

# 向电子邮件通讯订阅者添加订阅者

常见的营销工具是电子邮件通讯。它们是向多个目标发送信息的便捷方式。一个好的通讯系统很难实现，推荐的方法是使用市场上可用的。一个著名的是 MailChimp ([`mailchimp.com/`](https://mailchimp.com/))。

MailChimp 有很多可能性，但与本书相关的有趣之一是其 API，可以编写脚本来自动化工具。这个 RESTful API 可以通过 Python 访问。在这个食谱中，我们将看到如何向现有列表添加更多的订阅者。

# 准备就绪

由于我们将使用 MailChimp，因此需要有一个可用的帐户。您可以在[`login.mailchimp.com/signup/`](https://login.mailchimp.com/signup/)上创建一个免费帐户。

创建帐户后，请确保至少有一个我们将向其添加订阅者的列表。作为注册的一部分，可能已经创建了。它将显示在列表下：

![](img/05182753-f7ff-4002-8e63-40cdbcfe5e36.png)

列表将包含已订阅的用户。

对于 API，我们将需要一个 API 密钥。转到帐户|额外|API 密钥并创建一个新的：

![](img/a2174005-2a85-4c91-880a-a14ec426ca66.png)

我们将使用`requests`模块来访问 API。将其添加到您的虚拟环境中：

```py
$ echo "requests==2.18.3" >> requirements.txt
$ pip install -r requirements.txt
```

MailChimp API 使用**DC**（数据中心）的概念，您的帐户使用它。这可以从您的 API 的最后几位数字中获得，或者从 MailChimp 管理站点的 URL 开头获得。例如，在所有先前的截图中，它是`us19`。

# 如何做...

1.  导入`requests`模块：

```py
>>> import requests
```

1.  定义身份验证和基本 URL。基本 URL 需要在开头加上您的`dc`（例如`us19`）：

```py
>>> API = 'your secret key'
>>> BASE = 'https://<dc>.api.mailchimp.com/3.0'
>>> auth = ('user', API)
```

1.  获取所有列表：

```py
>>> url = f'{BASE}/lists'
>>> response = requests.get(url, auth=auth)
>>> result = response.json()
```

1.  过滤列表以获取所需列表的`href`：

```py
>>> LIST_NAME = 'Your list name'
>>> this_list = [l for l in result['lists'] if l['name'] == LIST_NAME][0]
>>> list_url = [l['href'] for l in this_list['_links'] if l['rel'] == 'self'][0]
```

1.  使用列表 URL，您可以获取列表成员的 URL：

```py
>>> response = requests.get(list_url, auth=auth)
>>> result = response.json()
>>> result['stats']
{'member_count': 1, 'unsubscribe_count': 0, 'cleaned_count': 0, ...}
>>> members_url = [l['href'] for l in result['_links'] if l['rel'] == 'members'][0]
```

1.  可以通过向`members_url`发出`GET`请求来检索成员列表：

```py
>>> response = requests.get(members_url, json=new_member, auth=auth)
>>> result = response.json()
>>> len(result['members'])
1
```

1.  向列表添加新成员：

```py
>>> new_member = {
    'email_address': 'test@test.com',
    'status': 'subscribed',
}
>>> response = requests.post(members_url, json=new_member, auth=auth)
```

1.  使用`GET`获取用户列表会获取到所有用户：

```py
>>> response = requests.post(members_url, json=new_member, auth=auth)
>>> result = response.json()
>>> len(result['members'])
2
```

# 工作原理...

在第 1 步导入 requests 模块后，在第 2 步定义连接的基本值，即基本 URL 和凭据。请注意，对于身份验证，我们只需要 API 密钥作为密码，以及任何用户（如 MailChimp 文档所述：[`developer.mailchimp.com/documentation/mailchimp/guides/get-started-with-mailchimp-api-3/`](https://developer.mailchimp.com/documentation/mailchimp/guides/get-started-with-mailchimp-api-3/)）。

第 3 步检索所有列表，调用适当的 URL。结果以 JSON 格式返回。调用包括具有定义凭据的`auth`参数。所有后续调用都将使用该`auth`参数进行身份验证。

第 4 步显示了如何过滤返回的列表以获取感兴趣的特定列表的 URL。每个返回的调用都包括一系列与相关信息的`_links`列表，使得可以通过 API 进行遍历。

在第 5 步调用列表的 URL。这将返回列表的信息，包括基本统计信息。类似于第 4 步的过滤，我们检索成员的 URL。

由于尺寸限制和显示相关数据，未显示所有检索到的元素。请随时进行交互式分析并了解它们。数据构造良好，遵循 RESTful 的可发现性原则；再加上 Python 的内省能力，使其非常易读和易懂。

第 6 步检索成员列表，向`members_url`发出`GET`请求，可以将其视为单个用户。这可以在网页界面的*Getting Ready*部分中看到。

第 7 步创建一个新用户，并在`members_url`上发布`json`参数中传递的信息，以便将其转换为 JSON 格式。第 7 步检索更新后的数据，显示列表中有一个新用户。

# 还有更多...

完整的 MailChimp API 非常强大，可以执行大量任务。请查看完整的 MailChimp 文档以发现所有可能性：[`developer.mailchimp.com/`](https://developer.mailchimp.com/)。

简要说明一下，超出了本书的范围，请注意向自动列表添加订阅者的法律影响。垃圾邮件是一个严重的问题，有新的法规来保护客户的权利，如 GPDR。确保您有用户的许可才能给他们发送电子邮件。好消息是，MailChimp 自动实现了帮助解决这个问题的工具，如自动退订按钮。

一般的 MailChimp 文档也非常有趣，展示了许多可能性。MailChimp 能够管理通讯和一般的分发列表，但也可以定制生成流程，安排发送电子邮件，并根据参数（如生日）自动向您的受众发送消息。

# 另请参阅

+   *发送单个电子邮件*配方

+   *发送交易电子邮件*配方

# 通过电子邮件发送通知

在这个配方中，我们将介绍如何发送将发送给客户的电子邮件。作为对用户操作的响应发送的电子邮件，例如确认电子邮件或警报电子邮件，称为*交易电子邮件*。由于垃圾邮件保护和其他限制，最好使用外部工具来实现这种类型的电子邮件。

在这个配方中，我们将使用 Mailgun ([`www.mailgun.com`](https://www.mailgun.com))，它能够发送这种类型的电子邮件，并与之通信。

# 准备工作

我们需要在 Mailgun 中创建一个帐户。转到[`signup.mailgun.com`](https://signup.mailgun.com/new/signup)创建一个。请注意，信用卡信息是可选的。

注册后，转到域以查看是否有沙箱环境。我们可以使用它来测试功能，尽管它只会向注册的测试电子邮件帐户发送电子邮件。API 凭据将显示在那里：

![](img/f287dcdc-a84e-4cd0-9df3-abae2afae51a.png)

我们需要注册帐户，以便我们将作为*授权收件人*收到电子邮件。您可以在此处添加：

![](img/eacf6ffc-c628-4baf-8292-c4eec436fd98.png)

要验证帐户，请检查授权收件人的电子邮件并确认。电子邮件地址现在已准备好接收测试邮件：

![](img/485590b4-1022-4f34-82ad-2859dac25be9.png)

我们将使用`requests`模块来连接 Mailgun API。在虚拟环境中安装它：

```py
$ echo "requests==2.18.3" >> requirements.txt
$ pip install -r requirements.txt
```

一切准备就绪，可以发送电子邮件，但请注意只发送给授权收件人。要能够在任何地方发送电子邮件，我们需要设置域。请参阅 Mailgun 文档：[`documentation.mailgun.com/en/latest/quickstart-sending.html#verify-your-domain`](https://documentation.mailgun.com/en/latest/quickstart-sending.html#verify-your-domain)。

# 如何做...

1.  导入`requests`模块：

```py
>>> import requests
```

1.  准备凭据，以及要发送和接收的电子邮件。请注意，我们正在使用模拟发件人：

```py
>>> KEY = 'YOUR-SECRET-KEY'
>>> DOMAIN = 'YOUR-DOMAIN.mailgun.org'
>>> TO = 'YOUR-AUTHORISED-RECEIVER'
```

```py

>>> FROM = f'sender@{DOMAIN}'
>>> auth = ('api', KEY)
```

1.  准备要发送的电子邮件。这里有 HTML 版本和备用纯文本版本：

```py
>>> text = "Hi!\nThis is the text version linking to https://www.packtpub.com/\nCheers!"
>>> html = '''<html><head></head><body>
...     <p>Hi!<br>
...        This is the HTML version linking to <a href="https://www.packtpub.com/">Packt</a><br>
...     </p>  
...   </body></html>'''
```

1.  设置要发送到 Mailgun 的数据：

```py
>>> data = {
...     'from': f'Sender <{FROM}>',
...     'to': f'Jaime Buelta <{TO}>',
...     'subject': 'An interesting email!',
...     'text': text,
...     'html': html,
... }
```

1.  调用 API：

```py
>>> response = requests.post(f"https://api.mailgun.net/v3/{DOMAIN}/messages", auth=auth, data=data)
>>> response.json()
{'id': '<YOUR-ID.mailgun.org>', 'message': 'Queued. Thank you.'}
```

1.  检索事件并检查电子邮件是否已发送：

```py
>>> response_events = requests.get(f'https://api.mailgun.net/v3/{DOMAIN}/events', auth=auth)
>>> response_events.json()['items'][0]['recipient'] == TO
True
>>> response_events.json()['items'][0]['event']
'delivered'
```

1.  邮件应该出现在您的收件箱中。由于它是通过沙箱环境发送的，请确保在直接显示时检查您的垃圾邮件文件夹。

# 它是如何工作的...

第 1 步导入`requests`模块以供以后使用。第 2 步定义了凭据和消息中的基本信息，并应从 Mailgun Web 界面中提取，如前所示。

第 3 步定义将要发送的电子邮件。第 4 步将信息结构化为 Mailgun 所期望的方式。请注意`html`和`text`字段。默认情况下，它将设置 HTML 为首选项，并将纯文本选项作为备选项。`TO`和`FROM`的格式应为`Name <address>`格式。您可以使用逗号将多个收件人分隔在`TO`中。

在第 5 步进行 API 调用。这是对消息端点的`POST`调用。数据以标准方式传输，并使用`auth`参数进行基本身份验证。请注意第 2 步中的定义。所有对 Mailgun 的调用都应包括此参数。它返回一条消息，通知您它已成功排队了消息。

在第 6 步，通过`GET`请求调用检索事件。这将显示最新执行的操作，其中最后一个将是最近的发送。还可以找到有关交付的信息。

# 还有更多...

要发送电子邮件，您需要设置用于发送电子邮件的域，而不是使用沙箱环境。您可以在这里找到说明：[`documentation.mailgun.com/en/latest/quickstart-sending.html#verify-your-domain`](https://documentation.mailgun.com/en/latest/quickstart-sending.html#verify-your-domain)。这需要您更改 DNS 记录以验证您是其合法所有者，并提高电子邮件的可交付性。

电子邮件可以以以下方式包含附件：

```py
attachments = [("attachment", ("attachment1.jpg", open("image.jpg","rb").read())),
               ("attachment", ("attachment2.txt", open("text.txt","rb").read()))]
response = requests.post(f"https://api.mailgun.net/v3/{DOMAIN}/messages",
                         auth=auth, files=attachments, data=data)
```

数据可以包括常规信息，如`cc`或`bcc`，但您还可以使用`o:deliverytime`参数将交付延迟最多三天：

```py
import datetime
import email.utils
delivery_time = datetime.datetime.now() + datetime.timedelta(days=1)
data = {
    ...
    'o:deliverytime': email.utils.format_datetime(delivery_time),
}
```

Mailgun 还可以用于接收电子邮件并在其到达时触发流程，例如，根据规则转发它们。查看 Mailgun 文档以获取更多信息。

完整的 Mailgun 文档可以在这里找到，[`documentation.mailgun.com/en/latest/quickstart.html`](https://documentation.mailgun.com/en/latest/quickstart.html)。一定要检查他们的*最佳实践*部分([`documentation.mailgun.com/en/latest/best_practices.html#email-best-practices`](https://documentation.mailgun.com/en/latest/best_practices.html#email-best-practices))，以了解发送电子邮件的世界以及如何避免被标记为垃圾邮件。

# 另请参阅

+   *使用电子邮件模板*配方

+   *发送单个电子邮件*配方

# 生成短信

最广泛使用的通信渠道之一是短信。短信非常方便用于分发信息。

短信可以用于营销目的，也可以用作警报或发送通知的方式，或者最近非常常见的是作为实施双因素身份验证系统的一种方式。

我们将使用 Twilio，这是一个提供 API 以轻松发送短信的服务。

# 准备就绪

我们需要在[`www.twilio.com/`](https://www.twilio.com/)为 Twilio 创建一个帐户。转到该页面并注册一个新帐户。

您需要按照说明设置一个电话号码来接收消息。您需要输入发送到此电话的代码或接听电话以验证此线路。

创建一个新项目并检查仪表板。从那里，您将能够创建第一个电话号码，能够接收和发送短信：

![](img/93858574-0d63-4bac-b762-e8e0f5435059.png)

一旦号码配置完成，它将出现在所有产品和服务 | 电话号码*.*的活动号码部分。

在主仪表板上，检查`ACCOUNT SID`和`AUTH TOKEN`。稍后将使用它们。请注意，您需要显示身份验证令牌。

我们还需要安装`twilio`模块。将其添加到您的虚拟环境中：

```py
$ echo "twilio==6.16.1" >> requirements.txt
$ pip install -r requirements.txt
```

请注意，接收者电话号码只能是经过试用账户验证的号码。您可以验证多个号码；请参阅[`support.twilio.com/hc/en-us/articles/223180048-Adding-a-Verified-Phone-Number-or-Caller-ID-with-Twilio`](https://support.twilio.com/hc/en-us/articles/223180048-Adding-a-Verified-Phone-Number-or-Caller-ID-with-Twilio)上的文档。

# 如何做...

1.  从`twilio`模块导入`Client`：

```py
>>> from twilio.rest import Client
```

1.  在之前从仪表板获取的身份验证凭据。还要设置您的 Twilio 电话号码；例如，这里我们设置了`+353 12 345 6789`，一个虚假的爱尔兰号码。它将是您国家的本地号码：

```py
>>> ACCOUNT_SID = 'Your account SID'
>>> AUTH_TOKEN = 'Your secret token'
>>> FROM = '+353 12 345 6789'
```

1.  启动`client`以访问 API：

```py
>>> client = Client(ACCOUNT_SID, AUTH_TOKEN)
```

1.  向您授权的电话号码发送一条消息。请注意`from_`末尾的下划线：

```py
>>> message = client.messages.create(body='This is a test message from Python!', 
                                     from_=FROM, 
                                     to='+your authorised number')
```

1.  您将收到一条短信到您的手机：

![](img/89dfa2a3-a312-4789-bbbe-74a394a84b48.png)

# 它是如何工作的...

使用 Twilio 客户端发送消息非常简单。

在第 1 步，我们导入`Client`，并准备在第 2 步配置的凭据和电话号码。

第 3 步使用适当的身份验证创建客户端，并在第 4 步发送消息。

请注意，`to`号码需要是试用帐户中经过身份验证的号码之一，否则将产生错误。您可以添加更多经过身份验证的号码；请查看 Twilio 文档。

从试用帐户发送的所有消息都将在短信中包含该详细信息，正如您在第 5 步中所看到的。

# 还有更多...

在某些地区（在撰写本文时为美国和加拿大），短信号码具有发送 MMS 消息（包括图像）的功能。要将图像附加到消息中，请添加`media_url`参数和要发送的图像的 URL：

```py
client.messages.create(body='An MMS message',
                       media_url='http://my.image.com/image.png', 
                       from_=FROM, 
                       to='+your authorised number')
```

客户端基于 RESTful API，并允许您执行多个操作，例如创建新的电话号码，或首先获取一个可用的号码，然后购买它：

```py
available_numbers = client.available_phone_numbers("IE").local.list()
number = available_numbers[0]
new_number = client.incoming_phone_numbers.create(phone_number=number.phone_number)
```

查看文档以获取更多可用操作，但大多数仪表板的点按操作都可以以编程方式执行。

Twilio 还能够执行其他电话服务，如电话呼叫和文本转语音。请在完整文档中查看。

完整的 Twilio 文档在此处可用：[`www.twilio.com/docs/`](https://www.twilio.com/docs/)。

# 另请参阅

+   *接收短信*配方

+   *创建 Telegram 机器人*配方

# 接收短信

短信也可以自动接收和处理。这使得可以提供按请求提供信息的服务（例如，发送 INFO GOALS 以接收足球联赛的结果），但也可以进行更复杂的流程，例如在机器人中，它可以与用户进行简单的对话，从而实现诸如远程配置恒温器之类的丰富服务。

每当 Twilio 接收到您注册的电话号码之一的短信时，它会执行对公开可用的 URL 的请求。这在服务中进行配置，这意味着它应该在您的控制之下。这会产生一个问题，即在互联网上有一个在您控制之下的 URL。这意味着仅仅您的本地计算机是行不通的，因为它是不可寻址的。我们将使用 Heroku（[`heroku.com`](http://heroku.com)）来提供一个可用的服务，但也有其他选择。Twilio 文档中有使用`grok`的示例，它允许通过在公共地址和您的本地开发环境之间创建隧道来进行本地开发。有关更多详细信息，请参见此处：[`www.twilio.com/blog/2013/10/test-your-webhooks-locally-with-ngrok.html`](https://www.twilio.com/blog/2013/10/test-your-webhooks-locally-with-ngrok.html)。

这种操作方式在通信 API 中很常见。值得注意的是，Twilio 有一个 WhatsApp 的 beta API，其工作方式类似。请查看文档以获取更多信息：[`www.twilio.com/docs/sms/whatsapp/quickstart/python`](https://www.twilio.com/docs/sms/whatsapp/quickstart/python)。

# 准备就绪

我们需要在[`www.twilio.com/`](https://www.twilio.com/)为 Twilio 创建一个帐户。有关详细说明，请参阅*准备就绪*部分中*生成短信*配方。

对于这个配方，我们还需要在 Heroku（[`www.heroku.com/`](https://www.heroku.com/)）中设置一个 Web 服务，以便能够创建一个能够接收发送给 Twilio 的短信的 Webhook。因为这个配方的主要目标是短信部分，所以在设置 Heroku 时我们将简洁一些，但您可以参考其出色的文档。它非常易于使用：

1.  在 Heroku 中创建一个帐户。

1.  您需要安装 Heroku 的命令行界面（所有平台的说明都在[`devcenter.heroku.com/articles/getting-started-with-python#set-up`](https://devcenter.heroku.com/articles/getting-started-with-python#set-up)），然后登录到命令行：

```py
$ heroku login
Enter your Heroku credentials.
Email: your.user@server.com
Password:
```

1.  从[`github.com/datademofun/heroku-basic-flask`](https://github.com/datademofun/heroku-basic-flask)下载一个基本的 Heroku 模板。我们将把它用作服务器的基础。

1.  将`twilio`客户端添加到`requirements.txt`文件中：

```py
$ echo "twilio" >> requirements.txt
```

1.  用 GitHub 中的`app.py`替换`app.py`：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/app.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/app.py)。

您可以保留现有的`app.py`来检查模板示例和 Heroku 的工作原理。查看[`github.com/datademofun/heroku-basic-flask`](https://github.com/datademofun/heroku-basic-flask)中的 README。

1.  完成后，将更改提交到 Git：

```py
$ git add .
$ git commit -m 'first commit'
```

1.  在 Heroku 中创建一个新服务。它将随机生成一个新的服务名称（我们在这里使用`service-name-12345`）。此 URL 是可访问的：

```py
$ heroku create
Creating app... done, ⬢ SERVICE-NAME-12345
https://service-name-12345.herokuapp.com/ | https://git.heroku.com/service-name-12345.git
```

1.  部署服务。在 Heroku 中，部署服务会将代码推送到远程 Git 服务器：

```py
$ git push heroku master
...
remote: Verifying deploy... done.
To https://git.heroku.com/service-name-12345.git
 b6cd95a..367a994 master -> master
```

1.  检查 Webhook URL 的服务是否正在运行。请注意，它显示为上一步的输出。您也可以在浏览器中检查：

```py
$ curl https://service-name-12345.herokuapp.com/
All working!
```

# 如何做...

1.  转到 Twilio 并访问 PHONE NUMBER 部分。配置 Webhook URL。这将使 URL 在每次收到短信时被调用。转到 All Products and Services | Phone Numbers 中的 Active Numbers 部分，并填写 Webhook。请注意 Webhook 末尾的`/sms`。单击保存：

![](img/278c3fc0-7ec7-4567-815a-060e27cd40f0.png)

1.  服务现在已经启动并可以使用。向您的 Twilio 电话号码发送短信，您应该会收到自动回复：

![](img/9853424c-d223-4767-88b5-55a161028b3a.png)

请注意，模糊的部分应该用您的信息替换。

如果您有试用账户，您只能向您授权的电话号码之一发送消息，所以您需要从它们发送文本。

# 它是如何工作的...

第 1 步设置了 Webhook，因此 Twilio 在电话线上收到短信时调用您的 Heroku 应用程序。

让我们看看`app.py`中的代码，看看它是如何工作的。这里为了清晰起见对其进行了编辑；请在[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/app.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/app.py)中查看完整文件：

```py
...
@app.route('/')
def homepage():
    return 'All working!'

@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():
    from_number = request.form['From']
    body = request.form['Body']
    resp = MessagingResponse()
    msg = (f'Awwwww! Thanks so much for your message {from_number}, '
           f'"{body}" to you too.')

    resp.message(msg)
    return str(resp)
...
```

`app.py`可以分为三个部分——文件开头的 Python 导入和 Flask 应用程序的启动，这只是设置 Flask（此处不显示）；调用`homepage`，用于测试服务器是否正常工作；和`sms_reply`，这是魔术发生的地方。

`sms_reply`函数从`request.form`字典中获取发送短信的电话号码以及消息的正文。然后，在`msg`中组成一个响应，将其附加到一个新的`MessagingResponse`，并返回它。

我们正在将用户的消息作为一个整体使用，但请记住第一章中提到的解析文本的所有技术，*让我们开始自动化之旅*。它们都适用于在此处检测预定义操作或任何其他文本处理。

返回的值将由 Twilio 发送回发送者，产生步骤 2 中看到的结果。

# 还有更多...

要能够生成自动对话，对话的状态应该被存储。对于高级状态，它可能应该被存储在数据库中，生成一个流程，但对于简单情况，将信息存储在`session`中可能足够了。会话能够在 cookies 中存储信息，这些信息在相同的来去电话号码组合之间是持久的，允许您在消息之间检索它。

例如，此修改将返回不仅发送正文，还有先前的正文。只包括相关部分：

```py
app = Flask(__name__)
app.secret_key = b'somethingreallysecret!!!!'
... 
@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():
    from_number = request.form['From']
    last_message = session.get('MESSAGE', None)
    body = request.form['Body']
    resp = MessagingResponse()
    msg = (f'Awwwww! Thanks so much for your message {from_number}, '
           f'"{body}" to you too. ')
    if last_message:
        msg += f'Not so long ago you said "{last_message}" to me..'
    session['MESSAGE'] = body
    resp.message(msg)
    return str(resp)
```

上一个`body`存储在会话的`MESSAGE`键中，会话会被保留。注意使用会话数据需要秘密密钥的要求。阅读此处的信息：[`flask.pocoo.org/docs/1.0/quickstart/?highlight=session#sessions`](http://flask.pocoo.org/docs/1.0/quickstart/?highlight=session#sessions)。

要在 Heroku 中部署新版本，将新的`app.py`提交到 Git，然后执行`git push heroku master`。新版本将自动部署！

因为这个食谱的主要目标是演示如何回复，Heroku 和 Flask 没有详细描述，但它们都有很好的文档。Heroku 的完整文档可以在这里找到：[`devcenter.heroku.com/categories/reference`](https://devcenter.heroku.com/categories/reference)，Flask 的文档在这里：[`flask.pocoo.org/docs/`](http://flask.pocoo.org/docs/)。

请记住，使用 Heroku 和 Flask 只是为了方便这个食谱，因为它们是很好和易于使用的工具。有多种替代方案，只要您能够公开一个 URL，Twilio 就可以调用它。还要检查安全措施，以确保对此端点的请求来自 Twilio：[`www.twilio.com/docs/usage/security#validating-requests`](https://www.twilio.com/docs/usage/security#validating-requests)。

Twilio 的完整文档可以在这里找到：[`www.twilio.com/docs/`](https://www.twilio.com/docs/)。

# 另请参阅

+   *生成短信*食谱

+   *创建 Telegram 机器人*食谱

# 创建一个 Telegram 机器人

Telegram Messenger 是一个即时通讯应用程序，对创建机器人有很好的支持。机器人是旨在产生自动对话的小型应用程序。机器人的重要承诺是作为可以产生任何类型对话的机器，完全无法与人类对话区分开来，并通过*Turing 测试*，但这个目标对大部分来说是相当雄心勃勃且不现实的。

图灵测试是由艾伦·图灵于 1951 年提出的。两个参与者，一个人类和一个人工智能（机器或软件程序），通过文本（就像在即时通讯应用程序中）与一个人类评委进行交流，评委决定哪一个是人类，哪一个不是。如果评委只能猜对一半的时间，就无法轻易区分，因此人工智能通过了测试。这是对衡量人工智能的最早尝试之一。

但是，机器人也可以以更有限的方式非常有用，类似于需要按*2*来检查您的账户，按*3*来报告遗失的卡片的电话系统。在这个食谱中，我们将看到如何生成一个简单的机器人，用于显示公司的优惠和活动。

# 准备就绪

我们需要为 Telegram 创建一个新的机器人。这是通过一个名为**BotFather**的界面完成的，它是一个特殊的 Telegram 频道，允许我们创建一个新的机器人。您可以通过此链接访问该频道：[`telegram.me/botfather`](https://telegram.me/botfather)。通过您的 Telegram 帐户访问它。

运行`/start`以启动界面，然后使用`/newbot`创建一个新的机器人。界面会要求您输入机器人的名称和用户名，用户名应该是唯一的。

一旦设置好，它将给您以下内容：

+   您的机器人的 Telegram 频道-`https:/t.me/<yourusername>`。

+   允许访问机器人的令牌。复制它，因为稍后会用到。

如果丢失令牌，可以生成一个新的令牌。阅读 BotFather 的文档。

我们还需要安装 Python 模块`telepot`，它包装了 Telegram 的 RESTful 接口：

```py
$ echo "telepot==12.7" >> requirements.txt
$ pip install -r requirements.txt
```

从 GitHub 上下载`telegram_bot.py`脚本：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/telegram_bot.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/telegram_bot.py)。

# 如何做...

1.  将生成的令牌设置到`telegram_bot.py`脚本的第 6 行的`TOKEN`常量中：

```py
TOKEN = '<YOUR TOKEN>'
```

1.  启动机器人：

```py
$ python telegram_bot.py
```

1.  使用 URL 在手机上打开 Telegram 频道并启动它。您可以使用`help`，`offers`和`events`命令：

![](img/e8a122ba-dadf-4e78-9c32-d29e8d56449c.png)

# 工作原理...

第 1 步设置要用于您特定频道的令牌。在第 2 步中，我们在本地启动机器人。

让我们看看`telegram_bot.py`中的代码结构：

```py
IMPORTS

TOKEN

# Define the information to return per command
def get_help():
def get_offers():
def get_events():
COMMANDS = {
    'help': get_help,
    'offers': get_offers,
    'events': get_events,
}

class MarketingBot(telepot.helper.ChatHandler):
...

# Create and start the bot

```

`MarketingBot`类创建了一个与 Telegram 进行通信的接口：

+   当频道启动时，将调用`open`方法

+   当收到消息时，将调用`on_chat_message`方法

+   如果有一段时间没有回应，将调用`on_idle`

在每种情况下，`self.sender.sendMessage`方法用于向用户发送消息。大部分有趣的部分都发生在`on_chat_message`中：

```py
def on_chat_message(self, msg):
    # If the data sent is not test, return an error
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text':
        self.sender.sendMessage("I don't understand you. "
                                "Please type 'help' for options")
        return

    # Make the commands case insensitive
    command = msg['text'].lower()
```

```py

    if command not in COMMANDS:
        self.sender.sendMessage("I don't understand you. "
                                "Please type 'help' for options")
        return

    message = COMMANDS[command]()
    self.sender.sendMessage(message)
```

首先，它检查接收到的消息是否为文本，如果不是，则返回错误消息。它分析接收到的文本，如果是定义的命令之一，则执行相应的函数以检索要返回的文本。

然后，将消息发送回用户。

第 3 步显示了从与机器人交互的用户的角度来看这是如何工作的。

# 还有更多...

您可以使用`BotFather`接口向您的 Telegram 频道添加更多信息，头像图片等。

为了简化我们的界面，我们可以创建一个自定义键盘来简化机器人。在定义命令之后创建它，在脚本的第 44 行左右：

```py
from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton
keys = [[KeyboardButton(text=text)] for text in COMMANDS]
KEYBOARD = ReplyKeyboardMarkup(keyboard=keys)
```

请注意，它正在创建一个带有三行的键盘，每行都有一个命令。然后，在每个`sendMessage`调用中添加生成的`KEYBOARD`作为`reply_markup`，例如如下所示：

```py
 message = COMMANDS[command]()
 self.sender.sendMessage(message, reply_markup=KEYBOARD)
```

这将键盘替换为仅有定义的按钮，使界面非常明显：

![](img/c7d4fc8c-6861-4cb5-b4e2-72ac8dd1108c.png)

这些更改可以在 GitHub 的`telegram_bot_custom_keyboard.py`文件中下载，链接在这里：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/telegram_bot_custom_keyboard.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter08/telegram_bot_custom_keyboard.py)。

您可以创建其他类型的自定义界面，例如内联按钮，甚至是创建游戏的平台。查看 Telegram API 文档以获取更多信息。

与 Telegram 的交互也可以通过 webhook 完成，方式与*接收短信*配方中介绍的类似。在`telepot`文档中查看 Flask 的示例：[`github.com/nickoala/telepot/tree/master/examples/webhook`](https://github.com/nickoala/telepot/tree/master/examples/webhook)。

通过`telepot`可以设置 Telegram webhook。这要求您的服务位于 HTTPS 地址后，以确保通信是私密的。这可能对于简单的服务来说有点棘手。您可以在 Telegram 文档中查看有关设置 webhook 的文档：[`core.telegram.org/bots/api#setwebhook`](https://core.telegram.org/bots/api#setwebhook)。

电报机器人的完整 API 可以在这里找到：[`core.telegram.org/bots`](https://core.telegram.org/bots)。

`telepot`模块的文档可以在这里找到：[`telepot.readthedocs.io/en/latest/`](https://telepot.readthedocs.io/en/latest/)。

# 另请参阅

+   *生成短信*配方

+   *接收短信*配方
