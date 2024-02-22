# 第九章：为什么不自动化您的营销活动呢？

在本章中，我们将介绍与营销活动相关的以下配方：

+   检测机会

+   创建个性化优惠券代码

+   通过用户的首选渠道向客户发送通知

+   准备销售信息

+   生成销售报告

# 介绍

在本章中，我们将创建一个完整的营销活动，逐步进行每个自动步骤。我们将在一个项目中利用本书中的所有概念和配方，这将需要不同的步骤。

让我们举个例子。对于我们的项目，我们的公司希望设置一个营销活动来提高参与度和销售额。这是一个非常值得赞扬的努力。为此，我们可以将行动分为几个步骤：

1.  我们希望检测启动活动的最佳时机，因此我们将从不同来源收到关键词的通知，这将帮助我们做出明智的决定

1.  该活动将包括生成个人代码以发送给潜在客户

1.  这些代码的部分将直接通过用户的首选渠道发送给他们，即短信或电子邮件

1.  为了监控活动的结果，将编制销售信息并生成销售报告

本章将逐步介绍这些步骤，并提出基于本书介绍的模块和技术的综合解决方案。

尽管这些示例是根据现实生活中的需求创建的，但请注意，您的特定环境总会让您感到意外。不要害怕尝试、调整和改进您的系统，随着对系统的了解越来越多，迭代是创建出色系统的方法。

让我们开始吧！

# 检测机会

在这个配方中，我们提出了一个分为几个步骤的营销活动：

1.  检测启动活动的最佳时机

1.  生成个人代码以发送给潜在客户

1.  通过用户的首选渠道直接发送代码，即短信或电子邮件

1.  整理活动的结果，并生成带有结果分析的销售报告

这个配方展示了活动的第一步。

我们的第一阶段是检测启动活动的最佳时间。为此，我们将监视一系列新闻网站，搜索包含我们定义关键词之一的新闻。任何与这些关键词匹配的文章都将被添加到一份报告中，并通过电子邮件发送。

# 做好准备

在这个配方中，我们将使用本书中之前介绍的几个外部模块，`delorean`、`requests`和`BeautifulSoup`。如果尚未添加到我们的虚拟环境中，我们需要将它们添加进去：

```py
$ echo "delorean==1.0.0" >> requirements.txt
$ echo "requests==2.18.3" >> requirements.txt
$ echo "beautifulsoup4==4.6.0" >> requirements.txt
$ echo "feedparser==5.2.1" >> requirements.txt
$ echo "jinja2==2.10" >> requirements.txt
$ echo "mistune==0.8.3" >> requirements.txt
$ pip install -r requirements.txt
```

您需要列出一些 RSS 源，我们将从中获取数据。

在我们的示例中，我们使用以下源，这些源都是知名新闻网站上的技术源：

[`feeds.reuters.com/reuters/technologyNews`](http://feeds.reuters.com/reuters/technologyNews)

[`rss.nytimes.com/services/xml/rss/nyt/Technology.xml`](http://rss.nytimes.com/services/xml/rss/nyt/Technology.xml)

[`feeds.bbci.co.uk/news/science_and_environment/rss.xml`](http://feeds.bbci.co.uk/news/science_and_environment/rss.xml)

下载`search_keywords.py`脚本，该脚本将从 GitHub 执行操作，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/search_keywords.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/search_keywords.py)。

您还需要下载电子邮件模板，可以在[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/email_styling.html`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/email_styling.html)和[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/email_template.md`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/email_template.md)找到。

在[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/config-opportunity.ini`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/config-opportunity.ini)中有一个配置模板。

你需要一个有效的用户名和密码来使用电子邮件服务。在第八章的*发送单独的电子邮件*示例中检查。

# 如何做...

1.  创建一个`config-opportunity.ini`文件，格式如下。记得填写你的详细信息：

```py
[SEARCH]
keywords = keyword, keyword
feeds = feed, feed

[EMAIL]
user = <YOUR EMAIL USERNAME>
password = <YOUR EMAIL PASSWORD>
from = <EMAIL ADDRESS FROM>
to = <EMAIL ADDRESS TO>
```

你可以使用 GitHub 上的模板[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/config-opportunity.ini`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/config-opportunity.ini)来搜索关键词`cpu`和一些测试源。记得用你自己的账户信息填写`EMAIL`字段。

1.  调用脚本生成电子邮件和报告：

```py
$ python search_keywords.py config-opportunity.ini
```

1.  检查`to`电子邮件，你应该收到一份包含找到的文章的报告。它应该类似于这样：

![](img/83e66ce4-be5b-46e2-96f3-5a4c5390865c.png)

# 工作原理...

在步骤 1 中创建脚本的适当配置后，通过调用`search_keywords.py`在步骤 2 中完成网页抓取和发送电子邮件的结果。

让我们看一下`search_keywords.py`脚本。代码分为以下几部分：

+   `IMPORTS`部分使所有 Python 模块可供以后使用。它还定义了`EmailConfig namedtuple`来帮助处理电子邮件参数。

+   `READ TEMPLATES`检索电子邮件模板并将它们存储以供以后在`EMAIL_TEMPLATE`和`EMAIL_STYLING`常量中使用。

+   `__main__`块通过获取配置参数、解析配置文件，然后调用主函数来启动过程。

+   `main`函数组合了其他函数。首先，它检索文章，然后获取正文并发送电子邮件。

+   `get_articles`遍历所有的源，丢弃任何超过一周的文章，检索每一篇文章，并搜索关键词的匹配。返回所有匹配的文章，包括链接和摘要的信息。

+   `compose_email_body`使用电子邮件模板编写电子邮件正文。注意模板是 Markdown 格式，它被解析为 HTML，以便在纯文本和 HTML 中提供相同的信息。

+   `send_email`获取正文信息，以及用户名/密码等必要信息，最后发送电子邮件。

# 还有更多...

从不同来源检索信息的主要挑战之一是在所有情况下解析文本。一些源可能以不同的格式返回信息。

例如，在我们的示例中，你可以看到路透社的摘要包含 HTML 信息，这些信息在最终的电子邮件中被渲染。如果你遇到这种问题，你可能需要进一步处理返回的数据，直到它变得一致。这可能高度依赖于预期的报告质量。

在开发自动任务时，特别是处理多个输入源时，预计会花费大量时间以一致的方式清理输入。但另一方面，要找到平衡，并牢记最终的接收者。例如，如果邮件是要由你自己或一个理解的队友接收，你可以比对待重要客户的情况更宽容一些。

另一种可能性是增加匹配的复杂性。在这个示例中，检查是用简单的`in`完成的，但请记住，第一章中的所有技术，包括所有正则表达式功能，都可以供您使用。

此脚本可以通过定时作业自动化，如《第二章》中所述，《自动化任务变得容易》。尝试每周运行一次！

# 另请参阅

+   在《第一章》的“添加命令行参数”中，《让我们开始我们的自动化之旅》

+   在《第一章》的“介绍正则表达式”中，《让我们开始我们的自动化之旅》

+   在《第二章》的“准备任务”中，《自动化任务变得容易》

+   在《第二章》的“设置定时作业”中，《自动化任务变得容易》

+   在《第三章》的“解析 HTML”中，《第一个网络爬虫应用程序》

+   在《第三章》的“爬取网络”中，《第一个网络爬虫应用程序》

+   在《第三章》的“构建您的第一个网络爬虫应用程序”中，订阅提要的食谱

+   在《第八章》的“发送个人电子邮件”中，《处理通信渠道》

# 创建个性化优惠券代码

在本章中，我们将一个营销活动分为几个步骤：

1.  检测最佳时机启动活动

1.  生成要发送给潜在客户的个人代码

1.  通过用户首选的渠道，即短信或电子邮件，直接发送代码给用户

1.  收集活动的结果

1.  生成带有结果分析的销售报告

这个食谱展示了活动的第 2 步。

在发现机会后，我们决定为所有客户生成一项活动。为了直接促销并避免重复，我们将生成 100 万个独特的优惠券，分为三批：

+   一半的代码将被打印并在营销活动中分发

+   30 万代码将被保留，以备将来在活动达到一些目标时使用

+   其余的 20 万将通过短信和电子邮件直接发送给客户，我们稍后会看到

这些优惠券可以在在线系统中兑换。我们的任务是生成符合以下要求的正确代码：

+   代码需要是唯一的

+   代码需要可打印且易于阅读，因为一些客户将通过电话口述它们

+   在检查代码之前应该有一种快速丢弃代码的方法（避免垃圾邮件攻击）

+   代码应以 CSV 格式呈现以供打印

# 做好准备

从 GitHub 上下载`create_personalised_coupons.py`脚本，该脚本将在 CSV 文件中生成优惠券，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/create_personalised_coupons.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/create_personalised_coupons.py)。

# 如何做...

1.  调用`create_personalised_coupons.py`脚本。根据您的计算机速度，运行时间可能需要一两分钟。它将在屏幕上显示生成的代码：

```py
$ python create_personalised_coupons.py
Code: HWLF-P9J9E-U3
Code: EAUE-FRCWR-WM
Code: PMW7-P39MP-KT
...
```

1.  检查它是否创建了三个 CSV 文件，其中包含代码`codes_batch_1.csv`，`codes_batch_2.csv`和`codes_batch_3.csv`，每个文件都包含正确数量的代码：

```py
$ wc -l codes_batch_*.csv
  500000 codes_batch_1.csv
  300000 codes_batch_2.csv
  200000 codes_batch_3.csv
 1000000 total
```

1.  检查每个批次文件是否包含唯一代码。您的代码将是唯一的，并且与此处显示的代码不同：

```py
$ head codes_batch_2.csv
9J9F-M33YH-YR
7WLP-LTJUP-PV
WHFU-THW7R-T9
...
```

# 它是如何工作的...

步骤 1 调用生成所有代码的脚本，步骤 2 检查结果是否正确。步骤 3 显示代码存储的格式。让我们分析`create_personalised_coupons.py`脚本。

总之，它具有以下结构：

```py
# IMPORTS

# FUNCTIONS
def random_code(digits)
def checksum(code1, code2)
def check_code(code)
def generate_code()

# SET UP TASK

# GENERATE CODES

# CREATE AND SAVE BATCHES
```

不同的功能一起工作来创建代码。`random_code`生成一组随机字母和数字的组合，取自`CHARACTERS`。该字符串包含所有可供选择的有效字符。

字符的选择被定义为易于打印且不易混淆的符号。例如，很难区分字母 O 和数字 0，或数字 1 和字母 I，这取决于字体。这可能取决于具体情况，因此如有必要，请进行打印测试以定制字符。但是避免使用所有字母和数字，因为这可能会引起混淆。如有必要，增加代码的长度。

`checksum`函数基于两个代码生成一个额外的数字，这个过程称为**哈希**，在计算中是一个众所周知的过程，尤其是在密码学中。

哈希的基本功能是从一个输入产生一个较小且不可逆的输出，这意味着很难猜测，除非已知输入。哈希在计算中有很多常见的应用，通常在底层使用。例如，Python 字典广泛使用哈希。

在我们的示例中，我们将使用 SHA256，这是一个众所周知的快速哈希算法，包含在 Python 的`hashlib`模块中：

```py
def checksum(code1, code2):
    m = hashlib.sha256()
    m.update(code1.encode())
    m.update(code2.encode())
    checksum = int(m.hexdigest()[:2], base=16)
    digit = CHARACTERS[checksum % len(CHARACTERS)]
    return digit
```

两个代码作为输入添加，然后将哈希的两个十六进制数字应用于`CHARACTERS`，以获得其中一个可用字符。这些数字被转换为数字（因为它们是十六进制的），然后我们应用`模`运算符来确保获得其中一个可用字符。

这个校验和的目的是能够快速检查代码是否正确，并且丢弃可能的垃圾邮件。我们可以再次对代码执行操作，以查看校验和是否相同。请注意，这不是加密哈希，因为在操作的任何时候都不需要秘密。鉴于这个特定的用例，这种（低）安全级别对我们的目的来说可能是可以接受的。

密码学是一个更大的主题，确保安全性强可能会很困难。密码学中涉及哈希的主要策略可能是仅存储哈希以避免以可读格式存储密码。您可以在这里阅读有关此的快速介绍：[`crackstation.net/hashing-security.htm`](https://crackstation.net/hashing-security.htm)。

`generate_code`函数然后生成一个随机代码，由四位数字、五位数字和两位校验和组成，用破折号分隔。第一个数字使用前九个数字按顺序生成（四位然后五位），第二个数字将其反转（五位然后四位）。

`check_code`函数将反转过程，并在代码正确时返回`True`，否则返回`False`。

有了基本元素之后，脚本开始定义所需的批次——500,000、300,000 和 200,000。

所有的代码都是在同一个池中生成的，称为`codes`。这是为了避免在池之间产生重复。请注意，由于过程的随机性，我们无法排除生成重复代码的可能性，尽管这很小。我们允许最多重试三次，以避免生成重复代码。代码被添加到一个集合累加器中，以确保它们的唯一性，并加快检查代码是否已经存在的速度。

`sets`是 Python 在底层使用哈希的另一个地方，因此它将要添加的元素进行哈希处理，并将其与已经存在的元素的哈希进行比较。这使得在集合中进行检查非常快速。

为了确保过程是正确的，每个代码都经过验证并打印出来，以显示生成代码的进度，并允许检查一切是否按预期工作。

最后，代码被分成适当数量的批次，每个批次保存在单独的`.csv`文件中。 代码使用`.pop()`从`codes`中逐个删除，直到`batch`达到适当大小为止：

```py
batch = [(codes.pop(),) for _ in range(batch_size)]
```

请注意，前一行创建了一个包含单个元素的适当大小行的批次。每一行仍然是一个列表，因为对于 CSV 文件来说应该是这样。

然后，创建一个文件，并使用`csv.writer`将代码存储为行。

作为最后的测试，验证剩余的`codes`是否为空。

# 还有更多...

在这个食谱中，流程采用了直接的方法。这与第二章中*准备运行任务*食谱中介绍的原则相反，*简化任务变得更容易*。请注意，与那里介绍的任务相比，此脚本旨在运行一次以生成代码，然后结束。它还使用了定义的常量，例如`BATCHES`，用于配置。

鉴于这是一个独特的任务，设计为仅运行一次，花时间将其构建成可重用的组件可能不是我们时间的最佳利用方式。

过度设计肯定是可能的，而在实用设计和更具未来导向性的方法之间做出选择可能并不容易。要对维护成本保持现实，并努力找到自己的平衡。

同样，这个食谱中的校验和设计旨在提供一种最小的方式来检查代码是否完全虚构或看起来合法。鉴于代码将被检查系统，这似乎是一个明智的方法，但要注意您特定的用例。

我们的代码空间是`22 个字符** 9 个数字= 1,207,269,217,792 个可能的代码`，这意味着猜测其中一个百万个生成的代码的概率非常小。也不太可能产生相同的代码两次，但尽管如此，我们通过最多三次重试来保护我们的代码。

这些检查以及检查每个代码的验证以及最终没有剩余代码的检查在开发这种脚本时非常有用。这确保了我们朝着正确的方向前进，事情按计划进行。只是要注意，在某些情况下`asserts`可能不会被执行。

如 Python 文档所述，如果使用`-O`命令运行 Python 代码，则`assert`命令将被忽略。请参阅此处的文档[`docs.python.org/3/reference/simple_stmts.html#the-assert-statement`](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement)。通常情况下不会这样做，但如果是这种情况可能会令人困惑。避免过度依赖`asserts`。

学习加密的基础并不像你可能认为的那么困难。有一些基本模式是众所周知且易于学习的。一个很好的介绍文章是这篇[`thebestvpn.com/cryptography/`](https://thebestvpn.com/cryptography/)。Python 也集成了大量的加密函数；请参阅文档[`docs.python.org/3/library/crypto.html`](https://docs.python.org/3/library/crypto.html)。最好的方法是找一本好书，知道虽然这是一个难以真正掌握的主题，但绝对是可以掌握的。

# 另请参阅

+   第一章中的*介绍正则表达式*食谱，*让我们开始自动化之旅*

+   第四章中的*读取 CSV 文件*食谱，*搜索和阅读本地文件*

# 向客户发送他们首选渠道的通知

在本章中，我们介绍了一个分为几个步骤的营销活动：

1.  检测最佳推出活动的时机

1.  生成要发送给潜在客户的个别代码

1.  直接将代码发送给用户，通过他们首选的渠道，短信或电子邮件

1.  收集活动的结果

1.  生成带有结果分析的销售报告

这个食谱展示了活动的第 3 步。

一旦我们的代码为直接营销创建好，我们需要将它们分发给我们的客户。

对于这个食谱，从包含所有客户及其首选联系方式信息的 CSV 文件中，我们将使用先前生成的代码填充文件，然后通过适当的方法发送通知，其中包括促销代码。

# 做好准备

在这个示例中，我们将使用已经介绍过的几个模块——`delorean`、`requests`和`twilio`。如果尚未添加到我们的虚拟环境中，我们需要将它们添加进去：

```py
$ echo "delorean==1.0.0" >> requirements.txt
$ echo "requests==2.18.3" >> requirements.txt
$ echo "twilio==6.16.3" >> requirements.txt
$ pip install -r requirements.txt
```

我们需要定义一个`config-channel.ini`文件，其中包含我们用于 Mailgun 和 Twilio 的服务的凭据。可以在 GitHub 上找到此文件的模板：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/config-channel.ini`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/config-channel.ini)。

有关如何获取凭据的信息，请参阅*通过电子邮件发送通知*和*生成短信*的示例第八章，*处理通信渠道*

文件的格式如下：

```py
[MAILGUN]
KEY = <YOUR KEY>
DOMAIN = <YOUR DOMAIN>
FROM = <YOUR FROM EMAIL>
[TWILIO]
ACCOUNT_SID = <YOUR SID>
AUTH_TOKEN = <YOUR TOKEN>
FROM = <FROM TWILIO PHONE NUMBER>
```

为了描述所有目标联系人，我们需要生成一个 CSV 文件`notifications.csv`，格式如下：

| Name | Contact Method | Target | Status | Code | Timestamp |
| --- | --- | --- | --- | --- | --- |
| John Smith | PHONE | +1-555-12345678 | `NOT-SENT` |  |  |
| Paul Smith | EMAIL | `paul.smith@test.com` | `NOT-SENT` |  |  |
| … |  |  |  |  |  |

请注意`Code`列为空，所有状态应为`NOT-SENT`或空。

如果您正在使用 Twilio 和 Mailgun 的测试帐户，请注意其限制。例如，Twilio 只允许您向经过身份验证的电话号码发送消息。您可以创建一个只包含两三个联系人的小型 CSV 文件来测试脚本。

应该准备好在 CSV 文件中使用的优惠券代码。您可以使用 GitHub 上的`create_personalised_coupons.py`脚本生成多个批次，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/create_personalised_coupons.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/create_personalised_coupons.py)。

从 GitHub 上下载要使用的脚本`send_notifications.py`，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/send_notifications.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/send_notifications.py)。

# 操作步骤...

1.  运行`send_notifications.py`以查看其选项和用法：

```py
$ python send_notifications.py --help
usage: send_notifications.py [-h] [-c CODES] [--config CONFIG_FILE] notif_file

positional arguments:
  notif_file notifications file

optional arguments:
  -h, --help show this help message and exit
  -c CODES, --codes CODES
                        Optional file with codes. If present, the file will be
                        populated with codes. No codes will be sent
  --config CONFIG_FILE config file (default config.ini)
```

1.  将代码添加到`notifications.csv`文件中：

```py
$ python send_notifications.py --config config-channel.ini notifications.csv -c codes_batch_3.csv 
$ head notifications.csv
Name,Contact Method,Target,Status,Code,Timestamp
John Smith,PHONE,+1-555-12345678,NOT-SENT,CFXK-U37JN-TM,
Paul Smith,EMAIL,paul.smith@test.com,NOT-SENT,HJGX-M97WE-9Y,
...
```

1.  最后，发送通知：

```py
$ python send_notifications.py --config config-channel.ini notifications.csv
$ head notifications.csv
Name,Contact Method,Target,Status,Code,Timestamp
John Smith,PHONE,+1-555-12345678,SENT,CFXK-U37JN-TM,2018-08-25T13:08:15.908986+00:00
Paul Smith,EMAIL,paul.smith@test.com,SENT,HJGX-M97WE-9Y,2018-08-25T13:08:16.980951+00:00
...
```

1.  检查电子邮件和电话，以验证消息是否已收到。

# 工作原理...

第 1 步展示了脚本的使用。总体思路是多次调用它，第一次用于填充代码，第二次用于发送消息。如果出现错误，可以再次执行脚本，只会重试之前未发送的消息。

`notifications.csv`文件获取将在第 2 步中注入的代码。这些代码最终将在第 3 步中发送。

让我们分析`send_notifications.py`的代码。这里只显示了最相关的部分：

```py
# IMPORTS

def send_phone_notification(...):
def send_email_notification(...):
def send_notification(...):

def save_file(...):
def main(...):

if __name__ == '__main__':
    # Parse arguments and prepare configuration
    ...
```

主要函数逐行遍历文件，并分析每种情况下要执行的操作。如果条目为`SENT`，则跳过。如果没有代码，则尝试填充。如果尝试发送，则会附加时间戳以记录发送或尝试发送的时间。

对于每个条目，整个文件都会被保存在名为`save_file`的文件中。注意文件光标定位在文件开头，然后写入文件，并刷新到磁盘。这样可以在每次条目操作时覆盖文件，而无需关闭和重新打开文件。

为什么要为每个条目写入整个文件？这是为了让您可以重试。如果其中一个条目产生意外错误或超时，甚至出现一般性故障，所有进度和先前的代码都将被标记为已发送，并且不会再次发送。这意味着可以根据需要重试操作。对于大量条目，这是确保在过程中出现问题不会导致我们重新发送消息给客户的好方法。

对于要发送的每个代码，`send_notification` 函数决定调用 `send_phone_notification` 或 `send_email_notification`。在两种情况下都附加当前时间。

如果无法发送消息，两个 `send` 函数都会返回错误。这允许您在生成的 `notifications.csv` 中标记它，并稍后重试。

`notifications.csv` 文件也可以手动更改。例如，假设电子邮件中有拼写错误，这就是错误的原因。可以更改并重试。

`send_email_notification` 根据 Mailgun 接口发送消息。有关更多信息，请参阅第八章中的*通过电子邮件发送通知*配方，*处理通信渠道*。请注意这里发送的电子邮件仅为文本。

`send_phone_notification` 根据 Twilio 接口发送消息。有关更多信息，请参阅第八章中的*生成短信*配方，*处理通信渠道*。

# 还有更多...

时间戳的格式故意以 ISO 格式编写，因为它是可解析的格式。这意味着我们可以轻松地以这种方式获取一个正确的对象，就像这样：

```py
>>> import datetime
>>> timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
>>> timestamp
'2018-08-25T14:13:53.772815+00:00'
>>> datetime.datetime.fromisoformat(timestamp)
datetime.datetime(2018, 9, 11, 21, 5, 41, 979567, tzinfo=datetime.timezone.utc)
```

这使您可以轻松地解析时间戳。

ISO 8601 时间格式在大多数编程语言中都得到很好的支持，并且非常精确地定义了时间，因为它包括时区。如果可以使用它，这是记录时间的绝佳选择。

`send_notification` 中用于路由通知的策略非常有趣：

```py
# Route each of the notifications
METHOD = {
    'PHONE': send_phone_notification,
    'EMAIL': send_email_notification,
}
try:
    method = METHOD[entry['Contact Method']]
    result = method(entry, config)
except KeyError:
    result = 'INVALID_METHOD'
```

`METHOD` 字典将每个可能的 `Contact Method` 分配给具有相同定义的函数，接受条目和配置。

然后，根据特定的方法，从字典中检索并调用函数。请注意 `method` 变量包含要调用的正确函数。

这类似于其他编程语言中可用的 `switch` 操作。也可以通过 `if...else` 块来实现。对于这种简单的代码，字典方法使代码非常易读。

`invalid_method` 函数被用作默认值。如果 `Contact Method` 不是可用的方法之一（`PHONE` 或 `EMAIL`），将引发 `KeyError`，捕获并将结果定义为 `INVALID METHOD`。

# 另请参阅

+   第八章中的*通过电子邮件发送通知*配方，*处理通信渠道*

+   第八章中的*生成短信*配方，*处理通信渠道*

# 准备销售信息

在本章中，我们介绍了一个分为几个步骤的营销活动：

1.  检测启动广告活动的最佳时机

1.  生成要发送给潜在客户的个人代码

1.  直接通过用户首选的渠道，短信或电子邮件，发送代码

1.  收集广告活动的结果

1.  生成带有结果分析的销售报告

这个配方展示了广告活动的第 4 步。

向用户发送信息后，我们需要收集商店的销售日志，以监控情况和广告活动的影响有多大。

销售日志作为与各个关联商店的单独文件报告，因此在这个配方中，我们将看到如何将所有信息汇总到一个电子表格中，以便将信息作为一个整体处理。

# 做好准备

对于这个配方，我们需要安装以下模块：

```py
$ echo "openpyxl==2.5.4" >> requirements.txt
$ echo "parse==1.8.2" >> requirements.txt
$ echo "delorean==1.0.0" >> requirements.txt
$ pip install -r requirements.txt
```

我们可以从 GitHub 上获取这个配方的测试结构和测试日志：[`github.com/PacktPublishing/Python-Automation-Cookbook/tree/master/Chapter09/sales`](https://github.com/PacktPublishing/Python-Automation-Cookbook/tree/master/Chapter09/sales)。请下载包含大量测试日志的完整`sales`目录。为了显示结构，我们将使用`tree`命令（[`mama.indstate.edu/users/ice/tree/`](http://mama.indstate.edu/users/ice/tree/)），它在 Linux 中默认安装，并且可以在 macOs 中使用`brew`安装（[`brew.sh/`](https://brew.sh/)）。您也可以使用图形工具来检查目录。

我们还需要`sale_log.py`模块和`parse_sales_log.py`脚本，可以在 GitHub 上找到：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/parse_sales_log.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/parse_sales_log.py)。

# 如何做...

1.  检查`sales`目录的结构。每个子目录代表一个商店提交了其销售日志的期间：

```py
$ tree sales
sales
├── 345
│   └── logs.txt
├── 438
│   ├── logs_1.txt
│   ├── logs_2.txt
│   ├── logs_3.txt
│   └── logs_4.txt
└── 656
 └── logs.txt
```

1.  检查日志文件：

```py
$ head sales/438/logs_1.txt
[2018-08-27 21:05:55+00:00] - SALE - PRODUCT: 12346 - PRICE: $02.99 - NAME: Single item - DISCOUNT: 0%
[2018-08-27 22:05:55+00:00] - SALE - PRODUCT: 12345 - PRICE: $07.99 - NAME: Family pack - DISCOUNT: 20%
...
```

1.  调用`parse_sales_log.py`脚本生成存储库：

```py
$ python parse_sales_log.py sales -o report.xlsx
```

1.  检查生成的 Excel 结果，`report.xlsx`：

![](img/f0a6152c-2b86-4262-9f0f-cf5283ba2602.png)

# 它是如何工作的...

步骤 1 和 2 展示了数据的结构。步骤 3 调用`parse_sales_log.py`来读取所有日志文件并解析它们，然后将它们存储在 Excel 电子表格中。电子表格的内容在步骤 4 中显示。

让我们看看`parse_sales_log.py`的结构：

```py
# IMPORTS
from sale_log import SaleLog

def get_logs_from_file(shop, log_filename):
    with open(log_filename) as logfile:
        logs = [SaleLog.parse(shop=shop, text_log=log)
                for log in logfile]
    return logs

def main(log_dir, output_filename):
    logs = []
    for dirpath, dirnames, filenames in os.walk(log_dir):
        for filename in filenames:
            # The shop is the last directory
            shop = os.path.basename(dirpath)
            fullpath = os.path.join(dirpath, filename)
            logs.extend(get_logs_from_file(shop, fullpath))

    # Create and save the Excel sheet
    xlsfile = openpyxl.Workbook()
    sheet = xlsfile['Sheet']
    sheet.append(SaleLog.row_header())
    for log in logs:
        sheet.append(log.row())
    xlsfile.save(output_filename)

if __name__ == '__main__':
  # PARSE COMMAND LINE ARGUMENTS AND CALL main()

```

命令行参数在第一章中有解释，*让我们开始自动化之旅*。请注意，导入包括`SaleLog`。

主要函数遍历整个目录并通过`os.walk`获取所有文件。您可以在第二章中获取有关`os.walk`的更多信息，*简化任务自动化*。然后将每个文件传递给`get_logs_from_file`来解析其日志并将它们添加到全局`logs`列表中。

注意，特定商店存储在最后一个子目录中，因此可以使用`os.path.basename`来提取它。

完成日志列表后，使用`openpyxl`模块创建一个新的 Excel 表。`SaleLog`模块有一个`.row_header`方法来添加第一行，然后所有日志都被转换为行格式使用`.row`。最后，文件被保存。

为了解析日志，我们创建一个名为`sale_log.py`的模块。这个模块抽象了解析和处理一行的过程。大部分都很简单，并且正确地结构化了每个不同的参数，但是解析方法需要一点注意：

```py
    @classmethod
    def parse(cls, shop, text_log):
        '''
        Parse from a text log with the format
        ...
        to a SaleLog object
        '''
        def price(string):
            return Decimal(string)

        def isodate(string):
            return delorean.parse(string)

        FORMAT = ('[{timestamp:isodate}] - SALE - PRODUCT: {product:d} '
                  '- PRICE: ${price:price} - NAME: {name:D} '
                  '- DISCOUNT: {discount:d}%')

        formats = {'price': price, 'isodate': isodate}
        result = parse.parse(FORMAT, text_log, formats)

        return cls(timestamp=result['timestamp'],
                   product_id=result['product'],
                   price=result['price'],
                   name=result['name'],
                   discount=result['discount'],
                   shop=shop)
```

`sale_log.py`是一个*classmethod*，意味着可以通过调用`SaleLog.parse`来使用它，并返回类的新元素。

Classmethods 被调用时，第一个参数存储类，而不是通常存储在`self`中的对象。约定是使用`cls`来表示它。在最后调用`cls(...)`等同于`SaleFormat(...)`，因此它调用`__init__`方法。

该方法使用`parse`模块从模板中检索值。请注意，`timestamp`和`price`这两个元素具有自定义解析。`delorean`模块帮助我们解析日期，价格最好描述为`Decimal`以保持适当的分辨率。自定义过滤器应用于`formats`参数。

# 还有更多...

`Decimal`类型在 Python 文档中有详细描述：[`docs.python.org/3/library/decimal.html`](https://docs.python.org/3/library/decimal.html)。

完整的`openpyxl`可以在这里找到：[`openpyxl.readthedocs.io/en/stable/`](https://openpyxl.readthedocs.io/en/stable/)。还要检查第六章，*电子表格的乐趣*，以获取有关如何使用该模块的更多示例。

完整的`parse`文档可以在这里找到：[`github.com/r1chardj0n3s/parse`](https://github.com/r1chardj0n3s/parse)。第一章中也更详细地描述了这个模块。

# 另请参阅

+   第一章中的*使用第三方工具—parse*配方，*让我们开始自动化之旅*

+   第四章中的*爬取和搜索目录*配方，*搜索和读取本地文件*

+   第四章中的*读取文本文件*配方，*搜索和读取本地文件*

+   第六章中的*更新 Excel 电子表格*配方，*电子表格的乐趣*

# 生成销售报告

在这一章中，我们提出了一个分为几个步骤的营销活动：

1.  检测最佳推出活动的时机

1.  生成个人代码以发送给潜在客户

1.  直接将代码通过用户首选的渠道，短信或电子邮件发送给用户

1.  收集活动的结果

1.  生成带有结果分析的销售报告

这个配方展示了活动的第 5 步。

作为最后一步，所有销售的信息都被汇总并显示在销售报告中。

在这个配方中，我们将看到如何利用从电子表格中读取、创建 PDF 和生成图表，以便自动生成全面的报告，以分析我们活动的表现。

# 准备工作

在这个配方中，我们将在虚拟环境中需要以下模块：

```py
$ echo "openpyxl==2.5.4" >> requirements.txt
$ echo "fpdf==1.7.2" >> requirements.txt
$ echo "delorean==1.0.0" >> requirements.txt
$ echo "PyPDF2==1.26.0" >> requirements.txt
$ echo "matplotlib==2.2.2" >> requirements.txt
$ pip install -r requirements.txt
```

我们需要在 GitHub 上的`sale_log.py`模块，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/sale_log.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/sale_log.py)。

输入电子表格是在前一个配方中生成的，准备销售信息。在那里查找更多信息。

您可以从 GitHub 上下载用于生成输入电子表格的脚本`parse_sales_log.py`，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/parse_sales_log.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/parse_sales_log.py)。

从 GitHub 上下载原始日志文件，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/tree/master/Chapter09/sales`](https://github.com/PacktPublishing/Python-Automation-Cookbook/tree/master/Chapter09/sales)。请下载完整的`sales`目录。

从 GitHub 上下载`generate_sales_report.py`脚本，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/generate_sales_report.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter09/generate_sales_report.py)。

# 如何做...

1.  检查输入文件和使用`generate_sales_report.py`：

```py
$ ls report.xlsx
report.xlsx
$ python generate_sales_report.py --help
usage: generate_sales_report.py [-h] input_file output_file

positional arguments:
  input_file
  output_file

optional arguments:
  -h, --help show this help message and exit
```

1.  使用输入文件和输出文件调用`generate_sales_report.py`脚本：

```py
$ python generate_sales_report.py report.xlsx output.pdf
```

1.  检查`output.pdf`输出文件。它将包含三页，第一页是简要摘要，第二页和第三页是按天和按商店的销售图表：

![](img/d1492287-b767-48b3-b522-fa5c294f4eb2.png)

第二页显示了每天的销售图表：

![](img/28b396ae-1c0d-4831-b27d-843ca484f302.png)

第三页按商店划分销售额：

![](img/b3cbce7b-0a65-40b0-adff-515c6d66bee8.png)

# 它是如何工作的

第 1 步显示如何使用脚本，第 2 步在输入文件上调用它。让我们来看一下`generate_sales_report.py`脚本的基本结构：

```py
# IMPORTS
def generate_summary(logs):

def aggregate_by_day(logs):
def aggregate_by_shop(logs):

def graph(...):

def create_summary_brief(...):

def main(input_file, output_file):
  # open and read input file
  # Generate each of the pages calling the other calls
  # Group all the pdfs into a single file
  # Write the resulting PDF

if __name__ == '__main__':
  # Compile the input and output files from the command line
  # call main
```

有两个关键元素——以不同方式（按商店和按天）聚合日志以及在每种情况下生成摘要。摘要是通过`generate_summary`生成的，它从日志列表中生成一个带有聚合信息的字典。日志的聚合是在`aggregate_by`函数中以不同的样式完成的。

`generate_summary`生成一个包含聚合信息的字典，包括开始和结束时间，所有日志的总收入，总单位，平均折扣，以及相同数据按产品进行的详细分解。

通过从末尾开始理解脚本会更好。主要函数将所有不同的操作组合在一起。读取每个日志并将其转换为本地的`SaleLog`对象。

然后，它将每个页面生成为一个中间的 PDF 文件：

+   `create_summary_brief`生成一个关于所有数据的总摘要。

+   日志被`aggregate_by_day`。创建一个摘要并生成一个图表。

+   日志被`aggregate_by_shop`。创建一个摘要并生成一个图表。

使用`PyPDF2`将所有中间 PDF 页面合并成一个文件。最后，删除中间页面。

`aggregate_by_day`和`aggregate_by_shop`都返回一个包含每个元素摘要的列表。在`aggregate_by_day`中，我们使用`.end_of_day`来检测一天何时结束，以区分一天和另一天。

`graph`函数执行以下操作：

1.  准备要显示的所有数据。这包括每个标签（日期或商店）的单位数量，以及每个标签的总收入。

1.  创建一个顶部图表，显示按产品分割的总收入，以堆叠条形图的形式。为了能够做到这一点，同时计算总收入时，还计算了基线（下一个堆叠位置的位置）。

1.  它将图表的底部部分分成与产品数量相同的图表，并显示每个标签（日期或商店）上销售的单位数量。

为了更好地显示，图表被定义为 A4 纸的大小。它还允许我们使用`skip_labels`在第二个图表的 X 轴上打印每个*X*标签中的一个，以避免重叠。这在显示日期时很有用，并且设置为每周只显示一个标签。

生成的图表被保存到文件中。

`create_summary_brief`使用`fpdf`模块保存一个包含总摘要信息的文本 PDF 页面。

`create_summary_brief`中的模板和信息被故意保持简单，以避免使这个配方复杂化，但可以通过更好的描述性文本和格式进行复杂化。有关如何使用`fpdf`的更多详细信息，请参阅第五章，“生成精彩报告”。

如前所示，`main`函数将所有 PDF 页面分组并合并成一个单一文档，然后删除中间页面。

# 还有更多...

此配方中包含的报告可以扩展。例如，可以在每个页面中计算平均折扣，并显示为一条线：

```py
# Generate a data series with the average discount
discount = [summary['average_discount'] for _, summary in full_summary]
....
# Print the legend
# Plot the discount in a second axis
plt.twinx()
plt.plot(pos, discount,'o-', color='green')
plt.ylabel('Average Discount')
```

但要小心，不要在一个图表中放入太多信息。这可能会降低可读性。在这种情况下，另一个图表可能是更好的显示方式。

在创建第二个轴之前小心打印图例，否则它将只显示第二个轴上的信息。

图表的大小和方向可以决定是否使用更多或更少的标签，以便清晰可读。这在使用`skip_labels`避免混乱时得到了证明。请注意生成的图形，并尝试通过更改大小或在某些情况下限制标签来适应该领域可能出现的问题。

例如，可能的限制是最多只能有三种产品，因为在我们的图表中打印第二行的四个图表可能会使文本难以辨认。请随意尝试并检查代码的限制。

完整的`matplotlib`文档可以在[`matplotlib.org/`](https://matplotlib.org/)找到。

`delorean`文档可以在这里找到：[`delorean.readthedocs.io/en/latest/`](https://delorean.readthedocs.io/en/latest/)

`openpyxl`的所有文档都可以在[`openpyxl.readthedocs.io/en/stable/`](https://openpyxl.readthedocs.io/en/stable/)找到。 

PyPDF2 的 PDF 操作模块的完整文档可以在[`pythonhosted.org/PyPDF2/`](https://pythonhosted.org/PyPDF2/)找到，`pyfdf`的文档可以在[`pyfpdf.readthedocs.io/en/latest/`](https://pyfpdf.readthedocs.io/en/latest/)找到。

本食谱利用了第五章中提供的不同概念和技术，用于 PDF 创建和操作，《第六章](404a9dc7-22f8-463c-9f95-b480dc17518d.xhtml)中的*与电子表格玩耍*，用于电子表格阅读，以及第七章中的*开发令人惊叹的图表*，用于图表创建。查看它们以了解更多信息。

# 另请参阅

+   在第五章中的*聚合 PDF*报告食谱

+   在第六章中的*读取 Excel*电子表格食谱

+   在第七章中的*绘制堆叠条形图*食谱

+   在《开发令人惊叹的图表》第七章中的*显示多行*食谱

+   在《开发令人惊叹的图表》第七章中的*添加图例和注释*食谱

+   在《开发令人惊叹的图表》第七章中的*组合图表*食谱

+   在《开发令人惊叹的图表》第七章中的*保存图表*食谱
