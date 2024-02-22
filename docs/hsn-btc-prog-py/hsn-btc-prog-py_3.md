# 第三章：以编程方式赚取比特币

在本章中，我们将学习如何在我们的网站上开始接受比特币作为支付方式。我们还将学习如何构建基于 API 的微服务以赚取比特币，并探索比特币交易机器人。

# 在您的网站上接受比特币

在本节中，我们将学习以下主题：

+   如何在我们的网站上启用比特币支付

+   介绍 BitPay，第三方比特币 API 服务

+   如何生成比特币支付按钮

+   如何在我们的网站上添加比特币支付按钮

有很多第三方 API 可用于网站上快速启用比特币支付，其中最流行的之一是 BitPay。

# 介绍 BitPay

BitPay 可用于以多种不同的方式接受付款，包括以下方式：

+   您可以使用比特币在电子商务网站上接受在线付款

+   将比特币与许多不同的电子商务解决方案集成

+   与购物车集成

+   您可以显示启用比特币的支付按钮，这对于在博客或播客上接受捐赠非常有效

# 如何生成比特币支付按钮

按照以下步骤生成比特币支付按钮：

1.  首先，注册并登录 BitPay，网址为[`bitpay.com/`](https://bitpay.com/)。

1.  接下来，转到“支付工具|支付按钮”页面并创建支付按钮：

![](img/93f292a5-40e9-4e2c-ae66-6b23f2881b74.png)

1.  新字段需要安全服务器 POST 的 SERVER IPN，并且用户点击时会支付金额。您将在页面底部看到按钮的预览：

![](img/4759a749-5b1d-4089-9b53-a1530f49c767.png)

1.  要将此按钮添加到网站，只需复制 HTML 代码并粘贴到所需的网页中。

# 如何在您的网站上添加比特币支付按钮

按照以下步骤将支付按钮添加到您的网站页面：

1.  在代码编辑器中打开您网站页面的源代码。

1.  粘贴我们在上一节从 BitPay 网站复制的 HTML 代码，保存文件并重新加载网页。

1.  以下截图显示了网页上的支付按钮，用户可以使用它发送付款：

![](img/dcbe546f-cdb5-464e-b2f1-28e9e4a0f1be.png)

# 构建和发布启用比特币的 API

在本节中，我们将学习以下主题：

+   介绍 21.co 市场

+   开始使用 21.co SDK

+   开始为比特币出售服务

# 21.co 市场

21.co 是一个平台，托管了一个虚拟市场，开发人员可以在其中创建并出售微服务以换取比特币。有关更多信息，请参阅[`earn.com/`](https://earn.com/)。

我们将演示如何加入这个市场并出售微服务以赚取比特币。

为此，我们将使用 21.co 的 SDK。

# 21.co SDK

21.co SDK 目前支持 Ubuntu 和 macOS。本节将演示在 AWS 上运行 Ubuntu 的情况。

您可以按照[`aws.amazon.com/premiumsupport/knowledge-center/create-linux-instance/`](https://aws.amazon.com/premiumsupport/knowledge-center/create-linux-instance/)上的说明在 AWS 上创建 AWS Ubuntu 14.x 实例。

创建 AWS 实例后，请按照 AWS 文档页面上的说明连接到它，网址为[`docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html`](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html)。

连接到 AWS 实例后，安装 21.co SDK。您可以通过执行以下命令来完成：

```py
curl https: //21.co | sh
```

安装 SDK 后，通过执行以下命令登录到您的 21.co 帐户：

```py
 21 login
```

如果用户没有 21.co 登录，则必须在 21.co 网站上创建帐户。登录后，首先加入 21.co 节点到 21.co 虚拟市场。您可以通过执行以下命令来完成：

```py
21 market join
```

通过执行以下命令可以实现用户加入请求的状态：

```py
21 market status
```

接下来，通过执行以下命令测试安装：

```py
21 doctor
```

上述命令将显示所有测试都已通过，并且节点已设置并加入了 21.co 市场。

为了获取比特币余额，执行以下命令：

```py
21 status 
```

上述代码显示了在 21.co 账户中持有的比特币余额。

# 在 21.co 市场上出售微服务

21.co 的 SDK 捆绑了一些服务。要启动所有这些服务，请执行以下命令：

```py
21 sell start --all
```

可能会提示安装依赖项。如果是这样，您应该继续并执行。

有时，用户可能需要注销并重新登录以进行更改。

在 21.co 市场上出售微服务，请执行以下步骤：

1.  执行以下命令：

```py
21 sell start --all
```

它将显示所有可在虚拟市场上开始销售的微服务的列表。

1.  要查看服务的状态，请执行以下命令：

```py
21 sell status
```

1.  完成服务后，或者如果要停止它，请运行以下命令：

```py
21 sell stop -all
```

1.  要查看节点上发生的所有活动，请使用以下命令：

```py
21 log
```

这是如何通过在 21.co 市场上出售和列出微服务来赚取比特币的演示。

# 构建比特币交易机器人

在本节中，我们将学习以下主题：

+   如何获取比特币的当前出价和要价

+   如何决定是否买入或卖出比特币

+   触发比特币交易建议警报

实际的比特币买卖不会被涵盖，因为涉及实际货币。但是，我们将专注于根据我们设置的条件尝试买入或卖出比特币时发送电子邮件警报。

我们将使用比特币价格 API 模块来获取比特币价格。它可以在 GitHub 上找到[`github.com/dursk/bitcoin-price-api`](https://github.com/dursk/bitcoin-price-api)。

# 触发比特币交易建议警报

为了设置比特币交易建议警报，请按照以下步骤进行：

1.  首先，通过导入名为`exchanges`的比特币价格 API 开始：

```py
#!/usr/bin/python

# import modules
# Make sure to copy the exchanges from https://github.com/dursk/bitcoin-price-api
# to the same location as this script
from exchanges.bitfinex import Bitfinex
```

1.  还要导入`smtplib`，我们将用它来触发比特币价格警报。在这里，我们定义了一个名为`trigger_email`的函数。然后设置服务器用户和电子邮件详细信息：

```py
import smtplib

# Function to send email
def trigger_email(msg):
 # Change these to your email details
 email_user = "bitcoin.harish@gmail.com"
 email_password = "bitcoin1"
 smtp_server = 'smtp.gmail.com'
 smtp_port = 587
 email_from = "bitcoin.harish@gmail.com"
 email_to = "bitcoin.harish@gmail.com"
```

1.  使用`smtplib`，发送`sendmail`函数发送价格警报电子邮件，如下面的代码所示：

```py
# login to the email server
 server = smtplib.SMTP(smtp_server, smtp_port)
 server.starttls()
 server.login(email_user, email_password)

 # send email
 server.sendmail(email_from, email_to, msg)
 server.quit()
```

1.  接下来，为比特币定义买入和卖出价格阈值。使用这些阈值来决定是否卖出或买入比特币：

```py
# define buy and sell thresholds for Bitcoin. These values you have to change according to the current price of the bitcoin.
buy_thresh = 6500
sell_thresh = 6500
```

1.  接下来，我们从 Bitfinex 比特币交易所使用我们在`bitcoin_trade.py`脚本中导入的`exchanges`模块获取当前比特币价格和当前出价。我们也可以使用其他交易所，如 CoinDesk，但目前我们将使用 Bitfinex。我们将在`btc_sell_price`和`btc_buy_price`中获取这些价格。

```py
# get Bitcoin prices
btc_sell_price = Bitfinex().get_current_bid()
btc_buy_price = Bitfinex().get_current_ask()
```

1.  一旦我们得到了当前的价格，我们可以将它们与之前设置的阈值价格进行比较。

1.  如果买入价格低于卖出阈值，我们调用`trigger_email`函数发送买入触发电子邮件警报：

```py
# Trigger Buy email if buy price is lower than threshold
if btc_buy_price < buy_thresh:
email_msg = """
 Bitcoin Buy Price is %s which is lower than
 threshold price of %s.
 Good time to buy!""" % (btc_buy_price, buy_thresh)

trigger_email(email_msg)
```

1.  如果卖出价格高于卖出阈值，我们调用`trigger_email`函数发送卖出触发电子邮件警报：

```py
# Trigger sell email if sell price is higher than threshold
if btc_sell_price > sell_thresh:

  email_msg = """
 Bitcoin sell Price is %s which is higher than
 threshold price of %s.
 Good time to sell!""" % (btc_sell_price, sell_thresh)

trigger_email(email_msg)
```

# 如何获取比特币的当前出价和要价

谷歌搜索是搜索当前出价的最简单方法。为了实现比特币的买卖，应相应地触发两者。

# 买入比特币的触发

以下是获取当前出价的步骤：

1.  首先，检查比特币价格在线。

1.  修改脚本，使买入警报首先触发。将买入阈值设置为高于当前价格。在这里，我们将买入阈值设置为`6500`，如下面的代码所示：

```py
# define buy and sell thresholds for Bitcoin
buy_thresh = 6500
sell_thresh = 6500
```

1.  保存脚本并执行它。以下屏幕截图显示了执行的脚本：

![](img/f2b014be-60ca-42e1-9d35-afb6b8934f25.png)

脚本已执行，买入警报应该已经发出。在电子邮件中检查。

以下屏幕截图显示，根据我们在脚本中设置的标准，我们已收到比特币警报电子邮件，建议我们购买比特币：

![](img/f282db35-a16e-418a-9130-b5617860f644.png)

# 出售比特币的触发

1.  最初，我们应该将出售门槛设置为低于当前价格。例如，让我们将`6400`作为门槛，并再次执行脚本。以下代码显示了`sell_thresh`设置为`6400`：

```py
# define buy and sell thresholds for bitcoin
buy_thresh = 6400
sell_thresh = 6400
```

现在，出售警报应该执行。再次在电子邮件中验证。

1.  验证后，我们应该看到我们已收到电子邮件警报，建议我们出售比特币，因为当前的要价高于我们愿意出售的价格：

![](img/7411d4ff-e5e5-4329-8a85-ad1028db7e5d.png)

1.  脚本已准备好。您现在可以将其设置为在各种操作系统上自动运行。在 Windows 上，请使用任务计划程序。

1.  从“操作”菜单中，选择“创建任务...”，并将其命名为“比特币交易警报”，如下面的屏幕截图所示：

![](img/16930c9b-5dc9-4cf0-98f6-b24e9ef7d6a6.png)

1.  从“触发器”选项卡中，单击“新建...”，如下面的屏幕截图所示：

![](img/db4a9f31-fc7c-4b26-9200-e69ce721a1b3.png)

1.  选择“每日”单选按钮。

1.  然后，在高级设置中，选择在所需的分钟数或小时数后重复任务。在这里，我们将其设置为每 1 小时，如下面的屏幕截图所示：

![](img/25ab6125-89c7-4b6c-9179-7726eae677db.png)

1.  接下来，从“操作”选项卡中，单击“新建...”按钮。

1.  通过单击“浏览...”按钮选择要在任务运行时执行的脚本。现在，此任务将每小时自动运行，并将检查比特币价格，并发送电子邮件建议我们是否买入或卖出比特币。

![](img/7f832c54-9ef9-424e-9a59-9839c8943f5d.png)

您还可以选择直接从脚本中触发交易，使用任何比特币交易所 API，例如 coinbase.com。由于涉及实际资金，用户需要小心处理。

# 总结

在本章中，我们探讨了如何在网站上启用比特币支付，向您介绍了 BitPay，学习了如何生成比特币支付按钮，以及如何将支付按钮添加到我们的网站上。我们还介绍了 21.co 市场和比特币的买卖服务，以及编写了一个简单的比特币交易机器人。我们学会了如何获取比特币的当前竞价和要价。我们还学会了如何决定是买入还是卖出比特币，以及如何发送电子邮件警报，建议我们是否执行该线程。

在下一章中，我们将学习如何对比特币数据进行数据分析。
