# 第二章：使用 Python 编程比特币和区块链

本章重点介绍使用树莓派比特币工具来使用 Python 编程比特币，并以编程方式与区块链 API 进行交互。读者还将对挖掘比特币及其初始阶段的过程有一个大致的了解。

在本章中，我们将学习以下主题：

+   使用 Python 编程比特币

+   创建多重签名比特币地址

+   使用 Python 进行区块链 API 编程

+   安装 Blockchain.info

+   Python 库

+   学习挖掘比特币

+   如何挖掘比特币

+   挖掘比特币的困难增加

# 使用 Python 编程比特币

在本节中，我们将介绍以下主题：

+   树莓派比特币工具库及如何开始使用它

+   如何生成私钥和公钥

+   如何从生成的私钥和公钥创建一个简单的比特币地址

要使用 Python 开始比特币，必须在系统中安装 Python 3.x 和名为 Pi 比特币工具的比特币 Python 库。

# Pi 比特币工具库

要安装 Pi 比特币工具库，请打开命令行程序并执行以下命令：

```py
pip install bitcoin
```

这个库最好的地方是，您不需要在计算机上安装比特币节点就可以开始使用它。

此库连接到比特币网络，并从诸如 Blockchain.info 之类的地方获取数据。

我们将首先用 Python 编写比特币的等价物。在`hello_bitcoin.py`脚本中，使用 Python 创建了一个新的比特币地址的演示。

按照以下步骤运行程序：

1.  导入比特币库：

```py
#!/usr/bin/env python
'''
Title - Hello Bitcoin
This program demonstrates the creation of
- private key,
- public key
- and a bitcoin address.
'''

# import bitcoin
from bitcoin import *
```

1.  使用随机密钥函数生成私钥：

```py
my_private_key = random_key()
```

1.  在屏幕上显示私钥：

```py
print("Private Key: %s\n" % my_private_key)
```

# 如何生成私钥和公钥

使用私钥生成公钥。通过将生成的私钥传递给`privtopub`函数来执行此步骤，如下所示：

```py
# Generate Public Key
my_public_key = privtopub(my_private_key)
print("Public Key: %s\n" % my_public_key)
```

现在，使用公钥生成比特币地址。通过将生成的公钥传递给`pubtoaddr`函数来实现：

```py
# Create a bitcoin address
my_bitcoin_address = pubtoaddr(my_public_key)
print("Bitcoin Address: %s\n" % my_bitcoin_address)
```

以下屏幕截图显示了生成的私钥、公钥和比特币地址：

![](img/95a83334-f039-40a7-b63c-7fe85b072209.png)

比特币地址

比特币地址是一次性令牌。就像人们使用电子邮件地址发送和接收电子邮件一样，您可以使用此比特币地址发送和接收比特币。但与电子邮件地址不同，人们有许多不同的比特币地址，每个交易应使用唯一的地址。

# 创建多重签名比特币地址

多重签名地址是与多个私钥关联的地址；因此，我们需要创建三个私钥。

按照以下步骤创建多重签名比特币地址：

1.  创建三个私钥：

```py
#!/usr/bin/env python
'''
Title - Create multi-signature address

This program demonstrates the creation of
Multi-signature bitcoin address.
'''
# import bitcoin
from bitcoin import *

# Create Private Keys
my_private_key1 = random_key()
my_private_key2 = random_key()
my_private_key3 = random_key()

print("Private Key1: %s" % my_private_key1)
print("Private Key2: %s" % my_private_key2)
print("Private Key3: %s" % my_private_key3)
print('\n')
```

1.  使用`privtopub`函数从这些私钥创建三个公钥：

```py
# Create Public keys
my_public_key1 = privtopub(my_private_key1)
my_public_key2 = privtopub(my_private_key2)
my_public_key3 = privtopub(my_private_key3)

print("Public Key1: %s" % my_public_key1)
print("Public Key2: %s" % my_public_key2)
print("Public Key3: %s" % my_public_key3)
print('\n')
```

1.  生成公钥后，通过将这三个公钥传递给`mk_ multi-sig_script`函数来创建`multisig`。将生成的`multisig`传递给`addr`脚本函数以创建多重签名比特币地址。

```py
# Create Multi-signature address
my_multi_sig = mk_multisig_script(my_private_key1, my_private_key2, my_private_key3, 2,3)
my_multi_address = scriptaddr(my_multi_sig)
print("Multi signature address: %s" % my_multi_address)
```

1.  打印多重签名地址并执行脚本。

以下屏幕截图显示了`multisig`比特币地址的输出：

![](img/655bd911-28dd-4e14-8b3d-6bc99d5897e9.png)多重签名地址在组织中非常有用，因为没有单个个人被信任授权花费比特币。

您还可以查看现有比特币地址的交易历史。我们将首先从 Blockchain.info 获取有效地址。

以下屏幕截图显示了比特币区块的复制地址：

![](img/b05f0400-25ff-44fb-b015-2a45d1caa95e.png)

将复制的地址传递给`history`函数，如下面的代码所示，以及输出以获取比特币地址的历史记录，包括交易信息：

```py
!/usr/bin/env python
'''
Title - Bitcoin Transaction History

This program demonstrates listing history of a bitcoin address.
'''
# import bitcoin
from bitcoin import *

#View address transaction history
a_valid_bitcoin_address = '329e5RtfraHHNPKGDMXNxtuS4QjZTXqBDg'
print(history(a_valid_bitcoin_address))
```

![](img/643ab693-37fd-405e-ac80-f208c2264469.png)

# 使用 Python 进行区块链 API 编程

Blockchain.info 是最受欢迎的区块链和比特币网络浏览器和钱包提供商之一。通过网络，您可以查看区块级别并查看所有已发生的交易。例如，通过转到特定的区块—即区块＃536081—您可以查看所有交易，以及一些其他信息，如以下截图所示：

![](img/4b65c000-f000-444c-a656-9d5ee8203042.png)

以下截图显示了统计数据（DATA | Stats）。这很棒，也很有用；但是，对于基于此数据构建应用程序或进行分析的开发人员来说，以编程方式获取这些数据非常重要：

![](img/7f93a608-5230-47aa-96f3-f3316d4badbf.png)

以下截图显示了市场数据（DATA | Markets）：

![](img/b9e447e7-76b5-4edf-b319-8647abd739d4.png)

# 安装 Blockchain.info Python 库

以下是安装`blockchain` Python 库的步骤：

1.  在计算机上打开命令行程序。

1.  运行`pip install blockchain`命令来安装`blockchain`库。

以下截图显示了比特币的安装：

![](img/5142839b-84bb-48c4-b41c-c190bed038fb.png)

# 从 Blockchain.info 获取比特币汇率

以下步骤显示了比特币汇率的方法：

1.  首先从`blockchain`库中导入`exchangerates`类：

```py
#!/usr/bin/env python

# import blockchain library
from blockchain import exchangerates
```

1.  汇率定义了一个`get_ticker`方法，它返回字典对象中的汇率数据。调用此方法并保存结果对象。我们拥有的`ticker`字典对象具有货币符号作为键：

```py
# get the Bitcoin rates in various currencies
ticker = exchangerates.get_ticker()
```

1.  通过运行这些键，可以获取有关各种汇率的数据。例如，可以通过获取`p15min`最小值来获取每种货币的最新比特币汇率：

```py
# print the Bitcoin price for every currency
print("Bitcoin Prices in various currencies:")
for k in ticker:
 print(k, ticker[k].p15min)
```

以下截图显示了各种货币及其相应的比特币汇率，即时或过去 15 分钟内：

![](img/4d7fc642-d069-4c37-aadb-d8af215813d7.png)

特定货币也可以转换为比特币。例如，您可以传递`to_btc`方法，并传递我们要转换为`btc`的货币和金额，并将结果作为比特币获取。以下代码显示了如何对 100 欧元进行此操作：

```py
# Getting Bitcoin value for a particular amount and currency
btc = exchangerates.to_btc('EUR', 100)
print("\n100 euros in Bitcoin: %s " % btc)
```

以下截图显示了 100 欧元的比特币输出：

![](img/1158be8b-bc2a-4e51-a5ae-bf4cc372a452.png)

# 统计

比特币区块链库的下一个类称为`statistics`。

有许多方法可以调用以获取各种区块链统计数据，例如以下截图所示：

![](img/88786f17-cb0e-41b2-987b-263f52032aec.png)

您可以按以下方式调用不同的方法：

+   导入相关类，调用`statistics`上的`get`方法，并保存该对象。例如，要获取比特币交易量，我们应该从创建的`stats`对象中获取`trade_volume_btc`属性，如以下代码所示：

```py
#!/usr/bin/env python

# import blockchain library
from blockchain import statistics

# get the stats object
stats = statistics.get()

# get and print Bitcoin trade volume
print("Bitcoin Trade Volume: %s\n" % stats.trade_volume_btc)
```

以下截图显示了比特币交易量：

![](img/0afc2cff-a58c-4db9-b615-1306f961c7a3.png)

+   要获取总挖掘的比特币，请在`stats`对象上调用`btc_mined`属性，如下所示：

```py
# get and print Bitcoin mined
print("Bitcoin mined: %s\n" % stats.btc_mined)
```

以下截图显示了挖掘的比特币数量的输出：

![](img/a4adbf74-d9cb-4e6e-9e37-99b5059be2b3.png)

+   要获取比特币市场价格，请使用`stats`类，调用市场价格并将其附加到特定货币：

```py
# get and print Bitcoin market price in usd
print("Bitcoin market price: %s\n" % stats.market_price_usd)
```

+   当前比特币价格以美元显示如下：

![](img/6880238e-87d8-4731-a923-f5d5602e21e5.png)

# 区块浏览器方法

对于区块浏览器方法，首先从`blockchain`库中导入相关类。要获取特定的区块，请调用以下代码中显示的`get_block`方法。它期望将一个区块作为参数传递。

```py
# import blockchain library
from blockchain import blockexplorer

# get a particular block
block = blockexplorer.get_block('')
```

通过从网络上获取一个示例区块，从 Blockchain.info 上复制这个区块的哈希（区块＃536081），并将其传递给`get_block`方法，如下面的屏幕截图所示：

![](img/104d6267-98ec-448b-996d-46eec72c1d77.png)

现在让我们获取有关这个区块的一些信息。例如，可以通过在创建的`block`对象上分别使用`fee`、`size`和`transactions`属性来获取区块费用、区块大小和区块交易，如下面的代码所示：

```py
#!/usr/bin/env python

# import blockchain library
from blockchain import blockexplorer

# get a particular block
block = blockexplorer.get_block('0000000000000000002e90b284607359f3415647626447643b9b880ee00e41fa')

print("Block Fee: %s\n" % block.fee)
print("Block size: %s\n" % block.size)
print("Block transactions: %s\n" % block.transactions)

# get the latest block
block = blockexplorer.get_latest_block()
```

以下屏幕截图显示了区块费用、区块大小和区块交易输出：

![](img/26124509-7ea5-4f74-9286-caf2f0d01f20.png)

Blockchain.info 库中还有许多可用的功能；其中一些与钱包、创建钱包等更相关。

要进一步探索这个库，请访问链接[`github.com/blockchain/api-v1-client-python`](https://github.com/blockchain/api-v1-client-python)。

# 学习挖掘比特币

比特币挖掘的一些特点如下：

+   比特币挖矿是将比特币交易数据添加到比特币全球公共账本的过程。每个比特币矿工都与其他矿工一起合作，通过处理专门的分析和算术问题，将未完成的交易汇总到一个区块中。

+   为了获得准确性并解决问题，比特币矿工获取他们处理的所有交易。

+   除了交易费用，矿工还会收到每个他们挖掘的区块的额外奖励。任何人都可以通过运行计算机程序参与比特币挖矿。除了在传统计算机上运行外，一些公司设计了专门的比特币挖矿硬件，可以更快地处理交易并构建区块。

可以选择在[`www.bitcoin.com/`](https://www.bitcoin.com/)上云端挖掘比特币。

这些计划的过程表明，比特币挖矿的难度正在增加，随着时间的推移变得更加昂贵。

一些公司购买专门的硬件来挖掘比特币。其中一种硬件是来自 21.co 公司的 21 比特币计算机。因此，这种硬件预先安装了必要的软件。

# 如何挖掘比特币

还有许多可用的比特币挖矿软件，可以在任何机器上运行。然而，它可能不再那么高效。例如，让我们去[`www.bitcoinx.com/bitcoin-mining-software/`](http://www.bitcoinx.com/bitcoin-mining-software/)查看这样的软件的长列表。它们可以在各种操作系统上运行：Windows、Linux 和 macOS。有基于 UI 的比特币矿工，也有基于命令行的比特币矿工，例如 Python 中的实现 Pyminer。

# 比特币挖矿难度增加

由于竞争和困难的增加，挖掘比特币时必须牢记许多因素，如下列表所示：

+   由于竞争，比特币的价格日益昂贵

+   全球许多超级计算机正在竞争挖掘下一个区块和比特币。

+   随着比特币矿工数量的增加，开始挖掘新比特币变得更加困难和昂贵

例如，以下屏幕截图显示了比特币挖矿难度的增加情况；有关更多信息，请参考[`bitcoinwisdom.com/bitcoin/difficulty`](https://bitcoinwisdom.com/bitcoin/difficulty)。这张图表显示了过去两个月的数值。这一最近的趋势反映了比特币创立时开始的难度增加：

![](img/5587c8c8-bb4a-46f1-a115-6d7841464ab5.png)

# 总结

在本章中，我们学习了如何使用 Python 开始编程比特币。我们探索了使用 Python 进行 Blockchain.info API 编程，以获取统计数据和其他比特币市场数据。

我们还学习了如何开始挖掘比特币。我们看了看挖掘比特币的各种方式，并了解到由于竞争日益激烈和难度增加，比特币挖矿可能并不适合每个人。

在下一章中，我们将学习如何通过在网站上接受比特币、运行基于 API 的微服务，或者构建比特币交易机器人来开始以编程方式运行比特币。
