# 第十二章。Grokkers，火星和敏捷配置

敏捷性在 Grok 中非常重要，并且为了使应用程序运行而需要做更少的配置是**敏捷**的关键。在 Grok 术语中，**grokker**是一段代码，允许开发者通过在代码中声明而不是使用 ZCML 配置文件来使用框架功能。在本章中，我们介绍了用于创建 grokkers 的库**火星**，并演示了如何为我们自己的应用程序创建一个简单的 grokker。我们将涵盖的主题包括：

+   什么是火星

+   为什么需要它以及 Grok 如何使用它

+   什么是 grokker

+   如何创建一个 grokker

# 敏捷配置

正如我们在本书一开始所解释的，当我们不使用 Grok 而使用 Zope Toolkit 时，我们必须使用 ZCML 来配置一切。这意味着我们必须为代码中的每个视图、视图组件、适配器、订阅者和注释添加 ZCML 指令。这里有很多标记，所有这些都必须与代码一起维护。当我们想到这一点时，敏捷性并不是首先想到的。

Grok 的开发者从经验中知道，Zope Toolkit 和**Zope 组件架构**（**ZCA**）使开发者能够创建高级面向对象系统。这种力量是以新开发者进入门槛提高为代价的。

另一个证明是 Zope Toolkit 采用问题的是，它对显式配置的强调。ZCML 允许开发者在其应用程序配置中非常明确和灵活，但它需要为配置单独的文件，并且需要更多的时间来创建、维护和理解。你只需要更多的时间来理解一个应用程序，因为你必须查看不同的代码片段，然后查阅 ZCML 文件以了解它们是如何相互关联的。

Grok 被设计成这样的方式，如果开发者在其代码中遵循某些约定，则不需要配置文件。相反，Grok 分析 Python 代码以使用这些约定，然后“理解”它们。幕后，一切连接正如如果配置是用 ZCML 编写的，但开发者甚至不需要考虑这一点。

由于这个过程被称为“理解”，因此 Grok 应用程序的代码干净且统一。整个配置都在代码中，以指令和组件的形式存在，因此更容易遵循，开发起来更有趣。

Grok 确实比单独的 Zope Toolkit 更敏捷，但它不是它的子集或“简化版”。Zope Toolkit 的所有功能都对开发者可用。甚至在需要时，可以使用 ZCML 进行显式配置，就像我们在上一章配置 SMTP 邮件器时所看到的那样。

# 火星库

Grok 中执行代码 'grokking' 的部分已被提取到一个名为 Martian 的独立库中。这个库提供了一个框架，允许以 Python 代码的形式表达配置，形式为声明性语句。想法是，通常，可以检查代码的结构，并且大多数它所需的配置步骤都可以从这个结构中推断出来。火星人通过使用指令来注释代码，使配置要求更加明显。

Martian 被发布为一个独立库，因为尽管它是 Grok 的关键部分，但它可以为任何类型的框架添加声明性配置。例如，`repoze.bfg`（[`bfg.repoze.org`](http://bfg.repoze.org)），一个基于 Zope 概念的最小化 Web 框架，使用 Martian 可选地允许在没有 ZCML 的情况下进行视图配置。

在程序启动时，火星人读取模块中的 Python 代码并分析所有类，以查看它们是否属于一个 'grokked' 基类（或其子类）。如果是，火星人将从类注册信息中检索信息以及其中可能包含的任何指令。然后，这些信息被用于在 ZCA 注册表中执行组件注册，这与 ZCML 机制类似。这个过程被称为 'grokking'，正如你所见，它允许在框架内快速注册插件。Grokkers 允许我们再次在同一个句子中写出 "agility" 和 "Zope Toolkit"，而无需带有讽刺意味。

# 理解 grokkers

Grokker 是一个包含要 grokked 的基类、一系列配置该类的指令以及使用 Martian 执行注册过程的实际代码的包。

让我们看看一个常规的 Grok 视图定义：

```py
class AddUser(grok.View):
grok.context(Interface)
grok.template('master')

```

在此代码中，`grok.View` 是一个已 grokked 的类，这意味着在程序启动时的 "Grok 时间" 来临时，它将被火星人找到，'grokked' 并注册到 ZCA。`grok.context` 和 `grok.template` 声明是该类可用的配置指令。实际的 'grokking' 是通过与已 grokked 类关联的一段代码来完成的，该代码将一个命名适配器注册到 ZCA 注册表中，该适配器是通过 `grok.context` 指令传入的接口。注册是通过使用类名来命名视图，以及将作为 `grok.template` 指令参数传递的任何字符串值来命名相关模板来完成的。

这就是 grokking 的全部意义，所以如果我们有三个必需的部分，我们就可以轻松地制作出自己的 grokkers。

## 已 grokked 的类

任何类都可以被 grokked；没有特殊要求。这使得开发者更容易开始，并且与它们一起工作要少得多。想象一下，我们有一些 `Mailer` 类想要 grok。它可以像这样简单：

```py
class Mailer(object):
pass

```

当然，它可以像需要的那样复杂，但重点是它不需要那么复杂。

## 指令

一旦我们有一个想要解析的类，我们就定义可能需要的指令来配置它。再次强调，这里没有强制要求。我们可能不需要指令就能完成配置，但大多数情况下我们可能需要几个指令。

```py
class hostname(martian.Directive):
scope = CLASS
default = 'localhost'
class port(martian.Directive):
scope = CLASS
default = 25

```

指令确实需要继承自`martian.Directive`子类。此外，它们至少需要指定一个作用域，可能还需要一个默认值。在这里，我们定义了两个指令`hostname`和`port`，这些指令将被用来配置邮件发送器。

## 类 grokker

我们 grokker 的最后一部分是执行实际注册的部分，它以继承自`martian.ClassGrokker`的类的形式出现：

```py
class MailGrokker(martian.ClassGrokker):
martian.component(Mailer)
martian.directive(hostname)
martian.directive(port)
def execute(self, class_, hostname, port, **kw):
register_mailer(class_, hostname, port)

```

grokker 类将解析的类与其指令连接起来，并执行解析或注册。它必须包含一个`execute`方法，该方法将负责任何配置操作。

`martian.component`指令将 grokker 与要解析的类（在这种情况下为`Mailer`）连接起来。`martian.directive`指令用于将我们之前定义的各种指令与这个 grokker 关联起来。

最后，`execute`方法接收基类和代码中声明的指令值，并执行最终的注册。请注意，`register_mailer`方法（实际上在这里执行工作）在前面代码中不存在，因为我们只想展示 grokker 的结构。

## 你将永远需要的唯一 ZCML

一旦 grokker 可用，它必须在启动时由 Grok 注册机制进行配置以初始化和使用。为此，我们必须在名为`meta.zcml`的文件中使用一些 ZCML：

```py
<configure >
<grok:grok package=".meta" />
</configure>

```

如果我们的`MailGrokker`类位于`meta.py`文件中，它将由 Grok 机制初始化。

# 为 zope.sendmail 配置创建我们自己的 grokker

现在我们知道了 grokker 的结构，让我们创建一个用于 SMTP 邮件发送器的 grokker，这个 SMTP 邮件发送器是我们之前在第十一章关于添加电子邮件通知的部分所使用的。

我们想要的是一个简单的`MailGrokker`类声明，包含`hostname`、`port`、`username`、`password`和`delivery type`指令。这将使我们能够避免使用 ZCML 来配置邮件发送器，正如我们在上一节所要求的那样。

我们需要创建一个新的包，这样我们的 grokker 就可以独立于`todo_plus`代码，并且可以在其他地方自由使用。

## 创建包

我们在第十一章的*创建新包*部分执行了这些步骤。如果您有任何疑问，请参阅该部分以获取详细信息。

要创建包，请进入我们主要`todo`应用的`src`目录，并输入：

```py
$ ../bin/paster create -t basic_package mailgrokker 

```

这将创建一个`mailgrokker`目录。现在，导航到这个目录，并将`grok`、`martian`以及`zope.sendmail`包添加到`install_requires`声明中：

```py
install_requires=[
'grok',
'martian',
'zope.sendmail',
],

```

这样，我们确保在安装`mailgrokker`后，所需的包是存在的。我们还必须将我们的新`mailgrokker`包添加到项目顶层的主`buildout.cfg`文件中，紧接在`todo_plus`下面。在 egg 和 develop 部分都这样做。

## 编写我们的解析器

首先，我们将添加一个`configure.zcml`文件，它就像`todo_plus`包中的那个一样。实际上，我们可以从那里复制它：

```py
<configure   >
<include package="grok" />
<includeDependencies package="." />
<grok:grok package="." />
</configure>

```

我们解析的类将位于`component.py`文件中。在这里，我们只使用一个基类，但一个解析器项目可以包含多个基类，并且按照惯例，它们在这里定义：

```py
import grok
class Mailer(object):
grok.baseclass()

```

这是一个没有方法的简单基类。使用`grok.baseclass`指令将其标记为基类，尽管这不是强制性的。

配置指令存储在一个名为`directives.py`的文件中。

```py
import martian
class name(martian.Directive):
scope = martian.CLASS
store = martian.ONCE
class hostname(martian.Directive):
scope = martian.CLASS
store = martian.ONCE
default = 'localhost'
class port(martian.Directive):
scope = martian.CLASS
store = martian.ONCE
default = '25'
class username(martian.Directive):
scope = martian.CLASS
store = martian.ONCE
default = None
class password(martian.Directive):
scope = martian.CLASS
store = martian.ONCE
default = None
class delivery(martian.Directive):
scope = martian.CLASS
store = martian.ONCE
default = 'queued'
class permission(martian.Directive):
scope = martian.CLASS
store = martian.ONCE
default = 'zope.Public'

```

这非常直接。我们只需定义我们需要的所有指令，然后添加一个`martian.CLASS`作用域。每个指令都有自己的默认值，这取决于其目的。通过查看代码，每个指令的意图应该是显而易见的，除了`delivery`指令。这个指令是必需的，因为`zope.sendmail`包括两种不同的交付机制`direct`和`queued`。

接下来是主要的解析器类，我们将将其添加到`meta.py`文件中。首先，是`import`语句。注意这里我们导入了`martian`以及`GrokError`，这是一个异常，如果解析失败，我们可以抛出它。我们还导入了我们将从`zope.sendmail`库中使用的所有内容。

```py
import martian
from martian.error import GrokError
from zope.component import getGlobalSiteManager
from zope.sendmail.delivery import QueuedMailDelivery, DirectMailDelivery
from zope.sendmail.delivery import QueueProcessorThread
from zope.sendmail.interfaces import IMailer, IMailDelivery
from zope.sendmail.mailer import SMTPMailer
from zope.sendmail.zcml import _assertPermission
from mailgrokker.components import Mailer
from mailgrokker.directives import name, hostname, port, username, password, delivery, permission

```

`register_mailer`函数创建一个`zope.sendmail` SMTP 邮件发送对象，并将其注册为名为`IMailer`的命名实用工具，名称来自`name`指令。注意使用`getGlobalSiteManager`函数，这实际上是一个获取组件注册表的华丽名称。我们使用注册表的`registerUtility`函数添加我们新创建的`SMTPMailer`实例。

```py
def register_mailer(class_, name, hostname, port, username, password, delivery, permission):
sm = getGlobalSiteManager()
mailer = SMTPMailer(hostname, port, username, password)
sm.registerUtility(mailer, IMailer, name)

```

继续使用`register_mailer`代码，我们现在使用传递的参数选定的交付机制来决定是否初始化一个`DirectMailDelivery`实例或一个`QueuedMailDelivery`实例。无论哪种方式，我们都将结果注册为一个实用工具。

在`queue`交付机制的情况下，一个将负责从主应用程序代码中单独发送电子邮件的线程被启动。

```py
if delivery=='direct':
mail_delivery = DirectMailDelivery(mailer)
_assertPermission(permission, IMailDelivery, mail_delivery)
sm.registerUtility(mail_delivery, IMailDelivery, name)
elif delivery=='queued':
mail_delivery = QueuedMailDelivery(name)
_assertPermission(permission, IMailDelivery, mail_delivery)
sm.registerUtility(mail_delivery, IMailDelivery, name)
thread = QueueProcessorThread()
thread.setMailer(mailer)
thread.setQueuePath(name)
thread.start()
else:
raise GrokError("Available delivery methods are 'direct' and 'queued'. Delivery method %s is not defined.",class_)

```

`MailGrokker`类声明了添加到`directives`模块的所有指令，并将其与它将要解析的`Mailer`类关联起来。然后它定义了`execute`方法，该方法将调用`register_mailer`函数以执行所需的`zope.sendmail`注册。

```py
class MailGrokker(martian.ClassGrokker):
martian.component(Mailer)
martian.directive(name)
martian.directive(hostname)
martian.directive(port)
martian.directive(username)
martian.directive(password)
martian.directive(delivery)
martian.directive(permission)
def execute(self, class_, config, name, hostname, port, username, password, delivery, permission, **kwds):
config.action(
discriminator = ('utility', IMailer, name),
callable = register_mailer,
args = (class_, name, hostname, port, username, password, delivery, permission),
order = 5
)
return True

```

上述代码与我们之前展示的代码的唯一区别是，我们不是直接调用`register_mailer`函数，而是将其包裹在`config.action`对象中。这样做是为了让 Grok 在代码加载后以任意顺序执行注册，而不是在初始化每个包时执行。这防止了任何配置冲突，并允许我们具体指定注册条件。

例如，`discriminator`参数，可能是空的，在这种情况下，是一个包含字符串`utility`、接口`IMailer`和`name`指令值的元组。如果任何其他 grokker 包使用这个相同的判别器，Grok 将发出冲突错误条件。

`action`的`order`参数用于指定调用动作的顺序，尽管这里只是添加了用于演示的目的。`callable`参数是执行注册的函数，而`args`参数包含传递给它的参数。

我们现在在`meta`模块中有了我们的 grokker，需要告诉 Grok 在这里找到它，我们通过添加之前讨论过的小的`meta.zcml`文件来完成：

```py
<configure >
<grok:grok package=".meta" />
</configure>

```

最后，编辑现有的`__init__.py`文件，该文件位于`src/mailgrokker/mailgrokker`目录中，使其看起来像以下代码：

```py
from mailgrokker.directives import name, hostname, port, username, password, del ivery, permission
from mailgrokker.components import Mailer

```

这将允许我们通过导入主要的`mailgrokker`模块来简单地使用指令，就像`grok.*`指令那样工作。

## 使用 mailgrokker

现在我们已经完成了我们的 grokker，唯一缺少的是展示如何在应用程序中使用它。我们将将其添加到`todo_plus`包中。在该文件的底部插入以下行：

```py
import mailgrokker
class TodoMailer(mailgrokker.Mailer):
mailgrokker.name('todoplus')
mailgrokker.hostname('smtp.example.com')
mailgrokker.username('cguardia')
mailgrokker.password('password')

```

显然，你应该用你的`smtp`服务器的实际值替换这里显示的值。你可能还想删除我们在之前的`configure.zcml`文件中放置的邮件发送器配置。

完成。我们现在创建了一个小的 grokker 包，可以在我们的任何应用程序中使用，以便轻松配置电子邮件提交。

# 摘要

在本章中，我们学习了关于 Martian 库以及它是如何使 Grok 成为一个敏捷框架的。我们现在准备好讨论如何调试我们的应用程序。
