# 第九章。路由食谱

在本章中，我们将介绍以下食谱：

+   使用 `routes.py` 创建更干净的 URL

+   创建一个简单的路由器

+   添加 URL 前缀

+   将应用程序与域名关联

+   省略应用程序名称

+   从 URL 中删除应用程序名称和控制器

+   在 URL 中将下划线替换为连字符

+   映射 `favicons.ico` 和 `robots.txt`

+   使用 URL 指定语言

# 简介

在其核心，web2py 包含一个将 URL 映射到函数调用的分发器。这种映射称为路由，并且可以进行配置。这可能是有必要的，为了缩短 URL，或者将 web2py 应用程序作为现有应用程序的替代品部署，而不希望破坏旧的外部链接。web2py 随带两个路由器，即双向路由配置。旧的那个使用正则表达式匹配传入的 URL 并将其映射到应用/控制器/函数。而新风格的路由器则采用更全面的方法。

# 使用 routes.py 创建更干净的 URL

在 web2py 中，默认情况下，传入的 URL 被解释为 `http://domain.com/application/controller/function/arg1/arg2?var1=val1&var2=val2`。

即，URL 的前三个元素被解释为 web2py 应用程序名称、控制器名称和函数名称，剩余的路径元素保存在 `request.args`（一个 **列表**）中，查询字符串保存在 `request.vars`（一个 **字典**）中。

如果传入的 URL 路径元素少于三个，则使用默认值填充缺失的元素：`/init/default/index`，或者如果没有名为 `init` 的应用程序，则使用 `welcome` 应用程序填充：`/welcome/default/index`。web2py 的 `URL()` 函数从其组成部分创建 URL 路径（默认情况下，没有方案或域名）：应用程序、控制器、函数、参数列表和变量字典。结果通常用于网页中的 `href` 链接，以及重定向函数的参数。

作为其路由逻辑的一部分，web2py 还支持 URL 重写，其中配置文件 `routes.py` 指定 `URL()` 重写它生成的 URL 的规则，以及 web2py 解释传入的 URL 的规则。有两种独立的重写机制，这取决于在 `routes.py` 中配置的是哪一个。

一个使用正则表达式模式匹配来重写 URL 字符串，而另一个使用路由参数字典来控制重写。我们分别称它们为 **基于模式的路由器** 和 **基于参数的路由器**（有时它们分别被称为旧路由器和新路由器，但这些术语描述性不强，我们将避免使用它们）。

以下部分提供了一个旧路由器的使用示例。本章的其余部分提供了一个新路由器的使用示例。

## 准备工作

通常，Web URL 的结构如下 `http://host/app/controller/function/args`。

现在想象一个应用程序，其中每个用户都有自己的主页。例如：`http://host/app/default/home/johndoe`，其中 `home` 是渲染页面的动作，而 `johndoe` 是 `request.args(0)`，它告诉 web2py 我们正在寻找哪个用户。虽然这是可能的，但拥有如下外观的 URL 会更好：

`http://host/johndoe/home`。

这可以通过使用 web2py 的基于模式的路由机制来实现。

我们将假设以下名为 `pages` 的最小化应用程序。

在 `models/db.py` 中添加以下代码：

```py
db = DAL('sqlite://storage.sqlite')
from gluon.tools import *
auth = Auth(db)
auth.settings.extra_fields = [Field('html','text'),Field('css','te
xt')]
auth.define_tables(username=True)

```

将以下代码和常规脚手架文件添加到 `controllers/default.py` 中：

```py
def index():
	return locals()

def user():
	return dict(form=auth())

def home():
	return db.auth_user(username=request.args(0)).html

def css():
	response.headers['content-type']='text/css'
	return db.auth_user(username=request.args(0)).css

```

## 如何实现...

我们通过在主 web2py 文件夹中创建/编辑 `routes.py` 来实现以下规则：

```py
routes_in = (
	# make sure you do not break admin
	('/admin','/admin'),
	('/admin/$anything','/admin/$anything'),
	# make sure you do not break appadmin
	('/$app/appadmin','/$app/appadmin'),
	('/$app/appadmin/$anything','/$app/appadmin/$anything'),
	# map the specific urls for this the "pages" app
	('/$username/home','/pages/default/home/$username'),
	('/$username/css','/pages/default/css/$username'),
	# leave everything else unchanged
)

routes_out = (
	# make sure you do not break admin
	('/admin','/admin'),
	('/admin/$anything','/admin/$anything'),
	# make sure you do not break appadmin
	('/$app/appadmin','/$app/appadmin'),
	('/$app/appadmin/$anything','/$app/appadmin/$anything'),
	# map the specific urls for this the "pages" app
	('/pages/default/home/$username','/$username/home'),
	('/pages/default/css/$username','/$username/css'),
	# leave everything else unchanged
)

```

注意，`$app` 是正则表达式 `(? P<app>\w+)` 的快捷方式，它将匹配不包含斜杠的任何内容。`$username` 是 `(? P<username>\w+)` 的快捷方式。同样，您可以使用其他变量。`$anything` 是特殊的，因为它对应着不同的正则表达式，`(? P<app>.*)`；即，它将匹配直到 URL 结尾的任何内容。

代码的关键部分如下：

```py
routes_in=(
	...
	('/$username/home','/pages/default/home/$username'),
	...
)
routes_out=(
	...
	('/pages/default/home/$username','/$username/home'),
	...
)

```

这些映射了 `home` 的请求。然后我们对 `css` 动作做同样的处理。其余的代码实际上不是必需的，但它确保您不会意外地破坏 `admin` 和 `appadmin` URL。

# 创建一个简单的路由器

本章的此部分和下一部分将处理新的基于参数的路由器，它通常更容易配置，并且有效地处理大多数常见的重写任务。如果可能，请尝试使用基于参数的路由器，但如果您需要更多控制特殊 URL 重写任务，请查看基于模式的路由器。

使用基于参数的路由器的起点是将文件 `router.example.py` 复制到 `web2py` 的 `base` 目录中的 `routes.py`。(`routes.example.py` 文件对于基于模式的路由器也具有相同的作用。)该 `example` 文件包含其各自路由系统的基本文档；更多文档可在 web2py 书籍的在线版本中找到，第四章*核心：URL 重写和错误路由*。

每当 `routes.py` 发生更改时，您必须重新启动 web2py，或者如果管理员应用程序可用，加载以下 URL，以便新的配置生效：

`http://yourdomain.com/admin/default/reload_routes`

### 注意

示例路由文件包含一组 Python `doctests`。当您更改路由配置时，请向 `routes.py` 中的 `doctests` 添加或编辑，以检查您的配置是否符合预期。

我们想要解决的第一个问题是，在可能的情况下，我们想要从可见 URL 中消除默认应用程序和控制器。

## 如何实现...

1.  将 `router.example.py` 复制到主 web2py 文件夹中的 `routes.py`，并按以下方式编辑。找到 `routers` 字典：

    ```py
    routers = dict(
    	# base router
    	BASE = dict(
    		default_application = 'welcome',
    	),
    )

    ```

1.  将`default_application`从`welcome`更改为你的应用程序名称。如果你的默认控制器和函数没有命名为`default`和`index`，请指定这些默认值：

    ```py
    routers = dict(
    	# base router
    	BASE = dict(
    		default_application = 'myapp',
    		default_controller = 'mycontroller',
    		default_function = 'myfunction',
    	),
    )

    ```

# 添加 URL 前缀

通常，当你在一个生产服务器上运行 web2py 时，相同的 URL 可能被多个应用程序或服务共享，你需要添加一个额外的`PATH_INFO`前缀来识别 web2py 服务。例如：

[`example.com/php/`](http://example.com/php)

[`example.com/web2py/app/default/index`](http://example.com/web2py/app/default/index)

在这里，`web2py/`标识 web2py 服务，而`php/`标识一个 php 服务，映射是由网络服务执行的。你可能想从`PATH_INFO`中消除额外的`web2py/`。

## 如何操作...

当你指定`path_prefix`时，它被添加到由`URL()`生成的所有 URL 之前，并从所有传入的 URL 中移除。例如，如果你想你的外部 URL 看起来像`http://example.com/web2py/app/default/index`，你可以这样做：

```py
routers = dict(
	# base router
	BASE = dict(
		default_application = 'myapp',
		path_prefix = 'web2py',
	),
)

```

# 将应用程序与域名关联

通常，你想将特定的域名与特定的 web2py 应用程序关联起来，以便将指定域名指向的传入 URL 路由到适当的应用程序，而无需在 URL 中包含应用程序名称。再次强调，参数化路由器非常有用。

## 如何操作...

使用基于参数的路由器的域名功能：

```py
routers = dict(
	BASE = dict(
		domains = {
			"domain1.com" : "app1",
			"www.domain1.com" : "app1",
			"domain2.com" : "app2",
		},
		exclusive_domain = True,
	),
	# app1 = dict(...),
	# app2 = dict(...),
)

```

在此示例中，`domain1.com`和`domain2.com`由同一个物理主机提供服务。配置指定了将`domain1.com`（在这种情况下，其子域名`www`）的 URL 路由到`app1`，将`domain2.com`的 URL 路由到`app2`。如果`exclusive_domain`（可选）设置为`True`，那么来自除`domain2.com`（以及类似地对于`app1`）之外的域的请求尝试使用 URL 生成指向`app2`的 URL 将失败，除非它们明确提供主机名到 URL。

注意，你也可以使用以下方式，进一步配置两个应用的路径：

```py
app1 = dict(...),
app2 = dict(...),

```

# 省略应用程序名称

如果你正在使用参数化路由器，你可能想从静态文件的可视 URL 中省略默认应用程序名称。

## 如何操作...

这很简单；你只需按照以下方式打开`map_static`标志：

```py
routers = dict(
	# base router
	BASE = dict(
		default_application = 'myapp',
		map_static = True,
	),
)

```

或者，如果你正在使用特定应用程序的路由字典，为每个应用程序（例如以下示例中的`myapp`）打开`map_static`标志：

```py
routers = dict(
	# base router
	BASE = dict(
		default_application = 'myapp',
	),
	myapp = dict(
		map_static = True,
	),
)

```

# 从 URL 中移除应用程序名称和控制器

有时候，你想使用参数化路由器的 URL 解析，但又不想重写可见的 URL。再次强调，你可以使用参数化路由器，但请禁用 URL 重写。

## 如何操作...

在`routes.py`中找到路由器的`dict`，如下所示：

```py
routers = dict(
	# base router
	BASE = dict(
		default_application = 'welcome',
	),
)

```

找到它后，将其更改为以下内容：

```py
routers = dict(
	# base router
	BASE = dict(
		applications = None,
		controllers = None,
	),
)

```

## 它是如何工作的...

将 `applications` 和 `controllers` 设置为 `None`（函数和 `languages` 默认设置为 `None`），告诉参数路由器不要省略可见 URL 中的相应部分。web2py 的默认 URL 解析比许多应用可能需要的更严格，因为它假设 URL 组件可能用于文件名。参数路由器更紧密地遵循 HTTP URL RFCs，这使得它对需要更多异国情调字符在它们的参数或查询字符串中的应用程序更友好。本食谱中的 `null` 路由器启用此解析，而实际上不重写 URL。

# 在 URL 中将下划线替换为破折号

URL 中的下划线可能看起来很丑，当 URL 被下划线时，它们可能很难看到，就像它们通常在网页上那样。破折号是一个更视觉上吸引人的替代品，但你不能在函数名中使用破折号，因为它们还必须是合法的 Python 标识符。你可以使用参数路由器，将破折号映射为 `_!`

参数路由器的 `map_hyphen` 标志将应用、控制器和函数名称中的下划线转换为可见 URL 中的破折号，并在接收到 URL 时将其转换回下划线。`Args, vars`（查询字符串）和可能的语言选择器不受影响，因为破折号在这些字段中是允许的。因此，以下 URL：

`http://some_controller/some_function`

将显示为以下内容：

`http://some-controller/some-function`

虽然内部控制器和函数名称保留了它们的下划线。

## 如何做到这一点...

打开 `map_hyphen` 标志。在路由器指令中添加以下代码：

```py
routers = dict(
	# base router
	BASE = dict(
		default_application = 'myapp',
	),
	myapp = dict(
		map_hyphen = True,
	),
)

```

# 映射 favicon.ico 和 robots.txt

一些特殊文件，如 `robots.txt` 和 `favicon.ico`，作为 URL 的根路径直接访问。因此，它们必须从 `root` 文件夹映射到应用的 `static` 文件夹中。

## 如何做到这一点...

默认情况下，基于参数的路由器将 `root_static` 设置如下：

```py
routers = dict(
	# base router
	BASE = dict(
		default_application = 'myapp',
		root_static = ['favicon.ico', 'robots.txt']
	),
)

```

这指定了要服务的文件来自默认应用的静态目录。

# 使用 URL 指定语言

第二章中的“使用 cookies 设置语言”食谱描述了如何将用户语言偏好保存到 cookie 中。在这个食谱中，我们描述了如何做类似的事情——将用户语言偏好**存储**在 URL 中。这种方法的一个优点是，然后可以保存包含语言偏好的链接。

## 如何做到这一点...

参数路由器支持 URL 中的可选 `language` 字段，作为应用名称之后的字段：

[`domain.com/app/lang/controller/function`](http://domain.com/app/lang/controller/function)

语言字段遵循常规省略规则：如果使用默认语言，参数路由器将省略语言标识符，如果省略不会造成歧义。

基于 URL 的语言处理通常会在特定应用的参数路由器中指定，设置`default_language`和`languages`如下：

```py
routers = dict(
	# base router
	BASE = dict(
		default_application = app,
	),
	app = dict(
		default_language = 'en',
		languages = ['en', 'it', 'pt', 'pt-br'],
	),
)

```

要使用`URL()`指定出站 URL 的语言，将`request.lang`设置为支持的任何一种语言。对于入站请求，`request.lang`将被设置为入站 URL 指定的语言。与语言在 cookie 中的设置类似，在使用翻译之前，使用`T.force`强制在模型文件中使用所需的翻译。例如，在你的模型中，你可以执行以下操作：

```py
T.force(request.lang)

```
