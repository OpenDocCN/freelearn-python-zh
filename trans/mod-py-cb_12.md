# 第十二章。Web 服务

在本章中，我们将查看以下配方：

+   使用 WSGI 实现 Web 服务

+   使用 Flask 框架进行 RESTful API

+   解析请求中的查询字符串

+   使用 urllib 进行 REST 请求

+   解析 URL 路径

+   解析 JSON 请求

+   为 Web 服务实施身份验证

# 介绍

提供 Web 服务涉及解决几个相互关联的问题。必须遵循一些适用的协议，每个协议都有其独特的设计考虑。Web 服务的核心是定义 HTTP 的各种标准。

HTTP 涉及两方；客户端和服务器：

+   客户端向服务器发出请求

+   服务器向客户端发送响应

这种关系是高度不对称的。我们期望服务器处理来自多个客户端的并发请求。因为客户端请求是异步到达的，服务器不能轻易区分那些来自单个人类用户的请求。通过设计提供会话令牌（或 cookie）来跟踪人类当前状态的服务器来实现人类用户会话的概念。

HTTP 协议是灵活和可扩展的。HTTP 的一个流行用例是以网页的形式提供内容。网页通常被编码为 HTML 文档，通常包含指向图形、样式表和 JavaScript 代码的链接。我们已经在第九章的*读取 HTML 文档*中查看了解析 HTML 的信息，*输入/输出、物理格式和逻辑布局*。

提供网页内容进一步分解为两种内容：

+   静态内容本质上是文件的下载。诸如 GUnicorn、NGINGX 或 Apache HTTPD 之类的程序可以可靠地提供静态文件。每个 URL 定义了文件的路径，服务器将文件下载到浏览器。

+   动态内容是根据需要由应用程序构建的。在这种情况下，我们将使用 Python 应用程序响应请求构建唯一的 HTML（或可能是图形）。

HTTP 的另一个非常流行的用例是提供 Web 服务。在这种情况下，标准的 HTTP 请求和响应将以 HTML 以外的格式交换数据。编码信息的最流行格式之一是 JSON。我们已经在第九章的*读取 JSON 文档*中查看了处理 JSON 文档的信息，*输入/输出、物理格式和逻辑布局*。

Web 服务可以被视为使用 HTTP 提供动态内容的一种变体。客户端可以准备 JSON 文档。服务器包括一个创建 JSON 表示的 Python 应用程序。

在某些情况下，服务的焦点非常狭窄。将服务和数据库持久性捆绑到一个单一的包中是可能的。这可能涉及创建一个具有基于 NGINX 的 Web 界面以及使用 MongoDB 或 Elastic 的数据库的服务器。整个包 - Web 服务加持久性 - 可以称为**微服务**。

Web 服务交换的文档编码了对象状态的表示。JavaScript 中的客户端应用程序可能具有发送到服务器的对象状态。Python 中的服务器可能会将对象状态的表示传输给客户端。这被称为**表述性状态转移**（**REST**）。使用 REST 处理的服务通常被称为 RESTful。

处理 HTML 或 JSON 的 HTTP 可以设计为一系列转换函数。思路如下：

```py
    response = F(request, persistent state) 

```

响应是通过某个函数`F(r, s)`从请求中构建的，该函数依赖于服务器上数据库中的请求加上一些持久状态。

这些函数形成了围绕核心服务的嵌套外壳或包装器。例如，核心处理可能被包装以确保发出请求的用户被授权更改数据库状态。我们可以总结如下：

```py
    response = auth(F(request, persistent state)) 

```

授权处理可能被包装在处理中，以验证用户的凭据。所有这些可能进一步包装在一个外壳中，以确保客户端应用程序软件期望以 JSON 表示形式进行响应。像这样使用多个层可以为许多不同的核心服务提供一致的操作。整个过程可能开始看起来像这样：

```py
    response = JSON( user( auth( F(request, persistent state) ) ) ) 

```

这种设计自然适用于一系列转换函数。这个想法为我们提供了一些指导，指导我们设计包括许多协议和创建有效响应的许多规则的复杂 Web 服务的方式。

一个良好的 RESTful 实现还应该提供关于服务的大量信息。提供此信息的一种方式是通过 OpenAPI 规范。有关 OpenAPI（Swagger）规范的信息，请参阅[`swagger.io/specification/`](http://swagger.io/specification/)。

OpenAPI 规范的核心是 JSON 模式规范。有关更多信息，请参阅[`json-schema.org`](http://json-schema.org)。

这两个基本思想如下：

1.  我们以 JSON 格式编写了发送到服务的请求和服务提供的响应的规范。

1.  我们在固定的 URL 上提供规范，通常是`/swagger.json`。客户端可以查询此 URL 以确定服务的详细信息。

创建 Swagger 文档可能具有挑战性。`swagger-spec-validator`项目可以提供帮助。请参阅[`pypi.python.org/pypi/swagger-spec-validator`](https://pypi.python.org/pypi/swagger-spec-validator)。这是一个 Python 包，我们可以使用它来确认 Swagger 规范是否符合 OpenAPI 要求。

在本章中，我们将探讨创建 RESTful Web 服务以及提供静态或动态内容的一些方法。

# 使用 WSGI 实现 Web 服务

许多 Web 应用程序将具有多个层。这些层通常可以总结为三种常见模式：

+   演示层可能在移动设备或网站上运行。这是可见的外部视图。

+   应用层通常实现为 Web 服务。该层对 Web 或移动演示进行处理。

+   持久层处理数据的保留和事务状态，以及来自单个用户的多个会话中的数据。这将支持应用程序层。

基于 Python 的网站或 Web 服务应用程序将遵守**Web 服务网关接口**（**WSGI**）标准。这为前端 Web 服务器（如 Apache HTTPD、NGINX 或 GUnicorn）提供了一种统一的方式来使用 Python 提供动态内容。

Python 有各种各样的 RESTful API 框架。在*使用 Flask 框架创建 RESTful API*的示例中，我们将看到 Flask。然而，在某些情况下，核心 WSGI 功能可能是我们所需要的。

我们如何创建支持遵循 WSGI 标准的分层组合的应用程序？

## 准备就绪

WSGI 标准定义了一个可组合的 Web 应用程序的总体框架。其背后的想法是定义每个应用程序，使其能够独立运行，并可以轻松连接到其他应用程序。整个网站是由一系列外壳或包装器构建的。

这是一种基本的 Web 服务器开发方法。WSGI 不是一个复杂的框架；它是一个最小的标准。我们将在*使用 Flask 框架创建 RESTful API*的示例中探讨一些简化设计的方法。

Web 服务的本质是 HTTP 请求和响应。服务器接收请求并创建响应。HTTP 请求包括几个数据部分：

+   资源的 URL。URL 可以像`http://www.example.com:8080/?query#fragment`这样复杂。URL 有几个部分：

+   方案`http`：以`:`结束。

+   主机`www.example.com`：这是以`//`为前缀的。它可能包括一个可选的端口号。在这种情况下，它是`8080`。

+   资源的路径：在本例中是`/`字符。路径以某种形式是必需的。它通常比简单的`/`更复杂。

+   以`?`为前缀的查询字符串：在本例中，查询字符串只是带有没有值的键`query`。

+   以`#`为前缀的片段标识符：在本例中，片段是`fragment`。对于 HTML 文档，这可以是特定标签的`id`值；浏览器将滚动到命名标签。

几乎所有这些 URL 元素都是可选的。我们可以利用查询字符串（或片段）来提供有关请求的附加格式信息。

WSGI 标准要求解析 URL。各种片段放入环境中。每个片段将被分配一个单独的键：

+   **方法**：常见的 HTTP 方法包括`HEAD`，`OPTIONS`，`GET`，`POST`，`PUT`和`DELETE`。

+   **请求标头**：标头是支持请求的附加信息。例如，标头用于定义可以接受的内容类型。

+   **附加内容**：请求可能包括来自 HTML 表单的输入，或要上传的文件。

HTTP 响应在许多方面类似于请求。它包含响应标头和响应正文。标头将包括诸如内容的编码，以便客户端可以正确地呈现它的细节。如果服务器提供 HTML 内容并维护服务器会话，那么 cookie 将作为每个请求和响应的一部分在标头中发送。

WSGI 旨在帮助创建可以用于构建更大更复杂应用程序的应用程序组件。WSGI 应用程序通常充当包装器，保护其他应用程序免受错误请求、未经授权的用户或未经身份验证的用户的影响。为了做到这一点，每个 WSGI 应用程序必须遵循一个共同的标准定义。每个应用程序必须是一个函数或可调用对象，并具有以下签名：

```py
    def application(environ, start_response): 
        start_response('200 OK', [('Content-Type', 'text/plain')]) 
        return iterable_strings 

```

`environ`参数是一个包含有关请求的信息的字典。这包括所有 HTTP 细节，加上操作系统上下文，加上 WSGI 服务器上下文。`start_response`参数是一个必须在返回响应正文之前调用的函数。这提供了响应的状态和标头。

WSGI 应用程序函数的返回值是 HTTP 响应正文。这通常是一系列字符串或字符串值的可迭代对象。这里的想法是，WSGI 应用程序可能是一个更大容器的一部分，该容器将从服务器向客户端流式传输响应，因为响应正在构建。

由于所有 WSGI 应用程序都是可调用函数，它们可以很容易地组合。一个复杂的网络服务器可能有几个 WSGI 组件来处理身份验证、授权、标准标头、审计日志、性能监控等细节。这些方面通常独立于底层内容；它们是所有网络应用程序或 RESTful 服务的通用特性。

我们将看一个相对简单的网络服务，它可以从牌组或鞋子中发出纸牌。我们将依赖于来自第六章*类和对象的基础*的*使用 __slots__ 优化小对象*配方中的`Card`类定义。这是核心的`Card`类，带有等级和花色信息：

```py
    class Card: 
        __slots__ = ('rank', 'suit') 
        def __init__(self, rank, suit): 
            self.rank = int(rank) 
            self.suit = suit 
        def __repr__(self): 
            return ("Card(rank={self.rank!r}, " 
             "suit={self.suit!r})").format(self=self) 
        def to_json(self): 
            return { 
                "__class__": "Card",  
                'rank': self.rank,  
                'suit': self.suit} 

```

我们为纸牌定义了一个小的基类。该类的每个实例都有两个属性，`rank`和`suit`。我们省略了哈希和比较方法的定义。要遵循第七章*更高级的类设计*中的*创建具有可排序对象的类*配方，这个类需要许多额外的特殊方法。这个配方将避免这些复杂性。

我们定义了一个`to_json()`方法，用于将这个复杂对象序列化为一致的 JSON 格式。该方法发出`Card`状态的字典表示。如果我们想要从 JSON 表示中反序列化`Card`对象，我们还需要创建一个`object_hook`函数。不过，对于这个示例，我们不需要它，因为我们不会接受`Card`对象作为输入。

我们还需要一个`Deck`类作为`Card`实例的容器。该类的一个实例可以创建`Card`实例，同时充当一个有状态的对象，可以发牌。以下是类定义：

```py
    import random 
     class Deck: 
        SUITS = ( 
            '\N{black spade suit}', 
            '\N{white heart suit}', 
            '\N{white diamond suit}', 
            '\N{black club suit}', 
        ) 

        def __init__(self, n=1): 
            self.n = n 
            self.create_deck(self.n) 

        def create_deck(self, n=1): 
            self.cards = [ 
                Card(r,s)  
                    for r in range(1,14)  
                        for s in self.SUITS  
                            for _ in range(n) 
            ] 
            random.shuffle(self.cards) 
            self.offset = 0 

        def deal(self, hand_size=5): 
            if self.offset + hand_size > len(self.cards): 
                self.create_deck(self.n) 
            hand = self.cards[self.offset:self.offset+hand_size] 
            self.offset += hand_size 
            return hand 

```

`create_deck()`方法使用生成器来创建所有 52 种组合的十三个等级和四种花色。每种花色由一个单字符定义：♣，♢，♡或♠。示例使用`\N{}`序列来拼写 Unicode 字符名称。

如果在创建`Deck`实例时提供了`n`的值，容器将创建 52 张牌的多个副本。这种多副牌鞋有时用于通过减少洗牌时间来加快游戏速度。一旦`Card`实例的序列被创建，就会使用`random`模块对其进行洗牌。对于可重复的测试用例，可以提供一个固定的种子。

`deal()`方法将使用`self.offset`的值来确定从哪里开始发牌。这个值从`0`开始，并在每发一手牌后递增。`hand_size`参数决定下一手牌有多少张。该方法通过递增`self.offset`的值来更新对象的状态，以便牌只被发一次。

以下是使用这个类创建`Card`对象的一种方法：

```py
 **>>> from ch12_r01 import deck_factory 
>>> import random 
>>> import json 

>>> random.seed(2) 
>>> deck = Deck() 
>>> cards = deck.deal(5) 
>>> cards   
[Card(rank=4, suit='♠'), Card(rank=8, suit='♡'), 
 Card(rank=3, suit='♡'), Card(rank=6, suit='♡'), 
 Card(rank=2, suit='♣')]** 

```

为了创建一个合理的测试，我们提供了一个固定的种子值。脚本使用`Deck()`创建了一副牌。然后我们可以从牌组中发出五张`Card`实例。

为了将其作为 Web 服务的一部分使用，我们还需要以 JSON 表示形式产生有用的输出。以下是一个示例，展示了这样的输出：

```py
 **>>> json_cards = list(card.to_json() for card in deck.deal(5)) 
>>> print(json.dumps(json_cards, indent=2, sort_keys=True))** 

    [ 
      { 
        "__class__": "Card", 
        "rank": 2, 
        "suit": "\u2662" 
      }, 
      { 
        "__class__": "Card", 
        "rank": 13, 
        "suit": "\u2663" 
      }, 
      { 
        "__class__": "Card", 
        "rank": 7, 
        "suit": "\u2662" 
      }, 
      { 
        "__class__": "Card", 
        "rank": 6, 
        "suit": "\u2662" 
      }, 
      { 
        "__class__": "Card", 
        "rank": 7, 
        "suit": "\u2660" 
      } 
    ] 

```

我们使用`deck.deal(5)`来从牌组中发 5 张牌。表达式`list(card.to_json() for card in deck.deal(5))`将使用每个`Card`对象的`to_json()`方法来发出该对象的小字典表示。然后将字典结构的列表序列化为 JSON 表示形式。`sort_keys=True`选项对于创建可重复的测试用例很方便。对于 RESTful Web 服务通常不是必需的。

## 如何做...

1.  导入所需的模块和对象。我们将使用`HTTPStatus`类，因为它定义了常用的 HTTP 状态码。需要`json`模块来生成 JSON 响应。我们还将使用`os`模块来初始化随机数种子：

```py
        from http import HTTPStatus 
        import json 
        import os 
        import random 

```

1.  导入或定义底层类，`Card`和`Deck`。通常，最好将这些定义为一个单独的模块。基本功能应该存在并在 Web 服务环境之外进行测试。这样做的想法是 Web 服务应该包装现有的、可工作的软件。

1.  创建所有会话共享的对象。`deck`的值是一个模块全局变量：

```py
        random.seed(os.environ.get('DEAL_APP_SEED')) 
        deck = Deck() 

```

我们依赖`os`模块来检查环境变量。如果环境变量`DEAL_APP_SEED`被定义，我们将使用该字符串值来生成随机数。否则，我们将依赖`random`模块的内置随机化特性。

1.  将目标 WSGI 应用程序定义为一个函数。该函数将通过发一手牌来响应请求，然后创建`Card`信息的 JSON 表示形式：

```py
        def deal_cards(environ, start_response): 
            global deck 
            hand_size = int(environ.get('HAND_SIZE', 5)) 
            cards = deck.deal(hand_size) 
            status = "{status.value} {status.phrase}".format(
             status=HTTPStatus.OK) 
            headers = [('Content-Type', 'application/json;charset=utf-8')] 
            start_response(status, headers) 
            json_cards = list(card.to_json() for card in cards) 
            return [json.dumps(json_cards, indent=2).encode('utf-8')] 

```

`deal_cards()`函数从`deck`中发牌下一组牌。操作系统环境可以定义`HAND_SIZE`环境变量来改变发牌的大小。全局`deck`对象用于执行相关处理。

响应的状态行是一个字符串，其中包含 HTTP 状态为`OK`的数值和短语。这可以跟随标头。这个例子包括`Content-Type`标头，向客户端提供信息；内容是一个 JSON 文档，这个文档的字节使用`utf-8`进行编码。最后，文档本身是这个函数的返回值。

1.  出于演示和调试目的，构建一个运行 WSGI 应用程序的服务器是有帮助的。我们将使用`wsgiref`模块的服务器。在 Werkzeug 中定义了良好的服务器。像 GUnicorn 这样的服务器甚至更好：

```py
        from wsgiref.simple_server import make_server 
        httpd = make_server('', 8080, deal_cards) 
        httpd.serve_forever() 

```

服务器运行后，我们可以打开浏览器查看`http://localhost:8080/`。这将返回一批五张卡片。每次刷新，我们都会得到不同的一批卡片。

这是因为在浏览器中输入 URL 会执行一个带有最小一组标头的`GET`请求。由于我们的 WSGI 应用程序不需要任何特定的标头，并且对任何 HTTP 方法都有响应，它将返回一个结果。

结果是一个 JSON 文档，表示从当前牌组中发出的五张卡片。每张卡片都用一个类名`rank`和`suit`表示：

```py
    [ 
      { 
        "__class__": "Card", 
        "suit": "\u2663", 
        "rank": 6 
      }, 
      { 
        "__class__": "Card", 
        "suit": "\u2662", 
        "rank": 8 
      }, 
      { 
        "__class__": "Card", 
        "suit": "\u2660", 
        "rank": 8 
      }, 
      { 
        "__class__": "Card", 
        "suit": "\u2660", 
        "rank": 10 
      }, 
      { 
        "__class__": "Card", 
        "suit": "\u2663", 
        "rank": 11 
      } 
    ] 

```

我们可以创建带有聪明的 JavaScript 程序的网页来获取一批卡片。这些网页和 JavaScript 程序可以用于动画处理，并包括卡片图像的图形。

## 工作原理...

WSGI 标准定义了 Web 服务器和应用程序之间的接口。这是基于 Apache HTTPD 的**公共网关接口**（**CGI**）。CGI 旨在运行 shell 脚本或单独的二进制文件。WSGI 是对这一传统概念的增强。

WSGI 标准使用环境字典定义了各种信息：

+   字典中的许多键反映了一些初步解析和数据转换后的请求。

+   `REQUEST_METHOD`：HTTP 请求方法，如`GET`或`POST`。

+   `SCRIPT_NAME`：请求 URL 路径的初始部分。这通常被视为整体应用程序对象或函数。

+   `PATH_INFO`：请求 URL 路径的其余部分，指定资源的位置。在这个例子中，不执行路径解析。

+   `QUERY_STRING`：请求 URL 中跟随`?`后的部分，如果有的话：

+   `CONTENT_TYPE`：HTTP 请求中任何 Content-Type 标头值的内容。

+   `CONTENT_LENGTH`：HTTP 请求中任何 Content-Length 标头值的内容。

+   `SERVER_NAME`和`SERVER_PORT`：请求中的服务器名称和端口号。

+   `SERVER_PROTOCOL`：客户端用于发送请求的协议版本。通常情况下，这可能是类似于`HTTP/1.0`或`HTTP/1.1`的内容。

+   **HTTP 标头**：这些标头将以`HTTP_`开头，并且以全部大写字母包含标头名称的键。

通常，请求的内容不是从服务器创建有意义的响应所需的唯一数据。通常需要额外的信息。这些信息通常包括另外两种类型的数据：

+   **操作系统环境**：在服务启动时存在的环境变量为服务器提供配置详细信息。这可能提供一个包含静态内容的目录路径。它可能提供用于验证用户的信息。

+   **WSGI 服务器上下文**：这些键以`wsgi.`开头，始终为小写。值包括一些关于遵循 WSGI 标准的服务器内部状态的附加信息。有两个特别有趣的对象，用于上传文件和日志支持：

+   `wsgi.input`：它是一个类似文件的对象。可以从中读取 HTTP 请求体字节。这通常需要根据`Content-Type`标头进行解码。

+   `wsgi.errors`：这是一个类似文件的对象，可以将错误输出写入其中。这是服务器的日志。

WSGI 函数的返回值可以是序列对象或可迭代对象。返回可迭代对象是构建非常大的文档并通过多个较小的缓冲区下载的方法。

此示例 WSGI 应用程序不检查请求路径。可以使用任何路径来检索一手牌。更复杂的应用程序可能会解析路径以确定有关所请求的手牌大小或应该从中发牌的牌组大小的信息。

## 还有更多...

Web 服务可以被视为连接到嵌套外壳或层中的一些常见部分。WSGI 应用程序的统一接口鼓励可重用功能的这种组合。

有许多常见的技术用于保护和生成动态内容。这些技术是 Web 服务应用程序的横切关注点。我们有以下几种选择：

+   我们可以在单个应用程序中编写许多`if`语句。

+   我们可以提取常见的编程并创建一个将安全性问题与内容构建分离的通用包装器

包装器只是另一个不直接产生结果的 WSGI 应用程序。相反，包装器将产生结果的工作交给另一个 WSGI 应用程序。

例如，我们可能需要一个确认期望 JSON 响应的包装器。此包装器将区分人类为中心的 HTML 请求和面向应用程序的 JSON 请求。

为了创建更灵活的应用程序，通常使用可调用对象而不是简单的函数是有帮助的。这样做可以使各种应用程序和包装器的配置更加灵活。我们将将 JSON 过滤器的概念与可调用对象结合起来。

这个对象的概述如下：

```py
    class JSON_Filter: 
        def __init__(self, json_app): 
            self.json_app = json_app 
        def __call__(self, environ, start_response): 
            return json_app(environ, start_response) 

```

通过提供另一个应用程序，`json_app`，我们将从这个类定义中创建一个可调用对象。

我们将像这样使用它：

```py
    json_wrapper = JSON_Filter(deal_cards) 

```

这将包装原始的`deal_cards()`WSGI 应用程序。现在我们可以将复合`json_wrapper`对象用作 WSGI 应用程序。当服务器调用`json_wrapper(environ, start_response)`时，将调用对象的`__call__()`方法，在这个例子中，将请求传递给`deal_cards()`函数。

以下是更完整的包装器应用程序。此包装器将检查 HTTP Accept 标头中的字符`"json"`。它还将检查查询字符串以查看是否进行了`?$format=json`的 JSON 格式请求。此类的一个实例可以配置为引用`deal_cards()`WSGI 应用程序：

```py
    from urllib.parse import parse_qs 
    class JSON_Filter: 
        def __init__(self, json_app): 
            self.json_app = json_app 
        def __call__(self, environ, start_response): 
            if 'HTTP_ACCEPT' in environ: 
                if 'json' in environ['HTTP_ACCEPT']: 
                    environ['$format'] = 'json' 
                    return self.json_app(environ, start_response) 
            decoded_query = parse_qs(environ['QUERY_STRING']) 
            if '$format' in decoded_query: 
                if decoded_query['$format'][0].lower() == 'json': 
                    environ['$format'] = 'json' 
                    return self.json_app(environ, start_response) 
            status = "{status.value}         {status.phrase}".format(status=HTTPStatus.BAD_REQUEST) 
            headers = [('Content-Type', 'text/plain;charset=utf-8')] 
            start_response(status, headers) 
            return ["Request doesn't include ?$format=json or Accept     header".encode('utf-8')] 

```

`__call__()`方法检查 Accept 标头以及查询字符串。如果 HTTP Accept 标头中的字符串`json`出现在任何位置，则调用给定的应用程序。环境将更新以包括此包装器使用的标头信息。

如果 HTTP Accept 标头不存在或不需要 JSON 响应，则会检查查询字符串。这种回退可能会有所帮助，因为很难更改浏览器发送的标头；使用查询字符串是 Accept 标头的浏览器友好替代方案。`parse_qs()`函数将查询字符串分解为键和值的字典。如果查询字符串中有`$format`作为键，则会检查其值是否包含`'json'`。如果是这样，则环境将使用查询字符串中找到的格式信息进行更新。

在这两种情况下，调用被包装的应用程序时会修改环境。被包装的函数只需要检查 WSGI 环境中的格式信息。这个包装器对象返回响应而不进行任何进一步的修改。

如果请求不要求 JSON，则会发送`400 BAD REQUEST`响应，并附带简单的文本消息。这将提供一些关于为什么查询不可接受的指导。

我们将使用`JSON_Filter`包装类定义如下：

```py
    json_wrapper = JSON_Filter(deal_cards) 
    httpd = make_server('', 8080, json_wrapper) 

```

我们没有从`deal_cards()`创建服务器，而是创建了一个引用`deal_cards()`函数的`JSON_Filter`类的实例。这将几乎与之前显示的版本完全相同。重要的区别是这需要一个 Accept 头或者一个像这样的 URL：`http://localhost:8080/?$format=json`。

### 提示

这个示例有一个微妙的语义问题。`GET`方法改变了服务器的状态。这通常是一个坏主意。

因为我们在浏览器中查看，很难解决问题。这里几乎没有可用的调试支持。这意味着`print()`函数以及日志消息对于调试是必不可少的。由于 WSGI 的工作方式，将打印到`sys.stderr`是必不可少的。使用 Flask 更容易，我们将在*使用 Flask 框架进行 RESTful API*的示例中展示。

HTTP 支持许多方法，包括`GET`，`POST`，`PUT`和`DELETE`。通常，将这些方法映射到数据库**CRUD**操作是明智的；使用`POST`进行创建，使用`GET`进行检索，使用`PUT`进行更新，使用`DELETE`进行删除。这意味着`GET`操作不会改变数据库的状态。

这导致了一个观点，即 Web 服务的`GET`操作应该是幂等的。一系列`GET`操作而没有其他`POST`，`PUT`或`DELETE`操作应该每次返回相同的结果。在这个示例中，每个`GET`都返回不同的结果。这是使用`GET`来处理卡片的一个语义问题。

对于我们演示基础知识的目的，这个区别是微不足道的。在一个更大更复杂的 Web 应用程序中，这个区别是一个重要的考虑因素。由于发牌服务不是幂等的，有一种观点认为它应该使用`POST`方法访问。

为了方便在浏览器中进行探索，我们避免检查 WSGI 应用程序中的方法。

## 另请参阅

+   Python 有各种各样的 RESTful API 框架。在*使用 Flask 框架进行 RESTful API*的示例中，我们将看一下 Flask 框架。

+   有三个地方可以查找有关 WSGI 标准的详细信息：

+   **PEP 3333**：请参阅[`www.python.org/dev/peps/pep-3333/`](https://www.python.org/dev/peps/pep-3333/)。

+   **Python 标准库**：它包括`wsgiref`模块。这是标准库中的参考实现。

+   **Werkzeug 项目**：请参阅[`werkzeug.pocoo.org`](http://werkzeug.pocoo.org)。这是一个具有众多 WSGI 实用程序的外部库。这被广泛用于实现适当的 WSGI 应用程序。

+   另请参阅[`docs.oasis-open.org/odata/odata-json-format/v4.0/odata-json-format-v4.0.html`](http://docs.oasis-open.org/odata/odata-json-format/v4.0/odata-json-format-v4.0.html)以获取有关为 Web 服务格式化数据的 JSON 的更多信息。

# 使用 Flask 框架进行 RESTful API

在*使用 WSGI 实现 Web 服务*的示例中，我们看到了如何使用 Python 标准库中可用的 WSGI 组件构建 RESTful API 和微服务。这导致了大量的编程来处理许多常见情况。

我们如何简化所有常见的 Web 应用程序编程并消除样板代码？

## 准备工作

首先，我们需要将 Flask 框架添加到我们的环境中。这通常依赖于使用`pip`安装 Flask 的最新版本以及其他相关项目，`itsdangerous`，`Jinja2`，`click`，`MarkupSafe`和`Werkzeug`。

安装看起来像下面这样：

```py
 **slott$ sudo pip3.5 install flask** 

 **Password:** 

 **Collecting flask** 

 **Downloading Flask-0.11.1-py2.py3-none-any.whl (80kB)** 

 **100% |████████████████████████████████| 81kB 3.6MB/s** 

 **Collecting itsdangerous>=0.21 (from flask)** 

 **Downloading itsdangerous-0.24.tar.gz (46kB)** 

 **100% |████████████████████████████████| 51kB 8.6MB/s** 

 **Requirement already satisfied (use --upgrade to upgrade): Jinja2>=2.4 in /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages (from flask)** 

 **Collecting click>=2.0 (from flask)** 

 **Downloading click-6.6.tar.gz (283kB)** 

 **100% |████████████████████████████████| 286kB 4.0MB/s** 

 **Collecting Werkzeug>=0.7 (from flask)** 

 **Downloading Werkzeug-0.11.10-py2.py3-none-any.whl (306kB)** 

 **100% |████████████████████████████████| 307kB 3.8MB/s** 

 **Requirement already satisfied (use --upgrade to upgrade): MarkupSafe in /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages (from Jinja2>=2.4->flask)** 

 **Installing collected packages: itsdangerous, click, Werkzeug, flask** 

 **Running setup.py install for itsdangerous ... done** 

 **Running setup.py install for click ... done** 

 **Successfully installed Werkzeug-0.11.10 click-6.6 flask-0.11.1 itsdangerous-0.24** 

```

我们可以看到`Jinja2`和`MarkupSafe`已经安装。缺少的元素被`pip`找到，下载并安装。Windows 用户不会使用`sudo`命令。

Flask 允许我们大大简化我们的网络服务应用程序。我们不需要创建一个大型且可能复杂的 WSGI 兼容函数或可调用对象，而是可以创建一个具有单独函数的模块。每个函数可以处理特定的 URL 路径模式。

我们将查看与*使用 WSGI 实现网络服务*食谱中相同的核心发牌功能。`Card`类定义了一个简单的扑克牌。`Deck`类定义了一副牌。

因为 Flask 为我们处理 URL 解析的细节，所以我们可以很容易地创建一个更复杂的网络服务。我们将定义一个路径，看起来像这样：

`/dealer/hand/?cards=5`。

这个路由有三个重要的信息：

+   路径的第一部分`/dealer/`是整个网络服务。

+   路径的下一部分`hand/`是一个特定的资源，一手牌。

+   查询字符串`?cards=5`定义了查询的 cards 参数。这是请求的手牌大小。这限制在 1 到 52 张牌的范围内。超出范围的值将得到`400`状态码，因为查询无效。

## 如何做...

1.  从`flask`包中导入一些核心定义。`Flask`类定义了整个应用程序。`request`对象保存当前的 web 请求：

```py
        from flask import Flask, request, jsonify, abort 
        from http import HTTPStatus 

```

`jsonify()`函数将从 Flask 视图函数返回一个 JSON 格式对象。`abort()`函数返回一个 HTTP 错误状态并结束请求的处理。

1.  导入底层类`Card`和`Deck`。理想情况下，这些应该从一个单独的模块中导入。应该可以在 web 服务环境之外测试所有功能：

```py
        from ch12_r01 import Card, Deck 

```

为了正确洗牌，我们还需要`random`模块：

```py
        import random 

```

1.  创建`Flask`对象。这是整个网络服务应用程序。我们将称 Flask 应用程序为`dealer`，并且还将将对象分配给全局变量`dealer`：

```py
        dealer = Flask('dealer') 

```

1.  创建应用程序中使用的任何对象。这些可以分配给`Flask`对象`dealer`作为属性。确保创建一个不会与 Flask 的内部属性冲突的唯一名称。另一种方法是使用模块全局变量。

有状态的全局对象必须能够在多线程环境中工作，或者必须显式禁用线程：

```py
        import os 
        random.seed(os.environ.get('DEAL_APP_SEED')) 
        deck = Deck() 

```

对于这个示例，`Deck`类的实现不是线程安全的，所以我们将依赖于单线程服务器。`deal()`方法应该使用`threading`模块中的`Lock`类来定义一个独占锁，以确保与并发线程的正确操作。

1.  定义一个路由-到执行特定请求的视图函数的 URL 模式。这是一个装饰器，直接放在函数的前面。它将把函数绑定到 Flask 应用程序：

```py
        @dealer.route('/dealer/hand/') 

```

1.  定义视图函数，检索数据或更新应用程序状态。在这个例子中，函数两者都做：

```py
        def deal(): 
            try: 
                hand_size = int(request.args.get('cards', 5)) 
                assert 1 <= hand_size < 53 
            except Exception as ex: 
                abort(HTTPStatus.BAD_REQUEST) 
            cards = deck.deal(hand_size) 
            response = jsonify([card.to_json() for card in cards]) 
            return response 

```

Flask 解析 URL 中`?`后面的字符串-查询字符串-以创建`request.args`值。客户端应用程序或浏览器可以使用查询字符串设置此值，例如`?cards=13`。这将为桥牌发牌 13 张牌。

如果查询字符串中的手牌大小值不合适，`abort()`函数将结束处理并返回`400`的 HTTP 状态码。这表示请求不可接受。这是一个最小的响应，没有更详细的内容。

应用程序的真正工作是一个简单的语句，`cards = dealer.deck.deal(hand_size)`。这里的想法是在 web 框架中包装现有功能。可以在没有 web 应用程序的情况下测试这些功能。

响应由`jsonify()`函数处理：这将创建一个响应对象。响应的主体将是以 JSON 表示的 Python 对象。如果我们需要向响应添加标头，我们可以更新`response.headers`以包含其他信息。

1.  定义运行服务器的主程序：

```py
        if __name__ == "__main__": 
            dealer.run(use_reloader=True, threaded=False, debug=True) 

```

我们包含了`debug=True`选项，以在浏览器和 Flask 日志文件中提供丰富的调试信息。服务器运行后，我们可以打开浏览器查看`http://localhost:5000/`。这将返回一批五张卡片。每次刷新，我们都会得到不同的一批卡片。

这是因为在浏览器中输入 URL 会执行一个带有最小一组标头的`GET`请求。由于我们的 WSGI 应用程序不需要任何特定的标头，并且响应所有的 HTTP 方法，它将返回一个结果。

结果是一个包含五张卡片的 JSON 文档。每张卡片由一个类名、`rank`和`suit`信息表示：

```py
    [ 
      { 
        "__class__": "Card", 
        "suit": "\u2663", 
        "rank": 6 
      }, 
      { 
        "__class__": "Card", 
        "suit": "\u2662", 
        "rank": 8 
      }, 
      { 
        "__class__": "Card", 
        "suit": "\u2660", 
        "rank": 8 
      }, 
      { 
        "__class__": "Card", 
        "suit": "\u2660", 
        "rank": 10 
      }, 
      { 
        "__class__": "Card", 
        "suit": "\u2663", 
        "rank": 11 
      } 
    ] 

```

要查看超过五张卡片，可以修改 URL。例如，这将返回一个桥牌手：`http://127.0.0.1:5000/dealer/hand/?cards=13`。

## 它是如何工作的...

Flask 应用程序由一个带有许多个别视图函数的应用程序对象组成。在这个食谱中，我们创建了一个单独的视图函数`deal()`。应用程序通常有许多函数。一个复杂的网站可能有许多应用程序，每个应用程序都有许多函数。

路由是 URL 模式和视图函数之间的映射。这使得可能有包含视图函数使用的参数的路由。

`@flask.route`装饰器是用于将每个路由和视图函数添加到整个 Flask 实例中的技术。视图函数根据路由模式绑定到整个应用程序中。

`Flask`对象的`run()`方法执行以下类型的处理。这并不完全是 Flask 的工作方式，但它提供了各种步骤的大致轮廓：

+   它等待 HTTP 请求。Flask 遵循 WSGI 标准，请求以字典的形式到达。有关 WSGI 的更多信息，请参阅*使用 WSGI 实现 Web 服务*食谱。

+   它从 WSGI 环境中创建一个 Flask`Request`对象。`request`对象包含来自请求的所有信息，包括所有 URL 元素、查询字符串元素和任何附加的文档。

+   Flask 然后检查各种路由，寻找与请求路径匹配的路由。

+   如果找到路由，则执行视图函数。该函数创建一个`Response`对象。这是视图函数的返回值。

+   如果找不到路由，则会自动发送`404 NOT FOUND`响应。

+   遵循 WSGI 模式准备状态和标头以开始发送响应。然后提供从视图函数返回的`Response`对象作为字节流。

Flask 应用程序可以包含许多方法，这使得提供 Web 服务非常容易。Flask 将其中一些方法公开为与请求或会话隐式绑定的独立函数。这使得编写视图函数稍微简单一些。

## 还有更多...

在*使用 WSGI 实现 Web 服务*食谱中，我们将应用程序包装在一个通用测试中，确认请求具有两个属性中的一个。我们使用了以下两条规则：

+   一个要求 JSON 的 Accept 标头

+   其中包含`$format=json`的查询字符串

如果我们正在编写一个复杂的 RESTful 应用程序服务器，我们通常希望对所有视图函数应用这种类型的测试。我们不想重复这个测试的代码。

当然，我们可以将*使用 WSGI 实现 Web 服务*食谱中的 WSGI 解决方案与 Flask 应用程序结合起来构建一个复合应用程序。我们也可以完全在 Flask 中完成这个任务。纯 Flask 解决方案比 WSGI 解决方案稍微简单一些，因此更受欢迎。

我们已经看到了 Flask 的`@flask.route`装饰器。Flask 还有许多其他装饰器，可以用来定义请求和响应处理中的各个阶段。为了对传入的请求应用测试，我们可以使用`@flask.before_request`装饰器。所有带有此装饰的函数将在处理请求之前被调用：

```py
    @dealer.before_request 
    def check_json(): 
        if 'json' in request.headers.get('Accept'): 
        return 
        if 'json' == request.args.get('$format'): 
            return 
        return abort(HTTPStatus.BAD_REQUEST) 

```

当`@flask.before_request`装饰器未能返回值（或返回`None`）时，处理将继续。路由将被检查，并且将评估视图函数。

在这个例子中，如果接受头包括`json`或者`$format`查询参数是`json`，那么函数返回`None`。这意味着正常的视图函数将被找到来处理请求。

当`@flask.before_request`装饰器返回一个值时，这就是最终结果，处理停止。在这个例子中，`check_json()`函数可能返回一个`abort()`响应，这将停止处理。`abort()`响应成为 Flask 应用程序的最终响应。这使得返回错误消息非常容易。

现在我们可以使用浏览器的地址栏输入以下 URL：

`http://127.0.0.1:5000/dealer/hand/?cards=13&$format=json`

这将返回一个 13 张牌的手，并且请求现在明确要求以 JSON 格式返回结果。尝试其他值作为`$format`以及完全省略`$format`键也是有益的。

### 提示

这个例子有一个微妙的语义问题。`GET`方法改变了服务器的状态。这通常是一个坏主意。

HTTP 支持一些与数据库 CRUD 操作相对应的方法。创建使用`POST`，检索使用`GET`，更新使用`PUT`，删除映射到`DELETE`。

这个想法导致了 Web 服务`GET`操作应该是幂等的想法。一系列`GET`操作——没有其他`POST`，`PUT`或`DELETE`——应该每次返回相同的结果。在这个例子中，每个`GET`都返回不同的结果。由于发牌服务不是幂等的，应该使用`POST`方法访问它。

为了方便使用浏览器进行探索，我们避免在 Flask 路由中检查方法。理想情况下，路由装饰器应该如下所示：

```py
    @dealer.route('/dealer/hand/', methods=['POST']) 

```

这样做会使得使用浏览器查看服务是否工作变得困难。在*使用 urllib 进行 REST 请求*中，我们将看到如何创建客户端，并切换到使用`POST`进行方法。

## 另见

+   有关 Web 服务的背景，请参阅*使用 WSGI 实现 Web 服务*。

+   有关 Flask 的详细信息，请参阅[`flask.pocoo.org/docs/0.11/`](http://flask.pocoo.org/docs/0.11/)。

+   请参阅[`www.packtpub.com/web-development/learning-flask-framework`](https://www.packtpub.com/web-development/learning-flask-framework)了解更多关于 Flask 框架的信息。另外，[`www.packtpub.com/web-development/mastering-flask`](https://www.packtpub.com/web-development/mastering-flask)有更多关于掌握 Flask 的信息。

# 解析请求中的查询字符串

URL 是一个复杂的对象。它至少包含六个单独的信息。可以通过可选元素包含更多信息。

例如`http://127.0.0.1:5000/dealer/hand/?cards=13&$format=json`的 URL 有几个字段：

+   `http`是方案。`https`用于使用加密套接字进行安全连接。

+   `127.0.0.1`可以称为授权，尽管网络位置更常用。这个特定的 IP 地址意味着本地主机，是本地主机的一种回环。本地主机的名称映射到这个 IP 地址。

+   `5000`是端口号，是授权的一部分。

+   `/dealer/hand/`是资源的路径。

+   `cards=13&$format=json`是一个查询字符串，它与路径由`?`字符分隔开。

查询字符串可能非常复杂。虽然不是官方标准，但查询字符串可能有重复的键。以下查询字符串是有效的，尽管可能令人困惑：

```py
    ?cards=13&cards=5 

```

我们重复了`cards`键。Web 服务将提供 13 张牌和 5 张牌。

[*作者不知道有任何手牌大小不同的纸牌游戏。缺乏一个好的用户故事使得这个例子有些牵强。*]

重复键的能力破坏了 URL 查询字符串和内置 Python 字典之间简单映射的可能性。这个问题有几种可能的解决方案：

+   字典中的每个键必须与包含所有值的`list`相关联。对于最常见的情况，即键不重复的情况，这很麻烦；每个列表只有一个项目。这个解决方案是通过`urllib.parse`中的`parse_qs()`实现的。

+   每个键只保存一次，第一个（或最后一个）值被保留，其他值被丢弃。这太糟糕了。

+   不使用字典。相反，查询字符串可以解析为*(键，值)*对的列表。这也允许键重复。对于具有唯一键的常见情况，列表可以转换为字典。对于不常见的情况，可以以其他方式处理重复的键。这是由`urllib.parse`中的`parse_qsl()`实现的。

有没有更好的方法来处理查询字符串？我们是否可以有一个更复杂的结构，行为类似于字典，对于常见情况具有单个值，并且对于罕见情况具有重复键和多个值的更复杂对象？

## 准备工作

Flask 依赖于另一个项目`Werkzeug`。当我们使用`pip`安装 Flask 时，要求将导致`pip`也安装 Werkzeug 工具包。Werkzeug 有一个数据结构，提供了处理查询字符串的绝佳方式。

我们将修改*使用 Flask 框架进行 RESTful API*配方中的示例，以使用更复杂的查询字符串。我们将添加一个第二个路由，用于发放多手牌。每手牌的大小将在允许重复键的查询字符串中指定。

## 如何做...

1.  从*使用 Flask 框架进行 RESTful API*配方开始。我们将向现有 Web 应用程序添加一个新的视图函数。

1.  定义一个路由——一个 URL 模式——到执行特定请求的视图函数。这是一个装饰器，直接放在函数前面。它将把函数绑定到 Flask 应用程序上：

```py
        @dealer.route('/dealer/hands/') 

```

1.  定义一个视图函数，响应发送到特定路由的请求：

```py
        def multi_hand(): 

```

1.  在视图函数中，使用`get()`方法提取唯一键的值，或者使用适用于内置 dict 类型的普通`[]`语法。这会返回单个值，而不会出现列表的复杂情况，其中列表只有一个元素的常见情况。

1.  对于重复的键，使用`getlist()`方法。这会将每个值作为列表返回。以下是一个查找查询字符串的视图函数，例如`?card=5&card=5`来发放两手五张牌：

```py
        try: 
            hand_sizes = request.args.getlist('cards', type=int) 
            if len(hand_sizes) == 0: 
                hand_sizes = [13,13,13,13] 
            assert all(1 <= hand_size < 53 for hand_size in hand_sizes) 
        except Exception as ex: 
            dealer.logger.exception(ex) 
            abort(HTTPStatus.BAD_REQUEST) 

        hands = [deck.deal(hand_size) for hand_size in hand_sizes] 
        response = jsonify( 
            [ 
                {'hand':i, 
                 'cards':[card.to_json() for card in hand] 
                } for i, hand in enumerate(hands) 
            ] 
        ) 
        return response 

```

这个函数将从查询字符串中获取所有`cards`键。如果值都是整数，并且每个值都在 1 到 52 的范围内（包括 1 和 52），那么这些值就是有效的，视图函数将返回一个结果。如果查询中没有`cards`键值，那么将发放 13 张牌的四手牌。

响应将是每手牌的 JSON 表示，作为一个小字典，有两个键：手牌 ID 和手牌上的牌。

1.  定义一个运行服务器的主程序：

```py
        if __name__ == "__main__": 
            dealer.run(use_reloader=True, threaded=False) 

```

服务器运行后，我们可以打开浏览器查看这个 URL：

`http://localhost:5000/?cards=5&cards=5&$format=json`

结果是一个 JSON 文档，其中有两手五张牌。我们省略了一些细节，以强调响应的结构：

```py
    [ 
      { 
        "cards": [ 
          { 
            "__class__": "Card", 
            "rank": 11, 
            "suit": "\u2660" 
          }, 
          { 
            "__class__": "Card", 
            "rank": 8, 
            "suit": "\u2662" 
          }, 
          ... 
        ], 
        "hand": 0 
      }, 
      { 
        "cards": [ 
          { 
            "__class__": "Card", 
            "rank": 3, 
            "suit": "\u2663" 
          }, 
          { 
            "__class__": "Card", 
            "rank": 9, 
            "suit": "\u2660" 
          }, 
          ... 
        ], 
        "hand": 1 
      } 
    ] 

```

因为 Web 服务解析查询字符串，向查询字符串添加更复杂的手牌大小是微不足道的。示例包括基于*使用 Flask 框架进行 RESTful API*配方的`$format=json`。

如果实现了`@dealer.before_request`函数`check_json()`来检查 JSON，那么就需要`$format`。如果未实现`@dealer.before_request`函数`check_json()`，那么查询字符串中的附加信息将被忽略。

## 它是如何工作的...

Werkzeug 的`Multidict`类是一个非常方便的数据结构。这是内置字典的扩展。它允许为给定的键有多个不同的值。

我们可以使用`collections`模块中的`defaultdict`类构建类似的东西。定义将是`defaultdict(list)`。这个定义的问题是每个键的值都是一个列表，即使列表只有一个项目作为值。

`Multidict`类提供的优势是`get()`方法的变体。当一个键有多个副本时，`get()`方法返回第一个值，或者当键只出现一次时返回唯一的值。这也有一个默认参数。这个方法与内置的`dict`类的方法相对应。

然而，`getlist()`方法返回给定键的所有值的列表。这种方法是`Multidict`类的独特方法。我们可以使用这种方法来解析更复杂的查询字符串。

用于验证查询字符串的常见技术是在验证时弹出项目。这是通过`pop()`和`poplist()`方法完成的。这些方法将从`Multidict`类中删除键。如果在检查所有有效键后仍然存在键，则这些额外的键可以被视为语法错误，并且 Web 请求将被拒绝并显示`abort(HTTPStatus.BAD_REQUEST)`。

## 还有更多...

查询字符串使用相对简单的语法规则。使用`=`作为键和值之间的标点符号的一个或多个键值对。每对之间的分隔符是`&`字符。由于其他字符在解析 URL 时的含义，还有一个重要的规则——键和值必须被编码。

URL 编码规则要求用 HTML 实体替换某些字符。这种技术称为百分比编码。这意味着当我们将`&`放入查询字符串的值中时，它必须被编码为`%26`，下面是一个显示这种编码的示例：

```py
 **>>> from urllib.parse import urlencode 
>>> urlencode( {'n':355,'d':113} ) 
'n=355&d=113' 
>>> urlencode( {'n':355,'d':113,'note':'this&that'} ) 
'n=355&d=113&note=this%26that'** 

```

值`this&that`被编码为`this%26that`。

有一小部分字符必须应用`%`编码规则。这来自*RFC 3986*，参见*第 2.2 节*，*保留字符*。列表包括这些字符：

```py
! * ' ( ) ; : @ & = + $ , / ? # [ ] % 

```

通常，与网页关联的 JavaScript 代码将处理编码查询字符串。如果我们在 Python 中编写 API 客户端，我们需要使用`urlencode()`函数来正确编码查询字符串。Flask 会自动处理解码。

查询字符串有一个实际的大小限制。例如，Apache HTTPD 有一个`LimitRequestLine`配置参数，默认值为`8190`。这将限制整个 URL 的大小。

在 OData 规范（[`docs.oasis-open.org/odata/odata/v4.0/`](http://docs.oasis-open.org/odata/odata/v4.0/)）中，建议查询选项使用几种类型的值。该规范建议我们的 Web 服务应支持以下类型的查询选项：

+   对于标识实体或实体集合的 URL，可以使用`$expand`和`$select`选项。扩展结果意味着查询将提供额外的细节。选择查询将对集合施加额外的条件。

+   标识集合的 URL 应支持`$filter`、`$search`、`$orderby`、`$count`、`$skip`和`$top`选项。这对于返回单个项目的 URL 没有意义。`$filter`和`$search`选项接受用于查找数据的复杂条件。`$orderby`选项定义了对结果施加的特定顺序。

`$count`选项从根本上改变了查询。它将返回项目的计数而不是项目本身。

`$top`和`$skip`选项用于浏览数据。如果计数很大，通常使用`$top`选项将结果限制为在网页上显示的特定数量。`$skip`选项的值确定将显示哪一页数据。例如，`$top=20$skip=40`将是结果的第 3 页-跳过 40 后的前 20 个。

通常，所有 URL 都应支持`$format`选项以指定结果的格式。我们一直专注于 JSON，但更复杂的服务可能提供 CSV 输出，甚至 XML。

## 另请参阅

+   请参阅*使用 Flask 框架进行 RESTful API*配方，了解如何使用 Flask 进行 Web 服务的基础知识。

+   在*使用 urllib 进行 REST 请求*配方中，我们将看看如何编写一个能够准备复杂查询字符串的客户端应用程序。

# 使用 urllib 进行 REST 请求

Web 应用程序有两个基本部分：

+   **客户端**：这可以是用户的浏览器，但也可能是移动设备应用程序。在某些情况下，Web 服务器可能是其他 Web 服务器的客户端。

+   **服务器**：这提供了我们一直在寻找的 Web 服务和资源，即*使用 WSGI 实现 Web 服务*，*使用 Flask 框架进行 RESTful API*和*解析请求中的查询字符串*配方，以及其他配方，如*解析 JSON 请求*和*为 Web 服务实现身份验证*。

基于浏览器的客户端通常是用 JavaScript 编写的。移动应用程序是用各种语言编写的，重点是 Android 设备的 Java 和 iOS 设备的 Objective-C 和 Swift。

有几个用户故事涉及用 Python 编写的 RESTful API 客户端。我们如何创建一个 Python 程序，作为 RESTful Web 服务的客户端？

## 准备就绪

我们假设我们有一个基于*使用 Flask 框架进行 RESTful API*或*解析请求中的查询字符串*配方的 Web 服务器。我们可以以以下方式为该服务器的行为编写正式规范：

```py
    { 
      "swagger": "2.0", 
      "info": { 
        "title": "dealer", 
        "version": "1.0" 
      }, 
      "schemes": ["http"], 
      "host": "127.0.0.1:5000", 
      "basePath": "/dealer", 
      "consumes": ["application/json"], 
      "produces": ["application/json"], 
      "paths": { 
        "/hands": { 
          "get": { 
            "parameters": [ 
              { 
                "name": "cards", 
                "in": "query", 
                "description": "number of cards in each hand", 
                "type": "array", 
                "items": {"type": "integer"}, 
                "collectionFormat": "multi", 
                "default": [13, 13, 13, 13] 
              } 
            ], 
            "responses": { 
              "200": { 
                "description":  
                "one hand of cards for each `hand` value in the query string" 
              } 
            } 
          } 
        }, 
        "/hand": { 
          "get": { 
            "parameters": [ 
              { 
                "name": "cards", 
                "in": "query", 
                "type": "integer", 
                "default": 5 
              } 
            ], 
            "responses": { 
              "200": { 
                "description":  
                "One hand of cards with a size given by the `hand` value in the query string" 
              } 
            } 
          } 
        } 
      } 
    } 

```

本文档为我们提供了如何使用 Python 的`urllib`模块来消耗这些服务的指导。它还描述了预期的响应应该是什么，为我们提供了如何处理这些响应的指导。

规范中的某些字段定义了基本 URL。特别是这三个字段提供了这些信息：

```py
      "schemes": ["http"], 
      "host": "127.0.0.1:5000", 
      "basePath": "/dealer", 

```

`produces`和`consumes`字段提供了帮助构建和验证 HTTP 标头的信息。请求的`Content-Type`标头必须是服务器消耗的**多用途互联网邮件扩展**（**MIME**）类型。同样，请求的 Accept 标头必须指定服务器生成的 MIME 类型。在这两种情况下，我们将提供`application/json`。

详细的服务定义在规范的`paths`部分中提供。例如，`/hands`路径显示了如何请求多个手的详细信息。路径详细信息是`basePath`值的后缀。

当 HTTP 方法为`get`时，参数是在查询中提供的。查询中的`cards`参数提供了一个整数卡的数量，并且可以多次重复。

响应将至少包括所描述的响应。在这种情况下，HTTP 状态将是`200`，响应的正文具有最少的描述。可以为响应提供更正式的模式定义，我们将在此示例中省略。

## 如何做...

1.  导入所需的`urllib`组件。我们将发出 URL 请求，并构建更复杂的对象，如查询字符串。我们将需要`urllib.request`和`urllib.parse`模块来实现这两个功能。由于预期的响应是 JSON 格式，因此`json`模块也将很有用：

```py
        import urllib.request 
        import urllib.parse 
        import json 

```

1.  定义将要使用的查询字符串。在这种情况下，所有值恰好是固定的。在更复杂的应用程序中，一些值可能是固定的，而另一些可能基于用户输入：

```py
        query = {'hand': 5} 

```

1.  使用查询构建完整 URL 的各个部分：

```py
        full_url = urllib.parse.ParseResult( 
            scheme="http", 
            netloc="127.0.0.1:5000", 
            path="/dealer" + "/hand/", 
            params=None, 
            query=urllib.parse.urlencode(query), 
            fragment=None 
        ) 

```

在这种情况下，我们使用`ParseResult`对象来保存 URL 的相关部分。这个类对于缺少的项目并不优雅，所以我们必须为 URL 的未使用部分提供明确的`None`值。

我们可以在脚本中使用`"http://127.0.0.1:5000/dealer/hand/?cards=5"`。然而，这种紧凑的字符串很难更改。在发出请求时，它作为一个紧凑的消息很有用，但不太适合制作灵活、可维护和可测试的程序。

使用这个长构造函数的优点是为 URL 的每个部分提供明确的值。在更复杂的应用程序中，这些单独的部分是从先前显示的 JSON Swagger 规范文档的分析中构建的：

1.  构建最终的`Request`实例。我们将使用从各种部分构建的 URL。我们将明确提供一个 HTTP 方法（浏览器通常使用`GET`作为默认值）。此外，我们可以提供明确的头部：

```py
        request = urllib.request.Request( 
            url = urllib.parse.urlunparse(full_url), 
            method = "GET", 
            headers = { 
                'Accept': 'application/json', 
            } 
        ) 

```

我们已经提供了 HTTP Accept 头部来声明服务器将产生的 MIME 类型结果，并被客户端接受。我们已经提供了 HTTP `Content-Type`头部来声明服务器消耗的请求，并由我们的客户端脚本提供。

1.  打开一个上下文来处理响应。`urlopen()`函数发出请求，处理 HTTP 协议的所有复杂性。最终的`result`对象可用于作为响应进行处理：

```py
        with urllib.request.urlopen(request) as response: 

```

1.  一般来说，响应的三个属性特别重要：

```py
        print(response.status) 
        print(response.headers) 
        print(json.loads(response.read().decode("utf-8"))) 

```

`status`是最终的状态码。我们期望一个正常请求的 HTTP 状态码为`200`。`headers`包括响应的所有头部。例如，我们可能想要检查`response.headers['Content-Type']`是否真的是`application/json`。

`response.read()`的值是从服务器下载的字节。我们经常需要解码这些字节以获得正确的 Unicode 字符。`utf-8`编码方案非常常见。我们可以使用`json.loads()`从 JSON 文档创建一个 Python 对象。

当我们运行这个时，我们会看到以下输出：

```py
 **200 
Content-Type: application/json 
Content-Length: 367 
Server: Werkzeug/0.11.10 Python/3.5.1 
Date: Sat, 23 Jul 2016 19:46:35 GMT 

[{'suit': '♠', 'rank': 4, '__class__': 'Card'}, 
 {'suit': '♡', 'rank': 4, '__class__': 'Card'}, 
 {'suit': '♣', 'rank': 9, '__class__': 'Card'}, 
 {'suit': '♠', 'rank': 1, '__class__': 'Card'}, 
 {'suit': '♠', 'rank': 2, '__class__': 'Card'}]** 

```

初始的`200`是状态，显示一切都正常工作。服务器提供了四个头部。最后，内部 Python 对象是一组小字典，提供了有关已发牌的卡片的信息。

要重建`Card`对象，我们需要使用一个稍微聪明的 JSON 解析器。参见第九章中的*读取 JSON 文档*配方，*输入/输出、物理格式和逻辑布局*。

## 它是如何工作的...

我们通过几个明确的步骤构建了请求：

1.  查询数据最初是一个简单的带有键和值的字典。

1.  `urlencode()`函数将查询数据转换为查询字符串，正确编码。

1.  整个 URL 最初作为`ParseResult`对象中的各个组件开始。这使得每个部分都是可见的，并且可以更改。对于这个特定的 API，这些部分基本上是固定的。在其他 API 中，URL 的路径和查询部分可能都具有动态值。

1.  整个请求是由 URL、方法和头部字典构建的。这个例子没有提供单独的文档作为请求的主体。如果发送复杂的文档，或者上传文件，也可以通过向`Request`对象提供详细信息来完成。

逐步组装对于简单的应用程序并不是必需的。在简单的情况下，URL 的字面字符串值可能是可以接受的。在另一个极端，一个更复杂的应用程序可能会打印出中间结果作为调试辅助，以确保请求被正确构造。

这样详细说明的另一个好处是提供一个方便的单元测试途径。有关更多信息，请参见第十一章 ，*测试*。我们经常可以将 Web 客户端分解为请求构建和请求处理。可以仔细测试请求构建，以确保所有元素都设置正确。请求处理可以使用不涉及与远程服务器的实时连接的虚拟结果进行测试。

## 还有更多...

用户身份验证通常是 Web 服务的重要组成部分。对于基于 HTML 的网站——强调用户交互——人们希望服务器能够理解通过会话的长时间运行的事务序列。用户将进行一次身份验证（通常使用用户名和密码），服务器将使用这些信息直到用户注销或会话过期。

对于 RESTful Web 服务，很少有会话的概念。每个请求都是单独处理的，服务器不需要维护复杂的长时间运行的事务状态。这个责任转移到了客户端应用程序。客户端需要进行适当的请求来构建一个可以呈现为单个事务的复杂文档。

对于 RESTful API，每个请求可能包括身份验证信息。我们将在*为 Web 服务实现身份验证*配方中详细讨论这一点。现在，我们将通过标题提供额外的细节。这将与我们的 RESTful 客户端脚本很好地契合。

有许多提供身份验证信息给 Web 服务器的方法：

+   一些服务使用 HTTP 的`Authorization`标题。当与基本机制一起使用时，客户端可以在每个请求中提供用户名和密码。

+   一些服务将发明一个全新的标题，名称为 API 密钥。该标题的值可能是一个复杂的字符串，其中包含有关请求者的编码信息。

+   一些服务将发明一个名为`X-Auth-Token`的标题。这可能在多步操作中使用，其中用户名和密码凭据作为初始请求的一部分发送。结果将包括一个字符串值（令牌），可用于后续 API 请求。通常，令牌具有短暂的过期时间，并且必须更新。

通常，这些方法需要**安全套接字层**（**SSL**）协议。这可以作为`https`方案使用。为了处理 SSL 协议，服务器（有时也是客户端）必须具有适当的证书。这些证书用作客户端和服务器之间的协商的一部分，以建立加密套接字对。

所有这些身份验证技术都有一个共同的特点——它们依赖于在标题中发送附加信息。它们在使用的标题和发送的信息方面略有不同。在最简单的情况下，我们可能会有以下内容：

```py
    request = urllib.request.Request( 
        url = urllib.parse.urlunparse(full_url), 
        method = "GET", 
        headers = { 
            'Accept': 'application/json', 
            'X-Authentication': 'seekrit password', 
        } 
    ) 

```

这个假设的请求将是针对需要在`X-Authentication`标题中提供密码的 Web 服务。在*为 Web 服务实现身份验证*配方中，我们将向 Web 服务器添加身份验证功能。

### OpenAPI（Swagger）规范

许多服务器将明确提供规范作为固定的标准 URL 路径`/swagger.json`的文件。OpenAPI 规范以前被称为**Swagger**，提供接口的文件名反映了这一历史。

如果提供，我们可以以以下方式获取网站的 OpenAPI 规范：

```py
    swagger_request = urllib.request.Request( 
        url = 'http://127.0.0.1:5000/dealer/swagger.json', 
        method = "GET", 
        headers = { 
            'Accept': 'application/json', 
        } 
    ) 

    from pprint import pprint 
    with urllib.request.urlopen(swagger_request) as response: 
        swagger = json.loads(response.read().decode("utf-8")) 
        pprint(swagger) 

```

一旦我们有了规范，我们可以使用它来获取服务或资源的详细信息。我们可以使用规范中的技术信息来构建 URL、查询字符串和标题。

### 将 Swagger 添加到服务器

对于我们的小型演示服务器，需要一个额外的视图函数来提供 OpenAPI Swagger 规范。我们可以更新`ch12_r03.py`模块以响应对`swagger.json`的请求。

有几种处理这些重要信息的方法：

1.  一个单独的静态文件。这就是这个配方中显示的内容。这是提供所需内容的一种非常简单的方式。

这是一个我们可以添加的视图函数，它将发送一个文件。当然，我们还需要将规范放入命名文件中：

```py
        from flask import send_file 
        @dealer.route('/dealer/swagger.json') 
        def swagger(): 
            response = send_file('swagger.json', mimetype='application/json') 
            return response 

```

这种方法的缺点是规范与实现模块分开。

1.  将规范嵌入模块中的大块文本。例如，我们可以将规范提供为模块本身的文档字符串。这提供了一个可见的地方来放置重要的文档，但这使得在模块级别包含文档字符串测试用例更加困难。

这个视图函数发送模块文档字符串，假设该字符串是一个有效的 JSON 文档：

```py
        from flask import make_response 
        @dealer.route('/dealer/swagger.json') 
        def swagger(): 
            response = make_response(__doc__.encode('utf-8')) 
            response.headers['Content-Type'] = 'application/json' 
            return response 

```

这种方法的缺点是需要检查文档字符串的语法以确保其是有效的 JSON。这除了验证模块实现实际上是否符合规范之外。

1.  在适当的 Python 语法中创建一个 Python 规范对象。然后可以将其编码为 JSON 并传输。这个视图函数发送一个 `specification` 对象。这将是一个有效的 Python 对象，可以序列化为 JSON 表示法：

```py
        from flask import make_response 
        import json 
        @dealer.route('/dealer/swagger.json') 
        def swagger3(): 
            response = make_response( 
                json.dumps(specification, indent=2).encode('utf-8')) 
            response.headers['Content-Type'] = 'application/json' 
            return response 

```

在所有情况下，拥有正式规范可用有几个好处：

1.  客户端应用程序可以下载规范以微调其处理。

1.  当包含示例时，规范成为客户端和服务器的一系列测试用例。

1.  规范的各种细节也可以被服务器应用程序用来提供验证规则、默认值和其他细节。

## 另见

+   *解析请求中的查询字符串* 配方介绍了核心 Web 服务

+   *为 Web 服务实现身份验证* 配方将添加身份验证以使服务更安全

# 解析 URL 路径

URL 是一个复杂的对象。它至少包含六个单独的信息片段。可以包括更多作为可选值。

诸如 `http://127.0.0.1:5000/dealer/hand/player_1?$format=json` 的 URL 具有几个字段：

+   `http` 是方案。`https` 用于使用加密套接字进行安全连接。

+   `127.0.0.1` 可以称为权限，尽管网络位置更常用。这个特定的 IP 地址意味着本地主机，是一种回环到本地主机的方式。localhost 映射到这个 IP 地址。

+   `5000` 是端口号，是权限的一部分。

+   /dealer/hand/player_1 是资源的路径。

+   `$format=json` 是一个查询字符串。

资源的路径可能非常复杂。在 RESTful Web 服务中，使用路径信息来标识资源组、单个资源甚至资源之间的关系是很常见的。

我们如何处理复杂的路径解析？

## 准备就绪

大多数 Web 服务提供对某种资源的访问。在*使用 Flask 框架实现 RESTful API* 和*解析请求中的查询字符串* 配方中，资源在 URL 路径上被标识为手或手。这在某种程度上是误导性的。

实际上，这些 Web 服务涉及两个资源：

+   一副牌，可以洗牌以产生一个或多个随机手

+   一只手，被视为对请求的瞬态响应

更让事情变得更加混乱的是，手资源是通过 `GET` 请求而不是更常见的 `POST` 请求创建的。这很令人困惑，因为不会预期 `GET` 请求改变服务器的状态。

对于简单的探索和技术尖刺，`GET` 请求是有帮助的。因为浏览器可以发出 `GET` 请求，这是探索 Web 服务设计某些方面的好方法。

重新设计可以提供对 `Deck` 类的随机实例的显式访问。牌组的一个特性将是牌的手。这与将 `Deck` 视为集合和 `Hands` 作为集合内资源的想法相一致：

+   `/dealer/decks`：`POST`请求将创建一个新的牌组对象。对这个请求的响应被用来标识唯一的牌组。

+   `/dealer/deck/{id}/hands`：对此的`GET`请求将从给定的牌组标识符获取一个手牌对象。查询字符串将指定多少张牌。查询字符串可以使用`$top`选项来限制返回多少手牌。它还可以使用`$skip`选项跳过一些手牌，并获取以后的手牌的牌。

这些查询将需要一个 API 客户端。它们不能轻松地从浏览器中完成。一个可能的方法是使用 Postman 作为 Chrome 浏览器的插件。我们将利用*使用 urllib 进行 REST 请求*的方法作为处理这些更复杂 API 的客户端的起点。

## 如何做...

我们将把这分解成两部分：服务器和客户端。

### 服务器

1.  从*解析请求中的查询字符串*的模板开始，作为 Flask 应用程序的模板。我们将改变那个例子中的视图函数：

```py
        from flask import Flask, jsonify, request, abort, make_response 
        from http import HTTPStatus 
        dealer = Flask('dealer') 

```

1.  导入任何额外的模块。在这种情况下，我们将使用`uuid`模块为洗牌后的牌组创建一个唯一的键：

```py
        import uuid 

```

我们还将使用 Werkzeug 的`BadRequest`响应。这使我们能够提供详细的错误消息。这比对于错误请求使用`abort(400)`要好一点：

```py
        from werkzeug.exceptions import BadRequest 

```

1.  定义全局状态。这包括牌组的集合。它还包括随机数生成器。为了测试目的，有一种方法可以强制使用特定的种子值：

```py
        import os 
        import random 
        random.seed(os.environ.get('DEAL_APP_SEED')) 
        decks = {} 

```

1.  定义一个路由——到执行特定请求的视图函数的 URL 模式。这是一个装饰器，直接放在函数的前面。它将把函数绑定到 Flask 应用程序：

```py
        @dealer.route('/dealer/decks', methods=['POST']) 

```

我们已经定义了牌组资源，并将路由限制为只处理`HTTP POST`请求。这缩小了这个特定端点的语义——`POST`请求通常意味着 URL 将在服务器上创建新的东西。在这个例子中，它在牌组集合中创建了一个新实例。

1.  定义支持这个资源的视图函数：

```py
        def make_deck(): 
            id = str(uuid.uuid1()) 
            decks[id]= Deck() 
            response_json = jsonify( 
                status='ok', 
                id=id 
            ) 
            response = make_response(response_json, HTTPStatus.CREATED) 
            return response 

```

`uuid1()`函数将基于当前主机和随机种子序列生成器创建一个通用唯一 ID。这个字符串版本是一个长的十六进制字符串，看起来像`93b8fc06-5395-11e6-9e73-38c9861bf556`。

我们将使用这个字符串作为创建`Deck`的新实例的键。响应将是一个带有两个字段的小 JSON 文档：

+   `status`字段将是`'ok'`，因为一切都正常。这使我们可以提供其他包括警告或错误的状态信息。

+   `id`字段具有刚刚创建的牌组的 ID 字符串。这允许服务器拥有多个并发游戏，每个游戏都由一个牌组 ID 区分。

响应是使用`make_response()`函数创建的，这样我们就可以提供`201 CREATED`的 HTTP 状态，而不是默认的`200 OK`。这种区别很重要，因为这个请求改变了服务器的状态。

1.  定义一个需要参数的路由。在这种情况下，路由将包括要处理的特定牌组 ID：

```py
        @dealer.route('/dealer/decks/<id>/hands', methods=['GET']) 

```

`<id>`使这成为一个路径模板，而不是一个简单的文字路径。Flask 将解析`/`字符并分隔`<id>`字段。

1.  定义一个视图函数，其参数与模板匹配。由于模板包含`<id>`，视图函数也有一个名为`id`的参数：

```py
        def get_hands(id): 
            if id not in decks: 
                dealer.logger.debug(id) 
                return make_response( 
                    'ID {} not found'.format(id), HTTPStatus.NOT_FOUND) 
            try: 
                cards = int(request.args.get('cards',13)) 
                top = int(request.args.get('$top',1)) 
                skip = int(request.args.get('$skip',0)) 
                assert skip*cards+top*cards <= len(decks[id].cards), \ 
                    "$skip, $top, and cards larger than the deck" 
            except ValueError as ex: 
                return BadRequest(repr(ex)) 
            subset = decks[id].cards[skip*cards:(skip+top)*cards] 
            hands = [subset[h*cards:(h+1)*cards] for h in range(top)] 
            response = jsonify( 
                [ 
                    {'hand':i, 'cards':[card.to_json() for card in hand]} 
                     for i, hand in enumerate(hands) 
                ] 
            ) 
            return response 

```

如果`id`参数的值不是牌组集合的键之一，函数将生成`404 NOT FOUND`响应。这个函数使用`BadRequest`而不是`abort()`函数，以包括解释性的错误消息。我们也可以在 Flask 中使用`make_response()`函数。

此函数还从查询字符串中提取`$top`、`$skip`和`cards`的值。在此示例中，所有值都恰好是整数，因此对每个值使用`int()`函数。对查询参数执行了一个基本的合理性检查。实际上需要进行额外的检查，鼓励读者思考可能使用的所有可能的不良参数。

`subset`变量是正在发牌的牌组部分。我们已经对牌组进行了切片，以在`skip`组`cards`后开始；我们在这个切片中只包括`top`组`cards`。从该切片中，`hands`序列将子集分解为`top`数量的手牌，每个手牌中都有`cards`。通过`jsonify()`函数将此序列转换为 JSON，并返回。

默认状态是`200 OK`，这是合适的，因为此查询是幂等的`GET`请求。每次发送查询时，将返回相同的一组牌。

1.  定义一个运行服务器的主程序：

```py
        if __name__ == "__main__": 
            dealer.run(use_reloader=True, threaded=False) 

```

### 客户端

这将类似于*使用 urllib 进行 REST 请求*食谱中的客户端模块：

1.  导入用于处理 RESTful API 的基本模块：

```py
        import urllib.request 
        import urllib.parse 
        import json 

```

1.  有一系列步骤来进行`POST`请求，以创建一个新的洗牌牌组。首先通过手动创建`ParseResult`对象来定义 URL 的各个部分。稍后将将其合并为单个字符串：

```py
        full_url = urllib.parse.ParseResult( 
            scheme="http", 
            netloc="127.0.0.1:5000", 
            path="/dealer" + "/decks", 
            params=None, 
            query=None, 
            fragment=None 
        ) 

```

1.  从 URL、方法和标头构建`Request`对象：

```py
        request = urllib.request.Request( 
            url = urllib.parse.urlunparse(full_url), 
            method = "POST", 
            headers = { 
                'Accept': 'application/json', 
            } 
        ) 

```

默认方法是`GET`，这对于此 API 请求是不合适的。

1.  发送请求并处理响应对象。出于调试目的，打印状态和标头信息可能会有所帮助。通常，我们只需要确保状态是预期的`201`。

响应文档应该是 Python 字典的 JSON 序列化，具有两个字段，状态和 ID。此客户端在使用`id`字段中的值之前确认响应中的状态为`ok`：

```py
        with urllib.request.urlopen(request) as response: 
            # print(response.status) 
            assert response.status == 201 
            # print(response.headers) 
            document = json.loads(response.read().decode("utf-8")) 

        print(document) 
        assert document['status'] == 'ok' 
        id = document['id'] 

```

在许多 RESTful API 中，将会有一个位置标头，它提供了一个链接到创建的对象的 URL。

1.  创建一个 URL，其中包括将 ID 插入 URL 路径以及提供一些查询字符串参数。这是通过创建一个模拟查询字符串的字典，然后使用`ParseResult`对象构建 URL 来完成的：

```py
        query = {'$top': 4, 'cards': 13} 

        full_url = urllib.parse.ParseResult( 
            scheme="http", 
            netloc="127.0.0.1:5000", 
            path="/dealer" + "/decks/{id}/hands".format(id=id), 
            params=None, 
            query=urllib.parse.urlencode(query), 
            fragment=None 
        ) 

```

我们使用`"/decks/{id}/hands/".format(id=id)`将`id`值插入路径。另一种方法是使用`"/".join(["", "decks", id, "hands", ""])`。请注意，空字符串是强制`"/"`出现在开头和结尾的一种方法。

1.  使用完整 URL、方法和标准标头创建`Request`对象：

```py
        request = urllib.request.Request( 
            url = urllib.parse.urlunparse(full_url), 
            method = "GET", 
            headers = { 
                'Accept': 'application/json', 
            } 
        ) 

```

1.  发送请求并处理响应。我们将确认响应为`200 OK`。然后可以解析响应以获取所请求手牌的详细信息：

```py
        with urllib.request.urlopen(request) as response: 
            # print(response.status) 
            assert response.status == 200 
            # print(response.headers) 
            cards = json.loads(response.read().decode("utf-8")) 

        print(cards) 

```

当我们运行此代码时，它将创建一个新的`Deck`实例。然后它将发出四手牌，每手 13 张牌。查询定义了每手的确切数量和每手中的牌数。

## 工作原理...

服务器定义了遵循集合和集合实例的常见模式的两个路由。通常使用复数名词`decks`来定义集合路径。使用复数名词意味着 CRUD 操作侧重于在集合内创建实例。

在这种情况下，使用`POST`方法的`/dealer/decks`路径实现了创建操作。通过编写一个额外的视图函数来处理`/dealer/decks`路径的`GET`方法，可以支持检索。这将公开牌组集合中的所有牌组实例。

如果支持删除，可以使用`DELETE`方法的`/dealer/decks`。更新（使用`PUT`方法）似乎不符合创建随机牌组的服务器的想法。

在 `/dealer/decks` 集合中，特定的牌堆由 `/dealer/decks/<id>` 路径标识。设计要求使用 `GET` 方法从给定的牌堆中获取几手牌。

剩下的 CRUD 操作——创建、更新和删除——对于这种类型的 `Deck` 对象并没有太多意义。一旦创建了 `Deck` 对象，客户端应用程序就可以查询各种手牌。

### 牌堆切片

发牌算法对一副牌进行了几次切片。这些切片是基于一副牌的大小，*D* ，必须包含足够的牌来满足手数，*h* ，以及每手的牌数，*c* 。手数和每手的牌数必须不大于牌的大小：

*h* × *c* ≤ *D*

发牌的社交仪式通常涉及切牌，这是由非发牌玩家进行的非常简单的洗牌。传统上，每隔 *h* 张牌分配给每个手牌 *H* [n] ：

*H[n] =* { *D[n]* [+] *[h]* [×] *[i]*  :0 ≤ *i* < *c* }

在前面的公式中，*H[n=0]*  手中有牌 *H[0] = { D[0] , D[h] , D[2h] , ..., D[c×h ] }* ，*H[n=1]*  手中有牌 *H[1] = { D[1] , D[1+h] , D[1+2h] , ..., D[1+c×h] }* ，依此类推。这种发牌方式看起来比简单地将每个玩家的下一批 *c* 张牌发给他们更公平。

这并不是真正必要的，我们的 Python 程序以稍微更容易用 Python 计算的批次发牌：

*H[n]* = { *D* [*n* × *c* +1] : 0 ≤ *i* < *c* }

Python 代码创建了手 *H[n=0]*  ，其中有牌 *H* [0] *=* { *D* [0] *, D* [1] *, D* [2] *, ..., D[c-]* [1] }，手 *H[n=1]*  有牌 *H* [0 ] *=* { *D[c] , D[c+]* [1] *, D[c+]* [2] *, ..., D* [2c- *1*] }，依此类推。对于一副随机的牌，这与任何其他分配牌的方式一样公平。在 Python 中，这稍微简单一些，因为它涉及到列表切片。有关切片的更多信息，请参阅第四章中的 *切片和切块列表* 配方，*内置数据结构 – 列表、集合、字典* 。

### 客户端

这个交易的客户端是一系列 RESTful 请求：

1.  理想情况下，操作从 `GET` 到 `swagger.json` 开始，以获取服务器的规范。根据服务器的不同，这可能会很简单：

```py
        with urllib.request.urlopen('http://127.0.0.1:5000/dealer/swagger.json') as         response 
            swagger = json.loads(response.read().decode("utf-8")) 

```

1.  然后，有一个 `POST` 来创建一个新的 `Deck` 实例。这需要创建一个 `Request` 对象，以便可以将方法设置为 `POST` 。

1.  然后，有一个 `GET` 来从牌堆实例中获取一些手牌。这可以通过调整 URL 作为字符串模板来完成。将 URL 作为一组单独字段而不是一个简单的字符串进行处理，稍微更一般化。

有两种处理 RESTful 应用程序错误的方法：

+   对于未找到的资源，使用 `abort(HTTPStatus.NOT_FOUND)` 等简单的状态响应。

+   对于某种方式无效的请求，使用 `make_response(message, HTTPStatus.BAD_REQUEST)` 。消息可以提供所需的详细信息。

对于一些其他状态码，比如 `403 Forbidden` ，我们可能不想提供太多细节。在授权问题的情况下，提供太多细节通常是一个坏主意。对于这种情况，`abort(HTTPStatus.FORBIDDEN)` 可能是合适的。

## 还有更多...

我们将看一些应该考虑添加到服务器的功能：

+   在接受标头中检查 `JSON`

+   提供 Swagger 规范

使用标头来区分 RESTful API 请求和对服务器的其他请求是很常见的。接受标头可以提供一个 MIME 类型，用于区分对 JSON 内容的请求和对面向用户的内容的请求。

`@dealer.before_request` 装饰器可用于注入一个过滤每个请求的函数。这个过滤器可以根据以下要求区分适当的 RESTful API 请求：

+   接受标头包括一个包含 `json` 的 MIME 类型。通常，完整的 MIME 字符串是 `application/json` 。

+   此外，我们可以为`swagger.json`文件做一个例外。这可以被视为一个 RESTful API 请求，而不考虑任何其他指示。

这是实现这一点的额外代码：

```py
    @dealer.before_request 
    def check_json(): 
        if request.path == '/dealer/swagger.json': 
            return 
        if 'json' in request.headers.get('Accept', '*/*'): 
            return 
        return abort(HTTPStatus.BAD_REQUEST) 

```

这个过滤器将简单地返回一个不详细的`400 BAD REQUEST`响应。提供更明确的错误消息可能会泄露关于服务器实现的太多信息。然而，如果有帮助的话，我们可以用`make_response()`替换`abort()`来返回更详细的错误。

### 提供 Swagger 规范

一个行为良好的 RESTful API 为各种可用的服务提供了 OpenAPI 规范。这通常打包在`/swagger.json`路由中。这并不一定意味着有一个字面上的文件可用。相反，这个路径被用作一个重点，以提供遵循 Swagger 2.0 规范的详细接口规范的 JSON 表示。

我们已经定义了路由`/swagger.json`，并将函数`swagger3()`绑定到这个路由。这个函数将创建一个全局对象`specification`的 JSON 表示：

```py
    @dealer.route('/dealer/swagger.json') 
    def swagger3(): 
        response = make_response(json.dumps(specification, indent=2).encode('utf-8')) 
        response.headers['Content-Type'] = 'application/json' 
        return response 

```

`specification`对象的大纲如下。重要细节已被替换为`...`以强调整体结构。细节如下：

```py
    specification = { 
        'swagger': '2.0', 
        'info': { 
            'title': '''Python Cookbook\nChapter 12, recipe 5.''', 
            'version': '1.0' 
        }, 
        'schemes': ['http'], 
        'host': '127.0.0.1:5000', 
        'basePath': '/dealer', 
        'consumes': ['application/json'], 
        'produces': ['application/json'], 
        'paths': { 
            '/decks': {...} 
            '/decks/{id}/hands': {...} 
        } 
    } 

```

这两个路径对应于服务器中的两个`@dealer.route`装饰器。这就是为什么通常有助于从 Swagger 规范开始设计服务器，然后构建代码以满足规范。

注意小的语法差异。Flask 使用`/decks/<id>/hands`，而 OpenAPI Swagger 规范使用`/decks/{id}/hands`。这一小细节意味着我们不能在 Python 和 Swagger 文档之间轻松地复制和粘贴。

这是`/decks`路径。这显示了来自查询字符串的输入参数。它还显示了包含牌组 ID 信息的`201`响应的细节：

```py
    '/decks': { 
     'post': { 
        'parameters': [ 
          { 
            'name': 'size', 
            'in': 'query', 
            'type': 'integer', 
            'default': 1, 
                'description': '''number of decks to build and shuffle''' 
          } 
        ], 
        'responses': { 
          '201': { 
            'description': '''Create and shuffle a deck. Returns a unique deck id.''', 
            'schema': { 
              'type': 'object', 
                'properties': { 
                  'status': {'type': 'string'}, 
                  'id': {'type': 'string'} 
                } 
              } 
            }, 
          '400': { 
            'description': '''Request doesn't accept a JSON response''' 
          } 
        } 
      } 

```

`/decks/{id}/hands`路径具有类似的结构。它定义了查询字符串中可用的所有参数。它还定义了各种响应；一个包含卡片的`200`响应，并在未找到 ID 值时定义了`404`响应。

我们省略了每个路径的参数的一些细节。我们还省略了关于牌组结构的细节。然而，大纲总结了 RESTful API：

+   `swagger`键必须设置为`2.0`。

+   `info`键可以提供大量信息。这个例子只有最低要求。

+   `schemes`、`host`和`basePath`字段定义了此服务使用的 URL 的一些常见元素。

+   `consumes`字段说明了请求的`Content-Type`应该包括什么。

+   `produces`字段同时说明了请求的 Accept 头必须说明什么，以及响应的`Content-Type`将是什么。

+   `paths`字段标识了在此服务器上提供响应的所有路径。这显示了`/decks`和`/decks/{id}/hands`路径。

`swagger3()`函数将这个 Python 对象转换为 JSON 表示法并返回它。这实现了似乎是下载`swagger.json`文件的功能。内容指定了 RESTful API 服务器提供的资源。

### 使用 Swagger 规范

在客户端编程中，我们使用简单的字面值来构建 URL。示例看起来像下面这样：

```py
    full_url = urllib.parse.ParseResult( 
        scheme="http", 
        netloc="127.0.0.1:5000", 
        path="/dealer" + "/decks", 
        params=None, 
        query=None, 
        fragment=None 
    ) 

```

其中的一部分可以来自 Swagger 规范。例如，我们可以使用`specification['host']`和`specification['basePath']`来代替`netloc`值和`path`值的第一部分。这种对 Swagger 规范的使用可以提供一点额外的灵活性。

Swagger 规范是为了供人们用于做设计决策的工具消费而设计的。其真正目的是驱动 API 的自动化测试。通常，Swagger 规范会包含详细的示例，可以帮助澄清如何编写客户端应用程序。

## 另请参阅

+   有关更多 RESTful web 服务示例，请参阅*使用 urllib 进行 REST 请求*和*解析请求中的查询字符串*配方

# 解析 JSON 请求

许多 web 服务涉及请求创建新的持久对象或对现有持久对象进行更新。为了执行这些操作，应用程序将需要来自客户端的输入。

RESTful web 服务通常会接受 JSON 文档形式的输入（和产生输出）。有关 JSON 的更多信息，请参阅第九章中的*阅读 JSON 文档*配方，*输入/输出、物理格式和逻辑布局*

我们如何解析来自 web 客户端的 JSON 输入？验证输入的简单方法是什么？

## 准备工作

我们将扩展 Flask 应用程序，从“解析请求中的查询字符串”配方中添加用户注册功能；这将添加一个玩家，然后玩家可以请求卡片。玩家是一个资源，将涉及基本的 CRUD 操作：

+   客户端可以对`/players`路径执行`POST`以创建新玩家。这将包括描述玩家的文档有效负载。服务将验证文档，如果有效，创建一个新的持久`Player`实例。响应将包括分配给玩家的 ID。如果文档无效，将发送响应详细说明问题。

+   客户端可以对`/players`路径执行`GET`以获取玩家列表。

+   客户端可以对`/players/<id>`路径执行`GET`以获取特定玩家的详细信息。

+   客户端可以对`/players/<id>`路径执行`PUT`以更新特定玩家的详细信息。与初始的`POST`一样，这需要验证有效负载文档。

+   客户端可以对`/players/<id>`路径执行`DELETE`以删除玩家。

与“解析请求中的查询字符串”配方一样，我们将实现这些服务的客户端和服务器部分。服务器将处理基本的`POST`和`GET`操作。我们将把`PUT`和`DELETE`操作留给读者作为练习。

我们需要一个 JSON 验证器。请参阅[`pypi.python.org/pypi/jsonschema/2.5.1`](https://pypi.python.org/pypi/jsonschema/2.5.1)。这特别好。还有一个 Swagger 规范验证器也很有帮助。请参阅[`pypi.python.org/pypi/swagger-spec-validator`](https://pypi.python.org/pypi/swagger-spec-validator)。

如果我们安装`swagger-spec-validator`包，这也会安装`jsonschema`项目的最新副本。整个序列可能如下所示：

```py
 **MacBookPro-SLott:pyweb slott$ pip3.5 install swagger-spec-validator** 

 **Collecting swagger-spec-validator** 

 **Downloading swagger_spec_validator-2.0.2.tar.gz** 

 **Requirement already satisfied (use --upgrade to upgrade):** 

 **jsonschema in /Library/.../python3.5/site-packages** 

 **(from swagger-spec-validator)** 

 **Requirement already satisfied (use --upgrade to upgrade):** 

 **setuptools in /Library/.../python3.5/site-packages** 

 **(from swagger-spec-validator)** 

 **Requirement already satisfied (use --upgrade to upgrade):** 

 **six in /Library/.../python3.5/site-packages** 

 **(from swagger-spec-validator)** 

 **Installing collected packages: swagger-spec-validator** 

 **Running setup.py install for swagger-spec-validator ... done** 

 **Successfully installed swagger-spec-validator-2.0.2** 

```

我们使用`pip`命令安装了`swagger-spec-validator`包。此安装还检查了`jsonschema`，`setuptools`和`six`是否已安装。

有一个关于使用`--upgrade`的提示。使用类似以下命令升级包可能有所帮助：`pip install jsonschema --upgrade`。如果`jsonschema`的版本低于 2.5.0，则可能需要这样做。

## 如何做...

我们将这分解为三部分：Swagger 规范、服务器和客户端。

### Swagger 规范

1.  以下是 Swagger 规范的概要：

```py
        specification = { 
            'swagger': '2.0', 
            'info': { 
                'title': '''Python Cookbook\nChapter 12, recipe 6.''', 
                'version': '1.0' 
            }, 
            'schemes': ['http'], 
            'host': '127.0.0.1:5000', 
            'basePath': '/dealer', 
            'consumes': ['application/json'], 
            'produces': ['application/json'], 
            'paths': { 
                '/players': {...}, 
                '/players/{id}': {...}, 
            } 
            'definitions': { 
                'player: {..} 
            } 
        } 

```

首先的字段是 RESTful web 服务的基本样板。`paths`和`definitions`将填入服务的 URL 和模式定义。

1.  以下是用于验证新玩家的模式定义。这将放在整体规范的定义中：

```py
        'player': { 
            'type': 'object', 
            'properties': { 
                'name': {'type': 'string'}, 
                'email': {'type': 'string', 'format': 'email'}, 
                'year': {'type': 'integer'}, 
                'twitter': {'type': 'string', 'format': 'uri'} 
            } 
        } 

```

整体输入文档正式描述为对象类型。该对象有四个属性：

+   一个名字，这是一个字符串

+   一个电子邮件地址，这是一个特定格式的字符串

+   Twitter URL，这是一个特定格式的字符串

+   一年，这是一个数字

JSON 模式规范语言中有一些定义的格式。`email`和`url`格式被广泛使用。格式的完整列表包括`date-time`、`hostname`、`ipv4`、`ipv6`和`uri`。有关定义模式的详细信息，请参见[`json-schema.org/documentation.html`](http://json-schema.org/documentation.html)。

1.  这是用于创建新玩家或获取所有玩家集合的整体`players`路径：

```py
        '/players': { 
            'post': { 
                'parameters': [ 
                        { 
                            'name': 'player', 
                            'in': 'body', 
                            'schema': {'$ref': '#/definitions/player'} 
                        }, 
                    ], 
                'responses': { 
                    '201': {'description': 'Player created', }, 
                    '403': {'description': 'Player is invalid or a duplicate'} 
                } 
            }, 
            'get': { 
                'responses': { 
                    '200': {'description': 'All of the players defined so far'}, 
                } 
            } 
        }, 

```

该路径定义了两种方法——`post`和`get`。`post`方法有一个名为`player`的参数。这个参数是请求的主体，并且遵循定义部分提供的玩家模式。

`get`方法显示了没有任何参数或响应结构的正式定义。

1.  这是一个用于获取有关特定玩家的详细信息的路径的定义：

```py
        '/players/{id}': { 
            'get': { 
                'parameters': [ 
                    { 
                        'name': 'id', 
                        'in': 'path', 
                        'type': 'string' 
                    } 
                ], 
                'responses': { 
                    '200': { 
                        'description': 'The details of a specific player', 
                        'schema': {'$ref': '#/definitions/player'} 
                    }, 
                    '404': {'description': 'Player ID not found'} 
                } 
            } 
        }, 

```

该路径类似于*解析 URL 路径*配方中所示的路径。URL 中提供了`player`键。显示了当玩家 ID 有效时的响应细节。响应具有一个定义的模式，该模式还使用了定义部分中的玩家模式定义。

这个规范将成为服务器的一部分。它可以由在`@dealer.route('/swagger.json')`路由中定义的视图函数提供。通常最简单的方法是创建一个包含这个规范文档的文件。

### 服务器

1.  以*解析请求中的查询字符串*配方作为 Flask 应用程序的模板开始。我们将改变视图函数：

```py
        from flask import Flask, jsonify, request, abort, make_response 
        from http import HTTPStatus 

```

1.  导入所需的额外库。我们将使用 JSON 模式进行验证。我们还将计算字符串的哈希值，以作为 URL 中有用的外部标识符：

```py
        from jsonschema import validate 
        from jsonschema.exceptions import ValidationError 
        import hashlib 

```

1.  创建应用程序和玩家数据库。我们将使用一个简单的全局变量。一个更大的应用程序可能会使用一个适当的数据库服务器来保存这些信息：

```py
        dealer = Flask('dealer') 
        players = {} 

```

1.  定义用于发布到整体`players`集合的路由：

```py
        @dealer.route('/dealer/players', methods=['POST']) 

```

1.  定义将解析输入文档、验证内容，然后创建持久`player`对象的函数：

```py
        def make_player(): 
            document = request.json 
            player_schema = specification['definitions']['player'] 
            try: 
                validate(document, player_schema) 
            except ValidationError as ex: 
                return make_response(ex.message, 403) 

            id = hashlib.md5(document['twitter'].encode('utf-8')).hexdigest() 
            if id in players: 
                return make_response('Duplicate player', 403) 

            players[id] = document 

            response = make_response( 
                jsonify( 
                    status='ok', 
                    id=id 
                ), 
                201 
            ) 
            return response 

```

这个函数遵循一个常见的四步设计：

+   验证输入文档。模式被定义为整体 Swagger 规范的一部分。

+   创建一个密钥并确认它是唯一的。这是从数据中派生出来的一个密钥。我们也可以使用`uuid`模块创建唯一的密钥。

+   将新文档持久化到数据库中。在这个例子中，它只是一个单一的语句，`players[id] = document`。这遵循了 RESTful API 围绕已经提供了完整功能实现的类和函数构建的理念。

+   构建一个响应文档。

1.  定义一个运行服务器的主程序：

```py
        if __name__ == "__main__": 
            dealer.run(use_reloader=True, threaded=False) 

```

我们可以添加其他方法来查看多个玩家或单个玩家。这些将遵循*解析 URL 路径*配方的基本设计。我们将在下一节中看到这些。

### 客户端

这将类似于*解析 URL 路径*配方中的客户端模块：

1.  导入用于处理 RESTful API 的基本模块：

```py
        import urllib.request 
        import urllib.parse 
        import json 

```

1.  通过手动创建`ParseResult`对象来逐步创建 URL。稍后将把它合并成一个字符串：

```py
        full_url = urllib.parse.ParseResult( 
            scheme="http", 
            netloc="127.0.0.1:5000", 
            path="/dealer" + "/players", 
            params=None, 
            query=None, 
            fragment=None 
        ) 

```

1.  创建一个可以序列化为 JSON 文档并发布到服务器的对象。研究`swagger.json`可以了解这个文档的模式必须是什么样的。`文档`将包括必需的四个属性：

```py
        document = { 
            'name': 'Xander Bowers', 
            'email': 'x@example.com', 
            'year': 1985, 
            'twitter': 'https://twitter.com/PacktPub' 
        } 

```

1.  我们将结合 URL、文档、方法和标头来创建完整的请求。这将使用`urlunparse()`将 URL 部分合并成一个字符串。`Content-Type`标头通知服务器我们将提供一个 JSON 格式的文本文档：

```py
        request = urllib.request.Request( 
            url = urllib.parse.urlunparse(full_url), 
            method = "POST", 
            headers = { 
                'Accept': 'application/json', 
                'Content-Type': 'application/json;charset=utf-8', 
            }, 
            data = json.dumps(document).encode('utf-8') 
        ) 

```

我们已经包括了`charset`选项，它指定了用于从 Unicode 字符串创建字节的特定编码。由于`utf-8`编码是默认的，这是不需要的。在使用不同编码的罕见情况下，这显示了如何提供替代方案。

1.  发送请求并处理`response`对象。出于调试目的，打印`status`和`headers`信息可能会有所帮助。通常，我们只需要确保`status`是预期的`201 CREATED`：

```py
        with urllib.request.urlopen(request) as response: 
            # print(response.status) 
            assert response.status == 201 
            # print(response.headers) 
            document = json.loads(response.read().decode("utf-8")) 

        print(document) 
        assert document['status'] == 'ok' 
        id = document['id'] 

```

我们检查响应文档以确保它包含两个预期字段。

我们还可以在这个客户端中包含其他查询。我们可能想要检索所有玩家或检索特定的玩家。这些将遵循*解析 URL 路径*配方中所示的设计。

## 工作原理...

Flask 会自动检查传入的文档以解析它们。我们可以简单地使用`request.json`来利用 Flask 内置的自动 JSON 解析。

如果输入实际上不是 JSON，那么 Flask 框架将返回`400 BAD REQUEST`响应。当我们的服务器应用程序引用请求的`json`属性时，就会发生这种情况。我们可以使用`try`语句来捕获`400 BAD REQUEST`响应对象并对其进行更改，或者可能返回不同的响应。

我们使用`jsonschema`包来验证输入文档。这将检查 JSON 文档的许多特性：

+   它检查 JSON 文档的整体类型是否与模式的整体类型匹配。在这个例子中，模式要求一个对象，即`{}` JSON 结构。

+   对于模式中定义的每个属性并且在文档中存在的属性，它确认文档中的值是否与模式定义匹配。这意味着该值符合定义的 JSON 类型之一。如果有其他验证规则，比如格式、范围规范或数组的元素数量，也会进行检查。这个检查会递归地通过模式的所有级别进行。

+   如果有一个必需的字段列表，它会检查这些字段是否实际上都存在于文档中。

在这个配方中，我们将模式的细节保持在最低限度。在这个例子中省略的一个常见特性是必需属性的列表。我们还可以提供更详细的属性描述。例如，年份可能应该有一个最小值为`1900`。

在这个例子中，我们尽量将数据库更新处理保持在最低限度。在某些情况下，数据库插入可能涉及一个更复杂的过程，其中数据库客户端连接用于执行改变数据库服务器状态的命令。理想情况下，数据库处理应尽量保持在最低限度——应用程序特定的细节通常从一个单独的模块导入，并呈现为 RESTful API 资源。

在一个更大的应用程序中，可能会有一个包含所有玩家数据库处理的`player_db`模块。该模块将定义所有的类和函数。这通常会为`player`对象提供详细的模式定义。RESTful API 服务将导入这些类、函数和模式规范，并将其暴露给外部消费者。

## 还有更多...

Swagger 规范允许响应文档的示例。这通常在几个方面很有帮助：

+   开始设计作为响应一部分的示例文档是很常见的。编写描述文档的模式规范可能很困难，模式验证功能有助于确保规范与文档匹配。

+   一旦规范完成，下一步就是编写服务器端编程。有利于编写利用模式示例文档的单元测试。

+   对于 Swagger 规范的用户，可以使用响应的具体示例来设计客户端，并为客户端编程编写单元测试。

我们可以使用以下代码来确认服务器是否具有有效的 Swagger 规范。如果出现异常，要么没有 Swagger 文档，要么文档不符合 Swagger 模式：

```py
    from swagger_spec_validator import validate_spec_url
    validate_spec_url('http://127.0.0.1:5000/dealer/swagger.json') 

```

### 位置标头

`201 CREATED`响应包含了一份包含一些状态信息的小文档。状态信息包括分配给新创建记录的键。

`201 CREATED`响应通常还会在响应中包含一个额外的位置标头。此标头将提供一个 URL，可用于检索创建的文档。对于此应用程序，位置将是一个 URL，如以下示例：`http://127.0.0.1:5000/dealer/players/75f1bfbda3a8492b74a33ee28326649c`。

位置标头可以被客户端保存。完整的 URL 比从 URL 模板和值创建 URL 稍微简单。

服务器可以构建此标头如下：

```py
    response.headers['Location'] = url_for('get_player', id=str(id)) 

```

这依赖于 Flask 的`url_for()`函数。此函数接受视图函数的名称和来自 URL 路径的任何参数。然后，它使用视图函数的路由来构造完整的 URL。这将包括当前运行服务器的所有信息。插入标头后，可以返回`response`对象。

### 其他资源

服务器应该能够响应玩家列表。以下是一个最小实现，只是将数据转换为一个大的 JSON 文档：

```py
    @dealer.route('/dealer/players', methods=['GET']) 
    def get_players(): 
        response = make_response(jsonify(players)) 
        return response 

```

更复杂的实现将支持`$top`和`$skip`查询参数，以浏览玩家列表。此外，`$filter`选项可能有助于实现对玩家子集的搜索。

除了对所有玩家的通用查询外，我们还需要实现一个将返回单个玩家的方法。这种视图函数通常就像下面的代码一样简单：

```py
    @dealer.route('/dealer/players/<id>', methods=['GET']) 
    def get_player(id): 
        if id not in players: 
            return make_response("{} not found".format(id), 404) 

        response = make_response( 
            jsonify( 
                players[id] 
            ) 
        ) 
        return response 

```

此函数确认给定的 ID 是数据库中的正确键值。如果键不在数据库中，则将数据库文档转换为 JSON 表示并返回。

### 查询特定玩家

以下是定位数据库中特定值所需的客户端处理。这涉及多个步骤：

1.  首先，我们将为特定玩家创建 URL：

```py
        id = '75f1bfbda3a8492b74a33ee28326649c' 
        full_url = urllib.parse.ParseResult( 
            scheme="http", 
            netloc="127.0.0.1:5000", 
            path="/dealer" + "/players/{id}".format(id=id), 
            params=None, 
            query=None, 
            fragment=None 
        ) 

```

我们已经从信息片段构建了 URL。这被创建为一个`ParseResult`对象，具有单独的字段。

1.  给定 URL 后，我们可以创建一个`Request`对象：

```py
        request = urllib.request.Request( 
            url = urllib.parse.urlunparse(full_url), 
            method = "GET", 
            headers = { 
                'Accept': 'application/json', 
            } 
        ) 

```

1.  一旦我们有了`request`对象，我们就可以发出请求并检索响应。我们需要确认响应状态为`200`。如果是，我们就可以解析响应正文以获取描述给定玩家的 JSON 文档：

```py
        with urllib.request.urlopen(request) as response: 
            assert response.status == 200 
            player= json.loads(response.read().decode("utf-8")) 
        print(player) 

```

如果玩家不存在，`urlopen()`函数将引发异常。我们可以将其放在`try`语句中，以捕获可能引发的`403 NOT FOUND`异常，如果玩家 ID 不存在。

### 异常处理

这是所有客户端请求的一般模式。这包括显式的`try`语句：

```py
    try: 
        with urllib.request.urlopen(request) as response: 
            # print(response.status) 
            assert response.status == 201 
            # print(response.headers) 
            document = json.loads(response.read().decode("utf-8")) 

        # process the document here. 

    except urllib.error.HTTPError as ex: 
        print(ex.status) 
        print(ex.headers) 
        print(ex.read()) 

```

实际上有两种一般类型的异常：

+   **较低级别的异常**：此异常表示无法联系服务器。`ConnectionError`异常是此较低级别异常的常见示例。这是`OSError`异常的子类。

+   **来自 urllib 模块的 HTTPError 异常**：此异常表示整体 HTTP 协议运行正常，但来自服务器的响应不是成功的状态代码。成功通常是在`200`到`299`范围内的值。

+   `HTTPError`异常具有与正确响应类似的属性。它包括状态、标头和正文。

在某些情况下，`HTTPError`异常可能是服务器的几种预期响应之一。它可能不表示错误或问题。它可能只是另一个有意义的状态代码。

## 另请参阅

+   请参阅*解析 URL 路径*配方，了解其他 URL 处理示例。

+   *使用 urllib 进行 REST 请求*配方显示了其他查询字符串处理的示例。

# 为 Web 服务实现身份验证

总的来说，安全性是一个普遍的问题。应用程序的每个部分都会有安全性考虑。安全实施的部分将涉及两个密切相关的问题：

+   **身份验证**：客户端必须提供一些关于自己的证据。这可能涉及签名证书，也可能涉及像用户名和密码这样的凭据。它可能涉及多个因素，例如发送短信到用户应该有访问权限的电话。Web 服务器必须验证此身份验证。

+   **授权**：服务器必须定义权限区域，并将其分配给用户组。此外，必须将个别用户定义为授权组的成员。

虽然从技术上讲可以基于个人基础定义授权，但随着站点或应用程序的增长和变化，这往往变得笨拙。更容易为组定义安全性。在某些情况下，一个组可能（最初）只有一个人。

应用软件必须实施授权决策。对于 Flask，授权可以成为每个视图函数的一部分。个人与组的连接以及组与视图函数的连接定义了任何特定用户可用的资源。

令人困惑的是，HTTP 标准使用 HTTP `Authorization`头提供身份验证凭据。这可能会导致一些混淆，因为头的名称并不完全反映其目的。

有多种方式可以从 Web 客户端提供身份验证详细信息到 Web 服务器。以下是一些替代方案：

+   **证书**：加密的证书包括数字签名以及对**证书** **颁发机构**（**CA**）的引用：这些由**安全套接字层**（**SSL**）交换。在某些环境中，客户端和服务器都必须具有用于相互认证的证书。在其他环境中，服务器提供真实性证书，但客户端不提供。这在`https`方案中很常见。服务器不验证客户端的证书。

+   **静态 API 密钥或令牌**：Web 服务可能提供一个简单的固定密钥。这可能会附带保密建议，就像密码一样。

+   **用户名和密码**：Web 服务器可能通过用户名和密码识别用户。用户身份可能进一步通过电子邮件或短信消息进行确认。

+   **第三方身份验证**：这可能涉及使用 OpenID 等服务。有关详细信息，请参见[`openid.net`](http://openid.net)。这将涉及回调 URL，以便 OpenID 提供者可以返回通知信息。

此外，还有一个问题，即用户信息如何加载到 Web 服务器中。有些网站是自助服务的，用户提供一些最小的联系信息并被授予访问内容的权限。

在许多情况下，网站不是自助服务的。在允许访问之前，用户可能会经过仔细审查。访问可能涉及合同和访问数据或服务的费用。在某些情况下，一家公司将为其员工购买许可证，为一组特定的 Web 服务提供访问权限的用户列表。

这个示例将展示一个自助服务应用程序，其中没有定义一组用户。这意味着必须有一个 Web 服务来创建不需要任何身份验证的新用户。所有其他服务将需要一个经过适当身份验证的用户。

## 准备就绪

我们将使用`Authorization`头实现基于 HTTP 的身份验证的版本。这个主题有两种变体：

+   **HTTP 基本身份验证**：这使用简单的用户名和密码字符串。它依赖于 SSL 层来加密客户端和服务器之间的流量。

+   **HTTP 摘要身份验证**：这使用了用户名、密码和服务器提供的一次性随机数的更复杂的哈希。服务器计算预期的哈希值。如果哈希值匹配，则使用相同的字节来计算哈希，密码必须是有效的。这不需要 SSL。

SSL 经常被 Web 服务器用来建立它们的真实性。因为这项技术如此普遍，这意味着可以使用 HTTP 基本身份验证。这在 RESTful API 处理中是一个巨大的简化，因为每个请求都将包括`Authorization`头，并且客户端和服务器之间将使用安全套接字。

### 配置 SSL

获取和配置证书的详细信息超出了 Python 编程的范围。OpenSSL 软件包提供了用于创建用于配置安全服务器的自签名证书的工具。像 Comodo Group 和 Symantec 这样的 CA 提供了被 OS 供应商广泛认可的受信任的证书，以及 Mozilla 基金会。

使用 OpenSSL 创建证书有两个部分：

1.  创建一个私钥文件。通常使用以下 OS 级命令完成：

```py
 **slott$ openssl genrsa 1024 > ssl.key** 

 **Generating RSA private key, 1024 bit long modulus** 

 **.......++++++** 

 **..........................++++++** 

 **e is 65537 (0x10001)** 

```

`openssl genrsa 1024`命令创建了一个私钥文件，保存在名为`ssl.key`的文件中。

1.  使用密钥文件创建证书。以下命令是处理此事的一种方式：

```py
 **slott$ openssl req -new -x509 -nodes -sha1 -days 365 -key ssl.key > ssl.cert** 

```

您即将被要求输入将被合并到您的证书请求中的信息。您即将输入的是所谓的**Distinguished Name**（**DN**）。有相当多的字段，但您可以留下一些空白。对于某些字段，将有一个默认值。如果输入`.`，该字段将被留空。

```py
 **Country Name (2 letter code) [AU]:US** 

 **State or Province Name (full name) [Some-State]:Virginia** 

 **Locality Name (eg, city) []:** 

 **Organization Name (eg, company) [Internet Widgits Pty Ltd]:ItMayBeAHack** 

 **Organizational Unit Name (eg, section) []:** 

 **Common Name (e.g. server FQDN or YOUR name) []:Steven F. Lott** 

 **Email Address []:** 

```

`openssl req -new -x509 -nodes -sha1 -days 365 -key ssl.key`命令创建了私有证书文件，保存在`ssl.cert`中。这个证书是私下签署的，没有 CA。它只提供了有限的功能集。

这两个步骤创建了两个文件：`ssl.cert`和`ssl.key`。我们将在下面使用这些文件来保护服务器。

### 用户和凭据

为了让用户能够提供用户名和密码，我们需要在服务器上存储这些信息。关于用户凭据有一个非常重要的规则：

### 提示

永远不要存储凭据。永远不要。

显然，存储明文密码是对安全灾难的邀请。不太明显的是，我们甚至不能存储加密密码。当用于加密密码的密钥被破坏时，将导致所有用户身份的丢失。

如果我们不存储密码，如何检查用户的密码？

解决方案是存储哈希而不是密码。第一次创建密码时，服务器保存了哈希摘要。之后每次，用户的输入都会被哈希并与保存的哈希进行比较。如果两个哈希匹配，则密码必须是正确的。核心问题是从哈希中恢复密码的极端困难。

创建密码的初始哈希值有一个三步过程：

1.  创建一个随机的`salt`值。通常使用`os.urandom()`生成 16 字节。

1.  使用`salt`加上密码创建`hash`值。通常情况下，使用`hashlib`来实现。具体来说，使用`hashlib.pbkdf2_hmac()`。为此使用特定的摘要算法，例如`md5`或`sha224`。

1.  保存摘要名称、`salt`和哈希字节。通常这些被合并成一个看起来像`md5$salt$hash`的单个字符串。`md5`是一个文字。`$`分隔算法名称、`salt`和`hash`值。

当需要检查密码时，会遵循类似的过程：

1.  根据用户名，找到保存的哈希字符串。这将具有摘要算法名称、保存的盐和哈希字节的三部分结构。这些元素可以用`$`分隔。

1.  使用保存的盐加上用户提供的候选密码创建计算的`hash`值。

1.  如果计算的哈希字节与保存的哈希字节匹配，我们知道摘要算法和盐匹配；因此，密码也必须匹配。

我们将定义一个简单的类来保留用户信息以及哈希密码。我们可以使用 Flask 的`g`对象在请求处理期间保存用户信息。

### Flask 视图函数装饰器

有几种处理身份验证检查的替代方法：

+   如果每个路由都具有相同的安全要求，那么`@dealer.before_request`函数可以用于验证所有`Authorization`标头。这将需要一些异常处理，用于`/swagger.json`路由和允许未经授权的用户创建其新用户名和密码凭据的自助服务路由。

+   当一些路由需要身份验证而另一些不需要时，引入需要身份验证的路由的装饰器是很好的。

Python 装饰器是一个包装另一个函数以扩展其功能的函数。核心技术看起来像这样：

```py
    from functools import wraps 
    def decorate(function): 
        @wraps(function) 
        def decorated_function(*args, **kw): 
            # processing before 
            result = function(*args, **kw) 
            # processing after 
            return result 
        return decorated_function 

```

这个想法是用一个新函数`decorated_function`来替换给定的函数`function`。在装饰函数的主体内，它执行原始函数。在装饰的函数之前可以进行一些处理，在函数之后也可以进行一些处理。

在 Flask 上下文中，我们将在`@route`装饰器之后放置我们的装饰器：

```py
    @dealer.route('/path/to/resource') 
    @decorate 
    def view_function(): 
        return make_result('hello world', 200) 

```

我们用`@decorate`装饰器包装了`view_function()`。装饰器可以检查身份验证，以确保用户已知。我们可以在这些函数中进行各种处理。

## 如何做到这一点...

我们将这分解为四个部分：

+   定义`User`类

+   定义视图装饰器

+   创建服务器

+   创建一个示例客户端

### 定义用户类

这个类定义提供了一个单独的`User`对象的定义示例：

1.  导入所需的模块以创建和检查密码：

```py
        import hashlib 
        import os 
        import base64 

```

其他有用的模块包括`json`，以便可以正确序列化`User`对象。

1.  定义`User`类：

```py
        class User: 

```

1.  由于我们将更改密码生成和检查的某些方面，因此我们将作为整体类定义的一部分提供两个常量：

```py
        DIGEST = 'sha384' 
        ROUNDS = 100000 

```

我们将使用**SHA-384**摘要算法。这提供了 64 字节的摘要。我们将使用**基于密码的密钥派生函数 2**（**PBKDF2**）算法进行 100,000 轮。

1.  大多数情况下，我们将从 JSON 文档创建用户。这将是一个可以使用`**`转换为关键字参数值的字典：

```py
        def __init__(self, **document): 
            self.name = document['name'] 
            self.year = document['year'] 
            self.email = document['email'] 
            self.twitter = document['twitter'] 
            self.password = None 

```

请注意，我们不希望直接设置密码。相反，我们将单独设置密码，而不是创建用户文档时。

我们省略了其他授权细节，例如用户所属的组列表。我们还省略了一个指示密码需要更改的指示器。

1.  定义设置密码`hash`值的算法：

```py
        def set_password(self, password): 
            salt = os.urandom(30) 
            hash = hashlib.pbkdf2_hmac( 
                self.DIGEST, password.encode('utf-8'), salt, self.ROUNDS) 
            self.password = '$'.join( 
                [self.DIGEST, 
                 base64.urlsafe_b64encode(salt).decode('ascii'), 
                 base64.urlsafe_b64encode(hash).decode('ascii') 
                ] 
            ) 

```

我们使用`os.urandom()`构建了一个随机盐。然后，我们使用给定的摘要算法、密码和`salt`构建了完整的`hash`值。我们使用可配置的轮数。

请注意，哈希计算是按字节而不是 Unicode 字符进行的。我们使用`utf-8`编码将密码编码为字节。

我们使用摘要算法的名称、盐和编码的`hash`值组装了一个字符串。我们使用 URL 安全的`base64`编码字节，以便可以轻松显示完整的哈希密码值。它可以保存在任何类型的数据库中，因为它只使用`A-Z`，`a-z`，`0-9`，`-`和`_`。

请注意，`urlsafe_b64encode()`创建一个字节值的字符串。这些必须解码才能看到它们代表的 Unicode 字符。我们在这里使用 ASCII 编码方案，因为`base64`只使用六十四个标准 ASCII 字符。

1.  定义检查密码哈希值的算法：

```py
        def check_password(self, password): 
            digest, b64_salt, b64_expected_hash = self.password.split('$') 
            salt = base64.urlsafe_b64decode(b64_salt) 
            expected_hash = base64.urlsafe_b64decode(b64_expected_hash) 
            computed_hash = hashlib.pbkdf2_hmac( 
                digest, password.encode('utf-8'), salt, self.ROUNDS) 
            return computed_hash == expected_hash 

```

我们已经将密码哈希分解为`digest`、`salt`和`expected_hash`值。由于各部分都是`base64`编码的，因此必须对其进行解码以恢复原始字节。

请注意，哈希计算以字节而不是 Unicode 字符工作。我们使用`utf-8`编码将密码编码为字节。`hashlib.pbkdf2_hmac()`的计算结果与预期结果进行比较。如果它们匹配，那么密码必须是相同的。

这是这个类如何使用的演示：

```py
 **>>> details = {'name': 'xander', 'email': 'x@example.com', 
...     'year': 1985, 'twitter': 'https://twitter.com/PacktPub' } 
>>> u = User(**details) 
>>> u.set_password('OpenSesame') 
>>> u.check_password('opensesame') 
False 
>>> u.check_password('OpenSesame') 
True** 

```

这个测试用例可以包含在类 docstring 中。有关这种测试用例的更多信息，请参见第十一章中的*使用 docstring 进行测试*配方，*测试*。

在更复杂的应用程序中，可能还会有一个用户集合的定义。这通常使用某种数据库来定位用户和插入新用户。

### 定义视图装饰器

1.  从`functools`导入`@wraps`装饰器。这有助于通过确保新函数具有从被装饰的函数复制的原始名称和文档字符串来定义装饰器：

```py
        from functools import wraps 

```

1.  为了检查密码，我们需要`base64`模块来帮助分解`Authorization`头的值。我们还需要报告错误，并使用全局`g`对象更新 Flask 处理上下文：

```py
        import base64 
        from flask import g 
        from http import HTTPStatus 

```

1.  定义装饰器。所有装饰器都有这个基本的轮廓。我们将在下一步中替换`这里处理`部分：

```py
        def authorization_required(view_function): 
            @wraps(view_function) 
            def decorated_function(*args, **kwargs): 
                processing here 
            return decorated_function 

```

1.  以下是检查头的处理步骤。请注意，遇到的每个问题都会简单地中止处理，并将`401 UNAUTHORIZED`作为状态码。为了防止黑客探索算法，尽管根本原因不同，但所有结果都是相同的：

```py
        if 'Authorization' not in request.headers: 
            abort(HTTPStatus.UNAUTHORIZED) 
        kind, data = request.headers['Authorization'].split() 
        if kind.upper() != 'BASIC': 
            abort(HTTPStatus.UNAUTHORIZED) 
        credentials = base64.decode(data) 
        username, _, password = credentials.partition(':') 
        if username not in user_database: 
            abort(HTTPStatus.UNAUTHORIZED) 
        if not user_database[username].check_password(password): 
            abort(HTTPStatus.UNAUTHORIZED) 
        g.user = user_database[username] 
        return view_function(*args, **kwargs) 

```

必须成功通过一些条件：

+   必须存在`Authorization`头

+   标题必须指定基本身份验证

+   该值必须包括使用`base64`编码的`username:password`字符串

+   用户名必须是已知的用户名

+   从密码计算出的哈希值必须与预期的密码哈希值匹配

任何单个失败都会导致`401 UNAUTHORIZED`响应。

### 创建服务器

这与*解析 JSON 请求*配方中显示的服务器相似。有一些重要的修改：

1.  创建本地自签名证书或从证书颁发机构购买证书。对于这个配方，我们假设两个文件名分别是`ssl.cert`和`ssl.key`。

1.  导入构建服务器所需的模块。还要导入`User`类定义：

```py
        from flask import Flask, jsonify, request, abort, url_for 
        from ch12_r07_user import User 
        from http import HTTPStatus 

```

1.  包括`@authorization_required`装饰器定义。

1.  定义一个无需身份验证的路由。这将用于创建新用户。在*解析 JSON 请求*配方中定义了一个类似的视图函数。这个版本需要传入文档中的密码属性。这将是用于创建哈希的明文密码。明文密码不会保存在任何地方；只有哈希值会被保留：

```py
        @dealer.route('/dealer/players', methods=['POST']) 
        def make_player(): 
            try: 
                document = request.json 
            except Exception as ex: 
                # Document wasn't even JSON. We can fine-tune 
                # the error message here. 
                raise 
            player_schema = specification['definitions']['player'] 
            try: 
                validate(document, player_schema) 
            except ValidationError as ex: 
                return make_response(ex.message, 403) 

            id = hashlib.md5(document['twitter'].encode('utf-8')).hexdigest() 
            if id in user_database: 
                return make_response('Duplicate player', 403) 

            new_user = User(**document) 
            new_user.set_password(document['password']) 
            user_database[id] = new_user 

            response = make_response( 
                jsonify( 
                    status='ok', 
                    id=id 
                ), 
                201 
            ) 
            response.headers['Location'] = url_for('get_player', id=str(id)) 
            return response 

```

创建用户后，密码将单独设置。这遵循了一些应用程序设置的模式，其中用户是批量加载的。这个处理可能为每个用户提供一个临时密码，必须立即更改。

请注意，每个用户都被分配了一个神秘的 ID。分配的 ID 是从他们的 Twitter 句柄的十六进制摘要计算出来的。这是不寻常的，但它表明有很大的灵活性可用。

如果我们希望用户选择自己的用户名，我们需要将其添加到请求文档中。我们将使用该用户名而不是计算出的 ID 值。

1.  为需要身份验证的路由定义路由。在*解析 JSON 请求*配方中定义了一个类似的视图函数。这个版本使用`@authorization_required`装饰器：

```py
        @dealer.route('/dealer/players/<id>', methods=['GET']) 
        @authorization_required 
        def get_player(id): 
            if id not in user_database: 
                return make_response("{} not found".format(id), 404) 

            response = make_response( 
                jsonify( 
                    players[id] 
                ) 
            ) 
            return response 

```

大多数其他路由将具有类似的`@authorization_required`装饰器。一些路由，如`/swagger.json`路由，将不需要授权。

1.  `ssl`模块定义了`ssl.SSLContext`类。上下文可以加载以前创建的自签名证书和私钥文件。然后 Flask 对象的`run()`方法使用该上下文。这将从`http://127.0.01:5000`的 URL 中更改方案为`https://127.0.0.1:5000`：

```py
        import ssl 
        ctx = ssl.SSLContext(ssl.PROTOCOL_SSLv23) 
        ctx.load_cert_chain('ssl.cert', 'ssl.key') 
        dealer.run(use_reloader=True, threaded=False, ssl_context=ctx) 

```

### 创建一个示例客户端

1.  创建一个与自签名证书一起使用的 SSL 上下文：

```py
        import ssl 
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH) 
        context.check_hostname = False 
        context.verify_mode = ssl.CERT_NONE 

```

这个上下文可以用于所有`urllib`请求。这将礼貌地忽略证书上缺少 CA 签名。

这是我们如何使用这个上下文来获取 Swagger 规范的方式：

```py
        with urllib.request.urlopen(swagger_request, context=context) as response: 
            swagger = json.loads(response.read().decode("utf-8")) 
            pprint(swagger) 

```

1.  创建用于创建新玩家实例的 URL。请注意，我们必须使用`https`作为方案。我们已经构建了一个`ParseResult`对象，以便分别显示 URL 的各个部分：

```py
        full_url = urllib.parse.ParseResult( 
            scheme="https", 
            netloc="127.0.0.1:5000", 
            path="/dealer" + "/players", 
            params=None, 
            query=None, 
            fragment=None 
        ) 

```

1.  创建一个 Python 对象，将被序列化为 JSON 文档。这个模式类似于*解析 JSON 请求*食谱中显示的示例。这包括一个额外的属性，即纯文本：

```py
        password.document = { 
            'name': 'Hannah Bowers', 
            'email': 'h@example.com', 
            'year': 1987, 
            'twitter': 'https://twitter.com/PacktPub', 
            'password': 'OpenSesame' 
        } 

```

因为 SSL 层使用加密套接字，所以发送这样的纯文本密码是可行的。

1.  我们将 URL、文档、方法和标头组合成完整的`Request`对象。这将使用`urlunparse()`将 URL 部分合并为一个字符串。`Content-Type`标头通知服务器我们将以 JSON 表示法提供文本文档：

```py
        request = urllib.request.Request( 
            url = urllib.parse.urlunparse(full_url), 
            method = "POST", 
            headers = { 
                'Accept': 'application/json', 
                'Content-Type': 'application/json;charset=utf-8', 
            }, 
            data = json.dumps(document).encode('utf-8') 
        ) 

```

1.  我们可以发布此文档以创建新玩家：

```py
        try: 
            with urllib.request.urlopen(request, context=context) as response: 
                # print(response.status) 
                assert response.status == 201 
                # print(response.headers) 
                document = json.loads(response.read().decode("utf-8")) 

            print(document) 
            assert document['status'] == 'ok' 
            id = document['id'] 
        except urllib.error.HTTPError as ex: 
            print(ex.status) 
            print(ex.headers) 
            print(ex.read()) 

```

快乐路径将收到`201`状态响应，并且用户将被创建。响应将包括分配的用户 ID 和多余的状态代码。

如果用户是重复的，或者文档不匹配模式，那么将会引发`HTTPError`异常。这可能会有有用的错误消息可以显示。

1.  我们可以使用分配的 ID 和已知密码创建一个`Authorization`标头：

```py
        import base64 
        credentials = base64.b64encode(b'75f1bfbda3a8492b74a33ee28326649c:OpenSesame') 

```

`Authorization`标头有一个两个单词的值：`b"BASIC " + credentials`。单词`BASIC`是必需的。凭据必须是`username:password`字符串的`base64`编码。在这个例子中，用户名是在创建用户时分配的特定 ID。

1.  这是一个查询所有玩家的 URL。我们已经构建了一个`ParseResult`对象，以便分别显示 URL 的各个部分：

```py
        full_url = urllib.parse.ParseResult( 
            scheme="https", 
            netloc="127.0.0.1:5000", 
            path="/dealer" + "/players", 
            params=None, 
            query=None, 
            fragment=None 
        ) 

```

1.  我们可以将 URL、方法和标头组合成一个单独的`Request`对象。这包括`Authorization`标头，其中包含用户名和密码的`base64`编码：

```py
        request = urllib.request.Request( 
            url = urllib.parse.urlunparse(full_url), 
            method = "GET", 
            headers = { 
                'Accept': 'application/json', 
                'Authorization': b"BASIC " + credentials 
            } 
        ) 

```

1.  `Request`对象可用于从服务器进行查询并使用`urllib`处理响应：

```py
        request.urlopen(request, context=context) as response: 
            assert response.status == 200 
            # print(response.headers) 
            players = json.loads(response.read().decode("utf-8")) 

        pprint(players) 

```

预期状态是`200`。响应应该是一个已知`players`列表的 JSON 文档。

## 它是如何工作的...

这个食谱有三个部分：

+   **使用 SSL 提供安全通道**：这使得直接交换用户名和密码成为可能。我们可以使用更简单的 HTTP 基本身份验证方案，而不是更复杂的 HTTP 摘要身份验证。Web 服务使用各种其他身份验证方案；其中大多数需要 SSL。

+   **使用最佳的密码哈希实践**：以任何形式保存密码都是安全风险。我们不保存纯文本密码，甚至加密密码，而是只保存密码的计算哈希值和一个随机盐字符串。这确保我们几乎不可能从哈希值中逆向工程密码。

+   **使用装饰器**：它用于区分需要身份验证和不需要身份验证的路由。这允许在创建 Web 服务时具有很大的灵活性。

在所有路由都需要身份验证的情况下，我们可以将密码检查算法添加到`@dealer.before_request`函数中。这将集中所有身份验证检查。这也意味着需要一个单独的管理流程来定义用户和散列密码。

这里的关键是服务器上的安全检查是一个简单的`@authorization_required`装饰器。很容易确保它在所有视图函数中都存在。

## 还有更多...

这个服务器有一套相对简单的授权规则：

+   大多数路由需要有效用户。这是通过在视图函数中存在`@authorization_required`装饰器来实现的。

+   对于`/dealer/swagger.json`的`GET`和`/dealer/players`的`POST`不需要有效用户。这是通过缺少额外装饰器来实现的。

在许多情况下，我们将有一个更复杂的特权、组和用户配置。最小特权原则建议用户应该被分隔成组，并且每个组应该具有尽可能少的特权来实现他们的目标。

这通常意味着我们将有一个管理组来创建新用户，但没有其他访问权限来使用 RESTful Web 服务。用户可以访问 Web 服务，但无法创建任何其他用户。

这需要对我们的数据模型进行几处更改。我们应该定义用户组并将用户分配到这些组中：

```py
    class Group: 
        '''A collection of users.''' 
        pass 

    administrators = Group() 
    players = Group() 

```

然后我们可以扩展`User`的定义以包括组成员资格：

```py
    class GroupUser(User): 
        def __init__(self, *args, **kw): 
            super().__init__(*args, **kw) 
            self.groups = set() 

```

当我们创建`GroupUser`类的新实例时，我们也可以将它们分配到特定的组中：

```py
    u = GroupUser(**document) 
    u.groups = set(players) 

```

现在我们可以扩展我们的装饰器来检查经过身份验证的用户的`groups`属性。带参数的装饰器比无参数的装饰器复杂一些：

```py
    def group_member(group_instance): 
        def group_member_decorator(view_function): 
            @wraps(view_function) 
            def decorated_view_function(*args, **kw): 
                # Check Password and determine user 
                if group_instance not in g.user.groups: 
                    abort(HTTPStatus.UNAUTHORIZED) 
                return view_function(*args, **kw) 
            return decorated_view_function 
        return group_member_decorator 

```

带参数的装饰器通过创建一个包含参数的具体装饰器来工作。具体装饰器`group_member_decorator`将包装给定的视图函数。这将解析`Authorization`头，找到`GroupUser`实例并检查组成员资格。

我们使用`＃Check Password and determine user`作为一个重构函数来检查`Authorization`头的占位符。`@authorization_required`装饰器的核心功能需要被提取到一个独立的函数中，以便在多个地方使用。

然后我们可以使用这个装饰器如下：

```py
    @dealer.route('/dealer/players') 
    @group_member(administrators) 
    def make_player(): 
        etc. 

```

这缩小了每个单独视图函数的特权范围。它确保了 RESTful Web 服务遵循最小特权原则。

### 创建一个命令行界面

在与具有特殊管理员特权的站点一起工作时，我们经常需要提供一种创建初始管理用户的方式。然后，这个用户可以创建所有具有非管理特权的用户。这通常是通过在 Web 服务器上直接由管理用户运行的 CLI 应用程序来完成的。

Flask 支持使用装饰器定义必须在 RESTful Web 服务环境之外运行的命令。我们可以使用`@dealer.cli.command()`来定义一个从命令行运行的命令。例如，这个命令可以加载初始的管理用户。也可以创建一个命令来从列表中加载用户。

`getpass`模块是管理用户以不会在终端上回显的方式提供他们的初始密码的一种方式。这可以确保站点的凭据正在安全处理。

### 构建身份验证头

依赖于 HTTP 基本`Authorization`头的 Web 服务可以通过两种常见的方式来支持：

+   使用凭据构建`Authorization`头，并在每个请求中包含它。为此，我们需要提供字符串`username:password`的正确`base64`编码。这种替代方法的优势在于相对简单。

+   使用`urllib`功能自动提供`Authorization`头：

```py
        from urllib.request import HTTPBasicAuthHandler,         HTTPPasswordMgrWithDefaultRealm 
        auth_handler = urllib.request.HTTPBasicAuthHandler( 
            password_mgr=HTTPPasswordMgrWithDefaultRealm) 
        auth_handler.add_password( 
            realm=None, 
            uri='https://127.0.0.1:5000/', 
            user='Aladdin', 
            passwd='OpenSesame') 
        password_opener = urllib.request.build_opener(auth_handler) 

```

我们创建了一个`HTTPBasicAuthHandler`的实例。这个实例包含了可能需要的所有用户名和密码。对于从多个站点收集数据的复杂应用程序，可能需要向处理程序添加多组凭据。

现在，我们将使用`with password_opener(request) as response:`而不是`with urllib.request.urlopen(request) as response:`。`password_opener`对象会在请求中添加`Authorization`头。

这种替代方案的优势在于相对灵活。我们可以在不遇到任何困难的情况下切换到使用`HTTPDigestAuthHandler`。我们还可以添加额外的用户名和密码。

有时候领域信息会让人感到困惑。领域是多个 URL 的容器。当服务器需要身份验证时，它会响应`401`状态码。这个响应将包括一个`Authenticate`头，指定凭据必须属于的领域。由于领域包含多个站点 URL，领域信息往往非常静态。`HTTPBasicAuthHandler`使用领域和 URL 信息来选择在授权响应中提供哪些用户名和密码。

通常需要编写一个技术性的尝试连接的技术性尝试，并打印`401`响应中的头部，以查看领域字符串是什么。一旦领域已知，就可以构建`HTTPBasicAuthHandler`。另一种方法是使用一些浏览器中可用的开发者模式来检查头部并查看`401`响应的详细信息。

## 另请参阅

+   服务器的适当 SSL 配置通常涉及使用由 CA 签名的证书。这涉及一个以服务器为起点并包括为各种颁发证书的各种机构的证书链。

+   许多 Web 服务实现使用诸如 GUnicorn 或 NGINX 之类的服务器。这些服务器通常在我们的应用程序之外处理 HTTP 和 HTTPS 问题。它们还可以处理复杂的证书链和捆绑包。

+   有关详细信息，请参阅[`docs.gunicorn.org/en/stable/settings.html#ssl`](http://docs.gunicorn.org/en/stable/settings.html#ssl)和[`nginx.org/en/docs/http/configuring_https_servers.html`](http://nginx.org/en/docs/http/configuring_https_servers.html)。
